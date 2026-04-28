#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limiting-magnitude estimation utilities.

This module supports both background-based limiting estimates and injection/
recovery experiments by simulating PSF sources into science cutouts and
measuring the detection threshold required for robust photometry.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import logging
import os
import sys
import time
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import pandas as pd


@contextmanager
def _pool_or_serial(n_jobs: int):
    """Yield a ProcessPoolExecutor when n_jobs > 1, else None (serial; avoids fork on HPC)."""
    if n_jobs <= 1:
        yield None
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        yield pool


@lru_cache(maxsize=512)
def _flux_for_mag_cached(m: float, counts_ref: float, exposure_time: float) -> float:
    """
    ePSF ``flux`` parameter for instrumental magnitude *m* (cached).

    ``counts_ref`` is the aperture sum integrated over the exposure, same units
    as ``Aperture.counts_AP`` (e⁻ in the frame when the pipeline uses
    ``image * gain``).  ``m`` uses the same e⁻/s convention as ``mag(flux_AP)``.

    Returns the scale factor for ``epsf_model.evaluate(..., flux=...)`` such
    that the *in-aperture* integrated signal matches a source of magnitude *m*.
    """
    flux_e_per_s = 10.0 ** (-0.4 * m)
    aperture_e_in_frame = flux_e_per_s * float(exposure_time)
    return aperture_e_in_frame / float(counts_ref)


# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus

from functions import log_warning_from_exception

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from functions import (
    set_size,
    gauss_1d,  # used directly - NOT redefined inside methods
    flux_upper_limit,
    mag,
    points_in_circum,
    beta_aperture,
)
from aperture import (
    Aperture,
    resolve_exposure_time_seconds,
    resolve_gain_e_per_adu,
)
from plotting_utils import get_marker_size


def _effective_exposure_seconds(input_yaml: dict) -> float:
    """Exposure time [s]; same rules as ``Aperture.measure`` (required, no default)."""
    return resolve_exposure_time_seconds(None, input_yaml)


def _downsample_psf_flux_conserving(psf_os: np.ndarray, oversampling: int) -> np.ndarray:
    """
    Flux-conserving binning of an oversampled PSF onto detector pixels.

    Same convention as ``Limits._downsample_psf`` (block sum over oversample
    cells).  Used so ePSF oversampling matches photutils / pipeline PSF models.
    """
    osamp = int(oversampling)
    if osamp <= 1:
        return np.asarray(psf_os, dtype=float)
    H_os, W_os = psf_os.shape
    H = H_os // osamp
    W = W_os // osamp
    return (
        psf_os[: H * osamp, : W * osamp]
        .reshape(H, osamp, W, osamp)
        .sum(axis=(1, 3))
    )


def _render_epsf_on_cutout(
    epsf_model,
    height: int,
    width: int,
    x_0: float,
    y_0: float,
    flux: float,
    oversampling: int,
) -> np.ndarray:
    """
    Render the ePSF onto a full ``(height, width)`` detector grid.

    When ``oversampling > 1``, evaluate on a subpixel grid and bin down with
    flux-conserving downsampling.  Injection trials **must** use the same path
    as ``counts_ref`` calibration; otherwise ``flux_for_mag`` is wrong and
    limiting magnitudes can be biased (often spuriously deep).
    """
    osamp = max(1, int(oversampling))
    H, W = int(height), int(width)
    if osamp <= 1:
        gridy, gridx = np.indices((H, W))
        return np.asarray(
            epsf_model.evaluate(
                x=gridx,
                y=gridy,
                flux=float(flux),
                x_0=float(x_0),
                y_0=float(y_0),
            ),
            dtype=float,
        )
    gx_os = np.linspace(0, W - 1, W * osamp)
    gy_os = np.linspace(0, H - 1, H * osamp)
    gridx_os, gridy_os = np.meshgrid(gx_os, gy_os)
    psf_os = np.asarray(
        epsf_model.evaluate(
            x=gridx_os,
            y=gridy_os,
            flux=float(flux),
            x_0=float(x_0),
            y_0=float(y_0),
        ),
        dtype=float,
    )
    return _downsample_psf_flux_conserving(psf_os, osamp)


# Sigma multiplier passed to ``beta_aperture`` for injection trials and related
# completeness (same n as Connolly-style aperture significance). Keep in sync
# with any code that recomputes aperture beta to match injection (e.g. main.py).
BETA_APERTURE_SIGMA_N = 3.0


def _compute_p_det(df: pd.DataFrame, beta_n: float) -> pd.Series:
    """
    Compute detection probability (beta) for each site.

    Expects ``flux_AP`` and ``noiseSky`` in matching *per-second* units
    (e⁻/s and e⁻/s per pixel) as from ``Aperture.measure()``.
    """
    def _beta_row(row):
        if (np.isfinite(row["flux_AP"]) and row["flux_AP"] > 0
                and np.isfinite(row["noiseSky"]) and row["noiseSky"] > 0
                and np.isfinite(row["area"]) and row["area"] > 0):
            return beta_aperture(
                n=beta_n,
                flux_aperture=row["flux_AP"],
                sigma=row["noiseSky"],
                npix=row["area"],
            )
        return 0.0
    return df.apply(_beta_row, axis=1)


# ===========================================================================
# Module-level worker functions
# ===========================================================================
# Workers must live at module scope so pickle (used by multiprocessing) can
# find them by name.  Instance methods cannot be pickled on all platforms.


def _fake_aperture_worker(args):
    """
    Draw *n_trials* random fake apertures from unmasked background pixels and
    return their summed flux.

    Parameters
    ----------
    args : tuple
        (cutout_e, includ_zip, aperture_area, n_trials, seed)
        cutout_e      - gain-scaled cutout (2-D ndarray)
        includ_zip    - (N, 2) array of valid (row, col) background pixel coords
        aperture_area - number of pixels per synthetic aperture
        n_trials      - number of apertures to draw in this chunk
        seed          - RNG seed for reproducibility
    """
    cutout_e, includ_zip, aperture_area, n_trials, seed = args
    rng = np.random.default_rng(seed)
    # Sample pixel indices with replacement so every trial is independent.
    idx = rng.integers(
        0, len(includ_zip), size=(n_trials, aperture_area), endpoint=False
    )
    return np.nansum(cutout_e[includ_zip[idx, 0], includ_zip[idx, 1]], axis=1)


def _injection_worker(args):
    """
    Inject a scaled PSF at a jittered position and attempt recovery.

    Parameters
    ----------
    args : tuple
        (x_inj, y_inj, F_amp, cutout, oversampling,
         epsf_model, input_yaml, background_rms,
         snr_limit, beta_n, DETECTION_BETA_THRESH, recovery_method)

    Returns
    -------
    (detection_flag, beta_p) : (bool, float)
    """
    (
        x_inj,
        y_inj,
        F_amp,
        cutout,
        oversampling,
        epsf_model,
        input_yaml,
        background_rms,
        snr_limit,
        beta_n,
        DETECTION_BETA_THRESH,
        recovery_method,
    ) = args

    try:
        ny, nx = cutout.shape
        psf_img = _render_epsf_on_cutout(
            epsf_model, ny, nx, x_inj, y_inj, F_amp, oversampling
        )
        new_img = cutout + psf_img

        # Guard: if the combined image is all-NaN, photometry cannot proceed.
        n_finite = int(np.count_nonzero(np.isfinite(new_img)))
        if n_finite == 0:
            return False, 0.0, np.nan
        # Replace any residual NaN pixels with local background median
        # to prevent annulus sky measurement from returning NaN.
        if n_finite < new_img.size:
            fill_val = float(np.nanmedian(new_img))
            if not np.isfinite(fill_val):
                fill_val = 0.0
            new_img = np.where(np.isfinite(new_img), new_img, fill_val)

        # Compute beta using the canonical aperture-based formalism (n=3) so
        # beta thresholds remain comparable across runs/methods.
        ap = Aperture(input_yaml=input_yaml, image=new_img)
        trial_df = pd.DataFrame({"x_pix": [x_inj], "y_pix": [y_inj]})
        mres = ap.measure(
            sources=trial_df, plot=False, background_rms=background_rms, verbose=0
        )

        # noiseSky from Aperture.measure() is per-pixel RMS (not total noise)
        # beta_aperture uses sigma * sqrt(npix) internally to compute total noise
        # Validate noiseSky first before computing beta_p
        noise_sky = float(mres["noiseSky"].iloc[0])
        flux_ap = float(mres["flux_AP"].iloc[0])
        area_px = float(mres["area"].iloc[0])

        if not (np.isfinite(noise_sky) and noise_sky > 0):
            return False, 0.0, np.nan

        beta_p = beta_aperture(
            n=beta_n,
            flux_aperture=flux_ap,
            sigma=noise_sky,
            npix=area_px,
        )
        det_beta = beta_p >= DETECTION_BETA_THRESH

        # Detection SNR gate: either aperture SNR (legacy) or PSF-fit SNR.
        method = str(recovery_method).strip().upper() if recovery_method is not None else "AP"
        snr_val = np.nan
        flux_hat = np.nan  # Initialize before PSF block
        if method == "PSF":
            try:
                # Fast PSF-flux estimate using weighted least squares on a local stamp.
                phot_cfg = input_yaml.get("photometry") or {}
                fwhm_px = float(input_yaml.get("fwhm", 3.0))
                scale = float(phot_cfg.get("psf_fit_shape_vfaint_scale_fwhm", 2.0))
                half = int(np.ceil(max(3.0, scale * fwhm_px)))

                ny, nx = new_img.shape
                x0i = float(x_inj)
                y0i = float(y_inj)
                x1 = max(0, int(np.floor(x0i)) - half)
                x2 = min(nx, int(np.floor(x0i)) + half + 2)
                y1 = max(0, int(np.floor(y0i)) - half)
                y2 = min(ny, int(np.floor(y0i)) + half + 2)

                data = np.asarray(new_img[y1:y2, x1:x2], dtype=float)
                # Same PSF rendering as injection (handles oversampling consistently).
                psf_stamp = np.asarray(psf_img[y1:y2, x1:x2], dtype=float)
                if np.isfinite(F_amp) and abs(float(F_amp)) > 1e-30:
                    psf1 = psf_stamp / float(F_amp)
                else:
                    psf1 = _render_epsf_on_cutout(
                        epsf_model,
                        y2 - y1,
                        x2 - x1,
                        float(x0i - x1),
                        float(y0i - y1),
                        1.0,
                        oversampling,
                    )
                # Fit both PSF flux and a constant background level in the stamp:
                # data ~= flux * psf1 + bkg. This correctly handles any local mean
                # shift (including the uniform "bias") without forcing background=0
                # or relying on an annulus estimate.
                if background_rms is not None:
                    # RMS map must be in the same units per pixel as ``cutout`` (typically ADU);
                    # variance is then consistent with ``data`` in the WLS model.
                    rms = np.asarray(background_rms[y1:y2, x1:x2], dtype=float)
                    var = np.maximum(rms * rms, 1e-30)
                else:
                    # Fallback variance: robust scatter of the stamp.
                    sig = float(np.nanstd(data))
                    var = np.full_like(data, max(sig * sig, 1e-30), dtype=float)

                ok = np.isfinite(data) & np.isfinite(psf1) & np.isfinite(var) & (var > 0)
                if int(np.count_nonzero(ok)) >= 10:
                    w = 1.0 / var[ok]
                    a = np.vstack([psf1[ok].ravel(), np.ones(int(np.count_nonzero(ok)))])  # (2, N)
                    # Weighted normal equations: (A W A^T)^{-1} A W y
                    aw = a * w  # broadcast weights across rows
                    m = aw @ a.T  # 2x2
                    b = aw @ data[ok].ravel()  # 2,
                    try:
                        cov = np.linalg.inv(m)
                    except Exception:
                        cov = None
                    if cov is not None and np.all(np.isfinite(cov)):
                        theta = cov @ b  # (flux, bkg)
                        flux_hat = float(theta[0])
                        flux_err = float(np.sqrt(max(cov[0, 0], 0.0)))
                        if np.isfinite(flux_err) and flux_err > 0:
                            snr_val = flux_hat / flux_err
            except Exception:
                snr_val = np.nan
        elif method == "EMCEE":
            # Robust (but slow) PSF photometry recovery using the same MCMCFitter
            # used by the main PSF pipeline. For performance and stability, this
            # is best run serial (n_jobs=1) at the get_injected_limit level.
            try:
                from psf import MCMCFitter

                phot_cfg = input_yaml.get("photometry") or {}
                fwhm_px = float(input_yaml.get("fwhm", 3.0))
                scale = float(phot_cfg.get("psf_fit_shape_vfaint_scale_fwhm", 2.0))
                half = int(np.ceil(max(6.0, scale * fwhm_px)))

                ny, nx = new_img.shape
                x0i = float(x_inj)
                y0i = float(y_inj)
                x1 = max(0, int(np.floor(x0i)) - half)
                x2 = min(nx, int(np.floor(x0i)) + half + 2)
                y1 = max(0, int(np.floor(y0i)) - half)
                y2 = min(ny, int(np.floor(y0i)) + half + 2)

                stamp = np.asarray(new_img[y1:y2, x1:x2], dtype=float)
                gx, gy = np.meshgrid(np.arange(stamp.shape[1]), np.arange(stamp.shape[0]))
                # gx/gy are stamp-local coordinates (correct)

                # For EMCEE recovery we do not force background=0. The MCMC fitter
                # uses an uncertainty model; background offsets are handled via the
                # local background treatment already applied in Aperture-based beta
                # and via the PSF flux posterior.

                # Local RMS if available.
                rms_stamp = None
                if background_rms is not None:
                    rms_stamp = np.asarray(background_rms[y1:y2, x1:x2], dtype=float)
                    rms_stamp = np.abs(rms_stamp)

                model = epsf_model.copy()
                # Place the source at injected coordinates in stamp-local coordinates.
                model.x_0.value = float(x0i - x1)  # stamp-local
                model.y_0.value = float(y0i - y1)  # stamp-local
                model.flux.value = float(F_amp)

                # Delta (px) bound for centroid; keep it tight since injection pos is known.
                raw_delta = phot_cfg.get("emcee_delta", None)
                delta_px = float(raw_delta) if raw_delta is not None else 1.0

                fitter = MCMCFitter(
                    nwalkers=int(phot_cfg.get("emcee_nwalkers", 20)),
                    nsteps=int(phot_cfg.get("emcee_nsteps", 5000)),
                    delta=float(delta_px),
                    burnin_frac=float(phot_cfg.get("emcee_burnin_frac", 0.3)),
                    thin=int(phot_cfg.get("emcee_thin", 10)),
                    adaptive_tau_target=int(phot_cfg.get("emcee_adaptive_tau_target", 50)),
                    min_autocorr_N=int(phot_cfg.get("emcee_min_autocorr_N", 100)),
                    batch_steps=int(phot_cfg.get("emcee_batch_steps", 100)),
                    jitter_scale=float(phot_cfg.get("emcee_jitter_scale", 0.01)),
                    use_nddata_uncertainty=True,
                    gain=float(resolve_gain_e_per_adu(None, input_yaml)),
                    readnoise=float(input_yaml.get("read_noise", 0.0)),
                    background_rms=rms_stamp,
                )

                fitted = fitter(
                    model,
                    gx,
                    gy,
                    stamp,
                    use_nddata_uncertainty=True,
                )

                try:
                    pnames = list(fitted.param_names)
                    i_flux = pnames.index("flux")
                    flux_hat = float(fitted.parameters[i_flux])
                    flux_err = float(getattr(fitted, "stds", np.full_like(fitted.parameters, np.nan))[i_flux])
                    if np.isfinite(flux_hat) and np.isfinite(flux_err) and flux_err > 0:
                        snr_val = flux_hat / flux_err
                except Exception:
                    snr_val = np.nan
            except Exception:
                snr_val = np.nan

        if not np.isfinite(snr_val):
            # Fall back to aperture-measured SNR.
            try:
                snr_val = float(mres["SNR"].iloc[0])
            except Exception:
                return False, beta_p, np.nan

        # For PSF / EMCEE methods, use PSF-fit (or MCMC) SNR for the gate; AP uses aperture flux/SNR.
        if method in ("PSF", "EMCEE"):
            recovered_flux = flux_hat if np.isfinite(flux_hat) else np.nan
        else:
            recovered_flux = float(mres["flux_AP"].iloc[0])

        if method in ("PSF", "EMCEE"):
            effective_snr_limit = float(snr_limit) if snr_limit is not None else 3.0
            det_snr = np.isfinite(snr_val) and (snr_val >= effective_snr_limit)
            return det_snr, beta_p, recovered_flux
        else:
            # AP method: use beta detection, optionally with SNR gate if snr_limit is set
            if snr_limit is not None:
                det_snr = np.isfinite(snr_val) and (snr_val >= float(snr_limit))
                return (det_beta and det_snr), beta_p, recovered_flux
            else:
                return det_beta, beta_p, recovered_flux

    except Exception:
        return False, 0.0, np.nan


# ===========================================================================
# limits class
# ===========================================================================


class Limits:
    """
    Compute limiting magnitudes for a single astronomical image frame using:

    * PSF injection / recovery           (getInjectedLimit)
    """

    def __init__(self, input_yaml: dict, catalog=None):
        """
        Parameters
        ----------
        input_yaml : dict
            Pipeline configuration loaded from YAML.  Expected keys include
            ``fwhm``, ``scale``, ``gain``, ``exposure_time``, ``target_x_pix``,
            ``target_y_pix``, ``photometry.aperture_radius``,
            ``limiting_magnitude.inject_source_location``, ``fpath``.
        catalog : pd.DataFrame, optional
            Source catalog for plotting comparison in injection recovery.
        """
        self.input_yaml = input_yaml
        self.catalog = catalog

        # Optional RNG seed for reproducible limiting-magnitude experiments.
        seed = self.input_yaml.get("rng_seed", None)
        self._rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

    # -----------------------------------------------------------------------
    # Cutout helper
    # -----------------------------------------------------------------------

    def get_cutout(
        self, image: np.ndarray, position=None, *, scale_override: float | None = None
    ) -> np.ndarray | None:
        """
        Extract a square cutout centred on the target (or supplied) position.

        The half-size of the cutout is ``ceil(inject_source_location * fwhm + scale)``.
        If ``scale_override`` is provided, it is used instead of ``input_yaml['scale']``.

        Parameters
        ----------
        image : ndarray  - full science frame
        position : (x, y) tuple, optional
            Pixel coordinates for the cutout centre.  Defaults to
            ``target_x_pix`` / ``target_y_pix`` from config.

        Returns
        -------
        ndarray or None
        """
        logger = logging.getLogger(__name__)
        try:
            if position is None:
                tx = self.input_yaml["target_x_pix"]
                ty = self.input_yaml["target_y_pix"]
            else:
                tx, ty = position
            if not (np.isfinite(tx) and np.isfinite(ty)):
                logger.warning(
                    "getCutout: target position (%.2f, %.2f) is not finite; skipping cutout.",
                    tx,
                    ty,
                )
                return None

            fwhm = self.input_yaml["fwhm"]
            scale = (
                float(scale_override)
                if scale_override is not None
                else float(self.input_yaml["scale"])
            )
            location_fwhm_mult = self.input_yaml["limiting_magnitude"]["inject_source_location"]

            half = int(np.ceil(location_fwhm_mult * fwhm + scale))
            # If `image` is already a cutout (e.g. the shared target cutout),
            # the configured full-frame scale can request a region larger than
            # the provided array. Clamp so Cutout2D never expands beyond bounds.
            try:
                ny, nx = int(image.shape[0]), int(image.shape[1])
                max_half = int(max(1, min(ny, nx) // 2 - 1))
                half = int(min(max_half, max(1, half)))
            except Exception:
                half = int(max(1, half))
            # Allow partial cutouts when the requested box exceeds the image bounds.
            # This matters when callers pass an already-extracted local cutout into
            # limiting-magnitude routines (target_cutout); in that case full-frame
            # scale parameters can request a larger region than the provided array.
            cutout_obj = Cutout2D(
                image,
                position=(tx, ty),
                size=(2 * half, 2 * half),
                mode="partial",
                fill_value=np.nan,
            )
            data = cutout_obj.data.copy()
            # True position of (tx, ty) in cutout-local pixel coordinates.
            # position_cutout is (x, y) = (col, row) in cutout frame.
            true_cx = float(cutout_obj.position_cutout[0])
            true_cy = float(cutout_obj.position_cutout[1])
            # Replace NaN border pixels (from partial cutout) with the median
            # of valid pixels so aperture annuli do not produce NaN sky estimates.
            # This preserves the original data where valid and avoids NaN
            # propagation into photometry.
            if np.any(~np.isfinite(data)):
                valid_median = np.nanmedian(data)
                if not np.isfinite(valid_median):
                    valid_median = 0.0
                data = np.where(np.isfinite(data), data, valid_median)
            return data, true_cx, true_cy

        except Exception as exc:
            logger.debug(f"getCutout failed: {exc}")
            return None, np.nan, np.nan

    # -----------------------------------------------------------------------
    # PSF helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _downsample_psf(psf_os: np.ndarray, oversampling: int) -> np.ndarray:
        """
        Flux-conserving downsampling of an oversampled PSF.

        Delegates to module-level ``_downsample_psf_flux_conserving`` (single
        implementation used by injection rendering and diagnostics).
        """
        return _downsample_psf_flux_conserving(psf_os, oversampling)

    # -----------------------------------------------------------------------
    # PSF injection / recovery limiting magnitude
    # -----------------------------------------------------------------------

    def get_injected_limit(
        self,
        full_image: np.ndarray,
        position,
        epsf_model=None,
        initialGuess: float = -5.0,
        detection_limit: float | None = None,
        detection_cutoff: float | None = None,
        plot: bool = True,
        background_rms: np.ndarray = None,
        subtraction_ready: bool = False,
        zeropoint: float = None,
        n_jobs: int = None,
        image_zeropoint: dict = None,
    ) -> float:
        """
        Bracket-and-bisect search for the limiting magnitude by injecting
        artificial PSF sources and measuring their recoverability.

        A *single* ProcessPoolExecutor is created for the entire search and
        reused by the nested ``run_trials_at_mag`` helper - avoiding the
        ~40-60 Pool creations that the original code performed.

        Parameters
        ----------
        full_image      : full 2-D science image (pre-subtraction)
        position        : (x, y) target pixel coords used to centre the cutout
        epsf_model      : photutils ePSF model
        initialGuess    : starting instrumental magnitude for the bracket
        detection_limit : optional S/N detection threshold. If set, a trial is a detection only when (S/N >= detection_limit) AND (beta >= detection_cutoff). If null, beta-only detections are used.
        detection_cutoff: beta threshold for an injected source to count as detected (default 0.5; beta is computed with n=3 so beta=0.5 ~ S/N~3)
        plot            : save diagnostic completeness PDF
        background_rms  : full-frame RMS map (optional)
        subtraction_ready: unused placeholder
        zeropoint       : adds an apparent-magnitude axis to the plot
        n_jobs          : worker processes; None defaults to 1 (serial)

        Returns
        -------
        float - limiting instrumental magnitude, or np.nan on failure
        """
        logger = logging.getLogger(__name__)
        start_time = time.time()

        try:
            # Always write diagnostic completeness plots for limiting-magnitude runs.
            # This keeps behavior consistent and avoids "missing plot" surprises.
            plot = True

            lim_cfg = self.input_yaml.get("limiting_magnitude") or {}
            if detection_cutoff is None:
                detection_cutoff = float(
                    lim_cfg.get("beta_limit", 0.5)
                )
            # Convert beta threshold to an equivalent k-sigma flux threshold for the
            # beta formalism (using n=3). This is only a rough guide.
            k_sigma = None
            try:
                k_sigma = float(flux_upper_limit(n=3.0, sigma=1.0, beta_p=float(detection_cutoff)))
            except Exception:
                k_sigma = None
            logger.info(
                "Injected limiting magnitude: beta_limit=%.3f (~%s sigma; beta_aperture n=%g), "
                "snr_gate=%s (limiting_magnitude.detection_limit)",
                float(detection_cutoff),
                ("%.2f" % float(k_sigma)) if k_sigma is not None else "unknown",
                float(BETA_APERTURE_SIGMA_N),
                "off" if detection_limit is None else str(detection_limit),
            )
            # =================================================================
            # Validation
            # =================================================================
            if epsf_model is None:
                logger.info("No PSF model - skipping limiting magnitude")
                return np.nan

            if initialGuess is None or not np.isfinite(float(initialGuess)):
                initialGuess = -5.0

            # =================================================================
            # Extract cutouts (science + RMS).
            #
            # If `precutout=True`, the caller has already provided the cutout
            # to operate on (e.g. the shared local target cutout). In that case
            # we must NOT call get_cutout() again, since it uses full-frame scale
            # parameters and can shrink/pad the provided array.
            # =================================================================
            lim_cfg = self.input_yaml.get("limiting_magnitude") or {}
            growth_factor = float(lim_cfg.get("scale_growth_factor", 1.5))
            growth_max = int(lim_cfg.get("scale_growth_max_px", 600))

            base_scale = float(self.input_yaml.get("scale", 0))
            scale_used = base_scale

            # Minimum half-size: injection radius + aperture_radius + annulus_width + 5px margin.
            # annulus_width is typically 1.5 * aperture_radius from Aperture defaults.
            phot_cfg_pre = self.input_yaml.get("photometry", {})
            fwhm_pre = float(self.input_yaml.get("fwhm", 3.0))
            ap_r_pre = float(phot_cfg_pre.get("aperture_radius", fwhm_pre))
            r_base_pre = float((self.input_yaml.get("limiting_magnitude") or {}).get(
                "inject_source_location", 3.0)) * fwhm_pre
            annulus_w_pre = float(phot_cfg_pre.get("annulus_width", 1.5 * ap_r_pre))
            min_half_needed = r_base_pre + ap_r_pre + annulus_w_pre + 5.0
            location_fwhm_mult_pre = float((self.input_yaml.get("limiting_magnitude") or {}).get(
                "inject_source_location", 3.0))
            current_half = float(np.ceil(location_fwhm_mult_pre * fwhm_pre + base_scale))
            if current_half < min_half_needed:
                scale_used = max(base_scale, min_half_needed - location_fwhm_mult_pre * fwhm_pre)
                logger.info(
                    "Auto-increasing scale from %.1f to %.1f px to fit sky annulus "
                    "(need half=%.1f, had half=%.1f)",
                    base_scale, scale_used, min_half_needed, current_half,
                )

            # Capture original values in immutable local variables before defining closure
            _orig_frame = np.asarray(full_image, dtype=float)  # full science frame
            _orig_position = [float(position[0]), float(position[1])]  # full-frame coords (target position)
            _orig_background_rms = background_rms  # full-frame RMS map

            def _extract_cutouts(scale_px: float):
                result = self.get_cutout(
                    image=_orig_frame, position=_orig_position, scale_override=scale_px
                )
                if result is None:
                    return None, None, np.nan, np.nan
                img, cx, cy = result
                rms = None
                if _orig_background_rms is not None:
                    rms_result = self.get_cutout(
                        image=_orig_background_rms,
                        position=_orig_position,
                        scale_override=scale_px,
                    )
                    if rms_result is not None:
                        rms = np.abs(np.asarray(rms_result[0], dtype=float))
                return img, rms, cx, cy

            # Always create cutout internally from full image
            cutout_img, cutout_rms, cutout_cx, cutout_cy = _extract_cutouts(scale_used)
            if cutout_img is None and growth_factor > 1:
                # Try a slightly larger cutout if the initial one fails.
                for _ in range(2):
                    scale_used = min(
                        float(growth_max),
                        max(scale_used + 1.0, scale_used * growth_factor),
                    )
                    cutout_img, cutout_rms, cutout_cx, cutout_cy = _extract_cutouts(scale_used)
                    if cutout_img is not None:
                        break

            if cutout_img is None or not np.isfinite(cutout_cx) or not np.isfinite(cutout_cy):
                logger.warning("getCutout returned None or invalid centre; aborting")
                return np.nan

            cutout = cutout_img
            background_rms = cutout_rms

            # Target is at the true centre returned by get_cutout (accounts for partial cutouts)
            H, W = cutout.shape
            logger.info(
                "Cutout extracted: shape=(%d, %d), true_target_centre=(%.2f, %.2f), "
                "geometric_centre=(%.2f, %.2f), offset=(%.2f, %.2f px)",
                H, W,
                float(cutout_cx), float(cutout_cy),
                float((W - 1) / 2.0), float((H - 1) / 2.0),
                float(cutout_cx - (W - 1) / 2.0),
                float(cutout_cy - (H - 1) / 2.0),
            )

            # Calculate optimum aperture radius if not already set
            fwhm = float(self.input_yaml.get("fwhm", 3.0))
            phot_cfg = self.input_yaml.get("photometry", {})
            configured_radius = float(phot_cfg.get("aperture_radius", fwhm))

            # Use local copy of config to avoid mutating shared state
            import copy
            local_input_yaml = copy.deepcopy(self.input_yaml)
            # Canonical exposure and gain for every Aperture call and for flux_for_mag
            # (must match ``Aperture.measure`` resolution on this frame).
            _exp_canon = _effective_exposure_seconds(local_input_yaml)
            _gain_canon = resolve_gain_e_per_adu(None, local_input_yaml)
            local_input_yaml["exposure_time"] = _exp_canon
            local_input_yaml["gain"] = _gain_canon
            logger.info(
                "Limiting magnitude: using exposure_time=%.5g s, gain=%.5g e/ADU for "
                "aperture photometry and injection flux calibration",
                _exp_canon,
                _gain_canon,
            )

            def _exclude_target_overlap(df: pd.DataFrame,
                                         target_cx: float,
                                         target_cy: float,
                                         exclusion_r: float) -> pd.DataFrame:
                """
                Remove any injection sites whose aperture could overlap with the
                transient aperture at (target_cx, target_cy).

                exclusion_r = aperture_radius (transient) + aperture_radius (injected)
                              + fwhm (PSF wing buffer)
                Sites within exclusion_r pixels of the target are removed.
                """
                if len(df) == 0:
                    return df
                dx = df["x_pix"].to_numpy() - target_cx
                dy = df["y_pix"].to_numpy() - target_cy
                dist = np.sqrt(dx**2 + dy**2)
                return df[dist >= exclusion_r].copy()

            def _filter_edge_clearance(df: pd.DataFrame,
                                      cutout_w: float,
                                      cutout_h: float,
                                      margin: float) -> pd.DataFrame:
                """Remove sites too close to cutout edges."""
                if len(df) == 0:
                    return df
                mask = (
                    (df["x_pix"] >= margin) &
                    (df["x_pix"] <= cutout_w - margin) &
                    (df["y_pix"] >= margin) &
                    (df["y_pix"] <= cutout_h - margin)
                )
                return df[mask].copy()

            # If aperture_radius equals fwhm (default fallback), try to calculate optimum
            if configured_radius == fwhm:
                try:
                    # Detect sources in the cutout to use for optimum radius calculation
                    from photutils.detection import DAOStarFinder
                    daofind = DAOStarFinder(fwhm=fwhm, threshold=5.0 * np.nanstd(cutout))
                    sources = daofind(cutout)
                    if sources is not None and len(sources) >= 5:
                        sources_df = pd.DataFrame({
                            "x_pix": sources["xcentroid"],
                            "y_pix": sources["ycentroid"]
                        })
                        ap = Aperture(input_yaml=local_input_yaml, image=cutout)
                        _, optimum_radius_fwhm, _ = ap.measure_optimum_radius(
                            sources=sources_df,
                            plot=False,
                            background_rms=background_rms,
                            n_jobs=1,
                        )
                        optimum_radius_pixels = optimum_radius_fwhm * fwhm
                        # Update local config with calculated optimum radius
                        local_input_yaml["photometry"]["aperture_radius"] = optimum_radius_pixels
                        logger.info(f"Calculated optimum aperture radius: {optimum_radius_pixels:.2f} pixels ({optimum_radius_fwhm:.2f} FWHM)")
                    else:
                        logger.info(f"Insufficient sources for optimum radius, using configured: {configured_radius:.2f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate optimum aperture radius: {e}, using configured")
            
            # Get final aperture radius from local_input_yaml
            phot_cfg_local = local_input_yaml.get("photometry", {})
            aperture_radius_local = float(phot_cfg_local.get("aperture_radius", fwhm))
            
            # Exclusion zone: two aperture diameters + one FWHM buffer
            # Ensures zero overlap between transient aperture and injection aperture,
            # plus a PSF-wing buffer so transient flux cannot bias the injected recovery.
            target_exclusion_r = 2.0 * aperture_radius_local + fwhm
            logger.info(
                "Target exclusion zone: r=%.1f px (2*ap_r=%.1f + fwhm=%.1f)",
                target_exclusion_r, 2.0 * aperture_radius_local, fwhm
            )

            # Define injection radii early (needed for initial guess)
            fwhm_px = float(self.input_yaml.get("fwhm", 3.0))
            r_min = float(lim_cfg.get("inject_min_radius_fwhm", 2.0)) * fwhm_px
            r_max = float(lim_cfg.get("inject_max_radius_fwhm", 6.0)) * fwhm_px
            r_base = float(lim_cfg.get("inject_source_location", 3.0)) * fwhm_px
            
            # Enforce r_min >= target_exclusion_r so sites are never generated inside exclusion zone
            r_min = max(r_min, target_exclusion_r)
            logger.info(
                "Injection radii: r_min=%.1f px (excl_zone=%.1f), r_base=%.1f, r_max=%.1f",
                r_min, target_exclusion_r, r_base, r_max,
            )
            
            # Calculate minimum cutout size needed for all photometry operations
            # Need space for: target exclusion zone + edge clearance + injection radius
            edge_margin = max(3.0 * fwhm_px, 2.0 * aperture_radius_local)
            min_half_size = target_exclusion_r + r_max + edge_margin
            min_cutout_size = 2 * min_half_size
            
            # Current scale-based cutout size
            location_fwhm_mult = float(lim_cfg.get("inject_source_location", 3.0))
            current_half_size = int(np.ceil(location_fwhm_mult * fwhm_px + base_scale))
            current_cutout_size = 2 * current_half_size

            # Update scale if current cutout is too small
            if current_cutout_size < min_cutout_size:
                # Calculate the scale needed to achieve minimum cutout size
                needed_half_size = min_half_size
                needed_scale = needed_half_size - location_fwhm_mult * fwhm_px
                # Ensure scale is at least the base scale and is reasonable
                new_scale = max(base_scale, needed_scale, 10.0)  # minimum 10px scale
                
                logger.info(
                    "Cutout size update: current=%dx%d (half=%.1f), needed=%dx%d (half=%.1f), "
                    "scale: %.1f -> %.1f",
                    current_cutout_size, current_cutout_size, current_half_size,
                    min_cutout_size, min_cutout_size, min_half_size,
                    base_scale, new_scale
                )
                
                # Update the scale used for cutout extraction
                scale_used = new_scale
            else:
                logger.info(
                    "Cutout size sufficient: %dx%d (half=%.1f) >= needed %dx%d (half=%.1f)",
                    current_cutout_size, current_cutout_size, current_half_size,
                    min_cutout_size, min_cutout_size, min_half_size
                )

            # Data-driven initial guess from instrumental magnitudes measured
            # on an annulus around the target location.
            try:
                use_annulus_guess = bool(np.isclose(float(initialGuess), -5.0))
            except Exception:
                use_annulus_guess = True
            if use_annulus_guess:
                try:
                    # Prefer an S/N-based guess tied to the local background scatter:
                    # flux_guess ~= k * sigma_sky * sqrt(Npix).
                    # This is more stable than using a percentile of random annulus mags,
                    # especially when the background mean is shifted (bias) but scatter is unchanged.
                    # Start at 10 sigma to ensure we're in the detectable regime before searching fainter.
                    try:
                        guess_k = float(lim_cfg.get("initial_guess_sigma_mult", 10.0))
                    except Exception:
                        guess_k = 10.0
                    probe_n = int((lim_cfg.get("initial_guess_n_samples", 24)))
                    probe_n = max(8, min(probe_n, 72))
                    # Use r_base for probe radius
                    probe_r = r_base
                    pts = points_in_circum(probe_r, center=[cutout_cx, cutout_cy], n=probe_n)
                    probe_df = pd.DataFrame(
                        {"x_pix": [p[0] for p in pts], "y_pix": [p[1] for p in pts]}
                    )
                    probe_ap = Aperture(input_yaml=local_input_yaml, image=cutout)
                    probe_meas = probe_ap.measure(
                        sources=probe_df,
                        plot=False,
                        background_rms=background_rms,
                        verbose=0,
                    )
                    sigma = np.asarray(probe_meas.get("noiseSky", np.array([])), dtype=float)
                    area = np.asarray(probe_meas.get("area", np.array([])), dtype=float)
                    ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(area) & (area > 0)
                    if int(np.count_nonzero(ok)) >= 5 and np.isfinite(guess_k) and guess_k > 0:
                        sigma_med = float(np.nanmedian(sigma[ok]))
                        area_med = float(np.nanmedian(area[ok]))
                        flux_guess = float(guess_k) * sigma_med * float(np.sqrt(area_med))
                        if np.isfinite(flux_guess) and flux_guess > 0:
                            initialGuess = float(mag(flux_guess))
                            logger.info(
                                "Injected limiting magnitude: initial guess from %.1f*sigma*sqrt(Npix) = %.2f "
                                "(sigma=%.3g, Npix=%.1f, N=%d)",
                                float(guess_k),
                                float(initialGuess),
                                float(sigma_med),
                                float(area_med),
                                int(np.count_nonzero(ok)),
                            )
                    else:
                        logger.info(
                            "Injected limiting magnitude: sigma-based initial guess unavailable (N=%d); using %.2f",
                            int(np.count_nonzero(ok)),
                            float(initialGuess),
                        )
                except Exception as exc:
                    logger.debug(
                        "Injected limiting magnitude: annulus-based initial guess failed: %s",
                        exc,
                    )
            
            # Note: Initial guess flux will be logged after flux_for_mag is defined

            # Pixel grids for PSF evaluation will be constructed after final cutout shape is known
            # (moved here to avoid shape mismatch if grow loop changes cutout size)

            # =================================================================
            # Oversampling - compute grids ONCE
            # =================================================================
            raw_os = getattr(epsf_model, "oversampling", 1)
            if isinstance(raw_os, (list, tuple, np.ndarray)):
                oversampling = int(raw_os[0])
            elif np.isscalar(raw_os) and raw_os > 1:
                oversampling = int(raw_os)
            else:
                oversampling = 1

            # Grids will be constructed after final cutout shape is known

            # =================================================================
            # PSF calibration: flux=1 -> what instrumental magnitude?
            # =================================================================
            # Use true cutout centre for PSF evaluation (grid is cutout-sized)
            # cutout_cx, cutout_cy are the true target position from Cutout2D.position_cutout
            cx, cy = cutout_cx, cutout_cy
            # Must match ``_injection_worker`` rendering (especially oversampling > 1).
            psf_unit = _render_epsf_on_cutout(
                epsf_model, H, W, float(cx), float(cy), 1.0, oversampling
            )

            psf_ap = Aperture(input_yaml=local_input_yaml, image=psf_unit)
            psf_meas = psf_ap.measure(
                pd.DataFrame({"x_pix": [cutout_cx], "y_pix": [cutout_cy]}),  # Use same centre as PSF evaluation
                plot=False,
                verbose=0,
            )
            F_ref = float(psf_meas["flux_AP"].iloc[0])  # e-/s (same as mag/PSF pipeline)
            counts_ref = float(psf_meas["counts_AP"].iloc[0])  # integrated e- in frame (see Aperture.measure)

            # Guard: check if calibration failed
            if not (np.isfinite(counts_ref) and counts_ref > 0
                    and np.isfinite(F_ref) and F_ref > 0):
                logger.error(
                    f"PSF calibration failed: counts_ref={counts_ref:.4e}, F_ref={F_ref:.4e} "
                    f"(both must be finite and positive). "
                    f"PSF sum={np.sum(psf_unit):.4e}, shape={psf_unit.shape}, oversampling={oversampling}"
                )
                return np.nan

            exposure_time = float(local_input_yaml["exposure_time"])

            # Memoised so repeated calls at the same magnitude are free.
            def flux_for_mag(m: float) -> float:
                """Return ePSF flux parameter to inject for instrumental magnitude m."""
                return _flux_for_mag_cached(m, counts_ref, exposure_time)

            # Sanity: mag(counts_ref/exposure) == mag(F_ref); flux_for_mag at that m should return 1.0.
            mag_at_unit_flux = float(mag(float(F_ref)))
            f_roundtrip = float(flux_for_mag(mag_at_unit_flux))
            logger.info(
                "PSF injection calibration: counts_ref=%.6g (e- in aperture @ model flux=1), "
                "mag(F=1)=%.5f, flux_for_mag(mag(F=1))=%.6f (expect 1.0), "
                "flux_for_mag(initialGuess)=%.6g",
                float(counts_ref),
                mag_at_unit_flux,
                f_roundtrip,
                float(flux_for_mag(float(initialGuess))),
            )
            if np.isfinite(f_roundtrip) and abs(f_roundtrip - 1.0) > 0.02:
                logger.warning(
                    "PSF flux calibration round-trip differs from 1.0 (got %.6f). "
                    "Check exposure_time, oversampling, and that the science cutout "
                    "uses the same ADU/gain convention as Aperture.measure.",
                    f_roundtrip,
                )

            # =================================================================
            # Choose quiet injection sites
            # =================================================================
            DETECTION_BETA_THRESH = detection_cutoff
            snr_limit = float(detection_limit) if detection_limit is not None else None
            recovery_method = str(lim_cfg.get("recovery_method", "PSF")).strip().upper()
            # "AUTO" is resolved in main.py to AP vs PSF from do_aperture_ONLY; if unset, prefer PSF.
            if recovery_method in {"AUTO", "DEFAULT", "MATCH_TRANSIENT", "MATCH_TARGET"}:
                logger.warning(
                    "limiting_magnitude.recovery_method=%s was not pre-resolved; using PSF. "
                    "Set recovery_method explicitly or run from main (auto).",
                    recovery_method,
                )
                recovery_method = "PSF"
            elif recovery_method in {"MCMC", "EMCEE"}:
                recovery_method = "EMCEE"
            elif recovery_method in {"PSF", "AP"}:
                pass
            else:
                logger.warning(
                    "Unknown recovery_method %r; using PSF.",
                    recovery_method,
                )
                recovery_method = "PSF"
            completeness_target = float(lim_cfg.get("completeness_target", 0.5))
            completeness_target = max(0.0, min(1.0, completeness_target))
            if not np.isfinite(completeness_target):
                completeness_target = 0.5

            # Option to disable quiet site selection for more representative limiting magnitude
            use_quiet_sites = bool(lim_cfg.get("inject_use_quiet_sites", True))

            logger.info(
                "Injected limiting magnitude: recovery_method=%s, completeness_target=%.2f, use_quiet_sites=%s",
                str(recovery_method),
                float(completeness_target),
                str(use_quiet_sites),
            )
            completeness_solver = str(lim_cfg.get("completeness_solver", "bisect")).strip().lower()
            if completeness_solver not in {"bisect", "logistic_emcee"}:
                completeness_solver = "bisect"

            # emcee recovery should run serial (each trial is expensive and models
            # are not guaranteed to pickle cleanly across platforms).
            if recovery_method == "EMCEE":
                n_jobs = 1
            # Beta is computed using the canonical n=3 aperture formalism so that
            # beta thresholds are comparable across runs (beta=0.5 ~ S/N~3).
            beta_n = float(BETA_APERTURE_SIGMA_N)
            # Monte Carlo settings for injection/recovery.
            # Too few sites/jitters makes the completeness curve look jagged.
            sourceNum = int(lim_cfg.get("injection_n_sites", 30))
            sourceNum = max(8, min(sourceNum, 200))
            redo_default = int(lim_cfg.get("injection_jitter_repetitions", 5))
            redo_default = max(1, min(redo_default, 20))
            distance_factor = 1.0
            injection_df = pd.DataFrame()

            # Injection strategy (radii already defined above for initial guess)
            inj_strategy = str(lim_cfg.get("injection_strategy", "ring_quiet")).strip().lower()
            if inj_strategy not in {"ring_quiet", "annulus_random"}:
                inj_strategy = "ring_quiet"

            for attempt in range(3):
                inj_dist = float(np.clip(r_base * distance_factor, r_min, r_max))
                if inj_strategy == "annulus_random":
                    # Sample positions uniformly in area within the local annulus.
                    # (r ~ sqrt(u)) and theta ~ U[0, 2pi].
                    # Add safety margin for sub-pixel jitter
                    r_min_with_jitter = r_min + 1.0  # 1px safety margin for jitter
                    theta = self._rng.random(sourceNum) * (2.0 * np.pi)
                    rr = np.sqrt(self._rng.random(sourceNum)) * (r_max - r_min_with_jitter) + r_min_with_jitter
                    xran = cutout_cx + rr * np.cos(theta)
                    yran = cutout_cy + rr * np.sin(theta)
                    df = pd.DataFrame({"x_pix": xran, "y_pix": yran})
                else:
                    pts = points_in_circum(inj_dist, center=[cutout_cx, cutout_cy], n=sourceNum)
                    xran = [p[0] for p in pts]
                    yran = [p[1] for p in pts]
                    df = pd.DataFrame({"x_pix": xran, "y_pix": yran})

                ini_ap = Aperture(input_yaml=local_input_yaml, image=cutout)
                df = ini_ap.measure(
                    sources=df,
                    plot=False,
                    background_rms=cutout_rms,
                    verbose=0,
                )
                p_det = _compute_p_det(df, beta_n)
                
                # Filter: remove locations with existing S/N > 3 (bright sources)
                snr_existing = df.get("SNR", pd.Series(np.zeros(len(df)), index=df.index))
                snr_mask = snr_existing <= 3.0
                n_snr_filtered = (~snr_mask).sum()
                if n_snr_filtered > 0:
                    logger.info(
                        "Filtered %d injection sites with existing S/N > 3 (bright sources)",
                        int(n_snr_filtered),
                    )
                
                # Diagnostic logging for quiet point selection
                logger.info(
                    "Quiet point selection: n_sites=%d, p_det_range=[%.4f, %.4f], thresh=%.4f, "
                    "flux_AP_range=[%.4e, %.4e], noiseSky_range=[%.4e, %.4e]",
                    len(p_det),
                    float(np.nanmin(p_det)) if len(p_det) > 0 else np.nan,
                    float(np.nanmax(p_det)) if len(p_det) > 0 else np.nan,
                    float(DETECTION_BETA_THRESH),
                    float(np.nanmin(df["flux_AP"])) if "flux_AP" in df.columns and len(df) > 0 else np.nan,
                    float(np.nanmax(df["flux_AP"])) if "flux_AP" in df.columns and len(df) > 0 else np.nan,
                    float(np.nanmin(df["noiseSky"])) if "noiseSky" in df.columns and len(df) > 0 else np.nan,
                    float(np.nanmax(df["noiseSky"])) if "noiseSky" in df.columns and len(df) > 0 else np.nan,
                )
                # "Quiet" sites: choose positions with detection probability below
                # the same cutoff used later to define what counts as a detection.
                # Also apply S/N filter to avoid injecting on bright existing sources.
                # If disabled, use all sites for more representative limiting magnitude.
                if use_quiet_sites:
                    injection_df = df[(p_det < DETECTION_BETA_THRESH) & snr_mask].copy()
                else:
                    injection_df = df[snr_mask].copy()

                # Always apply exclusion zone regardless of use_quiet_sites
                # Target is at the true centre returned by get_cutout
                _target_cx = cutout_cx
                _target_cy = cutout_cy
                injection_df = _exclude_target_overlap(
                    injection_df, _target_cx, _target_cy, target_exclusion_r
                )

                # Add edge clearance to prevent injection sites from being too close to cutout boundaries
                injection_df = _filter_edge_clearance(
                    injection_df, W, H, edge_margin
                )

                if len(injection_df) > 0:
                    break
                # Expand outward within the allowed annulus before giving up.
                distance_factor = min(2.0, distance_factor * 1.5)
                logger.info(
                    "No quiet positions; retrying at larger radius (r=%.1f px)",
                    float(np.clip(r_base * distance_factor, r_min, r_max)),
                )

            # If still no quiet sites, grow the cutout scale and retry the quiet-site
            # search. This helps when the local cutout is dominated by host structure.
            if (
                len(injection_df) == 0
                and use_quiet_sites
                and growth_factor > 1
                and scale_used < float(growth_max)
            ):
                for _grow in range(3):
                    prev_scale = float(scale_used)
                    scale_used = min(
                        float(growth_max),
                        max(scale_used + 1.0, scale_used * growth_factor),
                    )
                    if float(scale_used) <= prev_scale + 1e-6:
                        break
                    logger.info(
                        "No quiet positions; growing cutout scale from %.1f to %.1f px and retrying",
                        prev_scale,
                        float(scale_used),
                    )
                    cutout_img, cutout_rms, cutout_cx, cutout_cy = _extract_cutouts(scale_used)
                    if cutout_img is None:
                        continue
                    cutout = cutout_img
                    background_rms = cutout_rms
                    # True centre is already returned by get_cutout via _extract_cutouts
                    H, W = cutout.shape

                    distance_factor = 1.0
                    injection_df = pd.DataFrame()
                    for attempt in range(3):
                        inj_dist = float(np.clip(r_base * distance_factor, r_min, r_max))
                        if inj_strategy == "annulus_random":
                            theta = self._rng.random(sourceNum) * (2.0 * np.pi)
                            r_min_with_jitter = r_min + 1.0  # 1px safety margin for jitter
                            rr = (
                                np.sqrt(self._rng.random(sourceNum)) * (r_max - r_min_with_jitter)
                                + r_min_with_jitter
                            )
                            xran = cutout_cx + rr * np.cos(theta)
                            yran = cutout_cy + rr * np.sin(theta)
                            df = pd.DataFrame({"x_pix": xran, "y_pix": yran})
                        else:
                            pts = points_in_circum(inj_dist, center=[cutout_cx, cutout_cy], n=sourceNum)
                            xran = [p[0] for p in pts]
                            yran = [p[1] for p in pts]
                            df = pd.DataFrame({"x_pix": xran, "y_pix": yran})

                        ini_ap = Aperture(input_yaml=local_input_yaml, image=cutout_img)
                        df = ini_ap.measure(
                            sources=df,
                            plot=False,
                            background_rms=cutout_rms,
                            verbose=0,
                        )
                        p_det = _compute_p_det(df, beta_n)
                        
                        # Filter: remove locations with existing S/N > 3 (bright sources)
                        snr_existing = df.get("SNR", pd.Series(np.zeros(len(df)), index=df.index))
                        snr_mask = snr_existing <= 3.0
                        
                        if use_quiet_sites:
                            injection_df = df[(p_det < DETECTION_BETA_THRESH) & snr_mask].copy()
                        else:
                            injection_df = df[snr_mask].copy()
                        
                        # Always apply exclusion zone
                        # Target is at the true centre returned by get_cutout
                        _target_cx_grow = cutout_cx
                        _target_cy_grow = cutout_cy
                        injection_df = _exclude_target_overlap(
                            injection_df,
                            _target_cx_grow,
                            _target_cy_grow,
                            target_exclusion_r,
                        )
                        
                        # Apply edge clearance for grown cutout
                        injection_df = _filter_edge_clearance(
                            injection_df, W, H, edge_margin
                        )
                        
                        if len(injection_df) > 0:
                            break
                        distance_factor = min(2.0, distance_factor * 1.5)
                    if len(injection_df) > 0:
                        break

            if len(injection_df) == 0:
                # Final fallback: still inject off-target (never at the transient position).
                logger.info(
                    "No quiet positions found in annulus; using off-target injection sites anyway (r=%.1f px).",
                    float(np.clip(r_base, r_min, r_max)),
                )
                inj_dist = float(np.clip(r_base, r_min, r_max))
                if inj_strategy == "annulus_random":
                    theta = self._rng.random(sourceNum) * (2.0 * np.pi)
                    rr = np.sqrt(self._rng.random(sourceNum)) * (r_max - r_min) + r_min
                    xran = cutout_cx + rr * np.cos(theta)
                    yran = cutout_cy + rr * np.sin(theta)
                    injection_df = pd.DataFrame({"x_pix": xran, "y_pix": yran})
                else:
                    pts = points_in_circum(inj_dist, center=[cutout_cx, cutout_cy], n=sourceNum)
                    injection_df = pd.DataFrame(
                        {"x_pix": [p[0] for p in pts], "y_pix": [p[1] for p in pts]}
                    )

                # Apply exclusion zone to fallback sites too
                injection_df = _exclude_target_overlap(
                    injection_df, cutout_cx, cutout_cy, target_exclusion_r
                )

                # Use current cutout dimensions for edge clearance
                H_fb, W_fb = cutout.shape  # always current
                # Apply edge clearance to fallback sites
                injection_df = _filter_edge_clearance(
                    injection_df, W_fb, H_fb, edge_margin
                )
                
                if len(injection_df) == 0:
                    logger.warning(
                        "All fallback sites within exclusion zone or edge clearance (r=%.1f px, margin=%.1f px); "
                        "increase inject_source_location or inject_min_radius_fwhm",
                        target_exclusion_r, edge_margin,
                    )
                    return np.nan

            if len(injection_df) > sourceNum:
                injection_df = injection_df.sample(sourceNum).reset_index(drop=True)

            H_final, W_final = cutout.shape

            # Recompute r_max and r_base after final cutout shape is known
            margin_r = float(np.ceil(fwhm_px))
            max_safe_r = min(
                cutout_cx - margin_r,
                W_final - 1 - cutout_cx - margin_r,
                cutout_cy - margin_r,
                H_final - 1 - cutout_cy - margin_r,
            )
            r_max_eff = min(r_max, max(r_min, float(max_safe_r)))
            r_base_eff = float(np.clip(r_base, r_min, r_max_eff))

            # injection_df is already built in cutout coordinates
            x_pix_arr = injection_df["x_pix"].to_numpy()
            y_pix_arr = injection_df["y_pix"].to_numpy()
            n_sites = len(injection_df)

            # Filter out-of-bounds sites instead of clipping to edge
            margin = float(np.ceil(fwhm))
            valid_mask = (
                (x_pix_arr >= margin) & (x_pix_arr <= W_final - 1 - margin) &
                (y_pix_arr >= margin) & (y_pix_arr <= H_final - 1 - margin)
            )
            if valid_mask.sum() < 3:
                logger.warning(
                    f"Too few injection sites within cutout bounds ({int(valid_mask.sum())}/{len(x_pix_arr)}); "
                    "consider increasing cutout size or reducing inject_max_radius_fwhm"
                )
            x_pix_arr = x_pix_arr[valid_mask]
            y_pix_arr = y_pix_arr[valid_mask]
            n_sites = len(x_pix_arr)
            if n_sites == 0:
                logger.warning("No valid injection sites after bounds filtering; aborting.")
                return np.nan

            # Default serial; cap workers to avoid HPC fork/resource limits.
            n_jobs = n_jobs if n_jobs is not None else 1
            n_jobs = max(1, min(n_jobs, 8))

            # =================================================================
            # Trial runner - captures the shared pool from the outer scope.
            #
            # KEY OPTIMISATION: the ProcessPoolExecutor is created once below
            # and referenced here.  The original code called Pool() inside
            # this closure on every single magnitude evaluation.
            # =================================================================

            def run_trials_at_mag(m: float, redo: int = None, pool=None, *, return_flags: bool = False):
                """
                Inject at *m* at all sites with *redo* sub-pixel jitter
                repetitions and return (mean_detection_rate, beta_array, recovered_flux_array[, det_flags]).
                """
                cache_key = f"{round(m, 4)}_{int(redo if redo is not None else redo_default)}"
                if cache_key in _trial_cache:
                    cached = _trial_cache[cache_key]
                    if return_flags:
                        return cached
                    return cached[:3]

                redo = int(redo_default if redo is None else redo)
                redo = max(1, min(redo, 50))
                # Per-magnitude deterministic seeding for reproducibility
                seed = int(abs(hash(round(m, 4)))) % (2**31)
                local_rng = np.random.default_rng(seed)
                dx = local_rng.random((n_sites, redo)) - 0.5
                dy = local_rng.random((n_sites, redo)) - 0.5
                F = flux_for_mag(m)

                # Vectorized jitter: build arrays directly instead of nested list comprehension
                k_idx, j_idx = np.divmod(np.arange(n_sites * redo), redo)
                x_inj_all = x_pix_arr[k_idx] + dx[k_idx, j_idx]
                y_inj_all = y_pix_arr[k_idx] + dy[k_idx, j_idx]

                tasks = [
                    (
                        x_inj_all[n],
                        y_inj_all[n],
                        F,
                        cutout,
                        oversampling,
                        epsf_model,
                        local_input_yaml,
                        background_rms,
                        snr_limit,
                        beta_n,
                        DETECTION_BETA_THRESH,
                        recovery_method,
                    )
                    for n in range(len(x_inj_all))
                ]

                if pool is not None:
                    results = list(pool.map(_injection_worker, tasks))
                else:
                    results = [_injection_worker(t) for t in tasks]

                det_flags = np.array([r[0] for r in results], dtype=bool)
                betas = np.array([r[1] for r in results], dtype=float)
                recovered_fluxes = np.array([r[2] for r in results], dtype=float)

                # DIAGNOSTIC: log detection rate at this magnitude
                F = flux_for_mag(m)
                logger.info(
                    "run_trials_at_mag(m=%.3f): F=%.4e, det_rate=%.3f (%d/%d)",
                    m, F, float(det_flags.mean()), int(det_flags.sum()), len(det_flags)
                )

                result = (float(det_flags.mean()), betas, recovered_fluxes)
                result_with_flags = (float(det_flags.mean()), betas, recovered_fluxes, det_flags)
                _trial_cache[cache_key] = result_with_flags
                if return_flags:
                    return result_with_flags
                return result

            # =================================================================
            # Single ProcessPoolExecutor for the ENTIRE search (or serial if n_jobs==1)
            # =================================================================
            inject_lmag = np.nan
            bracket_steps: list[tuple] = []
            bisect_steps: list[tuple] = []
            _trial_cache: dict[str, tuple] = {}

            with _pool_or_serial(n_jobs) as pool:

                # ---- Bracket phase ------------------------------------------
                step = 0.5
                max_steps = 30
                # Check if injection recovery plot is enabled
                plot_injection_recovery = (self.input_yaml.get("limiting_magnitude") or {}).get("plot_injection_recovery", False)

                if plot_injection_recovery:
                    # Start at artificially bright magnitude for visualization in the plot
                    m_bright = -10.0  # Very bright starting point
                    c_bright, _, f_bright = run_trials_at_mag(m_bright, pool=pool)
                    going_faint = c_bright >= completeness_target
                    m_faint, c_faint = m_bright, c_bright
                    bracket_steps.append((m_bright, c_bright, np.nanmedian(f_bright)))
                else:
                    # Skip the -10.0 trial; start from data-driven initial guess instead
                    m_bright = float(initialGuess) if np.isfinite(initialGuess) else -5.0
                    c_bright, _, f_bright = run_trials_at_mag(m_bright, pool=pool)
                    going_faint = c_bright >= completeness_target
                    m_faint, c_faint = m_bright, c_bright
                    bracket_steps.append((m_bright, c_bright, np.nanmedian(f_bright)))

                for _ in range(max_steps):
                    m_test = m_faint + step if going_faint else m_bright - step
                    c_test, _, f_test = run_trials_at_mag(m_test, pool=pool)
                    bracket_steps.append((m_test, c_test, np.nanmedian(f_test)))

                    if going_faint:
                        m_faint, c_faint = m_test, c_test
                        # Continue stepping fainter until we find undetected endpoint
                        if c_faint < completeness_target:
                            break
                    else:
                        # When stepping brighter (looking for detected end)
                        # Keep m_faint fixed (this is the undetected endpoint)
                        if c_test >= completeness_target:
                            # Found detected end
                            m_bright, c_bright = m_test, c_test
                            break
                        # else: keep stepping brighter; do NOT break

                bracketed = (c_bright >= completeness_target) and (
                    c_faint < completeness_target
                )
                if not bracketed:
                    # Retry with a small grid of initial guesses around the provided
                    # starting point (avoids the removed probabilistic-limit path).
                    for guess in (
                        float(initialGuess) - 2.0,
                        float(initialGuess) - 1.0,
                        float(initialGuess),
                        float(initialGuess) + 1.0,
                        float(initialGuess) + 2.0,
                    ):
                        m_bright = float(guess)
                        c_bright, _, f_bright = run_trials_at_mag(m_bright, pool=pool)
                        going_faint = c_bright >= completeness_target
                        m_faint, c_faint = m_bright, c_bright
                        bracket_steps.append((m_bright, c_bright, np.nanmedian(f_bright)))
                        for _ in range(35):
                            m_test = m_faint + step if going_faint else m_bright - step
                            c_test, _, f_test = run_trials_at_mag(m_test, pool=pool)
                            bracket_steps.append((m_test, c_test, np.nanmedian(f_test)))
                            if going_faint:
                                m_faint, c_faint = m_test, c_test
                                if c_faint < completeness_target:
                                    break
                            else:
                                # When stepping brighter, update based on detection
                                if c_test >= completeness_target:
                                    m_bright, c_bright = m_test, c_test
                                    break
                                # else: keep stepping brighter; do NOT break
                        bracketed = (c_bright >= completeness_target) and (
                            c_faint < completeness_target
                        )
                        if bracketed:
                            logger.info(
                                "Bracket recovered using initialGuess=%.2f",
                                float(guess),
                            )
                            break

                if not bracketed:
                    # Return NaN when bracketing fails - no default fallback limit
                    inject_lmag = np.nan
                    logger.warning(
                        f"Could not bracket cutoff; returning NaN"
                    )

                else:
                    # ---- Bisect phase ----------------------------------------
                    lo_m, lo_c = m_bright, c_bright
                    hi_m, hi_c = m_faint, c_faint
                    # Bracket endpoints already recorded in bracket_steps; no need to rerun
                    bisect_steps = [(lo_m, lo_c, np.nan), (hi_m, hi_c, np.nan)]

                    for _ in range(30):
                        mid_m = 0.5 * (lo_m + hi_m)
                        mid_c, _, mid_f = run_trials_at_mag(mid_m, pool=pool)
                        bisect_steps.append((mid_m, mid_c, np.nanmedian(mid_f)))

                        if mid_c >= completeness_target:
                            lo_m, lo_c = mid_m, mid_c
                        else:
                            hi_m, hi_c = mid_m, mid_c

                        if abs(hi_m - lo_m) < 0.02:
                            break

                    inject_lmag = 0.5 * (lo_m + hi_m)

                    # Optional: fit a smooth completeness curve with emcee and solve for m50.
                    if completeness_solver == "logistic_emcee":
                        try:
                            inject_lmag, m50_err = self._solve_m50_logistic_emcee(
                                run_trials_at_mag=run_trials_at_mag,
                                pool=pool,
                                m_guess=float(inject_lmag),
                                redo=3,
                                completeness_target=float(completeness_target),
                                lim_cfg=lim_cfg,
                            )
                            if np.isfinite(m50_err):
                                logger.info(
                                    "Injected limiting magnitude (logistic_emcee): m50=%.3f +/- %.3f (instrumental mag)",
                                    float(inject_lmag),
                                    float(m50_err),
                                )
                        except Exception as exc:
                            logger.warning(
                                "Injected limiting magnitude: logistic_emcee solver failed (%s); using bisection result.",
                                str(exc),
                            )

                # ---- Extended injection trials for plotting ----
                # Rely on bracket and bisect steps to determine magnitude range
                extended_steps = []

                # ---- Plot completeness curve (still inside pool context) -----
                if plot:
                    self._plot_completeness(
                        None,  # No sample_mags
                        None,  # No completeness_groups
                        None,  # No medians
                        bracket_steps,
                        bisect_steps,
                        inject_lmag,
                        completeness_target,
                        DETECTION_BETA_THRESH,
                        zeropoint,
                        recovery_method,
                        epsf_model=epsf_model,
                        cutout=cutout,
                        position=position,
                        background_rms=background_rms,
                        flux_for_mag=flux_for_mag,
                        image_zeropoint=image_zeropoint,
                        injection_df=injection_df,
                        F_ref=F_ref,
                        counts_ref=counts_ref,
                        exposure_time=exposure_time,
                        extended_steps=extended_steps,
                        orig_position=_orig_position,
                        target_name=self.input_yaml.get("target_name", None),
                        cutout_cx=cutout_cx,
                        cutout_cy=cutout_cy,
                    )

                    # Detection-limit demo plot removed by request.

                # Optional: EMCEE diagnostic plot for one representative injected trial.
                if (
                    plot
                    and recovery_method == "EMCEE"
                    and bool((lim_cfg.get("emcee_save_diagnostic_plot", True)))
                    and np.isfinite(inject_lmag)
                ):
                    try:
                        self._save_emcee_recovery_diagnostic(
                            cutout=cutout,
                            background_rms=background_rms,
                            epsf_model=epsf_model,
                            position=position,
                            m_inj=float(inject_lmag),
                            flux_for_mag=flux_for_mag,
                        )
                    except Exception:
                        pass

            # =================================================================
            # Log result
            # =================================================================
            elapsed = time.time() - start_time
            if np.isfinite(inject_lmag):
                app_str = ""
                if zeropoint is not None and np.isfinite(zeropoint):
                    app_mag = float(inject_lmag) + float(zeropoint)
                    app_str = f" ({app_mag:.3f} apparent)"
                zp_log = (
                    f"{float(zeropoint):.3f}"
                    if zeropoint is not None and np.isfinite(zeropoint)
                    else "n/a"
                )
                logger.info(
                    f"Limiting magnitude: instrumental={inject_lmag:.3f}{app_str}, "
                    f"zeropoint={zp_log}"
                )
                f_inst_per_s = 10.0 ** (-0.4 * float(inject_lmag))
                logger.info(
                    "  Context: m_inst = -2.5*log10(flux [e/s]) (same as mag(flux_AP)). "
                    "Negative m_inst only means the equivalent aperture rate f > 1 e/s; "
                    "it does not mean the source is bright in apparent terms. "
                    "At the solved limit, f ≈ %.3g e/s; interpret depth using apparent mag "
                    "+ zeropoint if the latter is trusted.",
                    f_inst_per_s,
                )
                if str(recovery_method).strip().upper() == "PSF" and snr_limit is None:
                    logger.info(
                        "  PSF recovery uses WLS fit S/N ≥ 3.0 by default. "
                        "Set limiting_magnitude.detection_limit for a stricter (shallower) report."
                    )
                logger.info(
                    f"Limiting mag ~ {inject_lmag:.3f}{app_str}  [{elapsed:.1f}s]"
                )
            else:
                logger.info(f"Limiting magnitude search failed [{elapsed:.1f}s]")

            # The limiting magnitude is already exposure-time-normalized via flux_for_mag
            return float(inject_lmag)

        except Exception as exc:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning(
                f"getInjectedLimit failed: {exc} "
                f"[{exc_type.__name__}, {fname}:{exc_tb.tb_lineno}]"
            )
            return np.nan

    def _save_emcee_recovery_diagnostic(
        self,
        *,
        cutout: np.ndarray,
        background_rms: np.ndarray | None,
        epsf_model,
        position,
        m_inj: float,
        flux_for_mag,
    ) -> None:
        """
        Save a simple emcee diagnostic plot (trace + marginal histograms)
        for one injected recovery near the final limiting magnitude.
        """
        import matplotlib.pyplot as plt

        lim_cfg = self.input_yaml.get("limiting_magnitude") or {}
        phot_cfg = self.input_yaml.get("photometry") or {}

        # Inject off-target (representative; avoid injecting on the transient).
        H, W = cutout.shape
        # Target is always at the centre of the cutout by construction.
        x0c = (W - 1) / 2.0
        y0c = (H - 1) / 2.0
        fwhm_px = float(self.input_yaml.get("fwhm", 3.0))
        r_min = float(lim_cfg.get("inject_min_radius_fwhm", 2.0)) * fwhm_px
        r_base = float(lim_cfg.get("inject_source_location", 3.0)) * fwhm_px
        r_inj = float(max(r_min, r_base))
        pts = points_in_circum(r_inj, center=(x0c, y0c), n=8)
        x0, y0 = float(pts[0][0]), float(pts[0][1])
        F = float(flux_for_mag(float(m_inj)))
        raw_os = getattr(epsf_model, "oversampling", 1)
        if isinstance(raw_os, (list, tuple, np.ndarray)):
            os_inj = int(raw_os[0])
        elif np.isscalar(raw_os) and raw_os > 1:
            os_inj = int(raw_os)
        else:
            os_inj = 1
        psf_add = _render_epsf_on_cutout(epsf_model, H, W, x0, y0, F, os_inj)
        img = np.asarray(cutout, dtype=float) + np.asarray(psf_add, dtype=float)

        # Small stamp for speed.
        half = int(np.ceil(max(6.0, 3.0 * fwhm_px)))
        xi = int(np.floor(x0))
        yi = int(np.floor(y0))
        x1 = max(0, xi - half)
        x2 = min(W, xi + half + 2)
        y1 = max(0, yi - half)
        y2 = min(H, yi + half + 2)
        stamp = np.asarray(img[y1:y2, x1:x2], dtype=float)
        gx, gy = np.meshgrid(np.arange(stamp.shape[1]), np.arange(stamp.shape[0]))

        rms_stamp = None
        if background_rms is not None:
            rms_stamp = np.asarray(background_rms[y1:y2, x1:x2], dtype=float)
            rms_stamp = np.abs(rms_stamp)

        from psf import MCMCFitter

        model = epsf_model.copy()
        model.x_0.value = float(x0 - x1)
        model.y_0.value = float(y0 - y1)
        model.flux.value = float(F)

        raw_delta = phot_cfg.get("emcee_delta", None)
        delta_px = float(raw_delta) if raw_delta is not None else 1.0

        fitter = MCMCFitter(
            nwalkers=int(phot_cfg.get("emcee_nwalkers", 20)),
            nsteps=int(phot_cfg.get("emcee_nsteps", 5000)),
            delta=float(delta_px),
            burnin_frac=float(phot_cfg.get("emcee_burnin_frac", 0.3)),
            thin=int(phot_cfg.get("emcee_thin", 10)),
            adaptive_tau_target=int(phot_cfg.get("emcee_adaptive_tau_target", 50)),
            min_autocorr_N=int(phot_cfg.get("emcee_min_autocorr_N", 100)),
            batch_steps=int(phot_cfg.get("emcee_batch_steps", 100)),
            jitter_scale=float(phot_cfg.get("emcee_jitter_scale", 0.01)),
            use_nddata_uncertainty=True,
            gain=float(resolve_gain_e_per_adu(None, self.input_yaml)),
            readnoise=float(self.input_yaml.get("read_noise", 0.0)),
            background_rms=rms_stamp,
        )
        fitter(model, gx, gy, stamp, use_nddata_uncertainty=True)

        chain = fitter.fit_info.get("samples", {}).get(0, None)
        if chain is None or np.asarray(chain).ndim != 2 or np.asarray(chain).shape[0] < 10:
            return
        chain = np.asarray(chain, dtype=float)
        # Assume parameters are (flux, x_0, y_0) in that order for the ePSF model.
        labels = ["flux", "x_0", "y_0"][: chain.shape[1]]

        fig, axes = plt.subplots(2, len(labels), figsize=(3.2 * len(labels), 4.6), constrained_layout=True)
        if len(labels) == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for j, lab in enumerate(labels):
            ax_t = axes[0, j]
            ax_h = axes[1, j]
            ax_t.plot(chain[:, j], lw=0.5, color="0.2")
            ax_t.set_title(lab)
            ax_t.set_xlabel("sample")
            ax_t.set_ylabel("value")

            v = chain[:, j]
            v = v[np.isfinite(v)]
            if v.size > 0:
                ax_h.hist(v, bins=40, histtype="step", color="0.2")
            ax_h.set_xlabel(lab)
            ax_h.set_ylabel("N")

        fpath = self.input_yaml.get("fpath", "frame")
        base = os.path.splitext(os.path.basename(str(fpath)))[0]
        outdir = os.path.dirname(str(fpath)) if os.path.dirname(str(fpath)) else "."
        save_png = os.path.join(outdir, f"EMCEE_InjectionDiag_{base}.png")
        fig.suptitle(f"EMCEE recovery diagnostic (m_inj={float(m_inj):.3f})", fontsize=10)
        fig.savefig(save_png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _solve_m50_logistic_emcee(
        self,
        *,
        run_trials_at_mag,
        pool,
        m_guess: float,
        redo: int,
        completeness_target: float,
        lim_cfg: dict,
    ) -> tuple[float, float]:
        """
        Fit a logistic completeness curve with emcee and return (m50, m50_err).

        We model P(det|m) = 1 / (1 + exp((m - m50)/s)), with s>0.
        Each injected trial is a Bernoulli outcome det in {0,1}.
        """
        import emcee
        import matplotlib.pyplot as plt

        nmags = int(lim_cfg.get("logistic_emcee_nmags", 9))
        nmags = max(5, min(nmags, 21))
        span = float(lim_cfg.get("logistic_emcee_span_mag", 1.5))
        span = max(0.5, min(span, 5.0))

        mags = np.linspace(float(m_guess) - span, float(m_guess) + span, nmags)
        outcomes_m = []
        outcomes_y = []
        for m in mags:
            _, _, _, flags = run_trials_at_mag(float(m), redo=int(redo), pool=pool, return_flags=True)
            outcomes_m.extend([float(m)] * int(len(flags)))
            outcomes_y.extend([1.0 if bool(v) else 0.0 for v in np.asarray(flags).ravel().tolist()])

        x = np.asarray(outcomes_m, dtype=float)
        y = np.asarray(outcomes_y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size < 30:
            raise RuntimeError(f"Too few trials for logistic_emcee fit (N={x.size}).")

        def log_prior(theta):
            m50, log_s = theta
            if not np.isfinite(m50) or not np.isfinite(log_s):
                return -np.inf
            s = np.exp(log_s)
            if s <= 0:
                return -np.inf
            # weakly-informative priors around the guessed transition
            if abs(m50 - float(m_guess)) > 5.0:
                return -np.inf
            if not (1e-3 <= s <= 5.0):
                return -np.inf
            return 0.0

        def log_like(theta):
            m50, log_s = theta
            s = np.exp(log_s)
            z = (x - m50) / s
            # P(det) = 1/(1+exp(z))
            p = 1.0 / (1.0 + np.exp(np.clip(z, -60, 60)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_like(theta)

        nwalkers = int(lim_cfg.get("logistic_emcee_nwalkers", 32))
        nwalkers = max(12, min(nwalkers, 96))
        nsteps = int(lim_cfg.get("logistic_emcee_nsteps", 2000))
        nsteps = max(400, min(nsteps, 20000))
        burn_frac = float(lim_cfg.get("logistic_emcee_burnin_frac", 0.3))
        burn_frac = max(0.1, min(burn_frac, 0.7))

        # init walkers around (m_guess, log(span/4))
        ndim = 2
        p0 = np.zeros((nwalkers, ndim), dtype=float)
        p0[:, 0] = float(m_guess) + 0.05 * np.random.randn(nwalkers)
        p0[:, 1] = np.log(max(0.1, span / 4.0)) + 0.25 * np.random.randn(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(p0, nsteps, progress=False)
        chain = sampler.get_chain()
        discard = int(burn_frac * chain.shape[0])
        flat = sampler.get_chain(discard=discard, flat=True)
        if flat.shape[0] < 50:
            raise RuntimeError("Too few posterior samples after burn-in.")

        m50_samp = flat[:, 0]
        m50 = float(np.nanmedian(m50_samp))
        m50_err = float(0.5 * (np.nanpercentile(m50_samp, 84) - np.nanpercentile(m50_samp, 16)))

        # Always write a diagnostic plot for the logistic completeness fit.
        try:
            fpath = str(self.input_yaml.get("fpath", "frame"))
            base = os.path.splitext(os.path.basename(fpath))[0]
            outdir = (
                str(self.input_yaml.get("write_dir"))
                if self.input_yaml.get("write_dir") not in (None, "", "None")
                else os.path.dirname(fpath)
            )
            if not outdir:
                outdir = "."
            os.makedirs(outdir, exist_ok=True)
            save_png = os.path.join(outdir, f"Completeness_LogisticEMCEE_{base}.png")

            # empirical completeness per mag point
            emp = []
            for m in mags:
                emp.append(
                    float(np.mean(y[x == float(m)]) if np.any(x == float(m)) else np.nan)
                )
            emp = np.asarray(emp, float)

            fig, ax = plt.subplots(figsize=set_size(340, 1))
            ax.plot(mags, emp, "o", ms=4, color="0.2", label="empirical")

            # median model curve
            s_med = float(np.exp(np.nanmedian(flat[:, 1])))
            mm = np.linspace(np.nanmin(mags), np.nanmax(mags), 200)
            p_med = 1.0 / (1.0 + np.exp(np.clip((mm - m50) / s_med, -60, 60)))
            ax.plot(mm, p_med, "-", lw=0.5, color="#1f77b4", label="logistic (median)")

            ax.axhline(float(completeness_target), color="0.6", lw=0.5, ls="--")
            ax.axvline(float(m50), color="k", lw=0.5, ls="--", label=f"m50={m50:.3f}")
            ax.set_xlabel("Injected ePSF instrumental magnitude")
            ax.set_ylabel("Recovery fraction")

            ax.invert_xaxis()
            ax.legend(loc="upper left", fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(save_png, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception:
            pass

        return m50, m50_err

    # -----------------------------------------------------------------------
    # S/N vs magnitude diagnostic
    # -----------------------------------------------------------------------

    def analyze_snr_vs_magnitude(
        self,
        sources: pd.DataFrame,
        zeropoint: float,
        flux_for_mag=None,
        snr_bins: list = None,
        plot: bool = True,
    ) -> dict:
        """
        Analyze the relationship between S/N and apparent magnitude for sources.

        Groups sources by S/N ranges and calculates statistics of their apparent magnitudes.
        This helps validate completeness plot calculations by showing what actual sources
        in the data have at different S/N levels.

        Parameters
        ----------
        sources : pd.DataFrame
            Source catalog with columns: SNR, flux_AP (or mag)
        zeropoint : float
            Photometric zeropoint to convert instrumental to apparent magnitude
        flux_for_mag : callable, optional
            Function to convert instrumental magnitude to flux (for consistency with limiting magnitude)
        snr_bins : list, optional
            S/N bin edges (default: [3, 5, 10, 20, 50, 100])
        plot : bool
            Whether to create a diagnostic plot

        Returns
        -------
        dict
            Dictionary with S/N bins and corresponding magnitude statistics
        """
        logger = logging.getLogger(__name__)
        
        if snr_bins is None:
            snr_bins = [3, 5, 10, 20, 50, 100]
        
        # Ensure SNR column exists
        if "SNR" not in sources.columns:
            logger.error("Source catalog missing SNR column")
            return {}
        
        # Calculate apparent magnitudes using same scale as limiting magnitude
        exposure_time = _effective_exposure_seconds(self.input_yaml)

        if "flux_AP" in sources.columns:
            # flux_AP is e-/s from Aperture.measure; mag() is instrumental in that system
            from functions import mag
            sources["apparent_mag"] = sources["flux_AP"].apply(mag) + zeropoint
            logger.info("Using pipeline convention: mag(flux_AP/s) + zeropoint")
        elif "mag" in sources.columns:
            # Use existing magnitude column (assuming it's apparent)
            sources["apparent_mag"] = sources["mag"]
        else:
            logger.error("Source catalog missing flux_AP or mag column")
            return {}
        
        # Filter out invalid magnitudes
        sources = sources[np.isfinite(sources["apparent_mag"]) & np.isfinite(sources["SNR"])]
        
        # Group by S/N bins
        results = {}
        for i in range(len(snr_bins) - 1):
            snr_min = snr_bins[i]
            snr_max = snr_bins[i + 1]
            
            bin_sources = sources[(sources["SNR"] >= snr_min) & (sources["SNR"] < snr_max)]
            
            if len(bin_sources) > 0:
                results[f"{snr_min}-{snr_max}"] = {
                    "count": len(bin_sources),
                    "snr_median": float(np.median(bin_sources["SNR"])),
                    "mag_median": float(np.median(bin_sources["apparent_mag"])),
                    "mag_mean": float(np.mean(bin_sources["apparent_mag"])),
                    "mag_std": float(np.std(bin_sources["apparent_mag"])),
                    "mag_min": float(np.min(bin_sources["apparent_mag"])),
                    "mag_max": float(np.max(bin_sources["apparent_mag"])),
                }
        
        logger.info(f"S/N vs magnitude analysis: {len(results)} bins with data")
        for bin_name, stats in results.items():
            logger.info(
                f"  S/N {bin_name}: n={stats['count']}, "
                f"mag={stats['mag_median']:.2f} +/- {stats['mag_std']:.2f}"
            )
        
        if plot:
            self._plot_snr_vs_magnitude(sources, snr_bins, results, zeropoint)
        
        return results

    def _plot_snr_vs_magnitude(
        self,
        sources: pd.DataFrame,
        snr_bins: list,
        results: dict,
        zeropoint: float,
    ) -> None:
        """
        Create diagnostic plot showing S/N vs apparent magnitude relationship.
        """
        import logging
        logger = logging.getLogger(__name__)
        import matplotlib.pyplot as plt
        from functions import set_size
        
        fpath = str(self.input_yaml.get("fpath", "frame"))
        base = os.path.splitext(os.path.basename(fpath))[0]
        write_dir = (
            str(self.input_yaml.get("write_dir"))
            if self.input_yaml.get("write_dir") not in (None, "", "None")
            else os.path.dirname(fpath)
        )
        if not write_dir:
            write_dir = "."
        
        fig, ax = plt.subplots(figsize=set_size(340, 1.5))
        
        # Scatter plot of all sources
        sc = ax.scatter(
            sources["SNR"],
            sources["apparent_mag"],
            s=get_marker_size('medium'),
            alpha=0.3,
            color="tab:blue",
            label="Sources",
        )
        
        # Add median markers for each S/N bin
        for bin_name, stats in results.items():
            ax.plot(
                stats["snr_median"],
                stats["mag_median"],
                "o",
                color="red",
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1.0,
                label=f"Median ({bin_name})" if bin_name == "3-5" else "",
            )
            # Add error bar
            ax.errorbar(
                stats["snr_median"],
                stats["mag_median"],
                yerr=stats["mag_std"],
                fmt="none",
                color="red",
                linewidth=0.5,
                capsize=3,
                alpha=0.7,
            )
        
        ax.set_xlabel("S/N", fontsize=10)
        ax.set_ylabel("Apparent magnitude [mag]", fontsize=10)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        fig.tight_layout()
        save_loc = os.path.join(write_dir, f"SNR_vs_Magnitude_{base}.png")
        fig.savefig(save_loc, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        logger.info(f"S/N vs magnitude plot saved to {save_loc}")

    # -----------------------------------------------------------------------
    # Diagnostic plot
    # -----------------------------------------------------------------------

    def _plot_completeness(
        self,
        sample_mags,
        completeness_groups,
        medians,
        bracket_steps,
        bisect_steps,
        inject_lmag,
        completeness_target,
        DETECTION_BETA_THRESH,
        zeropoint,
        recovery_method,
        epsf_model=None,
        cutout=None,
        position=None,
        background_rms=None,
        flux_for_mag=None,
        image_zeropoint=None,
        injection_df=None,
        F_ref=None,
        counts_ref=None,
        extended_steps=None,
        orig_position=None,
        target_name=None,
        exposure_time=None,
        cutout_cx=None,
        cutout_cy=None,
    ) -> None:
        """
        Plot completeness curve with injection examples.
        If sample_mags is None, only shows bracket/bisect trajectories (no bar chart).
        """
        logger = logging.getLogger(__name__)
        
        # Select appropriate zeropoint based on recovery method
        if image_zeropoint is not None:
            recovery_method_upper = str(recovery_method).strip().upper()
            if recovery_method_upper in ("PSF", "EMCEE"):
                selected_zeropoint = image_zeropoint.get("PSF", {}).get("zeropoint", zeropoint)
                logger.info(f"Using PSF zeropoint: {selected_zeropoint}")
            else:  # AP method
                selected_zeropoint = image_zeropoint.get("AP", {}).get("zeropoint", zeropoint)
                logger.info(f"Using AP zeropoint: {selected_zeropoint}")
        else:
            selected_zeropoint = zeropoint
            logger.info("Using provided zeropoint (image_zeropoint not available)")
        # Skip bar chart processing if sample_mags is None
        if sample_mags is not None:
            order = np.argsort(sample_mags)
            mags_sorted = sample_mags[order]
            groups_sorted = [completeness_groups[i] for i in order]
            medians_sorted = np.asarray([medians[i] for i in order], float)
        else:
            mags_sorted = None
            groups_sorted = None
            medians_sorted = None

        fpath = str(self.input_yaml.get("fpath", "frame"))
        base = os.path.splitext(os.path.basename(fpath))[0]
        write_dir = (
            str(self.input_yaml.get("write_dir"))
            if self.input_yaml.get("write_dir") not in (None, "", "None")
            else os.path.dirname(fpath)
        )
        if not write_dir:
            write_dir = "."
        os.makedirs(write_dir, exist_ok=True)

        # Use the project-wide plotting style for consistency.
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))
        except Exception:
            # Fall back silently if the style cannot be loaded.
            pass

        plt.ioff()
        
        # Create figure with main completeness plot on top, injection examples below
        fig = plt.figure(figsize=set_size(540, 1.5))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.5, 1, 1])
        
        # Top row: Main completeness plot (spans all columns)
        ax = fig.add_subplot(gs[0, :])

        # Bar chart removed - using only bracket and bisect search trajectories
        # Plot bracket and bisect search trajectories
        if bracket_steps:
            bracket_mags = np.array([step[0] for step in bracket_steps])
            bracket_rates = np.array([step[1] for step in bracket_steps]) * 100  # Convert to percentage
            ax.plot(
                bracket_mags,
                bracket_rates,
                marker='o',
                markersize=3,
                linestyle='-',
                color='#D55E00',
                alpha=0.7,
                lw=0.8,
                label='Bracket search',
                zorder=3,
            )

        if bisect_steps:
            bisect_mags = np.array([step[0] for step in bisect_steps])
            bisect_rates = np.array([step[1] for step in bisect_steps]) * 100  # Convert to percentage
            ax.plot(
                bisect_mags,
                bisect_rates,
                marker='s',
                markersize=3,
                linestyle='--',
                color='#0072B2',
                alpha=0.7,
                lw=0.8,
                label='Bisection',
                zorder=3,
            )

        if mags_sorted is not None:
            widths = 0.15
            n_trials = np.array([max(1, len(g)) for g in groups_sorted], dtype=float)
            p = np.clip(medians_sorted, 0.0, 1.0)
            p_percent = p * 100  # Convert to percentage
            # Binomial standard error (normal approx) for visual guidance.
            p_err = np.sqrt(np.maximum(p * (1.0 - p) / n_trials, 0.0)) * 100  # Convert to percentage

            ax.bar(
                mags_sorted,
                p_percent,
                width=widths,
                color="#B0C4DE",
                edgecolor="#4D4D4D",
                linewidth=0.5,
                zorder=2,
            )
            ax.errorbar(
                mags_sorted,
                p_percent,
                yerr=p_err,
                fmt="none",
                ecolor="#4D4D4D",
                elinewidth=0.5,
                capsize=2,
                zorder=3,
            )

        # Bracket and bisect search trajectories.
        if bracket_steps:
            bm, bc, _ = zip(*bracket_steps)  # Unpack 3 values: (mag, detection_rate, recovered_flux)
            bc_percent = [c * 100 for c in bc]  # Convert to percentage
            ax.plot(
                bm,
                bc_percent,
                "-",
                ms=get_marker_size('medium'),
                lw=0.5,
                color="#0000FF",
            )
            # Add arrows for jumps > 0.1 mag
            for i in range(len(bm) - 1):
                mag_jump = abs(bm[i+1] - bm[i])
                if mag_jump > 0.1:
                    ax.annotate('', xy=(bm[i+1], bc_percent[i+1]), xytext=(bm[i], bc_percent[i]),
                               arrowprops=dict(arrowstyle='->', color='#0000FF', lw=0.5))
        if bisect_steps:
            bm, bc, _ = zip(*bisect_steps)  # Unpack 3 values: (mag, detection_rate, recovered_flux)
            bc_percent = [c * 100 for c in bc]  # Convert to percentage
            ax.plot(
                bm,
                bc_percent,
                "--",
                ms=get_marker_size('medium'),
                lw=0.5,
                color="#00AA00",
            )
            # Add arrows for jumps > 0.1 mag
            for i in range(len(bm) - 1):
                mag_jump = abs(bm[i+1] - bm[i])
                if mag_jump > 0.1:
                    ax.annotate('', xy=(bm[i+1], bc_percent[i+1]), xytext=(bm[i], bc_percent[i]),
                               arrowprops=dict(arrowstyle='->', color='#00AA00', lw=0.5))

        # Reference lines.
        ax.axhline(50, color="0.7", lw=0.5, ls="--", zorder=0)
        ax.text(
            0.02,
            50,
            "50%",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            color="0.5",
        )

        if completeness_target != 0.5:
            target_percent = completeness_target * 100
            ax.axhline(target_percent, color="0.7", lw=0.5, ls="-.", zorder=0)
            ax.text(
                0.98,
                target_percent,
                f"{int(target_percent)}%",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                ha="right",
                color="0.5",
            )

        if np.isfinite(inject_lmag):
            ax.axvline(
                inject_lmag,
                color="k",
                lw=0.5,
                ls="--",
                label="Adopted limit",
            )
        
        # Add axis labels to main completeness plot
        ax.set_xlabel("Injected brightness [mag]", fontsize=9)
        ax.set_ylabel("Recovery fraction [%]", fontsize=9)
        

        # Optional apparent-magnitude secondary axis.
        secax = None
        if selected_zeropoint is not None:
            # Capture by value to avoid late-binding lambda bug
            _zp = float(selected_zeropoint)
            secax = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda m, zp=_zp: m + zp,
                    lambda m, zp=_zp: m - zp,
                ),
            )
            secax.set_xlabel("Apparent brightness [mag]")
            # The secondary axis direction follows from the monotonic transform;
            # do not call secax.invert_xaxis() as it would double-invert.

        fig.tight_layout()
        
        # Add injection cutout panels below main plot if data available
        if epsf_model is not None and cutout is not None and position is not None and \
           np.isfinite(inject_lmag) and flux_for_mag is not None:
            ny_c, nx_c = cutout.shape
            # Use actual target center from get_cutout (accounts for partial cutouts)
            # Fall back to geometric center if not provided
            if cutout_cx is not None and cutout_cy is not None:
                target_x = float(cutout_cx)
                target_y = float(cutout_cy)
                logger.info(f"Using true target center: ({target_x:.2f}, {target_y:.2f})")
            else:
                target_x   = (nx_c - 1) / 2.0
                target_y   = (ny_c - 1) / 2.0
                logger.info(f"Using geometric center: ({target_x:.2f}, {target_y:.2f})")

            # aperture_radius and fwhm for subpanel use
            fwhm            = float(self.input_yaml.get("fwhm", 3.0))
            phot_cfg        = self.input_yaml.get("photometry", {})
            aperture_radius = float(phot_cfg.get("aperture_radius", fwhm))
            lim_cfg         = self.input_yaml.get("limiting_magnitude") or {}

            # Define target magnitudes for subpanel injection
            # Convert instrumental limiting magnitude to apparent magnitude
            # selected_zeropoint already computed at top of function
            if selected_zeropoint is not None and np.isfinite(float(selected_zeropoint)):
                limiting_apparent_mag = inject_lmag + float(selected_zeropoint)
            else:
                limiting_apparent_mag = inject_lmag  # Fallback to instrumental if no zeropoint
            
            # Start at limiting magnitude, then brighter magnitudes
            mag_offsets = [0.0, -0.5, -1.0]  # Limiting mag, 0.2 mag brighter, 0.5 mag brighter
            mag_targets = [limiting_apparent_mag + offset for offset in mag_offsets]
            
            # Calculate background RMS for S/N scaling (robust estimator)
            from astropy.stats import mad_std
            if background_rms is not None:
                # Use median of provided RMS map (excludes sources)
                background_rms_scalar = float(np.nanmedian(background_rms))
            else:
                # Fallback: use MAD which is robust to sources
                background_rms_scalar = float(mad_std(cutout, ignore_nan=True))

            raw_os_plot = getattr(epsf_model, "oversampling", 1)
            if isinstance(raw_os_plot, (list, tuple, np.ndarray)):
                oversampling_plot = int(raw_os_plot[0])
            elif np.isscalar(raw_os_plot) and raw_os_plot > 1:
                oversampling_plot = int(raw_os_plot)
            else:
                oversampling_plot = 1

            # ── Demo injection site selection ─────────────────────────────────────────
            #
            # Priority:
            #   1. Quietest valid site from injection_df (lowest |flux_AP|).
            #   2. Random circumference point (never always index 0 = top-right).
            #
            # Valid = within cutout bounds (with fwhm margin) AND at least
            #         (2*aperture_radius + fwhm) from the transient position.
            # ─────────────────────────────────────────────────────────────────────────
            min_sep   = 2.0 * aperture_radius + fwhm
            px_margin = float(np.ceil(fwhm))
            inj_dist  = float(lim_cfg.get("inject_source_location", 3.0)) * fwhm

            demo_x = demo_y = None

            if injection_df is not None and len(injection_df) > 0:
                flux_col = "flux_AP" if "flux_AP" in injection_df.columns else None
                scored   = []
                for _, site in injection_df.iterrows():
                    sx, sy = float(site["x_pix"]), float(site["y_pix"])
                    # Bounds check with margin
                    if not (
                        px_margin <= sx <= nx_c - 1 - px_margin
                        and px_margin <= sy <= ny_c - 1 - px_margin
                    ):
                        continue
                    # Must be far enough from the transient
                    if float(np.hypot(sx - target_x, sy - target_y)) < min_sep:
                        continue
                    score = float(abs(site[flux_col])) if flux_col else 0.0
                    scored.append((score, sx, sy))

                if scored:
                    scored.sort(key=lambda t: t[0])   # ascending → quietest first
                    _, demo_x, demo_y = scored[0]

            if demo_x is None:
                # Try each circumference point; pick first one with full annulus clearance
                annulus_margin = aperture_radius + fwhm * 2.0  # must fit full annulus
                pts = points_in_circum(inj_dist, center=[target_x, target_y], n=8)
                for rng_idx in self._rng.permutation(len(pts)).tolist():
                    cx_try = float(pts[rng_idx][0])
                    cy_try = float(pts[rng_idx][1])
                    if (annulus_margin <= cx_try <= nx_c - 1 - annulus_margin and
                            annulus_margin <= cy_try <= ny_c - 1 - annulus_margin):
                        demo_x = cx_try
                        demo_y = cy_try
                        break
                if demo_x is None:
                    # Last resort: use a point 3 FWHM from center in x direction
                    demo_x = float(np.clip(target_x + 3.0 * fwhm, annulus_margin,
                                           nx_c - 1 - annulus_margin))
                    demo_y = float(np.clip(target_y, annulus_margin,
                                           ny_c - 1 - annulus_margin))
                    logger.warning(
                        "All circumference fallback points too close to edge; "
                        "using offset demo site (%.1f, %.1f)", demo_x, demo_y,
                    )

            for i, mag_target in enumerate(mag_targets):
                ax_inject = fig.add_subplot(gs[1, i])
                
                try:
                    # Create a cutout-sized grid
                    ny, nx = cutout.shape
                    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
                    logger.debug(f"Subpanel: cutout shape=({ny},{nx}), grid shape={y_grid.shape}, {x_grid.shape}")
                    
                    # Target center: use true center from get_cutout if available
                    if cutout_cx is not None and cutout_cy is not None:
                        x_center = float(cutout_cx)
                        y_center = float(cutout_cy)
                    else:
                        # Fallback to geometric center
                        x_center = (nx - 1) / 2.0
                        y_center = (ny - 1) / 2.0
                    
                    # Inject PSF at this magnitude
                    injected = cutout.copy()

                    # Calculate flux needed to achieve target magnitude
                    # flux_for_mag expects instrumental magnitude, so convert from apparent
                    if selected_zeropoint is not None:
                        inst_mag_target = mag_target - selected_zeropoint
                    else:
                        inst_mag_target = mag_target  # Fallback if no zeropoint
                    flux_adu = flux_for_mag(inst_mag_target)  # Total flux to inject

                    # Add PSF flux to cutout at demo location
                    inject_x = demo_x
                    inject_y = demo_y
                    # inject_x/inject_y are in cutout-local coordinates (from injection_df["x_pix"]/["y_pix"])

                    ny, nx = cutout.shape
                    psf_inject = _render_epsf_on_cutout(
                        epsf_model,
                        ny,
                        nx,
                        float(inject_x),
                        float(inject_y),
                        float(flux_adu),
                        oversampling_plot,
                    )
                    injected += psf_inject
                    
                    # Zoom centred on TARGET so it always appears in the centre of the panel.
                    # The zoom radius is large enough to include the injection site with margin.
                    inject_distance = float(np.sqrt(
                        (inject_x - x_center)**2 + (inject_y - y_center)**2
                    ))
                    # Center zoom between target and injection site
                    mid_x = (x_center + inject_x) / 2.0
                    mid_y = (y_center + inject_y) / 2.0
                    half_dist = inject_distance / 2.0
                    zoom_radius = half_dist + max(3.0 * fwhm, 2.0 * aperture_radius)

                    # Clamp to cutout boundaries
                    x0_zoom = int(max(0, np.floor(mid_x - zoom_radius)))
                    x1_zoom = int(min(nx, np.ceil(mid_x + zoom_radius)))
                    y0_zoom = int(max(0, np.floor(mid_y - zoom_radius)))
                    y1_zoom = int(min(ny, np.ceil(mid_y + zoom_radius)))

                    # Guard: zoom must have non-zero area
                    if x1_zoom <= x0_zoom or y1_zoom <= y0_zoom:
                        logger.warning("Zoom region has zero area; skipping subpanel.")
                        raise ValueError("Zero-area zoom region")

                    # Extract zoomed background_rms region for aperture measurement
                    background_rms_zoom = None
                    if background_rms is not None:
                        background_rms_zoom = background_rms[y0_zoom:y1_zoom, x0_zoom:x1_zoom]
                        background_rms_zoom = np.abs(np.asarray(background_rms_zoom, dtype=float))

                    # Short aliases used by the S/N block
                    x0_z, x1_z, y0_z, y1_z = x0_zoom, x1_zoom, y0_zoom, y1_zoom
                    bkgrms_zoom = background_rms_zoom

                    logger.debug(
                        "Subpanel zoom: cutout=(%dx%d), target=(%.1f,%.1f), "
                        "injection=(%.1f,%.1f), inject_distance=%.1f px, zoom_radius=%.1f px, "
                        "zoom=[%d:%d, %d:%d]",
                        nx, ny, x_center, y_center,
                        inject_x, inject_y, inject_distance, zoom_radius,
                        x0_zoom, x1_zoom, y0_zoom, y1_zoom,
                    )

                    # Display the full injected cutout, zoom with set_xlim/ylim
                    from astropy.visualization import ZScaleInterval
                    zscale = ZScaleInterval()
                    finite = injected[np.isfinite(injected)]
                    if finite.size:
                        lower, upper = np.percentile(finite, [0.5, 99.5])
                        vmin, vmax = zscale.get_limits(np.clip(injected, lower, upper))
                    else:
                        vmin, vmax = np.nanmin(injected), np.nanmax(injected)
                    im = ax_inject.imshow(injected, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                    ax_inject.set_xlim(x0_zoom, x1_zoom)
                    ax_inject.set_ylim(y0_zoom, y1_zoom)

                    # Mark transient position (center of cutout, where actual target is)
                    from matplotlib.patches import Circle
                    transient_marker = Circle((x_center, y_center), radius=aperture_radius,
                                            edgecolor='red', facecolor='none', linestyle='-', linewidth=0.5)
                    ax_inject.add_patch(transient_marker)
                    # Use target_name with TNS prefix if available, otherwise use '1'
                    if target_name and target_name.strip():
                        # Get TNS prefix from input_yaml if available
                        name_prefix = self.input_yaml.get("name_prefix", "")
                        objname = self.input_yaml.get("objname", target_name)
                        # Add prefix if it exists and is not already in the name
                        if name_prefix and name_prefix.strip() and not target_name.startswith(name_prefix):
                            transient_label = f"{name_prefix}{objname}"
                        else:
                            transient_label = target_name
                    else:
                        transient_label = '1'
                    ax_inject.text(x_center, y_center + aperture_radius, transient_label,
                                   color='red', fontsize=8, ha='center', va='bottom')

                    # Mark injected source location with aperture circle
                    # Circle centre is in cutout-local coordinates (imshow displays full cutout)
                    aperture_circle = Circle((inject_x, inject_y), radius=aperture_radius,
                                           edgecolor='white', facecolor='none', linestyle='--', linewidth=0.5)
                    ax_inject.add_patch(aperture_circle)
                    
                    try:
                        recovery_method_upper = str(recovery_method).strip().upper()

                        injected_zoom = injected[y0_zoom:y1_zoom, x0_zoom:x1_zoom]
                        inj_x_z = inject_x - x0_zoom
                        inj_y_z = inject_y - y0_zoom

                        if recovery_method_upper == "PSF":
                            # Fast weighted-least-squares PSF S/N — same estimator as _injection_worker.
                            phot_cfg_l = self.input_yaml.get("photometry") or {}
                            scale_v    = float(phot_cfg_l.get("psf_fit_shape_vfaint_scale_fwhm", 2.0))
                            half_s     = int(np.ceil(max(3.0, scale_v * fwhm)))
                            nyiz, nxiz = injected_zoom.shape
                            x0s = max(0, int(np.floor(inj_x_z)) - half_s)
                            x1s = min(nxiz, int(np.floor(inj_x_z)) + half_s + 2)
                            y0s = max(0, int(np.floor(inj_y_z)) - half_s)
                            y1s = min(nyiz, int(np.floor(inj_y_z)) + half_s + 2)
                            data_s   = np.asarray(injected_zoom[y0s:y1s, x0s:x1s], float)
                            # Construct stamp grid in CUTOUT-LOCAL coordinates
                            x0s_cut = x0s + x0_zoom  # cutout-local stamp left edge
                            x1s_cut = x1s + x0_zoom
                            y0s_cut = y0s + y0_zoom
                            y1s_cut = y1s + y0_zoom
                            gxs, gys = np.meshgrid(
                                np.arange(x0s_cut, x1s_cut),
                                np.arange(y0s_cut, y1s_cut),
                            )
                            psf1 = np.asarray(
                                epsf_model.evaluate(x=gxs, y=gys, flux=1.0,
                                                    x_0=inject_x, y_0=inject_y),
                                float,
                            )
                            if bkgrms_zoom is not None:
                                var_s = np.maximum(bkgrms_zoom[y0s:y1s, x0s:x1s] ** 2, 1e-30)
                            else:
                                sig_s = float(np.nanstd(data_s))
                                var_s = np.full_like(data_s, max(sig_s ** 2, 1e-30))
                            ok_s = np.isfinite(data_s) & np.isfinite(psf1) & (var_s > 0)
                            snr  = np.nan
                            if int(np.count_nonzero(ok_s)) >= 10:
                                w_s  = 1.0 / var_s[ok_s]
                                a_s  = np.vstack([psf1[ok_s].ravel(),
                                                  np.ones(int(np.count_nonzero(ok_s)))])
                                aw_s = a_s * w_s
                                m_s  = aw_s @ a_s.T
                                b_s  = aw_s @ data_s[ok_s].ravel()
                                try:
                                    cov_s = np.linalg.inv(m_s)
                                except Exception:
                                    cov_s = None
                                if cov_s is not None and np.all(np.isfinite(cov_s)):
                                    theta_s  = cov_s @ b_s
                                    flux_hat = float(theta_s[0])
                                    flux_err = float(np.sqrt(max(cov_s[0, 0], 0.0)))
                                    if np.isfinite(flux_err) and flux_err > 0:
                                        snr = flux_hat / flux_err

                        else:  # AP / EMCEE → aperture photometry
                            ap_obj   = Aperture(input_yaml=self.input_yaml, image=injected_zoom)
                            snr_df   = pd.DataFrame({"x_pix": [inj_x_z], "y_pix": [inj_y_z]})
                            snr_meas = ap_obj.measure(
                                sources=snr_df, plot=False,
                                background_rms=bkgrms_zoom, verbose=0,
                            )
                            snr = float(snr_meas["SNR"].iloc[0])
                            if not np.isfinite(snr):
                                fl  = float(snr_meas["flux_AP"].iloc[0])
                                ns  = float(snr_meas["noiseSky"].iloc[0])
                                snr = fl / ns if ns > 0 else 0.0

                        if not np.isfinite(snr):
                            raise ValueError("SNR not finite; using fallback")

                    except Exception as snr_exc:
                        logger.debug("Subpanel S/N fallback: %s", snr_exc)
                        sig   = float(np.nansum(
                            injected[y0_zoom:y1_zoom, x0_zoom:x1_zoom]
                            - cutout[y0_zoom:y1_zoom, x0_zoom:x1_zoom]
                        ))
                        noise = background_rms_scalar * float(
                            np.sqrt(np.pi * aperture_radius ** 2)
                        )
                        snr   = sig / noise if noise > 0 else 0.0

                    ax_inject.text(
                        inject_x, inject_y + aperture_radius,
                        f"S/N={snr:.1f}",
                        color="white", fontsize=8, ha="center", va="bottom",
                    )

                    # Set title with injected magnitude
                    ax_inject.set_title(f'Injected mag = {mag_target:.2f}', fontsize=9)

                    # Add recovered magnitude text in lower left corner
                    if selected_zeropoint is not None:
                        # Compute recovered apparent magnitude from flux
                        recovered_apparent = np.nan
                        if recovery_method_upper == "PSF" and 'flux_hat' in locals() and np.isfinite(flux_hat):
                            # PSF method: flux_hat is PSF flux parameter
                            if counts_ref is not None and exposure_time is not None and counts_ref > 0 and exposure_time > 0:
                                recovered_flux_adu_per_s = flux_hat * counts_ref / exposure_time
                                recovered_inst = -2.5 * np.log10(max(recovered_flux_adu_per_s, 1e-30))
                                recovered_apparent = recovered_inst + selected_zeropoint
                        elif recovery_method_upper in ["AP", "EMCEE"]:
                            # AP/EMCEE method: use aperture flux from snr_meas
                            try:
                                recovered_flux = float(snr_meas["flux_AP"].iloc[0])
                                recovered_inst = -2.5 * np.log10(max(recovered_flux, 1e-30))
                                recovered_apparent = recovered_inst + selected_zeropoint
                            except Exception:
                                pass

                        if np.isfinite(recovered_apparent):
                            ax_inject.text(
                                0.05, 0.05,
                                f"Rec: {recovered_apparent:.2f}",
                                transform=ax_inject.transAxes,
                                color="white", fontsize=7, ha="left", va="bottom",
                                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
                            )
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax_inject, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                    ax_inject.set_xlabel('X [pixels]', fontsize=8)
                    if i == 0:
                        ax_inject.set_ylabel('Y [pixels]', fontsize=8)
                    else:
                        ax_inject.set_ylabel('')
                    ax_inject.tick_params(labelsize=8)
                except Exception as e:
                    # If injection fails, just show the original cutout
                    import traceback
                    logger.warning(f"Subpanel injection failed for mag={mag_target:.2f}: {e}")
                    logger.debug(f"Subpanel injection traceback:\n{traceback.format_exc()}")
                    ny, nx = cutout.shape
                    from astropy.visualization import simple_norm
                    norm = simple_norm(cutout, 'sqrt', percent=99.5)
                    ax_inject.imshow(cutout, origin='lower', cmap='viridis', norm=norm)
                    ax_inject.set_xlim(0, nx)
                    ax_inject.set_ylim(0, ny)
                    ax_inject.set_title(f'Injection failed', fontsize=9)
                    ax_inject.set_xlabel('X [pixels]', fontsize=8)
                    if i == 0:
                        ax_inject.set_ylabel('Y [pixels]', fontsize=8)
                    else:
                        ax_inject.set_ylabel('')
                    ax_inject.tick_params(labelsize=8)

            # Add axis data: small grey lines at top of main plot showing injected magnitudes
            # Plot on secondary axis (apparent magnitude) for correct positioning
            if secax is not None:
                from matplotlib.transforms import blended_transform_factory
                # Get current y-axis limits to calculate the top 5% in data coordinates
                ymin, ymax = ax.get_ylim()
                y_bottom = ymax - 0.05 * (ymax - ymin)  # 5% from top in data coordinates
                y_top = ymax
                # Create blended transform: secondary axis for x, primary axis for y
                transform = blended_transform_factory(secax.transData, ax.transData)
                for mag_target in mag_targets:
                    # mag_targets are already in apparent magnitude
                    ax.plot(
                        [mag_target, mag_target],
                        [y_bottom, y_top],
                        color='red',
                        alpha=0.5,
                        linewidth=1.5,
                        linestyle='-',
                        zorder=1,
                        transform=transform,
                    )

        fig.tight_layout()
        save_loc_png = os.path.join(write_dir, f"Completeness_{base}.png")
        fig.savefig(save_loc_png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # ---- Plot injection recovery vs magnitude (apparent vs instrumental) ----
        if bracket_steps or bisect_steps or extended_steps:
            # Check configuration to see if injection recovery plot should be generated
            plot_injection_recovery = (self.input_yaml.get("limiting_magnitude") or {}).get("plot_injection_recovery", False)
            if plot_injection_recovery:
                self._plot_injection_recovery(
                    bracket_steps,
                    bisect_steps,
                    extended_steps,
                    inject_lmag,
                    zeropoint,
                    selected_zeropoint,
                    write_dir,
                    base,
                    position,
                    counts_ref=counts_ref,
                    exposure_time=exposure_time,
                    recovery_method=recovery_method,
                )

    def _plot_injection_recovery(
        self,
        bracket_steps,
        bisect_steps,
        extended_steps,
        inject_lmag,
        zeropoint,
        selected_zeropoint,
        write_dir,
        base,
        position,
        counts_ref=None,
        exposure_time=None,
        recovery_method=None,
    ) -> None:
        """
        Plot injected apparent magnitude vs recovered apparent magnitude.
        Shows photometry recovery: detected sources follow a 1:1 line,
        non-detected sources flatten out. Also plots catalog sources for comparison.
        """
        import matplotlib.pyplot as plt
        from plotting_utils import get_color, get_marker_size, get_alpha, get_line_width, get_okabe_color

        logger = logging.getLogger(__name__)

        # Use extended_steps if available, otherwise fall back to bracket/bisect steps
        all_steps = extended_steps if extended_steps else (bracket_steps + bisect_steps)
        if not all_steps:
            logger.warning("No injection steps to plot")
            return

        # Extract magnitudes, detection rates, and recovered fluxes
        inst_mags = np.array([step[0] for step in all_steps])
        det_rates = np.array([step[1] for step in all_steps])
        recovered_fluxes = np.array([step[2] for step in all_steps])

        logger.info(f"Debug: inst_mags range: {inst_mags.min():.2f} to {inst_mags.max():.2f}")
        logger.info(f"Debug: selected_zeropoint: {selected_zeropoint}")
        logger.info(f"Debug: det_rates range: {det_rates.min():.2f} to {det_rates.max():.2f}")

        # Convert to apparent magnitudes
        injected_apparent = inst_mags + selected_zeropoint
        logger.info(f"Debug: injected_apparent range: {injected_apparent.min():.2f} to {injected_apparent.max():.2f}")

        # Convert recovered flux to instrumental magnitude, then to apparent
        # The recovered flux units depend on recovery_method:
        # - AP method: flux_AP is already ADU/s (divided by exposure_time in Aperture.measure())
        # - PSF/EMCEE methods: flux_hat is the PSF flux parameter (dimensionless scaling factor)
        #   To convert to actual flux: flux_actual = flux_hat * counts_ref / exposure_time
        logger.info(f"Debug: recovery_method={recovery_method}, counts_ref={counts_ref}, exposure_time={exposure_time}")
        logger.info(f"Debug: recovered_fluxes sample: {recovered_fluxes[:3]}")

        recovery_method_upper = str(recovery_method).strip().upper() if recovery_method is not None else "AP"
        if recovery_method_upper == "AP":
            # AP method: flux_AP is already ADU/s, apply mag() directly
            recovered_inst = -2.5 * np.log10(np.maximum(recovered_fluxes, 1e-30))
        else:
            # PSF/EMCEE methods: flux_hat is PSF flux parameter (dimensionless)
            # Convert to actual flux: flux_actual = flux_hat * counts_ref / exposure_time
            if counts_ref is not None and exposure_time is not None and counts_ref > 0 and exposure_time > 0:
                recovered_flux_adu_per_s = recovered_fluxes * counts_ref / exposure_time
                recovered_inst = -2.5 * np.log10(np.maximum(recovered_flux_adu_per_s, 1e-30))
            else:
                # Fallback: treat as raw ADU divided by exposure_time
                if exposure_time is not None and exposure_time > 0:
                    recovered_flux_adu_per_s = recovered_fluxes / exposure_time
                    recovered_inst = -2.5 * np.log10(np.maximum(recovered_flux_adu_per_s, 1e-30))
                else:
                    recovered_inst = -2.5 * np.log10(np.maximum(recovered_fluxes, 1e-30))
        
        logger.info(f"Debug: recovered_inst sample: {recovered_inst[:3]}")
        recovered_apparent = recovered_inst + selected_zeropoint

        # Separate detected vs non-detected (use 50% threshold)
        detected_mask = det_rates >= 0.5
        detected_injected = injected_apparent[detected_mask]
        detected_recovered = recovered_apparent[detected_mask]
        nondet_injected = injected_apparent[~detected_mask]
        nondet_recovered = recovered_apparent[~detected_mask]

        # Compute errorbars: group by injected magnitude and compute std of recovered magnitudes
        unique_mags = np.unique(injected_apparent)
        recovered_means = []
        recovered_stds = []
        for mag in unique_mags:
            mask = injected_apparent == mag
            if np.sum(mask) > 0:
                recovered_means.append(np.mean(recovered_apparent[mask]))
                recovered_stds.append(np.std(recovered_apparent[mask]))
        recovered_means = np.array(recovered_means)
        recovered_stds = np.array(recovered_stds)

        # Get catalog sources for comparison
        catalog = getattr(self, 'catalog', None)
        transient_apparent = None
        zeropoint_apparent = None
        if catalog is not None and len(catalog) > 0:
            use_filter = self.input_yaml.get("imageFilter")
            if use_filter and use_filter in catalog.columns and "flux_AP" in catalog.columns:
                catalog_flux = catalog["flux_AP"].values
                catalog_apparent = catalog[use_filter].values
                # Filter out invalid values
                valid_mask = np.isfinite(catalog_flux) & np.isfinite(catalog_apparent)
                catalog_flux = catalog_flux[valid_mask]
                catalog_apparent = catalog_apparent[valid_mask]

                # Try to find transient/target source by position
                if position is not None and "x_pix" in catalog.columns and "y_pix" in catalog.columns:
                    tx, ty = position
                    catalog_x = catalog["x_pix"].values
                    catalog_y = catalog["y_pix"].values
                    # Find source closest to target position
                    distances = np.sqrt((catalog_x - tx)**2 + (catalog_y - ty)**2)
                    target_idx = np.argmin(distances)
                    if distances[target_idx] < 5.0:  # Within 5 pixels
                        transient_apparent = catalog_apparent[target_idx]

                # Get zeropoint calibration sources (sources used for zeropoint fitting)
                # These are typically the catalog sources that passed quality cuts
                zeropoint_apparent = catalog_apparent.copy()
            else:
                catalog_inst = None
                catalog_apparent = None
                zeropoint_apparent = None
        else:
            catalog_inst = None
            catalog_apparent = None
            zeropoint_apparent = None

        # Use the project-wide plotting style
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            style_path = os.path.join(dir_path, "autophot.mplstyle")
            if os.path.exists(style_path):
                plt.style.use(style_path)
        except Exception:
            pass

        plt.ioff()
        fig, ax = plt.subplots(figsize=set_size(340, 1))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

        # Plot zeropoint calibration sources (catalog sources used for zeropoint)
        if zeropoint_apparent is not None and len(zeropoint_apparent) > 0:
            ax.scatter(
                zeropoint_apparent,
                zeropoint_apparent,
                s=get_marker_size('medium'),
                c=get_okabe_color('green'),
                alpha=get_alpha('medium'),
                marker='s',
                edgecolors='none',
                label=f"Zeropoint sources [{len(zeropoint_apparent)}]",
                zorder=5,
            )

        # Plot transient/target source if available
        if transient_apparent is not None and np.isfinite(transient_apparent):
            ax.scatter(
                transient_apparent,
                transient_apparent,
                s=get_marker_size('medium'),
                c=get_okabe_color('vermilion'),
                alpha=get_alpha('dark'),
                marker='*',
                edgecolors='black',
                linewidth=0.5,
                label=f"Transient [{transient_apparent:.2f}]",
                zorder=25,
            )

        # Plot detected injected sources
        if len(detected_injected) > 0:
            ax.scatter(
                detected_injected,
                detected_recovered,
                s=get_marker_size('medium'),
                c=get_okabe_color('blue'),
                alpha=get_alpha('dark'),
                marker='o',
                edgecolors='black',
                linewidth=0.5,
                label=f"Detected injected [{len(detected_injected)}]",
                zorder=10,
            )

        # Plot non-detected injected sources
        if len(nondet_injected) > 0:
            ax.scatter(
                nondet_injected,
                nondet_recovered,
                s=get_marker_size('medium'),
                c=get_okabe_color('red'),
                alpha=get_alpha('medium'),
                marker='x',
                linewidth=1.0,
                label=f"Non-detected injected [{len(nondet_injected)}]",
                zorder=8,
            )

        # Plot 1:1 line (expected for perfect recovery)
        if len(injected_apparent) > 0:
            mag_range = np.linspace(np.min(injected_apparent), np.max(injected_apparent), 100)
            ax.plot(
                mag_range,
                mag_range,
                color=get_color('fit'),
                linestyle="--",
                lw=0.5,
                zorder=15,
                label="Expected (1:1)",
            )

        # Plot errorbars (mean recovered magnitude with std at each injected magnitude)
        if len(unique_mags) > 0:
            ax.errorbar(
                unique_mags,
                recovered_means,
                xerr=None,  # No error on injected magnitude (known value)
                yerr=recovered_stds,  # Error on recovered magnitude (measured)
                fmt='none',
                color=get_okabe_color('blue'),
                alpha=get_alpha('medium'),
                lw=0.5,
                capsize=2,
                zorder=7,
            )

        # Mark limiting magnitude as vertical red line
        if np.isfinite(inject_lmag):
            limit_apparent = inject_lmag + selected_zeropoint
            ax.axvline(
                x=limit_apparent,
                color=get_okabe_color('red'),
                linestyle="-.",
                lw=0.5,
                zorder=20,
            )
            ax.text(
                limit_apparent,
                0.95,
                f"Limiting\n{limit_apparent:.2f}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va='top',
                ha='right',
                fontsize=7,
                color=get_okabe_color('red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
            )

        # Mark transient magnitude as vertical blue line
        if transient_apparent is not None and np.isfinite(transient_apparent):
            ax.axvline(
                x=transient_apparent,
                color=get_okabe_color('blue'),
                linestyle="-",
                lw=0.5,
                zorder=20,
            )
            ax.text(
                transient_apparent,
                0.95,
                f"Transient\n{transient_apparent:.2f}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va='top',
                ha='right',
                fontsize=7,
                color=get_okabe_color('blue'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
            )

        # Labels and styling
        ax.set_xlabel("Injected Apparent Magnitude [mag]", fontsize=9)
        ax.set_ylabel("Recovered Apparent Magnitude [mag]", fontsize=9)
        ax.invert_xaxis()
        ax.invert_yaxis()

        # Set axis limits based on data range
        all_mags = np.concatenate([injected_apparent, recovered_apparent])
        if len(all_mags) > 0:
            mag_min = np.min(all_mags)
            mag_max = np.max(all_mags)
            margin = 0.5  # 0.5 mag margin
            ax.set_xlim(mag_max + margin, mag_min - margin)  # Inverted for magnitude
            ax.set_ylim(mag_max + margin, mag_min - margin)  # Inverted for magnitude
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2, fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.5, zorder=0, lw=0.5)

        fig.tight_layout()
        save_path = os.path.join(write_dir, f"InjectionRecovery_{base}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"Saved injection recovery plot to {save_path}")
