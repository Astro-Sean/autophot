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

    ``counts_ref`` **must be in e⁻** (aperture sum of the unit-flux PSF render
    after multiplying ADU by gain), matching ``Aperture.counts_AP``.  The ePSF
    model is built from ADU images, so its raw aperture integral is in ADU;
    the caller is responsible for multiplying by gain before passing here.

    ``m`` uses the same e⁻/s convention as ``mag(flux_AP)``.  The returned
    value is the dimensionless PSF flux scale factor such that the injected PSF
    carries exactly ``10^(-0.4*m) * exposure_time`` e⁻ inside the photometry
    aperture.
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
         snr_limit, beta_n, recovery_method)

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
        recovery_method,
    ) = args

    try:
        ny, nx = cutout.shape
        psf_img = _render_epsf_on_cutout(
            epsf_model, ny, nx, x_inj, y_inj, F_amp, oversampling
        )
        # Preserve no-data regions (NaNs) end-to-end.
        # - Never inject flux into invalid pixels.
        # - Never replace NaNs with finite fill values (keeps diagnostics honest
        #   and avoids bias near chip gaps / bands).
        invalid = ~np.isfinite(cutout)
        if np.any(invalid):
            psf_img = np.asarray(psf_img, dtype=float)
            psf_img[invalid] = 0.0
        new_img = cutout + psf_img
        if np.any(invalid):
            new_img = np.asarray(new_img, dtype=float)
            new_img[invalid] = np.nan

        # Guard: if the combined image is all-NaN, photometry cannot proceed.
        n_finite = int(np.count_nonzero(np.isfinite(new_img)))
        if n_finite == 0:
            return False, 0.0, np.nan

        # Compute beta using the canonical aperture-based formalism (n=3) so
        # beta thresholds remain comparable across runs/methods.
        ap = Aperture(input_yaml=input_yaml, image=new_img)
        trial_df = pd.DataFrame({"x_pix": [x_inj], "y_pix": [y_inj]})
        mres = ap.measure(
            sources=trial_df, plot=False, background_rms=background_rms, verbose=0
        )

        # noiseSky from Aperture.measure() is per-pixel RMS (not total noise)
        # beta_aperture uses sigma * sqrt(npix) internally to compute total noise
        # Validate noiseSky first before computing beta_p (beta is diagnostic only).
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

        # Recovery decision is S/N-only. Beta is still returned for optional
        # diagnostics/plots and backwards compatibility with existing outputs.
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
                # Variance model:
                #   var ~= background_rms^2 + source_poisson + readnoise^2
                # with everything expressed in the same *image units* as `data`.
                # This prevents PSF recovery from reporting overly optimistic S/N.
                if background_rms is not None:
                    rms = np.asarray(background_rms[y1:y2, x1:x2], dtype=float)
                    var = np.maximum(rms * rms, 0.0)
                else:
                    sig = float(np.nanstd(data))
                    var = np.full_like(data, max(sig * sig, 0.0), dtype=float)

                # Optional Poisson + read-noise contributions (conservative defaults).
                lim_cfg = input_yaml.get("limiting_magnitude") or {}
                include_poisson = bool(lim_cfg.get("psf_snr_include_poisson", True))
                include_readnoise = bool(lim_cfg.get("psf_snr_include_readnoise", True))

                gain_e_per_adu = float(resolve_gain_e_per_adu(None, input_yaml))
                gain_e_per_adu = gain_e_per_adu if np.isfinite(gain_e_per_adu) and gain_e_per_adu > 0 else np.nan

                if include_poisson and np.isfinite(gain_e_per_adu) and gain_e_per_adu > 0:
                    # Source model (ADU) for the stamp using the injected-normalised PSF.
                    # Use injected flux as a first-order approximation in the weights.
                    model_adu = np.maximum(float(F_amp) * np.asarray(psf1, dtype=float), 0.0)
                    var = var + (model_adu / gain_e_per_adu)

                if include_readnoise and np.isfinite(gain_e_per_adu) and gain_e_per_adu > 0:
                    rn_e = float(input_yaml.get("read_noise", 0.0))
                    rn_adu = (rn_e / gain_e_per_adu) if np.isfinite(rn_e) and rn_e > 0 else 0.0
                    if rn_adu > 0:
                        var = var + (rn_adu * rn_adu)

                var = np.maximum(var, 1e-30)

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
                            # Use signed S/N: a negative flux_hat (source on sky hole)
                            # must NOT pass the detection threshold >= snr_limit > 0.
                            snr_val = float(flux_hat) / flux_err
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
                        snr_val = np.abs(flux_hat) / flux_err
                except Exception:
                    snr_val = np.nan
            except Exception:
                snr_val = np.nan

        # For PSF / EMCEE methods we require method-consistent S/N.
        # Do not silently fall back to AP S/N here, which can make limits too deep.
        if method in ("PSF", "EMCEE"):
            recovered_flux = flux_hat if np.isfinite(flux_hat) else np.nan
            if not np.isfinite(snr_val):
                # snr_val is NaN when the WLS/MCMC fit failed (e.g. too few valid
                # pixels in the stamp).  Return non-detection rather than crashing.
                # Note: this can happen for sites near chip edges/gaps where the PSF
                # stamp footprint is larger than the aperture disk used by site filtering.
                # Such sites should ideally be rejected by an upstream stamp-validity
                # check; treat this as a conservative non-detection.
                return False, beta_p, recovered_flux
        else:
            # AP recovery: aperture S/N is the method-consistent detection statistic.
            if not np.isfinite(snr_val):
                try:
                    snr_val = float(mres["SNR"].iloc[0])
                except Exception:
                    return False, beta_p, np.nan
            recovered_flux = float(mres["flux_AP"].iloc[0])

        effective_snr_limit = float(snr_limit) if snr_limit is not None else 3.0
        det_snr = np.isfinite(snr_val) and (snr_val >= effective_snr_limit)
        det_flux = np.isfinite(recovered_flux) and (float(recovered_flux) > 0.0)
        return (det_snr and det_flux), beta_p, recovered_flux

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
        _return_details: bool = False,
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
        detection_limit : optional S/N detection threshold. If null, defaults to S/N >= 3.0.
        detection_cutoff: legacy beta threshold (unused; retained for backwards compatibility)
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
            lim_cfg = self.input_yaml.get("limiting_magnitude") or {}
            # Beta-limit support is retained for backwards compatibility, but limiting
            # magnitude detection is now S/N-only (see _injection_worker).
            if detection_cutoff is None:
                detection_cutoff = float(lim_cfg.get("beta_limit", 0.5))
            effective_snr_limit = float(detection_limit) if detection_limit is not None else 3.0
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
            
            # For difference images and locally background-subtracted stamps, the local
            # annulus median can legitimately be negative. Flooring the local background
            # to 0 (the default in some configs) biases flux positive and can make every
            # candidate look like a real source (|S/N|>3 everywhere).
            #
            # Default for limiting-magnitude injection: do NOT floor negative local
            # backgrounds. This preserves the empirical noise statistics.
            lim_cfg = self.input_yaml.get("limiting_magnitude") or {}
            phot_cfg_local = local_input_yaml.get("photometry") or {}
            floor_local_bkg = bool(
                lim_cfg.get("enforce_nonnegative_local_background_for_injection", False)
            )
            phot_cfg_local["enforce_nonnegative_local_background"] = bool(floor_local_bkg)
            local_input_yaml["photometry"] = phot_cfg_local
            
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

            def _filter_aperture_validity(
                df: pd.DataFrame,
                image: np.ndarray,
                *,
                aperture_radius_pix: float,
            ) -> pd.DataFrame:
                """
                Keep sites whose *aperture footprint* has no invalid pixels.

                Policy:
                - Reject if the aperture circle intersects NaN/Inf pixels.
                - Optionally reject exact zeros (common SWarp no-coverage fill).
                """
                if df is None or len(df) == 0:
                    return df
                if image is None or np.ndim(image) != 2:
                    return df
                Hn, Wn = int(image.shape[0]), int(image.shape[1])
                r = float(max(1.0, aperture_radius_pix))
                r_int = int(np.ceil(r))
                imgf = np.asarray(image, dtype=float)
                avoid_zero_pixels = bool(lim_cfg.get("inject_avoid_zero_pixels", True))

                keep = np.zeros(len(df), dtype=bool)
                xs = np.asarray(df["x_pix"], dtype=float)
                ys = np.asarray(df["y_pix"], dtype=float)
                for i, (x, y) in enumerate(zip(xs, ys)):
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    xi = int(np.round(x))
                    yi = int(np.round(y))
                    if xi < 0 or yi < 0 or xi >= Wn or yi >= Hn:
                        continue
                    x0 = max(0, xi - r_int)
                    x1 = min(Wn, xi + r_int + 1)
                    y0 = max(0, yi - r_int)
                    y1 = min(Hn, yi + r_int + 1)
                    stamp = imgf[y0:y1, x0:x1]
                    if stamp.size == 0:
                        continue
                    yy, xx = np.indices(stamp.shape)
                    cx0 = float(x) - float(x0)
                    cy0 = float(y) - float(y0)
                    in_ap = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= r**2
                    vals = stamp[in_ap]
                    if vals.size == 0:
                        continue
                    ok = np.isfinite(vals)
                    if avoid_zero_pixels:
                        ok &= (vals != 0.0)
                    if bool(np.all(ok)):
                        keep[i] = True
                return df[keep].copy()

            def _filter_annulus_finite_support(
                df: pd.DataFrame,
                image: np.ndarray,
                *,
                annulus_in_pix: float,
                annulus_out_pix: float,
            ) -> pd.DataFrame:
                """
                Allow NaNs in the annulus, but require enough finite pixels to
                estimate local background/noise.
                """
                if df is None or len(df) == 0:
                    return df
                if image is None or np.ndim(image) != 2:
                    return df
                imgf = np.asarray(image, dtype=float)
                avoid_zero_pixels = bool(lim_cfg.get("inject_avoid_zero_pixels", True))

                min_frac = float(lim_cfg.get("inject_min_finite_annulus_frac", 0.05))
                min_frac = float(max(0.0, min(1.0, min_frac)))
                min_pix = int(lim_cfg.get("inject_min_finite_annulus_pix", 10))
                min_pix = int(max(0, min_pix))

                keep = np.zeros(len(df), dtype=bool)
                for i, (x, y) in enumerate(zip(df["x_pix"].to_numpy(float), df["y_pix"].to_numpy(float))):
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    ann = CircularAnnulus(
                        (float(x), float(y)),
                        r_in=float(annulus_in_pix),
                        r_out=float(annulus_out_pix),
                    )
                    try:
                        vals = ann.to_mask(method="center").get_values(imgf)
                    except Exception:
                        continue
                    if vals is None:
                        continue
                    vals = np.asarray(vals, dtype=float)
                    total = int(vals.size)
                    if total <= 0:
                        continue
                    # TOLERANT CHECK: Annulus can have some NaNs, but needs minimum valid pixels
                    # Require at least 50% of annulus pixels to be valid (same as aperture.py)
                    annulus_valid_fraction = 0.5
                    ok = np.isfinite(vals)
                    if avoid_zero_pixels:
                        ok &= (vals != 0.0)
                    n_ok = int(np.count_nonzero(ok))
                    if n_ok >= min_pix and (n_ok / float(total)) >= annulus_valid_fraction:
                        keep[i] = True
                return df[keep].copy()

            def _filter_pixel_statistics(
                df: pd.DataFrame,
                image: np.ndarray,
                *,
                aperture_radius_pix: float,
                rms_map: np.ndarray | None = None,
                max_variance_ratio: float = 3.0,
                max_abs_mean_sigma: float = 2.5,
            ) -> pd.DataFrame:
                """
                Reject sites whose pixel statistics indicate contamination from
                bright-star template subtraction residuals.

                Two independent tests are applied (each configurable via
                ``limiting_magnitude`` YAML keys):

                1. **Variance test** (``inject_max_variance_ratio``):
                   Compute the sample variance of the pixels inside the aperture
                   disk.  If an RMS map is available, compare against the median
                   expected variance from that map.  Otherwise compare against the
                   cutout-wide background variance estimated from the annulus
                   immediately outside the aperture.  A residual halo has elevated
                   pixel-to-pixel scatter even when its mean is near zero.
                   Reject if:  var_ap / var_ref  >  max_variance_ratio

                2. **Mean-bias test** (``inject_max_abs_mean_sigma``):
                   Estimate the local background mean from the annulus pixels
                   (same annulus used for photometry).  Reject if the absolute
                   mean offset in the aperture exceeds ``max_abs_mean_sigma``
                   times the expected per-pixel RMS.  This catches bright
                   positive/negative star-subtraction pedestals.
                   Reject if:  |mean_ap - mean_annulus| / sigma_ref  >  max_abs_mean_sigma

                Both tests must PASS for a site to be kept.  Sites that fail
                either test are dropped with a debug-level log entry.

                Thresholds can be relaxed (or tests disabled) via config:
                  inject_max_variance_ratio: float (default 3.0; 0 = disable)
                  inject_max_abs_mean_sigma: float (default 2.5; 0 = disable)
                """
                if df is None or len(df) == 0:
                    return df
                if image is None or np.ndim(image) != 2:
                    return df

                # Read configurable thresholds (allow YAML override)
                _var_ratio_thr = float(
                    lim_cfg.get("inject_max_variance_ratio", max_variance_ratio)
                )
                _mean_sigma_thr = float(
                    lim_cfg.get("inject_max_abs_mean_sigma", max_abs_mean_sigma)
                )
                _use_var_test = _var_ratio_thr > 0.0
                _use_mean_test = _mean_sigma_thr > 0.0

                if not (_use_var_test or _use_mean_test):
                    return df

                # Minimum fraction of annulus pixels that must be finite to
                # produce a reliable background/noise estimate.
                _annulus_valid_fraction = 0.5

                Hn, Wn = int(image.shape[0]), int(image.shape[1])
                imgf = np.asarray(image, dtype=float)
                r = float(max(1.0, aperture_radius_pix))
                r_int = int(np.ceil(r))
                # Annulus for local background reference: 1.5*r_in to 3.0*r_out
                ann_in = r * 1.5
                ann_out = r * 3.0

                # Global background variance fallback: sigma-clipped MAD of finite pixels
                _global_finite = imgf[np.isfinite(imgf)]
                if _global_finite.size >= 20:
                    _global_mad = float(np.nanmedian(np.abs(_global_finite - np.nanmedian(_global_finite))))
                    _global_sigma = max(_global_mad * 1.4826, 1e-30)
                    _global_var = _global_sigma ** 2
                else:
                    _global_var = 1.0

                keep = np.zeros(len(df), dtype=bool)
                n_drop_var = 0
                n_drop_mean = 0
                xs = np.asarray(df["x_pix"], dtype=float)
                ys = np.asarray(df["y_pix"], dtype=float)

                for i, (x, y) in enumerate(zip(xs, ys)):
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    xi = int(np.round(x))
                    yi = int(np.round(y))
                    if xi < 0 or yi < 0 or xi >= Wn or yi >= Hn:
                        continue

                    # --- Aperture pixels ---
                    x0 = max(0, xi - r_int)
                    x1 = min(Wn, xi + r_int + 1)
                    y0 = max(0, yi - r_int)
                    y1 = min(Hn, yi + r_int + 1)
                    stamp = imgf[y0:y1, x0:x1]
                    if stamp.size == 0:
                        continue
                    yy, xx = np.indices(stamp.shape)
                    cx0 = float(x) - float(x0)
                    cy0 = float(y) - float(y0)
                    in_ap = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= r ** 2
                    ap_vals = stamp[in_ap]
                    ap_vals = ap_vals[np.isfinite(ap_vals)]
                    if ap_vals.size < 4:
                        # Too few pixels to test — treat as valid (NaN filter already ran)
                        keep[i] = True
                        continue
                    mean_ap = float(np.mean(ap_vals))
                    var_ap = float(np.var(ap_vals))

                    # --- Reference variance ---
                    # Prefer RMS map if available (most accurate for difference images)
                    if rms_map is not None:
                        rms_stamp = np.asarray(rms_map, dtype=float)[y0:y1, x0:x1]
                        rms_ap_vals = rms_stamp[in_ap]
                        rms_ap_vals = rms_ap_vals[np.isfinite(rms_ap_vals) & (rms_ap_vals > 0)]
                        if rms_ap_vals.size >= 2:
                            var_ref = float(np.median(rms_ap_vals) ** 2)
                        else:
                            var_ref = _global_var
                        sigma_ref = float(np.sqrt(max(var_ref, 1e-60)))
                        # Annulus mean still needed for mean-bias test
                        try:
                            ann = CircularAnnulus(
                                (float(x), float(y)), r_in=ann_in, r_out=ann_out
                            )
                            ann_vals = ann.to_mask(method="center").get_values(imgf)
                            if ann_vals is not None:
                                ann_vals = np.asarray(ann_vals, dtype=float)
                                ann_vals = ann_vals[np.isfinite(ann_vals)]
                            # Require at least _annulus_valid_fraction of annulus pixels to be finite.
                            if ann_vals is not None and ann_vals.size >= 4:
                                total_annulus = ann_vals.size
                                if ann_vals.size >= (total_annulus * _annulus_valid_fraction):
                                    mean_annulus = float(np.median(ann_vals))
                                else:
                                    mean_annulus = 0.0
                            else:
                                mean_annulus = 0.0
                        except Exception:
                            mean_annulus = 0.0
                    else:
                        # Annulus-based reference
                        try:
                            ann = CircularAnnulus(
                                (float(x), float(y)), r_in=ann_in, r_out=ann_out
                            )
                            ann_vals = ann.to_mask(method="center").get_values(imgf)
                            if ann_vals is not None:
                                ann_vals = np.asarray(ann_vals, dtype=float)
                                ann_vals = ann_vals[np.isfinite(ann_vals)]
                            # Require at least _annulus_valid_fraction of annulus pixels to be finite.
                            if ann_vals is not None:
                                total_annulus = ann_vals.size
                                if ann_vals.size >= (total_annulus * _annulus_valid_fraction):
                                    mean_annulus = float(np.median(ann_vals))
                                    mad = float(np.median(np.abs(ann_vals - mean_annulus)))
                                    sigma_ref = max(mad * 1.4826, 1e-30)
                                    var_ref = sigma_ref ** 2
                                else:
                                    mean_annulus = 0.0
                                    var_ref = _global_var
                                    sigma_ref = float(np.sqrt(max(var_ref, 1e-60)))
                            else:
                                mean_annulus = 0.0
                                var_ref = _global_var
                                sigma_ref = float(np.sqrt(max(var_ref, 1e-60)))
                        except Exception:
                            mean_annulus = 0.0
                            var_ref = _global_var
                            sigma_ref = float(np.sqrt(max(var_ref, 1e-60)))

                    # --- Variance test ---
                    if _use_var_test and var_ref > 1e-60:
                        ratio = var_ap / var_ref
                        if ratio > _var_ratio_thr:
                            n_drop_var += 1
                            logger.debug(
                                "Site (%.1f,%.1f) rejected: var_ap/var_ref=%.2f > %.2f (elevated pixel scatter — likely star-subtraction residual).",
                                x, y, ratio, _var_ratio_thr,
                            )
                            continue

                    # --- Mean-bias test ---
                    if _use_mean_test and sigma_ref > 1e-60:
                        bias = abs(mean_ap - mean_annulus) / sigma_ref
                        if bias > _mean_sigma_thr:
                            n_drop_mean += 1
                            logger.debug(
                                "Site (%.1f,%.1f) rejected: |mean_ap - mean_annulus|/sigma=%.2f > %.2f (significant local mean bias — likely star-subtraction pedestal).",
                                x, y, bias, _mean_sigma_thr,
                            )
                            continue

                    keep[i] = True

                n_drop = n_drop_var + n_drop_mean
                return df[keep].copy()

            def _robust_site_snr(df: pd.DataFrame) -> pd.DataFrame:
                """
                Ensure candidate-site rows have a finite `SNR` for filtering.

                `Aperture.measure()` can yield NaN SNR for some sites even when they are
                visually quiet. For site selection only, compute a fallback:

                    SNR ~= |flux_AP| / (noiseSky * sqrt(area))

                and drop rows where the underlying inputs are non-finite.
                """
                if df is None or len(df) == 0:
                    return df
                if "SNR" not in df.columns:
                    df["SNR"] = np.nan
                flux_ap = np.asarray(df.get("flux_AP", np.nan), dtype=float)
                noise = np.asarray(df.get("noiseSky", np.nan), dtype=float)
                area = np.asarray(df.get("area", np.nan), dtype=float)
                snr = np.asarray(df.get("SNR", np.nan), dtype=float)
                valid = (
                    np.isfinite(flux_ap)
                    & np.isfinite(noise)
                    & (noise > 0)
                    & np.isfinite(area)
                    & (area > 0)
                )
                snr_fb = np.full_like(flux_ap, np.nan, dtype=float)
                snr_fb[valid] = np.abs(flux_ap[valid]) / (noise[valid] * np.sqrt(area[valid]))
                use_fb = (~np.isfinite(snr)) & np.isfinite(snr_fb)
                if np.any(use_fb):
                    df.loc[use_fb, "SNR"] = snr_fb[use_fb]
                n_before = int(len(df))
                df = df[valid].copy()
                n_drop = int(n_before - len(df))
                return df

            def _calculate_annulus_statistics(
                x: float,
                y: float,
                image: np.ndarray,
                annulus_in_pix: float,
                annulus_out_pix: float,
            ) -> tuple[float, float]:
                """
                Calculate annulus statistics (mean, std) for a given position.
                
                Parameters
                ----------
                x, y : float
                    Center position in pixels
                image : np.ndarray
                    Image array
                annulus_in_pix, annulus_out_pix : float
                    Inner and outer radius of annulus in pixels
                    
                Returns
                -------
                mean, std : tuple[float, float]
                    Mean and standard deviation of annulus pixels (NaN if invalid)
                """
                try:
                    ann = CircularAnnulus((x, y), r_in=annulus_in_pix, r_out=annulus_out_pix)
                    ann_vals = ann.to_mask(method="center").get_values(image)
                    if ann_vals is not None:
                        ann_vals = np.asarray(ann_vals, dtype=float)
                        ann_vals = ann_vals[np.isfinite(ann_vals)]
                        if ann_vals.size >= 4:
                            mean = float(np.median(ann_vals))
                            # Use robust std estimator (MAD scaled to normal distribution)
                            mad = np.median(np.abs(ann_vals - mean))
                            std = float(1.4826 * mad)
                            return mean, std
                except Exception:
                    pass
                return np.nan, np.nan

            # If aperture_radius equals fwhm (default fallback), try to calculate optimum
            if configured_radius == fwhm:
                try:
                    # Detect sources in the cutout to use for optimum radius calculation
                    from photutils.detection import DAOStarFinder
                    daofind = DAOStarFinder(fwhm=fwhm, threshold=5.0 * np.nanstd(cutout))
                    sources = daofind(cutout)
                    if sources is not None and len(sources) >= 5:
                        _xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
                        _ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
                        sources_df = pd.DataFrame({
                            "x_pix": sources[_xcol],
                            "y_pix": sources[_ycol]
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
                    else:
                        pass  # Insufficient sources for optimum radius, using configured
                except Exception as e:
                    logger.warning(f"Failed to calculate optimum aperture radius: {e}, using configured")
            
            # Get final aperture radius from local_input_yaml
            phot_cfg_local = local_input_yaml.get("photometry", {})
            aperture_radius_local = float(phot_cfg_local.get("aperture_radius", fwhm))

            # Annulus outer radius (match Aperture.measure defaults) for NaN clearance filtering.
            try:
                crowded_local = bool(phot_cfg_local.get("crowded", False))
            except Exception:
                crowded_local = False
            gap_fwhm = float(
                phot_cfg_local.get("annulus_gap_fwhm", 0.5 if crowded_local else 0.75)
            )
            width_fwhm = float(
                phot_cfg_local.get("annulus_width_fwhm", 1.5 if crowded_local else 2.0)
            )
            annulus_in_local = float(
                np.ceil(aperture_radius_local + gap_fwhm * float(fwhm))
            )
            annulus_out_local = float(
                np.ceil(annulus_in_local + width_fwhm * float(fwhm))
            )
            # Exclusion zone: two aperture diameters + one FWHM buffer
            # Ensures zero overlap between transient aperture and injection aperture,
            # plus a PSF-wing buffer so transient flux cannot bias the injected recovery.
            target_exclusion_r = 2.0 * aperture_radius_local + fwhm
            
            # Define injection radii early (needed for initial guess)
            fwhm_px = float(self.input_yaml.get("fwhm", 3.0))
            r_min = float(lim_cfg.get("inject_min_radius_fwhm", 2.0)) * fwhm_px
            r_max = float(lim_cfg.get("inject_max_radius_fwhm", 6.0)) * fwhm_px
            r_base = float(lim_cfg.get("inject_source_location", 3.0)) * fwhm_px
            
            # Enforce r_min >= target_exclusion_r so sites are never generated inside exclusion zone
            r_min = max(r_min, target_exclusion_r)
            # Ensure valid sampling range; if exclusion zone consumes the whole
            # injection annulus, expand r_max to leave at least one FWHM of room.
            if r_min >= r_max:
                r_max = r_min + max(fwhm_px, 2.0)
                        
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
                
                                
                # Update the scale used for cutout extraction
                scale_used = new_scale
            
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
                    probe_df = _filter_aperture_validity(
                        probe_df, cutout, aperture_radius_pix=float(aperture_radius_local)
                    )
                    probe_df = _filter_annulus_finite_support(
                        probe_df,
                        cutout,
                        annulus_in_pix=float(annulus_in_local),
                        annulus_out_pix=float(annulus_out_local),
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
                                "Injected limiting magnitude: initial guess from %.1f*sigma*sqrt(Npix) = %.2f (sigma=%.3g, Npix=%.1f, N=%d)",
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

            # Calibration: integrate the unit-flux PSF inside the aperture disk WITHOUT
            # local background subtraction.  Running Aperture.measure() on a pure PSF
            # image causes its local annulus estimator to subtract PSF-wing flux,
            # making counts_ref smaller than the true enclosed flux and biasing
            # flux_for_mag() to inject too much signal -> limit is spuriously shallow.
            # Using exact pixel-fraction photometry on a zero-background image avoids this.
            _psf_ap_obj = CircularAperture(
                (float(cx), float(cy)), r=float(aperture_radius_local)
            )
            _psf_phot = _psf_ap_obj.do_photometry(psf_unit, method="exact")
            counts_ref_adu = float(_psf_phot[0][0])  # integrated PSF flux in aperture (ADU)
            # Convert to e⁻ to match Aperture.measure which operates on image*gain.
            # Without this, _flux_for_mag_cached divides e⁻ by ADU giving F_amp that
            # is gain× too large, making every injected source gain× too bright and the
            # recovered limiting magnitude ~2.5*log10(gain) mag spuriously deep.
            counts_ref = counts_ref_adu * float(_gain_canon)  # now in e⁻
            # F_ref = counts_ref (e⁻) / exposure_time → e⁻/s, same units as flux_AP
            F_ref = counts_ref / float(local_input_yaml["exposure_time"])
            exposure_time = float(local_input_yaml["exposure_time"])

            # Guard: check if calibration failed
            if not (np.isfinite(counts_ref) and counts_ref > 0
                    and np.isfinite(F_ref) and F_ref > 0):
                logger.error(
                    f"PSF calibration failed: counts_ref_adu={counts_ref_adu:.4e}, "
                    f"counts_ref(e-)={counts_ref:.4e}, F_ref={F_ref:.4e} "
                    f"(both must be finite and positive). "
                    f"PSF sum={np.sum(psf_unit):.4e}, shape={psf_unit.shape}, oversampling={oversampling}, "
                    f"gain={_gain_canon:.4g} e/ADU"
                )
                return np.nan

            # Memoised so repeated calls at the same magnitude are free.
            def flux_for_mag(m: float) -> float:
                """Return ePSF flux parameter to inject for instrumental magnitude m."""
                return _flux_for_mag_cached(m, counts_ref, exposure_time)

            # Sanity round-trip: flux_for_mag(mag(F_ref)) should equal 1.0.
            mag_at_unit_flux = float(mag(float(F_ref)))
            f_roundtrip = float(flux_for_mag(mag_at_unit_flux))
            if np.isfinite(f_roundtrip) and abs(f_roundtrip - 1.0) > 1e-6:
                logger.warning(
                    "PSF flux calibration round-trip differs from 1.0 (got %.8f). "
                    "Check oversampling and that flux_for_mag uses the same aperture radius as recovery.",
                    f_roundtrip,
                )

            # =================================================================
            # Choose quiet injection sites
            # =================================================================
            snr_limit = float(detection_limit) if detection_limit is not None else None
            effective_snr_limit = float(snr_limit) if snr_limit is not None else 3.0
            recovery_method = str(lim_cfg.get("recovery_method", "PSF")).strip().upper()
            # "AUTO" is resolved in main.py to AP vs PSF from do_aperture_ONLY; if unset, prefer PSF.
            if recovery_method in {"AUTO", "DEFAULT", "MATCH_TRANSIENT", "MATCH_TARGET"}:
                logger.warning(
                    "limiting_magnitude.recovery_method=%s was not pre-resolved; using PSF. Set recovery_method explicitly or run from main (auto).",
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

            completeness_solver = str(lim_cfg.get("completeness_solver", "bisect")).strip().lower()
            if completeness_solver not in {"bisect", "logistic_emcee"}:
                completeness_solver = "bisect"

            # emcee recovery should run serial (each trial is expensive and models
            # are not guaranteed to pickle cleanly across platforms).
            if recovery_method == "EMCEE":
                n_jobs = 1
            # Beta is still computed with the canonical n=3 aperture formalism for
            # diagnostics/plots, but it is not used for detection gating.
            beta_n = float(BETA_APERTURE_SIGMA_N)
            # Monte Carlo settings for injection/recovery.
            # Too few sites/jitters makes the completeness curve look jagged.
            sourceNum = int(lim_cfg.get("injection_n_sites", 30))
            sourceNum = max(8, min(sourceNum, 200))
            redo_default = int(lim_cfg.get("injection_jitter_repetitions", 5))
            redo_default = max(1, min(redo_default, 20))
            injection_df = pd.DataFrame()

            # -----------------------------------------------------------------
            # Fully sample the environment around the transient and choose the
            # quietest sites (lowest |SNR|) for injection.
            # -----------------------------------------------------------------
            n_candidates = int(lim_cfg.get("inject_candidate_n_sites", 2000))
            n_candidates = max(200, min(n_candidates, 20000))
            n_quiet = int(lim_cfg.get("inject_quiet_n_sites", 100))
            n_quiet = max(10, min(n_quiet, 500))

            # Sample uniformly in area within [r_min, r_max].
            r_min_with_jitter = float(r_min) + 1.0
            r_min_with_jitter = min(r_min_with_jitter, float(r_max))
            theta = self._rng.random(n_candidates) * (2.0 * np.pi)
            rr = (
                np.sqrt(self._rng.random(n_candidates))
                * (float(r_max) - float(r_min_with_jitter))
                + float(r_min_with_jitter)
            )
            cand_df = pd.DataFrame(
                {
                    "x_pix": cutout_cx + rr * np.cos(theta),
                    "y_pix": cutout_cy + rr * np.sin(theta),
                }
            )

            # Apply geometric constraints first.
            n0 = int(len(cand_df))
            cand_df = _exclude_target_overlap(
                cand_df, cutout_cx, cutout_cy, target_exclusion_r
            )
            n1 = int(len(cand_df))
            cand_df = _filter_edge_clearance(cand_df, W, H, edge_margin)
            n2 = int(len(cand_df))

            # Validity: aperture must be fully finite; annulus can contain NaNs but
            # must have enough finite pixels to estimate background/noise.
            cand_df = _filter_aperture_validity(
                cand_df, cutout, aperture_radius_pix=float(aperture_radius_local)
            )
            n3 = int(len(cand_df))
            cand_df = _filter_annulus_finite_support(
                cand_df,
                cutout,
                annulus_in_pix=float(annulus_in_local),
                annulus_out_pix=float(annulus_out_local),
            )
            n4 = int(len(cand_df))

            # Pixel-statistics filter: reject sites contaminated by bright-star
            # template subtraction residuals (elevated local variance or biased mean).
            cand_df = _filter_pixel_statistics(
                cand_df,
                cutout,
                aperture_radius_pix=float(aperture_radius_local),
                rms_map=cutout_rms,
            )
            n5 = int(len(cand_df))

            logger.info(f"Found {n5} valid candidate sites for injection")

            if len(cand_df) == 0:
                logger.warning(
                    "No valid candidate sites after NaN/edge/exclusion filtering; cannot run injected limiting magnitude."
                )
                return np.nan

            if use_quiet_sites:
                # Calculate target annulus statistics to use as reference for site selection
                target_mean, target_std = _calculate_annulus_statistics(
                    cutout_cx, cutout_cy, cutout,
                    float(annulus_in_local), float(annulus_out_local)
                )
                logger.info(
                    "Target annulus statistics: mean=%.3f, std=%.3f",
                    target_mean, target_std
                )
                
                # Stage 1: Measure S/N at each candidate site (no jitter).
                ini_ap = Aperture(input_yaml=local_input_yaml, image=cutout)
                cand_df = ini_ap.measure(
                    sources=cand_df,
                    plot=False,
                    background_rms=cutout_rms,
                    verbose=0,
                )
                cand_df = _robust_site_snr(cand_df)
                
                # Calculate annulus statistics for each candidate site
                ann_means = []
                ann_stds = []
                for _, row in cand_df.iterrows():
                    mean, std = _calculate_annulus_statistics(
                        row["x_pix"], row["y_pix"], cutout,
                        float(annulus_in_local), float(annulus_out_local)
                    )
                    ann_means.append(mean)
                    ann_stds.append(std)
                cand_df = cand_df.assign(
                    _annulus_mean=ann_means,
                    _annulus_std=ann_stds
                )
                
                # Score candidates by combined metric:
                # 1. Primary: |SNR| (quietness) - lower is better
                # 2. Secondary: Similarity to target annulus statistics
                snr_col = np.asarray(cand_df.get("SNR", np.nan), dtype=float)
                mean_col = np.asarray(cand_df.get("_annulus_mean", np.nan), dtype=float)
                std_col = np.asarray(cand_df.get("_annulus_std", np.nan), dtype=float)
                
                # Calculate similarity scores (lower is more similar)
                mean_diff = np.abs(mean_col - target_mean) if np.isfinite(target_mean) else np.zeros_like(mean_col)
                std_diff = np.abs(std_col - target_std) if np.isfinite(target_std) else np.zeros_like(std_col)
                
                # Normalize differences by target values (relative difference)
                mean_sim = mean_diff / (np.abs(target_mean) + 1e-6) if np.isfinite(target_mean) and target_mean != 0 else mean_diff
                std_sim = std_diff / (np.abs(target_std) + 1e-6) if np.isfinite(target_std) and target_std != 0 else std_diff
                
                # Combined score: weighted sum of SNR and similarity
                # Weight SNR more heavily (0.7) but still consider similarity (0.3)
                similarity_weight = 0.3
                snr_weight = 0.7
                combined_score = snr_weight * np.abs(snr_col) + similarity_weight * (mean_sim + std_sim)
                
                cand_df = cand_df.assign(_combined_score=combined_score)
                cand_df = cand_df[np.isfinite(cand_df["_combined_score"])].copy()
                
                if len(cand_df) == 0:
                    logger.warning(
                        "All candidate sites have non-finite scores; cannot find injection sites."
                    )
                    return np.nan

                # Select the n_quiet candidates with best combined score (quiet + similar to target)
                cand_df = cand_df.sort_values("_combined_score", ascending=True)
                # Apply SNR filter as a hard constraint (must be quiet enough)
                quiet_mask = np.abs(snr_col) <= float(effective_snr_limit)
                cand_quiet = cand_df[quiet_mask].copy()
                stage1_chosen = cand_quiet.head(n_quiet) if len(cand_quiet) >= n_quiet else cand_df.head(n_quiet)
                
                # Log statistics about the selected sites
                if len(stage1_chosen) > 0:
                    selected_mean = np.mean(np.abs(stage1_chosen["SNR"]))
                    selected_mean_sim = np.mean(np.abs(stage1_chosen["_annulus_mean"] - target_mean)) if np.isfinite(target_mean) else 0
                    selected_std_sim = np.mean(np.abs(stage1_chosen["_annulus_std"] - target_std)) if np.isfinite(target_std) else 0
                    logger.info(
                        "Stage 1 site selection: sampled %d candidates -> selected %d sites with best combined score (|S/N|<=%.3g).",
                        int(n_candidates),
                        int(len(stage1_chosen)),
                        float(effective_snr_limit),
                    )
                    logger.info(
                        "Selected sites: mean |S/N|=%.3f, mean similarity (mean=%.3f, std=%.3f)",
                        selected_mean, selected_mean_sim, selected_std_sim
                    )
                else:
                    logger.warning(
                        "No candidate sites passed the S/N filter (|S/N|<=%.3g); cannot find injection sites.",
                        float(effective_snr_limit),
                    )

                # Stage 2: Jitter the selected candidates, measure S/N, apply K-means spatial
                # uniformity selection on the quietest jittered positions.
                _jitter_rng = np.random.default_rng(42)
                _jitter_dx = _jitter_rng.uniform(-0.5, 0.5, (len(stage1_chosen), redo_default))
                _jitter_dy = _jitter_rng.uniform(-0.5, 0.5, (len(stage1_chosen), redo_default))
                jittered_rows = []
                for i, (x0, y0) in enumerate(zip(stage1_chosen["x_pix"], stage1_chosen["y_pix"])):
                    for j in range(redo_default):
                        jittered_rows.append({
                            "x_pix": x0 + _jitter_dx[i, j],
                            "y_pix": y0 + _jitter_dy[i, j],
                        })
                jittered_df = pd.DataFrame(jittered_rows)
                jittered_df = ini_ap.measure(
                    sources=jittered_df,
                    plot=False,
                    background_rms=cutout_rms,
                    verbose=0,
                )
                jittered_df = _robust_site_snr(jittered_df)
                jittered_df["_abs_snr"] = np.abs(jittered_df["SNR"])
                jittered_df = jittered_df[np.isfinite(jittered_df["_abs_snr"])].copy()
                jittered_df = jittered_df.sort_values("_abs_snr", ascending=True)
                pool_size = min(3 * n_quiet, len(jittered_df))
                pool = jittered_df.head(pool_size).copy()

                try:
                    from sklearn.cluster import KMeans
                    coords = np.column_stack([pool["x_pix"].values, pool["y_pix"].values])
                    kmeans = KMeans(n_clusters=n_quiet, random_state=42, n_init=10)
                    kmeans.fit(coords)
                    chosen_indices = []
                    for center in kmeans.cluster_centers_:
                        distances = np.sum((coords - center) ** 2, axis=1)
                        chosen_indices.append(np.argmin(distances))
                    jittered_chosen = pool.iloc[chosen_indices].copy()
                    logger.info(
                        "Stage 2 jittered quiet selection: %d candidates x %d jitters -> %d jittered -> %d spatially uniform quiet sites via K-means from pool of %d.",
                        int(len(stage1_chosen)), int(redo_default),
                        int(len(jittered_df)), int(len(jittered_chosen)), int(pool_size),
                    )
                except Exception as exc:
                    logger.warning(
                        "K-means spatial selection failed (%s); falling back to simple selection.",
                        str(exc),
                    )
                    jittered_chosen = pool.head(n_quiet)
                    logger.info(
                        "Stage 2 jittered quiet selection: %d candidates x %d jitters -> using lowest %d (K-means fallback).",
                        int(len(stage1_chosen)), int(redo_default), int(len(jittered_chosen)),
                    )
                injection_df = jittered_chosen[["x_pix", "y_pix"]].reset_index(drop=True)

            else:
                # Representative-site mode (inject_use_quiet_sites=False):
                # Use a spatially uniform random draw from the full candidate pool.
                # This gives a limit representative of the actual local background,
                # rather than cherry-picked quiet patches which bias the depth faint.
                n_draw = min(n_quiet, len(cand_df))
                injection_df = cand_df[["x_pix", "y_pix"]].sample(
                    n=n_draw, random_state=42, replace=False
                ).reset_index(drop=True)
                
            # Use only these jittered positions for injection trials.
            sourceNum = int(len(injection_df))

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

            # injection_df contains jittered positions from quiet-site selection
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

            # injection_df already contains jittered positions from quiet-site selection,
            # so x_pix_arr/y_pix_arr are the final trial positions (no further jittering needed).
            _x_inj_all = x_pix_arr
            _y_inj_all = y_pix_arr
            
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
                Inject at *m* at all pre-jittered sites and return summary statistics.

                Jittering is done during quiet-site selection, so the positions in
                _x_inj_all/_y_inj_all are already the final trial positions.

                To keep memory bounded during bracketing/bisection, we cache only
                scalars per magnitude (detection rate + median beta + median
                recovered flux). If `return_flags=True`, we additionally return
                the per-trial detection flags (needed for the optional logistic
                fit), but we still avoid caching large per-trial arrays.
                """
                cache_key = f"{round(m, 4)}"
                if cache_key in _trial_cache:
                    cached = _trial_cache[cache_key]
                    if return_flags:
                        return cached
                    return cached[:3]

                F = flux_for_mag(m)
                x_inj_all = _x_inj_all
                y_inj_all = _y_inj_all

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

                # Per-trial progress line with visual completeness bar.
                _rate = float(det_flags.mean()) if len(det_flags) else 0.0
                _n_det = int(det_flags.sum())
                _n_tot = len(det_flags)
                _bar_width = 20
                logger.info(
                    "  inject m=%+7.3f | %5.1f%%  (%d/%d detected)",
                    m, 100.0 * _rate, _n_det, _n_tot,
                )

                det_rate = float(det_flags.mean()) if len(det_flags) else 0.0
                beta_med = float(np.nanmedian(betas)) if betas.size else np.nan
                flux_med = float(np.nanmedian(recovered_fluxes)) if recovered_fluxes.size else np.nan

                if return_flags:
                    # Cache only the scalars; return flags (uncached) for caller use.
                    _trial_cache[cache_key] = (det_rate, beta_med, flux_med)
                    return det_rate, beta_med, flux_med, det_flags

                _trial_cache[cache_key] = (det_rate, beta_med, flux_med)
                return det_rate, beta_med, flux_med

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
                    f_faint = f_bright
                    bracket_steps.append((m_bright, c_bright, f_bright))
                else:
                    # Skip the -10.0 trial; start from data-driven initial guess instead
                    m_bright = float(initialGuess) if np.isfinite(initialGuess) else -5.0
                    c_bright, _, f_bright = run_trials_at_mag(m_bright, pool=pool)
                    going_faint = c_bright >= completeness_target
                    m_faint, c_faint = m_bright, c_bright
                    f_faint = f_bright
                    bracket_steps.append((m_bright, c_bright, f_bright))

                for _ in range(max_steps):
                    m_test = m_faint + step if going_faint else m_bright - step
                    c_test, _, f_test = run_trials_at_mag(m_test, pool=pool)
                    bracket_steps.append((m_test, c_test, f_test))

                    if going_faint:
                        # Track the last detected point so bisection starts from the
                        # final detected bracket endpoint (not the initial guess).
                        prev_m, prev_c, prev_f = m_faint, c_faint, f_faint
                        m_faint, c_faint = m_test, c_test
                        f_faint = f_test
                        # Continue stepping fainter until we find undetected endpoint
                        if c_faint < completeness_target:
                            # The detected endpoint is the previous step.
                            m_bright, c_bright, f_bright = prev_m, prev_c, prev_f
                            break
                    else:
                        # When stepping brighter (looking for detected end)
                        # Keep m_faint fixed (this is the undetected endpoint)
                        if c_test >= completeness_target:
                            # Found detected end
                            m_bright, c_bright = m_test, c_test
                            f_bright = f_test
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
                        f_faint = f_bright
                        bracket_steps.append((m_bright, c_bright, f_bright))
                        for _ in range(35):
                            m_test = m_faint + step if going_faint else m_bright - step
                            c_test, _, f_test = run_trials_at_mag(m_test, pool=pool)
                            bracket_steps.append((m_test, c_test, f_test))
                            if going_faint:
                                prev_m, prev_c, prev_f = m_faint, c_faint, f_faint
                                m_faint, c_faint = m_test, c_test
                                f_faint = f_test
                                if c_faint < completeness_target:
                                    m_bright, c_bright, f_bright = prev_m, prev_c, prev_f
                                    break
                            else:
                                # When stepping brighter, update based on detection
                                if c_test >= completeness_target:
                                    m_bright, c_bright = m_test, c_test
                                    f_bright = f_test
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
                    all_bracket_mags = [t[0] for t in bracket_steps if np.isfinite(t[0])]
                    m_lo_tried = float(min(all_bracket_mags)) if all_bracket_mags else np.nan
                    m_hi_tried = float(max(all_bracket_mags)) if all_bracket_mags else np.nan
                    logger.warning(
                        "Could not bracket %.0f%% completeness threshold "
                        "(searched m=[%.2f, %.2f], brightest det_rate=%.1f%%, "
                        "faintest det_rate=%.1f%%); returning NaN.",
                        100.0 * completeness_target,
                        m_lo_tried, m_hi_tried,
                        100.0 * float(c_bright),
                        100.0 * float(c_faint),
                    )

                else:
                    # ---- Bisect phase ----------------------------------------
                    lo_m, lo_c = m_bright, c_bright
                    hi_m, hi_c = m_faint, c_faint
                                        # Bracket endpoints already recorded in bracket_steps, but include
                    # them here so the plotted trajectory clearly straddles 50%.
                    # (f_bright/f_faint are medians from run_trials_at_mag).
                    try:
                        lo_f = float(f_bright) if "f_bright" in locals() else np.nan
                    except Exception:
                        lo_f = np.nan
                    try:
                        hi_f = float(f_faint) if "f_faint" in locals() else np.nan
                    except Exception:
                        hi_f = np.nan
                    bisect_steps = [(lo_m, lo_c, lo_f), (hi_m, hi_c, hi_f)]

                    for _ in range(30):
                        mid_m = 0.5 * (lo_m + hi_m)
                        mid_c, _, mid_f = run_trials_at_mag(mid_m, pool=pool)
                        bisect_steps.append((mid_m, mid_c, mid_f))

                        if mid_c >= completeness_target:
                            lo_m, lo_c = mid_m, mid_c
                        else:
                            hi_m, hi_c = mid_m, mid_c

                        if abs(hi_m - lo_m) < 0.02:
                            break

                    # Ensure the final bracketing endpoints used for interpolation are
                    # included in the plotted bisection trajectory.
                    try:
                        xs = np.asarray([t[0] for t in bisect_steps], dtype=float)
                        has_lo = bool(np.any(np.isfinite(xs) & np.isclose(xs, lo_m, atol=1e-12, rtol=0.0)))
                        has_hi = bool(np.any(np.isfinite(xs) & np.isclose(xs, hi_m, atol=1e-12, rtol=0.0)))
                        if not has_lo:
                            bisect_steps.append((lo_m, lo_c, np.nan))
                        if not has_hi:
                            bisect_steps.append((hi_m, hi_c, np.nan))
                    except Exception:
                        pass

                    # Estimate m at exactly completeness_target by interpolating
                    # between the final bracketing points (lo, hi). This is more
                    # faithful than returning the last midpoint when the recovery
                    # fraction is quantized by a finite number of trials.
                    inject_lmag = 0.5 * (lo_m + hi_m)
                    try:
                        denom = float(hi_c - lo_c)
                        if np.isfinite(denom) and abs(denom) > 0:
                            w = float((completeness_target - lo_c) / denom)
                            if np.isfinite(w):
                                w = float(np.clip(w, 0.0, 1.0))
                                inject_lmag = float(lo_m + w * (hi_m - lo_m))
                    except Exception:
                        pass
                    logger.info(
                        "    Bisection converged: m50=%.4f (bracket width=%.4f mag)",
                        float(inject_lmag), abs(hi_m - lo_m),
                    )

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
                        detection_cutoff,
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
                        snr_limit=effective_snr_limit,
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
                # Select appropriate zeropoint for logging.
                log_zeropoint = zeropoint
                if image_zeropoint is not None:
                    recovery_method_upper = str(recovery_method).strip().upper() if recovery_method else "AP"
                    if recovery_method_upper in ("PSF", "EMCEE"):
                        log_zeropoint = image_zeropoint.get("PSF", {}).get("zeropoint", zeropoint)
                    else:
                        log_zeropoint = image_zeropoint.get("AP", {}).get("zeropoint", zeropoint)

                zp_log = (
                    f"{float(log_zeropoint):.3f}"
                    if log_zeropoint is not None and np.isfinite(float(log_zeropoint) if log_zeropoint is not None else np.nan)
                    else "n/a"
                )
                app_str = ""
                if log_zeropoint is not None:
                    try:
                        if np.isfinite(float(log_zeropoint)):
                            app_mag = float(inject_lmag) + float(log_zeropoint)
                            app_str = f"  apparent={app_mag:.3f}"
                    except (TypeError, ValueError):
                        pass

                n_trials_total = len(_trial_cache)
                logger.info(
                    "Limiting magnitude: inst=%.3f%s  ZP=%s  "
                    "method=%s  completeness=%.0f%%  trials=%d  [%.1fs]",
                    float(inject_lmag), app_str, zp_log,
                    str(recovery_method), 100.0 * completeness_target,
                    n_trials_total, elapsed,
                )
            else:
                logger.info(
                    "\u26a0 Limiting magnitude search failed  [%.1fs]", elapsed
                )

            # The limiting magnitude is already exposure-time-normalized via flux_for_mag
            result_mag = float(inject_lmag)
            if _return_details:
                return {
                    "inject_lmag": result_mag,
                    "bracket_steps": bracket_steps,
                    "bisect_steps": bisect_steps,
                    "completeness_target": completeness_target,
                    "detection_cutoff": detection_cutoff,
                    "zeropoint": zeropoint,
                    "recovery_method": recovery_method,
                    "snr_limit": effective_snr_limit,
                    "image_zeropoint": image_zeropoint,
                    # Objects needed for injection cutout inset panels
                    "epsf_model": epsf_model,
                    "cutout": cutout,
                    "position": position,
                    "background_rms": background_rms,
                    "flux_for_mag": flux_for_mag,
                    "injection_df": injection_df,
                    "F_ref": F_ref,
                    "counts_ref": counts_ref,
                    "exposure_time": exposure_time,
                    "cutout_cx": cutout_cx,
                    "cutout_cy": cutout_cy,
                }
            return result_mag

        except Exception as exc:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning(
                f"getInjectedLimit failed: {exc} "
                f"[{exc_type.__name__}, {fname}:{exc_tb.tb_lineno}]"
            )
            if _return_details:
                return {
                    "inject_lmag": np.nan,
                    "bracket_steps": [],
                    "bisect_steps": [],
                    "completeness_target": locals().get('completeness_target', 0.5),
                    "detection_cutoff": detection_cutoff,
                    "zeropoint": zeropoint,
                    "recovery_method": None,
                    "snr_limit": None,
                    "image_zeropoint": image_zeropoint,
                }
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
        # Preserve no-data regions: do not inject into NaNs and keep them NaN.
        invalid = ~np.isfinite(cutout)
        psf_add = np.asarray(psf_add, dtype=float)
        if np.any(invalid):
            psf_add[invalid] = 0.0
        img = np.asarray(cutout, dtype=float) + psf_add
        if np.any(invalid):
            img = np.asarray(img, dtype=float)
            img[invalid] = np.nan

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
                try:
                    be = np.histogram_bin_edges(v, bins="fd")
                except Exception:
                    be = 40
                ax_h.hist(v, bins=be, histtype="step", color="0.2")
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
        detection_cutoff,
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
        snr_limit=None,
        multi_snr_details=None,
        fig=None,
        gs=None,
        inset_row=1,
        draw_main_plot=True,
    ) -> None:
        """
        Plot completeness curve with injection examples.
        If sample_mags is None, only shows bracket/bisect trajectories (no bar chart).
        If multi_snr_details is provided, draws all S/N thresholds' trajectories
        with distinct colors on the same plot.
        If fig/gs are provided, uses them instead of creating a new figure.
        inset_row controls which GridSpec row to use for insets (default=1).
        draw_main_plot=False skips the main completeness plot (for adding inset rows only).
        """
        logger = logging.getLogger(__name__)
        
        # Select appropriate zeropoint based on recovery method
        if image_zeropoint is not None:
            recovery_method_upper = str(recovery_method).strip().upper()
            if recovery_method_upper in ("PSF", "EMCEE"):
                selected_zeropoint = image_zeropoint.get("PSF", {}).get("zeropoint", zeropoint)
            else:  # AP method
                selected_zeropoint = image_zeropoint.get("AP", {}).get("zeropoint", zeropoint)
        else:
            selected_zeropoint = zeropoint
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

        # Create figure or use provided one
        owns_figure = fig is None
        if owns_figure:
            # Create figure with main completeness plot on top, injection examples below
            fig = plt.figure(figsize=set_size(540, 1.5))
            gs = GridSpec(3, 4, figure=fig, height_ratios=[1.5, 1, 1])
        else:
            # Use provided figure and gridspec
            if gs is None:
                raise ValueError("gs must be provided when fig is provided")

        # Main completeness plot (spans all columns in row 0)
        if draw_main_plot:
            ax = fig.add_subplot(gs[0, :])

        # Plot bracket and bisect search trajectories below (scatter + arrows).
        # Avoid drawing a single polyline through all steps here, as it can make
        # the bisection look like it "jumps" across large intervals.

        if draw_main_plot:
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

            # Adopted limit: use the interpolated m50 (inject_lmag) rather than the
            # last evaluated step (which can be above/below 50%).
            adopted_mag = float(inject_lmag) if np.isfinite(inject_lmag) else np.nan

            # ---- Bracket/bisection search trajectories & adopted-limit lines ----
            # Color palette for multi-S/N thresholds (up to 6 distinct)
            _palette = [
                ("#2196F3", "#1565C0"),  # blue / dark-blue
                ("#F44336", "#B71C1C"),  # red / dark-red
                ("#4CAF50", "#1B5E20"),  # green / dark-green
                ("#FF9800", "#E65100"),  # orange / dark-orange
                ("#9C27B0", "#4A148C"),  # purple / dark-purple
                ("#00BCD4", "#006064"),  # cyan / dark-cyan
            ]
            _markers = ["o", "s", "^", "D", "v", "P"]

            if multi_snr_details is not None and len(multi_snr_details) > 1:
                # --- Multi-SNR: draw all thresholds with distinct colours ---
                for idx, detail in enumerate(multi_snr_details):
                    d_snr = detail.get("snr_limit")
                    d_bracket = detail.get("bracket_steps", [])
                    d_bisect = detail.get("bisect_steps", [])
                    d_lmag = detail.get("inject_lmag", np.nan)
                    d_ctarget = detail.get("completeness_target", 0.5)

                    c_brk, c_bis = _palette[idx % len(_palette)]
                    mk = _markers[idx % len(_markers)]
                    tag = f"S/N$\\geq${d_snr:.0f}" if d_snr is not None else f"#{idx+1}"

                    if d_bracket:
                        bm, bc, _ = zip(*d_bracket)
                        bc_pct = np.asarray(bc, float) * 100.0
                        bm = np.asarray(bm, float)
                        ax.scatter(bm, bc_pct, s=16, color=c_brk, alpha=0.85,
                                   edgecolors="none", marker=mk,
                                   label=f"Bracket ({tag})", zorder=4)
                        for i in range(len(bm) - 1):
                            ax.annotate("", xy=(bm[i+1], bc_pct[i+1]),
                                        xytext=(bm[i], bc_pct[i]),
                                        arrowprops=dict(arrowstyle="->", color=c_brk,
                                                        lw=0.5, alpha=0.6))

                    if d_bisect:
                        bm, bc, _ = zip(*d_bisect)
                        bc_pct = np.asarray(bc, float) * 100.0
                        bm = np.asarray(bm, float)
                        ax.scatter(bm, bc_pct, s=16, color=c_bis, alpha=0.85,
                                   edgecolors="none", marker=mk,
                                   label=f"Bisection ({tag})", zorder=5)
                        for i in range(len(bm) - 1):
                            ax.annotate("", xy=(bm[i+1], bc_pct[i+1]),
                                        xytext=(bm[i], bc_pct[i]),
                                        arrowprops=dict(arrowstyle="->", color=c_bis,
                                                        lw=0.5, alpha=0.6))

                    d_adopted = float(d_lmag) if np.isfinite(d_lmag) else np.nan
                    if np.isfinite(d_adopted):
                        ax.axvline(d_adopted, color=c_brk, lw=0.8, ls="--",
                                   label=f"m50 ({tag})", zorder=6, alpha=0.9)
                        ax.scatter([d_adopted], [float(d_ctarget) * 100.0],
                                   s=32, marker="D", c=c_brk,
                                   edgecolors="white", linewidth=0.4, zorder=7)
            else:
                # --- Single-SNR: original behaviour ---
                snr_label = ""
                if snr_limit is not None:
                    snr_label = f" (S/N>={snr_limit:.0f})"

                if bracket_steps:
                    bm, bc, _ = zip(*bracket_steps)
                    bc_percent = np.asarray(bc, float) * 100.0
                    bm = np.asarray(bm, float)
                    ax.scatter(bm, bc_percent, s=14, color="#0000FF", alpha=0.8,
                               edgecolors="none", label=f"Bracket search{snr_label}", zorder=4)
                    for i in range(len(bm) - 1):
                        ax.annotate("", xy=(bm[i+1], bc_percent[i+1]),
                                    xytext=(bm[i], bc_percent[i]),
                                    arrowprops=dict(arrowstyle="->", color="#0000FF",
                                                    lw=0.5, alpha=0.7))

                if bisect_steps:
                    bm, bc, _ = zip(*bisect_steps)
                    bc_percent = np.asarray(bc, float) * 100.0
                    bm = np.asarray(bm, float)
                    ax.scatter(bm, bc_percent, s=14, color="#00AA00", alpha=0.8,
                               edgecolors="none", label=f"Bisection{snr_label}", zorder=5)
                    for i in range(len(bm) - 1):
                        ax.annotate("", xy=(bm[i+1], bc_percent[i+1]),
                                    xytext=(bm[i], bc_percent[i]),
                                    arrowprops=dict(arrowstyle="->", color="#00AA00",
                                                    lw=0.5, alpha=0.7))

                if np.isfinite(adopted_mag):
                    ax.axvline(adopted_mag, color="k", lw=0.6, ls="--",
                               label=f"Adopted limit (m50){snr_label}", zorder=6)
                    try:
                        ax.scatter([adopted_mag], [float(completeness_target) * 100.0],
                                   s=28, marker="D", c="k", edgecolors="white",
                                   linewidth=0.4, zorder=7)
                    except Exception:
                        pass

            # Reference lines.
            ax.axhline(50, color="0.7", lw=0.5, ls="--", zorder=0)
            ax.text(0.02, 50, "50%", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left", color="0.5")

            if completeness_target != 0.5:
                target_percent = completeness_target * 100
                ax.axhline(target_percent, color="0.7", lw=0.5, ls="-.", zorder=0)
                ax.text(0.98, target_percent, f"{int(target_percent)}%",
                        transform=ax.get_yaxis_transform(), va="bottom",
                        ha="right", color="0.5")

            # Add axis labels and title to main completeness plot
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

            ncol = 2 if (multi_snr_details is not None and len(multi_snr_details) > 1) else 1
            ax.legend(loc="best", fontsize=7, frameon=False, ncol=ncol)

            if owns_figure:
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

            # ------------------------------------------------------------------
            # Choose demo injection magnitudes based on *recovery fraction* so the
            # 3 cutouts remain meaningful even when m50 lies near the search edge.
            #
            # Requested: show ~100%, 80%, and 50% recoverable injections.
            # ------------------------------------------------------------------
            zp_ok = selected_zeropoint is not None and np.isfinite(float(selected_zeropoint))
            zp_val = float(selected_zeropoint) if zp_ok else 0.0

            def _mag_at_recovery(p_target: float) -> float:
                """Return instrumental magnitude at which completeness ~ p_target."""
                steps = []
                for s in (bracket_steps or []) + (bisect_steps or []):
                    try:
                        m0 = float(s[0])
                        c0 = float(s[1])
                    except Exception:
                        continue
                    if np.isfinite(m0) and np.isfinite(c0):
                        steps.append((m0, float(np.clip(c0, 0.0, 1.0))))
                if len(steps) < 2:
                    return float(inject_lmag)

                # Sort bright -> faint (increasing mag). Enforce a monotone non-increasing
                # completeness curve to make interpolation stable under Monte-Carlo noise.
                steps.sort(key=lambda t: t[0])
                m_arr = np.asarray([t[0] for t in steps], dtype=float)
                c_arr = np.asarray([t[1] for t in steps], dtype=float)
                c_mono = np.maximum.accumulate(c_arr[::-1])[::-1]  # non-increasing vs mag

                # Find first crossing from above to below p_target.
                for j in range(1, len(m_arr)):
                    c0, c1 = c_mono[j - 1], c_mono[j]
                    if (c0 >= p_target) and (c1 <= p_target) and (c0 != c1):
                        w = float((p_target - c0) / (c1 - c0))
                        w = float(np.clip(w, 0.0, 1.0))
                        return float(m_arr[j - 1] + w * (m_arr[j] - m_arr[j - 1]))

                # If we never cross, clamp to range end that is "closest" in probability.
                # p high -> choose brightest; p low -> choose faintest.
                if p_target >= float(np.nanmax(c_mono)):
                    return float(m_arr[0])
                return float(m_arr[-1])

            m50_inst = float(inject_lmag)
            m80_inst = float(_mag_at_recovery(0.80))
            m100_inst = float(_mag_at_recovery(0.99))  # "100%" in practice

            inst_targets = [m50_inst, m80_inst, m100_inst]
            mag_targets = [m + zp_val for m in inst_targets] if zp_ok else inst_targets
            
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
            #   1. Quietest valid site from injection_df (lowest |SNR|).
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
                snr_col = "SNR" if "SNR" in injection_df.columns else None
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
                    # Avoid invalid/no-data pixels for demo site.
                    xi = int(np.round(sx))
                    yi = int(np.round(sy))
                    if xi < 0 or yi < 0 or xi >= nx_c or yi >= ny_c:
                        continue
                    v0 = float(cutout[yi, xi])
                    if not np.isfinite(v0) or (
                        False and v0 == 0.0
                    ):
                        continue
                    # Consistent with main injection-site selection: score by |SNR|.
                    if snr_col is not None:
                        try:
                            score = float(abs(float(site[snr_col])))
                        except Exception:
                            score = float("inf")
                    else:
                        score = float("inf")
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
                        xi = int(np.round(cx_try))
                        yi = int(np.round(cy_try))
                        if 0 <= xi < nx_c and 0 <= yi < ny_c:
                            v0 = float(cutout[yi, xi])
                            if not np.isfinite(v0) or (
                                False and v0 == 0.0
                            ):
                                continue
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
                        "All circumference fallback points too close to edge; using offset demo site (%.1f, %.1f)", demo_x, demo_y,
                    )

            for i, mag_target in enumerate(mag_targets):
                ax_inject = fig.add_subplot(gs[inset_row, i])
                
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
                    # Use only hardware mask (NaN/inf pixels) for injection - don't mask out zero-valued pixels
                    # which could be valid bright sources
                    invalid_mask = ~np.isfinite(injected)
                    # Ensure no-data stays NaN even after injection.
                    injected = np.asarray(injected, dtype=float)
                    injected[invalid_mask] = np.nan

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
                    # Do not inject into invalid pixels.
                    try:
                        psf_inject = np.asarray(psf_inject, dtype=float)
                        psf_inject[invalid_mask] = 0.0
                    except Exception:
                        pass
                    injected += psf_inject
                    # Re-impose invalid mask for display (do not let injection fill no-data areas).
                    injected_disp = np.asarray(injected, dtype=float).copy()
                    injected_disp[invalid_mask] = np.nan
                    
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
                        "Subpanel zoom: cutout=(%dx%d), target=(%.1f,%.1f), injection=(%.1f,%.1f), inject_distance=%.1f px, zoom_radius=%.1f px, "
                        "zoom=[%d:%d, %d:%d]",
                        nx, ny, x_center, y_center,
                        inject_x, inject_y, inject_distance, zoom_radius,
                        x0_zoom, x1_zoom, y0_zoom, y1_zoom,
                    )

                    # Display the full injected cutout, zoom with set_xlim/ylim
                    from astropy.visualization import ZScaleInterval
                    zscale = ZScaleInterval()
                    finite = injected_disp[np.isfinite(injected_disp)]
                    if finite.size:
                        lower, upper = np.percentile(finite, [0.5, 99.5])
                        vmin, vmax = zscale.get_limits(np.clip(injected_disp, lower, upper))
                    else:
                        vmin, vmax = np.nanmin(injected_disp), np.nanmax(injected_disp)
                    cmap = plt.get_cmap("grey").copy()
                    cmap.set_bad(color="white")
                    im = ax_inject.imshow(
                        np.ma.array(injected_disp, mask=~np.isfinite(injected_disp)),
                        origin="lower",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
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
                                        snr = np.abs(flux_hat) / flux_err

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
                                snr = np.abs(fl) / ns if ns > 0 else 0.0

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

                    # Set title with injected magnitude and target recovery level.
                    # Panel order matches mag_targets: 50%, 80%, ~100%.
                    try:
                        _lbl = ["50%", "80%", "100%"][int(i)]
                    except Exception:
                        _lbl = ""
                    ax_inject.set_title(
                        f"Mag$_{{in}}$ = {mag_target:.2f}" + (f" ({_lbl})" if _lbl else ""),
                        fontsize=9,
                    )

                    # Add recovered magnitude text in lower left corner
                    if selected_zeropoint is not None:
                        # Compute recovered apparent magnitude from flux
                        recovered_apparent = np.nan
                        if recovery_method_upper == "PSF" and 'flux_hat' in locals() and np.isfinite(flux_hat):
                            # PSF method: flux_hat is PSF flux parameter
                            if counts_ref is not None and exposure_time is not None and counts_ref > 0 and exposure_time > 0:
                                recovered_flux_e_per_s = flux_hat * counts_ref / exposure_time
                                recovered_inst = -2.5 * np.log10(max(recovered_flux_e_per_s, 1e-30))
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
                                f"Mag$_{{out}}$: {recovered_apparent:.2f}",
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
                    cmap = plt.get_cmap("grey").copy()
                    cmap.set_bad(color="white")
                    # Use only hardware mask (NaN/inf pixels) for plotting - don't mask out zero-valued pixels
                    cut_disp = np.asarray(cutout, dtype=float).copy()
                    ax_inject.imshow(
                        np.ma.array(cut_disp, mask=~np.isfinite(cut_disp)),
                        origin="lower",
                        cmap=cmap,
                        norm=norm,
                    )
                    ax_inject.set_xlim(0, nx)
                    ax_inject.set_ylim(0, ny)
                    ax_inject.set_title(f'Injection failed', fontsize=9)
                    ax_inject.set_xlabel('X [pixels]', fontsize=8)
                    if i == 0:
                        ax_inject.set_ylabel('Y [pixels]', fontsize=8)
                    else:
                        ax_inject.set_ylabel('')
                    ax_inject.tick_params(labelsize=8)

            # Fourth panel (bottom right, same row as injection panels): original cutout with injection site markers
            if cutout is not None and injection_df is not None and len(injection_df) > 0:
                ax_sites = fig.add_subplot(gs[inset_row, 3])
                ny, nx = cutout.shape

                # Display original cutout (no injections)
                from astropy.visualization import simple_norm
                norm = simple_norm(cutout, 'sqrt', percent=99.5)
                cmap = plt.get_cmap("grey").copy()
                cmap.set_bad(color="white")
                # Use only hardware mask (NaN/inf pixels) for plotting - don't mask out zero-valued pixels
                cut_disp = np.asarray(cutout, dtype=float).copy()
                ax_sites.imshow(
                    np.ma.array(cut_disp, mask=~np.isfinite(cut_disp)),
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                )
                ax_sites.set_xlim(0, nx)
                ax_sites.set_ylim(0, ny)

                # Mark target position
                from matplotlib.patches import Circle
                target_marker = Circle((target_x, target_y), radius=aperture_radius,
                                      edgecolor='red', facecolor='none', linestyle='-', linewidth=0.5)
                ax_sites.add_patch(target_marker)
                ax_sites.text(target_x, target_y + aperture_radius, transient_label,
                             color='red', fontsize=8, ha='center', va='bottom')

                # Mark all injection sites
                for _, site in injection_df.iterrows():
                    sx, sy = float(site["x_pix"]), float(site["y_pix"])
                    if not (np.isfinite(sx) and np.isfinite(sy)):
                        continue
                    # Circle marker for each injection site
                    site_circle = Circle((sx, sy), radius=aperture_radius,
                                        edgecolor='cyan', facecolor='none', linestyle='--', linewidth=0.5)
                    ax_sites.add_patch(site_circle)
                    # Small cross at center
                    ax_sites.plot([sx], [sy], '+', color='cyan', markersize=4, markeredgewidth=0.5)

                ax_sites.set_title("Injection sites", fontsize=9)
                ax_sites.set_xlabel('X [pixels]', fontsize=8)
                ax_sites.set_ylabel('Y [pixels]', fontsize=8)
                ax_sites.tick_params(labelsize=8)

            # Add axis data: small grey lines at top of main plot showing injected magnitudes
            # Only when main plot is being drawn (ax and secax exist)
            if draw_main_plot:
                from matplotlib.transforms import blended_transform_factory
                ymin, ymax = ax.get_ylim()
                y_bottom = ymax - 0.05 * (ymax - ymin)  # 5% from top in data coordinates
                y_top = ymax
                if secax is not None:
                    xvals = list(mag_targets)  # already apparent mags
                    xtrans = secax.transData
                else:
                    # Convert apparent targets back to instrumental for the primary axis.
                    if selected_zeropoint is not None and np.isfinite(float(selected_zeropoint)):
                        xvals = [float(m) - float(selected_zeropoint) for m in mag_targets]
                    else:
                        xvals = list(mag_targets)
                    xtrans = ax.transData
                transform = blended_transform_factory(xtrans, ax.transData)
                for xv in xvals:
                    if not np.isfinite(xv):
                        continue
                    ax.plot(
                        [xv, xv],
                        [y_bottom, y_top],
                        color="red",
                        alpha=0.55,
                        linewidth=1.2,
                        linestyle="--",
                        marker=None,
                        markevery=None,
                        zorder=6,
                        clip_on=False,
                        transform=transform,
                    )

        if owns_figure:
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

        
        # Convert to apparent magnitudes
        injected_apparent = inst_mags + selected_zeropoint

        # Convert recovered flux to instrumental magnitude, then to apparent
        # The recovered flux units depend on recovery_method:
        # - AP method: flux_AP is already e-/s (divided by exposure_time in Aperture.measure()).
        # - PSF/EMCEE methods: flux_hat is the PSF flux parameter (dimensionless scaling factor)
        #   To convert to physical flux rate: flux_e_per_s = flux_hat * counts_ref / exposure_time
        
        recovery_method_upper = str(recovery_method).strip().upper() if recovery_method is not None else "AP"
        if recovery_method_upper == "AP":
            # AP method: flux_AP is already e-/s, apply mag() directly
            recovered_inst = -2.5 * np.log10(np.maximum(recovered_fluxes, 1e-30))
        else:
            # PSF/EMCEE methods: flux_hat is PSF flux parameter (dimensionless)
            # Convert to physical flux rate: flux_e_per_s = flux_hat * counts_ref / exposure_time
            if counts_ref is not None and exposure_time is not None and counts_ref > 0 and exposure_time > 0:
                recovered_flux_e_per_s = recovered_fluxes * counts_ref / exposure_time
                recovered_inst = -2.5 * np.log10(np.maximum(recovered_flux_e_per_s, 1e-30))
            else:
                # Fallback: treat as raw integrated e- divided by exposure_time.
                if exposure_time is not None and exposure_time > 0:
                    recovered_flux_e_per_s = recovered_fluxes / exposure_time
                    recovered_inst = -2.5 * np.log10(np.maximum(recovered_flux_e_per_s, 1e-30))
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

    # -----------------------------------------------------------------------
    # Combined completeness plot for multi-S/N thresholds
    # -----------------------------------------------------------------------

    def _plot_completeness_combined(self, all_details: list) -> None:
        """
        Draw a single completeness plot with bracket/bisect trajectories
        from multiple S/N threshold runs overlaid in distinct colors,
        plus injection cutout inset panels for EACH S/N threshold.

        Parameters
        ----------
        all_details : list of dicts returned by get_injected_limit(_return_details=True)
        """
        logger = logging.getLogger(__name__)

        if not all_details:
            return

        # Create figure with enough rows: 1 main plot + 1 row per threshold
        n = len(all_details)
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=set_size(540, 1 + 0.5 * n))
        gs = GridSpec(1 + n, 4, figure=fig, height_ratios=[1.5] + [1] * n)

        # Use the first (primary) threshold's detail to drive the main plot
        # and first inset row, then add additional rows for remaining thresholds.
        primary = all_details[0]

        # Main plot + first inset row (primary threshold)
        self._plot_completeness(
            None,  # No sample_mags
            None,  # No completeness_groups
            None,  # No medians
            primary.get("bracket_steps", []),
            primary.get("bisect_steps", []),
            primary.get("inject_lmag", np.nan),
            primary.get("completeness_target", 0.5),
            primary.get("detection_cutoff"),
            primary.get("zeropoint"),
            primary.get("recovery_method"),
            epsf_model=primary.get("epsf_model"),
            cutout=primary.get("cutout"),
            position=primary.get("position"),
            background_rms=primary.get("background_rms"),
            flux_for_mag=primary.get("flux_for_mag"),
            image_zeropoint=primary.get("image_zeropoint"),
            injection_df=primary.get("injection_df"),
            F_ref=primary.get("F_ref"),
            counts_ref=primary.get("counts_ref"),
            exposure_time=primary.get("exposure_time"),
            extended_steps=[],
            orig_position=None,
            target_name=self.input_yaml.get("target_name", None),
            cutout_cx=primary.get("cutout_cx"),
            cutout_cy=primary.get("cutout_cy"),
            snr_limit=primary.get("snr_limit"),
            multi_snr_details=all_details,
            fig=fig,
            gs=gs,
            inset_row=1,
            draw_main_plot=True,
        )

        # Add inset rows for remaining thresholds
        for idx, detail in enumerate(all_details[1:], start=2):
            self._plot_completeness(
                None, None, None,
                detail.get("bracket_steps", []),
                detail.get("bisect_steps", []),
                detail.get("inject_lmag", np.nan),
                detail.get("completeness_target", 0.5),
                detail.get("detection_cutoff"),
                detail.get("zeropoint"),
                detail.get("recovery_method"),
                epsf_model=primary.get("epsf_model"),  # Shared
                cutout=primary.get("cutout"),  # Shared
                position=primary.get("position"),  # Shared
                background_rms=primary.get("background_rms"),  # Shared
                flux_for_mag=primary.get("flux_for_mag"),  # Shared
                image_zeropoint=primary.get("image_zeropoint"),  # Shared
                injection_df=primary.get("injection_df"),  # Shared
                F_ref=primary.get("F_ref"),  # Shared
                counts_ref=primary.get("counts_ref"),  # Shared
                exposure_time=primary.get("exposure_time"),  # Shared
                extended_steps=[],
                orig_position=None,
                target_name=self.input_yaml.get("target_name", None),
                cutout_cx=primary.get("cutout_cx"),  # Shared
                cutout_cy=primary.get("cutout_cy"),  # Shared
                snr_limit=detail.get("snr_limit"),
                multi_snr_details=None,
                fig=fig,
                gs=gs,
                inset_row=idx,
                draw_main_plot=False,
            )

        # Add S/N threshold labels on the left side of each inset row
        for idx, detail in enumerate(all_details, start=1):
            d_snr = detail.get("snr_limit")
            if d_snr is not None:
                # Add label as figure text to avoid GridSpec cell conflicts
                try:
                    # Compute position: left of each inset row
                    n_rows = 1 + len(all_details)
                    y_pos = 1.0 - (idx + 0.5) / n_rows  # Center of each row
                    fig.text(
                        0.02, y_pos, f"S/N ≥ {d_snr:.0f}",
                        fontsize=10, fontweight="bold",
                        ha="left", va="center", rotation=90,
                    )
                except Exception:
                    pass

        fig.tight_layout()
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
        save_loc_png = os.path.join(write_dir, f"Completeness_{base}.png")
        fig.savefig(save_loc_png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Combined completeness plot saved: %s", save_loc_png)

    # -----------------------------------------------------------------------
    # Multi-S/N injection limiting magnitude
    # -----------------------------------------------------------------------

    def get_injected_limits_multi_snr(
        self,
        full_image: np.ndarray,
        position,
        epsf_model=None,
        initialGuess: float = -5.0,
        snr_thresholds: list = None,
        detection_cutoff: float | None = None,
        plot: bool = True,
        background_rms: np.ndarray = None,
        subtraction_ready: bool = False,
        zeropoint: float = None,
        n_jobs: int = None,
        image_zeropoint: dict = None,
    ) -> dict:
        """
        Calculate limiting magnitudes for multiple S/N thresholds using injection/recovery.

        This function runs the injection limiting magnitude analysis for each specified
        S/N threshold and returns structured results with all limits and comparisons.

        Parameters
        ----------
        full_image      : full 2-D science image (pre-subtraction)
        position        : (x, y) target pixel coords used to centre the cutout
        epsf_model      : photutils ePSF model
        initialGuess    : starting instrumental magnitude for the bracket
        snr_thresholds  : list of S/N thresholds (e.g., [3.0, 5.0]). If None, uses config
        detection_cutoff: legacy beta threshold (unused; retained for backwards compatibility)
        plot            : save diagnostic completeness PDF for each S/N threshold
        background_rms  : full-frame RMS map (optional)
        subtraction_ready: unused placeholder
        zeropoint       : adds an apparent-magnitude axis to the plot
        n_jobs          : worker processes; None defaults to 1 (serial)

        Returns
        -------
        dict - results for each S/N threshold with limiting magnitudes and comparisons
        """
        logger = logging.getLogger(__name__)
        
        # Get S/N thresholds from config if not specified
        if snr_thresholds is None:
            lim_cfg = self.input_yaml.get("limiting_magnitude") or {}
            snr_thresholds = lim_cfg.get("snr_thresholds", [3.0])
        
        # Ensure we have valid thresholds
        if not snr_thresholds or not isinstance(snr_thresholds, list):
            logger.warning("Invalid snr_thresholds, using default [3.0]")
            snr_thresholds = [3.0]
        
        logger.info(f"Calculating injection limits for S/N thresholds: {snr_thresholds}")
        
        results = {}
        # Collect per-threshold details for the combined completeness plot
        all_details = []
        
        # Calculate limiting magnitude for each S/N threshold
        # Suppress individual plots; we'll draw one combined plot at the end.
        for snr in snr_thresholds:
            try:
                logger.info(f"Calculating limiting magnitude for S/N >= {snr}")
                detail = self.get_injected_limit(
                    full_image=full_image,
                    position=position,
                    epsf_model=epsf_model,
                    initialGuess=initialGuess,
                    detection_limit=snr,
                    detection_cutoff=detection_cutoff,
                    plot=False,  # suppress individual plots
                    background_rms=background_rms,
                    subtraction_ready=subtraction_ready,
                    zeropoint=zeropoint,
                    n_jobs=n_jobs,
                    image_zeropoint=image_zeropoint,
                    _return_details=True,
                )
                limit = detail["inject_lmag"]
                all_details.append(detail)
                results[f'snr_{snr}'] = {
                    'limiting_mag': limit,
                    'snr_threshold': snr,
                    'valid': np.isfinite(limit)
                }
                logger.info(f"S/N {snr} limiting magnitude: {limit:.3f}")
            except Exception as e:
                logger.error(f"Failed to calculate S/N {snr} limiting magnitude: {e}")
                results[f'snr_{snr}'] = {
                    'limiting_mag': np.nan,
                    'snr_threshold': snr,
                    'valid': False,
                    'error': str(e)
                }
        
        # Add comparison metrics if we have multiple valid results
        valid_results = {k: v for k, v in results.items() if v.get('valid', False)}
        if len(valid_results) >= 2:
            sorted_thresholds = sorted([float(k.split('_')[1]) for k in valid_results.keys()])
            
            comparisons = {}
            for i in range(len(sorted_thresholds) - 1):
                snr_low = sorted_thresholds[i]
                snr_high = sorted_thresholds[i + 1]
                
                mag_low = valid_results[f'snr_{snr_low}']['limiting_mag']
                mag_high = valid_results[f'snr_{snr_high}']['limiting_mag']
                
                if np.isfinite(mag_low) and np.isfinite(mag_high):
                    delta_mag = mag_high - mag_low
                    comparisons[f'snr_{snr_high}_vs_{snr_low}'] = {
                        'delta_mag': delta_mag,
                        'snr_low': snr_low,
                        'snr_high': snr_high,
                        'mag_low': mag_low,
                        'mag_high': mag_high
                    }
            
            results['comparisons'] = comparisons
            logger.info(f"Generated {len(comparisons)} S/N threshold comparisons")
        
        # Adaptive threshold recommendation
        lim_cfg = self.input_yaml.get("limiting_magnitude") or {}
        if lim_cfg.get("adaptive_snr_selection", False):
            adaptive_threshold = 5.0 if 5.0 in snr_thresholds else max(snr_thresholds)
            results['adaptive_threshold'] = adaptive_threshold
            logger.info(f"Adaptive S/N threshold recommendation: {adaptive_threshold}")
        
        # Generate a single combined completeness plot with all S/N thresholds
        if plot and all_details:
            try:
                self._plot_completeness_combined(all_details)
            except Exception as exc:
                logger.warning("Combined completeness plot failed: %s", exc)
        
        return results
