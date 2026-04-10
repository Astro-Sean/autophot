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
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache


@contextmanager
def _pool_or_serial(n_jobs: int):
    """Yield a ProcessPoolExecutor when n_jobs > 1, else None (serial; avoids fork on HPC)."""
    if n_jobs <= 1:
        yield None
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        yield pool


# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    border_msg,
)
from aperture import Aperture

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
        (x_inj, y_inj, F_amp, gridx, gridy, cutout,
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
        gridx,
        gridy,
        cutout,
        epsf_model,
        input_yaml,
        background_rms,
        snr_limit,
        beta_n,
        DETECTION_BETA_THRESH,
        recovery_method,
    ) = args

    try:
        psf_img = epsf_model.evaluate(
            x=gridx, y=gridy, flux=F_amp, x_0=x_inj, y_0=y_inj
        )
        new_img = cutout + psf_img

        # Compute beta using the canonical aperture-based formalism (n=3) so
        # beta thresholds remain comparable across runs/methods.
        ap = Aperture(input_yaml=input_yaml, image=new_img)
        trial_df = pd.DataFrame({"x_pix": [x_inj], "y_pix": [y_inj]})
        mres = ap.measure(
            sources=trial_df, plot=False, background_rms=background_rms, verbose=0
        )

        beta_p = beta_aperture(
            n=beta_n,
            flux_aperture=float(mres["flux_AP"].iloc[0]),
            sigma=float(mres["noiseSky"].iloc[0]),
            npix=float(mres["area"].iloc[0]),
        )
        det_beta = beta_p >= DETECTION_BETA_THRESH

        # Detection SNR gate: either aperture SNR (legacy) or PSF-fit SNR.
        method = str(recovery_method).strip().upper() if recovery_method is not None else "AP"
        snr_val = np.nan
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
                gx, gy = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
                psf1 = np.asarray(
                    epsf_model.evaluate(x=gx, y=gy, flux=1.0, x_0=x0i, y_0=y0i),
                    dtype=float,
                )
                # Fit both PSF flux and a constant background level in the stamp:
                # data ~= flux * psf1 + bkg. This correctly handles any local mean
                # shift (including the uniform "bias") without forcing background=0
                # or relying on an annulus estimate.
                if background_rms is not None:
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
                # Place the source at injected coordinates in stamp coordinates.
                model.x_0.value = float(x0i - x1)
                model.y_0.value = float(y0i - y1)
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
                    gain=float(input_yaml.get("gain", 1.0)),
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
                return False, beta_p

        # For PSF method, use snr_limit if set, otherwise default to 3 sigma
        # For AP method, only apply SNR gate if snr_limit is explicitly set
        if method == "PSF":
            effective_snr_limit = float(snr_limit) if snr_limit is not None else 3.0
            det_snr = np.isfinite(snr_val) and (snr_val >= effective_snr_limit)
            return (det_beta and det_snr), beta_p
        else:
            # AP method: only check SNR if snr_limit is set
            if snr_limit is not None:
                det_snr = np.isfinite(snr_val) and (snr_val >= float(snr_limit))
                return (det_beta and det_snr), beta_p
            else:
                return det_beta, beta_p

    except Exception:
        return False, 0.0


# ===========================================================================
# limits class
# ===========================================================================


class Limits:
    """
    Compute limiting magnitudes for a single astronomical image frame using:

    * PSF injection / recovery           (getInjectedLimit)
    """

    def __init__(self, input_yaml: dict):
        """
        Parameters
        ----------
        input_yaml : dict
            Pipeline configuration loaded from YAML.  Expected keys include
            ``fwhm``, ``scale``, ``gain``, ``exposure_time``, ``target_x_pix``,
            ``target_y_pix``, ``photometry.aperture_radius``,
            ``limiting_magnitude.inject_source_location``, ``fpath``.
        """
        self.input_yaml = input_yaml

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
            mag_factor = self.input_yaml["limiting_magnitude"]["inject_source_location"]

            half = int(np.ceil(mag_factor * fwhm + scale))
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
            cutout = Cutout2D(
                image,
                position=(tx, ty),
                size=(2 * half, 2 * half),
                mode="partial",
                fill_value=np.nan,
            )
            return cutout.data

        except Exception as exc:
            logger.debug(f"getCutout failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # PSF helpers
    # -----------------------------------------------------------------------

    def _normalize_psf(self, epsf_model, shape: tuple):
        """
        Ensure the PSF model integrates to unity over *shape* pixels.

        Parameters
        ----------
        epsf_model : photutils ePSF model
        shape      : (height, width)

        Returns
        -------
        epsf_model  (modified in-place and returned for convenience)
        """
        grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        centre = shape[0] // 2
        psf_data = epsf_model.evaluate(
            x=grid[0], y=grid[1], flux=1.0, x_0=centre, y_0=centre
        )
        total = np.nansum(psf_data)
        if not np.isclose(total, 1.0, rtol=0.01):
            epsf_model.flux = 1.0 / total
        return epsf_model

    @staticmethod
    def _downsample_psf(psf_os: np.ndarray, oversampling: int) -> np.ndarray:
        """
        Flux-conserving block-average downsampling of an oversampled PSF.

        Reshapes into (H, os, W, os) blocks and takes the mean over the
        oversampling axes - exact for integer oversampling factors.

        Parameters
        ----------
        psf_os      : 2-D ndarray  (H*os, W*os)
        oversampling: int

        Returns
        -------
        psf_unit : 2-D ndarray  (H, W),  sum ~ 1/(os^2) * sum(psf_os)
        """
        H_os, W_os = psf_os.shape
        H = H_os // oversampling
        W = W_os // oversampling
        # Trim to exact multiple then reshape and average.
        return (
            psf_os[: H * oversampling, : W * oversampling]
            .reshape(H, oversampling, W, oversampling)
            .mean(axis=(1, 3))
        )

    # -----------------------------------------------------------------------
    # PSF injection / recovery limiting magnitude
    # -----------------------------------------------------------------------

    def get_injected_limit(
        self,
        cutout: np.ndarray,
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
        precutout: bool = False,
    ) -> float:
        """
        Bracket-and-bisect search for the limiting magnitude by injecting
        artificial PSF sources and measuring their recoverability.

        A *single* ProcessPoolExecutor is created for the entire search and
        reused by the nested ``run_trials_at_mag`` helper - avoiding the
        ~40-60 Pool creations that the original code performed.

        Parameters
        ----------
        cutout          : full 2-D science image (pre-subtraction)
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
                "Injected limiting magnitude: beta_limit=%.3f (~%s sigma; n=3 beta formalism), "
                "snr_gate=%s (limiting_magnitude.detection_limit), beta_n=3",
                float(detection_cutoff),
                ("%.2f" % float(k_sigma)) if k_sigma is not None else "unknown",
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

            def _extract_cutouts(scale_px: float):
                img = self.get_cutout(
                    image=cutout, position=position, scale_override=scale_px
                )
                if img is None:
                    return None, None
                rms = None
                if background_rms is not None:
                    rms = self.get_cutout(
                        image=background_rms,
                        position=position,
                        scale_override=scale_px,
                    )
                    if rms is not None:
                        rms = np.abs(np.asarray(rms, dtype=float))
                return img, rms

            if bool(precutout):
                cutout_img = np.asarray(cutout, dtype=float)
                cutout_rms = (
                    None
                    if background_rms is None
                    else np.abs(np.asarray(background_rms, dtype=float))
                )
                # In precutout mode, treat `position` as already in cutout coordinates.
                # We still recenter internally below to the extracted cutout centre.
            else:
                cutout_img, cutout_rms = _extract_cutouts(scale_used)
            if cutout_img is None and growth_factor > 1:
                # Try a slightly larger cutout if the initial one fails.
                for _ in range(2):
                    scale_used = min(
                        float(growth_max),
                        max(scale_used + 1.0, scale_used * growth_factor),
                    )
                    if bool(precutout):
                        break
                    cutout_img, cutout_rms = _extract_cutouts(scale_used)
                    if cutout_img is not None:
                        break

            if cutout_img is None:
                logger.warning("getCutout returned None; aborting")
                return np.nan

            cutout = cutout_img
            background_rms = cutout_rms

            # Re-centre position to the middle of the extracted cutout.
            H, W = cutout.shape
            position = [W / 2.0, H / 2.0]

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
                    try:
                        guess_k = float(lim_cfg.get("initial_guess_sigma_mult", 5.0))
                    except Exception:
                        guess_k = 5.0
                    probe_n = int((lim_cfg.get("initial_guess_n_samples", 24)))
                    probe_n = max(8, min(probe_n, 72))
                    probe_r = (
                        float(lim_cfg.get("inject_source_location", 3))
                        * float(self.input_yaml.get("fwhm", 3.0))
                    )
                    pts = points_in_circum(probe_r, center=position, n=probe_n)
                    probe_df = pd.DataFrame(
                        {"x_pix": [p[0] for p in pts], "y_pix": [p[1] for p in pts]}
                    )
                    probe_ap = Aperture(input_yaml=self.input_yaml, image=cutout)
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

            # Pixel grids for PSF evaluation (integer pixels, no oversampling yet).
            gridx, gridy = np.meshgrid(np.arange(W), np.arange(H))

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

            logger.info(f"PSF oversampling factor: {oversampling}x")

            if oversampling > 1:
                # Fine grid spanning exactly the same pixel extent as gridx/gridy.
                gx_os = np.linspace(0, W - 1, W * oversampling)
                gy_os = np.linspace(0, H - 1, H * oversampling)
                gridx_os, gridy_os = np.meshgrid(gx_os, gy_os)
            else:
                gridx_os, gridy_os = gridx, gridy

            # =================================================================
            # PSF calibration: flux=1 -> what instrumental magnitude?
            # =================================================================
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            psf_os = epsf_model.evaluate(
                x=gridx_os, y=gridy_os, flux=1.0, x_0=cx, y_0=cy
            )
            logger.info(
                f"Oversampled PSF shape={psf_os.shape}, " f"sum={np.nansum(psf_os):.4f}"
            )

            psf_unit = (
                self._downsample_psf(psf_os, oversampling)
                if oversampling > 1
                else psf_os
            )
            psf_unit = psf_unit / np.nansum(psf_unit)  # exact unit normalisation

            psf_ap = Aperture(input_yaml=self.input_yaml, image=psf_unit)
            psf_meas = psf_ap.measure(
                pd.DataFrame({"x_pix": [W / 2.0], "y_pix": [H / 2.0]}),
                plot=False,
                verbose=0,
            )
            F_ref = float(psf_meas["flux_AP"].iloc[0])
            m_ref = mag(F_ref)
            logger.info(f"PSF calibration: flux={F_ref:.4e} -> m_ref={m_ref:.3f}")

            # Memoised so repeated calls at the same magnitude are free.
            @lru_cache(maxsize=512)
            def flux_for_mag(m: float) -> float:
                return 10.0 ** (-0.4 * (m - m_ref))

            # =================================================================
            # Choose quiet injection sites
            # =================================================================
            DETECTION_BETA_THRESH = detection_cutoff
            snr_limit = float(detection_limit) if detection_limit is not None else None
            recovery_method = str(lim_cfg.get("recovery_method", "PSF")).strip().upper()
            # Accept common synonyms and any letter case.
            if recovery_method in {"MCMC", "EMCEE"}:
                recovery_method = "EMCEE"
            elif recovery_method in {"PSF", "AP"}:
                recovery_method = recovery_method
            else:
                recovery_method = "PSF"
            completeness_target = float(lim_cfg.get("completeness_target", 0.5))
            completeness_target = max(0.0, min(1.0, completeness_target))
            if not np.isfinite(completeness_target):
                completeness_target = 0.5

            logger.info(
                "Injected limiting magnitude: recovery_method=%s, completeness_target=%.2f",
                str(recovery_method),
                float(completeness_target),
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
            beta_n = 3.0
            # Monte Carlo settings for injection/recovery.
            # Too few sites/jitters makes the completeness curve look jagged.
            sourceNum = int(lim_cfg.get("injection_n_sites", 30))
            sourceNum = max(8, min(sourceNum, 200))
            redo_default = int(lim_cfg.get("injection_jitter_repetitions", 5))
            redo_default = max(1, min(redo_default, 20))
            distance_factor = 1.0
            injection_df = pd.DataFrame()

            # Injection radii (in pixels): keep injected sources near-by but
            # never on top of the transient itself.
            fwhm_px = float(self.input_yaml.get("fwhm", 3.0))
            r_min = float(lim_cfg.get("inject_min_radius_fwhm", 2.0)) * fwhm_px
            r_max = float(lim_cfg.get("inject_max_radius_fwhm", 6.0)) * fwhm_px
            r_base = float(lim_cfg.get("inject_source_location", 3.0)) * fwhm_px
            r_base = float(np.clip(r_base, r_min, r_max))
            inj_strategy = str(lim_cfg.get("injection_strategy", "ring_quiet")).strip().lower()
            if inj_strategy not in {"ring_quiet", "annulus_random"}:
                inj_strategy = "ring_quiet"

            for attempt in range(3):
                inj_dist = float(np.clip(r_base * distance_factor, r_min, r_max))
                if inj_strategy == "annulus_random":
                    # Sample positions uniformly in area within the local annulus.
                    # (r ~ sqrt(u)) and theta ~ U[0, 2pi).
                    theta = self._rng.random(sourceNum) * (2.0 * np.pi)
                    rr = np.sqrt(self._rng.random(sourceNum)) * (r_max - r_min) + r_min
                    xran = position[0] + rr * np.cos(theta)
                    yran = position[1] + rr * np.sin(theta)
                    df = pd.DataFrame({"x_pix": xran, "y_pix": yran})
                else:
                    pts = points_in_circum(inj_dist, center=position, n=sourceNum)
                    xran = [p[0] for p in pts]
                    yran = [p[1] for p in pts]
                    df = pd.DataFrame({"x_pix": xran, "y_pix": yran})

                ini_ap = Aperture(input_yaml=self.input_yaml, image=cutout)
                df = ini_ap.measure(
                    sources=df,
                    plot=False,
                    background_rms=background_rms,
                    verbose=0,
                )
                p_det = df.apply(
                    lambda row: beta_aperture(
                        n=beta_n,
                        flux_aperture=row["flux_AP"],
                        sigma=row["noiseSky"],
                        npix=row["area"],
                    ),
                    axis=1,
                )
                # "Quiet" sites: choose positions with detection probability below
                # the same cutoff used later to define what counts as a detection.
                injection_df = df[p_det < DETECTION_BETA_THRESH].copy()

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
                    cutout_img, cutout_rms = _extract_cutouts(scale_used)
                    if cutout_img is None:
                        continue
                    cutout = cutout_img
                    background_rms = cutout_rms

                    distance_factor = 1.0
                    injection_df = pd.DataFrame()
                    for attempt in range(3):
                        inj_dist = float(np.clip(r_base * distance_factor, r_min, r_max))
                        if inj_strategy == "annulus_random":
                            theta = self._rng.random(sourceNum) * (2.0 * np.pi)
                            rr = (
                                np.sqrt(self._rng.random(sourceNum)) * (r_max - r_min)
                                + r_min
                            )
                            xran = position[0] + rr * np.cos(theta)
                            yran = position[1] + rr * np.sin(theta)
                            df = pd.DataFrame({"x_pix": xran, "y_pix": yran})
                        else:
                            pts = points_in_circum(inj_dist, center=position, n=sourceNum)
                            xran = [p[0] for p in pts]
                            yran = [p[1] for p in pts]
                            df = pd.DataFrame({"x_pix": xran, "y_pix": yran})

                        ini_ap = Aperture(input_yaml=self.input_yaml, image=cutout)
                        df = ini_ap.measure(
                            sources=df,
                            plot=False,
                            background_rms=background_rms,
                            verbose=0,
                        )
                        p_det = df.apply(
                            lambda row: beta_aperture(
                                n=beta_n,
                                flux_aperture=row["flux_AP"],
                                sigma=row["noiseSky"],
                                npix=row["area"],
                            ),
                            axis=1,
                        )
                        injection_df = df[p_det < DETECTION_BETA_THRESH].copy()
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
                    xran = position[0] + rr * np.cos(theta)
                    yran = position[1] + rr * np.sin(theta)
                    injection_df = pd.DataFrame({"x_pix": xran, "y_pix": yran})
                else:
                    pts = points_in_circum(inj_dist, center=position, n=sourceNum)
                    injection_df = pd.DataFrame(
                        {"x_pix": [p[0] for p in pts], "y_pix": [p[1] for p in pts]}
                    )

            if len(injection_df) > sourceNum:
                injection_df = injection_df.sample(sourceNum).reset_index(drop=True)

            x_pix_arr = injection_df["x_pix"].to_numpy()
            y_pix_arr = injection_df["y_pix"].to_numpy()
            n_sites = len(injection_df)

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
            rng = self._rng

            def run_trials_at_mag(m: float, redo: int = None, pool=None, *, return_flags: bool = False):
                """
                Inject at *m* at all sites with *redo* sub-pixel jitter
                repetitions and return (mean_detection_rate, beta_array[, det_flags]).
                """
                redo = int(redo_default if redo is None else redo)
                redo = max(1, min(redo, 50))
                dx = rng.random((n_sites, redo)) - 0.5
                dy = rng.random((n_sites, redo)) - 0.5
                F = flux_for_mag(m)

                tasks = [
                    (
                        x_pix_arr[k] + dx[k, j],
                        y_pix_arr[k] + dy[k, j],
                        F,
                        gridx,
                        gridy,
                        cutout,
                        epsf_model,
                        self.input_yaml,
                        background_rms,
                        snr_limit,
                        beta_n,
                        DETECTION_BETA_THRESH,
                        recovery_method,
                    )
                    for k in range(n_sites)
                    for j in range(redo)
                ]

                if pool is not None:
                    results = list(pool.map(_injection_worker, tasks))
                else:
                    results = [_injection_worker(t) for t in tasks]

                det_flags = np.array([r[0] for r in results], dtype=bool)
                betas = np.array([r[1] for r in results], dtype=float)
                if return_flags:
                    return float(det_flags.mean()), betas, det_flags
                return float(det_flags.mean()), betas

            # =================================================================
            # Single ProcessPoolExecutor for the ENTIRE search (or serial if n_jobs==1)
            # =================================================================
            inject_lmag = np.nan
            bracket_steps: list[tuple] = []
            bisect_steps: list[tuple] = []

            with _pool_or_serial(n_jobs) as pool:

                # ---- Bracket phase ------------------------------------------
                step = 0.5
                max_steps = 30
                m_bright = float(initialGuess)
                c_bright, _ = run_trials_at_mag(m_bright, pool=pool)
                going_faint = c_bright >= completeness_target
                m_faint, c_faint = m_bright, c_bright

                bracket_steps.append((m_bright, c_bright))

                for _ in range(max_steps):
                    m_test = m_faint + step if going_faint else m_bright - step
                    c_test, _ = run_trials_at_mag(m_test, pool=pool)
                    bracket_steps.append((m_test, c_test))

                    if going_faint:
                        m_faint, c_faint = m_test, c_test
                        if c_faint < completeness_target:
                            break
                    else:
                        m_bright, c_bright = m_test, c_test
                        if c_bright >= completeness_target:
                            break

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
                        c_bright, _ = run_trials_at_mag(m_bright, pool=pool)
                        going_faint = c_bright >= completeness_target
                        m_faint, c_faint = m_bright, c_bright
                        bracket_steps.append((m_bright, c_bright))
                        for _ in range(35):
                            m_test = m_faint + step if going_faint else m_bright - step
                            c_test, _ = run_trials_at_mag(m_test, pool=pool)
                            bracket_steps.append((m_test, c_test))
                            if going_faint:
                                m_faint, c_faint = m_test, c_test
                                if c_faint < completeness_target:
                                    break
                            else:
                                m_bright, c_bright = m_test, c_test
                                if c_bright >= completeness_target:
                                    break
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
                    # Conservative estimate: faintest mag tried when stepping faint, else brightest when stepping bright
                    inject_lmag = float(m_faint if going_faint else m_bright)
                    logger.info(
                        f"Could not bracket cutoff; using conservative limit {inject_lmag:.3f}"
                    )

                else:
                    # ---- Bisect phase ----------------------------------------
                    lo_m, lo_c = m_bright, c_bright
                    hi_m, hi_c = m_faint, c_faint
                    bisect_steps = [(lo_m, lo_c), (hi_m, hi_c)]

                    for _ in range(30):
                        mid_m = 0.5 * (lo_m + hi_m)
                        mid_c, _ = run_trials_at_mag(mid_m, pool=pool)
                        bisect_steps.append((mid_m, mid_c))

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

                # ---- Plot completeness curve (still inside pool context) -----
                if plot:
                    try:
                        span = float(lim_cfg.get("completeness_plot_span_mag", 0.8))
                    except Exception:
                        span = 0.8
                    span = max(0.3, min(span, 2.5))
                    try:
                        nm = int(lim_cfg.get("completeness_plot_nmags", 9))
                    except Exception:
                        nm = 9
                    nm = max(5, min(nm, 21))
                    m_c = float(inject_lmag) if np.isfinite(inject_lmag) else float(initialGuess)
                    sample_mags = np.linspace(m_c - span, m_c + span, nm)

                    completeness_groups, medians = [], []
                    for m in sample_mags:
                        # For plotting completeness, use the actual detection flags
                        # (det = beta>=thresh and optional SNR gate), not the beta values.
                        _, _betas, det_flags = run_trials_at_mag(
                            m, pool=pool, return_flags=True
                        )
                        completeness_groups.append(
                            np.asarray(det_flags, dtype=float)
                        )
                        medians.append(float(np.mean(det_flags)))

                    self._plot_completeness(
                        sample_mags,
                        completeness_groups,
                        medians,
                        bracket_steps,
                        bisect_steps,
                        inject_lmag,
                        completeness_target,
                        DETECTION_BETA_THRESH,
                        zeropoint,
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
                try:
                    if zeropoint is not None and np.isfinite(float(zeropoint)):
                        app_str = (
                            f" ({float(zeropoint) + float(inject_lmag):.3f} apparent)"
                        )
                except Exception:
                    app_str = ""
                logger.info(
                    f"Limiting mag ~ {inject_lmag:.3f}{app_str}  [{elapsed:.1f}s]"
                )
            else:
                logger.info(f"Limiting magnitude search failed [{elapsed:.1f}s]")

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
        x0c, y0c = float(position[0]), float(position[1])
        fwhm_px = float(self.input_yaml.get("fwhm", 3.0))
        r_min = float(lim_cfg.get("inject_min_radius_fwhm", 2.0)) * fwhm_px
        r_base = float(lim_cfg.get("inject_source_location", 3.0)) * fwhm_px
        r_inj = float(max(r_min, r_base))
        pts = points_in_circum(r_inj, center=(x0c, y0c), n=8)
        x0, y0 = float(pts[0][0]), float(pts[0][1])
        gridx, gridy = np.meshgrid(np.arange(W), np.arange(H))
        F = float(flux_for_mag(float(m_inj)))
        img = np.asarray(cutout, dtype=float) + np.asarray(
            epsf_model.evaluate(x=gridx, y=gridy, flux=F, x_0=x0, y_0=y0),
            dtype=float,
        )

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
            gain=float(self.input_yaml.get("gain", 1.0)),
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
            _, _, flags = run_trials_at_mag(float(m), redo=int(redo), pool=pool, return_flags=True)
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

            fig, ax = plt.subplots(figsize=set_size(540, 1))
            ax.plot(mags, emp, "o", ms=4, color="0.2", label="empirical")

            # median model curve
            s_med = float(np.exp(np.nanmedian(flat[:, 1])))
            mm = np.linspace(np.nanmin(mags), np.nanmax(mags), 200)
            p_med = 1.0 / (1.0 + np.exp(np.clip((mm - m50) / s_med, -60, 60)))
            ax.plot(mm, p_med, "-", lw=1.0, color="#1f77b4", label="logistic (median)")

            ax.axhline(float(completeness_target), color="0.6", lw=0.6, ls="--")
            ax.axvline(float(m50), color="k", lw=0.6, ls="--", label=f"m50={m50:.3f}")
            ax.set_xlabel("Injected ePSF instrumental magnitude")
            ax.set_ylabel("Recovery fraction")
            ax.set_ylim(-0.05, 1.05)
            ax.invert_xaxis()
            ax.legend(loc="upper left", fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(save_png, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except Exception:
            pass

        return m50, m50_err

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
    ) -> None:
        """
        Save a completeness curve PNG next to the FITS file.

        Extracted from getInjectedLimit to keep that method focused on the
        search logic and to make the plot code independently testable.
        """
        order = np.argsort(sample_mags)
        mags_sorted = sample_mags[order]
        groups_sorted = [completeness_groups[i] for i in order]
        medians_sorted = np.asarray([medians[i] for i in order], float)

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
        fig, ax = plt.subplots(figsize=set_size(540, 1))

        # Bar chart of recovery fraction (what the solver actually uses).
        # The previous boxplot-of-0/1 outcomes was visually confusing and
        # looked "wrong" even when the recovery fractions were fine.
        widths = 0.15
        n_trials = np.array([max(1, len(g)) for g in groups_sorted], dtype=float)
        p = np.clip(medians_sorted, 0.0, 1.0)
        # Binomial standard error (normal approx) for visual guidance.
        p_err = np.sqrt(np.maximum(p * (1.0 - p) / n_trials, 0.0))

        ax.bar(
            mags_sorted,
            p,
            width=widths,
            color="#B0C4DE",
            edgecolor="#4D4D4D",
            linewidth=0.6,
            label="Sampled recovery",
            zorder=2,
        )
        ax.errorbar(
            mags_sorted,
            p,
            yerr=p_err,
            fmt="none",
            ecolor="#4D4D4D",
            elinewidth=0.6,
            capsize=2,
            zorder=3,
        )

        # Bracket and bisect search trajectories.
        if bracket_steps:
            bm, bc = zip(*bracket_steps)
            ax.plot(
                bm,
                bc,
                "-",
                ms=3.5,
                lw=0.8,
                color="#0000FF",
                label="Bracket path",
            )
        if bisect_steps:
            bm, bc = zip(*bisect_steps)
            ax.plot(
                bm,
                bc,
                "--",
                ms=3.5,
                lw=0.8,
                color="#00AA00",
                label="Bisect path",
            )

        # Reference lines.
        ax.axhline(0.5, color="0.7", lw=0.5, ls="--", zorder=0)
        ax.text(
            0.02,
            0.5,
            "50%",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            color="0.5",
        )

        if completeness_target != 0.5:
            ax.axhline(completeness_target, color="0.7", lw=0.5, ls="-.", zorder=0)
            ax.text(
                0.98,
                completeness_target,
                f"{int(100 * completeness_target)}%",
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

        ax.set_xlabel("Injected ePSF brightness [mag]")
        ax.set_ylabel("Recovery fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.invert_xaxis()
        ax.legend(loc="upper left", fontsize=7, frameon=False)

        # Optional apparent-magnitude secondary axis.
        if zeropoint is not None:
            secax = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda m: m + zeropoint,
                    lambda m: m - zeropoint,
                ),
            )
            secax.set_xlabel("Apparent brightness [mag]")
            secax.invert_xaxis()

        fig.tight_layout()
        save_loc_png = os.path.join(write_dir, f"Completeness_{base}.png")
        fig.savefig(save_loc_png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
