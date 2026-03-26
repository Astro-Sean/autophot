#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSF construction, fitting, and diagnostics.

This module builds an oversampled PSF model, performs PSF fitting (including
optional MCMC-based sampling), and generates diagnostic plots used by the
autophot pipeline.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import inspect
import logging
import os
import time
import traceback
import copy
from dataclasses import dataclass
from math import ceil

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import corner
import emcee
from emcee import autocorr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.fft import fft2, fftshift
from scipy.optimize import least_squares
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Astropy
# ---------------------------------------------------------------------------
import astropy.units as u
from astropy.io import fits
from astropy.nddata import NDData, StdDevUncertainty
from astropy.stats import SigmaClip, mad_std, sigma_clipped_stats
from astropy.table import QTable, Table
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.fitting import TRFLSQFitter, SLSQPLSQFitter

# ---------------------------------------------------------------------------
# Photutils
# ---------------------------------------------------------------------------
from photutils.background import LocalBackground, MedianBackground, MMMBackground
from photutils.centroids import centroid_com, centroid_2dg, centroid_sources
from photutils.psf import (
    EPSFBuilder,
    EPSFFitter,
    EPSFStars,
    extract_stars,
    PSFPhotometry,
    SourceGrouper,
    IterativePSFPhotometry,
    ImagePSF,
)
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_threshold
from photutils.utils import calc_total_error
from photutils.utils.cutouts import overlap_slices

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from functions import border_msg, set_size, log_warning_from_exception

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _nddata_clone(nd: NDData, data: Optional[np.ndarray] = None) -> NDData:
    """
    Clone an NDData object without relying on NDData.copy() (not present in older astropy).
    Preserves uncertainty, unit, wcs, and meta.
    """
    new_data = np.array(nd.data, copy=True) if data is None else data
    unc = getattr(nd, "uncertainty", None)
    new_unc = None
    if unc is not None:
        try:
            # StdDevUncertainty supports copy on the array; keep unitless array.
            arr = np.array(getattr(unc, "array", unc), copy=True)
            new_unc = StdDevUncertainty(arr)
        except Exception:
            new_unc = copy.deepcopy(unc)
    return NDData(
        new_data,
        uncertainty=new_unc,
        unit=getattr(nd, "unit", None),
        wcs=getattr(nd, "wcs", None),
        meta=copy.deepcopy(getattr(nd, "meta", {})),
    )


# ===========================================================================
# Module-level utilities (shared by multiple classes / methods)
# ===========================================================================


def _odd(n: int) -> int:
    """Return *n* if odd, else *n* + 1."""
    n = int(n)
    return n + (n % 2 == 0)


def get_quartic_kernel(oversample: int = 4, kernel_size: int = None) -> np.ndarray:
    """
    Quartic 2-D smoothing kernel for ePSF construction (Anderson & King 2000).

    Fits a 2-D quartic polynomial via least-squares to an impulse target (1
    at centre, 0 elsewhere) and returns the normalised result.

    Parameters
    ----------
    oversample   : int   oversampling factor; governs default kernel_size
    kernel_size  : int or None  odd kernel side length; defaults to
                   min(2*oversample + 1, 7)

    Returns
    -------
    ndarray  (kernel_size, kernel_size)  normalised kernel
    """
    if kernel_size is None:
        kernel_size = min(2 * oversample + 1, 7)
    if kernel_size % 2 == 0:
        kernel_size += 1

    half = kernel_size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]

    target = np.zeros((kernel_size, kernel_size))
    target[half, half] = 1.0

    def _residuals(p):
        model = (
            1
            + p[0] * x
            + p[1] * y
            + p[2] * x**2
            + p[3] * x * y
            + p[4] * y**2
            + p[5] * x**3
            + p[6] * x**2 * y
            + p[7] * x * y**2
            + p[8] * y**3
            + p[9] * x**4
            + p[10] * x**3 * y
            + p[11] * x**2 * y**2
            + p[12] * x * y**3
            + p[13] * y**4
        )
        return (model - target).ravel()

    res = least_squares(_residuals, np.zeros(14))
    p = res.x
    kernel = (
        1
        + p[0] * x
        + p[1] * y
        + p[2] * x**2
        + p[3] * x * y
        + p[4] * y**2
        + p[5] * x**3
        + p[6] * x**2 * y
        + p[7] * x * y**2
        + p[8] * y**3
        + p[9] * x**4
        + p[10] * x**3 * y
        + p[11] * x**2 * y**2
        + p[12] * x * y**3
        + p[13] * y**4
    )
    return kernel / kernel.sum()


def get_smoothing_kernel(
    fwhm: float, oversample: int = 4, kernel_size: int = None
) -> np.ndarray:
    """
    Return the quartic smoothing kernel for ePSF construction.

    *fwhm* is accepted for API compatibility but not used - the quartic
    polynomial kernel is preferred over a Gaussian for ePSF smoothing.
    """
    return get_quartic_kernel(oversample, kernel_size)


def centroid_com_with_error(data, mask=None, error=None, xpeak=None, ypeak=None):
    """
    Centre-of-mass centroid compatible with centroid_sources_with_error.

    Returns (x, y, x_err, y_err) as Python floats.

    Notes
    -----
    Extracted from the psf class because it is passed as a *callable* to
    centroid_sources_with_error - it never needs self.
    """
    # `photutils.centroid_com` can return NaNs when the input data contains
    # substantial negative values (common after background subtraction).
    # Here we compute a non-negative "flux" (offset by -nanmin and clip) and
    # derive both the centroid and uncertainty from that.
    yy, xx = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]

    flux = np.array(data, dtype=float, copy=True)
    if mask is not None:
        flux = np.where(mask, 0.0, flux)

    finite = np.isfinite(flux)
    if not np.any(finite):
        return np.nan, np.nan, np.nan, np.nan

    offset = float(np.nanmin(flux[finite]))
    flux = flux - offset
    flux[~np.isfinite(flux)] = 0.0
    flux[flux < 0] = 0.0

    total = float(np.sum(flux))
    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan

    xcen = float(np.sum(flux * xx) / total)
    ycen = float(np.sum(flux * yy) / total)

    # Variance about centroid under the same flux model.
    xvar = float(np.sum(flux * (xx - xcen) ** 2) / total)
    yvar = float(np.sum(flux * (yy - ycen) ** 2) / total)

    # COM uncertainty proxy; preserve original behaviour/scale.
    xerr = float(np.sqrt(xvar / total)) if xvar >= 0 else np.nan
    yerr = float(np.sqrt(yvar / total)) if yvar >= 0 else np.nan

    return xcen, ycen, xerr, yerr


def centroid_2dg_with_error(data, mask=None, error=None, xpeak=None, ypeak=None):
    """
    2D Gaussian centroid compatible with centroid_sources_with_error (4-tuple API).
    Uses error for weighting when provided; returns (x, y, nan, nan) for uncertainties.
    """
    xcen, ycen = centroid_2dg(data, mask=mask, error=error)
    return (
        float(np.atleast_1d(xcen)[0]),
        float(np.atleast_1d(ycen)[0]),
        np.nan,
        np.nan,
    )


def _centroid_sources_with_error_impl(
    data,
    xpos,
    ypos,
    box_size=11,
    footprint=None,
    mask=None,
    centroid_func=centroid_com_with_error,
    **kwargs,
):
    """
    Standalone centroid_sources that returns (x, y, xerr, yerr). Used by
    EPSFBuilder.centroid_sources_with_error and by catalog.recenter when error is provided.
    """
    xpos = np.atleast_1d(xpos)
    ypos = np.atleast_1d(ypos)

    if isinstance(box_size, tuple):
        box_size = box_size[0]
    box_size = _odd(int(box_size))

    if xpos.ndim != 1 or ypos.ndim != 1:
        raise ValueError("xpos and ypos must be 1-D arrays.")

    if (
        np.any(xpos < 0)
        or np.any(ypos < 0)
        or np.any(xpos > data.shape[1] - 1)
        or np.any(ypos > data.shape[0] - 1)
    ):
        raise ValueError("xpos/ypos contain points outside image bounds.")

    if footprint is None:
        if box_size is None:
            raise ValueError("box_size or footprint must be defined.")
        ny = nx = int(box_size)
        if ny % 2 == 0 or nx % 2 == 0:
            raise ValueError("box_size values must be odd integers")
        footprint = np.ones((ny, nx), dtype=bool)
    else:
        footprint = np.asanyarray(footprint, dtype=bool)
        if footprint.ndim != 2:
            raise ValueError("footprint must be 2-D.")

    spec = inspect.signature(centroid_func)
    if "mask" not in spec.parameters:
        raise ValueError('centroid_func must accept a "mask" keyword.')

    centroid_kwargs = {k: v for k, v in kwargs.items() if k in spec.parameters}

    xcentroids, ycentroids, xerrs, yerrs = [], [], [], []

    for xp, yp in zip(xpos, ypos, strict=True):
        slices_large, slices_small = overlap_slices(
            data.shape, footprint.shape, (yp, xp)
        )
        data_cutout = data[slices_large]
        footprint_mask = ~footprint[slices_small]

        mask_cutout = (
            np.logical_or(mask[slices_large], footprint_mask)
            if mask is not None
            else footprint_mask
        )

        if np.all(mask_cutout):
            raise ValueError(
                f"Cutout for source at ({xp:.1f}, {yp:.1f}) is fully masked."
            )

        centroid_kwargs["mask"] = mask_cutout
        err_arr = centroid_kwargs.get("error")
        if err_arr is not None:
            centroid_kwargs["error"] = err_arr[slices_large]

        centroid_kwargs.pop("xpeak", None)
        centroid_kwargs.pop("ypeak", None)

        try:
            xcen, ycen, xerr, yerr = centroid_func(data_cutout, **centroid_kwargs)
        except (ValueError, TypeError):
            xcen, ycen, xerr, yerr = np.nan, np.nan, np.nan, np.nan

        xcentroids.append(xcen + slices_large[1].start)
        ycentroids.append(ycen + slices_large[0].start)
        xerrs.append(xerr)
        yerrs.append(yerr)

    return (
        np.array(xcentroids),
        np.array(ycentroids),
        np.array(xerrs),
        np.array(yerrs),
    )


# ---------------------------------------------------------------------------
# Covariance container (replaces the original one-liner class)
# ---------------------------------------------------------------------------


@dataclass
class Covariance:
    """Thin wrapper around a covariance matrix for fitter compatibility."""

    cov_matrix: np.ndarray


# ===========================================================================
# MCMCFitter
# ===========================================================================


class MCMCFitter:
    """
    Adaptive emcee-based fitter with the same __call__ signature as the
    photutils least-squares fitters.

    Changes vs. original
    --------------------
    * No re-imports inside __call__ or run_mcmc.
    * logger obtained once at top of __call__ rather than twice.
    * Covariance now a dataclass (no behavioural change).
    """

    # When nsteps is None, run until convergence (tau < adaptive_tau_target) up to this cap.
    _max_steps_when_auto = 50_000

    def __init__(
        self,
        nwalkers: int = 50,
        nsteps: Optional[int] = 5000,
        delta: float = 10.0,
        burnin_frac: float = 0.3,
        thin: int = 10,
        random_state=None,
        inplace=None,
        adaptive_tau_target: int = 50,
        min_autocorr_N: int = 100,
        batch_steps: int = 100,
        jitter_scale: float = 0.01,
        use_nddata_uncertainty: bool = False,
        gain: float = 1.0,
        readnoise: float = 0.0,
        background_rms=None,
    ):
        self.nwalkers = int(nwalkers)
        self.nsteps = int(nsteps) if nsteps is not None else None
        self.delta = float(delta)
        self.burnin_frac = float(burnin_frac)
        self.thin = int(thin)
        self.adaptive_tau_target = adaptive_tau_target
        self.min_autocorr_N = min_autocorr_N
        self.batch_steps = int(batch_steps)
        self.jitter_scale = float(jitter_scale)
        self.random_state = (
            np.random.RandomState(random_state)
            if random_state is not None
            else np.random
        )
        self.sampler = None
        self.fit_info = {"samples": {}, "per_source": []}
        self.counter = 0
        # When True, __call__ uses these for per-pixel variance (e.g. when invoked by photutils without kwargs)
        self._use_nddata_uncertainty = bool(use_nddata_uncertainty)
        self._gain = float(gain)
        self._readnoise = float(readnoise)
        self._background_rms = background_rms

    # ---- Prior / likelihood -----------------------------------------------

    def _in_bounds(self, value, bounds):
        lo = -np.inf if bounds[0] is None else bounds[0]
        hi = np.inf if bounds[1] is None else bounds[1]
        return lo <= value <= hi

    def log_prior(self, params, model):
        for name, param_value in zip(model.param_names, params):
            if not self._in_bounds(param_value, getattr(model, name).bounds):
                return -np.inf
            n = name.lower()
            if any(k in n for k in ("flux", "amplitude", "amp")) and param_value < 0:
                return -np.inf
            if (
                any(k in n for k in ("sigma", "stddev", "fwhm", "alpha", "beta"))
                and param_value <= 0
            ):
                return -np.inf
        return 0.0

    def log_likelihood(self, params, model, x, y, data, weights, noise_variance):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        data = np.asarray(data, float)

        if weights is not None:
            weights = np.asarray(weights, float)
            if weights.shape != data.shape:
                try:
                    weights = np.broadcast_to(weights, data.shape)
                except Exception:
                    return -np.inf

        if np.isscalar(noise_variance):
            var = np.full(data.shape, float(noise_variance))
        else:
            var = np.asarray(noise_variance, float)
            if var.shape != data.shape:
                try:
                    var = np.broadcast_to(var, data.shape)
                except Exception:
                    return -np.inf

        model.parameters = params
        mu = model(x, y)
        resid = data - mu

        if weights is not None:
            var = var / np.clip(weights, 1e-12, None) ** 2

        if not np.all(var > 0):
            return -np.inf

        return -0.5 * np.sum(resid**2 / var + np.log(2.0 * np.pi * var))

    def log_posterior(self, params, model, x, y, data, weights, noise_variance):
        lp = self.log_prior(params, model)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(
            params, model, x, y, data, weights, noise_variance
        )

    # ---- Walker initialisation --------------------------------------------

    def _jitter_within_bounds(self, initial_params, model, scale: float = None):
        """Scatter walkers near initial params (scale ~ 1% of typical magnitude) for better mixing."""
        if scale is None:
            scale = self.jitter_scale
        ndim = len(initial_params)
        pos = np.empty((self.nwalkers, ndim), float)
        for i in range(self.nwalkers):
            trial = initial_params + scale * self.random_state.randn(ndim)
            for j, name in enumerate(model.param_names):
                lo, hi = getattr(model, name).bounds
                lo = -1e18 if lo is None else lo
                hi = 1e18 if hi is None else hi
                if trial[j] < lo:
                    trial[j] = lo + (lo - trial[j])
                if trial[j] > hi:
                    trial[j] = hi - (trial[j] - hi)
            pos[i] = trial
        return pos

    # ---- Noise variance helper --------------------------------------------

    def _recompute_noise_variance(
        self,
        data,
        gain=1.0,
        readnoise=0.0,
        background_rms=None,
        psf_error_floor_frac=0.0,
        smooth_variance_sigma=0.0,
    ):
        """
        Compute per-pixel sigma in electrons matching create_nddata_with_fitting_weights.
        """
        image_e = np.asarray(data, float) * float(gain)
        bkg_rms_e = (
            np.zeros_like(image_e)
            if background_rms is None
            else np.asarray(background_rms, float) * float(gain)
        )
        bkg_error = np.sqrt(bkg_rms_e**2 + float(readnoise) ** 2)
        total_error = calc_total_error(
            data=image_e, bkg_error=bkg_error, effective_gain=1.0
        )

        if psf_error_floor_frac > 0:
            psf_floor = float(psf_error_floor_frac) * np.clip(image_e, 0.0, None)
            total_error = np.sqrt(total_error**2 + psf_floor**2)

        if smooth_variance_sigma > 0:
            total_error = gaussian_filter(
                total_error, float(smooth_variance_sigma), mode="nearest"
            )

        return np.maximum(total_error, 1e-12)

    # ---- MCMC run ---------------------------------------------------------

    def run_mcmc(
        self,
        model,
        x,
        y,
        data,
        weights,
        initial_params,
        noise_variance,
        log,
        maxiters=None,
    ):
        pos0 = self._jitter_within_bounds(initial_params, model)
        ndim = len(initial_params)

        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            self.log_posterior,
            args=(model, x, y, data, weights, noise_variance),
        )

        total_steps = 0
        max_steps = (
            self.nsteps if self.nsteps is not None else self._max_steps_when_auto
        )
        if self.nsteps is None:
            log.info(
                "[MCMC] nsteps not set; running until convergence (max %d steps).",
                max_steps,
            )

        while total_steps < max_steps:
            batch_n = min(self.batch_steps, max_steps - total_steps)
            self.sampler.run_mcmc(
                pos0 if total_steps == 0 else None,
                batch_n,
                progress=False,
            )
            total_steps += batch_n

            # Start autocorrelation checks only after enough steps per walker.
            if total_steps >= self.min_autocorr_N:
                try:
                    # get_chain() is (nsteps, nwalkers, ndim); time axis is 0
                    tau = autocorr.integrated_time(self.sampler.get_chain(), axis=0)
                    tau_est = np.nanmean(tau)
                    if np.isfinite(tau_est) and tau_est < self.adaptive_tau_target:
                        log.info(
                            "[MCMC] Converged at %d steps, tau=%.1f",
                            total_steps,
                            tau_est,
                        )
                        break
                except Exception as exc:
                    log.debug("[MCMC] tau estimation failed: %s", exc)

        acc = float(np.mean(self.sampler.acceptance_fraction))
        tau_arr = None
        try:
            tau_arr = self.sampler.get_autocorr_time(quiet=True)
            log.info(f"[MCMC] acc={acc:.3f}  tau~{np.nanmean(tau_arr):.1f}")
        except Exception:
            log.info(f"[MCMC] acc={acc:.3f}")
        self._last_tau = tau_arr  # for discard/thin in __call__

        if not 0.15 <= acc <= 0.7:
            log.warning("[MCMC] Suboptimal acceptance; consider tuning delta/nwalkers")

    # ---- __call__ ---------------------------------------------------------

    def __call__(
        self,
        model,
        x,
        y,
        data,
        weights=None,
        initial_params=None,
        maxiters=None,
        noise_variance=1.0,
        inplace=None,
        gain=1.0,
        readnoise=0.0,
        background_rms=None,
        psf_error_floor_frac=0.0,
        smooth_variance_sigma=0.0,
        mask=None,
        use_nddata_uncertainty=False,
    ):
        t0 = time.time()
        log = logging.getLogger(__name__)

        # Use instance noise settings when fitter was created for MCMC from fit() (photutils does not pass these)
        if getattr(self, "_use_nddata_uncertainty", False):
            use_nddata_uncertainty = True
            gain = self._gain
            readnoise = self._readnoise
            background_rms = self._background_rms

        initial_params = (
            np.array(model.parameters, float)
            if initial_params is None
            else np.array(initial_params, float)
        )
        fitted_model = model.copy()

        for pname, p0 in zip(fitted_model.param_names, initial_params):
            if pname in ("x_0", "y_0"):
                getattr(fitted_model, pname).bounds = (p0 - self.delta, p0 + self.delta)

        if use_nddata_uncertainty:
            # _recompute_noise_variance returns per-pixel sigma; likelihood expects variance (sigma^2)
            sigma = self._recompute_noise_variance(
                data,
                gain=gain,
                readnoise=readnoise,
                background_rms=background_rms,
                psf_error_floor_frac=psf_error_floor_frac,
                smooth_variance_sigma=smooth_variance_sigma,
            )
            noise_variance = np.asarray(sigma, float) ** 2
            weights = None  # use our variance only; avoid double-counting if caller passed weights
        else:
            # If the caller provides an explicit per-pixel variance map, treat it
            # as authoritative and ignore extra weights to avoid ambiguity.
            if weights is not None and not np.isscalar(noise_variance):
                try:
                    noise_variance_arr = np.asarray(noise_variance, float)
                    if noise_variance_arr.size > 1:
                        log.debug(
                            "[MCMC] Per-pixel noise_variance provided; ignoring weights to avoid duplicate uncertainty scaling."
                        )
                        weights = None
                except Exception:
                    pass

        log.info(f"[MCMC] Fitting source {self.counter + 1} (adaptive)")
        self.run_mcmc(
            fitted_model,
            x,
            y,
            data,
            weights,
            initial_params,
            noise_variance,
            log,
            maxiters=maxiters,
        )

        discard = int(max(0, round(self.burnin_frac * self.sampler.iteration)))
        thin = max(1, self.thin)
        if discard >= self.sampler.iteration:
            discard = max(0, self.sampler.iteration // 2)
        # Use integrated autocorrelation time when available for principled burn-in/thin
        tau_arr = getattr(self, "_last_tau", None)
        if tau_arr is not None and np.any(np.isfinite(tau_arr)):
            tau_max = float(np.nanmax(tau_arr))
            if tau_max >= 1:
                discard = max(
                    discard, min(int(2 * tau_max), self.sampler.iteration // 2)
                )
                # Ensure thinning is not weaker than tau/2 (reduce correlated samples).
                thin = max(1, thin, int(np.ceil(tau_max / 2)))

        chain = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        logp = self.sampler.get_log_prob(discard=discard, thin=thin, flat=True)

        if chain.shape[0] == 0:
            log.error("[MCMC] No samples after burn-in/thinning.")
            per16 = per50 = per84 = best_params = initial_params.copy()
        else:
            per16 = np.nanpercentile(chain, 16, axis=0)
            per50 = np.nanpercentile(chain, 50, axis=0)
            per84 = np.nanpercentile(chain, 84, axis=0)
            best_params = per50

        fitted_model.parameters = best_params

        perr_lo = best_params - per16
        perr_hi = per84 - best_params
        perr_sym = np.maximum(0.5 * (perr_lo + perr_hi), 0.0)

        fitted_model.stds = perr_sym
        fitted_model.cov_matrix = Covariance(np.diag(perr_sym**2))

        record = {
            "src_index": self.counter,
            "param_names": list(fitted_model.param_names),
            "p16": per16.copy(),
            "p50": per50.copy(),
            "p84": per84.copy(),
        }
        self.fit_info["per_source"].append(record)
        self.fit_info["samples"][self.counter] = chain
        self.fit_info["log_prob"] = logp
        self.fit_info["param_errs"] = perr_sym
        self.fit_info["param_errs_lower"] = perr_lo
        self.fit_info["param_errs_upper"] = perr_hi
        self.fit_info["burnin_discard"] = discard
        self.fit_info["thin"] = thin
        self.fit_info["acceptance_fraction"] = float(
            np.mean(self.sampler.acceptance_fraction)
        )
        self.fit_info["total_steps"] = int(self.sampler.iteration)

        self.counter += 1
        log.info(f"[MCMC] elapsed={time.time() - t0:.3f}s")
        return fitted_model


# ===========================================================================
# psf class
# ===========================================================================


class PSF:
    """
    Point Spread Function construction, fitting, and diagnostics.

    Methods
    -------
    centroid_sources_with_error() - centroid with uncertainty propagation
    create_nddata_with_fitting_weights() - NDData with per-pixel sigma
    robust_extract_stars()   - PSF-star cutout extraction with centroiding
    build()                  - ePSF construction via EPSFBuilder
    fit()                    - tiered-SNR PSF photometry
    plot()                   - diagnostic PSF/residual panels
    plot_oversampled_PSF()   - oversampled PSF image with projections
    _create_psf_visualization() - star-cutout + ePSF summary plot
    _plot_mcmc_corner()      - corner plot for MCMC fitter output
    """

    def __init__(self, input_yaml: dict, image: np.ndarray, header=None):
        self.input_yaml = input_yaml
        self.image = image
        self.header = header

    # -----------------------------------------------------------------------
    # Centroiding
    # -----------------------------------------------------------------------

    def centroid_sources_with_error(
        self,
        data,
        xpos,
        ypos,
        box_size=11,
        footprint=None,
        mask=None,
        centroid_func=centroid_com_with_error,
        **kwargs,
    ):
        """
        centroid_sources equivalent that also returns centroid uncertainties.

        Returns
        -------
        xcentroid, ycentroid, xerr, yerr : np.ndarray
        """
        return _centroid_sources_with_error_impl(
            data,
            xpos,
            ypos,
            box_size=box_size,
            footprint=footprint,
            mask=mask,
            centroid_func=centroid_func,
            **kwargs,
        )

    # -----------------------------------------------------------------------
    # NDData creation
    # -----------------------------------------------------------------------

    def create_nddata_with_fitting_weights(
        self,
        image: np.ndarray,
        gain: float = 1.0,
        read_noise: float = 0.0,
        background_rms: np.ndarray = None,
        is_background_subtracted: bool = True,
        psf_error_floor_frac: float = 0.0,
        smooth_variance_sigma: float = 0.0,
        mask: np.ndarray = None,
    ) -> NDData:
        """
        Build an NDData with per-pixel StdDevUncertainty in electrons.

        The error model is:
            sigma_total = sqrt(sigma_Poisson2 + sigma_sky2 + sigma_read2)
        optionally plus a PSF-model-mismatch floor and optional smoothing.
        """
        image_e = np.asarray(image, float) * float(gain)
        bkg_rms_e = (
            np.zeros_like(image_e)
            if background_rms is None
            else np.asarray(background_rms, float) * float(gain)
        )
        bkg_error = np.sqrt(bkg_rms_e**2 + float(read_noise) ** 2)
        total_error = calc_total_error(
            data=image_e, bkg_error=bkg_error, effective_gain=1.0
        )

        if psf_error_floor_frac > 0:
            psf_floor = float(psf_error_floor_frac) * np.clip(image_e, 0.0, None)
            total_error = np.sqrt(total_error**2 + psf_floor**2)

        if smooth_variance_sigma > 0:
            total_error = gaussian_filter(
                total_error, float(smooth_variance_sigma), mode="nearest"
            )

        np.maximum(total_error, 1e-12, out=total_error)

        return NDData(
            data=image_e,
            uncertainty=StdDevUncertainty(total_error, unit=u.electron),
            unit=u.electron,
            mask=mask,
        )

    # -----------------------------------------------------------------------
    # Star extraction
    # -----------------------------------------------------------------------

    def robust_extract_stars(
        self, ndimage, df, cutout_shape, fwhm, fit_boxsize, log, weightmap=None
    ):
        """
        Extract centroided star cutouts for EPSFBuilder.

        Parameters
        ----------
        ndimage      : NDData
        df           : DataFrame with 'x_pix', 'y_pix'
        cutout_shape : (ny, nx) tuple
        fwhm         : float or None
        fit_boxsize  : int  (used only for _odd enforcement here)
        log          : Logger
        weightmap    : 2-D array or None

        Returns
        -------
        (EPSFStars, Table)
        """
        try:
            cutout_shape = (_odd(cutout_shape[0]), _odd(cutout_shape[1]))
            fit_boxsize = _odd(fit_boxsize)

            ny, nx = ndimage.data.shape
            hy, hx = cutout_shape[0] // 2, cutout_shape[1] // 2

            x = df["x_pix"].to_numpy(float)
            y = df["y_pix"].to_numpy(float)
            keep = (x >= hx) & (x < nx - hx) & (y >= hy) & (y < ny - hy)

            dropped = int((~keep).sum())
            if dropped:
                log.info(f"[robust_extract_stars] Dropped {dropped} near-edge sources")

            x, y = x[keep], y[keep]
            if x.size == 0:
                log.error("[robust_extract_stars] No candidates after edge filter")
                return EPSFStars([]), Table()

            # Centroiding.
            phot_cfg = self.input_yaml.get("photometry", {}) or {}
            undersampled_thr = float(
                phot_cfg.get("undersampled_fwhm_threshold", 2.5)
            )
            undersampled_mode = bool(
                self.input_yaml.get(
                    "undersampled_mode",
                    bool(fwhm is not None and float(fwhm) <= undersampled_thr),
                )
            )
            cen_func = centroid_2dg if undersampled_mode else centroid_com
            cen_box = _odd(max(7, int(np.ceil(3.0 * max(1.0, fwhm or 2.0)))))

            if weightmap is None:
                weightmap = 1.0 / ndimage.uncertainty.array

            try:
                x_cen, y_cen = centroid_sources(
                    ndimage.data, x, y, box_size=cen_box, centroid_func=cen_func
                )
            except Exception as exc:
                log.warning(
                    f"[robust_extract_stars] Centroiding failed ({exc}); using COM"
                )
                x_cen, y_cen = centroid_sources(
                    ndimage.data, x, y, box_size=cen_box, centroid_func=centroid_com
                )

            good = np.isfinite(x_cen) & np.isfinite(y_cen)
            n_bad = int((~good).sum())
            if n_bad:
                log.info(f"[robust_extract_stars] Dropped {n_bad} non-finite centroids")
            x_cen, y_cen = x_cen[good], y_cen[good]

            if x_cen.size == 0:
                log.error("[robust_extract_stars] No valid centroids")
                return EPSFStars([]), Table()

            stars_tbl = Table({"x": x_cen, "y": y_cen})
            epsfstars = extract_stars(ndimage, stars_tbl, size=cutout_shape)
            log.info(f"[robust_extract_stars] Extracted {len(epsfstars)} cutouts")

            # Assign per-pixel weights if provided.
            weight_nddata = NDData(weightmap)
            weight_cutouts = extract_stars(weight_nddata, stars_tbl, size=cutout_shape)

            if len(weight_cutouts) == len(epsfstars):
                for star, wcut in zip(epsfstars, weight_cutouts):
                    star.weights = wcut.data.astype(float)
            else:
                log.warning(
                    "[robust_extract_stars] Weight/star count mismatch; skipping weights"
                )

            return epsfstars, stars_tbl

        except Exception:
            log.error("[robust_extract_stars] Fatal:\n" + traceback.format_exc())
            return EPSFStars([]), Table()

    # -----------------------------------------------------------------------
    # FFT outlier detection
    # -----------------------------------------------------------------------

    def compute_power_spectrum(self, cutout: np.ndarray) -> np.ndarray:
        """2-D power spectrum of *cutout* via FFT."""
        return np.abs(fftshift(fft2(cutout))) ** 2

    def detect_fft_outliers(
        self,
        epsfstars,
        n_sigma: float = 4.0,
        use_mad: bool = True,
        use_median_deviation: bool = True,
    ) -> EPSFStars:
        """
        Remove ePSF cutouts whose power spectrum deviates from the typical (median) spectrum.

        Uses a robust per-star scalar: either median or mean of per-pixel normalized
        deviations. Scale (dispersion) is MAD or std per pixel across stars. This avoids
        rejecting good stars when a few contaminants inflate the variance.
        """
        spectra = []
        for st in epsfstars:
            d = st.data.copy()
            d -= np.nanmin(d)
            d[d < 0] = 0.0
            spectra.append(self.compute_power_spectrum(d))

        spectra = np.array(spectra, dtype=float)
        med = np.nanmedian(spectra, axis=0)
        if use_mad:
            # MAD along axis=0; 1.4826 for consistency with Gaussian
            scale = np.nanmedian(np.abs(spectra - med), axis=0)
            scale = 1.4826 * np.where(scale > 0, scale, np.nanmedian(scale))
        else:
            scale = np.nanstd(spectra, axis=0)
        scale = np.where(scale > 1e-12, scale, 1e-12)

        deviations = np.abs(spectra - med) / scale
        # Per-star scalar: reduce over both spatial axes (axis 1 and 2)
        if use_median_deviation:
            star_metric = np.nanmedian(deviations, axis=(1, 2))
        else:
            star_metric = np.nanmean(deviations, axis=(1, 2))

        kept = [
            st
            for i, st in enumerate(epsfstars)
            if np.isfinite(star_metric[i]) and star_metric[i] < n_sigma
        ]
        return EPSFStars(kept)

    # -----------------------------------------------------------------------
    # ePSF construction
    # -----------------------------------------------------------------------

    def build(
        self,
        psfSources,
        usePSFlist: bool = False,
        numSources: int = 250,
        mask=None,
        background_rms=None,
        threshold_limit: float = 5.0,
        SNR_limit: list = None,
        plot: bool = True,
        filename_prefix: str = "PSF_model_image",
        make_template_psf: bool = False,
    ):
        """
        Build an ePSF model from robustly selected, spatially uniform stars.

        Returns
        -------
        (epsf, df) or (None, None) on failure
        """
        if SNR_limit is None:
            SNR_limit = [5, 1e6]

        log = logging.getLogger(__name__)

        # ---- nested helpers ------------------------------------------------
        def _validate_epsfstars(epsfstars_obj, cutout_shape, fit_boxsize):
            cy = (cutout_shape[0] - 1) / 2.0
            cx = (cutout_shape[1] - 1) / 2.0
            lim = 0.45 * (fit_boxsize - 1)
            kept = []
            for st in epsfstars_obj:
                try:
                    data = np.asarray(st.data, float)
                    if (
                        data.ndim != 2
                        or not np.isfinite(data).all()
                        or np.nansum(data) <= 0
                    ):
                        continue
                    y0, x0 = getattr(st, "cutout_center", (cy, cx))
                    if not (np.isfinite(y0) and np.isfinite(x0)):
                        continue
                    if np.hypot(x0 - cx, y0 - cy) > lim:
                        continue
                    kept.append(st)
                except Exception:
                    continue
            return EPSFStars(kept) if kept else EPSFStars([])

        def _locate_columns(df):
            xcols = ["xcentroid", "x", "x_fit", "x0", "x_pix", "col"]
            ycols = ["ycentroid", "y", "y_fit", "y0", "y_pix", "row"]
            snrcols = ["SNR", "snr", "signal_to_noise", "flux_snr"]
            thrcols = ["threshold", "thresh", "det_thresh"]
            return (
                next((c for c in xcols if c in df.columns), None),
                next((c for c in ycols if c in df.columns), None),
                next((c for c in snrcols if c in df.columns), None),
                next((c for c in thrcols if c in df.columns), None),
            )

        def _select_uniform(df, num_keep, image_shape, xcol, ycol, snrcol, thrcol):
            if df.empty:
                return df
            ny, nx = image_shape
            work = df.copy()
            work = work[np.isfinite(work[xcol]) & np.isfinite(work[ycol])]
            if work.empty:
                return work

            if snrcol and snrcol in work.columns:
                work["_rank"] = work[snrcol].values
            elif thrcol and thrcol in work.columns:
                work["_rank"] = work[thrcol].values
            else:
                fluxcol = next(
                    (
                        c
                        for c in ("flux_psf", "flux_ap", "flux", "FLUX_AUTO")
                        if c in work.columns
                    ),
                    None,
                )
                work["_rank"] = (
                    work[fluxcol].values
                    if fluxcol
                    else np.random.default_rng(42).random(len(work))
                )
            # Grid size: aim for ~num_keep cells but at least 2×2
            side = int(np.ceil(np.sqrt(max(4, num_keep))))
            xedges = np.linspace(0, nx, side + 1)
            yedges = np.linspace(0, ny, side + 1)
            xi = np.clip(np.digitize(work[xcol].values, xedges) - 1, 0, side - 1)
            yi = np.clip(np.digitize(work[ycol].values, yedges) - 1, 0, side - 1)
            work["_cell"] = yi * side + xi
            # Shuffle cell order so we do not preferentially select from any
            # particular region purely due to indexing.
            rng = np.random.default_rng(42)
            cells = np.arange(side * side, dtype=int)
            rng.shuffle(cells)

            selected_idx = []
            # First pass: take the best source from each non-empty cell.
            for cell in cells:
                rows = work[work["_cell"] == cell]
                if rows.empty:
                    continue
                best_idx = rows["_rank"].astype(float).idxmax()
                selected_idx.append(best_idx)
                if len(selected_idx) >= num_keep:
                    break

            # If we still need more, take second-best, third-best, ... per cell
            # in additional passes before falling back to global ranking. This
            # keeps the sample more spatially uniform across the image.
            pass_num = 2
            while len(selected_idx) < num_keep:
                added_this_pass = 0
                for cell in cells:
                    rows = work[work["_cell"] == cell].sort_values(
                        "_rank", ascending=False
                    )
                    if rows.empty:
                        continue
                    # Skip any rows already chosen in previous passes
                    rows = rows[~rows.index.isin(selected_idx)]
                    if rows.empty:
                        continue
                    best_idx = rows["_rank"].astype(float).idxmax()
                    selected_idx.append(best_idx)
                    added_this_pass += 1
                    if len(selected_idx) >= num_keep:
                        break
                if added_this_pass == 0:
                    break

            selected = work.loc[selected_idx].copy()
            if len(selected) < num_keep:
                remaining = work.drop(index=selected.index).sort_values(
                    "_rank", ascending=False
                )
                selected = pd.concat(
                    [selected, remaining.head(num_keep - len(selected))], axis=0
                )
            return selected.drop(
                columns=["_rank", "_cell"], errors="ignore"
            ).reset_index(drop=True)

        # ---- main ----------------------------------------------------------
        try:
            df = (
                pd.read_csv(usePSFlist)
                if isinstance(usePSFlist, str)
                else psfSources.copy()
            )
            log.info(f"Building ePSF from {len(df)} sources")

            # Exclude saturated and streaky/elongated sources from PSF building.
            # For small candidate pools, apply these cuts adaptively so we do not
            # over-prune and end up with an unstable ePSF from too few stars.
            phot_cfg = self.input_yaml.get("photometry", {}) or {}
            saturate = float(self.input_yaml.get("saturate", np.inf))
            saturate_frac = float(phot_cfg.get("psf_saturate_fraction", 0.9))
            min_psf_candidates = int(phot_cfg.get("psf_min_candidates", 8))
            min_psf_candidates = max(4, min_psf_candidates)
            min_keep_frac_after_cut = float(
                phot_cfg.get("psf_build_min_keep_frac_after_cut", 0.60)
            )
            min_keep_frac_after_cut = max(0.10, min(0.95, min_keep_frac_after_cut))
            thr_cfg = phot_cfg.get("psf_threshold_limit", threshold_limit)
            threshold_limit_eff = (
                float(thr_cfg)
                if thr_cfg is not None and np.isfinite(float(thr_cfg))
                else None
            )
            snr_min_eff = float(phot_cfg.get("psf_snr_min", SNR_limit[0]))
            snr_max_eff = float(phot_cfg.get("psf_snr_max", SNR_limit[1]))

            n_before = len(df)
            # When the starting pool is modest, relax shape/saturation cuts slightly.
            if n_before <= max(2 * min_psf_candidates, 20):
                saturate_frac = max(saturate_frac, 0.97)
                log.info(
                    "Small PSF-candidate pool (%d): relaxed cuts to "
                    "saturate_frac=%.2f",
                    n_before,
                    saturate_frac,
                )
            if saturate < np.inf:
                peak_col = next(
                    (c for c in ["peak", "peak_flux", "FLUX_MAX"] if c in df.columns),
                    None,
                )
                if peak_col is not None:
                    sat_cut = saturate_frac * saturate
                    ok = df[peak_col] < sat_cut
                    n_sat = (~ok).sum()
                    if n_sat > 0:
                        n_keep = int(np.sum(ok))
                        if n_keep >= min_psf_candidates:
                            df = df[ok].copy()
                            log.info(
                                f"Excluded {n_sat} saturated PSF candidates (peak >= {saturate_frac:.2f} * saturate)"
                            )
                        else:
                            log.info(
                                "Skipping saturation cut: would leave only %d candidates (< %d).",
                                n_keep,
                                min_psf_candidates,
                            )
            if len(df) < n_before:
                log.info(
                    f"PSF candidates after saturation cuts: {len(df)}/{n_before}"
                )

            fpath = self.input_yaml["fpath"]
            write_dir = self.input_yaml["write_dir"]
            base = os.path.basename(fpath).split(".")[0]
            oversample = max(
                1, int(self.input_yaml["photometry"].get("psf_oversample", 1))
            )

            if make_template_psf:
                # Template/science header may lack APER, GAIN, FWHM; use fallbacks.
                _fwhm_h = float(self.header.get("FWHM", self.header.get("fwhm", 3.0)))
                _aper = self.header.get("APER", self.header.get("aper"))
                aperture_radius = float(
                    _aper
                    if _aper is not None
                    else self.input_yaml.get("photometry", {}).get("aperture_radius")
                    or 1.5 * _fwhm_h
                )
                _gain = self.header.get("GAIN", self.header.get("gain"))
                gain = float(
                    _gain if _gain is not None else self.input_yaml.get("gain", 1.0)
                )
                _fwhm = self.header.get("FWHM", self.header.get("fwhm"))
                fwhm = float(
                    _fwhm if _fwhm is not None else self.input_yaml.get("fwhm", _fwhm_h)
                )
            else:
                aperture_radius = float(
                    self.input_yaml["photometry"]["aperture_radius"]
                )
                gain = float(self.input_yaml["gain"])
                fwhm = float(self.input_yaml["fwhm"])

            scale = float(self.input_yaml["scale"])
            fwhm_input = float(fwhm)

            # Re-estimate FWHM from the selected PSF-star table when possible.
            # This keeps the ePSF build geometry tied to the actual stars used,
            # rather than a potentially stale global image FWHM.
            fwhm_cols = (
                "FWHM_IMAGE",
                "fwhm_image",
                "FWHM",
                "fwhm",
            )
            fwhm_candidates = None
            for col in fwhm_cols:
                if col in df.columns:
                    arr = np.asarray(df[col], float)
                    arr = arr[np.isfinite(arr) & (arr > 0)]
                    if arr.size >= 5:
                        lo, hi = np.nanpercentile(arr, [10, 90])
                        arr = arr[(arr >= lo) & (arr <= hi)]
                    if arr.size >= 3:
                        fwhm_candidates = arr
                        break
            if fwhm_candidates is not None and fwhm_candidates.size >= 3:
                fwhm_psf = float(np.nanmedian(fwhm_candidates))
                fwhm_guard_lo = 0.5 * fwhm_input
                fwhm_guard_hi = 2.0 * fwhm_input
                if np.isfinite(fwhm_input) and fwhm_input > 0:
                    if fwhm_psf < fwhm_guard_lo or fwhm_psf > fwhm_guard_hi:
                        log.warning(
                            "PSF-star FWHM median %.2f px is far from runtime FWHM %.2f px; clamping build FWHM to %.2f px.",
                            fwhm_psf,
                            fwhm_input,
                            float(np.clip(fwhm_psf, fwhm_guard_lo, fwhm_guard_hi)),
                        )
                    fwhm = float(np.clip(fwhm_psf, fwhm_guard_lo, fwhm_guard_hi))
                else:
                    fwhm = fwhm_psf
            else:
                fwhm = fwhm_input

            pixel_scale = self.input_yaml.get("pixel_scale", np.nan)  # arcsec/pix
            try:
                pixel_scale = float(pixel_scale)
            except Exception:
                pixel_scale = np.nan
            has_pixel_scale = np.isfinite(pixel_scale) and pixel_scale > 0
            fwhm_arcsec = float(fwhm * pixel_scale) if has_pixel_scale else np.nan

            # Adaptive geometric boost for coarse/undersampled data.
            # Coarser sampling generally needs larger windows to capture wings robustly.
            build_sampling_boost = 1.0
            if fwhm <= 2.5:
                build_sampling_boost += 0.20
            if has_pixel_scale and pixel_scale >= 0.8:
                build_sampling_boost += 0.10
            if has_pixel_scale and pixel_scale >= 1.2:
                build_sampling_boost += 0.10
            build_boost_cap = float(phot_cfg.get("psf_build_sampling_boost_max", 1.6))

            # Detect undersampling: FWHM <= 2 pixels is a typical regime where
            # a PSF spans only a handful of pixels across the core and is at
            # high risk of aliasing.  We *do not* change the oversampling
            # factor automatically (user controls this via config), but we
            # still adapt centroiding and cutout sizes below.
            undersampled_fwhm_threshold = float(
                phot_cfg.get("undersampled_fwhm_threshold", 2.5)
            )
            undersampled = fwhm <= undersampled_fwhm_threshold
            if undersampled:
                build_boost_cap_u = float(
                    phot_cfg.get("psf_build_sampling_boost_max_undersampled", 2.2)
                )
                build_boost_cap = max(build_boost_cap, build_boost_cap_u)
            build_sampling_boost = float(min(build_sampling_boost, max(1.0, build_boost_cap)))
            if undersampled and oversample <= 1:
                log.info(
                    "Image appears undersampled (FWHM=%.2f pix) and psf_oversample=%d; "
                    "consider increasing psf_oversample in the config if enough PSF "
                    "stars are available.",
                    fwhm,
                    oversample,
                )
            if undersampled and bool(
                phot_cfg.get("psf_disable_build_quality_cuts_undersampled", True)
            ):
                threshold_limit_eff = None
                snr_min_eff, snr_max_eff = 0.0, 1.0e12
                min_keep_frac_after_cut = min(min_keep_frac_after_cut, 0.30)
                log.info(
                    "Undersampled regime (FWHM <= %.2f px): disabling strict threshold/SNR candidate cuts for PSF build.",
                    undersampled_fwhm_threshold,
                )

            # Image preprocessing.
            img = np.array(self.image, float)
            img[~np.isfinite(img)] = np.nanmedian(img)
            if mask is not None and mask.shape == img.shape:
                img[mask.astype(bool)] = np.nanmedian(img)

            # Use a PSF-shape-aware centroiding choice: for clearly
            # undersampled data, a 2-D Gaussian centroid is generally more
            # stable than a pure centre-of-mass on a few bright pixels.
            recenter_func = centroid_2dg if undersampled else centroid_com
            log.info(
                f"FWHM={fwhm:.2f} pix  recenter={recenter_func.__name__}  "
                f"oversample={oversample}x"
            )

            # Centroid and fit windows scale with FWHM, but must remain odd.
            cen_box = _odd(max(7, int(np.ceil(2.0 * fwhm))))
            fit_boxsize_scale_base = float(
                phot_cfg.get("psf_fit_boxsize_scale_fwhm", 2.5)
            )
            fit_boxsize_scale = fit_boxsize_scale_base * build_sampling_boost
            fit_box_min_arcsec = float(phot_cfg.get("psf_fit_box_min_arcsec", 2.5))
            fit_box_min_px = (
                fit_box_min_arcsec / pixel_scale if has_pixel_scale else np.nan
            )
            fit_boxsize = _odd(
                max(
                    5,
                    int(fit_boxsize_scale * fwhm),
                    int(np.ceil(fit_box_min_px)) if np.isfinite(fit_box_min_px) else 0,
                )
            )

            # PSF cutout size: use both the configured angular scale and the
            # image-space FWHM to guarantee that even undersampled wings are
            # represented.  For undersampled images the FWHM constraint
            # dominates; for well-sampled data this reduces to the previous
            # behaviour.
            cutout_size_scale_base = float(
                phot_cfg.get("psf_cutout_size_scale_fwhm", 10.0)
            )
            cutout_size_scale = cutout_size_scale_base * build_sampling_boost
            cutout_min_arcsec = float(phot_cfg.get("psf_cutout_min_arcsec", 9.0))
            cutout_min_px = cutout_min_arcsec / pixel_scale if has_pixel_scale else np.nan
            cutout_n = _odd(
                max(
                    12,
                    int(2 * scale),  # legacy behaviour (arcsec scale)
                    int(cutout_size_scale * fwhm),  # configurable wing coverage
                    int(np.ceil(cutout_min_px)) if np.isfinite(cutout_min_px) else 0,
                )
            )
            cutout_shape = (cutout_n, cutout_n)
            if fit_boxsize >= cutout_n - 2:
                fit_boxsize = _odd(cutout_n - 3)

            log.info(f"Initial sources: {len(df)}")
            if threshold_limit_eff is not None and "threshold" in df.columns:
                mask_thr = df["threshold"] > threshold_limit_eff
                n_keep_thr = int(np.sum(mask_thr))
                min_keep_thr = max(
                    min_psf_candidates,
                    int(np.ceil(min_keep_frac_after_cut * max(1, len(df)))),
                )
                if n_keep_thr >= min_keep_thr:
                    df = df[mask_thr]
                else:
                    log.info(
                        "Skipping threshold cut (>%s): would leave %d candidates (< required %d).",
                        threshold_limit_eff,
                        n_keep_thr,
                        min_keep_thr,
                    )
            if "SNR" in df.columns:
                mask_snr = (df["SNR"] >= snr_min_eff) & (df["SNR"] <= snr_max_eff)
                n_keep_snr = int(np.sum(mask_snr))
                min_keep_snr = max(
                    min_psf_candidates,
                    int(np.ceil(min_keep_frac_after_cut * max(1, len(df)))),
                )
                if n_keep_snr >= min_keep_snr:
                    df = df[mask_snr]
                else:
                    log.info(
                        "Skipping strict SNR cut ([%s, %s]): would leave %d candidates (< required %d).",
                        snr_min_eff,
                        snr_max_eff,
                        n_keep_snr,
                        min_keep_snr,
                    )
            log.info(f"Sources after cuts: {len(df)}")

            if len(df) == 0:
                log.error("No PSF candidates after filtering.")
                return None, df

            xcol, ycol, snrcol, thrcol = _locate_columns(df)
            if xcol is None or ycol is None:
                log.error("Required coordinate columns not found.")
                return None, df

            # Optionally pre-select the highest-quality candidates (by SNR /
            # threshold / flux) before enforcing spatial uniformity, so that
            # the downsampled set is drawn from the best stars available.
            preselect_factor = float(phot_cfg.get("psf_preselect_factor", 1.3))
            if preselect_factor > 1.0 and len(df) > numSources:
                rank_col = None
                if snrcol and snrcol in df.columns:
                    rank_col = snrcol
                elif thrcol and thrcol in df.columns:
                    rank_col = thrcol
                else:
                    for c in ("flux_psf", "flux_ap", "flux", "FLUX_AUTO"):
                        if c in df.columns:
                            rank_col = c
                            break
                if rank_col is not None:
                    n_top = min(len(df), int(np.ceil(preselect_factor * numSources)))
                    if n_top > numSources:
                        df = (
                            df.sort_values(rank_col, ascending=False)
                            .head(n_top)
                            .reset_index(drop=True)
                        )
                        log.info(
                            "Preselecting top %d PSF candidates by %s before spatial downselection",
                            n_top,
                            rank_col,
                        )

            # Enforce a spatially uniform PSF-star sample to avoid the ePSF
            # being dominated by any single region of the detector (e.g. only
            # near the target).  This greatly reduces systematic PSF variation
            # and light-curve scatter across epochs.
            df_uniform = _select_uniform(
                df,
                num_keep=min(numSources, len(df)),
                image_shape=img.shape,
                xcol=xcol,
                ycol=ycol,
                snrcol=snrcol,
                thrcol=thrcol,
            )
            log.info(
                f"Using {len(df_uniform)}/{len(df)} PSF candidates after spatial downselection"
            )
            df = df_uniform

            if "SNR" in df.columns:
                df = df.sort_values("SNR", ascending=False).reset_index(drop=True)

            ndimage = self.create_nddata_with_fitting_weights(
                image=self.image.copy(),
                gain=float(self.input_yaml.get("gain", gain)),
                read_noise=float(self.input_yaml.get("readnoise", 0.0)),
                background_rms=background_rms,
            )

            epsfstars, stars_tbl = self.robust_extract_stars(
                ndimage, df, cutout_shape, fwhm, fit_boxsize, log
            )

            if len(epsfstars) == 0:
                log.error("All PSF-star candidates rejected.")
                return None, df

            # Optional: remove pathological PSF-star cutouts (blends, cosmic rays) via
            # power-spectrum comparison. Configurable; uses robust MAD + median deviation.
            fft_cfg = phot_cfg if phot_cfg is not None else {}
            do_fft = fft_cfg.get("psf_fft_rejection", True)
            fft_n_sigma = float(fft_cfg.get("psf_fft_n_sigma", 4.0))
            fft_min_frac = float(fft_cfg.get("psf_fft_min_frac_keep", 0.25))
            fft_min_abs = max(5, int(fft_cfg.get("psf_fft_min_stars", 10)))

            if do_fft and len(epsfstars) >= 8:
                epsfstars_clean = self.detect_fft_outliers(
                    epsfstars,
                    n_sigma=fft_n_sigma,
                    use_mad=fft_cfg.get("psf_fft_use_mad", True),
                    use_median_deviation=fft_cfg.get(
                        "psf_fft_use_median_deviation", True
                    ),
                )
                min_keep = max(fft_min_abs, int(fft_min_frac * len(epsfstars)))
                if len(epsfstars_clean) < min_keep:
                    # Retry with relaxed sigma before giving up
                    relaxed_sigma = fft_n_sigma * 1.5
                    epsfstars_relaxed = self.detect_fft_outliers(
                        epsfstars,
                        n_sigma=relaxed_sigma,
                        use_mad=fft_cfg.get("psf_fft_use_mad", True),
                        use_median_deviation=fft_cfg.get(
                            "psf_fft_use_median_deviation", True
                        ),
                    )
                    if len(epsfstars_relaxed) >= min_keep:
                        log.info(
                            f"FFT rejection at n_sigma={fft_n_sigma} left too few stars; "
                            f"using relaxed n_sigma={relaxed_sigma:.1f} -> {len(epsfstars_relaxed)} stars."
                        )
                        epsfstars_clean = epsfstars_relaxed
                if len(epsfstars_clean) >= min_keep:
                    log.info(
                        f"Rejected {len(epsfstars) - len(epsfstars_clean)} "
                        "PSF stars via FFT outlier filtering"
                    )
                    epsfstars = epsfstars_clean
                else:
                    log.info(
                        "FFT outlier rejection would leave too few stars "
                        f"({len(epsfstars_clean)}/{len(epsfstars)}); keeping original set."
                    )
            elif do_fft and len(epsfstars) < 8:
                log.info("Skipping FFT rejection (fewer than 8 PSF stars).")

            smooth_kernel = get_smoothing_kernel(fwhm, oversample=oversample)

            # Use a normalization radius that is at least a configurable multiple
            # of the PSF-build FWHM to preserve wing structure.
            norm_scale = float(phot_cfg.get("psf_norm_radius_scale_fwhm", 2.8))
            norm_min_px = float(phot_cfg.get("psf_norm_radius_min_pixels", 5.0))
            aperture_radius = float(
                max(aperture_radius, norm_scale * fwhm, norm_min_px)
            )

            log.info(
                "PSF build FWHM: input=%.2f px, used=%.2f px; normalisation radius=%.2f px (>= max[input, %.2f*FWHM, %.1f px]); oversample=x%d",
                fwhm_input,
                fwhm,
                aperture_radius,
                norm_scale,
                norm_min_px,
                oversample,
            )

            epsf_clip_sigma = float(phot_cfg.get("psf_build_sigma_clip_sigma", 4.0))
            epsf_clip_maxiters = int(
                max(1, phot_cfg.get("psf_build_sigma_clip_maxiters", 15))
            )
            sigma_clip_epsf = SigmaClip(
                sigma=epsf_clip_sigma,
                cenfunc=np.nanmedian,
                stdfunc=mad_std,
                maxiters=epsf_clip_maxiters,
            )
            log.info(
                border_msg("PSF build parameters")
                + "\n"
                + (
                    "  make_template_psf: {make_template_psf}\n"
                    "  candidates: initial={n_before}, after_cuts={n_after}, used_after_uniform={n_uniform}\n"
                    "  selection cuts: min_candidates={min_psf_candidates}, threshold_limit={threshold_limit}, "
                    "SNR_limit=[{snr_lo}, {snr_hi}], min_keep_frac_after_cut={min_keep_frac_after_cut:.2f}, "
                    "saturate_frac={saturate_frac:.2f}\n"
                    "  geometry: fwhm_input={fwhm_input:.2f}px, fwhm_used={fwhm_used:.2f}px, "
                    "fwhm_arcsec={fwhm_arcsec}, pixel_scale={pixel_scale}, "
                    "build_sampling_boost={build_sampling_boost:.2f}, "
                    "oversample=x{oversample}, recenter={recenter}, cen_box={cen_box}px, fit_box={fit_box}px, cutout={cutout}px\n"
                    "  normalization: aperture_radius={aperture_radius:.2f}px, "
                    "norm_scale={norm_scale:.2f}*FWHM, norm_min_px={norm_min_px:.1f}\n"
                    "  epsf builder: sigma_clip={epsf_clip_sigma:.2f}, sigma_clip_maxiters={epsf_clip_maxiters}, "
                    "maxiters=10, smoothing_kernel=quartic\n"
                    "  fft rejection: enabled={fft_enabled}, n_sigma={fft_n_sigma:.2f}, "
                    "min_frac_keep={fft_min_frac:.2f}, min_stars={fft_min_abs}"
                ).format(
                    make_template_psf=bool(make_template_psf),
                    n_before=int(n_before),
                    n_after=int(len(df)),
                    n_uniform=int(len(df_uniform)),
                    min_psf_candidates=int(min_psf_candidates),
                    threshold_limit=(
                        float(threshold_limit_eff)
                        if threshold_limit_eff is not None
                        else np.nan
                    ),
                    snr_lo=float(snr_min_eff),
                    snr_hi=float(snr_max_eff),
                    min_keep_frac_after_cut=float(min_keep_frac_after_cut),
                    saturate_frac=float(saturate_frac),
                    fwhm_input=float(fwhm_input),
                    fwhm_used=float(fwhm),
                    fwhm_arcsec=(
                        f"{fwhm_arcsec:.2f}\"" if np.isfinite(fwhm_arcsec) else "unknown"
                    ),
                    pixel_scale=(
                        f"{pixel_scale:.3f} arcsec/px"
                        if has_pixel_scale
                        else "unknown"
                    ),
                    build_sampling_boost=float(build_sampling_boost),
                    oversample=int(oversample),
                    recenter=str(recenter_func.__name__),
                    cen_box=int(cen_box),
                    fit_box=int(fit_boxsize),
                    cutout=int(cutout_n),
                    aperture_radius=float(aperture_radius),
                    norm_scale=float(norm_scale),
                    norm_min_px=float(norm_min_px),
                    epsf_clip_sigma=float(epsf_clip_sigma),
                    epsf_clip_maxiters=int(epsf_clip_maxiters),
                    fft_enabled=bool(do_fft),
                    fft_n_sigma=float(fft_n_sigma),
                    fft_min_frac=float(fft_min_frac),
                    fft_min_abs=int(fft_min_abs),
                )
            )
            log.info(
                "PSF build windows: fit_box=%.2f*FWHM (base %.2f, min %.2f arcsec) -> %d px, "
                "cutout=%.2f*FWHM (base %.2f, min %.2f arcsec) -> %d px, sigma_clip=%.2f (%d iters)",
                fit_boxsize_scale,
                fit_boxsize_scale_base,
                fit_box_min_arcsec,
                fit_boxsize,
                cutout_size_scale,
                cutout_size_scale_base,
                cutout_min_arcsec,
                cutout_n,
                epsf_clip_sigma,
                epsf_clip_maxiters,
            )
            epsf_builder = EPSFBuilder(
                oversampling=oversample,
                recentering_func=recenter_func,
                recentering_boxsize=cen_box,
                recentering_maxiters=100,
                fitter=EPSFFitter(fit_boxsize=fit_boxsize),
                maxiters=10,
                norm_radius=aperture_radius,
                sigma_clip=sigma_clip_epsf,
                smoothing_kernel=smooth_kernel,
                progress_bar=False,
            )

            # Initialise ePSF from a Gaussian.
            kernel = Gaussian2DKernel(
                x_stddev=oversample * fwhm / 2.355,
                x_size=oversample * cutout_n,
                y_size=oversample * cutout_n,
            )
            cutout_ctr = (oversample * cutout_n - 1) / 2.0
            init_epsf = ImagePSF(
                data=kernel.array,
                x_0=cutout_ctr,
                y_0=cutout_ctr,
                oversampling=oversample,
            )

            epsf, fitted_stars = epsf_builder.build_epsf(epsfstars, init_epsf=init_epsf)

            # Enforce non-negativity of the final ePSF model. Small negative
            # lobes can appear from noise or overfitting; clipping them and
            # renormalising keeps the PSF physically meaningful while
            # preserving its overall shape.
            try:
                arr = np.asarray(epsf.data, float)
                arr = np.clip(arr, 0.0, None)
                s = arr.sum()
                if s > 0:
                    arr /= s
                epsf.data = arr
            except Exception:
                # If anything goes wrong, keep the original EPSF.
                pass

            if oversample > 1:
                self.plot_oversampled_psf(
                    epsf,
                    oversample=oversample,
                    save_path=os.path.join(
                        write_dir, f"{filename_prefix}_{base}_psf_image.png"
                    ),
                )

            save_path = os.path.join(write_dir, f"{filename_prefix}_{base}.fits")
            fits.PrimaryHDU(np.asarray(epsf.data, float)).writeto(
                save_path, overwrite=True
            )
            self._create_psf_visualization(
                fitted_stars,
                epsf,
                cutout_shape[0],
                aperture_radius,
                write_dir,
                f"{filename_prefix}_{base}",
            )

            return epsf, df

        except Exception as exc:
            log.error(f"[build] Fatal: {exc}\n{traceback.format_exc()}")
            return None, None

    # -----------------------------------------------------------------------
    # Oversampled PSF plot
    # -----------------------------------------------------------------------

    def plot_oversampled_psf(
        self,
        psf_model: Any,
        oversample: int = 1,
        save_path: Optional[str] = None,
        cmap: str = "viridis",
        use_zscale: bool = True,
        zscale_contrast: float = 0.25,
    ):
        """
        Three-panel figure: oversampled PSF image + X/Y projections.
        """
        log = logging.getLogger(__name__)

        if oversample <= 1:
            log.info("Skipping PSF plot: oversample <= 1.")
            return None

        try:
            data = np.asarray(psf_model.data, float)
            if data.ndim != 2:
                log.warning("PSF data must be 2-D.")
                return None
            if not np.isfinite(data).all():
                data = data.copy()
                data[~np.isfinite(data)] = 0.0

            # Display limits.
            if use_zscale:
                try:
                    vmin, vmax = ZScaleInterval(contrast=zscale_contrast).get_limits(
                        data
                    )
                except Exception as exc:
                    log_warning_from_exception(
                        log, "ZScale failed; using min/max", exc
                    )
                    vmin, vmax = np.nanmin(data), np.nanmax(data)
            else:
                vmin, vmax = np.nanmin(data), np.nanmax(data)

            _style = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
            )
            if os.path.exists(_style):
                plt.style.use(_style)
            fig, ax = plt.subplots(figsize=set_size(540, 1))
            divider = make_axes_locatable(ax)
            ax_R = divider.append_axes("right", size="25%", pad=0.25, sharey=ax)
            ax_B = divider.append_axes("bottom", size="25%", pad=0.25, sharex=ax)

            ny, nx = data.shape
            extent = [0, nx, 0, ny]

            im = ax.imshow(
                data,
                extent=extent,
                origin="lower",
                cmap=cmap,
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
            )

            cx, cy = nx // 2, ny // 2
            ax.axvline(cx, color="white", lw=0.5, alpha=0.8, ls="--")
            ax.axhline(cy, color="white", lw=0.5, alpha=0.8, ls="--")
            ax.set_title(f"Oversample={oversample}x", fontsize=8, pad=2)
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")
            ax.set_xticks([])

            x_phys = np.arange(nx)
            y_phys = np.arange(ny)
            hx = data.mean(axis=0)
            hy = data.mean(axis=1)

            ax_B.step(x_phys, hx, color="black", lw=0.5, where="mid")
            ax_R.step(hy, y_phys, color="black", lw=0.5, where="mid")
            ax_B.axvline(cx, color="black", lw=0.5, alpha=0.8, ls="--")
            ax_R.axhline(cy, color="black", lw=0.5, alpha=0.8, ls="--")
            ax_B.set_ylabel("Intensity")
            ax_B.set_xlabel("X Pixels")
            ax_R.yaxis.tick_right()
            ax_R.tick_params(axis="x", rotation=90)

            for axis in (ax_R, ax_B):
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                if axis is ax_R:
                    axis.xaxis.set_major_formatter(fmt)
                    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                else:
                    axis.yaxis.set_major_formatter(fmt)
                    axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                return None

            # plt.tight_layout()
            plt.show()
            return fig

        except Exception as exc:
            log.error(f"PSF plot failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # PSF visualisation (star cutouts + ePSF)
    # -----------------------------------------------------------------------

    def _create_psf_visualization(
        self, stars, epsf, star_shape, aperture_radius, write_dir, base
    ):
        """Grid of star cutouts alongside the final ePSF model."""
        num_stars = len(stars)
        ncols = int(np.ceil(np.sqrt(num_stars)))
        nrows = ceil(num_stars / ncols)
        # Right side: 2D and 3D each span multiple rows, block centred vertically
        span_2d = 2
        span_3d = 2
        block_height = span_2d + span_3d
        total_rows = max(nrows, block_height)

        _style = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
        )
        if os.path.exists(_style):
            plt.style.use(_style)
        plt.ioff()
        fig = plt.figure(figsize=set_size(540, 1.3))
        # Centre the right-hand block: equal empty rows above and below
        start_row = (total_rows - block_height) // 2
        height_ratios = [1] * total_rows
        # Slightly taller rows for the 2D/3D block so the panels are bigger
        for r in range(start_row, start_row + block_height):
            height_ratios[r] = 1.15
        # Main grid: left block (stars), gap column, right block (2D + 3D)
        grid = fig.add_gridspec(
            nrows=total_rows,
            ncols=ncols + 3,
            width_ratios=[1] * ncols + [0.12] + [1, 1],
            height_ratios=height_ratios,
            hspace=0,
            wspace=0,
        )
        # Left: star cutouts with no spacing between them
        left_sub = grid[:, 0:ncols].subgridspec(nrows, ncols, hspace=0, wspace=0)
        # Right: 2D and 3D with vertical spacing between them
        right_sub = grid[start_row : start_row + block_height, -2:].subgridspec(
            2, 1, hspace=0.2
        )

        xx, yy = np.mgrid[:star_shape, :star_shape]
        psf_model = epsf.evaluate(
            xx, yy, x_0=star_shape / 2, y_0=star_shape / 2, flux=1
        )

        all_data = np.concatenate([np.ravel(s) for s in stars] + [psf_model.ravel()])
        vmin, vmax = ZScaleInterval().get_limits(all_data)
        norm = ImageNormalize(vmin=vmin, vmax=vmax)

        for i in range(num_stars):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(left_sub[row, col])
            ax.imshow(
                stars[i],
                origin="lower",
                cmap="viridis",
                norm=norm,
                interpolation="none",
            )
            ctr = [stars[i].shape[1] / 2, stars[i].shape[0] / 2]
            ax.add_patch(
                Circle(
                    ctr,
                    aperture_radius,
                    color="#FF0000",
                    ls="-",
                    fill=False,
                    lw=0.5,
                )
            )
            ax.text(
                0.98,
                0.98,
                f"{i+1}/{num_stars}",
                transform=ax.transAxes,
                va="top",
                ha="right",
                color="white",
                fontsize=6,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        vmin_p, vmax_p = ZScaleInterval().get_limits(psf_model)
        norm_p = ImageNormalize(vmin=vmin_p, vmax=vmax_p)

        # 2D ePSF image (right side, top of right subgrid)
        ax_right = fig.add_subplot(right_sub[0, 0])
        im = ax_right.imshow(
            psf_model, origin="lower", cmap="viridis", norm=norm_p, interpolation="none"
        )
        ax_right.set_title("2D ePSF", fontsize=8, pad=2)
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        ctr = [psf_model.shape[1] / 2, psf_model.shape[0] / 2]
        ax_right.add_patch(
            Circle(
                ctr,
                aperture_radius,
                color="#FF0000",
                ls="-",
                fill=False,
                lw=0.5,
            )
        )

        cbar = fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((-1, 1))
        cbar.formatter.set_useMathText(True)
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.yaxis.get_offset_text().set_fontsize(6)

        # 3D histogram of the ePSF (bottom of right subgrid)
        ax_right_3d = fig.add_subplot(right_sub[1, 0], projection="3d")
        # Bar positions: (x, y) = pixel indices, z = 0 at base, height = PSF value
        z_flat = np.maximum(psf_model.ravel(), 0.0)
        x_bar = yy.ravel()
        y_bar = xx.ravel()
        z_base = np.zeros_like(z_flat)
        dx = dy = np.ones_like(z_flat)
        norm_3d = plt.Normalize(vmin=vmin_p, vmax=vmax_p)
        colors = plt.cm.viridis(norm_3d(z_flat))
        ax_right_3d.bar3d(
            x_bar,
            y_bar,
            z_base,
            dx,
            dy,
            z_flat,
            color=colors,
            shade=True,
        )
        ax_right_3d.set_title("3D ePSF", fontsize=8, pad=2)
        ax_right_3d.set_xlabel("")
        ax_right_3d.set_ylabel("")
        ax_right_3d.set_zlabel("")
        ax_right_3d.set_xticks([])
        ax_right_3d.set_yticks([])
        ax_right_3d.set_zticks([])
        ax_right_3d.view_init(elev=35, azim=135)

        # Standardized PNG output.
        psf_sources_png = os.path.join(write_dir, f"PSF_Sources_{base}.png")
        fig.savefig(
            psf_sources_png,
            bbox_inches="tight",
            dpi=150,
            facecolor="white",
        )
        plt.close()
        return 1

    # -----------------------------------------------------------------------
    # Column name helper
    # -----------------------------------------------------------------------

    def _first_present(self, tbl, names, unit=None, default=np.nan):
        """
        Return the first available column from *tbl*, converted to a float array.

        Works with both astropy Tables (.colnames) and pandas DataFrames (.columns).
        """
        # Unified column-name lookup for both Table and DataFrame.
        col_names = tbl.colnames if hasattr(tbl, "colnames") else list(tbl.columns)
        for nm in names:
            if nm in col_names:
                col = tbl[nm]
                try:
                    return np.asarray(
                        col.to_value(unit) if unit is not None else col, float
                    )
                except Exception:
                    return np.asarray(col, float)
        return np.full(len(tbl), default, float)

    # -----------------------------------------------------------------------
    # PSF fitting
    # -----------------------------------------------------------------------

    def fit(
        self,
        epsf_model,
        sources: pd.DataFrame,
        plot: bool = False,
        plotTarget: bool = False,
        forcePhotometry: bool = False,
        ignore_sources=None,
        is_target_fit: bool = False,
        background_rms=None,
        xy_bounds=None,
        iterative: bool = False,
    ) -> pd.DataFrame:
        """
        Tiered-SNR PSF photometry with optional MCMC error propagation.

        Returns
        -------
        updated : DataFrame  (copy of sources with fit columns added)
        """
        log = logging.getLogger(__name__)
        t0 = time.perf_counter()
        log.info(border_msg(f"PSF photometry on {len(sources)} sources"))

        if epsf_model is None:
            log.info("No ePSF model; returning sources unchanged.")
            return sources

        fwhm = float(self.input_yaml.get("fwhm", 3.0))
        exposure_time = float(self.input_yaml.get("exposure_time", 30.0))
        image_filter = self.input_yaml.get("imageFilter", "unk")
        scale = self.input_yaml.get("scale", 11.5)
        aperture_radius = float(
            self.input_yaml.get("photometry", {}).get(
                "aperture_radius", max(1.5 * fwhm, 2.0)
            )
        )
        phot_cfg = self.input_yaml.get("photometry", {}) or {}

        undersampled_fwhm_threshold = float(
            phot_cfg.get("undersampled_fwhm_threshold", 2.5)
        )
        undersampled = fwhm <= undersampled_fwhm_threshold

        pixel_scale = self.input_yaml.get("pixel_scale", np.nan)  # arcsec / px
        try:
            pixel_scale = float(pixel_scale)
        except Exception:
            pixel_scale = np.nan
        has_pixel_scale = np.isfinite(pixel_scale) and pixel_scale > 0

        # Fit box sizes chosen to enclose several FWHM in each regime.
        # We also apply an adaptive boost for undersampled/coarse-scale data.
        fs_bright_scale = float(phot_cfg.get("psf_fit_shape_bright_scale_fwhm", 3.0))
        fs_faint_scale = float(phot_cfg.get("psf_fit_shape_faint_scale_fwhm", 2.5))
        fs_vfaint_scale = float(phot_cfg.get("psf_fit_shape_vfaint_scale_fwhm", 2.0))
        fit_sampling_boost = 1.0
        if fwhm <= 2.5:
            fit_sampling_boost += 0.20
        if has_pixel_scale and pixel_scale >= 0.8:
            fit_sampling_boost += 0.10
        if has_pixel_scale and pixel_scale >= 1.2:
            fit_sampling_boost += 0.10
        fit_sampling_boost_max = float(
            phot_cfg.get("psf_fit_sampling_boost_max", 1.6)
        )
        fit_sampling_boost = float(
            min(max(1.0, fit_sampling_boost), max(1.0, fit_sampling_boost_max))
        )
        fs_bright_scale *= fit_sampling_boost
        fs_faint_scale *= fit_sampling_boost
        fs_vfaint_scale *= fit_sampling_boost
        target_shape_boost = float(phot_cfg.get("psf_target_fit_shape_boost", 1.5))
        if is_target_fit:
            target_shape_boost = max(1.0, target_shape_boost)
            fs_bright_scale *= target_shape_boost
            fs_faint_scale *= target_shape_boost
            fs_vfaint_scale *= target_shape_boost

        bright_min_arcsec = float(
            phot_cfg.get("psf_fit_shape_bright_min_arcsec", 3.0)
        )
        faint_min_arcsec = float(phot_cfg.get("psf_fit_shape_faint_min_arcsec", 2.5))
        vfaint_min_arcsec = float(
            phot_cfg.get("psf_fit_shape_vfaint_min_arcsec", 2.0)
        )
        bright_min_px = bright_min_arcsec / pixel_scale if has_pixel_scale else np.nan
        faint_min_px = faint_min_arcsec / pixel_scale if has_pixel_scale else np.nan
        vfaint_min_px = vfaint_min_arcsec / pixel_scale if has_pixel_scale else np.nan

        fit_shape_bright = (
            _odd(
                max(
                    7,
                    int(fs_bright_scale * fwhm),
                    int(np.ceil(bright_min_px)) if np.isfinite(bright_min_px) else 0,
                )
            ),
        ) * 2
        fit_shape_faint = (
            _odd(
                max(
                    7,
                    int(fs_faint_scale * fwhm),
                    int(np.ceil(faint_min_px)) if np.isfinite(faint_min_px) else 0,
                )
            ),
        ) * 2
        fit_shape_vfaint = (
            _odd(
                max(
                    7,
                    int(fs_vfaint_scale * fwhm),
                    int(np.ceil(vfaint_min_px)) if np.isfinite(vfaint_min_px) else 0,
                )
            ),
        ) * 2

        # For undersampled data, enforce a minimum box that still captures
        # the first Airy ring / broader wings, helping the PSF distinguish
        # true point sources from neighbouring structure.
        if undersampled:
            fit_shape_bright = (_odd(max(fit_shape_bright[0], 9)),) * 2
            fit_shape_faint = (_odd(max(fit_shape_faint[0], 9)),) * 2
            fit_shape_vfaint = (_odd(max(fit_shape_vfaint[0], 7)),) * 2

        if xy_bounds is None:
            # Default xy-bounds are configured in arcseconds and converted to pixels.
            # Fallback to legacy FWHM-based bounds when pixel_scale is unavailable.
            cfg_xy_bounds_arcsec = phot_cfg.get("fitting_xy_bounds", None)
            pixel_scale = self.input_yaml.get("pixel_scale", None)  # arcsec / pixel
            if cfg_xy_bounds_arcsec is not None:
                try:
                    cfg_xy_bounds_arcsec = float(cfg_xy_bounds_arcsec)
                    pixel_scale = (
                        float(pixel_scale)
                        if pixel_scale is not None and np.isfinite(pixel_scale)
                        else np.nan
                    )
                    if np.isfinite(pixel_scale) and pixel_scale > 0:
                        xy_bounds = cfg_xy_bounds_arcsec / pixel_scale
                    else:
                        # Allow slightly larger centroid excursions for undersampled data,
                        # where sub-pixel shifts can be substantial relative to the FWHM.
                        xy_bounds = 3.0 * fwhm if undersampled else 2.0 * fwhm
                        log.info(
                            "fitting_xy_bounds is set (%.3f arcsec) but pixel_scale is unavailable; "
                            "using legacy xy_bounds=%.2f px fallback.",
                            cfg_xy_bounds_arcsec,
                            xy_bounds,
                        )
                except Exception:
                    xy_bounds = 3.0 * fwhm if undersampled else 2.0 * fwhm
            else:
                # Legacy fallback when no explicit config is provided.
                xy_bounds = 3.0 * fwhm if undersampled else 2.0 * fwhm

        def _effective_xy_bounds_for_shape(fit_shape):
            """
            Constrain search bounds to be consistent with the PSF fit cutout.
            A source cannot be robustly re-centered beyond roughly half the
            fit box width; keep margin to avoid edge-dominated solutions.
            """
            half_box = 0.5 * (min(fit_shape) - 1)
            frac_limit = 0.98 if undersampled else 0.90
            max_from_shape = max(0.5, frac_limit * half_box)
            return float(max(0.5, min(float(xy_bounds), max_from_shape)))

        xb_bright = _effective_xy_bounds_for_shape(fit_shape_bright)
        xb_faint = _effective_xy_bounds_for_shape(fit_shape_faint)
        xb_vfaint = _effective_xy_bounds_for_shape(fit_shape_vfaint)
        cfg_xy_arcsec_log = phot_cfg.get("fitting_xy_bounds", None)
        pix_scale_log = self.input_yaml.get("pixel_scale", None)
        if (
            cfg_xy_arcsec_log is not None
            and pix_scale_log is not None
            and np.isfinite(float(pix_scale_log))
            and float(pix_scale_log) > 0
        ):
            log.info(
                "PSF fit bounds config: fitting_xy_bounds=%.3f arcsec, pixel_scale=%.3f arcsec/px, "
                "base_xy_bounds=%.2f px, effective per tier (bright/faint/very-faint)=%.2f/%.2f/%.2f px",
                float(cfg_xy_arcsec_log),
                float(pix_scale_log),
                float(xy_bounds),
                xb_bright,
                xb_faint,
                xb_vfaint,
            )
        else:
            log.info(
                "PSF fit bounds config: fitting_xy_bounds=%s arcsec, pixel_scale=%s arcsec/px, "
                "base_xy_bounds=%.2f px, effective per tier (bright/faint/very-faint)=%.2f/%.2f/%.2f px",
                str(cfg_xy_arcsec_log),
                str(pix_scale_log),
                float(xy_bounds),
                xb_bright,
                xb_faint,
                xb_vfaint,
            )

        log.info(
            f"Fit shapes: bright={fit_shape_bright}, faint={fit_shape_faint}, "
            f"very-faint={fit_shape_vfaint}  xy_bounds={xy_bounds:.1f} "
            f"(undersampled={undersampled})"
        )
        if is_target_fit:
            log.info(
                "Target-fit shape scaling: bright/faint/very-faint=%.2f/%.2f/%.2f * FWHM "
                "(fit_sampling_boost=%.2f, target_boost=%.2f, pixel_scale=%s arcsec/px).",
                fs_bright_scale,
                fs_faint_scale,
                fs_vfaint_scale,
                fit_sampling_boost,
                target_shape_boost,
                f"{pixel_scale:.3f}" if has_pixel_scale else "unknown",
            )

        epsf_model_copy = epsf_model.copy()

        ndimage = self.create_nddata_with_fitting_weights(
            self.image.copy(),
            gain=float(self.input_yaml.get("gain", 1.0)),
            read_noise=float(self.input_yaml.get("read_noise", 0.0)),
            background_rms=background_rms,
        )

        if sources is None or len(sources) == 0:
            log.info("No sources supplied to PSF fit; returning unchanged.")
            return sources
        if ("x_pix" not in sources.columns) or ("y_pix" not in sources.columns):
            log.warning(
                "PSF fit requires x_pix/y_pix columns; skipping PSF fit for this source table."
            )
            return sources

        flux_col = None
        for c in ("counts_AP", "flux_AP", "counts", "flux"):
            if c in sources.columns:
                flux_col = c
                break
        if flux_col is None:
            log.warning(
                "PSF fit requires an aperture-flux column (counts_AP/flux_AP); skipping PSF fit."
            )
            return sources

        x_all = np.asarray(sources["x_pix"], float)
        y_all = np.asarray(sources["y_pix"], float)
        flux_all = np.asarray(sources[flux_col], float)
        orig_idx = np.arange(len(sources)).astype(int)
        valid = np.isfinite(x_all) & np.isfinite(y_all)

        x, y, flux_ap = x_all[valid], y_all[valid], flux_all[valid]
        idx_keep = orig_idx[valid]

        init_params = QTable(
            {
                "x": x,
                "y": y,
                "flux": u.Quantity(np.clip(flux_ap, 1e-6, 1e12), u.electron),
                "__row": idx_keep,
            }
        )

        if len(init_params) == 0:
            log.info("No valid sources; returning unchanged.")
            return sources

        bkgrmsval = float(
            np.nanmedian(background_rms)
            if background_rms is not None
            else np.nanmedian(ndimage.uncertainty.array)
        )
        area = np.pi * max(2.0, aperture_radius) ** 2
        snr = (
            np.asarray(sources["SNR"], float)[valid]
            if "SNR" in sources.columns
            else flux_ap / np.sqrt(np.maximum(flux_ap, 0.0) + bkgrmsval**2 * area)
        )

        bright_mask = snr >= 8.0
        faint_mask = (snr >= 4.0) & (snr < 8.0)
        vfaint_mask = snr < 4.0

        # ---- Fitter --------------------------------------------------------
        # Emcee is driven only by perform_emcee_fitting_s2n. Used only when is_target_fit is True.
        # perform_emcee_fitting_s2n 0 or None: never use emcee (LSQ only).
        # perform_emcee_fitting_s2n > 0: for target fit, use emcee only if target S/N is below threshold.
        # Planar/surface background fitting was removed. PSF fitting always uses
        # a constant local background estimate.
        emcee_s2n = phot_cfg.get("perform_emcee_fitting_s2n", 0)
        if emcee_s2n is None:
            emcee_s2n = 0
        else:
            emcee_s2n = float(emcee_s2n)
        use_emcee_for_all = False
        use_emcee_tiered = False
        if is_target_fit and emcee_s2n > 0 and len(init_params) == 1:
            target_snr = float(snr[0])
            use_emcee_for_all = target_snr < emcee_s2n
            log.info(
                "Target S/N=%.2f; %s (low-S/N threshold=%.1f).",
                target_snr,
                "MCMC" if use_emcee_for_all else "LSQ",
                emcee_s2n,
            )
        elif is_target_fit and emcee_s2n > 0 and len(init_params) > 1:
            use_emcee_tiered = True

        lsq_fitter = TRFLSQFitter()
        emcee_fitter = None
        if use_emcee_for_all or use_emcee_tiered:
            try:
                emcee_delta = phot_cfg.get("emcee_delta")
                delta = float(emcee_delta) if emcee_delta is not None else xy_bounds
                emcee_nsteps = phot_cfg.get("emcee_nsteps", 5000)
                emcee_fitter = MCMCFitter(
                    nwalkers=int(phot_cfg.get("emcee_nwalkers", 20)),
                    nsteps=int(emcee_nsteps) if emcee_nsteps is not None else None,
                    delta=delta,
                    burnin_frac=float(phot_cfg.get("emcee_burnin_frac", 0.3)),
                    thin=int(phot_cfg.get("emcee_thin", 10)),
                    adaptive_tau_target=int(
                        phot_cfg.get("emcee_adaptive_tau_target", 50)
                    ),
                    min_autocorr_N=int(phot_cfg.get("emcee_min_autocorr_N", 100)),
                    batch_steps=int(phot_cfg.get("emcee_batch_steps", 100)),
                    jitter_scale=float(phot_cfg.get("emcee_jitter_scale", 0.01)),
                    use_nddata_uncertainty=True,
                    gain=float(self.input_yaml.get("gain", 1.0)),
                    readnoise=float(self.input_yaml.get("read_noise", 0.0)),
                    background_rms=bkgrmsval,
                )
            except Exception as exc:
                log_warning_from_exception(
                    log, "MCMC unavailable; using LSQ for all sources", exc
                )
                emcee_fitter = None
                use_emcee_for_all = False
                use_emcee_tiered = False

        # Per-tier: use emcee if (emcee for all) or (tier has sources with S/N < emcee_s2n).
        # Tier SNR bounds: bright >= 8, faint [4, 8), very faint < 4.
        def use_emcee_for_tier(label):
            if use_emcee_for_all:
                return True
            if not use_emcee_tiered or emcee_fitter is None:
                return False
            tier_min_snr = {"bright": 8.0, "faint": 4.0, "very faint": 0.0}.get(
                label, 0.0
            )
            return tier_min_snr < emcee_s2n

        any_emcee_used = use_emcee_for_all or (
            use_emcee_tiered and emcee_fitter is not None
        )

        # In undersampled data, the effective PSF footprint in pixels is
        # small; use a more conservative grouping distance to avoid
        # over-merging genuinely distinct neighbours.
        group_sep = 3.0 * fwhm if undersampled else 2.0 * fwhm
        group_maker = SourceGrouper(min_separation=max(group_sep, 1.0))

        def _fit_plane_from_annulus(
            data2d: np.ndarray,
            x0: float,
            y0: float,
            inner_r: float,
            outer_r: float,
        ) -> Optional[tuple[float, float, float]]:
            """
            Disabled: surface plane fitting (planar background via annulus-fit
            and subtraction) has been removed; this helper always returns None.
            """
            # Surface plane fitting removed (cleanup): this helper is intentionally
            # disabled so no planar surface subtraction is applied.
            return None

        def _subtract_plane_in_fit_box(
            data2d: np.ndarray,
            coef: tuple[float, float, float],
            x0: float,
            y0: float,
            fit_shape: tuple[int, int],
        ) -> bool:
            """Disabled: planar surface subtraction removed."""
            # Surface plane fitting removed (cleanup): this helper is intentionally
            # disabled so no planar surface subtraction is applied.
            return False

        # ---- Fit helper ----------------------------------------------------
        # Background: photutils subtracts per-source local background (annulus inner_r..outer_r)
        # from each fit-shape cutout before fitting the PSF, so the model is fit to
        # (data - local_bkg). NDData uncertainty includes background_rms in the noise model.
        def _psf_fit(mask, inner_r, outer_r, fit_shape, fitter, use_emcee_this_tier):
            if not np.any(mask):
                return None, None

            localbkg = LocalBackground(inner_r, outer_r, bkg_estimator=MMMBackground())
            xy_bounds_this = _effective_xy_bounds_for_shape(fit_shape)

            if not iterative:
                nd_for_fit = ndimage
                if False:
                    # Work on a copy so other tiers/fits see the original image.
                    work = np.array(ndimage.data, dtype=float, copy=True)
                    sub_init_tmp = init_params[mask].to_pandas()
                    n_ok = 0
                    slopes = []
                    # For the target fit (single source), subtract the fitted plane
                    # over a larger local stamp around the target (comparable to the
                    # local background region), instead of just the tiny PSF fit box.
                    if is_target_fit and int(np.count_nonzero(mask)) == 1:
                        _x = float(sub_init_tmp["x"].to_numpy()[0])
                        _y = float(sub_init_tmp["y"].to_numpy()[0])
                        coef = _fit_plane_from_annulus(work, _x, _y, inner_r, outer_r)
                        if coef is not None:
                            # Use a box ~2*outer_r on a side around the target to
                            # capture the same local environment as the annulus-based
                            # background estimate, but without touching the full frame.
                            side = max(3, int(2.0 * float(outer_r)))
                            side = _odd(side)
                            if _subtract_plane_in_fit_box(
                                work, coef, _x, _y, (side, side)
                            ):
                                n_ok = 1
                                slopes.append((coef[1], coef[2]))
                                log.info(
                                    "Planar background subtraction applied in a local stamp for target fit "
                                    "(inner=%.1f px, outer=%.1f px, box=%dx%d). "
                                    "Slopes: bx=%.3g, by=%.3g ADU/pix.",
                                    float(inner_r),
                                    float(outer_r),
                                    side,
                                    side,
                                    float(coef[1]),
                                    float(coef[2]),
                                )
                    else:
                        for _x, _y in zip(
                            sub_init_tmp["x"].to_numpy(), sub_init_tmp["y"].to_numpy()
                        ):
                            coef = _fit_plane_from_annulus(
                                work, float(_x), float(_y), inner_r, outer_r
                            )
                            if coef is None:
                                continue
                            if _subtract_plane_in_fit_box(
                                work, coef, float(_x), float(_y), fit_shape
                            ):
                                n_ok += 1
                                slopes.append((coef[1], coef[2]))
                    nd_for_fit = _nddata_clone(ndimage, data=work)
                    if n_ok > 0:
                        bx_med = (
                            float(np.nanmedian([s[0] for s in slopes]))
                            if slopes
                            else 0.0
                        )
                        by_med = (
                            float(np.nanmedian([s[1] for s in slopes]))
                            if slopes
                            else 0.0
                        )
                        log.info(
                            "Planar background subtraction applied to %d/%d sources (inner=%.1f px, outer=%.1f px). "
                            "Median slopes: bx=%.3g, by=%.3g ADU/pix.",
                            n_ok,
                            int(np.count_nonzero(mask)),
                            float(inner_r),
                            float(outer_r),
                            bx_med,
                            by_med,
                        )
                psfphot = PSFPhotometry(
                    psf_model=epsf_model_copy,
                    fitter=fitter,
                    fitter_maxiters=1000,
                    fit_shape=fit_shape,
                    aperture_radius=aperture_radius,
                    localbkg_estimator=localbkg,
                    xy_bounds=xy_bounds_this,
                    progress_bar=False,
                )
                sub_init = init_params[mask]
                res = psfphot(nd_for_fit, init_params=sub_init)
            else:
                finder = DAOStarFinder(
                    np.nanmedian(detect_threshold(ndimage.data, nsigma=3.0))
                    * u.electron,
                    fwhm,
                )
                psfphot = IterativePSFPhotometry(
                    psf_model=epsf_model_copy,
                    fitter=fitter,
                    fitter_maxiters=100,
                    fit_shape=fit_shape,
                    aperture_radius=aperture_radius,
                    localbkg_estimator=localbkg,
                    grouper=group_maker,
                    xy_bounds=xy_bounds_this,
                    progress_bar=False,
                    finder=finder,
                    maxiters=3,
                )
                sub_init = init_params[mask]
                res = psfphot(ndimage, init_params=sub_init)

                # Match back to input positions via KD-tree.
                sub_init = sub_init.to_pandas()
                res_df = res.to_pandas()
                if res_df.empty:
                    return None, psfphot
                tree = cKDTree(np.vstack((res_df["x_fit"], res_df["y_fit"])).T)
                dists, idxs = tree.query(np.vstack((sub_init["x"], sub_init["y"])).T)
                match = dists <= xy_bounds_this
                res = res_df.iloc[idxs[match]].reset_index(drop=True)
                sub_init = sub_init.iloc[match].reset_index(drop=True)

            sub_init = (
                sub_init.to_pandas() if hasattr(sub_init, "to_pandas") else sub_init
            )
            res = res.to_pandas() if hasattr(res, "to_pandas") else res

            res["idx"] = (
                sub_init["__row"].to_numpy()
                if "__row" in sub_init.columns
                else np.flatnonzero(mask)
            )
            if "flags" not in res.columns:
                res["flags"] = 0

            # LSQ robustness retry:
            # Re-fit only pathological rows with a slightly larger fit box but
            # tighter centroid bounds to reduce divergence/outlier solutions.
            if (not use_emcee_this_tier) and (not res.empty):
                bad = (
                    ~np.isfinite(self._first_present(res, ["x_fit", "xcenter_fit", "x_0_fit"]))
                    | ~np.isfinite(self._first_present(res, ["y_fit", "ycenter_fit", "y_0_fit"]))
                    | ~np.isfinite(self._first_present(res, ["flux_fit", "flux"]))
                    | (self._first_present(res, ["flux_fit", "flux"]) <= 0)
                    | (np.asarray(res.get("flags", 0), int) != 0)
                )
                if np.any(bad):
                    bad_ids = np.asarray(res.loc[bad, "idx"], int)
                    if "__row" in sub_init.columns:
                        retry_sel = np.isin(
                            np.asarray(sub_init["__row"], int), bad_ids
                        )
                    else:
                        retry_sel = np.zeros(len(sub_init), dtype=bool)
                    if np.any(retry_sel):
                        retry_init = init_params[mask][retry_sel]
                        fit_shape_retry = tuple(_odd(int(s) + 2) for s in fit_shape)
                        xy_bounds_retry = max(0.5, 0.7 * float(xy_bounds_this))
                        log.info(
                            "LSQ retry for %d/%d problematic fits (fit_shape %s -> %s, xy_bounds %.2f -> %.2f).",
                            int(np.sum(retry_sel)),
                            len(sub_init),
                            str(fit_shape),
                            str(fit_shape_retry),
                            float(xy_bounds_this),
                            float(xy_bounds_retry),
                        )
                        try:
                            psfphot_retry = PSFPhotometry(
                                psf_model=epsf_model_copy,
                                fitter=fitter,
                                fitter_maxiters=1500,
                                fit_shape=fit_shape_retry,
                                aperture_radius=aperture_radius,
                                localbkg_estimator=localbkg,
                                xy_bounds=xy_bounds_retry,
                                progress_bar=False,
                            )
                            res_retry = psfphot_retry(nd_for_fit, init_params=retry_init)
                            res_retry = (
                                res_retry.to_pandas()
                                if hasattr(res_retry, "to_pandas")
                                else res_retry
                            )
                            retry_init_df = (
                                retry_init.to_pandas()
                                if hasattr(retry_init, "to_pandas")
                                else retry_init
                            )
                            if not res_retry.empty:
                                res_retry["idx"] = (
                                    retry_init_df["__row"].to_numpy()
                                    if "__row" in retry_init_df.columns
                                    else np.arange(len(res_retry), dtype=int)
                                )
                                if "flags" not in res_retry.columns:
                                    res_retry["flags"] = 0
                                # Prefer retry solutions for the retried ids.
                                keep = ~np.isin(
                                    np.asarray(res["idx"], int),
                                    np.asarray(res_retry["idx"], int),
                                )
                                res = pd.concat(
                                    [res.loc[keep], res_retry], ignore_index=True
                                )
                        except Exception as exc:
                            log.info("LSQ retry skipped (non-fatal): %s", exc)

            # Propagate MCMC per-source errors.
            if use_emcee_this_tier and isinstance(
                getattr(fitter, "fit_info", None), dict
            ):
                per_src = fitter.fit_info.get("per_source")
                for nm in ("x_fit_err", "y_fit_err", "flux_fit_err"):
                    if nm not in res.columns:
                        res[nm] = np.nan

                # MCMCFitter.fit_info is cumulative across fitter calls (e.g.
                # different SNR tiers). The current `res` only corresponds to the
                # most recent fitted batch, so take the last `len(res)` records.
                if isinstance(per_src, (list, tuple)) and len(per_src) >= len(res):
                    per_src_batch = per_src[-len(res) :]
                    if len(per_src_batch) == len(res):
                        for i, rec in enumerate(per_src_batch):
                            names = list(rec.get("param_names", []))
                            p16 = np.asarray(rec.get("p16", []), float)
                            p50 = np.asarray(rec.get("p50", []), float)
                            p84 = np.asarray(rec.get("p84", []), float)
                            if p16.size == 0:
                                continue
                            sigma = 0.5 * ((p84 - p50) + (p50 - p16))

                            def _get_sigma(keys):
                                for k in keys:
                                    if k in names:
                                        return float(max(sigma[names.index(k)], 0.0))
                                return np.nan

                            res.at[i, "x_fit_err"] = _get_sigma(
                                ("x", "x_0", "x_mean", "xcenter")
                            )
                            res.at[i, "y_fit_err"] = _get_sigma(
                                ("y", "y_0", "y_mean", "ycenter")
                            )
                            res.at[i, "flux_fit_err"] = _get_sigma(
                                ("flux", "amplitude", "amp")
                            )

            return res, psfphot

        # ---- Dispatch per SNR tier -----------------------------------------
        results = []
        psfphot_last = None
        # In crowded fields, push the background annulus further out from the core
        # and narrow its width so it samples background rather than neighbour PSF wings.
        phot_cfg = self.input_yaml.get("photometry", {}) or {}
        crowded_field = bool(phot_cfg.get("crowded_field", False))

        if crowded_field:
            bright_inner = aperture_radius + 4.0 * fwhm
            bright_outer = bright_inner + 2.0 * fwhm
            faint_inner = aperture_radius + 3.0 * fwhm
            faint_outer = faint_inner + 2.0 * fwhm
            vf_inner = aperture_radius + 3.0 * fwhm
            vf_outer = vf_inner + 2.0 * fwhm
        else:
            bright_inner = aperture_radius + 3.0 * fwhm
            bright_outer = aperture_radius + 6.0 * fwhm
            faint_inner = aperture_radius + 2.0 * fwhm
            faint_outer = aperture_radius + 5.0 * fwhm
            vf_inner = aperture_radius + 2.0 * fwhm
            vf_outer = aperture_radius + 5.0 * fwhm

        for mask, inner_r, outer_r, fshape, label in [
            (bright_mask, bright_inner, bright_outer, fit_shape_bright, "bright"),
            (faint_mask, faint_inner, faint_outer, fit_shape_faint, "faint"),
            (vfaint_mask, vf_inner, vf_outer, fit_shape_vfaint, "very faint"),
        ]:
            if np.any(mask):
                use_emcee_this = use_emcee_for_tier(label)
                tier_fitter = (
                    emcee_fitter
                    if use_emcee_this and emcee_fitter is not None
                    else lsq_fitter
                )
                log.info(
                    "Fitting %d %s sources (%s)",
                    int(mask.sum()),
                    label,
                    "MCMC" if use_emcee_this else "LSQ",
                )
                res, psfphot_last = _psf_fit(
                    mask, inner_r, outer_r, fshape, tier_fitter, use_emcee_this
                )
                if res is not None:
                    results.append(res)

        if not results:
            return sources

        combined = pd.concat(results)
        idx_out = np.asarray(combined["idx"], int)

        x_fit = self._first_present(combined, ["x_fit", "xcenter_fit", "x_0_fit"])
        y_fit = self._first_present(combined, ["y_fit", "ycenter_fit", "y_0_fit"])
        x_err = self._first_present(combined, ["x_fit_err", "x_err", "xcenter_fit_err"])
        y_err = self._first_present(combined, ["y_fit_err", "y_err", "ycenter_fit_err"])
        flux_fit = self._first_present(combined, ["flux_fit", "flux"], unit=u.electron)
        flux_err = self._first_present(
            combined, ["flux_fit_err", "flux_err", "flux_uncertainty"], unit=u.electron
        )

        # Enforce positive fitted fluxes: negative solutions are unphysical for
        # a positive PSF and indicate overfitting or local background issues.
        # Clip at a tiny floor and propagate to errors.
        with np.errstate(invalid="ignore"):
            flux_fit = np.clip(flux_fit, 1e-6, np.inf)
            flux_err = np.where(np.isfinite(flux_err), flux_err, np.nan)
        cfit_out = self._first_present(combined, ["cfit"])
        qfit_out = self._first_present(combined, ["qfit"])
        chi2_out = self._first_present(combined, ["reduced_chi2", "chi2_red"])
        flags_out = np.asarray(combined["flags"], int)

        df_out = pd.DataFrame(
            {
                "x_fit": x_fit,
                "y_fit": y_fit,
                "x_fit_err": x_err,
                "y_fit_err": y_err,
                "flux_fit_e": flux_fit,
                "flux_err_e": flux_err,
                "flags": flags_out,
                "cfit": cfit_out,
                "qfit": qfit_out,
                "reduced_chi2": chi2_out,
            },
            index=idx_out,
        ).sort_index()

        updated = sources.copy()
        for col in (
            "x_fit",
            "y_fit",
            "x_fit_err",
            "y_fit_err",
            "flux_PSF",
            "flux_PSF_err",
            "flags",
            "cfit",
            "qfit",
            "reduced_chi2",
            "fwhm_psf",
        ):
            if col not in updated.columns:
                updated[col] = np.nan
        updated["fwhm_psf"] = fwhm

        row_pos = df_out.index.to_numpy()
        updated.iloc[
            row_pos,
            updated.columns.get_indexer(["x_fit", "y_fit", "x_fit_err", "y_fit_err"]),
        ] = df_out[["x_fit", "y_fit", "x_fit_err", "y_fit_err"]].to_numpy()
        # Flux in e/s (same convention as aperture) so AP and PSF mags share the same zeropoint.
        # flux_fit_e is total flux in electrons from the PSF fit; divide by exposure_time for e/s.
        updated.iloc[row_pos, updated.columns.get_indexer(["flux_PSF"])] = (
            df_out["flux_fit_e"].to_numpy() / exposure_time
        )
        updated.iloc[row_pos, updated.columns.get_indexer(["flux_PSF_err"])] = (
            df_out["flux_err_e"].to_numpy() / exposure_time
        )

        for col in ("flags", "cfit", "qfit", "reduced_chi2"):
            updated.iloc[row_pos, updated.columns.get_indexer([col])] = df_out[
                col
            ].to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            inst_col = f"inst_{image_filter}_PSF"
            inst_err_col = f"inst_{image_filter}_PSF_err"
            updated[inst_col] = -2.5 * np.log10(updated["flux_PSF"])
            valid_flux = (updated["flux_PSF"] > 0) & np.isfinite(updated["flux_PSF"])
            mag_err = np.full(len(updated), np.nan)
            mag_err[valid_flux] = (2.5 / np.log(10.0)) * (
                updated.loc[valid_flux, "flux_PSF_err"]
                / updated.loc[valid_flux, "flux_PSF"]
            )
            updated[inst_err_col] = mag_err

        log.info(f"Fitted {len(updated)} sources in {time.perf_counter() - t0:.2f}s")

        if plot or plotTarget:
            try:
                self.plot(
                    updated,
                    ndimage,
                    psfphot_last,
                    plotTarget=plotTarget,
                    scale=scale,
                    aperture_radius=aperture_radius,
                )
            except Exception as exc:
                log.info(f"Plotting failed: {exc}")

        if plotTarget and any_emcee_used:
            try:
                self._plot_mcmc_corner(
                    emcee_fitter,
                    0,
                    writedir=self.input_yaml.get("write_dir"),
                    truths=[
                        sources.get("x_pix", pd.Series([np.nan])).iloc[0],
                        sources.get("y_pix", pd.Series([np.nan])).iloc[0],
                        sources.get("counts_AP", pd.Series([np.nan])).iloc[0],
                    ],
                )
            except Exception as exc:
                log_warning_from_exception(log, "Corner plot failed", exc)

        return updated

    # -----------------------------------------------------------------------
    # Diagnostic plot
    # -----------------------------------------------------------------------

    def plot(
        self,
        sources: pd.DataFrame,
        ndimage: NDData,
        psfphot=None,
        epsf=None,
        plotTarget: bool = False,
        scale: float = 1.0,
        aperture_radius: float = 7.0,
    ) -> None:
        """
        Multi-panel diagnostic: science image, PSF residual, ePSF model.
        """
        fpath = self.input_yaml["fpath"]
        base = os.path.splitext(os.path.basename(fpath))[0]
        write_dir = os.path.dirname(fpath)
        log = logging.getLogger(__name__)

        try:
            nd_for_plot = ndimage

            first_image = np.asarray(nd_for_plot.data)
            uncertainty = (
                np.asarray(nd_for_plot.uncertainty.array)
                if nd_for_plot.uncertainty is not None
                else np.zeros_like(first_image)
            )

            if psfphot is not None:
                subtracted = psfphot.make_residual_image(
                    nd_for_plot.data * nd_for_plot.unit,
                    psf_shape=first_image.shape,
                    include_localbkg=True,
                )
                second_image = np.asarray(subtracted.data)
                fitted_model = first_image - second_image
                second_label = "Residual Flux"
            else:
                second_image = fitted_model = None
                second_label = None

            # Summary of fitted local background for annotation (PSF model = PSF + local bkg).
            bkg_summary = None
            for col in ("local_background", "local_bkg", "background", "bkg"):
                if col in sources.columns:
                    vals = np.asarray(sources[col], float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        bkg_summary = (
                            float(np.median(vals)),
                            float(np.min(vals)),
                            float(np.max(vals)),
                        )
                        break

            ncols = (
                1 + (1 if psfphot is not None else 0) + (1 if epsf is not None else 0)
            )
            _style = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
            )
            if os.path.exists(_style):
                plt.style.use(_style)
            golden_ratio = (5**0.5 + 1) / 2
            width_in = 5.5 * ncols
            aspect = 5.0 * golden_ratio / width_in
            width_pt = width_in * 72.27
            fig = plt.figure(figsize=set_size(width_pt, aspect=aspect))
            gs = GridSpec(1, ncols, width_ratios=[1] * ncols, wspace=0.45)

            ax_list, cax_list = [], []
            for k in range(ncols):
                ax = fig.add_subplot(gs[0, k])
                divider = make_axes_locatable(ax)
                ax_R = divider.append_axes("right", size="20%", pad=0.15, sharey=ax)
                ax_B = divider.append_axes("bottom", size="20%", pad=0.15, sharex=ax)
                cax = divider.append_axes("top", size="5%", pad=0.05)
                ax.tick_params(axis="x", labelbottom=False)
                ax_R.tick_params(axis="y", labelleft=False)
                if k > 0:
                    ax.tick_params(axis="y", labelleft=False)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                for _a in (ax_R, ax_B):
                    _a.tick_params(axis="both", labelsize=8)
                # Rotate tick labels on projection axes to avoid overlap
                ax_B.tick_params(axis="x", labelrotation=30)
                ax_R.tick_params(axis="x", labelrotation=30)
                ax_list.append((ax, ax_R, ax_B))
                cax_list.append(cax)

            x_center = (
                np.nanmean(sources["x_pix"])
                if "x_pix" in sources
                else ndimage.data.shape[1] / 2
            )
            y_center = (
                np.nanmean(sources["y_pix"])
                if "y_pix" in sources
                else ndimage.data.shape[0] / 2
            )

            x0 = max(int(np.floor(x_center - scale)), 0)
            x1 = min(int(np.ceil(x_center + scale)), ndimage.data.shape[1])
            y0 = max(int(np.floor(y_center - scale)), 0)
            y1 = min(int(np.ceil(y_center + scale)), ndimage.data.shape[0])

            # ---- Panel 1: science image ------------------------------------
            ax1, ax1_R, ax1_B = ax_list[0]
            cutout1 = first_image[y0:y1, x0:x1]
            unc_cut = uncertainty[y0:y1, x0:x1]

            if plotTarget:
                # Scale to +/- 3 sigma (robust) so faint PSF is visible despite bright contaminants
                _, med1, std1 = sigma_clipped_stats(cutout1, sigma=3, maxiters=5)
                std1 = max(std1, np.finfo(float).tiny)
                vmin1 = med1 - 3 * std1
                vmax1 = med1 + 3 * std1
                norm1 = ImageNormalize(vmin=vmin1, vmax=vmax1, stretch=LinearStretch())
            else:
                norm1 = ImageNormalize(
                    cutout1, interval=ZScaleInterval(), stretch=LinearStretch()
                )
            im1 = ax1.imshow(
                first_image,
                origin="lower",
                cmap="viridis",
                norm=norm1,
                interpolation=None,
            )

            if "x_pix" in sources and "y_pix" in sources:
                ax1.scatter(
                    sources["x_pix"],
                    sources["y_pix"],
                    marker="+",
                    c="cyan",
                    s=30,
                    lw=0.5,
                    label="Input position",
                    zorder=10,
                )

            if "x_fit" in sources and "y_fit" in sources:
                for _, row in sources.iterrows():
                    ax1.add_patch(
                        Circle(
                            (row["x_fit"], row["y_fit"]),
                            aperture_radius,
                            edgecolor="white",
                            facecolor="none",
                            lw=0.5,
                            ls="--",
                            alpha=0.8,
                        )
                    )
                    xe, ye = row.get("x_fit_err", np.nan), row.get("y_fit_err", np.nan)
                    if (
                        np.isfinite(xe)
                        and np.isfinite(ye)
                        and xe > 0
                        and ye > 0
                        and xe < scale
                        and ye < scale
                    ):
                        ax1.add_patch(
                            Ellipse(
                                (row["x_fit"], row["y_fit"]),
                                2 * xe,
                                2 * ye,
                                angle=0,
                                edgecolor="#FF0000",
                                facecolor="none",
                                lw=0.5,
                                alpha=1,
                                zorder=1,
                            )
                        )

            # ---- Projections helper ----------------------------------------
            # Right-panel step is drawn separately with _draw_right_step so the profile
            # is centered in the error band (step constant over same y-blocks as fill).
            def _proj(main_ax, ax_R, ax_B, data, color="blue", draw_right=True):
                xl, yl = main_ax.get_xlim(), main_ax.get_ylim()
                xi0, xi1 = max(int(xl[0]), 0), min(int(xl[1]), data.shape[1])
                yi0, yi1 = max(int(yl[0]), 0), min(int(yl[1]), data.shape[0])
                cut = data[yi0:yi1, xi0:xi1]
                ax_B.step(
                    np.arange(xi0, xi1),
                    cut.mean(axis=0),
                    color=color,
                    lw=0.5,
                    where="mid",
                )
                if draw_right:
                    ax_R.step(
                        cut.mean(axis=1),
                        np.arange(yi0, yi1),
                        color=color,
                        lw=0.5,
                        where="mid",
                    )
                ax_R.set_yticklabels([])

            def _draw_right_step(ax_R, hy, y0, y1, color="dodgerblue"):
                """Draw right-panel profile as a step constant over the same y-blocks as
                fill_betweenx(..., step='mid'), so the line is centered in the error band.
                """
                n = len(hy)
                if n == 0:
                    return
                # Blocks match fill_betweenx step='mid': [(y[i-1]+y[i])/2, (y[i]+y[i+1])/2]
                y_edges = np.empty(2 * n)
                for i in range(n):
                    y_edges[2 * i] = (y0 + i - 0.5) if i > 0 else y0
                    y_edges[2 * i + 1] = (y0 + i + 0.5) if i < n - 1 else (y1 - 1)
                x_step = np.repeat(hy, 2)
                ax_R.plot(x_step, y_edges, color=color, lw=0.5)

            for ax, _, _ in ax_list:
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)

            _proj(ax1, ax1_R, ax1_B, first_image, draw_right=False)

            # Error shading on science panel: SE of mean = sqrt(sum(sigma^2))/N (not mean(sigma)).
            hx = cutout1.mean(axis=0)
            hy = cutout1.mean(axis=1)
            n_rows, n_cols = unc_cut.shape[0], unc_cut.shape[1]
            exh = np.sqrt(np.nansum(unc_cut**2, axis=0)) / max(n_rows, 1)
            eyh = np.sqrt(np.nansum(unc_cut**2, axis=1)) / max(n_cols, 1)
            y_vals = np.arange(y0, y1, dtype=float)
            kw_bottom = dict(
                facecolor="dodgerblue", edgecolor="none", alpha=0.3, step="mid"
            )
            # Right panel: fill uses step='mid' (constant in y-blocks); profile drawn as
            # step over same y-blocks so line is centered in the band.
            kw_right = dict(
                facecolor="dodgerblue", edgecolor="none", alpha=0.3, step="mid"
            )
            ax1_B.fill_between(np.arange(x0, x1), hx - exh, hx + exh, **kw_bottom)
            ax1_R.fill_betweenx(y_vals, hy - eyh, hy + eyh, **kw_right)
            _draw_right_step(ax1_R, hy, y0, y1, color="dodgerblue")
            # Set right-panel xlim so the error band is visible (same as panel 2).
            _lo_r1 = np.nanmin(hy - eyh)
            _hi_r1 = np.nanmax(hy + eyh)
            _margin_r1 = (
                max((_hi_r1 - _lo_r1) * 0.05, 1e-9)
                if np.isfinite(_hi_r1 - _lo_r1)
                else 0.1
            )
            ax1_R.set_xlim(_lo_r1 - _margin_r1, _hi_r1 + _margin_r1)

            # ---- Panel 2: residual -----------------------------------------
            idx = 1
            if psfphot is not None:
                ax2, ax2_R, ax2_B = ax_list[idx]
                cutout2 = second_image[y0:y1, x0:x1]
                if plotTarget:
                    _, med2, std2 = sigma_clipped_stats(cutout2, sigma=3, maxiters=5)
                    std2 = max(std2, np.finfo(float).tiny)
                    vmin2 = med2 - 3 * std2
                    vmax2 = med2 + 3 * std2
                    norm2 = ImageNormalize(
                        vmin=vmin2, vmax=vmax2, stretch=LinearStretch()
                    )
                else:
                    norm2 = ImageNormalize(
                        cutout2, interval=ZScaleInterval(), stretch=LinearStretch()
                    )
                im2 = ax2.imshow(
                    second_image,
                    origin="lower",
                    cmap="viridis",
                    norm=norm2,
                    interpolation=None,
                )
                _proj(ax2, ax2_R, ax2_B, second_image, draw_right=False)

                # Error shading on residual panel (same SE of mean as science).
                hx2 = cutout2.mean(axis=0)
                hy2 = cutout2.mean(axis=1)
                ax2_B.fill_between(np.arange(x0, x1), hx2 - exh, hx2 + exh, **kw_bottom)
                ax2_R.fill_betweenx(y_vals, hy2 - eyh, hy2 + eyh, **kw_right)
                _draw_right_step(ax2_R, hy2, y0, y1, color="dodgerblue")

                # Zoom bottom and right panels onto the fit profile (scale to data range).
                _lo_b = np.nanmin(hx2 - exh)
                _hi_b = np.nanmax(hx2 + exh)
                _margin_b = (
                    max((_hi_b - _lo_b) * 0.05, 1e-9)
                    if np.isfinite(_hi_b - _lo_b)
                    else 0.1
                )
                ax2_B.set_ylim(_lo_b - _margin_b, _hi_b + _margin_b)
                _lo_r = np.nanmin(hy2 - eyh)
                _hi_r = np.nanmax(hy2 + eyh)
                _margin_r = (
                    max((_hi_r - _lo_r) * 0.05, 1e-9)
                    if np.isfinite(_hi_r - _lo_r)
                    else 0.1
                )
                ax2_R.set_xlim(_lo_r - _margin_r, _hi_r + _margin_r)

                _proj(ax1, ax1_R, ax1_B, fitted_model, color="#FF0000")
                idx += 1

            # ---- Panel 3: ePSF ---------------------------------------------
            if epsf is not None:
                ax3, ax3_R, ax3_B = ax_list[idx]
                norm3 = ImageNormalize(
                    epsf.data, interval=ZScaleInterval(), stretch=LinearStretch()
                )
                im3 = ax3.imshow(
                    epsf.data,
                    origin="lower",
                    cmap="viridis",
                    norm=norm3,
                    interpolation=None,
                )
                ax3.set_title(
                    f"Oversampled ePSF (x{epsf.oversampling})", fontsize=7, pad=2
                )

            # ---- Colorbars -------------------------------------------------
            cbar1 = fig.colorbar(im1, cax=cax_list[0], orientation="horizontal")
            cbar1.set_label("Science Flux")
            cax_list[0].xaxis.set_ticks_position("top")
            cax_list[0].xaxis.set_label_position("top")

            if psfphot is not None:
                cbar2 = fig.colorbar(im2, cax=cax_list[1], orientation="horizontal")
                cbar2.set_label(second_label)
                cax_list[1].xaxis.set_ticks_position("top")
                cax_list[1].xaxis.set_label_position("top")

            if epsf is not None:
                cbar3 = fig.colorbar(im3, cax=cax_list[-1], orientation="horizontal")
                cbar3.set_label("Normalised PSF")
                cax_list[-1].xaxis.set_ticks_position("top")
                cax_list[-1].xaxis.set_label_position("top")

            for _ax, _ax_R, _ax_B in ax_list:
                _ax_B.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
                _ax_R.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            ax1.legend(loc="upper right", frameon=False, labelcolor="w")
            save_name_png = (
                f"PSF_Target_{base}.png" if plotTarget else f"PSF_Subtractions_{base}.png"
            )
            plt.savefig(
                os.path.join(write_dir, save_name_png),
                bbox_inches="tight",
                dpi=150,
                facecolor="white",
            )
            plt.close(fig)

        except Exception as exc:
            log.error(f"plot() failed: {exc}", exc_info=True)

    # -----------------------------------------------------------------------
    # MCMC corner plot
    # -----------------------------------------------------------------------

    def _plot_mcmc_corner(
        self,
        fitter,
        sourceidx: int,
        writedir: str = None,
        write_dir: str = None,
        base: str = "psffit",
        truths=None,
        orig=None,
        bounds=None,
    ):
        """Corner plot for MCMC posterior of one source."""
        log = logging.getLogger(__name__)

        # Accept both spellings.
        if writedir is None:
            writedir = write_dir

        fitinfo = getattr(fitter, "fitinfo", None) or getattr(fitter, "fit_info", None)
        if not isinstance(fitinfo, dict):
            log.warning("No fitter.fit_info dict; skipping corner plot.")
            return None

        samples_dict = fitinfo.get("samples")
        if samples_dict is None or sourceidx not in samples_dict:
            log.warning(f"No MCMC samples for source {sourceidx}.")
            return None

        samples_all = np.asarray(samples_dict[sourceidx], float)
        if (
            samples_all.ndim != 2
            or samples_all.shape[0] < 10
            or not np.isfinite(samples_all).all()
        ):
            log.warning(f"Invalid samples for source {sourceidx}.")
            return None

        npar_all = samples_all.shape[1]

        # Recover parameter names.
        param_names_all = None
        per_src = fitinfo.get("per_source")
        if isinstance(per_src, (list, tuple)) and len(per_src) > sourceidx:
            rec = per_src[sourceidx]
            if isinstance(rec, dict):
                param_names_all = rec.get("param_names") or rec.get("paramnames")

        if not param_names_all or len(param_names_all) != npar_all:
            param_names_all = getattr(fitter, "param_names", None) or [
                f"p{i}" for i in range(npar_all)
            ]

        param_names_all = [str(p) for p in param_names_all]

        def _find_idx(candidates):
            for nm in candidates:
                if nm in param_names_all:
                    return param_names_all.index(nm)
            return None

        x_idx = _find_idx(("x_0", "x0", "x", "x_mean", "xcenter"))
        y_idx = _find_idx(("y_0", "y0", "y", "y_mean", "ycenter"))
        f_idx = _find_idx(("flux", "amplitude", "amp"))

        if (x_idx is None or y_idx is None or f_idx is None) and npar_all == 3:
            n0, n1, n2 = param_names_all
            if (
                any(k in n0.lower() for k in ("flux", "amplitude", "amp"))
                and "x" in n1.lower()
                and "y" in n2.lower()
            ):
                f_idx, x_idx, y_idx = 0, 1, 2

        if x_idx is None or y_idx is None or f_idx is None:
            log.warning(
                f"Cannot identify X/Y/Flux in {param_names_all}; skipping corner plot."
            )
            return None

        sel_idx = [x_idx, y_idx, f_idx]
        sel_names = [param_names_all[i] for i in sel_idx]
        samples = samples_all[:, sel_idx]
        labels = ["X [pix]", "Y [pix]", "Flux"]

        def _as_sel_array(param_value):
            if param_value is None:
                return None
            if isinstance(param_value, dict):
                out = np.full(3, np.nan)
                for k, nm in enumerate(sel_names):
                    if nm in param_value and param_value[nm] is not None:
                        out[k] = float(param_value[nm])
                return out
            arr = np.asarray(param_value, float).ravel()
            return arr if arr.size == 3 else None

        truths_arr = _as_sel_array(truths)
        orig_arr = _as_sel_array(orig)

        bnd = dict(bounds) if isinstance(bounds, dict) else {}
        if (not bnd) and (orig_arr is not None) and hasattr(fitter, "delta"):
            delta = float(fitter.delta)
            for k in (0, 1):  # X and Y only
                bnd[sel_names[k]] = (
                    (orig_arr[k] - delta, orig_arr[k] + delta)
                    if np.isfinite(orig_arr[k])
                    else (None, None)
                )

        cfg = getattr(self, "input_yaml", {}) or {}
        fpath = cfg.get("fpath", "image")
        writedir = writedir or cfg.get("write_dir", ".")
        os.makedirs(writedir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(fpath))[0]
        outpath_png = os.path.join(writedir, f"PSF_MCMC_Corner_{stem}.png")

        fig = corner.corner(
            samples,
            labels=labels,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            title_fmt=".3f",
            truths=truths_arr,
        )
        axgrid = np.array(fig.axes, dtype=object).reshape((3, 3))

        for i in range(3):
            lo, hi = bnd.get(sel_names[i], (None, None))
            ax1d = axgrid[i, i]
            if lo is not None or hi is not None:
                x0_lim, x1_lim = ax1d.get_xlim()
                ax1d.axvspan(
                    x0_lim if lo is None else lo,
                    x1_lim if hi is None else hi,
                    color="0.25",
                    alpha=0.12,
                    zorder=0,
                )
                if lo is not None:
                    ax1d.axvline(lo, color="0.25", lw=0.5, ls="--")
                if hi is not None:
                    ax1d.axvline(hi, color="0.25", lw=0.5, ls="--")
            if orig_arr is not None and np.isfinite(orig_arr[i]):
                ax1d.axvline(orig_arr[i], color="C1", lw=0.5, ls=":")

            for j in range(i):
                ax2d = axgrid[i, j]
                for bkey, fn in (
                    (sel_names[j], ax2d.axvline),
                    (sel_names[i], ax2d.axhline),
                ):
                    blo, bhi = bnd.get(bkey, (None, None))
                    if blo is not None:
                        fn(blo, color="0.25", lw=0.5, ls="--")
                    if bhi is not None:
                        fn(bhi, color="0.25", lw=0.5, ls="--")

                if orig_arr is not None:
                    if np.isfinite(orig_arr[j]):
                        ax2d.axvline(orig_arr[j], color="C1", lw=0.5, ls=":")
                    if np.isfinite(orig_arr[i]):
                        ax2d.axhline(orig_arr[i], color="C1", lw=0.5, ls=":")

        fig.suptitle("PSF MCMC posterior", y=0.98, fontsize=10)
        fig.savefig(
            outpath_png,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        log.info("Saved MCMC corner plot: %s", outpath_png)
        return outpath_png