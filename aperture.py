#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aperture photometry and aperture correction.

This module measures fluxes using circular apertures, estimates an
optimum-radius (curve-of-growth style) when requested, and computes
aperture-correction factors used to calibrate the photometry.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import logging
import warnings
import multiprocessing

# Safeguard: force BLAS/OpenMP to 1 thread before importing numpy (avoids exhausting
# process/thread limits when using multiprocessing on HPC; OpenBLAS often defaults to 128).
for _env in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env] = "1"

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scipy.optimize
from scipy.stats import mstats, median_abs_deviation

from astropy.stats import (
    biweight_midvariance,
    sigma_clipped_stats,
    mad_std,
    biweight_location,
    sigma_clip,
)
from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus,
)
from photutils.utils import calc_total_error
from photutils.profiles import CurveOfGrowth
from astropy.visualization import ImageNormalize, ZScaleInterval

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from functions import (
    log_step,
    log_exception,
    log_warning_from_exception,
    mag,
    set_size,
    resolve_verbose_level,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NSOURCES = 10  # minimum source count to justify spawning worker processes
MAX_WORKERS_DEFAULT = (
    16  # cap on default n_jobs to avoid exhausting HPC process/thread limits
)


def resolve_exposure_time_seconds(exposure_time, input_yaml: dict) -> float:
    """
    Return a valid exposure time in seconds.

    Parameters
    ----------
    exposure_time : float or None
        If not ``None``, this value is used (must be finite and > 0).
    input_yaml : dict
        Must contain ``exposure_time`` when *exposure_time* is ``None`` - normally
        set from the FITS header in ``main`` before any photometry.

    Raises
    ------
    ValueError
        If no finite exposure > 0 s is available.
    """
    if exposure_time is not None:
        try:
            et = float(exposure_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid exposure_time argument: {exposure_time!r}") from exc
        if np.isfinite(et) and et > 0:
            return et
        raise ValueError(f"exposure_time must be finite and > 0 s, got {exposure_time!r}")
    raw = input_yaml.get("exposure_time")
    if raw is None:
        raise ValueError(
            "exposure_time is required: pass it to measure(), or set "
            "input_yaml['exposure_time'] (normally from the FITS EXPTIME header "
            "before running photometry)."
        )
    try:
        et = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input_yaml['exposure_time'] is not numeric: {raw!r}") from exc
    if not np.isfinite(et) or et <= 0:
        raise ValueError(
            f"input_yaml['exposure_time'] must be finite and > 0 s, got {raw!r}"
        )
    return et


def exposure_seconds_from_header(header, preferred_keys=None):
    """
    Read a positive, finite exposure time in seconds from a FITS ``header``.

    Parameters
    ----------
    header : mapping
        FITS header (supports ``in`` / ``__getitem__``).
    preferred_keys : sequence of str, optional
        Tried first in order (e.g. telescope.yml ``exptime`` mapping), then
        standard alternates.

    Returns
    -------
    (exposure_time, key_used) : (float, str)

    Raises
    ------
    ValueError
        If no usable exposure keyword is found.
    """
    keys = []
    if preferred_keys is not None:
        for k in preferred_keys:
            if not k or k == "not_given_by_user":
                continue
            if k not in keys:
                keys.append(k)
    for alt in (
        "EXPTIME",
        "EXPOSURE",
        "TEXP",
        "EXPTIME0",
        "TEXPTIME",
        "INTTIME",
        "EXPTIM",
    ):
        if alt not in keys:
            keys.append(alt)
    for key in keys:
        if key not in header:
            continue
        raw = header[key]
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0:
            return val, key
    raise ValueError(
        "No valid exposure time in FITS header; tried keys "
        f"{keys[:12]!r}. Add EXPTIME or EXPOSURE (seconds)."
    )


def resolve_gain_e_per_adu(gain, input_yaml: dict) -> float:
    """
    Return a valid detector gain in electrons per ADU.

    Parameters
    ----------
    gain : float or None
        If not ``None``, this value is used (must be finite and > 0).
    input_yaml : dict
        Must contain ``gain`` when *gain* is ``None`` - normally set from the
        FITS header in ``main`` before any photometry.

    Raises
    ------
    ValueError
        If no finite gain > 0 e-/ADU is available.
    """
    if gain is not None:
        try:
            g = float(gain)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid gain argument: {gain!r}") from exc
        if np.isfinite(g) and g > 0:
            return g
        raise ValueError(f"gain must be finite and > 0 e-/ADU, got {gain!r}")
    raw = input_yaml.get("gain")
    if raw is None:
        raise ValueError(
            "gain is required: pass it to measure(), or set "
            "input_yaml['gain'] (normally from the FITS GAIN header before "
            "running photometry)."
        )
    try:
        g = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input_yaml['gain'] is not numeric: {raw!r}") from exc
    if not np.isfinite(g) or g <= 0:
        raise ValueError(
            f"input_yaml['gain'] must be finite and > 0 e-/ADU, got {raw!r}"
        )
    return g


def gain_e_per_adu_from_header(header, preferred_keys=None):
    """
    Read a positive, finite gain (e-/ADU) from a FITS ``header``.

    Parameters
    ----------
    header : mapping
        FITS header (supports ``in`` / ``__getitem__``).
    preferred_keys : sequence of str, optional
        Tried first in order (e.g. telescope.yml ``gain`` mapping), then
        standard alternates.

    Returns
    -------
    (gain, key_used) : (float, str)

    Raises
    ------
    ValueError
        If no usable gain keyword is found.
    """
    keys = []
    if preferred_keys is not None:
        for k in preferred_keys:
            if not k or k == "not_given_by_user":
                continue
            if k not in keys:
                keys.append(k)
    # Pan-STARRS / PS1 hierarchy cards (see utils/fix_panstarrs_headers.py).
    for alt in (
        "GAIN",
        "gain",
        "EGAIN",
        "CONADU",
        "CELL.GAIN",
        "DET.GAIN",
    ):
        if alt not in keys:
            keys.append(alt)
    for key in keys:
        if key not in header:
            continue
        raw = header[key]
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0:
            return val, key
    raise ValueError(
        "No valid detector gain (e-/ADU) in FITS header; tried keys "
        f"{keys[:12]!r}. Add GAIN or the instrument-specific keyword."
    )


def _resolve_n_jobs(n_jobs, half_cpus=False):
    """Resolve and cap worker count for multiprocessing (safeguards HPC process/thread limits).

    ``n_jobs is None`` defaults to 1 (serial). ``half_cpus`` is kept for call-site compatibility
    and is ignored.
    """
    if n_jobs is None:
        return 1
    return min(MAX_WORKERS_DEFAULT, max(1, int(n_jobs)))


# ===========================================================================
# Module-level worker functions  (MUST be at module scope for pickle)
# ===========================================================================

# Shared per-worker state: the full image, error map, phot table, masks, and
# all aperture/annulus masks are broadcast ONCE via Pool(initializer=...)
# instead of being pickled into every per-source task.
_AP_CTX: dict = {}


def _measure_init_shared(
    aperture_masks, annulus_masks, image_e, error, read_noise_sq,
    inv_exposure_time, area, phot, gain, enforce_nonnegative_local_bkg,
    verbose, defects_mask,
):
    _AP_CTX.update(
        aperture_masks=aperture_masks,
        annulus_masks=annulus_masks,
        image_e=image_e,
        error=error,
        read_noise_sq=read_noise_sq,
        inv_exposure_time=inv_exposure_time,
        area=area,
        phot=phot,
        gain=gain,
        enforce_nonnegative_local_bkg=enforce_nonnegative_local_bkg,
        verbose=verbose,
        defects_mask=defects_mask,
    )


def _measure_worker_shared(i):
    """Pool entry point: only the source index is pickled per task."""
    c = _AP_CTX
    return _measure_worker(
        (
            i,
            c["aperture_masks"],
            c["annulus_masks"],
            c["image_e"],
            c["error"],
            c["read_noise_sq"],
            c["inv_exposure_time"],
            c["area"],
            c["phot"],
            c["gain"],
            c["enforce_nonnegative_local_bkg"],
            c["verbose"],
            c["defects_mask"],
        )
    )


def _optimum_radius_init_shared(
    fwhm, radii, image, error, norm_factor, stability_threshold,
    use_moffat_cog, moffat_beta, mask,
):
    _AP_CTX.update(
        opt_fwhm=fwhm,
        opt_radii=radii,
        opt_image=image,
        opt_error=error,
        opt_norm_factor=norm_factor,
        opt_stability_threshold=stability_threshold,
        opt_use_moffat_cog=use_moffat_cog,
        opt_moffat_beta=moffat_beta,
        opt_mask=mask,
    )


def _optimum_radius_worker_shared(args):
    """Pool entry point: only (idx, x_pix, y_pix) is pickled per task."""
    idx, x_pix, y_pix = args
    c = _AP_CTX
    return _optimum_radius_worker(
        (
            idx,
            x_pix,
            y_pix,
            c["opt_fwhm"],
            c["opt_radii"],
            c["opt_image"],
            c["opt_error"],
            c["opt_norm_factor"],
            c["opt_stability_threshold"],
            c["opt_use_moffat_cog"],
            c["opt_moffat_beta"],
            c["opt_mask"],
        )
    )


def _measure_worker(args):
    """
    Perform aperture photometry for a single source.

    Parameters
    ----------
    args : tuple
        (i, aperture_masks, annulus_masks, image_e, error,
         read_noise_sq, inv_exposure_time, area, phot, gain,
         enforce_nonnegative_local_bkg, verbose, defects_mask)

    Returns
    -------
    dict  - result keyed by 'idx'; contains 'fail_reason' on failure.
    """
    (
        i,
        aperture_masks,
        annulus_masks,
        image_e,
        error,
        read_noise_sq,
        inv_exposure_time,
        area,
        phot,
        gain,
        enforce_nonnegative_local_bkg,
        verbose,
        defects_mask,
    ) = args

    try:
        ap_mask = aperture_masks[i]
        an_mask = annulus_masks[i]

        def _finite_mask_values(mask_obj, img2d):
            """
            Return pixel values under an ApertureMask, excluding padded bounding-box
            zeros introduced by mask multiplication.
            """
            try:
                cut = mask_obj.multiply(img2d)
                w = np.asarray(mask_obj.data, dtype=float)
                vals = np.asarray(cut, dtype=float)[w > 0]
                return vals
            except Exception:
                # Fallback to photutils helper (may include bbox-padding zeros).
                return mask_obj.get_values(img2d)

        ap_pix = _finite_mask_values(ap_mask, image_e)
        bkg_pix = _finite_mask_values(an_mask, image_e)

        # Apply hardware defects mask (trails, streaks, saturation, NaN)
        # so bad pixels are excluded from aperture flux and annulus background.
        if defects_mask is not None:
            ap_mask_vals = _finite_mask_values(ap_mask, defects_mask)
            bkg_mask_vals = _finite_mask_values(an_mask, defects_mask)
            if len(ap_mask_vals) == len(ap_pix):
                ap_pix = np.where(ap_mask_vals > 0, np.nan, ap_pix)
            if len(bkg_mask_vals) == len(bkg_pix):
                bkg_pix = np.where(bkg_mask_vals > 0, np.nan, bkg_pix)

        # Optional per-pixel uncertainty (e.g. from Background2D / calc_total_error).
        ap_err_pix = None
        if error is not None:
            ap_err_pix = _finite_mask_values(ap_mask, error)
            if defects_mask is not None:
                ap_err_mask_vals = _finite_mask_values(ap_mask, defects_mask)
                if len(ap_err_mask_vals) == len(ap_err_pix):
                    ap_err_pix = np.where(ap_err_mask_vals > 0, np.nan, ap_err_pix)

        # Remove NaNs/infs. Do NOT discard exact zeros here: difference images
        # and locally background-subtracted stamps can legitimately contain 0-valued
        # pixels. However, if the aperture is dominated by exact zeros, it's likely
        # SWarp padding rather than real data.

        # STRICT CHECK: Aperture must have NO NaNs (measurement region must be clean)
        ap_has_nan = not np.all(np.isfinite(ap_pix))
        if ap_has_nan:
            return {"idx": i, "fail_reason": "aperture_has_nan"}

        # Check for SWarp-padded zero regions in the aperture
        ap_zero_frac = float(np.mean(ap_pix == 0.0))
        if ap_zero_frac > 0.5:
            return {"idx": i, "fail_reason": "aperture_swarp_padding"}

        # TOLERANT CHECK: Annulus can have some NaNs, but needs minimum valid pixels
        # Also filter exact-zero pixels: SWarp pads uncovered regions with 0.0
        # (not NaN), and these produce zero-variance backgrounds (bkg_invalid).
        # On science images with real sky background, exact 0.0 ADU is unphysical;
        # on difference images the sky is already subtracted to ~0 but has noise,
        # so zero-variance zeros are still SWarp padding, not real pixels.
        n_annulus_total = bkg_pix.size  # before NaN/zero filtering
        bkg_pix = bkg_pix[np.isfinite(bkg_pix)]
        bkg_pix = bkg_pix[bkg_pix != 0.0]
        # Require at least 50% of *annulus* pixels to be valid for background
        # estimation, with an absolute floor of 10 pixels (a heavily clipped
        # edge annulus could otherwise pass with a handful of pixels).
        # NOTE: the denominator must be the annulus pixel count, not the
        # aperture count -- the annulus is typically ~4-6x larger, so using
        # len(ap_pix) here would accept sites with only ~10% valid annulus.
        annulus_valid_fraction = 0.5
        if bkg_pix.size < max(10.0, annulus_valid_fraction * n_annulus_total):
            return {"idx": i, "fail_reason": "annulus_too_many_nans"}

        if ap_pix.size == 0:
            return {"idx": i, "fail_reason": "empty_aperture"}

        # Median background with MAD scatter: resistant to outliers, handles negatives cleanly.
        bkg_value = np.median(bkg_pix)
        bkg_value_used = (
            max(float(bkg_value), 0.0)
            if bool(enforce_nonnegative_local_bkg)
            else float(bkg_value)
        )
        local_bkg_floored = bool(
            bool(enforce_nonnegative_local_bkg) and np.isfinite(bkg_value) and bkg_value < 0
        )
        empirical_std = 1.4826 * np.median(np.abs(bkg_pix - bkg_value))

        if empirical_std <= 0 or not np.isfinite(empirical_std):
            return {"idx": i, "fail_reason": "bkg_invalid"}

        row = phot.iloc[i]
        raw_aperture_sum = row.aperture_sum
        if not np.isfinite(raw_aperture_sum):
            return {"idx": i, "fail_reason": "aperture_sum_invalid"}

        # IMPORTANT: `photutils.aperture_photometry` (used upstream) integrates
        # flux with an exact aperture model by default (fractional edge pixels).
        # The background subtraction must therefore use the same *geometric*
        # area (pi*r^2), not the integer pixel count from a "center" mask.
        # Using mask-summed area here can under-subtract background and inflate
        # fluxes (and breaks "quiet site" selection for injections).
        effective_area = float(area)
        aperture_bkg = bkg_value_used * effective_area
        aperture_sum = raw_aperture_sum - aperture_bkg

        # Peak pixel within the aperture: use the raw (unweighted) pixel
        # values covered by the mask.  mask.multiply() returns weight*value,
        # so a brightest pixel landing on a fractional-weight edge pixel of
        # an "exact" aperture would be biased low; cutout()+w>0 selects the
        # raw values instead.  (ap_pix is already guaranteed all-finite and
        # defect-free by the checks above.)
        try:
            _ap_raw = np.asarray(ap_mask.cutout(image_e), dtype=float)[
                np.asarray(ap_mask.data, dtype=float) > 0
            ]
            raw_max = float(np.nanmax(_ap_raw)) if _ap_raw.size else np.nan
            if not np.isfinite(raw_max):
                raw_max = np.max(ap_pix)
        except Exception:
            raw_max = np.max(ap_pix)
        max_val = raw_max - bkg_value_used

        # Variance model: prefer fully propagated per-pixel uncertainties
        # when available; otherwise fall back to a simple Poisson+sky+read-noise
        # approximation based on the empirical background standard deviation.
        sqrt_var = np.nan
        # Preferred path: photutils' aperture_sum_err, which uses the same
        # fractional-pixel (exact) aperture geometry as raw_aperture_sum.
        # The center-mask error sum below drops edge pixels entirely and
        # undercounts the aperture noise by ~perimeter/area.
        if error is not None and "aperture_sum_err" in phot.columns:
            _ase = row.get("aperture_sum_err", np.nan)
            if np.isfinite(_ase) and _ase > 0:
                sqrt_var = float(_ase)
        if not np.isfinite(sqrt_var) and ap_err_pix is not None:
            # error array is in electrons; propagate by summing variances.
            # Use only finite values to avoid contamination from bad pixels
            ap_err_finite = ap_err_pix[np.isfinite(ap_err_pix)]
            if len(ap_err_finite) > 0:
                var_from_error = np.nansum(ap_err_finite.astype(float) ** 2)
                if var_from_error > 0 and np.isfinite(var_from_error):
                    sqrt_var = np.sqrt(var_from_error)

        if not np.isfinite(sqrt_var):
            # Fallback variance: |source| + area * sigma_sky^2
            # empirical_std is the MAD-based per-pixel scatter from the
            # annulus, which already includes read noise.  Do NOT add
            # read_noise_sq again -- that double-counts it.
            source_flux = abs(aperture_sum)
            sky_var = max(empirical_std**2, 0.0)  # No artificial floor; use measured background
            total_var = source_flux + effective_area * sky_var
            if total_var > 0 and np.isfinite(total_var):
                sqrt_var = np.sqrt(total_var)

        if np.isfinite(sqrt_var) and sqrt_var > 0:
            snr = aperture_sum / sqrt_var
        else:
            sqrt_var = np.nan
            snr = 0.0

        # Aperture sum is integrated over the exposure in image_e units (e- in frame);
        # flux is the rate in e-/s for use with mag() and PSF outputs (also e-/s).
        # Aperture-only flux (for total light include aperture correction).
        flux_ap = aperture_sum * inv_exposure_time

        try:
            mag_val = mag(flux_ap)
        except Exception:
            mag_val = np.nan

        # Magnitude uncertainty: use the standard linear approximation
        # (2.5/ln(10)) * (flux_err/flux) = 1.0857/SNR, consistent with PSF
        # photometry (psf.py) and _compute_delta_mag (zeropoint.py).
        # The old snr_err(snr) = 2.5*log10(1 + 1/SNR) is the exact lower
        # error bar and is always smaller than the linear approx, causing
        # AP magnitude errors to be systematically underestimated relative
        # to PSF at the same SNR (15% bias at SNR=5, 25% at SNR=3).
        mag_err_val = np.nan
        try:
            if (
                np.isfinite(flux_ap)
                and flux_ap > 0
                and np.isfinite(snr)
                and snr > 0
            ):
                mag_err_val = (2.5 / np.log(10.0)) / snr
        except Exception:
            mag_err_val = np.nan

        # maxPixel_err is the uncertainty of a SINGLE pixel (the brightest
        # pixel in the aperture), not the aperture sum.  The variance of a
        # single pixel is source_e + sky_e + read^2 ~ |max_val| + std^2
        # (max_val = raw_max - bkg is the sky-subtracted peak; empirical_std^2
        # already contains sky+read noise).  Using |raw_max| here would
        # double-count the sky term (raw_max ~ source + sky).
        # Do NOT scale sky by aperture area (that would overestimate by
        # sqrt(area) and bias the m_peak_err weights in FWHM fitting).
        max_flux_err = np.sqrt(np.abs(max_val) + empirical_std**2) * inv_exposure_time

        return {
            "idx": i,
            "maxPixel": max_val * inv_exposure_time,
            "maxPixel_err": max_flux_err,
            "area": effective_area,
            "counts_AP": aperture_sum,
            "flux_AP": flux_ap,
            "flux_AP_err": (
                sqrt_var * inv_exposure_time if np.isfinite(sqrt_var) else np.nan
            ),
            "sky_bkg_total": aperture_bkg,
            "sky_bkg_total_flux": aperture_bkg * inv_exposure_time,
            "noiseSky": empirical_std * inv_exposure_time,
            "threshold": max_val / empirical_std,
            "SNR": snr,
            "bkg_std_method": "MAD",
            "local_bkg_raw": float(bkg_value),
            "local_bkg_used": float(bkg_value_used),
            "local_bkg_floored": local_bkg_floored,
            "mag": mag_val,
            "mag_err": mag_err_val,
        }

    except Exception as exc:
        if verbose >= 2:
            logging.getLogger(__name__).error(
                f"Error processing source {i}: {exc}", exc_info=True
            )
        return {"idx": i, "fail_reason": str(exc)}


def _cog_profile_worker(args):
    """
    Compute a normalised Curve-of-Growth profile for one source.

    Parameters
    ----------
    args : tuple
        (idx, x_pix, y_pix, fwhm, radii, image, error)

    Returns
    -------
    dict or None
    """
    idx, x_pix, y_pix, fwhm, radii, image, error = args
    try:
        xycen = np.array([x_pix, y_pix])
        cog = CurveOfGrowth(
            image, xycen, radii, error=error, mask=None, method="subpixel"
        )
        cog.normalize()
        return {"idx": idx, "radii": cog.radii, "profile": cog.profile}
    except Exception:
        return None


def _optimum_radius_worker(args):
    """
    Compute the per-source optimum aperture radius at the target encircled
    energy fraction (norm_factor), with an SNR guard.

    Parameters
    ----------
    args : tuple
        (idx, x_pix, y_pix, fwhm, radii, image, error,
         norm_factor, stability_threshold, use_moffat_cog, moffat_beta)

    Returns
    -------
    dict or None
    """
    (
        idx,
        x_pix,
        y_pix,
        fwhm,
        radii,
        image,
        error,
        norm_factor,
        stability_threshold,
        use_moffat_cog,
        moffat_beta,
        mask,
    ) = (
        args
    )

    try:
        xycen = np.array([x_pix, y_pix])
        cog = CurveOfGrowth(
            image, xycen, radii, error=error, mask=mask, method="subpixel"
        )
        cog.normalize()

        norm_profile = cog.profile
        norm_profile_err = cog.profile_error

        if use_moffat_cog:
            # Analytic encircled-energy inversion for a circular Moffat profile:
            # E(r) = 1 - [1 + (r/alpha)^2]^(1-beta), beta>1.
            b = max(float(moffat_beta), 1.01)
            alpha = float(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / b) - 1.0))
            ee = float(np.clip(norm_factor, 1e-6, 1 - 1e-6))
            r_at_norm = alpha * np.sqrt((1.0 - ee) ** (1.0 / (1.0 - b)) - 1.0)
        else:
            r_at_norm = cog.calc_radius_at_ee(norm_factor)
        if not np.isfinite(r_at_norm):
            return None

        opt_r_fwhm = float(r_at_norm / fwhm)

        # SNR guard at the chosen radius.
        r_pix = opt_r_fwhm * fwhm
        idx_r = int(np.argmin(np.abs(cog.radii - r_pix)))
        enc_f = norm_profile[idx_r]
        # profile_error is an empty array when no error map was supplied;
        # guard against indexing it (returns scalar error when present).
        enc_err = (
            norm_profile_err[idx_r]
            if norm_profile_err is not None and np.size(norm_profile_err) > idx_r
            else None
        )
        if (
            enc_err is not None
            and np.isfinite(enc_err)
            and enc_err > 0
            and (enc_f / enc_err) < 3
        ):
            return None

        # Mean slope inside the per-source optimum radius (monotonicity proxy).
        within_opt = (cog.radii / fwhm) <= opt_r_fwhm
        if np.count_nonzero(within_opt) > 1:
            mean_slope = float(
                np.nanmean(np.gradient(norm_profile[within_opt], cog.radii[within_opt]))
            )
        else:
            mean_slope = float("nan")

        # Local tail deviation and excess above 1 (neighbour contamination).
        beyond_local = (cog.radii / fwhm) > opt_r_fwhm
        tail_max_dist = (
            float(np.nanmax(np.abs(norm_profile[beyond_local] - 1.0)))
            if np.any(beyond_local)
            else float("nan")
        )
        # Excess above 1 at large radius: CoG increasing there indicates a neighbouring source.
        tail_excess = (
            float(max(0.0, np.nanmax(norm_profile[beyond_local]) - 1.0))
            if np.any(beyond_local)
            else 0.0
        )

        # Surrounding environment: MAD-based scatter in an annulus just outside the star.
        # Prefer low local std (clean background, no bright neighbour or gradient).
        local_env_std = float("nan")
        try:
            inner_r = max(r_at_norm * 1.2, 2.0 * fwhm)
            outer_r = 4.0 * fwhm
            if outer_r > inner_r + fwhm * 0.5:
                annulus = CircularAnnulus(xycen, inner_r, outer_r)
                amask = annulus.to_mask(method="center")
                apix = amask.get_values(image)
                apix = apix[np.isfinite(apix)]
                if len(apix) >= 10:
                    local_env_std = float(mad_std(apix))
        except Exception:
            pass

        return {
            "idx": idx,
            "optimum_radius": opt_r_fwhm,
            "mean_slope": mean_slope,
            "tail_max_dist_local": tail_max_dist,
            "tail_excess": tail_excess,
            "local_env_std": local_env_std,
            "profile": norm_profile,
        }

    except Exception:
        return None


# ===========================================================================
# Aperture class
# ===========================================================================


class Aperture:
    """
    Circular aperture photometry with background annulus subtraction.

    Provides:
    * measure()               - per-source flux / magnitude measurements
    * measure_optimum_radius() - data-driven aperture radius selection
    * compute_aperture_correction() - CoG-based aperture correction

    **Flux / count conventions in ``measure()`` output**

    Photometry is performed on ``self.image * gain`` (``image_e``), so sums are in
    *electrons* integrated over the exposure, unless the caller uses gain=1.0
    and treats ADU as a proxy.

    * ``counts_AP`` - background-subtracted aperture sum over the *full* exposure
      (e- in the frame, i.e. integrated, not a rate).
    * ``flux_AP`` - per-second rate ``counts_AP / exposure_time`` (e-/s), matching
      ``functions.mag()`` and PSF photometry's ``flux_PSF`` (also e-/s).
    * ``noiseSky`` - local sky RMS per *pixel* in the same *rate* units (e-/s per
      pixel), consistent with ``flux_AP`` in ``beta_aperture``-style S/N.
    """

    def __init__(self, input_yaml: dict, image: np.ndarray, verbose: int | None = None):
        """
        Parameters
        ----------
        input_yaml : dict   pipeline configuration
        image      : ndarray  2-D science image
        verbose    : int      0 = quiet, 1 = normal, 2 = debug; None reads
                     ``global_verbose_level`` from *input_yaml*.
        """
        self.input_yaml = input_yaml
        self.image = image
        if verbose is None:
            verbose = (input_yaml or {}).get("global_verbose_level", 1)
        self.verbose = resolve_verbose_level(verbose)

    # -----------------------------------------------------------------------
    # Background statistics helpers
    # -----------------------------------------------------------------------

    def optimal_background_std_estimation(
        self, bkg_pixels: np.ndarray, verbose: bool = False
    ):
        """
        Estimate the background standard deviation with a cascade of
        outlier-resistant estimators (best -> worst).

        Parameters
        ----------
        bkg_pixels : 1-D ndarray
        verbose    : bool

        Returns
        -------
        (std_bkg, method_name) : (float, str)
        """
        logger = logging.getLogger(__name__)

        def _percentile_std(data):
            p16, p84 = np.percentile(data, [16, 84])
            return 0.5 * (p84 - p16)

        def _winsorized_std(data):
            return float(np.sqrt(mstats.winsorize(data, limits=(0.05, 0.05)).var()))

        methods = [
            ("MAD", lambda d: mad_std(d, ignore_nan=True)),
            ("Biweight", lambda d: np.sqrt(biweight_midvariance(d, c=6.0))),
            ("Percentile", _percentile_std),
            ("Winsorized", _winsorized_std),
            ("SigmaClip", lambda d: sigma_clipped_stats(d, sigma=3, maxiters=5)[2]),
        ]

        for name, fn in methods:
            try:
                std_value = fn(bkg_pixels)
                if std_value > 0 and np.isfinite(std_value):
                    if verbose:
                        logger.debug(
                            f"Background std [{name}]: sigma = {std_value:.3f}"
                        )
                    return std_value, name
            except Exception as exc:
                if verbose:
                    logger.debug("%s failed: %s", name, exc)

        return np.nan, "None"

    def enhanced_background_estimation(
        self, bkg_pixels: np.ndarray, image_gain: float = 1.0, verbose: bool = False
    ):
        """
        Estimate background level and standard deviation with Poisson sanity check.

        Returns
        -------
        (bkg_level, std, method_name) or (nan, nan, 'InsufficientData')
        """
        logger = logging.getLogger(__name__)
        finite = np.isfinite(bkg_pixels)
        if finite.sum() < 10:
            return np.nan, np.nan, "InsufficientData"

        clean = bkg_pixels[finite]
        bkg_lvl = biweight_location(clean, c=6.0)

        # Biweight midvariance with Poisson sanity check.
        try:
            bw_var = biweight_midvariance(clean, c=6.0, M=bkg_lvl)
            if bw_var > 0 and np.isfinite(bw_var):
                emp_std = np.sqrt(bw_var)
                if bkg_lvl > 0:
                    poisson_std = np.sqrt(bkg_lvl / image_gain)
                    ratio = emp_std / poisson_std
                    if 0.5 < ratio < 5.0:
                        return bkg_lvl, emp_std, "Biweight"
                    if verbose:
                        logger.debug("Biweight ratio outside range: %.2f", ratio)
        except Exception:
            pass

        # MAD fallback.
        try:
            ms = mad_std(clean)
            if ms > 0 and np.isfinite(ms):
                return bkg_lvl, ms, "MAD"
        except Exception:
            pass

        # Percentile fallback.
        try:
            p16, p84 = np.percentile(clean, [16, 84])
            ps = 0.5 * (p84 - p16)
            if ps > 0:
                return bkg_lvl, ps, "Percentile"
        except Exception:
            pass

        return bkg_lvl, np.nan, "Failed"

    # -----------------------------------------------------------------------
    # Aperture photometry
    # -----------------------------------------------------------------------

    def measure(
        self,
        sources: pd.DataFrame,
        ap_size: float = None,
        exposure_time: float = None,
        read_noise: float = None,
        gain: float = None,
        plot: bool = False,
        background_rms: np.ndarray = None,
        saveTarget: bool = False,
        verbose: int | None = None,
        n_jobs: int = None,
        mask: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Measure aperture photometry for all sources in *sources*.

        Parameters
        ----------
        sources        : DataFrame with 'x_pix', 'y_pix' columns
        ap_size        : aperture radius (pixels); read from config if None
        exposure_time  : seconds; read from config if None
        read_noise     : electrons; read from config if None
        gain           : e-/ADU; from ``input_yaml`` if ``None`` (must be set; normally from FITS header in ``main``)
        plot           : save per-source diagnostic PDF
        background_rms : 2-D RMS map for error model
        saveTarget     : use filename stem (not index) when naming plot files
        verbose        : 0 quiet, 1 normal, 2 debug; None reads
                         ``global_verbose_level`` from ``input_yaml``
        n_jobs         : worker processes; None -> 1 (serial)

        Returns
        -------
        sources : DataFrame (in-place columns added / updated)

        ``counts_AP`` and ``flux_AP`` follow the conventions described in
        :class:`Aperture` (integrated e- in frame, and e-/s, respectively, when
        the pipeline uses ``image * gain`` as in this implementation).
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        pd.options.mode.chained_assignment = None

        logger = logging.getLogger(__name__)
        if verbose is None:
            verbose = (self.input_yaml or {}).get("global_verbose_level", 1)
        verbose = resolve_verbose_level(verbose)

        # ---- Configuration -------------------------------------------------
        fwhm = float(self.input_yaml["fwhm"])
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError(f"Invalid FWHM={fwhm}; expected finite and >0.")
        gain = resolve_gain_e_per_adu(gain, self.input_yaml)
        exposure_time = resolve_exposure_time_seconds(exposure_time, self.input_yaml)
        # Use explicit read_noise if provided (including 0.0); fall back to
        # input_yaml only when None.  Using `or` would ignore read_noise=0.0.
        if read_noise is None:
            read_noise = float(self.input_yaml.get("read_noise", 0.0))
        else:
            read_noise = float(read_noise)
        if ap_size is None or not np.isfinite(float(ap_size)) or ap_size <= 0:
            ap_size = self.input_yaml["photometry"]["aperture_radius"]
        ap_size = float(ap_size)
        if not np.isfinite(ap_size) or ap_size <= 0:
            raise ValueError(
                f"Invalid aperture radius={ap_size}; expected finite and >0."
            )

        crowded = self.input_yaml.get("photometry", {}).get("crowded_field", False)
        enforce_nonnegative_local_bkg = bool(
            self.input_yaml.get("photometry", {}).get(
                "enforce_nonnegative_local_background", False
            )
        )
        
        # Configurable annulus radii (in units of FWHM)
        # gap_fwhm: distance from aperture edge to annulus inner edge
        # width_fwhm: width of the annulus
        phot_cfg = self.input_yaml.get("photometry", {})
        if crowded:
            gap_fwhm = float(phot_cfg.get("annulus_gap_fwhm", 0.5))  # Default: 0.5 FWHM from aperture edge
            width_fwhm = float(phot_cfg.get("annulus_width_fwhm", 1.5))  # Default: 1.5 FWHM width
        else:
            gap_fwhm = float(phot_cfg.get("annulus_gap_fwhm", 0.75))  # Default: 0.75 FWHM from aperture edge
            width_fwhm = float(phot_cfg.get("annulus_width_fwhm", 2.0))  # Default: 2.0 FWHM width
        
        # Calculate annulus radii: inner = aperture + gap, outer = inner + width
        annulusIN = float(np.ceil(ap_size + gap_fwhm * fwhm))
        annulusOUT = float(np.ceil(annulusIN + width_fwhm * fwhm))
        
        area = np.pi * ap_size**2

        image_e = self.image * gain
        image_e[~np.isfinite(image_e)] = np.nan

        inv_exp_time = 1.0 / exposure_time
        read_noise_sq = read_noise**2

        filt = self.input_yaml["imageFilter"]
        mag_col = f"inst_{filt}_AP"
        err_col = f"inst_{filt}_AP_err"

        # ---- Ensure output columns exist -----------------------------------
        float_cols = [
            "maxPixel",
            "maxPixel_err",
            "counts_AP",
            "flux_AP",
            "flux_AP_err",
            "sky_bkg_total",
            "sky_bkg_total_flux",
            "noiseSky",
            "SNR",
            "SNR_err",
            mag_col,
            err_col,
            "threshold",
            "area",
        ]
        str_cols = ["bkg_std_method", "fail_reason"]
        for col in float_cols:
            if col not in sources.columns:
                sources[col] = np.nan
        for col in str_cols:
            if col not in sources.columns:
                sources[col] = ""

        # ---- Error model ---------------------------------------------------
        if background_rms is not None:
            # Replace NaN values (chip gaps, interpolation failures) with
            # the median of finite values so calc_total_error doesn't break.
            _bkg_rms = np.asarray(background_rms, dtype=float)
            if np.any(np.isnan(_bkg_rms)):
                _finite_median = float(np.nanmedian(_bkg_rms))
                if not np.isfinite(_finite_median) or _finite_median <= 0:
                    _finite_median = 1.0
                _bkg_rms = np.where(np.isfinite(_bkg_rms), _bkg_rms, _finite_median)
            _bkg_rms = np.abs(_bkg_rms)
            # calc_total_error already drops the Poisson term for negative
            # pixels (returns only bkg_error), which is physically correct for
            # difference images: negative pixels are noise fluctuations with
            # no source photons.  Do NOT wrap with abs() or maximum(,0) -- that
            # would add spurious Poisson noise for negative pixels.
            image_e_pois = np.where(
                np.isfinite(image_e), image_e, np.nan
            )
            error = calc_total_error(
                image_e_pois, _bkg_rms * gain, effective_gain=1
            )
        else:
            error = None

        # ---- Validate source positions -------------------------------------
        if not {"x_pix", "y_pix"}.issubset(sources.columns):
            raise ValueError(
                "sources must contain 'x_pix' and 'y_pix' columns."
            )
        x, y = sources["x_pix"].values, sources["y_pix"].values
        valid_mask = (
            (x >= 0) & (x < self.image.shape[1]) & (y >= 0) & (y < self.image.shape[0])
        )
        n_dropped = int((~valid_mask).sum())
        if n_dropped > 0:
            logger.info(
                "Dropped %d out-of-bounds/non-finite source(s) before photometry.",
                n_dropped,
            )
        sources = sources[valid_mask].reset_index(drop=True)
        if sources.empty:
            if verbose:
                logger.warning("No valid sources within image bounds.")
            return sources

        positions = list(zip(sources["x_pix"], sources["y_pix"]))

        # ---- Aperture objects & photutils photometry -----------------------
        apertures_obj = CircularAperture(positions, r=ap_size)
        annuli_obj = CircularAnnulus(positions, r_in=annulusIN, r_out=annulusOUT)

        phot = aperture_photometry(image_e, apertures_obj, error=error, mask=mask).to_pandas()
        # Use unweighted (center) pixel inclusion for consistency across the pipeline.
        # This avoids fractional-weight median biases and makes AP/PSF background
        # conventions comparable on difference images.
        aperture_masks = [ap.to_mask(method="center") for ap in apertures_obj]
        # Background estimation should use *unweighted* annulus pixels. Using the
        # "exact" (fractional) masks scales edge pixels by <1 and biases the
        # annulus median low, which then mis-subtracts sky and confuses diagnostics.
        annulus_masks = [an.to_mask(method="center") for an in annuli_obj]

        n_jobs = _resolve_n_jobs(n_jobs, half_cpus=True)

        # ---- Dispatch (parallel for large catalogs) ------------------------
        # Pooled path: broadcast shared state once per worker via initializer;
        # only the source index is pickled per task (image/masks/phot can be
        # tens of MB -- pickling them per task dominates runtime otherwise).
        if len(sources) >= NSOURCES:
            with Pool(
                processes=n_jobs,
                initializer=_measure_init_shared,
                initargs=(
                    aperture_masks,
                    annulus_masks,
                    image_e,
                    error,
                    read_noise_sq,
                    inv_exp_time,
                    area,
                    phot,
                    gain,
                    enforce_nonnegative_local_bkg,
                    verbose,
                    mask,
                ),
            ) as pool:
                results = pool.map(_measure_worker_shared, range(len(sources)))
        else:
            args_list = [
                (
                    i,
                    aperture_masks,
                    annulus_masks,
                    image_e,
                    error,
                    read_noise_sq,
                    inv_exp_time,
                    area,
                    phot,
                    gain,
                    enforce_nonnegative_local_bkg,
                    verbose,
                    mask,
                )
                for i in range(len(sources))
            ]
            results = [_measure_worker(a) for a in args_list]

        # ---- Batch update DataFrame (one dict -> update is much faster) -----
        # Collect all successful results into column-keyed lists, then assign.
        updates: dict[str, list] = {c: [np.nan] * len(sources) for c in float_cols}
        for c in str_cols:
            updates[c] = [""] * len(sources)

        fail_count = 0
        floored_count = 0
        floored_values = []
        for res in results:
            i = res.pop("idx")
            if "fail_reason" in res:
                updates["fail_reason"][i] = res["fail_reason"]
                fail_count += 1
                continue
            if bool(res.get("local_bkg_floored", False)):
                floored_count += 1
                raw_bkg = res.get("local_bkg_raw", np.nan)
                if np.isfinite(raw_bkg):
                    floored_values.append(float(raw_bkg))

            updates["maxPixel"][i] = res["maxPixel"]
            updates["maxPixel_err"][i] = res["maxPixel_err"]
            updates["area"][i] = res["area"]
            updates["counts_AP"][i] = res["counts_AP"]
            updates["flux_AP"][i] = res["flux_AP"]
            updates["flux_AP_err"][i] = res["flux_AP_err"]
            updates["sky_bkg_total"][i] = res["sky_bkg_total"]
            updates["sky_bkg_total_flux"][i] = res["sky_bkg_total_flux"]
            updates["noiseSky"][i] = res["noiseSky"]
            updates["threshold"][i] = res["threshold"]
            updates["SNR"][i] = res["SNR"]
            updates["bkg_std_method"][i] = res["bkg_std_method"]
            updates[mag_col][i] = res["mag"]
            updates[err_col][i] = res["mag_err"]

        # Single DataFrame assignment per column - avoids .at[] overhead.
        for col, vals in updates.items():
            sources[col] = vals

        if verbose >= 1 and fail_count > 0:
            # Collect fail reasons for diagnostic output
            fail_reasons = []
            for i in range(len(sources)):
                fr = updates.get("fail_reason", [""] * len(sources))[i]
                if fr:
                    fail_reasons.append(f"source #{i}: {fr}")
            # For single-source (target) fits, always show the reason
            if len(sources) == 1 and fail_reasons:
                logger.warning(
                    f"Target photometry failed: {fail_reasons[0]}"
                )
            else:
                logger.warning(
                    f"{fail_count}/{len(results)} sources failed. "
                    f"Fail reasons: {'; '.join(fail_reasons[:5])}"
                    + (f" ... ({len(fail_reasons)-5} more)" if len(fail_reasons) > 5 else "")
                )
        if verbose >= 1 and floored_count > 0:
            min_raw_bkg = (
                float(np.nanmin(floored_values))
                if len(floored_values) > 0
                else float("nan")
            )
            logger.warning(
                "Aperture photometry: floored negative local background to 0 for %d/%d sources (min raw local background=%g).",
                int(floored_count),
                int(len(results)),
                min_raw_bkg,
            )

        # ---- Per-source diagnostic plots -----------------------------------
        if plot:
            # When saveTarget=True and there are multiple target rows, produce
            # a single multi-target plot (1 row x N columns) instead of
            # individual per-source plots that would overwrite each other.
            _multi_target = (
                saveTarget
                and len(sources) > 1
                and "x_pix" in sources.columns
                and "y_pix" in sources.columns
            )
            if _multi_target:
                try:
                    _centers = []
                    _names = []
                    for i in range(len(sources)):
                        if sources.at[i, "fail_reason"]:
                            continue
                        _centers.append((
                            float(sources.at[i, "x_pix"]),
                            float(sources.at[i, "y_pix"]),
                        ))
                        # Use _additional_target_name if available, else generic
                        _nm = sources.at[i, "_additional_target_name"] if "_additional_target_name" in sources.columns else None
                        if not _nm or str(_nm) == "nan":
                            if i == 0:
                                _nm = self.input_yaml.get("target_name", "Main target")
                                # Use "Main target" for generic placeholder names
                                if _nm in ("Transient", "Center of Field", "Primary", None, "nan"):
                                    _nm = "Main target"
                            else:
                                _nm = f"Sub target {i}"
                        _names.append(str(_nm))
                    if _centers:
                        self._generate_multi_target_plot(
                            image=image_e,
                            target_centers=_centers,
                            target_names=_names,
                            ap_size=ap_size,
                            annulusIN=annulusIN,
                            annulusOUT=annulusOUT,
                            fwhm=fwhm,
                            error=error,
                            mask=mask,
                        )
                except Exception as exc:
                    logger.exception("Multi-target aperture plot failed: %s", exc)
                    # Fall back to per-source plots
                    _multi_target = False

            if not _multi_target:
                for i in range(len(sources)):
                    if sources.at[i, "fail_reason"]:
                        continue
                    try:
                        self._generate_plot(
                            image=image_e,
                            cutout_center=(
                                float(sources.at[i, "x_pix"]),
                                float(sources.at[i, "y_pix"]),
                            ),
                            ap_size=ap_size,
                            annulusIN=annulusIN,
                            annulusOUT=annulusOUT,
                            fwhm=fwhm,
                            saveTarget=saveTarget,
                            index=i,
                            error=error,
                            mask=mask,
                        )
                    except Exception as exc:
                        logger.exception("Plot failed for source %s: %s", i, exc)

        return sources

    # -----------------------------------------------------------------------
    # Multi-target diagnostic plot (1 row x N columns, one per target)
    # -----------------------------------------------------------------------

    def _generate_multi_target_plot(
        self,
        image,
        target_centers,
        target_names,
        ap_size,
        annulusIN,
        annulusOUT,
        fwhm,
        error=None,
        mask=None,
    ):
        """
        Multi-target diagnostic: 1 row x N columns, one column per target.

        Each column has the same 3-panel layout (main image + right profile +
        bottom profile) zoomed around that target.  All other target positions
        are marked with crosshairs on every panel so contamination is visible.

        Parameters
        ----------
        image            : 2-D array (electrons)
        target_centers   : list of (x, y) tuples in full-image coordinates
        target_names     : list of str labels (primary first)
        ap_size, annulusIN, annulusOUT, fwhm : aperture/annulus parameters
        error, mask      : optional 2-D arrays matching *image*
        """
        logger = logging.getLogger(__name__)
        plt.ioff()
        from plotting_utils import apply_autophot_mplstyle
        apply_autophot_mplstyle()

        fpath = self.input_yaml["fpath"]
        write_dir = os.path.dirname(fpath)
        base = os.path.splitext(os.path.basename(fpath))[0]

        n_targets = len(target_centers)

        zoom_bounds = []
        for cx, cy in target_centers:
            zoom_size = 1.25 * (annulusOUT + fwhm)
            x_min = max(0, int(np.floor(cx - zoom_size)))
            x_max = min(image.shape[1], int(np.ceil(cx + zoom_size)))
            y_min = max(0, int(np.floor(cy - zoom_size)))
            y_max = min(image.shape[0], int(np.ceil(cy + zoom_size)))
            zoom_bounds.append((x_min, x_max, y_min, y_max))

        # Size the figure from the cutout aspect so the equal-aspect
        # image panels keep square pixels; a fixed figsize would stretch
        # or shrink them inside mismatched cells. Each gridspec cell
        # holds the main panel plus the 20% side/bottom profiles the
        # divider carves out of it: cell = 1.2 * main + pad.
        ax_h = 2.8
        pad_in = 0.15
        main_w = [
            ax_h * (x1 - x0) / max(y1 - y0, 1)
            for (x0, x1, y0, y1) in zoom_bounds
        ]
        col_w = [1.2 * w + pad_in for w in main_w]
        cell_h = 1.2 * ax_h + pad_in
        gap_in = 0.60
        left_in, right_in = 0.62, 0.72
        bottom_in, top_in = 0.78, 0.45
        fig_w = left_in + sum(col_w) + gap_in * (n_targets - 1) + right_in
        fig_h = bottom_in + cell_h + top_in

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            1,
            n_targets,
            figure=fig,
            width_ratios=col_w,
            left=left_in / fig_w,
            right=1.0 - right_in / fig_w,
            bottom=bottom_in / fig_h,
            top=1.0 - top_in / fig_h,
            wspace=gap_in / (sum(col_w) / n_targets),
        )

        # Use a shared normalisation computed from ALL target zoom regions
        # so that panels remain comparable even when targets are far apart
        # and in different brightness regimes.
        _all_finite = []
        for _xb in zoom_bounds:
            _x0, _x1, _y0, _y1 = _xb
            _cut = image[_y0:_y1, _x0:_x1]
            _all_finite.append(_cut[np.isfinite(_cut)])
        _all_finite = np.concatenate(_all_finite) if _all_finite else np.array([])
        if _all_finite.size > 0:
            _med = float(np.nanmedian(_all_finite))
            _std = float(np.nanstd(_all_finite))
            if _std <= 0 or not np.isfinite(_std):
                _std = 1.0
            vmin_shared = _med - 3 * _std
            vmax_shared = _med + 3 * _std
        else:
            vmin_shared, vmax_shared = 0.0, 1.0

        plot_zero_as_nan = bool(
            (self.input_yaml.get("plotting") or {}).get("plot_zero_as_nan", True)
        )

        # Target marker colors: primary=white, others=muted teal (was neon cyan)
        _target_colors = ["white"] + ["#17A2B8"] * (n_targets - 1)

        for col_idx, (cx, cy) in enumerate(target_centers):
            x_min, x_max, y_min, y_max = zoom_bounds[col_idx]
            zoom_image = image[y_min:y_max, x_min:x_max]
            zoom_error = (
                error[y_min:y_max, x_min:x_max]
                if error is not None
                else np.zeros_like(zoom_image)
            )
            zoom_mask = (
                mask[y_min:y_max, x_min:x_max]
                if mask is not None
                else None
            )

            ax_main = fig.add_subplot(gs[0, col_idx])
            divider = make_axes_locatable(ax_main)
            ax_right = divider.append_axes("right", size="20%", pad=0.15, sharey=ax_main)
            ax_bottom = divider.append_axes("bottom", size="20%", pad=0.15, sharex=ax_main)

            ax_main.tick_params(axis="x", labelbottom=False)
            ax_right.tick_params(axis="y", labelleft=False)
            ax_right.yaxis.tick_right()
            ax_right.xaxis.tick_top()
            ax_main.xaxis.tick_top()
            ax_bottom.tick_params(axis="both", labelsize=8)
            ax_right.tick_params(axis="both", labelsize=8)
            ax_bottom.tick_params(axis="x", labelrotation=30)
            ax_right.tick_params(axis="x", labelrotation=30)

            _title = target_names[col_idx] if col_idx < len(target_names) else f"Target {col_idx}"
            ax_main.set_title(_title, fontsize=9, pad=4)

            # Guard: skip profiles for tiny regions
            if zoom_image.shape[0] < 5 or zoom_image.shape[1] < 5:
                logger.warning(
                    f"Zoom region too small for profile plotting: {zoom_image.shape}"
                )
                norm = ImageNormalize(zoom_image, interval=ZScaleInterval())
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad(color="none")
                zmask = ~np.isfinite(zoom_image)
                zoom_disp = np.ma.array(zoom_image, mask=zmask)
                ax_main.imshow(zoom_disp, origin="lower", norm=norm, cmap=cmap, aspect="equal")
                from plotting_utils import overlay_mask_hatch
                overlay_mask_hatch(ax_main, zmask)
                ax_main.set_xlim(0, zoom_image.shape[1])
                ax_main.set_ylim(0, zoom_image.shape[0])
                cx_local = cx - x_min
                cy_local = cy - y_min
                for radius, color, ls in [
                    (ap_size, "#00AA00", "-"),
                    (annulusIN, "#D94F4F", "--"),
                    (annulusOUT, "#D94F4F", "--"),
                ]:
                    ax_main.add_patch(
                        Circle((cx_local, cy_local), radius, ec=color, fc="none", lw=0.5, ls=ls)
                    )
                continue

            from plotting_utils import overlay_mask_hatch

            norm = ImageNormalize(vmin=vmin_shared, vmax=vmax_shared)
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad(color="none")
            zmask = ~np.isfinite(zoom_image)
            if plot_zero_as_nan:
                zmask |= (np.asarray(zoom_image, dtype=float) == 0.0)
            zoom_disp = np.ma.array(zoom_image, mask=zmask)

            ax_main.imshow(zoom_disp, origin="lower", norm=norm, cmap=cmap, aspect="equal")
            overlay_mask_hatch(ax_main, zmask)
            ax_main.set_xlim(0, zoom_image.shape[1])
            ax_main.set_ylim(0, zoom_image.shape[0])

            cx_local = cx - x_min
            cy_local = cy - y_min
            for radius, color, ls in [
                (ap_size, "#00AA00", "-"),
                (annulusIN, "#D94F4F", "--"),
                (annulusOUT, "#D94F4F", "--"),
            ]:
                ax_main.add_patch(
                    Circle(
                        (cx_local, cy_local),
                        radius,
                        ec=color,
                        fc="none",
                        lw=0.6 if color == "#D94F4F" else 0.5,
                        ls=ls,
                        zorder=5,
                    )
                )

            # Mark ALL other targets with crosshairs on this panel
            for _ti, (_tcx, _tcy) in enumerate(target_centers):
                if _ti == col_idx:
                    continue
                _tcx_local = _tcx - x_min
                _tcy_local = _tcy - y_min
                # Only mark if within the zoom region
                if 0 <= _tcx_local < zoom_image.shape[1] and 0 <= _tcy_local < zoom_image.shape[0]:
                    _r = fwhm / 2.0
                    ax_main.plot(
                        [_tcx_local - _r, _tcx_local + _r],
                        [_tcy_local, _tcy_local],
                        color=_target_colors[_ti], lw=1.0, alpha=0.8, zorder=10,
                    )
                    ax_main.plot(
                        [_tcx_local, _tcx_local],
                        [_tcy_local - _r, _tcy_local + _r],
                        color=_target_colors[_ti], lw=1.0, alpha=0.8, zorder=10,
                    )
                    _tname = target_names[_ti] if _ti < len(target_names) else f"T{_ti}"
                    ax_main.annotate(
                        _tname, (_tcx_local, _tcy_local + _r + 2),
                        color=_target_colors[_ti], fontsize=6, ha="center", zorder=11,
                    )

            kw = dict(ls=":", color="white", lw=0.5, alpha=0.7)
            ax_main.axvline(cx_local, **kw)
            ax_main.axhline(cy_local, **kw)
            ax_bottom.axvline(cx_local, **kw)
            ax_right.axhline(cy_local, **kw)

            finite = np.isfinite(zoom_image)
            if zoom_mask is not None:
                finite &= ~np.asarray(zoom_mask, dtype=bool)
            # Exclude mask-flagged pixels from the mean (matching the
            # finite-pixel error counts below).
            masked_image = np.where(finite, zoom_image, np.nan)
            hx = np.nanmean(masked_image, axis=0)
            hy = np.nanmean(masked_image, axis=1)
            err2 = np.nan_to_num(
                np.asarray(zoom_error, dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0,
            ) ** 2
            err2 = np.where(finite, err2, 0.0)
            n_col = np.sum(finite, axis=0).astype(float)
            n_row = np.sum(finite, axis=1).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                hx_err = np.sqrt(np.sum(err2, axis=0)) / np.where(n_col > 0, n_col, np.nan)
                hy_err = np.sqrt(np.sum(err2, axis=1)) / np.where(n_row > 0, n_row, np.nan)

            x_range = np.arange(0, zoom_image.shape[1])
            y_range = np.arange(0, zoom_image.shape[0])

            kw_step = dict(color="dodgerblue", where="mid", lw=0.5, marker=None, markersize=0)
            ax_bottom.step(x_range, hx, **kw_step)
            ax_bottom.fill_between(
                x_range, hx - hx_err, hx + hx_err,
                color="dodgerblue", alpha=0.5, step="mid",
            )
            ax_bottom.plot(x_range, hx - hx_err, color="dodgerblue", lw=0.3, alpha=0.7, drawstyle="steps-mid")
            ax_bottom.plot(x_range, hx + hx_err, color="dodgerblue", lw=0.3, alpha=0.7, drawstyle="steps-mid")

            n_y = len(y_range)
            if n_y > 0:
                y_edges_r = np.empty(2 * n_y)
                for i in range(n_y):
                    y_edges_r[2 * i] = i - 0.5 if i > 0 else 0
                    y_edges_r[2 * i + 1] = i + 0.5 if i < n_y - 1 else (n_y - 1)
                ax_right.plot(np.repeat(hy, 2), y_edges_r, color="dodgerblue", lw=0.5)
                ax_right.fill_betweenx(
                    y_range, hy - hy_err, hy + hy_err,
                    color="dodgerblue", alpha=0.5, step="mid"
                )
                ax_right.plot(np.repeat(hy - hy_err, 2), y_edges_r, color="dodgerblue", lw=0.3, alpha=0.7)
                ax_right.plot(np.repeat(hy + hy_err, 2), y_edges_r, color="dodgerblue", lw=0.3, alpha=0.7)

            try:
                ann = CircularAnnulus(
                    (cx, cy), r_in=float(annulusIN), r_out=float(annulusOUT)
                )
                ann_mask = ann.to_mask(method="center")
                bkg_pix = ann_mask.get_values(image)
                bkg_pix = bkg_pix[np.isfinite(bkg_pix)]
                if bkg_pix.size > 0:
                    bkg_level = float(np.median(bkg_pix))
                    kw_bkg = dict(color="#FFA500", lw=0.9, ls="--", alpha=0.95)
                    ax_bottom.axhline(bkg_level, **kw_bkg, marker=None)
                    ax_right.axvline(bkg_level, **kw_bkg, marker=None)
            except Exception:
                pass

            ax_bottom.set_xlabel("X position (pixels)")
            ax_bottom.set_ylabel("Flux [e$^-$]")
            ax_right.set_xlabel("Flux [e$^-$]")
            ax_right.yaxis.set_label_position("right")
            ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
            ax_right.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))

            # Add legend to the first panel only
            if col_idx == 0:
                from matplotlib.lines import Line2D as _Line2D
                from matplotlib.patches import Patch as _Patch
                _legend_handles = [
                    _Patch(facecolor="none", edgecolor="#00AA00", lw=0.5, label="Aperture"),
                    _Patch(facecolor="none", edgecolor="#D94F4F", lw=0.5, ls="--", label="Annulus"),
                ]
                for _ti, _tname in enumerate(target_names):
                    _mc = _target_colors[_ti] if _ti < len(_target_colors) else "#17A2B8"
                    _legend_handles.append(
                        _Line2D([0], [0], marker="x", color=_mc,
                                markersize=6, markeredgewidth=1.5,
                                linestyle="None", label=_tname)
                    )
                _n_leg = len(_legend_handles)
                _leg_ncol = 3 if _n_leg >= 8 else (2 if _n_leg >= 5 else 1)
                ax_main.legend(
                    handles=_legend_handles, loc="upper left",
                    frameon=False, fontsize=7, ncol=_leg_ncol,
                )

        from plotting_utils import get_plot_ext
        save_name = f"Aperture_Target_{base}{get_plot_ext(self.input_yaml)}"
        save_name = os.path.join(write_dir, save_name)
        fig.savefig(save_name, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)
        logger.info(
            "Saved multi-target aperture plot: %s (%d targets)",
            os.path.basename(save_name), n_targets,
        )

    # -----------------------------------------------------------------------
    # Diagnostic plot
    # -----------------------------------------------------------------------

    def _generate_plot(
        self,
        image,
        cutout_center,
        ap_size,
        annulusIN,
        annulusOUT,
        fwhm,
        saveTarget,
        index,
        error=None,
        mask=None,
    ):
        """
        Three-panel diagnostic: main image + right / bottom flux profiles.
        """
        logger = logging.getLogger(__name__)
        plt.ioff()
        from plotting_utils import apply_autophot_mplstyle
        apply_autophot_mplstyle()

        fpath = self.input_yaml["fpath"]
        write_dir = os.path.dirname(fpath)
        base = os.path.splitext(os.path.basename(fpath))[0]

        cx, cy = cutout_center

        # FIX 3 & 5: compute zoom bounds arithmetically, not from axes state
        zoom_size = 1.25 * (annulusOUT + fwhm)
        x_min = max(0, int(np.floor(cx - zoom_size)))
        x_max = min(image.shape[1], int(np.ceil(cx + zoom_size)))
        y_min = max(0, int(np.floor(cy - zoom_size)))
        y_max = min(image.shape[0], int(np.ceil(cy + zoom_size)))

        zoom_image = image[y_min:y_max, x_min:x_max]
        zoom_error = (
            error[y_min:y_max, x_min:x_max]
            if error is not None
            else np.zeros_like(zoom_image)
        )
        zoom_mask = (
            mask[y_min:y_max, x_min:x_max]
            if mask is not None
            else None
        )

        # Size the figure from the cutout aspect so the equal-aspect
        # image panel keeps square pixels; the divider carves the 20%
        # side/bottom profiles out of the main cell (cell = 1.2 * main
        # + pad), and the margins hold their tick labels.
        zoom_h_px, zoom_w_px = zoom_image.shape
        ax_h = 3.0
        ax_w = ax_h * (zoom_w_px / max(zoom_h_px, 1))
        pad_in = 0.15
        cell_w = 1.2 * ax_w + pad_in
        cell_h = 1.2 * ax_h + pad_in
        left_in, right_in = 0.62, 0.78
        bottom_in, top_in = 0.85, 0.55
        fig_w = left_in + cell_w + right_in
        fig_h = bottom_in + cell_h + top_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax_main = fig.add_axes(
            [
                left_in / fig_w,
                bottom_in / fig_h,
                cell_w / fig_w,
                cell_h / fig_h,
            ]
        )
        divider = make_axes_locatable(ax_main)
        ax_right = divider.append_axes("right", size="20%", pad=0.15, sharey=ax_main)
        ax_bottom = divider.append_axes("bottom", size="20%", pad=0.15, sharex=ax_main)

        ax_main.tick_params(axis="x", labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.yaxis.tick_right()
        ax_right.xaxis.tick_top()
        ax_main.xaxis.tick_top()
        ax_bottom.tick_params(axis="both", labelsize=8)
        ax_right.tick_params(axis="both", labelsize=8)
        ax_bottom.tick_params(axis="x", labelrotation=30)
        ax_right.tick_params(axis="x", labelrotation=30)

        # Guard: skip profiles for tiny regions
        if zoom_image.shape[0] < 5 or zoom_image.shape[1] < 5:
            logger.warning(
                f"Zoom region too small for profile plotting: {zoom_image.shape}"
            )
            norm = ImageNormalize(zoom_image, interval=ZScaleInterval())
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad(color="none")
            # Use only hardware mask (NaN/inf pixels) for plotting - don't mask out zero-valued pixels
            zmask = ~np.isfinite(zoom_image)
            zoom_disp = np.ma.array(zoom_image, mask=zmask)
            # FIX 1: use cutout-local coordinates for simpler alignment
            ax_main.imshow(
                zoom_disp,
                origin="lower",
                norm=norm,
                cmap=cmap,
                aspect="equal",
            )
            from plotting_utils import overlay_mask_hatch
            overlay_mask_hatch(ax_main, zmask)
            ax_main.set_xlim(0, zoom_image.shape[1])
            ax_main.set_ylim(0, zoom_image.shape[0])

            # Convert full-image coordinates to cutout-local coordinates
            cx_local = cx - x_min
            cy_local = cy - y_min

            for radius, color, ls in [
                (ap_size, "#00AA00", "-"),
                (annulusIN, "#D94F4F", "--"),
                (annulusOUT, "#D94F4F", "--"),
            ]:
                ax_main.add_patch(
                    Circle((cx_local, cy_local), radius, ec=color, fc="none", lw=0.5, ls=ls)
                )
            from matplotlib.patches import Patch
            fig.legend(
                handles=[
                    Patch(facecolor="none", edgecolor="#00AA00", lw=0.8, label="Aperture"),
                    Patch(facecolor="none", edgecolor="#D94F4F", lw=0.8, ls="--", label="Annulus"),
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=2,
                fontsize=8,
                frameon=False,
            )
            label = base if saveTarget else index
            from plotting_utils import get_plot_ext
            _ext = get_plot_ext(self.input_yaml)
            save_name = (
                f"Aperture_Target_{base}{_ext}" if saveTarget
                else f"Aperture_Source_{label}{_ext}"
            )
            save_name = os.path.join(write_dir, save_name)
            fig.savefig(save_name, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close(fig)
            return

        from plotting_utils import overlay_mask_hatch

        norm = ImageNormalize(zoom_image, interval=ZScaleInterval())
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="none")
        plot_zero_as_nan = bool(
            (self.input_yaml.get("plotting") or {}).get("plot_zero_as_nan", True)
        )
        zmask = ~np.isfinite(zoom_image)
        if plot_zero_as_nan:
            zmask |= (np.asarray(zoom_image, dtype=float) == 0.0)
        zoom_disp = np.ma.array(zoom_image, mask=zmask)

        # FIX 1: render zoom cutout without extent for simpler alignment
        ax_main.imshow(
            zoom_disp,
            origin="lower",
            norm=norm,
            cmap=cmap,
            aspect="equal",
        )
        overlay_mask_hatch(ax_main, zmask)
        ax_main.set_xlim(0, zoom_image.shape[1])
        ax_main.set_ylim(0, zoom_image.shape[0])

        # Convert full-image coordinates to cutout-local coordinates
        cx_local = cx - x_min
        cy_local = cy - y_min

        for radius, color, ls in [
            (ap_size, "#00AA00", "-"),
            (annulusIN, "#D94F4F", "--"),
            (annulusOUT, "#D94F4F", "--"),
        ]:
            ax_main.add_patch(
                Circle(
                    (cx_local, cy_local),
                    radius,
                    ec=color,
                    fc="none",
                    lw=0.6 if color == "#D94F4F" else 0.5,
                    ls=ls,
                    zorder=5,
                )
            )

        kw = dict(ls=":", color="white", lw=0.5, alpha=0.7)
        ax_main.axvline(cx_local, **kw)
        ax_main.axhline(cy_local, **kw)
        ax_bottom.axvline(cx_local, **kw)
        ax_right.axhline(cy_local, **kw)

        # Profiles: NaNs should remain NaNs in the image, but projections should
        # still be well-defined on finite pixels. Use finite counts per row/col
        # to compute mean profiles and propagate uncertainty consistently.
        finite = np.isfinite(zoom_image)
        if zoom_mask is not None:
            finite &= ~np.asarray(zoom_mask, dtype=bool)
        # Exclude mask-flagged pixels from the mean so they cannot bias the
        # profile (matching the finite-pixel error counts below).
        masked_image = np.where(finite, zoom_image, np.nan)
        hx = np.nanmean(masked_image, axis=0)
        hy = np.nanmean(masked_image, axis=1)

        # Variance of the mean profile: Var(mean) = sum(sigma_i^2) / N^2 for finite pixels.
        # Use the image finite mask so masked/no-data pixels do not dilute uncertainties.
        err2 = np.nan_to_num(
            np.asarray(zoom_error, dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        ) ** 2
        err2 = np.where(finite, err2, 0.0)
        n_col = np.sum(finite, axis=0).astype(float)
        n_row = np.sum(finite, axis=1).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            hx_err = np.sqrt(np.sum(err2, axis=0)) / np.where(n_col > 0, n_col, np.nan)
            hy_err = np.sqrt(np.sum(err2, axis=1)) / np.where(n_row > 0, n_row, np.nan)

        # Coordinate arrays in cutout-local coordinates
        x_range = np.arange(0, zoom_image.shape[1])
        y_range = np.arange(0, zoom_image.shape[0])

        # Some matplotlib styles set a default marker for lines; explicitly
        # disable markers so projections remain clean.
        kw_step = dict(color="dodgerblue", where="mid", lw=0.5, marker=None, markersize=0)
        ax_bottom.step(x_range, hx, **kw_step)
        ax_bottom.fill_between(
            x_range, hx - hx_err, hx + hx_err,
            color="dodgerblue", alpha=0.5, step="mid",
        )
        ax_bottom.plot(x_range, hx - hx_err, color="dodgerblue", lw=0.3, alpha=0.7, drawstyle="steps-mid")
        ax_bottom.plot(x_range, hx + hx_err, color="dodgerblue", lw=0.3, alpha=0.7, drawstyle="steps-mid")

        # Right panel: step() and plot(drawstyle="steps-mid") step along the
        # x-axis (value), but we need stepping along y-axis (coordinate) to
        # match fill_betweenx(step="mid").  Manually construct horizontal step
        # edges so the profile and error envelope align with the fill.
        n_y = len(y_range)
        if n_y > 0:
            y_edges_r = np.empty(2 * n_y)
            for i in range(n_y):
                y_edges_r[2 * i] = i - 0.5 if i > 0 else 0
                y_edges_r[2 * i + 1] = i + 0.5 if i < n_y - 1 else (n_y - 1)
            ax_right.plot(np.repeat(hy, 2), y_edges_r, color="dodgerblue", lw=0.5)
            ax_right.fill_betweenx(
                y_range, hy - hy_err, hy + hy_err,
                color="dodgerblue", alpha=0.5, step="mid"
            )
            ax_right.plot(np.repeat(hy - hy_err, 2), y_edges_r, color="dodgerblue", lw=0.3, alpha=0.7)
            ax_right.plot(np.repeat(hy + hy_err, 2), y_edges_r, color="dodgerblue", lw=0.3, alpha=0.7)

        bias_applied = False
        try:
            ann = CircularAnnulus(
                (cx, cy), r_in=float(annulusIN), r_out=float(annulusOUT)
            )
            ann_mask = ann.to_mask(method="center")
            bkg_pix = ann_mask.get_values(image)
            bkg_pix = bkg_pix[np.isfinite(bkg_pix)]
            if bkg_pix.size > 0:
                bkg_level_raw = float(np.median(bkg_pix))
                enforce_nn = bool(
                    (self.input_yaml.get("photometry") or {}).get(
                        "enforce_nonnegative_local_background", False
                    )
                )
                bkg_level_used = (
                    max(bkg_level_raw, 0.0)
                    if (enforce_nn and np.isfinite(bkg_level_raw))
                    else bkg_level_raw
                )
                kw_bkg = dict(color="#FFA500", lw=0.9, ls="--", alpha=0.95)
                ax_bottom.axhline(bkg_level_used, **kw_bkg, marker=None)
                ax_right.axvline(bkg_level_used, **kw_bkg, marker=None)
                if enforce_nn and np.isfinite(bkg_level_raw) and bkg_level_raw < 0:
                    bias_applied = True
                    kw_bkg_raw = dict(color="#FFA500", lw=0.7, ls=":", alpha=0.8)
                    ax_bottom.axhline(bkg_level_raw, **kw_bkg_raw, marker=None)
                    ax_right.axvline(bkg_level_raw, **kw_bkg_raw, marker=None)
        except Exception:
            pass

        ax_bottom.set_xlabel("X position (pixels)")
        ylabel = "Flux [e$^-$] + BIAS" if bias_applied else "Flux [e$^-$]"
        ax_bottom.set_ylabel(ylabel)
        ax_right.set_xlabel(ylabel)
        ax_right.yaxis.set_label_position("right")
        ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
        ax_right.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))

        from matplotlib.patches import Patch
        fig.legend(
            handles=[
                Patch(facecolor="none", edgecolor="#00AA00", lw=0.8, label="Aperture"),
                Patch(facecolor="none", edgecolor="#D94F4F", lw=0.8, ls="--", label="Annulus"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

        label = base if saveTarget else index
        from plotting_utils import get_plot_ext
        _ext = get_plot_ext(self.input_yaml)
        save_name = (
            f"Aperture_Target_{base}{_ext}" if saveTarget
            else f"Aperture_Source_{label}{_ext}"
        )
        save_name = os.path.join(write_dir, save_name)
        fig.savefig(save_name, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def interpolate_nans(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate (and extrapolate) NaN values in *y*.

        Parameters
        ----------
        x, y : array-like

        Returns
        -------
        y_interp : ndarray
        """
        x = np.asarray(x)
        y = np.asarray(y, dtype=float)

        nan_mask = np.isnan(y)
        # BUG FIX: original used `len(nan_indices) == 0` which checks the
        # *length* of the boolean array (always == len(y)), never the count
        # of True entries.  Fixed to `.any()`.
        if not nan_mask.any():
            return y
        # interp1d(kind="linear") requires at least two valid points.
        if int(np.count_nonzero(~nan_mask)) < 2:
            return y

        interp = interp1d(
            x[~nan_mask], y[~nan_mask], kind="linear", fill_value="extrapolate"
        )
        y_out = y.copy()
        y_out[nan_mask] = interp(x[nan_mask])
        return y_out

    # -----------------------------------------------------------------------
    # Optimum aperture radius
    # -----------------------------------------------------------------------

    def measure_optimum_radius(
        self,
        sources: pd.DataFrame,
        plot: bool = True,
        norm_factor: float = 0.8,
        aperture_norm_factor: float = 0.95,
        sigma: float = 5.0,
        background_rms: np.ndarray = None,
        max_radius: float = 3.5,
        stability_threshold: float = 0.2,
        max_tail_excess: float = 0.8,
        min_tail_flux: float = 0.8,
        fwhm_uncertainty: float = 0.5,
        n_jobs: int = None,
        crowded: bool = False,
        mask: np.ndarray = None,
    ):
        """
        Data-driven selection of the optimal aperture radius.

        Selection logic
        ---------------
        1. Per-source optimum radii via CoG + SNR guard.
        2. Preliminary global radius = sigma-clipped median of per-source radii.
        3. Concentration gate: require a minimum enclosed-flux fraction at a
           fixed core radius (default 1.7 * FWHM) to reject very broad sources.
        3. Global stability (adaptive): primarily reject contaminated tails
           (profile > 1 + excess) beyond the preliminary global radius.
           Broad but monotonic profiles are retained.
        4. After final optimum_radius is set: apply a gentle final tail screen,
           then keep a minimum number/fraction of best-behaved stars.
           CoG is normalized 0->1 at aperture radius = max_radius (default 3.5 FWHM).
        5. Final global radius = sigma-clipped median of stable sources.
        6. Tail check w.r.t. final optimum (min_tail_flux / max_tail_excess).
        7. Optional EE model refinement at *aperture_norm_factor*.
        8. optimum_scale = ceil(optimum_radius * fwhm) + 0.5

        Returns
        -------
        (filtered_sources, optimum_radius, optimum_scale)
        """
        logger = logging.getLogger(__name__)
        phot_cfg = self.input_yaml.get("photometry", {}) or {}

        if not {"x_pix", "y_pix"}.issubset(sources.columns):
            raise ValueError(
                "sources must contain 'x_pix' and 'y_pix' columns."
            )

        # Crowded fields: use a fixed aperture radius of ~1.5 FWHM; a
        # data-driven radius search is unstable or fails outright in very
        # dense regions.  This radius is in FWHM units and can be overridden
        # by `photometry.crowded_optimum_radius_fwhm` in the config.
        if crowded:
            fixed_radius = float(phot_cfg.get("crowded_optimum_radius_fwhm", 1.5))
            fwhm = float(self.input_yaml["fwhm"])
            optimum_radius = fixed_radius
            optimum_scale = max(7, int(np.ceil(fixed_radius * fwhm))) + 0.5
            logger.info(
                "Crowded field: skipping data-driven optimum-radius search; using fixed aperture radius of %.2f FWHM (%.2f pixels).",
                optimum_radius,
                optimum_radius * fwhm,
            )
            return sources, optimum_radius, optimum_scale

        # Sparse/normal fields: use data-driven optimisation with a slightly
        # larger default fallback radius.
        fallback_radius = 1.7
        optimum_radius = fallback_radius
        optimum_scale = (
            max(7, int(np.ceil(fallback_radius * self.input_yaml["fwhm"]))) + 0.5
        )

        # ---- SNR pre-filter (slightly relaxed for crowded) ------------------------------------------------
        # Skip entirely when no SNR column is available (e.g. raw finder
        # tables passed from limits.py) rather than raising KeyError.
        snr_min = 3.0 if crowded else 5.0
        if "SNR" in sources.columns:
            sources = sources[(sources["SNR"] > snr_min) & (sources["SNR"] < 10000)].copy()
        else:
            logger.info("No SNR column; skipping optimum-radius SNR pre-filter.")
        sources.reset_index(inplace=True)
        n_sources = len(sources)

        if n_sources == 0:
            logger.warning("No sources passed SNR cut. Using default radius/scale.")
            return sources, optimum_radius, optimum_scale

        fwhm = float(self.input_yaml["fwhm"])
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError(f"Invalid FWHM={fwhm}; expected finite and >0.")
        radii_fwhm = np.arange(0.05, max_radius + 1e-9, 0.1)
        radii = radii_fwhm * fwhm
        logger.info(log_step(f"Optimum aperture: {n_sources} sources"))

        sources["optimum_radius"] = np.nan
        sources["mean_slope"] = np.nan
        sources["tail_max_dist_local"] = np.nan
        sources["tail_max_dist_global"] = np.nan
        sources["tail_excess_global"] = np.nan
        sources["stable_beyond_global"] = False
        sources["local_env_std"] = np.nan
        sources["enc_flux_core"] = np.nan
        sources["tail_outer_std"] = np.nan
        sources["tail_outer_slope"] = np.nan

        gain = resolve_gain_e_per_adu(None, self.input_yaml)
        if background_rms is not None:
            # Replace NaN values (chip gaps) with median of finite values.
            _bkg_rms = np.asarray(background_rms, dtype=float)
            if np.any(np.isnan(_bkg_rms)):
                _finite_median = float(np.nanmedian(_bkg_rms))
                if not np.isfinite(_finite_median) or _finite_median <= 0:
                    _finite_median = 1.0
                _bkg_rms = np.where(np.isfinite(_bkg_rms), _bkg_rms, _finite_median)
            _bkg_rms = np.abs(_bkg_rms)
            # Convert both data and bkg_error to electrons so total_error is in e-.
            # Matches the convention used in Aperture.measure() (image_e = image*gain,
            # bkg_error = background_rms_ADU * gain, effective_gain=1).
            _image_e_opt = np.where(
                np.isfinite(self.image), self.image * gain, np.nan
            )
            error = calc_total_error(
                _image_e_opt, _bkg_rms * gain, effective_gain=1
            )
            # The CoG workers run on self.image (ADU), so convert the error
            # map back to ADU: profile_error is normalised by a flux in ADU
            # units and must carry matching units for the SNR guard.
            error = error / gain
        else:
            error = None
        n_jobs = _resolve_n_jobs(n_jobs, half_cpus=False)
        use_moffat_cog = bool(phot_cfg.get("optimum_radius_use_moffat_cog", False))
        moffat_beta = float(
            phot_cfg.get(
                "optimum_radius_moffat_beta",
                phot_cfg.get("psf_init_moffat_beta", 4.765),
            )
        )
        logger.info(
            "Optimum-radius CoG mode: %s (moffat_beta=%g)",
            "moffat" if use_moffat_cog else "empirical",
            moffat_beta,
        )

        # ---- Parallel CoG analysis -----------------------------------------
        if n_sources < NSOURCES:
            args_list = [
                (
                    idx,
                    row["x_pix"],
                    row["y_pix"],
                    fwhm,
                    radii,
                    self.image,
                    error,
                    norm_factor,
                    stability_threshold,
                    use_moffat_cog,
                    moffat_beta,
                    mask,
                )
                for idx, row in sources.iterrows()
            ]
            results = [_optimum_radius_worker(a) for a in args_list]
        else:
            # Broadcast image/error/radii once per worker; only (idx, x, y) is
            # pickled per task.
            with Pool(
                processes=n_jobs,
                initializer=_optimum_radius_init_shared,
                initargs=(
                    fwhm,
                    radii,
                    self.image,
                    error,
                    norm_factor,
                    stability_threshold,
                    use_moffat_cog,
                    moffat_beta,
                    mask,
                ),
            ) as pool:
                results = pool.map(
                    _optimum_radius_worker_shared,
                    [
                        (idx, row["x_pix"], row["y_pix"])
                        for idx, row in sources.iterrows()
                    ],
                )

        # ---- Collect results -----------------------------------------------
        # KEY OPTIMISATION: profiles are already in `results`; no second Pool
        # needed for the plot pass.  Store them here for reuse.
        profiles_map: dict[int, np.ndarray] = {}
        for res in results:
            if res is None:
                continue
            idx = res["idx"]
            sources.at[idx, "optimum_radius"] = res["optimum_radius"]
            sources.at[idx, "mean_slope"] = res["mean_slope"]
            sources.at[idx, "tail_max_dist_local"] = res["tail_max_dist_local"]
            sources.at[idx, "local_env_std"] = res.get("local_env_std", np.nan)
            profiles_map[idx] = res["profile"]

        # ---- Preliminary global radius (no stability cut yet) --------------
        prelim_mask = np.isfinite(sources["optimum_radius"].values) & np.isfinite(
            sources["mean_slope"].values
        )
        if not prelim_mask.any():
            logger.warning("No valid sources for global radius. Using default.")
            return sources.iloc[[]], optimum_radius, optimum_scale

        prelim_r = sources.loc[prelim_mask, "optimum_radius"].to_numpy(float)
        prelim_s = sources.loc[prelim_mask, "mean_slope"].to_numpy(float)

        try:
            cr = sigma_clip(prelim_r, sigma=sigma, stdfunc=mad_std)
            cs = sigma_clip(prelim_s, sigma=sigma, stdfunc=mad_std)
            prelim_keep = ~np.asanyarray(cr.mask) & ~np.asanyarray(cs.mask)
        except Exception:
            prelim_keep = np.ones(len(prelim_r), dtype=bool)

        if not prelim_keep.any():
            logger.warning(
                "Preliminary sigma-clip rejected all sources. Using default."
            )
            return sources.iloc[[]], fallback_radius, optimum_scale

        global_optimum_pre = float(np.nanmedian(prelim_r[prelim_keep]))

        # ---- Global stability filter (tail w.r.t. preliminary radius) -------
        # Key change: avoid rejecting intrinsically broad stars just because
        # their profile is still <1 beyond the preliminary global radius.
        # Use one-sided tail-excess checks (contamination proxy), with adaptive
        # tolerances based on the current epoch's source distribution.
        beyond_global = (radii / fwhm) > global_optimum_pre
        finite_tail_excess = []
        for idx in range(len(sources)):
            prof = profiles_map.get(idx)
            if prof is None or not beyond_global.any():
                continue
            tail_region = prof[beyond_global]
            tmd = float(np.nanmax(np.abs(tail_region - 1.0)))
            tail_excess_global = float(max(0.0, np.nanmax(tail_region) - 1.0))
            sources.at[idx, "tail_max_dist_global"] = tmd
            sources.at[idx, "tail_excess_global"] = tail_excess_global
            if np.isfinite(tail_excess_global):
                finite_tail_excess.append(tail_excess_global)

        # Adaptive upper tolerance: if this epoch is noisier/crowded, avoid
        # over-pruning while still rejecting clear positive-tail contamination.
        if len(finite_tail_excess) > 0:
            tail_excess_arr = np.asarray(finite_tail_excess, float)
            adaptive_tail_excess = float(
                np.nanpercentile(tail_excess_arr, 90) + 0.05
            )
            tail_excess_limit = max(max_tail_excess, adaptive_tail_excess)
        else:
            tail_excess_limit = max_tail_excess

        for idx in range(len(sources)):
            te = float(sources.at[idx, "tail_excess_global"])
            ms = float(sources.at[idx, "mean_slope"])
            if np.isfinite(te) and np.isfinite(ms) and ms > 0 and te <= tail_excess_limit:
                sources.at[idx, "stable_beyond_global"] = True

        stable_mask = sources["stable_beyond_global"].values & np.isfinite(
            sources["optimum_radius"].values
        )
        # Concentration gate to reject very broad sources:
        # require a minimum enclosed-flux fraction within a fixed core radius.
        core_radius_fwhm = float(
            phot_cfg.get("optimum_radius_core_radius_fwhm", 1.7)
        )
        core_flux_min = float(phot_cfg.get("optimum_radius_core_flux_min", 0.5))
        core_radius_fwhm = max(0.5, min(float(max_radius), core_radius_fwhm))
        core_flux_min = max(0.05, min(0.95, core_flux_min))
        radii_fwhm_arr = radii / fwhm
        core_pass_mask = np.zeros(len(sources), dtype=bool)
        for idx in range(len(sources)):
            prof = profiles_map.get(idx)
            if prof is None:
                continue
            try:
                enc_core = float(
                    np.interp(
                        core_radius_fwhm,
                        radii_fwhm_arr,
                        np.asarray(prof, float),
                    )
                )
            except Exception:
                enc_core = np.nan
            sources.at[idx, "enc_flux_core"] = enc_core
            if np.isfinite(enc_core) and enc_core >= core_flux_min:
                core_pass_mask[idx] = True
        stable_mask &= core_pass_mask
        n_core_rej = int(np.count_nonzero(~core_pass_mask & np.isfinite(sources["optimum_radius"].values)))
        if n_core_rej > 0:
            logger.info(
                "Concentration gate rejected %d broad sources (enc_flux(%.2f*FWHM) < %.2f).",
                n_core_rej,
                core_radius_fwhm,
                core_flux_min,
            )

        # Ensure a minimum stable pool: rescue high-SNR, near-global-radius stars
        # when adaptive stability is still too strict.
        min_keep_abs = int(phot_cfg.get("optimum_radius_min_keep_abs", 8))
        min_keep_frac = float(phot_cfg.get("optimum_radius_min_keep_frac", 0.25))
        min_keep_abs = max(3, min_keep_abs)
        min_keep_frac = max(0.05, min(0.9, min_keep_frac))
        min_keep = min(n_sources, max(min_keep_abs, int(np.ceil(min_keep_frac * n_sources))))

        if np.count_nonzero(stable_mask) < min_keep:
            candidates = sources[
                np.isfinite(sources["optimum_radius"].values)
                & np.isfinite(sources["mean_slope"].values)
                & core_pass_mask
            ].copy()
            if not candidates.empty:
                # Prefer high SNR and radii close to preliminary global value.
                dr = np.abs(candidates["optimum_radius"].values - global_optimum_pre)
                snr_vals = np.asarray(candidates.get("SNR", pd.Series(np.ones(len(candidates)))), float)
                snr_rank = np.argsort(np.argsort(-snr_vals))
                dr_rank = np.argsort(np.argsort(dr))
                score = snr_rank + dr_rank
                candidates["_score"] = score
                rescue_idx = (
                    candidates.sort_values("_score")
                    .head(min_keep)
                    .index.values
                )
                stable_mask[rescue_idx] = True
                logger.info(
                    "Adaptive rescue kept %d sources for optimum-radius stability (target minimum=%d).",
                    int(np.count_nonzero(stable_mask)),
                    min_keep,
                )

        if not stable_mask.any():
            # Crowded fallback: use median of all preliminary radii (clipped) instead of failing
            if crowded and prelim_mask.any():
                r_clip = np.clip(prelim_r, 0.5, 2.0)
                fallback_from_data = float(np.nanmedian(r_clip))
                if np.isfinite(fallback_from_data):
                    logger.warning(
                        "No globally stable sources; using median of preliminary radii (crowded): %.2f FWHM",
                        fallback_from_data,
                    )
                    optimum_scale = (
                        max(
                            12,
                            int(np.ceil(fallback_from_data * self.input_yaml["fwhm"])),
                        )
                        + 0.5
                    )
                    return sources.iloc[[]], fallback_from_data, optimum_scale
            logger.warning("No globally stable sources. Using default.")
            return sources.iloc[[]], fallback_radius, optimum_scale

        # Use all globally stable sources (as many as possible) with a mild
        # sanity cut on radius; no additional behaviour-score culling.
        opt_r_arr = sources.loc[stable_mask, "optimum_radius"].to_numpy(float)
        slopes_arr = sources.loc[stable_mask, "mean_slope"].to_numpy(float)
        kept_indices = np.where(stable_mask)[0]
        profiles = np.array(
            [profiles_map[i] for i in kept_indices if i in profiles_map], dtype=float
        )

        # Keep only reasonable radii but otherwise retain the full stable set.
        min_r_ok = 0.5
        max_r_ok = max_radius
        if len(opt_r_arr) > 0:
            radius_ok = (opt_r_arr >= min_r_ok) & (opt_r_arr <= max_r_ok)
            if radius_ok.any():
                opt_r_arr = opt_r_arr[radius_ok]
                slopes_arr = slopes_arr[radius_ok]
                kept_indices = kept_indices[radius_ok]
                if profiles.size > 0:
                    profiles = profiles[radius_ok]

        logger.info(
            f"Using {len(kept_indices)} sources with stable profiles "
            f"for optimum-radius and PSF selection."
        )

        # ---- Final sigma-clipping ------------------------------------------
        # Gentle clip: only drop extreme radius outliers, keep almost all stable profiles.
        sigma_radius = max(sigma, 10.0)
        try:
            cr_f = sigma_clip(opt_r_arr, sigma=sigma_radius, stdfunc=mad_std)
            final_mask = ~np.asanyarray(cr_f.mask)
        except Exception:
            final_mask = np.ones(len(opt_r_arr), dtype=bool)

        if final_mask.any():
            final_indices = kept_indices[final_mask]
            filtered_sources = sources.iloc[final_indices].copy()
            optimum_radius = float(np.nanmedian(opt_r_arr[final_mask]))
        else:
            logger.warning(
                "Final sigma-clip rejected all stable sources. Using default."
            )
            return sources.iloc[[]], fallback_radius, optimum_scale

        if len(final_indices) < len(kept_indices):
            logger.info(
                "Radius sanity + gentle sigma-clip: %d -> %d sources.",
                len(kept_indices),
                len(final_indices),
            )

        # ---- Tail check w.r.t. final optimum radius -----------------------
        # Gentle final screen: prioritize rejecting positive-tail contamination.
        # Do not strongly penalize broad (still-rising) but otherwise smooth profiles.
        beyond_final = (radii / fwhm) > optimum_radius
        if beyond_final.any():
            tail_excess_vals = []
            tail_min_vals = []
            tail_by_idx = {}
            outer_start_fwhm = float(
                phot_cfg.get("optimum_radius_outer_start_fwhm", 2.2)
            )
            outer_start_fwhm = max(optimum_radius + 0.1, outer_start_fwhm)
            outer_radii_mask = (radii / fwhm) >= outer_start_fwhm
            for i in final_indices:
                prof = profiles_map.get(i)
                if prof is None:
                    continue
                tail = prof[beyond_final]
                t_min = float(np.nanmin(tail))
                t_excess = float(max(0.0, np.nanmax(tail) - 1.0))
                if np.any(outer_radii_mask):
                    outer_prof = np.asarray(prof[outer_radii_mask], float)
                    outer_r = np.asarray((radii / fwhm)[outer_radii_mask], float)
                    outer_std = float(np.nanstd(outer_prof))
                    if len(outer_prof) > 1 and np.isfinite(outer_prof).all():
                        outer_slope = float(
                            np.nanmean(np.gradient(outer_prof, outer_r))
                        )
                    else:
                        outer_slope = np.nan
                else:
                    outer_std = np.nan
                    outer_slope = np.nan
                sources.at[i, "tail_outer_std"] = outer_std
                sources.at[i, "tail_outer_slope"] = outer_slope
                tail_by_idx[i] = (t_min, t_excess, outer_std, outer_slope)
                if np.isfinite(t_excess):
                    tail_excess_vals.append(t_excess)
                if np.isfinite(t_min):
                    tail_min_vals.append(t_min)

            tail_excess_limit_final = max_tail_excess
            if len(tail_excess_vals) > 0:
                tail_excess_limit_final = max(
                    max_tail_excess,
                    float(np.nanpercentile(np.asarray(tail_excess_vals, float), 90) + 0.05),
                )
            min_tail_flux_final = min_tail_flux
            if len(tail_min_vals) > 0:
                # Keep this permissive to avoid excluding broad stars.
                adaptive_min_tail = float(np.nanpercentile(np.asarray(tail_min_vals, float), 5) - 0.1)
                min_tail_flux_final = min(min_tail_flux, adaptive_min_tail)
            # Outer tail should be flat/negligible (no bright source contamination).
            outer_std_max = float(phot_cfg.get("optimum_radius_outer_std_max", 0.05))
            outer_slope_abs_max = float(
                phot_cfg.get("optimum_radius_outer_slope_abs_max", 0.03)
            )
            # Undersampled data: the normalised CoG is derived from only a
            # handful of pixels per radius step, so outer-tail statistics are
            # intrinsically noisier; relax the flatness tolerances to avoid
            # discarding genuine stars on a noise-dominated diagnostic.
            _us_thr = float(phot_cfg.get("undersampled_fwhm_threshold", 2.5))
            if float(fwhm) <= _us_thr:
                _tail_relax = float(
                    phot_cfg.get("optimum_radius_undersampled_tail_relax", 2.5)
                )
                outer_std_max *= max(1.0, _tail_relax)
                outer_slope_abs_max *= max(1.0, _tail_relax)
                tail_excess_limit_final += 0.5 * (_tail_relax - 1.0)
            outer_std_max = max(0.005, outer_std_max)
            outer_slope_abs_max = max(0.001, outer_slope_abs_max)

            n_pre_tail_screen = len(final_indices)
            still_ok = []
            for i in final_indices:
                if i not in tail_by_idx:
                    still_ok.append(i)
                    continue
                t_min, t_excess, outer_std, outer_slope = tail_by_idx[i]
                outer_ok = (
                    np.isfinite(outer_std)
                    and outer_std <= outer_std_max
                    and np.isfinite(outer_slope)
                    and np.abs(outer_slope) <= outer_slope_abs_max
                )
                if (
                    np.isfinite(t_min)
                    and t_min >= min_tail_flux_final
                    and np.isfinite(t_excess)
                    and t_excess <= tail_excess_limit_final
                    and outer_ok
                ):
                    still_ok.append(i)
            if len(still_ok) < len(final_indices):
                logger.info(
                    "Final tail screen: %d -> %d sources "
                    "(outer_std<=%.3f, |outer_slope|<=%.3f, "
                    "tail_excess<=%.2f, min_tail_flux>=%.2f).",
                    n_pre_tail_screen,
                    len(still_ok),
                    outer_std_max,
                    outer_slope_abs_max,
                    tail_excess_limit_final,
                    min_tail_flux_final,
                )
                final_indices = np.array(still_ok, dtype=int)
                filtered_sources = sources.iloc[final_indices].copy()
                if len(final_indices) > 0:
                    optimum_radius = float(
                        np.nanmedian(
                            sources.loc[final_indices, "optimum_radius"].values
                        )
                    )
                # Enforce minimum retention after final tail screen.
                if len(final_indices) < min_keep:
                    fallback_candidates = sources.loc[
                        kept_indices[np.isin(kept_indices, np.where(core_pass_mask)[0])]
                    ].copy()
                    if not fallback_candidates.empty:
                        snr_vals = np.asarray(
                            fallback_candidates.get(
                                "SNR", pd.Series(np.ones(len(fallback_candidates)))
                            ),
                            float,
                        )
                        dr = np.abs(
                            fallback_candidates["optimum_radius"].values - optimum_radius
                        )
                        rank = np.argsort(np.argsort(-snr_vals)) + np.argsort(
                            np.argsort(dr)
                        )
                        fallback_candidates["_rank"] = rank
                        rescued = (
                            fallback_candidates.sort_values("_rank")
                            .head(min_keep)
                            .index.values
                        )
                        final_indices = np.asarray(rescued, dtype=int)
                        filtered_sources = sources.iloc[final_indices].copy()
                        optimum_radius = float(
                            np.nanmedian(
                                sources.loc[final_indices, "optimum_radius"].values
                            )
                        )
                        logger.info(
                            "Final tail screen was over-restrictive; rescued to %d sources.",
                            len(final_indices),
                        )

        # ---- Optional EE model refinement ----------------------------------
        fine_r, fine_profile = None, None
        good_profiles = np.array(
            [profiles_map[i] for i in final_indices if i in profiles_map], dtype=float
        )
        if len(final_indices) > 0 and good_profiles.size > 0:
            try:
                if good_profiles.ndim == 2 and good_profiles.shape[0] > 0:
                    mean_profile = np.nanmedian(good_profiles, axis=0)

                    # Use non-parametric smoothed median profile instead of
                    # parametric model for better representation of the true
                    # curve of growth (handles complex PSF shapes: core + wings).
                    from scipy.interpolate import make_interp_spline
                    
                    # Fine grid for smooth plotting
                    fine_r = np.linspace(0, radii[-1], 500)
                    
                    # Smooth the median profile using a smoothing spline.
                    # Use a small smoothing factor to avoid overfitting but
                    # reduce noise from the discrete radii sampling.
                    try:
                        # Use 3rd order spline with moderate smoothing
                        tck = make_interp_spline(
                            radii, mean_profile, k=3,
                        )
                        fine_profile = tck(fine_r)
                        # Clip to valid range [0, 1]
                        fine_profile = np.clip(fine_profile, 0.0, 1.0)
                        # Ensure monotonic (non-decreasing)
                        fine_profile = np.maximum.accumulate(fine_profile)
                    except Exception:
                        # Fallback to linear interpolation
                        fine_profile = np.interp(fine_r, radii, mean_profile)
                        fine_profile = np.clip(fine_profile, 0.0, 1.0)
                        fine_profile = np.maximum.accumulate(fine_profile)
                    
                    # Refine optimum radius from smoothed profile.  np.interp
                    # silently returns fine_r[-1] if the profile never reaches
                    # aperture_norm_factor -- detect that and warn.
                    r_target_pix = np.interp(aperture_norm_factor, fine_profile, fine_r)
                    if (
                        float(np.nanmax(fine_profile)) < aperture_norm_factor
                    ):
                        logger.warning(
                            "Median CoG never reaches %.0f%% encircled flux "
                            "(max=%.3f); optimum radius saturated at the search "
                            "limit (%.2f FWHM).",
                            100.0 * aperture_norm_factor,
                            float(np.nanmax(fine_profile)),
                            float(radii[-1] / fwhm),
                        )
                    if np.isfinite(r_target_pix) and r_target_pix > 0:
                        optimum_radius = float(r_target_pix / fwhm)
            except Exception as exc:
                log_warning_from_exception(logger, "EE model fit failed", exc)

        # ---- Optimum scale -------------------------------------------------
        # optimum_radius is in FWHM units; convert to pixels before adding a
        # +2*FWHM margin so PSF-star cutouts retain surrounding context.
        optimum_scale = max(12, int(np.ceil((optimum_radius + 2.0) * fwhm))) + 0.5
        if (2 * optimum_scale) % 2 == 0:
            optimum_scale += 0.5

        # ---- Plotting (reuses profiles already in profiles_map) ------------
        if plot:
            # No second Pool - profiles were computed in the analysis pass.
            from plotting_utils import apply_autophot_mplstyle
            try:
                apply_autophot_mplstyle()
            except Exception:
                pass

            from plotting_utils import get_plot_ext, safe_tight_layout
            save_loc = os.path.join(
                self.input_yaml["write_dir"],
                f'Optimum_Aperture_{self.input_yaml["base"]}{get_plot_ext(self.input_yaml)}',
            )
            fig = plt.figure(figsize=set_size(340, 1.5))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)

            kept_set = set(final_indices.tolist())
            # Plot all profiles; grey-out rejected ones.
            for idx, prof in profiles_map.items():
                in_kept = idx in kept_set
                if in_kept:
                    ax1.plot(radii / fwhm, prof, color="tab:blue", alpha=0.6, lw=0.8)
                else:
                    ax1.plot(radii / fwhm, prof, color="grey", alpha=0.3, lw=0.5)

            if fine_r is not None:
                ax1.plot(fine_r / fwhm, fine_profile, ls="--", color="black")

            ax1.axvline(
                global_optimum_pre, color="black", ls=":",
            )
            ax1.axvline(optimum_radius, color="black", ls="--",
            )
            # Vertical text labels on the lines, placed at bottom of upper plot
            ax1.text(
                global_optimum_pre, 0.02,
                r"$f_{{enc}}={:.0f}\%$".format(norm_factor * 100),
                rotation=90, va="bottom", ha="right", fontsize=7,
                color="black",
            )
            ax1.text(
                optimum_radius, 0.02,
                r"$f_{{enc}}={:.0f}\%$".format(aperture_norm_factor * 100),
                rotation=90, va="bottom", ha="right", fontsize=7,
                color="black",
            )
            ax1.set_ylabel("Normalized Flux")
            plt.setp(ax1.get_xticklabels(), visible=False)

            per_source = (
                sources.loc[list(kept_set), "optimum_radius"].values
                if kept_set
                else np.array([])
            )
            other = sources.loc[
                [i for i in range(len(sources)) if i not in kept_set], "optimum_radius"
            ].values
            all_radii = np.concatenate(
                [np.atleast_1d(per_source), np.atleast_1d(other)]
            )
            all_radii = all_radii[np.isfinite(all_radii)]

            if len(all_radii) > 0:
                # Freedman-Diaconis rule for bin edges (shared between selected/rejected).
                # Constrain to [0, max_radius] since radii are in FWHM units here.
                all_clip = all_radii[(all_radii >= 0.0) & (all_radii <= float(max_radius))]
                if all_clip.size == 0:
                    all_clip = all_radii
                try:
                    bins = np.histogram_bin_edges(all_clip, bins="fd")
                except Exception:
                    # Fallback: simple heuristic that behaves well for small n
                    n_bins = min(25, max(8, int(np.ceil(len(all_radii) / 4))))
                    r_min, r_max = float(np.nanmin(all_radii)), float(np.nanmax(all_radii))
                    r_range = max(r_max - r_min, 0.1)
                    bins = np.linspace(
                        max(0, r_min - 0.02 * r_range),
                        min(max_radius, r_max + 0.02 * r_range),
                        n_bins + 1,
                    )
                n_selected = int(np.count_nonzero(np.isfinite(per_source)))
                n_rejected = int(np.count_nonzero(np.isfinite(other)))
                if len(per_source) > 0:
                    ax2.hist(
                        per_source,
                        bins=bins,
                        facecolor="tab:blue",
                        alpha=0.85,
                        label=f"Selected (N={n_selected})",
                        zorder=1,
                    )
                if len(other) > 0:
                    ax2.hist(
                        other,
                        bins=bins,
                        facecolor="grey",
                        alpha=0.5,
                        label=f"Rejected (N={n_rejected})",
                        zorder=0,
                    )
                ax2.axvline(optimum_radius, color="black", ls="--", label="Final")
                ax2.legend(loc="upper right", frameon=False, fontsize=8)

            ax2.set_xlabel("Aperture Radius [FWHM]")
            ax2.set_ylabel("Count")
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlim(-0.05, max_radius + 0.05)

            safe_tight_layout(fig)

            fig.savefig(save_loc, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close(fig)

        logger.info(
            f"Returning {len(filtered_sources)} sources, "
            f"optimum radius: {optimum_radius:.2f} * FWHM, "
            f"scale: {optimum_scale}"
        )
        return filtered_sources, optimum_radius, optimum_scale

    # -----------------------------------------------------------------------
    # Aperture correction
    # -----------------------------------------------------------------------

    def compute_aperture_correction(
        self,
        image: np.ndarray,
        sources: pd.DataFrame,
        n_samples: int = 25,
        fwhm: float = None,
        ap_size: float = None,
        max_radius: float = 5.0,
        background_rms: np.ndarray = None,
        plot: bool = True,
        mask: np.ndarray = None,
    ):
        """
        Compute the aperture correction (ap_size -> inf) via Curve of Growth.

        Parameters
        ----------
        image          : 2-D ndarray
        sources        : DataFrame with 'x_pix', 'y_pix', 'flux_AP'
        n_samples      : bright stars to include
        fwhm           : PSF FWHM in pixels (required)
        ap_size        : science aperture radius in pixels (required)
        max_radius     : CoG extent in FWHM units
        background_rms : optional 2-D RMS map
        plot           : save histogram PDF

        Returns
        -------
        (correction, correction_err) : (float, float)  [mag]
        """
        logger = logging.getLogger(__name__)
        write_dir = self.input_yaml["write_dir"]
        base_name = self.input_yaml["base"]

        if len(sources) < 5:
            logger.warning("Too few sources [%s] for aperture correction.", len(sources))
            return np.nan, np.nan

        if not {"x_pix", "y_pix", "flux_AP"}.issubset(sources.columns):
            raise ValueError(
                "sources must contain 'x_pix', 'y_pix', and 'flux_AP' columns."
            )

        if fwhm is None or ap_size is None:
            raise ValueError("fwhm and ap_size are required.")
        fwhm = float(fwhm)
        ap_size = float(ap_size)
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError(f"Invalid fwhm={fwhm}; expected finite and >0.")
        if not np.isfinite(ap_size) or ap_size <= 0:
            raise ValueError(f"Invalid ap_size={ap_size}; expected finite and >0.")

        gain = resolve_gain_e_per_adu(None, self.input_yaml)
        radii = np.arange(0.05, max_radius, 0.1) * fwhm
        if background_rms is not None:
            # Replace NaN values (chip gaps) with median of finite values.
            _bkg_rms = np.asarray(background_rms, dtype=float)
            if np.any(np.isnan(_bkg_rms)):
                _finite_median = float(np.nanmedian(_bkg_rms))
                if not np.isfinite(_finite_median) or _finite_median <= 0:
                    _finite_median = 1.0
                _bkg_rms = np.where(np.isfinite(_bkg_rms), _bkg_rms, _finite_median)
            _bkg_rms = np.abs(_bkg_rms)
            # Convert both data and bkg_error to electrons (matches Aperture.measure convention).
            _image_e_ac = np.where(
                np.isfinite(image), image * gain, np.nan
            )
            error = calc_total_error(
                _image_e_ac, _bkg_rms * gain, effective_gain=1
            )
            # CurveOfGrowth runs on `image` (ADU) below; keep the error map
            # in matching units.
            error = error / gain
        else:
            error = None

        selected = sources.sort_values("flux_AP", ascending=False).head(n_samples)

        corrections = []
        for _, row in selected.iterrows():
            try:
                xycen = np.array([row["x_pix"], row["y_pix"]])
                cog = CurveOfGrowth(
                    image, xycen, radii, error=error, mask=mask, method="subpixel"
                )
                cog.normalize()
                # np.interp silently clamps to the profile endpoints when
                # ap_size falls outside the measured radii (e.g. ap_size >
                # max_radius*fwhm -> frac=1 -> correction=0).  Guard instead.
                if not (cog.radii[0] <= ap_size <= cog.radii[-1]):
                    continue
                frac = np.interp(ap_size, cog.radii, cog.profile)
                if 0 < frac <= 1:
                    corrections.append(-2.5 * np.log10(1.0 / frac))
            except Exception as exc:
                log_warning_from_exception(logger, "Skipping star", exc)

        if not corrections:
            logger.warning("No valid aperture corrections computed.")
            return np.nan, np.nan

        corrections = np.asarray(corrections, dtype=float)

        # Sigma-clip outliers.
        clipped = sigma_clip(
            corrections,
            sigma=3,
            masked=True,
            cenfunc=np.nanmedian,
            stdfunc=mad_std,
        )
        corrections = corrections[~clipped.mask]
        if corrections.size == 0:
            logger.warning("Aperture-correction sigma-clip rejected all stars.")
            return np.nan, np.nan
        correction = float(np.nanmedian(corrections))
        # Use the standard error of the median (SE = 1.858 * MAD / sqrt(N)),
        # not the population std.  The correction is a median estimate, so its
        # uncertainty scales as 1/sqrt(N), not as the scatter itself.
        # Using std overestimates the error when many sources are available
        # (e.g. 25 sources: std ~ 0.05 mag, SE ~ 0.01 mag).
        _n_corr = len(corrections)
        if _n_corr >= 2:
            _mad_corr = float(median_abs_deviation(corrections, nan_policy="omit"))
            correction_err = float(1.858 * _mad_corr / np.sqrt(_n_corr))
        else:
            correction_err = float(np.nanstd(corrections))
        logger.info("Aperture correction: %.3f +/- %.3f (N=%d, SE of median)", correction, correction_err, _n_corr)

        if plot:
            plt.ioff()
            from plotting_utils import apply_autophot_mplstyle, get_plot_color, get_plot_ext, safe_tight_layout
            apply_autophot_mplstyle()
            fig, ax = plt.subplots(figsize=set_size(540, aspect=1.2))
            try:
                be = np.histogram_bin_edges(corrections, bins="fd")
            except Exception:
                be = 15
            ax.hist(corrections, bins=be, alpha=0.7, color=get_plot_color('hist_primary'), edgecolor="black")
            ax.axvline(
                correction, color="r", ls="--", label=f"Median: {correction:.3f}"
            )
            ax.set_xlabel("Aperture Correction [mag]")
            ax.set_ylabel("Number of Sources")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      frameon=False, fontsize=8)
            safe_tight_layout(fig)
            png_path = os.path.join(
                write_dir,
                f"Aperture_Correction_{base_name}{get_plot_ext(self.input_yaml)}",
            )
            fig.savefig(
                png_path,
                bbox_inches="tight",
                dpi=150,
                facecolor="white",
            )
            plt.close(fig)

        return correction, correction_err
