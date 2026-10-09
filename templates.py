# -*- coding: utf-8 -*-
"""
Optimized script for astronomical image processing.

Handles template downloading (PanSTARRS, 2MASS), image alignment, masking,
cropping, flux calibration, and image subtraction (ZOGY / SFFT / HOTPANTS).

Key optimizations over the original:
  - Consolidated FITS I/O to avoid redundant reads of the same file.
  - Fixed mask-logic bug in find_largest_available_area (was OR, now AND).
  - Fixed robust_outlier_mask returning indices in sorted vs. original order.
  - Fixed ConstrainedSlopeRegressor reference (was self.Class, now module-level).
  - Fixed variable-shadowing bug in download_panstarrs_template.
  - Moved pure helper functions to module level (distance, _odd, find_conda_env).
  - Replaced legacy np.random calls with np.random.default_rng().
  - Used pathlib.Path consistently for file path handling.
  - Added dataclass-based parameter containers for clarity.

Created on Thu Oct 27 11:27:05 2022
@author: seanbrennan
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import gc
import glob
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve

# =============================================================================
# Third-Party Library Imports
# =============================================================================
import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from scipy.ndimage import (
    binary_dilation,
    uniform_filter,
    shift as _ndimage_shift,
)
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.linear_model import RANSACRegressor
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

    _SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    # Keep template download usable without scikit-learn; the
    # regression/color-term paths error at runtime instead.
    class _BaseEstimatorPlaceholder:
        """Stub used when scikit-learn is unavailable."""

        pass

    class _RegressorMixinPlaceholder:
        """Stub used when scikit-learn is unavailable."""

        pass

    BaseEstimator = _BaseEstimatorPlaceholder
    RegressorMixin = _RegressorMixinPlaceholder
    RANSACRegressor = None
    check_X_y = None
    check_array = None
    check_is_fitted = None
    _SKLEARN_AVAILABLE = False

# =============================================================================
# Astropy Imports
# =============================================================================
import astropy.wcs as astropy_wcs_module
from astropy import units as u
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS

# =============================================================================
# Astroquery Imports
# =============================================================================
try:
    from astroquery.sdss import SDSS  # type: ignore

    _ASTROQUERY_AVAILABLE = True
except ModuleNotFoundError:
    SDSS = None
    _ASTROQUERY_AVAILABLE = False

# =============================================================================
# Photutils Imports
# =============================================================================
try:
    from photutils.aperture import CircularAperture, RectangularAperture
    from photutils.segmentation import (
        detect_sources,
        SourceCatalog,
        make_2dgaussian_kernel,
        SegmentationImage,
    )
    from photutils.detection import DAOStarFinder

    _PHOTUTILS_AVAILABLE = True
except ModuleNotFoundError:
    # Keep template download importable without photutils; the
    # alignment/photometry paths fail at runtime instead.
    CircularAperture = None
    RectangularAperture = None
    detect_sources = None
    SourceCatalog = None
    make_2dgaussian_kernel = None
    SegmentationImage = None
    DAOStarFinder = None
    _PHOTUTILS_AVAILABLE = False

# =============================================================================
# Reproject Imports
# =============================================================================
try:
    from reproject import reproject_interp, reproject_adaptive, reproject_exact

    _REPROJECT_AVAILABLE = True
except ModuleNotFoundError:
    reproject_interp = None
    reproject_adaptive = None
    reproject_exact = None
    _REPROJECT_AVAILABLE = False

from functools import lru_cache
from typing import NamedTuple

# =============================================================================
# Custom / Local Module Imports
# =============================================================================
try:
    from catalog import Catalog
except ModuleNotFoundError:
    Catalog = None
try:
    from functions import (
        log_step,
        distance_to_uniform_row_col,
        get_header,
        get_image,
        get_image_stats,
        save_to_fits,
        remove_wcs_from_header,
    )
except (ModuleNotFoundError, ImportError):
    # Download-only use does not require the full photometry stack.
    def log_step(message: Any, *args: Any, **kwargs: Any) -> str:
        m = str(message).strip()
        if not m:
            return ""
        return f"\n---- {m} ----"

    distance_to_uniform_row_col = None
    get_header = None
    get_image = None
    get_image_stats = None
    save_to_fits = None
    remove_wcs_from_header = None
from wcs import get_wcs
try:
    from utils import run_IDC
except (ModuleNotFoundError, ImportError):
    run_IDC = None

from functions import clean_subprocess_log, log_warning_from_exception, safe_fits_write, cap_console_lines, STATUS, _sfft_post_anomaly_retry_viable
try:
    from functions import invalidate_fits_cache
except ImportError:
    def invalidate_fits_cache(fpath=None):
        return None
# =============================================================================
# External Tool Imports
# =============================================================================
try:
    import legacystamps  # Optional: only required for Legacy Survey templates

    _HAS_LEGACYSTAMPS = True
except ImportError:
    legacystamps = None  # type: ignore[assignment]
    _HAS_LEGACYSTAMPS = False

# =============================================================================
# Optional Alignment Packages
# =============================================================================
try:
    import spalipy

    _HAS_SPALIPY = True
except ImportError:
    spalipy = None  # type: ignore[assignment]
    _HAS_SPALIPY = False

try:
    import tweakwcs  # noqa: F401

    _HAS_TWEAKWCS = True
except ImportError:
    tweakwcs = None  # type: ignore[assignment]
    _HAS_TWEAKWCS = False

try:
    from image_registration import chi2_shift
    from image_registration.fft_tools import shift as _imgreg_shift

    _HAS_IMGREG = True
except ImportError:
    chi2_shift = None  # type: ignore[assignment]
    _imgreg_shift = None  # type: ignore[assignment]
    _HAS_IMGREG = False

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
cap_console_lines(logging.getLogger().handlers)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Constants
# =============================================================================

# Unseeded so random sequences differ across images (replaces np.random.*)
RNG = np.random.default_rng(seed=None)

# Sentinel value used to represent "no data" in FITS images after alignment.
# 0.0 is used so that SExtractor (inside SFFT/HOTPANTS) does not treat the
# no-coverage border as a giant connected source, which causes "Pixel stack
# overflow" and prevents real sources from being detected.  SFFT's own
# internal mask already excludes pixels with |value| < 1.1e-20 (i.e. 0.0).
NO_DATA_SENTINEL = 0.0


class AlignmentResult(NamedTuple):
    """Structured result from an alignment attempt.

    ``success`` is derived from the output paths: every alignment method
    must produce an aligned template, so a falsy ``template_path`` means
    the attempt failed.  ``quality`` is 'pass', 'warn', or 'fail' under the
    shared gate in ``_classify_alignment_quality`` ('fail' also marks
    unverifiable products when verification was required).
    """
    science_path: Optional[str]
    template_path: Optional[str]
    method_used: Optional[str]
    median_offset_px: Optional[float]
    rms_px: Optional[float] = None
    p90_px: Optional[float] = None
    coverage_ok: Optional[bool] = None
    n_matches: Optional[int] = None
    finite_fraction: Optional[float] = None
    quality: str = "fail"
    message: str = ""

    @property
    def success(self) -> bool:
        return bool(self.template_path)


# PanSTARRS filter names
PANSTARRS_FILTERS = frozenset({"g", "r", "i", "z",'y','w'})

# Valid 2MASS band identifiers (case-insensitive input, stored upper)
TWOMASS_VALID_BANDS = frozenset({"J", "H", "KS", "K"})
TWOMASS_MAX_SIZE_ARCMIN = 15

# Default sigma for sigma-clipped statistics throughout the pipeline
DEFAULT_SIGMA_CLIP = 3.0

# Default FWHM padding multiplier for masking around bright / saturated sources
DEFAULT_FWHM_PADDING_MULTIPLIER = 3


def _pad_psf_to_image(psf: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Zero-pad a PSF stamp to *target_shape* with the PSF centred.

    ZOGY's FFT-based subtraction requires the PSF to be the same shape as
    the science/reference images.  The PSF stamp is placed at the centre
    of a zero-filled array of the target shape.

    A Tukey edge taper is applied to the PSF stamp before padding to
    smooth any non-zero values at the stamp edges to zero.  Without this,
    truncated PSF wings create a sharp cutoff that produces sinc-like
    ripples in Fourier space, appearing as correlated noise patterns in
    the difference image.
    """
    psf = np.asarray(psf, dtype=float)
    th, tw = target_shape
    ph, pw = psf.shape
    if ph == th and pw == tw:
        # Still needs ifftshift so the (centred) PSF zero-lag sits at [0,0],
        # matching the FFT convolution convention used below.
        return np.fft.ifftshift(psf)

    # Apply Tukey edge taper: smoothly taper the outer ~20% of each axis
    # to zero.  Only taper if there are non-zero values near the edges
    # (avoid modifying already well-tapered PSFs).
    _edge_frac = 0.2
    _edge_h = max(1, int(ph * _edge_frac))
    _edge_w = max(1, int(pw * _edge_frac))
    _needs_taper = False
    if ph > 2 * _edge_h:
        _top = np.max(np.abs(psf[:_edge_h, :]))
        _bot = np.max(np.abs(psf[-_edge_h:, :]))
        if _top > 1e-6 * np.max(np.abs(psf)) or _bot > 1e-6 * np.max(np.abs(psf)):
            _needs_taper = True
    if pw > 2 * _edge_w:
        _left = np.max(np.abs(psf[:, :_edge_w]))
        _right = np.max(np.abs(psf[:, -_edge_w:]))
        if _left > 1e-6 * np.max(np.abs(psf)) or _right > 1e-6 * np.max(np.abs(psf)):
            _needs_taper = True
    if _needs_taper:
        _win_y = np.ones(ph)
        _win_x = np.ones(pw)
        if ph > 2 * _edge_h:
            _ramp = 0.5 * (1 - np.cos(np.pi * np.arange(_edge_h) / _edge_h))
            _win_y[:_edge_h] = _ramp
            _win_y[-_edge_h:] = _ramp[::-1]
        if pw > 2 * _edge_w:
            _ramp = 0.5 * (1 - np.cos(np.pi * np.arange(_edge_w) / _edge_w))
            _win_x[:_edge_w] = _ramp
            _win_x[-_edge_w:] = _ramp[::-1]
        psf = psf * np.outer(_win_y, _win_x)
        # Tapering changes the sum; restore unit normalization (ZOGY assumes it).
        _s = float(np.nansum(psf))
        if _s > 0:
            psf = psf / _s

    out = np.zeros(target_shape, dtype=float)
    # Place PSF center at (th//2, tw//2) so ifftshift moves it to [0,0]
    y0 = th // 2 - ph // 2
    x0 = tw // 2 - pw // 2
    y1 = y0 + ph
    x1 = x0 + pw
    # Clip to bounds in case PSF is somehow larger than the image
    sy0 = max(0, -y0)
    sx0 = max(0, -x0)
    sy1 = ph - max(0, y1 - th)
    sx1 = pw - max(0, x1 - tw)
    out[max(0, y0):min(th, y1), max(0, x0):min(tw, x1)] = psf[sy0:sy1, sx0:sx1]
    # Move PSF center to [0,0] for FFT-based convolution.
    # np.fft.fft2 treats [0,0] as the origin; a centered PSF introduces
    # a circular shift of (th//2, tw//2) in convolution results.
    out = np.fft.ifftshift(out)
    return out


def _zogy_matched_aperture_scale(
    science: np.ndarray,
    reference: np.ndarray,
    positions,
    fwhm: float,
    sat_sci: float = np.nan,
    sat_ref: float = np.nan,
) -> float:
    """Flux scale (ref -> sci) from aperture photometry at matched positions.

    Per-pixel ratios mix different PSF shapes (core vs wing signal differs
    between a sharp science PSF and a broader template PSF), which biases
    the median pixel ratio by a few percent -- enough to leave residuals
    proportional to source flux.  Integrating over a wide aperture at
    positions shared by both images cancels the PSF difference: encircled
    energy within ~2 FWHM is nearly complete for both.

    Returns (scale, n_ratios): NaN scale when no usable ratio survives,
    so callers can fall back to the pixel-ratio estimator.  The ratio
    count lets the caller sanity-check an estimate that rests on only
    one or two stars.
    """
    if positions is None or len(positions) == 0:
        return np.nan, 0
    try:
        fwhm = float(fwhm)
    except (TypeError, ValueError):
        fwhm = 0.0
    if not (np.isfinite(fwhm) and fwhm > 0):
        fwhm = 10.0
    r_ap = 2.0 * fwhm
    r_in = 2.5 * fwhm
    r_out = 3.5 * fwhm
    half = int(np.ceil(r_out)) + 1
    h, w = science.shape
    ratios = []
    for pos in positions:
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, IndexError, ValueError, KeyError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = round(x), round(y)
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            continue
        c_sci = science[yi - half : yi + half + 1, xi - half : xi + half + 1]
        c_ref = reference[yi - half : yi + half + 1, xi - half : xi + half + 1]
        # Skip sources with a clipped core on either side: the measured
        # ratio is biased toward the saturated image's ceiling.
        try:
            if (
                np.isfinite(float(sat_sci))
                and np.nanmax(c_sci) >= 0.9 * float(sat_sci)
            ) or (
                np.isfinite(float(sat_ref))
                and np.nanmax(c_ref) >= 0.9 * float(sat_ref)
            ):
                continue
        except (TypeError, ValueError):
            pass
        yy, xx = np.mgrid[
            yi - half : yi + half + 1, xi - half : xi + half + 1
        ].astype(float)
        r = np.hypot(xx - x, yy - y)
        finite = np.isfinite(c_sci) & np.isfinite(c_ref)
        ann = finite & (r > r_in) & (r <= r_out)
        ap = finite & (r <= r_ap)
        if ann.sum() < 20 or ap.sum() < 10:
            continue
        sky_sci = float(np.median(c_sci[ann]))
        sky_ref = float(np.median(c_ref[ann]))
        std_sci = 1.4826 * float(np.median(np.abs(c_sci[ann] - sky_sci)))
        std_ref = 1.4826 * float(np.median(np.abs(c_ref[ann] - sky_ref)))
        f_sci = float(np.sum(c_sci[ap] - sky_sci))
        f_ref = float(np.sum(c_ref[ap] - sky_ref))
        # A matched position can be blank sky in one frame (e.g. a cosmic
        # ray detected only in the science image); its aperture sum is then
        # annulus noise and the ratio is garbage.  Require a real aperture
        # detection on both sides.
        n_ap = np.sqrt(float(ap.sum()))
        if f_sci < 5.0 * std_sci * n_ap or f_ref < 5.0 * std_ref * n_ap:
            continue
        if f_sci > 0 and f_ref > 0:
            ratios.append(f_sci / f_ref)
    if len(ratios) < 1:
        return np.nan, 0
    ratios = np.asarray(ratios)
    med = float(np.median(ratios))
    if not (np.isfinite(med) and med > 0):
        return np.nan, 0
    # One 3-sigma clip rejects blends/crowded apertures before the median.
    mad = float(np.median(np.abs(ratios - med))) * 1.4826
    if mad > 0:
        keep = np.abs(ratios - med) < 3.0 * mad
        if keep.sum() >= 3:
            med = float(np.median(ratios[keep]))
    return med, int(len(ratios))


def _zogy_exposure_flux_scale(
    science_exposure,
    reference_exposure,
    science_gain,
    reference_gain,
):
    try:
        science_exposure = float(science_exposure)
        reference_exposure = float(reference_exposure)
        science_gain = float(science_gain)
        reference_gain = float(reference_gain)
    except (TypeError, ValueError):
        return np.nan
    values = (
        science_exposure,
        reference_exposure,
        science_gain,
        reference_gain,
    )
    if not all(np.isfinite(value) and value > 0 for value in values):
        return np.nan
    scale = (science_exposure / reference_exposure) * (
        reference_gain / science_gain
    )
    return scale if np.isfinite(scale) and scale > 0 else np.nan


def _convolve_variance_map(variance, kernel):
    variance = np.asarray(variance, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    result = np.fft.ifft2(
        np.fft.fft2(variance) * np.fft.fft2(kernel**2)
    ).real
    return np.maximum(result, 0.0)


def _zogy_star_psf(
    image: np.ndarray,
    positions,
    guess_fwhm: float = 8.0,
    saturate: float = np.nan,
):
    """Median-stacked native PSF stamp + FWHM from matched star cutouts.

    Bright matched sources measure the *actual* field PSF directly --
    independent of ePSF convergence or the header/SExtractor FWHM (both
    under-measure broad plateau PSFs on resampled frames).  Each cutout
    is recentred on its own centroid before stacking -- matched positions
    carry ~1 px catalogue centring error that would otherwise smear the
    core into the wings and inflate the stack width.

    Returns ``(stamp, fwhm, kept_positions)``: a unit-normalized stamp
    built as the 2D median of the subpixel-recentred, peak-normalized
    cutouts (the real resampled-frame PSF shape -- an analytic model
    can carry 2-5x wrong far-wing flux that a matching kernel then
    prints as core/ring residuals), the half-maximum FWHM measured on
    that stamp, and the positions that survived the per-star
    consistency filter (cosmic rays and hot pixels masquerading as
    matched sources must not collapse the stack).
    ``kept_positions`` is ``None`` only when vetting never ran (no
    positions offered); a run that rejected everything returns ``[]``
    so callers can tell "no usable stars" apart from "no vetting".
    """
    if positions is None or len(positions) < 1:
        return None, float("nan"), None
    try:
        guess = float(guess_fwhm)
    except (TypeError, ValueError):
        guess = 8.0
    if not (np.isfinite(guess) and guess > 0):
        guess = 8.0
    half = int(np.ceil(3.5 * guess)) + 2
    h, w = image.shape
    sky_r_in = half - int(np.ceil(0.5 * guess)) - 1
    radii = np.arange(0.0, half - 1, 0.5)
    profs = []
    shifted = []
    kept_pos = []
    for pos in positions:
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, IndexError, ValueError, KeyError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = round(x), round(y)
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            continue
        c = image[yi - half : yi + half + 1, xi - half : xi + half + 1]
        if not np.isfinite(c).any():
            continue
        gy, gx = np.mgrid[-half : half + 1, -half : half + 1]
        r = np.hypot(gx, gy)
        sky_m = (r > sky_r_in) & np.isfinite(c)
        if sky_m.sum() < 20:
            continue
        sky = float(np.nanmedian(c[sky_m]))
        sky_std = 1.4826 * float(np.nanmedian(np.abs(c[sky_m] - sky)))
        if not np.isfinite(sky_std) or sky_std < 0:
            sky_std = 0.0
        # Matched positions carry catalogue centring error (a bad match
        # or the integer round is enough): profiling about the catalog
        # position smears the core into the wings and broadens every
        # member of the stack.  Recentre on the cutout's own centroid.
        _cb = int(np.ceil(1.5 * guess)) + 2
        _sub = c[half - _cb : half + _cb + 1, half - _cb : half + _cb + 1]
        _wsub = np.clip(np.where(np.isfinite(_sub), _sub - sky, 0.0), 0.0, None)
        _wsum = float(_wsub.sum())
        if _wsum > 0:
            _sgy, _sgx = np.mgrid[-_cb : _cb + 1, -_cb : _cb + 1]
            cx = float((_wsub * _sgx).sum() / _wsum)
            cy = float((_wsub * _sgy).sum() / _wsum)
        else:
            cx = cy = 0.0
        r_c = np.hypot(gx - cx, gy - cy)
        _pk_m = (r_c <= 2.0) & np.isfinite(c)
        peak = (
            float(np.nanmax(c[_pk_m])) - sky if _pk_m.any()
            else float(c[half, half]) - sky
        )
        # A matched science position can land on blank sky in the other
        # frame (e.g. a cosmic ray); its noise "profile" then measures a
        # garbage width and can still slip through the FWHM window.
        if not np.isfinite(peak) or peak <= 0 or peak < 10.0 * sky_std:
            continue
        # A saturated star has a clipped core whose flat-topped profile
        # reads wider than the true PSF -- it would inflate the stack.
        if np.isfinite(saturate) and saturate > 0:
            _cp = float(np.nanmax(c))
            if np.isfinite(_cp) and _cp >= 0.9 * saturate:
                continue
        # The header saturation gate misses clipped stars whenever the
        # SATURATE card is absent or bogus (GROND r carries 1e10).  A
        # flat top is self-evident: the median over the core disk sits
        # within ~3% of the peak, where even a FWHM~15 px unsaturated
        # profile keeps the median below ~0.95.
        _core_m = (r_c <= 2.5) & np.isfinite(c)
        if _core_m.sum() >= 5:
            _core_med = float(np.nanmedian(c[_core_m]) - sky) / peak
            if np.isfinite(_core_med) and _core_med > 0.97:
                continue
        with warnings.catch_warnings():
            # Empty radial bins (all-NaN cutout corners, gaps past the
            # cutout edge) are expected; nanmedian of an empty slice is
            # NaN, which is what the code wants, but it warns.
            warnings.simplefilter("ignore", RuntimeWarning)
            prof = np.array(
                [
                    np.nanmedian(
                        c[
                            np.isfinite(c)
                            & (r_c >= ri - 0.5)
                            & (r_c < ri + 0.5)
                        ]
                    )
                    - sky
                    for ri in radii
                ]
            ) / peak
        profs.append(prof)
        # Subpixel-recentred, peak-normalized cutout for the 2D median
        # stack: keeps the real PSF shape (ellipticity, resampled-frame
        # wings) that the azimuthal median profile erases.
        _cs = np.where(np.isfinite(c), c - sky, 0.0)
        _cs = _ndimage_shift(
            _cs, (-cy, -cx), order=3, mode="constant", cval=0.0
        )
        shifted.append(_cs / peak)
        kept_pos.append((x, y))
    if len(profs) < 1:
        return None, float("nan"), []
    profs = np.asarray(profs)

    def _prof_fwhm(p):
        # Per-star width, anchored on the guess: a matched-source list
        # can be dominated by cosmic-ray spikes (~1-3 px), which would
        # otherwise form the "consistent" ensemble and exclude real stars.
        i = np.nonzero(p < 0.5)[0]
        if i.size == 0 or i[0] == 0:
            return np.nan
        j = int(i[0])
        frac = (p[j - 1] - 0.5) / (p[j - 1] - p[j] + 1e-30)
        return 2.0 * (radii[j - 1] + frac * 0.5)

    indiv = np.array([_prof_fwhm(p) for p in profs])
    near = np.isfinite(indiv) & (indiv > 0.45 * guess) & (indiv < 1.8 * guess)
    logger.debug(
        "ZOGY star-stack: %d cutouts measured (guess %.2f px), "
        "%d inside [%.2f, %.2f] px; per-star FWHMs: %s",
        len(profs),
        guess,
        int(near.sum()),
        0.45 * guess,
        1.8 * guess,
        ", ".join(f"{f:.2f}" if np.isfinite(f) else "nan" for f in indiv),
    )
    if near.sum() < 1:
        # Rescue pass: when the guess FWHM is wrong (a mis-measured image
        # FWHM can be off by ~2x on sparse junk-rich fields) the window
        # excludes every real star.  If the finite per-star widths instead
        # form a dominant tight clump at a plausible seeing width, trust
        # the clump over the guess.  The >=2.5 px floor keeps consistent
        # CR-spike ensembles (~1-2 px) from passing as stars.
        _fin = np.isfinite(indiv) & (indiv > 0)
        if _fin.sum() >= 3:
            _med_f = float(np.median(indiv[_fin]))
            _sig_f = max(
                1.4826 * float(np.median(np.abs(indiv[_fin] - _med_f))),
                0.15 * _med_f,
            )
            _clump = _fin & (np.abs(indiv - _med_f) < 2.5 * _sig_f)
            _clump_med = (
                float(np.median(indiv[_clump])) if _clump.any() else 0.0
            )
            # The clump itself must be tight: a scattered ensemble whose
            # median happens to sit far from the guess is junk, not a
            # mis-guessed stellar locus - a bad substitute stamp would do
            # more harm than no stack at all.
            _clump_sig = (
                1.4826
                * float(np.median(np.abs(indiv[_clump] - _clump_med)))
                if _clump.any()
                else np.inf
            )
            if (
                _clump.sum() >= max(3, int(np.ceil(0.6 * _fin.sum())))
                and _clump_med >= 2.5
                and np.isfinite(_clump_sig)
                and _clump_sig <= 0.25 * _clump_med
            ):
                logger.debug(
                    "ZOGY star-stack: no cutout inside [%.2f, %.2f] px "
                    "around the %.2f px guess, but %d/%d FWHMs clump at "
                    "%.2f px; re-anchored on the ensemble.",
                    0.45 * guess, 1.8 * guess, guess,
                    int(_clump.sum()), int(_fin.sum()), _clump_med,
                )
                near = _clump
    if near.sum() < 1:
        # No cutout looks like the expected stellar width -- the position
        # list is dominated by spikes/galaxies; no stack.  A single
        # survivor is still used: one bright-star cutout is a far better
        # PSF than a blind analytic guess.
        return None, float("nan"), []
    profs = profs[near]
    shifted = [s for s, k in zip(shifted, near) if k]
    kept_pos = [p for p, k in zip(kept_pos, near) if k]
    med = np.nanmedian(profs, axis=0)
    # Half-maximum crossing with linear interpolation.
    i = np.nonzero(med < 0.5)[0]
    if i.size == 0 or i[0] == 0:
        fwhm = float("nan")
    else:
        j = int(i[0])
        frac = (med[j - 1] - 0.5) / (med[j - 1] - med[j] + 1e-30)
        fwhm = 2.0 * (radii[j - 1] + frac * 0.5)
    # 2D median of the recentred, peak-normalized cutouts.  Residual
    # sky offset comes out of the far annulus; negative clips keep
    # median-of-few noise dips from feeding the FFT denominators.
    stamp = np.nanmedian(np.asarray(shifted), axis=0)
    n = stamp.shape[0]
    gy, gx = np.mgrid[0:n, 0:n]
    rr = np.hypot(gx - n // 2, gy - n // 2)
    _edge = stamp[rr > sky_r_in]
    if _edge.size:
        stamp = stamp - float(np.nanmedian(_edge))
    stamp = np.where(np.isfinite(stamp), stamp, 0.0)
    stamp = np.clip(stamp, 0.0, None)
    # A median of few members can peak ~1 px off the stamp centre;
    # left as-is, ifftshift splits the PSF across the padded array
    # corners and both the kernel and the fidelity probe misread it.
    _rc = int(np.ceil(1.5 * guess))
    _core = stamp[
        n // 2 - _rc : n // 2 + _rc + 1, n // 2 - _rc : n // 2 + _rc + 1
    ]
    _csum = float(_core.sum())
    if _csum > 0:
        _cgy, _cgx = np.mgrid[-_rc : _rc + 1, -_rc : _rc + 1]
        _dx = float((_core * _cgx).sum() / _csum)
        _dy = float((_core * _cgy).sum() / _csum)
        if np.isfinite(_dx) and np.isfinite(_dy) and (
            abs(_dx) > 0.05 or abs(_dy) > 0.05
        ):
            stamp = _ndimage_shift(
                stamp, (-_dy, -_dx), order=3, mode="constant", cval=0.0
            )
    fwhm_s = _measure_stamp_fwhm(stamp)
    if np.isfinite(fwhm_s) and fwhm_s > 0:
        fwhm = fwhm_s
    logger.debug(
        "ZOGY star-stack kept positions: %s",
        ", ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in kept_pos),
    )
    total = float(stamp.sum())
    if not np.isfinite(total) or total <= 0:
        # The individual cutouts passed vetting even though their
        # median stack cannot be normalised - the positions are still
        # the vetted subset.
        return None, float("nan"), kept_pos
    return stamp / total, fwhm, kept_pos


def _zogy_star_stack_consistent(star_fwhm, reference_fwhms, max_frac=0.12):
    try:
        star_fwhm = float(star_fwhm)
        max_frac = float(max_frac)
    except (TypeError, ValueError):
        return True
    if not (np.isfinite(star_fwhm) and star_fwhm > 0):
        return True
    references = []
    for value in reference_fwhms:
        try:
            reference = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(reference) and reference > 0:
            references.append(reference)
    if not references:
        return True
    # Only a mutually-consistent set of references can veto the stack.
    # When the image FWHM and the model-stamp width disagree with each
    # other (a degenerate analytic/ePSF stamp is the usual culprit), the
    # stack is the tiebreaker: it is a direct measurement on the field
    # stars, and rejecting it would let the bad stamp survive repair.
    med = float(np.median(references))
    consistent = [
        ref for ref in references
        if abs(ref - med) / med <= max_frac
    ]
    if not consistent:
        # Loose sanity bound still applies: a stack nowhere near any
        # reference is a spike ensemble, not a PSF.
        return min(references) * 0.5 <= star_fwhm <= max(references) * 2.0
    if any(
        abs(star_fwhm - reference) / reference <= max_frac
        for reference in consistent
    ):
        return True
    # A stack narrower than the reference ensemble is not a contaminant
    # signature: galaxies, saturation flat-tops and centroid jitter all
    # broaden a stack, and nothing real is narrower than the true PSF.
    # The references are not independent probes either -- a fallback
    # model stamp is sized on the image FWHM, so two agreeing numbers
    # can still be one biased measurement (galaxy-inflated detection
    # FWHMs are a documented failure mode).  A coherent stack reading
    # narrower is evidence the references are biased, not that the stars
    # are junk; trust it within a bounded offset -- past ~40% the stack
    # is a different source population, not a width disagreement.
    ref_lo = min(consistent)
    return star_fwhm < ref_lo and star_fwhm >= ref_lo * 0.6


def _load_psf_stamp_native(fpath: str) -> np.ndarray:
    """Load a saved ePSF FITS stamp resampled to native detector pixels.

    ``PSF.build`` saves the ePSF on its oversampled grid and records the
    grid convention in the header (``OVERSAMP``, ``PSFNPIX``, ``PSFX0`` /
    ``PSFY0``).  ZOGY requires a native-resolution PSF: for ``OVERSAMP > 1``
    the stamp is extracted on the native grid via the photutils ePSF
    mapping ``over = c_over + k * (native - c_native)`` -- an exact strided
    extraction when the offsets are integral (always the case for odd
    cutouts under the photutils convention), falling back to cubic
    interpolation otherwise.  Files without the keywords (older runs) are
    returned unchanged, i.e. assumed native.
    """
    with fits.open(fpath) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header
    try:
        k = int(hdr.get("OVERSAMP", 1) or 1)
    except Exception:
        k = 1
    if k <= 1 or data.ndim != 2:
        return data

    n = int(hdr.get("PSFNPIX", 0) or 0)
    if n <= 0:
        # Recover the native size from the oversampled shape:
        # S = k*n + 1 (even k) or S = k*n (odd k) both give round((S-1)/k).
        n = max(1, int(round((min(data.shape) - 1) / k)))
    cy = float(hdr.get("PSFY0", (data.shape[0] - 1) / 2.0))
    cx = float(hdr.get("PSFX0", (data.shape[1] - 1) / 2.0))
    half = (n - 1) / 2.0
    off_y = cy - k * half
    off_x = cx - k * half
    if (
        abs(off_y - round(off_y)) < 1e-6
        and abs(off_x - round(off_x)) < 1e-6
        and int(round(off_y)) + k * (n - 1) < data.shape[0]
        and int(round(off_x)) + k * (n - 1) < data.shape[1]
    ):
        native = data[int(round(off_y)) :: k, int(round(off_x)) :: k][:n, :n]
    else:
        from scipy.ndimage import map_coordinates

        cc = np.arange(n) - half
        gy, gx = np.meshgrid(cy + k * cc, cx + k * cc, indexing="ij")
        native = map_coordinates(
            data, [gy, gx], order=3, mode="constant", cval=0.0
        )
    logger.info(
        "ZOGY: extracted native %dx%d PSF stamp from %dx-oversampled ePSF %s.",
        int(native.shape[0]), int(native.shape[1]), k, os.path.basename(fpath),
    )
    return np.asarray(native, dtype=float)


def _pick_psf_file(files, image_fpath):
    files = list(files or [])
    if not files:
        return None
    base = os.path.splitext(os.path.basename(str(image_fpath)))[0]
    exact = [
        path
        for path in files
        if os.path.splitext(os.path.basename(path))[0].endswith("_" + base)
    ]
    candidates = exact or files
    try:
        return max(candidates, key=os.path.getmtime)
    except OSError:
        return candidates[0]


def _measure_stamp_fwhm(stamp: np.ndarray) -> float:
    """FWHM of a centred PSF stamp from its azimuthal median profile.

    Half-pixel radial bins (median per bin) with linear interpolation of
    the half-maximum crossing.  Median binning matters on ePSF stamps:
    they are often flat-topped or asymmetric, and a coarse mean profile
    under-measures them non-uniformly -- enough to flip the ZOGY AUTO
    convolution direction.  Returns NaN when the stamp is unusable
    (non-finite peak, flat profile, or the crossing lies outside).
    """
    s = np.asarray(stamp, dtype=float)
    if s.ndim != 2 or s.size == 0 or not np.isfinite(s).any():
        return float("nan")
    peak_i = np.nanargmax(s)
    peak = float(s.flat[peak_i])
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    py, px = np.unravel_index(peak_i, s.shape)
    yy, xx = np.mgrid[0 : s.shape[0], 0 : s.shape[1]]
    r = np.hypot(yy - py, xx - px)
    good = np.isfinite(s)
    r_i = (r[good] * 2.0).astype(int).ravel()
    v = s[good].ravel()
    n_bins = int(r_i.max()) + 1
    # Median per half-pixel bin; a mean is dragged by asymmetric wings.
    # Lexsort by (bin, value) gives every bin pre-sorted, so all medians
    # come from index arithmetic -- a per-bin mask loop costs minutes on
    # image-sized stamps (ZOGY kernel probes run at full image shape).
    order = np.lexsort((v, r_i))
    ri_s = r_i[order]
    v_s = v[order]
    seg_a = np.flatnonzero(np.r_[True, np.diff(ri_s) != 0])
    seg_b = np.r_[seg_a[1:], ri_s.size]
    lo = (seg_a + seg_b - 1) // 2
    hi = (seg_a + seg_b) // 2
    prof = np.full(n_bins, np.nan)
    prof[ri_s[seg_a]] = 0.5 * (v_s[lo] + v_s[hi])
    prof[0] = peak
    half = 0.5 * peak
    finite = np.isfinite(prof)
    if not finite[0]:
        return float("nan")
    below = np.nonzero(finite & (prof < half))[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    # Back up to the previous finite bin so gaps in the profile do not
    # break the interpolation.
    j = i - 1
    while j >= 0 and not finite[j]:
        j -= 1
    if j < 0:
        return float("nan")
    p0, p1 = float(prof[j]), float(prof[i])
    if not p1 < p0:
        return float("nan")
    # Floor binning puts bin k over r in [0.5k, 0.5k+0.5), so each profile
    # value is evaluated at the bin centre (+0.25 px) -- skipping the
    # offset under-measures every FWHM by ~0.5 px.
    r_cross = 0.5 * (j + (half - p0) / (p1 - p0) * (i - j)) + 0.25
    return 2.0 * r_cross


# Negative-mass fraction of a unit-normalized Wiener PSF-matching kernel
# above which the convolution is treated as deconvolution.  Pure
# smoothing kernels carry ~1e-3; observed sharpening kernels reach 0.5-0.9.
_ZOGY_KERNEL_NEG_TOL = 0.05


def _zogy_wiener_kernel(source_psf, target_psf, eps_rel=1e-2):
    """PSF-matching kernel that convolves ``source_psf`` onto ``target_psf``.

    Wiener regularization (``eps_rel`` times the source's peak spectral
    power) keeps high-frequency noise from blowing up the kernel.

    Returns ``(kernel, neg_frac)``: the unit-normalized kernel and its
    absolute negative mass.  ``neg_frac`` is the direct signature of a
    sharpening kernel -- a smoothing kernel is nearly all-positive, while
    deconvolving kernels carry large negative sidelobes that leave dipole
    residuals at bright sources.
    """
    src_hat = np.fft.fft2(source_psf)
    src_abs2 = np.abs(src_hat) ** 2
    eps_w = float(eps_rel) * float(np.max(src_abs2))
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.real(
            np.fft.ifft2(
                np.fft.fft2(target_psf)
                * np.conj(src_hat)
                / (src_abs2 + eps_w)
            )
        )
    ksum = float(np.sum(kernel))
    if not np.isfinite(ksum) or abs(ksum) <= 1e-30:
        return kernel, np.inf
    kernel = kernel / ksum
    return kernel, float(-np.sum(kernel[kernel < 0]))


# Regularization grid for the adaptive Wiener kernel: the weakest damping
# that still produces a clean kernel is used first, because stronger eps
# broadens the kernel beyond the target PSF and leaves monopole residuals.
_ZOGY_WIENER_EPS_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _zogy_wiener_kernel_adaptive(source_psf, target_psf,
                                 neg_tol=_ZOGY_KERNEL_NEG_TOL,
                                 half_width=0, fid_tol=None,
                                 neg_cap=None):
    """Wiener kernel at the damping level that best reproduces the target.

    Steps up the eps grid; for each candidate the realised convolution
    ``source * kernel`` is compared against the target width (when
    ``fid_tol`` is given).  Sidelobe cleanliness alone is not
    sufficient: damping strong enough to clean a mismatched PSF pair
    (eps ~ 1, the correlation-kernel limit) over-broadens the result by
    ~20% and leaves positive-core/negative-ring residuals at every
    source (GROND TDP4 59585 r).  A candidate that misses the target
    width is unusable no matter how clean it is.

    Preference order: the faithful candidate with the smallest width
    error whose windowed sidelobe mass stays under ``neg_cap`` - the
    residuals scale with the width error, so between two admissible
    candidates fidelity wins over cleanliness (TDP6 60881 r: eps=1e-3
    at 3% width error / 0.19 sidelobes beats eps=1e-2 at 11% / 0.125);
    failing that, the least-negative faithful candidate (the caller's
    sidelobe screen judges whether it may be applied); failing that,
    the least-negative candidate overall so the veto still sees the
    honest diagnostic.

    Returns ``(kernel, neg_frac, eps_rel)``.
    """
    faithful = (None, np.inf, _ZOGY_WIENER_EPS_GRID[0])
    fid_best = (None, np.inf, np.inf, _ZOGY_WIENER_EPS_GRID[0])
    best = (None, np.inf, _ZOGY_WIENER_EPS_GRID[-1])
    for eps in _ZOGY_WIENER_EPS_GRID:
        try:
            k, neg = _zogy_wiener_kernel(
                source_psf, target_psf, eps_rel=eps
            )
        except Exception:
            continue
        if k is None or not np.isfinite(k).all() or not np.isfinite(neg):
            continue
        if neg < best[1]:
            best = (k, neg, eps)
        if fid_tol is not None:
            kw = (
                _zogy_window_kernel(k, half_width)
                if half_width
                else k
            )
            fe = _zogy_kernel_fid_err(source_psf, kw, target_psf)
            if not np.isfinite(fe) or fe > fid_tol:
                # This damping level no longer convolves the source
                # onto the target; skip it as an applied candidate.
                continue
            if neg < faithful[1]:
                faithful = (k, neg, eps)
            # The cap is judged on the windowed kernel - the part that
            # actually prints onto the image.
            neg_w = float(-np.nansum(kw[kw < 0])) if kw is not None else neg
            cap = neg_cap if neg_cap is not None else np.inf
            if (
                np.isfinite(neg_w)
                and neg_w <= cap
                and (fe, neg, eps) < (fid_best[2], fid_best[1], fid_best[3])
            ):
                fid_best = (k, neg, fe, eps)
            if neg <= neg_tol:
                return k, neg, eps
        elif neg <= neg_tol:
            return k, neg, eps
    if fid_tol is not None:
        if fid_best[0] is not None:
            return fid_best[0], fid_best[1], fid_best[3]
        if faithful[0] is not None:
            return faithful
    return best


# Negative-mass fraction (of the *unwindowed* Wiener kernel) above which a
# direction is treated as inherently deconvolving even when the FWHM
# ordering claims otherwise.  A near-delta kernel (matched PSFs) carries
# ~0.5 of sinc sidelobes and noisy broadening kernels reach ~0.6, while
# genuine sharpening kernels carry >=0.8 (TDP6_12: ~0.85; SN2024pba
# 2025-02-22 r: 1.38).
_ZOGY_KERNEL_NEG_HARD = 0.7
_ZOGY_NEAR_MATCH_KERNEL_NEG_LIMIT = 0.25
# Sidelobe ceiling for a kernel adopted through a veto override: the
# flipped-to direction must be trustworthy in its own right, not merely
# cleaner than the vetoed one.  Wiener kernels adopted this way are held
# to _ZOGY_KERNEL_NEG_TOL (a genuine smoother is ~1e-3); an LSQ kernel may
# carry legitimate shift-absorption mass, documented at ~0.05-0.3 -- the
# 0.25-0.39 kernels applied on J2344 r-band (2026-10) left sombrero
# residuals and ~1.0 dipole fractions.
_ZOGY_FLIP_LSQ_NEG_CAP = 0.30


def _zogy_taper_stamp(stamp, fwhm, r_in_f=2.0, r_out_f=4.0):
    """Cosine-taper a PSF stamp's wings for matching-kernel construction.

    Empirical stamps (ePSF models, star stacks) carry noise in their
    outskirts that the Fourier-space Wiener kernel turns into far-field
    sidelobes -- the dipole residuals seen around bright stars.  Zeroing
    the stamp beyond ~r_out_f FWHM with a raised-cosine taper starting at
    ~r_in_f FWHM keeps the informative core while suppressing that noise.

    Applied ONLY to the stamps used to build the matching kernel; the
    PSFs entering the ZOGY filter itself are left untapered.  Returns the
    (re-normalised) stamp; non-finite or unmeasurable inputs pass through
    unchanged.
    """
    s = np.asarray(stamp, dtype=float)
    if s.ndim != 2 or not np.isfinite(s).any():
        return s
    try:
        f = float(fwhm)
    except (TypeError, ValueError):
        return s
    if not (np.isfinite(f) and f > 0):
        return s
    ny, nx = s.shape
    py, px = np.unravel_index(np.nanargmax(s), s.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - py, xx - px)
    r_in = r_in_f * f
    r_out = max(r_out_f * f, r_in + 1.0)
    if r_in >= 0.5 * min(ny, nx):
        # The whole stamp sits inside the flat part of the taper.
        return s
    w = np.ones_like(s)
    band = (r > r_in) & (r < r_out)
    w[band] = 0.5 * (1.0 + np.cos(np.pi * (r[band] - r_in) / (r_out - r_in)))
    w[r >= r_out] = 0.0
    out = s * w
    # NaN borders are routine on photutils ePSF stamps (NaN written
    # outside the support region): they land where the taper attenuates
    # anyway, and would otherwise force the kernel probe to refuse the
    # stamp.  Non-finite pixels in the untouched core (w == 1) are left
    # alone so a genuinely corrupt stamp still poisons the kernel and is
    # refused.
    out[~np.isfinite(out) & (w < 1.0)] = 0.0
    tot = float(np.nansum(out))
    if np.isfinite(tot) and tot > 0:
        out = out / tot
    return out


def _zerolag_stamp_fwhm(arr):
    """FWHM of a PSF wherever its peak sits in a periodic layout.

    Kernel probes work on zero-lag (ifftshifted) padded arrays, where
    the PSF is split across the four array corners and a plain radial
    profile about the argmax mixes quarter-arcs at different radii -
    under-reading the width by ~25% (the observed "convolved 12 px vs
    target 5 px" fidelity veto on TDP6 60881 r).  Rolling the argmax to
    the array centre reassembles the PSF before measuring and is a
    no-op for already-centred stamps.
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).any():
        return float("nan")
    py, px = np.unravel_index(np.nanargmax(a), a.shape)
    a = np.roll(
        a, (a.shape[0] // 2 - py, a.shape[1] // 2 - px), axis=(0, 1)
    )
    return _measure_stamp_fwhm(a)


def _zogy_kernel_fid_err(source_psf, kernel, target_psf):
    """Fractional width error of a matching kernel's realised result.

    Convolves ``source_psf`` with ``kernel`` (both padded, FFT-order)
    and compares the convolved FWHM against ``target_psf``.  A kernel
    whose only acceptable damping level destroys the convolution -
    e.g. eps ~ 1 collapses the Wiener ratio into the correlation
    kernel, over-broadening by ~20% - passes the negative-mass screen
    while no longer doing the job, which prints a positive-core /
    negative-ring residual at every source.  Returns NaN when either
    width is unmeasurable.
    """
    try:
        conv = np.real(
            np.fft.ifft2(
                np.fft.fft2(source_psf) * np.fft.fft2(kernel)
            )
        )
    except Exception:
        return float("nan")
    f_c = _zerolag_stamp_fwhm(conv)
    f_t = _zerolag_stamp_fwhm(target_psf)
    if not (np.isfinite(f_c) and np.isfinite(f_t) and f_t > 0):
        return float("nan")
    return abs(f_c - f_t) / f_t


def _zogy_window_kernel(kernel, half_width):
    """Truncate a padded matching kernel to compact support around its peak.

    The Wiener kernel spans the whole padded image; its far-field
    sidelobes are regularisation noise that print dipole residuals around
    stars.  Keeping only the +-half_width neighbourhood of the peak
    (wrap-safe, since the zero-lag peak sits at a corner in the padded
    convention) and renormalising to unit sum removes them without
    touching the PSF-matching core.
    """
    k = np.asarray(kernel, dtype=float)
    if k.ndim != 2 or not np.isfinite(k).all():
        return k
    try:
        hw = int(half_width)
    except (TypeError, ValueError):
        return k
    if hw <= 0:
        return k
    ny, nx = k.shape
    py, px = np.unravel_index(np.argmax(k), k.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    dy = np.minimum(np.abs(yy - py), ny - np.abs(yy - py))
    dx = np.minimum(np.abs(xx - px), nx - np.abs(xx - px))
    out = np.where((dy <= hw) & (dx <= hw), k, 0.0)
    s = float(np.sum(out))
    if np.isfinite(s) and abs(s) > 1e-30:
        out = out / s
    return out


def _zogy_lsq_fit_positions(matching_sources, star_pos, star_ref_pos):
    """Positions to feed the pixel-domain LSQ kernel fit.

    A matched position that failed the star-stack vetting in either
    frame (blank sky, flat-topped, CR spike) gives the solve a garbage
    stamp, and the resulting overfit kernel is exactly the ringy map
    the sidelobe cap then vetoes.  Fit on positions that survived in
    BOTH frames; when only one side's vetting ran, use it; only an
    unvetted matched list is used raw.  A too-small vetted set leaves
    the solver's own min_stamps gate to decline the fit rather than
    reintroducing rejected stamps.
    """
    out = []
    for msrc in matching_sources or []:
        try:
            mxy = np.asarray([msrc["x"], msrc["y"]], dtype=float)
        except (TypeError, IndexError, KeyError):
            try:
                mxy = np.asarray(msrc, dtype=float)
            except (TypeError, ValueError):
                continue
        if mxy.size >= 2 and np.isfinite(mxy[:2]).all():
            out.append((float(mxy[0]), float(mxy[1])))
    if star_pos is None and star_ref_pos is None:
        return out
    for vlist in (star_pos, star_ref_pos):
        if vlist is None:
            continue
        vset = {
            (round(float(p[0]), 1), round(float(p[1]), 1))
            for p in vlist
        }
        out = [
            p for p in out
            if (round(p[0], 1), round(p[1], 1)) in vset
        ]
    return out


def _zogy_lsq_kernel(
    source_img,
    target_img,
    positions,
    half_width,
    invalid_mask=None,
    lam_rel=1e-3,
    min_stamps=3,
    max_stamps=60,
    margin=None,
    rej_clip=4.0,
    neg_cap=None,
):
    """Solve the matching kernel by regularized least squares on pixels.

    Bramich (2008) delta-basis approach: every kernel pixel is a free
    parameter and ``target ~= (K * source) + b0`` is fitted on stamps
    around ``positions`` (0-based x, y on the shared resampled grid).
    A Laplacian Tikhonov penalty on the kernel grid damps the unconstrained
    high-frequency noise an unregularized pixel kernel parks in its wings
    (Hu & Wang 2024; Becker et al. 2012), and the kernel centre is exempt
    from the penalty so the delta-like peak is not shrunk.  Because the
    kernel is fitted to image pixels rather than a PSF ratio, it absorbs
    sub-pixel misregistration -- the residual dipoles a Wiener kernel
    cannot represent.

    ``invalid_mask`` (True = unusable) removes residual pixels and rejects
    residual pixels whose kernel footprint overlaps mask on >50% of its
    pixels, so filled/masked regions do not steer the fit.  A single
    re-solve pass drops stamps whose residual RMS is a ``rej_clip``-MAD
    outlier (unknown variables/neighbours hiding in the source list).

    The raw kernel sum is a free photometric factor (SFFT's
    ConstPhotRatio equivalent): it is checked for sanity and returned in
    ``info["ksum_raw"]`` so the caller can fold it into the flux scale;
    the returned kernel keeps the raw sum.

    Returns ``(kernel, info)``: ``kernel`` is the centred (nk, nk) array
    or None when the solve fails its sanity gates; ``info`` carries
    ``b0`` (differential background to add to the convolved image),
    ``neg`` (negative mass of the unit-normalized kernel), ``ksum_raw``,
    ``lam_eff``, ``n_stamps``, ``n_pix``, and ``resid_ratio``.
    """
    info = {
        "ok": False,
        "b0": 0.0,
        "neg": np.inf,
        "ksum_raw": np.nan,
        "lam_eff": 0.0,
        "n_stamps": 0,
        "n_pix": 0,
        "resid_ratio": np.nan,
    }
    src = np.asarray(source_img, dtype=np.float64)
    tgt = np.asarray(target_img, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2:
        return None, info
    try:
        hw = int(half_width)
    except (TypeError, ValueError):
        return None, info
    if hw < 2:
        return None, info
    nk = 2 * hw + 1
    nk2 = nk * nk
    if margin is None:
        margin = hw
    sh = hw + int(margin)
    ny, nx = src.shape
    if ny <= 2 * (sh + hw) or nx <= 2 * (sh + hw):
        return None, info

    mask = None
    if invalid_mask is not None:
        try:
            mask = np.asarray(invalid_mask, dtype=bool)
        except (TypeError, ValueError):
            mask = None
        if mask is not None and mask.shape != src.shape:
            mask = None

    pts = []
    if positions is None:
        positions = []
    for pos in positions:
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, IndexError, ValueError, KeyError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cy, cx = int(round(y)), int(round(x))
        if (
            cy - sh - hw < 0
            or cy + sh + hw >= ny
            or cx - sh - hw < 0
            or cx + sh + hw >= nx
        ):
            continue
        pts.append((cy, cx))
    if len(pts) > max_stamps:
        step = int(np.ceil(len(pts) / max_stamps))
        pts = pts[::step][:max_stamps]
    if len(pts) < min_stamps:
        return None, info

    from numpy.lib.stride_tricks import sliding_window_view

    def _stamp_rows(cy, cx):
        """Design rows for one stamp; each row maps the source stencil
        to one residual pixel of the target stamp."""
        reg = src[cy - sh - hw : cy + sh + hw + 1,
                  cx - sh - hw : cx + sh + hw + 1]
        tst = tgt[cy - sh : cy + sh + 1, cx - sh : cx + sh + 1]
        win = sliding_window_view(reg, (nk, nk))
        ok = np.isfinite(tst)
        if mask is not None:
            ok &= ~mask[cy - sh : cy + sh + 1, cx - sh : cx + sh + 1]
            mreg = mask[
                cy - sh - hw : cy + sh + hw + 1,
                cx - sh - hw : cx + sh + hw + 1,
            ].astype(np.float32)
            mfrac = sliding_window_view(mreg, (nk, nk)).mean(axis=(2, 3))
            ok &= mfrac < 0.5
        ok &= np.isfinite(win).all(axis=(2, 3))
        if not ok.any():
            return None
        return win[ok].reshape(-1, nk2), tst[ok].ravel()

    n_aug = nk2 + 1  # +1 for the differential background term b0

    # Laplacian penalty on the kernel grid; the centre pixel is exempt so
    # the penalty shrinks sidelobe noise, not the delta-like peak.
    Lap = np.zeros((nk2, nk2))
    for a in range(nk):
        for b in range(nk):
            i = a * nk + b
            if a == hw and b == hw:
                continue
            Lap[i, i] = 4.0
            if a > 0:
                Lap[i, i - nk] = -1.0
            if a < nk - 1:
                Lap[i, i + nk] = -1.0
            if b > 0:
                Lap[i, i - 1] = -1.0
            if b < nk - 1:
                Lap[i, i + 1] = -1.0
    P = Lap.T @ Lap

    def _solve(pt_list, lam):
        C = np.zeros((n_aug, n_aug))
        d = np.zeros(n_aug)
        n_rows = 0
        for cy, cx in pt_list:
            rows = _stamp_rows(cy, cx)
            if rows is None:
                continue
            A, y = rows
            C[:nk2, :nk2] += A.T @ A
            C[:nk2, nk2] += A.sum(axis=0)
            C[nk2, nk2] += A.shape[0]
            d[:nk2] += A.T @ y
            d[nk2] += y.sum()
            n_rows += A.shape[0]
        if n_rows < 5 * nk2:
            return None, 0, 0.0
        C[nk2, :nk2] = C[:nk2, nk2]
        trace = float(np.trace(C[:nk2, :nk2]))
        if not np.isfinite(trace) or trace <= 0:
            return None, n_rows, 0.0
        lam_eff = float(lam) * trace / nk2
        M = C.copy()
        M[:nk2, :nk2] += lam_eff * P
        M += 1e-10 * trace / nk2 * np.eye(n_aug)
        try:
            x = np.linalg.solve(M, d)
        except np.linalg.LinAlgError:
            try:
                x = np.linalg.lstsq(M, d, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None, n_rows, 0.0
        return x, n_rows, lam_eff

    def _finalize(x, n_pix, lam_eff, lam):
        """Rejection re-solve, kernel sanity gates, info fill."""
        # One rejection pass: a stamp dominated by a masked neighbour, an
        # unlisted variable, or a resampling edge inflates its residual RMS;
        # refit without it rather than let it bend the global kernel.
        kept = pts
        res_rms = []
        for cy, cx in pts:
            rows = _stamp_rows(cy, cx)
            if rows is None:
                res_rms.append(np.inf)
                continue
            A, y = rows
            res = y - A @ x[:nk2] - x[nk2]
            res_rms.append(float(np.std(res)))
        res_rms = np.asarray(res_rms)
        finite_rms = res_rms[np.isfinite(res_rms)]
        if finite_rms.size >= max(min_stamps + 1, 4):
            med = float(np.median(finite_rms))
            mad = 1.4826 * float(np.median(np.abs(finite_rms - med)))
            cut = med + max(rej_clip * mad, 0.5 * med)
            kept = [
                p for p, r in zip(pts, res_rms)
                if np.isfinite(r) and r <= cut
            ]
            if len(kept) >= min_stamps and len(kept) < len(pts):
                x, n_pix, lam_eff = _solve(kept, lam)
                if x is None:
                    return None, info

        k = np.asarray(x[:nk2], dtype=np.float64).reshape(nk, nk)
        b0 = float(x[nk2])
        if not np.isfinite(k).all() or not np.isfinite(b0):
            return None, info
        ksum = float(k.sum())
        i2 = dict(info)
        i2["ksum_raw"] = ksum
        i2["n_pix"] = int(n_pix)
        i2["n_stamps"] = len(kept)
        i2["b0"] = b0
        i2["lam_eff"] = lam_eff
        # A kernel sum far from unity means the solver is compensating a
        # broken flux scale or a wrong direction, not matching PSFs.
        if not np.isfinite(ksum) or ksum < 0.25 or ksum > 4.0:
            return None, i2
        neg = float(-np.sum(k[k < 0])) / abs(ksum)
        i2["neg"] = neg
        i2["resid_ratio"] = float(
            np.median(res_rms[np.isfinite(res_rms)]) / np.nanstd(tgt)
        ) if np.nanstd(tgt) > 0 else np.nan
        i2["ok"] = True
        return k, i2

    # When neg_cap is set the solve escalates the damping until the
    # kernel's sidelobe mass fits under it - a few-stamp ensemble needs
    # far more damping than the fiducial lam_rel (TDP6 60881 r: neg 1.7
    # at lam=1e-3, 0.2 at lam=0.1).  The first satisfying level is the
    # weakest damping that stays clean; if none qualifies, the
    # least-negative attempt is returned so the caller's veto reports an
    # honest diagnostic.
    try:
        _cap = float(neg_cap) if neg_cap is not None else None
    except (TypeError, ValueError):
        _cap = None
    lams = [float(lam_rel)]
    if _cap is not None:
        lams += [float(lam_rel) * m for m in (10.0, 100.0, 1000.0)]
    best = (None, np.inf, dict(info))
    for lam_eff_rel in lams:
        x, n_pix, lam_eff = _solve(pts, lam_eff_rel)
        if x is None:
            continue
        k, i2 = _finalize(x, n_pix, lam_eff, lam_eff_rel)
        if k is None:
            continue
        i2["lam_rel"] = lam_eff_rel
        if i2["neg"] < best[1]:
            best = (k, i2["neg"], i2)
        if _cap is None or i2["neg"] <= _cap:
            return k, i2
    if best[0] is not None:
        return best[0], best[2]
    return None, info


def _pack_lsq_kernel(kernel, shape):
    """Pack a centred (nk, nk) LSQ kernel into the FFT convolution layout.

    The solver's convention is ``target(p) ~= sum_d k[d] * source(p + d)``
    with d in +-hw -- a correlation.  ``ifft2(fft2(img) * fft2(Kpack))``
    computes a convolution (kernel reflected), so the packed array holds
    the flipped kernel with its centre at index (0, 0) and the negative
    offsets wrapped to the far edges, matching the packed Wiener-kernel
    layout used elsewhere in this module.
    """
    k = np.asarray(kernel, dtype=np.float64)
    nk = k.shape[0]
    hw = nk // 2
    kpack = np.zeros(shape)
    iy = (np.arange(nk) - hw) % shape[0]
    ix = (np.arange(nk) - hw) % shape[1]
    kpack[np.ix_(iy, ix)] = k[::-1, ::-1]
    return kpack


def _zogy_subtract(
    N,
    R,
    Pn,
    Pr,
    sn,
    sr,
    fn=1.0,
    fr=None,
    sn_map=None,
    sr_map=None,
    nan_mask=None,
    gain_n=1.0,
    gain_r=1.0,
    source_var_n=None,
    source_var_r=None,
):
    """Core ZOGY subtraction (Zackay, Ofek & Gal-Yam 2016, ApJ, 830, 27).

    Computes the proper difference image D, the score image S, and the
    corrected significance Scorr using Equations 1-13 from the paper.
    Uses only numpy FFTs -- no pyfftw or external dependencies.

    Parameters
    ----------
    N : np.ndarray
        Science (new) image, background-subtracted.
    R : np.ndarray
        Reference image, background-subtracted.
    Pn : np.ndarray
        Science PSF, same shape as N (use _pad_psf_to_image).
    Pr : np.ndarray
        Reference PSF, same shape as R.
    sn : float
        Science background noise RMS (scalar, used if sn_map is None).
    sr : float
        Reference background noise RMS (scalar, used if sr_map is None).
    fn : float
        Science flux-based zero point (default 1.0).
    fr : float or None
        Reference flux-based zero point. If None, set to fn (same
        instrument/filter).
    sn_map, sr_map : np.ndarray or None
        Optional per-pixel noise RMS maps.  When provided, the variance
        images Vn/Vr use per-pixel variance (sn_map**2 / sr_map**2)
        instead of the scalar sn**2 / sr**2, following the ZOGY paper
        more closely for spatially varying noise (e.g. near chip gaps,
        bright galaxy backgrounds).
    nan_mask : np.ndarray(bool) or None
        Optional boolean mask of pixels to zero in both images (NaN
        regions).  When None, falls back to the (R==0)|(N==0) heuristic.
    gain_n, gain_r : float
        Detector gains in electrons per ADU for N and R.
    source_var_n, source_var_r : np.ndarray or None
        Optional source Poisson variance maps in the respective image units.
        Use these when a source image was pre-convolved, since its variance
        must be convolved with the squared matching kernel.

    Returns
    -------
    D : np.ndarray (float)
        Proper difference image (Eq. 1), in science-image flux units.
    S : np.ndarray (float)
        Score image (Eq. 2).
    Scorr : np.ndarray (float)
        Corrected significance (Eq. 13), with astrometric noise ignored
        (approximation valid for well-aligned images).
    P_D : np.ndarray (float)
        PSF of the difference image (Eq. 12), in native image pixels,
        same shape as D.  Used for photometry when forceconv=AUTO
        (geometric-mean PSF, no pre-convolution).
    """
    if fr is None:
        fr = fn
    if not (np.isfinite(fn) and fn > 0) or not (np.isfinite(fr) and fr > 0):
        raise ValueError(
            f"ZOGY flux zero-points must be positive and finite (fn={fn}, fr={fr})"
        )
    try:
        gain_n = float(gain_n)
        gain_r = float(gain_r)
    except (TypeError, ValueError) as exc:
        raise ValueError("ZOGY gains must be positive and finite") from exc
    if not (
        np.isfinite(gain_n)
        and gain_n > 0
        and np.isfinite(gain_r)
        and gain_r > 0
    ):
        raise ValueError("ZOGY gains must be positive and finite")

    N = np.asarray(N, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    Pn = np.asarray(Pn, dtype=np.float64)
    Pr = np.asarray(Pr, dtype=np.float64)

    # NaNs in a PSF stamp would propagate through fft2 and silently
    # produce an all-NaN difference image.  Zero them before use.
    if not np.isfinite(Pn).all():
        Pn = np.nan_to_num(Pn, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(Pr).all():
        Pr = np.nan_to_num(Pr, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize PSFs to unit sum (ZOGY assumes normalized PSFs)
    _pn_sum = float(np.nansum(Pn))
    _pr_sum = float(np.nansum(Pr))
    if _pn_sum > 0:
        Pn = Pn / _pn_sum
    if _pr_sum > 0:
        Pr = Pr / _pr_sum

    # Zero out non-overlapping / NaN regions
    if nan_mask is not None:
        # Include any non-finite input pixels too: a stray NaN outside the
        # caller's mask would otherwise poison every output pixel through
        # the FFTs below.
        mask_zero = (
            np.asarray(nan_mask, dtype=bool)
            | ~np.isfinite(N)
            | ~np.isfinite(R)
        )
    else:
        # Non-finite pixels must always be masked here: an unmasked NaN
        # would poison every output pixel through the FFTs below.
        mask_zero = ~np.isfinite(N) | ~np.isfinite(R)
        z = (N == 0) | (R == 0)
        if np.any(z):
            # Resampling pads uncovered regions with *contiguous* exact
            # zeros, while a genuine zero-flux pixel (e.g. integer data
            # minus an integer sky) is isolated.  Only mask zeros
            # embedded in a mostly-zero neighbourhood so real zeros are
            # not nulled along with the other image's flux at those
            # pixels.  The dilation catches the thin partial-zero rim
            # around a padded block whose 5x5 window is <60% zero.
            z_neighbour = uniform_filter(
                z.astype(np.float32), size=5, mode="nearest"
            )
            z_block = binary_dilation(z_neighbour > 0.6, iterations=2)
            mask_zero |= z & z_block
    N = np.where(mask_zero, 0.0, N)
    R = np.where(mask_zero, 0.0, R)

    sn2 = sn ** 2
    sr2 = sr ** 2
    fn2 = fn ** 2
    fr2 = fr ** 2

    # FFTs (use numpy FFT -- no pyfftw dependency)
    N_hat = np.fft.fft2(N)
    R_hat = np.fft.fft2(R)
    Pn_hat = np.fft.fft2(Pn)
    Pr_hat = np.fft.fft2(Pr)

    Pn_hat2_abs = np.abs(Pn_hat) ** 2
    Pr_hat2_abs = np.abs(Pr_hat) ** 2

    denominator = (sn2 * fr2) * Pr_hat2_abs + (sr2 * fn2) * Pn_hat2_abs

    # Guard against a degenerate denominator (e.g. sn = sr = 0, or PSFs with
    # no Fourier power): the relative floor would be zero and the division
    # below would silently produce NaN/inf everywhere.
    _denom_max = float(np.max(denominator))
    if not np.isfinite(_denom_max) or _denom_max <= 0:
        raise ValueError(
            "ZOGY denominator is degenerate (sn=%.4g, sr=%.4g); "
            "check the noise estimates and PSF models." % (sn, sr)
        )

    # Avoid division by zero / noise amplification at high frequencies.
    # A relative floor (not absolute 1e-30) prevents the denominator from
    # going to near-zero at high spatial frequencies where both PSFs have
    # low power, which would amplify noise in D_hat and P_D_hat.
    _denom_floor = 1e-12 * _denom_max
    denominator = np.where(denominator < _denom_floor, _denom_floor, denominator)

    fD = (fr * fn) / np.sqrt(sn2 * fr2 + sr2 * fn2)

    # Difference image D (Eq. 1)
    D_hat = (fr * (Pr_hat * N_hat) - fn * (Pn_hat * R_hat)) / np.sqrt(denominator)
    D = np.real(np.fft.ifft2(D_hat)) / fD

    # PSF of the difference image
    P_D_hat = (fr * fn / fD) * (Pr_hat * Pn_hat) / np.sqrt(denominator)

    # Score image S (Eq. 2)
    S_hat = fD * D_hat * np.conj(P_D_hat)
    S = np.real(np.fft.ifft2(S_hat))

    # Variance images for Scorr (Eqs. 25-31)
    kr_hat = (fr * fn2) * np.conj(Pr_hat) * Pn_hat2_abs / denominator
    kr = np.real(np.fft.ifft2(kr_hat))
    kr2 = kr ** 2
    kr2_hat = np.fft.fft2(kr2)

    kn_hat = (fn * fr2) * np.conj(Pn_hat) * Pr_hat2_abs / denominator
    kn = np.real(np.fft.ifft2(kn_hat))
    kn2 = kn ** 2
    kn2_hat = np.fft.fft2(kn2)

    # Source Poisson variance is in ADU squared and depends on detector gain.
    # Supplied maps preserve the variance after matching-kernel convolution.
    if source_var_n is None:
        source_var_n = np.maximum(N, 0.0) / gain_n
    else:
        source_var_n = np.asarray(source_var_n, dtype=np.float64)
        if source_var_n.shape != N.shape:
            raise ValueError("science source variance map shape does not match N")
        if not np.isfinite(source_var_n).all() or np.any(source_var_n < 0):
            raise ValueError(
                "science source variance map must be finite and nonnegative"
            )
        source_var_n = np.where(mask_zero, 0.0, source_var_n)
    if source_var_r is None:
        source_var_r = np.maximum(R, 0.0) / gain_r
    else:
        source_var_r = np.asarray(source_var_r, dtype=np.float64)
        if source_var_r.shape != R.shape:
            raise ValueError("reference source variance map shape does not match R")
        if not np.isfinite(source_var_r).all() or np.any(source_var_r < 0):
            raise ValueError(
                "reference source variance map must be finite and nonnegative"
            )
        source_var_r = np.where(mask_zero, 0.0, source_var_r)

    if sn_map is not None:
        Vn = source_var_n + np.asarray(sn_map, dtype=np.float64) ** 2
    else:
        Vn = source_var_n + sn2
    if sr_map is not None:
        Vr = source_var_r + np.asarray(sr_map, dtype=np.float64) ** 2
    else:
        Vr = source_var_r + sr2
    Vn_hat = np.fft.fft2(Vn)
    Vr_hat = np.fft.fft2(Vr)

    VSn = np.real(np.fft.ifft2(Vn_hat * kn2_hat))
    VSr = np.real(np.fft.ifft2(Vr_hat * kr2_hat))

    V_S = VSr + VSn
    V_S = np.where(V_S > 0, V_S, 1e-30)

    Scorr = S / np.sqrt(V_S)

    # Difference-image PSF (Eq. 12): needed for photometry when no
    # pre-convolution is applied (forceconv=AUTO).  P_D is in native
    # image pixels, same shape as D.
    P_D = np.real(np.fft.ifft2(P_D_hat))

    return D, S, Scorr, P_D


# =============================================================================
# Dataclass-Based Parameter Containers
# =============================================================================
# Dataclasses replace the former loose dicts: named fields, defaults in one
# place, and easy conversion to/from dicts for YAML compatibility.


@dataclass
class MaskParams:
    """Parameters controlling image mask creation."""

    saturation_level: float = 2**16
    """Pixel value above which a source is considered saturated."""

    saturate_frac: float = 0.90
    """Fraction of saturation_level above which a source is considered saturated (for non-linear regime avoidance)."""

    fwhm: int = 5
    """Full width at half-maximum of the PSF in pixels."""

    npixels: int = 8
    """Minimum number of connected pixels to be detected as a source."""

    padding: int = 10
    """Pixel padding around each masked source bounding box."""

    snr_limit: int = 3000
    """Signal-to-noise ratio ceiling (sources above this are flagged)."""

    detection_sigma: float = 5.0
    """Number of sigma above background to set the detection threshold."""

    local_bkg_width_factor: int = 15
    """Multiplied by FWHM to set local background annulus width."""


@dataclass
class SubtractionParams:
    """Parameters controlling image subtraction."""

    method: str = "sfft"
    """Algorithm: 'hotpants', 'sfft', or 'zogy'."""

    kernel_order: int = 0
    """Spatial kernel polynomial order for HOTPANTS / SFFT."""

    sfft_crowded_method: bool = True
    """If True, use SFFT crowded-field (ECP); False uses sparse (ESP). The subtract() path defaults to ESP for typical fields unless crowded_field is set."""

    sfft_bg_order: int = 0
    """SFFT background spatial polynomial order (0=constant). SFFT's BGPolyOrder handles the constant background difference between images; BACK_TYPE=MANUAL with BACK_VALUE=0.0 tells SExtractor not to subtract background."""

    zogy_template_psf_independent: bool = True
    """If True (default), select PSF stars for the reference image independently from the
    reference image (better quality reference PSF). If False, use the same matched stars
    for both science and reference PSFs (original behaviour; reference stars may be poor)."""

    hotpants_exe_loc: str = "hotpants"
    """Filesystem path to the HOTPANTS executable, or a command on PATH (default: 'hotpants')."""


@dataclass
class FluxMatchParams:
    """Parameters for flux-consistency matching between catalogs (outlier-resistant fit)."""

    flux_key: str = "flux_AP"
    """Column name for aperture flux."""

    flux_key_err: str = "flux_AP_err"
    """Column name for aperture flux uncertainty."""

    mag_residual_threshold: float = 0.3
    """Maximum allowed magnitude residual for an inlier (stricter for the outlier-resistant fit)."""

    min_samples_fraction: float = 0.7
    """Minimum fraction of sources required by RANSAC (increased for the outlier-resistant fit)."""

    max_trials: int = 2000
    """RANSAC maximum iteration count (increased for better sampling)."""

    min_absolute_samples: int = 3
    """Hard floor on the number of RANSAC samples (lowered to support sparse fields)."""

    use_percentile_cut: bool = False
    """Whether to apply a brightness percentile trim after RANSAC."""

    percentiles: Tuple[float, float] = (5.0, 99.0)
    """Lower and upper percentile bounds when percentile cut is active."""

    enforce_slope_constraint: bool = True
    """If True, penalise slopes far from 1.0 in the magnitude fit."""

    fix_slope_to_one: bool = True
    """If True, fix slope to 1 when fitting science vs reference (only fit intercept)."""

    use_spatial_thinning: bool = True
    """If True, apply spatial thinning to avoid over-clustered regions. Set False to use all available stars."""

    spatial_n_bins: int = 8
    """Number of bins per axis for spatial grid when use_spatial_thinning=True."""

    spatial_max_per_bin: int = 10
    """Maximum number of sources to keep per spatial bin when use_spatial_thinning=True."""


@dataclass
class ReprojectConfig:
    """Cached reproject configuration extracted once from input_yaml."""
    method: str = "exact"
    roundtrip: bool = True
    interp_order: Any = "bicubic"
    parallel: bool = True
    conserve_flux: bool = False
    center_jacobian: bool = False
    undersampled_fwhm_threshold: float = 2.5
    @classmethod
    def from_yaml(cls, input_yaml: Dict[str, Any]) -> "ReprojectConfig":
        cfg = input_yaml.get("alignment", {})
        phot_cfg = input_yaml.get("photometry", {}) or {}
        # reproject accepts an int worker count.  ``parallel=True`` would
        # claim os.cpu_count() workers inside every image-level worker, so
        # divide the machine by nCPU to avoid oversubscription.
        _parallel = cfg.get("reproject_parallel", True)
        if _parallel is True:
            _ncpu = max(1, int(input_yaml.get("nCPU", 1) or 1))
            _parallel = max(1, (os.cpu_count() or 1) // _ncpu)
        return cls(
            method=str(cfg.get("reproject_method", "exact")).lower().strip(),
            roundtrip=bool(cfg.get("reproject_roundtrip_coords", True)),
            interp_order=_normalize_reproject_interp_order(
                cfg.get("reproject_interp_order", "bicubic")
            ),
            parallel=_parallel,
            conserve_flux=bool(cfg.get("reproject_adaptive_conserve_flux", False)),
            center_jacobian=bool(cfg.get("reproject_adaptive_center_jacobian", False)),
            undersampled_fwhm_threshold=float(
                phot_cfg.get("undersampled_fwhm_threshold", 2.5)
            ),
        )


def _prepare_projection_header(header_in: fits.Header) -> fits.Header:
    """
    Build a projection header for reproject, preserving SIP/PV/TPV distortion
    keywords. Appends -SIP to TAN CTYPEs when SIP coefficients are present.
    """
    hdr = header_in.copy()
    sip_keys = ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")
    if any(k in hdr for k in sip_keys):
        for ctype_key in ("CTYPE1", "CTYPE2"):
            cval = str(hdr.get(ctype_key, ""))
            if cval and "TAN" in cval and not cval.endswith("-SIP"):
                hdr[ctype_key] = f"{cval}-SIP"
    return hdr


def _distortion_summary(header_in: fits.Header) -> str:
    """Return a compact string describing WCS distortion keywords present."""
    keys = list(header_in.keys())
    pv_count = sum(1 for k in keys if str(k).startswith("PV"))
    sip_count = sum(
        1 for k in keys
        if str(k).startswith(("A_", "B_", "AP_", "BP_", "SIP_"))
    )
    c1 = str(header_in.get("CTYPE1", ""))
    c2 = str(header_in.get("CTYPE2", ""))
    return f"CTYPE1={c1} CTYPE2={c2} PV={pv_count} SIP={sip_count}"


def _introspect_reproject_adaptive() -> Dict[str, Any]:
    """
    Inspect reproject_adaptive signature once at import time.
    Returns dict of supported optional kwargs for the installed version.
    """
    if reproject_adaptive is None:
        return {}
    import inspect as _inspect
    extras: Dict[str, Any] = {}
    try:
        sig = _inspect.signature(reproject_adaptive)
        params = sig.parameters
        if "conserve_flux" in params:
            extras["_has_conserve_flux"] = True
        if "center_jacobian" in params:
            extras["_has_center_jacobian"] = True
        if "despike_jacobian" in params:
            extras["_has_despike_jacobian"] = True
    except (TypeError, ValueError):
        pass
    return extras


# Introspect once at import time - not on every alignment call
_REPROJECT_ADAPTIVE_EXTRAS: Dict[str, Any] = _introspect_reproject_adaptive()


def _build_adaptive_kwargs(cfg: ReprojectConfig) -> Dict[str, Any]:
    """Build extra kwargs for reproject_adaptive from config and introspection cache."""
    kwargs: Dict[str, Any] = {}
    if _REPROJECT_ADAPTIVE_EXTRAS.get("_has_conserve_flux"):
        kwargs["conserve_flux"] = cfg.conserve_flux
    if _REPROJECT_ADAPTIVE_EXTRAS.get("_has_center_jacobian"):
        kwargs["center_jacobian"] = cfg.center_jacobian
    if _REPROJECT_ADAPTIVE_EXTRAS.get("_has_despike_jacobian"):
        kwargs["despike_jacobian"] = True
    return kwargs


def _detect_sextractor_sources(data_or_path, input_yaml=None, fwhm_pix=3.0,
                               thresh=2.0, fwhm_min=None, ell_max=0.5,
                               return_errors=False):
    """Detect sources using SExtractor.

    Accepts either a FITS file path (str) or an in-memory 2D array.
    For in-memory arrays, a temporary FITS file is written, SExtractor is
    run, and the temp file is cleaned up.

    Returns (xy, flux, fwhm) arrays or (None, ...) if no sources found.
    If ``return_errors=True``, returns (xy, flux, fwhm, errx, erry) where
    errx/erry are the 1-sigma centroid uncertainties decomposed from
    SExtractor's error ellipse (ERRAWIN/ERRBWIN/ERRTHETAWIN).

    All coordinates are 0-based (SExtractor's 1-based XWIN_IMAGE is
    converted to 0-based to match the rest of the codebase).

    This is the standard source detection for all alignment-related code,
    ensuring consistent centroiding across spalipy, compute_alignment_rms,
    and the alignment offset diagnostic plot.
    """
    _n_err = 5 if return_errors else 3
    _none = (None,) * _n_err
    # Lower fwhm_min for undersampled images: SExtractor FWHM_IMAGE can be
    # ~1.0 px for critically sampled data; the old fixed 1.5 px cut removed
    # real sources.
    if fwhm_min is None:
        fwhm_min = 0.8 if (fwhm_pix and fwhm_pix < 2.0) else 1.5
    try:
        from utils.run_sex import SExtractorWrapper
    except ModuleNotFoundError:
        return _none

    _tmp_path = None
    try:
        if isinstance(data_or_path, (str, os.PathLike)):
            fits_path = str(data_or_path)
        else:
            # SExtractor runs on files; write the array to a temp FITS.
            # NaN pixels are filled with the median - SExtractor can crash on NaN.
            import tempfile
            data = np.asarray(data_or_path, dtype=np.float32)
            if data.ndim != 2:
                return _none
            nan_mask = ~np.isfinite(data)
            if nan_mask.any():
                data = np.where(nan_mask, float(np.nanmedian(data)), data)
            _tmp_fd, _tmp_path = tempfile.mkstemp(suffix=".fits")
            os.close(_tmp_fd)
            from astropy.io import fits as _fits
            hdr = _fits.Header()
            hdr["NAXIS1"] = data.shape[1]
            hdr["NAXIS2"] = data.shape[0]
            _fits.PrimaryHDU(data, header=hdr).writeto(
                _tmp_path, overwrite=True, output_verify="silentfix+ignore",
            )
            fits_path = _tmp_path

        if input_yaml is None:
            input_yaml = {"fwhm": fwhm_pix, "saturate": 65000.0}

        sex = SExtractorWrapper(config=input_yaml)
        _fwhm, sources, _scale = sex.run(
            fits_path=fits_path,
            use_FWHM=fwhm_pix,
            return_raw=True,
            use_for_matching=True,
            detect_thresh=thresh,
        )

        if sources is None or len(sources) == 0:
            return _none

        # SExtractor positions are 1-based; convert to 0-based.
        x = np.asarray(sources["XWIN_IMAGE"], float) - 1.0
        y = np.asarray(sources["YWIN_IMAGE"], float) - 1.0
        flux = np.asarray(sources["FLUX_AUTO"], float)
        fwhm = np.asarray(sources["FWHM_IMAGE"], float)
        ell = np.asarray(sources["ELLIPTICITY"], float)

        good = (fwhm >= fwhm_min) & (fwhm <= 30) & (ell < ell_max) & np.isfinite(x) & np.isfinite(y)
        if not good.any():
            return _none

        xy = np.column_stack([x[good], y[good]])
        flux = flux[good]
        fwhm = fwhm[good]

        if return_errors:
            # Decompose SExtractor error ellipse into x/y sigma.
            # SExtractorWrapper guarantees these columns exist (with fallbacks),
            # but guard defensively in case of truncated catalogs.
            _erra_col = "ERRAWIN_IMAGE" if "ERRAWIN_IMAGE" in sources.colnames else None
            _errb_col = "ERRBWIN_IMAGE" if "ERRBWIN_IMAGE" in sources.colnames else None
            _errt_col = "ERRTHETAWIN_IMAGE" if "ERRTHETAWIN_IMAGE" in sources.colnames else None
            if _erra_col and _errb_col and _errt_col:
                erra = np.asarray(sources[_erra_col], float)[good]
                errb = np.asarray(sources[_errb_col], float)[good]
                theta = np.asarray(sources[_errt_col], float)[good]
                theta_rad = np.deg2rad(theta)
                cos_t = np.cos(theta_rad)
                sin_t = np.sin(theta_rad)
                errx = np.sqrt((erra * cos_t) ** 2 + (errb * sin_t) ** 2)
                erry = np.sqrt((erra * sin_t) ** 2 + (errb * cos_t) ** 2)
            else:
                # Fallback: estimate position error from FWHM (rough)
                _pos_err = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                errx = _pos_err
                erry = _pos_err
            return xy, flux, fwhm, errx, erry

        return xy, flux, fwhm

    except Exception as _det_err:
        logger.debug("_detect_sextractor_sources failed: %s", _det_err, exc_info=True)
        return _none
    finally:
        if _tmp_path is not None:
            try:
                os.unlink(_tmp_path)
            except OSError:
                pass


def compute_alignment_rms(
    sci_data: np.ndarray,
    ref_data: np.ndarray,
    fwhm_pixels: float,
    input_yaml: Optional[Dict[str, Any]] = None,
    sci_xy_override: Optional[np.ndarray] = None,
    return_per_quadrant: bool = False,
) -> Optional[tuple]:
    """
    Compute alignment quality from pre-loaded arrays.

    Uses SExtractor for source detection (consistent with the rest of the
    pipeline).  Mutual nearest-neighbour star matching resists false
    matches in crowded fields.

    If *sci_xy_override* is provided, science source detection is skipped
    and the supplied (N, 2) array of pixel positions is used instead.  This
    is critical for images with ghost/stacking artifacts: re-detecting
    sources in the science image picks up artifacts that create false
    matches and inflate the RMS.  By passing in pre-matched science
    sources (e.g. from spalipy's RA/DEC matching step), only real sources
    with known reference counterparts are used.

    Returns (median_offset, rms, p90) tuple in pixels, or None if measurement fails.

    If ``return_per_quadrant=True``, returns (median_offset, rms, p90,
    quadrant_rms) where ``quadrant_rms`` is a dict with keys 'q1_rms',
    'q2_rms', 'q3_rms', 'q4_rms', 'max_rms', 'max_quadrant'.
    The quadrants are defined as:
      Q1 = top-left,    Q2 = top-right,
      Q3 = bottom-left, Q4 = bottom-right
    """
    if sci_data.shape != ref_data.shape:
        return None

    try:
        fwhm = min(max(float(fwhm_pixels), 2.0), 8.0)

        # Adaptive detection threshold: lower for sparse fields to ensure
        # enough sources for reliable RMS measurement.  The first pass at
        # thresh=5.0 may find too few sources; fall back to 3.0 if needed.
        _align_thresh = 5.0
        if input_yaml is not None:
            _ts_cfg = input_yaml.get("template_subtraction", {}) or {}
            _align_thresh = float(_ts_cfg.get("alignment_rms_detect_thresh", 5.0) or 5.0)

        # request centroid errors where available; fakes patched in tests
        # may still return a 3-tuple, so unpack defensively.
        def _det_xy_err(data, thresh):
            _d = _detect_sextractor_sources(
                data, input_yaml=input_yaml, fwhm_pix=fwhm,
                thresh=thresh, return_errors=True,
            )
            _xy = _d[0]
            _ex = _d[3] if len(_d) >= 5 else None
            _ey = _d[4] if len(_d) >= 5 else None
            return _xy, _ex, _ey

        sci_ex = sci_ey = ref_ex = ref_ey = None
        if sci_xy_override is not None:
            sci_xy = np.asarray(sci_xy_override, float)
            if sci_xy.ndim != 2 or sci_xy.shape[1] != 2 or len(sci_xy) < 5:
                return None
        else:
            sci_xy, sci_ex, sci_ey = _det_xy_err(sci_data, _align_thresh)
        ref_xy, ref_ex, ref_ey = _det_xy_err(ref_data, _align_thresh)

        # Sparse-field fallback: retry at 3.0 when too few sources at thresh.
        if (sci_xy is None or len(sci_xy) < 10) and _align_thresh > 3.0 and sci_xy_override is None:
            sci_xy, sci_ex, sci_ey = _det_xy_err(sci_data, 3.0)
        if (ref_xy is None or len(ref_xy) < 10) and _align_thresh > 3.0:
            ref_xy, ref_ex, ref_ey = _det_xy_err(ref_data, 3.0)

        if sci_xy is None or ref_xy is None:
            return None
        if len(sci_xy) < 5 or len(ref_xy) < 5:
            return None

        max_sep = float(max(2.5, 2.5 * fwhm))
        tree_ref = cKDTree(ref_xy)
        tree_sci = cKDTree(sci_xy)

        d_sr, i_sr = tree_ref.query(sci_xy, k=1)
        d_rs, i_rs = tree_sci.query(ref_xy, k=1)

        idx_s = np.arange(len(sci_xy), dtype=int)
        mutual = (i_rs[i_sr] == idx_s) & np.isfinite(d_sr)
        if not np.any(mutual):
            return None

        d_mut = d_sr[mutual]
        _mut_mask = np.isfinite(d_mut) & (d_mut <= max_sep)
        d_mut = d_mut[_mut_mask]
        _mut_idx = np.where(_mut_mask)[0]
        if len(d_mut) < 10:
            # Wide-radius coherent-offset pass.  When few pairs survive the
            # nominal separation cut, "too few detections" (a genuinely
            # sparse field) must be distinguished from "no nearby
            # counterparts" (a coherent misregistration larger than
            # max_sep).  Mutual nearest neighbours whose pairwise offsets
            # cluster tightly around their median are evidence of a real
            # displacement; scattered offsets indicate unrelated sources.
            # Without this check a gross misregistration produced zero
            # surviving matches and was silently accepted as unverifiable.
            _dx_all = sci_xy[mutual, 0] - ref_xy[i_sr[mutual], 0]
            _dy_all = sci_xy[mutual, 1] - ref_xy[i_sr[mutual], 1]
            if len(_dx_all) >= 3:
                _mdx = float(np.nanmedian(_dx_all))
                _mdy = float(np.nanmedian(_dy_all))
                _resid = np.hypot(_dx_all - _mdx, _dy_all - _mdy)
                _coh = np.isfinite(_resid) & (_resid <= max_sep)
                _n_coh = int(np.sum(_coh))
                if _n_coh >= max(3, int(np.ceil(0.6 * len(_dx_all)))):
                    # Coherent offset cluster -> measurable misregistration
                    # (or a good sparse-field alignment).  Report metrics on
                    # the cluster so the caller's quality gates see the true
                    # offset instead of silently accepting an unverified
                    # product.
                    _mut_idx = np.where(_coh)[0]
                    d_mut = np.hypot(_dx_all[_coh], _dy_all[_coh])
                    logger.info(
                        "Alignment RMS: coherent offset cluster found "
                        "(n=%d of %d mutual pairs, median offset=%.2f px, "
                        "max_sep=%.1f px).",
                        _n_coh, len(_dx_all), float(np.hypot(_mdx, _mdy)),
                        max_sep,
                    )
                else:
                    return None
            else:
                return None

        # Per-axis matched offsets (pre-clipping)
        _dx_mut = sci_xy[mutual, 0][_mut_idx] - ref_xy[i_sr[mutual], 0][_mut_idx]
        _dy_mut = sci_xy[mutual, 1][_mut_idx] - ref_xy[i_sr[mutual], 1][_mut_idx]

        # MAD-based clipping: plain sigma_clip is self-defeating here because
        # false matches (ghost/stacking artifacts) inflate the std.  MAD is
        # resistant to up to 50% outliers.
        _med_d = float(np.nanmedian(d_mut))
        _mad_d = float(np.nanmedian(np.abs(d_mut - _med_d)))
        _robust_sigma = 1.4826 * _mad_d if _mad_d > 0 else float(np.nanstd(d_mut))
        if _robust_sigma < 1e-10:
            _robust_sigma = 1e-10
        _clip_mask = np.abs(d_mut - _med_d) > 3.0 * _robust_sigma
        # Iterate to convergence (recompute MAD on kept sources), max 4 rounds.
        for _ in range(4):
            _kept = ~_clip_mask
            if np.sum(_kept) < 5:
                break
            _med_d = float(np.nanmedian(d_mut[_kept]))
            _mad_d = float(np.nanmedian(np.abs(d_mut[_kept] - _med_d)))
            _robust_sigma = 1.4826 * _mad_d if _mad_d > 0 else 1e-10
            _new_clip = np.abs(d_mut - _med_d) > 3.0 * _robust_sigma
            if np.array_equal(_new_clip, _clip_mask):
                break
            _clip_mask = _new_clip
        d_clipped = d_mut[~_clip_mask]
        if len(d_clipped) < 5:
            d_clipped = d_mut  # fallback: too few after clipping

        # Median offset = magnitude of the median per-axis offset vector.
        # This measures the systematic offset, not the typical per-source
        # distance (which includes centroid noise and is always >= this).
        # Matches the post-SWarp verification logic in run_IDC.py.
        # Use post-clipping data for consistency with RMS and P90.
        _kept_mask = ~_clip_mask
        _med_dx = float(np.nanmedian(_dx_mut[_kept_mask])) if np.sum(_kept_mask) > 0 else float(np.nanmedian(_dx_mut))
        _med_dy = float(np.nanmedian(_dy_mut[_kept_mask])) if np.sum(_kept_mask) > 0 else float(np.nanmedian(_dy_mut))
        median_offset = float(np.sqrt(_med_dx**2 + _med_dy**2))
        p90 = float(np.nanpercentile(d_clipped, 90.0))
        rms = float(np.sqrt(np.mean(d_clipped**2)))

        # Error-weighted residuals: combine the SExtractor centroid errors
        # (quadrature sum of the two catalogs' per-axis sigmas) so faint or
        # extended sources do not dominate the scatter estimate.  Wrms~1
        # means the measured residuals match the formal centroid errors.
        _weighted = None
        if (
            sci_ex is not None and sci_ey is not None
            and ref_ex is not None and ref_ey is not None
        ):
            try:
                _src_idx = np.where(mutual)[0][_mut_idx][_kept_mask]
                _ref_idx = i_sr[mutual][_mut_idx][_kept_mask]
                _dx_k = _dx_mut[_kept_mask]
                _dy_k = _dy_mut[_kept_mask]
                _sx2 = (
                    np.asarray(sci_ex, float)[_src_idx] ** 2
                    + np.asarray(ref_ex, float)[_ref_idx] ** 2
                )
                _sy2 = (
                    np.asarray(sci_ey, float)[_src_idx] ** 2
                    + np.asarray(ref_ey, float)[_ref_idx] ** 2
                )
                _wok = np.isfinite(_sx2) & np.isfinite(_sy2) & (_sx2 > 0) & (_sy2 > 0)
                if int(np.sum(_wok)) >= 5:
                    _chi2 = float(np.sum(
                        _dx_k[_wok] ** 2 / _sx2[_wok]
                        + _dy_k[_wok] ** 2 / _sy2[_wok]
                    ))
                    _dof = max(1, 2 * int(np.sum(_wok)) - 2)
                    _wrms = float(np.sqrt(np.mean(0.5 * (
                        _dx_k[_wok] ** 2 / _sx2[_wok]
                        + _dy_k[_wok] ** 2 / _sy2[_wok]
                    ))))
                    _weighted = {
                        "chi2": _chi2,
                        "chi2_dof": _chi2 / _dof,
                        "wrms_norm": _wrms,
                        "n_weighted": int(np.sum(_wok)),
                    }
            except Exception:
                _weighted = None

        if _weighted is not None:
            logger.log(
                STATUS,
                "Alignment RMS:\tmed=%.3f px\n"
                "                  dx=%.3f, dy=%.3f\n"
                "                  rms=%.3f px p90=%.3f px n=%d (of %d, %d clipped)\n"
                "                  max=%.2f px\n"
                "                  chi2/dof=%.2f wrms_norm=%.2f (n=%d weighted)",
                median_offset, _med_dx, _med_dy, rms, p90, len(d_clipped), len(d_mut),
                len(d_mut) - len(d_clipped), max_sep,
                _weighted["chi2_dof"], _weighted["wrms_norm"],
                _weighted["n_weighted"],
            )
        else:
            logger.log(
                STATUS,
                "Alignment RMS:\tmed=%.3f px\n"
                "                  dx=%.3f, dy=%.3f\n"
                "                  rms=%.3f px p90=%.3f px n=%d (of %d, %d clipped)\n"
                "                  max=%.2f px",
                median_offset, _med_dx, _med_dy, rms, p90, len(d_clipped), len(d_mut),
                len(d_mut) - len(d_clipped), max_sep,
            )

        # Per-quadrant alignment quality.
        # Split the matched sources into 2x2 spatial quadrants and compute
        # the RMS for each.  This detects spatially-varying alignment errors
        # (e.g. edge distortion, spline overfitting in one region) that the
        # global RMS would average away.
        if return_per_quadrant:
            _quadrant_rms = _compute_per_quadrant_rms(
                sci_xy, ref_xy, i_sr, mutual, _mut_idx,
                _clip_mask, sci_data.shape,
            )
            if _weighted is not None:
                _quadrant_rms.update(_weighted)
            return median_offset, rms, p90, _quadrant_rms

        return median_offset, rms, p90

    except Exception:
        logger.debug("compute_alignment_rms failed", exc_info=True)
        return None


def _compute_per_quadrant_rms(
    sci_xy: np.ndarray,
    ref_xy: np.ndarray,
    i_sr: np.ndarray,
    mutual: np.ndarray,
    mut_idx: np.ndarray,
    clip_mask: np.ndarray,
    image_shape: tuple,
) -> dict:
    """
    Compute per-quadrant alignment RMS from matched sources.

    Quadrants:
      Q1 = top-left (x < cx, y < cy)
      Q2 = top-right (x >= cx, y < cy)
      Q3 = bottom-left (x < cx, y >= cy)
      Q4 = bottom-right (x >= cx, y >= cy)

    Returns dict with per-quadrant RMS, max RMS, and max quadrant label.
    Quadrants with < 3 sources get NaN RMS.
    """
    _ny, _nx = image_shape[:2]
    _cx = _nx / 2.0
    _cy = _ny / 2.0

    # Matched science source positions (after mutual + clip filtering)
    _matched_sci = sci_xy[mutual][mut_idx]
    _kept = ~clip_mask
    _matched_sci_kept = _matched_sci[_kept]
    _matched_ref = ref_xy[i_sr[mutual]][mut_idx][_kept]

    _dx = _matched_sci_kept[:, 0] - _matched_ref[:, 0]
    _dy = _matched_sci_kept[:, 1] - _matched_ref[:, 1]
    _d = np.sqrt(_dx**2 + _dy**2)

    _qx = _matched_sci_kept[:, 0] < _cx
    _qy = _matched_sci_kept[:, 1] < _cy
    _q_labels = np.where(
        _qx & _qy, "q1",
        np.where(~_qx & _qy, "q2",
                 np.where(_qx & ~_qy, "q3", "q4")),
    )

    _result = {}
    _max_rms = 0.0
    _max_quad = "none"
    for _ql in ("q1", "q2", "q3", "q4"):
        _sel = _q_labels == _ql
        _n = int(np.sum(_sel))
        if _n >= 3:
            _qrms = float(np.sqrt(np.mean(_d[_sel] ** 2)))
        else:
            _qrms = float("nan")
        _result[f"{_ql}_rms"] = _qrms
        _result[f"{_ql}_n"] = _n
        if np.isfinite(_qrms) and _qrms > _max_rms:
            _max_rms = _qrms
            _max_quad = _ql

    _result["max_rms"] = _max_rms if _max_rms > 0 else float("nan")
    _result["max_quadrant"] = _max_quad

    # Spatial coverage diagnostics.  Matched sources confined to a small
    # region of the detector cannot validate rotation/scale over the full
    # field even if their local residuals are excellent.  Coverage is deemed
    # adequate when the matched sample spans at least 25% of the image in
    # both axes and occupies more than one quadrant.
    _result["n_matched"] = int(len(_matched_sci_kept))
    if len(_matched_sci_kept) >= 2:
        _span_x = (
            float(_matched_sci_kept[:, 0].max() - _matched_sci_kept[:, 0].min())
            / max(float(_nx), 1.0)
        )
        _span_y = (
            float(_matched_sci_kept[:, 1].max() - _matched_sci_kept[:, 1].min())
            / max(float(_ny), 1.0)
        )
        _n_q_occupied = sum(
            1 for _ql in ("q1", "q2", "q3", "q4") if _result.get(f"{_ql}_n", 0) >= 1
        )
    else:
        _span_x = _span_y = 0.0
        _n_q_occupied = 0
    _result["n_quadrants"] = int(_n_q_occupied)
    _result["span_x"] = _span_x
    _result["span_y"] = _span_y
    _result["coverage_ok"] = bool(
        _span_x >= 0.25 and _span_y >= 0.25 and _n_q_occupied >= 2
    )

    if not _result["coverage_ok"] and len(_matched_sci_kept) >= 2:
        logger.warning(
            "Alignment verification: %d matched sources cover only "
            "%.0f%%x%.0f%% of the field (%d quadrants) - rotation/scale "
            "cannot be validated outside the cluster.",
            len(_matched_sci_kept), 100.0 * _span_x, 100.0 * _span_y,
            _n_q_occupied,
        )

    if np.isfinite(_max_rms) and _max_rms > 0:
        logger.info(
            "Per-quadrant alignment RMS: Q1=%.3f Q2=%.3f\n"
            "                            Q3=%.3f Q4=%.3f px (max=%s=%.3f px)",
            _result["q1_rms"], _result["q2_rms"],
            _result["q3_rms"], _result["q4_rms"],
            _max_quad, _max_rms,
        )

    return _result


# -----------------------------------------------------------------------------
# Shared alignment-quality policy
# -----------------------------------------------------------------------------

_P95_KEY_DEPRECATION_WARNED = False


def _alignment_gate_thresholds(
    input_yaml: Optional[Dict[str, Any]],
    fwhm_pix: float,
) -> Tuple[float, float, float]:
    """Return the FWHM-scaled (max_offset, max_rms, max_p90) pixel gates.

    The P90 gate historically lived under the misnamed
    ``alignment_max_p95_px`` key (it has always gated the 90th percentile of
    matched separations).  ``alignment_max_p90_px`` is now preferred; the old
    key is still honoured with a one-time deprecation warning.
    """
    global _P95_KEY_DEPRECATION_WARNED
    ts_cfg = (input_yaml or {}).get("template_subtraction", {}) or {}
    max_offset = float(ts_cfg.get("alignment_max_offset_px", 0.5))
    max_rms = float(ts_cfg.get("alignment_max_rms_px", 0.75))
    if "alignment_max_p90_px" in ts_cfg:
        max_p90 = float(ts_cfg["alignment_max_p90_px"])
    else:
        max_p90 = float(ts_cfg.get("alignment_max_p95_px", 1.5))
        if "alignment_max_p95_px" in ts_cfg and not _P95_KEY_DEPRECATION_WARNED:
            logger.warning(
                "template_subtraction.alignment_max_p95_px is deprecated: "
                "it gates the P90 statistic, not P95. "
                "Rename it to alignment_max_p90_px."
            )
            _P95_KEY_DEPRECATION_WARNED = True
    # Scale the gates with FWHM so the same fractional-pixel tolerance
    # applies to sharp and broad PSFs.
    scale = max(0.5, min(3.0, float(fwhm_pix) / 3.0))
    return max_offset * scale, max_rms * scale, max_p90 * scale


def _classify_alignment_quality(
    median_offset: Optional[float],
    rms: Optional[float],
    p90: Optional[float],
    coverage_ok: Optional[bool],
    max_offset: float,
    max_rms: float,
    max_p90: float,
) -> Tuple[str, List[str]]:
    """Classify a measured alignment into 'pass' / 'warn' / 'fail' /
    'unverified' under one policy shared by every backend.

    - 'fail': any finite metric exceeds its configured gate.
    - 'unverified': no finite metric was produced (sparse field, detection
      failure) - the product cannot be confirmed good or bad.
    - 'warn': metrics pass but matched-source coverage is poor, so the
      measurement may not represent the whole field.
    - 'pass': metrics inside gates with adequate coverage.
    """
    measured = [
        v for v in (median_offset, rms, p90)
        if v is not None and np.isfinite(v)
    ]
    if not measured:
        return "unverified", []
    reasons = []
    if median_offset is not None and np.isfinite(median_offset) and median_offset > max_offset:
        reasons.append("offset=%.3f px (> %.2f px)" % (median_offset, max_offset))
    if rms is not None and np.isfinite(rms) and rms > max_rms:
        reasons.append("RMS=%.3f px (> %.2f px)" % (rms, max_rms))
    if p90 is not None and np.isfinite(p90) and p90 > max_p90:
        reasons.append("P90=%.3f px (> %.2f px)" % (p90, max_p90))
    if reasons:
        return "fail", reasons
    if coverage_ok is False:
        return "warn", ["matched-source coverage limited"]
    return "pass", []


def _alignment_gate_accept(
    quality: str,
    input_yaml: Optional[Dict[str, Any]],
) -> bool:
    """Decide whether an alignment of the given quality may be used.

    'fail' always rejects.  'unverified' and 'warn' are accepted by default
    (best-available fallback) unless the caller opts into strict mode via
    ``alignment_require_verification`` / ``alignment_fail_on_poor_coverage``.
    """
    ts_cfg = (input_yaml or {}).get("template_subtraction", {}) or {}
    if quality == "fail":
        return False
    if quality == "unverified" and bool(
        ts_cfg.get("alignment_require_verification", False)
    ):
        return False
    if quality == "warn" and bool(
        ts_cfg.get("alignment_fail_on_poor_coverage", False)
    ):
        return False
    return True


def verify_common_grid(
    sci_header: fits.Header,
    tpl_header: fits.Header,
    shape: Tuple[int, int],
    tolerance_px: float = 0.05,
) -> bool:
    """Check that two headers map the same pixel grid to the same sky.

    Tests the four corners and the image centre: each science pixel is
    transformed to sky and back through the template WCS.  A residual larger
    than ``tolerance_px`` at any test point means the aligned products do
    not share a common grid and subtraction would be invalid.

    Returns True when the grids match.  When either header has no usable
    WCS the check cannot run and True is returned (unverifiable, not
    mismatched) so WCS-free pixel-aligned products still subtract.
    """
    try:
        sci_wcs = get_wcs(sci_header)
        tpl_wcs = get_wcs(tpl_header)
        if sci_wcs is None or tpl_wcs is None:
            logger.info(
                "verify_common_grid: WCS unavailable on one or both headers; "
                "grid check skipped."
            )
            return True
        ny, nx = shape
        test_points = np.array(
            [
                [0.0, 0.0],
                [nx - 1.0, 0.0],
                [0.0, ny - 1.0],
                [nx - 1.0, ny - 1.0],
                [(nx - 1.0) / 2.0, (ny - 1.0) / 2.0],
            ]
        )
        for x, y in test_points:
            ra, dec = sci_wcs.all_pix2world(x, y, 0)
            xt, yt = tpl_wcs.all_world2pix(ra, dec, 0)
            if not np.isfinite([xt, yt]).all():
                return False
            if np.hypot(xt - x, yt - y) > tolerance_px:
                return False
        return True
    except Exception:
        logger.debug("verify_common_grid failed", exc_info=True)
        return True


def _wcs_footprints_overlap(
    header1: fits.Header,
    shape1: tuple,
    header2: fits.Header,
    shape2: tuple,
    margin_frac: float = 0.05,
) -> bool:
    """Return True when two image WCS footprints overlap on the sky.

    Compares the angular separation of the two field centres against the
    sum of their angular radii (plus a fractional margin).  Unlike an
    RA/Dec bounding box this is correct for fields crossing RA=0, near the
    poles, and for rotated footprints.  Returns False when either WCS is
    unusable (the caller then treats the fields as non-overlapping rather
    than attempting sky-independent matching).
    """
    try:
        def _center_radius(hdr, shape):
            w = get_wcs(hdr)
            if w is None:
                return None
            ny, nx = shape
            # Headers under test need not carry NAXIS cards; pass the array
            # shape explicitly so calc_footprint does not require them.
            corners = w.calc_footprint(axes=(ny, nx))
            if corners is None or not np.all(np.isfinite(corners)):
                return None
            ra_c, dec_c = w.all_pix2world(
                (nx - 1.0) / 2.0, (ny - 1.0) / 2.0, 0
            )
            if not np.isfinite([ra_c, dec_c]).all():
                return None
            center = SkyCoord(float(ra_c) * u.deg, float(dec_c) * u.deg)
            corner_coords = SkyCoord(
                corners[:, 0] * u.deg, corners[:, 1] * u.deg
            )
            radius = float(np.max(center.separation(corner_coords).deg))
            if not np.isfinite(radius) or radius <= 0:
                return None
            return center, radius

        c1 = _center_radius(header1, shape1)
        c2 = _center_radius(header2, shape2)
        if c1 is None or c2 is None:
            return False
        margin = margin_frac * (c1[1] + c2[1])
        sep = c1[0].separation(c2[0]).deg
        return bool(sep <= c1[1] + c2[1] + margin)
    except Exception:
        return False


def _atomic_fits_write(fpath: str, data: np.ndarray, header: fits.Header) -> None:
    """Write FITS via a sibling temp file then ``os.replace``.

    A crash mid-``writeto`` would otherwise leave a truncated file that a
    later stage could mistake for a valid aligned product.
    """
    tmp_path = f"{fpath}.tmp"
    safe_fits_write(tmp_path, np.asarray(data, dtype=np.float32), header)
    os.replace(tmp_path, fpath)
    try:
        invalidate_fits_cache(fpath)
    except Exception:
        pass


def _write_alignment_report(
    attempts: List["AlignmentResult"],
    out_dir: str,
    science_fpath: str,
    template_fpath: str,
    requested_method: Optional[str] = None,
) -> None:
    """Write ``Alignment_<base>.json`` recording every attempted method.

    The report gives each backend's metrics and the shared pass/warn/fail
    classification so a subtraction's alignment provenance survives even
    when the log is lost.
    """
    try:
        def _f(v):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return fv if np.isfinite(fv) else None

        records = [
            {
                "method": a.method_used,
                "success": bool(a.success),
                "quality": a.quality,
                "median_offset_px": _f(a.median_offset_px),
                "rms_px": _f(a.rms_px),
                "p90_px": _f(a.p90_px),
                "n_matches": a.n_matches,
                "coverage_ok": a.coverage_ok,
                "finite_fraction": _f(a.finite_fraction),
                "message": a.message or "",
            }
            for a in attempts
            if a is not None
        ]
        base = os.path.splitext(os.path.basename(science_fpath))[0]
        out_path = os.path.join(out_dir, f"Alignment_{base}.json")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(
                {
                    "science": os.path.basename(science_fpath),
                    "template": os.path.basename(template_fpath),
                    "requested_method": requested_method,
                    "attempts": records,
                },
                fh,
                indent=2,
            )
        os.replace(tmp_path, out_path)
        logger.info("Alignment report written to %s", out_path)
    except Exception:
        logger.debug("alignment report write failed", exc_info=True)


def _pad_to_shape(image, target_shape, fill=np.nan, mask=None):
    """Pad ``image`` up to ``target_shape`` on the high side of each axis.

    Axes already >= the target extent are left alone (the array is never
    cropped).  ``mask`` (boolean, same shape as ``image``) is padded with
    True so the added border counts as invalid.

    Returns ``(padded_image, padded_mask_or_None)``.
    """
    pad = tuple(
        (0, max(int(t) - int(s), 0)) for s, t in zip(image.shape, target_shape)
    )
    if not any(p[1] for p in pad):
        return image, mask
    image = np.pad(image, pad, mode="constant", constant_values=fill)
    if mask is not None:
        mask = np.pad(mask, pad, mode="constant", constant_values=True)
    return image, mask


def _merge_detection_lists(
    xy_a, flux_a, fwhm_a, xy_b, flux_b, fwhm_b, radius=1.5
):
    """Union two detection lists, deduplicated at ``radius`` px.

    The first list wins on overlap so a deeper detection pass only adds
    sources, never replaces shallower-pass centroids.
    """
    if xy_a is None or len(xy_a) == 0:
        return xy_b, flux_b, fwhm_b
    if xy_b is None or len(xy_b) == 0:
        return xy_a, flux_a, fwhm_a
    from scipy.spatial import cKDTree

    dists, _ = cKDTree(xy_a).query(xy_b, k=1, distance_upper_bound=radius)
    new_rows = np.isinf(dists)
    if not new_rows.any():
        return xy_a, flux_a, fwhm_a
    return (
        np.vstack([xy_a, xy_b[new_rows]]),
        np.concatenate([flux_a, flux_b[new_rows]]),
        np.concatenate([fwhm_a, fwhm_b[new_rows]]),
    )


def _close_group_survivor_indices(xy, flux, min_sep):
    """Indices keeping the brightest member of each close-pair group.

    Same crowding rule spalipy applies internally, but it removes every
    member of a pair closer than ``min_sep``; keeping the brightest member
    of each connected group preserves one anchor per group while retaining
    the anti-ambiguity intent (a survivor's 2nd-nearest neighbour in the
    other catalog is then >2*max_match_dist once both sides are filtered).
    """
    n = len(xy)
    if n < 2 or min_sep <= 0:
        return np.arange(n)
    from scipy.spatial import cKDTree

    pairs = cKDTree(xy).query_pairs(float(min_sep))
    if not pairs:
        return np.arange(n)
    parent = np.arange(n)

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[_find(b)] = _find(a)
    groups = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)
    flux = np.asarray(flux, float)
    keep = []
    for idxs in groups.values():
        idxs = np.asarray(idxs)
        keep.append(int(idxs[np.argmax(flux[idxs])]))
    return np.sort(np.asarray(keep))


def _finite_coverage_floor(cfg_min_ff, src_finite_px, out_size):
    """Finite-fraction floor for a warped template.

    A stamp template can only fill its own footprint on the science grid,
    so the gate checks that the warp retained most of the attainable
    coverage (20% allowed for resample edge losses) rather than an
    absolute fraction the source can never reach.  Returns
    ``(floor, attainable_fraction)``.
    """
    attainable = float(src_finite_px) / max(1.0, float(out_size))
    return min(float(cfg_min_ff), 0.8 * attainable), attainable


def _reproject_template(
    science_image: np.ndarray,
    science_header: fits.Header,
    template_image: np.ndarray,
    template_header: fits.Header,
    output_path: str,
    cfg: ReprojectConfig,
    fwhm_pixels: float = 3.0,
    input_yaml: Optional[Dict[str, Any]] = None,
) -> AlignmentResult:
    """
    Reproject template onto the science pixel grid.

    Tries ``cfg.method`` first, then the remaining reproject backends in
    order.  The alignment-RMS diagnostic runs on the in-memory reprojected
    array (not a second read of the output FITS) so its metrics can be
    stored in the output header.

    Parameters
    ----------
    science_image, science_header : ndarray, Header
        Science image data and header (already loaded by caller).
    template_image, template_header : ndarray, Header
        Template data and header (already loaded by caller).
    output_path : str
        Destination path for the reprojected template FITS.
    cfg : ReprojectConfig
        Pre-built configuration (extracted from input_yaml once by caller).
    fwhm_pixels : float
        PSF FWHM for alignment quality diagnostic.

    Returns
    -------
    AlignmentResult
        science_path is None on failure.
    """
    _FAIL = AlignmentResult(None, None, "reproject", None, None, None)

    shape_out = science_image.shape

    # reproject needs real WCS objects; raw headers drop SIP distortion terms.
    from wcs import get_wcs
    template_proj = get_wcs(template_header)
    science_proj = get_wcs(science_header)
    if template_proj is None or science_proj is None:
        logger.error("get_wcs failed for template or science header")
        return _FAIL

    # Log footprints so overlap failures are diagnosable from the log.
    try:
        h, w = science_image.shape
        sci_corners = science_proj.calc_footprint().flatten()
        ref_corners = template_proj.calc_footprint().flatten()
        logger.info(
            "Reproject WCS check: science footprint RA=[%.3f,%.3f] Dec=[%.3f,%.3f], template footprint RA=[%.3f,%.3f] Dec=[%.3f,%.3f]",
            sci_corners[0::2].min(), sci_corners[0::2].max(),
            sci_corners[1::2].min(), sci_corners[1::2].max(),
            ref_corners[0::2].min(), ref_corners[0::2].max(),
            ref_corners[1::2].min(), ref_corners[1::2].max(),
        )
    except Exception as wcs_exc:
        logger.debug("Could not compute WCS footprint: %s", wcs_exc)

    logger.info(
        "Reproject distortion: template[%s] -> science[%s]",
        _distortion_summary(template_header),
        _distortion_summary(science_header),
    )

    adaptive_kwargs = _build_adaptive_kwargs(cfg)

    # BUG 91: Auto-enable center_jacobian for large rotation differences.
    # The default Jacobian approximation is inaccurate when rotation between
    # input and output is large (>30 deg), causing sub-pixel misregistration.
    try:
        sci_rot = np.degrees(np.arctan2(
            science_proj.wcs.cd[0, 1], science_proj.wcs.cd[0, 0]
        ))
        tpl_rot = np.degrees(np.arctan2(
            template_proj.wcs.cd[0, 1], template_proj.wcs.cd[0, 0]
        ))
        rot_diff = abs(sci_rot - tpl_rot)
        if rot_diff > 180:
            rot_diff = 360 - rot_diff
        if rot_diff > 30.0 and not adaptive_kwargs.get("center_jacobian"):
            if _REPROJECT_ADAPTIVE_EXTRAS.get("_has_center_jacobian"):
                adaptive_kwargs["center_jacobian"] = True
                logger.info(
                    "Large rotation (%.1f deg) - enabling center_jacobian for "
                    "more accurate reproject_adaptive resampling.",
                    rot_diff,
                )
    except Exception:
        pass

    # BUG 90: Auto-downgrade interpolation for undersampled images.
    # Bicubic/biquadratic introduce ringing artifacts when PSF is undersampled
    # (FWHM < threshold). Bilinear is safer in that regime.
    _us_thresh = getattr(cfg, "undersampled_fwhm_threshold", 2.5)
    interp_order_eff = cfg.interp_order
    if isinstance(fwhm_pixels, (int, float)) and fwhm_pixels > 0 and fwhm_pixels < _us_thresh:
        if interp_order_eff in ("bicubic", "biquadratic"):
            logger.info(
                "Undersampled image (FWHM=%.2f px < %.1f) - downgrading %s to bilinear "
                "to avoid ringing artifacts.",
                fwhm_pixels, _us_thresh, interp_order_eff,
            )
            interp_order_eff = "bilinear"

    all_methods = ("exact", "adaptive", "interp")
    if cfg.method in all_methods:
        fallbacks = [cfg.method] + [m for m in all_methods if m != cfg.method]
    else:
        logger.warning(
            "Unknown reproject_method=%r; defaulting to adaptive with fallbacks.",
            cfg.method,
        )
        fallbacks = ["exact", "adaptive", "interp"]

    aligned: Optional[np.ndarray] = None
    footprint: Optional[np.ndarray] = None
    used_method: Optional[str] = None
    last_exc: Optional[Exception] = None

    for m in fallbacks:
        try:
            if m == "adaptive":
                aligned, footprint = reproject_adaptive(
                    (template_image, template_proj),
                    science_proj,
                    shape_out=shape_out,
                    roundtrip_coords=cfg.roundtrip,
                    parallel=cfg.parallel,
                    **adaptive_kwargs,
                )
            elif m == "interp":
                aligned, footprint = reproject_interp(
                    (template_image, template_proj),
                    science_proj,
                    shape_out=shape_out,
                    roundtrip_coords=True,
                    order=interp_order_eff,
                )
            else:  # exact
                aligned, footprint = reproject_exact(
                    (template_image, template_proj),
                    science_proj,
                    shape_out=shape_out,
                    parallel=cfg.parallel,
                )
            used_method = m
            break
        except Exception as _e:
            last_exc = _e
            log_warning_from_exception(logger, f"Reproject ({m}) failed", _e)

    if used_method is None or aligned is None or footprint is None:
        logger.error("All reproject methods failed; last error: %s", last_exc)
        return _FAIL

    # Check for zero footprint coverage (WCS mismatch - no overlap)
    fp_mask = footprint.astype(bool)
    n_footprint = np.sum(fp_mask)
    n_total = fp_mask.size
    if n_footprint == 0:
        logger.error(
            f"Reproject has zero footprint coverage\n"
            f"    ({n_footprint}/{n_total} pixels). Template and science\n"
            f"    WCS do not overlap. This is likely due to SCAMP failure\n"
            f"    causing large WCS shift. Returning failure to trigger\n"
            f"    fallback to AstroAlign."
        )
        return _FAIL

    _min_overlap = float(
        (input_yaml or {}).get("template_subtraction", {}).get(
            "alignment_min_overlap_fraction", 0.0
        )
        or 0.0
    )
    if n_total and (n_footprint / n_total) < _min_overlap:
        logger.error(
            "Reproject footprint coverage %.1f%% below minimum %.1f%%; "
            "too little overlap for a meaningful subtraction.",
            100.0 * n_footprint / n_total,
            100.0 * _min_overlap,
        )
        return _FAIL

    logger.info("Reproject footprint coverage: %s/%s pixels (%.1f%)", n_footprint, n_total, 100*n_footprint/n_total)

    # Mask non-footprint pixels with NaN (preserves chip gaps)
    aligned[~fp_mask] = np.nan
    to_write = np.asarray(aligned, dtype=np.float32)

    # ------------------------------------------------------------------
    # Alignment quality diagnostic - computed on the in-memory array before
    # writing so the metrics can be stored in the output header (ALIG*) for
    # downstream SFFT kernel sizing and photometry provenance.
    # ------------------------------------------------------------------
    _quad = None
    try:
        align_metrics = compute_alignment_rms(
            science_image, to_write, fwhm_pixels,
            input_yaml=input_yaml,
            return_per_quadrant=True,
        )
        if align_metrics is not None and len(align_metrics) == 4:
            median_offset, rms_offset, p90_offset, _quad = align_metrics
        else:
            median_offset, rms_offset, p90_offset = (
                align_metrics[:3] if align_metrics else (None, None, None)
            )
            _quad = None
    except Exception:
        median_offset, rms_offset, p90_offset = None, None, None
        _quad = None

    coverage_ok = _quad.get("coverage_ok") if _quad is not None else None
    if coverage_ok is False:
        logger.warning(
            "Reproject verification: matched sources cover only "
            "%.0f%%x%.0f%% of the field - coverage-limited verification.",
            100.0 * _quad.get("span_x", 0.0),
            100.0 * _quad.get("span_y", 0.0),
        )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    hdr = template_header.copy()
    hdr = remove_wcs_from_header(hdr)
    from functions import copy_wcs_from_header
    copy_wcs_from_header(science_header, hdr)
    hdr["NAXIS1"] = aligned.shape[1]
    hdr["NAXIS2"] = aligned.shape[0]
    try:
        if median_offset is not None and np.isfinite(median_offset):
            hdr["ALIGMED"] = (float(median_offset), "Alignment median offset (px)")
        if rms_offset is not None and np.isfinite(rms_offset):
            hdr["ALIGRMS"] = (float(rms_offset), "Alignment RMS (px)")
        if p90_offset is not None and np.isfinite(p90_offset):
            hdr["ALIGP90"] = (float(p90_offset), "Alignment P90 offset (px)")
        hdr["ALIGMETH"] = (f"reproject/{used_method}", "Alignment method used")
        if _quad is not None:
            _qmax = _quad.get("max_rms", np.nan)
            if np.isfinite(_qmax) and _qmax > 0:
                hdr["ALIGQMAX"] = (float(_qmax), "Max per-quadrant alignment RMS (px)")
                hdr["ALIGQREG"] = (
                    str(_quad.get("max_quadrant", "none")),
                    "Worst alignment quadrant",
                )
            hdr["ALIGCOV"] = (
                int(bool(_quad.get("coverage_ok", True))),
                "Matched-source coverage adequate (1=yes, 0=clustered)",
            )
    except Exception:
        pass
    _atomic_fits_write(output_path, to_write, hdr)
    # The write may have overwritten the staged template in place; refresh any
    # sibling weight map still on the pre-resampling grid (the reproject
    # footprint doubles as a coverage weight on the new grid).
    try:
        from functions import refresh_sibling_weight_map

        refresh_sibling_weight_map(
            output_path,
            weight_data=np.asarray(footprint, dtype=np.float32),
            logger=logger,
        )
    except Exception:
        pass

    logger.info("Reproject alignment succeeded (method=%s).", used_method)
    return AlignmentResult(
        science_path=None,  # filled in by caller
        template_path=output_path,
        method_used=f"reproject/{used_method}",
        median_offset_px=median_offset,
        rms_px=rms_offset,
        p90_px=p90_offset,
        coverage_ok=coverage_ok,
        n_matches=_quad.get("n_matched") if _quad else None,
        finite_fraction=n_footprint / n_total if n_total else None,
        quality="pass",
    )


# =============================================================================
# Module-Level Helper Functions
# =============================================================================
# These are pure functions with no side effects, so they belong at module scope
# rather than being redefined inside methods on every call.


def _as_bool(value: Any, default: bool = False) -> bool:
    """Coerce YAML-friendly values to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y", "on"):
            return True
        if v in ("0", "false", "f", "no", "n", "off"):
            return False
        return default
    return default


def _select_forceconv(
    cfg_value: Any,
    science_fwhm: float,
    template_fwhm: float,
    auto_tol: float = 0.05,
) -> Tuple[str, bool, str]:
    """Resolve the SFFT convolution direction.

    Parameters
    ----------
    cfg_value : str
        Configured direction: 'REF' (DIFF = SCI - conv(REF); transient keeps
        the science PSF), 'SCI' (DIFF = conv(SCI) - REF; diff has the
        reference PSF), or 'AUTO'.
    science_fwhm, template_fwhm : float
        FWHMs measured on the aligned products (pixels) -- the values used
        for kernel sizing, not raw FITS headers.
    auto_tol : float
        Fractional FWHM difference below which the images are treated as
        equal and AUTO keeps REF (the PSF-conserving convention).

    Returns
    -------
    (direction, deconvolves, note)
        direction: 'REF' or 'SCI' to pass to run_sfft.py (AUTO is resolved
        here so SFFT never decides from potentially stale header FWHMs --
        SWarp resampling can flip apparent PSF ordering, BUG 122).
        deconvolves: True when the chosen direction must sharpen the
        convolved image (convolved side is broader than the target by more
        than auto_tol), which produces an unstable kernel with negative
        sidelobes when few matched sources constrain the fit.
        note: short human-readable explanation for logging.
    """
    cfg = str(cfg_value or "REF").strip().upper()
    fs = float(science_fwhm) if np.isfinite(science_fwhm) else 0.0
    ft = float(template_fwhm) if np.isfinite(template_fwhm) else 0.0
    tol = float(auto_tol) if np.isfinite(auto_tol) else 0.05
    tol = max(0.0, min(tol, 0.5))

    if cfg == "AUTO":
        if ft > 0 and fs > 0 and ft > fs * (1.0 + tol):
            # Template broader: convolve the sharper science up to the
            # reference PSF (well-posed blur).  Diff gets reference PSF.
            return "SCI", False, (
                f"AUTO->SCI: template FWHM {ft:.2f} > science {fs:.2f} "
                f"(>{tol:.0%}); convolve science (well-posed, diff has "
                "reference PSF)"
            )
        if fs > 0 and ft > 0 and fs > ft * (1.0 + tol):
            # Science broader: convolve the sharper reference down to the
            # science PSF (well-posed blur).  Diff keeps science PSF.
            return "REF", False, (
                f"AUTO->REF: science FWHM {fs:.2f} > template {ft:.2f} "
                f"(>{tol:.0%}); convolve template (diff keeps science PSF)"
            )
        return "REF", False, (
            f"AUTO->REF: FWHMs nearly equal (sci={fs:.2f}, ref={ft:.2f}, "
            f"tol={tol:.0%}); keep science-PSF convention"
        )

    direction = cfg if cfg in ("REF", "SCI") else "REF"
    # Deconvolution check: the convolved side must end up *sharper* than it
    # started (kernel removes width rather than adding it).
    if direction == "REF":
        deconvolves = ft > 0 and fs > 0 and ft > fs * (1.0 + tol)
    else:
        deconvolves = fs > 0 and ft > 0 and fs > ft * (1.0 + tol)
    note = (
        "configured direction requires deconvolution"
        if deconvolves
        else "configured direction"
    )
    return direction, deconvolves, note


def _apply_sfft_deconvolution_veto(
    direction: str,
    deconvolves: bool,
    enabled: bool = True,
) -> Tuple[str, bool, bool]:
    """Flip an impossible SFFT convolution direction when enabled.

    A direction that must sharpen the convolved image produces the same
    negative-sidelobe kernel the ZOGY probe vetoes.  The opposite direction is
    always a broadening convolution for a non-trivial PSF difference, so it is
    safer than honoring an explicit sharpening request.
    """
    direction = str(direction).strip().upper()
    if not enabled or not deconvolves or direction not in ("REF", "SCI"):
        return direction, bool(deconvolves), False
    return ("SCI" if direction == "REF" else "REF"), False, True


def _auto_sfft_kernel_order(
    rel_diff: float,
    n_eff: int,
    max_order: int,
    last_resort: bool = False,
) -> int:
    """Choose conservative SFFT spatial-kernel freedom.

    A larger source list makes higher orders feasible, but does not show that
    the matching kernel varies spatially.  Add freedom only for measured PSF
    mismatch or a known low-quality alignment, then cap it by kernel size.
    """
    max_order = max(0, min(int(max_order), 3))
    if n_eff >= 80 and rel_diff > 0.6:
        return min(2, max_order)
    if n_eff >= 30 and (rel_diff > 0.4 or last_resort):
        return min(1, max_order)
    return 0


def _diff_resid_at_sources(
    diff_path: Any,
    xy_list: Any,
    radius: float = 4.0,
) -> Optional[float]:
    """Median absolute aperture residual at matched-source positions.

    Measures how much flux remains at calibration-source locations in a
    difference image -- the direct observable for over/under-subtraction.
    Returns None when the metric cannot be evaluated.
    """
    try:
        if not diff_path or not os.path.isfile(str(diff_path)):
            return None
        if xy_list is None or len(xy_list) == 0:
            return None
        data = np.asarray(fits.getdata(str(diff_path)), dtype=float)
        if data.size == 0 or not np.isfinite(data).any():
            return None
        ny, nx = data.shape
        r = max(2.0, float(radius))
        resids = []
        for x, y in xy_list:
            xi, yi = int(round(float(x))), int(round(float(y)))
            x0, x1 = max(0, xi - int(np.ceil(r))), min(nx, xi + int(np.ceil(r)) + 1)
            y0, y1 = max(0, yi - int(np.ceil(r))), min(ny, yi + int(np.ceil(r)) + 1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            m = np.hypot(xx - float(x), yy - float(y)) <= r
            vals = data[y0:y1, x0:x1][m]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 5:
                resids.append(float(np.sum(vals)))
        if len(resids) < 2:
            return None
        return float(np.median(np.abs(resids)))
    except Exception:
        return None


def euclidean_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Return the Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def ensure_odd(n: int) -> int:
    """Round *n* up to the nearest odd integer."""
    return n + (n % 2 == 0)


def find_conda_env(env_name: str) -> Optional[str]:
    """
    Look up a conda environment by name and return its prefix path.

    Returns None if the environment is not found or conda is unavailable.
    """
    try:
        proc = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in proc.stdout.splitlines():
            if env_name in line:
                return line.split()[1]
    except FileNotFoundError:
        logger.debug("conda not found on PATH")
    return None


def _normalize_reproject_interp_order(order: Any) -> Any:
    """
    Map YAML / user strings to values accepted by ``reproject.reproject_interp``.

    Reproject accepts int 0-5 or specific strings (e.g. ``'bicubic'``); unknown
    strings fall back to ``'bilinear'``.
    """
    if isinstance(order, int):
        return max(0, min(5, int(order)))
    s = str(order).lower().strip()
    if s in ("nearest", "nearest-neighbor", "nn"):
        return "nearest-neighbor"
    if s in ("bilinear", "linear", "1"):
        return "bilinear"
    if s in ("biquadratic", "2"):
        return "biquadratic"
    if s in ("bicubic", "cubic", "3"):
        return "bicubic"
    if s.isdigit():
        return max(0, min(5, int(s)))
    return "bilinear"


def read_fits(
    fpath: str,
    *,
    as_float: bool = True,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Read a FITS file and return (data, header) in a single I/O operation.

    Parameters
    ----------
    fpath : str
        Path to the FITS file.
    as_float : bool
        If True, cast data to float64 (avoids integer-overflow issues
        in later arithmetic).

    Returns
    -------
    data : np.ndarray
    header : fits.Header
    """
    with fits.open(fpath, ignore_missing_end=True, lazy_load_hdus=True) as hdul:
        hdul[0].verify("silentfix+ignore")
        data = hdul[0].data
        header = hdul[0].header.copy()
        # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
        if data is not None and data.dtype.kind != 'f':
            data = data.astype(np.float32)
        # ESO multi-extension FITS often have empty primary (NAXIS=0); use first HDU with data
        if data is None and len(hdul) > 1:
            for i in range(1, len(hdul)):
                if hdul[i].data is not None and getattr(hdul[i].data, "ndim", 0) >= 2:
                    data = hdul[i].data
                    header = hdul[i].header.copy()
                    break
        if data is None:
            raise ValueError(f"No image data found in {fpath} (primary and extensions)")
        if as_float:
            data = data.astype(np.float64)
    return data, header


def write_fits(
    fpath: str,
    data: np.ndarray,
    header: fits.Header,
    *,
    overwrite: bool = True,
) -> None:
    """Write *data* and *header* to a FITS file with silent verification."""
    from functions import safe_fits_write
    safe_fits_write(fpath, data, header, overwrite=overwrite)


def flux_to_mag(
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert linear flux (and its uncertainty) to magnitudes.

    Uses the standard relation: mag = -2.5 * log10(flux).
    Handles non-positive flux gracefully by returning NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux)
        # Non-positive flux -> NaN error rather than inf/nan from the ratio.
        mag_err = np.where(flux > 0, (2.5 / np.log(10)) * (flux_err / flux), np.nan)
    return mag, mag_err


def clean_fits_nans(fpath: str, output_dir: str = None) -> str:
    """
    Replace NaN / Inf pixels in a FITS image with NO_DATA_SENTINEL.

    Creates a temporary file with cleaned data and returns its path.
    The original file is NOT modified to prevent cross-image contamination.

    This is necessary before feeding images to external tools (HOTPANTS, SFFT)
    that cannot handle IEEE special values.

    Args:
        fpath: Path to the input FITS file.
        output_dir: Directory for the temporary file. If None, uses the parent directory of fpath.

    Returns:
        str: Path to the cleaned temporary file.
    """
    with fits.open(fpath) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
        bad = ~np.isfinite(data)
        if bad.any():
            data[bad] = NO_DATA_SENTINEL

    # Never modify the original file in place.
    if output_dir is None:
        output_dir = str(Path(fpath).parent)
    fd, tmp_path = tempfile.mkstemp(suffix=".fits", prefix="cleaned_", dir=output_dir)
    try:
        with os.fdopen(fd, 'wb') as f:
            hdu = fits.PrimaryHDU(data, header=header)
            hdu.writeto(f, output_verify="silentfix+ignore")
    except Exception:
        os.close(fd)
        raise
    return tmp_path


def deduplicate_points(
    points: List[Tuple[float, float]],
    min_sep: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Remove near-duplicate (x, y) positions from a list.

    Keeps the first occurrence and discards any subsequent point that lies
    within *min_sep* pixels of an already-kept point.
    """
    kept: List[Tuple[float, float]] = []
    for pt in points:
        if all(euclidean_distance(pt, k) >= min_sep for k in kept):
            kept.append(pt)
    return kept


def _read_sfft_matching_positions(path: str) -> Optional[List[Tuple[float, float]]]:
    """Read finite SFFT matched positions in 0-based image coordinates."""
    if not path or not os.path.isfile(path):
        return None
    try:
        sources = pd.read_csv(path)
    except Exception:
        return None
    if {"X_IMAGE_REF_SCI_MEAN", "Y_IMAGE_REF_SCI_MEAN"}.issubset(
        sources.columns
    ):
        xy = sources[["X_IMAGE_REF_SCI_MEAN", "Y_IMAGE_REF_SCI_MEAN"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float) - 1.0
    elif {"x_pix", "y_pix"}.issubset(sources.columns):
        xy = sources[["x_pix", "y_pix"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
    else:
        return None
    return [tuple(row) for row in xy[np.isfinite(xy).all(axis=1)]]


def _validate_noise_map(
    path: Optional[str],
    expected_shape: Tuple[int, int],
    label: str,
    ref_wcs=None,
) -> Optional[str]:
    """
    Validate a noise/RMS map against the image grid it will be used with.

    Returns the path when it exists, has the expected shape, contains
    finite pixels, and (when it carries a WCS) shares the image grid;
    otherwise returns None and logs a warning.  A mismatched map is worse
    than no map: backends either reject it with a warning (HOTPANTS) or
    silently misapply it.
    """
    if not path:
        return None
    path = str(path)
    if not os.path.isfile(path):
        logger.warning("%s noise map missing: %s; ignoring.", label, path)
        return None
    try:
        data = np.asarray(fits.getdata(path), dtype=float)
    except Exception as e:
        logger.warning(
            "%s noise map unreadable (%s): %s; ignoring.", label, path, e
        )
        return None
    if data.shape != tuple(expected_shape):
        logger.warning(
            "%s noise-map shape %s does not match image shape %s; "
            "ignoring map (it was not resampled with the aligned image).",
            label,
            data.shape,
            tuple(expected_shape),
        )
        return None
    if not np.isfinite(data).any():
        logger.warning("%s noise map has no finite pixels; ignoring.", label)
        return None
    if ref_wcs is not None:
        # A same-shape map can still describe a different pixel grid when
        # the image was resampled after the map was written (the staged
        # template weight keeps the pre-alignment template WCS).  When the
        # map carries no WCS of its own, shape agreement is the only
        # evidence available, so the map is kept.
        try:
            noise_wcs = get_wcs(fits.getheader(path))
        except Exception:
            noise_wcs = None
        if noise_wcs is not None and not _wcs_on_common_grid(
            ref_wcs, noise_wcs, data.shape
        ):
            logger.warning(
                "%s noise map WCS disagrees with the aligned image grid; "
                "ignoring map (it was not resampled with the image).",
                label,
            )
            return None
    return path


def _wcs_on_common_grid(
    wcs_a,
    wcs_b,
    shape: Tuple[int, int],
    tol_px: float = 0.5,
) -> bool:
    """
    Check that two WCS describe the same pixel grid.

    Probes the frame corners and center: a pixel is mapped to sky through
    *wcs_a* and back to pixels through *wcs_b*.  Both frames share a grid
    when every probed round trip lands within *tol_px*.
    """
    ny, nx = shape
    pts = [
        (0.0, 0.0),
        (nx - 1.0, 0.0),
        (0.0, ny - 1.0),
        (nx - 1.0, ny - 1.0),
        (nx / 2.0, ny / 2.0),
    ]
    try:
        for px, py in pts:
            ra, dec = wcs_a.all_pix2world(px, py, 0)
            xb, yb = wcs_b.all_world2pix(ra, dec, 0)
            if not (np.isfinite(xb) and np.isfinite(yb)):
                return False
            if float(np.hypot(xb - px, yb - py)) > tol_px:
                return False
    except Exception:
        return False
    return True


def _ensure_diff_wcs(
    diff_header,
    science_header,
    shape: Tuple[int, int],
    target_xy: Optional[Tuple[float, float]] = None,
    tol_px: float = 0.5,
):
    """
    Return a difference-image header guaranteed to carry a WCS consistent
    with the science pixel grid.

    Backends should copy the science header, but external tools can drop or
    rewrite WCS.  When the difference WCS is missing or maps sky
    coordinates to a different pixel position than the science WCS by more
    than *tol_px* at any probed point, the science WCS is injected: the
    difference image is defined on the science pixel grid by construction.
    """
    try:
        sci_wcs = get_wcs(science_header)
        diff_wcs = get_wcs(diff_header)
    except Exception:
        return diff_header

    if sci_wcs is None:
        if diff_wcs is None:
            logger.warning(
                "No usable WCS in science or difference header; "
                "downstream RA/Dec->pixel mapping will fail."
            )
        return diff_header

    need_replace = diff_wcs is None
    worst = np.nan
    if not need_replace:
        ny, nx = shape
        pts = [
            (0.0, 0.0),
            (nx - 1.0, 0.0),
            (0.0, ny - 1.0),
            (nx - 1.0, ny - 1.0),
            (nx / 2.0, ny / 2.0),
        ]
        if target_xy is not None and len(target_xy) >= 2:
            pts.append((float(target_xy[0]), float(target_xy[1])))
        try:
            worst = 0.0
            for px, py in pts:
                ra, dec = sci_wcs.all_pix2world(px, py, 0)
                xd, yd = diff_wcs.all_world2pix(ra, dec, 0)
                if not (np.isfinite(xd) and np.isfinite(yd)):
                    worst = np.inf
                    break
                worst = max(worst, float(np.hypot(xd - px, yd - py)))
        except Exception:
            need_replace = True
        if not np.isfinite(worst) or worst > tol_px:
            need_replace = True

    if need_replace:
        try:
            diff_header.update(sci_wcs.to_header(relax=True))
            if diff_wcs is None:
                logger.warning(
                    "Difference image has no usable WCS; "
                    "replaced with science WCS (same pixel grid)."
                )
            else:
                logger.warning(
                    "Difference-image WCS disagrees with science grid "
                    "(worst offset %s px > %.2f); replaced with science WCS.",
                    f"{worst:.3f}" if np.isfinite(worst) else "non-finite",
                    tol_px,
                )
        except Exception as e:
            logger.warning(
                "Could not inject science WCS into difference header: %s", e
            )
    return diff_header


# =============================================================================
# Largest Rectangle from Target
# =============================================================================


def largest_rectangle_from_target(
    mask: np.ndarray,
    target_pixel: Tuple[int, int],
    max_expand: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """
    Expand an axis-aligned rectangle outward from *target_pixel* in *mask*.

    The rectangle grows one pixel per iteration in each cardinal direction
    as long as the entire new edge consists of valid (True) pixels.

    Parameters
    ----------
    mask : np.ndarray
        2-D boolean array where True marks valid pixels.
    target_pixel : tuple of int
        (row, col) seed position.
    max_expand : int or None
        Maximum expansion radius.  Defaults to max(mask.shape).

    Returns
    -------
    (min_row, min_col, max_row, max_col)
        Bounding box of the largest rectangle found.

    Raises
    ------
    ValueError
        If the target is out of bounds or lies on an invalid pixel.
    """
    rows, cols = mask.shape
    tr, tc = int(target_pixel[0]), int(target_pixel[1])

    if not (0 <= tr < rows and 0 <= tc < cols):
        raise ValueError(
            f"Target pixel ({tr}, {tc}) is outside image bounds " f"({rows}x{cols})."
        )
    if not mask[tr, tc]:
        raise ValueError("Target pixel is not in a valid (True) region.")

    if max_expand is None:
        max_expand = max(rows, cols)

    min_row, max_row = tr, tr
    min_col, max_col = tc, tc

    for _ in range(max_expand):
        expanded = False

        if min_row > 0 and mask[min_row - 1, min_col : max_col + 1].all():
            min_row -= 1
            expanded = True

        if max_row < rows - 1 and mask[max_row + 1, min_col : max_col + 1].all():
            max_row += 1
            expanded = True

        if min_col > 0 and mask[min_row : max_row + 1, min_col - 1].all():
            min_col -= 1
            expanded = True

        if max_col < cols - 1 and mask[min_row : max_row + 1, max_col + 1].all():
            max_col += 1
            expanded = True

        if not expanded:
            break

    return min_row, min_col, max_row, max_col


# =============================================================================
# Fill Masked Regions in FITS
# =============================================================================


def fill_masked_regions_in_fits(
    image_fpath: str,
    mask_fpath: str,
    output_fpath: Optional[str] = None,
    *,
    apply_sigma_clip: bool = True,
    sigma: float = DEFAULT_SIGMA_CLIP,
    use_poisson: bool = False,
) -> str:
    """
    Replace masked pixels with synthetic noise matching the background.

    Reads *image_fpath* and *mask_fpath*, computes background statistics
    from unmasked pixels, fills masked pixels with Gaussian (or Poisson)
    noise drawn from those statistics, and writes the result to
    *output_fpath*.

    Parameters
    ----------
    image_fpath : str
        Input science image.
    mask_fpath : str
        Boolean mask (True = pixel to fill).
    output_fpath : str or None
        Destination path.  Defaults to ``<image>_filled.fits``.
    apply_sigma_clip : bool
        Use sigma-clipped statistics for the background estimate.
    sigma : float
        Clipping threshold in standard deviations.
    use_poisson : bool
        If True, draw from a Poisson distribution instead of Gaussian.

    Returns
    -------
    str
        Path to the written output file.
    """
    image_data, header = read_fits(image_fpath)
    mask_data, _ = read_fits(mask_fpath)
    mask_data = mask_data.astype(bool)

    if image_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: image {image_data.shape} vs mask {mask_data.shape}."
        )

    bg_pixels = image_data[~mask_data]
    if bg_pixels.size == 0:
        raise ValueError("No unmasked pixels available for background estimation.")

    if apply_sigma_clip:
        from functions import biweight_sky_sigma

        mean, _, _ = sigma_clipped_stats(bg_pixels, sigma=sigma)
        std = biweight_sky_sigma(bg_pixels)
    else:
        mean, std = float(np.mean(bg_pixels)), float(np.std(bg_pixels))

    n_fill = int(mask_data.sum())
    if use_poisson:
        noise = RNG.poisson(lam=max(mean, 0), size=n_fill).astype(float)
    else:
        noise = RNG.normal(loc=mean, scale=max(std, 0), size=n_fill)

    # Fill in-place; image_data is a private copy from read_fits.
    image_data[mask_data] = noise

    if output_fpath is None:
        output_fpath = image_fpath.replace(".fits", "_filled.fits")

    write_fits(output_fpath, image_data, header)
    return output_fpath


# =============================================================================
# Download PanSTARRS Template
# =============================================================================


def download_panstarrs_template(
    ra: float,
    dec: float,
    size: int,
    template_folder: str,
    band: str = "r",
) -> Optional[str]:
    """
    Download a Pan-STARRS stacked image cutout for a single band.

    Uses the PS1 image-cutout service at STScI.  The pixel scale is fixed
    at 0.25 arcsec/pixel, so *size* (given in arcminutes) is converted
    accordingly.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in decimal degrees (J2000).
    size : int
        Cutout side length in arcminutes.
    template_folder : str
        Root directory under which band-specific sub-folders are created.
    band : str
        One of 'g', 'r', 'i', 'z', 'y', 'w'.

    Returns
    -------
    str or None
        Path to the downloaded FITS file, or None on failure.
    """
    warnings.filterwarnings("ignore", category=AstropyWarning)

    band = band.strip().lower()
    if band not in PANSTARRS_FILTERS:
        logger.info("Band '%s' not available in PanSTARRS [griz]", band)
        return None

    # Convert arcmin to PS1 pixels (0.25 arcsec/pixel)
    size_px = int(size * 60 / 0.25)
    logger.info("Searching for %s-band image from PanSTARRS", band)

    try:
        # Step 1: query the filename service
        svc = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = f"{svc}?ra={ra}&dec={dec}&size={size_px}&format=fits&sep=,&filters={band}"

        with requests.Session() as session:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            lines = [line.decode("utf-8") for line in resp.iter_lines()]

        df = pd.DataFrame(
            [line.split(",") for line in lines[1:]],
            columns=lines[0].split(","),
        )
        if df.empty:
            logger.info("No %s-band image found", band)
            return None

        # Step 2: build cutout URL
        # NOTE: Original code shadowed the `f` parameter here - fixed by
        #       renaming the outer parameter to `band`.
        cutout_base = (
            f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
            f"?ra={ra}&dec={dec}&size={size_px}&format=fits&filters={band}"
        )
        # Sort by filter priority (relevant when multiple filters match)
        filter_order = [band.find(x) for x in df["filter"]]
        df = df.iloc[np.argsort(filter_order)].reset_index(drop=True)
        urls = [f"{cutout_base}&red={fn}" for fn in df["filename"]]

        if not urls:
            logger.info("Cannot build download URL for %s-band", band)
            return None

        # Step 3: prepare output path (defer mkdir until write succeeds to
        # avoid leaving empty band subfolders on download failure)
        sub_folder = Path(template_folder) / f"{band}_template"
        template_fpath = sub_folder / f"panstarrs_{band}_band_template.fits"

        if template_fpath.exists():
            logger.info(
                "Template already exists at %s - skipping download", template_fpath
            )
            return str(template_fpath)

        # Step 4: download and repackage
        with fits.open(urls[0], ignore_missing_end=True, lazy_load_hdus=True) as hdu:
            hdu.verify("silentfix+ignore")
            src_header = hdu[0].header

            new_header = fits.PrimaryHDU().header
            new_header.update(
                {
                    "TELESCOP": "PS1",
                    "INSTRUME": "GPC1",
                    "FILTER": band,
                    "GAIN": src_header.get("CELL.GAIN", src_header.get("GAIN", 1.0)),
                    "MJD-OBS": src_header.get("MJD-OBS", 0.0),
                    "EXPTIME": src_header.get("EXPTIME", 1.0),
                }
            )
            template_wcs = get_wcs(src_header)
            from functions import update_header_from_wcs
            update_header_from_wcs(new_header, template_wcs)

            # Create the subfolder only when we have data ready to write
            sub_folder.mkdir(parents=True, exist_ok=True)
            write_fits(str(template_fpath), hdu[0].data, new_header)

    except Exception:
        logger.exception("Error downloading PanSTARRS template")
        return None

    return str(template_fpath)


# =============================================================================
# Download SDSS Template
# =============================================================================

SDSS_FILTERS = ("u", "g", "r", "i", "z")


def download_sdss_template(
    ra: float,
    dec: float,
    size: int,
    template_folder: str,
    f: str = "r",
) -> Optional[str]:
    """
    Download an SDSS image cutout for a single band for use as a template in
    template subtraction.

    Uses astroquery.sdss.SDSS to fetch the frame containing the coordinates,
    then extracts a cutout of the requested size (arcmin). SDSS pixel scale
    is ~0.396 arcsec/pixel.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in decimal degrees (J2000).
    size : int
        Cutout side length in arcminutes.
    template_folder : str
        Root directory under which band-specific sub-folders are created.
    f : str
        Filter band: one of 'u', 'g', 'r', 'i', 'z'.

    Returns
    -------
    str or None
        Path to the downloaded FITS file, or None on failure.
    """
    warnings.filterwarnings("ignore", category=AstropyWarning)

    if not _ASTROQUERY_AVAILABLE or SDSS is None:
        logger.warning(
            "SDSS template download requested but 'astroquery' is not installed; skipping."
        )
        return None

    band = str(f).strip().lower()
    if band not in SDSS_FILTERS:
        logger.info("Band '%s' not available in SDSS [ugriz]; skipping.", band)
        return None

    # Compute output path (defer mkdir until write succeeds to avoid
    # leaving empty band subfolders on download failure)
    sub_folder = Path(template_folder) / f"{band}_template"
    template_fpath = sub_folder / f"sdss_{band}_band_template.fits"

    if template_fpath.exists():
        logger.info(
            "SDSS template already exists at %s - skipping download", template_fpath
        )
        return str(template_fpath)

    logger.info(
        "Downloading %s-band template from SDSS for (%.4f, %.4f)", band, ra, dec
    )

    try:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        # get_images returns list of HDUList; we get the full frame for this band.
        images = SDSS.get_images(
            coordinates=coord,
            band=band,
            cache=True,
            show_progress=False,
        )
        if not images or len(images) == 0:
            logger.warning(
                "No SDSS %s-band image found for (%.4f, %.4f)", band, ra, dec
            )
            return None

        hdul = images[0]
        data = np.asarray(hdul[0].data, dtype=float)
        src_header = hdul[0].header
        template_wcs = get_wcs(src_header)
        # SDSS scale ~0.396 arcsec/pixel; size in arcmin -> half-size in pixels
        scale_arcsec_px = 0.396
        half_size_px = (float(size) * 60 / 2) / scale_arcsec_px
        position = template_wcs.world_to_pixel(coord)
        xc, yc = float(position[0]), float(position[1])
        if not (0 <= xc < data.shape[1] and 0 <= yc < data.shape[0]):
            logger.warning("Target (%.4f, %.4f) outside SDSS frame", ra, dec)
            return None
        from functions import nan_crop
        new_header = fits.PrimaryHDU().header
        new_header.update(
            {
                "TELESCOP": "SDSS",
                "INSTRUME": "Imaging",
                "FILTER": band,
                "GAIN": src_header.get("GAIN", 1.0),
                "MJD-OBS": src_header.get("MJD-OBS", 0.0),
                "EXPTIME": src_header.get("EXPTIME", 1.0),
            }
        )
        # Copy WCS keywords from source header, then nan_crop will update CRPIX
        from functions import copy_wcs_from_header
        copy_wcs_from_header(new_header, src_header)
        cutout_size = int(2 * half_size_px)
        cutout_data, new_header = nan_crop(
            data, new_header, xc, yc, cutout_size, cutout_size
        )
        # Create the subfolder only when we have data ready to write
        sub_folder.mkdir(parents=True, exist_ok=True)
        write_fits(str(template_fpath), cutout_data, new_header)
        logger.debug("SDSS template written: %s", template_fpath)
    except Exception:
        logger.exception("Error downloading SDSS template")
        return None

    return str(template_fpath)


# =============================================================================
# Download Legacy Survey Template
# =============================================================================

# Legacy Survey bands (gri only); order matches multi-extension FITS from legacystamps
LEGACY_FILTERS = ("g", "r", "i")
LEGACY_BAND_INDEX = {b: i for i, b in enumerate(LEGACY_FILTERS)}

# Retries for flaky Legacy Survey downloads (IncompleteRead / connection drops)
LEGACY_DOWNLOAD_ATTEMPTS = 3
LEGACY_DOWNLOAD_RETRY_DELAY = 10  # seconds

# Legacy Survey cutout URL; 0.262 arcsec/pixel (nanomaggy), size in pixels
LEGACY_CUTOUT_PIXSCALE = 0.262
LEGACY_CUTOUT_LAYER = "ls-dr10"
LEGACY_CUTOUT_BASE = "https://www.legacysurvey.org/viewer/fits-cutout/"
# Use subimage API (no server-side resampling). Subimage returns 19+ extensions and empty primary;
# we expect either 3D primary or 3x2D extensions, so standard cutout is used.
LEGACY_USE_SUBIMAGE = False
LEGACY_DOWNLOAD_CHUNK_SIZE = 2**20  # 1 MB for progress bar updates


def _legacy_cutout_url(ra: float, dec: float, size_arcmin: float, bands: str) -> str:
    """Build the Legacy Survey viewer FITS cutout URL."""
    size_pix = int(round(float(size_arcmin) * 60.0 / LEGACY_CUTOUT_PIXSCALE))
    params = {
        "ra": ra,
        "dec": dec,
        "layer": LEGACY_CUTOUT_LAYER,
        "pixscale": LEGACY_CUTOUT_PIXSCALE,
        "bands": bands,
        "size": size_pix,
    }
    url = LEGACY_CUTOUT_BASE + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    if LEGACY_USE_SUBIMAGE:
        url = url + "&subimage"
    return url


def _download_legacy_cutout(
    ra: float,
    dec: float,
    size_arcmin: float,
    bands: str,
    dest_path: Path,
) -> bool:
    """
    Download a Legacy Survey FITS cutout via HTTP with optional progress bar.
    Returns True if the file was written successfully, False otherwise.
    """
    url = _legacy_cutout_url(ra, dec, size_arcmin, bands)
    try:
        logger.info("Downloading Legacy Survey cutout to %s ...", dest_path.name)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = resp.headers.get("content-length")
        total = int(total) if total is not None else None
        pbar = (
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=True,
                desc="Legacy cutout",
                ncols=100,
            )
            if tqdm is not None
            else None
        )
        try:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=LEGACY_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        if pbar is not None:
                            pbar.update(len(chunk))
        finally:
            if pbar is not None:
                pbar.close()
        if dest_path.exists() and dest_path.stat().st_size > 0:
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            logger.debug("Legacy Survey cutout written (%s MB).", f"{size_mb:.1f}")
            return True
        return False
    except (requests.RequestException, OSError):
        return False


def download_legacy_template(
    ra: float,
    dec: float,
    size: float,
    template_folder: str,
    band: str = "r",
) -> Optional[str]:
    """
    Download a Legacy Survey image cutout for use as a template.

    Downloads the combined multi-band FITS (gri) from the Legacy Survey
    viewer cutout service once, then saves g, r, and i band images into
    their template subfolders. Size is interpreted as arcminutes.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in decimal degrees (J2000).
    size : float
        Cutout half-width in arcminutes.
    template_folder : str
        Root directory under which band-specific sub-folders are created.
    band : str
        One of 'g', 'r', 'i'.

    Returns
    -------
    str or None
        Path to the downloaded FITS file for the requested band, or None on failure.
    """
    if not _HAS_LEGACYSTAMPS:
        logger.warning(
            "Legacy Survey template download requested but 'legacystamps' is not installed; skipping."
        )
        return None

    band = str(band).strip().lower()
    if band not in LEGACY_FILTERS:
        logger.info("Band '%s' not available in Legacy Survey [gri]; skipping.", band)
        return None

    # Subfolder names must match find_templates (g_template, etc.).
    # Defer mkdir of template_base until a file is ready to write so a
    # failed download does not leave an empty templates/ folder.
    template_base = Path(template_folder)
    out_path = template_base / f"{band}_template" / f"legacy_{band}_band_template.fits"

    if out_path.exists():
        logger.info("Legacy template already exists at %s - skipping", out_path)
        return str(out_path)

    # Combined multi-band file (same naming as Legacy Survey cutout layer)
    combined_name = f"legacystamps_{ra:.6f}_{dec:.6f}_{LEGACY_CUTOUT_LAYER}.fits"
    combined_path = template_base / combined_name

    # Track whether template_base existed before this call so we can clean up
    # an empty folder if the download fails.
    _template_base_existed = template_base.exists()

    if not combined_path.exists():
        logger.info(
            "Downloading Legacy Survey template (gri) for (%.4f, %.4f), size=%.2f arcmin",
            ra,
            dec,
            size,
        )
        # mkdir just before download; needed to write combined_path.
        template_base.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, LEGACY_DOWNLOAD_ATTEMPTS + 1):
            try:
                ok = _download_legacy_cutout(
                    ra=ra,
                    dec=dec,
                    size_arcmin=size,
                    bands="gri",
                    dest_path=combined_path,
                )
                if ok and combined_path.exists():
                    break
                last_error = RuntimeError("Download did not create expected file")
            except Exception as e:
                last_error = e
                if combined_path.exists():
                    try:
                        combined_path.unlink()
                        logger.info(
                            "Removed partial Legacy Survey file after failed download"
                        )
                    except OSError:
                        pass
                if attempt < LEGACY_DOWNLOAD_ATTEMPTS:
                    logger.warning(
                        "Legacy Survey download attempt %d/%d failed, retrying in %ds ...",
                        attempt,
                        LEGACY_DOWNLOAD_ATTEMPTS,
                        LEGACY_DOWNLOAD_RETRY_DELAY,
                    )
                    time.sleep(LEGACY_DOWNLOAD_RETRY_DELAY)
        if not combined_path.exists():
            logger.error(
                "Legacy Survey cutout download failed after %d attempt(s): %s",
                LEGACY_DOWNLOAD_ATTEMPTS,
                last_error,
            )
            # Clean up empty template_base if we created it and download failed
            if not _template_base_existed:
                try:
                    _remaining = list(template_base.iterdir())
                    if not _remaining:
                        template_base.rmdir()
                        logger.debug(
                            "Removed empty templates folder after failed download: %s",
                            template_base,
                        )
                except OSError:
                    pass
            return None

    # Load combined FITS once and save g, r, i band images
    # Support both formats: single 3D HDU (standard) or one 2D HDU per band (subimage)
    try:
        with fits.open(combined_path) as hdul:
            primary = hdul[0].data
            header = hdul[0].header.copy()
            if primary is not None and primary.ndim == 3:
                data = np.asarray(primary, dtype=float)
            elif len(hdul) >= 3 and all(
                hdul[i].data is not None and hdul[i].data.ndim == 2
                for i in range(1, min(4, len(hdul)))
            ):
                # Legacy viewer cutouts commonly store the bands as extensions 1..3,
                # with an empty primary HDU. Do NOT include HDU[0] here.
                data = np.stack(
                    [np.asarray(hdul[i].data, dtype=float) for i in range(1, 4)],
                    axis=0,
                )
            else:
                raise ValueError(
                    "Legacy FITS: expected 3D primary or 3+ 2D extensions, got "
                    f"primary ndim={getattr(primary, 'ndim', None)}, next={len(hdul)}"
                )
    except Exception:
        logger.exception("Error reading Legacy Survey FITS: %s", combined_path)
        if combined_path.exists():
            try:
                combined_path.unlink()
                logger.info("Removed truncated or corrupt Legacy Survey file for retry")
            except OSError:
                pass
        return None

    if data.ndim != 3:
        logger.warning(
            "Legacy Survey FITS expected 3D (bands, ny, nx), got ndim=%d; skipping band %s",
            data.ndim,
            band,
        )
        return None

    # Read the band order from the header rather than assuming gri.
    # The Legacy cutout header typically provides BANDS='gri' and BAND0/BAND1/BAND2.
    header_bands = None
    try:
        hb = header.get("BANDS")
        if isinstance(hb, str) and hb.strip():
            header_bands = tuple(hb.strip())
    except Exception:
        header_bands = None
    if not header_bands:
        # Fall back to BAND0/BAND1/... keywords.
        band_tokens = []
        for k in ("BAND0", "BAND1", "BAND2"):
            v = header.get(k)
            if isinstance(v, str) and v.strip():
                band_tokens.append(v.strip().lower())
        header_bands = tuple(band_tokens) if band_tokens else None

    if header_bands:
        band_to_idx = {b: i for i, b in enumerate(header_bands)}
        logger.info("Legacy cutout band order from header: %s", "".join(header_bands))
    else:
        band_to_idx = dict(LEGACY_BAND_INDEX)
        logger.info(
            "Legacy cutout band order not found; assuming %s", "".join(LEGACY_FILTERS)
        )

    n_bands = min(len(LEGACY_FILTERS), data.shape[0])
    saved = []

    for b in LEGACY_FILTERS[:n_bands]:
        idx = band_to_idx.get(b)
        if idx is None:
            continue
        if idx >= data.shape[0]:
            continue
        sub_folder = template_base / f"{b}_template"
        sub_folder.mkdir(parents=True, exist_ok=True)
        band_path = sub_folder / f"legacy_{b}_band_template.fits"

        band_data = np.asarray(data[idx], dtype=float)
        band_header = header.copy()
        band_header["FILTER"] = b
        band_header["TELESCOP"] = "LEGACY"
        band_header["INSTRUME"] = "SURVEY"
        band_header["NAXIS"] = 2
        if "NAXIS3" in band_header:
            del band_header["NAXIS3"]

        try:
            safe_fits_write(str(band_path), band_data, band_header)
            saved.append(b)

        except Exception:
            logger.exception("Error writing Legacy template to %s", band_path)

    if saved:
        logger.debug("Legacy band template(s) written: %s", ", ".join(saved))

    if out_path.exists():
        return str(out_path)
    logger.warning(
        "Band '%s' not written from Legacy FITS (shape %s).", band, data.shape
    )
    return None


# =============================================================================
# Download 2MASS Template
# =============================================================================


def download_2mass_template(
    ra: float,
    dec: float,
    size: float,
    template_folder: str,
    band: str = "J",
) -> Optional[Dict[str, str]]:
    """
    Download a 2MASS image cutout via IRSA's Simple Image Access service.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in decimal degrees.
    size : float
        Cutout half-width in arcminutes (max 15).
    template_folder : str
        Root directory for templates.
    band : str
        'J', 'H', or 'Ks' (case-insensitive; 'K' is accepted as alias).

    Returns
    -------
    dict
        ``{band_upper: fpath}`` mapping.

    Raises
    ------
    ValueError
        If *band* is invalid or *size* exceeds the service limit.
    """
    band = band.upper()
    if band not in TWOMASS_VALID_BANDS:
        raise ValueError(
            f"Invalid 2MASS band '{band}'. Choose from {sorted(TWOMASS_VALID_BANDS)}."
        )
    if size > TWOMASS_MAX_SIZE_ARCMIN:
        raise ValueError(
            f"Maximum 2MASS cutout size is {TWOMASS_MAX_SIZE_ARCMIN} arcmin."
        )

    # 2MASS queries use 'KS'; 'K' is accepted as an alias.
    query_band = "KS" if band == "K" else band

    # Compute output path (defer mkdir until write succeeds to avoid
    # leaving empty band subfolders on download failure)
    sub_folder = Path(template_folder) / f"{query_band}_template"
    out_path = sub_folder / f"2MASS_{query_band}_band_template.fits"

    if out_path.exists():
        return {query_band: str(out_path)}

    sia_url = "https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia"
    params = {
        "POS": f"{ra},{dec}",
        "SIZE": size / 60,  # convert arcmin -> deg
        "BAND": query_band,
        "FORMAT": "image/fits",
    }

    try:
        resp = requests.get(sia_url, params=params, timeout=30)
        resp.raise_for_status()
        table = Table.read(BytesIO(resp.content), format="votable")

        if len(table) == 0:
            raise RuntimeError("No image returned for the given coordinates and band.")

        img_resp = requests.get(table[0]["download"], timeout=30)
        img_resp.raise_for_status()
        # Create the subfolder only when we have data ready to write
        sub_folder.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_resp.content)

        # The SIA response has no instrument keywords; stamp them here.
        with fits.open(str(out_path), mode="update") as hdul:
            hdr = hdul[0].header
            hdr["TELESCOP"] = "2MASS"
            hdr["INSTRUME"] = "Survey"
            hdr["FILTER"] = query_band
            hdul.flush()

    except Exception:
        logger.exception("Error downloading 2MASS template")
        return None

    return {query_band: str(out_path)}


# =============================================================================
# PSF stamp chi2 workers  (module scope so multiprocessing can pickle them)
# =============================================================================

# Shared per-worker state broadcast ONCE via Pool(initializer=...) instead of
# being pickled into every per-source task (image + PSF model can be MBs).
_STAMP_CTX: dict = {}


def _psf_stamp_init_shared(ctx):
    _STAMP_CTX.clear()
    _STAMP_CTX.update(ctx)


def _psf_stamp_task(i):
    """Pool entry point: only the source index is pickled per task."""
    c = _STAMP_CTX
    return _psf_stamp_eval(c["xs"][i], c["ys"][i], c)


def _psf_stamp_eval(x, y, ctx):
    """Reduced chi^2 of the PSF-model fit on one stamp.

    Returns NaN when the fit cannot be made (off-frame, masked, too few
    valid pixels, unrenderable model) and +inf for a non-positive fitted
    amplitude.  ``ctx`` carries the invariants shared across positions
    (image, mask, PSF model, fit-box geometry, gain).
    """
    from astropy.stats import biweight_scale
    from limits import _render_epsf_on_cutout
    from photutils.centroids import centroid_2dg

    img = ctx["img"]
    mask = ctx["mask"]
    h = ctx["h"]
    h2 = ctx["h2"]
    ring = ctx["ring"]
    ring_r2 = ctx["ring_r2"]
    gain = ctx["gain"]
    mff = ctx["model_floor_frac"]
    osamp = ctx["osamp"]
    ny, nx = img.shape

    if not (np.isfinite(x) and np.isfinite(y)):
        return np.nan
    xi, yi = int(round(x)), int(round(y))
    if yi - h < 0 or yi + h >= ny or xi - h < 0 or xi + h >= nx:
        return np.nan
    stamp = img[yi - h : yi + h + 1, xi - h : xi + h + 1]
    valid = np.isfinite(stamp)
    if mask is not None:
        m = mask[yi - h : yi + h + 1, xi - h : xi + h + 1]
        valid &= ~np.asarray(m, dtype=bool)
    if int(valid.sum()) < 12:
        return np.nan

    # Local sky scatter from a ring outside the fit box.
    sig0 = np.nan
    if yi - h2 >= 0 and yi + h2 < ny and xi - h2 >= 0 and xi + h2 < nx:
        wide = img[yi - h2 : yi + h2 + 1, xi - h2 : xi + h2 + 1]
        ring_pix = wide[ring & np.isfinite(wide)]
        if ring_pix.size >= 8:
            sig0 = float(biweight_scale(ring_pix, ignore_nan=True))
    if not np.isfinite(sig0) or sig0 <= 0:
        edge_pix = stamp[valid & (ring_r2 > (0.6 * h) ** 2)]
        if edge_pix.size >= 6:
            sig0 = float(biweight_scale(edge_pix, ignore_nan=True))
    if not np.isfinite(sig0) or sig0 <= 0:
        return np.nan

    # Re-centre the model on the stamp centroid so a small catalog
    # position error is not mistaken for a PSF-shape mismatch.
    lx, ly = x - (xi - h), y - (yi - h)
    try:
        _cen = centroid_2dg(stamp, mask=~valid)
        if (
            _cen is not None
            and np.isfinite(_cen[0])
            and np.isfinite(_cen[1])
            and 0 <= _cen[0] <= 2 * h
            and 0 <= _cen[1] <= 2 * h
        ):
            lx, ly = float(_cen[0]), float(_cen[1])
    except Exception:
        pass

    _m = ctx["psf_model"]
    if ctx["psf_is_gridded"]:
        try:
            from psf import epsf_at_position as _epsf_at_pos

            _m = _epsf_at_pos(_m, x, y)
        except Exception:
            _m = None
        if _m is None:
            return np.nan
    try:
        P = np.asarray(
            _render_epsf_on_cutout(
                _m, 2 * h + 1, 2 * h + 1, lx, ly, 1.0, osamp
            ),
            dtype=float,
        )
    except Exception:
        return np.nan
    valid &= np.isfinite(P)
    if int(valid.sum()) < 12:
        return np.nan

    # WLS fit of stamp = A*P + c.  Two passes: the first uses the
    # data for the Poisson term, the second the fitted model.
    d = np.where(valid, stamp, 0.0)
    Pv = np.where(valid, P, 0.0)
    A = 0.0
    chi2_red = np.nan
    for _pass in range(2):
        var = sig0 ** 2 + np.clip(A * Pv, 0.0, None) / gain
        if mff > 0:
            var = var + (mff * np.abs(A) * Pv) ** 2
        w = np.where(valid, 1.0 / np.maximum(var, 1e-12), 0.0)
        Spp = float(np.sum(w * Pv * Pv))
        Sp = float(np.sum(w * Pv))
        S = float(np.sum(w))
        Spd = float(np.sum(w * Pv * d))
        Sd = float(np.sum(w * d))
        det = Spp * S - Sp * Sp
        if not np.isfinite(det) or abs(det) < 1e-20:
            break
        A = (S * Spd - Sp * Sd) / det
        c = (Spp * Sd - Sp * Spd) / det
        dof = int(valid.sum()) - 2
        chi2_red = (
            float(np.sum(w * (d - A * Pv - c) ** 2)) / dof
            if dof > 0
            else np.nan
        )
    if not np.isfinite(A) or A <= 0:
        return np.inf
    return chi2_red


# =============================================================================
# Constrained Slope Regressor
# =============================================================================
# Canonical implementation lives in zeropoint.PenalisedSlopeRegressor.
# Alias kept here so all call sites in this file continue to work unchanged.

from zeropoint import PenalisedSlopeRegressor as ConstrainedSlopeRegressor  # noqa: E402
from zeropoint import FixedSlopeRegressor  # noqa: E402


# =============================================================================
# Templates Class - Main Processing Pipeline
# =============================================================================


class Templates:
    """
    End-to-end handler for template-based image subtraction.

    Responsibilities:
      - Source masking (saturated, bright, extended objects).
      - Image alignment (SWarp / AstroAlign / WCS reproject).
      - Image cropping to the maximal overlap region.
      - Flux calibration between science and template catalogs.
      - Image subtraction (ZOGY / SFFT / HOTPANTS).

    Parameters
    ----------
    input_yaml : dict
        Pipeline configuration dictionary.  Expected keys include
        ``target_ra``, ``target_dec``, ``target_x_pix``, ``target_y_pix``,
        ``imageFilter``, ``fits_dir``, ``scale``, ``fpath``, and
        ``template_subtraction`` (a nested dict with subtraction settings).
    """

    def __init__(self, input_yaml: Dict[str, Any]):
        self.input_yaml = input_yaml

    # -----------------------------------------------------------------
    # Masking utilities
    # -----------------------------------------------------------------

    @staticmethod
    def apply_mask_to_fits(fits_file: str, mask_file: str) -> None:
        """
        Set pixels to NaN wherever *mask_file* is True, modifying
        *fits_file* on disk.
        """
        with fits.open(fits_file, mode="update") as hdul_data:
            with fits.open(mask_file) as hdul_mask:
                mask = hdul_mask[0].data.astype(bool)
            hdul_data[0].data[mask] = np.nan
            hdul_data.flush()

    @staticmethod
    def create_source_mask(
        dataframe: pd.DataFrame,
        shape: Tuple[int, int],
        radius: int = 7,
        nsources: int = 10,
    ) -> np.ndarray:
        """
        Build a binary mask with circular apertures at source positions.

        Returns an *inverse* mask: 1 = unmasked, 0 = masked.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Must contain 'x_pix' and 'y_pix' columns.
        shape : (int, int)
            Image dimensions (height, width).
        radius : int
            Aperture radius in pixels.
        nsources : int
            Use only the first *nsources* rows of *dataframe*.
        """
        mask = np.zeros(shape, dtype=np.int32)
        subset = dataframe.head(nsources)
        logger.info("Creating source mask with %d sources", len(subset))

        positions = list(zip(subset["x_pix"].values, subset["y_pix"].values))
        if positions:
            aps = CircularAperture(positions, r=radius)
            for ap_mask in aps.to_mask(method="center"):
                img = ap_mask.to_image(shape=shape)
                if img is not None:
                    mask += img.astype(np.int32)

        # Collapse overlapping apertures
        mask = np.clip(mask, 0, 1)
        return 1 - mask  # invert: 1=keep, 0=masked

    @staticmethod
    def does_box_overlap(
        source: Dict[str, float],
        point: Tuple[float, float],
        padding: int = 10,
    ) -> bool:
        """Check whether *point* falls inside the padded bounding box of *source*."""
        x, y = point
        return (source["bbox_xmin"] - padding) <= x <= (
            source["bbox_xmax"] + padding
        ) and (source["bbox_ymin"] - padding) <= y <= (source["bbox_ymax"] + padding)

    @staticmethod
    def find_non_uniform_center(
        img: np.ndarray,
    ) -> Tuple[float, float, int, int, int, int]:
        """
        Locate the bounding box of the non-uniform (data-bearing) region.

        Uniform rows/columns (where every pixel equals the first element)
        are assumed to be padding from a prior reproject or mosaic step.

        Returns
        -------
        (center_y, center_x, top_row, bottom_row, left_col, right_col)
        """
        # NaN-aware uniformity test: NaN == NaN is False, so the plain
        # equality check flagged every NaN-padded row as "data".  A row or
        # column counts as data-bearing only when its finite pixels span a
        # range larger than atol; all-NaN, all-zero, and constant-value
        # padding rows/columns are excluded.  A small tolerance also
        # absorbs resampling/compression noise in the padding.
        atol = 1e-8
        finite = np.isfinite(img)
        img_f = np.where(finite, img, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_ptp = np.nanmax(img_f, axis=1) - np.nanmin(img_f, axis=1)
            col_ptp = np.nanmax(img_f, axis=0) - np.nanmin(img_f, axis=0)
        non_uniform_rows = finite.any(axis=1) & (
            np.nan_to_num(row_ptp, nan=0.0) > atol
        )
        non_uniform_cols = finite.any(axis=0) & (
            np.nan_to_num(col_ptp, nan=0.0) > atol
        )

        row_idx = np.where(non_uniform_rows)[0]
        col_idx = np.where(non_uniform_cols)[0]

        if row_idx.size == 0 or col_idx.size == 0:
            # All-uniform or all-invalid image: there is no padding to
            # find, so report the full frame instead of raising IndexError.
            ny, nx = img.shape
            logger.debug(
                "find_non_uniform_center: image has no non-uniform region; "
                "using full frame bounds."
            )
            return ny / 2.0, nx / 2.0, 0, ny - 1, 0, nx - 1

        top, bottom = int(row_idx[0]), int(row_idx[-1])
        left, right = int(col_idx[0]), int(col_idx[-1])

        center_y = (top + bottom) / 2.0
        center_x = (left + right) / 2.0

        return center_y, center_x, top, bottom, left, right

    @staticmethod
    def _trim_nan_boundaries(image_data, header, buffer_pixels=10):
        """
        Trim image to remove NaN boundary regions.

        Parameters
        ----------
        image_data : np.ndarray
            2D image array (can contain NaNs)
        header : fits.Header
            FITS header to update with new WCS after trimming
        buffer_pixels : int
            Minimum buffer around valid data region

        Returns
        -------
        trimmed_data : np.ndarray
            Image with NaN boundaries removed
        trimmed_header : fits.Header
            Updated header with corrected WCS
        trim_info : dict
            Information about trimming performed
        """
        from astropy.wcs import WCS

        valid_mask = ~np.isnan(image_data)

        if not np.any(valid_mask) or np.all(valid_mask):
            return image_data, header, {"trimmed": False}

        rows_with_valid = np.any(valid_mask, axis=1)
        cols_with_valid = np.any(valid_mask, axis=0)

        if not np.any(rows_with_valid) or not np.any(cols_with_valid):
            return image_data, header, {"trimmed": False}

        y_min, y_max = np.where(rows_with_valid)[0][[0, -1]]
        x_min, x_max = np.where(cols_with_valid)[0][[0, -1]]

        y_min = max(0, y_min - buffer_pixels)
        y_max = min(image_data.shape[0] - 1, y_max + buffer_pixels)
        x_min = max(0, x_min - buffer_pixels)
        x_max = min(image_data.shape[1] - 1, x_max + buffer_pixels)

        trimmed_data = image_data[y_min:y_max+1, x_min:x_max+1]
        trimmed_header = header.copy()

        try:
            wcs = get_wcs(trimmed_header)
            # Shift the WCS reference pixel by the crop offset.
            if 'CRPIX1' in trimmed_header:
                trimmed_header['CRPIX1'] -= x_min
            if 'CRPIX2' in trimmed_header:
                trimmed_header['CRPIX2'] -= y_min

            trimmed_header['NAXIS1'] = trimmed_data.shape[1]
            trimmed_header['NAXIS2'] = trimmed_data.shape[0]

            # The CD matrix stays valid after a CRPIX shift; leave it alone.

            trimmed_header.add_history(f'Trimmed: removed NaN boundaries [{x_min}:{x_max+1},{y_min}:{y_max+1}]')
        except Exception as wcs_exc:
            # WCS update failed; still fix the dimensions.
            trimmed_header['NAXIS1'] = trimmed_data.shape[1]
            trimmed_header['NAXIS2'] = trimmed_data.shape[0]
            trimmed_header.add_history(f'Trimmed: removed NaN boundaries (WCS update failed)')
        
        trim_info = {
            "trimmed": True,
            "x_slice": (x_min, x_max + 1),
            "y_slice": (y_min, y_max + 1),
            "original_shape": image_data.shape,
            "trimmed_shape": trimmed_data.shape
        }
        
        return trimmed_data, trimmed_header, trim_info

    def find_bright_sources(
        self,
        header: fits.Header,
        usefilter: Optional[List[str]] = None,
        magCutoff: Optional[List[float]] = None,
        catalogName: str = "refcat",
    ) -> Optional[pd.DataFrame]:
        """
        Query an astrometric catalog for bright sources within the field.

        Parameters
        ----------
        header : fits.Header
            FITS header with WCS information.
        usefilter : list of str
            Photometric band(s) to query (default: ['J']).
        magCutoff : list of float
            Magnitude ceiling(s) per filter (default: [13]).
        catalogName : str
            Catalog identifier understood by ``catalog.download()``.

        Returns
        -------
        pd.DataFrame or None
            Two-column frame (x_pix, y_pix) of bright sources.
        """
        if usefilter is None:
            usefilter = ["J"]
        if magCutoff is None:
            magCutoff = [13.0]

        try:
            logger.info(
                log_step(f"Mask bright sources: {catalogName}")
            )
            target = SkyCoord(
                self.input_yaml["target_ra"],
                self.input_yaml["target_dec"],
                unit=(u.deg, u.deg),
            )
            imageWCS = get_wcs(header)

            # Size the cone to cover (and be limited by) the detector
            # footprint so bright-source masking is not biased to the
            # field centre; works off the pixel scale when the image has
            # no WCS.
            _fbs_radius = 10.0
            try:
                from catalog import (
                    _catalog_cone_radius_arcmin,
                    _pixscale_arcsec,
                )

                _nx = int(header.get("NAXIS1") or 0)
                _ny = int(header.get("NAXIS2") or 0)
                _fbs_radius = _catalog_cone_radius_arcmin(
                    [
                        {
                            "wcs": imageWCS,
                            "shape": (_ny, _nx)
                            if _nx > 0 and _ny > 0
                            else None,
                            "pixscale": _pixscale_arcsec(header)
                            or self.input_yaml.get("pixel_scale"),
                        }
                    ],
                    target,
                    default_arcmin=10.0,
                )
            except Exception:
                pass

            sequenceData = Catalog(input_yaml=self.input_yaml)
            bright_sources_catalog = sequenceData.download(
                target,
                catalogName=catalogName,
                radius=_fbs_radius,
            )
            if bright_sources_catalog is None:
                return None

            bright_sources_catalog = sequenceData.clean(
                selectedCatalog=bright_sources_catalog,
                catalogName=catalogName,
                image_wcs=imageWCS,
                get_local_sources=False,
                border=0,
                full_clean=False,
                usefilter=usefilter,
                magCutoff=magCutoff,
            )
            return bright_sources_catalog[["x_pix", "y_pix"]]

        except Exception:
            logger.exception("Error finding bright sources")
            return None

    def create_image_mask(
        self,
        data: np.ndarray,
        params: Optional[MaskParams] = None,
        *,
        # Legacy keyword interface (used if params is None)
        sat_lvl: float = 2**16,
        saturate_frac: float = 0.90,
        fwhm: int = 5,
        npixels: int = 8,
        padding: int = 10,
        snr_limit: int = 3000,
        create_source_mask: bool = True,
        ignore_position: Optional[List[Tuple[float, float]]] = None,
        remove_large_sources: bool = False,
        mask_bright_catalog_overlaps: bool = False,
        bright_sources: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Build a binary mask flagging problematic pixels/sources in *data*.

        The mask includes:
          - Saturated pixels (``data >= saturate_frac * sat_lvl``), dilated
            to cover the bleeding wings.  Runs even when source detection
            finds nothing.
          - Negative-peak sources (likely artefacts).
          - Every detected source when ``create_source_mask`` is True.
          - Anomalously large sources (optional, via sigma-clip on area, disabled by default).
          - Sources overlapping known bright-catalog objects (optional, disabled by default).

        Parameters
        ----------
        data : np.ndarray
            2-D science image.
        params : MaskParams or None
            Structured parameter set.  If None, falls back to keyword args.
        create_source_mask : bool
            If True, mask all detected sources; if False (typical for
            subtraction inputs), keep unsaturated point sources unmasked so
            they remain usable for kernel fitting and flux calibration.
        ignore_position : list of (x, y)
            Positions whose enclosing mask component should *not* be masked
            (typically the transient target).  Masked pixels are blanked to
            NaN in the difference image, so masking the target would erase
            the transient itself.
        mask_bright_catalog_overlaps : bool
            If True, mask sources overlapping bright catalog objects (default False).

        Returns
        -------
        mask : np.ndarray (int)
            Binary mask (1 = masked, 0 = good).
        masked_centres : list of (x, y)
            Centres of all masked components/sources.
        """
        if ignore_position is None:
            ignore_position = []

        # Positions that must stay unmasked (typically the transient target).
        # A masked footprint containing one of these positions is dropped:
        # masked regions are blanked to NaN in the difference image, so
        # masking the target would erase the transient itself.
        ignore_xy = []
        for _pos in ignore_position:
            try:
                _px, _py = float(_pos[0]), float(_pos[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(_px) and np.isfinite(_py):
                ignore_xy.append((_px, _py))

        def _hits_ignore(footprint: np.ndarray) -> bool:
            if not ignore_xy:
                return False
            ny_, nx_ = footprint.shape
            for _px, _py in ignore_xy:
                _ix, _iy = int(round(_px)), int(round(_py))
                if 0 <= _ix < nx_ and 0 <= _iy < ny_ and footprint[_iy, _ix]:
                    return True
            return False

        # Be strict about the input type for photutils.
        data = np.asarray(data)
        if data.ndim != 2:
            raise TypeError(
                f"create_image_mask expected 2-D image data, got shape {data.shape}"
            )

        # Resolve parameters: prefer dataclass, fall back to kwargs
        if params is not None:
            sat_lvl = params.saturation_level
            saturate_frac = params.saturate_frac
            fwhm = params.fwhm
            padding = params.padding
            snr_limit = params.snr_limit

        masked_centres: List[Tuple[float, float]] = []
        mask = np.zeros(data.shape, dtype=np.int32)

        try:
            if data.shape[0] < fwhm or data.shape[1] < fwhm:
                logger.warning(
                    f"create_image_mask: image shape {data.shape} is smaller than FWHM {fwhm}; returning empty mask."
                )
                return mask, masked_centres

            fwhm = int(fwhm)
            if fwhm <= 0:
                logger.warning("create_image_mask: invalid FWHM %s; setting to default 5.", fwhm)
                fwhm = 5
            if sat_lvl <= 0:
                logger.warning("create_image_mask: invalid sat_lvl %s; setting to 2^16.", sat_lvl)
                sat_lvl = 2**16
            padding = int(padding)
            if padding < 0:
                logger.warning("create_image_mask: invalid padding %s; setting to 0.", padding)
                padding = 0

            # Images that are mostly NaN cannot yield a useful mask.
            nan_mask = ~np.isfinite(data)
            nan_fraction = np.sum(nan_mask) / data.size
            if nan_fraction > 0.9:
                logger.warning(
                    f"create_image_mask: image has {100*nan_fraction:.1f}% NaNs; returning empty mask."
                )
                return mask, masked_centres
            elif nan_fraction > 0.5:
                logger.warning(
                    f"create_image_mask: image has {100*nan_fraction:.1f}% NaNs; mask may be unreliable."
                )

            finite_data = data[np.isfinite(data)]
            if len(finite_data) == 0:
                logger.warning("create_image_mask: no finite pixels in image; returning empty mask.")
                return mask, masked_centres

            # --- Saturation masking ---
            # Pixels at or above saturate_frac * sat_lvl are non-linear; the
            # subtraction kernel cannot model them, and their unmasked
            # footprints leave DIFF ~= SCI artefacts that read as transients.
            # Dilate to cover the bleeding wings and record the component
            # centres for downstream residual diagnostics.  This is done
            # before source detection so saturated fields that fail the
            # segmentation steps below still get masked.
            sat_threshold = saturate_frac * sat_lvl
            if np.isfinite(sat_threshold) and sat_threshold > 0:
                sat_pixels = np.isfinite(data) & (data >= sat_threshold)
                if np.any(sat_pixels):
                    dilate_r = max(
                        2,
                        int(
                            np.ceil(
                                DEFAULT_FWHM_PADDING_MULTIPLIER * fwhm
                            )
                        ),
                    )
                    _gy, _gx = np.ogrid[
                        -dilate_r : dilate_r + 1, -dilate_r : dilate_r + 1
                    ]
                    _disk = (_gx ** 2 + _gy ** 2) <= dilate_r ** 2
                    sat_mask = binary_dilation(sat_pixels, structure=_disk)

                    from scipy.ndimage import (
                        center_of_mass as _com,
                        label as _ndi_label,
                    )

                    _lab, _nlab = _ndi_label(sat_mask)

                    # Drop dilated components that contain an ignore position.
                    if ignore_xy and _nlab > 0:
                        _drop = np.zeros(_nlab + 1, dtype=bool)
                        for _px, _py in ignore_xy:
                            _ix, _iy = int(round(_px)), int(round(_py))
                            if (
                                0 <= _ix < sat_mask.shape[1]
                                and 0 <= _iy < sat_mask.shape[0]
                            ):
                                _drop[_lab[_iy, _ix]] = True
                        if np.any(_drop[1:]):
                            sat_mask[_drop[_lab]] = False
                            _lab, _nlab = _ndi_label(sat_mask)

                    mask[sat_mask] = 1
                    n_sat = int(np.count_nonzero(sat_mask))
                    if n_sat > 0:
                        logger.info(
                            "create_image_mask: masked %d saturated pixels "
                            "(>= %.4g ADU, dilate=%d px).",
                            n_sat, sat_threshold, dilate_r,
                        )
                        for _li in range(1, _nlab + 1):
                            _cy_c, _cx_c = _com(
                                sat_mask, labels=_lab, index=_li
                            )
                            if np.isfinite(_cx_c) and np.isfinite(_cy_c):
                                masked_centres.append(
                                    (float(_cx_c), float(_cy_c))
                                )

            from functions import biweight_stdfunc

            _, image_median, image_std = sigma_clipped_stats(
                finite_data,
                sigma=DEFAULT_SIGMA_CLIP,
                cenfunc=np.nanmedian,
                stdfunc=biweight_stdfunc,
            )

            if not np.isfinite(image_std) or image_std <= 0:
                logger.warning(
                    f"create_image_mask: invalid image_std {image_std}; setting to robust estimate."
                )
                image_std = np.nanstd(finite_data)
                if not np.isfinite(image_std) or image_std <= 0:
                    image_std = 1.0  # last-resort default

            # Kernel builder requires an even FWHM.
            if fwhm % 2 != 0:
                fwhm += 1

            # Detection threshold and minimum connected area
            npixels_det_raw = float(np.pi * (float(fwhm) / 2.0) ** 2)
            npixels_det = int(npixels_det_raw)
            if npixels_det < 5:
                logger.debug(
                    "create_image_mask: computed npixels=%d from fwhm=%s (raw=%g); clamping to 5.",
                    int(npixels_det),
                    str(fwhm),
                    float(npixels_det_raw),
                )
                npixels_det = 5
            # Cap at ~1% of the image (max 100 px) to keep detections point-like.
            max_npixels = min(100, data.shape[0] * data.shape[1] // 100)
            if npixels_det > max_npixels:
                npixels_det = max_npixels

            threshold = 3.0 * image_std + image_median
            if not np.isfinite(threshold):
                logger.warning("create_image_mask: threshold is NaN; using percentile-based threshold.")
                threshold = np.percentile(finite_data, 95)
            if threshold <= image_median:
                logger.warning("create_image_mask: threshold %s <= median %s; using 2*std.", threshold, image_median)
                threshold = 2.0 * image_std + image_median

            # detect_sources cannot handle NaN; fill with the image median.
            data_for_detection = data.copy()
            data_for_detection[~np.isfinite(data_for_detection)] = image_median

            seg = detect_sources(
                data_for_detection, threshold, n_pixels=npixels_det
            )
            if seg is None:
                logger.warning(
                    "create_image_mask: detect_sources returned None; returning empty mask."
                )
                return mask, masked_centres

            # Photutils >= 1.1 expects a SegmentationImage for SourceCatalog;
            # older versions may return a plain array, so normalise here.
            if not isinstance(seg, SegmentationImage):
                seg_arr = np.asarray(getattr(seg, "data", seg))
                if seg_arr.ndim != 2:
                    raise TypeError(
                        f"create_image_mask: expected 2-D segmentation array, got shape {seg_arr.shape}"
                    )
                seg = SegmentationImage(seg_arr)

            if seg.n_labels == 0:
                logger.warning("create_image_mask: no sources detected; returning empty mask.")
                return mask, masked_centres

            cat = SourceCatalog(
                data_for_detection, seg, local_bkg_width=15 * fwhm
            )
            tbl = cat.to_table().to_pandas()

            if len(tbl) == 0:
                logger.warning("create_image_mask: source catalog is empty; returning empty mask.")
                return mask, masked_centres

            is_negative = tbl["max_value"] < 0

            # Mask negative-peak sources (likely artefacts) directly from their
            # segmentation footprints BEFORE removing them from tbl.
            # Skip footprints containing an ignore position (the target):
            # masked pixels are blanked in the difference image.
            for bad_src in tbl[is_negative].itertuples(index=False):
                seg_pixels = seg.data == bad_src.label
                if _hits_ignore(seg_pixels):
                    continue
                mask[seg_pixels] = 1
                cx = (bad_src.bbox_xmin + bad_src.bbox_xmax) / 2
                cy = (bad_src.bbox_ymin + bad_src.bbox_ymax) / 2
                masked_centres.append((float(cx), float(cy)))

            # Optional: mask every detected source (not just artefacts).
            # Callers that need a pure defect mask pass
            # create_source_mask=False so point sources remain available
            # for kernel fitting and flux calibration.
            if create_source_mask:
                for src in tbl.itertuples(index=False):
                    seg_pixels = seg.data == src.label
                    if _hits_ignore(seg_pixels):
                        continue
                    mask[seg_pixels] = 1
                    cx = (src.bbox_xmin + src.bbox_xmax) / 2
                    cy = (src.bbox_ymin + src.bbox_ymax) / 2
                    masked_centres.append((float(cx), float(cy)))

            if remove_large_sources:
                # Handle case where all areas are the same (sigma_clip will fail)
                if len(tbl["area"]) > 1 and np.std(tbl["area"]) > 0:
                    clipped = sigma_clip(tbl["area"], sigma=10, maxiters=10)
                    is_large = clipped.mask
                else:
                    is_large = np.zeros(len(tbl), dtype=bool)
                tbl = tbl[~(is_negative | is_large)]
            else:
                tbl = tbl[~(is_negative)]

            if len(tbl) == 0:
                logger.warning("create_image_mask: all sources filtered out; returning empty mask.")
                return mask, masked_centres

            # --- Mask bright-catalog overlaps (optional) ---
            if mask_bright_catalog_overlaps and bright_sources is not None and len(bright_sources) > 0:
                bright_xy = list(zip(bright_sources["x_pix"].values,
                                     bright_sources["y_pix"].values))
                for src in tbl.itertuples(index=False):
                    cx = (src.bbox_xmin + src.bbox_xmax) / 2
                    cy = (src.bbox_ymin + src.bbox_ymax) / 2
                    src_dict = src._asdict()
                    overlaps = any(
                        self.does_box_overlap(src_dict, bxy, padding)
                        for bxy in bright_xy
                    )
                    if overlaps:
                        seg_pixels = seg.data == src.label
                        mask[seg_pixels] = 1
                        masked_centres.append((cx, cy))

            # NOTE: Do NOT mask remaining unsaturated sources - they are needed
            # for flux calibration and kernel fitting. Only saturated/negative
            # sources and bright catalog overlaps should be masked.

            mask = np.clip(mask, 0, 1)

        except Exception:
            logger.exception("Error in create_image_mask")
            raise

        return mask, masked_centres

    # -----------------------------------------------------------------
    # Template discovery
    # -----------------------------------------------------------------

    def get_template(self) -> Optional[str]:
        """
        Locate a pre-downloaded template FITS file for the current filter.

        Searches under ``<fits_dir>/templates/<filter>_template/`` for
        files ending in .fits / .fts / .fit that contain "_template" in
        their name (excluding PSF models and weight maps).

        Returns
        -------
        str or None
            Path to the first matching file, or None if not found.
        """
        use_filter = self.input_yaml.get("imageFilter")
        if not use_filter:
            logger.info("Image filter not specified in input YAML")
            return None

        logger.info(log_step(f"Template for filter {use_filter}"))
        fits_root = Path(self.input_yaml.get("fits_dir", None) or "") / "templates"
        use_filter = str(use_filter).strip()

        # Prefer <f>_template; also accept the legacy <f>p_template folder
        # (Pan-STARRS/Sloan naming convention).
        dir_labels = [f"{use_filter}_template"]
        if use_filter in {"u", "g", "r", "i", "z"}:
            dir_labels.append(f"{use_filter}p_template")


        candidate_dirs = [fits_root / d for d in dir_labels]
        logger.debug(
            "Expected %s-band template folder(s): %s",
            use_filter,
            ", ".join(str(d) for d in candidate_dirs),
        )

        for template_dir in candidate_dirs:
            if not template_dir.is_dir():
                continue
            candidates = [
                p
                for p in template_dir.iterdir()
                if p.suffix.lower() in {".fits", ".fts", ".fit"}
                and "PSF_model_" not in p.name
                and ".weight" not in p.name.lower()
            ]
            if not candidates:
                logger.info("No template files found in %s", template_dir)
                continue
            
            # Prefer exact <f>_template.fits / <f>p_template.fits names over
            # arbitrary .fits files sitting in the folder.
            preferred_names = [f"{use_filter}_template.fits"]
            if use_filter in {"u", "g", "r", "i", "z"}:
                preferred_names.append(f"{use_filter}p_template.fits")
            
            preferred_candidate = None
            for pref_name in preferred_names:
                preferred_candidate = next((p for p in candidates if p.name == pref_name), None)
                if preferred_candidate:
                    result = str(preferred_candidate)
                    logger.debug("Template filepath: %s", result)
                    return result
            
            result = str(candidates[0])
            if template_dir.name.endswith("_template"):
                logger.debug("Template filepath (legacy folder): %s", result)
            else:
                logger.debug("Template filepath: %s", result)
            return result

        logger.info(
            "Template directory does not exist: %s",
            ", ".join(str(d) for d in candidate_dirs),
        )
        return None

    # -----------------------------------------------------------------
    # Alignment
    # -----------------------------------------------------------------

    def align(
        self,
        scienceFpath: str,
        templateFpath: str,
        imageCatalog: Optional[Any] = None,
        center: Optional[Any] = None,
        method: str = "spalipy",
    ) -> AlignmentResult:
        """
        Align science and template images to a common pixel grid.

        Cascade order (first success wins):
          - ``swarp``: try SCAMP+SWarp, then WCS reproject, then AstroAlign.
          - ``astroalign``: try AstroAlign, then WCS reproject, then SCAMP+SWarp.
          - ``reproject``: try WCS reproject, then AstroAlign, then SCAMP+SWarp.
          - ``spalipy``: try spalipy (spline-warp), then SCAMP+SWarp, reproject, AstroAlign.
          - ``tweakwcs``: try tweakwcs (STScI WCS tweaking + reproject), then SCAMP+SWarp, reproject, AstroAlign.
          - ``chi2_shift``: try chi2_shift (cross-correlation for extended fields), then SCAMP+SWarp, reproject, AstroAlign.

        Images are loaded once and passed to module-level helpers to avoid
        redundant FITS I/O. ReprojectConfig and projection headers are built
        once per call and shared across fallback attempts.

        Returns
        -------
        AlignmentResult
            ``result.success`` is False when every method failed - the
            caller must then disable subtraction rather than subtract the
            unaligned inputs.  Per-attempt metrics are also written to
            ``Alignment_<science base>.json`` next to the science image.
        """
        sci_name = Path(scienceFpath).name
        ref_name = Path(templateFpath).name
        logger.info(
            log_step(f"Align: {sci_name} vs {ref_name}")
        )

        try:
            scienceDir = Path(scienceFpath).parent
            new_templateFpath = str(scienceDir / Path(templateFpath).name)

            # Load once; both images are reused across every fallback strategy.
            scienceImage, scienceHeader = read_fits(scienceFpath)
            templateImage, templateHeader = read_fits(templateFpath)

            # Fail fast on non-2D input before running any strategy.
            if scienceImage.ndim != 2 or templateImage.ndim != 2:
                logger.error(
                    "align: expected 2D images, got science=%s template=%s",
                    scienceImage.shape,
                    templateImage.shape,
                )
                return AlignmentResult(
                    None, None, None, None,
                    message="non-2D input",
                )

            # input_yaml["fwhm"] still holds the YAML config default at this
            # point - main.py only writes the measured value after template
            # subtraction.  The measured FWHM is stamped on the science header;
            # prefer it so the FWHM-scaled quality gates are calibrated to the
            # real PSF rather than the 3 px placeholder.
            fwhm_pix = float(self.input_yaml.get("fwhm", 3.0))
            try:
                _sci_hdr_fwhm = scienceHeader.get(
                    "FWHM", scienceHeader.get("fwhm", scienceHeader.get("FWHMPIX"))
                )
                if (
                    _sci_hdr_fwhm is not None
                    and np.isfinite(float(_sci_hdr_fwhm))
                    and float(_sci_hdr_fwhm) > 0
                ):
                    fwhm_pix = float(_sci_hdr_fwhm)
            except (TypeError, ValueError):
                pass
            _ts_cfg = self.input_yaml.get("template_subtraction", {}) or {}

            # Shared by every reproject attempt below.
            reproject_cfg = ReprojectConfig.from_yaml(self.input_yaml)

            # One quality policy for every backend: FWHM-scaled offset/RMS/
            # P90 gates feeding the shared pass/warn/fail classifier.
            _gates = _alignment_gate_thresholds(self.input_yaml, fwhm_pix)

            def _fail(method_name, msg=""):
                return AlignmentResult(
                    None, None, method_name, None,
                    quality="fail", message=msg,
                )

            _attempts: List[AlignmentResult] = []

            def _attempt(fn) -> AlignmentResult:
                res = fn()
                _attempts.append(res)
                return res

            def _finish(result: AlignmentResult) -> AlignmentResult:
                # Provenance: every attempted backend and its gate outcome is
                # recorded next to the science image even when the run fails.
                _write_alignment_report(
                    _attempts,
                    str(Path(scienceFpath).parent),
                    scienceFpath,
                    templateFpath,
                    requested_method=str(method),
                )
                # Stamp the shared gate classification onto the aligned
                # template so downstream provenance does not depend on which
                # backend produced the file.
                if result.success and result.template_path:
                    try:
                        with fits.open(
                            result.template_path, mode="update", memmap=False
                        ) as _hdl:
                            _h = _hdl[0].header
                            _h["ALIGQUAL"] = (
                                str(result.quality),
                                "Alignment quality class (pass/warn/unverified)",
                            )
                            if result.n_matches is not None:
                                _h["ALIGNM"] = (
                                    int(result.n_matches),
                                    "Matched alignment sources",
                                )
                            _hdl.flush()
                        invalidate_fits_cache(result.template_path)
                    except Exception:
                        logger.debug(
                            "align: failed to stamp quality header",
                            exc_info=True,
                        )
                return result

            # The science image is untouched by the template-only resampling
            # methods (reproject/spalipy/chi2_shift/tweakwcs), so its source
            # catalog is detected once and reused in every post-alignment
            # quality measurement instead of re-running SExtractor per method.
            _sci_det_cache: Dict[str, Any] = {"done": False, "xy": None}

            def _science_xy() -> Optional[np.ndarray]:
                if not _sci_det_cache["done"]:
                    _sci_det_cache["done"] = True
                    try:
                        _thresh = float(
                            _ts_cfg.get("alignment_rms_detect_thresh", 5.0)
                            or 5.0
                        )
                        _xy, _, _ = _detect_sextractor_sources(
                            scienceImage,
                            input_yaml=self.input_yaml,
                            fwhm_pix=min(max(float(fwhm_pix), 2.0), 8.0),
                            thresh=_thresh,
                        )
                        if _xy is not None and len(_xy) >= 5:
                            _sci_det_cache["xy"] = _xy
                    except Exception:
                        _sci_det_cache["xy"] = None
                return _sci_det_cache["xy"]

            # ------------------------------------------------------------------
            # Strategy helpers
            # ------------------------------------------------------------------

            def _swarp() -> AlignmentResult:
                if run_IDC is None:
                    logger.info("run_IDC not available; skipping SWarp.")
                    return _fail("swarp", "run_IDC not available")
                logger.info("Attempting SWarp + SCAMP alignment.")
                idc = run_IDC.ImageDistortionCorrector(input_yaml=self.input_yaml)
                
                ts_cfg = self.input_yaml.get("template_subtraction", {})
                skip_resampling = bool(ts_cfg.get("skip_resampling", False))
                resample_mode = str(ts_cfg.get("resample_mode", "")).strip().lower()
                
                # Backward compatibility: skip_resampling=True maps to "wcs_only"
                if not resample_mode:
                    resample_mode = "wcs_only" if skip_resampling else "common_grid"
                
                valid_modes = ("common_grid", "native_scale", "wcs_only")
                if resample_mode not in valid_modes:
                    logger.warning(
                        f"Invalid resample_mode '{resample_mode}'; falling back to 'common_grid'. "
                        f"Valid modes: {valid_modes}"
                    )
                    resample_mode = "common_grid"
                
                if resample_mode != "common_grid":
                    logger.info(
                        f"Using alignment resample_mode='{resample_mode}' for template subtraction"
                    )
                
                res = idc.align_and_resample_both_images(
                    scienceFpath, templateFpath, resample_mode=resample_mode
                )
                if not res or not res.get("science_aligned"):
                    logger.info("SWarp alignment did not produce aligned outputs.")
                    return _fail("swarp", "no aligned outputs")
                sci_al = res["science_aligned"]
                ref_al = res["reference_aligned"]
                # Extra read just for the RMS diagnostic (not the alignment I/O).
                _swarp_ok = True
                _smed = None
                _srms = None
                _sp90 = None
                _squad_rms = None
                _squality = "unverified"
                _sreasons: List[str] = []
                try:
                    sci_al_data, _ = read_fits(sci_al)
                    ref_al_data, _ = read_fits(ref_al)
                    _swarp_align = compute_alignment_rms(
                        sci_al_data, ref_al_data, fwhm_pix,
                        input_yaml=self.input_yaml,
                        return_per_quadrant=True,
                    )
                    if _swarp_align is not None and len(_swarp_align) == 4:
                        _smed, _srms, _sp90, _squad_rms = _swarp_align
                    else:
                        _smed, _srms, _sp90 = _swarp_align[:3] if _swarp_align else (None, None, None)
                        _squad_rms = None
                    # Uniform quality gate (backup to post_swarp_verify in
                    # run_IDC): same thresholds and pass/warn/fail policy for
                    # every backend.
                    _squality, _sreasons = _classify_alignment_quality(
                        _smed, _srms, _sp90,
                        _squad_rms.get("coverage_ok") if _squad_rms else None,
                        *_gates,
                    )
                    if _squality == "fail":
                        logger.warning(
                            "SCAMP+SWarp alignment rejected: %s. "
                            "Falling back to next alignment method.",
                            "; ".join(_sreasons),
                        )
                        _swarp_ok = False
                    elif _squality == "unverified":
                        logger.warning(
                            "SCAMP+SWarp alignment could not be independently "
                            "verified (insufficient matched sources)."
                        )
                    if _squad_rms is not None and _squad_rms.get("coverage_ok") is False:
                        logger.warning(
                            "SCAMP+SWarp verification: matched sources cover only "
                            "%.0f%%x%.0f%% of the field - rotation/scale verified "
                            "on a limited region only.",
                            100.0 * _squad_rms.get("span_x", 0.0),
                            100.0 * _squad_rms.get("span_y", 0.0),
                        )
                except Exception:
                    logger.debug("swarp: quality measurement failed", exc_info=True)
                if not _swarp_ok:
                    return _fail("swarp", "; ".join(_sreasons))
                if not _alignment_gate_accept(_squality, self.input_yaml):
                    logger.warning(
                        "SCAMP+SWarp alignment rejected: quality=%s "
                        "(strict verification gate enabled).",
                        _squality,
                    )
                    return _fail("swarp", f"quality={_squality}")
                method_used = res.get("alignment_method", "scamp_swarp")
                logger.log(STATUS, "Alignment succeeded (method: %s).", method_used)
                # Store alignment RMS in the aligned reference header for
                # downstream SFFT kernel sizing and photometry provenance.
                try:
                    _alig_kw = {}
                    if _smed is not None and np.isfinite(_smed):
                        _alig_kw["ALIGMED"] = (float(_smed), "Alignment median offset (px)")
                    if _srms is not None and np.isfinite(_srms):
                        _alig_kw["ALIGRMS"] = (float(_srms), "Alignment RMS (px)")
                    if _sp90 is not None and np.isfinite(_sp90):
                        _alig_kw["ALIGP90"] = (float(_sp90), "Alignment P90 offset (px)")
                    _alig_kw["ALIGMETH"] = (str(method_used), "Alignment method used")
                    if _squad_rms is not None:
                        _sqmax = _squad_rms.get("max_rms", np.nan)
                        if np.isfinite(_sqmax) and _sqmax > 0:
                            _alig_kw["ALIGQMAX"] = (float(_sqmax), "Max per-quadrant alignment RMS (px)")
                            _alig_kw["ALIGQREG"] = (
                                str(_squad_rms.get("max_quadrant", "none")),
                                "Worst alignment quadrant",
                            )
                        _alig_kw["ALIGCOV"] = (
                            int(bool(_squad_rms.get("coverage_ok", True))),
                            "Matched-source coverage adequate (1=yes, 0=clustered)",
                        )
                    if _alig_kw and os.path.isfile(ref_al):
                        with fits.open(ref_al, mode="update", memmap=False) as _hdl:
                            for _k, _v in _alig_kw.items():
                                _hdl[0].header[_k] = _v
                            _hdl.flush()
                except Exception:
                    logger.debug("swarp: failed to write quality header", exc_info=True)
                # Update target coordinates to reflect new WCS after alignment
                self._update_target_coordinates_after_alignment(sci_al, method_used)
                return AlignmentResult(
                    sci_al, ref_al, method_used, _smed,
                    rms_px=_srms, p90_px=_sp90,
                    coverage_ok=(
                        _squad_rms.get("coverage_ok") if _squad_rms else None
                    ),
                    n_matches=(
                        _squad_rms.get("n_matched") if _squad_rms else None
                    ),
                    quality=_squality,
                    message="; ".join(_sreasons),
                )

            def _astroalign() -> AlignmentResult:
                if run_IDC is None:
                    logger.info("run_IDC not available; skipping AstroAlign.")
                    return _fail("astroalign", "run_IDC not available")
                logger.info("Attempting AstroAlign.")
                idc = run_IDC.ImageDistortionCorrector(input_yaml=self.input_yaml)
                res = idc.align_with_astroalign(scienceFpath, templateFpath)
                if not res or res.get("rejected") or not res.get("science_aligned"):
                    logger.info("AstroAlign did not produce aligned outputs.")
                    return _fail("astroalign", "no aligned outputs")
                sci_al = res["science_aligned"]
                ref_al = res["reference_aligned"]
                _aa_ok = True
                _amed = None
                _arms = None
                _ap90 = None
                _aa_quad = None
                _aquality = "unverified"
                _areasons: List[str] = []
                try:
                    sci_al_data, _ = read_fits(sci_al)
                    ref_al_data, _ = read_fits(ref_al)
                    _aa_metrics = compute_alignment_rms(
                        sci_al_data, ref_al_data, fwhm_pix,
                        input_yaml=self.input_yaml,
                        return_per_quadrant=True,
                    )
                    if _aa_metrics is not None and len(_aa_metrics) == 4:
                        _amed, _arms, _ap90, _aa_quad = _aa_metrics
                    elif _aa_metrics is not None:
                        _amed, _arms, _ap90 = _aa_metrics[:3]
                    _aquality, _areasons = _classify_alignment_quality(
                        _amed, _arms, _ap90,
                        _aa_quad.get("coverage_ok") if _aa_quad else None,
                        *_gates,
                    )
                    if _aquality == "fail":
                        logger.warning(
                            "AstroAlign alignment rejected: %s. "
                            "Falling back to next alignment method.",
                            "; ".join(_areasons),
                        )
                        _aa_ok = False
                    elif _aquality == "unverified":
                        logger.warning(
                            "AstroAlign alignment could not be independently "
                            "verified (insufficient matched sources)."
                        )
                    if _aa_quad is not None and _aa_quad.get("coverage_ok") is False:
                        logger.warning(
                            "AstroAlign verification: matched sources cover only "
                            "%.0f%%x%.0f%% of the field - rotation/scale verified "
                            "on a limited region only.",
                            100.0 * _aa_quad.get("span_x", 0.0),
                            100.0 * _aa_quad.get("span_y", 0.0),
                        )
                except Exception:
                    logger.debug("astroalign: quality measurement failed", exc_info=True)
                if not _aa_ok:
                    return _fail("astroalign", "; ".join(_areasons))
                if not _alignment_gate_accept(_aquality, self.input_yaml):
                    logger.warning(
                        "AstroAlign alignment rejected: quality=%s "
                        "(strict verification gate enabled).",
                        _aquality,
                    )
                    return _fail("astroalign", f"quality={_aquality}")
                method_used = res.get("alignment_method", "astroalign")
                logger.log(STATUS, "Alignment succeeded (method: %s).", method_used)
                # Store alignment RMS in the aligned reference header for
                # downstream SFFT kernel sizing and photometry provenance.
                try:
                    _alig_kw = {}
                    if _amed is not None and np.isfinite(_amed):
                        _alig_kw["ALIGMED"] = (float(_amed), "Alignment median offset (px)")
                    if _arms is not None and np.isfinite(_arms):
                        _alig_kw["ALIGRMS"] = (float(_arms), "Alignment RMS (px)")
                    if _ap90 is not None and np.isfinite(_ap90):
                        _alig_kw["ALIGP90"] = (float(_ap90), "Alignment P90 offset (px)")
                    _alig_kw["ALIGMETH"] = (str(method_used), "Alignment method used")
                    if _aa_quad is not None:
                        _alig_kw["ALIGCOV"] = (
                            int(bool(_aa_quad.get("coverage_ok", True))),
                            "Matched-source coverage adequate (1=yes, 0=clustered)",
                        )
                    if _alig_kw and os.path.isfile(ref_al):
                        with fits.open(ref_al, mode="update", memmap=False) as _hdl:
                            for _k, _v in _alig_kw.items():
                                _hdl[0].header[_k] = _v
                            _hdl.flush()
                except Exception:
                    logger.debug("astroalign: failed to write quality header", exc_info=True)
                # Update target coordinates to reflect new WCS after alignment
                self._update_target_coordinates_after_alignment(sci_al, method_used)
                return AlignmentResult(
                    sci_al, ref_al, method_used, _amed,
                    rms_px=_arms, p90_px=_ap90,
                    coverage_ok=(
                        _aa_quad.get("coverage_ok") if _aa_quad else None
                    ),
                    n_matches=(
                        _aa_quad.get("n_matched") if _aa_quad else None
                    ),
                    quality=_aquality,
                    message="; ".join(_areasons),
                )

            def _reproject() -> AlignmentResult:
                result = _reproject_template(
                    science_image=scienceImage,
                    science_header=scienceHeader,
                    template_image=templateImage,
                    template_header=templateHeader,
                    output_path=new_templateFpath,
                    cfg=reproject_cfg,
                    fwhm_pixels=fwhm_pix,
                    input_yaml=self.input_yaml,
                )
                if result.template_path is None:
                    return result._replace(
                        method_used="reproject",
                        quality="fail",
                        message=result.message or "reproject produced no output",
                    )

                _quality, _reasons = _classify_alignment_quality(
                    result.median_offset_px, result.rms_px, result.p90_px,
                    result.coverage_ok, *_gates,
                )
                if _quality == "unverified":
                    logger.warning(
                        "Reproject alignment quality could not be verified "
                        "(insufficient sources); accepting as best available fallback."
                    )
                if not _alignment_gate_accept(_quality, self.input_yaml):
                    logger.warning(
                        "Reproject alignment rejected: %s. "
                        "Falling back to next alignment method.",
                        "; ".join(_reasons) if _reasons else _quality,
                    )
                    return result._replace(
                        template_path=None, quality=_quality,
                        message="; ".join(_reasons) or _quality,
                    )

                # Only the template is resampled; the science WCS is
                # unchanged, so target coordinates need no update.
                method_used = result.method_used or "reproject"
                logger.log(STATUS, "Alignment succeeded (method: %s).", method_used)
                return result._replace(
                    science_path=scienceFpath,
                    quality=_quality,
                    message="; ".join(_reasons),
                )

            def _check_sub_tile_feasibility(det, shape, sub_tile, min_per_tile=4):
                """Check if spalipy's sub-tile splitting will have enough sources.

                spalipy splits the *source* image (template) into sub_tile x
                sub_tile grid and requires >=4 sources per sub-tile for quad
                construction.  When the template only partially overlaps the
                science image, matched sources cluster in the overlap region
                and some sub-tiles may be empty.

                This simulates the splitting and reduces sub_tile if needed.
                """
                if sub_tile <= 1:
                    return sub_tile
                try:
                    coo = np.column_stack([
                        np.asarray(det["x"], float),
                        np.asarray(det["y"], float),
                    ])
                    width = shape[1]
                    height = shape[0]
                    sub_w = width / sub_tile
                    sub_h = height / sub_tile
                    counts = []
                    for i in range(sub_tile):
                        cx = width * (2 * i + 1) / (sub_tile * 2)
                        for j in range(sub_tile):
                            cy = height * (2 * j + 1) / (sub_tile * 2)
                            mask = (
                                (np.abs(cx - coo[:, 0]) <= sub_w / 2) &
                                (np.abs(cy - coo[:, 1]) <= sub_h / 2)
                            )
                            counts.append(int(mask.sum()))
                    _min_count = min(counts)
                    if _min_count >= min_per_tile:
                        logger.info(
                            "spalipy: sub_tile=%d feasible (per-tile sources: %s).",
                            sub_tile, counts,
                        )
                        return sub_tile
                    else:
                        logger.info(
                            "spalipy: sub_tile=%d infeasible (per-tile sources: %s, "
                            "min=%d < %d needed); reducing to sub_tile=1.",
                            sub_tile, counts, _min_count, min_per_tile,
                        )
                        return 1
                except Exception:
                    return sub_tile

            def _spalipy() -> AlignmentResult:
                """Align template to science using spalipy (spline-warp registration).

                spalipy uses quad-based asterism matching for an initial affine
                transform, then fits 2D spline surfaces to the residual field
                to correct non-homogeneous/optical distortion.  This handles
                spatially-varying distortion that a single affine or polynomial
                cannot - the failure mode that per-quadrant verification detects.

                Source detection uses SExtractor for consistency with the rest
                of the pipeline.  Only the template is resampled; the science
                image is unchanged.
                """
                if not _HAS_SPALIPY:
                    logger.info("spalipy not installed; skipping spalipy alignment.")
                    return _fail("spalipy", "spalipy not installed")
                try:
                    from spalipy import Spalipy

                    # --- Source detection with SExtractor ---
                    # SExtractor's deblending is reliable enough that a single
                    # low-threshold run finds plenty of sources - no need for
                    # the adaptive multi-threshold scanning SEP required.
                    _sp_cfg = (
                        (self.input_yaml.get("template_subtraction") or {})
                        if isinstance(self.input_yaml, dict)
                        else {}
                    )
                    _sp_detect_thresh = float(_sp_cfg.get("spalipy_detect_thresh", 2.0) or 2.0)
                    _sp_min_pool = int(_sp_cfg.get("spalipy_detect_min_sources", 20) or 20)
                    _sp_match_radius_cfg = _sp_cfg.get("spalipy_match_radius_arcsec", None)
                    if _sp_match_radius_cfg is not None:
                        _sp_match_radius = float(_sp_match_radius_cfg)
                    else:
                        # Adaptive: 3x FWHM in arcsec, floored at 10"
                        # (handles WCS offsets in poorly plate-solved images)
                        _sp_pix_scale = float(self.input_yaml.get("pixel_scale", 1.0) or 1.0)
                        _sp_match_radius = max(3.0 * fwhm_pix * _sp_pix_scale, 10.0)

                    def _detect_for_spalipy(data, min_sources=20):
                        """Detect sources via SExtractor, return spalipy-format Table.

                        A wide ellipticity cut (0.9) is used from the start so
                        extended sources count as alignment anchors - the
                        registration needs reliable centroids, not stellar
                        profiles.  When the pool is thinner than ``min_sources``
                        a deeper pass is run and merged (deduplicated) so it can
                        only add anchors.
                        """
                        xy, flux, fwhm = _detect_sextractor_sources(
                            data, input_yaml=self.input_yaml, fwhm_pix=fwhm_pix,
                            thresh=_sp_detect_thresh, ell_max=0.9,
                        )
                        _xy_len = len(xy) if xy is not None else 0
                        if _xy_len < min_sources and _sp_detect_thresh > 0.5:
                            _low_thresh = max(0.5, _sp_detect_thresh * 0.5)
                            logger.info(
                                "spalipy: only %d sources; retrying with "
                                "lower detection threshold (%.1f -> %.1f).",
                                _xy_len, _sp_detect_thresh, _low_thresh,
                            )
                            xy_low, flux_low, fwhm_low = _detect_sextractor_sources(
                                data, input_yaml=self.input_yaml, fwhm_pix=fwhm_pix,
                                thresh=_low_thresh, ell_max=0.9,
                            )
                            _n_low = len(xy_low) if xy_low is not None else 0
                            xy, flux, fwhm = _merge_detection_lists(
                                xy, flux, fwhm, xy_low, flux_low, fwhm_low,
                            )
                            _xy_len = len(xy) if xy is not None else 0
                            logger.info(
                                "spalipy: low-threshold pass found %d sources; "
                                "merged pool now %d.",
                                _n_low, _xy_len,
                            )
                        if xy is None or len(xy) < 4:
                            return None
                        return Table({
                            "x": xy[:, 0],
                            "y": xy[:, 1],
                            "flux": flux,
                            "fwhm": fwhm,
                            "flag": np.zeros(len(xy), dtype=int),
                        })

                    def _dedup_close_pairs(det, min_sep):
                        """Subset a detection table to its brightest-per-close-group
                        members (see ``_close_group_survivor_indices``)."""
                        if det is None or len(det) < 2:
                            return det
                        _coo = np.column_stack([
                            np.asarray(det["x"], float),
                            np.asarray(det["y"], float),
                        ])
                        return det[
                            _close_group_survivor_indices(
                                _coo, det["flux"], min_sep
                            )
                        ]

                    def _wcs_overlap_filter(det, det_header, tgt_header, tgt_shape):
                        """Keep detections whose sky positions land inside the
                        other image's pixel footprint (plus a margin).

                        The margin covers moderate WCS header offsets so real
                        overlap-region sources are not clipped by a slightly
                        stale pointing record.
                        """
                        from astropy.wcs import WCS as _WCSf
                        _margin = max(50.0, 0.1 * float(max(tgt_shape)))
                        _w_det = _WCSf(det_header)
                        _w_tgt = _WCSf(tgt_header)
                        _ra, _dec = _w_det.all_pix2world(det["x"], det["y"], 0)
                        _px, _py = _w_tgt.all_world2pix(_ra, _dec, 0)
                        _in = (
                            np.isfinite(_px) & np.isfinite(_py)
                            & (_px >= -_margin) & (_px < tgt_shape[1] + _margin)
                            & (_py >= -_margin) & (_py < tgt_shape[0] + _margin)
                        )
                        return det[_in]

                    sci_det = _detect_for_spalipy(scienceImage, _sp_min_pool)
                    tpl_det = _detect_for_spalipy(templateImage, _sp_min_pool)
                    if sci_det is None or tpl_det is None:
                        logger.info("spalipy: insufficient sources for alignment.")
                        return _fail("spalipy", "insufficient sources")
                    if len(sci_det) < 4 or len(tpl_det) < 4:
                        logger.info(
                            "spalipy: too few sources (sci=%d, tpl=%d; need >=4).",
                            len(sci_det), len(tpl_det),
                        )
                        return _fail("spalipy", "too few sources")

                    # --- RA/DEC source matching ---
                    # Detect sources in the science image, convert to RA/DEC
                    # using the science WCS, then match with template sources
                    # (also in RA/DEC) using a sky matching radius.  Only
                    # these matched pairs are passed to spalipy.  This avoids
                    # spalipy's quad matching finding wrong transforms (e.g.
                    # scale=0.777 when the true scale is 1.0) from false quad
                    # matches between unrelated sources in non-overlapping
                    # sky regions.
                    try:
                        import astropy.units as u
                        from astropy.coordinates import (
                            SkyCoord,
                            match_coordinates_sky,
                        )
                        from astropy.wcs import WCS as _WCS

                        _sci_wcs = _WCS(scienceHeader)
                        _tpl_wcs = _WCS(templateHeader)

                        _sci_ra, _sci_dec = _sci_wcs.all_pix2world(
                            sci_det["x"], sci_det["y"], 0,
                        )
                        _sci_sky = SkyCoord(_sci_ra * u.deg, _sci_dec * u.deg)

                        _tpl_ra, _tpl_dec = _tpl_wcs.all_pix2world(
                            tpl_det["x"], tpl_det["y"], 0,
                        )
                        _tpl_sky = SkyCoord(_tpl_ra * u.deg, _tpl_dec * u.deg)

                        # Nearest template source for each science source.
                        _match_radius = _sp_match_radius * u.arcsec
                        _idx_tpl, _sep2d, _ = match_coordinates_sky(
                            _sci_sky, _tpl_sky, nthneighbor=1,
                        )
                        _matched = _sep2d < _match_radius

                        # Require mutual nearest neighbours to reject one-to-many matches.
                        _idx_sci_rev, _sep2d_rev, _ = match_coordinates_sky(
                            _tpl_sky, _sci_sky, nthneighbor=1,
                        )
                        _mutual = _matched & (
                            _idx_sci_rev[_idx_tpl] == np.arange(len(sci_det))
                        )

                        _n_sci_before = len(sci_det)
                        _n_tpl_before = len(tpl_det)
                        _sci_det_all = sci_det
                        _tpl_det_all = tpl_det
                        _sci_matched = sci_det[_mutual]
                        _tpl_matched = tpl_det[_idx_tpl[_mutual]]

                        logger.info(
                            "spalipy: RA/DEC matching sci %d + tpl %d -> %d matched pairs "
                            "(radius=%.1f arcsec).",
                            _n_sci_before, _n_tpl_before, len(_sci_matched),
                            _match_radius.value,
                        )

                        if len(_sci_matched) < 4:
                            # Few RA/DEC matches can mean non-overlapping
                            # fields OR a WCS offset larger than the match
                            # radius.  When the footprints still overlap, let
                            # spalipy's internal quad matching try the
                            # detections inside the shared footprint: feeding
                            # the full lists would let the brightest-N quad
                            # anchors all sit outside the overlap, where they
                            # can never match.
                            if _wcs_footprints_overlap(
                                scienceHeader, scienceImage.shape,
                                templateHeader, templateImage.shape,
                            ):
                                try:
                                    _sci_ov = _wcs_overlap_filter(
                                        _sci_det_all, scienceHeader,
                                        templateHeader, templateImage.shape,
                                    )
                                    _tpl_ov = _wcs_overlap_filter(
                                        _tpl_det_all, templateHeader,
                                        scienceHeader, scienceImage.shape,
                                    )
                                except Exception:
                                    _sci_ov = _tpl_ov = None
                                if (
                                    _sci_ov is not None
                                    and len(_sci_ov) >= 4
                                    and len(_tpl_ov) >= 4
                                ):
                                    sci_det, tpl_det = _sci_ov, _tpl_ov
                                else:
                                    sci_det = _sci_det_all
                                    tpl_det = _tpl_det_all
                                logger.warning(
                                    "spalipy: only %d RA/DEC matched sources\n"
                                    "    but WCS footprints overlap - WCS\n"
                                    "    offset may exceed the match radius.\n"
                                    "    Falling back to spalipy internal quad\n"
                                    "    matching (sci=%d, tpl=%d\n"
                                    "    footprint-filtered detections).",
                                    len(_sci_matched),
                                    len(sci_det), len(tpl_det),
                                )
                            else:
                                logger.info(
                                    "spalipy: too few RA/DEC matched sources "
                                    "(%d; need >=4) and footprints do not overlap.",
                                    len(_sci_matched),
                                )
                                return _fail(
                                    "spalipy", "too few RA/DEC matches"
                                )
                        else:
                            sci_det = _sci_matched
                            tpl_det = _tpl_matched

                    except Exception as _match_err:
                        logger.warning(
                            "spalipy: RA/DEC matching failed (%s); "
                            "falling back to overlap filtering.",
                            _match_err,
                        )
                        # Fallback: WCS overlap filtering
                        try:
                            _n_sci_before = len(sci_det)
                            _n_tpl_before = len(tpl_det)
                            sci_det = _wcs_overlap_filter(
                                sci_det, scienceHeader,
                                templateHeader, templateImage.shape,
                            )
                            tpl_det = _wcs_overlap_filter(
                                tpl_det, templateHeader,
                                scienceHeader, scienceImage.shape,
                            )

                            if len(sci_det) < _n_sci_before or len(tpl_det) < _n_tpl_before:
                                logger.info(
                                    "spalipy: WCS overlap filter sci %d->%d, "
                                    "tpl %d->%d sources.",
                                    _n_sci_before, len(sci_det),
                                    _n_tpl_before, len(tpl_det),
                                )
                        except Exception as _wcs_filt_exc:
                            # WCS filtering is best-effort, but a broken
                            # transform should still be diagnosable.
                            logger.debug(
                                "spalipy: WCS overlap filter skipped: %s",
                                _wcs_filt_exc,
                            )

                        if len(sci_det) < 4 or len(tpl_det) < 4:
                            logger.info(
                                "spalipy: too few overlapping sources "
                                "(sci=%d, tpl=%d; need >=4).",
                                len(sci_det), len(tpl_det),
                            )
                            return _fail("spalipy", "too few overlapping sources")

                    logger.info(
                        "Attempting spalipy alignment (sci=%d sources, tpl=%d sources).",
                        len(sci_det), len(tpl_det),
                    )

                    # Spalipy(source, template_data=..., source_det=..., template_det=...)
                    # transforms source -> template grid.  We want template -> science,
                    # so science is the template and template is the source.

                    # --- Reflection detection: flip template if needed ---
                    # spalipy's similarity transform [[a,-b],[b,a]] has positive
                    # determinant and can only represent rotations, NOT reflections.
                    # If the template is reflected relative to the science (det(A)<0
                    # where A = inv(CD_s) @ CD_t), flip the template in x to
                    # convert the reflection into a rotation spalipy can handle.
                    _tpl_img = templateImage
                    _tpl_det = tpl_det
                    try:
                        from astropy.wcs import WCS as _WCS2
                        _sw = _WCS2(scienceHeader)
                        _tw = _WCS2(templateHeader)
                        _A = np.linalg.inv(_sw.pixel_scale_matrix) @ _tw.pixel_scale_matrix
                        if np.linalg.det(_A) < 0:
                            logger.info(
                                "spalipy: template is REFLECTED relative to science "
                                "(det(A)=%.3f). Flipping template in x.",
                                np.linalg.det(_A),
                            )
                            _tpl_img = np.fliplr(templateImage).copy()
                            _tpl_det = tpl_det.copy()
                            _tpl_det["x"] = (templateImage.shape[1] - 1) - tpl_det["x"]
                    except Exception as _refl_exc:
                        # Reflection check is best-effort, but a broken
                        # WCS read should still be diagnosable.
                        logger.debug(
                            "spalipy: reflection check skipped: %s", _refl_exc
                        )

                    # Replace NaNs with median - spalipy can't handle NaNs.
                    _tpl_nan = ~np.isfinite(_tpl_img)
                    _sci_nan = ~np.isfinite(scienceImage)
                    _tpl_fill = np.where(_tpl_nan, float(np.nanmedian(_tpl_img)), _tpl_img).astype(np.float32)
                    _sci_fill = np.where(_sci_nan, float(np.nanmedian(scienceImage)), scienceImage).astype(np.float32)

                    _med_fwhm = float(np.median(
                        np.concatenate([sci_det["fwhm"], _tpl_det["fwhm"]])
                    ))

                    # spalipy parameters: adaptive defaults with YAML overrides.
                    # See https://github.com/Lyalpha/spalipy for parameter docs.
                    _yaml_n_quad = _sp_cfg.get("spalipy_n_quad_det")
                    _yaml_min_match = _sp_cfg.get("spalipy_min_n_match")
                    _yaml_hash_dist = _sp_cfg.get("spalipy_max_quad_hash_dist")
                    _yaml_min_sep = _sp_cfg.get("spalipy_min_sep")
                    _yaml_interp = _sp_cfg.get("spalipy_interp_order")
                    _yaml_sub_tile = _sp_cfg.get("spalipy_sub_tile")
                    _yaml_spline = _sp_cfg.get("spalipy_spline_order")
                    _yaml_max_match_dist = _sp_cfg.get("spalipy_max_match_dist")
                    _yaml_min_quad_sep = _sp_cfg.get("spalipy_min_quad_sep")
                    _yaml_quad_edge_buffer = _sp_cfg.get("spalipy_quad_edge_buffer")
                    _yaml_max_quad_cand = _sp_cfg.get("spalipy_max_quad_cand")
                    _yaml_patience = _sp_cfg.get("spalipy_patience_quad_cand")

                    # --- Image geometry for parameter scaling ---
                    _tpl_h, _tpl_w = _tpl_fill.shape
                    _sci_h, _sci_w = scienceImage.shape
                    _min_img_dim = float(min(_tpl_w, _tpl_h, _sci_w, _sci_h))

                    # max_match_dist: maximum matching distance in template
                    # (science) pixel frame after affine transform.  spalipy
                    # also requires the 2nd-nearest match to be >2x this
                    # distance (anti-double-match).  Centroid uncertainty is
                    # ~FWHM/10, so scale with FWHM.  Too large -> false matches
                    # pass the 2nd-nearest test; too small -> real matches
                    # rejected.  Clamp to [2, 5] px.
                    if _yaml_max_match_dist is not None:
                        _max_match_dist = float(_yaml_max_match_dist)
                    else:
                        _max_match_dist = float(np.clip(0.5 * _med_fwhm, 2.0, 5.0))

                    # min_sep: minimum separation between detections used in
                    # alignment.  Removes crowded/blended sources.  Set to
                    # max(1 FWHM, 2*max_match_dist) per spalipy's default
                    # logic (min_sep defaults to 2*max_match_dist when None).
                    if _yaml_min_sep is not None:
                        _det_sep = float(_yaml_min_sep)
                    else:
                        _det_sep = max(float(_med_fwhm), 2.0 * _max_match_dist)

                    # Crowding dedup applied here so downstream sizing sees
                    # the real usable pool.  spalipy's own min_sep filter
                    # drops BOTH members of every close pair; keeping the
                    # brightest member of each close group recovers one
                    # anchor per group while preserving the anti-ambiguity
                    # intent (the survivor's 2nd-nearest neighbour in the
                    # other list is then >2*max_match_dist once both lists
                    # are deduplicated).  Applying it ourselves also makes
                    # the surviving count exact, so min_n_match cannot trip
                    # the constructor-time "detections < min_n_match" guard.
                    _n_sci_dedup = len(sci_det)
                    _n_tpl_dedup = len(_tpl_det)
                    sci_det = _dedup_close_pairs(sci_det, _det_sep)
                    _tpl_det = _dedup_close_pairs(_tpl_det, _det_sep)
                    _n_sources = min(len(sci_det), len(_tpl_det))
                    if len(sci_det) < _n_sci_dedup or len(_tpl_det) < _n_tpl_dedup:
                        logger.info(
                            "spalipy: crowding dedup (min_sep=%.1f px) "
                            "sci %d->%d, tpl %d->%d usable sources.",
                            _det_sep, _n_sci_dedup, len(sci_det),
                            _n_tpl_dedup, len(_tpl_det),
                        )
                    if _n_sources < 4:
                        logger.info(
                            "spalipy: too few non-crowded sources "
                            "(sci=%d, tpl=%d; need >=4).",
                            len(sci_det), len(_tpl_det),
                        )
                        return _fail("spalipy", "too few non-crowded sources")

                    # n_quad_det: number of detections used for quad construction.
                    # C(n_quad_det, 4) quads are made per sub-tile, so this has
                    # O(n^4) performance impact.  For sparse fields, use all
                    # sources (more quads = more chances to find a match).
                    # For moderate and dense fields, 25 gives C(25,4)=12650
                    # quads - a dense hash space so the correct quad is
                    # rarely missed.
                    if _yaml_n_quad is not None:
                        _n_quad = int(_yaml_n_quad)
                    else:
                        if _n_sources <= 25:
                            _n_quad = _n_sources
                        else:
                            _n_quad = 25

                    # min_n_match: minimum matched sources for alignment.
                    # Lower for sparse fields so spalipy doesn't reject valid
                    # matches.  Default floor is 6 (4 for the affine minimum
                    # plus margin), but when fewer sources than the floor are
                    # available, lower to the source count (floored at 4).
                    # This lets spalipy attempt alignment in very sparse fields
                    # instead of failing immediately.
                    if _yaml_min_match is not None:
                        _min_match = int(_yaml_min_match)
                    else:
                        _min_match = max(6, min(_n_sources // 3, 20))
                    if _min_match > _n_sources:
                        _min_match = max(4, _n_sources)
                        logger.info(
                            "spalipy: lowering min_n_match to %d (only %d "
                            "sources available).",
                            _min_match, _n_sources,
                        )

                    # max_quad_hash_dist: tolerance for quad hash matching.
                    # Scaled by FWHM to account for centroid uncertainty.
                    if _yaml_hash_dist is not None:
                        _hash_dist = float(_yaml_hash_dist)
                    else:
                        _hash_dist = max(0.005, 2.0 * _med_fwhm / 50.0)

                    # min_quad_sep: minimum distance between detections in a
                    # quad.  spalipy defaults to 50 px, which is too large for
                    # small templates (e.g. 486x501).  Quads should span a
                    # meaningful fraction of the image for astrometric
                    # constraining power.  Scale with min image dimension,
                    # clamped to [10, 50] px.
                    if _yaml_min_quad_sep is not None:
                        _min_quad_sep = float(_yaml_min_quad_sep)
                    else:
                        _min_quad_sep = float(np.clip(_min_img_dim / 4.0, 10.0, 50.0))

                    # quad_edge_buffer: exclude detections within this many
                    # pixels of the template edge from quad construction.
                    # When the template only partially overlaps the science
                    # image, sources near the template edge may be truncated
                    # or have poor centroids.  Set to ~1 FWHM when partial
                    # overlap is detected, 0 otherwise.
                    if _yaml_quad_edge_buffer is not None:
                        _quad_edge_buffer = int(_yaml_quad_edge_buffer)
                    else:
                        _quad_edge_buffer = 0
                        try:
                            _tpl_wcs_area = _tpl_w * _tpl_h
                            # Partial-overlap check via WCS footprints.
                            from astropy.wcs import WCS as _WCS3
                            _sw3 = _WCS3(scienceHeader)
                            _tw3 = _WCS3(templateHeader)
                            _sci_corners = _sw3.calc_footprint()
                            _tpl_corners = _tw3.calc_footprint()
                            _tpl_in_sci = _sw3.all_world2pix(
                                _tpl_corners[:, 0], _tpl_corners[:, 1], 0
                            )
                            _tpl_x_min = float(np.min(_tpl_in_sci[0]))
                            _tpl_x_max = float(np.max(_tpl_in_sci[0]))
                            _tpl_y_min = float(np.min(_tpl_in_sci[1]))
                            _tpl_y_max = float(np.max(_tpl_in_sci[1]))
                            _covers_x = (_tpl_x_min <= 0) and (_tpl_x_max >= _sci_w)
                            _covers_y = (_tpl_y_min <= 0) and (_tpl_y_max >= _sci_h)
                            if not (_covers_x and _covers_y):
                                _quad_edge_buffer = max(1, int(_med_fwhm))
                                logger.info(
                                    "spalipy: partial overlap detected "
                                    "(tpl covers sci x:[%.0f,%.0f]/%d, y:[%.0f,%.0f]/%d); "
                                    "setting quad_edge_buffer=%d px.",
                                    _tpl_x_min, _tpl_x_max, _sci_w,
                                    _tpl_y_min, _tpl_y_max, _sci_h,
                                    _quad_edge_buffer,
                                )
                        except Exception as _qeb_exc:
                            logger.debug(
                                "spalipy: quad_edge_buffer skipped: %s",
                                _qeb_exc,
                            )

                    # max_quad_cand: maximum quad candidates to try for the
                    # affine transform.  Each candidate is one cross-match
                    # plus a least-squares refit - cheap relative to quad
                    # construction - so a generous budget is affordable.
                    # patience_quad_cand: stop after this many consecutive
                    # non-improving candidates; spalipy's default of 2 aborts
                    # the search before ambiguous hash-order lists are
                    # explored, so we ask for more tries.
                    if _yaml_max_quad_cand is not None:
                        _max_quad_cand = int(_yaml_max_quad_cand)
                    else:
                        _max_quad_cand = 20
                    _patience = (
                        int(_yaml_patience) if _yaml_patience is not None else 4
                    )

                    # interp_order: spline interpolation order for resampling.
                    # Lower order for undersampled images (less smooth interpolation).
                    if _yaml_interp is not None:
                        _interp_order = int(_yaml_interp)
                    else:
                        _interp_order = 2 if _med_fwhm < 3.0 else 3

                    # sub_tile: number of sub-tiles for affine fitting.
                    # Higher values model spatially-varying distortion better
                    # but need enough sources per tile.  spalipy splits the
                    # *source* image (template) into sub_tile x sub_tile grid
                    # and requires >=4 sources per sub-tile for quad construction.
                    # When the template only partially overlaps the science
                    # image, matched sources cluster in the overlap region and
                    # some sub-tiles may be empty.
                    #
                    # We simulate spalipy's sub-tile splitting on the template
                    # image to check that each sub-tile has enough sources.
                    if _yaml_sub_tile is not None:
                        _sub_tile = int(_yaml_sub_tile)
                    else:
                        _sub_tile = 2 if _n_sources >= 200 else 1
                        if _sub_tile > 1:
                            _sub_tile = _check_sub_tile_feasibility(
                                _tpl_det, _tpl_fill.shape, _sub_tile,
                            )

                    # Spline order: SmoothBivariateSpline requires at least
                    # (kx+1)*(ky+1) matched sources (not detected sources).
                    # After quad matching, typically only ~70-85% of detected
                    # sources are matched.  We use a conservative estimate:
                    # require 1.5x the minimum to account for unmatched
                    # sources.  If the initial order fails at runtime, the
                    # retry loop below progressively reduces it.
                    #
                    #   spline_order=3 -> needs 16 matched -> require 24 detected
                    #   spline_order=2 -> needs  9 matched -> require 14 detected
                    #   spline_order=1 -> needs  4 matched -> require  6 detected
                    #   spline_order=0 -> affine only, no minimum
                    _min_detected_for_order = {3: 24, 2: 14, 1: 6}
                    _spline_order = 0
                    if _yaml_spline is not None:
                        _spline_order = max(0, min(3, int(_yaml_spline)))
                    else:
                        for _o in (3, 2, 1):
                            if _n_sources >= _min_detected_for_order.get(_o, 0):
                                _spline_order = _o
                                break

                    logger.debug(
                        "spalipy: hash_dist=%.4f match_dist=%.2f min_quad_sep=%.1f "
                        "edge_buf=%d max_cand=%d min_match=%d n_quad=%d "
                        "sub_tile=%d spline_order=%d (FWHM=%.1f, n_sources=%d).",
                        _hash_dist, _max_match_dist, _min_quad_sep,
                        _quad_edge_buffer, _max_quad_cand, _min_match, _n_quad,
                        _sub_tile, _spline_order, _med_fwhm, _n_sources,
                    )

                    # Try align with progressively lower spline orders.
                    # spalipy only catches dfitpackError internally, not
                    # ValueError, so we catch it here and retry.
                    _spalipy_orders_to_try = [o for o in (3, 2, 1, 0) if o <= _spline_order]
                    sp = None
                    _try_sub_tile = _sub_tile
                    _aligned_ok = False
                    # spalipy tiles the template-role (science) detections for
                    # quad construction using the *source* image's shape.  When
                    # the source image is smaller than the science frame,
                    # science detections beyond the source's pixel bounds are
                    # silently dropped from the template quadlist - for a
                    # centred cutout this removes nearly all anchors and the
                    # quad-hash match fails.  Pad the source data/mask up to
                    # the science frame so the tiling sees the correct extent.
                    # The pad uses the median fill (not NaN - the NaN would
                    # propagate through the order>1 spline prefilter and blank
                    # the whole resample) and is masked, so warped pad pixels
                    # are re-masked to NaN downstream.
                    _sp_src_shape = _tpl_fill.shape
                    _tpl_fill, _tpl_nan = _pad_to_shape(
                        _tpl_fill, scienceImage.shape,
                        fill=float(np.median(_tpl_fill)), mask=_tpl_nan,
                    )
                    if _tpl_fill.shape != _sp_src_shape:
                        logger.info(
                            "spalipy: source image %s smaller than the science "
                            "frame %s - padded so template-detection tiling "
                            "uses the correct extent.",
                            _sp_src_shape, _tpl_fill.shape,
                        )
                    # spalipy logs per-quad progress through the root logger;
                    # keep only warnings on normal runs.
                    from functions import quiet_root_logger
                    import inspect as _inspect
                    _spalipy_sig = _inspect.signature(Spalipy.__init__).parameters
                    while not _aligned_ok:
                        for _try_order in _spalipy_orders_to_try:
                            _sp_kwargs = {
                                "source_mask": _tpl_nan if _tpl_nan.any() else None,
                                "template_data": _sci_fill,
                                "source_det": _tpl_det,
                                "template_det": sci_det,
                                "output_shape": scienceImage.shape,
                                "min_n_match": _min_match,
                                "n_quad_det": _n_quad,
                                "max_quad_hash_dist": _hash_dist,
                                "max_match_dist": _max_match_dist,
                                "min_quad_sep": _min_quad_sep,
                                "quad_edge_buffer": _quad_edge_buffer,
                                "max_quad_cand": _max_quad_cand,
                                "patience_quad_cand": _patience,
                                "min_sep": _det_sep,
                                "interp_order": _interp_order,
                                "sub_tile": _try_sub_tile,
                                "spline_order": _try_order,
                                "cval": np.nan,
                            }
                            # pyproject pins spalipy>=3.5; drop kwargs the
                            # installed version predates.
                            _sp_kwargs = {
                                k: v for k, v in _sp_kwargs.items()
                                if k in _spalipy_sig
                            }
                            with quiet_root_logger():
                                sp = Spalipy(_tpl_fill, **_sp_kwargs)
                            try:
                                with quiet_root_logger():
                                    sp.align()
                            except Exception as _spalipy_err:
                                _err_msg = str(_spalipy_err)
                                if "length of x, y and z" in _err_msg and _try_order > 0:
                                    logger.info(
                                        "spalipy: spline_order=%d failed (not enough "
                                        "matched sources); retrying with order=%d.",
                                        _try_order, _try_order - 1,
                                    )
                                    sp._aligned_data = None
                                    continue
                                elif "Not enough detections" in _err_msg and _try_sub_tile > 1:
                                    logger.info(
                                        "spalipy: sub_tile=%d failed (not enough "
                                        "detections in sub-tile); retrying with sub_tile=1.",
                                        _try_sub_tile,
                                    )
                                    _try_sub_tile = 1
                                    sp._aligned_data = None
                                    break  # break inner for, restart with sub_tile=1
                                else:
                                    logger.warning(
                                        "spalipy: align() raised %s: %s",
                                        type(_spalipy_err).__name__, _spalipy_err,
                                        exc_info=True,
                                    )
                                    sp._aligned_data = None
                                    _aligned_ok = True  # break outer while too
                                    break
                            # A converged spline is not necessarily a
                            # supported one: a cubic warp fit to too few
                            # matched sources interpolates between anchors
                            # it invented.  Require ~2x the (order+1)^2
                            # matched-source minimum before keeping the
                            # order; demote and retry otherwise.
                            _n_match_now = 0
                            try:
                                _sdm = getattr(sp, "_source_det_matched", None)
                                if _sdm is not None:
                                    for _e in _sdm:
                                        if (
                                            _e is not None
                                            and hasattr(_e, "__len__")
                                        ):
                                            _n_match_now += len(_e)
                            except Exception:
                                _n_match_now = 0
                            _min_match_for_order = {3: 32, 2: 18, 1: 8}
                            if (
                                _try_order > 0
                                and _n_match_now > 0
                                and _n_match_now < _min_match_for_order[_try_order]
                            ):
                                logger.info(
                                    "spalipy: %d matched sources cannot support "
                                    "spline_order=%d; retrying with order=%d.",
                                    _n_match_now, _try_order, _try_order - 1,
                                )
                                sp._aligned_data = None
                                continue
                            _aligned_ok = True
                            break  # success
                        else:
                            # All spline orders exhausted without success
                            _aligned_ok = True

                    if sp.aligned_data is None:
                        logger.info("spalipy did not produce aligned output.")
                        return _fail("spalipy", "no aligned output")

                    try:
                        _n_matched = 0
                        _sdm = getattr(sp, "_source_det_matched", None)
                        if _sdm:
                            for _e in _sdm:
                                if _e is not None and hasattr(_e, "__len__"):
                                    _n_matched += len(_e)
                        logger.info(
                            "spalipy: transform scale=%.3f rot=%.1f deg "
                            "(%d matched sources).",
                            sp.affine_transform.scale,
                            sp.affine_transform.rotation,
                            _n_matched,
                        )
                        # Post-spline residuals: the true accuracy of the
                        # combined affine + distortion correction.  Computed
                        # manually - sp._residuals() hits a broadcasting bug
                        # in this spalipy version for 2D coordinate input.
                        _sdm0 = sp._source_det_matched[0]
                        _tdm0 = sp._template_det_matched[0]
                        if _sdm0 is not None and _tdm0 is not None and len(_sdm0):
                            _src_coo = np.column_stack([
                                np.asarray(_sdm0["x"], float),
                                np.asarray(_sdm0["y"], float),
                            ])
                            _tpl_coo = np.column_stack([
                                np.asarray(_tdm0["x"], float),
                                np.asarray(_tdm0["y"], float),
                            ])
                            _aff = sp.affine_transform.apply_transform(_src_coo)
                            _dx = _tpl_coo[:, 0] - _aff[:, 0]
                            _dy = _tpl_coo[:, 1] - _aff[:, 1]
                            _sbsx = getattr(sp, "_sbs_x", None)
                            _sbsy = getattr(sp, "_sbs_y", None)
                            if (
                                _sbsx is not None and _sbsy is not None
                                and _sbsx[0] is not None and _sbsy[0] is not None
                            ):
                                _dx = _dx - _sbsx[0].ev(
                                    _tpl_coo[:, 0], _tpl_coo[:, 1]
                                )
                                _dy = _dy - _sbsy[0].ev(
                                    _tpl_coo[:, 0], _tpl_coo[:, 1]
                                )
                            logger.info(
                                "spalipy: post-fit residuals "
                                "dx=%.3f+/-%.3f px dy=%.3f+/-%.3f px.",
                                float(np.median(_dx)), float(np.std(_dx)),
                                float(np.median(_dy)), float(np.std(_dy)),
                            )
                    except Exception as _res_exc:
                        # Telemetry only - never fail the warp over it.
                        logger.debug(
                            "spalipy: residual stats skipped: %s", _res_exc
                        )

                    aligned_template = np.asarray(sp.aligned_data, dtype=np.float32)

                    # Re-mask pixels that were invalid in the source: cval=nan
                    # marks unmapped output pixels; sp.aligned_mask carries the
                    # warped source_mask.
                    _aligned_nan = ~np.isfinite(aligned_template)
                    try:
                        _warped_mask = getattr(sp, "aligned_mask", None)
                        if _warped_mask is not None:
                            _aligned_nan = _aligned_nan | np.asarray(
                                _warped_mask
                            ).astype(bool)
                    except Exception as _mask_exc:
                        logger.debug(
                            "spalipy: warped mask merge skipped: %s",
                            _mask_exc,
                        )
                    if _aligned_nan.any():
                        aligned_template = np.where(_aligned_nan, np.nan, aligned_template)

                    # Sanity gates on the warped product: a degenerate
                    # spalipy result (wrong shape, mostly empty, or constant)
                    # must not reach subtraction.
                    if aligned_template.shape != scienceImage.shape:
                        logger.warning(
                            "spalipy output shape %s != science shape %s; "
                            "rejecting.",
                            aligned_template.shape, scienceImage.shape,
                        )
                        return _fail("spalipy", "output shape mismatch")
                    _finite_frac = float(np.mean(np.isfinite(aligned_template)))
                    _min_ff, _attainable_ff = _finite_coverage_floor(
                        _ts_cfg.get("alignment_min_finite_fraction", 0.25),
                        float(np.count_nonzero(~_tpl_nan)),
                        aligned_template.size,
                    )
                    _min_px = int(
                        _ts_cfg.get("alignment_min_valid_pixels", 1000)
                    )
                    _valid_px = aligned_template[np.isfinite(aligned_template)]
                    if _finite_frac < _min_ff or _valid_px.size < _min_px:
                        logger.warning(
                            "spalipy output nearly empty (finite=%.1f%% "
                            "vs attainable %.1f%%, n=%d < %d); rejecting.",
                            100.0 * _finite_frac, 100.0 * _attainable_ff,
                            _valid_px.size, _min_px,
                        )
                        return _fail("spalipy", "insufficient finite coverage")
                    if float(np.nanstd(_valid_px)) <= 0:
                        logger.warning(
                            "spalipy output is constant; rejecting."
                        )
                        return _fail("spalipy", "degenerate constant output")

                    # No post-alignment sub-pixel shift correction: an
                    # empirical shift applied after spalipy's spline-warp can
                    # introduce subtraction dipoles (same reason the SCAMP+SWarp
                    # shift correction was removed).  The quality gate below
                    # measures the actual spalipy output.

                    # Alignment quality measurement.
                    # Use the pre-matched science sources (sci_det) instead of
                    # re-detecting in the science image.  sci_det was already
                    # RA/DEC matched to the reference, so it contains only real
                    # sources -- no ghost/stacking artifacts that would create
                    # false matches and inflate the RMS.
                    #
                    # spalipy is now held to the same pass/warn/fail gates as
                    # every other backend: a result with metrics beyond the
                    # configured limits is rejected so the cascade can fall
                    # back rather than committing to a poor alignment.
                    _med_off = None
                    _rms_off = None
                    _p90_off = None
                    _quad_rms = None
                    _sp_quality = "unverified"
                    _sp_reasons: List[str] = []
                    try:
                        _sci_xy_for_rms = np.column_stack([
                            np.asarray(sci_det["x"], float),
                            np.asarray(sci_det["y"], float),
                        ])
                        _align_result = compute_alignment_rms(
                            scienceImage, aligned_template, fwhm_pix,
                            input_yaml=self.input_yaml,
                            sci_xy_override=_sci_xy_for_rms,
                            return_per_quadrant=True,
                        )
                        if _align_result is not None and len(_align_result) == 4:
                            _med_off, _rms_off, _p90_off, _quad_rms = _align_result
                        else:
                            _med_off, _rms_off, _p90_off = _align_result[:3] if _align_result else (None, None, None)
                            _quad_rms = None
                        logger.info(
                            "spalipy: alignment RMS median=%.3f px rms=%.3f px.",
                            _med_off, _rms_off,
                        )
                        _sp_quality, _sp_reasons = _classify_alignment_quality(
                            _med_off, _rms_off, _p90_off,
                            _quad_rms.get("coverage_ok") if _quad_rms else None,
                            *_gates,
                        )
                    except Exception:
                        logger.debug("spalipy: quality measurement failed", exc_info=True)

                    if not _alignment_gate_accept(_sp_quality, self.input_yaml):
                        logger.warning(
                            "spalipy alignment rejected: %s. "
                            "Falling back to next alignment method.",
                            "; ".join(_sp_reasons) if _sp_reasons else _sp_quality,
                        )
                        return _fail(
                            "spalipy",
                            "; ".join(_sp_reasons) or _sp_quality,
                        )
                    if _sp_quality != "pass":
                        logger.warning(
                            "spalipy alignment quality=%s%s; metrics recorded "
                            "in header.",
                            _sp_quality,
                            (" (%s)" % "; ".join(_sp_reasons)) if _sp_reasons else "",
                        )

                    # Write aligned template under the science WCS.
                    hdr = templateHeader.copy()
                    hdr = remove_wcs_from_header(hdr)
                    from functions import copy_wcs_from_header
                    copy_wcs_from_header(scienceHeader, hdr)
                    hdr["NAXIS1"] = aligned_template.shape[1]
                    hdr["NAXIS2"] = aligned_template.shape[0]
                    # Store alignment quality in header for downstream SFFT
                    # kernel sizing and photometry provenance.
                    try:
                        if _med_off is not None and np.isfinite(_med_off):
                            hdr["ALIGMED"] = (float(_med_off), "Alignment median offset (px)")
                        if _rms_off is not None and np.isfinite(_rms_off):
                            hdr["ALIGRMS"] = (float(_rms_off), "Alignment RMS (px)")
                        if _p90_off is not None and np.isfinite(_p90_off):
                            hdr["ALIGP90"] = (float(_p90_off), "Alignment P90 offset (px)")
                        hdr["ALIGMETH"] = ("spalipy", "Alignment method used")
                        # Per-quadrant RMS: store the maximum quadrant RMS
                        # so SFFT can boost the kernel in regions with
                        # spatially-varying alignment errors.
                        if _quad_rms is not None:
                            _qmax = _quad_rms.get("max_rms", np.nan)
                            if np.isfinite(_qmax) and _qmax > 0:
                                hdr["ALIGQMAX"] = (float(_qmax), "Max per-quadrant alignment RMS (px)")
                                hdr["ALIGQREG"] = (
                                    str(_quad_rms.get("max_quadrant", "none")),
                                    "Worst alignment quadrant",
                                )
                            hdr["ALIGCOV"] = (
                                int(bool(_quad_rms.get("coverage_ok", True))),
                                "Matched-source coverage adequate (1=yes, 0=clustered)",
                            )
                    except Exception:
                        logger.debug("spalipy: failed to write quality header", exc_info=True)
                    _atomic_fits_write(new_templateFpath, aligned_template, hdr)
                    try:
                        from functions import refresh_sibling_weight_map

                        refresh_sibling_weight_map(new_templateFpath)
                    except Exception:
                        pass

                    # Diagnostic plot: matched sources side-by-side
                    # NOTE: when the template was flipped for reflection
                    # correction, _tpl_matched_xy and _tpl_det are in the
                    # FLIPPED frame.  We pass the flipped image and flipped
                    # all-detections so the plot circles align correctly.
                    try:
                        _sci_matched_xy = None
                        _tpl_matched_xy = None
                        if (
                            hasattr(sp, "_source_det_matched")
                            and sp._source_det_matched
                            and sp._source_det_matched[0] is not None
                            and hasattr(sp, "_template_det_matched")
                            and sp._template_det_matched
                            and sp._template_det_matched[0] is not None
                        ):
                            _src_m = sp._source_det_matched[0]
                            _tpl_m = sp._template_det_matched[0]
                            _tpl_matched_xy = np.column_stack([
                                np.asarray(_src_m["x"], float),
                                np.asarray(_src_m["y"], float),
                            ])
                            _sci_matched_xy = np.column_stack([
                                np.asarray(_tpl_m["x"], float),
                                np.asarray(_tpl_m["y"], float),
                            ])
                        if _sci_matched_xy is not None and len(_sci_matched_xy) > 0:
                            from plot import Plot as _Plot
                            _plot_inst = _Plot(input_yaml={
                                "fpath": scienceFpath,
                                "fwhm": float(np.median(
                                    np.concatenate([sci_det["fwhm"], _tpl_det["fwhm"]]
                                ))),
                                "plot_format": (self.input_yaml or {}).get("plot_format", "png"),
                            })
                            _plot_inst.plot_match_sources(
                                sci_image=scienceImage,
                                tpl_image=_tpl_img,
                                sci_matched_xy=_sci_matched_xy,
                                tpl_matched_xy=_tpl_matched_xy,
                                sci_all_xy=np.column_stack([
                                    np.asarray(sci_det["x"], float),
                                    np.asarray(sci_det["y"], float),
                                ]),
                                tpl_all_xy=np.column_stack([
                                    np.asarray(_tpl_det["x"], float),
                                    np.asarray(_tpl_det["y"], float),
                                ]),
                                method_label="spalipy",
                                sci_fwhm=float(np.median(sci_det["fwhm"])),
                                tpl_fwhm=float(np.median(_tpl_det["fwhm"])),
                            )
                    except Exception as _plot_err:
                        logger.debug("Match sources plot skipped: %s", _plot_err)

                    method_used = "spalipy"
                    logger.log(STATUS, "Alignment succeeded (method: %s).", method_used)
                    # Science image unchanged - no target coordinate update needed
                    return AlignmentResult(
                        scienceFpath, new_templateFpath, method_used,
                        _med_off,
                        rms_px=_rms_off, p90_px=_p90_off,
                        coverage_ok=(
                            _quad_rms.get("coverage_ok") if _quad_rms else None
                        ),
                        n_matches=_n_matched or (
                            _quad_rms.get("n_matched") if _quad_rms else None
                        ),
                        finite_fraction=_finite_frac,
                        quality=_sp_quality,
                        message="; ".join(_sp_reasons),
                    )

                except Exception as _e:
                    log_warning_from_exception(logger, "spalipy alignment failed", _e)
                    return _fail("spalipy", str(_e))

            def _tweakwcs() -> AlignmentResult:
                """Align template to science using tweakwcs (STScI WCS tweaking).

                tweakwcs computes corrections to WCS objects to minimize mismatch
                between image source catalogs and reference catalogs.  It uses
                tangent-plane linear corrections with sigma-clipped fitting -
                the same approach as HST/JWST pipeline alignment.

                After tweaking the template WCS, reproject is used to resample
                the template onto the science pixel grid.
                """
                if not _HAS_TWEAKWCS:
                    logger.info("tweakwcs not installed; skipping tweakwcs alignment.")
                    return _fail("tweakwcs", "tweakwcs not installed")
                if not _REPROJECT_AVAILABLE:
                    logger.info("reproject not available; tweakwcs requires reproject for resampling.")
                    return _fail("tweakwcs", "reproject not available")
                try:
                    from tweakwcs.correctors import FITSWCSCorrector
                    from tweakwcs.imalign import align_wcs
                    from tweakwcs.matchutils import XYXYMatch

                    # --- Detect sources in both images using SExtractor ---
                    fwhm = min(max(float(fwhm_pix), 2.0), 8.0)

                    def _detect_for_tweakwcs(data):
                        xy, _, _ = _detect_sextractor_sources(
                            data, input_yaml=self.input_yaml,
                            fwhm_pix=fwhm, thresh=5.0, fwhm_min=1.5,
                        )
                        if xy is None or len(xy) < 4:
                            return None
                        return Table({"x": xy[:, 0], "y": xy[:, 1]})

                    sci_cat = _detect_for_tweakwcs(scienceImage)
                    tpl_cat = _detect_for_tweakwcs(templateImage)
                    if sci_cat is None or tpl_cat is None:
                        logger.info("tweakwcs: insufficient sources for alignment.")
                        return _fail("tweakwcs", "insufficient sources")

                    logger.info(
                        "Attempting tweakwcs alignment (sci=%d sources, tpl=%d sources).",
                        len(sci_cat), len(tpl_cat),
                    )

                    # Build the reference catalog in sky coordinates.
                    # _detect_sextractor_sources already returns 0-based coords.
                    sci_wcs = WCS(scienceHeader)
                    sci_ra, sci_dec = sci_wcs.all_pix2world(
                        np.asarray(sci_cat["x"]),
                        np.asarray(sci_cat["y"]),
                        0,
                    )
                    ref_cat = Table({
                        "RA": np.asarray(sci_ra, float),
                        "DEC": np.asarray(sci_dec, float),
                    })

                    tpl_wcs = WCS(templateHeader)
                    tpl_corrector = FITSWCSCorrector(
                        tpl_wcs,
                        {"wcsname": str(templateHeader.get("WCSNAME", "TPL"))},
                    )
                    # Attach the template source catalog (pixel coords)
                    tpl_corrector.meta["catalog"] = tpl_cat

                    # Create a reference corrector for the science WCS
                    # (defines the tangent plane for the reference catalog)
                    sci_corrector = FITSWCSCorrector(
                        sci_wcs,
                        {"wcsname": str(scienceHeader.get("WCSNAME", "SCI"))},
                    )

                    # Match and align: template is aligned to science reference
                    match = XYXYMatch(
                        searchrad=10.0,
                        separation=5.0,
                        tolerance=2.0,
                        use2dhist=True,
                    )
                    align_wcs(
                        [tpl_corrector],
                        refcat=ref_cat,
                        ref_tpwcs=sci_corrector,
                        match=match,
                        fitgeom="general",
                        nclip=3,
                        sigma=(3.0, "rmse"),
                    )

                    corrected_wcs = tpl_corrector.wcs

                    # reproject cannot propagate NaN; fill with 0.
                    tpl_data = np.where(
                        np.isfinite(templateImage), templateImage, 0.0
                    ).astype(np.float32)

                    aligned_tpl, footprint = reproject_exact(
                        (tpl_data, corrected_wcs),
                        sci_wcs,
                        shape_out=scienceImage.shape,
                    )
                    fp_mask = footprint.astype(bool)
                    if fp_mask.sum() == 0:
                        logger.info("tweakwcs: zero footprint coverage after reproject.")
                        return _fail("tweakwcs", "zero footprint coverage")
                    # Zero-filled pixels contribute 0 flux but full overlap
                    # area, so an output pixel partly over an invalid region
                    # is diluted by the valid fraction - and a pixel fully
                    # inside a hole reads as real ~zero data.  Reproject the
                    # validity map identically: correct the dilution where
                    # most input was real and mask where it was not.
                    try:
                        _vmap, _ = reproject_exact(
                            (
                                np.isfinite(templateImage).astype(np.float32),
                                corrected_wcs,
                            ),
                            sci_wcs,
                            shape_out=scienceImage.shape,
                        )
                        _vf = np.asarray(_vmap, dtype=np.float32)
                        _ok = fp_mask & np.isfinite(_vf)
                        _majority = _ok & (_vf > 0.5)
                        aligned_tpl[_majority] = (
                            aligned_tpl[_majority] / _vf[_majority]
                        )
                        fp_mask &= _majority
                    except Exception as _ve:
                        logger.debug(
                            "tweakwcs valid-fraction correction skipped: %s",
                            _ve,
                        )
                    aligned_tpl[~fp_mask] = np.nan

                    # Compute alignment quality BEFORE writing so the metrics
                    # can be stored in the output header for downstream SFFT
                    # kernel sizing and photometry provenance.
                    to_write = np.asarray(aligned_tpl, dtype=np.float32)
                    _tmed = None
                    _trms = None
                    _tp90 = None
                    _tquad = None
                    _tweak_ok = True
                    _tquality = "unverified"
                    _treasons: List[str] = []
                    try:
                        _tmetrics = compute_alignment_rms(
                            scienceImage, to_write, fwhm_pix,
                            input_yaml=self.input_yaml,
                            sci_xy_override=_science_xy(),
                            return_per_quadrant=True,
                        )
                        if _tmetrics is not None and len(_tmetrics) == 4:
                            _tmed, _trms, _tp90, _tquad = _tmetrics
                        elif _tmetrics is not None:
                            _tmed, _trms, _tp90 = _tmetrics[:3]
                        _tquality, _treasons = _classify_alignment_quality(
                            _tmed, _trms, _tp90,
                            _tquad.get("coverage_ok") if _tquad else None,
                            *_gates,
                        )
                        if _tquality == "fail":
                            logger.warning(
                                "tweakwcs alignment rejected: %s. "
                                "Falling back to next alignment method.",
                                "; ".join(_treasons),
                            )
                            _tweak_ok = False
                        # tweakwcs fits a corrected WCS to the matched sources;
                        # a solution verified only on a clustered subset cannot
                        # be trusted over the full field, so reject it and let
                        # SCAMP+SWarp (which has its own residual check) try.
                        if _tquad is not None and _tquad.get("coverage_ok") is False:
                            logger.warning(
                                "tweakwcs alignment rejected: %d matched\n"
                                "    sources cover only %.0f%%x%.0f%% of the\n"
                                "    field - the WCS correction is\n"
                                "    underconstrained outside the cluster.\n"
                                "    Falling back to next alignment method.",
                                _tquad.get("n_matched", 0),
                                100.0 * _tquad.get("span_x", 0.0),
                                100.0 * _tquad.get("span_y", 0.0),
                            )
                            _tweak_ok = False
                    except Exception:
                        logger.debug("tweakwcs: quality measurement failed", exc_info=True)

                    if not _tweak_ok:
                        return _fail(
                            "tweakwcs",
                            "; ".join(_treasons) or "coverage/quality gate",
                        )
                    if not _alignment_gate_accept(_tquality, self.input_yaml):
                        logger.warning(
                            "tweakwcs alignment rejected: quality=%s "
                            "(strict verification gate enabled).",
                            _tquality,
                        )
                        return _fail("tweakwcs", f"quality={_tquality}")

                    # Write aligned template with science WCS + quality keywords
                    hdr = templateHeader.copy()
                    hdr = remove_wcs_from_header(hdr)
                    from functions import copy_wcs_from_header
                    copy_wcs_from_header(scienceHeader, hdr)
                    hdr["NAXIS1"] = to_write.shape[1]
                    hdr["NAXIS2"] = to_write.shape[0]
                    try:
                        if _tmed is not None and np.isfinite(_tmed):
                            hdr["ALIGMED"] = (float(_tmed), "Alignment median offset (px)")
                        if _trms is not None and np.isfinite(_trms):
                            hdr["ALIGRMS"] = (float(_trms), "Alignment RMS (px)")
                        if _tp90 is not None and np.isfinite(_tp90):
                            hdr["ALIGP90"] = (float(_tp90), "Alignment P90 offset (px)")
                        hdr["ALIGMETH"] = ("tweakwcs", "Alignment method used")
                    except Exception:
                        pass
                    _atomic_fits_write(new_templateFpath, to_write, hdr)
                    try:
                        from functions import refresh_sibling_weight_map

                        refresh_sibling_weight_map(new_templateFpath)
                    except Exception:
                        pass

                    method_used = "tweakwcs"
                    logger.log(STATUS, "Alignment succeeded (method: %s).", method_used)
                    return AlignmentResult(
                        scienceFpath, new_templateFpath, method_used, _tmed,
                        rms_px=_trms, p90_px=_tp90,
                        coverage_ok=(
                            _tquad.get("coverage_ok") if _tquad else None
                        ),
                        n_matches=(
                            _tquad.get("n_matched") if _tquad else None
                        ),
                        finite_fraction=float(
                            np.count_nonzero(fp_mask) / fp_mask.size
                        ),
                        quality=_tquality,
                        message="; ".join(_treasons),
                    )

                except Exception as _e:
                    log_warning_from_exception(logger, "tweakwcs alignment failed", _e)
                    return _fail("tweakwcs", str(_e))

            def _chi2_shift() -> AlignmentResult:
                """Align template to science using chi2_shift cross-correlation.

                Uses the ``image_registration`` package's ``chi2_shift`` which
                performs DFT-upsampling cross-correlation with chi-squared
                error estimation.  This works on **extended emission**
                (nebulae, galaxy-dominated fields) where there are no point
                sources to match - the gap that all source-based methods
                cannot fill.

                Only a translation is computed; the template is shifted and
                written with the science WCS.  A translation-only shift is
                only meaningful on a shared pixel grid, so mismatched shapes
                or grossly different WCS grids skip to a resampling fallback
                instead of being top-left cropped (which would silently
                assume pixel (0,0) is the same sky position in both images).
                """
                if not _HAS_IMGREG:
                    logger.info("image_registration not installed; skipping chi2_shift alignment.")
                    return _fail("chi2_shift", "image_registration not installed")
                try:
                    if scienceImage.shape != templateImage.shape:
                        logger.info(
                            "chi2_shift requires a common pixel grid "
                            "(science %s vs template %s); skipping.",
                            scienceImage.shape, templateImage.shape,
                        )
                        return _fail("chi2_shift", "shape mismatch")
                    # Loose common-grid check: the two WCSes must share
                    # scale/orientation/origin within the multi-pixel offset
                    # chi2_shift exists to correct.
                    _grid_tol = max(5.0, 2.0 * float(fwhm_pix))
                    if not verify_common_grid(
                        scienceHeader, templateHeader, scienceImage.shape,
                        tolerance_px=_grid_tol,
                    ):
                        logger.info(
                            "chi2_shift: template and science WCS grids differ "
                            "by more than %.1f px; translation cannot fix a "
                            "geometric mismatch - skipping.",
                            _grid_tol,
                        )
                        return _fail("chi2_shift", "WCS grids differ")

                    # Zero-fill invalid pixels for the correlation only; the
                    # validity mask is tracked separately and re-applied to
                    # the shifted product below so no-data regions (NaN chip
                    # gaps, SWarp zero padding) can never leak into the
                    # aligned image as finite values.
                    _valid_tpl = np.isfinite(templateImage) & (
                        np.abs(templateImage) >= 1.1e-20
                    )
                    sci_data = np.where(
                        np.isfinite(scienceImage), scienceImage, 0.0
                    ).astype(np.float64)
                    tpl_data = np.where(
                        _valid_tpl, templateImage, 0.0
                    ).astype(np.float64)

                    logger.info("Attempting chi2_shift cross-correlation alignment.")

                    xoff, yoff, exoff, eyoff = chi2_shift(
                        sci_data, tpl_data,
                        noise=None,
                        upsample_factor="auto",
                        return_error=True,
                        zeromean=True,
                    )

                    xoff = float(xoff)
                    yoff = float(yoff)
                    total_offset = float(np.sqrt(xoff**2 + yoff**2))

                    logger.info(
                        "chi2_shift: offset=(%.3f, %.3f) px, total=%.3f px, "
                        "error=(%.4f, %.4f) px",
                        xoff, yoff, total_offset, float(exoff or 0), float(eyoff or 0),
                    )

                    # Quality gate: reject spurious correlations.
                    # Offsets >= 100 px or non-finite are almost certainly
                    # failed cross-correlations (noise peak), not real shifts.
                    # The offset magnitude itself is NOT a valid rejection
                    # criterion: chi2_shift exists precisely to correct
                    # multi-pixel WCS offsets in source-free fields.  Gate on
                    # the formal error instead - a real correlation peak has
                    # a small error, a noise peak a large one.
                    if not np.isfinite(total_offset) or total_offset >= 100.0:
                        logger.warning(
                            "chi2_shift alignment rejected: offset=%.2f px is "
                            "non-finite or >= 100 px (spurious correlation). "
                            "Falling back to next alignment method.",
                            total_offset,
                        )
                        return _fail("chi2_shift", f"spurious offset {total_offset:.1f} px")
                    _err_off = float(np.hypot(
                        float(exoff) if exoff is not None else np.nan,
                        float(eyoff) if eyoff is not None else np.nan,
                    ))
                    _max_err = max(1.0, 0.5 * float(fwhm_pix))
                    if not np.isfinite(_err_off) or _err_off > _max_err:
                        logger.warning(
                            "chi2_shift alignment rejected: formal error=%.2f px "
                            "(> %.2f px) - likely a noise peak, not a real shift. "
                            "Falling back to next alignment method.",
                            _err_off, _max_err,
                        )
                        return _fail("chi2_shift", f"formal error {_err_off:.2f} px")

                    # Shift the template to match the science image, and shift
                    # its validity mask separately.  shiftnd takes
                    # (y_shift, x_shift) - we shift tpl by (-yoff, -xoff).
                    # The mask uses linear interpolation + a 0.5 threshold so
                    # the invalid boundary follows the sub-pixel shift; the
                    # vacated edge (which shiftnd fills by wraparound) gets
                    # cval=0 and is correctly marked invalid.
                    aligned_tpl = _imgreg_shift.shiftnd(
                        np.where(_valid_tpl, templateImage, 0.0),
                        (-yoff, -xoff),
                    )
                    aligned_tpl = np.asarray(aligned_tpl, dtype=np.float32)
                    _aligned_valid = _ndimage_shift(
                        _valid_tpl.astype(np.float32), (-yoff, -xoff),
                        order=1, mode="constant", cval=0.0,
                    )
                    aligned_tpl[_aligned_valid < 0.5] = np.nan

                    # Write with science WCS
                    hdr = templateHeader.copy()
                    hdr = remove_wcs_from_header(hdr)
                    from functions import copy_wcs_from_header
                    copy_wcs_from_header(scienceHeader, hdr)
                    hdr["NAXIS1"] = aligned_tpl.shape[1]
                    hdr["NAXIS2"] = aligned_tpl.shape[0]
                    # Store alignment quality in header for downstream SFFT
                    # kernel sizing and photometry provenance.  _cmed defaults
                    # to the applied shift; the post-shift source-match
                    # residual replaces it when measurable (it is None in the
                    # source-free fields chi2_shift exists for).
                    _cmed = total_offset
                    _crms = None
                    _cp90 = None
                    _cquad = None
                    _cmetrics = None
                    try:
                        _cmetrics = compute_alignment_rms(
                            scienceImage, aligned_tpl, fwhm_pix,
                            input_yaml=self.input_yaml,
                            sci_xy_override=_science_xy(),
                            return_per_quadrant=True,
                        )
                        if _cmetrics is not None and len(_cmetrics) == 4:
                            _cmed, _crms, _cp90, _cquad = _cmetrics
                        elif _cmetrics is not None:
                            _cmed, _crms, _cp90 = _cmetrics[:3]
                    except Exception:
                        logger.debug("chi2_shift: RMS computation failed", exc_info=True)
                    if _cmetrics:
                        _cquality, _creasons = _classify_alignment_quality(
                            _cmed, _crms, _cp90,
                            _cquad.get("coverage_ok") if _cquad else None,
                            *_gates,
                        )
                    else:
                        # chi2_shift exists for source-free fields where no
                        # match metrics are measurable; unverified is the
                        # honest classification.
                        _cquality, _creasons = "unverified", []
                    try:
                        if _cmed is not None and np.isfinite(_cmed):
                            hdr["ALIGMED"] = (float(_cmed), "Alignment median offset (px)")
                        if _crms is not None and np.isfinite(_crms):
                            hdr["ALIGRMS"] = (float(_crms), "Alignment RMS (px)")
                        if _cp90 is not None and np.isfinite(_cp90):
                            hdr["ALIGP90"] = (float(_cp90), "Alignment P90 offset (px)")
                        hdr["ALIGMETH"] = ("chi2_shift", "Alignment method used")
                    except Exception:
                        pass
                    _atomic_fits_write(new_templateFpath, aligned_tpl, hdr)
                    try:
                        from functions import refresh_sibling_weight_map

                        refresh_sibling_weight_map(new_templateFpath)
                    except Exception:
                        pass

                    method_used = "chi2_shift"
                    logger.log(STATUS, "Alignment succeeded (method: %s).", method_used)
                    return AlignmentResult(
                        scienceFpath, new_templateFpath, method_used, _cmed,
                        rms_px=_crms, p90_px=_cp90,
                        coverage_ok=(
                            _cquad.get("coverage_ok") if _cquad else None
                        ),
                        n_matches=(
                            _cquad.get("n_matched") if _cquad else None
                        ),
                        finite_fraction=float(np.mean(np.isfinite(aligned_tpl))),
                        quality=_cquality,
                        message="; ".join(_creasons),
                    )

                except Exception as _e:
                    log_warning_from_exception(logger, "chi2_shift alignment failed", _e)
                    return _fail("chi2_shift", str(_e))

            # ------------------------------------------------------------------
            # Cascade
            # ------------------------------------------------------------------
            if method == "swarp":
                out = _attempt(_swarp)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_reproject)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_astroalign)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_chi2_shift)
                if out[0] and out[1]:
                    return _finish(out)
                logger.error(
                    "All alignment methods failed (swarp->reproject->astroalign->chi2_shift). "
                    "Skipping template subtraction: aligned inputs are required "
                    "for a meaningful difference image."
                )
                return _finish(_fail(str(method), "all methods failed"))

            if method == "astroalign":
                out = _attempt(_astroalign)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_reproject)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_swarp)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_chi2_shift)
                if out[0] and out[1]:
                    return _finish(out)
                logger.error(
                    "All alignment methods failed (astroalign->reproject->swarp->chi2_shift). "
                    "Skipping template subtraction: aligned inputs are required "
                    "for a meaningful difference image."
                )
                return _finish(_fail(str(method), "all methods failed"))

            if method == "reproject":
                logger.warning(
                    "alignment_method='reproject' selected. spalipy ('spalipy') is the most robust "
                    "alignment method. Consider switching to spalipy."
                )
                out = _attempt(_reproject)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_swarp)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_astroalign)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_chi2_shift)
                if out[0] and out[1]:
                    return _finish(out)
                logger.error(
                    "All alignment methods failed (reproject->swarp->astroalign->chi2_shift). "
                    "Skipping template subtraction: aligned inputs are required "
                    "for a meaningful difference image."
                )
                return _finish(_fail(str(method), "all methods failed"))

            if method == "spalipy":
                # spalipy: spline-warp registration for non-homogeneous distortion.
                out = _attempt(_spalipy)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_swarp)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_reproject)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_astroalign)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_chi2_shift)
                if out[0] and out[1]:
                    return _finish(out)
                logger.error(
                    "All alignment methods failed (spalipy->swarp->reproject->astroalign->chi2_shift). "
                    "Skipping template subtraction: aligned inputs are required "
                    "for a meaningful difference image."
                )
                return _finish(_fail(str(method), "all methods failed"))

            if method == "tweakwcs":
                # tweakwcs: STScI WCS tweaking + reproject (HST/JWST approach).
                out = _attempt(_tweakwcs)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_swarp)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_reproject)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_astroalign)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_chi2_shift)
                if out[0] and out[1]:
                    return _finish(out)
                logger.error(
                    "All alignment methods failed (tweakwcs->swarp->reproject->astroalign->chi2_shift). "
                    "Skipping template subtraction: aligned inputs are required "
                    "for a meaningful difference image."
                )
                return _finish(_fail(str(method), "all methods failed"))

            if method == "chi2_shift":
                # chi2_shift: cross-correlation for extended-source-dominated fields.
                out = _attempt(_chi2_shift)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_swarp)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_reproject)
                if out[0] and out[1]:
                    return _finish(out)
                out = _attempt(_astroalign)
                if out[0] and out[1]:
                    return _finish(out)
                logger.error(
                    "All alignment methods failed (chi2_shift->swarp->reproject->astroalign). "
                    "Skipping template subtraction: aligned inputs are required "
                    "for a meaningful difference image."
                )
                return _finish(_fail(str(method), "all methods failed"))

            # Unknown method: default to spalipy (RA/DEC pre-matching +
            # spline-warp handles non-homogeneous distortion).
            logger.warning(
                "Unknown alignment_method=%r; defaulting to spalipy -> swarp -> reproject -> astroalign "
                "(spalipy is the most robust alignment method).",
                method,
            )
            out = _attempt(_spalipy)
            if out[0] and out[1]:
                return _finish(out)
            out = _attempt(_swarp)
            if out[0] and out[1]:
                return _finish(out)
            out = _attempt(_reproject)
            if out[0] and out[1]:
                return _finish(out)
            out = _attempt(_astroalign)
            if out[0] and out[1]:
                return _finish(out)
            out = _attempt(_chi2_shift)
            if out[0] and out[1]:
                return _finish(out)
            logger.error(
                "All alignment methods failed. "
                "Skipping template subtraction: aligned inputs are required "
                "for a meaningful difference image."
            )
            return _finish(_fail(str(method), "all methods failed"))

        except Exception:
            logger.exception("Error during alignment")
            return AlignmentResult(
                None, None, str(method), None,
                quality="fail",
                message="unhandled exception in align()",
            )

    def _update_target_coordinates_after_alignment(
        self, aligned_image_path: str, method_name: str
    ) -> bool:
        """
        Update target pixel coordinates in input_yaml to reflect the WCS after alignment.

        After alignment, the WCS changes (resampling onto a common grid). The target
        pixel coordinates must be recalculated using the new WCS to ensure subsequent
        operations (subtraction, photometry) use correct coordinates.

        Args
        ----
        aligned_image_path : str
            Path to the aligned science image.
        method_name : str
            Name of the alignment method (for logging).

        Returns
        -------
        bool
            True if update succeeded, False otherwise.
        """
        try:
            aligned_data, aligned_header = read_fits(aligned_image_path)
            aligned_wcs = get_wcs(aligned_header)

            # Reject a corrupt WCS (NaN at the pixel origin).
            test_ra, test_dec = aligned_wcs.all_pix2world([0], [0], 0)
            if not (np.isfinite(test_ra[0]) and np.isfinite(test_dec[0])):
                logger.error(
                    "WCS validation failed after %s alignment: WCS produces NaN at pixel (0,0). This indicates a corrupted WCS header.",
                    method_name
                )
                return False

            # RA/Dec are invariant under regridding; only the pixel mapping changes.
            target_ra = self.input_yaml["target_ra"]
            target_dec = self.input_yaml["target_dec"]

            new_target_x, new_target_y = aligned_wcs.all_world2pix(
                target_ra, target_dec, 0
            )

            if not (np.isfinite(new_target_x) and np.isfinite(new_target_y)):
                logger.error(
                    "WCS validation failed after %s alignment: Target RA/Dec conversion produced NaN pixel coordinates "
                    "(RA=%.6f, Dec=%.6f). This indicates a corrupted WCS header.",
                    method_name, target_ra, target_dec
                )
                return False

            old_target_x = self.input_yaml.get("target_x_pix", np.nan)
            old_target_y = self.input_yaml.get("target_y_pix", np.nan)

            self.input_yaml["target_x_pix"] = float(new_target_x)
            self.input_yaml["target_y_pix"] = float(new_target_y)

            # Warn when the target lands within 50 px of an edge.
            h, w = aligned_data.shape
            margin = 50  # pixels
            if not (margin <= new_target_x < w - margin and margin <= new_target_y < h - margin):
                logger.warning(
                    "Target after %s alignment is close to image edge: (%.1f, %.1f) in %dx%d image (margin=%d px). "
                    "This may cause issues with subtraction/photometry.",
                    method_name, new_target_x, new_target_y, w, h, margin
                )

            logger.info(
                "Target coords updated after %s alignment:\t(%.1f, %.1f) -> (%.1f, %.1f) px | "
                "WCS CRPIX=(%.1f, %.1f) CRVAL=(%.6f, %.6f)",
                method_name, old_target_x, old_target_y, new_target_x, new_target_y,
                aligned_header.get("CRPIX1", np.nan), aligned_header.get("CRPIX2", np.nan),
                aligned_header.get("CRVAL1", np.nan), aligned_header.get("CRVAL2", np.nan)
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to update target coordinates after %s alignment: %s",
                method_name, e
            )
            return False

    # -----------------------------------------------------------------
    # FITS processing helpers
    # -----------------------------------------------------------------

    @staticmethod
    def process_fits_file(
        data: np.ndarray,
        header: fits.Header,
        coords: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, WCS]:
        """
        Extract a rectangular cutout and return the updated WCS.

        Parameters
        ----------
        coords : (min_row, min_col, max_row, max_col)
        """
        from functions import nan_crop
        # Use exact float centre; nan_crop handles the rounding.
        center_x = (coords[3] + coords[1]) / 2.0
        center_y = (coords[2] + coords[0]) / 2.0
        ny = coords[2] - coords[0] + 1
        nx = coords[3] - coords[1] + 1
        new_header = header.copy()
        cropped, new_header = nan_crop(data, new_header, center_x, center_y, ny, nx)
        return cropped, get_wcs(new_header)

    @staticmethod
    def largest_histogram_rectangle(
        heights: List[float],
    ) -> Tuple[float, Tuple[int, float, int]]:
        """
        Classic O(n) stack-based algorithm for the largest rectangle in a
        histogram.  Used as a subroutine by ``find_largest_available_area``.

        Returns
        -------
        (max_area, (left_col, height, right_col))
        """
        stack: List[int] = []
        max_area = 0.0
        best = (0, 0.0, 0)

        for i in range(len(heights) + 1):
            while stack and (i == len(heights) or heights[i] < heights[stack[-1]]):
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    left = (stack[-1] + 1) if stack else 0
                    best = (left, h, i - 1)
            stack.append(i)

        return max_area, best

    def find_largest_available_area(
        self,
        image: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """
        Find the largest axis-aligned rectangle of *valid* pixels.

        A pixel is valid if it is finite AND non-zero.

        **BUG FIX**: The original code used ``(~isnan) | (!=0)`` which is
        True for almost all pixels.  Changed to AND so that only pixels
        that are *both* finite and non-zero are considered valid.

        Returns
        -------
        (row_start, col_start, row_end, col_end)
        """
        valid = np.isfinite(image) & (image != 0)
        rows, cols = valid.shape
        height = np.zeros(cols, dtype=int)

        best_area = 0
        best_coords = (0, 0, 0, 0)

        for r in range(rows):
            # height[c] = run length of consecutive valid pixels ending at row r.
            height = np.where(valid[r], height + 1, 0)

            area, (left, h, right) = self.largest_histogram_rectangle(height.tolist())
            if area > best_area:
                best_area = area
                best_coords = (r - int(h) + 1, left, r, right)

        return best_coords

    # -----------------------------------------------------------------
    # Cropping
    # -----------------------------------------------------------------

    def crop(
        self,
        scienceFpath: str,
        templateFpath: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Crop science (and optionally template) images to the maximal
        overlapping, data-bearing rectangle centred on the target.

        When both images are provided, the crop region is the intersection
        of both non-uniform regions.

        Returns
        -------
        (cropped_science, cropped_template_or_None)
        """
        sci_name = Path(scienceFpath).name
        ref_name = Path(templateFpath).name if templateFpath is not None else "None"
        logger.info(
            log_step(f"Crop: {sci_name} vs {ref_name}")
        )

        target_ra = self.input_yaml["target_ra"]
        target_dec = self.input_yaml["target_dec"]
        target_x_pix = np.floor(self.input_yaml["target_x_pix"])
        target_y_pix = np.floor(self.input_yaml["target_y_pix"])

        # --- Load science image (single I/O) ---
        scienceImage, scienceHeader = read_fits(scienceFpath)
        imageWCS = get_wcs(scienceHeader)
        
        # Preserve FWHM and APER from original header for alignment
        original_fwhm = scienceHeader.get("FWHM", scienceHeader.get("fwhm"))
        original_aper = scienceHeader.get("APER")

        cy, cx, top, bot, left, right = self.find_non_uniform_center(scienceImage)
        # find_non_uniform_center returns inclusive bounds, so add 1.
        height = bot - top + 1
        width = right - left + 1

        # Ensure even dimensions (needed for FFT-based subtraction later)
        height -= height % 2
        width -= width % 2

        scienceDir = Path(scienceFpath).parent
        cropped_scienceFpath = str(scienceDir / Path(scienceFpath).name)

        # ------------------------------------------------------------------
        # Science-only crop (no template)
        # ------------------------------------------------------------------
        if templateFpath is None:
            if not (np.isfinite(cx) and np.isfinite(cy)):
                logger.error(
                    f"Invalid crop center coordinates: cx={cx}, cy={cy}. "
                    f"This indicates find_non_uniform_center returned invalid values. "
                    f"Image shape: {scienceImage.shape}"
                )
                raise ValueError(
                    f"Invalid crop center: cx={cx}, cy={cy} (must be finite)"
                )
            from functions import nan_crop
            scienceImage, scienceHeader = nan_crop(
                scienceImage, scienceHeader,
                np.floor(cx), np.floor(cy),
                height, width,
            )

            # Second pass: tighten to the largest all-valid rectangle.
            coords = self.find_largest_available_area(scienceImage)
            scienceImage_tmp, scienceHeader_newwcs = self.process_fits_file(
                scienceImage,
                scienceHeader,
                coords,
            )
            target_x_pix, target_y_pix = scienceHeader_newwcs.all_world2pix(
                target_ra,
                target_dec,
                0,
            )

            border = 10
            h, w = scienceImage_tmp.shape
            if (
                border <= target_x_pix < w - border
                and border <= target_y_pix < h - border
            ):
                from functions import update_header_from_wcs
                scienceImage = scienceImage_tmp
                update_header_from_wcs(scienceHeader, scienceHeader_newwcs)

            # Restore FWHM and APER from original header for alignment
            if original_fwhm is not None:
                scienceHeader["FWHM"] = original_fwhm
            if original_aper is not None:
                scienceHeader["APER"] = original_aper
            
            write_fits(cropped_scienceFpath, scienceImage, scienceHeader)
            return cropped_scienceFpath, None

        # ------------------------------------------------------------------
        # Joint science + template crop
        # ------------------------------------------------------------------
        templateImage, templateHeader = read_fits(templateFpath)
        templateWCS = get_wcs(templateHeader)
        cropped_templateFpath = str(scienceDir / Path(templateFpath).name)
        
        # Preserve FWHM and APER from original headers for alignment
        original_template_fwhm = templateHeader.get("FWHM", templateHeader.get("fwhm"))
        original_template_aper = templateHeader.get("APER")

        cy_t, cx_t, top_t, bot_t, left_t, right_t = self.find_non_uniform_center(
            templateImage
        )
        # find_non_uniform_center returns inclusive bounds, so add 1.
        height_t = bot_t - top_t + 1
        width_t = right_t - left_t + 1

        # Use the smaller dimension from either image
        if height_t < height:
            height, cy = height_t, cy_t
        if width_t < width:
            width, cx = width_t, cx_t

        size = (height - height % 2, width - width % 2)
        # Pass the exact fractional centre; nan_crop rounds deterministically.
        position = (cx, cy)
        
        if not (np.isfinite(cx) and np.isfinite(cy)):
            logger.error(
                f"Invalid crop center coordinates: cx={cx}, cy={cy}.\n"
                f"    This indicates find_non_uniform_center returned\n"
                f"    invalid values. Science image shape:\n"
                f"    {scienceImage.shape}, template image shape:\n"
                f"    {templateImage.shape}"
            )
            raise ValueError(
                f"Invalid crop center: cx={cx}, cy={cy} (must be finite)"
            )

        # Distance from the target to the uniform padding edge, per image.
        d_uniform_template = distance_to_uniform_row_col(
            templateImage, x=target_x_pix, y=target_y_pix
        )
        d_uniform_science = distance_to_uniform_row_col(
            scienceImage, x=target_x_pix, y=target_y_pix
        )

        if np.isfinite(d_uniform_template) or np.isfinite(d_uniform_science):
            # Bail out (no crop) if the WCS cannot map the crop centre.
            try:
                test_ra, test_dec = imageWCS.all_pix2world([cx], [cy], 0)
                if not (np.isfinite(test_ra[0]) and np.isfinite(test_dec[0])):
                    logger.error(
                        f"Invalid WCS transformation at crop center\n"
                        f"    (cx={cx}, cy={cy}): RA={test_ra[0]},\n"
                        f"    Dec={test_dec[0]}. This will cause the crop\n"
                        f"    to fail. Falling back to no crop."
                    )
                    return scienceFpath, templateFpath
            except Exception as e:
                logger.error("WCS validation failed: %s. Falling back to no crop.", e)
                return scienceFpath, templateFpath
            
            # nan_crop preserves WCS distortion keywords that a plain
            # slice+CRPIX update would lose.
            from functions import nan_crop
            scienceImage, scienceHeader = nan_crop(
                scienceImage, scienceHeader,
                position[0], position[1],
                size[0], size[1],
            )
            templateImage, templateHeader = nan_crop(
                templateImage, templateHeader,
                position[0], position[1],
                size[0], size[1],
            )

            # Mark shared invalid regions (NaNs from chip gaps or out-of-bounds)
            mask = np.isnan(templateImage) | np.isnan(scienceImage)
            templateImage[mask] = np.nan
            scienceImage[mask] = np.nan

            # Tighten to the largest all-valid rectangle.
            coords = self.find_largest_available_area(scienceImage)
            scienceImage_tmp, scienceHeader_newwcs = self.process_fits_file(
                scienceImage,
                scienceHeader,
                coords,
            )
            templateImage_tmp, templateHeader_newwcs = self.process_fits_file(
                templateImage,
                templateHeader,
                coords,
            )

            target_x_pix, target_y_pix = scienceHeader_newwcs.all_world2pix(
                target_ra,
                target_dec,
                0,
            )
            border = self.input_yaml.get("scale", 0)
            h, w = scienceImage_tmp.shape

            if (
                border <= target_x_pix < w - border
                and border <= target_y_pix < h - border
            ):
                from functions import update_header_from_wcs
                scienceImage = scienceImage_tmp
                templateImage = templateImage_tmp
                update_header_from_wcs(scienceHeader, scienceHeader_newwcs)
                update_header_from_wcs(templateHeader, templateHeader_newwcs)
            else:
                logger.info("Target too close to border; keeping initial crop")

            # Safety sweep: inf pixels from the crop become NaN.
            templateImage[~np.isfinite(templateImage)] = np.nan
            scienceImage[~np.isfinite(scienceImage)] = np.nan

        # Restore FWHM and APER from original headers for alignment
        if original_fwhm is not None:
            scienceHeader["FWHM"] = original_fwhm
        if original_aper is not None:
            scienceHeader["APER"] = original_aper
        if original_template_fwhm is not None:
            templateHeader["FWHM"] = original_template_fwhm
        if original_template_aper is not None:
            templateHeader["APER"] = original_template_aper

        write_fits(cropped_templateFpath, templateImage, templateHeader)
        write_fits(cropped_scienceFpath, scienceImage, scienceHeader)

        # Diagnostic: after joint crop, the same sky target should map to
        # similar pixels in both images if WCS updates were propagated correctly.
        try:
            sci_w = get_wcs(scienceHeader)
            ref_w = get_wcs(templateHeader)
            sx, sy = sci_w.all_world2pix(target_ra, target_dec, 0)
            tx, ty = ref_w.all_world2pix(target_ra, target_dec, 0)
            logger.info(
                "Post-crop WCS target mapping: science=(%.2f, %.2f) template=(%.2f, %.2f) delta=(%.2f, %.2f) px",
                float(sx),
                float(sy),
                float(tx),
                float(ty),
                float(sx - tx),
                float(sy - ty),
            )
        except Exception as exc:
            log_warning_from_exception(
                logger, "Post-crop WCS target-mapping diagnostic failed", exc
            )

        return cropped_scienceFpath, cropped_templateFpath

    # -----------------------------------------------------------------
    # PSF helpers
    # -----------------------------------------------------------------

    @staticmethod
    def pad_psf(psf: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Symmetrically zero-pad *psf* to *target_shape*.

        The padding value is NaN to preserve chip gaps and invalid regions.
        """
        dy = target_shape[0] - psf.shape[0]
        dx = target_shape[1] - psf.shape[1]
        pad_width = (
            (dy // 2, dy - dy // 2),
            (dx // 2, dx - dx // 2),
        )
        return np.pad(psf, pad_width, mode="constant", constant_values=np.nan)

    @staticmethod
    def determine_kernel_order(
        sci_fwhm: float,
        ref_fwhm: float,
        n_sources: int = 0,
    ) -> Tuple[int, str]:
        """
        Choose an appropriate spatial-kernel polynomial order based on
        the number of matched sources available to constrain the spatial
        variation of the kernel across the field of view.

        The DFT kernel handles the PSF shape difference; the polynomial
        order only controls how the kernel varies spatially.  Order 0
        (constant) works well for small fields with good alignment.

        Auto-selection is capped at order 1 to avoid excessive RAM usage.
        Order 2+ scales as (n_terms x kernel_pixels)^2 and can require
        >7 GB for typical kernel sizes.  Users can override via YAML.

        Returns
        -------
        (order, explanation) : (int, str)
        """
        if n_sources < 20:
            return 0, "Few sources (<20). Constant kernel - sufficient for small fields."
        else:
            return 1, "Sufficient sources (>=20). Linear spatial variation."

    # -----------------------------------------------------------------
    # Outlier detection (rolling MAD/std)
    # -----------------------------------------------------------------

    @staticmethod
    def robust_outlier_mask(
        values: np.ndarray,
        window_size: int = 50,
        n_sigma: int = 3,
        use_mad: bool = True,
        min_for_window: int = 20,
    ) -> np.ndarray:
        """
        Identify outliers against a rolling local statistic (MAD or std).

        **BUG FIX**: The original returned a mask in *sorted* order
        instead of the *original* order.  Now we argsort, compute the
        mask on sorted values, and invert the permutation before
        returning.

        Parameters
        ----------
        values : np.ndarray
            1-D array of measurements.
        window_size : int
            Rolling-window width.
        n_sigma : int
            Number of MAD/sigma for the inlier threshold.
        use_mad : bool
            If True use MAD; otherwise standard deviation.
        min_for_window : int
            If fewer values than this, use a global statistic instead.

        Returns
        -------
        np.ndarray of bool
            True = inlier (keep), False = outlier.
        """
        n = len(values)

        def _mad_sigma(v):
            # scale="normal" applies the 1.4826 factor - the raw MAD is a
            # factor ~1.5 tighter than a sigma, so using it directly would
            # over-reject inliers at the n_sigma threshold.
            return median_abs_deviation(v, scale="normal")

        # --- Global fallback for small samples ---
        if n < min_for_window:
            med = np.median(values)
            scale = _mad_sigma(values) if use_mad else np.std(values)
            # A zero scale (identical values) must not flag everything as an
            # outlier: use a tiny floor so only genuinely deviant points fail.
            if not np.isfinite(scale) or scale <= 0:
                scale = np.finfo(float).eps
            return np.abs(values - med) <= n_sigma * scale

        # --- Rolling-window approach ---
        # Sort, but remember the original order so we can unsort at the end.
        sort_idx = np.argsort(values)
        sorted_vals = values[sort_idx]

        s = pd.Series(sorted_vals)
        rolling_med = s.rolling(window_size, center=True, min_periods=1).median().values

        if use_mad:
            rolling_scale = (
                s.rolling(window_size, center=True, min_periods=1)
                .apply(_mad_sigma, raw=True)
                .values
            )
        else:
            rolling_scale = (
                s.rolling(window_size, center=True, min_periods=1).std().values
            )

        # pandas .values may be read-only; copy before in-place edits.
        rolling_scale = np.array(rolling_scale, copy=True)

        # Edge windows can yield NaN scale; fall back to the global scale.
        global_scale = (
            _mad_sigma(sorted_vals) if use_mad else np.std(sorted_vals)
        )
        rolling_scale[np.isnan(rolling_scale)] = global_scale
        # Floor the scale: a window of identical values gives scale=0, which
        # would otherwise flag every value in the window as an outlier.
        rolling_scale[~(rolling_scale > 0)] = np.finfo(float).eps

        residuals = np.abs(sorted_vals - rolling_med)
        inlier_sorted = residuals <= n_sigma * rolling_scale

        # Un-sort: place the mask back in the original element order.
        inlier_original = np.empty(n, dtype=bool)
        inlier_original[sort_idx] = inlier_sorted
        return inlier_original

    # -----------------------------------------------------------------
    # Flux-consistency matching
    # -----------------------------------------------------------------

    @staticmethod
    def _catalog_xy(
        df: pd.DataFrame,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Best available pixel positions for a matched catalog.

        ``x_pix``/``y_pix`` are centroid-refined by the caller; ``x``/``y``
        and ``x_centroid``/``y_centroid`` are fallbacks.
        """
        for xc, yc in (
            ("x_pix", "y_pix"),
            ("x", "y"),
            ("x_centroid", "y_centroid"),
        ):
            if xc in df.columns and yc in df.columns:
                x = pd.to_numeric(df[xc], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(df[yc], errors="coerce").to_numpy(dtype=float)
                return x, y
        return None, None

    @staticmethod
    def _psf_stamp_chi2(
        image: Optional[np.ndarray],
        x_pos: Optional[np.ndarray],
        y_pos: Optional[np.ndarray],
        psf_model,
        fwhm: float,
        gain: float = 1.0,
        mask: Optional[np.ndarray] = None,
        model_floor_frac: float = 0.05,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Reduced chi^2 of a PSF-model fit on a small stamp per position.

        For each ``(x, y)`` the PSF model is re-centred on the stamp's own
        2D-Gaussian centroid -- so the metric measures *shape* mismatch
        (blends, extended sources, artifacts) rather than centroid error --
        then amplitude and a constant sky offset are fit by weighted least
        squares.  Pixel weights use the local sky MAD plus a Poisson term
        and a fractional model-mismatch floor, so bright, well-fit stars
        are not rejected for sub-percent ePSF inaccuracies.

        Units: ``image`` must be in ADU and ``gain`` in electrons/ADU
        (the pipeline convention).  The Poisson term ``A*P/gain`` is then
        the model source counts converted back to ADU-space variance;
        passing an electron-space image would underestimate it by gain^2.

        Returns ``NaN`` where the fit cannot be made (off-frame, fully
        masked, too few valid pixels) and ``+inf`` where the fitted
        amplitude is non-positive.  ``NaN`` means "unverifiable" -- the
        caller decides whether such sources are kept.

        ``n_jobs`` > 1 spreads the per-source fits over a process pool for
        large position lists; any pool failure falls back to the serial
        loop.
        """
        n = 0 if x_pos is None else len(x_pos)
        chi2 = np.full(n, np.nan)
        if (
            image is None
            or psf_model is None
            or x_pos is None
            or y_pos is None
            or n == 0
            or not np.isfinite(fwhm)
            or fwhm <= 0
        ):
            return chi2

        img = np.asarray(image, dtype=float)
        _os = getattr(psf_model, "oversampling", 1)
        osamp = max(1, int(np.max(np.atleast_1d(_os)))) if _os is not None else 1
        h = max(4, int(np.ceil(1.5 * fwhm)))  # ~3xFWHM fit box
        h2 = h + 3  # outer ring supplies the local sky MAD
        yy, xx = np.indices((2 * h + 1, 2 * h + 1))
        wy, wx = np.indices((2 * h2 + 1, 2 * h2 + 1))
        gain = float(gain) if np.isfinite(gain) and gain > 0 else 1.0

        # A GriddedPSFModel must be pinned to the local ePSF at each
        # source's detector position before stamp rendering - the render
        # call uses stamp-local coordinates, which would otherwise select
        # the wrong grid cell.
        _psf_is_gridded = False
        try:
            from photutils.psf import GriddedPSFModel as _GPM

            _psf_is_gridded = isinstance(psf_model, _GPM)
        except Exception:
            _psf_is_gridded = False

        ctx = {
            "img": img,
            "mask": mask,
            "psf_model": psf_model,
            "psf_is_gridded": _psf_is_gridded,
            "xs": np.asarray(x_pos, dtype=float),
            "ys": np.asarray(y_pos, dtype=float),
            "osamp": osamp,
            "h": h,
            "h2": h2,
            "ring_r2": (xx - h) ** 2 + (yy - h) ** 2,
            "ring": (wx - h2) ** 2 + (wy - h2) ** 2 > h ** 2,
            "gain": gain,
            "model_floor_frac": float(model_floor_frac),
        }

        try:
            n_jobs = max(1, int(n_jobs))
        except Exception:
            n_jobs = 1
        # The pool is only worthwhile for large candidate lists; below the
        # threshold its startup cost exceeds the serial loop's runtime.
        if n_jobs > 1 and n >= 64:
            try:
                from aperture import _resolve_n_jobs
                from multiprocessing import Pool

                workers = _resolve_n_jobs(n_jobs)
                if workers > 1:
                    with Pool(
                        processes=workers,
                        initializer=_psf_stamp_init_shared,
                        initargs=(ctx,),
                    ) as pool:
                        vals = pool.map(_psf_stamp_task, range(n))
                    chi2 = np.asarray(vals, dtype=float)
                    return chi2
            except Exception:
                logger.debug(
                    "Parallel PSF stamp chi2 failed; using serial path.",
                    exc_info=True,
                )

        for i in range(n):
            chi2[i] = _psf_stamp_eval(
                float(ctx["xs"][i]), float(ctx["ys"][i]), ctx
            )
        return chi2

    def find_flux_consistent_sources(
        self,
        catalog_img: pd.DataFrame,
        catalog_tpl: pd.DataFrame,
        params: Optional[FluxMatchParams] = None,
        make_plot: bool = True,
        psf_vetting: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Tuple[float, float]]:
        """
        Cross-match science and template photometric catalogs, rejecting
        outliers via RANSAC and MAD-based statistics.

        The procedure:
          1. Convert aperture fluxes to instrumental magnitudes.
          2. Apply error and positivity cuts.
          3. Remove rolling-window outliers.
          4. Fit a linear relation with the slope fixed to 1 (intercept
             only) using RANSAC.
          5. Optionally trim to a central percentile range and re-fit.
          6. Within the flux-consistent inliers, fit a straight line in
             log space ``log10(S/N_ref) = a*log10(S/N_sci) + b``
             (least squares + MAD clip + unweighted refit) and flag the
             S/N-consistent subset.
          7. Return inlier catalog rows (flux- AND S/N-consistent, and
             PSF-well-fit when ``psf_vetting`` is supplied) and a compact
             fit tuple ``(mag_slope, flux_scale)`` where:
             - ``mag_slope`` is the fitted slope in magnitude space
             - ``flux_scale`` is ``10**(-0.4 * intercept)``, i.e. the multiplicative
               flux ratio corresponding to zero-point offset.

        Parameters
        ----------
        catalog_img, catalog_tpl : pd.DataFrame
            Matched source catalogs (same length, same row ordering).
        params : FluxMatchParams or None
            Structured parameters.  Uses defaults if None.
        make_plot : bool
            Save a diagnostic PDF alongside the science image.
        psf_vetting : dict or None
            Optional PSF-fit-quality veto.  Recognised keys:
            ``image_sci``/``image_tpl`` (pixel arrays),
            ``psf_sci``/``psf_tpl`` (PSF models, e.g. photutils ImagePSF),
            ``fwhm_sci``/``fwhm_tpl`` (pixels), ``gain_sci``/``gain_tpl``
            (e-/ADU, default 1), ``mask_sci``/``mask_tpl`` (optional,
            True = bad pixel), ``chi2_max`` (reduced-chi2 rejection
            threshold, default ``sfft_psf_chi2_max`` = 5.0),
            ``model_floor_frac`` (fractional model-mismatch noise floor,
            default ``sfft_psf_model_floor_frac`` = 0.05),
            ``model_floor_frac_sci``/``model_floor_frac_tpl`` (per-side
            overrides - use a larger floor where the vetting model is
            analytic and cannot capture real PSF structure, else bright
            stars are vetoed for model error), ``min_keep`` (minimum
            surviving inlier sources, default ``sfft_psf_min_keep`` = 8)
            and ``min_keep_frac`` (minimum surviving fraction of the
            robust inlier pool, default ``sfft_psf_min_keep_frac`` = 0.5).
            A source is vetoed when its PSF-stamp reduced chi2 exceeds
            ``chi2_max`` in either image; sources that cannot be measured
            (edge/masked stamps) are kept.  The veto is adaptive: if it
            would leave fewer than ``min_keep`` sources or less than
            ``min_keep_frac`` of the robust pool, the best-fitting vetoed
            sources (lowest worst-image chi2) are restored until the
            floor is met and the relaxed effective threshold is logged.
            When omitted the selection is unchanged.

        Returns
        -------
        (inlier_df, (mag_slope, flux_scale))
        """
        if params is None:
            params = FluxMatchParams()

        # Allow YAML override of mag_residual_threshold for SFFT source
        # selection.  The default 0.3 mag is permissive; users can tighten
        # it (e.g. 0.15) to reject variable/mismatched sources more
        # aggressively before passing them to SFFT.
        _ts_cfg_fc = self.input_yaml.get("template_subtraction", {}) or {}
        _yaml_thresh = float(_ts_cfg_fc.get("sfft_flux_mag_residual_threshold", 0.0))
        if _yaml_thresh > 0:
            params.mag_residual_threshold = _yaml_thresh

        empty = pd.DataFrame(columns=catalog_img.columns)
        nan_fit = (np.nan, np.nan)

        try:
            n_img, n_tpl = len(catalog_img), len(catalog_tpl)
            logger.info(
                "Finding consistent sources: image [%d] vs reference [%d]",
                n_img,
                n_tpl,
            )

            # --- Validation ---
            if n_img != n_tpl:
                logger.info("Catalog length mismatch")
                return empty, nan_fit
            if (
                params.flux_key not in catalog_img
                or params.flux_key_err not in catalog_img
            ):
                logger.info("Missing flux columns in catalog")
                return empty, nan_fit
            if n_img < params.min_absolute_samples:
                logger.info("Not enough sources for analysis")
                return empty, nan_fit

            # --- Positive flux + SNR filter ---
            ok = (
                (catalog_img[params.flux_key].values > 0)
                & (catalog_tpl[params.flux_key].values > 0)
                & (catalog_img["threshold"] > 3)
                & (catalog_tpl["threshold"] > 3)
            )
            n_positive = int(ok.sum())
            if not ok.any():
                logger.info("No sources pass positivity + SNR cuts")
                return empty, nan_fit

            # --- Saturation filter (exclude non-linear sources) ---
            # Reject sources entering the non-linear regime (peak >=
            # nonlinear_peak_frac * saturate, same convention as
            # zeropoint/main.py), not just fully saturated ones.
            try:
                from main import SATURATE_INTERNAL_FALLBACK
            except ImportError:
                SATURATE_INTERNAL_FALLBACK = np.inf
            saturate = self.input_yaml.get("saturate", SATURATE_INTERNAL_FALLBACK)
            nonlinear_frac = self.input_yaml.get("zeropoint", {}).get(
                "nonlinear_peak_frac", 0.85
            )
            if (
                np.isfinite(saturate)
                and saturate > 0
                and "peak_flux" in catalog_img.columns
                and "peak_flux" in catalog_tpl.columns
            ):
                peak_img = catalog_img["peak_flux"].values
                peak_tpl = catalog_tpl["peak_flux"].values
                sat_thresh = nonlinear_frac * saturate
                sat_ok = (peak_img < sat_thresh) & (peak_tpl < sat_thresh)
                n_sat = int(np.sum(ok & ~sat_ok))
                if n_sat > 0:
                    logger.info(
                        "Removed %d non-linear sources from flux comparison (peak_flux >= %.2f x saturate).",
                        n_sat,
                        nonlinear_frac,
                    )
                ok = ok & sat_ok

            n_flux_eligible = int(ok.sum())
            f_img = catalog_img[params.flux_key].values[ok]
            f_tpl = catalog_tpl[params.flux_key].values[ok]
            fe_img = catalog_img[params.flux_key_err].values[ok]
            fe_tpl = catalog_tpl[params.flux_key_err].values[ok]

            mag_img, me_img = flux_to_mag(f_img, fe_img)
            mag_tpl, me_tpl = flux_to_mag(f_tpl, fe_tpl)

            good = (me_img < 1) & (me_tpl < 1)
            n_measurable = int(good.sum())
            if not good.any():
                logger.info("No sources pass error threshold")
                return empty, nan_fit

            mag_img, mag_tpl = mag_img[good], mag_tpl[good]
            me_img, me_tpl = me_img[good], me_tpl[good]
            mag_err = np.sqrt(me_img ** 2 + me_tpl ** 2)
            indices = catalog_img.index[ok][good]

            # Per-source S/N in each image, aligned with the post-'good'
            # arrays and tracked through the outlier/spatial masks below.
            with np.errstate(divide="ignore", invalid="ignore"):
                snr_img = np.abs(f_img[good]) / np.maximum(np.abs(fe_img[good]), 1e-30)
                snr_tpl = np.abs(f_tpl[good]) / np.maximum(np.abs(fe_tpl[good]), 1e-30)

            # --- PSF-fit quality veto ---
            # A source can be flux-consistent yet poorly fit by the PSF model
            # in one image (blend, extended morphology, cosmic ray, masked-
            # neighbour contamination); SFFT kernel priors must be clean
            # point sources in BOTH images.  When the caller supplies the
            # images + PSF models via ``psf_vetting``, require a converged
            # low-chi2 model stamp fit on each side.  Sources whose stamp
            # cannot be measured (edge, fully masked) are kept -- consistent
            # with the pipeline's "cannot verify -> keep" convention -- and
            # reported in the log so the reason for rejection stays visible.
            psf_ok = np.ones(len(mag_img), dtype=bool)
            psf_chi2_sci = np.full(len(mag_img), np.nan)
            psf_chi2_tpl = np.full(len(mag_img), np.nan)
            psf_q = np.full(len(mag_img), np.nan)
            _pv_chi2_max = 0.0
            _pv_min_keep = 0
            _pv_min_keep_frac = 0.0
            if psf_vetting:
                _pv_chi2_max = float(
                    psf_vetting.get(
                        "chi2_max",
                        _ts_cfg_fc.get("sfft_psf_chi2_max", 5.0),
                    )
                )
                _pv_floor = float(
                    psf_vetting.get(
                        "model_floor_frac",
                        _ts_cfg_fc.get("sfft_psf_model_floor_frac", 0.05),
                    )
                )
                # Per-side floors: an analytic model (Moffat at the image
                # FWHM) cannot capture real PSF structure, so its expected
                # mismatch is much larger than an empirical ePSF's.  The
                # caller sets model_floor_frac_<side> when that side's
                # model is analytic; without it bright stars - the best
                # kernel anchors - are vetoed for model error, not
                # morphology.
                _pv_floor_sci = float(
                    psf_vetting.get("model_floor_frac_sci", _pv_floor)
                )
                _pv_floor_tpl = float(
                    psf_vetting.get("model_floor_frac_tpl", _pv_floor)
                )
                _pv_min_keep = int(
                    psf_vetting.get(
                        "min_keep",
                        _ts_cfg_fc.get("sfft_psf_min_keep", 8),
                    )
                )
                _pv_min_keep_frac = float(
                    psf_vetting.get(
                        "min_keep_frac",
                        _ts_cfg_fc.get("sfft_psf_min_keep_frac", 0.5),
                    )
                )
                if _pv_chi2_max > 0:
                    _pv_x, _pv_y = self._catalog_xy(catalog_img.loc[indices])
                    _pv_x_t, _pv_y_t = self._catalog_xy(catalog_tpl.loc[indices])
                    _pv_njobs = int(
                        (self.input_yaml or {}).get("n_jobs", 1) or 1
                    )
                    try:
                        psf_chi2_sci = self._psf_stamp_chi2(
                            psf_vetting.get("image_sci"),
                            _pv_x,
                            _pv_y,
                            psf_vetting.get("psf_sci"),
                            psf_vetting.get("fwhm_sci", np.nan),
                            gain=psf_vetting.get("gain_sci", 1.0),
                            mask=psf_vetting.get("mask_sci"),
                            model_floor_frac=_pv_floor_sci,
                            n_jobs=_pv_njobs,
                        )
                    except Exception:
                        logger.debug(
                            "Science-side PSF quality fit failed; skipping.",
                            exc_info=True,
                        )
                    try:
                        psf_chi2_tpl = self._psf_stamp_chi2(
                            psf_vetting.get("image_tpl"),
                            _pv_x_t,
                            _pv_y_t,
                            psf_vetting.get("psf_tpl"),
                            psf_vetting.get("fwhm_tpl", np.nan),
                            gain=psf_vetting.get("gain_tpl", 1.0),
                            mask=psf_vetting.get("mask_tpl"),
                            model_floor_frac=_pv_floor_tpl,
                            n_jobs=_pv_njobs,
                        )
                    except Exception:
                        logger.debug(
                            "Reference-side PSF quality fit failed; skipping.",
                            exc_info=True,
                        )
                    bad_sci = ~np.isnan(psf_chi2_sci) & (
                        psf_chi2_sci > _pv_chi2_max
                    )
                    bad_tpl = ~np.isnan(psf_chi2_tpl) & (
                        psf_chi2_tpl > _pv_chi2_max
                    )
                    psf_ok = ~(bad_sci | bad_tpl)
                    # Worst-image chi2 per source, used to rank vetoed
                    # sources if the keep floor forces a partial restore.
                    # fmax ignores NaN (unmeasurable) so an unverifiable
                    # side does not punish an otherwise-good source.
                    psf_q = np.fmax(psf_chi2_sci, psf_chi2_tpl)
                    n_bad_s = int(bad_sci.sum())
                    n_bad_t = int(bad_tpl.sum())
                    n_unmeas = int(
                        (np.isnan(psf_chi2_sci) | np.isnan(psf_chi2_tpl)).sum()
                    )
                    logger.info(
                        "PSF-fit quality (chi2_red <= %.2f, floors sci=%.2f "
                        "tpl=%.2f): %d rejected in science, %d in reference, "
                        "%d unmeasurable (kept)",
                        _pv_chi2_max, _pv_floor_sci, _pv_floor_tpl,
                        n_bad_s, n_bad_t, n_unmeas,
                    )

            # --- MAD-based outlier removal on paired magnitude residuals ---
            if params.fix_slope_to_one:
                _pair_residual = mag_tpl - mag_img
                _pair_center = float(np.nanmedian(_pair_residual))
                _pair_scale = float(
                    median_abs_deviation(_pair_residual, scale="normal")
                )
                _pair_limit = np.maximum(
                    max(float(params.mag_residual_threshold), 1e-6),
                    3.0 * mag_err,
                )
                _pair_limit = np.maximum(_pair_limit, 5.0 * _pair_scale)
                robust_mask = np.abs(_pair_residual - _pair_center) <= _pair_limit
            else:
                robust_mask = np.ones(len(mag_img), dtype=bool)
            n_pair_residual = int(robust_mask.sum())
            keep_mask = robust_mask & psf_ok

            # Adaptive relaxation: the veto must never starve the source
            # pool.  If fewer than min_keep inlier sources survive -- or
            # the veto stripped more than (1 - min_keep_frac) of the
            # robust pool, which signals systematic model-limited
            # rejection rather than real pathologies -- restore just
            # enough vetoed sources, best (lowest) worst-image chi2
            # first, so n_want inliers remain.  This keeps the veto for
            # genuinely bad fits while guaranteeing a usable prior set;
            # the relaxation is logged with the effective threshold so
            # it stays diagnosable.
            _n_want = min(
                max(
                    _pv_min_keep,
                    params.min_absolute_samples,
                    int(np.ceil(_pv_min_keep_frac * int(robust_mask.sum()))),
                ),
                int(robust_mask.sum()),
            )
            if int(keep_mask.sum()) < _n_want:
                cand = np.where(robust_mask & ~psf_ok)[0]
                if cand.size:
                    q_rank = np.where(np.isnan(psf_q), -np.inf, psf_q)
                    order = cand[np.argsort(q_rank[cand], kind="stable")]
                    n_restore = min(_n_want - int(keep_mask.sum()), order.size)
                    restore = order[:n_restore]
                    psf_ok[restore] = True
                    keep_mask = robust_mask & psf_ok
                    _chi2_eff = (
                        float(q_rank[restore].max())
                        if n_restore > 0
                        else _pv_chi2_max
                    )
                    logger.warning(
                        "PSF-quality veto relaxed to keep %d sources: "
                        "effective chi2_red threshold %.2f -> %.2f "
                        "(%d best-fitting vetoed sources restored)",
                        _n_want, _pv_chi2_max, _chi2_eff, int(n_restore),
                    )
            n_psf_vetted = int(keep_mask.sum())
            mag_img_r = mag_img[keep_mask]
            mag_tpl_r = mag_tpl[keep_mask]
            mag_err_r = mag_err[keep_mask]
            me_img_r = me_img[keep_mask]
            me_tpl_r = me_tpl[keep_mask]
            snr_img_r = snr_img[keep_mask]
            snr_tpl_r = snr_tpl[keep_mask]
            idx_r = indices[keep_mask]

            # --- Optional spatial thinning to avoid over-clustered regions ---
            # Skip for sparse fields - thinning can remove valid sources when
            # we already have too few for reliable kernel fitting.
            if (
                params.use_spatial_thinning
                and len(mag_img_r) >= 15
                and {"x", "y"}.issubset(catalog_img.columns)
            ):
                x_pos = catalog_img.loc[idx_r, "x"].to_numpy(dtype=float)
                y_pos = catalog_img.loc[idx_r, "y"].to_numpy(dtype=float)
                spatial_mask = self._spatially_uniform_mask(
                    x_pos,
                    y_pos,
                    value=mag_img_r,
                    n_bins=params.spatial_n_bins,
                    max_per_bin=params.spatial_max_per_bin,
                    prefer_small_value=True,
                )
                if spatial_mask.any():
                    mag_img_r = mag_img_r[spatial_mask]
                    mag_tpl_r = mag_tpl_r[spatial_mask]
                    mag_err_r = mag_err_r[spatial_mask]
                    me_img_r = me_img_r[spatial_mask]
                    me_tpl_r = me_tpl_r[spatial_mask]
                    snr_img_r = snr_img_r[spatial_mask]
                    snr_tpl_r = snr_tpl_r[spatial_mask]
                    idx_r = idx_r[spatial_mask]
                    x_pos = x_pos[spatial_mask]
                    y_pos = y_pos[spatial_mask]

            if len(mag_img_r) < params.min_absolute_samples:
                logger.info("Too few sources after robust filtering")
                return empty, nan_fit

            # --- RANSAC / fallback fit ---
            X = mag_img_r.reshape(-1, 1)
            y = mag_tpl_r
            thresh = params.mag_residual_threshold
            slope, intercept = np.nan, np.nan
            inliers = np.zeros(len(y), dtype=bool)
            method_used = "None"

            # NOTE: use module-level regressors, not self.*
            # FixedSlopeRegressor pins the slope to exactly 1.0 (intercept-
            # only fit).  PenalisedSlopeRegressor takes (slope_constraint,
            # slope_tolerance): tolerance 0.5 softly constrains the slope,
            # inf leaves it free.
            if params.fix_slope_to_one:
                _slope_tol = 0.0
            elif params.enforce_slope_constraint:
                _slope_tol = 0.5
            else:
                _slope_tol = np.inf

            if len(y) >= 4:
                try:
                    if params.fix_slope_to_one:
                        base_est = FixedSlopeRegressor(slope=1.0)
                    else:
                        base_est = ConstrainedSlopeRegressor(
                            slope_constraint=1.0,
                            slope_tolerance=_slope_tol,
                        )

                    # Fit RANSAC on the high-S/N subset only: mag err < 0.2
                    # corresponds to S/N > ~5.4 (me ~ 1.086/SNR).
                    snr_mask = mag_err_r < 0.2
                    if snr_mask.sum() >= 4:
                        X_snr = X[snr_mask]
                        y_snr = y[snr_mask]
                        ransac = RANSACRegressor(
                            estimator=base_est,
                            residual_threshold=thresh,
                            max_trials=params.max_trials,
                            min_samples=max(
                                int(params.min_samples_fraction * len(y_snr)),
                                params.min_absolute_samples,
                                4,
                            ),
                            random_state=42,
                        )
                        ransac.fit(X_snr, y_snr)
                        # Apply the fitted model to the FULL catalog to identify
                        # all inliers, not just the high-S/N subset used for fitting.
                        slope = ransac.estimator_.slope_
                        intercept = ransac.estimator_.intercept_
                        residuals_all = y - (slope * X.ravel() + intercept)
                        inliers = np.abs(residuals_all) < thresh
                        method_used = "RANSAC (S/N > 5 fit, all-source inliers)"
                    else:
                        # Too few high-S/N sources: fit all sources instead.
                        ransac = RANSACRegressor(
                            estimator=base_est,
                            residual_threshold=thresh,
                            max_trials=params.max_trials,
                            min_samples=max(
                                int(params.min_samples_fraction * len(y)),
                                params.min_absolute_samples,
                                4,
                            ),
                            random_state=42,
                        )
                        ransac.fit(X, y)
                        inliers = ransac.inlier_mask_
                        if inliers.any():
                            slope = ransac.estimator_.slope_
                            intercept = ransac.estimator_.intercept_
                            method_used = "RANSAC (all sources)"

                    # Inliers clustered in a narrow magnitude range make the
                    # RANSAC fit unreliable; fall back to the median offset.
                    if inliers.sum() >= 10:
                        mag_range = np.nanpercentile(mag_img_r[inliers], [5, 95])
                        mag_span = mag_range[1] - mag_range[0]
                        total_mag_range = np.nanpercentile(mag_img_r, [5, 95])
                        total_mag_span = total_mag_range[1] - total_mag_range[0]
                        if total_mag_span > 0 and mag_span / total_mag_span < 0.3:
                            logger.warning(
                                f"RANSAC inliers are clustered in magnitude space (span={mag_span:.2f} vs total={total_mag_span:.2f}); falling back to median offset"
                            )
                            diffs = y - X.ravel()
                            median_diff = np.nanmedian(diffs)
                            inliers = np.abs(diffs - median_diff) < params.mag_residual_threshold
                            slope, intercept = 1.0, median_diff
                            method_used = "Median offset (RANSAC clustered)"
                        else:
                            # 3-sigma clip on inlier residuals; catches
                            # outliers RANSAC admitted.
                            residuals = y[inliers] - (slope * X[inliers].ravel() + intercept)
                            residual_std = np.std(residuals)
                            if residual_std > 0:
                                inlier_residuals = np.abs(residuals) < 3 * residual_std
                                n_clipped = int(np.sum(~inlier_residuals))
                                if n_clipped > 0:
                                    logger.info(
                                        "Post-RANSAC sigma-clipping removed %d additional outliers (3 sigma)",
                                        n_clipped,
                                    )
                                    inlier_indices = np.where(inliers)[0]
                                    inliers[inlier_indices[~inlier_residuals]] = False
                except Exception as exc:
                    logger.debug("RANSAC failed: %s", exc)

            if method_used == "None" and len(y) >= 2:
                diffs = y - X.ravel()
                median_diff = np.nanmedian(diffs)
                inliers = np.abs(diffs - median_diff) < params.mag_residual_threshold
                slope, intercept = 1.0, median_diff
                method_used = "Median offset"

            # --- Optional percentile refinement ---
            final_inliers = inliers.copy()
            if params.use_percentile_cut and inliers.sum() >= 5:
                lo, hi = np.nanpercentile(mag_img_r[inliers], params.percentiles)
                central = inliers & (mag_img_r >= lo) & (mag_img_r <= hi)
                if central.sum() >= params.min_absolute_samples:
                    if _SKLEARN_AVAILABLE:
                        if params.fix_slope_to_one:
                            est = FixedSlopeRegressor(slope=1.0)
                        else:
                            est = ConstrainedSlopeRegressor(
                                slope_constraint=1.0,
                                slope_tolerance=_slope_tol,
                            )
                        est.fit(
                            mag_img_r[central].reshape(-1, 1), mag_tpl_r[central]
                        )
                        slope, intercept = est.slope_, est.intercept_
                        final_inliers = central
                    else:
                        logger.debug(
                            "Skipping percentile refinement; scikit-learn not available."
                        )

            # --- Optional bin-wise consistency filter for low-S/N magnitude regimes ---
            # Only reject bins where very few sources are inliers (avoids isolated
            # "lucky" inliers in noisy regimes). No continuity constraint: allow
            # inliers across the full magnitude range so the flux comparison is not
            # over-restricted. Use a modest majority threshold to avoid over-masking.
            # Skip for sparse fields - bins with 1-2 sources make the inlier fraction
            # meaningless and can remove valid sources.
            bin_majority_frac = 0.25  # reject bin only if inlier fraction below this
            if final_inliers.sum() >= params.min_absolute_samples and len(mag_img_r) >= 15:
                try:
                    bin_width = 0.5  # mag
                    mag_min = float(np.nanmin(mag_img_r))
                    mag_max = float(np.nanmax(mag_img_r))
                    if (
                        np.isfinite(mag_min)
                        and np.isfinite(mag_max)
                        and mag_max > mag_min
                    ):
                        edges = np.arange(mag_min, mag_max + bin_width, bin_width)
                        min_bin_count = max(params.min_absolute_samples, 5)
                        for i in range(len(edges) - 1):
                            in_bin = (mag_img_r >= edges[i]) & (
                                mag_img_r < edges[i + 1]
                            )
                            count = int(in_bin.sum())
                            if count < min_bin_count:
                                continue
                            frac_inlier = float(final_inliers[in_bin].mean())
                            if frac_inlier < bin_majority_frac:
                                final_inliers[in_bin] = False
                except Exception:
                    logger.debug(
                        "Bin-wise inlier refinement skipped due to an error.",
                        exc_info=True,
                    )

            # If we modified the inlier set after the RANSAC/percentile fit (e.g. via
            # bin-wise refinement), recompute the intercept on the final inliers so the
            # diagnostic line passes through the plotted inlier cloud.
            if np.isfinite(slope) and final_inliers.sum() >= max(3, params.min_absolute_samples):
                try:
                    diffs_final = mag_tpl_r[final_inliers] - (slope * mag_img_r[final_inliers])
                    intercept = float(np.nanmedian(diffs_final))
                except Exception:
                    pass

            # --- Linearity check ---
            # Verify that the magnitude relationship is truly linear (not
            # just that residuals are small).  A non-linear relationship
            # indicates variable sources, mismatched sources, or systematic
            # flux-dependent bias (e.g., different aperture sizes for
            # different FWHM).  SFFT assumes all prior sources are
            # non-variable stars with a constant flux ratio.
            if final_inliers.sum() >= 5:
                _mi = mag_img_r[final_inliers]
                _mt = mag_tpl_r[final_inliers]
                _resid = _mt - (slope * _mi + intercept)

                # Pearson correlation: should be very high (> 0.95) for
                # a truly linear relationship between non-variable stars.
                _corr_pearson = float(np.corrcoef(_mi, _mt)[0, 1])

                # Residual-magnitude correlation: if residuals correlate
                # with magnitude, the relationship is non-linear (curved).
                # Spearman is used because it is insensitive to outlier magnitude.
                from scipy.stats import spearmanr
                _corr_resid, _ = spearmanr(_mi, _resid)
                _corr_resid = float(_corr_resid) if np.isfinite(_corr_resid) else 0.0

                _min_corr = float(
                    _ts_cfg_fc.get("sfft_flux_linearity_min_corr", 0.90)
                )
                _max_resid_corr = float(
                    _ts_cfg_fc.get("sfft_flux_linearity_max_resid_corr", 0.50)
                )

                logger.info(
                    "Flux linearity: Pearson r=%.3f, residual-mag Spearman rho=%.3f "
                    "(thresholds: min_r=%.2f, max_resid_rho=%.2f)",
                    _corr_pearson, _corr_resid, _min_corr, _max_resid_corr,
                )

                if _corr_pearson < _min_corr:
                    logger.warning(
                        "Flux linearity check FAILED: Pearson r=%.3f < %.3f.\n"
                        "    Magnitude relationship is not linear -- sources\n"
                        "    may include variables, mismatches, or\n"
                        "    flux-dependent bias. Proceeding with inliers but\n"
                        "    SFFT may reject many sources.",
                        _corr_pearson, _min_corr,
                    )

                if abs(_corr_resid) > _max_resid_corr:
                    logger.warning(
                        "Flux linearity check FAILED:\n"
                        "    residual-mag Spearman rho=%.3f > %.3f.\n"
                        "    Residuals correlate with magnitude -- the flux\n"
                        "    relationship is non-linear (curved). This\n"
                        "    indicates systematic flux-dependent bias (e.g.,\n"
                        "    different aperture sizes for different FWHM) or a\n"
                        "    large fraction of variable sources. SFFT kernel\n"
                        "    fit may be poor.",
                        abs(_corr_resid), _max_resid_corr,
                    )
                    # Flag non-linear sources: reject inliers whose residuals
                    # follow the trend (keep only those that don't follow it).
                    # This removes sources in the magnitude range where the
                    # non-linearity is strongest.
                    #
                    # Two safeguards for sparse fields:
                    #  * a minimum inlier count -- with only a handful of
                    #    sources, a rank correlation is easily inflated by a
                    #    few faint, high-error points and removing priors
                    #    costs SFFT more than a marginal non-linearity does.
                    #  * error-normalised rejection -- the quadratic trend is
                    #    fit with 1/sigma_mag weights and a source is removed
                    #    only where the trend amplitude is significant both
                    #    against the detrended scatter AND relative to that
                    #    source's own magnitude error (heteroscedastic-safe).
                    _min_trend_src = int(
                        _ts_cfg_fc.get("sfft_flux_linearity_min_trend_sources", 20)
                    )
                    if final_inliers.sum() < _min_trend_src:
                        logger.info(
                            "Non-linearity correction skipped: only %d inliers "
                            "(< %d) -- too few sources to trust a residual-mag "
                            "trend; keeping all priors.",
                            final_inliers.sum(), _min_trend_src,
                        )
                    else:
                        _sig_res = np.asarray(
                            mag_err_r[final_inliers], dtype=float
                        )
                        _sig_fin = _sig_res[np.isfinite(_sig_res) & (_sig_res > 0)]
                        _sig_floor = (
                            float(np.median(_sig_fin)) if _sig_fin.size else 0.1
                        )
                        _sig_res = np.where(
                            np.isfinite(_sig_res) & (_sig_res > 0),
                            _sig_res, _sig_floor,
                        )
                        # Weighted quadratic fit (polyfit w multiplies the
                        # residuals, so w = 1/sigma gives chi-square weighting).
                        _poly_coef = np.polyfit(_mi, _resid, 2, w=1.0 / _sig_res)
                        _poly_pred = np.polyval(_poly_coef, _mi)
                        _poly_resid = _resid - _poly_pred
                        _poly_std = np.std(_poly_resid)
                        if _poly_std > 0:
                            # Reject sources where the quadratic trend explains
                            # most of the residual AND is significant relative
                            # to the source's own photometric error.
                            _trend_bad = (np.abs(_poly_pred) > 2.0 * _poly_std) & (
                                np.abs(_poly_pred) > 2.0 * _sig_res
                            )
                            _n_trend_bad = int(_trend_bad.sum())
                            if _n_trend_bad > 0 and (_final_inliers_count := final_inliers.sum()) - _n_trend_bad >= 5:
                                _trend_bad_indices = np.where(final_inliers)[0][_trend_bad]
                                final_inliers[_trend_bad_indices] = False
                                logger.info(
                                    "Non-linearity correction: removed %d sources "
                                    "following quadratic trend (%d inliers remain).",
                                    _n_trend_bad, final_inliers.sum(),
                                )

            # --- S/N-consistency check (second stage) ---
            # The comparison runs in two stages: the magnitude (flux)
            # inliers are found first, then the S/N-consistent subset is
            # identified *within* them.  A source can be flux-consistent
            # yet S/N-inconsistent when its noise estimate is anomalous in
            # one image (blending, chip-edge, masked-neighbour
            # contamination); such sources are poor SFFT kernel priors.
            #
            # sigma_SNR: propagating the flux measurement error alone
            # gives sigma_SNR = sigma_F/sigma_F = 1 (S/N is a
            # noise-normalised flux).  The dominant term at high S/N is
            # the uncertainty of the noise model itself: sigma_SNR ~
            # SNR * rho, with rho the fractional uncertainty of the
            # flux-error estimate (~10% for aperture photometry,
            # background-estimation dominated).
            _snr_rel = float(_ts_cfg_fc.get("sfft_snr_model_relerr", 0.10))
            snr_slope_fit, snr_intercept_fit = np.nan, np.nan
            snr_resid_all = np.full(len(mag_img_r), np.nan)
            is_snr_inlier = final_inliers.copy()

            _ok_snr = (
                final_inliers
                & np.isfinite(snr_img_r)
                & np.isfinite(snr_tpl_r)
                & (snr_img_r > 0)
                & (snr_tpl_r > 0)
            )
            if int(_ok_snr.sum()) >= 4:
                try:
                    _lx = np.log10(snr_img_r[_ok_snr])
                    _ly = np.log10(snr_tpl_r[_ok_snr])

                    # Straight-line fit in log space
                    # log10(S/N_ref) = a*log10(S/N_sci) + b via least
                    # squares.  Degenerate x (S/N_sci nearly constant)
                    # leaves the slope unidentifiable: fall back to the
                    # proportional (slope=1) median offset so outlier
                    # clipping still works.
                    _degen = float(np.ptp(_lx)) < 1e-6
                    if _degen:
                        _a, _b = 1.0, float(np.nanmedian(_ly - _lx))
                    else:
                        _coef = np.polyfit(_lx, _ly, 1)
                        _a, _b = float(_coef[0]), float(_coef[1])
                        if not (np.isfinite(_a) and np.isfinite(_b)):
                            _a, _b = 1.0, float(np.nanmedian(_ly - _lx))
                            _degen = True

                    _resid = _ly - (_a * _lx + _b)
                    _mad = float(
                        np.nanmedian(np.abs(_resid - np.nanmedian(_resid)))
                    )
                    _tol = (
                        max(3.0 * 1.4826 * _mad, 0.1)
                        if np.isfinite(_mad)
                        else 0.1
                    )
                    _keep = np.abs(_resid - np.nanmedian(_resid)) < _tol

                    # Unweighted least-squares refit on the clipped
                    # subset; skipped for degenerate x where the slope is
                    # unidentifiable.
                    if int(_keep.sum()) >= 4 and not _degen:
                        _coef = np.polyfit(_lx[_keep], _ly[_keep], 1)
                        if np.isfinite(_coef).all():
                            _a, _b = float(_coef[0]), float(_coef[1])
                            _resid = _ly - (_a * _lx + _b)
                            _keep = (
                                np.abs(_resid - np.nanmedian(_resid)) < _tol
                            )

                    if (
                        int(_keep.sum()) >= 4
                        and np.isfinite(_a)
                        and np.isfinite(_b)
                    ):
                        snr_slope_fit, snr_intercept_fit = _a, _b
                        is_snr_inlier = np.zeros(len(mag_img_r), dtype=bool)
                        is_snr_inlier[np.where(_ok_snr)[0][_keep]] = True
                        snr_resid_all[_ok_snr] = _resid
                        logger.info(
                            "S/N consistency: slope=%.3f, offset=%.3f dex, "
                            "inliers=%d/%d flux-consistent sources",
                            snr_slope_fit,
                            snr_intercept_fit,
                            int(is_snr_inlier.sum()),
                            int(final_inliers.sum()),
                        )
                    else:
                        logger.info(
                            "S/N-consistency: too few inliers after "
                            "clipping; keeping flux inliers."
                        )
                except Exception:
                    logger.debug(
                        "S/N-consistency fit failed; keeping flux inliers.",
                        exc_info=True,
                    )

            # --- Build result DataFrame ---
            result = catalog_img.loc[idx_r].copy()
            result["mag_img"] = mag_img_r
            result["mag_tpl"] = mag_tpl_r
            result["snr_img"] = snr_img_r
            result["snr_tpl"] = snr_tpl_r
            result["mag_err_img"] = me_img_r
            result["mag_err_tpl"] = me_tpl_r
            result["snr_img_err"] = np.sqrt(
                1.0 + (_snr_rel * snr_img_r) ** 2
            )
            result["snr_tpl_err"] = np.sqrt(
                1.0 + (_snr_rel * snr_tpl_r) ** 2
            )
            result["snr_residual"] = snr_resid_all
            result["mag_residual"] = y - (slope * mag_img_r + intercept)
            result["is_snr_inlier"] = is_snr_inlier
            result["is_inlier"] = final_inliers
            result["is_robust"] = True
            result["psf_chi2_sci"] = psf_chi2_sci[keep_mask]
            result["psf_chi2_tpl"] = psf_chi2_tpl[keep_mask]
            result["is_psf_inlier"] = psf_ok[keep_mask]

            non_robust_idx = indices[~keep_mask]
            nr = catalog_img.loc[non_robust_idx].copy()
            nr["mag_img"] = mag_img[~keep_mask]
            nr["mag_tpl"] = mag_tpl[~keep_mask]
            nr["snr_img"] = snr_img[~keep_mask]
            nr["snr_tpl"] = snr_tpl[~keep_mask]
            nr["mag_err_img"] = me_img[~keep_mask]
            nr["mag_err_tpl"] = me_tpl[~keep_mask]
            nr["snr_img_err"] = np.sqrt(
                1.0 + (_snr_rel * nr["snr_img"].to_numpy(dtype=float)) ** 2
            )
            nr["snr_tpl_err"] = np.sqrt(
                1.0 + (_snr_rel * nr["snr_tpl"].to_numpy(dtype=float)) ** 2
            )
            nr["snr_residual"] = np.nan
            nr["mag_residual"] = np.nan
            nr["is_snr_inlier"] = False
            nr["is_inlier"] = False
            # A PSF-vetoed source can still be a flux inlier; keep the two
            # flags separate so the diagnostic plot can distinguish them.
            nr["is_robust"] = robust_mask[~keep_mask]
            nr["psf_chi2_sci"] = psf_chi2_sci[~keep_mask]
            nr["psf_chi2_tpl"] = psf_chi2_tpl[~keep_mask]
            nr["is_psf_inlier"] = psf_ok[~keep_mask]

            full = pd.concat([result, nr])

            # Keep mag-space fit as authoritative. For convenience we also return
            # the equivalent multiplicative flux scale from the intercept.
            # NOTE: this is not a full linear flux model when slope != 1.
            mag_slope = float(slope)
            flux_scale = 10 ** (-0.4 * intercept) if np.isfinite(intercept) else np.nan

            logger.info(
                "Fit [%s]: slope=%.3f, intercept=%.3f, inliers=%d/%d, "
                "kept=%d/%d (PSF-vetoed=%d)",
                method_used,
                slope,
                intercept,
                final_inliers.sum(),
                len(y),
                len(mag_img_r),
                len(mag_img),
                int((~psf_ok).sum()),
            )
            if np.isfinite(intercept) and abs(intercept) > 0.5:
                logger.debug(
                    "RANSAC intercept=%.3f mag indicates flux scaling between science and template "
                    "(expected for different exposure times/zeropoints; handled by subtraction pipeline).",
                    intercept,
                )

            # --- Optional diagnostic plot ---
            if make_plot and "fpath" in self.input_yaml:
                self._plot_flux_comparison(
                    full,
                    result,
                    mag_err,
                    mag_err_r,
                    mag_img_r,
                    slope,
                    intercept,
                    final_inliers,
                    (snr_slope_fit, snr_intercept_fit),
                )

            # Final selection: flux-consistent AND S/N-consistent.  Guard
            # against starving the pipeline when the S/N stage rejects too
            # many sources.
            _sel = (
                full["is_inlier"].to_numpy(bool)
                & full["is_snr_inlier"].to_numpy(bool)
            )
            if int(_sel.sum()) < params.min_absolute_samples:
                logger.warning(
                    "S/N-consistency stage left %d sources (< %d); "
                    "returning flux-consistent inliers only.",
                    int(_sel.sum()),
                    params.min_absolute_samples,
                )
                _sel = full["is_inlier"].to_numpy(bool)

            logger.info(
                "Flux-match stages: input=%d positive/SNR=%d unsaturated=%d "
                "measurable=%d pair-residual=%d PSF/RANSAC candidates=%d "
                "flux-inliers=%d S/N-inliers=%d output=%d",
                n_img,
                n_positive,
                n_flux_eligible,
                n_measurable,
                n_pair_residual,
                n_psf_vetted,
                int(final_inliers.sum()),
                int(is_snr_inlier.sum()),
                int(_sel.sum()),
            )
            return (
                full[_sel],
                (mag_slope, flux_scale),
            )

        except Exception:
            logger.exception("Flux consistency check failed")
            return empty, nan_fit

    def _plot_flux_comparison(
        self,
        full: pd.DataFrame,
        robust: pd.DataFrame,
        mag_err_all: np.ndarray,
        mag_err_robust: np.ndarray,
        mag_img_robust: np.ndarray,
        slope: float,
        intercept: float,
        inliers: np.ndarray,
        snr_fit: Tuple[float, float] = (np.nan, np.nan),
    ) -> None:
        """Save a diagnostic magnitude/S/N-comparison plot to disk.

        The top panel shows the instrumental-magnitude relation with the
        fitted line; the bottom panel shows the S/N relation restricted to
        the flux-consistent sources, with S/N error bars and a fitted
        power-law slope.
        """
        from matplotlib import pyplot as plt
        from functions import set_size
        from plotting_utils import (
            apply_autophot_mplstyle, get_ransac_color, get_marker_size,
            get_alpha, get_line_width, ransac_grid, ransac_savefig,
            set_mag_axes_inverted_xy, get_plot_ext,
        )

        plt.ioff()
        apply_autophot_mplstyle()

        has_snr = (
            {"snr_img", "snr_tpl"}.issubset(robust.columns)
            and np.isfinite(robust["snr_img"].to_numpy(dtype=float)).any()
            and np.isfinite(robust["snr_tpl"].to_numpy(dtype=float)).any()
        )
        if has_snr:
            fig, (ax, ax_snr) = plt.subplots(
                2, 1,
                figsize=set_size(540, 1.7),
                gridspec_kw={"height_ratios": [3, 2]},
            )
        else:
            fig, ax = plt.subplots(figsize=set_size(540, 1))

        # Per-axis magnitude errors: science error on x, reference on y.
        # Fall back to the combined error for older result frames.
        if {"mag_err_img", "mag_err_tpl"}.issubset(robust.columns):
            xerr_all = robust["mag_err_img"].to_numpy(dtype=float)
            yerr_all = robust["mag_err_tpl"].to_numpy(dtype=float)
        else:
            xerr_all = np.asarray(mag_err_robust, dtype=float)
            yerr_all = xerr_all

        # Outliers first so inliers sit on top
        rej = ~robust["is_inlier"].to_numpy(bool)
        if rej.any():
            ax.errorbar(
                robust.loc[rej, "mag_img"],
                robust.loc[rej, "mag_tpl"],
                xerr=xerr_all[rej],
                yerr=yerr_all[rej],
                fmt="x",
                color=get_ransac_color('outliers'),
                ecolor="lightgrey",
                alpha=get_alpha('medium'),
                markersize=get_marker_size('medium'),
                capsize=get_marker_size('medium') / 4,
                elinewidth=0.5,
                label=f"Outliers [{np.sum(rej)}]",
            )
        sel = ~rej
        ax.errorbar(
            robust.loc[sel, "mag_img"],
            robust.loc[sel, "mag_tpl"],
            xerr=xerr_all[sel],
            yerr=yerr_all[sel],
            fmt="o",
            markersize=get_marker_size('medium'),
            color=get_ransac_color('flux_comparison'),
            ecolor="lightgrey",
            elinewidth=0.5,
            capsize=get_marker_size('medium') / 4,
            alpha=get_alpha('dark'),
            label=f"Inliers [{np.sum(sel)}]",
        )
        # Sources vetoed by the PSF-fit quality stage land in the
        # non-kept partition of `full`; show them distinctly so a poor
        # model fit is not mistaken for a flux outlier.
        if "is_psf_inlier" in full.columns:
            psf_bad = ~full["is_psf_inlier"].to_numpy(bool)
            if "mag_img" in full.columns and "mag_tpl" in full.columns:
                _pm = full["mag_img"].to_numpy(dtype=float)
                _pt = full["mag_tpl"].to_numpy(dtype=float)
                psf_bad &= np.isfinite(_pm) & np.isfinite(_pt)
            if psf_bad.any():
                ax.plot(
                    full.loc[psf_bad, "mag_img"],
                    full.loc[psf_bad, "mag_tpl"],
                    linestyle="none",
                    marker="D",
                    markersize=get_marker_size('medium') + 1,
                    markerfacecolor="none",
                    markeredgecolor=get_ransac_color('color_term'),
                    markeredgewidth=1.2,
                    alpha=get_alpha('medium'),
                    label=f"PSF-rejected [{int(psf_bad.sum())}]",
                )
        xx = np.linspace(mag_img_robust.min(), mag_img_robust.max(), 100)
        yy = slope * xx + intercept
        if np.isfinite(slope) and abs(float(slope) - 1.0) < 0.02:
            fit_lbl = rf"Fit: $m_{{\rm ref}} = m_{{\rm sci}} {intercept:+.3f}$ (slope$\approx$1)"
        else:
            fit_lbl = rf"Fit: $m_{{\rm ref}} = {slope:.3f}\,m_{{\rm sci}} {intercept:+.3f}$"
        if sel.sum() > 0:
            residuals = robust.loc[sel, "mag_tpl"].values - (slope * robust.loc[sel, "mag_img"].values + intercept)
            intercept_error = np.std(residuals) / np.sqrt(sel.sum())
            ax.fill_between(
                xx, yy - intercept_error, yy + intercept_error,
                color=get_ransac_color('error_band'), alpha=get_alpha('very_light'),
            )
        ax.plot(
            xx, yy,
            color=get_ransac_color('fit'),
            linestyle="--",
            lw=get_line_width("medium"),
            label=fit_lbl,
        )
        ax.set(
            xlabel="Science Instrumental Magnitude [mag]",
            ylabel="Reference Instrumental Magnitude [mag]",
        )
        ax.legend(
            loc="upper left", ncol=1, fontsize=8, frameon=False,
        )
        ransac_grid(ax)
        set_mag_axes_inverted_xy(ax)

        if has_snr:
            s_i = robust["snr_img"].to_numpy(dtype=float)
            s_t = robust["snr_tpl"].to_numpy(dtype=float)
            ok_s = np.isfinite(s_i) & np.isfinite(s_t) & (s_i > 0) & (s_t > 0)

            if {"snr_img_err", "snr_tpl_err"}.issubset(robust.columns):
                se_i = robust["snr_img_err"].to_numpy(dtype=float)
                se_t = robust["snr_tpl_err"].to_numpy(dtype=float)
            else:
                se_i = se_t = np.ones(len(s_i))

            # Cascade: flux-consistent sources first, then the
            # S/N-consistent subset within them (both from
            # find_flux_consistent_sources).
            flux_in = robust["is_inlier"].to_numpy(bool) & ok_s
            if "is_snr_inlier" in robust.columns:
                snr_in = robust["is_snr_inlier"].to_numpy(bool) & ok_s
            else:
                snr_in = flux_in.copy()
            snr_slope_f, snr_intercept_f = snr_fit

            # Flux outliers (faint), then flux-consistent but
            # S/N-inconsistent sources, then consistent sources on top.
            flux_out = ok_s & ~flux_in
            snr_out = flux_in & ~snr_in
            if flux_out.any():
                ax_snr.errorbar(
                    s_i[flux_out], s_t[flux_out],
                    xerr=se_i[flux_out], yerr=se_t[flux_out],
                    fmt="x", ms=get_marker_size('medium'),
                    color=get_ransac_color('outliers'),
                    ecolor="lightgrey", elinewidth=0.5,
                    capsize=get_marker_size('medium') / 4,
                    alpha=get_alpha('very_light'),
                    label=f"Flux outliers [{int(flux_out.sum())}]",
                )
            if snr_out.any():
                ax_snr.errorbar(
                    s_i[snr_out], s_t[snr_out],
                    xerr=se_i[snr_out], yerr=se_t[snr_out],
                    fmt="x", ms=get_marker_size('medium'),
                    color=get_ransac_color('color_term'),
                    ecolor="lightgrey", elinewidth=0.5,
                    capsize=get_marker_size('medium') / 4,
                    alpha=get_alpha('medium'),
                    label=f"S/N outliers [{int(snr_out.sum())}]",
                )
            ax_snr.errorbar(
                s_i[snr_in], s_t[snr_in],
                xerr=se_i[snr_in], yerr=se_t[snr_in],
                fmt="o", ms=get_marker_size('medium'),
                color=get_ransac_color('flux_comparison'),
                ecolor="lightgrey", elinewidth=0.5,
                capsize=get_marker_size('medium') / 4,
                alpha=get_alpha('dark'),
                label=f"Consistent [{int(snr_in.sum())}]",
            )

            # Fitted power law S/N_ref = 10^b * (S/N_sci)^a -- a straight
            # line on the log-log axes.
            if np.isfinite(snr_slope_f) and np.isfinite(snr_intercept_f):
                _xs_src = s_i[flux_in] if flux_in.any() else s_i[ok_s]
                lo = float(np.nanmin(_xs_src))
                hi = float(np.nanmax(_xs_src))
                if hi > lo > 0:
                    xs = np.logspace(np.log10(lo), np.log10(hi), 100)
                    ys = (10.0 ** snr_intercept_f) * xs ** snr_slope_f
                    if snr_in.sum() >= 2:
                        _res = (
                            np.log10(s_t[snr_in])
                            - (
                                snr_slope_f * np.log10(s_i[snr_in])
                                + snr_intercept_f
                            )
                        )
                        _band = float(
                            np.std(_res) / np.sqrt(snr_in.sum())
                        )
                        if np.isfinite(_band) and _band > 0:
                            ax_snr.fill_between(
                                xs,
                                ys * 10.0 ** (-_band),
                                ys * 10.0 ** (_band),
                                color=get_ransac_color('error_band'),
                                alpha=get_alpha('very_light'),
                            )
                    ax_snr.plot(
                        xs, ys,
                        color=get_ransac_color('fit'), linestyle="-",
                        lw=get_line_width('medium'),
                        label=(
                            rf"Fit: $\log\,(S/N)_{{\rm ref}} = "
                            rf"{snr_slope_f:.2f}\,\log\,(S/N)_{{\rm sci}} "
                            rf"{snr_intercept_f:+.2f}$"
                        ),
                    )
            ax_snr.set_xscale("log")
            ax_snr.set_yscale("log")
            ax_snr.set_xlabel("Science Image S/N")
            ax_snr.set_ylabel("Reference Image S/N")
            ax_snr.legend(
                loc="upper left", ncol=1, fontsize=8, frameon=False,
            )
            ransac_grid(ax_snr)

        base = Path(self.input_yaml["fpath"]).stem
        out = Path(self.input_yaml["fpath"]).parent / f"Flux_Comparison_{base}{get_plot_ext(self.input_yaml)}"
        ransac_savefig(fig, str(out))
        plt.close(fig)

    # -----------------------------------------------------------------
    # Neighbour-based bright-outlier masking
    # -----------------------------------------------------------------

    @staticmethod
    def mask_bright_outliers_relative_to_neighbors(
        cat: pd.DataFrame,
        flux_key: str = "flux_AP",
        sep_thresh: float = 10.0,
        contrast_thresh: float = 5.0,
    ) -> np.ndarray:
        """
        Flag sources that are much brighter than their local neighbours.

        Uses a KD-tree for O(n log n) spatial neighbour queries.

        Returns
        -------
        np.ndarray of bool
            True = keep, False = bright outlier.
        """
        coords = np.column_stack([cat["x"].values, cat["y"].values])
        fluxes = cat[flux_key].to_numpy()
        tree = cKDTree(coords)
        neighbours = tree.query_ball_point(coords, sep_thresh)

        keep = np.ones(len(cat), dtype=bool)
        for i, nbrs in enumerate(neighbours):
            # query_ball_point includes the query point itself.
            nbrs = [j for j in nbrs if j != i]
            if not nbrs:
                continue
            local_med = np.median(fluxes[nbrs])
            if local_med > 0 and fluxes[i] / local_med > contrast_thresh:
                keep[i] = False

        return keep

    # -----------------------------------------------------------------
    # Spatial downsampling helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _spatially_uniform_mask(
        x: np.ndarray,
        y: np.ndarray,
        value: Optional[np.ndarray] = None,
        n_bins: int = 10,
        max_per_bin: int = 10,
        prefer_small_value: bool = True,
    ) -> np.ndarray:
        """
        Construct a boolean mask that enforces an approximately uniform
        spatial sampling across the detector.

        Parameters
        ----------
        x, y : array_like
            Pixel coordinates for each source.
        value : array_like or None
            Optional "priority" value used to rank sources within each
            spatial bin (e.g. magnitude or SNR). When provided, sources
            are sorted so that either the smallest (brightest) or largest
            (highest SNR) values are kept depending on *prefer_small_value*.
        n_bins : int
            Number of bins per axis for the spatial grid.
        max_per_bin : int
            Maximum number of sources to keep per spatial bin.
        prefer_small_value : bool
            If True, keep the smallest *value* entries in each bin; if
            False, keep the largest.

        Returns
        -------
        mask : np.ndarray of bool
            True for sources retained after spatial thinning.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = len(x)
        mask = np.zeros(n, dtype=bool)

        if n == 0:
            return mask

        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            # No usable coordinates; keep everything.
            return np.ones(n, dtype=bool)

        idx_all = np.arange(n)[finite]
        x_f = x[finite]
        y_f = y[finite]

        # Normalise to [0, 1] to build a regular grid.
        x_min, x_max = float(x_f.min()), float(x_f.max())
        y_min, y_max = float(y_f.min()), float(y_f.max())
        dx = (x_max - x_min) or 1.0
        dy = (y_max - y_min) or 1.0

        xi = np.clip(((x_f - x_min) / dx * n_bins).astype(int), 0, n_bins - 1)
        yi = np.clip(((y_f - y_min) / dy * n_bins).astype(int), 0, n_bins - 1)

        if value is not None:
            cell_value_scores = np.asarray(value, float)[finite]
        else:
            # Flat priority when no value is supplied.
            cell_value_scores = np.zeros_like(x_f)

        for i in range(n_bins):
            for j in range(n_bins):
                cell = (xi == i) & (yi == j)
                if not np.any(cell):
                    continue
                cell_idx = np.where(cell)[0]
                if cell_idx.size <= max_per_bin:
                    mask[idx_all[cell_idx]] = True
                    continue

                # Rank within the cell.
                order = np.argsort(cell_value_scores[cell_idx])
                if not prefer_small_value:
                    order = order[::-1]
                keep_local = cell_idx[order[:max_per_bin]]
                mask[idx_all[keep_local]] = True

        # Safety: if spatial thinning removed everything, fall back to
        # keeping all sources.
        if not mask.any():
            return np.ones(n, dtype=bool)
        return mask

    # -----------------------------------------------------------------
    # Image subtraction
    # -----------------------------------------------------------------

    def subtract(
        self,
        scienceFpath: str,
        templateFpath: str,
        method: str = "sfft",
        kernel_order: int = 0,
        matching_sources: Optional[List[Tuple[float, float]]] = None,
        masked_sources: Optional[List[Tuple[float, float]]] = None,
        common_sources: Optional[List[Tuple[float, float]]] = None,
        stamp_loc: Optional[str] = None,
        scienceNoise: Optional[str] = None,
        templateNoise: Optional[str] = None,
        background_defects_mask: Optional[np.ndarray] = None,
        scale: Optional[int] = None,
        science_psf: Optional[str] = None,
        template_psf: Optional[str] = None,
    ) -> Tuple[
        Optional[str], Optional[np.ndarray], Optional[List[Tuple[float, float]]], Optional[int]
    ]:
        """
        Subtract the template from the science image.

        Supports three backends that are tried in cascade when a method
        fails:
          - **ZOGY** (Zackay, Ofek & Gal-Yam 2016) via a self-contained numpy implementation.
          - **SFFT** (Hu et al. 2022) via an external conda environment.
          - **HOTPANTS** (Becker 2015) via a compiled executable.

        The function:
          1. Loads both images in a single I/O pass.
          2. Builds individual + combined masks (saturated, NaN, extended).
          3. Runs the requested subtraction backend.
          4. Validates the output difference image.

        Parameters
        ----------
        scienceFpath, templateFpath : str
            Input FITS file paths.
        method : str
            Initial backend choice ('sfft', 'hotpants', 'zogy').
        kernel_order : int
            Spatial-variation polynomial order.
        matching_sources : list of (x, y) or None
            Kernel-prior positions on the aligned science grid, in
            **0-based** NumPy pixel coordinates.  The SFFT adapter converts
            them to 1-based SExtractor coordinates internally.
        masked_sources : list of (x, y) or None
            Positions to ban from kernel fitting, in **1-based**
            SExtractor/FITS coordinates (the convention ``run_sfft.py``
            expects for ``XY_PriorBan``).
        stamp_loc : str or None
            Path to a stamp-selection file (HOTPANTS ``-ssf``, 1-based
            coordinates).  When None, the HOTPANTS adapter builds one from
            ``matching_sources`` with the target excluded.
        scienceNoise, templateNoise : str or None
            Optional external noise-map FITS files.  They must live on the
            same pixel grid as the image actually fed to the backend; the
            template weight map is not resampled by ``align()``, so a
            mismatched map is dropped with a warning rather than
            misapplied.  Used by HOTPANTS (``-ini``/``-tni``).
        science_psf, template_psf : str or None
            Explicit PSF-stamp paths that take precedence over the
            ``PSF_model_*`` file discovery (e.g. ZOGY-specific analytic
            stamps written beside the run outputs).  Only ZOGY consumes
            PSF models; other backends ignore them.

        Returns
        -------
        (differenceFpath, visualization_mask, masked_centers, kernel_half_width)
            Paths and arrays, or (None, None, None, None) on failure.
        """
        if matching_sources is None:
            matching_sources = []
        if masked_sources is None:
            masked_sources = []
        if common_sources is None:
            common_sources = []

        # Normalise method once: the backend dispatch below compares against
        # lowercase names, so a config value like "SFFT" would otherwise be
        # silently skipped and fall through to the validation failure.
        method = str(method or "sfft").strip().lower()
        if method not in ("sfft", "hotpants", "zogy"):
            logger.warning(
                "Unknown subtraction method %r; defaulting to 'sfft'.", method
            )
            method = "sfft"

        t0 = time.time()
        sci_name = Path(scienceFpath).name
        ref_name = Path(templateFpath).name
        logger.info(
            log_step(f"Image subtraction: {sci_name} - {ref_name}")
        )

        prepared_template_fpath: Optional[str] = None
        template_work_fpath: str = str(templateFpath)
        _sci_clean_path: Optional[str] = None
        _ref_clean_path: Optional[str] = None
        _sci_prepared_path: Optional[str] = None
        _tpl_sfft_path: Optional[str] = None
        _diff_part_path: Optional[str] = None
        kernel_half_width: Optional[int] = None

        try:
            # =============================================================
            # 1. Load images (one read each)
            # =============================================================
            scienceImage, scienceHeader = read_fits(scienceFpath)
            templateImage, templateHeader = read_fits(templateFpath)
            scienceDir = Path(scienceFpath).parent
            base_name = Path(scienceFpath).name
            differenceFpath = str(scienceDir / f"diff_{base_name}")

            # ---------------------------------------------------------
            # Pre-subtraction grid validation.
            # Every backend subtracts pixel-by-pixel, so both inputs must
            # be 2-D images resampled onto a common grid.  A finite but
            # wrong-shaped or wrongly-projected template would silently
            # corrupt the difference image and downstream target
            # photometry.
            # ---------------------------------------------------------
            if scienceImage.ndim != 2 or templateImage.ndim != 2:
                logger.error(
                    "Subtraction requires 2-D images; got science ndim=%d, "
                    "template ndim=%d.",
                    scienceImage.ndim,
                    templateImage.ndim,
                )
                return None, None, None, None
            if scienceImage.shape != templateImage.shape:
                logger.error(
                    "Science/template shape mismatch %s vs %s; "
                    "template was not resampled to the science grid.",
                    scienceImage.shape,
                    templateImage.shape,
                )
                return None, None, None, None
            _sci_wcs_v = get_wcs(scienceHeader)
            _tpl_wcs_v = get_wcs(templateHeader)
            if _sci_wcs_v is None or _tpl_wcs_v is None:
                logger.error(
                    "Subtraction requires valid science/template WCS "
                    "(science=%s, template=%s).",
                    "ok" if _sci_wcs_v is not None else "missing",
                    "ok" if _tpl_wcs_v is not None else "missing",
                )
                return None, None, None, None
            _grid_tol = float(
                self.input_yaml.get("template_subtraction", {}).get(
                    "common_grid_tol_px", 0.5
                )
            )
            if not _wcs_on_common_grid(
                _sci_wcs_v, _tpl_wcs_v, scienceImage.shape, tol_px=_grid_tol
            ):
                logger.error(
                    "Science and template do not share a common pixel grid "
                    "(tolerance %.3f px); refusing to subtract.",
                    _grid_tol,
                )
                return None, None, None, None

            def _ensure_prepared_template_path() -> str:
                nonlocal prepared_template_fpath, template_work_fpath
                if prepared_template_fpath is None:
                    fd, tmp_path = tempfile.mkstemp(
                        prefix="template_prepared_",
                        suffix=".fits",
                        dir=str(scienceDir),
                    )
                    os.close(fd)
                    prepared_template_fpath = tmp_path
                    template_work_fpath = tmp_path
                    logger.info(
                        "Using temporary prepared template: %s",
                        os.path.basename(tmp_path),
                    )
                return template_work_fpath

            # Sky subtraction is backend-specific:
            #   - SFFT sparse flavor requires sky-subtracted inputs (Hu et al.
            #     2022 S3.2: masked regions become zero, so a sky pedestal
            #     would create step edges at mask boundaries).  Applied inside
            #     the SFFT branch below, to dedicated temp files, so a
            #     SFFT -> HOTPANTS fallback never mixes sky-subtracted and
            #     original inputs.
            #   - ZOGY subtracts its own sigma-clipped median internally.
            #   - HOTPANTS models the background itself (-bgo) and must see
            #     the original images.

            # Keep interpolation to the WCS reproject stage only.

            # Extract instrument scalars in a consistent way.
            # Prefer pipeline config (`input_yaml`) for science FWHM/gain/read-noise because
            # the header may not be updated after alignment/cropping steps. Fall back to
            # common FITS keys if needed.
            def _hdr_float(hdr: fits.Header, keys: list[str], default: float) -> float:
                for k in keys:
                    if k in hdr:
                        try:
                            v = float(hdr.get(k))
                            if np.isfinite(v):
                                return v
                        except Exception:
                            continue
                return float(default)

            # FWHMs must be in resampled-image pixels; SWarp resampling to a
            # common pixel scale can shift the pixel FWHM, so the template
            # FWHM is rescaled below.
            #
            # Prefer the header FWHM over input_yaml["fwhm"] for the science image.
            # The input_yaml FWHM comes from SExtractor and can be inflated by galaxy
            # contamination (e.g., 7.85px vs true 3.99px). The header FWHM comes from
            # the careful measure_image step (PSF fitting on point sources only).
            # Using the inflated FWHM causes kernel_order=0 (constant kernel) when the
            # true PSF difference is large (e.g., 70%), leading to flux scaling
            # mismatches and dipoles in the subtraction.
            _sci_hdr_fwhm = _hdr_float(scienceHeader, ["FWHM", "fwhm"], 0.0)
            if _sci_hdr_fwhm and _sci_hdr_fwhm > 0:
                science_fwhm_orig = _sci_hdr_fwhm
            else:
                science_fwhm_orig = float(self.input_yaml.get("fwhm", 3.0))
            template_fwhm_orig = float(_hdr_float(templateHeader, ["FWHM", "fwhm"], 3.0))

            # Get pixel scales from WCS to check if resampling changed the scale
            try:
                from astropy.wcs.utils import proj_plane_pixel_scales
                sci_wcs = get_wcs(scienceHeader)
                tpl_wcs = get_wcs(templateHeader)
                sci_pix_scale = float(proj_plane_pixel_scales(sci_wcs)[0] * 3600.0)  # arcsec/pixel
                tpl_pix_scale = float(proj_plane_pixel_scales(tpl_wcs)[0] * 3600.0)  # arcsec/pixel
                logger.info(
                    "Pixel scales for kernel sizing: science=%.4f, template=%.4f arcsec/px",
                    sci_pix_scale, tpl_pix_scale
                )
                # If pixel scales differ significantly, adjust FWHM to common scale
                # Assume science pixel scale is the reference (images resampled to science scale)
                # FWHM_sci_px = FWHM_tpl_px * tpl_pix_scale / sci_pix_scale
                # (angular FWHM is invariant: fwhm_px * pix_scale = constant).
                if tpl_pix_scale > 0 and sci_pix_scale > 0 and abs(tpl_pix_scale - sci_pix_scale) / sci_pix_scale > 0.01:
                    scale_factor = tpl_pix_scale / sci_pix_scale
                    template_fwhm = template_fwhm_orig * scale_factor
                    logger.info(
                        "Template FWHM adjusted for pixel scale: %.2f -> %.2f px (scale factor=%.3f)",
                        template_fwhm_orig, template_fwhm, scale_factor
                    )
                else:
                    template_fwhm = template_fwhm_orig
                science_fwhm = science_fwhm_orig
            except Exception as e:
                logger.debug("Could not get pixel scales for FWHM adjustment: %s", e)
                science_fwhm = science_fwhm_orig
                template_fwhm = template_fwhm_orig

            from aperture import gain_e_per_adu_from_header, resolve_gain_e_per_adu

            try:
                science_gain = resolve_gain_e_per_adu(None, self.input_yaml)
            except ValueError:
                try:
                    science_gain = gain_e_per_adu_from_header(
                        scienceHeader, None
                    )[0]
                except ValueError:
                    science_gain = 1.0
            try:
                template_gain = gain_e_per_adu_from_header(templateHeader, None)[0]
            except ValueError:
                template_gain = 1.0

            # Saturation: header may be missing or effectively "no saturation".
            # Treat missing / non-finite / non-positive values as "no hard limit"
            # by mapping them to np.inf. Downstream code interprets np.inf as
            # "do not mask on saturation".
            # Use consistent fallback values with main.py
            try:
                from main import SATURATE_INTERNAL_FALLBACK
                from functions import SATURATE_SENTINEL_MIN
            except ImportError:
                SATURATE_INTERNAL_FALLBACK = np.inf
                SATURATE_SENTINEL_MIN = 1e8

            def _safe_saturate(hdr: fits.Header) -> float:
                saturate_raw_value = None
                for k in ("SATURATE", "saturate", "SATLEVEL", "SATUR"):
                    if k in hdr:
                        saturate_raw_value = hdr.get(k)
                        break
                if saturate_raw_value is None:
                    saturate_raw_value = SATURATE_INTERNAL_FALLBACK
                try:
                    sat = float(saturate_raw_value)
                except Exception:
                    sat = SATURATE_INTERNAL_FALLBACK
                # Sentinel-scale placeholders (>= 1e8) are stored "no limit"
                # markers, not detector levels; map to inf like a missing key.
                if not np.isfinite(sat) or sat <= 0 or sat >= SATURATE_SENTINEL_MIN:
                    return SATURATE_INTERNAL_FALLBACK
                return sat

            science_saturate = _safe_saturate(scienceHeader)
            template_saturate = _safe_saturate(templateHeader)

            # NOTE: template background subtraction is disabled
            # (template_bg_median commented out), so no saturation
            # adjustment is needed here.

            # Optional: inpaint broken/saturated template cores (cosmetic).
            # Can reduce subtraction artefacts around very bright stars, but
            # does not recover lost flux.  Off by default; YAML-controlled.
            ts_cfg_inp = (
                (self.input_yaml.get("template_subtraction") or {})
                if isinstance(self.input_yaml, dict)
                else {}
            )
            # Saturation fraction for source filtering (same convention as PSF
            # building).  0.98 is higher than the photometry cut so that only
            # truly saturated cores are masked, not bright point sources.
            subtraction_saturate_frac = float(
                ts_cfg_inp.get("subtraction_saturate_fraction", 0.98)
            )
            _inpaint_mask = None
            if _as_bool(ts_cfg_inp.get("inpaint_template_cores", False), False) and np.isfinite(
                template_saturate
            ):
                try:
                    from utils.inpaint import InpaintConfig, inpaint_saturated_cores

                    cfg = InpaintConfig(
                        enabled=True,
                        method=str(ts_cfg_inp.get("inpaint_method", "biharmonic")),
                        saturate_frac=float(
                            ts_cfg_inp.get("inpaint_saturate_frac", 0.90)
                        ),
                        dilate_radius=int(ts_cfg_inp.get("inpaint_dilate_radius", 6)),
                        max_mask_fraction=float(
                            ts_cfg_inp.get("inpaint_max_mask_fraction", 0.01)
                        ),
                    )
                    before = templateImage
                    templateImage, _mask_used = inpaint_saturated_cores(
                        templateImage,
                        saturate=float(template_saturate),
                        cfg=cfg,
                    )
                    if np.any(_mask_used):
                        # The filled pixels are synthetic smooth values, not
                        # measurements - keep them finite for the subtraction
                        # engines but record the footprint so it can join the
                        # template exclusion mask below.
                        _inpaint_mask = np.asarray(_mask_used, dtype=bool)
                        write_fits(
                            _ensure_prepared_template_path(),
                            templateImage,
                            templateHeader,
                        )
                        logger.info(
                            "Inpainted template saturated cores: %.3f%% pixels (method=%s, dilate=%d px).",
                            float(np.mean(_mask_used)) * 100.0,
                            cfg.method,
                            cfg.dilate_radius,
                        )
                except Exception as _e:
                    logger.info(
                        "Template inpainting skipped/failed (non-fatal): %s", _e
                    )
            # Read noise: prefer pipeline config (science) and common FITS keys (template).
            # main.py writes RDNOISE; some surveys use READNOISE.
            science_readnoise = float(
                self.input_yaml.get(
                    "read_noise",
                    _hdr_float(scienceHeader, ["RDNOISE", "READNOISE", "rdnoise", "readnoise"], 1.0),
                )
            )
            template_readnoise = float(
                _hdr_float(templateHeader, ["RDNOISE", "READNOISE", "rdnoise", "readnoise"], 1.0)
            )

            # Compute the subtraction kernel half-width from the PSF FWHMs of both images.
            #
            # The convolution kernel that transforms the sharper PSF into the broader one has a
            # FWHM of sqrt(FWHM_broad^2 - FWHM_narrow^2).  The kernel stamp must be large enough
            # to contain this kernel, so kernel_hw >= ceil(multiplier * FWHM_conv).
            # We also enforce a floor of ceil(FWHM_broad) so the kernel always covers at least
            # one full PSF footprint, which is required for flux-conserving photometry.
            #
            # The multiplier is configurable and can be overridden entirely via
            # kernel_hw_override for debugging. For undersampled images (FWHM < undersampled_fwhm_threshold), the
            # multiplier is automatically increased to capture extended PSF wings.
            ts_cfg_ker = self.input_yaml.get("template_subtraction", {})
            # YAML defines sfft_kernel_hw_min/max; the unprefixed names are
            # kept only as a backward-compatible fallback.
            KER_HW_MIN = int(
                ts_cfg_ker.get(
                    "sfft_kernel_hw_min", ts_cfg_ker.get("kernel_hw_min", 3)
                )
            )
            KER_HW_MAX = int(
                ts_cfg_ker.get(
                    "sfft_kernel_hw_max", ts_cfg_ker.get("kernel_hw_max", 50)
                )
            )
            fwhm_ref = float(template_fwhm)
            fwhm_sci = float(science_fwhm)
            fwhm_broad = max(fwhm_ref, fwhm_sci)
            fwhm_narrow = min(fwhm_ref, fwhm_sci)

            # BUG 120: Estimate effective source count early for kernel floor
            # adaptation.  SFFT self-matches when pipeline provides too few
            # sources.  SFFT's SExtractor typically finds 25-35 sources, but
            # after cross-matching and quality filtering only ~15-20 survive
            # for kernel fitting.
            #
            # When the pipeline provides too few priors, SFFT will self-match
            # using its own SExtractor detection.  Using n_eff=10 (the old
            # default) when SFFT actually has 20+ sources causes:
            #   - kernel_order=0 (too rigid for large PSF differences)
            #   - floor_mult=1.5 (too small to contain PSF wings)
            #   - source-count cap too aggressive (shrinks kernel below PSF)
            #
            # Use 20 as the self-match estimate, which matches the typical
            # post-filtering source count.  This prevents under-sizing the
            # kernel in the common case where SFFT self-matches successfully.
            _n_matched_early = len(matching_sources) if matching_sources else 0
            _min_prior_early = int(ts_cfg_ker.get("sfft_min_prior_sources", 3) or 3)
            _sfft_self_match_early = _n_matched_early < _min_prior_early
            n_eff = 20 if _sfft_self_match_early else _n_matched_early

            # Override: user directly specifies kernel half-width in pixels
            _ker_hw_override = ts_cfg_ker.get("kernel_hw_override", None)
            if _ker_hw_override is not None:
                try:
                    ker_hw = int(_ker_hw_override)
                    logger.info(
                        "Kernel half-width overridden by config: %d px (kernel_hw_override)",
                        ker_hw,
                    )
                except (TypeError, ValueError):
                    _ker_hw_override = None

            if _ker_hw_override is None:
                # User multiplier; fall back to 2.5 on missing/bad input.
                try:
                    _mult = float(ts_cfg_ker.get("kernel_hw_fwhm_multiplier") or 2.5)
                    if not np.isfinite(_mult) or _mult <= 0:
                        _mult = 2.5
                    # Clamp to [1, 5] so a config typo cannot explode the kernel.
                    _mult = max(1.0, min(_mult, 5.0))
                except (TypeError, ValueError):
                    _mult = 2.5

                # FWHM of the convolution kernel (quadrature difference)
                if fwhm_broad > fwhm_narrow and fwhm_broad > 0:
                    fwhm_conv = np.sqrt(max(fwhm_broad ** 2 - fwhm_narrow ** 2, 0.0))
                else:
                    fwhm_conv = fwhm_broad  # identical PSFs: kernel ~ delta function; use broad as floor

                # Adaptive multiplier: boost for undersampled images to capture PSF wings
                # Uses configurable undersampled_fwhm_threshold (default 2.5 px) for consistency
                _us_thr_ker = float(
                    (self.input_yaml.get("photometry", {}) or {}).get(
                        "undersampled_fwhm_threshold", 2.5
                    )
                )
                _mult_effective = _mult
                if fwhm_broad < _us_thr_ker:
                    _mult_effective = min(_mult * 1.5, 4.0)  # cap boost at 4.0 total
                    logger.debug(
                        "Undersampled PSF detected (FWHM=%.2f < %.1f px); "
                        "boosting kernel multiplier: %.2f -> %.2f",
                        fwhm_broad, _us_thr_ker, _mult, _mult_effective,
                    )

                # A very large PSF difference usually means bad PSF measurement.
                if fwhm_conv > 10.0:
                    logger.warning(
                        "Large PSF FWHM difference detected (%.1f px). "
                        "Subtraction quality may be degraded. "
                        "Consider using better-matched templates.",
                        fwhm_conv,
                    )

                # --- Kernel half-width sizing (ringing-prevention) ---
                # Ringing artifacts in the difference image are caused by kernel
                # overfitting: the delta-function basis has (2*hw+1)^2 free
                # parameters per spatial order, and when the kernel is too large
                # relative to the number of sources, the outer pixels fit noise
                # instead of the PSF difference.
                #
                # The convolution kernel FWHM is sqrt(FWHM_broad^2 - FWHM_narrow^2).
                # The kernel half-width only needs to contain this convolution PSF,
                # NOT the full broad PSF.  Using 2.5*FWHM_broad as a floor (as was
                # done previously) inflates the kernel massively when PSFs differ:
                #   FWHM_sci=12, FWHM_ref=6 -> floor=30px (3721 params for 25 sources!)
                #   but FWHM_conv=10.4, so 2*FWHM_conv=21px would suffice.
                #
                # New strategy:
                #   1. Primary: hw = ceil(mult * FWHM_conv)  -- contains the kernel
                #   2. Floor: hw >= ceil(floor_mult * FWHM_broad) -- captures PSF wings
                #      but floor_mult is adaptive: 1.5 for sparse, 2.0 for dense
                #   3. Source-count cap: hw <= max_hw_for_sources(n_eff)
                #      ensures the kernel is not under-constrained
                #
                # The source-count cap prevents overfitting.  With the delta-function
                # basis, each source provides ~(2*hw+1)^2 pixels of data, but the
                # effective number of independent constraints is much less (PSF is
                # smooth, outer pixels are noise-dominated).  We require at least
                # ~5 sources per 100 kernel pixels as a heuristic.
                ker_hw_from_conv = int(np.ceil(_mult_effective * fwhm_conv))

                # Adaptive floor multiplier based on source count.
                # Fewer sources -> smaller floor to prevent overfitting.
                _floor_mult = 2.0
                if n_eff < 15:
                    _floor_mult = 1.5
                elif n_eff < 30:
                    _floor_mult = 1.75
                ker_hw_floor = int(np.ceil(_floor_mult * fwhm_broad))

                ker_hw = max(ker_hw_from_conv, ker_hw_floor)

                # Source-count cap: prevent kernel from being severely
                # under-constrained.  With the delta-function basis, each source
                # provides ~(2*hw+1)^2 pixels, so the constraint ratio is
                # ~n_eff / spatial_terms.  The cap only trims the FLOOR (PSF wing
                # coverage), never the convolution term (which is the physical
                # minimum needed to contain the PSF difference).
                # Heuristic: (2*hw+1)^2 should not exceed 100 * n_eff.
                # For n_eff=25: max kernel pixels = 2500 -> hw <= 24
                # For n_eff=50: max kernel pixels = 5000 -> hw <= 35
                _max_hw_pixels = 100 * max(n_eff, 5)
                _max_hw_from_sources = int((np.sqrt(_max_hw_pixels) - 1) / 2)
                # Enforce a PHYSICAL MINIMUM: the kernel must contain at least
                # ceil(1.5 * fwhm_broad) to enclose the broader PSF's core.
                # The source-count cap can only reduce the floor down to this
                # minimum, never below it.  A kernel smaller than the PSF
                # itself cannot model the PSF difference, causing dipoles and
                # flux scaling mismatch regardless of how few sources are
                # available.
                _phys_min_hw = int(np.ceil(1.5 * fwhm_broad))
                _hw_floor_capped = max(
                    _phys_min_hw,
                    min(ker_hw_floor, _max_hw_from_sources),
                )
                ker_hw = max(ker_hw_from_conv, _hw_floor_capped)

                # For sparse fields, also cap the convolution term.
                # A large under-constrained kernel (e.g., 55x55=3025 params
                # for 8 sources = 378 params/source) fits noise rather than
                # the PSF, producing worse residuals than a smaller kernel
                # that under-fits the PSF but is better constrained.
                # The physical minimum is a hard constraint -- never go below
                # it regardless of source count.
                _sparse_cap_thresh = int(
                    ts_cfg_ker.get("sfft_sparse_field_threshold", 10) or 10
                )
                if n_eff < _sparse_cap_thresh:
                    _ker_hw_before_sparse_cap = ker_hw
                    ker_hw = max(
                        _phys_min_hw,
                        min(ker_hw, _max_hw_from_sources),
                    )
                    if ker_hw != _ker_hw_before_sparse_cap:
                        logger.info(
                            "Kernel sizing: sparse field (%d sources < %d) -- "
                            "capping kernel_hw at %d px (was %d, max_hw_sources=%d) "
                            "to prevent under-constrained kernel fit.",
                            n_eff, _sparse_cap_thresh,
                            ker_hw, _ker_hw_before_sparse_cap, _max_hw_from_sources,
                        )

                ker_hw = max(KER_HW_MIN, min(KER_HW_MAX, ker_hw))

                # Alignment-quality-aware kernel boost.
                # When the aligned template header carries an alignment RMS
                # (ALIGRMS keyword from spalipy/SWarp), poor alignment means
                # the kernel must compensate for residual astrometric offsets.
                # A kernel that is too small will produce dipole residuals at
                # source positions because it cannot model the spatially-
                # varying flux mismatch caused by sub-pixel misregistration.
                #
                # The boost scales with the ratio of alignment RMS to FWHM:
                # a 1 px offset on a 3 px FWHM image is far more damaging than
                # on a 8 px FWHM image.  We boost by ceil(rms / fwhm_broad *
                # fwhm_broad) = ceil(rms) as a minimum, but scale up when the
                # RMS is a significant fraction of the PSF.  The maximum boost
                # is capped at ceil(0.5 * fwhm_broad) to avoid excessive kernels.
                #
                # Additionally, when per-quadrant RMS is available (ALIGQMAX),
                # spatially-varying alignment errors require an extra boost
                # because the global RMS underestimates the worst-region error.
                _align_rms_boost = 0
                try:
                    _aligrms = float(templateHeader.get("ALIGRMS", 0.0))
                    _aligqmax = float(templateHeader.get("ALIGQMAX", 0.0))
                    # Use the larger of global RMS and max-quadrant RMS.
                    _align_rms_eff = max(_aligrms, _aligqmax)
                    if np.isfinite(_align_rms_eff) and _align_rms_eff > 0.3:
                        # Scale boost with RMS relative to FWHM.
                        # A sub-pixel RMS (< 1 px) still gets at least 1 px boost
                        # because even sub-pixel misregistration causes dipoles.
                        # Cap at ceil(0.5 * fwhm_broad) to avoid excessive kernels.
                        _max_boost = int(np.ceil(0.5 * fwhm_broad))
                        _align_rms_boost = max(
                            1,
                            int(np.ceil(_align_rms_eff)),
                        )
                        _align_rms_boost = min(_align_rms_boost, _max_boost)
                        # Extra boost when spatial variation is significant
                        # (quadrant max >> global RMS means non-uniform errors).
                        if (
                            np.isfinite(_aligqmax)
                            and np.isfinite(_aligrms)
                            and _aligqmax > 0
                            and _aligrms > 0
                            and _aligqmax / _aligrms > 1.5
                        ):
                            _align_rms_boost = min(_align_rms_boost + 1, _max_boost + 1)
                            logger.info(
                                "Kernel sizing: spatial alignment variation detected "
                                "(ALIGQMAX=%.3f >> ALIGRMS=%.3f, ratio=%.2f) -> "
                                "extra +1 px kernel boost.",
                                _aligqmax, _aligrms, _aligqmax / _aligrms,
                            )
                        _ker_hw_before_boost = ker_hw
                        ker_hw = max(KER_HW_MIN, min(KER_HW_MAX, ker_hw + _align_rms_boost))
                        if ker_hw != _ker_hw_before_boost:
                            logger.info(
                                "Kernel sizing: alignment RMS=%.3f px (qmax=%.3f) -> "
                                "boosting kernel_hw by %d px (%d -> %d) to compensate "
                                "for residual astrometric offsets.",
                                _aligrms, _aligqmax, _align_rms_boost,
                                _ker_hw_before_boost, ker_hw,
                            )
                except Exception:
                    pass

                logger.info(
                    "Kernel sizing: FWHM sci=%.2f ref=%.2f conv=%.2f px -> "
                    "kernel_hw=%d px (hw_conv=%d, floor=%d, n_eff=%d)",
                    fwhm_sci, fwhm_ref, fwhm_conv,
                    ker_hw, ker_hw_from_conv, ker_hw_floor, n_eff,
                )
                logger.debug(
                    "Kernel sizing detail: floor_mult=%.2f capped=%d clamp=[%d,%d] "
                    "max_hw_sources=%d align_boost=%d",
                    _floor_mult, _hw_floor_capped, KER_HW_MIN, KER_HW_MAX,
                    _max_hw_from_sources, _align_rms_boost,
                )

            # Both SFFT and HOTPANTS use the same physics-based kernel half-width
            # (ker_hw). The caller's `scale` (pipeline source-detection cutout
            # size, typically 4*FWHM) is NOT used for kernel sizing -- it was
            # incorrectly used for HOTPANTS in the past (BUG 136), causing
            # oversized kernels and under-constrained fits in sparse fields.
            sfft_kernel_hw = max(ker_hw, 5)
            # Always use the physics-based kernel half-width.  The caller's
            # `scale` is the pipeline source-detection cutout size (~5*FWHM),
            # which is larger than required and caused oversized, under-
            # constrained kernels in sparse fields (BUG 136).
            scale = sfft_kernel_hw
            # Store the actual kernel half-width for the return value.
            # This is used by plot.py to draw the kernel-size overlay.
            kernel_half_width = sfft_kernel_hw
            target_location = [
                (self.input_yaml["target_x_pix"], self.input_yaml["target_y_pix"])
            ]

            # =============================================================
            # 2. Locate PSF models
            # =============================================================
            # Gridded-PSF runs place model FITS under PSF_MODELS/; plain
            # runs write them beside the other outputs.  Per-cell stamps
            # (``_cellxIyJ``) are not field models -- exclude them so the
            # mtime fallback cannot pick one.
            def _field_psf_globs(root, prefix):
                files = glob.glob(str(root / f"{prefix}*fits")) + glob.glob(
                    str(root / "PSF_MODELS" / f"{prefix}*fits")
                )
                return [
                    f for f in files
                    if "_cellx" not in os.path.basename(f)
                ]

            science_psf_files = _field_psf_globs(scienceDir, "PSF_model_image")
            template_psf_files = _field_psf_globs(
                scienceDir, "PSF_model_template"
            )
            science_psf = science_psf or _pick_psf_file(
                science_psf_files, scienceFpath
            )
            template_psf = template_psf or _pick_psf_file(
                template_psf_files, templateFpath
            )

            # =============================================================
            # 3. Build masks
            # =============================================================
            logger.debug("Building science and template masks...")

            # NaN / sentinel masks
            # NOTE: the original (abs(x) < 1.1e-20) & (x != 0) condition was
            # logically impossible - any float that close to zero IS 0.0 in
            # IEEE 754 and will never satisfy != 0.  The guard has been removed
            # so that true near-zero sentinel pixels (written by SWarp/SFFT in
            # no-coverage regions) are correctly marked as invalid.
            template_mask_nans = (
                (np.abs(templateImage) < 1.1e-20)
                | (~np.isfinite(templateImage))
            ).astype(np.int32)

            science_mask_nans = (
                (np.abs(scienceImage) < 1.1e-20)
                | (~np.isfinite(scienceImage))
            ).astype(np.int32)

            # Template mask is deliberately shallow: large sources, catalog
            # overlaps, and generous padding would remove pixels needed to
            # calibrate the kernel.
            template_seg_mask, template_seg_centers = self.create_image_mask(
                templateImage,
                sat_lvl=template_saturate,
                saturate_frac=subtraction_saturate_frac,
                fwhm=template_fwhm,
                create_source_mask=False,
                ignore_position=target_location,
                remove_large_sources=False,
                mask_bright_catalog_overlaps=False,
                padding=int(2 * template_fwhm),  # smaller than science padding
            )
            # Synthetic inpainted pixels must not feed kernel-stamp selection
            # or survive as real diff flux - the seg mask propagates them to
            # universal_mask/tpl_mask_hp without touching the image values.
            if (
                _inpaint_mask is not None
                and _inpaint_mask.shape == template_seg_mask.shape
            ):
                template_seg_mask = (
                    template_seg_mask.astype(np.int32)
                    | _inpaint_mask.astype(np.int32)
                ).astype(bool)
            # Science mask: saturated/negative sources only; large sources and
            # catalog overlaps are left alone so point sources stay available.
            science_seg_mask, science_seg_centers = self.create_image_mask(
                scienceImage,
                sat_lvl=science_saturate,
                saturate_frac=subtraction_saturate_frac,
                fwhm=science_fwhm,
                create_source_mask=False,
                ignore_position=target_location,
                remove_large_sources=False,
                mask_bright_catalog_overlaps=False,
                padding=int(DEFAULT_FWHM_PADDING_MULTIPLIER * science_fwhm),
            )

            # Combined mask: NaN/invalid (essential) + segmentation-based source masks + background defects
            mask_essential = np.clip(
                science_mask_nans + template_mask_nans,
                0,
                1,
            ).astype(bool)
            mask_sources = np.clip(
                science_seg_mask.astype(np.int32) + template_seg_mask.astype(np.int32),
                0,
                1,
            ).astype(bool)

            # Include background defects mask if provided (saturation streaks, satellite trails, etc.)
            if background_defects_mask is not None:
                background_defects_mask = background_defects_mask.astype(bool)
                if background_defects_mask.shape != scienceImage.shape:
                    # The mask lives on a different pixel grid than the
                    # aligned science image.  Cropping by array origin would
                    # silently misplace defects (alignment may shift, crop,
                    # or resample the frame), so discard it instead: a wrong
                    # mask is worse than no mask.
                    logger.warning(
                        "Background defects mask shape %s != aligned science "
                        "shape %s; discarding rather than applying an "
                        "unregistered mask.",
                        background_defects_mask.shape, scienceImage.shape,
                    )
                    background_defects_mask = None
            if background_defects_mask is not None:
                # A compact defect component sitting exactly on a known
                # target is a suspect flag, not a data hole (e.g. an
                # isolated point source read as a cosmic-ray cluster whose
                # flag blob lands in the defects mask).  Drop only small
                # components whose target pixel is finite in the science
                # image; real holes (NaN/sentinel) stay masked.
                _defect_rescue_xy = []
                for _pos in target_location or []:
                    try:
                        _px, _py = float(_pos[0]), float(_pos[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if np.isfinite(_px) and np.isfinite(_py):
                        _defect_rescue_xy.append((_px, _py))
                _defect_sci_wcs = get_wcs(scienceHeader)
                if _defect_sci_wcs is not None:
                    for _at in (
                        self.input_yaml.get("_additional_targets_resolved")
                        or []
                    ):
                        try:
                            _ax, _ay = _defect_sci_wcs.all_world2pix(
                                float(_at["ra"]), float(_at["dec"]), 0
                            )
                            if np.isfinite(_ax) and np.isfinite(_ay):
                                _defect_rescue_xy.append(
                                    (float(_ax), float(_ay))
                                )
                        except Exception:
                            continue
                if _defect_rescue_xy:
                    from functions import drop_mask_components_at_positions

                    # Both the compactness extent and the area cap scale
                    # with the science FWHM: a false-flag footprint grows
                    # with the PSF, while a real elongated defect (column,
                    # trail, streak) crossing the position is kept by the
                    # extent check.
                    _cr_cfg = self.input_yaml.get("cosmic_rays", {})
                    try:
                        _rescue_fwhm = float(science_fwhm)
                    except (TypeError, ValueError):
                        _rescue_fwhm = np.nan
                    if np.isfinite(_rescue_fwhm) and _rescue_fwhm > 0:
                        _defect_rescue_extent = (
                            float(
                                _cr_cfg.get(
                                    "cr_defect_target_extent_fwhm", 2.5
                                )
                            )
                            * _rescue_fwhm
                            + 2.0
                        )
                        _defect_rescue_max = max(
                            int(
                                _cr_cfg.get(
                                    "cr_defect_target_component_max_px", 150
                                )
                            ),
                            int(
                                np.ceil(np.pi * _defect_rescue_extent**2)
                            ),
                        )
                    else:
                        _defect_rescue_extent = None
                        _defect_rescue_max = int(
                            _cr_cfg.get(
                                "cr_defect_target_component_max_px", 150
                            )
                        )
                    background_defects_mask, _dropped_defect = (
                        drop_mask_components_at_positions(
                            background_defects_mask,
                            _defect_rescue_xy,
                            max_area_px=_defect_rescue_max,
                            max_extent_px=_defect_rescue_extent,
                            reference_image=scienceImage,
                        )
                    )
                    if _dropped_defect:
                        logger.warning(
                            "Dropped %d small defect component(s) covering a "
                            "target position - suspect flag, not a data hole; "
                            "those pixels stay usable for subtraction and "
                            "photometry.",
                            len(_dropped_defect),
                        )
                mask_essential = mask_essential | background_defects_mask
                logger.info(
                    "Included background defects mask (saturation streaks, satellite trails) in universal mask."
                )

            # universal_mask_full keeps all layers for reference/logging
            universal_mask_full = (mask_essential | mask_sources).astype(np.int32)

            # universal_mask passed to SFFT: NaN/invalid + background defects +
            # saturated/negative source footprints (from mask_sources).
            # Unsaturated point-source masks are excluded to preserve flux
            # calibration stars for kernel fitting.
            universal_mask = (mask_essential | mask_sources).astype(np.int32)
            logger.info(
                "Universal mask for subtraction: %.1f%% masked "
                "(NaN/defects=%.1f%% + sources=%.1f%%).",
                np.mean(universal_mask) * 100.0,
                np.mean(mask_essential) * 100.0,
                np.mean(mask_sources) * 100.0,
            )

            visualization_mask = universal_mask.copy()

            # Save mask to scienceDir (not templateDir) to prevent crosstalk
            # when multiple science images use the same template
            mask_loc = os.path.join(scienceDir, f"universal_mask_{base_name}")
            save_to_fits(universal_mask.astype(int), mask_loc)

            masked_centers = deduplicate_points(
                science_seg_centers + template_seg_centers,
                min_sep=5.0,
            )

            masked_percentage = np.sum(universal_mask) / universal_mask.size * 100
            logger.debug("Masked %.3f%% of pixels before subtraction", masked_percentage)

            # --- Filter matching sources against the universal mask ---
            # The mask only suppresses these pixels in the subtracted
            # output; every backend can still fit kernels/flux scales on
            # saturated or defect-contaminated stamps unless the position
            # list itself is filtered.  Applies to SFFT priors, HOTPANTS
            # stamps, and ZOGY scale/star-PSF positions alike.
            if (
                matching_sources
                and int(np.sum(universal_mask)) > 0
                and np.asarray(universal_mask).shape == scienceImage.shape
            ):
                _um = np.asarray(universal_mask, dtype=bool)
                _ny_m, _nx_m = _um.shape
                _stamp_r = max(
                    int(sfft_kernel_hw),
                    int(2 * max(science_fwhm, template_fwhm)),
                )
                _ts_cfg_m = self.input_yaml.get("template_subtraction", {}) or {}
                _max_frac_m = float(
                    _ts_cfg_m.get("sfft_defects_max_frac", 0.15)
                )
                _min_keep_m = int(_ts_cfg_m.get("sfft_min_prior_sources", 3))
                # Integral image gives O(1) per-source stamp means.
                _integral_m = np.zeros(
                    (_ny_m + 1, _nx_m + 1), dtype=np.float64
                )
                np.cumsum(
                    np.cumsum(_um.astype(np.float64), axis=0),
                    axis=1,
                    out=_integral_m[1:, 1:],
                )
                _kept_m = []
                _n_rej_m = 0
                for _ms in matching_sources:
                    try:
                        _mxy = (_ms["x"], _ms["y"])
                    except (TypeError, IndexError, KeyError):
                        _mxy = _ms
                    _mxi = int(round(float(_mxy[0])))
                    _myi = int(round(float(_mxy[1])))
                    _y0m = max(0, _myi - _stamp_r)
                    _y1m = min(_ny_m, _myi + _stamp_r + 1)
                    _x0m = max(0, _mxi - _stamp_r)
                    _x1m = min(_nx_m, _mxi + _stamp_r + 1)
                    _area_m = (_y1m - _y0m) * (_x1m - _x0m)
                    if _area_m <= 0:
                        _n_rej_m += 1
                        continue
                    _frac_m = float(
                        _integral_m[_y1m, _x1m]
                        - _integral_m[_y0m, _x1m]
                        - _integral_m[_y1m, _x0m]
                        + _integral_m[_y0m, _x0m]
                    ) / _area_m
                    if _frac_m > _max_frac_m:
                        _n_rej_m += 1
                    else:
                        _kept_m.append(_ms)
                if _n_rej_m > 0 and len(_kept_m) >= _min_keep_m:
                    logger.info(
                        "Universal-mask filtering: rejected %d/%d matching "
                        "sources on defects (stamp_radius=%d px, "
                        "max_frac=%.2f, kept=%d).",
                        _n_rej_m,
                        len(matching_sources),
                        _stamp_r,
                        _max_frac_m,
                        len(_kept_m),
                    )
                    matching_sources = _kept_m
                elif _n_rej_m > 0:
                    logger.warning(
                        "Universal-mask filtering would leave %d sources "
                        "(< %d minimum); keeping all %d sources.",
                        len(_kept_m),
                        _min_keep_m,
                        len(matching_sources),
                    )

            # =============================================================
            # 4. Background statistics on unmasked pixels
            # =============================================================
            scienceMean, scienceMedian, scienceSTD = get_image_stats(
                scienceImage[~universal_mask.astype(bool)]
            )
            templateMean, templateMedian, templateSTD = get_image_stats(
                templateImage[~universal_mask.astype(bool)]
            )

            # 4a. Kernel order selection: use user value if provided, else
            #     derive a sensible default from the PSF FWHM ratio.
            ts_cfg = self.input_yaml.get("template_subtraction", {})
            if "template_subtraction" not in self.input_yaml:
                self.input_yaml["template_subtraction"] = ts_cfg

            # If the global photometry config has flagged the image as crowded,
            # propagate that information into the template_subtraction block so
            # that SFFT and related options default to crowded-safe behaviour.
            phot_cfg = self.input_yaml.get("photometry", {})
            if phot_cfg.get("crowded_field", False):
                if "crowded_field" not in ts_cfg:
                    ts_cfg["crowded_field"] = True
                    logger.info(
                        "photometry.crowded_field=True; enabling crowded mode for template_subtraction/SFFT."
                    )
                # Unless the user has explicitly chosen otherwise, also enable
                # the SFFT crowded kernel (ECP) for crowded fields.
                if "sfft_crowded_method" not in ts_cfg:
                    ts_cfg["sfft_crowded_method"] = True

            # Automatically switch to crowded mode from source density unless sfft_crowded_auto is False
            run_crowded_auto = ts_cfg.get("sfft_crowded_auto", False)
            if run_crowded_auto:
                ny, nx = scienceImage.shape[0], scienceImage.shape[1]
                try:
                    sci_wcs = get_wcs(scienceHeader)
                    from astropy.wcs.utils import proj_plane_pixel_scales
                    pixel_scale_arcsec = float(
                        proj_plane_pixel_scales(sci_wcs)[0] * 3600.0
                    )
                except Exception:
                    pixel_scale_arcsec = 0.3
                area_sq_arcmin = (ny * nx) * (pixel_scale_arcsec / 60.0) ** 2
                n_src = len(matching_sources)
                density = n_src / area_sq_arcmin if area_sq_arcmin > 0 else 0.0
                min_sources = ts_cfg.get("sfft_crowded_min_sources", 300)
                min_density = ts_cfg.get("sfft_crowded_min_density", 1.5)
                is_crowded = n_src >= min_sources or density >= min_density
                # Default to sparse (ESP): it performs better on typical
                # fields.  Users can set sfft_crowded_method: true for dense
                # fields.
                if "sfft_crowded_method" not in ts_cfg:
                    ts_cfg["sfft_crowded_method"] = False
                if is_crowded:
                    logger.info(
                        "Image detected as crowded (%d sources, %.2f per sq arcmin) -> using SFFT crowded (ECP)",
                        n_src,
                        density,
                    )
                else:
                    logger.info(
                        "Image sparse (%d sources, %.2f per sq arcmin) -> using SFFT sparse (ESP)",
                        n_src,
                        density,
                    )
            elif "sfft_crowded_method" not in ts_cfg and "crowded_field" not in ts_cfg:
                ts_cfg["sfft_crowded_method"] = (
                    False  # default to ESP (sparse) for better performance on typical fields
                )
            # Kernel polynomial order controls spatial variation of the solved
            # kernel, not the size of the PSF correction itself.  Source count
            # limits what can be fitted, while PSF mismatch and alignment
            # quality decide whether extra spatial terms are needed.
            #
            # RAM scaling (SFFT linear system):
            #   order 0:  1 term  ->  manageable
            #   order 1:  3 terms ->  moderate
            #   order 2:  6 terms ->  ~7 GB (auto caps at 2)
            #   order 3: 10 terms ->  ~20 GB (user must set explicitly)
            # null/None defaults to 0 (constant kernel, minimal RAM).
            _raw_kernel = ts_cfg.get("kernel_order", 0)
            _is_auto = isinstance(_raw_kernel, str) and _raw_kernel.strip().lower() == "auto"
            # null/None -> 0 (constant).  Only "auto" string triggers auto-select.
            # Numeric strings like "0", "1", "2" are valid user overrides.
            if _is_auto:
                user_kernel = None
            elif isinstance(_raw_kernel, str):
                try:
                    user_kernel = int(_raw_kernel.strip())
                except ValueError:
                    user_kernel = None
            elif _raw_kernel is not None:
                user_kernel = _raw_kernel
            else:
                user_kernel = None
            n_matched = len(matching_sources) if matching_sources else 0

            # n_eff was computed earlier for kernel floor adaptation (BUG 120).
            # It uses 20 for self-match cases (SFFT typically finds 20+ sources
            # via its own SExtractor), consistent with the kernel floor fix.
            # Recompute _sfft_self_match for logging purposes.
            _min_prior = int(ts_cfg.get("sfft_min_prior_sources", 3) or 3)
            _sfft_self_match = n_matched < _min_prior

            if user_kernel is not None and user_kernel >= 0:
                # User override: keep it, but warn when under-constrained.
                kernel_order = min(int(user_kernel), 3)
                min_src_for_order = {0: 5, 1: 20, 2: 50, 3: 80}
                needed = min_src_for_order.get(kernel_order, 0)
                if n_eff < needed:
                    logger.warning(
                        "User kernel_order=%d may be under-constrained (only %d matched sources, recommend %d). Consider lowering kernel_order.",
                        kernel_order, n_eff, needed,
                    )
                if kernel_order >= 2:
                    logger.warning(
                        "User kernel_order=%d: orders >=2 use significant RAM "
                        "(~7 GB for order 2, ~20 GB for order 3 with typical "
                        "kernel sizes). Ensure sufficient memory is available.",
                        kernel_order,
                    )
                # Auto-boost: when user sets kernel_order=0 but the PSF
                # difference is large, a constant kernel cannot model the
                # spatially-varying kernel shape.  Boost to 1 with a warning.
                # This prevents dipole residuals and flux scaling mismatches
                # that occur when a constant kernel is used for a large PSF
                # difference (e.g., FWHM_sci=13 vs FWHM_ref=5).
                _rel_diff_check = abs(science_fwhm - template_fwhm) / max(
                    (science_fwhm + template_fwhm) / 2, 0.1
                )
                if (
                    kernel_order == 0
                    and _rel_diff_check > 0.4
                    and n_eff >= 30
                ):
                    kernel_order = 1
                    logger.warning(
                        "Auto-boosting kernel_order from 0 to 1:\n"
                        "    large PSF rel_diff=%.2f (FWHM_sci=%.1f,\n"
                        "    FWHM_ref=%.1f) with %d matched sources requires\n"
                        "    a spatially-varying kernel. A constant kernel\n"
                        "    (order 0) causes flux scaling mismatches and\n"
                        "    dipole residuals. Set kernel_order: \"auto\" in\n"
                        "    YAML to suppress this boost.",
                        _rel_diff_check, science_fwhm, template_fwhm, n_eff,
                    )
            else:
                # The polynomial order controls how the kernel varies SPATIALLY
                # across the field of view - NOT how the PSF difference is
                # modelled.  The DFT kernel itself handles the PSF shape
                # difference at each position.  The polynomial just determines
                # how many independent spatial terms are used.
                #
                # Order 0 (constant kernel) works well for small fields where
                # the kernel is uniform across the image.  Order 1 (linear)
                # allows the kernel to vary linearly across the field.
                #
                # RAM CONSTRAINT: SFFT's linear system scales as
                #   (n_poly_terms x kernel_pixels)^2
                # With KerHW=35 (kernel 71x71 = 5041 px):
                #   order 0:  1 term  ->  ~5K unknowns  ->  manageable
                #   order 1:  3 terms ->  ~15K unknowns ->  moderate
                #   order 2:  6 terms ->  ~30K unknowns ->  ~7 GB matrix
                #   order 3: 10 terms ->  ~50K unknowns ->  ~20 GB matrix
                # Order 3+ can exhaust RAM on typical machines.  Auto-selection
                # is capped at order 2 for small kernels (KerHW <= 25), order 1
                # otherwise.  Users can override with kernel_order in YAML.
                rel_diff = abs(science_fwhm - template_fwhm) / max((science_fwhm + template_fwhm) / 2, 0.1)

                # BUG 121: With KerHW=20, order 2 needs only ~812 MB
                # (6x1681=10086 unknowns).  Allow order 2 for small kernels
                # to model spatially-varying astrometric residuals.
                #
                # The cap must reflect the kernel SFFT will actually use.
                # When sfft_auto_kernel_size lets run_sfft.py size the kernel
                # (non-sparse fields), the auto size is
                # max(ceil(mult_eff*FWHM_broad), ceil(3*FWHM_conv/2.355)) and
                # ignores our source-count cap, so it can exceed ker_hw.
                _ker_hw_for_cap = ker_hw
                if (
                    _as_bool(
                        ts_cfg_ker.get("sfft_auto_kernel_size", True), True
                    )
                    and _ker_hw_override is None
                    and _n_matched_early
                    >= int(
                        ts_cfg_ker.get("sfft_sparse_field_threshold", 10) or 10
                    )
                ):
                    try:
                        _est_auto_hw = max(
                            int(np.ceil(_mult_effective * fwhm_broad)),
                            int(np.ceil(3.0 * fwhm_conv / 2.3548200450309493)),
                        )
                        _ker_hw_for_cap = max(ker_hw, _est_auto_hw)
                    except Exception:
                        pass
                _max_auto_order = 2 if _ker_hw_for_cap <= 25 else 1

                # Source count sets how much spatial freedom is feasible, but
                # it is not evidence that the kernel varies across the field.
                # Higher orders fit edge noise and masked-source systematics,
                # so auto starts constant and adds spatial terms only for a
                # large mismatch or a known low-quality alignment.
                _is_last_resort = "last_resort" in Path(scienceFpath).name.lower()
                kernel_order = _auto_sfft_kernel_order(
                    rel_diff,
                    n_eff,
                    _max_auto_order,
                    last_resort=_is_last_resort,
                )

                if kernel_order > 0:
                    logger.info(
                        "Boosting kernel_order to %d (PSF rel_diff=%.2f, "
                        "n_eff=%d, last_resort=%s).",
                        kernel_order,
                        rel_diff,
                        n_eff,
                        _is_last_resort,
                    )

                logger.info(
                    "Auto-selected kernel_order=%d (PSF rel_diff=%.2f, %d matched sources%s)",
                    kernel_order, rel_diff, n_matched,
                    f", SFFT self-match ~{n_eff}" if _sfft_self_match else "",
                )

            # Validate background polynomial order (bg_order)
            bg_order = ts_cfg.get("sfft_bg_order", 0)
            if bg_order is None:
                bg_order = 0
            else:
                bg_order = int(bg_order)
                if bg_order > 2:
                    logger.warning(
                        "Background polynomial order %d is unusually high (recommended 0-2). High orders may cause overfitting and instability.",
                        bg_order
                    )
                    bg_order = min(bg_order, 2)
                ts_cfg["sfft_bg_order"] = bg_order

            # Validate StarExt_iter
            star_ext_iter = ts_cfg.get("sfft_star_ext_iter", None)
            if star_ext_iter is not None and star_ext_iter > 0:
                star_ext_iter = int(star_ext_iter)
                if star_ext_iter > 6:
                    logger.warning(
                        "StarExt_iter %d is unusually high (recommended 1-6). High values may cause over-deblending and slow performance.",
                        star_ext_iter
                    )
                    star_ext_iter = min(star_ext_iter, 6)
                ts_cfg["sfft_star_ext_iter"] = star_ext_iter

            # 4b. SFFT: default is sparse (ESP) for better performance; crowded (ECP) only if explicitly enabled.

            # =============================================================
            # 5. Run subtraction backend
            # =============================================================
            # Track which backend actually produced the difference image.
            # The cascade is linear (zogy -> sfft -> hotpants); each helper
            # returns "done" on success or the next method to try.  Recorded
            # in the diff header (SUBALGO/SUBREQ/SUBFALL) for provenance --
            # the helpers mutate `method`, so without this the metadata can
            # only record the literal string "done".
            requested_method = method
            backend_used: Optional[str] = None

            if method == "zogy":
                _res = self._subtract_zogy(
                    scienceFpath,
                    template_work_fpath,
                    differenceFpath,
                    scienceHeader,
                    science_psf,
                    template_psf,
                    science_saturate,
                    template_saturate,
                    method,
                    science_fwhm=science_fwhm,
                    template_fwhm=template_fwhm,
                    matching_sources=matching_sources,
                    kernel_half_width=kernel_half_width,
                    universal_mask=universal_mask,
                    masked_sources=masked_sources,
                    target_xy=target_location,
                    science_gain=science_gain,
                    template_gain=template_gain,
                )
                if _res == "done":
                    backend_used = "zogy"
                method = _res

            if method == "sfft":
                # SFFT sparse flavor requires sky-subtracted inputs
                # (Hu et al. 2022 S3.2): its masking replaces non-source
                # pixels with zero, so a sky pedestal creates step edges at
                # mask boundaries that corrupt the FFT-based kernel solution.
                # Applied ONLY here, to dedicated temp files, so a later
                # HOTPANTS fallback still sees the original images (HOTPANTS
                # models the background itself via -bgo; ZOGY subtracts its
                # own median internally).  A shared pre-subtraction
                # previously left SFFT->HOTPANTS fallbacks mixing original
                # science with a sky-subtracted template.
                _sci_sfft_in = str(scienceFpath)
                _tpl_sfft_in = str(template_work_fpath)
                ts_cfg_sky = self.input_yaml.get("template_subtraction", {})
                _sky_subtract = _as_bool(
                    ts_cfg_sky.get(
                        "sky_subtract",
                        ts_cfg_sky.get("sfft_sky_subtract", True),
                    ),
                    True,
                )
                if _sky_subtract:
                    from astropy.stats import sigma_clipped_stats as _scs
                    for _img_label, _img_data, _img_hdr, _is_sci in [
                        ("science", scienceImage, scienceHeader, True),
                        ("template", templateImage, templateHeader, False),
                    ]:
                        _invalid = ~np.isfinite(_img_data) | (
                            np.abs(_img_data) < 1.1e-20
                        )
                        if _invalid.all():
                            logger.debug(
                                "Sky subtraction skipped for %s: all pixels invalid.",
                                _img_label,
                            )
                            continue
                        _, _sky_median, _ = _scs(
                            _img_data, mask=_invalid, sigma=3, maxiters=5
                        )
                        if not (np.isfinite(_sky_median) and abs(_sky_median) > 1e-10):
                            continue
                        _img_data = _img_data - _sky_median
                        # Restore NaN at invalid/sentinel positions: subtracting
                        # the median would otherwise turn 0-sentinel pixels
                        # (SWarp no-coverage, chip gaps) into -median, which
                        # escapes both the |x|<1.1e-20 sentinel test and the
                        # ~isfinite test in the mask construction.
                        _img_data = np.where(_invalid, np.nan, _img_data)
                        if _is_sci:
                            scienceImage = _img_data
                            fd, _sci_tmp = tempfile.mkstemp(
                                prefix="science_skysub_",
                                suffix=".fits",
                                dir=str(scienceDir),
                            )
                            os.close(fd)
                            write_fits(_sci_tmp, scienceImage, scienceHeader)
                            if os.path.exists(_sci_tmp):
                                _sci_prepared_path = _sci_tmp
                                _sci_sfft_in = _sci_tmp
                            else:
                                logger.warning(
                                    "Sky-subtracted science temp file not found after write: %s",
                                    _sci_tmp,
                                )
                            logger.info(
                                "Science image sky-subtracted for SFFT (median %.4g ADU removed).",
                                float(_sky_median),
                            )
                        else:
                            templateImage = _img_data
                            fd, _tpl_tmp = tempfile.mkstemp(
                                prefix="template_skysub_",
                                suffix=".fits",
                                dir=str(scienceDir),
                            )
                            os.close(fd)
                            write_fits(_tpl_tmp, templateImage, templateHeader)
                            if os.path.exists(_tpl_tmp):
                                _tpl_sfft_path = _tpl_tmp
                                _tpl_sfft_in = _tpl_tmp
                            else:
                                logger.warning(
                                    "Sky-subtracted template temp file not found after write: %s",
                                    _tpl_tmp,
                                )
                            logger.info(
                                "Template image sky-subtracted for SFFT (median %.4g ADU removed).",
                                float(_sky_median),
                            )

                # Clean input files to prevent SFFT from modifying originals in-place
                # (matches HOTPANTS behavior and prevents crosstalk)
                if not os.path.exists(_sci_sfft_in):
                    logger.warning(
                        "SFFT science input does not exist: %s - "
                        "falling back to original science path.",
                        _sci_sfft_in,
                    )
                    _sci_sfft_in = str(scienceDir / sci_name)
                if not os.path.exists(_tpl_sfft_in):
                    logger.warning(
                        "SFFT template input does not exist: %s - "
                        "falling back to working template path.",
                        _tpl_sfft_in,
                    )
                    _tpl_sfft_in = str(template_work_fpath)
                sci_clean = clean_fits_nans(_sci_sfft_in, str(scienceDir))
                ref_clean = clean_fits_nans(_tpl_sfft_in, str(scienceDir))
                _sci_clean_path = sci_clean
                _ref_clean_path = ref_clean
                # Write per-image FWHM into the cleaned FITS headers so that
                # run_sfft.py reads the correct per-image FWHM (it reads from
                # headers for detect_minarea and diagnostic logging).  Without
                # this, stale or missing FWHM keywords cause wrong source
                # detection area sizing.
                try:
                    for _clean_path, _clean_fwhm, _clean_label in [
                        (sci_clean, science_fwhm, "science"),
                        (ref_clean, template_fwhm, "template"),
                    ]:
                        if _clean_path and os.path.isfile(_clean_path) and _clean_fwhm and np.isfinite(float(_clean_fwhm)) and float(_clean_fwhm) > 0:
                            with fits.open(_clean_path, mode="update", memmap=False) as _hdl:
                                _hdl[0].header["FWHM"] = float(_clean_fwhm)
                                _hdl.flush()
                except Exception as _fwhm_e:
                    logger.debug("Could not write FWHM to cleaned SFFT input FITS: %s", _fwhm_e)

                _res = self._subtract_sfft(
                    sci_clean,
                    ref_clean,
                    differenceFpath,
                    mask_loc,
                    scienceDir,
                    base_name,
                    masked_sources,
                    matching_sources,
                    kernel_order,
                    sfft_kernel_hw,
                    method,
                    science_fwhm,
                    template_fwhm,
                    science_gain,
                    template_gain,
                    science_saturate,
                    template_saturate,
                )
                if _res == "done":
                    backend_used = "sfft"
                method = _res

            if method == "hotpants":
                # HOTPANTS always receives the original science image and the
                # working template (inpainted if enabled) -- the SFFT-only
                # sky subtraction above never rewrote either path, so no
                # restore step is needed and the fallback cannot mix
                # sky-subtracted and original inputs.
                #
                # Per-image masks: the universal mask conflates science-only
                # defects (hardware mask) with template-side defects.  A
                # science bad column is not a template defect; HOTPANTS
                # accepts separate -imi/-tmi maps and rejects a substamp when
                # either image is masked in its footprint.
                sci_mask_hp = np.clip(
                    science_mask_nans.astype(np.int32)
                    + science_seg_mask.astype(np.int32)
                    + (
                        background_defects_mask.astype(np.int32)
                        if background_defects_mask is not None
                        else 0
                    ),
                    0,
                    1,
                )
                tpl_mask_hp = np.clip(
                    template_mask_nans.astype(np.int32)
                    + template_seg_mask.astype(np.int32),
                    0,
                    1,
                )
                mask_sci_loc = os.path.join(
                    scienceDir, f"hotpants_mask_sci_{base_name}"
                )
                mask_tpl_loc = os.path.join(
                    scienceDir, f"hotpants_mask_tpl_{base_name}"
                )
                save_to_fits(sci_mask_hp.astype(int), mask_sci_loc)
                save_to_fits(tpl_mask_hp.astype(int), mask_tpl_loc)

                # Noise maps must live on the same pixel grid as the images
                # actually fed to HOTPANTS.  The science map was built from
                # the aligned science image; the template weight map (when
                # present) was never resampled by align(), so a grid mismatch
                # is detected here and the map is dropped rather than
                # silently misapplied.
                sci_noise_v = _validate_noise_map(
                    scienceNoise,
                    scienceImage.shape,
                    "science",
                    ref_wcs=_sci_wcs_v,
                )
                tpl_noise_v = _validate_noise_map(
                    templateNoise,
                    templateImage.shape,
                    "template",
                    ref_wcs=_tpl_wcs_v,
                )

                success = self._subtract_hotpants(
                    scienceFpath,
                    template_work_fpath,
                    differenceFpath,
                    mask_loc,
                    scienceDir,
                    base_name,
                    scienceMedian,
                    scienceSTD,
                    templateMedian,
                    templateSTD,
                    science_saturate,
                    template_saturate,
                    science_readnoise,
                    template_readnoise,
                    science_fwhm,
                    template_fwhm,
                    kernel_order,
                    stamp_loc,
                    sci_noise_v,
                    sfft_kernel_hw,
                    templateNoise=tpl_noise_v,
                    sci_mask_loc=mask_sci_loc,
                    tpl_mask_loc=mask_tpl_loc,
                    matching_sources=matching_sources,
                    target_xy=(
                        target_location[0] if target_location else None
                    ),
                    masked_sources=masked_sources,
                )
                if not success:
                    return None, None, None, None
                backend_used = "hotpants"

            # =============================================================
            # 6. Validate output
            # =============================================================
            if not (
                os.path.isfile(differenceFpath) and os.path.getsize(differenceFpath) > 0
            ):
                logger.error("Difference file missing or empty")
                return None, None, None, None

            diff_data, diff_header = read_fits(differenceFpath)

            # The difference image must live on the science pixel grid: the
            # caller maps target RA/Dec through its WCS for photometry, so a
            # wrong-shaped or wrongly-projected file must not pass silently.
            if diff_data.shape != scienceImage.shape:
                logger.error(
                    "Difference image shape %s != science shape %s; "
                    "subtraction output unusable.",
                    diff_data.shape, scienceImage.shape,
                )
                return None, None, None, None

            # WCS validation.  Backends should copy the science header, but
            # external tools can drop or rewrite WCS.  When the diff WCS is
            # missing or disagrees with the science grid, inject the science
            # WCS -- the difference image is defined on the science pixel
            # grid by construction.
            diff_header = _ensure_diff_wcs(
                diff_header,
                scienceHeader,
                diff_data.shape,
                target_xy=target_location[0] if target_location else None,
            )

            # Backend provenance: which backend produced this file, what was
            # requested, and how many fallbacks occurred.
            if backend_used is not None:
                _cascade = ("zogy", "sfft", "hotpants")
                try:
                    _nfallback = max(
                        0,
                        _cascade.index(backend_used)
                        - _cascade.index(requested_method),
                    )
                except ValueError:
                    _nfallback = 0
                diff_header["SUBALGO"] = backend_used
                diff_header["SUBREQ"] = requested_method
                diff_header["SUBFALL"] = int(_nfallback)

            # ------------------------------------------------------------------
            # Preserve "no data" regions through subtraction backends.
            #
            # Some subtraction tools (notably SFFT variants) can emit 0-valued pixels
            # where inputs contained NaNs (chip gaps / no-coverage). Downstream
            # photometry and diagnostics must treat those pixels as invalid, so we
            # re-impose the combined NaN mask from the original aligned inputs.
            # ------------------------------------------------------------------
            combined_nan_mask = None
            diff_invalid_mask = None
            try:
                # Match the sentinel test used for mask_nans above: both
                # non-finite pixels and |x| < 1.1e-20 zero-sentinels
                # (SWarp no-coverage) are invalid in the inputs.
                combined_nan_mask = (
                    (~np.isfinite(scienceImage))
                    | (np.abs(scienceImage) < 1.1e-20)
                    | (~np.isfinite(templateImage))
                    | (np.abs(templateImage) < 1.1e-20)
                )
                # A finite pixel inside the universal mask can keep raw
                # input flux in the diff (e.g. a saturated star whose
                # template footprint was blanked leaves DIFF ~= SCI) and
                # read as a transient downstream, so the full mask - not
                # just no-data inputs - is invalidated here.  Applies to
                # every backend, including the HOTPANTS fallback.
                diff_invalid_mask = combined_nan_mask.copy()
                if (
                    universal_mask is not None
                    and universal_mask.shape == diff_invalid_mask.shape
                ):
                    diff_invalid_mask |= universal_mask.astype(bool)
                elif universal_mask is not None:
                    logger.warning(
                        "universal_mask shape %s != diff shape %s; masking "
                        "no-data inputs only.",
                        universal_mask.shape,
                        diff_invalid_mask.shape,
                    )
                if np.any(diff_invalid_mask) and diff_data.shape == diff_invalid_mask.shape:
                    diff_data = np.asarray(diff_data, dtype=float)
                    n_leak = int(
                        np.count_nonzero(np.isfinite(diff_data) & diff_invalid_mask)
                    )
                    diff_data[diff_invalid_mask] = np.nan
                    n_src_only = int(
                        np.count_nonzero(diff_invalid_mask & ~combined_nan_mask)
                    )
                    logger.info(
                        "Invalidated %d finite masked pixels in difference image "
                        "(mask=%d px: %d no-data + %d source/defect-only).",
                        n_leak,
                        int(np.count_nonzero(diff_invalid_mask)),
                        int(np.count_nonzero(combined_nan_mask)),
                        n_src_only,
                    )
            except Exception:
                # Non-fatal: continue with raw backend output.
                pass

            _finite_diff = np.isfinite(diff_data)
            if not _finite_diff.any() or np.std(diff_data[_finite_diff]) < 1e-5:
                logger.error(
                    "Difference image is invalid (all NaN or near-zero variance). Subtraction backend may have written a bad file; treat as failure and use "
                    "original science image."
                )
                return None, None, None, None

            # =============================================================
            # 6a. Subtraction quality validation
            # =============================================================
            # Comprehensive difference-image quality assessment including:
            #   - Global statistics (median, std, RMS)
            #   - Dipole detection around known sources
            #   - Bright-star residual flux
            #   - Spatial background variation
            #   - Residual autocorrelation
            #   - Edge artifacts
            #   - Structured quality score with pass/downgrade/fail classification
            #   - Machine-readable JSON manifest for provenance
            diff_quality_metrics = None
            try:
                from utils.difference_quality import (
                    assess_difference_image,
                    write_quality_manifest,
                    write_quality_to_fits_header,
                    QualityConfig,
                )

                # Build quality mask: NaN, universal mask, target region
                quality_mask = ~np.isfinite(diff_data) | (np.abs(diff_data) < 1.1e-20) | universal_mask.astype(bool)
                try:
                    _tx = float(self.input_yaml.get("target_x_pix", np.nan))
                    _ty = float(self.input_yaml.get("target_y_pix", np.nan))
                    _fwhm_excl = float(self.input_yaml.get("fwhm", science_fwhm))
                    _excl_r = max(3.0 * _fwhm_excl, 15.0)
                    if np.isfinite(_tx) and np.isfinite(_ty):
                        yy, xx = np.ogrid[:diff_data.shape[0], :diff_data.shape[1]]
                        target_mask = (xx - _tx) ** 2 + (yy - _ty) ** 2 < _excl_r ** 2
                        quality_mask = quality_mask | target_mask
                except Exception:
                    pass  # Non-fatal: skip target exclusion if it fails

                # Build quality config from pipeline config
                _ts_cfg_q = self.input_yaml.get("template_subtraction", {}) or {}
                _qcfg = QualityConfig(
                    dipole_n_sigma=float(_ts_cfg_q.get("diffqual_dipole_n_sigma", 5.0)),
                    dipole_radius_fwhm=float(_ts_cfg_q.get("diffqual_dipole_radius_fwhm", 1.5)),
                    dipole_max_fraction_pass=float(_ts_cfg_q.get("diffqual_dipole_max_frac_pass", 0.05)),
                    dipole_max_fraction_fail=float(_ts_cfg_q.get("diffqual_dipole_max_frac_fail", 0.20)),
                    dipole_min_checked=int(_ts_cfg_q.get("diffqual_dipole_min_checked", 10)),
                    dipole_wilson_z=float(_ts_cfg_q.get("diffqual_dipole_wilson_z", 1.645)),
                    bright_star_n_sigma=float(_ts_cfg_q.get("diffqual_bright_star_n_sigma", 50.0)),
                    bright_star_max_residual_sigma=float(_ts_cfg_q.get("diffqual_bright_star_max_resid_sigma", 3.0)),
                    edge_max_std_ratio=float(_ts_cfg_q.get("diffqual_edge_max_std_ratio", 2.0)),
                    pass_threshold=float(_ts_cfg_q.get("diffqual_pass_threshold", 0.75)),
                    downgrade_threshold=float(_ts_cfg_q.get("diffqual_downgrade_threshold", 0.50)),
                )

                # Sources to check for structured residuals: the matched
                # calibrators AND the masked (bright/saturated) centers.
                # The worst kernel-sidelobe residuals appear at the
                # bright stars, whose cores are masked but whose
                # sidelobe rings are not -- checking only the clean
                # matched calibrators let a deconvolving-kernel diff
                # pass QA (SN2024pba 2025-02-22 r).
                _matching_priors = list(matching_sources or [])
                _matching_for_quality = _matching_priors
                if backend_used == "sfft":
                    _sfft_base = (
                        Path(base_name).stem.replace(" ", "_")
                        .replace(".", "_")
                        .replace("_APT", "")
                        .replace("_ERROR", "")
                    )
                    _sfft_match_csv = scienceDir / (
                        f"SFFT_Matching_Sources_{_sfft_base}.csv"
                    )
                    _sfft_actual_sources = _read_sfft_matching_positions(
                        str(_sfft_match_csv)
                    )
                    if _sfft_actual_sources is not None:
                        _matching_for_quality = _sfft_actual_sources
                        logger.info(
                            "SFFT kernel sources: %d pipeline priors -> %d "
                            "sources used by the adopted run.",
                            len(_matching_priors),
                            len(_matching_for_quality),
                        )
                _n_matching_sources = len(_matching_for_quality)

                _dipole_sources = list(_matching_for_quality)
                if masked_centers:
                    _dipole_sources = deduplicate_points(
                        _dipole_sources + list(masked_centers), min_sep=2.0
                    )
                _bright_stars = _dipole_sources

                # Determine the difference-image FWHM for quality checks
                _diff_fwhm = science_fwhm
                _convd_hdr = str(diff_header.get("CONVD", "")).strip().upper()
                _diff_fwhm_hdr = float(diff_header.get("DIFFFWHM", 0.0))
                if _diff_fwhm_hdr > 0:
                    _diff_fwhm = _diff_fwhm_hdr

                # Metadata for provenance
                _qmeta = {
                    "algorithm": str(backend_used or "unknown"),
                    "forceconv": str(diff_header.get("FORCECON", "")),
                    "kernel_order": int(kernel_order),
                    "kernel_half_width": int(kernel_half_width) if kernel_half_width else 0,
                    "science_fwhm": float(science_fwhm),
                    "template_fwhm": float(template_fwhm),
                    "n_matching_sources": _n_matching_sources,
                    "n_matching_priors": len(_matching_priors),
                    "flux_scale_conv": float(diff_header.get("FSCAL_CONV", 0.0)),
                    "flux_scale_phot": float(diff_header.get("FSCAL_PHOT", 0.0)),
                    "flux_scale_discrep_pct": float(diff_header.get("FSCAL_DISC", 0.0)),
                }

                diff_quality_metrics = assess_difference_image(
                    diff_data,
                    quality_mask,
                    source_positions=_dipole_sources,
                    bright_star_positions=_bright_stars,
                    fwhm=_diff_fwhm,
                    cfg=_qcfg,
                    metadata=_qmeta,
                )

                # Write quality keywords to the difference image FITS header
                # NOTE: The actual write is deferred to after the final
                # write_fits() call below, because write_fits would
                # overwrite the header and lose these keywords.
                pass

                # Write machine-readable JSON manifest
                _manifest_path = os.path.join(
                    str(scienceDir), f"diff_quality_{base_name}.json"
                )
                _manifest_meta = {
                    "science_file": str(scienceFpath),
                    "template_file": str(templateFpath),
                    "difference_file": str(differenceFpath),
                    "science_fwhm": float(science_fwhm),
                    "template_fwhm": float(template_fwhm),
                    "kernel_order": int(kernel_order),
                    "kernel_half_width": int(kernel_half_width) if kernel_half_width else 0,
                    "n_matching_sources": _n_matching_sources,
                    "n_matching_priors": len(_matching_priors),
                    "n_masked_sources": len(masked_sources) if masked_sources else 0,
                    "masked_percentage": float(masked_percentage),
                    "universal_mask_frac": float(np.mean(universal_mask)),
                }
                write_quality_manifest(
                    diff_quality_metrics, _manifest_path, extra_metadata=_manifest_meta
                )

                # Log warnings for degraded quality
                if diff_quality_metrics.quality_class == "fail":
                    logger.error(
                        "Subtraction quality FAILED (score=%.3f):\n"
                        "    dipoles=%d/%d checked (%.1f%%),\n"
                        "    bright_star_resid=%.1f sigma,\n"
                        "    bg_spatial_std=%.3f, edge_ratio=%.2f.\n"
                        "    Downstream photometry should be treated with\n"
                        "    caution.",
                        diff_quality_metrics.quality_score,
                        diff_quality_metrics.dipole_count,
                        diff_quality_metrics.dipole_checked,
                        diff_quality_metrics.dipole_fraction * 100,
                        diff_quality_metrics.bright_star_residual_rms,
                        diff_quality_metrics.background_spatial_std,
                        diff_quality_metrics.edge_std_ratio,
                    )
                elif diff_quality_metrics.quality_class == "downgrade":
                    logger.warning(
                        "Subtraction quality DOWNGRADED (score=%.3f): "
                        "dipoles=%d/%d checked (%.1f%%), bright_star_resid=%.1f sigma, "
                        "bg_spatial_std=%.3f, edge_ratio=%.2f.",
                        diff_quality_metrics.quality_score,
                        diff_quality_metrics.dipole_count,
                        diff_quality_metrics.dipole_checked,
                        diff_quality_metrics.dipole_fraction * 100,
                        diff_quality_metrics.bright_star_residual_rms,
                        diff_quality_metrics.background_spatial_std,
                        diff_quality_metrics.edge_std_ratio,
                    )

            except ImportError:
                logger.debug(
                    "utils.difference_quality not available; "
                    "falling back to basic quality checks."
                )
                # Fallback: basic checks (original behaviour)
                try:
                    quality_mask = ~np.isfinite(diff_data) | (np.abs(diff_data) < 1.1e-20) | universal_mask.astype(bool)
                    valid_pixels = diff_data[~quality_mask]
                    valid_pixels = valid_pixels[np.isfinite(valid_pixels)]
                    if len(valid_pixels) > 0:
                        diff_median = np.median(valid_pixels)
                        diff_std = np.nanstd(valid_pixels)
                        diff_rms = np.sqrt(np.mean(valid_pixels ** 2))
                        logger.info(
                            "Subtraction quality (basic): median=%.3f std=%.3f rms=%.3f | valid=%d px",
                            diff_median, diff_std, diff_rms, len(valid_pixels)
                        )
                        if abs(diff_median) > 0.1 * diff_std:
                            logger.warning(
                                "Subtraction has systematic offset (median=%.3f, std=%.3f).",
                                diff_median, diff_std
                            )
                except Exception as e:
                    logger.warning("Subtraction quality validation failed (non-fatal): %s", e)
            except Exception as e:
                logger.warning("Subtraction quality validation failed (non-fatal): %s", e)

            # Write the masked difference image back (no background zeroing --
            # the subtraction backend's native output is preserved so photometry
            # sees the true pixel values including any DC offset from the sky).
            try:
                if diff_invalid_mask is not None and np.any(diff_invalid_mask) and diff_data.shape == diff_invalid_mask.shape:
                    diff_data = np.asarray(diff_data, dtype=float)
                    diff_data[diff_invalid_mask] = np.nan
            except Exception:
                pass
            # Write to a sibling temp file and rename atomically so a crash
            # mid-write cannot leave a truncated difference image that the
            # caller would mistake for a valid product.
            _diff_part_path = f"{differenceFpath}.part"
            write_fits(_diff_part_path, diff_data, diff_header)
            os.replace(_diff_part_path, differenceFpath)
            _diff_part_path = None

            # Write quality keywords to the difference image FITS header
            # AFTER the final write_fits so they are not overwritten.
            if diff_quality_metrics is not None:
                try:
                    from utils.difference_quality import (
                        write_quality_to_fits_header as _wqfh,
                    )
                    _wqfh(differenceFpath, diff_quality_metrics)
                except Exception as _wq_e:
                    logger.warning(
                        "Failed to write quality keywords to FITS header: %s",
                        _wq_e,
                    )

            elapsed = time.time() - t0
            logger.info("Image subtraction completed in %.1f s", elapsed)

            return differenceFpath, visualization_mask, masked_centers, kernel_half_width

        except Exception:
            logger.exception("Unhandled error in subtract()")
            return None, None, None, None
        finally:
            if prepared_template_fpath:
                try:
                    os.remove(prepared_template_fpath)
                except OSError:
                    pass
            # Clean up temporary cleaned_ files created by clean_fits_nans.
            # Use the dedicated path variables (not `method`) so cleanup is
            # unconditional even when _subtract_sfft mutates `method` on fallback.
            for _tmp in (_sci_clean_path, _ref_clean_path, _sci_prepared_path, _tpl_sfft_path, _diff_part_path):
                try:
                    if _tmp and os.path.exists(_tmp):
                        os.remove(_tmp)
                        logger.debug("Cleaned up temp file: %s", _tmp)
                except (OSError, NameError, UnboundLocalError) as e:
                    logger.warning("Failed to clean up temp file %s: %s", _tmp, e)

    # ----- Private subtraction-backend methods -----

    def _subtract_zogy(
        self,
        scienceFpath,
        templateFpath,
        differenceFpath,
        scienceHeader,
        science_psf,
        template_psf,
        science_saturate,
        template_saturate,
        method,
        science_fwhm=0.0,
        template_fwhm=0.0,
        matching_sources=None,
        kernel_half_width=None,
        universal_mask=None,
        masked_sources=None,
        target_xy=None,
        science_gain=None,
        template_gain=None,
    ) -> str:
        """Attempt ZOGY subtraction; return next method to try on failure."""
        logger.info("Starting ZOGY subtraction...")
        try:
            if not science_psf or not template_psf:
                raise ValueError("PSF models required for ZOGY are missing")
            science_data = np.asarray(fits.getdata(scienceFpath), dtype=float)
            reference_data = np.asarray(fits.getdata(templateFpath), dtype=float)
            # Stamps may be oversampled ePSF grids (OVERSAMP > 1); ZOGY
            # needs native-resolution PSFs.
            science_psf_data = _load_psf_stamp_native(science_psf)
            reference_psf_data = _load_psf_stamp_native(template_psf)
            if science_data.shape != reference_data.shape:
                raise ValueError(
                    f"ZOGY requires same image shapes: science {science_data.shape} vs reference {reference_data.shape}"
                )
            from aperture import gain_e_per_adu_from_header, resolve_gain_e_per_adu

            if science_gain is None:
                try:
                    science_gain = resolve_gain_e_per_adu(None, self.input_yaml)
                except ValueError:
                    try:
                        science_gain = gain_e_per_adu_from_header(
                            scienceHeader, None
                        )[0]
                    except ValueError:
                        science_gain = 1.0
            if template_gain is None:
                try:
                    template_gain = gain_e_per_adu_from_header(
                        fits.getheader(templateFpath), None
                    )[0]
                except (OSError, ValueError):
                    template_gain = 1.0

            # The universal mask (saturated/negative seg footprints, defect
            # regions, inpainted cores) marks pixels that carry real but
            # unusable flux: the FFT convolution has no masking facility, so
            # those pixels are median-filled alongside the NaN regions and
            # excluded from the flux-scale/star-PSF estimators.
            _um = None
            if universal_mask is not None:
                try:
                    _um_arr = np.asarray(universal_mask, dtype=bool)
                except (TypeError, ValueError):
                    _um_arr = None
                if _um_arr is not None and _um_arr.shape == science_data.shape:
                    _um = _um_arr
                else:
                    logger.warning(
                        "ZOGY: ignoring universal mask with shape %s (image is %s).",
                        getattr(_um_arr, "shape", None),
                        science_data.shape,
                    )

            # The ZOGY math runs via the self-contained _zogy_subtract()
            # (numpy FFTs only, no external ZOGY package needed).
            ts_cfg = self.input_yaml.get("template_subtraction", {})

            # -----------------------------------------------------------------
            # Sky subtraction: ZOGY assumes background-subtracted images
            # (Zackay et al. 2016, Eq. 1 requires N and R to be pure source
            # flux with no DC offset).  Without sky subtraction, the constant
            # background creates a coherent Fourier component at zero
            # frequency that contaminates the difference image.
            # -----------------------------------------------------------------
            from astropy.stats import sigma_clipped_stats as _scs

            _zogy_sky_subtract = _as_bool(
                ts_cfg.get("sky_subtract", ts_cfg.get("zogy_sky_subtract", True)), True
            )
            if _zogy_sky_subtract:
                for _label, _data in [("science", science_data), ("reference", reference_data)]:
                    _finite = np.isfinite(_data)
                    if _um is not None:
                        _finite &= ~_um
                    if _finite.sum() > 100:
                        _, _sky, _ = _scs(_data[_finite], sigma=3, maxiters=5)
                        if np.isfinite(_sky) and abs(_sky) > 1e-10:
                            if _label == "science":
                                science_data = science_data - _sky
                            else:
                                reference_data = reference_data - _sky
                            logger.info(
                                "ZOGY: %s sky-subtracted (median %.4g removed).",
                                _label, float(_sky),
                            )

            # Noise RMS via biweight scale: a plain std is inflated by
            # sources, which would bias ZOGY's sn/sr noise terms, while
            # MAD staircases on quantized (compressed) stacks.
            from functions import biweight_sky_sigma as _rbs
            # Masked pixels carry real but unusable flux (saturated cores);
            # including them would inflate the noise scale.
            if _um is not None:
                _sci_finite = science_data[np.isfinite(science_data) & ~_um]
                _ref_finite = reference_data[np.isfinite(reference_data) & ~_um]
            else:
                _sci_finite = science_data[np.isfinite(science_data)]
                _ref_finite = reference_data[np.isfinite(reference_data)]
            _sn = _rbs(_sci_finite) if _sci_finite.size > 100 else (float(np.std(_sci_finite)) if _sci_finite.size > 0 else 1.0)
            _sr = _rbs(_ref_finite) if _ref_finite.size > 100 else (float(np.std(_ref_finite)) if _ref_finite.size > 0 else 1.0)
            _sn = float(_sn) if _sn and _sn > 0 else 1.0
            _sr = float(_sr) if _sr and _sr > 0 else 1.0
            # Snapshot the unconvolved science noise: CONVD=SCI rescales
            # _sn by the kernel norm, but the downstream background_rms
            # map that VSCALE rescales is in unconvolved units, so the
            # measured factor must be sigma_diff/sigma_sci.
            _sn_sci = _sn

            # Sanitise the measured FWHMs once: they drive the star-stack
            # window, the PSF plausibility gate, the forceconv resolution,
            # and several header cards -- a NaN/None would otherwise poison
            # all of them.
            def _fwhm_val(v):
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return 0.0
                return f if np.isfinite(f) and f > 0 else 0.0

            _fs_f = _fwhm_val(science_fwhm)
            _ft_f = _fwhm_val(template_fwhm)

            # Variable sources (masked_sources, 1-based FITS convention)
            # and the target position (0-based) must not enter the flux
            # scale or star-stack: a variable calibrator biases the scale,
            # and a transient in the stack distorts the measured PSFs.
            _ban_xy = []
            for _bp_arr, _shift in (
                (masked_sources, -1.0),
                (target_xy, 0.0),
            ):
                if _bp_arr is None:
                    continue
                try:
                    for _bp in np.atleast_2d(
                        np.asarray(_bp_arr, dtype=float)
                    ):
                        if _bp.size >= 2 and np.isfinite(_bp[:2]).all():
                            _ban_xy.append(
                                (float(_bp[0]) + _shift, float(_bp[1]) + _shift)
                            )
                except (TypeError, ValueError):
                    continue
            if _ban_xy and matching_sources:
                _excl_r = max(3.0 * max(_fs_f, _ft_f), 15.0)
                _ban = np.asarray(_ban_xy)
                _kept_b = []
                for _msrc in matching_sources:
                    try:
                        _mxy = np.asarray(
                            [_msrc["x"], _msrc["y"]], dtype=float
                        )
                    except (TypeError, IndexError, KeyError):
                        _mxy = np.asarray(_msrc, dtype=float)
                    if _mxy.size < 2 or not np.isfinite(_mxy[:2]).all():
                        continue
                    if (
                        np.min(
                            np.hypot(
                                _ban[:, 0] - _mxy[0], _ban[:, 1] - _mxy[1]
                            )
                        )
                        <= _excl_r
                    ):
                        continue
                    _kept_b.append(_msrc)
                _n_ban = len(matching_sources) - len(_kept_b)
                if _n_ban:
                    logger.info(
                        "ZOGY: excluded %d matched source(s) within %.1f px "
                        "of variable/target positions.",
                        _n_ban,
                        _excl_r,
                    )
                    matching_sources = _kept_b

            # Field PSF measured directly off the bright matched stars.  It
            # is the most truthful width available to this routine: header/
            # SExtractor FWHMs under-measure broad plateau PSFs on resampled
            # frames, and non-converged ePSFs can be outright garbage.
            # The filtered positions also feed the aperture flux-scale
            # estimate so spikes cannot poison it.
            _star_sci_stamp, _star_sci_f, _star_pos = _zogy_star_psf(
                science_data, matching_sources, _fs_f or 8.0,
                saturate=science_saturate,
            )
            _star_ref_stamp, _star_ref_f, _star_ref_pos = _zogy_star_psf(
                reference_data, matching_sources, _ft_f or 8.0,
                saturate=template_saturate,
            )
            for _tag, _stamp, _stack_f, _image_f, _model in (
                ("science", _star_sci_stamp, _star_sci_f, _fs_f, science_psf_data),
                ("template", _star_ref_stamp, _star_ref_f, _ft_f, reference_psf_data),
            ):
                if _stamp is None:
                    continue
                _model_f = _measure_stamp_fwhm(_model)
                if not _zogy_star_stack_consistent(
                    _stack_f, (_image_f, _model_f)
                ):
                    logger.warning(
                        "ZOGY %s star-stack PSF rejected (FWHM %.2f px; "
                        "image/model widths %.2f/%.2f px).",
                        _tag,
                        _stack_f,
                        _image_f,
                        _model_f,
                    )
                    if _tag == "science":
                        _star_sci_stamp, _star_sci_f = None, float("nan")
                        # A vetoed ensemble's members are not trusted
                        # stars: [] (not None) so downstream consumers
                        # still know vetting ran.
                        _star_pos = []
                    else:
                        _star_ref_stamp, _star_ref_f = None, float("nan")
                        _star_ref_pos = []
            # Either validated list beats the raw one: matched-source
            # catalogs can carry cosmic-ray spikes that are blank sky in
            # the other frame.  An empty vetted list falls through to
            # the other frame's list, then to raw matches - the
            # aperture-scale estimator has its own robustification,
            # unlike the LSQ kernel solve.
            _scale_pos = (
                _star_pos
                if _star_pos
                else _star_ref_pos
                if _star_ref_pos
                else matching_sources
            )

            # -----------------------------------------------------------------
            # Flux-scale matching: ZOGY requires both images in the same flux
            # units (fn = fr).  When exposure times differ (e.g. 330s sci vs
            # 62s ref), the raw pixel values differ by the exposure-time ratio
            # (~5.3x) plus transparency/airmass differences (~6.6x total).
            # Without scaling, sources don't cancel in the difference image.
            # Scale the template to match the science image.  Preferred
            # estimator: aperture photometry at the matched-source positions
            # (shared pixel grid after resampling; integrating ~2 FWHM
            # apertures cancels the PSF-shape difference that biases
            # per-pixel ratios).  Fallback: median ratio of bright source
            # pixels, then EXPTIME ratio.
            # -----------------------------------------------------------------
            _flux_scale_src = "none"
            _ap_fwhm = max(
                _star_sci_f
                if np.isfinite(_star_sci_f) and _star_sci_f > 0
                else _fs_f,
                _star_ref_f
                if np.isfinite(_star_ref_f) and _star_ref_f > 0
                else _ft_f,
            )
            _flux_scale, _n_scale_ratios = _zogy_matched_aperture_scale(
                science_data,
                reference_data,
                _scale_pos,
                _ap_fwhm,
                sat_sci=science_saturate,
                sat_ref=template_saturate,
            )
            # The validated PSF-pool subset can collapse to a handful of
            # positions (rejected stamps, saturated cores), leaving the
            # median resting on one or two ratios.  Retry with the full
            # matched-source list (already screened for variables and the
            # target) before accepting a sparse estimate.
            if (
                _n_scale_ratios < 3
                and matching_sources is not None
                and len(matching_sources) > len(_scale_pos)
            ):
                _s2, _n2 = _zogy_matched_aperture_scale(
                    science_data,
                    reference_data,
                    matching_sources,
                    _ap_fwhm,
                    sat_sci=science_saturate,
                    sat_ref=template_saturate,
                )
                if _n2 > _n_scale_ratios:
                    _flux_scale, _n_scale_ratios = _s2, _n2
            if np.isfinite(_flux_scale) and _flux_scale > 0:
                _flux_scale_src = "aperture"
            _both_finite = np.isfinite(science_data) & np.isfinite(reference_data)
            if _um is not None:
                _both_finite &= ~_um
            _bright_mask = _both_finite & (science_data > 5 * _sn) & (reference_data > 5 * _sr)
            # Exclude near-saturated pixels: a clipped core on one side
            # drags the median ratio away from the true photometric scale.
            for _sat_data, _sat_val in (
                (science_data, science_saturate),
                (reference_data, template_saturate),
            ):
                try:
                    _sat_lim = 0.9 * float(_sat_val)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(_sat_lim) and _sat_lim > 0:
                    _bright_mask &= _sat_data < _sat_lim
            # The pixel-ratio ensemble is also needed as a sanity check
            # when the aperture scale rests on only a star or two: a
            # single variable or mismatched source must not rescale the
            # whole subtraction unchecked.
            _pixel_scale = np.nan
            if _bright_mask.sum() > 50:
                _flux_ratios = science_data[_bright_mask] / reference_data[_bright_mask]
                _pixel_scale = float(np.median(_flux_ratios))
                # Inlier refinement: keep ratios within +-30% of the median.
                # Guard the threshold when the median is <= 0 (pathological
                # inputs can produce a non-positive scale; an absolute
                # tolerance keeps the comparison meaningful).
                if np.isfinite(_pixel_scale) and _pixel_scale > 0:
                    _tol = 0.3 * _pixel_scale
                    _valid = np.abs(_flux_ratios - _pixel_scale) < _tol
                    if _valid.sum() > 20:
                        _pixel_scale = float(np.median(_flux_ratios[_valid]))
            if not (np.isfinite(_flux_scale) and _flux_scale > 0):
                if np.isfinite(_pixel_scale) and _pixel_scale > 0:
                    _flux_scale = _pixel_scale
                    _flux_scale_src = "pixel-ratio"
            elif _n_scale_ratios < 3:
                # Cross-check the sparse aperture estimate against the
                # pixel ensemble.  PSF-shape differences bias the pixel
                # ratio by a few percent, so only a large disagreement
                # (>30%) marks the aperture ratios as untrustworthy.
                if (
                    np.isfinite(_pixel_scale)
                    and _pixel_scale > 0
                    and abs(_flux_scale - _pixel_scale) / _pixel_scale > 0.3
                ):
                    logger.warning(
                        "ZOGY: aperture flux scale %.4g from only %d "
                        "ratio(s) disagrees with the pixel-ratio ensemble "
                        "%.4g by >30%%; using the ensemble value.",
                        _flux_scale,
                        _n_scale_ratios,
                        _pixel_scale,
                    )
                    _flux_scale = _pixel_scale
                    _flux_scale_src = "pixel-ratio"
                else:
                    logger.warning(
                        "ZOGY: aperture flux scale %.4g rests on only %d "
                        "matched ratio(s); treating as low-confidence.",
                        _flux_scale,
                        _n_scale_ratios,
                    )
            if not np.isfinite(_flux_scale) or _flux_scale <= 0:
                from aperture import (
                    exposure_seconds_from_header,
                    resolve_exposure_time_seconds,
                )

                try:
                    _sci_exptime = resolve_exposure_time_seconds(
                        None, self.input_yaml
                    )
                    _sci_gain = float(science_gain)
                    _ref_hdr = fits.getheader(templateFpath)
                    _ref_exptime = exposure_seconds_from_header(_ref_hdr)[0]
                    _ref_gain = float(template_gain)
                    _flux_scale = _zogy_exposure_flux_scale(
                        _sci_exptime,
                        _ref_exptime,
                        _sci_gain,
                        _ref_gain,
                    )
                except (OSError, ValueError):
                    _flux_scale = np.nan
                if np.isfinite(_flux_scale) and _flux_scale > 0:
                    _flux_scale_src = "exptime-gain"
                    logger.info(
                        "ZOGY: aperture and pixel-ratio flux scales invalid; "
                        "using exposure/gain ratio %.4g.",
                        _flux_scale,
                    )
                else:
                    _flux_scale = 1.0
                    logger.warning(
                        "ZOGY: could not estimate flux scale (aperture, pixel "
                        "ratio and exposure/gain all invalid); assuming scale=1.0."
                    )

            logger.info(
                "ZOGY flux scale: %.4g (template -> science, estimator=%s, "
                "n_ratios=%d, bright pixels=%d)",
                _flux_scale,
                _flux_scale_src,
                int(_n_scale_ratios),
                int(_bright_mask.sum()),
            )
            # The measured photometric scale, before the LSQ kernel-sum
            # fold below adjusts the *applied* scale -- kept separately so
            # the FSCAL_PHOT/FSCAL_CONV header cards (and the quality JSON
            # built from them) report what SFFT already does.
            _flux_scale_phot = _flux_scale

            # Scale template data and noise to match science flux units
            reference_data = reference_data * _flux_scale
            _sr = _sr * _flux_scale

            # -----------------------------------------------------------------
            # Per-pixel noise maps (optional): compute local background RMS
            # maps to capture spatially varying noise (e.g. near chip gaps,
            # bright galaxy backgrounds).  When disabled or too slow, fall
            # back to scalar noise (original behaviour).
            # -----------------------------------------------------------------
            _sn_map = None
            _sr_map = None
            _use_noise_maps = _as_bool(
                ts_cfg.get("zogy_per_pixel_noise", False), False
            )
            if _use_noise_maps:
                try:
                    def _local_noise_map(data, scalar_noise, box_size=64):
                        """Estimate per-pixel noise via local biweight scale
                        on a coarse grid, then bilinear upsample."""
                        h, w = data.shape
                        # Coarse grid: local scale in box_size tiles
                        ny = max(1, h // box_size)
                        nx = max(1, w // box_size)
                        _noise_grid = np.full((ny, nx), scalar_noise)
                        for iy in range(ny):
                            y0 = iy * box_size
                            y1 = min(y0 + box_size, h)
                            for ix in range(nx):
                                x0 = ix * box_size
                                x1 = min(x0 + box_size, w)
                                _tile = data[y0:y1, x0:x1]
                                _tile_ok = np.isfinite(_tile)
                                if _um is not None:
                                    _tile_ok &= ~_um[y0:y1, x0:x1]
                                _tile_finite = _tile[_tile_ok]
                                if _tile_finite.size > 50:
                                    _tile_std = _rbs(_tile_finite)
                                    if _tile_std and _tile_std > 0:
                                        _noise_grid[iy, ix] = float(_tile_std)
                        # Bilinear upsample to full image size
                        from scipy.ndimage import zoom
                        _zoom_y = h / ny
                        _zoom_x = w / nx
                        _noise_map = zoom(_noise_grid, (_zoom_y, _zoom_x), order=1)
                        # Crop/pad to exact shape
                        if _noise_map.shape[0] > h:
                            _noise_map = _noise_map[:h]
                        if _noise_map.shape[1] > w:
                            _noise_map = _noise_map[:, :w]
                        if _noise_map.shape[0] < h or _noise_map.shape[1] < w:
                            _pad = np.full((h, w), scalar_noise)
                            _pad[:_noise_map.shape[0], :_noise_map.shape[1]] = _noise_map
                            _noise_map = _pad
                        return _noise_map

                    try:
                        _noise_box = int(
                            ts_cfg.get("zogy_noise_map_box_size", 64)
                        )
                    except (TypeError, ValueError):
                        _noise_box = 64
                    if _noise_box < 16:
                        logger.warning(
                            "zogy_noise_map_box_size=%r invalid; using 64 px.",
                            ts_cfg.get("zogy_noise_map_box_size"),
                        )
                        _noise_box = 64
                    _sn_map = _local_noise_map(science_data, _sn, _noise_box)
                    _sr_map = _local_noise_map(reference_data, _sr, _noise_box)
                    logger.info(
                        "ZOGY: computed per-pixel noise maps (box=%d px, "
                        "sci range=[%.4g, %.4g], ref range=[%.4g, %.4g]).",
                        _noise_box,
                        float(np.nanmin(_sn_map)), float(np.nanmax(_sn_map)),
                        float(np.nanmin(_sr_map)), float(np.nanmax(_sr_map)),
                    )
                except Exception as _e:
                    logger.info(
                        "ZOGY: per-pixel noise map computation failed (%s); "
                        "using scalar noise.", _e,
                    )
                    _sn_map = None
                    _sr_map = None

            # -----------------------------------------------------------------
            # PSF normalization: ensure PSFs sum to 1 before padding.
            # EPSFBuilder normalizes to unit sum, but the saved FITS may have
            # been re-scaled or truncated.  Log the sum for diagnostics.
            # -----------------------------------------------------------------
            _pn_sum = float(np.nansum(science_psf_data))
            _pr_sum = float(np.nansum(reference_psf_data))
            if _pn_sum > 0 and abs(_pn_sum - 1.0) > 0.01:
                logger.info(
                    "ZOGY: science PSF sum=%.4f (renormalizing to 1.0).", _pn_sum
                )
                science_psf_data = science_psf_data / _pn_sum
            if _pr_sum > 0 and abs(_pr_sum - 1.0) > 0.01:
                logger.info(
                    "ZOGY: reference PSF sum=%.4f (renormalizing to 1.0).", _pr_sum
                )
                reference_psf_data = reference_psf_data / _pr_sum

            # -----------------------------------------------------------------
            # PSF plausibility gate.  A degenerate ePSF (few stars, no
            # convergence) can carry a wildly wrong width -- 1.7 px against
            # a 9.6 px true PSF on TDP5_4 -- which turns either convolution
            # direction into a sharpening kernel and produces the dipole
            # residuals seen as oversubtraction.  Bright matched sources
            # measure the field PSF directly; a stamp more than ~15% off
            # that width (or carrying substantial negative mass) is
            # replaced by the star-stacked stamp, or by an analytic Moffat
            # at the best available width when no stack exists.
            # -----------------------------------------------------------------

            _min_psf_stars = int(
                ts_cfg.get("zogy_min_psf_stars", 3) or 3
            )

            def _moffat_stamp(fwhm):
                n = 2 * int(np.ceil(3.0 * fwhm)) + 1
                yy, xx = np.mgrid[0:n, 0:n]
                a = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / 3.5) - 1.0))
                m = (
                    1.0
                    + ((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / a**2
                ) ** -3.5
                return m / m.sum()

            def _repair_stamp(stamp, star_stamp, star_f, img_f, tag, star_n=0):
                sf = _measure_stamp_fwhm(stamp)
                # Negative mass is the signature of a sick stamp: ePSF
                # outskirts hover at +-noise and can collectively carry more
                # flux than the PSF itself, and the kernel built from such a
                # stamp inherits the ringing.  Healthy stamps sit at ~0.
                _tot = float(np.nansum(stamp))
                neg_mass = (
                    float(-np.nansum(stamp[stamp < 0]) / _tot)
                    if stamp.size and abs(_tot) > 1e-30
                    else np.inf
                )
                unsanitary = (
                    not np.isfinite(sf)
                    or sf < 1.0
                    or not np.isfinite(neg_mass)
                    or neg_mass > 0.25
                )
                sub, ref_f, sub_tag = None, 0.0, ""
                if (
                    star_stamp is not None
                    and np.isfinite(star_f)
                    and star_f > 0
                    and star_n >= _min_psf_stars
                ):
                    # A vetted stack of enough stars is the truest stamp:
                    # it carries the real resampled-frame PSF shape
                    # (wing flux, ellipticity) that an analytic model can
                    # get wrong by ~2x in the far wings even when its
                    # FWHM agrees (TDP6 60881 r: model-vs-stack shape
                    # error printed ~14% cores + negative rings on every
                    # bright star).  The kernel maps the stamp onto real
                    # stars, so feed it the real stars.
                    logger.info(
                        "ZOGY %s PSF: using star-stack stamp "
                        "(%d stars, FWHM %.2f px; model stamp %.2f px, "
                        "neg mass %.2f).",
                        tag, star_n, star_f,
                        sf if np.isfinite(sf) else -1.0,
                        neg_mass if np.isfinite(neg_mass) else -1.0,
                    )
                    return star_stamp, star_f
                if np.isfinite(star_f) and star_f > 0:
                    # The star stack is the most truthful field PSF and
                    # arrives pre-vetted; a stamp >15% off its width is a
                    # mismatch, not measurement noise.  Analytic-composite
                    # fits on sparse pools sit ~18% narrow (TDP5_10:
                    # 4.6 px stamp vs a 5.6 px stack), which used to slip
                    # under the old 25% gate and poisoned the kernel.
                    dev = (
                        abs(sf - star_f) / star_f if np.isfinite(sf) else np.inf
                    )
                    bad = unsanitary or dev > 0.15
                    sub, ref_f = star_stamp, star_f
                    sub_tag = "star-stack"
                else:
                    # With no star stack the image FWHM is the only
                    # reference, and it can be wrong on resampled data --
                    # only reject physically unsanitary stamps.
                    bad = unsanitary
                    if np.isfinite(img_f) and img_f > 0:
                        sub, ref_f = _moffat_stamp(img_f), img_f
                        sub_tag = "analytic-Moffat"
                if not bad or sub is None:
                    if bad:
                        logger.warning(
                            "ZOGY %s PSF stamp looks degenerate (FWHM %.2f, "
                            "neg mass %.2f) but no substitute is "
                            "available; keeping it.",
                            tag, sf if np.isfinite(sf) else -1.0,
                            neg_mass if np.isfinite(neg_mass) else -1.0,
                        )
                    return stamp, sf
                logger.warning(
                    "ZOGY %s PSF stamp rejected (FWHM %.2f vs reference "
                    "%.2f px, neg mass %.2f); substituting a %s "
                    "stamp for the subtraction inputs.",
                    tag, sf if np.isfinite(sf) else -1.0, ref_f,
                    neg_mass, sub_tag,
                )
                return sub, ref_f

            science_psf_data, _fwhm_s_psf = _repair_stamp(
                science_psf_data, _star_sci_stamp, _star_sci_f, _fs_f,
                "science",
                star_n=len(_star_pos) if _star_pos is not None else 0,
            )
            reference_psf_data, _fwhm_t_psf = _repair_stamp(
                reference_psf_data, _star_ref_stamp, _star_ref_f, _ft_f,
                "template",
                star_n=len(_star_ref_pos) if _star_ref_pos is not None else 0,
            )
            # Effective PSF widths drive the AUTO direction choice.  The
            # star-stack width is measured on the real field stars and is
            # the most truthful value when it exists -- even when the saved
            # stamp was kept (agreement within tolerance), the stack's
            # width is the better ordering statistic.
            if np.isfinite(_star_sci_f) and _star_sci_f > 0:
                _fwhm_s_psf = _star_sci_f
            if np.isfinite(_star_ref_f) and _star_ref_f > 0:
                _fwhm_t_psf = _star_ref_f
            _fwhm_s_psf = _fwhm_val(_fwhm_s_psf) or _fs_f
            _fwhm_t_psf = _fwhm_val(_fwhm_t_psf) or _ft_f

            # Persist the effective stamps actually handed to the kernel
            # builder -- the repair step may have substituted a star stack,
            # and post-mortem debugging needs the stamps that were used,
            # not the pre-repair models saved upstream.
            try:
                _eff_dir = os.path.dirname(str(differenceFpath))
                _eff_base = os.path.splitext(
                    os.path.basename(str(differenceFpath))
                )[0]
                for _eff_tag, _eff_arr, _eff_f in (
                    ("science", science_psf_data, _fwhm_s_psf),
                    ("template", reference_psf_data, _fwhm_t_psf),
                ):
                    fits.writeto(
                        os.path.join(
                            _eff_dir,
                            f"ZOGY_PSF_effective_{_eff_tag}_{_eff_base}.fits",
                        ),
                        np.asarray(_eff_arr, dtype=float),
                        fits.Header({"FWHMPIX": float(_eff_f or np.nan)}),
                        overwrite=True,
                    )
            except Exception as _e:
                logger.debug(
                    "ZOGY: could not write effective PSF stamps (%s).", _e
                )

            # Pad PSFs to the image shape (ZOGY requires same-shape FFTs)
            _psf_sci = _pad_psf_to_image(science_psf_data, science_data.shape)
            _psf_ref = _pad_psf_to_image(reference_psf_data, reference_data.shape)

            # -----------------------------------------------------------------
            # NaN handling: replace NaNs with local median instead of zero.
            # Zero-filling NaN regions creates artificial step functions at
            # NaN boundaries that produce FFT ringing artifacts in the
            # difference image.  Median-filling provides a smoother transition.
            # MUST come before pre-convolution so _sci_clean/_ref_clean exist.
            # -----------------------------------------------------------------
            _nan_mask = np.isfinite(science_data) & np.isfinite(reference_data)
            _nan_regions = ~_nan_mask
            # Masked-but-finite pixels (saturated cores, defects) must not
            # reach the FFT either: their non-linear flux prints as kernel
            # sidelobes across the frame.  They are filled like NaNs and
            # re-masked in the output difference image downstream.
            _fill_regions = (
                _nan_regions | _um if _um is not None else _nan_regions
            )

            if _fill_regions.any():
                _fill_method = str(
                    ts_cfg.get("zogy_nan_fill", "median")
                ).strip().lower()
                if _fill_method not in ("median", "zero"):
                    logger.warning(
                        "Unknown zogy_nan_fill=%r; using 'median'.",
                        _fill_method,
                    )
                    _fill_method = "median"
                if _fill_method == "median":
                    _sci_ok = np.isfinite(science_data) & ~_fill_regions
                    _ref_ok = np.isfinite(reference_data) & ~_fill_regions
                    _sci_med = float(
                        np.nanmedian(science_data[_sci_ok])
                        if _sci_ok.any()
                        else np.nanmedian(
                            science_data[np.isfinite(science_data)]
                        )
                    )
                    _ref_med = float(
                        np.nanmedian(reference_data[_ref_ok])
                        if _ref_ok.any()
                        else np.nanmedian(
                            reference_data[np.isfinite(reference_data)]
                        )
                    )
                    _sci_clean = np.where(
                        _fill_regions, _sci_med, science_data
                    )
                    _ref_clean = np.where(
                        _fill_regions, _ref_med, reference_data
                    )
                    _sci_clean = np.where(
                        np.isfinite(_sci_clean), _sci_clean, _sci_med
                    )
                    _ref_clean = np.where(
                        np.isfinite(_ref_clean), _ref_clean, _ref_med
                    )
                    logger.info(
                        "ZOGY: filled %d invalid/masked pixels with median "
                        "(sci=%.4g, ref=%.4g).",
                        int(_fill_regions.sum()), _sci_med, _ref_med,
                    )
                else:
                    _sci_clean = np.where(_fill_regions, 0.0, science_data)
                    _ref_clean = np.where(_fill_regions, 0.0, reference_data)
                    _sci_clean = np.nan_to_num(_sci_clean, nan=0.0)
                    _ref_clean = np.nan_to_num(_ref_clean, nan=0.0)
                    logger.info(
                        "ZOGY: filled %d invalid/masked pixels with zero.",
                        int(_fill_regions.sum()),
                    )
            else:
                _sci_clean = science_data.copy()
                _ref_clean = reference_data.copy()

            # -----------------------------------------------------------------
            # Pre-convolution: convolve one image to match the other's PSF
            # so the ZOGY difference image has a known, single PSF.
            #
            # With forceconv=REF, convolve the reference to match
            # the science PSF.  The difference image then has the science PSF,
            # so the science ePSF model can be used directly for photometry
            # without any PSF mismatch correction.  This matches the SFFT
            # forceconv=REF convention (Bramich 2008, Hu et al. 2022).
            # forceconv=REF is the pipeline default; AUTO picks the
            # sharper-to-broader direction per field.
            #
            # The convolution kernel in Fourier space is:
            #   K_hat = Pn_hat / Pr_hat  (transforms Pr -> Pn)
            # To avoid deconvolution noise when the reference PSF is broader
            # (high-frequency zeros in Pr_hat), we use Wiener-like
            # regularization: K_hat = Pn_hat * conj(Pr_hat) / (|Pr_hat|^2 + eps)
            # -----------------------------------------------------------------
            _zogy_fc_cfg = str(
                ts_cfg.get("forceconv", ts_cfg.get("zogy_forceconv", "REF"))
            ).strip().upper()
            if _zogy_fc_cfg not in ("REF", "SCI", "AUTO"):
                logger.warning(
                    "Unknown forceconv=%r for ZOGY; defaulting to REF.",
                    _zogy_fc_cfg,
                )
                _zogy_fc_cfg = "REF"
            _zogy_convolved = None  # track which image was convolved

            # Probe both candidate Wiener kernels before resolving the
            # direction.  AUTO's only job is to avoid a sharpening kernel,
            # and the honest test is the kernel's own negative mass rather
            # than any FWHM estimate: resampled-template image FWHMs and
            # plateaued ePSF stamp widths both lie.  On TDP6_12 the image
            # FWHMs (sci 7.65 > ref 7.30) resolved REF while the PSF models
            # required sharpening -- the resulting kernel carried ~85%
            # negative mass and left dipole residuals at every bright star.
            #
            # The kernels are built from wing-tapered stamps and windowed
            # to compact support: stamp noise beyond ~4 FWHM and the padded
            # kernel's far-field ringing otherwise print as sidelobe
            # residuals around bright sources (SN2024pba 2025-02-22 r).
            _kern_win = 0
            try:
                _kern_win = int(kernel_half_width) if kernel_half_width else 0
            except (TypeError, ValueError):
                _kern_win = 0
            _kern_win = max(
                _kern_win,
                int(np.ceil(3.0 * max(_fwhm_s_psf, _fwhm_t_psf))),
                3,
            )
            _psf_sci_k = _pad_psf_to_image(
                _zogy_taper_stamp(science_psf_data, _fwhm_s_psf),
                science_data.shape,
            )
            _psf_ref_k = _pad_psf_to_image(
                _zogy_taper_stamp(reference_psf_data, _fwhm_t_psf),
                reference_data.shape,
            )

            # Kernel-solver configuration.  Two candidates are probed per
            # direction: the Wiener PSF-ratio kernel (adaptive
            # regularization) and the pixel-domain LSQ kernel, which is
            # the Bramich/SFFT delta basis with a Laplacian Tikhonov
            # penalty -- it absorbs sub-pixel misregistration that the
            # PSF-ratio kernel cannot represent.
            _lsq_on = _as_bool(ts_cfg.get("zogy_lsq_kernel", True), True)
            try:
                _lsq_lam = float(ts_cfg.get("zogy_lsq_lambda", 1e-3) or 1e-3)
            except (TypeError, ValueError):
                _lsq_lam = 1e-3
            try:
                _lsq_hwcap = int(ts_cfg.get("zogy_lsq_kernel_hw_max", 12) or 12)
            except (TypeError, ValueError):
                _lsq_hwcap = 12
            try:
                _lsq_min = int(ts_cfg.get("zogy_lsq_min_sources", 3) or 3)
            except (TypeError, ValueError):
                _lsq_min = 3
            _adapt_eps = _as_bool(
                ts_cfg.get("zogy_wiener_adaptive_eps", True), True
            )
            _kfit_xy = _zogy_lsq_fit_positions(
                matching_sources, _star_pos, _star_ref_pos
            )
            _kern_meta = {}

            def _build_dir_kernel(_d):
                """Best matching kernel for one direction.

                The veto diagnostic must be an honest deconvolution test,
                so the reported negative mass always comes from the
                fixed-eps Wiener probe (or the LSQ kernel when it is
                selected): strong Wiener damping turns a sharpening
                kernel into a smooth correlation bump whose clean neg
                mass would otherwise smuggle a deconvolving direction
                past the veto.  The adaptive-eps kernel is therefore only
                used as the applied kernel, never as the diagnostic.

                The LSQ pixel-domain kernel is preferred when its
                sidelobe mass stays under the LSQ cap and the Wiener
                probe does not flag deconvolution: it is the only
                candidate that can compensate residual misalignment.
                """
                _src = _psf_ref_k if _d == "REF" else _psf_sci_k
                _tgt = _psf_sci_k if _d == "REF" else _psf_ref_k
                _meta = {"method": "none", "b0": 0.0}
                _kern_meta[_d] = _meta
                _eps_w = 1e-2
                _kw_full = None
                _neg_w = np.inf
                try:
                    # Fixed-eps probe: the veto's deconvolution test.
                    _kp, _neg_w = _zogy_wiener_kernel(_src, _tgt)
                    if (
                        _kp is None
                        or not np.isfinite(_kp).all()
                        or not np.isfinite(_neg_w)
                    ):
                        _neg_w = np.inf
                    else:
                        _ksum = float(np.sum(_kp))
                        if not (np.isfinite(_ksum) and abs(_ksum) > 1e-30):
                            _neg_w = np.inf
                except Exception as _e:
                    logger.info(
                        "ZOGY: %s-direction kernel probe failed (%s).",
                        _d, _e,
                    )
                    _neg_w = np.inf
                # Applied Wiener kernel: weakest damping that stays
                # clean AND faithful - the ladder only accepts a
                # candidate whose realised convolution reproduces the
                # target width (over-damped kernels look clean while
                # over-broadening the result by ~20%, which prints
                # positive-core/negative-ring residuals at every
                # source).  Its neg mass is logged but not used for the
                # veto (over-damped kernels look artificially clean).
                _fid_tol = float(
                    ts_cfg.get("zogy_kernel_fidelity_tol", 0.15) or 0.15
                )
                _neg_app_cap = float(
                    ts_cfg.get("zogy_kernel_applied_neg_cap", 0.30) or 0.30
                )
                if _adapt_eps:
                    try:
                        _ka, _neg_ka, _eps_w = _zogy_wiener_kernel_adaptive(
                            _src, _tgt,
                            half_width=_kern_win,
                            fid_tol=_fid_tol,
                            neg_cap=_neg_app_cap,
                        )
                    except Exception:
                        _ka = None
                    if _ka is not None and np.isfinite(_ka).all():
                        _kw_full = _zogy_window_kernel(_ka, _kern_win)
                else:
                    if np.isfinite(_neg_w):
                        _kw_full = _zogy_window_kernel(_kp, _kern_win)
                _meta["wiener_eps"] = _eps_w

                # Applied-kernel screens on the windowed kernel - the
                # part that actually prints onto the convolved image:
                #
                # 1. Fidelity: the convolved source must reproduce the
                #    target width.  Also rejects the adaptive ladder's
                #    last-resort return, which is the least-negative
                #    candidate even when no candidate stayed faithful.
                # 2. Sidelobe mass, but only when the direction does
                #    real convolution work: a faithful kernel can still
                #    carry deep negative lobes when the PSF mismatch
                #    demands genuine wing deconvolution (TDP4 59585 r:
                #    neg 0.44 at the faithful damping level), and
                #    applying it rings every bright source.  The cap is
                #    waived for width-matched stamp pairs, where the
                #    kernel is a near-idempotent blob whose sidelobe
                #    "mass" is regularisation ripple - the
                #    _near_match_ringy probe veto already guards that
                #    case at the direction level.
                # The direction then falls through to the LSQ candidate
                # or the canonical-ZOGY veto.
                if _kw_full is not None:
                    _neg_app = float(
                        -np.sum(_kw_full[_kw_full < 0])
                    )
                    _meta["wiener_applied_neg"] = _neg_app
                    _fid_err = _zogy_kernel_fid_err(_src, _kw_full, _tgt)
                    _meta["wiener_fid_err"] = _fid_err
                    _fw_s = _fwhm_t_psf if _d == "REF" else _fwhm_s_psf
                    _fw_t = _fwhm_s_psf if _d == "REF" else _fwhm_t_psf
                    _fc_tol0 = float(
                        ts_cfg.get("sfft_forceconv_auto_tol", 0.05)
                        or 0.05
                    )
                    _dir_near = (
                        _fw_s > 0
                        and _fw_t > 0
                        and abs(_fw_s - _fw_t)
                        <= _fc_tol0 * max(_fw_s, _fw_t)
                    )
                    if not np.isfinite(_fid_err) or _fid_err > _fid_tol:
                        try:
                            _conv_dbg = np.real(
                                np.fft.ifft2(
                                    np.fft.fft2(_src) * np.fft.fft2(_kw_full)
                                )
                            )
                            _fc_dbg = _zerolag_stamp_fwhm(_conv_dbg)
                            _ft_dbg = _zerolag_stamp_fwhm(_tgt)
                        except Exception:
                            _fc_dbg = _ft_dbg = float("nan")
                        logger.info(
                            "ZOGY: %s-direction Wiener kernel fails the "
                            "fidelity check (convolved %.2f px vs target "
                            "%.2f px, %.1f%% off, limit %.1f%%); not usable.",
                            _d,
                            _fc_dbg,
                            _ft_dbg,
                            100.0 * _fid_err if np.isfinite(_fid_err) else -1.0,
                            100.0 * _fid_tol,
                        )
                        _kw_full = None
                    elif (
                        not _dir_near
                        and (
                            not np.isfinite(_neg_app)
                            or _neg_app > _neg_app_cap
                        )
                    ):
                        logger.info(
                            "ZOGY: %s-direction Wiener kernel carries "
                            "%.0f%% negative sidelobe mass within the "
                            "applied window (cap %.0f%%); it would ring "
                            "at every bright source - refusing it.",
                            _d,
                            100.0 * _neg_app if np.isfinite(_neg_app) else -1.0,
                            100.0 * _neg_app_cap,
                        )
                        _kw_full = None

                _kl_full = None
                _neg_l = np.inf
                _kc = None
                _linfo = {}
                _lsq_cap = float(
                    ts_cfg.get("zogy_lsq_neg_cap", 0.45) or 0.45
                )
                if _lsq_on and _kfit_xy:
                    _s_img = _ref_clean if _d == "REF" else _sci_clean
                    _t_img = _sci_clean if _d == "REF" else _ref_clean
                    try:
                        _kc, _linfo = _zogy_lsq_kernel(
                            _s_img,
                            _t_img,
                            _kfit_xy,
                            min(_kern_win, _lsq_hwcap),
                            invalid_mask=_fill_regions,
                            lam_rel=_lsq_lam,
                            min_stamps=_lsq_min,
                            neg_cap=_lsq_cap,
                        )
                    except Exception as _e:
                        logger.info(
                            "ZOGY: %s-direction LSQ kernel solve failed "
                            "(%s).",
                            _d, _e,
                        )
                        _kc, _linfo = None, {}
                    if _kc is not None:
                        _kl_full = _pack_lsq_kernel(
                            _kc, _sci_clean.shape
                        )
                        _neg_l = float(_linfo.get("neg", np.inf))
                        _meta["b0"] = float(_linfo.get("b0", 0.0))
                        _meta["ksum_raw"] = _linfo.get("ksum_raw", np.nan)
                        _meta["n_stamps"] = _linfo.get("n_stamps", 0)
                        _meta["n_pix"] = _linfo.get("n_pix", 0)
                        _meta["resid_ratio"] = _linfo.get(
                            "resid_ratio", np.nan
                        )

                # Prefer the LSQ kernel below a sanity cap: a smoothing
                # kernel that absorbs a sub-pixel shift legitimately
                # carries ~0.05-0.3 sidelobe mass, so comparing directly
                # against the Wiener candidate's neg mass would always
                # pick the shape-only kernel that cannot fix the
                # misregistration.  A truly impossible (sharpening) fit
                # shows neg >> this cap (measured ~0.9 on the real ZTF
                # epoch, ~5 on synthetic TDP-like data).  The fixed-eps
                # Wiener probe gates the LSQ choice as well: when the
                # PSF-ratio kernel screams deconvolution, no pixel-domain
                # fit is trusted for that direction either.
                _lsq_rr_max = float(
                    ts_cfg.get("zogy_lsq_resid_cap", 0.5) or 0.5
                )
                # Width-fidelity check on the LSQ candidate too: a fit
                # that passes neg/resid caps but lands its convolved PSF
                # well off the target width is not trustworthy.  NaN
                # (unmeasurable) is tolerated -- the pixel residual is
                # still evidence.
                _fid_l = (
                    _zogy_kernel_fid_err(_src, _kl_full, _tgt)
                    if _kl_full is not None
                    else np.nan
                )
                _meta["lsq_fid_err"] = _fid_l
                _lsq_ok = (
                    _kl_full is not None
                    and _neg_l <= _lsq_cap
                    and (
                        not np.isfinite(_neg_w)
                        or _neg_w <= _ZOGY_KERNEL_NEG_HARD
                    )
                    and (
                        np.isnan(_meta.get("resid_ratio", np.nan))
                        or _meta["resid_ratio"] <= _lsq_rr_max
                    )
                    and (
                        not np.isfinite(_fid_l)
                        or _fid_l <= max(_fid_tol, 0.20)
                    )
                )
                if not _lsq_ok:
                    # An absent or rejected LSQ candidate used to be
                    # silent, leaving no trace of why the Wiener path
                    # won; log the deciding term.
                    if _kc is None:
                        logger.debug(
                            "ZOGY: %s-direction LSQ kernel unavailable "
                            "(n_stamps=%s, ksum=%s).",
                            _d,
                            _linfo.get("n_stamps", "?"),
                            _linfo.get("ksum_raw", "?"),
                        )
                    elif not _lsq_ok:
                        logger.info(
                            "ZOGY: %s-direction LSQ kernel rejected "
                            "(neg=%.3f cap %.2f, fid=%.3f, "
                            "resid_ratio=%.3f cap %.2f, lam_rel=%s, "
                            "probe neg %.3f).",
                            _d,
                            _neg_l,
                            _lsq_cap,
                            _fid_l,
                            _meta.get("resid_ratio", float("nan")),
                            _lsq_rr_max,
                            f"{_linfo.get('lam_rel'):.3g}"
                            if _linfo.get("lam_rel") is not None
                            else "?",
                            _neg_w,
                        )
                if _lsq_ok:
                    _meta["method"] = "lsq"
                    _meta["lam_rel"] = _linfo.get("lam_rel")
                    return _kl_full, _neg_l
                if _kw_full is not None:
                    _meta["method"] = "wiener"
                    _meta["b0"] = 0.0
                return _kw_full, _neg_w

            _neg_by_dir = {}
            _kernel_by_dir = {}
            for _dirn in ("REF", "SCI"):
                _k, _neg = _build_dir_kernel(_dirn)
                _neg_by_dir[_dirn] = _neg
                if _k is not None:
                    _kernel_by_dir[_dirn] = _k

            if _zogy_fc_cfg == "AUTO" and len(_neg_by_dir) == 2:
                _neg_ref = _neg_by_dir["REF"]
                _neg_sci = _neg_by_dir["SCI"]
                _lsq_cap_auto = float(
                    ts_cfg.get("zogy_lsq_neg_cap", 0.45) or 0.45
                )

                def _dir_tol(_d):
                    return (
                        _lsq_cap_auto
                        if _kern_meta.get(_d, {}).get("method") == "lsq"
                        else _ZOGY_KERNEL_NEG_TOL
                    )

                _ref_ok = _neg_ref <= _dir_tol("REF")
                _sci_ok = _neg_sci <= _dir_tol("SCI")
                if _ref_ok != _sci_ok:
                    # Exactly one kernel is a clean smoother -- trust the
                    # probe over any width estimate.
                    _zogy_forceconv = "REF" if _ref_ok else "SCI"
                    _zogy_fc_note = (
                        f"AUTO->{_zogy_forceconv}: kernel probe picked "
                        "the smoothing direction "
                        f"(neg mass REF={_neg_ref:.3f}, "
                        f"SCI={_neg_sci:.3f})"
                    )
                else:
                    # Both clean, or both carry sidelobes (PSF-shape
                    # mismatch rather than a width inversion): defer to
                    # the effective-PSF width ordering, which is the
                    # truthful direction after stamp repair.
                    (
                        _zogy_forceconv,
                        _zogy_fc_deconvolves,
                        _zogy_fc_note,
                    ) = _select_forceconv(
                        "AUTO", _fwhm_s_psf, _fwhm_t_psf,
                        auto_tol=float(
                            ts_cfg.get("sfft_forceconv_auto_tol", 0.05) or 0.05
                        ),
                    )
                    _zogy_fc_note += (
                        f" [kernel neg mass REF={_neg_ref:.3f}, "
                        f"SCI={_neg_sci:.3f}]"
                    )
            else:
                (
                    _zogy_forceconv,
                    _zogy_fc_deconvolves,
                    _zogy_fc_note,
                ) = _select_forceconv(
                    _zogy_fc_cfg,
                    _fwhm_s_psf,
                    _fwhm_t_psf,
                    auto_tol=float(
                        ts_cfg.get("sfft_forceconv_auto_tol", 0.05) or 0.05
                    ),
                )

            # Deconvolution guard.  A matching kernel that has to sharpen
            # the convolved image -- or whose sidelobes are extreme no
            # matter what the FWHMs claim -- prints dipole residuals at
            # every source rather than cancelling them.  Never apply such
            # a kernel: flip to the opposite direction when its kernel is
            # cleaner, otherwise skip pre-convolution entirely so the
            # canonical ZOGY equation handles the PSF mismatch natively
            # (the diff PSF then becomes the geometric-mean P_D).  A
            # broadening kernel with moderate sidelobes from noisy stamps
            # is still usable -- the direction is well-posed.
            _fc_tol = float(
                ts_cfg.get("sfft_forceconv_auto_tol", 0.05) or 0.05
            )
            if _zogy_forceconv in ("REF", "SCI"):
                _order_sharp = (
                    (
                        _zogy_forceconv == "REF"
                        and _fwhm_t_psf > _fwhm_s_psf * (1.0 + _fc_tol)
                    )
                    or (
                        _zogy_forceconv == "SCI"
                        and _fwhm_s_psf > _fwhm_t_psf * (1.0 + _fc_tol)
                    )
                )
                _neg_sel = _neg_by_dir.get(_zogy_forceconv)
                # An LSQ kernel carrying moderate sidelobe mass from
                # shift compensation is not a ringy Wiener kernel; it is
                # only untrustworthy beyond the LSQ cap.
                _neg_tol_sel = (
                    float(ts_cfg.get("zogy_lsq_neg_cap", 0.45) or 0.45)
                    if _kern_meta.get(_zogy_forceconv, {}).get("method")
                    == "lsq"
                    else _ZOGY_KERNEL_NEG_TOL
                )
                _ringy = _neg_sel is None or _neg_sel > _neg_tol_sel
                _extreme = (
                    _neg_sel is not None and _neg_sel > _ZOGY_KERNEL_NEG_HARD
                )
                _near_match = (
                    _fs_f > 0
                    and _ft_f > 0
                    and _fwhm_s_psf > 0
                    and _fwhm_t_psf > 0
                    and abs(_fs_f - _ft_f) <= _fc_tol * max(_fs_f, _ft_f)
                    and abs(_fwhm_s_psf - _fwhm_t_psf)
                    <= _fc_tol * max(_fwhm_s_psf, _fwhm_t_psf)
                )
                _near_match_ringy = (
                    _near_match
                    and _neg_sel is not None
                    and np.isfinite(_neg_sel)
                    and _neg_sel > _ZOGY_NEAR_MATCH_KERNEL_NEG_LIMIT
                )
                _veto = (
                    _zogy_forceconv not in _kernel_by_dir
                    or (_order_sharp and _ringy)
                    or _extreme
                    or _near_match_ringy
                )
                if _veto:
                    _opp = "SCI" if _zogy_forceconv == "REF" else "REF"
                    _neg_opp = _neg_by_dir.get(_opp)
                    _neg_cap_opp = (
                        _ZOGY_FLIP_LSQ_NEG_CAP
                        if _kern_meta.get(_opp, {}).get("method") == "lsq"
                        else _ZOGY_KERNEL_NEG_TOL
                    )
                    _near_match_both_ringy = (
                        _near_match_ringy
                        and (
                            _neg_opp is None
                            or not np.isfinite(_neg_opp)
                            or _neg_opp > _ZOGY_NEAR_MATCH_KERNEL_NEG_LIMIT
                        )
                    )
                    if _near_match_both_ringy:
                        logger.warning(
                            "ZOGY: PSF widths agree within %.1f%% but both "
                            "matching kernels are ringy (negative mass %.2f/%s); "
                            "skipping pre-convolution and using canonical ZOGY.",
                            100.0 * _fc_tol,
                            _neg_sel,
                            f"{_neg_opp:.2f}"
                            if _neg_opp is not None and np.isfinite(_neg_opp)
                            else "unavailable",
                        )
                        _zogy_fc_note += (
                            "; near-matched PSFs with ringy kernels, "
                            "canonical ZOGY"
                        )
                        _zogy_forceconv = "NONE"
                        _zogy_fc_deconvolves = False
                    elif (
                        _opp in _kernel_by_dir
                        and _neg_opp is not None
                        and np.isfinite(_neg_opp)
                        and _neg_opp <= _neg_cap_opp
                        and (_neg_sel is None or _neg_opp < _neg_sel)
                    ):
                        logger.warning(
                            "ZOGY forceconv=%s requires deconvolution or a\n"
                            "    ringy kernel (neg mass %s); overriding to\n"
                            "    forceconv=%s whose kernel is cleaner (neg\n"
                            "    mass %.2f).  The difference image carries\n"
                            "    the %s PSF.",
                            _zogy_forceconv,
                            "%.2f" % _neg_sel
                            if _neg_sel is not None
                            else "non-finite",
                            _opp,
                            _neg_opp,
                            "reference" if _opp == "SCI" else "science",
                        )
                        _zogy_fc_note += (
                            f"; overridden to {_opp} (kernel neg mass "
                            f"{_neg_opp:.2f})"
                        )
                        _zogy_forceconv = _opp
                        _zogy_fc_deconvolves = False
                    else:
                        logger.warning(
                            "ZOGY forceconv=%s requires deconvolution or\n"
                            "    a ringy kernel (neg mass %s) and the %s\n"
                            "    direction is no better (neg mass %s);\n"
                            "    skipping pre-convolution -- canonical ZOGY\n"
                            "    handles the PSF mismatch natively and the\n"
                            "    diff PSF becomes the geometric-mean P_D.",
                            _zogy_forceconv,
                            "%.2f" % _neg_sel
                            if _neg_sel is not None
                            else "non-finite",
                            _opp,
                            "%.2f" % _neg_opp
                            if _neg_opp is not None
                            else "unusable",
                        )
                        _zogy_fc_note += (
                            "; pre-convolution vetoed (ringy kernel), "
                            "canonical ZOGY"
                        )
                        _zogy_forceconv = "NONE"
                        _zogy_fc_deconvolves = False

            # When a kernel was probed, its negative mass is the ground
            # truth for whether the resolved direction deconvolves --
            # regardless of what the FWHM ordering claimed.  An LSQ
            # kernel legitimately carries ~0.05-0.15 sidelobe mass when
            # it absorbs a sub-pixel shift, so it is judged against the
            # LSQ cap instead of the Wiener tolerance.
            if _zogy_forceconv in _neg_by_dir:
                _neg_gate = (
                    float(ts_cfg.get("zogy_lsq_neg_cap", 0.45) or 0.45)
                    if _kern_meta.get(_zogy_forceconv, {}).get("method")
                    == "lsq"
                    else _ZOGY_KERNEL_NEG_TOL
                )
                _zogy_fc_deconvolves = (
                    _neg_by_dir[_zogy_forceconv] > _neg_gate
                )
            logger.info(
                "ZOGY forceconv=%s (cfg=%s, eff FWHM: sci=%.2f ref=%.2f, "
                "image FWHM: %.2f/%.2f; %s).",
                _zogy_forceconv, _zogy_fc_cfg, _fwhm_s_psf, _fwhm_t_psf,
                _fs_f, _ft_f, _zogy_fc_note,
            )
            if _zogy_fc_deconvolves:
                _neg_msg = ""
                if _zogy_forceconv in _neg_by_dir:
                    _neg_msg = (
                        f"  Kernel negative mass: {_neg_by_dir[_zogy_forceconv]:.2f}."
                    )
                logger.warning(
                    "ZOGY forceconv=%s applies a kernel with substantial\n"
                    "    negative mass (%.2f px -> %.2f px) -- a noisy\n"
                    "    broadening kernel, not a sharpening one, but its\n"
                    "    sidelobes can still leave residuals at bright\n"
                    "    sources.%s",
                    _zogy_forceconv,
                    _fwhm_t_psf if _zogy_forceconv == "REF" else _fwhm_s_psf,
                    _fwhm_s_psf if _zogy_forceconv == "REF" else _fwhm_t_psf,
                    _neg_msg,
                )

            _source_var_n = np.maximum(_sci_clean, 0.0) / science_gain
            _source_var_r = (
                np.maximum(_ref_clean, 0.0) * _flux_scale / template_gain
            )
            _source_var_n[_fill_regions] = 0.0
            _source_var_r[_fill_regions] = 0.0

            if _zogy_forceconv in ("REF", "SCI") and _fwhm_s_psf and _fwhm_t_psf:
                try:
                    # Reuse the probed kernel when available; the direction
                    # was already decided from its negative mass.
                    _conv_kernel = _kernel_by_dir.get(_zogy_forceconv)
                    if _conv_kernel is None:
                        _conv_kernel, _ = _build_dir_kernel(_zogy_forceconv)
                    if _conv_kernel is None or not np.isfinite(_conv_kernel).all():
                        raise ValueError("non-finite matching kernel")
                    _kmeta = _kern_meta.get(_zogy_forceconv, {})
                    _kmeth = _kmeta.get("method", "wiener")
                    _eps_used = (
                        _kmeta.get("wiener_eps", 1e-2)
                        if _kmeth != "lsq"
                        else _kmeta.get("lam_rel") or _lsq_lam
                    )
                    # The LSQ solve carries a differential-background term:
                    # target ~= K*source + b0, so b0 is added to the
                    # convolved image for the model to reproduce the
                    # target side in both directions.
                    _b0 = float(_kmeta.get("b0", 0.0) or 0.0)
                    if not np.isfinite(_b0):
                        _b0 = 0.0
                    # The raw LSQ kernel sum is a free photometric factor
                    # (SFFT ConstPhotRatio equivalent): fold it into the
                    # flux scale so residual scale error is absorbed by
                    # the kernel rather than printed as a monopole
                    # residual at every source.  REF: the raw kernel is
                    # applied to the already-scaled reference, so the
                    # effective scale is _flux_scale * s.  SCI: convolve
                    # science with the unit-normalized kernel (the
                    # transient flux must keep the science scale) and
                    # rescale the reference term by 1/s.
                    if _kmeth == "lsq":
                        _ksum_l = float(_kmeta.get("ksum_raw", 1.0) or 1.0)
                        if not np.isfinite(_ksum_l) or _ksum_l <= 0:
                            _ksum_l = 1.0
                        if _zogy_forceconv == "SCI":
                            _conv_kernel = _conv_kernel / _ksum_l
                            _ref_clean = _ref_clean / _ksum_l
                            _sr = _sr / _ksum_l
                            _source_var_r = _source_var_r / _ksum_l**2
                            if _sr_map is not None:
                                _sr_map = _sr_map / _ksum_l
                            _b0 = _b0 / _ksum_l
                            _flux_scale = _flux_scale / _ksum_l
                        else:
                            _flux_scale = _flux_scale * _ksum_l
                        if abs(_ksum_l - 1.0) > 0.15:
                            logger.warning(
                                "ZOGY: LSQ kernel sum %.3f deviates >15%% "
                                "from unity -- the prior flux scale was "
                                "off; folded into the effective scale "
                                "(%.4g).",
                                _ksum_l, _flux_scale,
                            )
                        else:
                            logger.info(
                                "ZOGY: LSQ kernel sum %.3f folded into "
                                "flux scale (effective %.4g).",
                                _ksum_l, _flux_scale,
                            )

                    # Convolution noise propagation: for white pixel noise,
                    # convolving with kernel k scales the per-pixel RMS by
                    # sqrt(sum(k^2)).  The ZOGY denominator and Scorr variance
                    # must use the POST-convolution noise, otherwise the
                    # matched filter is mis-weighted and significances are
                    # overestimated (deconvolution) or underestimated
                    # (smoothing).  This captures the marginal noise; the
                    # induced pixel-pixel correlation remains unmodeled.
                    _conv_noise_factor = float(
                        np.sqrt(np.sum(_conv_kernel ** 2))
                    )
                    if not np.isfinite(_conv_noise_factor) or _conv_noise_factor <= 0:
                        _conv_noise_factor = 1.0

                    if _zogy_forceconv == "REF":
                        # Convolve reference to match science PSF
                        _ref_clean_conv = np.real(
                            np.fft.ifft2(
                                np.fft.fft2(_ref_clean) * np.fft.fft2(_conv_kernel)
                            )
                        )
                        _ref_clean = _ref_clean_conv + _b0
                        _source_var_r = _convolve_variance_map(
                            _source_var_r, _conv_kernel
                        )
                        # Now both PSFs are the science PSF
                        _psf_ref = _psf_sci.copy()
                        _zogy_convolved = "REF"
                        _sr = _sr * _conv_noise_factor
                        if _sr_map is not None:
                            _sr_map = _sr_map * _conv_noise_factor
                        logger.info(
                            "ZOGY: convolved reference to science PSF "
                            "(FWHM %.2f -> %.2f px, kernel=%s, %s=%.2g, "
                            "b0=%.4g). "
                            "Reference noise scaled by sqrt(sum k^2)=%.3f.",
                            _fwhm_t_psf, _fwhm_s_psf, _kmeth,
                            "eps" if _kmeth != "lsq" else "lam",
                            _eps_used,
                            _b0, _conv_noise_factor,
                        )
                    else:
                        # Convolve science to match reference PSF
                        _sci_clean_conv = np.real(
                            np.fft.ifft2(
                                np.fft.fft2(_sci_clean) * np.fft.fft2(_conv_kernel)
                            )
                        )
                        _sci_clean = _sci_clean_conv + _b0
                        _source_var_n = _convolve_variance_map(
                            _source_var_n, _conv_kernel
                        )
                        _psf_sci = _psf_ref.copy()
                        _zogy_convolved = "SCI"
                        _sn = _sn * _conv_noise_factor
                        if _sn_map is not None:
                            _sn_map = _sn_map * _conv_noise_factor
                        logger.info(
                            "ZOGY: convolved science to reference PSF "
                            "(FWHM %.2f -> %.2f px, kernel=%s, %s=%.2g, "
                            "b0=%.4g). "
                            "Science noise scaled by sqrt(sum k^2)=%.3f.",
                            _fwhm_s_psf, _fwhm_t_psf, _kmeth,
                            "eps" if _kmeth != "lsq" else "lam",
                            _eps_used,
                            _b0, _conv_noise_factor,
                        )
                except Exception as _conv_e:
                    logger.info(
                        "ZOGY: pre-convolution failed (%s); using standard "
                        "ZOGY (geometric-mean PSF).", _conv_e,
                    )
                    _zogy_convolved = None
            elif _zogy_forceconv in ("REF", "SCI"):
                logger.info(
                    "ZOGY: pre-convolution skipped (missing FWHM values); "
                    "using standard ZOGY (geometric-mean PSF)."
                )

            logger.info(
                "Running ZOGY subtraction (sr=%.4g, sn=%.4g, flux_scale=%.4g, "
                "noise_maps=%s, sky_sub=%s)...",
                _sr, _sn, _flux_scale,
                "yes" if _sn_map is not None else "no",
                "yes" if _zogy_sky_subtract else "no",
            )
            # nan_mask=None: the NaN regions have already been filled
            # (median or zero) above.  Passing _nan_regions would zero the
            # median-filled pixels and recreate the step-function boundaries
            # that median-filling is meant to avoid (FFT ringing).  The
            # (R==0)|(N==0) fallback inside _zogy_subtract still catches any
            # remaining exact-zero pixels, and the caller re-imposes NaN on
            # the output difference image.
            _D, _S, _Scorr, _P_D = _zogy_subtract(
                _sci_clean, _ref_clean, _psf_sci, _psf_ref, _sn, _sr,
                sn_map=_sn_map,
                sr_map=_sr_map,
                nan_mask=None,
                gain_n=science_gain,
                gain_r=template_gain / _flux_scale,
                source_var_n=_source_var_n,
                source_var_r=_source_var_r,
            )
            if _zogy_convolved is not None:
                # After pre-convolution both images share (approximately) the
                # same PSF, so ZOGY's Eq. 1 reduces algebraically to the plain
                # difference N - R_conv divided by the common noise.  The
                # Fourier ratio P_hat/|P_hat| contributes nothing except phase
                # ringing wherever the empirical PSF's transform is not
                # strictly positive: on real data this smears sharp unmasked
                # sources (cosmic rays, detector artifacts) into extended
                # dipole blobs.  The direct difference is the honest output
                # for the claimed CONVD PSF and keeps such features compact.
                diff_image = _sci_clean - _ref_clean
            else:
                diff_image = _D
            # Add ZOGY metadata to the header BEFORE writing so it survives
            # the subsequent background-zeroing rewrite in subtract().
            # When pre-convolution was applied (CONVD=REF or SCI), the
            # difference image has that image's PSF, not the geometric mean.
            _zogy_hdr = scienceHeader.copy()
            _zogy_hdr["FORCECON"] = "ZOGY"
            _zogy_hdr["FWHM_SCI"] = _fwhm_s_psf
            _zogy_hdr["FWHM_REF"] = _fwhm_t_psf
            _zogy_hdr["FLXSCALE"] = (
                float(_flux_scale),
                "effective template->science flux scale applied",
            )
            # Same provenance cards the SFFT path writes: the measured
            # photometric scale versus the scale actually realised by the
            # (LSQ-folded) convolution.  The quality JSON reads these.
            _zogy_hdr["FSCAL_PHOT"] = (
                float(_flux_scale_phot),
                "measured photometric template->science flux scale",
            )
            _zogy_hdr["FSCAL_CONV"] = (
                float(_flux_scale),
                "realised convolution template->science flux scale",
            )
            _fs_disc = 0.0
            if np.isfinite(_flux_scale_phot) and np.isfinite(_flux_scale):
                _fs_disc = float(
                    abs(_flux_scale - _flux_scale_phot)
                    / max(abs(_flux_scale), abs(_flux_scale_phot), 1e-10)
                    * 100.0
                )
            _zogy_hdr["FSCAL_DISC"] = (
                _fs_disc,
                "photometric vs convolution flux scale discrepancy [%]",
            )
            if _zogy_convolved == "REF":
                # Reference was convolved to science PSF -> diff has science PSF
                _zogy_hdr["CONVD"] = "REF"
                _zogy_hdr["FWHM"] = _fwhm_s_psf
                _zogy_hdr["DIFFFWHM"] = _fwhm_s_psf
                _diff_stamp_src = science_psf_data
            elif _zogy_convolved == "SCI":
                # Science was convolved to reference PSF -> diff has reference PSF
                _zogy_hdr["CONVD"] = "SCI"
                _zogy_hdr["FWHM"] = _fwhm_t_psf
                _zogy_hdr["DIFFFWHM"] = _fwhm_t_psf
                _diff_stamp_src = reference_psf_data
            else:
                # Standard ZOGY: geometric mean PSF
                _zogy_hdr["CONVD"] = "ZOGY"
                _zogy_fwhm = 0.0
                if _fwhm_s_psf and _fwhm_t_psf:
                    _zogy_fwhm = np.sqrt(_fwhm_s_psf * _fwhm_t_psf)
                    _zogy_hdr["FWHM"] = _zogy_fwhm
                    _zogy_hdr["DIFFFWHM"] = _zogy_fwhm
                _diff_stamp_src = None

            # Kernel provenance for QA: which solver produced the applied
            # kernel, its sidelobe mass, and the LSQ fit diagnostics (raw
            # kernel sum = free photometric factor; b0 = fitted
            # differential background; n_stamps/n_pix = fit support).
            _km_sel = _kern_meta.get(_zogy_convolved, {})
            if _zogy_convolved in ("REF", "SCI") and _km_sel:
                _zogy_hdr["KERNMETH"] = str(_km_sel.get("method", "wiener"))
                if _km_sel.get("method") == "lsq":
                    _zogy_hdr["KERNSUM"] = float(
                        _km_sel.get("ksum_raw", np.nan)
                    )
                    _zogy_hdr["KERNB0"] = float(_km_sel.get("b0", 0.0))
                    _zogy_hdr["KERNNST"] = int(_km_sel.get("n_stamps", 0))
                _zogy_hdr["KERNEPS"] = float(
                    _km_sel.get("wiener_eps", 1e-2)
                )
            _neg_applied = _neg_by_dir.get(_zogy_convolved)
            if _neg_applied is not None and np.isfinite(_neg_applied):
                _zogy_hdr["KERNNG"] = float(_neg_applied)

            # Write the PSF the difference image actually has so main.py
            # can build an ImagePSF for photometry.  For the pre-convolved
            # paths this is the (possibly repaired) target stamp; for
            # canonical ZOGY it is extracted from P_D.  The upstream ePSF
            # file cannot be trusted here -- stamp repair may have
            # replaced it.
            try:
                _diffpsf_path = os.path.join(
                    os.path.dirname(str(differenceFpath)),
                    f"diff_psf_{os.path.splitext(os.path.basename(str(differenceFpath)))[0]}.fits",
                )
                if _diff_stamp_src is not None:
                    _psf_stamp = np.asarray(_diff_stamp_src, dtype=float)
                    _zogy_stamp_fwhm = float(
                        _zogy_hdr.get("DIFFFWHM", 0.0)
                    )
                else:
                    _pd = np.asarray(_P_D, dtype=float)
                    _ny, _nx = _pd.shape
                    # P_D from ifft2 has the PSF peak at [0,0] (FFT origin).
                    # fftshift moves it to the center of the full array.
                    _pd = np.fft.fftshift(_pd)
                    _zogy_fwhm = float(_zogy_fwhm)
                    # Stamp size: ~5x FWHM or 25px, whichever is larger
                    _stamp_hw = (
                        max(int(np.ceil(5.0 * _zogy_fwhm)), 25)
                        if _zogy_fwhm
                        else 25
                    )
                    _stamp_hw = min(_stamp_hw, _ny // 2, _nx // 2)
                    _cy, _cx = _ny // 2, _nx // 2
                    _psf_stamp = _pd[
                        _cy - _stamp_hw: _cy + _stamp_hw + 1,
                        _cx - _stamp_hw: _cx + _stamp_hw + 1,
                    ]
                    # Measure the FWHM on P_D itself rather than trusting the
                    # Gaussian geometric mean: the diff PSF departs from
                    # sqrt(Fn*Fr) for non-Gaussian PSFs, and DIFFFWHM drives
                    # the caller's aperture rescale.
                    _meas_fwhm = _measure_stamp_fwhm(_psf_stamp)
                    _zogy_stamp_fwhm = _zogy_fwhm
                    if np.isfinite(_meas_fwhm) and _meas_fwhm > 0:
                        if (
                            _zogy_fwhm
                            and abs(_meas_fwhm - _zogy_fwhm) / _zogy_fwhm > 0.15
                        ):
                            logger.info(
                                "ZOGY: measured P_D FWHM %.2f px differs from "
                                "geometric-mean %.2f px; using measured value.",
                                _meas_fwhm, _zogy_fwhm,
                            )
                        _zogy_stamp_fwhm = float(_meas_fwhm)
                        _zogy_hdr["FWHM"] = _zogy_stamp_fwhm
                        _zogy_hdr["DIFFFWHM"] = _zogy_stamp_fwhm
                # A kept ePSF stamp can carry NaN outskirts; ImagePSF
                # downstream cannot digest them.
                if not np.isfinite(_psf_stamp).all():
                    _psf_stamp = np.nan_to_num(
                        _psf_stamp, nan=0.0, posinf=0.0, neginf=0.0
                    )
                # Normalize to unit sum
                _ps_sum = float(np.nansum(_psf_stamp))
                if _ps_sum > 0:
                    _psf_stamp = _psf_stamp / _ps_sum
                _psf_hdr = fits.Header()
                _psf_hdr["FWHM"] = (
                    _zogy_stamp_fwhm if _zogy_stamp_fwhm else 0.0
                )
                _psf_hdr["ORIGIN"] = "ZOGY"
                write_fits(_diffpsf_path, _psf_stamp, _psf_hdr)
                _zogy_hdr["DIFFPSF"] = _diffpsf_path
                logger.info(
                    "ZOGY: wrote difference-image PSF stamp (%dx%d px, "
                    "FWHM=%.2f) to %s",
                    _psf_stamp.shape[0], _psf_stamp.shape[1],
                    float(_zogy_stamp_fwhm) if _zogy_stamp_fwhm else 0.0,
                    _diffpsf_path,
                )
            except Exception as _psf_e:
                logger.warning(
                    "ZOGY: failed to write diff PSF stamp: %s", _psf_e
                )
            if _as_bool(ts_cfg.get("zogy_write_score_images", False), False):
                # S/Scorr are ZOGY's matched-filter detection maps and the
                # only products that use the per-pixel noise maps; D itself
                # is unchanged by zogy_per_pixel_noise.  Written as
                # diagnostics only.
                try:
                    _diff_dir = os.path.dirname(str(differenceFpath))
                    _diff_base = os.path.splitext(
                        os.path.basename(str(differenceFpath))
                    )[0]
                    _s_path = os.path.join(
                        _diff_dir, f"zogy_score_{_diff_base}.fits"
                    )
                    _sc_path = os.path.join(
                        _diff_dir, f"zogy_scorr_{_diff_base}.fits"
                    )
                    write_fits(_s_path, np.asarray(_S, dtype=float), _zogy_hdr)
                    write_fits(_sc_path, np.asarray(_Scorr, dtype=float), _zogy_hdr)
                    _zogy_hdr["DIFFS"] = _s_path
                    _zogy_hdr["DIFFSCOR"] = _sc_path
                    logger.info(
                        "ZOGY: wrote score images to %s and %s",
                        os.path.basename(_s_path), os.path.basename(_sc_path),
                    )
                except Exception as _sc_e:
                    logger.warning(
                        "ZOGY: failed to write score images: %s", _sc_e
                    )
            # Propagate the measured diff noise into the error model the same
            # way run_sfft does: the diff carries Var ~ sigma_sci^2 +
            # sigma_ref^2*|k|^2, so the science-image RMS map underestimates
            # it (up to ~sqrt(2) for a matched-depth template).  VSCALE lets
            # main.py rescale background_rms to the measured level without
            # touching the photometrically-calibrated pixels.
            try:
                _dv = np.asarray(diff_image, dtype=float)
                # Filled pixels carry synthetic (median) values, not real
                # difference signal -- exclude them from the noise estimate.
                _dvalid = np.isfinite(_dv) & ~_fill_regions
                if np.count_nonzero(_dvalid) > 1000 and np.isfinite(_sn_sci) and _sn_sci > 0:
                    _q25, _q75 = np.percentile(_dv[_dvalid], [25.0, 75.0])
                    _sig_diff = (_q75 - _q25) / 1.3489795003921634
                    if np.isfinite(_sig_diff) and _sig_diff > 0:
                        _vscale = float(_sig_diff / _sn_sci)
                        # Apply the measured factor even when large:
                        # a shallow template scaled by the flux ratio
                        # legitimately produces a noisy diff, and
                        # leaving VSCALE=1.0 understates every downstream
                        # photometric error by that factor.  Only a
                        # pathological ratio (noise model diverged,
                        # sigma_sci ~ 0) is refused - the IQR cannot be
                        # trusted at that extreme either.
                        if 1.05 < _vscale < 1e4:
                            _zogy_hdr["VSCALE"] = (
                                round(_vscale, 6),
                                "Noise-model factor sigma_diff/sigma_sci (IQR)",
                            )
                            if _vscale > 5.0:
                                logger.warning(
                                    "ZOGY: difference noise %.1fx the "
                                    "science noise model - VSCALE applied "
                                    "anyway so photometric errors reflect "
                                    "the measured diff; the excess likely "
                                    "marks a degraded subtraction "
                                    "(check quality).",
                                    _vscale,
                                )
                            else:
                                logger.info(
                                    "ZOGY: VSCALE=%.4f written to diff header.",
                                    _vscale,
                                )
                        elif _vscale >= 1e4:
                            logger.warning(
                                "ZOGY: difference noise %.1fx the science "
                                "noise model - pathological; leaving "
                                "VSCALE=1.0 (check subtraction quality).",
                                _vscale,
                            )
            except Exception as _vs_e:
                logger.debug("ZOGY VSCALE measurement failed: %s", _vs_e)
            write_fits(
                str(differenceFpath), np.asarray(diff_image, dtype=float), _zogy_hdr
            )
            logger.info("ZOGY subtraction succeeded")
            return "done"
        except Exception as exc:
            log_warning_from_exception(
                logger, "ZOGY failed, falling back to SFFT", exc
            )
            return "sfft"

    def _subtract_sfft(
        self,
        scienceFpath,
        template_work_fpath,
        outputFpath,
        mask_loc,
        scienceDir,
        base_name,
        masked_sources,
        matching_sources,
        kernel_order,
        scale,
        method,
        science_fwhm,
        template_fwhm,
        science_gain,
        template_gain,
        science_saturate,
        template_saturate,
    ) -> str:
        """Attempt SFFT subtraction; return next method to try on failure."""
        # Use finite saturation for SFFT (FITS/MeLOn cannot use inf)
        _saturate_fallback = 1e30

        # Mutable retry state -- initialised before the try block so the
        # exception handler can reference them without NameError.
        ts_sub = (self.input_yaml or {}).get("template_subtraction") or {}
        current_excluded: list = list(masked_sources or [])
        current_matching_sources: list = list(matching_sources or [])
        # Track the log of the currently-adopted run so flux-scaling metadata
        # reflects the diff that is actually on disk.
        _active_log_holder: list = [None]

        _sol_path = os.path.join(str(scienceDir), "SFFT_Solution.fits")

        def _backup_sfft_outputs() -> dict:
            """Copy the diff (and its SFFT solution) aside before a retry."""
            bak = {}
            for p in (str(outputFpath), _sol_path):
                if p and os.path.isfile(p):
                    bp = p + ".firstpass.bak"
                    try:
                        shutil.copy2(p, bp)
                        bak[p] = bp
                    except Exception:
                        pass
            return bak

        def _restore_sfft_outputs(bak: dict) -> None:
            for p, bp in bak.items():
                try:
                    os.replace(bp, p)
                except Exception:
                    pass

        def _discard_sfft_backups(bak: dict) -> None:
            for bp in bak.values():
                try:
                    if os.path.isfile(bp):
                        os.remove(bp)
                except Exception:
                    pass

        def _diff_is_valid(path) -> bool:
            """Check a difference image is present and non-degenerate."""
            try:
                if not path or not os.path.isfile(path):
                    return False
                d = np.asarray(fits.getdata(path), dtype=float)
                fin = np.isfinite(d)
                if int(fin.sum()) < 100:
                    return False
                return float(np.std(d[fin])) > 0.0
            except Exception:
                return False

        sat_sci = (
            float(science_saturate)
            if np.isfinite(science_saturate)
            else _saturate_fallback
        )
        sat_ref = (
            float(template_saturate)
            if np.isfinite(template_saturate)
            else _saturate_fallback
        )
        try:
            script = Path(__file__).parent / "utils" / "run_sfft.py"
            ts_sub = self.input_yaml.get("template_subtraction") or {}
            # Allow user to control whether variable sources are passed to SFFT
            pass_masked_sources = _as_bool(
                ts_sub.get("sfft_pass_masked_sources", False), False
            )
            # Only pass variable sources when the user enables it.  The
            # transient is ALWAYS banned from the kernel fit regardless.
            if pass_masked_sources:
                excluded = list(masked_sources or [])
            else:
                excluded = []
            # Always ban the transient position from SFFT's kernel fit.
            # The transient is a new source not present in the template; its
            # pixels bias the DFT-based kernel solution, causing flux scaling
            # mismatch and dipole residuals at the transient/host position.
            # XY_PriorBan expects 1-based SExtractor coordinates.
            _tx = float(self.input_yaml.get("target_x_pix", 0) or 0)
            _ty = float(self.input_yaml.get("target_y_pix", 0) or 0)
            if _tx > 0 and _ty > 0:
                excluded.append((_tx + 1.0, _ty + 1.0))
            current_excluded[:] = list(excluded)
            current_matching_sources[:] = list(matching_sources)
            # Save the ORIGINAL matching sources before the post-anomaly
            # feedback may modify them.  The post-anomaly feedback replaces
            # current_matching_sources with SFFT-vetted sources and may remove
            # all of them, which would prevent the ConstPhotRatio retry from
            # firing (n_matched_local >= 3 check fails with 0 sources).
            _orig_matching_sources = list(matching_sources)
            _orig_n_matching = len(matching_sources)

            phot_cfg = self.input_yaml.get("photometry", {})

            # ForceConv: which image to convolve to match the other's PSF.
            # REF => DIFF = SCI - conv(REF): transients keep the science PSF.
            # SCI => DIFF = conv(SCI) - REF: difference has reference PSF.
            #
            # AUTO is resolved HERE from the measured post-alignment
            # FWHMs -- we do NOT pass AUTO through to SFFT, whose internal
            # AUTO selects on header FWHMs that SWarp LANCZOS3 resampling can
            # flip relative to the true PSF ordering (BUG 122).  Pipeline AUTO
            # convolves the sharper image up to the broader PSF whenever the
            # FWHM difference exceeds sfft_forceconv_auto_tol, avoiding a
            # deconvolving kernel; within the tolerance it keeps REF (the
            # Bramich 2008 / Hu et al. 2022 science-PSF convention, so the
            # difference image keeps the science PSF and the science ePSF
            # model is used directly for photometry).
            #
            # Users can override with forceconv in YAML (REF/SCI/AUTO).  An
            # explicit direction that requires deconvolution is vetoed to the
            # broadening direction by default; sfft_veto_deconvolution=False
            # restores the old warning-only behaviour.
            # Backward compat: fall back to sfft_forceconv if forceconv is absent.
            _fc_cfg = str(
                ts_sub.get("forceconv", ts_sub.get("sfft_forceconv", "REF"))
            ).strip().upper()
            _fc_tol = float(ts_sub.get("sfft_forceconv_auto_tol", 0.05) or 0.05)
            _fc_requested, _fc_deconvolves, _fc_note = _select_forceconv(
                _fc_cfg, science_fwhm, template_fwhm, auto_tol=_fc_tol
            )
            _fc_veto_enabled = _as_bool(
                ts_sub.get("sfft_veto_deconvolution", True), True
            )
            forceconv, _fc_deconvolves, _fc_vetoed = _apply_sfft_deconvolution_veto(
                _fc_requested, _fc_deconvolves, enabled=_fc_veto_enabled
            )
            if _fc_vetoed:
                _fc_note += f"; vetoed {_fc_requested}->{forceconv} deconvolution"
            logger.info(
                "SFFT ForceConv=%s (cfg=%s, FWHM: sci=%.2f ref=%.2f; %s).",
                forceconv, _fc_cfg, science_fwhm, template_fwhm, _fc_note,
            )
            if _fc_cfg not in ("REF", "SCI", "AUTO"):
                logger.warning("Unknown forceconv=%r; defaulting to REF.", _fc_cfg)
            if _fc_vetoed:
                logger.warning(
                    "SFFT ForceConv=%s requires deconvolution: the convolved\n"
                    "    image (FWHM=%.2f) is broader than the target (%.2f).\n"
                    "    Overriding to ForceConv=%s, a broadening\n"
                    "    convolution. The difference image carries the %s\n"
                    "    PSF.",
                    _fc_requested,
                    template_fwhm if _fc_requested == "REF" else science_fwhm,
                    science_fwhm if _fc_requested == "REF" else template_fwhm,
                    forceconv,
                    "reference" if forceconv == "SCI" else "science",
                )
            elif _fc_deconvolves:
                logger.warning(
                    "SFFT ForceConv=%s requires deconvolution: the convolved\n"
                    "    image (FWHM=%.2f) is broader than the target (%.2f).\n"
                    "    The kernel must remove width, producing negative\n"
                    "    sidelobes that are unstable with few matched\n"
                    "    sources. Consider forceconv=SCI or AUTO (note: SCI\n"
                    "    direction gives the difference image the\n"
                    "    *reference* PSF).",
                    forceconv,
                    template_fwhm if forceconv == "REF" else science_fwhm,
                    science_fwhm if forceconv == "REF" else template_fwhm,
                )

            # Background polynomial order: default to 0 unless the user explicitly overrides
            bg_order = ts_sub.get("sfft_bg_order", 0)
            allow_bg_override = _as_bool(
                ts_sub.get("sfft_allow_crowded_bg_order_override", False), False
            )
            # Star extension iterations: None = use SFFT defaults (4 for sparse, 2 for crowded)
            star_ext_iter = ts_sub.get("sfft_star_ext_iter", None)
            # Allow user to control photometric scaling behavior.
            # In SFFT, ConstPhotRatio controls whether the *solved* flux
            # scaling is spatially constant: True fits a single global flux
            # scale; False fits a spatially-varying scaling polynomial (like
            # HOTPANTS).  It does NOT pin the kernel sum to the SExtractor-
            # measured photometric ratio (FSCAL_PHOT) -- the realised kernel
            # scaling (FSCAL_CONV) comes from the least-squares kernel fit in
            # both modes.  FSCAL_PHOT is only a diagnostic, and is biased when
            # the two images have different PSFs (a fixed aperture on the
            # broader image loses more wing flux) or when matched sources sit
            # on structured background (BACKPHOTO_TYPE=LOCAL).
            const_phot_ratio = _as_bool(
                ts_sub.get("sfft_const_phot_ratio", False), False
            )
            # Auto-enable ConstPhotRatio=True for very sparse fields.
            # With very few matched sources the spatially-varying scaling
            # polynomial (ConstPhotRatio=False) is under-constrained and can
            # absorb source noise as fake spatial structure; the constant
            # model removes those degrees of freedom.  This is a variance-
            # reduction choice -- it does not correct a wrong kernel scale.
            #
            # The user can disable this auto-behaviour by setting
            # sfft_const_phot_ratio_sparse_threshold to 0, or by explicitly
            # setting sfft_const_phot_ratio: True (skip auto, use always).
            _cpr_sparse_thresh = int(
                ts_sub.get("sfft_const_phot_ratio_sparse_threshold", 5)
            ) if ts_sub.get("sfft_const_phot_ratio_sparse_threshold") is not None else 5
            if not const_phot_ratio and _cpr_sparse_thresh > 0:
                # Will be checked after we know the matching source count.
                # For now, flag that auto-enable is eligible.
                _cpr_auto_eligible = True
            else:
                _cpr_auto_eligible = False
            # crowded_field is a shortcut: when True, use SFFT crowded (ECP) unless
            # the user *explicitly* forces sparse via `force_sparse_sfft`.
            sfft_crowded = ts_sub.get(
                "sfft_crowded_method",
                ts_sub.get("crowded_field", False),
            )
            force_sparse = _as_bool(ts_sub.get("force_sparse_sfft", False), False)

            if phot_cfg.get("crowded_field", False) and not force_sparse:
                if not sfft_crowded:
                    logger.info(
                        "photometry.crowded_field=True; forcing SFFT crowded (ECP) mode for subtraction."
                    )
                sfft_crowded = True
            elif force_sparse:
                logger.info(
                    "force_sparse_sfft=True in template_subtraction; using SFFT sparse (ESP) even though crowded_field=%s.",
                    phot_cfg.get("crowded_field", False),
                )

            if sfft_crowded:
                sfft_method = "crowded"
            else:
                sfft_method = "sparse"

            logger.info("Starting SFFT subtraction via %s method...", sfft_method)
            logger.debug(
                "SFFT photometric scaling: ConstPhotRatio=%s",
                const_phot_ratio,
            )

            # `scale` was computed above in subtract() using the physics-based quadrature
            # formula: kernel_hw = ceil(mult * sqrt(FWHM_broad^2 - FWHM_narrow^2)), floored at
            # FWHM_broad.  Use it directly; the fallback path below covers the rare case where
            # scale was not passed into _subtract_sfft.
            KER_HW_MIN = 3
            KER_HW_MAX = 50
            logger.debug(
                "SFFT kernel sizing: scale=%s, science_fwhm=%.2f, template_fwhm=%.2f",
                scale,
                float(science_fwhm),
                float(template_fwhm),
            )
            # `scale` here is the kernel_hw already computed in subtract() via the
            # physics-based quadrature formula.  Use it directly.
            if scale is not None and int(scale) > 0:
                kernel_half_width = min(int(scale), KER_HW_MAX)
                logger.debug(
                    "SFFT kernel half-width: %d px (from physics-based sizing)",
                    kernel_half_width,
                )
            else:
                # Pure fallback: no scale passed; use quadrature formula directly.
                # Use the same multiplier and adaptive floor as subtract() for
                # consistency.
                fwhm_ref_fb = float(template_fwhm)
                fwhm_sci_fb = float(science_fwhm)
                fwhm_broad_fb = max(fwhm_ref_fb, fwhm_sci_fb)
                fwhm_narrow_fb = min(fwhm_ref_fb, fwhm_sci_fb)
                if fwhm_broad_fb > fwhm_narrow_fb and fwhm_broad_fb > 0:
                    fwhm_conv_fb = np.sqrt(max(fwhm_broad_fb ** 2 - fwhm_narrow_fb ** 2, 0.0))
                else:
                    fwhm_conv_fb = fwhm_broad_fb
                _fb_mult = float(ts_sub.get("kernel_hw_fwhm_multiplier") or 2.5)
                if not np.isfinite(_fb_mult) or _fb_mult <= 0:
                    _fb_mult = 2.5
                _fb_mult = max(1.0, min(_fb_mult, 5.0))
                ker_hw_conv = int(np.ceil(_fb_mult * fwhm_conv_fb))
                _fm_fb = 2.0  # consistent with subtract() default floor
                ker_hw_floor = int(np.ceil(_fm_fb * fwhm_broad_fb))
                kernel_half_width = max(KER_HW_MIN, min(KER_HW_MAX, max(ker_hw_conv, ker_hw_floor)))
                logger.info(
                    "SFFT kernel half-width: %d px (fallback quadrature formula, "
                    "FWHM_sci=%.1f FWHM_ref=%.1f FWHM_conv=%.1f mult=%.2f)",
                    kernel_half_width, fwhm_sci_fb, fwhm_ref_fb, fwhm_conv_fb, _fb_mult,
                )
            kernel_half_width = max(KER_HW_MIN, min(KER_HW_MAX, kernel_half_width))

            # Decide whether to pass our computed kernel_half_width to
            # run_sfft.py or let run_sfft.py auto-size it.
            #
            # run_sfft.py auto-sizes conservatively, taking the max of:
            #   hw_broad = ceil(mult * fwhm_broad)       -- full PSF support (SFFT/ip_diffim)
            #   hw_conv  = ceil(3 * fwhm_conv / 2.355)   -- 3-sigma PSF-difference lobe
            #   kernel_hw = max(hw_broad, hw_conv)
            #
            # This is the SFFT (Hu et al. 2022) and ip_diffim convention:
            # the kernel must contain the BROADER PSF, not just the convolution
            # (difference) PSF.  The templates.py formula uses fwhm_conv as the
            # primary term, which can produce a smaller kernel that doesn't
            # fully contain the PSF wings, causing dipole residuals, flux
            # scaling mismatch, and over/undersubtraction.
            #
            # By default, let run_sfft.py auto-size (pass 0).  The
            # templates.py-computed kernel_half_width is still used for:
            #   - logging and diagnostics
            #   - the kernel_half_width return value
            #   - the universal-mask stamp radius for source filtering
            #
            # If kernel_hw_override is set, always pass it (user knows what
            # they want).  If sfft_auto_kernel_size is False, pass the
            # templates.py-computed value (backward compat).
            _sfft_pass_kernel_hw = kernel_half_width
            _user_override_hw = ts_sub.get("kernel_hw_override", None)
            _auto_kernel = _as_bool(
                ts_sub.get("sfft_auto_kernel_size", True), True
            )
            # For sparse fields, run_sfft.py's auto-size formula
            # (hw_broad = ceil(mult * FWHM_broad)) can produce a kernel
            # that is too large relative to the number of sources, leading
            # to an under-constrained kernel fit.  The templates.py formula
            # uses FWHM_conv as the primary term with an adaptive floor
            # and source-count cap, producing a smaller, better-constrained
            # kernel.  When the field is sparse (< 10 vetted sources),
            # pass the templates.py-computed kernel_hw directly instead of
            # letting run_sfft.py auto-size.
            _n_matching_sfft = len(current_matching_sources)
            _sparse_threshold = int(ts_sub.get("sfft_sparse_field_threshold", 10) or 10)
            _sparse_field = _n_matching_sfft < _sparse_threshold

            # Auto-enable ConstPhotRatio=True for very sparse fields.  With
            # very few matched sources (< sfft_const_phot_ratio_sparse_threshold)
            # the spatially-varying scaling polynomial is under-constrained;
            # a constant flux scale removes those degrees of freedom.
            if _cpr_auto_eligible and _n_matching_sfft < _cpr_sparse_thresh:
                const_phot_ratio = True
                logger.info(
                    "SFFT: auto-enabling ConstPhotRatio=True for very\n"
                    "    sparse field (%d matching sources < %d threshold):\n"
                    "    constant flux-scaling model instead of a spatial\n"
                    "    polynomial the source count cannot constrain.",
                    _n_matching_sfft,
                    _cpr_sparse_thresh,
                )
            if _user_override_hw is not None:
                _sfft_pass_kernel_hw = int(_user_override_hw)
                logger.info(
                    "SFFT: passing kernel_hw=%d (user override).",
                    _sfft_pass_kernel_hw,
                )
            elif _auto_kernel and not _sparse_field:
                _sfft_pass_kernel_hw = 0  # let run_sfft.py auto-size
                logger.info(
                    "SFFT: letting run_sfft.py auto-size kernel "
                    "(templates.py computed %d px for diagnostics, "
                    "%d vetted sources).",
                    kernel_half_width,
                    _n_matching_sfft,
                )
            elif _auto_kernel and _sparse_field:
                # Sparse field: pass templates.py-computed kernel_hw to
                # avoid an under-constrained kernel from run_sfft.py's
                # auto-size (which uses FWHM_broad as primary term).
                _sfft_pass_kernel_hw = kernel_half_width
                logger.info(
                    "SFFT: sparse field (%d vetted sources < %d threshold)\n"
                    "    -- passing templates.py kernel_hw=%d px instead of\n"
                    "    auto-sizing (run_sfft.py would use ~%d px =\n"
                    "    ceil(%.1f*FWHM_broad), under-constrained for %d\n"
                    "    sources).",
                    _n_matching_sfft,
                    _sparse_threshold,
                    kernel_half_width,
                    int(np.ceil(float(ts_sub.get("kernel_hw_fwhm_multiplier", 2.5) or 2.5)
                                * max(float(science_fwhm), float(template_fwhm)))),
                    float(ts_sub.get("kernel_hw_fwhm_multiplier", 2.5) or 2.5),
                    _n_matching_sfft,
                )
            else:
                logger.info(
                    "SFFT: passing templates.py kernel_hw=%d px "
                    "(sfft_auto_kernel_size=False).",
                    kernel_half_width,
                )

            def _serialize_xy_pairs(xy_list) -> str:
                if not xy_list:
                    return "[]"
                coords = ",".join(f"[{float(x):.3f},{float(y):.3f}]" for x, y in xy_list)
                return f"[{coords}]"

            def _build_sfft_cmd(run_excluded, run_matching, template_fp, diff_fp):
                min_sources_for_prior = int(ts_sub.get("sfft_min_prior_sources", 3) or 3)
                if len(run_matching) < min_sources_for_prior:
                    logger.warning(
                        "Only %d vetted point-source priors are available (minimum=%d); "
                        "not substituting SFFT automatic detections.",
                        len(run_matching),
                        min_sources_for_prior,
                    )
                match_str = _serialize_xy_pairs(run_matching)
                excl_str = _serialize_xy_pairs(run_excluded)

                cmd_local = [
                    sys.executable,
                    str(script),
                    "-sci",
                    str(scienceFpath),
                    "-ref",
                    str(template_fp),
                    "-diff",
                    str(diff_fp),
                    "-mask",
                    str(mask_loc),
                    "-out_base",
                    out_base,
                ]
                
                # Always pass -masked_sources: the transient is always banned,
                # and variable sources are included only when sfft_pass_masked_sources=True.
                if len(run_excluded) > 0:
                    cmd_local.extend(["-masked_sources", excl_str])
                
                cmd_local.extend([
                    "-forceconv",
                    forceconv,
                    "-kernel_order",
                    str(kernel_order),
                    "-bg_order",
                    str(bg_order),
                    "-allow_crowded_bg_order_override",
                    "true" if allow_bg_override else "false",
                    "-star_ext_iter",
                    str(star_ext_iter) if star_ext_iter is not None else "0",
                    "-constphotratio",
                    "true" if const_phot_ratio else "false",
                    "-matching_sources",
                    match_str,
                    "-kernel_half_width",
                    str(_sfft_pass_kernel_hw),
                    "-gain_sci",
                    str(float(science_gain)),
                    "-gain_ref",
                    str(float(template_gain)),
                    "-saturate_sci",
                    str(sat_sci),
                    "-saturate_ref",
                    str(sat_ref),
                ])

                # Optional: finer background mesh for SExtractor/SFFT.
                back_size = ts_sub.get("sfft_back_size", None)
                back_filt = ts_sub.get("sfft_back_filtersize", None)
                back_phototype = ts_sub.get("sfft_backphototype", "LOCAL")
                detect_thresh = ts_sub.get("sfft_detect_thresh", None)
                if back_size is not None:
                    cmd_local += ["-back_size", str(int(back_size))]
                if back_filt is not None:
                    cmd_local += ["-back_filtersize", str(int(back_filt))]
                if back_phototype is not None:
                    cmd_local += ["-backphototype", str(back_phototype).upper()]
                if detect_thresh is not None:
                    cmd_local += ["-detect_thresh", str(float(detect_thresh))]

                # Compute backend: numpy (CPU) or cupy (NVIDIA GPU).
                # run_sfft.py falls back to Numpy when cupy is missing.
                _sfft_backend = str(
                    ts_sub.get("sfft_backend", "numpy")
                ).strip().lower()
                cmd_local += ["-backend", _sfft_backend]
                if _sfft_backend in ("cupy", "gpu", "cuda"):
                    cmd_local += [
                        "-cuda_device",
                        str(ts_sub.get("sfft_cuda_device", "0")),
                    ]

                # SFFT source-rejection controls.
                # Default: exclude blended sources (FLAGS & 2) which bias flux scaling.
                # Fallback to permissive flags is handled by the retry logic below.
                only_flags_cfg = ts_sub.get("sfft_only_flags", [0, 1, 16, 17])
                if only_flags_cfg is None:
                    cmd_local += ["-only_flags", "none"]
                elif isinstance(only_flags_cfg, (list, tuple)):
                    cmd_local += ["-only_flags", ",".join(str(int(v)) for v in only_flags_cfg)]
                else:
                    cmd_local += ["-only_flags", str(only_flags_cfg)]

                if ts_sub.get("sfft_cvrej_magd_thresh", None) is not None:
                    cmd_local += [
                        "-cvrej_magd_thresh",
                        str(float(ts_sub["sfft_cvrej_magd_thresh"])),
                    ]
                if ts_sub.get("sfft_evrej_ratio_thresh", None) is not None:
                    cmd_local += [
                        "-evrej_ratio_thresh",
                        str(float(ts_sub["sfft_evrej_ratio_thresh"])),
                    ]
                if ts_sub.get("sfft_evrej_safe_magdev", None) is not None:
                    cmd_local += [
                        "-evrej_safe_magdev",
                        str(float(ts_sub["sfft_evrej_safe_magdev"])),
                    ]
                if ts_sub.get("sfft_pac_ratio_thresh", None) is not None:
                    cmd_local += [
                        "-pac_ratio_thresh",
                        str(float(ts_sub["sfft_pac_ratio_thresh"])),
                    ]

                # New SFFT v1.5.0+ features (enabled by default)
                if ts_sub.get("sfft_use_bspline_kernel", False):
                    cmd_local += ["-use_bspline_kernel", "true"]
                if ts_sub.get("sfft_decorrelate_noise", False):
                    cmd_local += ["-decorrelate_noise", "true"]
                if ts_sub.get("sfft_save_decorrelated", False):
                    cmd_local += ["-save_decorrelated", "true"]

                # Kernel regularization (B-Spline path, Cupy backend only).
                # "auto" lets run_sfft.py decide from the matched-source count.
                _reg_mode = str(
                    ts_sub.get("sfft_regularize_kernel", "auto")
                ).strip().lower()
                if _reg_mode not in ("auto", "true", "false"):
                    _reg_mode = "auto"
                cmd_local += ["-regularize_kernel", _reg_mode]
                cmd_local += [
                    "-regularize_lambda",
                    str(float(ts_sub.get("sfft_regularize_lambda", 1e-6))),
                ]
                cmd_local += [
                    "-regularize_sparse_threshold",
                    str(int(ts_sub.get("sfft_regularize_sparse_threshold", 30))),
                ]

                # Variable star rejection: enable flags must be passed alongside
                # their thresholds, otherwise the thresholds are ignored and variable
                # stars bias the kernel fit, leaving residuals at point source positions.
                coarse_var_rej = _as_bool(ts_sub.get("sfft_coarse_var_rejection", False), False)
                elab_var_rej = _as_bool(ts_sub.get("sfft_elabo_var_rejection", False), False)
                cmd_local += ["-coarse_var_rejection", "true" if coarse_var_rej else "false"]
                cmd_local += ["-elabo_var_rejection", "true" if elab_var_rej else "false"]

                # Kernel half-width limits
                kernel_hw_min = ts_sub.get("sfft_kernel_hw_min", 3)
                kernel_hw_max = ts_sub.get("sfft_kernel_hw_max", 50)
                cmd_local += ["-kernel_hw_min", str(int(kernel_hw_min))]
                cmd_local += ["-kernel_hw_max", str(int(kernel_hw_max))]

                # Forward the configured FWHM multiplier so run_sfft.py auto-sizing
                # matches the pipeline's physics-based kernel sizing.
                _ker_mult = ts_sub.get("kernel_hw_fwhm_multiplier", None)
                if _ker_mult is not None:
                    cmd_local += ["-kernel_hw_fwhm_multiplier", str(float(_ker_mult))]

                # Prior source validation
                min_prior_sources = ts_sub.get("sfft_min_prior_sources", 3)
                cmd_local += ["-min_prior_sources", str(int(min_prior_sources))]
                cmd_local += [
                    "-allow_unvetted_source_retry",
                    "true" if _as_bool(
                        ts_sub.get("sfft_allow_unvetted_source_retry", True), True
                    ) else "false",
                    "-self_match_min_sources",
                    str(int(ts_sub.get("sfft_self_match_min_sources", 10) or 10)),
                ]

                # Cross-match tolerance factor: DIVIDES SFFT's auto tolerance
                # (~1.6*max(FWHM) ~ 12px). Default 2.0 -> ~6px, enforcing
                # stricter positional overlap between sci and ref sources.
                # Higher values = tighter matching.
                _match_tol_factor = ts_sub.get("sfft_match_tol_factor", 2.0)
                cmd_local += ["-match_tol_factor", str(float(_match_tol_factor))]

                # Point-source ellipticity threshold for SFFT's own source vetting.
                _ps_min_ellip = ts_sub.get("sfft_point_source_min_ellip", 0.3)
                cmd_local += ["-point_source_min_ellip", str(float(_ps_min_ellip))]

                # Diagnostic-plot format (SFFT_Matching output) follows the
                # top-level plot_format config.
                cmd_local += [
                    "-plot_format",
                    str((self.input_yaml or {}).get("plot_format", "png")),
                ]

                if sfft_crowded:
                    cmd_local.append("-crowded")
                return cmd_local

            out_base = (
                Path(base_name).stem.replace(" ", "_")
                .replace(".", "_")
                .replace("_APT", "")
                .replace("_ERROR", "")
            )
            post_anomaly_csv = scienceDir / f"SFFT_PostAnomaly_Sources_{out_base}.csv"
            matching_sources_csv = scienceDir / f"SFFT_Matching_Sources_{out_base}.csv"
            log_path = scienceDir / f"SFFT_{Path(base_name).stem}.txt"
            # Force single process/CPU: one thread for BLAS/OpenMP and common env limits.
            sfft_env = {**os.environ}
            for _k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "OPENMP_NUM_THREADS",
            ):
                sfft_env[_k] = "1"

            # SFFT's internal MeLOn wrapper calls `sex` (SExtractor) via PATH.
            # The esoreflex SExtractor (v2.5.0) lacks GAIN_KEY support required
            # by SFFT. Prepend the conda env bin dir (containing sex v2.28.2)
            # so SFFT finds the correct SExtractor.
            _py_bin = os.path.dirname(sys.executable)
            if _py_bin and os.path.isfile(os.path.join(_py_bin, "sex")):
                sfft_env["PATH"] = _py_bin + os.pathsep + sfft_env.get("PATH", "")

            cmd = _build_sfft_cmd(current_excluded, current_matching_sources, template_work_fpath, outputFpath)
            sfft_timeout = float(ts_sub.get("sfft_timeout", 600))
            with open(log_path, "w") as lf:
                subprocess.run(
                    cmd, check=True, text=True, stdout=lf, stderr=lf, env=sfft_env,
                    timeout=sfft_timeout
                )
            clean_subprocess_log(log_path)
            _active_log_holder[0] = log_path

            # Optional one-pass feedback: exclude SFFT post-anomaly sources and rerun.
            use_post_anom_feedback = _as_bool(
                ts_sub.get("sfft_use_post_anomaly_feedback", True), True
            )
            post_anom_min_count = int(ts_sub.get("sfft_post_anomaly_min_count", 1))
            post_anom_max_frac = float(ts_sub.get("sfft_post_anomaly_max_fraction", 0.80))
            post_anom_match_radius_px = float(
                ts_sub.get("sfft_post_anomaly_match_radius_px", 1.5)
            )
            if use_post_anom_feedback and post_anomaly_csv.exists():
              retry_log_path = None
              try:
                try:
                    df_anom = pd.read_csv(post_anomaly_csv)
                    xcol = (
                        "X_IMAGE_REF_SCI_MEAN"
                        if "X_IMAGE_REF_SCI_MEAN" in df_anom.columns
                        else None
                    )
                    ycol = (
                        "Y_IMAGE_REF_SCI_MEAN"
                        if "Y_IMAGE_REF_SCI_MEAN" in df_anom.columns
                        else None
                    )
                    if xcol and ycol:
                        xy = df_anom[[xcol, ycol]].apply(pd.to_numeric, errors="coerce")
                        xy = xy.replace([np.inf, -np.inf], np.nan).dropna()
                        post_anom_xy = [
                            (float(v[0]) - 1.0, float(v[1]) - 1.0)
                            for v in xy.to_numpy(float)
                        ]
                    else:
                        post_anom_xy = []
                except Exception:
                    post_anom_xy = []

                n_post = len(post_anom_xy)

                # Load the sources SFFT actually used for the kernel fit
                # before computing the anomaly fraction.  Anomalies are a
                # subset of SFFT's own SubSource list, so that list - not
                # the pipeline priors - is the correct denominator.
                sfft_vetted_sources = []
                if matching_sources_csv.exists():
                    try:
                        df_match = pd.read_csv(matching_sources_csv)
                        _mx = (
                            "X_IMAGE_REF_SCI_MEAN"
                            if "X_IMAGE_REF_SCI_MEAN" in df_match.columns
                            else None
                        )
                        _my = (
                            "Y_IMAGE_REF_SCI_MEAN"
                            if "Y_IMAGE_REF_SCI_MEAN" in df_match.columns
                            else None
                        )
                        if _mx and _my:
                            _mxy = df_match[[_mx, _my]].apply(
                                pd.to_numeric, errors="coerce"
                            )
                            _mxy = _mxy.replace(
                                [np.inf, -np.inf], np.nan
                            ).dropna()
                            sfft_vetted_sources = [
                                (float(v[0]) - 1.0, float(v[1]) - 1.0)
                                for v in _mxy.to_numpy(float)
                            ]
                    except Exception:
                        sfft_vetted_sources = []

                n_ref = max(
                    1, len(sfft_vetted_sources) or len(current_matching_sources)
                )
                frac_post = float(n_post) / float(n_ref)
                # High anomaly fraction (frac_post > max_frac) does NOT mean
                # we should skip the retry.  It means the kernel was bad
                # (e.g., constant kernel with large PSF difference).  This is
                # exactly when the retry is most valuable: SFFT-vetted sources
                # replace the bad priors, anomaly sources are banned, and the
                # kernel gets a second chance.  Only skip when there are zero
                # anomalies (nothing to fix) or below the minimum count.
                _high_anomaly = frac_post > post_anom_max_frac
                if _high_anomaly:
                    logger.warning(
                        "SFFT post-anomaly fraction=%.2f exceeds\n"
                        "    threshold=%.2f (%d/%d sources anomalous).\n"
                        "    High anomaly fraction indicates a bad kernel\n"
                        "    fit - proceeding with retry using SFFT-vetted\n"
                        "    sources.",
                        frac_post, post_anom_max_frac, n_post, n_ref,
                    )
                if n_post >= post_anom_min_count:

                    # --- Improve matching sources using SFFT-vetted results ---
                    # After the first SFFT pass, SFFT writes the sources it
                    # actually used for the kernel fit to
                    # SFFT_Matching_Sources_<base>.csv.  These are vetted by
                    # SFFT's own SExtractor + cross-match + quality checks
                    # (PostAnomaly, CVREJ, EVREJ), so they are better vetted
                    # than the pipeline's original priors.  Use them as the
                    # matching sources for the retry, minus any near
                    # post-anomaly sources.
                    #
                    # However, when ALL SFFT-vetted sources coincide with
                    # anomaly positions (which happens when the kernel is
                    # under-constrained and every source has a residual),
                    # replacing the priors would leave zero sources for the
                    # retry.  Check whether the removal would leave enough
                    # sources BEFORE committing to the replacement.
                    _min_for_retry = max(
                        2, int(ts_sub.get("sfft_min_prior_sources", 3) or 3)
                    )

                    # Determine which source list to use for the retry.
                    # Prefer SFFT-vetted sources, but keep pipeline priors
                    # if the SFFT list is empty or would be emptied by
                    # anomaly removal.
                    _candidate_sources = list(current_matching_sources)
                    _surviving_vetted = []
                    if sfft_vetted_sources:
                        # Check how many SFFT-vetted sources survive anomaly
                        # removal before committing to the replacement.
                        _anom_arr_check = np.asarray(post_anom_xy, float)
                        for x0, y0 in sfft_vetted_sources:
                            _dist2 = (_anom_arr_check[:, 0] - float(x0)) ** 2 + (
                                _anom_arr_check[:, 1] - float(y0)
                            ) ** 2
                            if not np.any(_dist2 <= post_anom_match_radius_px**2):
                                _surviving_vetted.append((x0, y0))
                        if len(_surviving_vetted) >= _min_for_retry:
                            logger.info(
                                "SFFT post-anomaly feedback: using %d SFFT-vetted "
                                "matching sources from first pass (replacing %d "
                                "pipeline priors, %d removed as anomaly-adjacent).",
                                len(_surviving_vetted),
                                len(current_matching_sources),
                                len(sfft_vetted_sources) - len(_surviving_vetted),
                            )
                            _candidate_sources = _surviving_vetted
                        else:
                            logger.warning(
                                "SFFT post-anomaly feedback: %d SFFT-vetted sources "
                                "would leave %d after anomaly removal (< %d). "
                                "Keeping %d original pipeline priors.",
                                len(sfft_vetted_sources),
                                len(_surviving_vetted),
                                _min_for_retry,
                                len(current_matching_sources),
                            )
                            # Keep original priors; anomaly positions are
                            # banned via current_excluded below.

                    # Remove matching sources too close to anomaly sources.
                    if post_anom_xy and _candidate_sources:
                        anom_arr = np.asarray(post_anom_xy, float)
                        filtered_matching = []
                        for x0, y0 in _candidate_sources:
                            dist2 = (anom_arr[:, 0] - float(x0)) ** 2 + (
                                anom_arr[:, 1] - float(y0)
                            ) ** 2
                            if np.any(dist2 <= post_anom_match_radius_px**2):
                                continue
                            filtered_matching.append((x0, y0))
                        dropped_matching = len(_candidate_sources) - len(
                            filtered_matching
                        )
                        current_matching_sources = filtered_matching
                    else:
                        dropped_matching = 0

                    # Safety: don't retry if too few matching sources remain
                    # after removing anomaly-adjacent sources.  SFFT needs at
                    # least sfft_min_prior_sources to produce a valid kernel.
                    #
                    # The anomaly positions ARE first-pass SubSources, so
                    # banning them removes constraints directly from the
                    # retry's SubSource catalog - _surviving_vetted predicts
                    # the retry's matched count.  When both that count and
                    # the pipeline priors sit at/below the kernel minimum
                    # the retry cannot reach a constrained solution (its
                    # prior pass dies at run_sfft's matched-source gate and
                    # the prior-free alternative was already evaluated in
                    # the first pass); skip it rather than spend a subprocess
                    # call on a guaranteed failure.
                    if not _sfft_post_anomaly_retry_viable(
                        len(_surviving_vetted),
                        bool(sfft_vetted_sources),
                        len(current_matching_sources),
                        _min_for_retry,
                    ):
                        logger.warning(
                            "SFFT post-anomaly feedback: banning %d anomaly "
                            "position(s) leaves %d/%d kernel constraints "
                            "(min=%d) with only %d pipeline prior(s); retry "
                            "cannot produce a constrained kernel. Skipping "
                            "retry; keeping first-pass result.",
                            n_post,
                            len(_surviving_vetted),
                            len(sfft_vetted_sources),
                            _min_for_retry,
                            len(current_matching_sources),
                        )
                    elif len(current_matching_sources) < _min_for_retry:
                        logger.warning(
                            "SFFT post-anomaly feedback: only %d matching "
                            "sources remain after removing %d anomaly-adjacent "
                            "sources (min=%d). Skipping retry; keeping "
                            "first-pass result.",
                            len(current_matching_sources),
                            dropped_matching,
                            _min_for_retry,
                        )
                    else:
                        # Extend prior-ban list (only if we're actually retrying).
                        current_excluded = current_excluded + [
                            (x + 1.0, y + 1.0) for x, y in post_anom_xy
                        ]

                        # When the anomaly fraction is high, the kernel was
                        # likely too small to model the PSF difference.  Boost
                        # the kernel half-width for the retry to give SFFT more
                        # freedom to fit the PSF.  This is especially important
                        # when sfft_auto_kernel_size=False (manual sizing).
                        _retry_kernel_boost = 0
                        if _high_anomaly:
                            _retry_kernel_boost = max(
                                3,
                                int(np.ceil(0.25 * kernel_half_width)),
                            )
                            _saved_pass_hw = _sfft_pass_kernel_hw
                            if _sfft_pass_kernel_hw > 0:
                                _sfft_pass_kernel_hw = min(
                                    KER_HW_MAX,
                                    _sfft_pass_kernel_hw + _retry_kernel_boost,
                                )
                            # If auto-sizing (pass 0), the boost is handled by
                            # run_sfft.py's own sizing; we just log it.
                            logger.info(
                                "SFFT post-anomaly retry: boosting kernel by %d px "
                                "(%d -> %d) due to high anomaly fraction.",
                                _retry_kernel_boost,
                                kernel_half_width,
                                kernel_half_width + _retry_kernel_boost,
                            )

                        logger.info(
                            "SFFT post-anomaly feedback: banning %d sources and "
                            "removing %d matching sources; rerunning subtraction.",
                            n_post,
                            dropped_matching,
                        )
                        cmd_retry = _build_sfft_cmd(
                            current_excluded,
                            current_matching_sources,
                            template_work_fpath,
                            outputFpath,
                        )
                        # Restore the pass value after building the retry cmd.
                        if _high_anomaly and _sfft_pass_kernel_hw > 0:
                            _sfft_pass_kernel_hw = _saved_pass_hw
                        retry_log_path = scienceDir / f"sfft_{Path(base_name).stem}_postanom_retry.txt"
                        # Back up the first-pass diff+solution before the retry
                        # overwrites them: a failed or degenerate retry must not
                        # destroy a valid first-pass result.
                        _bak_pa = _backup_sfft_outputs()
                        try:
                            with open(retry_log_path, "w") as lf:
                                subprocess.run(
                                    cmd_retry,
                                    check=True,
                                    text=True,
                                    stdout=lf,
                                    stderr=lf,
                                    env=sfft_env,
                                    timeout=sfft_timeout,
                                )
                            clean_subprocess_log(retry_log_path)
                            if _diff_is_valid(str(outputFpath)):
                                # Adopt the retry: flux-scaling metadata must be
                                # re-parsed from the retry log below.
                                _active_log_holder[0] = retry_log_path
                                logger.info(
                                    "SFFT post-anomaly retry produced a valid "
                                    "difference image; adopting retry result."
                                )
                            else:
                                logger.warning(
                                    "SFFT post-anomaly retry produced an invalid "
                                    "difference image; restoring first-pass result."
                                )
                                _restore_sfft_outputs(_bak_pa)
                        except Exception:
                            # A failed retry may leave a truncated/corrupt diff
                            # on disk -- restore the first-pass output before the
                            # outer handler logs "keeping first-pass result".
                            _restore_sfft_outputs(_bak_pa)
                            raise
                        finally:
                            _discard_sfft_backups(_bak_pa)
                elif n_post > 0:
                    logger.info(
                        "SFFT post-anomaly feedback skipped (count=%d below min=%d).",
                        n_post,
                        post_anom_min_count,
                    )
              except subprocess.CalledProcessError as exc_pa:
                # Retry subprocess exited non-zero.  str(exc_pa) would dump
                # the full command line - the actual SFFT error lives at the
                # tail of the retry log instead.
                logger.warning(
                    "SFFT post-anomaly feedback retry failed (exit=%s); "
                    "keeping first-pass SFFT result. Retry log: %s",
                    exc_pa.returncode,
                    retry_log_path,
                )
                try:
                    _pa_tail = (
                        Path(retry_log_path)
                        .read_text(errors="ignore")
                        .splitlines()[-25:]
                    )
                    if _pa_tail:
                        logger.warning(
                            "SFFT post-anomaly retry log tail:\n%s",
                            "\n".join(_pa_tail),
                        )
                except Exception:
                    pass
              except Exception as exc_pa:
                # Post-anomaly feedback retry failed.  Don't propagate - the
                # first-pass SFFT result is already written to outputFpath.
                # Keep the first-pass result rather than falling back to
                # HOTPANTS.
                log_warning_from_exception(
                    logger,
                    "SFFT post-anomaly feedback retry failed; "
                    "keeping first-pass SFFT result",
                    exc_pa,
                )

            # Parse SFFT log for flux scaling discrepancy warning.
            _conv_scale = None
            _phot_scale = None
            _discrep_pct = None
            # SFFT's MeLOn logs two independent flux scaling estimates:
            #   - Convolution-based: "The Flux Scaling through the Convolution ..."
            #   - Photometric:       "The approximated Flux Scaling from Photometry ..."
            # A large discrepancy (>3%) indicates the kernel integral doesn't match
            # the true flux ratio. This happens when PSFs are nearly identical (kernel
            # is delta-like, integral ~1.0 regardless of true flux ratio) or when too
            # few sources were used for kernel fitting (unconstrained solution).
            # Note: pre-scaling the reference doesn't fix this - SFFT re-estimates
            # both scalings from the pre-scaled input, preserving the relative mismatch.
            # The discrepancy is a diagnostic indicator of kernel quality, not a
            # correctable error. Dipoles from this are best addressed by increasing
            # source count or improving astrometric alignment.
            try:
                import re as _re
                # Prefer header values written by run_sfft.py from SFFT's own
                # return values (SFFT_FSCAL_MEAN + MAG_OFFSET); fall back to
                # log parsing for older outputs.
                try:
                    if outputFpath and os.path.isfile(outputFpath):
                        _fh = fits.getheader(outputFpath)
                        if _fh.get("FSCAL_CONV") is not None and _fh.get("FSCAL_PHOT") is not None:
                            _conv_scale = float(_fh["FSCAL_CONV"])
                            _phot_scale = float(_fh["FSCAL_PHOT"])
                            _discrep_pct = float(
                                _fh.get(
                                    "FSCAL_DISC",
                                    abs(_conv_scale - _phot_scale)
                                    / max(abs(_conv_scale), abs(_phot_scale), 1e-10)
                                    * 100.0,
                                )
                            )
                except Exception:
                    pass
                _parse_log = _active_log_holder[0] or log_path
                if _conv_scale is None and Path(_parse_log).exists():
                    _log_text = Path(_parse_log).read_text(errors="ignore")
                    # BUG 110: Include optional minus sign in the capture group
                    # so negative flux scaling values (e.g., -24.02) are detected.
                    _conv_match = _re.search(
                        r"Flux Scaling through the Convolution.*?\[(-?[\d.]+)", _log_text
                    )
                    _phot_match = _re.search(
                        r"Flux Scaling from Photometry.*?\[(-?[\d.]+)", _log_text
                    )
                    if _conv_match and _phot_match:
                        _conv_scale = float(_conv_match.group(1))
                        _phot_scale = float(_phot_match.group(1))
                        _discrep_pct = abs(_conv_scale - _phot_scale) / max(abs(_conv_scale), abs(_phot_scale), 1e-10) * 100.0
                        # Write both flux scalings to the diff header for downstream use
                        try:
                            if outputFpath and os.path.isfile(outputFpath):
                                with fits.open(outputFpath, mode="update", memmap=False) as _hdul:
                                    _hdul[0].header["HIERARCH FSCAL_CONV"] = float(_conv_scale)
                                    _hdul[0].header["HIERARCH FSCAL_PHOT"] = float(_phot_scale)
                                    _hdul[0].header["HIERARCH FSCAL_DISC"] = float(_discrep_pct)
                                    _hdul.flush()
                        except Exception:
                            pass
                if _conv_scale is not None and _phot_scale is not None:
                    if _conv_scale < 0:
                        logger.warning(
                            "SFFT convolution flux scaling is negative (%.4f). "
                            "Kernel solution may be unconstrained (likely too few "
                            "matched sources). Proceeding with SFFT result.",
                            _conv_scale,
                        )
                    elif _discrep_pct > 3.0:
                        _fc_msg = ""
                        # Read actual ForceConv from diff image header
                        # (SFFT writes FORCECON keyword).  When AUTO is
                        # used, the direction is decided inside SFFT based
                        # on measured post-resampling FWHMs.
                        _actual_fc = forceconv
                        try:
                            if outputFpath and os.path.isfile(outputFpath):
                                with fits.open(outputFpath, memmap=True) as _hdul:
                                    _actual_fc = str(
                                        _hdul[0].header.get("FORCECON", forceconv)
                                    ).strip().upper()
                        except Exception:
                            pass
                        if _actual_fc == "SCI":
                            _fc_msg = (
                                " ForceConv=SCI (SFFT measured science as sharper). "
                                "Discrepancy may be from nearly-identical post-SWarp "
                                "PSFs or too few sources for kernel fitting."
                            )
                        logger.warning(
                            "SFFT flux scaling discrepancy:\n"
                            "    convolution=%.4f vs photometric=%.4f (%.1f%% mismatch)\n"
                            "    NOTE: FSCAL_PHOT comes from SExtractor photometry on the\n"
                            "    *unconvolved* images (BACKPHOTO_TYPE=LOCAL) and is biased\n"
                            "    low when the reference PSF is broader (fixed apertures\n"
                            "    lose more wing flux) or matched sources sit on structured\n"
                            "    background -- FSCAL_CONV is measured through the actual\n"
                            "    convolved product and is usually the more trustworthy\n"
                            "    scale. Treat moderate discrepancies (<15%%) as a\n"
                            "    kernel-quality flag, not proof of wrong normalisation.\n"
                            "    Genuine kernel error produces dipole residuals at source\n"
                            "    positions; poor astrometric alignment or a deconvolving\n"
                            "    ForceConv direction are other causes.%s",
                            _conv_scale, _phot_scale, _discrep_pct, _fc_msg,
                        )
                    else:
                        logger.info(
                            "SFFT flux scaling consistent: convolution=%.4f vs photometric=%.4f "
                            "(%.1f%% match).",
                            _conv_scale, _phot_scale, _discrep_pct,
                        )
            except Exception:
                pass

            # --- Flux scaling discrepancy retry ---
            # If the first pass produced a large flux scaling discrepancy
            # (>3%), the kernel integral doesn't match the true flux ratio.
            # A higher kernel_order allows spatial variation that can better
            # model the PSF difference, reducing the discrepancy.  Retry once
            # with kernel_order+1 (capped at 2) if we have enough sources.
            _discrep_retry_thresh = float(
                ts_sub.get("sfft_flux_discrepancy_retry_pct", 3.0)
            )
            # Compute n_eff from the ORIGINAL matching sources (before
            # post-anomaly feedback may have emptied the list).  The flux
            # scaling discrepancy was computed from the original SFFT run
            # which used the original sources -- the retry decision should
            # be based on that count, not the post-anomaly-emptied list.
            _n_matched_local = _orig_n_matching
            _min_prior_local = int(ts_sub.get("sfft_min_prior_sources", 3) or 3)
            # When SFFT self-matched (fewer than min_prior priors supplied),
            # the vetted-prior count understates the sources SFFT actually
            # used.  Read the true matched count from the CSV written by
            # run_sfft.py; fall back to the prior count when unavailable.
            if _n_matched_local < _min_prior_local and matching_sources_csv.exists():
                try:
                    _n_sfft_actual = int(
                        len(pd.read_csv(matching_sources_csv).dropna(how="all"))
                    )
                    if _n_sfft_actual > _n_matched_local:
                        _n_matched_local = _n_sfft_actual
                except Exception:
                    pass
            _n_eff_local = _n_matched_local
            _do_discrep_retry = (
                _conv_scale is not None
                and _phot_scale is not None
                and _conv_scale > 0
                and _discrep_pct > _discrep_retry_thresh
                and kernel_order < 2
                and _n_eff_local >= 20
            )
            if _do_discrep_retry:
                _retry_kernel_order = min(kernel_order + 1, 2)
                logger.warning(
                    "SFFT flux scaling discrepancy=%.1f%% > %.1f%% threshold. "
                    "Retrying with kernel_order=%d (was %d) to improve kernel "
                    "spatial variation.",
                    _discrep_pct, _discrep_retry_thresh,
                    _retry_kernel_order, kernel_order,
                )
                _saved_kernel_order = kernel_order
                kernel_order = _retry_kernel_order
                _bak_dr = _backup_sfft_outputs()
                try:
                    cmd_discrep_retry = _build_sfft_cmd(
                        current_excluded,
                        _orig_matching_sources,
                        template_work_fpath,
                        outputFpath,
                    )
                    discrep_log_path = scienceDir / f"sfft_{Path(base_name).stem}_discrep_retry.txt"
                    with open(discrep_log_path, "w") as lf:
                        subprocess.run(
                            cmd_discrep_retry,
                            check=True,
                            text=True,
                            stdout=lf,
                            stderr=lf,
                            env=sfft_env,
                            timeout=sfft_timeout,
                        )
                    clean_subprocess_log(discrep_log_path)
                    # Prefer the header FSCAL values (written by run_sfft.py
                    # from SFFT's return values); fall back to log parsing.
                    _conv_scale2 = _phot_scale2 = _discrep_pct2 = None
                    try:
                        if outputFpath and os.path.isfile(outputFpath):
                            _rh = fits.getheader(outputFpath)
                            if _rh.get("FSCAL_CONV") is not None and _rh.get("FSCAL_PHOT") is not None:
                                _conv_scale2 = float(_rh["FSCAL_CONV"])
                                _phot_scale2 = float(_rh["FSCAL_PHOT"])
                                _discrep_pct2 = float(_rh.get("FSCAL_DISC", abs(_conv_scale2 - _phot_scale2) / max(abs(_conv_scale2), abs(_phot_scale2), 1e-10) * 100.0))
                    except Exception:
                        pass
                    # Re-parse the retry log for updated flux scaling
                    if _conv_scale2 is None and discrep_log_path.exists():
                        _retry_text = discrep_log_path.read_text(errors="ignore")
                        _rc = _re.search(
                            r"Flux Scaling through the Convolution.*?\[(-?[\d.]+)", _retry_text
                        )
                        _rp = _re.search(
                            r"Flux Scaling from Photometry.*?\[(-?[\d.]+)", _retry_text
                        )
                        if _rc and _rp:
                            _conv_scale2 = float(_rc.group(1))
                            _phot_scale2 = float(_rp.group(1))
                            _discrep_pct2 = abs(_conv_scale2 - _phot_scale2) / max(
                                abs(_conv_scale2), abs(_phot_scale2), 1e-10
                            ) * 100.0
                    if _conv_scale2 is not None and _phot_scale2 is not None:
                        try:
                            if outputFpath and os.path.isfile(outputFpath):
                                with fits.open(outputFpath, mode="update", memmap=False) as _hdul:
                                    _hdul[0].header["HIERARCH FSCAL_CONV"] = float(_conv_scale2)
                                    _hdul[0].header["HIERARCH FSCAL_PHOT"] = float(_phot_scale2)
                                    _hdul[0].header["HIERARCH FSCAL_DISC"] = float(_discrep_pct2)
                                    _hdul.flush()
                        except Exception:
                            pass
                        # Adopt the retry only if it actually improved the
                        # flux-scaling discrepancy AND produced a valid diff;
                        # otherwise restore the better first-pass result.
                        if _discrep_pct2 < _discrep_pct and _diff_is_valid(str(outputFpath)):
                            logger.info(
                                "SFFT discrepancy retry improved: %.1f%% -> %.1f%% "
                                "(kernel_order=%d).",
                                _discrep_pct, _discrep_pct2, _retry_kernel_order,
                            )
                        else:
                            logger.warning(
                                "SFFT discrepancy retry did not improve: %.1f%% -> %.1f%% "
                                "(kernel_order=%d). Restoring first-pass result.",
                                _discrep_pct, _discrep_pct2, _retry_kernel_order,
                            )
                            _restore_sfft_outputs(_bak_dr)
                    elif not _diff_is_valid(str(outputFpath)):
                        logger.warning(
                            "SFFT discrepancy retry produced an invalid difference "
                            "image; restoring first-pass result."
                        )
                        _restore_sfft_outputs(_bak_dr)
                    logger.info("SFFT subtraction succeeded (discrepancy retry)")
                    return "done"
                except Exception as exc_dr:
                    # A failed retry may leave a truncated diff on disk.
                    _restore_sfft_outputs(_bak_dr)
                    log_warning_from_exception(
                        logger,
                        "SFFT flux scaling discrepancy retry failed; "
                        "keeping first-pass result",
                        exc_dr,
                    )
                finally:
                    kernel_order = _saved_kernel_order
                    _discard_sfft_backups(_bak_dr)

            # --- ConstPhotRatio retry for sparse fields ---
            # When the field is sparse (< ~10 sources), a spatially-varying
            # flux-scaling polynomial (ConstPhotRatio=False) is under-
            # constrained and can absorb source noise as fake spatial
            # structure.  Retrying with ConstPhotRatio=True reduces the
            # scaling model to a single constant -- a variance-reduction
            # measure.
            #
            # NOTE: ConstPhotRatio does NOT pin the kernel sum to the
            # SExtractor photometric ratio (FSCAL_PHOT) -- the realised kernel
            # scaling (FSCAL_CONV) comes from the least-squares fit in both
            # modes, so the retry is not expected to reduce FSCAL_DISC.  The
            # adoption decision therefore compares the actual difference
            # images at the matched-source positions rather than the FSCAL
            # metric: the retry is kept only when it does not increase the
            # median absolute aperture residual there.
            #
            # The standard discrepancy retry (kernel_order+1) requires >= 20
            # sources and is skipped for sparse fields.
            _const_phot_retry_thresh = float(
                ts_sub.get("sfft_const_phot_retry_pct", 10.0)
            )
            _do_const_phot_retry = (
                not _do_discrep_retry  # standard retry didn't fire
                and _conv_scale is not None
                and _phot_scale is not None
                and _conv_scale > 0
                and _discrep_pct > _const_phot_retry_thresh
                and not const_phot_ratio  # not already enabled
                and _n_matched_local >= 3  # need at least a few sources
            )
            if _do_const_phot_retry:
                logger.warning(
                    "SFFT flux scaling discrepancy=%.1f%% > %.1f%% with\n"
                    "    only %d vetted sources. Retrying with\n"
                    "    ConstPhotRatio=True (constant flux-scaling model\n"
                    "    -- reduces spatial-polynomial overfit; does not\n"
                    "    repin the kernel sum).",
                    _discrep_pct, _const_phot_retry_thresh, _n_matched_local,
                )
                _saved_cpr = const_phot_ratio
                const_phot_ratio = True
                _bak_cpr = _backup_sfft_outputs()
                try:
                    # Residual at the vetted positions in the first-pass diff
                    # (0-based science coords; the diff is in science frame).
                    _resid_r = max(3.0, 1.5 * float(science_fwhm))
                    _resid_first = _diff_resid_at_sources(
                        _bak_cpr.get(str(outputFpath)),
                        _orig_matching_sources,
                        radius=_resid_r,
                    )
                    cmd_cpr_retry = _build_sfft_cmd(
                        current_excluded,
                        _orig_matching_sources,
                        template_work_fpath,
                        outputFpath,
                    )
                    cpr_log_path = scienceDir / f"sfft_{Path(base_name).stem}_constphot_retry.txt"
                    with open(cpr_log_path, "w") as lf:
                        subprocess.run(
                            cmd_cpr_retry,
                            check=True,
                            text=True,
                            stdout=lf,
                            stderr=lf,
                            env=sfft_env,
                            timeout=sfft_timeout,
                        )
                    clean_subprocess_log(cpr_log_path)
                    if not _diff_is_valid(str(outputFpath)):
                        logger.warning(
                            "SFFT ConstPhotRatio retry produced an invalid "
                            "difference image; restoring first-pass result."
                        )
                        _restore_sfft_outputs(_bak_cpr)
                    else:
                        # Adopt the retry only if it does not increase the
                        # residual at the matched-source positions -- the
                        # FSCAL metric cannot judge it (ConstPhotRatio does
                        # not repin the kernel sum).
                        _resid_retry = _diff_resid_at_sources(
                            str(outputFpath), _orig_matching_sources,
                            radius=_resid_r,
                        )
                        if (
                            _resid_first is not None
                            and _resid_retry is not None
                            and _resid_retry > 1.10 * _resid_first
                        ):
                            logger.warning(
                                "SFFT ConstPhotRatio retry increased matched-"
                                "source residuals (%.2f -> %.2f); restoring "
                                "first-pass result.",
                                _resid_first, _resid_retry,
                            )
                            _restore_sfft_outputs(_bak_cpr)
                            logger.info("SFFT subtraction succeeded (first pass kept)")
                            return "done"
                        # Read flux scaling from header (preferred) or retry log.
                        _conv_scale3 = _phot_scale3 = _discrep_pct3 = None
                        try:
                            _rh3 = fits.getheader(outputFpath)
                            if _rh3.get("FSCAL_CONV") is not None and _rh3.get("FSCAL_PHOT") is not None:
                                _conv_scale3 = float(_rh3["FSCAL_CONV"])
                                _phot_scale3 = float(_rh3["FSCAL_PHOT"])
                                _discrep_pct3 = float(_rh3.get("FSCAL_DISC", 0.0))
                        except Exception:
                            pass
                        if _conv_scale3 is None and cpr_log_path.exists():
                            _cpr_text = cpr_log_path.read_text(errors="ignore")
                            _rc2 = _re.search(
                                r"Flux Scaling through the Convolution.*?\[(-?[\d.]+)", _cpr_text
                            )
                            _rp2 = _re.search(
                                r"Flux Scaling from Photometry.*?\[(-?[\d.]+)", _cpr_text
                            )
                            if _rc2 and _rp2:
                                _conv_scale3 = float(_rc2.group(1))
                                _phot_scale3 = float(_rp2.group(1))
                                _discrep_pct3 = abs(_conv_scale3 - _phot_scale3) / max(
                                    abs(_conv_scale3), abs(_phot_scale3), 1e-10
                                ) * 100.0
                        if _conv_scale3 is not None:
                            try:
                                with fits.open(outputFpath, mode="update", memmap=False) as _hdul:
                                    _hdul[0].header["HIERARCH FSCAL_CONV"] = float(_conv_scale3)
                                    _hdul[0].header["HIERARCH FSCAL_PHOT"] = float(_phot_scale3)
                                    _hdul[0].header["HIERARCH FSCAL_DISC"] = float(_discrep_pct3)
                                    _hdul[0].header["CPHOTR"] = True
                                    _hdul.flush()
                            except Exception:
                                pass
                            logger.info(
                                "SFFT ConstPhotRatio retry: flux scaling discrepancy "
                                "%.1f%% -> %.1f%% (conv=%.4f phot=%.4f; "
                                "matched-source resid %s -> %s).",
                                _discrep_pct, _discrep_pct3,
                                _conv_scale3, _phot_scale3,
                                f"{_resid_first:.2f}" if _resid_first is not None else "n/a",
                                f"{_resid_retry:.2f}" if _resid_retry is not None else "n/a",
                            )
                    logger.info("SFFT subtraction succeeded (ConstPhotRatio retry)")
                    return "done"
                except Exception as exc_cpr:
                    _restore_sfft_outputs(_bak_cpr)
                    log_warning_from_exception(
                        logger,
                        "SFFT ConstPhotRatio retry failed; "
                        "keeping first-pass result",
                        exc_cpr,
                    )
                finally:
                    const_phot_ratio = _saved_cpr
                    _discard_sfft_backups(_bak_cpr)

            logger.info("SFFT subtraction succeeded")
            return "done"
        except Exception as exc:
            # If a valid difference image already exists on disk (first pass
            # succeeded but a later step raised), keep it rather than rerunning
            # SFFT or falling back to HOTPANTS with a worse/corrupt product.
            if _diff_is_valid(str(outputFpath)):
                logger.warning(
                    "SFFT raised after producing a valid difference image "
                    "(%s); keeping the existing result.",
                    exc,
                )
                # The abort may have skipped the FSCAL header write
                # (e.g. the matched-source veto fires after SFFT's own
                # subtraction completed).  Recover the scalings from the
                # run log so the flux-scale discrepancy still reaches the
                # photometric error budget and the quality manifest.
                try:
                    import re as _re_resc

                    _resc_log = _active_log_holder[0] or locals().get(
                        "log_path"
                    )
                    if _resc_log and Path(_resc_log).exists():
                        _resc_text = Path(_resc_log).read_text(errors="ignore")
                        _rc_resc = _re_resc.search(
                            r"Flux Scaling through the Convolution.*?\[(-?[\d.]+)",
                            _resc_text,
                        )
                        _rp_resc = _re_resc.search(
                            r"Flux Scaling from Photometry.*?\[(-?[\d.]+)",
                            _resc_text,
                        )
                        _resc_hdr = fits.getheader(outputFpath)
                        if _rc_resc and _rp_resc and (
                            _resc_hdr.get("FSCAL_CONV") is None
                            or _resc_hdr.get("FSCAL_PHOT") is None
                        ):
                            _conv_resc = float(_rc_resc.group(1))
                            _phot_resc = float(_rp_resc.group(1))
                            _disc_resc = abs(_conv_resc - _phot_resc) / max(
                                abs(_conv_resc), abs(_phot_resc), 1e-10
                            ) * 100.0
                            with fits.open(
                                outputFpath, mode="update", memmap=False
                            ) as _hdul:
                                _hdul[0].header["HIERARCH FSCAL_CONV"] = (
                                    _conv_resc
                                )
                                _hdul[0].header["HIERARCH FSCAL_PHOT"] = (
                                    _phot_resc
                                )
                                _hdul[0].header["HIERARCH FSCAL_DISC"] = (
                                    _disc_resc
                                )
                                _hdul.flush()
                            logger.info(
                                "Recovered SFFT flux scaling from log on "
                                "rescue path: conv=%.4f phot=%.4f "
                                "discrepancy=%.1f%%.",
                                _conv_resc,
                                _phot_resc,
                                _disc_resc,
                            )
                except Exception:
                    pass
                return "done"
            # If the failure happened before the command builder was defined
            # (early config parsing), no retry is possible.
            if not callable(locals().get("_build_sfft_cmd")):
                logger.warning(
                    "SFFT failed before command construction (%s); "
                    "falling back to HOTPANTS.",
                    exc,
                )
                return "hotpants"
            # --- Fallback: retry with permissive ONLY_FLAGS if restrictive
            # flags were used.  Restrictive flags (excluding blended sources)
            # can starve SFFT of sources in dense fields.  Retry once with
            # the full permissive set before falling back to HOTPANTS.
            _restrictive_flags = ts_sub.get("sfft_only_flags", [0, 1, 16, 17])
            _permissive_flags = [0, 1, 2, 3, 16, 17, 18, 19]
            _is_restrictive = (
                _restrictive_flags is not None
                and isinstance(_restrictive_flags, (list, tuple))
                and set(_restrictive_flags) != set(_permissive_flags)
            )
            if _is_restrictive:
                logger.warning(
                    "SFFT failed with restrictive only_flags=%s (%s). "
                    "Retrying with permissive flags=%s before HOTPANTS fallback.",
                    _restrictive_flags, str(exc), _permissive_flags,
                )
                _saved_flags = ts_sub.get("sfft_only_flags")
                ts_sub["sfft_only_flags"] = _permissive_flags
                try:
                    cmd_fallback = _build_sfft_cmd(
                        current_excluded,
                        current_matching_sources,
                        template_work_fpath,
                        outputFpath,
                    )
                    fallback_log_path = scienceDir / f"sfft_{Path(base_name).stem}_flags_retry.txt"
                    with open(fallback_log_path, "w") as lf:
                        subprocess.run(
                            cmd_fallback,
                            check=True,
                            text=True,
                            stdout=lf,
                            stderr=lf,
                            env=sfft_env,
                            timeout=sfft_timeout,
                        )
                    clean_subprocess_log(fallback_log_path)
                    logger.info(
                        "SFFT succeeded with permissive flags fallback."
                    )
                    return "done"
                except Exception as exc2:
                    log_warning_from_exception(
                        logger,
                        "SFFT permissive-flags retry also failed; "
                        "attempting ConstPhotRatio=True retry before HOTPANTS",
                        exc2,
                    )
                finally:
                    if _saved_flags is not None:
                        ts_sub["sfft_only_flags"] = _saved_flags
                    else:
                        ts_sub.pop("sfft_only_flags", None)
            else:
                log_warning_from_exception(
                    logger, "SFFT failed, attempting ConstPhotRatio retry before HOTPANTS", exc
                )

            # --- Last-resort retry with ConstPhotRatio=True before HOTPANTS.
            # When SFFT fails due to too few matched sources (e.g. 2 sources
            # with ConstPhotRatio=False which requires minimum 3), retrying
            # with ConstPhotRatio=True lowers the minimum to 2 and constrains
            # the flux scaling to the photometric ratio.  This can rescue
            # subtractions in very sparse fields where HOTPANTS would also
            # struggle.
            if not const_phot_ratio:
                logger.info(
                    "SFFT: retrying with ConstPhotRatio=True (was False) "
                    "to constrain flux scaling for sparse field."
                )
                _saved_cpr_exc = const_phot_ratio
                const_phot_ratio = True
                # Also use permissive flags for this last-resort retry.
                _saved_flags_exc = ts_sub.get("sfft_only_flags")
                ts_sub["sfft_only_flags"] = [0, 1, 2, 3, 16, 17, 18, 19]
                try:
                    cmd_cpr_exc = _build_sfft_cmd(
                        current_excluded,
                        current_matching_sources,
                        template_work_fpath,
                        outputFpath,
                    )
                    cpr_exc_log_path = scienceDir / f"sfft_{Path(base_name).stem}_constphot_exc_retry.txt"
                    with open(cpr_exc_log_path, "w") as lf:
                        subprocess.run(
                            cmd_cpr_exc,
                            check=True,
                            text=True,
                            stdout=lf,
                            stderr=lf,
                            env=sfft_env,
                            timeout=sfft_timeout,
                        )
                    clean_subprocess_log(cpr_exc_log_path)
                    # Parse flux scaling from the retry log
                    if cpr_exc_log_path.exists():
                        import re as _re_exc
                        _cpr_exc_text = cpr_exc_log_path.read_text(errors="ignore")
                        _rc_exc = _re_exc.search(
                            r"Flux Scaling through the Convolution.*?\[(-?[\d.]+)", _cpr_exc_text
                        )
                        _rp_exc = _re_exc.search(
                            r"Flux Scaling from Photometry.*?\[(-?[\d.]+)", _cpr_exc_text
                        )
                        if _rc_exc and _rp_exc:
                            _conv_exc = float(_rc_exc.group(1))
                            _phot_exc = float(_rp_exc.group(1))
                            _disc_exc = abs(_conv_exc - _phot_exc) / max(
                                abs(_conv_exc), abs(_phot_exc), 1e-10
                            ) * 100.0
                            try:
                                if outputFpath and os.path.isfile(outputFpath):
                                    with fits.open(outputFpath, mode="update", memmap=False) as _hdul:
                                        _hdul[0].header["HIERARCH FSCAL_CONV"] = float(_conv_exc)
                                        _hdul[0].header["HIERARCH FSCAL_PHOT"] = float(_phot_exc)
                                        _hdul[0].header["HIERARCH FSCAL_DISC"] = float(_disc_exc)
                                        _hdul[0].header["CPHOTR"] = True
                                        _hdul.flush()
                            except Exception:
                                pass
                            logger.info(
                                "SFFT ConstPhotRatio exception retry succeeded: "
                                "conv=%.4f phot=%.4f discrepancy=%.1f%%.",
                                _conv_exc, _phot_exc, _disc_exc,
                            )
                    logger.info("SFFT subtraction succeeded (ConstPhotRatio exception retry)")
                    return "done"
                except Exception as exc_cpr_exc:
                    log_warning_from_exception(
                        logger,
                        "SFFT ConstPhotRatio exception retry also failed; "
                        "falling back to HOTPANTS",
                        exc_cpr_exc,
                    )
                finally:
                    const_phot_ratio = _saved_cpr_exc
                    if _saved_flags_exc is not None:
                        ts_sub["sfft_only_flags"] = _saved_flags_exc
                    else:
                        ts_sub.pop("sfft_only_flags", None)

            return "hotpants"

    def _subtract_hotpants(
        self,
        scienceFpath,
        templateFpath,
        differenceFpath,
        mask_loc,
        scienceDir,
        base_name,
        scienceMedian,
        scienceSTD,
        templateMedian,
        templateSTD,
        science_saturate,
        template_saturate,
        science_readnoise,
        template_readnoise,
        science_fwhm,
        template_fwhm,
        kernel_order,
        stamp_loc,
        scienceNoise,
        scale,
        templateNoise=None,
        sci_mask_loc=None,
        tpl_mask_loc=None,
        matching_sources=None,
        target_xy=None,
        masked_sources=None,
    ) -> bool:
        """Attempt HOTPANTS subtraction. Returns True on success."""
        logger.info("Starting HOTPANTS subtraction...")
        try:
            ts = self.input_yaml.get("template_subtraction", {})
            exe_cfg = ts.get("hotpants_exe_loc")
            exe = exe_cfg.strip() if isinstance(exe_cfg, str) else ""
            if not exe:
                exe = "hotpants"

            # Resolve executable: allow either an explicit path or a command on PATH.
            resolved_exe = exe
            if os.path.sep in exe or exe.startswith("."):
                if not os.path.isfile(exe):
                    logger.warning(
                        "HOTPANTS executable '%s' was not found. Set template_subtraction.hotpants_exe_loc to a valid path or ensure 'hotpants' is on PATH.",
                        exe,
                    )
                    return False
            else:
                which = shutil.which(exe)
                if which is None:
                    logger.warning(
                        "HOTPANTS executable '%s' not found on PATH. Install HOTPANTS and/or set template_subtraction.hotpants_exe_loc to its full path.",
                        exe,
                    )
                    return False
                resolved_exe = which

            original_sci_path = scienceFpath
            original_ref_path = templateFpath
            scienceFpath = clean_fits_nans(scienceFpath, str(scienceDir))
            templateFpath = clean_fits_nans(templateFpath, str(scienceDir))

            # Convolution direction: by default, ALWAYS convolve the reference
            # (template) to match the science PSF.  This matches the SFFT and
            # ZOGY forceconv=REF convention (Bramich 2008, Hu et al. 2022):
            # the difference image retains the science PSF, so the science
            # ePSF model can be used directly for photometry without any
            # PSF mismatch correction.
            #
            # HOTPANTS -c t = convolve template (REF), -c i = convolve science (SCI).
            # REF is the pipeline default; AUTO is resolved pipeline-side
            # from the measured FWHMs (same rule as SFFT) so it picks the
            # sharper-to-broader direction per field.
            # Users can override with forceconv=SCI/AUTO in YAML.
            _hp_fc_cfg = str(
                ts.get("forceconv", ts.get("sfft_forceconv", "REF"))
            ).strip().upper()
            if _hp_fc_cfg == "AUTO":
                _hp_fc_resolved, _, _hp_fc_note = _select_forceconv(
                    "AUTO",
                    science_fwhm,
                    template_fwhm,
                    auto_tol=float(
                        ts.get("sfft_forceconv_auto_tol", 0.05) or 0.05
                    ),
                )
                logger.info("HOTPANTS forceconv %s", _hp_fc_note)
            elif _hp_fc_cfg in ("REF", "SCI"):
                _hp_fc_resolved = _hp_fc_cfg
            else:
                logger.warning(
                    "Unknown forceconv=%r for HOTPANTS; defaulting to REF.",
                    _hp_fc_cfg,
                )
                _hp_fc_resolved = "REF"
            if _hp_fc_resolved == "SCI":
                _hp_forceconv = "i"
                _hp_forceconv_kw = "SCI"
            else:
                _hp_forceconv = "t"
                _hp_forceconv_kw = "REF"
            logger.info(
                "HOTPANTS convolution: %s (diff will have %s PSF).",
                "convolving template (REF)" if _hp_forceconv_kw == "REF"
                else "convolving science (SCI)",
                "science" if _hp_forceconv_kw == "REF" else "reference",
            )

            # Kernel sizing: scale is the physics-based kernel half-width
            # (sfft_kernel_hw), computed from FWHM via quadrature formula in
            # subtract(). Both SFFT and HOTPANTS now use the same kernel sizing.
            # r = kernel half-width for HOTPANTS convolution kernel.
            # rss = substamp half-width (typically 3x r).
            MAX_KERNEL_HALF_WIDTH = 50
            logger.info(
                "HOTPANTS kernel sizing: scale=%s, science_fwhm=%.2f, template_fwhm=%.2f",
                scale,
                float(science_fwhm),
                float(template_fwhm),
            )
            if scale is not None and int(scale) > 0:
                r = ensure_odd(max(min(int(scale), MAX_KERNEL_HALF_WIDTH), 5))
                logger.info(
                    "HOTPANTS kernel half-width (-r) set from physics-based sizing: %d px (clamped to max %d)",
                    r,
                    MAX_KERNEL_HALF_WIDTH,
                )
            else:
                hotpants_fwhm = ensure_odd(
                    int(max(np.ceil(template_fwhm), np.ceil(science_fwhm)))
                )
                r = ensure_odd(max(int(1.5 * hotpants_fwhm), 5))
                logger.info(
                    "HOTPANTS kernel half-width (-r) set from FWHM: %d px (fallback, scale not provided)",
                    r,
                )
            rss = ensure_odd(max(3 * r, 11))


            # Guard against non-finite background statistics (e.g. when every
            # pixel is masked upstream): NaN lower-limit arguments would make
            # HOTPANTS fail with an opaque error.
            if not all(
                np.isfinite(v)
                for v in (scienceMedian, scienceSTD, templateMedian, templateSTD)
            ):
                logger.warning(
                    "HOTPANTS: non-finite background statistics "
                    "(sci_med=%s sci_std=%s tpl_med=%s tpl_std=%s); aborting.",
                    scienceMedian, scienceSTD, templateMedian, templateSTD,
                )
                return False

            # Read noise: HOTPANTS can misbehave with 0; use a small floor (e.g. 0.1 e-).
            rn_floor = 0.1
            science_readnoise = max(float(science_readnoise), rn_floor)
            template_readnoise = max(float(template_readnoise), rn_floor)

            # Upper limits (-iu, -tu): use a very high value so HOTPANTS does not
            # treat bright valid pixels as bad and cut out parts of the image.
            # Only the lower limits (-il, -tl) are used to exclude negative/low noise.
            ul = 1e30
            mask_abs = os.path.abspath(mask_loc)
            # Per-image masks (-imi/-tmi) when the caller built them; a
            # shared mask misattributes science-only defects to the
            # template and vice versa.
            imi_abs = (
                os.path.abspath(sci_mask_loc) if sci_mask_loc else mask_abs
            )
            tmi_abs = (
                os.path.abspath(tpl_mask_loc) if tpl_mask_loc else mask_abs
            )

            # Stamp positions for the kernel fit.  The target must never be
            # a stamp position: a bright transient in the kernel solution
            # biases the flux scale and can self-subtract.  This applies to
            # the caller-provided file too -- a bright transient can reach
            # it via the NaN-flux re-addition in main.py.  SFFT achieves the
            # same exclusion via XY_PriorBan.
            stamp_arg = None
            try:
                _excl_r = max(3.0 * float(science_fwhm), 15.0)
                # Kernel stamps must not sit on variable sources either: a
                # variable star in the fit biases the kernel like the
                # transient itself.  masked_sources arrive 1-based.
                _ban_pts = []
                if target_xy is not None:
                    _tarr = np.asarray(target_xy, dtype=float).ravel()
                    if _tarr.size >= 2 and np.isfinite(_tarr[:2]).all():
                        _ban_pts.append((float(_tarr[0]), float(_tarr[1])))
                if masked_sources is not None:
                    for _bp in np.atleast_2d(
                        np.asarray(masked_sources, dtype=float)
                    ):
                        if _bp.size >= 2 and np.isfinite(_bp[:2]).all():
                            _ban_pts.append(
                                (float(_bp[0]) - 1.0, float(_bp[1]) - 1.0)
                            )

                def _drop_target(pts_0b):
                    if not _ban_pts:
                        return pts_0b
                    return [
                        (x, y)
                        for x, y in pts_0b
                        if min(
                            np.hypot(x - _bx, y - _by)
                            for _bx, _by in _ban_pts
                        )
                        > _excl_r
                    ]

                _pts = None
                _from_file = False
                if stamp_loc and os.path.isfile(stamp_loc):
                    # Caller-provided stamp file: 1-based FITS coords.
                    _raw = []
                    with open(stamp_loc) as _sf_in:
                        for _line in _sf_in:
                            _cols = _line.split()
                            if len(_cols) < 2:
                                continue
                            try:
                                _raw.append(
                                    (float(_cols[0]) - 1.0,
                                     float(_cols[1]) - 1.0)
                                )
                            except ValueError:
                                continue
                    _kept = _drop_target(_raw)
                    _n_dropped = len(_raw) - len(_kept)
                    if _n_dropped > 0:
                        logger.info(
                            "HOTPANTS: excluded %d stamp(s) within %.1f px "
                            "of target/variable positions.",
                            _n_dropped, _excl_r,
                        )
                    # Respect the caller's list even when small: it was
                    # chosen deliberately.
                    _pts = _kept if _kept else None
                    _from_file = _pts is not None
                    if _raw and not _kept:
                        logger.warning(
                            "Position exclusion emptied the provided stamp "
                            "file; falling back to matching sources."
                        )

                if _pts is None and matching_sources:
                    _pts = _drop_target(
                        [
                            (float(ms[0]), float(ms[1]))
                            for ms in matching_sources
                        ]
                    )

                if _pts is not None:
                    _pts = deduplicate_points(_pts, min_sep=2.0)
                _n_min_stamps = 10
                if _pts and (_from_file or len(_pts) >= _n_min_stamps):
                    stamp_arg = os.path.join(
                        str(scienceDir), f"hotpants_stamps_{base_name}"
                    )
                    with open(stamp_arg, "w") as _sf:
                        for x, y in _pts:
                            # HOTPANTS stamp positions are 1-based
                            # FITS pixel coordinates.
                            _sf.write(f"{x + 1.0:.3f} {y + 1.0:.3f}\n")
                    logger.info(
                        "HOTPANTS stamp file: %d positions "
                        "(target excluded).",
                        len(_pts),
                    )
                else:
                    logger.warning(
                        "Too few stamp positions after target exclusion; "
                        "letting HOTPANTS choose substamps."
                    )
            except Exception as e:
                logger.warning(
                    "Could not build HOTPANTS stamp list: %s", e
                )
                stamp_arg = None

            timeout_sec = float(ts.get("hotpants_timeout", 100))
            args = [
                resolved_exe,
                "-inim",
                str(scienceFpath),
                "-tmplim",
                str(templateFpath),
                "-outim",
                str(differenceFpath),
                "-il",
                str(scienceMedian - 25 * scienceSTD),
                "-tl",
                str(templateMedian - 25 * templateSTD),
                "-tu",
                str(ul),
                "-iu",
                str(ul),
                "-tr",
                str(template_readnoise),
                "-ir",
                str(science_readnoise),
                "-imi",
                imi_abs,
                "-tmi",
                tmi_abs,
                "-n",
                "i",
                "-c",
                _hp_forceconv,
                "-v",
                "2",
                "-r",
                str(r),
                "-rss",
                str(rss),
                "-ko",
                str(kernel_order),
                # Background polynomial order: the pipeline always subtracts a
                # constant median from both images before subtraction, so the
                # background is ~0.  bg_order=0 (constant offset) is sufficient
                # to absorb any residual constant difference.  Higher orders
                # can overfit and introduce spatial structure that mimics
                # point-source residuals.
                "-bgo",
                str(max(int(ts.get("hotpants_bg_order", 0)), 0)),
            ]
            if stamp_arg:
                args += ["-ssf", stamp_arg]
                args += ["-savexy", str(scienceDir / "used_stamps.region")]
            if scienceNoise:
                args += ["-ini", str(scienceNoise)]
            if templateNoise:
                args += ["-tni", str(templateNoise)]

            log_path = scienceDir / f"HOTPANTS_{Path(base_name).stem}.txt"
            with open(log_path, "w") as lf:
                subprocess.run(
                    args,
                    text=True,
                    stdout=lf,
                    stderr=lf,
                    check=True,
                    timeout=timeout_sec,
                )
            clean_subprocess_log(log_path)

            logger.info("HOTPANTS subtraction succeeded")

            # Write FORCECON / CONVD / FWHM keywords to the diff header so the
            # main.py ePSF consistency block can detect the convolution direction
            # and convolve the science ePSF to match the diff-image PSF.
            try:
                with fits.open(differenceFpath, mode="update") as _hdul:
                    _hdr = _hdul[0].header
                    _hdr["FORCECON"] = _hp_forceconv_kw
                    _hdr["CONVD"] = _hp_forceconv_kw
                    _hdr["FWHM_SCI"] = float(science_fwhm)
                    _hdr["FWHM_REF"] = float(template_fwhm)
                    logger.info(
                        "HOTPANTS diff header: FORCECON=%s, CONVD=%s, FWHM_SCI=%.2f, FWHM_REF=%.2f",
                        _hp_forceconv_kw, _hp_forceconv_kw,
                        float(science_fwhm), float(template_fwhm),
                    )
                    # Same variance calibration as run_sfft/ZOGY: the diff
                    # noise exceeds the science-image RMS by
                    # sqrt(1 + (sigma_ref/sigma_sci)^2*|k|^2), so propagate the
                    # measured ratio via VSCALE for main.py to apply to
                    # background_rms.  clean_fits_nans marks invalid pixels
                    # with the 0.0 sentinel - exclude them from both IQRs.
                    try:
                        _dd = np.asarray(_hdul[0].data, dtype=np.float64)
                        _dv = _dd[np.isfinite(_dd) & (_dd != 0.0)]
                        _sd = np.asarray(
                            fits.getdata(scienceFpath), dtype=np.float64
                        )
                        _sv = _sd[np.isfinite(_sd) & (_sd != 0.0)]
                        if _dv.size > 1000 and _sv.size > 1000:
                            _q25d, _q75d = np.percentile(_dv, [25.0, 75.0])
                            _q25s, _q75s = np.percentile(_sv, [25.0, 75.0])
                            _sig_diff = (_q75d - _q25d) / 1.3489795003921634
                            _sig_sci = (_q75s - _q25s) / 1.3489795003921634
                            if (
                                np.isfinite(_sig_diff)
                                and np.isfinite(_sig_sci)
                                and _sig_diff > 0
                                and _sig_sci > 0
                            ):
                                _vscale = float(_sig_diff / _sig_sci)
                                # Measured factor is applied even when
                                # large (see the ZOGY path): understating
                                # it would leave photometric errors a
                                # factor too small on degraded fields.
                                if 1.05 < _vscale < 1e4:
                                    _hdr["VSCALE"] = (
                                        round(_vscale, 6),
                                        "Noise-model factor sigma_diff/sigma_sci (IQR)",
                                    )
                                    if _vscale > 5.0:
                                        logger.warning(
                                            "HOTPANTS: difference noise "
                                            "%.1fx the science noise "
                                            "model - VSCALE applied "
                                            "anyway; the excess likely "
                                            "marks a degraded "
                                            "subtraction.",
                                            _vscale,
                                        )
                                    else:
                                        logger.info(
                                            "HOTPANTS: VSCALE=%.4f written to diff header.",
                                            _vscale,
                                        )
                                elif _vscale >= 1e4:
                                    logger.warning(
                                        "HOTPANTS: difference noise %.1fx the science "
                                        "noise model - pathological; leaving VSCALE=1.0.",
                                        _vscale,
                                    )
                    except Exception as _vs_e:
                        logger.debug(
                            "HOTPANTS VSCALE measurement failed: %s", _vs_e
                        )
            except Exception as _he:
                logger.warning("Failed to write FORCECON keywords to HOTPANTS diff header: %s", _he)

            return True

        except Exception as exc:
            log_warning_from_exception(
                logger, "HOTPANTS subtraction failed", exc
            )
            return False
        finally:
            # Clean up temporary files created by clean_fits_nans
            try:
                if scienceFpath != original_sci_path and os.path.exists(scienceFpath):
                    os.remove(scienceFpath)
                    logger.debug("Cleaned up HOTPANTS temp file: %s", scienceFpath)
            except (OSError, NameError) as e:
                logger.warning("Failed to clean up HOTPANTS science temp file: %s", e)
            try:
                if templateFpath != original_ref_path and os.path.exists(templateFpath):
                    os.remove(templateFpath)
                    logger.debug("Cleaned up HOTPANTS temp file: %s", templateFpath)
            except (OSError, NameError) as e:
                logger.warning("Failed to clean up HOTPANTS template temp file: %s", e)
