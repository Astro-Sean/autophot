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
  - Added detailed docstrings and inline commentary throughout.

Created on Thu Oct 27 11:27:05 2022
@author: seanbrennan
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import gc
import glob
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
from scipy.ndimage import shift as ndimage_shift
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.linear_model import RANSACRegressor
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

    _SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    # Template download code should be usable even when scikit-learn isn't
    # installed; the regression/color-term sections will error at runtime.
    class _BaseEstimatorPlaceholder:
        pass

    class _RegressorMixinPlaceholder:
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
    # Allow module import for template downloading; alignment/photometry will
    # fail at runtime if these are required but missing.
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

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    phase_cross_correlation = None  # subpixel refinement disabled if skimage missing

# =============================================================================
# Custom / Local Module Imports
# =============================================================================
try:
    from catalog import Catalog
except ModuleNotFoundError:
    Catalog = None
try:
    from functions import (
        border_msg,
        distance_to_uniform_row_col,
        get_header,
        get_image,
        get_image_stats,
        save_to_fits,
    )
except (ModuleNotFoundError, ImportError):
    # Download-only use cases shouldn't require the full photometry stack.
    def border_msg(message: Any, *args: Any, **kwargs: Any) -> str:
        return str(message)

    distance_to_uniform_row_col = None
    get_header = None
    get_image = None
    get_image_stats = None
    save_to_fits = None
from wcs import get_wcs
try:
    from utils import run_IDC
except (ModuleNotFoundError, ImportError):
    run_IDC = None

from functions import log_warning_from_exception

# =============================================================================
# External Tool Imports
# =============================================================================
try:
    import legacystamps  # Optional: only required for Legacy Survey templates

    _HAS_LEGACYSTAMPS = True
except ImportError:
    legacystamps = None  # type: ignore[assignment]
    _HAS_LEGACYSTAMPS = False
try:
    # PyZOGY is only needed for ZOGY image subtraction.
    # Support both historical module naming styles used in different builds.
    try:
        from PyZOGY.subtract import run_subtraction
    except ImportError:
        from pyzogy.subtract import run_subtraction  # type: ignore[no-redef]
    _HAS_PYZOGY = True
except ImportError:
    run_subtraction = None  # type: ignore[assignment]
    _HAS_PYZOGY = False

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Constants
# =============================================================================

# Reproducible random number generator (replaces legacy np.random.* calls)
RNG = np.random.default_rng(seed=42)

# Sentinel value used to represent "no data" in FITS images after alignment.
# Chosen to be extremely small but non-zero so it passes finite checks
# but is trivially distinguishable from real flux.
NO_DATA_SENTINEL = 1e-30

# Supported PanSTARRS filter names
PANSTARRS_FILTERS = frozenset({"g", "r", "i", "z"})

# Valid 2MASS band identifiers (case-insensitive input, stored upper)
TWOMASS_VALID_BANDS = frozenset({"J", "H", "KS", "K"})
TWOMASS_MAX_SIZE_ARCMIN = 15

# Default sigma for sigma-clipped statistics throughout the pipeline
DEFAULT_SIGMA_CLIP = 3.0

# Default FWHM padding multiplier for masking around bright / saturated sources
DEFAULT_FWHM_PADDING_MULTIPLIER = 5


# =============================================================================
# Dataclass-Based Parameter Containers
# =============================================================================
# Using dataclasses gives us:
#   - Named, documented fields instead of loose dict keys
#   - Default values in one place
#   - Type checking via IDE / mypy
#   - Easy conversion to/from dicts for YAML compatibility


@dataclass
class MaskParams:
    """Parameters controlling image mask creation."""

    saturation_level: float = 2**16
    """Pixel value above which a source is considered saturated."""

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

    allow_subpixel_shifts: bool = False
    """If True, attempt sub-pixel alignment before subtraction."""

    sfft_crowded_method: bool = True
    """If True (default), use SFFT crowded-field (ECP). If False, use sparse (ESP). ECP is preferred; set False only to force ESP."""

    sfft_bg_order: int = 0
    """SFFT background spatial polynomial order (0=constant). Input images are assumed background-subtracted."""

    zogy_template_psf_independent: bool = True
    """If True (default), select PSF stars for the reference image independently from the
    reference image (better quality reference PSF). If False, use the same matched stars
    for both science and reference PSFs (original behaviour; reference stars may be poor)."""

    hotpants_exe_loc: str = "hotpants"
    """Filesystem path to the HOTPANTS executable, or a command on PATH (default: 'hotpants')."""


@dataclass
class FluxMatchParams:
    """Parameters for robust flux-consistency matching between catalogs."""

    flux_key: str = "flux_AP"
    """Column name for aperture flux."""

    flux_key_err: str = "flux_AP_err"
    """Column name for aperture flux uncertainty."""

    mag_residual_threshold: float = 0.4
    """Maximum allowed magnitude residual for an inlier (more relaxed by default)."""

    min_samples_fraction: float = 0.33
    """Minimum fraction of sources required by RANSAC."""

    max_trials: int = 500
    """RANSAC maximum iteration count."""

    min_absolute_samples: int = 3
    """Hard floor on the number of RANSAC samples."""

    use_percentile_cut: bool = False
    """Whether to apply a brightness percentile trim after RANSAC."""

    percentiles: Tuple[float, float] = (5.0, 99.0)
    """Lower and upper percentile bounds when percentile cut is active."""

    enforce_slope_constraint: bool = True
    """If True, penalise slopes far from 1.0 in the magnitude fit."""

    fix_slope_to_one: bool = True
    """If True, fix slope to 1 when fitting science vs reference (only fit intercept)."""


# =============================================================================
# Module-Level Helper Functions
# =============================================================================
# These are pure functions with no side effects, so they belong at module scope
# rather than being redefined inside methods on every call.


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
    fits.writeto(
        fpath,
        data,
        header,
        overwrite=overwrite,
        output_verify="silentfix+ignore",
    )


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
        mag_err = (2.5 / np.log(10)) * (flux_err / flux)
    return mag, mag_err


def clean_fits_nans(fpath: str) -> None:
    """
    Replace NaN / Inf pixels in a FITS image with NO_DATA_SENTINEL in-place.

    This is necessary before feeding images to external tools (HOTPANTS, SFFT)
    that cannot handle IEEE special values.
    """
    with fits.open(fpath, mode="update") as hdul:
        data = hdul[0].data
        bad = ~np.isfinite(data)
        if bad.any():
            data[bad] = NO_DATA_SENTINEL
            hdul.flush()


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


# =============================================================================
# Install PyZOGY
# =============================================================================


def install_pyzogy() -> None:
    """
    Download PyZOGY from GitHub into ~/Downloads and run ``setup.py install``.

    This is a convenience bootstrap for environments where PyZOGY is not
    available via pip.  It:
      1. Downloads the master-branch ZIP archive to a temp directory.
      2. Extracts it to ~/Downloads/PyZOGY.
      3. Runs ``python setup.py install`` inside the extracted folder.
    """
    repo_url = "https://github.com/dguevel/PyZOGY/archive/refs/heads/master.zip"
    downloads = Path.home() / "Downloads"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        logger.info("Downloading PyZOGY from GitHub...")
        zip_path = temp_dir_path / "PyZOGY-master.zip"
        urlretrieve(repo_url, str(zip_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir_path)

        dest = downloads / "PyZOGY"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(temp_dir_path / "PyZOGY-master", dest)
        logger.info("PyZOGY extracted to %s", dest)

        subprocess.run(
            [sys.executable, "setup.py", "install"],
            cwd=str(dest),
            check=True,
        )
        logger.info("PyZOGY installed successfully")


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

        # Try expanding upward
        if min_row > 0 and mask[min_row - 1, min_col : max_col + 1].all():
            min_row -= 1
            expanded = True

        # Try expanding downward
        if max_row < rows - 1 and mask[max_row + 1, min_col : max_col + 1].all():
            max_row += 1
            expanded = True

        # Try expanding left
        if min_col > 0 and mask[min_row : max_row + 1, min_col - 1].all():
            min_col -= 1
            expanded = True

        # Try expanding right
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
    # --- Single I/O pass per file ---
    image_data, header = read_fits(image_fpath)
    mask_data, _ = read_fits(mask_fpath)
    mask_data = mask_data.astype(bool)

    if image_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: image {image_data.shape} vs mask {mask_data.shape}."
        )

    # Estimate background from unmasked pixels
    bg_pixels = image_data[~mask_data]
    if bg_pixels.size == 0:
        raise ValueError("No unmasked pixels available for background estimation.")

    if apply_sigma_clip:
        mean, _, std = sigma_clipped_stats(bg_pixels, sigma=sigma)
    else:
        mean, std = float(np.mean(bg_pixels)), float(np.std(bg_pixels))

    # Generate synthetic noise
    n_fill = int(mask_data.sum())
    if use_poisson:
        noise = RNG.poisson(lam=max(mean, 0), size=n_fill).astype(float)
    else:
        noise = RNG.normal(loc=mean, scale=max(std, 0), size=n_fill)

    # Fill in-place (no copy needed - we own image_data)
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
        One of 'g', 'r', 'i', 'z'.

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

        # Step 3: prepare output path
        sub_folder = Path(template_folder) / f"{band}p_template"
        sub_folder.mkdir(parents=True, exist_ok=True)
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
            new_header.update(template_wcs.to_header(relax=True), relax=True)

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

    sub_folder = Path(template_folder) / f"{band}p_template"
    sub_folder.mkdir(parents=True, exist_ok=True)
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
        cutout = Cutout2D(
            data,
            (xc, yc),
            size=(2 * half_size_px, 2 * half_size_px),
            wcs=template_wcs,
            mode="partial",
            fill_value=np.nan,
        )
        cutout_data = cutout.data
        cutout_wcs = cutout.wcs
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
        if cutout_wcs is not None:
            new_header.update(cutout_wcs.to_header(relax=True), relax=True)
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

    Builds the Legacy Survey viewer cutout URL and downloads via wget. Downloads
    the combined multi-band FITS (gri) once, then saves g, r, and i band images
    into their template subfolders. Size is interpreted as arcminutes. Requires
    wget to be installed and on PATH.

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

    # Path for requested band; subfolders must match find_templates (gp_template, etc.)
    template_base = Path(template_folder)
    template_base.mkdir(parents=True, exist_ok=True)
    out_path = template_base / f"{band}p_template" / f"legacy_{band}_band_template.fits"

    if out_path.exists():
        logger.info("Legacy template already exists at %s - skipping", out_path)
        return str(out_path)

    # Combined multi-band file (same naming as Legacy Survey cutout layer)
    combined_name = f"legacystamps_{ra:.6f}_{dec:.6f}_{LEGACY_CUTOUT_LAYER}.fits"
    combined_path = template_base / combined_name

    if not combined_path.exists():
        logger.info(
            "Downloading Legacy Survey template (gri) for (%.4f, %.4f), size=%.2f arcmin",
            ra,
            dec,
            size,
        )
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

    # Determine band order from header when possible (more robust than assuming gri).
    # The Legacy cutout header typically provides BANDS='gri' and BAND0/BAND1/BAND2.
    header_bands = None
    try:
        hb = header.get("BANDS")
        if isinstance(hb, str) and hb.strip():
            header_bands = tuple(hb.strip())
    except Exception:
        header_bands = None
    if not header_bands:
        # Fallback: try BAND0/BAND1/... keywords
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
        sub_folder = template_base / f"{b}p_template"
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
            fits.writeto(
                str(band_path),
                band_data,
                band_header,
                overwrite=True,
                output_verify="silentfix+ignore",
            )
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

    # Normalise alias
    query_band = "KS" if band == "K" else band

    sub_folder = Path(template_folder) / f"{query_band}_template"
    sub_folder.mkdir(parents=True, exist_ok=True)
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
        out_path.write_bytes(img_resp.content)

        # Stamp standard header keywords
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
# Constrained Slope Regressor
# =============================================================================


class ConstrainedSlopeRegressor(BaseEstimator, RegressorMixin):
    """
    Ordinary-least-squares linear fit with an optional soft constraint
    that penalises slopes far from a target value, or a fixed slope.

    This is designed for the photometric magnitude-comparison fit where
    a slope of ~1.0 is physically expected (flux-consistent sources).
    When *fixed_slope* is set, only the intercept is fitted (slope fixed).

    Parameters
    ----------
    slope_constraint : float
        Target slope value (default 1.0).
    slope_tolerance : float
        Half-width of the zero-penalty zone around *slope_constraint*.
    enforce : bool
        If False, the constraint is disabled entirely.
    fixed_slope : float, optional
        If set, slope is fixed to this value and only intercept is fitted
        (mean(y - fixed_slope * x)). Takes precedence over enforce.
    """

    def __init__(
        self,
        slope_constraint: float = 1.0,
        slope_tolerance: float = 0.1,
        enforce: bool = True,
        fixed_slope: Optional[float] = None,
    ):
        self.slope_constraint = slope_constraint
        self.slope_tolerance = slope_tolerance
        self.enforce = enforce
        self.fixed_slope = fixed_slope

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstrainedSlopeRegressor":
        if check_X_y is None or not _SKLEARN_AVAILABLE:
            raise ModuleNotFoundError(
                "scikit-learn is required for ConstrainedSlopeRegressor."
            )
        X, y = check_X_y(X, y)
        x = X.ravel()

        if self.fixed_slope is not None:
            self.slope_ = float(self.fixed_slope)
            self.intercept_ = float(np.nanmean(y - self.fixed_slope * x))
            return self

        def _loss(params: np.ndarray) -> float:
            slope, intercept = params
            mse = np.mean((y - (slope * x + intercept)) ** 2)
            if self.enforce:
                violation = max(
                    0.0, abs(slope - self.slope_constraint) - self.slope_tolerance
                )
                return mse + 100.0 * violation
            return mse

        init = [1.0, float(np.mean(y) - np.mean(x))]
        result = minimize(_loss, init, method="L-BFGS-B")
        self.slope_, self.intercept_ = result.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if check_is_fitted is None or check_array is None or not _SKLEARN_AVAILABLE:
            raise ModuleNotFoundError(
                "scikit-learn is required for ConstrainedSlopeRegressor.predict."
            )
        check_is_fitted(self)
        X = check_array(X)
        return self.slope_ * X.ravel() + self.intercept_


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

        for _, row in subset.iterrows():
            ap = CircularAperture((row["x_pix"], row["y_pix"]), r=radius)
            mask += ap.to_mask().to_image(shape=shape).astype(np.int32)

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
        non_uniform_rows = ~np.all(img == img[:, 0:1], axis=1)
        non_uniform_cols = ~np.all(img == img[0:1, :], axis=0)

        row_idx = np.where(non_uniform_rows)[0]
        col_idx = np.where(non_uniform_cols)[0]

        top, bottom = int(row_idx[0]), int(row_idx[-1])
        left, right = int(col_idx[0]), int(col_idx[-1])

        center_y = (top + bottom) / 2.0
        center_x = (left + right) / 2.0

        return center_y, center_x, top, bottom, left, right

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
                border_msg(f"Masking bright sources from {catalogName} catalog")
            )
            target = SkyCoord(
                self.input_yaml["target_ra"],
                self.input_yaml["target_dec"],
                unit=(u.deg, u.deg),
            )
            imageWCS = get_wcs(header)

            sequenceData = Catalog(input_yaml=self.input_yaml)
            bright_sources_catalog = sequenceData.download(
                target, catalogName=catalogName
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
        fwhm: int = 5,
        npixels: int = 8,
        padding: int = 10,
        snr_limit: int = 3000,
        create_source_mask: bool = True,
        ignore_position: Optional[List[Tuple[float, float]]] = None,
        remove_large_sources: bool = True,
        bright_sources: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Build a binary mask flagging problematic sources in *data*.

        Sources are detected via sigma-clipped thresholding and
        segmentation.  The mask includes:
          - Saturated sources (peak > saturation level).
          - Negative-peak sources (likely artefacts).
          - Anomalously large sources (optional, via sigma-clip on area).
          - Sources overlapping known bright-catalog objects.

        Parameters
        ----------
        data : np.ndarray
            2-D science image.
        params : MaskParams or None
            Structured parameter set.  If None, falls back to keyword args.
        ignore_position : list of (x, y)
            Positions whose enclosing source should *not* be masked
            (typically the transient target).

        Returns
        -------
        mask : np.ndarray (int)
            Binary mask (1 = masked, 0 = good).
        masked_centres : list of (x, y)
            Centroids of all masked sources.
        """
        if ignore_position is None:
            ignore_position = []

        # Be strict about the input type for photutils.
        data = np.asarray(data)
        if data.ndim != 2:
            raise TypeError(
                f"create_image_mask expected 2-D image data, got shape {data.shape}"
            )

        # Resolve parameters: prefer dataclass, fall back to kwargs
        if params is not None:
            sat_lvl = params.saturation_level
            fwhm = params.fwhm
            padding = params.padding
            snr_limit = params.snr_limit

        masked_centres: List[Tuple[float, float]] = []
        mask = np.zeros(data.shape, dtype=np.int32)

        try:
            # --- Background statistics ---
            _, image_median, image_std = sigma_clipped_stats(
                data,
                sigma=DEFAULT_SIGMA_CLIP,
                cenfunc=np.nanmedian,
                stdfunc="mad_std",
            )

            # Ensure FWHM is even (required by kernel builder)
            fwhm = int(fwhm)
            if fwhm % 2 != 0:
                fwhm += 1

            # Detection threshold and minimum connected area
            npixels_det_raw = float(np.pi * (float(fwhm) / 2.0) ** 2)
            npixels_det = int(npixels_det_raw)
            if npixels_det < 5:
                logger.warning(
                    "create_image_mask: computed npixels=%d from fwhm=%s (raw=%g); clamping to 5.",
                    int(npixels_det),
                    str(fwhm),
                    float(npixels_det_raw),
                )
                npixels_det = 5
            threshold = 5.0 * image_std + image_median

            # --- Source detection via segmentation ---
            seg = detect_sources(data, threshold, npixels=npixels_det)
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

            cat = SourceCatalog(data, seg, localbkg_width=15 * fwhm)
            tbl = cat.to_table().to_pandas()

            # Flag categories
            is_saturated = tbl["max_value"] > sat_lvl
            is_negative = tbl["max_value"] < 0

            if remove_large_sources:
                clipped = sigma_clip(tbl["area"], sigma=10, maxiters=10)
                is_large = clipped.mask
                tbl = tbl[is_saturated | is_negative | is_large]
            else:
                tbl = tbl[is_saturated | is_negative]

            # --- Mask bright-catalog overlaps ---
            if bright_sources is not None:
                for _, src in tbl.iterrows():
                    cx = (src["bbox_xmin"] + src["bbox_xmax"]) / 2
                    cy = (src["bbox_ymin"] + src["bbox_ymax"]) / 2
                    overlaps = any(
                        self.does_box_overlap(src, (bs["x_pix"], bs["y_pix"]), padding)
                        for _, bs in bright_sources.iterrows()
                    )
                    if overlaps:
                        seg_pixels = seg.data == src["label"]
                        mask[seg_pixels] = 1
                        masked_centres.append((cx, cy))

            # --- Mask remaining flagged sources (skip if near target) ---
            invalid_bbox_count = 0
            for _, src in tbl.iterrows():
                cx = (src["bbox_xmin"] + src["bbox_xmax"]) / 2
                cy = (src["bbox_ymin"] + src["bbox_ymax"]) / 2

                if any(
                    self.does_box_overlap(src, pos, padding) for pos in ignore_position
                ):
                    continue

                # Guard against degenerate/invalid segmentation boxes.
                x_min = float(src["bbox_xmin"])
                x_max = float(src["bbox_xmax"])
                y_min = float(src["bbox_ymin"])
                y_max = float(src["bbox_ymax"])
                if not np.all(np.isfinite([x_min, x_max, y_min, y_max, cx, cy])):
                    invalid_bbox_count += 1
                    continue

                # Photutils bounding boxes can be tight; enforce at least 1 pixel.
                w = max(1.0, x_max - x_min)
                h = max(1.0, y_max - y_min)
                if w <= 0 or h <= 0:
                    invalid_bbox_count += 1
                    continue

                rect = RectangularAperture((cx, cy), w=w, h=h, theta=0)
                rect_img = rect.to_mask().to_image(shape=mask.shape)
                if rect_img is None:
                    invalid_bbox_count += 1
                    continue
                mask += rect_img.astype(np.int32)
                masked_centres.append((cx, cy))

            if invalid_bbox_count > 0:
                logger.warning(
                    "create_image_mask: skipped %d invalid/degenerate segmentation boxes.",
                    invalid_bbox_count,
                )

            # Collapse overlapping regions
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

        logger.info(border_msg(f"Finding template file for filter {use_filter}"))
        fits_root = Path(self.input_yaml.get("fits_dir", None) or "") / "templates"
        use_filter = str(use_filter).strip()

        # Prefer modern naming for ugriz, keep legacy *p_template as fallback.
        dir_labels = [f"{use_filter}_template"]
        if use_filter in {"u", "g", "r", "i", "z"}:
            dir_labels.append(f"{use_filter}p_template")

        candidate_dirs = [fits_root / d for d in dir_labels]
        logger.info(
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
            result = str(candidates[0])
            if template_dir.name.endswith("p_template"):
                logger.info(
                    "Template filepath (legacy folder): %s",
                    result,
                )
            else:
                logger.info("Template filepath: %s", result)
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
        method: str = "swarp",
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Align science and template images to a common pixel grid.

        Three strategies are tried in cascading order:
          1. **SWarp** (SCAMP+SWarp astrometric solution)
          2. **AstroAlign** (feature-based affine transform)
          3. **Reproject** (pure WCS-based reprojection)

        The *method* parameter controls which strategy is tried first,
        but the cascade always falls through to reproject as a last resort.

        Returns
        -------
        (science_out, template_out) : tuple of str or None
            Paths to aligned images, or (None, None) on complete failure.
        """
        sci_name = Path(scienceFpath).name
        ref_name = Path(templateFpath).name
        logger.info(
            border_msg(
                f"Aligning science and reference images ({sci_name} vs {ref_name})"
            )
        )

        try:
            scienceDir = Path(scienceFpath).parent
            new_templateFpath = str(scienceDir / Path(templateFpath).name)

            # Read headers (single I/O pass each)
            scienceImage, scienceHeader = read_fits(scienceFpath)
            templateImage, templateHeader = read_fits(templateFpath)
            imageWCS = get_wcs(scienceHeader)

            # ------------------------------------------------------------------
            # Helper: assess astrometric quality of an alignment
            # ------------------------------------------------------------------
            def _log_alignment_rms(
                sci_path: str,
                ref_path: str,
                fwhm_pixels: float,
            ) -> Optional[float]:
                """
                Log approximate alignment RMS (diagnostic only). Does not reject.
                """
                try:
                    sci_data, _ = read_fits(sci_path)
                    ref_data, _ = read_fits(ref_path)
                    if sci_data.shape != ref_data.shape:
                        return None
                    from astropy.stats import sigma_clipped_stats as _scs

                    _, med_sci, std_sci = _scs(sci_data, sigma=3.0)
                    _, med_ref, std_ref = _scs(ref_data, sigma=3.0)
                    fwhm = max(float(fwhm_pixels), 2.0)
                    daofind_sci = DAOStarFinder(fwhm=fwhm, threshold=5.0 * std_sci)
                    daofind_ref = DAOStarFinder(fwhm=fwhm, threshold=5.0 * std_ref)
                    tbl_sci = daofind_sci(sci_data - med_sci) or []
                    tbl_ref = daofind_ref(ref_data - med_ref) or []
                    if len(tbl_sci) < 5 or len(tbl_ref) < 5:
                        return None
                    sci_xy = np.vstack(
                        (tbl_sci["xcentroid"].data, tbl_sci["ycentroid"].data)
                    ).T
                    ref_xy = np.vstack(
                        (tbl_ref["xcentroid"].data, tbl_ref["ycentroid"].data)
                    ).T
                    if len(sci_xy) < 5 or len(ref_xy) < 5:
                        return None

                    # Use mutual nearest-neighbor pairs and clip by a generous
                    # distance tied to image FWHM. One-way nearest-neighbor
                    # pairing can overestimate RMS in crowded fields.
                    max_sep = float(max(2.5, 2.5 * fwhm))
                    tree_ref = cKDTree(ref_xy)
                    d_sr, i_sr = tree_ref.query(sci_xy, k=1)
                    tree_sci = cKDTree(sci_xy)
                    d_rs, i_rs = tree_sci.query(ref_xy, k=1)

                    if len(i_sr) == 0 or len(i_rs) == 0:
                        return None
                    idx_s = np.arange(len(sci_xy), dtype=int)
                    mutual = (i_rs[i_sr] == idx_s) & np.isfinite(d_sr)
                    if not np.any(mutual):
                        return None
                    d_mut = d_sr[mutual]
                    d_mut = d_mut[np.isfinite(d_mut) & (d_mut <= max_sep)]
                    if len(d_mut) < 10:
                        return None

                    med = float(np.nanmedian(d_mut))
                    p90 = float(np.nanpercentile(d_mut, 90.0))
                    rms = float(np.sqrt(np.mean(d_mut**2)))
                    logger.info(
                        "Alignment offset diagnostics: median=%.3f px rms=%.3f px p90=%.3f px (%d mutual pairs, max_sep=%.2f px).",
                        med,
                        rms,
                        p90,
                        int(len(d_mut)),
                        max_sep,
                    )
                    # Return a robust score for gating; RMS can be dominated by
                    # a small high-residual tail in crowded fields.
                    return med
                except Exception:
                    return None

            # ------------------------------------------------------------------
            # Strategy implementations
            # ------------------------------------------------------------------

            def _reproject() -> Tuple[Optional[str], Optional[str]]:
                def _prepare_projection_header(header_in):
                    """
                    Build a projection header for reproject while preserving
                    distortion keywords (SIP/PV/TPV) where possible.
                    """
                    hdr = header_in.copy()
                    sip_keys = ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")
                    if any(k in hdr for k in sip_keys):
                        for ctype_key in ("CTYPE1", "CTYPE2"):
                            cval = str(hdr.get(ctype_key, ""))
                            # SIP is defined for TAN projections; avoid invalid
                            # projection strings like RA---TPV-SIP.
                            if (
                                cval
                                and ("TAN" in cval)
                                and not cval.endswith("-SIP")
                            ):
                                hdr[ctype_key] = f"{cval}-SIP"
                    return hdr

                def _distortion_summary(header_in) -> str:
                    keys = list(header_in.keys())
                    pv = sum(1 for k in keys if str(k).startswith("PV"))
                    sip = sum(
                        1
                        for k in keys
                        if str(k).startswith(("A_", "B_", "AP_", "BP_", "SIP_"))
                    )
                    c1 = str(header_in.get("CTYPE1", ""))
                    c2 = str(header_in.get("CTYPE2", ""))
                    return f"CTYPE1={c1} CTYPE2={c2} PV={pv} SIP={sip}"

                align_cfg = self.input_yaml.get("alignment", {})
                method_name = (
                    str(align_cfg.get("reproject_method", "adaptive")).lower().strip()
                )
                # Reproject defaults roundtrip_coords=True; keeping True avoids
                # one-way WCS glitches that show up as systematic misalignment.
                roundtrip = bool(align_cfg.get("reproject_roundtrip_coords", True))
                interp_order_raw = align_cfg.get("reproject_interp_order", "bicubic")
                interp_order = _normalize_reproject_interp_order(interp_order_raw)
                parallel = bool(align_cfg.get("reproject_parallel", False))
                # conserve_flux=True rescales adaptive output; for similar plate
                # scales it can blur or skew template-science match vs. library default.
                conserve_flux = bool(
                    align_cfg.get("reproject_adaptive_conserve_flux", False)
                )
                center_jacobian = bool(
                    align_cfg.get("reproject_adaptive_center_jacobian", False)
                )
                logger.info(
                    "Aligning via WCS reproject (%s) order=%s roundtrip=%s parallel=%s "
                    "adaptive_conserve_flux=%s adaptive_center_jacobian=%s",
                    method_name,
                    interp_order,
                    roundtrip,
                    parallel,
                    conserve_flux,
                    center_jacobian,
                )
                try:
                    shape_out = scienceImage.shape
                    # Use FITS headers directly for reproject so distortion
                    # conventions (SIP/PV/TPV) are preserved in the transform.
                    template_proj = _prepare_projection_header(templateHeader)
                    science_proj = _prepare_projection_header(scienceHeader)
                    logger.info(
                        "Reproject distortion inputs: template[%s] -> science[%s]",
                        _distortion_summary(template_proj),
                        _distortion_summary(science_proj),
                    )
                    # Honour requested method; if it fails, fall back in a predictable order.
                    # (exact -> adaptive -> interp) gives robust behaviour and consistent logs.
                    if method_name in ("exact", "adaptive", "interp"):
                        fallbacks = [method_name] + [
                            m
                            for m in ("exact", "adaptive", "interp")
                            if m != method_name
                        ]
                    else:
                        logger.warning(
                            "Unknown reproject_method=%r; defaulting to 'exact' with fallbacks.",
                            method_name,
                        )
                        fallbacks = ["exact", "adaptive", "interp"]

                    last_exc = None
                    aligned = footprint = None
                    used_method = None
                    import inspect as _inspect

                    _adaptive_extras: Dict[str, Any] = {}
                    try:
                        _asig = _inspect.signature(reproject_adaptive)
                        if "conserve_flux" in _asig.parameters:
                            _adaptive_extras["conserve_flux"] = conserve_flux
                        if "center_jacobian" in _asig.parameters:
                            _adaptive_extras["center_jacobian"] = center_jacobian
                    except (TypeError, ValueError):
                        pass

                    for m in fallbacks:
                        try:
                            if m == "adaptive":
                                aligned, footprint = reproject_adaptive(
                                    (templateImage, template_proj),
                                    output_projection=science_proj,
                                    shape_out=shape_out,
                                    roundtrip_coords=roundtrip,
                                    parallel=parallel,
                                    **_adaptive_extras,
                                )
                            elif m == "interp":
                                aligned, footprint = reproject_interp(
                                    (templateImage, template_proj),
                                    output_projection=science_proj,
                                    shape_out=shape_out,
                                    roundtrip_coords=roundtrip,
                                    order=interp_order,
                                    parallel=parallel,
                                )
                            else:
                                aligned, footprint = reproject_exact(
                                    (templateImage, template_proj),
                                    output_projection=science_proj,
                                    shape_out=shape_out,
                                    parallel=parallel,
                                )
                            used_method = m
                            break
                        except Exception as _e:
                            last_exc = _e
                            log_warning_from_exception(
                                logger, f"Reproject ({m}) failed", _e
                            )

                    if used_method is None or aligned is None or footprint is None:
                        raise RuntimeError(
                            f"All reproject methods failed; last error: {last_exc}"
                        )

                    fp_mask = footprint.astype(bool)
                    aligned = np.nan_to_num(
                        aligned,
                        nan=NO_DATA_SENTINEL,
                        posinf=NO_DATA_SENTINEL,
                        neginf=NO_DATA_SENTINEL,
                    )
                    aligned[~fp_mask] = NO_DATA_SENTINEL

                    # Optional subpixel refinement to reduce residual misalignment
                    if (
                        align_cfg.get("reproject_subpixel_refine", False)
                        and phase_cross_correlation is not None
                    ):
                        try:
                            ref_valid = np.asarray(
                                np.isfinite(scienceImage)
                                & (np.abs(scienceImage) < 1e30),
                                dtype=bool,
                            )
                            mov_valid = np.asarray(
                                (aligned != NO_DATA_SENTINEL) & np.isfinite(aligned),
                                dtype=bool,
                            )
                            if np.sum(ref_valid) > 100 and np.sum(mov_valid) > 100:
                                ref_img = np.asarray(scienceImage, dtype=np.float64)
                                mov_img = np.asarray(aligned, dtype=np.float64)
                                shift_result = phase_cross_correlation(
                                    ref_img,
                                    mov_img,
                                    reference_mask=ref_valid,
                                    moving_mask=mov_valid,
                                )
                                subpix_shift = (
                                    shift_result[0]
                                    if isinstance(shift_result, tuple)
                                    else shift_result
                                )
                                subpix_shift = np.atleast_1d(
                                    np.asarray(subpix_shift, dtype=float)
                                )
                                if subpix_shift.size >= 2 and np.all(
                                    np.abs(subpix_shift[:2]) < 5.0
                                ):
                                    aligned = ndimage_shift(
                                        aligned,
                                        (
                                            float(subpix_shift[0]),
                                            float(subpix_shift[1]),
                                        ),
                                        order=3,
                                        mode="constant",
                                        cval=np.nan,
                                    )
                                    aligned = np.nan_to_num(
                                        aligned,
                                        nan=NO_DATA_SENTINEL,
                                        posinf=NO_DATA_SENTINEL,
                                        neginf=NO_DATA_SENTINEL,
                                    )
                                    logger.info(
                                        "Reproject subpixel refinement applied: shift (row, col) = (%.3f, %.3f)",
                                        float(subpix_shift[0]),
                                        float(subpix_shift[1]),
                                    )
                        except Exception as e:
                            log_warning_from_exception(
                                logger,
                                "Subpixel refinement failed (using WCS-aligned only)",
                                e,
                                exc_info=True,
                            )

                    hdr = templateHeader.copy()
                    hdr.update(science_proj, relax=True)
                    # Ensure header dimensions match the aligned array (science grid)
                    hdr["NAXIS1"] = aligned.shape[1]
                    hdr["NAXIS2"] = aligned.shape[0]
                    write_fits(new_templateFpath, aligned.astype(np.float32), hdr)

                    rms_pix = None
                    try:
                        fwhm_pix = float(self.input_yaml.get("fwhm", 3.0))
                        rms_pix = _log_alignment_rms(
                            scienceFpath, new_templateFpath, fwhm_pix
                        )
                    except Exception:
                        rms_pix = None

                    logger.info(
                        "Alignment via WCS reproject (%s) succeeded.", used_method
                    )
                    return scienceFpath, new_templateFpath
                except Exception as exc:
                    log_warning_from_exception(
                        logger, "Reproject failed", exc, exc_info=True
                    )
                    return None, None

            def _swarp() -> Tuple[Optional[str], Optional[str]]:
                logger.info("Attempting SWarp + SCAMP alignment.")
                idc = run_IDC.ImageDistortionCorrector(input_yaml=self.input_yaml)
                res = idc.align_and_resample_both_images(scienceFpath, templateFpath)
                if res and res.get("science_aligned"):
                    sci_al, ref_al = res["science_aligned"], res["reference_aligned"]
                    fwhm_pix = float(self.input_yaml.get("fwhm", 3.0))
                    _log_alignment_rms(sci_al, ref_al, fwhm_pix)
                    logger.info("SWarp alignment succeeded.")
                    return sci_al, ref_al
                logger.info("SWarp alignment did not produce aligned outputs.")
                return None, None

            def _astroalign() -> Tuple[Optional[str], Optional[str]]:
                logger.info("Attempting AstroAlign.")
                idc = run_IDC.ImageDistortionCorrector(input_yaml=self.input_yaml)
                res = idc.align_with_astroalign(scienceFpath, templateFpath)
                if res and res.get("science_aligned"):
                    sci_al, ref_al = res["science_aligned"], res["reference_aligned"]
                    fwhm_pix = float(self.input_yaml.get("fwhm", 3.0))
                    _log_alignment_rms(sci_al, ref_al, fwhm_pix)
                    logger.info("AstroAlign alignment succeeded.")
                    return sci_al, ref_al
                logger.info("AstroAlign did not produce aligned outputs.")
                return None, None

            # ------------------------------------------------------------------
            # Cascade
            # ------------------------------------------------------------------
            if method == "swarp":
                out = _swarp()
                if out[0]:
                    return out

            if method in ("astroalign", "swarp"):
                out = _astroalign()
                if out[0]:
                    return out

            if method == "reproject":
                out = _reproject()
                if out[0]:
                    return out
                # Reproject failed or failed quality gate: try feature/scamp fallback.
                out = _astroalign()
                if out[0]:
                    return out
                return _swarp()

            return _reproject()

        except Exception:
            logger.exception("Error during alignment")
            return None, None

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
        wcs_obj = WCS(header)
        center = ((coords[2] + coords[0]) // 2, (coords[3] + coords[1]) // 2)
        size = (coords[2] - coords[0] + 1, coords[3] - coords[1] + 1)
        cutout = Cutout2D(data, position=center, size=size, wcs=wcs_obj)
        return cutout.data, cutout.wcs

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
        # Corrected: AND instead of OR
        valid = np.isfinite(image) & (image != 0)
        rows, cols = valid.shape
        height = np.zeros(cols, dtype=int)

        best_area = 0
        best_coords = (0, 0, 0, 0)

        for r in range(rows):
            # Build histogram: cumulative height of consecutive valid pixels
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
            border_msg(
                f"Cropping science and reference images ({sci_name} vs {ref_name})"
            )
        )

        target_ra = self.input_yaml["target_ra"]
        target_dec = self.input_yaml["target_dec"]
        target_x_pix = np.floor(self.input_yaml["target_x_pix"])
        target_y_pix = np.floor(self.input_yaml["target_y_pix"])

        # --- Load science image (single I/O) ---
        scienceImage, scienceHeader = read_fits(scienceFpath)
        imageWCS = WCS(scienceHeader, relax=True)

        cy, cx, top, bot, left, right = self.find_non_uniform_center(scienceImage)
        height = bot - top
        width = right - left

        # Ensure even dimensions (needed for FFT-based subtraction later)
        height -= height % 2
        width -= width % 2

        scienceDir = Path(scienceFpath).parent
        cropped_scienceFpath = str(scienceDir / Path(scienceFpath).name)

        # ------------------------------------------------------------------
        # Science-only crop (no template)
        # ------------------------------------------------------------------
        if templateFpath is None:
            cutout = Cutout2D(
                scienceImage,
                position=(np.floor(cx), np.floor(cy)),
                size=(height, width),
                wcs=imageWCS,
                mode="partial",
                fill_value=NO_DATA_SENTINEL,
            )
            imageWCS = WCS(cutout.wcs.to_header(relax=True), relax=True)
            scienceHeader.update(imageWCS.to_header(relax=True), relax=True)
            scienceImage = cutout.data

            # Try tighter crop to largest valid rectangle
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
                scienceImage = scienceImage_tmp
                scienceHeader.update(scienceHeader_newwcs.to_header(relax=True), relax=True)

            write_fits(cropped_scienceFpath, scienceImage, scienceHeader)
            return cropped_scienceFpath, None

        # ------------------------------------------------------------------
        # Joint science + template crop
        # ------------------------------------------------------------------
        templateImage, templateHeader = read_fits(templateFpath)
        templateWCS = WCS(templateHeader, relax=True)
        cropped_templateFpath = str(scienceDir / Path(templateFpath).name)

        cy_t, cx_t, top_t, bot_t, left_t, right_t = self.find_non_uniform_center(
            templateImage
        )
        height_t = bot_t - top_t
        width_t = right_t - left_t

        # Use the smaller dimension from either image
        if height_t < height:
            height, cy = height_t, cy_t
        if width_t < width:
            width, cx = width_t, cx_t

        size = (height - height % 2, width - width % 2)
        position = (np.floor(cx), np.floor(cy))

        # Check proximity to padding edges
        d_uniform_template = distance_to_uniform_row_col(
            templateImage, x=target_x_pix, y=target_y_pix
        )
        d_uniform_science = distance_to_uniform_row_col(
            scienceImage, x=target_x_pix, y=target_y_pix
        )

        if np.isfinite(d_uniform_template) or np.isfinite(d_uniform_science):
            # Initial cutout
            scienceCutout = Cutout2D(
                scienceImage,
                position,
                size,
                wcs=imageWCS,
                mode="trim",
                fill_value=NO_DATA_SENTINEL,
            )
            scienceImage = scienceCutout.data
            imageWCS = WCS(scienceCutout.wcs.to_header(relax=True), relax=True)
            scienceHeader.update(imageWCS.to_header(relax=True), relax=True)

            templateCutout = Cutout2D(
                templateImage,
                position,
                size,
                wcs=templateWCS,
                mode="trim",
                fill_value=NO_DATA_SENTINEL,
            )
            templateImage = templateCutout.data
            templateWCS = WCS(templateCutout.wcs.to_header(relax=True), relax=True)
            templateHeader.update(templateWCS.to_header(relax=True), relax=True)

            # Mark shared invalid regions
            mask = (templateImage == NO_DATA_SENTINEL) | (
                scienceImage == NO_DATA_SENTINEL
            )
            templateImage[mask] = np.nan
            scienceImage[mask] = np.nan

            # Tight crop to largest valid rectangle
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
                scienceImage = scienceImage_tmp
                templateImage = templateImage_tmp
                scienceHeader.update(scienceHeader_newwcs.to_header(relax=True), relax=True)
                templateHeader.update(templateHeader_newwcs.to_header(relax=True), relax=True)
            else:
                logger.info("Target too close to border; keeping initial crop")

            # Final NaN cleanup
            templateImage[~np.isfinite(templateImage)] = np.nan
            scienceImage[~np.isfinite(scienceImage)] = np.nan

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

        The padding value is NO_DATA_SENTINEL rather than zero to avoid
        division-by-zero issues in Fourier-domain subtraction methods.
        """
        dy = target_shape[0] - psf.shape[0]
        dx = target_shape[1] - psf.shape[1]
        pad_width = (
            (dy // 2, dy - dy // 2),
            (dx // 2, dx - dx // 2),
        )
        return np.pad(psf, pad_width, mode="constant", constant_values=NO_DATA_SENTINEL)

    @staticmethod
    def determine_kernel_order(
        sci_fwhm: float,
        ref_fwhm: float,
    ) -> Tuple[int, str]:
        """
        Choose an appropriate spatial-kernel polynomial order based on
        how different the science and reference PSFs are.

        Returns
        -------
        (order, explanation) : (int, str)
        """
        avg = (sci_fwhm + ref_fwhm) / 2.0
        rel_diff = abs(sci_fwhm - ref_fwhm) / avg if avg > 0 else 0

        if rel_diff < 0.1:
            return 0, "PSFs very similar (<10% difference). Constant kernel."
        elif rel_diff < 0.3:
            return 1, "Moderate PSF difference (10-30%). Linear kernel."
        elif rel_diff < 0.5:
            return 2, "Significant PSF difference (30-50%). Quadratic kernel."
        else:
            return 3, "Large PSF difference (>50%). Cubic kernel."

    # -----------------------------------------------------------------
    # Robust outlier detection
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
        Identify outliers using a rolling robust statistic.

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

        # --- Global fallback for small samples ---
        if n < min_for_window:
            med = np.median(values)
            scale = median_abs_deviation(values) if use_mad else np.std(values)
            return np.abs(values - med) < n_sigma * scale

        # --- Rolling-window approach ---
        # Sort, but remember the original order so we can unsort at the end.
        sort_idx = np.argsort(values)
        sorted_vals = values[sort_idx]

        s = pd.Series(sorted_vals)
        rolling_med = s.rolling(window_size, center=True, min_periods=1).median().values

        if use_mad:
            rolling_scale = (
                s.rolling(window_size, center=True, min_periods=1)
                .apply(median_abs_deviation, raw=True)
                .values
            )
        else:
            rolling_scale = (
                s.rolling(window_size, center=True, min_periods=1).std().values
            )

        # Ensure writable array before in-place modification
        rolling_scale = np.array(rolling_scale, copy=True)

        # Fill NaN scale values with the global scale
        global_scale = (
            median_abs_deviation(sorted_vals) if use_mad else np.std(sorted_vals)
        )
        rolling_scale[np.isnan(rolling_scale)] = global_scale

        residuals = np.abs(sorted_vals - rolling_med)
        inlier_sorted = residuals < n_sigma * rolling_scale

        # Invert the sort permutation to return mask in original order
        inlier_original = np.empty(n, dtype=bool)
        inlier_original[sort_idx] = inlier_sorted
        return inlier_original

    # -----------------------------------------------------------------
    # Flux-consistency matching
    # -----------------------------------------------------------------

    def find_flux_consistent_sources(
        self,
        catalog_img: pd.DataFrame,
        catalog_tpl: pd.DataFrame,
        params: Optional[FluxMatchParams] = None,
        make_plot: bool = True,
    ) -> Tuple[pd.DataFrame, Tuple[float, float]]:
        """
        Cross-match science and template photometric catalogs, rejecting
        outliers via RANSAC and robust statistics.

        The procedure:
          1. Convert aperture fluxes to instrumental magnitudes.
          2. Apply error and positivity cuts.
          3. Remove rolling-window outliers.
          4. Fit a constrained linear relation (slope ~ 1) with RANSAC.
          5. Optionally trim to a central percentile range and re-fit.
          6. Return inlier catalog rows and a compact fit tuple
             ``(mag_slope, flux_scale)`` where:
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

        Returns
        -------
        (inlier_df, (mag_slope, flux_scale))
        """
        if params is None:
            params = FluxMatchParams()

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
            if not ok.any():
                logger.info("No sources pass positivity + SNR cuts")
                return empty, nan_fit

            # Extract matched arrays
            f_img = catalog_img[params.flux_key].values[ok]
            f_tpl = catalog_tpl[params.flux_key].values[ok]
            fe_img = catalog_img[params.flux_key_err].values[ok]
            fe_tpl = catalog_tpl[params.flux_key_err].values[ok]

            mag_img, me_img = flux_to_mag(f_img, fe_img)
            mag_tpl, me_tpl = flux_to_mag(f_tpl, fe_tpl)

            # Magnitude-error cut
            good = (me_img < 1) & (me_tpl < 1)
            if not good.any():
                logger.info("No sources pass error threshold")
                return empty, nan_fit

            mag_img, mag_tpl = mag_img[good], mag_tpl[good]
            mag_err = np.sqrt(me_img[good] ** 2 + me_tpl[good] ** 2)
            indices = catalog_img.index[ok][good]

            # --- Robust outlier removal in magnitude space ---
            robust_mask = self.robust_outlier_mask(
                mag_img,
                window_size=50,
                n_sigma=5,
                use_mad=True,
            )
            mag_img_r = mag_img[robust_mask]
            mag_tpl_r = mag_tpl[robust_mask]
            mag_err_r = mag_err[robust_mask]
            idx_r = indices[robust_mask]

            # --- Optional spatial thinning to avoid over-clustered regions ---
            if {"x", "y"}.issubset(catalog_img.columns):
                x_pos = catalog_img.loc[idx_r, "x"].to_numpy(dtype=float)
                y_pos = catalog_img.loc[idx_r, "y"].to_numpy(dtype=float)
                spatial_mask = self._spatially_uniform_mask(
                    x_pos,
                    y_pos,
                    value=mag_img_r,
                    n_bins=8,
                    max_per_bin=10,
                    prefer_small_value=True,
                )
                if spatial_mask.any():
                    mag_img_r = mag_img_r[spatial_mask]
                    mag_tpl_r = mag_tpl_r[spatial_mask]
                    mag_err_r = mag_err_r[spatial_mask]
                    idx_r = idx_r[spatial_mask]

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

            # NOTE: use module-level ConstrainedSlopeRegressor, not self.*
            if len(y) >= 4:
                try:
                    base_est = ConstrainedSlopeRegressor(
                        enforce=params.enforce_slope_constraint,
                        fixed_slope=1.0 if params.fix_slope_to_one else None,
                    )
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
                        method_used = "RANSAC"
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
                        est = ConstrainedSlopeRegressor(
                            enforce=params.enforce_slope_constraint,
                            fixed_slope=(
                                1.0 if params.fix_slope_to_one else None
                            ),
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
            bin_majority_frac = 0.25  # reject bin only if inlier fraction below this
            if final_inliers.sum() >= params.min_absolute_samples:
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

            # --- Build result DataFrame ---
            result = catalog_img.loc[idx_r].copy()
            result["mag_img"] = mag_img_r
            result["mag_tpl"] = mag_tpl_r
            result["mag_residual"] = y - (slope * mag_img_r + intercept)
            result["is_inlier"] = final_inliers
            result["is_robust"] = True

            non_robust_idx = indices[~robust_mask]
            nr = catalog_img.loc[non_robust_idx].copy()
            nr["mag_img"] = mag_img[~robust_mask]
            nr["mag_tpl"] = mag_tpl[~robust_mask]
            nr["mag_residual"] = np.nan
            nr["is_inlier"] = False
            nr["is_robust"] = False

            full = pd.concat([result, nr])

            # Keep mag-space fit as authoritative. For convenience we also return
            # the equivalent multiplicative flux scale from the intercept.
            # NOTE: this is not a full linear flux model when slope != 1.
            mag_slope = float(slope)
            flux_scale = 10 ** (-0.4 * intercept) if np.isfinite(intercept) else np.nan

            logger.info(
                "Fit [%s]: slope=%.3f, intercept=%.3f, " "inliers=%d/%d, robust=%d/%d",
                method_used,
                slope,
                intercept,
                final_inliers.sum(),
                len(y),
                len(mag_img_r),
                len(mag_img),
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
                )

            return (
                full[full["is_inlier"]].reset_index(drop=True),
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
    ) -> None:
        """Save a diagnostic magnitude-comparison plot to disk."""
        from matplotlib import pyplot as plt
        from functions import set_size

        plt.ioff()
        fig, ax = plt.subplots(figsize=set_size(340, 1))

        # All matched sources (faint, small markers to reduce overlap)
        ax.errorbar(
            full["mag_img"],
            full["mag_tpl"],
            xerr=mag_err_all,
            yerr=mag_err_all,
            fmt="o",
            color="lightgray",
            alpha=0.25,
            markersize=1.8,
            capsize=0,
            elinewidth=0.4,
            label="All",
        )
        # Robust candidates (before final inlier mask)
        ax.errorbar(
            robust["mag_img"],
            robust["mag_tpl"],
            xerr=mag_err_robust,
            yerr=mag_err_robust,
            fmt="o",
            color="steelblue",
            alpha=0.5,
            markersize=2.2,
            capsize=0,
            elinewidth=0.4,
            label="Robust",
        )
        # Final selected inliers - small outlined markers
        sel = robust["is_inlier"].to_numpy(bool)
        ax.errorbar(
            robust.loc[sel, "mag_img"],
            robust.loc[sel, "mag_tpl"],
            xerr=mag_err_robust[sel],
            yerr=mag_err_robust[sel],
            fmt="o",
            markersize=2.8,
            mfc="none",
            mec="#00AA00",
            ecolor="#00AA00",
            elinewidth=0.4,
            capsize=0,
            alpha=0.9,
            label="Selected inliers",
        )
        # Rejected robust points
        rej = ~sel
        ax.errorbar(
            robust.loc[rej, "mag_img"],
            robust.loc[rej, "mag_tpl"],
            xerr=mag_err_robust[rej],
            yerr=mag_err_robust[rej],
            fmt="x",
            color="#FF0000",
            alpha=0.5,
            markersize=2.2,
            capsize=0,
            elinewidth=0.4,
            label="Rejected",
        )
        xx = np.linspace(mag_img_robust.min(), mag_img_robust.max(), 100)
        ax.plot(xx, slope * xx + intercept, "b--", lw=0.5, label="Fit")
        ax.set(xlabel="Science mag", ylabel="Template mag")
        ax.legend(frameon=False, fontsize="small")
        ax.grid(alpha=0.3)
        ax.invert_xaxis()
        ax.invert_yaxis()

        base = Path(self.input_yaml["fpath"]).stem
        out = (
            Path(self.input_yaml["fpath"]).parent / f"flux_comparison_{base}.png"
        )
        fig.savefig(str(out), bbox_inches="tight", dpi=150)
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
            # Exclude self
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
        allow_subpixel_shifts: bool = False,
        common_sources: Optional[List[Tuple[float, float]]] = None,
        stamp_loc: Optional[str] = None,
        scienceNoise: Optional[str] = None,
        templateNoise: Optional[str] = None,
    ) -> Tuple[
        Optional[str], Optional[np.ndarray], Optional[List[Tuple[float, float]]]
    ]:
        """
        Subtract the template from the science image.

        Supports three backends that are tried in cascade when a method
        fails:
          - **ZOGY** (Zackay, Ofek & Gal-Yam 2016) via PyZOGY.
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
            Source positions used for kernel fitting.
        masked_sources : list of (x, y) or None
            Additional positions to exclude from fitting.
        stamp_loc : str or None
            Path to a stamp-selection file (HOTPANTS ``-ssf``).
        scienceNoise, templateNoise : str or None
            Optional external noise-map FITS files.

        Returns
        -------
        (differenceFpath, universal_mask, filtered_matching_sources)
            Paths and arrays, or (None, None, None) on failure.
        """
        if matching_sources is None:
            matching_sources = []
        if masked_sources is None:
            masked_sources = []
        if common_sources is None:
            common_sources = []

        t0 = time.time()
        sci_name = Path(scienceFpath).name
        ref_name = Path(templateFpath).name
        logger.info(border_msg(f"Starting image subtraction ({sci_name} - {ref_name})"))

        prepared_template_fpath: Optional[str] = None
        template_work_fpath: str = str(templateFpath)

        try:
            # =============================================================
            # 1. Load images (one read each)
            # =============================================================
            scienceImage, scienceHeader = read_fits(scienceFpath)
            templateImage, templateHeader = read_fits(templateFpath)
            scienceDir = Path(scienceFpath).parent
            base_name = Path(scienceFpath).name
            differenceFpath = str(scienceDir / f"diff_{base_name}")

            def _as_bool(value, default: bool = False) -> bool:
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
                    if v in ("0", "false", "f", "no", "n", "off", ""):
                        return False
                return default

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

            # Ensure reference (template) is background-subtracted so both inputs
            # to SFFT/HOTPANTS have comparable zero level (science is already
            # background-subtracted in main.py).
            template_invalid = ~np.isfinite(templateImage) | (
                np.abs(templateImage) < 1.1e-20
            )
            template_bg_median = 0.0
            if not template_invalid.all():
                _, template_bg_median, _ = sigma_clipped_stats(
                    templateImage, mask=template_invalid, sigma=3, maxiters=5
                )
                templateImage = templateImage - template_bg_median
                write_fits(_ensure_prepared_template_path(), templateImage, templateHeader)
                logger.info(
                    "Reference image background subtracted (median %.4g).",
                    float(template_bg_median),
                )

            # Keep interpolation to the WCS reproject stage only.

            # Extract relevant header values with sensible defaults
            science_fwhm = scienceHeader["fwhm"]
            template_fwhm = templateHeader.get("fwhm", 3)
            science_gain = float(scienceHeader.get("gain", 1))
            template_gain = float(templateHeader.get("gain", 1))

            # Saturation: header may be missing or effectively "no saturation".
            # Treat missing / non-finite / non-positive values as "no hard limit"
            # by mapping them to np.inf. Downstream code interprets np.inf as
            # "do not mask on saturation".
            def _safe_saturate(hdr: fits.Header) -> float:
                saturate_raw_value = hdr.get("saturate", np.inf)
                try:
                    sat = float(saturate_raw_value)
                except Exception:
                    sat = np.inf
                if not np.isfinite(sat) or sat <= 0:
                    return np.inf
                return sat

            science_saturate = _safe_saturate(scienceHeader)
            template_saturate = _safe_saturate(templateHeader)

            # -------------------------------------------------------------
            # Optional: rescale reference (template) image to science scale.
            #
            # Some templates are stored in normalized units (e.g. ~1e-3) while
            # science frames are in ADU-like counts. SFFT/HOTPANTS are linear,
            # but their internal source detection, background estimation, and
            # numeric conditioning are not strictly scale-invariant. When the
            # template dynamic range is extremely small, SFFT can become
            # unstable and appear "oversubtracted".
            #
            # This rescales the template to match the science robust sigma on
            # finite pixels and propagates the scaling into saturation and gain
            # (so e- = ADU * gain remains consistent).
            # -------------------------------------------------------------
            ts_cfg_scale = self.input_yaml.get("template_subtraction", {}) or {}
            if _as_bool(ts_cfg_scale.get("scale_template_to_science", False), False):
                logger.info(
                    "Template scaling enabled: will rescale template to match science robust sigma when scale mismatch is large."
                )
                try:
                    sci_invalid = ~np.isfinite(scienceImage) | (
                        np.abs(scienceImage) < 1.1e-20
                    )
                    ref_invalid = ~np.isfinite(templateImage) | (
                        np.abs(templateImage) < 1.1e-20
                    )
                    _, _, sci_std = sigma_clipped_stats(
                        scienceImage, mask=sci_invalid, sigma=3, maxiters=5
                    )
                    _, _, ref_std = sigma_clipped_stats(
                        templateImage, mask=ref_invalid, sigma=3, maxiters=5
                    )
                    sci_std = float(sci_std)
                    ref_std = float(ref_std)
                    if np.isfinite(sci_std) and np.isfinite(ref_std) and ref_std > 0:
                        scale_fac = sci_std / ref_std
                        # Only apply when the mismatch is large (avoid tiny jitter).
                        if scale_fac >= 50 or scale_fac <= 0.02:
                            old_gain = (
                                float(template_gain)
                                if np.isfinite(template_gain)
                                else template_gain
                            )
                            old_sat = (
                                float(template_saturate)
                                if np.isfinite(template_saturate)
                                else template_saturate
                            )
                            templateImage = templateImage * scale_fac
                            # Saturation scales with pixel units
                            if np.isfinite(template_saturate):
                                template_saturate = float(template_saturate) * scale_fac
                            # Gain is e/ADU; if ADU scaled by scale_fac, gain scales inversely.
                            if np.isfinite(template_gain) and template_gain > 0:
                                template_gain = float(template_gain) / scale_fac

                            write_fits(
                                _ensure_prepared_template_path(),
                                templateImage,
                                templateHeader,
                            )
                            logger.info(
                                "Rescaled template to science scale: sci_std=%.4g, ref_std=%.4g, factor=%.4g. "
                                "template_gain: %s -> %.4g; template_saturate: %s -> %s.",
                                sci_std,
                                ref_std,
                                scale_fac,
                                f"{old_gain:.4g}" if np.isfinite(old_gain) else "inf",
                                float(template_gain),
                                f"{old_sat:.4g}" if np.isfinite(old_sat) else "inf",
                                (
                                    f"{template_saturate:.4g}"
                                    if np.isfinite(template_saturate)
                                    else "inf"
                                ),
                            )
                        else:
                            logger.info(
                                "Template/science scale looks consistent (sci_std=%.4g, ref_std=%.4g, factor=%.4g); no rescaling applied.",
                                sci_std,
                                ref_std,
                                scale_fac,
                            )
                    else:
                        logger.info(
                            "Template scaling enabled but robust sigma could not be estimated (sci_std=%.4g, ref_std=%.4g); no rescaling applied.",
                            sci_std,
                            ref_std,
                        )
                except Exception as _e:
                    logger.info("Template rescaling skipped/failed (non-fatal): %s", _e)
            # Adjust template saturation into the background-subtracted frame:
            # after subtracting template_bg_median, saturated pixels move from
            # SATURATE to SATURATE - template_bg_median in data units.
            if np.isfinite(template_saturate) and template_bg_median != 0.0:
                template_saturate = template_saturate - float(template_bg_median)
                if template_saturate <= 0:
                    template_saturate = np.inf

            # Optional: inpaint broken/saturated template cores (cosmetic/robustness).
            # This can reduce subtraction artefacts around very bright stars, but does not
            # recover lost flux. Controlled by YAML to keep default behaviour unchanged.
            ts_cfg_inp = (
                (self.input_yaml.get("template_subtraction") or {})
                if isinstance(self.input_yaml, dict)
                else {}
            )
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
            science_readnoise = float(scienceHeader.get("READNOISE", 1))
            template_readnoise = float(templateHeader.get("READNOISE", 1))

            # NOTE: `scale` in default_input.yml is a plotting/cutout helper and
            # is not the same as SFFT's kernel half-width.
            # Oversubtraction artefacts can occur if we pass an uninitialized
            # (often 0) kernel size into SFFT.
            #
            # Instead, derive SFFT kernel half-width from the measured FWHMs,
            # mirroring SFFT's internal KerHWRatio-based default behaviour.
            KER_HW_MIN = 3
            KER_HW_MAX = 50
            fwhm_ref = float(template_fwhm)
            fwhm_sci = float(science_fwhm)
            ker_hw = int(
                max(KER_HW_MIN, min(KER_HW_MAX, round(2.0 * max(fwhm_ref, fwhm_sci))))
            )
            scale = ker_hw
            target_location = [
                (self.input_yaml["target_x_pix"], self.input_yaml["target_y_pix"])
            ]

            # =============================================================
            # 2. Locate PSF models
            # =============================================================
            science_psf_files = glob.glob(str(scienceDir / "PSF_model_image*fits"))
            template_psf_files = glob.glob(str(scienceDir / "PSF_model_template*fits"))
            science_psf = science_psf_files[0] if science_psf_files else None
            template_psf = template_psf_files[0] if template_psf_files else None

            # =============================================================
            # 3. Build masks
            # =============================================================
            logger.info("Building science and template masks...")

            # NaN / sentinel masks
            template_mask_nans = (
                ((np.abs(templateImage) < 1.1e-20) & (templateImage != 0))
                | (~np.isfinite(templateImage))
            ).astype(np.int32)

            science_mask_nans = (
                ((np.abs(scienceImage) < 1.1e-20) & (scienceImage != 0))
                | (~np.isfinite(scienceImage))
            ).astype(np.int32)

            # Segmentation-based source masks
            template_seg_mask, template_seg_centers = self.create_image_mask(
                templateImage,
                sat_lvl=template_saturate,
                fwhm=template_fwhm,
                create_source_mask=False,
                ignore_position=target_location,
                remove_large_sources=True,
                padding=int(DEFAULT_FWHM_PADDING_MULTIPLIER * template_fwhm),
            )
            science_seg_mask, science_seg_centers = self.create_image_mask(
                scienceImage,
                sat_lvl=science_saturate,
                fwhm=science_fwhm,
                create_source_mask=False,
                ignore_position=target_location,
                remove_large_sources=True,
                padding=int(DEFAULT_FWHM_PADDING_MULTIPLIER * science_fwhm),
            )

            # Combined mask: NaN/invalid (essential) + segmentation-based source masks
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
            universal_mask_full = (mask_essential | mask_sources).astype(np.int32)

            # Dynamic mask cap: if the full mask would leave too few good pixels for
            # subtraction (e.g. HOTPANTS needs usable stamps), use only the essential
            # (NaN/invalid) mask so that source masks are effectively excluded.
            ts_cfg_early = self.input_yaml.get("template_subtraction", {})
            max_masked_frac = float(
                ts_cfg_early.get("subtraction_max_masked_fraction", 0.90)
            )
            total_pix = universal_mask_full.size
            full_masked_frac = (
                np.sum(universal_mask_full) / total_pix if total_pix > 0 else 0.0
            )

            if full_masked_frac > max_masked_frac:
                universal_mask = np.where(mask_essential, 1, 0).astype(np.int32)
                reduced_frac = (
                    np.sum(universal_mask) / total_pix if total_pix > 0 else 0.0
                )
                logger.info(
                    "Subtraction mask capped: full mask would mask %.1f%% (limit %.0f%%); "
                    "using only NaN/invalid mask (%.1f%% masked) so subtraction has enough good pixels.",
                    full_masked_frac * 100.0,
                    max_masked_frac * 100.0,
                    reduced_frac * 100.0,
                )
            else:
                universal_mask = universal_mask_full

            templateDir = os.path.dirname(templateFpath)
            mask_loc = os.path.join(templateDir, "universal_mask.fits")
            save_to_fits(universal_mask.astype(int), mask_loc)

            # Deduplicate masked centres
            masked_centers = deduplicate_points(
                science_seg_centers + template_seg_centers,
                min_sep=5.0,
            )

            masked_percentage = np.sum(universal_mask) / universal_mask.size * 100
            logger.info("Masked %.3f%% of pixels before subtraction", masked_percentage)

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
                    pixel_scale_arcsec = float(
                        WCS.utils.proj_plane_pixel_scales(sci_wcs)[0] * 3600.0
                    )
                except Exception:
                    pixel_scale_arcsec = 0.3
                area_sq_arcmin = (ny * nx) * (pixel_scale_arcsec / 60.0) ** 2
                n_src = len(matching_sources)
                density = n_src / area_sq_arcmin if area_sq_arcmin > 0 else 0.0
                min_sources = ts_cfg.get("sfft_crowded_min_sources", 300)
                min_density = ts_cfg.get("sfft_crowded_min_density", 1.5)
                is_crowded = n_src >= min_sources or density >= min_density
                # Always use ECP (crowded); sparse (ESP) is no longer used by default.
                ts_cfg["sfft_crowded_method"] = True
                if is_crowded:
                    logger.info(
                        "Image detected as crowded (%d sources, %.2f per sq arcmin) -> using SFFT crowded (ECP)",
                        n_src,
                        density,
                    )
                else:
                    logger.info(
                        "Image sparse (%d sources, %.2f per sq arcmin) -> using SFFT crowded (ECP) anyway",
                        n_src,
                        density,
                    )
            elif "sfft_crowded_method" not in ts_cfg and "crowded_field" not in ts_cfg:
                ts_cfg["sfft_crowded_method"] = (
                    True  # default to ECP; sparse (ESP) only when explicitly set False
                )
            # Kernel polynomial order: default to 0 unless the user explicitly overrides
            user_kernel = ts_cfg.get("kernel_order", None)
            if user_kernel is None or user_kernel < 0:
                kernel_order = 0
            else:
                kernel_order = int(user_kernel)

            # 4b. SFFT: default is crowded (ECP); sparse (ESP) only if user sets sfft_crowded_method: false.

            # Filter matching sources that fall on masked pixels
            filtered_matching_sources = [
                (x, y)
                for x, y in matching_sources
                if (
                    np.isfinite(x)
                    and np.isfinite(y)
                    and not universal_mask[int(y), int(x)]
                )
            ]
            matching_sources = filtered_matching_sources
            logger.info(
                "%d matching sources remain after mask filtering",
                len(matching_sources),
            )

            # =============================================================
            # 5. Run subtraction backend
            # =============================================================

            if method.lower() in ("zogy", "pyzogy"):
                method = self._subtract_zogy(
                    scienceFpath,
                    template_work_fpath,
                    differenceFpath,
                    scienceHeader,
                    science_psf,
                    template_psf,
                    science_saturate,
                    template_saturate,
                    method,
                )

            if method == "sfft":
                method = self._subtract_sfft(
                    scienceFpath,
                    template_work_fpath,
                    differenceFpath,
                    mask_loc,
                    scienceDir,
                    base_name,
                    masked_sources,
                    masked_centers,
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
                )

            if method == "hotpants":
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
                    scienceNoise,
                )
                if not success:
                    return None, None, None

            # =============================================================
            # 6. Validate output
            # =============================================================
            if not (
                os.path.isfile(differenceFpath) and os.path.getsize(differenceFpath) > 0
            ):
                logger.error("Difference file missing or empty")
                return None, None, None

            diff_data, diff_header = read_fits(differenceFpath)
            if np.all(np.isnan(diff_data)) or np.nanstd(diff_data) < 1e-5:
                logger.error(
                    "Difference image is invalid (all NaN or near-zero variance). "
                    "Subtraction backend may have written a bad file; treat as failure and use "
                    "original science image."
                )
                return None, None, None

            # Zero the difference image background (sigma-clipped median) so it is ~0
            invalid = ~np.isfinite(diff_data) | (np.abs(diff_data) < 1.1e-20)
            invalid = invalid | (universal_mask.astype(bool))
            if not invalid.all():
                _, diff_median, _ = sigma_clipped_stats(
                    diff_data, mask=invalid, sigma=3, maxiters=5
                )
                diff_data = diff_data - float(diff_median)
                write_fits(differenceFpath, diff_data, diff_header)
                logger.info(
                    "Difference image background zeroed (subtracted median %.4g).",
                    float(diff_median),
                )

            elapsed = time.time() - t0
            logger.info("Image subtraction completed in %.1f s", elapsed)

            return differenceFpath, universal_mask, matching_sources

        except Exception:
            logger.exception("Unhandled error in subtract()")
            return None, None, None
        finally:
            if prepared_template_fpath:
                try:
                    os.remove(prepared_template_fpath)
                except OSError:
                    pass

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
    ) -> str:
        """Attempt ZOGY subtraction; return next method to try on failure."""
        logger.info("Starting ZOGY subtraction...")
        try:
            if not _HAS_PYZOGY or run_subtraction is None:
                raise RuntimeError(
                    "ZOGY subtraction requested but PyZOGY is not installed. "
                    "Install the optional dependency that provides PyZOGY, or switch "
                    "template_subtraction.method to 'sfft' or 'hotpants'."
                )
            if not science_psf or not template_psf:
                raise ValueError("PSF models required for ZOGY are missing")
            science_data = np.asarray(fits.getdata(scienceFpath), dtype=float)
            reference_data = np.asarray(fits.getdata(templateFpath), dtype=float)
            science_psf_data = np.asarray(fits.getdata(science_psf), dtype=float)
            reference_psf_data = np.asarray(fits.getdata(template_psf), dtype=float)
            if science_data.shape != reference_data.shape:
                raise ValueError(
                    f"ZOGY requires same image shapes: science {science_data.shape} vs reference {reference_data.shape}"
                )
            diff = run_subtraction(
                science_image=science_data,
                reference_image=reference_data,
                science_psf=science_psf_data,
                reference_psf=reference_psf_data,
                science_saturation=float(science_saturate),
                reference_saturation=float(template_saturate),
                max_iterations=10,
                use_pixels=False,
                size_cut=True,
                show=False,
                normalization="science",
            )
            diff_image = diff[0] if isinstance(diff, (tuple, list)) else diff
            write_fits(
                str(differenceFpath), np.asarray(diff_image, dtype=float), scienceHeader
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
        templateFpath,
        differenceFpath,
        mask_loc,
        scienceDir,
        base_name,
        masked_sources,
        masked_centers,
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
        def _as_bool(value, default: bool = False) -> bool:
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
                if v in ("0", "false", "f", "no", "n", "off", ""):
                    return False
            return default

        # Use finite saturation for SFFT (FITS/MeLOn cannot use inf)
        _saturate_fallback = 1e30
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
            excluded = masked_sources + masked_centers
            current_excluded = list(excluded)
            current_matching_sources = list(matching_sources)

            ts_sub = self.input_yaml["template_subtraction"]
            phot_cfg = self.input_yaml.get("photometry", {})

            # Enforce REF convolution for SFFT so DIFF = SCI - conv(REF).
            forceconv = "REF"

            # Background polynomial order: default to 0 unless the user explicitly overrides
            bg_order = ts_sub.get("sfft_bg_order", 0)
            allow_bg_override = _as_bool(
                ts_sub.get("sfft_allow_crowded_bg_order_override", False), False
            )
            # Enforce a global flux scale in SFFT (constant kernel-sum photometric ratio).
            # This intentionally disables spatially varying photometric scaling.
            const_phot_ratio = True

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
                        "photometry.crowded_field=True; forcing SFFT crowded (ECP) mode "
                        "for subtraction."
                    )
                sfft_crowded = True
            elif force_sparse:
                logger.info(
                    "force_sparse_sfft=True in template_subtraction; using SFFT sparse "
                    "(ESP) even though crowded_field=%s.",
                    phot_cfg.get("crowded_field", False),
                )

            if sfft_crowded:
                sfft_method = "crowded"
            else:
                sfft_method = "sparse"

            logger.info("Starting SFFT subtraction via %s method...", sfft_method)
            logger.info(
                "SFFT photometric scaling: ConstPhotRatio=%s",
                const_phot_ratio,
            )
            def _serialize_xy_pairs(xy_list) -> str:
                if not xy_list:
                    return "[]"
                coords = ",".join(f"[{float(x):.3f},{float(y):.3f}]" for x, y in xy_list)
                return f"[{coords}]"

            def _build_sfft_cmd(run_excluded, run_matching):
                # If fewer than 5 pipeline-matched sources, let SFFT perform matching.
                min_sources_for_prior = 5
                if len(run_matching) < min_sources_for_prior:
                    logger.info(
                        "Fewer than %d pipeline-matched sources (%d); letting SFFT perform source matching.",
                        min_sources_for_prior,
                        len(run_matching),
                    )
                    match_str = "[]"
                else:
                    match_str = _serialize_xy_pairs(run_matching)
                excl_str = _serialize_xy_pairs(run_excluded)

                cmd_local = [
                    sys.executable,
                    str(script),
                    "-sci",
                    str(scienceFpath),
                    "-ref",
                    str(templateFpath),
                    "-diff",
                    str(differenceFpath),
                    "-mask",
                    str(mask_loc),
                    "-masked_sources",
                    excl_str,
                    "-kernel_order",
                    str(kernel_order),
                    "-bg_order",
                    str(bg_order),
                    "-allow_crowded_bg_order_override",
                    "true" if allow_bg_override else "false",
                    "-constphotratio",
                    "true" if const_phot_ratio else "false",
                    "-matching_sources",
                    match_str,
                    "-kernel_half_width",
                    str(scale),
                    "-forceconv",
                    forceconv,
                    "-gain_sci",
                    str(float(science_gain)),
                    "-gain_ref",
                    str(float(template_gain)),
                    "-saturate_sci",
                    str(sat_sci),
                    "-saturate_ref",
                    str(sat_ref),
                ]

                # Optional: finer background mesh for SExtractor/SFFT.
                back_size = ts_sub.get("sfft_back_size", None)
                back_filt = ts_sub.get("sfft_back_filtersize", None)
                if back_size is not None:
                    cmd_local += ["-back_size", str(int(back_size))]
                if back_filt is not None:
                    cmd_local += ["-back_filtersize", str(int(back_filt))]

                # Robust SFFT source rejection controls.
                only_flags_cfg = ts_sub.get("sfft_only_flags", [0, 1, 2])
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
            log_path = scienceDir / f"sfft_{Path(base_name).stem}.txt"
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

            cmd = _build_sfft_cmd(current_excluded, current_matching_sources)
            with open(log_path, "w") as lf:
                subprocess.run(
                    cmd, check=True, text=True, stdout=lf, stderr=lf, env=sfft_env
                )

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
                        post_anom_xy = [tuple(v) for v in xy.to_numpy(float)]
                    else:
                        post_anom_xy = []
                except Exception:
                    post_anom_xy = []

                n_post = len(post_anom_xy)
                n_ref = max(1, len(current_matching_sources))
                frac_post = float(n_post) / float(n_ref)
                if n_post >= post_anom_min_count and frac_post <= post_anom_max_frac:
                    # Extend prior-ban list.
                    current_excluded = current_excluded + post_anom_xy

                    # Remove prior-selected matches too close to anomaly sources.
                    if current_matching_sources:
                        anom_arr = np.asarray(post_anom_xy, float)
                        filtered_matching = []
                        for x0, y0 in current_matching_sources:
                            dist2 = (anom_arr[:, 0] - float(x0)) ** 2 + (
                                anom_arr[:, 1] - float(y0)
                            ) ** 2
                            if np.any(dist2 <= post_anom_match_radius_px**2):
                                continue
                            filtered_matching.append((x0, y0))
                        dropped_matching = len(current_matching_sources) - len(
                            filtered_matching
                        )
                        current_matching_sources = filtered_matching
                    else:
                        dropped_matching = 0

                    logger.info(
                        "SFFT post-anomaly feedback: banning %d sources and removing %d prior matches; rerunning subtraction once.",
                        n_post,
                        dropped_matching,
                    )
                    cmd_retry = _build_sfft_cmd(
                        current_excluded,
                        current_matching_sources,
                    )
                    retry_log_path = scienceDir / f"sfft_{Path(base_name).stem}_postanom_retry.txt"
                    with open(retry_log_path, "w") as lf:
                        subprocess.run(
                            cmd_retry,
                            check=True,
                            text=True,
                            stdout=lf,
                            stderr=lf,
                            env=sfft_env,
                        )
                elif n_post > 0:
                    logger.info(
                        "SFFT post-anomaly feedback skipped (count=%d, fraction=%.2f, limits: min=%d, max_frac=%.2f).",
                        n_post,
                        frac_post,
                        post_anom_min_count,
                        post_anom_max_frac,
                    )

            logger.info("SFFT subtraction succeeded")
            return "done"
        except Exception as exc:
            log_warning_from_exception(
                logger, "SFFT failed, falling back to HOTPANTS", exc
            )
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
                        "HOTPANTS executable '%s' was not found. "
                        "Set template_subtraction.hotpants_exe_loc to a valid path or ensure 'hotpants' is on PATH.",
                        exe,
                    )
                    return False
            else:
                which = shutil.which(exe)
                if which is None:
                    logger.warning(
                        "HOTPANTS executable '%s' not found on PATH. "
                        "Install HOTPANTS and/or set template_subtraction.hotpants_exe_loc to its full path.",
                        exe,
                    )
                    return False
                resolved_exe = which

            # Sanitise inputs
            clean_fits_nans(scienceFpath)
            clean_fits_nans(templateFpath)

            hotpants_fwhm = ensure_odd(
                int(max(np.ceil(template_fwhm), np.ceil(science_fwhm)))
            )
            r = ensure_odd(max(int(1.5 * hotpants_fwhm), 5))
            rss = ensure_odd(max(3 * r, 11))

            # Read noise: HOTPANTS can misbehave with 0; use a small floor (e.g. 0.1 e-).
            rn_floor = 0.1
            science_readnoise = max(float(science_readnoise), rn_floor)
            template_readnoise = max(float(template_readnoise), rn_floor)

            # Upper limits (-iu, -tu): use a very high value so HOTPANTS does not
            # treat bright valid pixels as bad and cut out parts of the image.
            # Only the lower limits (-il, -tl) are used to exclude negative/low noise.
            ul = 1e30
            mask_abs = os.path.abspath(mask_loc)
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
                mask_abs,
                "-tmi",
                mask_abs,
                "-n",
                "i",
                "-c",
                "t",
                "-v",
                "2",
                "-r",
                str(r),
                "-rss",
                str(rss),
                "-ko",
                str(kernel_order),
                "-bgo",
                "0",
            ]
            if stamp_loc:
                args += ["-ssf", stamp_loc]
                args += ["-savexy", str(scienceDir / "used_stamps.region")]
            if scienceNoise:
                args += ["-ini", str(scienceNoise)]

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

            logger.info("HOTPANTS subtraction succeeded")
            return True

        except Exception as exc:
            log_warning_from_exception(
                logger, "HOTPANTS subtraction failed", exc
            )
            return False
