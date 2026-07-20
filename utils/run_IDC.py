"""
Drop-in module: ImageDistortionCorrector
Purpose:
  - Build robust SExtractor catalogs (optional PSFEx)
  - Cross-match science/reference sources into 1<->1 tables, using extended sources as alignment anchors
  - Align via SCAMP+SWarp, or Reproject (WCS), then AstroAlign on failure
  - Refine WCS with SCAMP (.head handling fixed)
  - Optimize convolution kernel if FWHM is available in the header
Key guarantees:
  - After filter_matched_sources(): same length catalogs, row i in science <-> row i in reference, shared MATCH_ID
External tools required on PATH:
  - sex, psfex (optional), scamp, swarp
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List, Optional, Tuple, Union

import astroalign as aa
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table

# Check if reproject is available for fallback alignment
try:
    from reproject import reproject_interp, reproject_adaptive
    HAS_REPROJECT = True
    # Introspect reproject_adaptive for optional kwargs
    import inspect as _inspect
    _ADAPTIVE_PARAMS = set(_inspect.signature(reproject_adaptive).parameters.keys())
except ImportError:
    HAS_REPROJECT = False
    _ADAPTIVE_PARAMS = set()
from astropy.visualization import ZScaleInterval
import astropy.wcs as WCS
from matplotlib.patches import Circle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import minimize
import astropy.units as u

sys.path.append(str(Path(__file__).parent.parent))
from functions import remove_wcs_from_header, log_warning_from_exception
from wcs import get_wcs, _normalize_projection_codes
from utils.run_sex import SExtractorWrapper


def _normalize_head_file(head_path: Union[str, Path]) -> bool:
    """Normalize SCAMP .head WCS so SWarp recognizes SIP/TPV distortion.

    SCAMP writes PV/SIP distortion keywords but often leaves CTYPE as
    plain RA---TAN/DEC--TAN.  SWarp (and astropy) may then treat the WCS
    as linear and ignore the distortion during resampling, causing
    subpixel misalignments across the field.  This helper rewrites the
    .head with consistent TPV or TAN-SIP CTYPE codes.

    Parameters
    ----------
    head_path : str or Path
        Path to the .head file to normalize in place.

    Returns
    -------
    bool
        True if the file was normalized successfully, False otherwise.
    """
    try:
        _hp = Path(head_path)
        if not _hp.is_file():
            return False
        _hdr = fits.Header.fromtextfile(str(_hp))
        _hdr_norm = _normalize_projection_codes(_hdr, inplace=False)
        if _hdr_norm.tostring() != _hdr.tostring():
            _hdr_norm.totextfile(str(_hp), overwrite=True)
        return True
    except Exception:
        return False


def _safe_world2pix(wcs_obj, ra, dec, origin=0):
    """
    ``wcs.all_world2pix`` with ``quiet=True`` so non-converging points
    return NaN instead of raising ``astropy.wcs.NoConvergence``.

    The ``quiet`` kwarg was added in astropy 4.0.  On older versions we fall
    back to the plain call wrapped in a try/except so callers never see the
    exception and can test ``np.isfinite`` instead.
    """
    try:
        return wcs_obj.all_world2pix(ra, dec, origin, quiet=True)
    except TypeError:
        # astropy < 4.0: quiet kwarg not supported
        try:
            return wcs_obj.all_world2pix(ra, dec, origin)
        except Exception:
            _nan = np.full_like(np.atleast_1d(ra), np.nan, dtype=float)
            return _nan, _nan.copy()
    except Exception:
        _nan = np.full_like(np.atleast_1d(ra), np.nan, dtype=float)
        return _nan, _nan.copy()


class ImageDistortionCorrector:
    """
    Process astronomical images with SExtractor, SCAMP, SWarp, and AstroAlign,
    using extended sources as alignment anchors and optimizing the convolution kernel if FWHM is available.
    """

    # --------------------------- Default tool configs ---------------------------
    DEFAULT_SEX_CONFIG = {
        "CATALOG_TYPE": "FITS_LDAC",
        "DETECT_THRESH": 1.5,
        "ANALYSIS_THRESH": 1.2,
        "DETECT_MINAREA": 3,
        "BACK_SIZE": 32,
        "DEBLEND_NTHRESH": 64,
        "BACK_TYPE": "MANUAL",
        "DEBLEND_MINCONT": 0.001,
        "BACK_FILTERSIZE": 5,
        "FILTER": "Y",
        "CLEAN": "Y",
        "CLEAN_PARAM": 1,
        "PHOT_APERTURES": 10,
        "VERBOSE_TYPE": "QUIET",
    }

    # Alignment-specific SExtractor config overrides: more sensitive detection for sparse fields
    # These are applied on top of DEFAULT_SEX_CONFIG
    ALIGNMENT_SEX_CONFIG = {
        "DETECT_THRESH": 0.8,  # Lower threshold for fainter sources in sparse fields
        "ANALYSIS_THRESH": 0.5,  # Lower analysis threshold
        "DETECT_MINAREA": 1,  # Smaller minimum area for compact sources
        "BACK_SIZE": 32,  # Smaller background mesh for better local estimate in sparse fields
        "DEBLEND_NTHRESH": 16,  # Fewer deblending thresholds to avoid splitting faint sources
        "BACK_FILTERSIZE": 3,
        "MEMORY_PIXSTACK": 300000,  # Increase pixel stack to avoid overflow warnings
        "CLEAN": "N",  # Disable cleaning to avoid removing faint sources
        "FILTER": "Y",  # Keep convolution filter enabled
    }

    # Maximum FWHM (pixels) for sources used in alignment; sources with FWHM > this are excluded
    ALIGNMENT_MAX_FWHM_PIX = 100.0

    # Crowded-field overrides: maximize source detection (tighter deblending, smaller back mesh, lower thresholds)
    CROWDED_SEX_CONFIG = {
        "DEBLEND_MINCONT": 0.0001,
        "DEBLEND_NTHRESH": 128,
        "BACK_SIZE": 16,
        "BACK_FILTERSIZE": 3,
        "DETECT_MINAREA": 2,
        "DETECT_THRESH": 1.2,
        "ANALYSIS_THRESH": 1.0,
    }

    DEFAULT_SCAMP_CONFIG = {
        "SOLVE_ASTROM": "Y",
        "SOLVE_PHOTOM": "N",  # Alignment-only; photometric solution unused and wastes time
        "REF_TIMEOUT": 60,
        "REF_SERVER": "vizier.cfa.harvard.edu",
        "DISTORT_DEGREES": None,  # Will be set from config
        "MATCH": "Y",
        "MATCH_RESOL": 0,
        "MATCH_FLIPPED": "Y",
        "WRITE_XML": "Y",
        "VERBOSE_TYPE": "LOG",
        # Input LDAC centroid columns (windowed, most accurate)
        "CENTROID_KEYS": "XWIN_IMAGE,YWIN_IMAGE",
        "CENTROIDERR_KEYS": "ERRAWIN_IMAGE,ERRBWIN_IMAGE,ERRTHETAWIN_IMAGE",
        "DISTORT_KEYS": "XWIN_IMAGE,YWIN_IMAGE",
        # Reference catalog column names (science LDAC used as ASTREFCAT_NAME).
        # The science LDAC contains XWIN_WORLD/YWIN_WORLD as its windowed world
        # coordinates — these are what SCAMP expects for ASTREFCENT_KEYS when the
        # reference catalog is an LDAC (not an online catalog like GAIA).
        "ASTREF_WEIGHT": 1,
        "ASTREFMAG_KEY": "MAG_AUTO",
        "ASTREFMAGERR_KEY": "MAGERR_AUTO",
        "ASTREFCENT_KEYS": "XWIN_WORLD,YWIN_WORLD",
        "ASTREFERR_KEYS": "ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD",
        "ELLIPTICITY_MAX": 0.5,
        "MOSAIC_TYPE": "UNCHANGED",
        "STABILITY_TYPE": "EXPOSURE",
        "SN_THRESHOLDS": "3.0,100000.0",
    }

    DEFAULT_SWARP_CONFIG = {
        "COMBINE": "N",
        "COMBINE_TYPE": "MEDIAN",
        "INTERPOLATE": "N",
        "OVERSAMPLING": 0,
        "SUBTRACT_BACK": "N",
        "RESAMPLE": "Y",
        "RESAMPLE_DIR": ".",
        "RESAMPLE_SUFFIX": ".resamp.fits",
        "WRITE_XML": "Y",
        "VERBOSE_TYPE": "LOG",
        "BLANK_BADPIXELS": "Y",
        "FILL_VALUE": "NAN",
        "CELESTIAL_TYPE": "NATIVE",
        "PROJECTION_TYPE": "TAN",
        "FSCALASTRO_TYPE": "NONE",
        "COPY_KEYWORDS": "TELESCOP,FILTER,INSTRUME,EXPTIME,GAIN,OBSMJD,RDNOISE,APER,FWHM",
    }

    # ---------------------------- Constructor / logger ----------------------------
    def __init__(
        self, input_yaml, verbose_level: int = 1, delete_originals: bool = True
    ):
        """
        Initialize the ImageDistortionCorrector.

        Args:
            input_yaml: Configuration YAML for the pipeline.
            verbose_level: Logging verbosity (0: quiet, 1: info, 2: verbose).
            delete_originals: Remove original FITS after creating *_align products.
        """
        self.verbose_level = verbose_level
        self.delete_originals = delete_originals
        self.logger = logging.getLogger(__name__)
        self.default_threads = 4
        self._executables: Dict[str, str] = {}
        self._temp_dirs: set[str] = set()
        self.cleanup_intermediate = True
        self.input_yaml = input_yaml
        # Allow user to keep SCAMP logs for debugging
        iy = getattr(self, "input_yaml", None) or {}
        ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
        self.preserve_scamp_logs = bool(ts.get("preserve_scamp_logs", False))
        # Initialize SExtractorWrapper for alignment (same as main pipeline)
        self.sextractor = SExtractorWrapper(input_yaml)

    # ---------------------------- GAIA cache helpers ----------------------------

    def _get_gaia_cache_dir(self) -> Path:
        """Return the persistent GAIA cache directory path."""
        iy = getattr(self, "input_yaml", None) or {}
        ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
        cache_dir = ts.get("gaia_cache_dir")
        if cache_dir:
            path = Path(cache_dir).expanduser()
        else:
            # Default: store cache alongside reduced data so it travels with the dataset.
            fits_dir = iy.get("fits_dir") if isinstance(iy, dict) else None
            if fits_dir:
                fits_path = Path(fits_dir)
                reduced_dir = fits_path.parent / (fits_path.name + "_REDUCED")
                path = reduced_dir / "scamp_gaia_cache"
            else:
                # Try to infer from current working directory if it ends with _REDUCED
                cwd = Path.cwd()
                if cwd.name.endswith("_REDUCED"):
                    path = cwd / "scamp_gaia_cache"
                else:
                    # Try fpath (output directory) if available
                    fpath = iy.get("fpath") if isinstance(iy, dict) else None
                    if fpath:
                        fpath_dir = Path(fpath).parent
                        if fpath_dir.name.endswith("_REDUCED"):
                            path = fpath_dir / "scamp_gaia_cache"
                        else:
                            path = Path.home() / ".autophot" / "scamp_gaia_cache"
                    else:
                        path = Path.home() / ".autophot" / "scamp_gaia_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _gaia_cache_ttl_seconds(self) -> float:
        """Return cache TTL in seconds (default 30 days)."""
        iy = getattr(self, "input_yaml", None) or {}
        ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
        days = ts.get("gaia_cache_ttl_days", 30)
        return float(days) * 86400.0

    def _compute_gaia_cache_key(self, catalog_path: str) -> str:
        """Compute a cache key from the target coordinates in input_yaml.

        All images of the same field share the same target RA/Dec, so they
        all get the same cache key.  Falls back to the catalog's median world
        coordinates only when target_ra/target_dec are not available.
        """
        cat_path = Path(catalog_path)
        try:
            # Prefer target coordinates from input_yaml — constant per field
            iy = getattr(self, "input_yaml", None) or {}
            target_ra = iy.get("target_ra")
            target_dec = iy.get("target_dec")

            if target_ra is not None and target_dec is not None:
                ra_r = round(float(target_ra), 2)
                dec_r = round(float(target_dec), 2)
                key_str = f"{ra_r:.2f}_{dec_r:.2f}_30.0"
                return hashlib.md5(key_str.encode()).hexdigest()

            # Fallback: median world coordinates from the catalog itself
            if not cat_path.exists():
                return hashlib.md5(cat_path.name.encode()).hexdigest()
            with fits.open(str(cat_path)) as hdul:
                data_hdu = None
                for hdu in hdul:
                    if hasattr(hdu, "columns"):
                        cols = [c.name for c in hdu.columns]
                        if "XWIN_WORLD" in cols and "YWIN_WORLD" in cols:
                            data_hdu = hdu
                            break
                        if "ALPHA_J2000" in cols and "DELTA_J2000" in cols:
                            data_hdu = hdu
                            break
                if data_hdu is None or len(data_hdu.data) == 0:
                    return hashlib.md5(cat_path.name.encode()).hexdigest()

                cols = [c.name for c in data_hdu.columns]
                ra = data_hdu.data["XWIN_WORLD" if "XWIN_WORLD" in cols else "ALPHA_J2000"]
                dec = data_hdu.data["YWIN_WORLD" if "YWIN_WORLD" in cols else "DELTA_J2000"]
                ra = np.asarray(ra, dtype=float)
                dec = np.asarray(dec, dtype=float)
                valid = np.isfinite(ra) & np.isfinite(dec)
                if not valid.any():
                    return hashlib.md5(cat_path.name.encode()).hexdigest()

                center_ra = float(np.median(ra[valid]))
                center_dec = float(np.median(dec[valid]))
                ra_r = round(center_ra, 2)
                dec_r = round(center_dec, 2)
                key_str = f"{ra_r:.2f}_{dec_r:.2f}_30.0"
                return hashlib.md5(key_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(cat_path.name.encode()).hexdigest()

    def _find_cached_gaia_catalog(self, cache_key: str) -> Optional[Path]:
        """Return path to cached GAIA catalog if it exists and is fresh."""
        cache_dir = self._get_gaia_cache_dir()
        cat_path = cache_dir / f"{cache_key}_gaia_dr3.cat"
        meta_path = cache_dir / f"{cache_key}_gaia_dr3.json"
        if not cat_path.exists():
            return None
        if not meta_path.exists():
            # No metadata: treat as stale to be safe
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            created = meta.get("created", 0)
            ttl = self._gaia_cache_ttl_seconds()
            if time.time() - created > ttl:
                self.logger.debug("GAIA cache entry expired for key %s", cache_key)
                return None
            self.logger.debug("GAIA cache hit: %s", cat_path)
            return cat_path
        except Exception:
            return None

    def _save_gaia_catalog_to_cache(
        self, temp_cat_path: str, cache_key: str
    ) -> Optional[Path]:
        """Move a SCAMP-downloaded GAIA catalog into the persistent cache.

        SCAMP's SAVE_REFCATALOG writes columns like X_WORLD/Y_WORLD/MAG,
        but our DEFAULT_SCAMP_CONFIG expects XWIN_WORLD/YWIN_WORLD/MAG_AUTO
        when ASTREF_CATALOG=FILE.  We rename the columns on the way into the
        cache so that subsequent cache hits work transparently.
        """
        temp_path = Path(temp_cat_path)
        if not temp_path.exists():
            return None
        cache_dir = self._get_gaia_cache_dir()
        cat_path = cache_dir / f"{cache_key}_gaia_dr3.cat"
        meta_path = cache_dir / f"{cache_key}_gaia_dr3.json"
        try:
            with fits.open(str(temp_path)) as hdul:
                new_hdul = fits.HDUList()
                for hdu in hdul:
                    if hasattr(hdu, "columns"):
                        # Rebuild table so we can safely add columns
                        tbl = Table(hdu.data)
                        rename_map = {
                            "X_WORLD": "XWIN_WORLD",
                            "Y_WORLD": "YWIN_WORLD",
                            "MAG": "MAG_AUTO",
                            "MAGERR": "MAGERR_AUTO",
                        }
                        changed = False
                        for old_name, new_name in rename_map.items():
                            if old_name in tbl.colnames and new_name not in tbl.colnames:
                                tbl.rename_column(old_name, new_name)
                                changed = True
                        # Add dummy ERRTHETA_WORLD if missing (ASTREFERR_KEYS expects it)
                        if (
                            "ERRA_WORLD" in tbl.colnames
                            and "ERRB_WORLD" in tbl.colnames
                            and "ERRTHETA_WORLD" not in tbl.colnames
                        ):
                            tbl["ERRTHETA_WORLD"] = np.zeros(len(tbl), dtype=np.float32)
                            changed = True
                        if changed:
                            tbl.meta["COMMENT"] = "Renamed GAIA columns for SCAMP FILE mode"
                        new_hdu = fits.table_to_hdu(tbl)
                        new_hdu.header["EXTNAME"] = hdu.name
                        new_hdul.append(new_hdu)
                    else:
                        new_hdul.append(hdu.copy())
                new_hdul.writeto(str(cat_path), overwrite=True)

            meta = {
                "created": time.time(),
                "catalog": "GAIA-DR3",
                "cache_key": cache_key,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            self.logger.info("Cached GAIA catalog to %s", cat_path)
            return cat_path
        except Exception as e:
            self.logger.warning("Failed to save GAIA catalog to cache: %s", e)
            return None

    # -------------------------------- Utilities --------------------------------
    @staticmethod
    def _header_indicates_resampled(header) -> bool:
        """Return True if the FITS header suggests the image was already resampled (e.g. by SWarp)."""
        for key in ("RESAMPLED", "SWARPED", "RESAMPLE"):
            if key not in header:
                continue
            header_value = header[key]
            if isinstance(header_value, bool) and header_value:
                return True
            if isinstance(header_value, str) and header_value.upper() in (
                "T",
                "TRUE",
                "Y",
                "YES",
                "1",
            ):
                return True
            if isinstance(header_value, (int, float)) and header_value:
                return True
        return False

    @staticmethod
    def _header_indicates_scamp(
        header, pattern: str = "Astrometric solution by SCAMP"
    ) -> bool:
        """Return True if the FITS header has HISTORY or COMMENT indicating a SCAMP astrometric solution."""
        pattern_lower = pattern.lower()
        for key in ("HISTORY", "COMMENT"):
            if key not in header:
                continue
            # Header can have multiple cards with the same keyword
            for card in header.cards:
                if card.keyword != key:
                    continue
                history_text = str(card.value).strip() if card.value is not None else ""
                if pattern_lower in history_text.lower():
                    return True
        return False

    @staticmethod
    def _ascii_safe(line: str) -> str:
        """Replace non-ASCII chars in a line with '?' safely (fixes ord() bug)."""
        return "".join((ch if ord(ch) < 128 else "?") for ch in line)

    def _create_conv_file(
        self,
        path: str,
        fwhm_pixels: float = 3.0,
        aperture_radius: Optional[float] = None,
        scale_half_width: Optional[int] = None,
    ) -> None:
        """
        Create a convolution kernel for SExtractor.

        Kernel half-width priority (highest to lowest):
          1. ``scale_half_width`` — the pipeline ``scale`` parameter (already half the
             cutout box size). Ensures the kernel covers the same footprint used for
             PSF/cutout extraction.
          2. ``aperture_radius`` — photometry aperture radius, so the matched-filter
             kernel exactly covers the detection aperture.
          3. ``1.7 × fwhm_pixels`` — default aperture formula from run_sex.py.

        ``fwhm_pixels`` always controls the Gaussian sigma regardless of which
        half-width source is used.

        Args:
            path: Path to save the convolution file.
            fwhm_pixels: FWHM in pixels (sets the Gaussian sigma).
            aperture_radius: Photometry aperture radius in pixels.
            scale_half_width: The pipeline ``scale`` parameter (half the cutout box) in pixels.
                Takes priority over aperture_radius when provided.
        """

        # Enforce realistic FWHM bounds: 2.5-15 pixels for convolution kernel
        fwhm_pixels = float(max(2.5, min(fwhm_pixels, 15.0)))
        # Kernel half-width priority: scale_half_width > aperture_radius > 1.7×FWHM.
        # 2*half_width+1 is always odd, so no even-size correction is needed.
        # SExtractor hard limit: 31×31 pixels.
        # Clamp half_width to reasonable range (3-15) to avoid oversized kernels.
        MAX_KERNEL_HALF_WIDTH = 15  # Gives max kernel_size = 31
        if scale_half_width is not None and int(scale_half_width) > 0:
            half_width = min(int(scale_half_width), MAX_KERNEL_HALF_WIDTH)
            self.logger.info(
                "Convolution kernel half-width set from scale: %d px (FWHM=%.1f px)",
                half_width,
                fwhm_pixels,
            )
        elif aperture_radius is not None and float(aperture_radius) > 0:
            half_width = min(int(np.ceil(float(aperture_radius))), MAX_KERNEL_HALF_WIDTH)
            self.logger.info(
                "Convolution kernel half-width set from aperture radius: %d px (FWHM=%.1f px)",
                half_width,
                fwhm_pixels,
            )
        else:
            half_width = int(np.ceil(1.7 * fwhm_pixels))
        kernel_size = max(3, 2 * half_width + 1)
        kernel_size = min(kernel_size, 31)  # SExtractor hard limit
        self.logger.info(
            "Creating convolution kernel: FWHM=%.1f px  size=%dx%d",
            fwhm_pixels,
            kernel_size,
            kernel_size,
        )
        center = kernel_size // 2
        sigma = fwhm_pixels / 2.355  # FWHM = 2.355 * sigma
        conv_text = f"CONV NORM\n# {kernel_size}x{kernel_size} convolution mask with FWHM = {fwhm_pixels:.1f} pixels\n"
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                weight_value = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
                conv_text += f"{weight_value:.6f} "
            conv_text += "\n"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(conv_text)

    @staticmethod
    def _create_nnw_file(path: str) -> None:
        """Default SExtractor stellarity network."""
        import re

        nnw_text = r"""
NNW
# Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
# inputs: 9 for profile parameters + 1 for seeing.
# outputs: Stellarity index (0.0 to 1.0)
 3 10 10  1
-1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01
 3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00
-3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00
 4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
-2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00
 4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e-01 -1.01592e-02 -7.54052e+00
 1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
-5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01
 2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01
 3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01
-8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
-3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00
-1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00
 0.00000e+00
 1.00000e+00
"""
        # Remove indentation introduced by quoting
        nindent = len(re.split("NNW", nnw_text.split("\n", 1)[1])[0])
        nnw_text = "\n".join([line[nindent:] for line in nnw_text.split("\n")[1:]])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(nnw_text)

    def _validate_output_dir(
        self, output_dir: Optional[str], prefix: Optional[str] = None
    ) -> str:
        """Create and register a temporary output directory if none provided."""
        if output_dir is None:
            output_dir = mkdtemp(prefix=prefix)
            self._temp_dirs.add(output_dir)
            self.logger.info(f"Created temporary directory: {output_dir}")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    def plot_sextractor_sources(
        self,
        image_path: str,
        catalog_path: str,
        output_plot_path: str = "sextractor_sources.png",
        cmap: str = "gray",
        figsize: Optional[tuple] = None,
        max_sources: int = 1000,
        **imshow_kwargs,
    ) -> None:
        """
        Plot an astronomical image and overlay circles around all sources detected by SExtractor.
        The circle radius is set to the source's FWHM.

        Args:
            image_path: Path to the FITS image.
            catalog_path: Path to the SExtractor catalog (FITS_LDAC).
            output_plot_path: Path to save the output plot.
            cmap: Colormap for the image.
            figsize: Figure size.
            max_sources: Maximum number of sources to plot.
            **imshow_kwargs: Additional arguments for imshow.
        """
        try:
            if figsize is None:
                # Prefer AutoPHoT figure sizing (golden-ratio based).
                from functions import set_size

                golden_ratio = (5**0.5 + 1) / 2
                figsize = set_size(340, aspect=golden_ratio)  # square-ish

            # Load image
            with fits.open(image_path) as hdul:
                data = hdul[0].data.astype(np.float32)

            # Load catalog
            with fits.open(catalog_path) as hdul:
                catalog = Table(hdul[2].data)

            # ZScale for contrast
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(data)

            # Plot image
            fig, ax = plt.subplots(figsize=figsize)
            # Render NaNs as white "no data" regions.
            cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
            try:
                cmap = cmap.copy()
            except Exception:
                pass
            try:
                cmap.set_bad(color="white")
            except Exception:
                pass
            im = ax.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", **imshow_kwargs
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Plot circles for each source
            for i, row in enumerate(catalog[:max_sources]):
                if (
                    "XWIN_IMAGE" in row.colnames
                    and "YWIN_IMAGE" in row.colnames
                    and "FWHM_IMAGE" in row.colnames
                ):
                    x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                    fwhm = row["FWHM_IMAGE"]
                    # Convert to 0-based if needed
                    x_0based, y_0based = x - 1, y - 1
                    circle = Circle(
                        (x_0based, y_0based),
                        fwhm,
                        facecolor="none",
                        edgecolor="#FF0000",
                        linewidth=0.5,
                        alpha=0.7,
                    )
                    ax.add_patch(circle)
                    ax.text(
                        x_0based,
                        y_0based - fwhm - 2,
                        str(i),
                        color="#FF0000",
                        fontsize=8,
                        ha="center",
                        va="top",
                    )

            plt.tight_layout()
            plt.savefig(output_plot_path, dpi=150)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting sources: {e}")
            raise

    def _check_executable(self, name: str) -> str:
        """Resolve tool path or raise. Cached for speed."""
        if name in self._executables:
            return self._executables[name]
        for cmd in (name, f"{name}.exe"):
            try:
                subprocess.run(
                    [cmd, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                    timeout=10,
                )
                self._executables[name] = cmd
                return cmd
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        raise FileNotFoundError(f"Executable not found: {name}")

    def _is_executable_available(self, name: str) -> bool:
        """Check if an executable is installed without raising. Cached for speed."""
        if name in self._executables:
            return True
        for cmd in (name, f"{name}.exe"):
            try:
                subprocess.run(
                    [cmd, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                    timeout=10,
                )
                self._executables[name] = cmd
                return True
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        return False

    @staticmethod
    def angular_distance(ra1, dec1, ra2, dec2):
        """Small-angle distance in degrees; uses cos(dec2) projection for RA."""
        dra = (ra1 - ra2) * np.cos(np.radians(dec2))
        ddec = dec1 - dec2
        return np.sqrt(dra**2 + ddec**2)

    def update_fits_catalog(
        self, catalog_path, target_ra, target_dec, sigma_deg=0.01, tbhdu=2
    ):
        """
        Inflate errors far from target to bias SCAMP matching near a position.
        Uses SExtractor-style column names: ALPHA_J2000, DELTA_J2000, ERRA_WORLD, ERRB_WORLD.
        """
        with fits.open(catalog_path, mode="update") as hdul:
            catalog = Table(hdul[tbhdu].data)
            required_cols = ["ALPHA_J2000", "DELTA_J2000", "ERRA_WORLD", "ERRB_WORLD"]
            for col in required_cols:
                if col not in catalog.colnames:
                    raise ValueError(f"Column '{col}' not found in catalog")
            ra = catalog["ALPHA_J2000"]
            dec = catalog["DELTA_J2000"]
            err_ra = catalog["ERRA_WORLD"]
            err_dec = catalog["ERRB_WORLD"]
            dist = self.angular_distance(ra, dec, target_ra, target_dec)
            weights = np.exp(-0.5 * (dist / sigma_deg) ** 2)
            min_err = 1e-4
            catalog["ERRA_WORLD"] = np.clip(err_ra / weights, min_err, None)
            catalog["ERRB_WORLD"] = np.clip(err_dec / weights, min_err, None)
            hdul[tbhdu].data = catalog.as_array()

    def clean_log(self, input_file, output_file=None):
        """Strip ANSI escape codes and blank lines from a log file."""
        import re

        with open(input_file, "r", errors="ignore") as f:
            content = f.read()
        ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean = ansi.sub("", content)
        clean = "\n".join([line for line in clean.split("\n") if line.strip()])
        output_file = input_file if output_file is None else output_file
        with open(output_file, "w") as f:
            f.write(clean)
        return output_file

    def determine_saturation_level(self, fits_path: str) -> float:
        """Heuristic saturation estimate for SExtractor SATUR_LEVEL."""
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            # Store original dtype for saturation check before conversion
            original_dtype = data.dtype
            # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
            if data.dtype.kind != 'f':
                data = data.astype(np.float32)
            if np.issubdtype(original_dtype, np.integer):
                max_possible = np.iinfo(original_dtype).max
                if np.nanmax(data) >= 0.99 * max_possible:
                    satur = float(max_possible)
            else:
                finite = data[np.isfinite(data)]
                if finite.size == 0:
                    satur = float(1e6)
                else:
                    hist, bins = np.histogram(finite, bins=1000)
                    peak_idx = int(np.argmax(hist))
                    pct = float(np.nanpercentile(finite, 99.99))
                    satur = float(max(bins[min(peak_idx, len(bins) - 2)], pct))
            # Cap at reasonable maximum (65535 = max for 16-bit unsigned)
            # SExtractor rejects SATUR_LEVEL values that are too large
            max_saturation = 65535
            if satur > max_saturation:
                self.logger.warning(
                    "Heuristic saturation %s exceeds SExtractor maximum; capping at %d",
                    satur, max_saturation
                )
                satur = max_saturation
            return satur

    # ----------------------------- SExtractor + PSFEx -----------------------------
    @staticmethod
    def _guess_map_weight_path(fits_image: str) -> Optional[str]:
        """
        Best-effort MAP_WEIGHT companion for SExtractor alignment.

        The photometry pipeline commonly writes ``<stem>.weight<ext>`` next to the
        science image. If present, using it in SExtractor improves robustness on
        mosaics with NaNs / no-coverage bands encoded as zeros.
        """
        try:
            p = Path(str(fits_image))
            candidates = [
                p.with_name(p.name + ".weight" + p.suffix),  # e.g. foo.fits -> foo.weight.fits
                p.with_suffix(".weight.fits"),  # legacy
            ]
            for c in candidates:
                if c.is_file():
                    return str(c)
        except Exception:
            return None
        return None

    def run_sextractor(
        self,
        fits_image: str,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
        SEEING_FWHM: float = 1.5,
        PIXEL_SCALE: float = 0,
        use_psfex: bool = False,
        aperture_radius=None,
        weight_path: Optional[str] = None,
        crowded: bool = False,
        scale: Optional[int] = None,
        for_alignment: bool = False,
        fwhm_pixels: Optional[float] = None,
    ) -> Dict:
        """
        Build a clean catalog using SExtractor, prioritizing extended sources.
        Optimizes the convolution kernel if FWHM is available in the header.

        If ``weight_path`` points to an existing FITS weight map, SExtractor is run with
        ``WEIGHT_TYPE MAP_WEIGHT`` so invalid pixels can be down-weighted (important for
        NaN-heavy stacks and SWarp no-coverage bands).
        If crowded is True, uses parameters tuned for crowded fields (tighter deblending,
        smaller background mesh, more deblend levels).
        If for_alignment is True, uses alignment-specific config with more sensitive detection.
        If fwhm_pixels is provided, use it directly instead of reading from header (avoids stale header values).
        """
        try:
            output_dir = self._validate_output_dir(output_dir, prefix="sex_")

            stem = Path(fits_image).stem
            conv_path = str(Path(output_dir) / f"{stem}_default.conv")
            nnw_path = str(Path(output_dir) / f"{stem}_default.nnw")
            self._create_nnw_file(nnw_path)

            # Get FWHM: prefer explicit parameter, then header, then default
            # Header FWHM can be stale from previous runs or instrument defaults
            if fwhm_pixels is None or fwhm_pixels <= 0:
                with fits.open(fits_image) as hdul:
                    header = hdul[0].header
                    fwhm_pixels = header.get("FWHM", 2.0)
            
            pixel_scale_header = None
            with fits.open(fits_image) as hdul:
                header = hdul[0].header
                pixel_scale_header = header.get("PIXSCALE", header.get("CDELT2", 0))
                if pixel_scale_header:
                    pixel_scale_header = abs(float(pixel_scale_header)) * 3600.0
                else:
                    pixel_scale_header = PIXEL_SCALE if PIXEL_SCALE > 0 else 1.0
                seeing_fwhm_arcsec = float(fwhm_pixels) * pixel_scale_header

            # Kernel sizing priority: scale > aperture_radius > 1.7×FWHM.
            # scale is already half the pipeline source-cutout box size, so it
            # directly gives the kernel half-width — the kernel covers the same
            # footprint as PSF/cutout work.
            _eff_aperture: Optional[float] = None
            if aperture_radius is not None and float(aperture_radius) > 0:
                _eff_aperture = float(aperture_radius)
            _scale_hw: Optional[int] = int(scale) if scale is not None and int(scale) > 0 else None
            
            # Always create convolution file for filtering
            self._create_conv_file(
                conv_path,
                fwhm_pixels=fwhm_pixels,
                aperture_radius=_eff_aperture,
                scale_half_width=_scale_hw,
            )

            final_config = self.DEFAULT_SEX_CONFIG.copy()
            if for_alignment:
                final_config.update(self.ALIGNMENT_SEX_CONFIG)
                self.logger.info(
                    "Using SExtractor alignment config overrides (more sensitive detection for sparse fields)"
                )
            elif crowded:
                final_config.update(self.CROWDED_SEX_CONFIG)
                self.logger.info(
                    "Using SExtractor crowded-field config overrides (tighter deblending, smaller back mesh)"
                )
            
            final_config.update(
                {
                    # 'CHECKIMAGE_NAME': 'check_seg.fits,check_aper.fits',
                    "SATUR_LEVEL": self.determine_saturation_level(fits_image),
                    "FILTER_NAME": conv_path,
                    "FILTER": "Y",
                    "STARNNW_NAME": nnw_path,
                    "NTHREADS": self.default_threads,
                    "PIXEL_SCALE": PIXEL_SCALE,
                    "CATALOG_TYPE": "FITS_LDAC",
                    "SEEING_FWHM": round(seeing_fwhm_arcsec, 3),
                }
            )

            # Optional MAP_WEIGHT (0/1 or continuous) for robust detection on masked mosaics.
            wpath = str(weight_path) if weight_path else ""
            if wpath and os.path.isfile(wpath):
                final_config["WEIGHT_TYPE"] = "MAP_WEIGHT"
                final_config["WEIGHT_IMAGE"] = wpath

            if config:
                final_config.update(config)
            if aperture_radius:
                final_config["PHOT_APERTURES"] = aperture_radius

            sex_cmd = self._check_executable("sex")
            param_file = str(Path(output_dir) / f"{stem}_sextractor.param")
            params = [
                "NUMBER",
                "X_IMAGE",
                "Y_IMAGE",
                "FLUX_AUTO",
                "FLUXERR_AUTO",
                "FLUX_APER",
                "FLUXERR_APER",
                "FWHM_IMAGE",
                "ELLIPTICITY",
                "ELONGATION",
                "CLASS_STAR",
                "FLAGS",
                "FLAGS_WEIGHT",
                "XWIN_IMAGE",
                "YWIN_IMAGE",
                "ERRAWIN_IMAGE",
                "ERRBWIN_IMAGE",
                "ERRTHETAWIN_IMAGE",
                "XWIN_WORLD",
                "YWIN_WORLD",
                "X_WORLD",
                "Y_WORLD",
                "ERRA_WORLD",
                "ERRB_WORLD",
                "ERRTHETA_WORLD",
                "MAG_AUTO",
                "MAGERR_AUTO",
                "ALPHA_J2000",
                "DELTA_J2000",
                "FLUX_RADIUS",
                "BACKGROUND",
                "ISOAREA_IMAGE",
                "SNR_WIN",
                "A_IMAGE",
                "B_IMAGE",
                "THETA_IMAGE",
            ]
            Path(param_file).write_text("\n".join(params))
            config_file = str(Path(output_dir) / f"{stem}_default.sex")
            with open(config_file, "w") as f:
                for k, v in final_config.items():
                    f.write(f"{k}\t{v}\n")

            # Use the input filename (without extension) for the catalog
            catalog_name = Path(fits_image).stem + ".cat"
            catalog_path = str(Path(output_dir) / catalog_name)

            # Build SExtractor command
            cmd1 = [
                sex_cmd,
                fits_image,
                "-c",
                config_file,
                "-PARAMETERS_NAME",
                param_file,
                "-CATALOG_NAME",
                catalog_path,
                "-NTHREADS",
                str(final_config["NTHREADS"]),
            ]

            subprocess.run(cmd1, check=True, text=True)

            tbhdu = 2
            with fits.open(catalog_path, mode="update") as hdul:
                catalog = Table(hdul[tbhdu].data)
                if len(catalog) > 0:
                    good = np.ones(len(catalog), dtype=bool)
                    filt = {
                        "SNR_WIN": lambda c: c > 1,
                    }
                    for col, cond in filt.items():
                        if col in catalog.colnames:
                            good &= cond(catalog[col])
                    # Alignment source selection:
                    # Default behavior prefers star-like sources by excluding very large FWHM objects.
                    # For sparse fields, extended sources (galaxies) can be useful alignment anchors.
                    # Opt-in via YAML:
                    #   template_subtraction.alignment_use_extended_sources: True
                    # Optionally override the max-FWHM cutoff:
                    #   template_subtraction.alignment_max_fwhm_pix: <float>
                    iy = getattr(self, "input_yaml", None) or {}
                    ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
                    use_ext = bool(
                        isinstance(ts, dict)
                        and ts.get("alignment_use_extended_sources", False)
                    )
                    max_fwhm = float(
                        ts.get("alignment_max_fwhm_pix", self.ALIGNMENT_MAX_FWHM_PIX)
                    ) if isinstance(ts, dict) else self.ALIGNMENT_MAX_FWHM_PIX
                    if not use_ext:
                        if "FWHM_IMAGE" in catalog.colnames:
                            would_keep = good & (catalog["FWHM_IMAGE"] <= max_fwhm)
                            n_removed = np.sum(
                                good & (catalog["FWHM_IMAGE"] > max_fwhm)
                            )
                            # If excluding would leave 0 sources, keep all (avoid forcing empty catalog)
                            if np.sum(would_keep) > 0:
                                # Sparse-field: if very few sources would remain, keep all
                                # regardless of FWHM. Extended sources (galaxies) are useful
                                # alignment anchors when the field is sparse.
                                if np.sum(would_keep) < 15 and n_removed > 0:
                                    self.logger.info(
                                        "Alignment SExtractor: only %d sources pass FWHM cut; "
                                        "keeping all %d (including %d extended) for sparse-field alignment.",
                                        int(np.sum(would_keep)),
                                        int(np.sum(good)),
                                        int(n_removed),
                                    )
                                else:
                                    good &= catalog["FWHM_IMAGE"] <= max_fwhm
                                    if n_removed > 0:
                                        self.logger.info(
                                            "Alignment SExtractor: excluding %d sources with FWHM > %.0f px",
                                            int(n_removed),
                                            float(max_fwhm),
                                        )
                            elif n_removed > 0:
                                self.logger.info(
                                    "Alignment SExtractor: all %d source(s) have FWHM > %.0f px; keeping them so alignment can proceed",
                                    int(n_removed),
                                    float(max_fwhm),
                                )
                    else:
                        self.logger.info(
                            "Alignment SExtractor: extended sources enabled (keeping large-FWHM objects)."
                        )
                    cleaned = catalog[good]
                    # cleaned = self.filter_well_defined_positions(cleaned)
                    if "SNR_WIN" in cleaned.colnames:
                        cleaned.sort("SNR_WIN", reverse=True)
                    # Add SCAMP-required columns if not present
                    try:
                        from wcs import get_wcs as _get_wcs_for_scamp
                        wcs = _get_wcs_for_scamp(fits.getheader(fits_image))
                        if wcs is None:
                            raise ValueError("get_wcs returned None")
                        if 'XWIN_WORLD' not in cleaned.colnames and 'XWIN_IMAGE' in cleaned.colnames:
                            # SExtractor uses 1-based pixel coordinates (FITS convention)
                            # astropy WCS pixel_to_world expects 0-based numpy convention
                            x_coords = np.asarray(cleaned['XWIN_IMAGE'], float) - 1.0
                            y_coords = np.asarray(cleaned['YWIN_IMAGE'], float) - 1.0
                            world_coords = wcs.pixel_to_world(x_coords, y_coords)
                            cleaned['XWIN_WORLD'] = world_coords.ra.deg
                            cleaned['YWIN_WORLD'] = world_coords.dec.deg
                            cleaned['X_WORLD'] = world_coords.ra.deg
                            cleaned['Y_WORLD'] = world_coords.dec.deg
                            cleaned['ALPHA_J2000'] = world_coords.ra.deg
                            cleaned['DELTA_J2000'] = world_coords.dec.deg
                    except Exception as e:
                        self.logger.warning(f"Could not compute world coordinates: {e}")
                    # Add MAG_AUTO and MAGERR_AUTO if not present
                    if 'MAG_AUTO' not in cleaned.colnames:
                        if 'MAG_APER' in cleaned.colnames:
                            cleaned['MAG_AUTO'] = cleaned['MAG_APER']
                        elif 'FLUX_AUTO' in cleaned.colnames:
                            cleaned['MAG_AUTO'] = -2.5 * np.log10(cleaned['FLUX_AUTO'])
                        else:
                            cleaned['MAG_AUTO'] = 0.0
                    if 'MAGERR_AUTO' not in cleaned.colnames:
                        if 'MAGERR_APER' in cleaned.colnames:
                            cleaned['MAGERR_AUTO'] = cleaned['MAGERR_APER']
                        else:
                            cleaned['MAGERR_AUTO'] = 0.1
                    # Add error columns if not present
                    if 'ERRAWIN_IMAGE' not in cleaned.colnames:
                        if 'FWHM_IMAGE' in cleaned.colnames:
                            cleaned['ERRAWIN_IMAGE'] = cleaned['FWHM_IMAGE'] * 0.1
                        else:
                            cleaned['ERRAWIN_IMAGE'] = 0.1
                    if 'ERRBWIN_IMAGE' not in cleaned.colnames:
                        cleaned['ERRBWIN_IMAGE'] = cleaned['ERRAWIN_IMAGE']
                    if 'ERRX2WIN_IMAGE' not in cleaned.colnames:
                        cleaned['ERRX2WIN_IMAGE'] = cleaned['ERRAWIN_IMAGE'] ** 2
                    if 'ERRY2WIN_IMAGE' not in cleaned.colnames:
                        cleaned['ERRY2WIN_IMAGE'] = cleaned['ERRAWIN_IMAGE'] ** 2
                    if 'ERRTHETAWIN_IMAGE' not in cleaned.colnames:
                        cleaned['ERRTHETAWIN_IMAGE'] = 0.1
                    if 'ERRX2WIN_WORLD' not in cleaned.colnames:
                        cleaned['ERRX2WIN_WORLD'] = cleaned['ERRX2WIN_IMAGE']
                    if 'ERRY2WIN_WORLD' not in cleaned.colnames:
                        cleaned['ERRY2WIN_WORLD'] = cleaned['ERRY2WIN_IMAGE']
                    pixel_scale = 0.1585 / 3600.0
                    if 'ERRA_WORLD' not in cleaned.colnames:
                        cleaned['ERRA_WORLD'] = cleaned['ERRAWIN_IMAGE'] * pixel_scale
                    if 'ERRDEC_WORLD' not in cleaned.colnames:
                        cleaned['ERRDEC_WORLD'] = cleaned['ERRAWIN_IMAGE'] * pixel_scale
                    if 'ERRB_WORLD' not in cleaned.colnames:
                        cleaned['ERRB_WORLD'] = cleaned['ERRBWIN_IMAGE'] * pixel_scale
                    if 'ERRX2_WORLD' not in cleaned.colnames:
                        cleaned['ERRX2_WORLD'] = cleaned['ERRX2WIN_WORLD']
                    if 'ERRY2_WORLD' not in cleaned.colnames:
                        cleaned['ERRY2_WORLD'] = cleaned['ERRY2WIN_WORLD']
                    hdul[tbhdu].data = cleaned.as_array()
                    hdul.flush()
                else:
                    cleaned = catalog

            self.logger.info(f"Sextractor found {len(cleaned)} sources")
            if len(cleaned) == 0 or "FWHM_IMAGE" not in cleaned.colnames:
                fwhm = (
                    float(fwhm_pixels)
                    if np.isfinite(fwhm_pixels) and fwhm_pixels > 0
                    else 2.5
                )
            else:
                from astropy.stats import sigma_clipped_stats
                _, fwhm, _ = sigma_clipped_stats(cleaned["FWHM_IMAGE"], sigma=3)
            if not np.isfinite(fwhm) or fwhm <= 0:
                fwhm = 2.5

        except Exception as e:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            self.logger.info(
                f"Type: {exc_type.__name__} File: {fname} Line: {lineno} Message: {e}"
            )
            raise

        return {
            "output_dir": output_dir,
            "catalog": cleaned,
            "catalog_path": catalog_path,
            "fwhm": fwhm,
            "checkimages": None,
            "config": final_config,
            "weight_path": weight_path,
        }

    def place_science_header_as_ahead(
        self,
        science_image: str,
        reference_dir: str,
        ahead_filename: str = "science.ahead",
    ) -> str:
        """
        Extract the header from the science image and save it as an `.ahead` file in the reference directory.
        This ensures the reference image uses the science image's WCS and metadata for alignment.

        Args:
            science_image: Path to the science image (FITS file).
            reference_dir: Path to the directory where the `.ahead` file will be saved.
            ahead_filename: Name of the output `.ahead` file.

        Returns:
            Path to the created `.ahead` file.
        """
        from astropy.io import fits
        import os

        # Ensure the reference directory exists
        os.makedirs(reference_dir, exist_ok=True)

        # Open the science image and extract its header
        with fits.open(science_image) as hdul:
            header = hdul[0].header

        # Save the header as an `.ahead` file in the reference directory
        ahead_path = os.path.join(reference_dir, ahead_filename)
        with open(ahead_path, "w") as f:
            for card in header.cards:
                # Use str(card) to get the full card as a string
                f.write(f"{str(card)}\n")

        self.logger.debug("Science header written as .ahead file: %s", ahead_path)

        return ahead_path

    def _extract_wcs_and_scale(self, fits_path):
        from astropy.wcs.utils import proj_plane_pixel_scales
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            wcs_obj = get_wcs(header)
            # Geometric mean of X and Y scales accounts for pixel aspect ratio
            # and CD-matrix rotation, giving the isotropic pixel scale.
            _scales = proj_plane_pixel_scales(wcs_obj)
            pixel_scale = float(np.sqrt(_scales[0] * _scales[1])) * 3600.0
            return wcs_obj, pixel_scale, header

    # ------------------------ Align + resample via SCAMP/SWarp ------------------------
    def align_and_resample_both_images(
        self,
        science_image: str,
        reference_image: str,
        output_dir: Optional[str] = None,
        science_already_resampled: bool = False,
        reference_already_resampled: bool = False,
        resample_mode: str = "common_grid",
    ) -> Optional[dict]:
        """
        Align and resample both science and reference images.
        SCAMP is run on the reference image only (against the science catalog) to compute
        the WCS correction. SWarp resampling behaviour is controlled by resample_mode:

        - "common_grid" (default): Both images resampled onto the science image's pixel
          grid (same center, pixel scale, and shape). Best for pixel-based operations.
        - "native_scale": Each image resampled independently keeping its native pixel
          scale, but aligned to the same sky center. Output shapes differ if pixel
          scales differ. Best for template subtraction algorithms that handle
          different pixel scales internally.
        - "wcs_only": No SWarp resampling; only SCAMP WCS correction is applied.
          Best when you want to avoid any resampling.

        Args:
            science_image: Path to science image.
            reference_image: Path to reference image.
            output_dir: Output directory for aligned images.
            science_already_resampled: If True, skip SWarp for science; use image as-is.
            reference_already_resampled: If True, skip SWarp for reference; use image as-is.
            resample_mode: "common_grid", "native_scale", or "wcs_only".
        Returns:
            Dictionary with paths to aligned images and alignment metadata.
        """
        try:
            # --- Check that SCAMP and SWarp are installed ---
            scamp_available = self._is_executable_available("scamp")
            swarp_available = self._is_executable_available("swarp")

            if not scamp_available:
                self.logger.warning(
                    "SCAMP executable not found. SCAMP WCS refinement will be skipped. "
                    "Falling back to WCS-based reproject alignment (no astrometric correction)."
                )
            if not swarp_available and resample_mode != "wcs_only":
                self.logger.warning(
                    "SWarp executable not found but resample_mode=%r requires it. "
                    "Falling back to WCS-based reproject alignment.",
                    resample_mode,
                )

            # If SCAMP is not installed, we can't do astrometric refinement.
            # Fall back to WCS-based reproject which only needs valid WCS headers.
            if not scamp_available or (
                not swarp_available and resample_mode != "wcs_only"
            ):
                self.logger.info(
                    "Missing SCAMP/SWarp — using WCS-based reproject fallback for alignment."
                )
                result = self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )
                if result is not None:
                    return result
                self.logger.error(
                    "WCS-based reproject and AstroAlign both failed. "
                    "Cannot align images without SCAMP/SWarp."
                )
                return None

            output_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(science_image).parent
            )
            # Make aligned directories unique per science image to prevent crosstalk
            # when multiple images are processed in the same output directory
            science_base_name = Path(science_image).stem
            science_aligned_dir = output_dir / f"aligned_sci_{science_base_name}"
            reference_aligned_dir = output_dir / f"aligned_ref_{science_base_name}"

            # Delete existing directories if they exist
            if science_aligned_dir.exists():
                shutil.rmtree(science_aligned_dir)
            if reference_aligned_dir.exists():
                shutil.rmtree(reference_aligned_dir)

            # Create new, empty directories
            science_aligned_dir.mkdir(parents=True, exist_ok=True)
            reference_aligned_dir.mkdir(parents=True, exist_ok=True)

            # Copy and clean images
            sci_image_copy = science_aligned_dir / "science_image.fits"
            ref_image_copy = reference_aligned_dir / "reference_image.fits"

            # Copy science and reference images (weight maps not used for alignment)
            shutil.copy2(science_image, sci_image_copy)
            shutil.copy2(reference_image, ref_image_copy)

            # Pre-extract cutout from large reference images centered on science region
            # This ensures: (1) reference covers science region, (2) faster processing,
            # (3) output shapes match science image exactly.
            # WARNING: do NOT cut references with SIP/TPV distortion.  Moving CRPIX
            # to the cutout center shifts the origin of the distortion polynomial,
            # so the same A/B or PV coefficients describe the wrong part of the
            # focal plane and produce subpixel misalignments across the field.
            try:
                with fits.open(sci_image_copy) as _sh, fits.open(ref_image_copy) as _rh:
                    _sci_h = _sh[0].header
                    _ref_h = _rh[0].header
                    _sci_w = get_wcs(_sci_h)
                    _ref_w = get_wcs(_ref_h)
                    _sci_shape = (_sci_h.get("NAXIS2", 0), _sci_h.get("NAXIS1", 0))
                    _ref_shape = (_ref_h.get("NAXIS2", 0), _ref_h.get("NAXIS1", 0))

                    _ref_has_sip = any(k in _ref_h for k in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"])
                    _ref_has_pv = any(str(k).startswith("PV") for k in _ref_h)

                    # Only cut if reference is significantly larger (>1.5x in either dimension)
                    # AND has no non-linear distortion (SIP/TPV).
                    if (_ref_shape[0] > _sci_shape[0] * 1.5 or _ref_shape[1] > _sci_shape[1] * 1.5) \
                            and not (_ref_has_sip or _ref_has_pv):
                        from astropy.nddata.utils import Cutout2D
                        
                        # Get science center in world coords.
                        # Use correct numpy 0-based center: (nx-1)/2, (ny-1)/2.
                        _scx = (_sci_shape[1] - 1) / 2.0
                        _scy = (_sci_shape[0] - 1) / 2.0
                        _sci_cra, _sci_cdec = _sci_w.all_pix2world([_scx], [_scy], 0)
                        
                        # Project science center into reference pixel coords
                        _rcx_arr, _rcy_arr = _safe_world2pix(_ref_w, _sci_cra[0], _sci_cdec[0], 0)
                        _rcx, _rcy = float(np.atleast_1d(_rcx_arr)[0]), float(np.atleast_1d(_rcy_arr)[0])
                        
                        # Cutout size: science size + 20% margin for alignment shifts
                        _cut_h = int(_sci_shape[0] * 1.2)
                        _cut_w = int(_sci_shape[1] * 1.2)
                        
                        if np.isfinite(_rcx) and np.isfinite(_rcy):
                            _ref_cutout = Cutout2D(
                                _rh[0].data,
                                (_rcx, _rcy),
                                (_cut_h, _cut_w),
                                wcs=_ref_w,
                                mode='partial',
                                fill_value=np.nan,
                            )
                            # Update reference image with cutout
                            _ref_header_out = _ref_h.copy()
                            from functions import update_header_from_wcs
                            update_header_from_wcs(_ref_header_out, _ref_cutout.wcs)
                            _ref_header_out['NAXIS1'] = _cut_w
                            _ref_header_out['NAXIS2'] = _cut_h
                            fits.writeto(
                                ref_image_copy, 
                                _ref_cutout.data.astype(_rh[0].data.dtype), 
                                _ref_header_out, 
                                overwrite=True
                            )
                            self.logger.info(
                                "Pre-cut reference to %dx%d centered on science region (was %dx%d)",
                                _cut_w, _cut_h, _ref_shape[1], _ref_shape[0]
                            )
                    elif _ref_has_sip or _ref_has_pv:
                        self.logger.info(
                            "Reference pre-cutout skipped: reference has %s distortion "
                            "and cutting would shift the distortion origin.",
                            "SIP" if _ref_has_sip else "TPV/PV",
                        )
            except Exception as _e:
                self.logger.debug("Reference pre-cutout skipped: %s", _e)

            # Check if science image has SIP or TPV distortion parameters from solve-field/SCAMP
            # solve-field returns WCS with SIP parameters, SCAMP may convert to TPV/PV
            # Neither corrects the image data itself
            # We need to run SWarp on the science image to remove distortions using the distortion WCS
            with fits.open(sci_image_copy) as hdul:
                sci_head_initial = hdul[0].header

            # Check for SIP distortion keywords (A_ORDER, B_ORDER, etc.)
            has_sip = any(key in sci_head_initial for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"])
            # Check for TPV/PV distortion keywords (PV_*)
            has_pv = any(key.startswith("PV_") for key in sci_head_initial)
            # Check CTYPE for distortion projection
            ctype1 = str(sci_head_initial.get("CTYPE1", "")).upper()
            ctype2 = str(sci_head_initial.get("CTYPE2", "")).upper()
            has_distortion_ctype = "TAN-SIP" in ctype1 or "TAN-SIP" in ctype2 or "TPV" in ctype1 or "TPV" in ctype2

            has_distortion = has_sip or has_pv or has_distortion_ctype

            # Detect reference image distortion order for SCAMP DISTORT_DEGREES.
            # SCAMP's .head file REPLACES the reference's WCS, so DISTORT_DEGREES
            # must be high enough to model the reference's existing distortion.
            # If it's too low, the .head degrades the WCS by replacing high-order
            # SIP/PV with a lower-order polynomial, introducing systematic offsets.
            ref_distort_order = 1  # default: linear only
            try:
                with fits.open(ref_image_copy) as _rh:
                    _ref_hdr = _rh[0].header
                    _ref_sip_order = max(
                        int(_ref_hdr.get("A_ORDER", 0)),
                        int(_ref_hdr.get("B_ORDER", 0)),
                    )
                    _ref_pv_count = sum(1 for k in _ref_hdr if k.startswith("PV_"))
                    if _ref_sip_order > 0:
                        ref_distort_order = _ref_sip_order
                    elif _ref_pv_count > 0:
                        # PV distortion: TPV with N PV keywords roughly maps to
                        # polynomial degree (PV_0_1, PV_1_1, PV_2_1, ... PV_n_1)
                        ref_distort_order = min(_ref_pv_count // 2, 4)
                    _ref_ctype = str(_ref_hdr.get("CTYPE1", "")).upper()
                    if "SIP" in _ref_ctype or "TPV" in _ref_ctype:
                        if ref_distort_order < 2:
                            ref_distort_order = max(ref_distort_order, 2)
            except Exception:
                pass

            # Detect science image distortion order for GAIA SCAMP DISTORT_DEGREES.
            # Same logic as ref_distort_order: SCAMP's .head REPLACES the science WCS,
            # so DISTORT_DEGREES must be high enough to model the science's existing
            # distortion.  If it's too low, the .head degrades the WCS by replacing
            # high-order SIP/PV with a lower-order polynomial, introducing systematic
            # offsets that propagate to the reference SCAMP as well.
            sci_distort_order = 1  # default: linear only
            try:
                with fits.open(sci_image_copy) as _sh:
                    _sci_hdr_dist = _sh[0].header
                    _sci_sip_order = max(
                        int(_sci_hdr_dist.get("A_ORDER", 0)),
                        int(_sci_hdr_dist.get("B_ORDER", 0)),
                    )
                    _sci_pv_count = sum(1 for k in _sci_hdr_dist if k.startswith("PV_"))
                    if _sci_sip_order > 0:
                        sci_distort_order = _sci_sip_order
                    elif _sci_pv_count > 0:
                        sci_distort_order = min(_sci_pv_count // 2, 4)
                    _sci_ctype_dist = str(_sci_hdr_dist.get("CTYPE1", "")).upper()
                    if "SIP" in _sci_ctype_dist or "TPV" in _sci_ctype_dist:
                        if sci_distort_order < 2:
                            sci_distort_order = max(sci_distort_order, 2)
            except Exception:
                pass

            # Note: We do NOT run a separate distortion correction step here.
            # The final SWarp on both images together will handle distortion correction
            # as part of the resampling process. Running distortion correction separately
            # can produce output with different dimensions due to SIP/TPV distortion.
            if has_distortion:
                distortion_type = "SIP" if has_sip else "TPV/PV" if has_pv else "CTYPE-based"
                self.logger.info(
                    f"Science image has {distortion_type} distortion. "
                    "Correction will happen during SWarp resampling."
                )
            # Use weight map if available
            sci_w = self._guess_map_weight_path(str(sci_image_copy))
            if sci_w:
                self.logger.info("Alignment SExtractor: using science MAP_WEIGHT %s", sci_w)
            with (
                fits.open(sci_image_copy) as hdul,
                fits.open(ref_image_copy) as hdul_ref,
            ):
                sci_data, sci_head = hdul[0].data, hdul[0].header
                ref_data, ref_head = hdul_ref[0].data, hdul_ref[0].header
                from astropy.wcs.utils import proj_plane_pixel_scales
                sci_wcs = get_wcs(sci_head)
                ref_wcs = get_wcs(ref_head)
                sci_shape = sci_data.shape
                ref_shape = ref_data.shape
                # Isotropic pixel scale: geometric mean of X and Y scales.
                _sci_scales = proj_plane_pixel_scales(sci_wcs)
                _ref_scales = proj_plane_pixel_scales(ref_wcs)
                sci_pix_scale = float(np.sqrt(_sci_scales[0] * _sci_scales[1])) * 3600.0
                ref_pix_scale = float(np.sqrt(_ref_scales[0] * _ref_scales[1])) * 3600.0
                # Use science image shape directly for output grid.
                # Both images are resampled to match science dimensions and pixel scale.
                output_width, output_height = sci_shape[1], sci_shape[0]
                
                # Compute world coordinates of image center for SWarp CENTER.
                # SWarp uses FITS 1-based convention; the center of an nx x ny image
                # is at ((nx+1)/2, (ny+1)/2) in FITS coordinates.
                _scx = (output_width + 1) / 2.0
                _scy = (output_height + 1) / 2.0
                center_ra, center_dec = sci_wcs.all_pix2world([_scx], [_scy], 1)
                center_ra = float(center_ra[0])
                center_dec = float(center_dec[0])

                # Use the original science image for SWarp. Do NOT create a copy with
                # modified CRPIX/CRVAL — distortion terms (SIP, PV) are evaluated
                # relative to CRPIX, so moving CRPIX changes the WCS mapping and
                # introduces a small systematic offset. COMBINE=Y already guarantees
                # full-size output at the requested IMAGE_SIZE.
                sci_image_for_swarp = sci_image_copy
                
                self.logger.info(
                    "Output grid: %dx%d  CENTER=(%.6f,%.6f)  PIXEL_SCALE=%.4f (science was %dx%d, reference was %dx%d)",
                    output_width, output_height, center_ra, center_dec, sci_pix_scale,
                    sci_shape[1], sci_shape[0], ref_shape[1], ref_shape[0]
                )
                
                # Verify reference covers the science region (diagnostic only —
                # never abort alignment if this check fails or WCS doesn't converge).
                try:
                    sci_corners = np.array(
                        [[0, 0],
                         [sci_shape[1] - 1, 0],
                         [sci_shape[1] - 1, sci_shape[0] - 1],
                         [0, sci_shape[0] - 1]]
                    )
                    sci_world_corners = sci_wcs.all_pix2world(sci_corners, 0)
                    # _safe_world2pix returns NaN for non-converging points instead
                    # of raising NoConvergence (common with high-order SIP distortion).
                    ref_x, ref_y = _safe_world2pix(
                        ref_wcs, sci_world_corners[:, 0], sci_world_corners[:, 1], 0
                    )
                    ref_in_bounds = (np.isfinite(ref_x) & np.isfinite(ref_y) &
                                     (ref_x >= 0) & (ref_x < ref_shape[1]) &
                                     (ref_y >= 0) & (ref_y < ref_shape[0]))
                    n_oob = int((~ref_in_bounds).sum())
                    if n_oob > 0:
                        severity = "severe" if n_oob >= 2 else "minor"
                        self.logger.warning(
                            "Reference image may not fully cover science image region "
                            "(%d/4 corners out of bounds, %s). "
                            "Science corners in ref pixels: X=%s, Y=%s. "
                            "Uncovered regions will produce NaN/zero borders in the "
                            "aligned template, increasing the masked fraction and "
                            "reducing sources available for SFFT kernel fitting.",
                            n_oob, severity,
                            np.round(ref_x, 1).tolist(), np.round(ref_y, 1).tolist(),
                        )
                    else:
                        self.logger.debug("Reference image fully covers science region")
                except Exception as _cov_exc:
                    self.logger.debug("Coverage check skipped: %s", _cov_exc)
                
                reference_already_scamp = self._header_indicates_scamp(ref_head)

            # Extract sources
            sci_aperture_radius = sci_head.get("APER", 7)
            ref_aperture_radius = ref_head.get("APER", 7)
            self.logger.info(
                f"Extracting sources with aperture radii: science = {sci_aperture_radius:.1f} pixels, "
                f"reference = {ref_aperture_radius:.1f} pixels"
            )

            # Use crowded-field SExtractor config when requested (better deblending in dense fields)
            iy = getattr(self, "input_yaml", None) or {}
            ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
            templates_cfg = iy.get("templates", {}) if isinstance(iy, dict) else {}
            phot = iy.get("photometry") if isinstance(iy, dict) else None
            phot_crowded = bool(
                isinstance(phot, dict) and phot.get("crowded_field", False)
            )
            sextractor_crowded = ts.get(
                "sextractor_crowded", templates_cfg.get("crowded_field", phot_crowded)
            )
            # For alignment, disable weight maps to avoid masking sources
            # Weight maps can mask faint sources that are needed for alignment
            sci_w = None
            ref_w = None
            self.logger.info("Alignment SExtractor: weight maps disabled to avoid masking sources")
            
            # Log science image statistics to debug detection issues
            with fits.open(sci_image_copy) as hdul:
                sci_data = hdul[0].data
                self.logger.info(
                    "Science image stats: shape=%s, min=%.2f, max=%.2f, mean=%.2f, median=%.2f, std=%.2f",
                    sci_data.shape, np.nanmin(sci_data), np.nanmax(sci_data),
                    np.nanmean(sci_data), np.nanmedian(sci_data), np.nanstd(sci_data)
                )
                # Check if image has been modified (e.g., by SWarp copy)
                self.logger.info(
                    "Science image header FWHM: %.2f, APER: %.2f, CRPIX1: %.2f, CRPIX2: %.2f",
                    hdul[0].header.get("FWHM", "N/A"),
                    hdul[0].header.get("APER", "N/A"),
                    hdul[0].header.get("CRPIX1", "N/A"),
                    hdul[0].header.get("CRPIX2", "N/A")
                )
                # Check if this is a cropped image (from templates.py)
                self.logger.info(
                    "Science image path: %s (original: %s)",
                    sci_image_copy, science_image
                )

            # Pass 1: measure FWHM (kernel sized from aperture/FWHM header only)
            # Use header FWHM if available (set by main pipeline after initial source detection)
            # This avoids using stale/instrument-default FWHM values
            sci_hdr_fwhm = None
            ref_hdr_fwhm = None
            try:
                sci_hdr_fwhm = float(fits.getheader(str(sci_image_copy)).get("FWHM", fits.getheader(str(sci_image_copy)).get("fwhm")))
            except Exception:
                pass
            try:
                ref_hdr_fwhm = float(fits.getheader(str(ref_image_copy)).get("FWHM", fits.getheader(str(ref_image_copy)).get("fwhm")))
            except Exception:
                pass
            
            # Use main pipeline's SExtractorWrapper for alignment (same as main source detection)
            # This ensures consistent source detection behavior
            self.logger.info("Using SExtractorWrapper.run for alignment (with sensitive detection for sparse fields)")
            
            # Convert pixel scale to arcsec/pixel for SExtractorWrapper
            sci_seeing_fwhm = (sci_hdr_fwhm if sci_hdr_fwhm else 2.0) * sci_pix_scale
            ref_seeing_fwhm = (ref_hdr_fwhm if ref_hdr_fwhm else 2.0) * ref_pix_scale
            
            # SExtractorWrapper saves catalog as <stem>_PYSEx_CAT.fits in the mdir
            # SCAMP expects .cat extension for FITS-LDAC catalogs
            # Use return_raw=True to get the raw FITS-LDAC file path and avoid pandas conversion
            # Catalog paths must match the image paths that filter_matched_sources expects
            sci_catalog_path = str(science_aligned_dir / "science_image_PYSEx_CAT.cat")
            ref_catalog_path = str(reference_aligned_dir / "reference_image_PYSEx_CAT.cat")
            # SExtractorWrapper will copy the FITS-LDAC catalog to these paths when return_raw=True
            # These are the actual SExtractor outputs with full metadata
            
            # Pass 1: initial source detection with header/default FWHM for scale estimation.
            # Pass 2 (below) re-runs with FWHM-based scale for alignment-quality catalogs.
            sci_fwhm, sci_catalog_raw, sci_scale = self.sextractor.run(
                fits_path=str(sci_image_copy),
                pixel_scale=sci_pix_scale,
                seeing_fwhm=sci_seeing_fwhm,
                catalog_type="FITS_LDAC",
                use_FWHM=sci_hdr_fwhm if sci_hdr_fwhm else 0.0,
                crowded=sextractor_crowded,
                use_for_matching=True,
                mdir=str(science_aligned_dir),
                return_raw=True,
                detect_thresh=1.0,
                analysis_thresh=0.8,
                detect_minarea=1,
                back_size=64,
            )
            
            ref_fwhm, ref_catalog_raw, ref_scale = self.sextractor.run(
                fits_path=str(ref_image_copy),
                pixel_scale=ref_pix_scale,
                seeing_fwhm=ref_seeing_fwhm,
                catalog_type="FITS_LDAC",
                use_FWHM=ref_hdr_fwhm if ref_hdr_fwhm else 0.0,
                crowded=sextractor_crowded,
                use_for_matching=True,
                mdir=str(reference_aligned_dir),
                return_raw=True,
                detect_thresh=1.0,
                analysis_thresh=0.8,
                detect_minarea=1,
                back_size=64,
            )
            
            # SExtractorWrapper with return_raw=True copies the FITS-LDAC catalog to the mdir
            # The catalogs are now at the expected paths with full SExtractor metadata
            # No need to manually write FITS-LDAC files
            
            # Use the raw Tables for downstream processing
            sci_catalog = sci_catalog_raw
            ref_catalog = ref_catalog_raw
            
            # Convert to expected format - use the FITS_LDAC catalog path
            sci_sex = {"fwhm": sci_fwhm, "catalog": sci_catalog, "catalog_path": sci_catalog_path}
            ref_sex = {"fwhm": ref_fwhm, "catalog": ref_catalog, "catalog_path": ref_catalog_path}
            
            self.logger.info("SExtractor pass-1: %d sci / %d ref sources", len(sci_catalog) if sci_catalog is not None else 0, len(ref_catalog) if ref_catalog is not None else 0)

            fwhm_sci_pix = float(sci_sex["fwhm"]) if "fwhm" in sci_sex else 2.5
            fwhm_ref_pix = float(ref_sex["fwhm"]) if "fwhm" in ref_sex else 2.5

            # Sanity check: cap alignment FWHM using the pipeline's header FWHM.
            # The header FWHM comes from the carefully measured measure_image step
            # and is far more reliable than SExtractor's estimate in the presence of
            # extended sources or galaxies that inflate the FWHM.
            # If the alignment SExtractor returns a value > 1.5× the header FWHM,
            # it's almost certainly contaminated by non-point sources.
            FWHM_INFLATION_FACTOR = 1.5
            if sci_hdr_fwhm and np.isfinite(sci_hdr_fwhm) and sci_hdr_fwhm > 0:
                if fwhm_sci_pix > FWHM_INFLATION_FACTOR * sci_hdr_fwhm:
                    self.logger.warning(
                        "Alignment science FWHM %.1f px is inflated (> %.1f x header FWHM %.1f px); "
                        "capping at header FWHM.",
                        fwhm_sci_pix, FWHM_INFLATION_FACTOR, sci_hdr_fwhm,
                    )
                    fwhm_sci_pix = sci_hdr_fwhm
            if ref_hdr_fwhm and np.isfinite(ref_hdr_fwhm) and ref_hdr_fwhm > 0:
                if fwhm_ref_pix > FWHM_INFLATION_FACTOR * ref_hdr_fwhm:
                    self.logger.warning(
                        "Alignment reference FWHM %.1f px is inflated (> %.1f x header FWHM %.1f px); "
                        "capping at header FWHM.",
                        fwhm_ref_pix, FWHM_INFLATION_FACTOR, ref_hdr_fwhm,
                    )
                    fwhm_ref_pix = ref_hdr_fwhm

            # Hard cap: if FWHM is still unreasonable (no header value available), use defaults
            MAX_REASONABLE_FWHM = 30.0
            if fwhm_sci_pix > MAX_REASONABLE_FWHM:
                self.logger.warning(
                    "Alignment FWHM %.1f px is unreasonable (> %.0f px); using default FWHM.",
                    fwhm_sci_pix, MAX_REASONABLE_FWHM,
                )
                fwhm_sci_pix = sci_hdr_fwhm if (sci_hdr_fwhm and sci_hdr_fwhm > 0) else 8.5
            if fwhm_ref_pix > MAX_REASONABLE_FWHM:
                self.logger.warning(
                    "Alignment FWHM %.1f px is unreasonable (> %.0f px); using default FWHM.",
                    fwhm_ref_pix, MAX_REASONABLE_FWHM,
                )
                fwhm_ref_pix = ref_hdr_fwhm if (ref_hdr_fwhm and ref_hdr_fwhm > 0) else 8.5

            # Pass 2: re-run with kernel sized from FWHM-based scale for alignment.
            # Use a smaller scale than the pipeline cutout scale to ensure proper
            # source detection. Alignment needs to detect point sources, not use
            # the large cutout scale used for PSF building/subtraction.
            # Use 1.5×FWHM as a reasonable alignment scale (covers PSF without being excessive)
            sci_scale = int(max(5, 1.5 * fwhm_sci_pix))
            ref_scale = int(max(5, 1.5 * fwhm_ref_pix))
            combined_scale = max(sci_scale, ref_scale)
            self.logger.info("SExtractor pass-2: scale=%d px (FWHM-based)", combined_scale)
            
            # Re-run with FWHM-based scale using SExtractorWrapper
            # The pass-2 catalogs will overwrite the pass-1 catalogs (same paths)
            sci_fwhm2, sci_catalog2_raw, sci_scale2 = self.sextractor.run(
                fits_path=str(sci_image_copy),
                pixel_scale=sci_pix_scale,
                seeing_fwhm=fwhm_sci_pix * sci_pix_scale,
                catalog_type="FITS_LDAC",
                use_FWHM=fwhm_sci_pix,
                crowded=sextractor_crowded,
                use_for_matching=True,
                mdir=str(science_aligned_dir),
                return_raw=True,  # Return raw FITS-LDAC Table for SCAMP compatibility
                detect_thresh=1.0,
                analysis_thresh=0.8,
                detect_minarea=1,
                back_size=64,
            )
            
            ref_fwhm2, ref_catalog2_raw, ref_scale2 = self.sextractor.run(
                fits_path=str(ref_image_copy),
                pixel_scale=ref_pix_scale,
                seeing_fwhm=fwhm_ref_pix * ref_pix_scale,
                catalog_type="FITS_LDAC",
                use_FWHM=fwhm_ref_pix,
                crowded=sextractor_crowded,
                use_for_matching=True,
                mdir=str(reference_aligned_dir),
                return_raw=True,  # Return raw FITS-LDAC Table for SCAMP compatibility
                detect_thresh=1.0,
                analysis_thresh=0.8,
                detect_minarea=1,
                back_size=64,
            )
            
            # SExtractorWrapper with return_raw=True copies the FITS-LDAC catalog to the mdir
            # The catalogs are now at the expected paths with full SExtractor metadata
            # No need to manually write FITS-LDAC files
            
            # Use the raw Tables for downstream processing
            sci_catalog2 = sci_catalog2_raw
            ref_catalog2 = ref_catalog2_raw
            
            # Convert to expected format - use the FITS_LDAC catalog path
            sci_sex = {"fwhm": sci_fwhm2, "catalog": sci_catalog2, "catalog_path": sci_catalog_path}
            ref_sex = {"fwhm": ref_fwhm2, "catalog": ref_catalog2, "catalog_path": ref_catalog_path}
            
            self.logger.info("SExtractor pass-2: %d sci / %d ref sources", len(sci_catalog2) if sci_catalog2 is not None else 0, len(ref_catalog2) if ref_catalog2 is not None else 0)

            # --- Sparse-field retry: if very few sources, re-run with maximum sensitivity ---
            _sparse_thresh = int(
                ts.get("alignment_sparse_field_retry_count", 15)
                if isinstance(ts, dict) else 15
            )
            _n_sci2 = len(sci_catalog2) if sci_catalog2 is not None else 0
            _n_ref2 = len(ref_catalog2) if ref_catalog2 is not None else 0
            if _n_sci2 < _sparse_thresh or _n_ref2 < _sparse_thresh:
                self.logger.warning(
                    "Sparse field detected (%d sci / %d ref sources < %d). "
                    "Re-running SExtractor with maximum sensitivity "
                    "(DETECT_THRESH=0.5, BACK_SIZE=32, CLEAN=N).",
                    _n_sci2, _n_ref2, _sparse_thresh,
                )
                # Adaptive BACK_SIZE: use min(32, max(16, image_size//8))
                with fits.open(str(sci_image_copy), memmap=False) as _h:
                    _sci_ny, _sci_nx = _h[0].data.shape
                with fits.open(str(ref_image_copy), memmap=False) as _h:
                    _ref_ny, _ref_nx = _h[0].data.shape
                _sci_back = min(32, max(16, min(_sci_nx, _sci_ny) // 8))
                _ref_back = min(32, max(16, min(_ref_nx, _ref_ny) // 8))

                if _n_sci2 < _sparse_thresh:
                    sci_fwhm3, sci_catalog3_raw, _ = self.sextractor.run(
                        fits_path=str(sci_image_copy),
                        pixel_scale=sci_pix_scale,
                        seeing_fwhm=fwhm_sci_pix * sci_pix_scale,
                        catalog_type="FITS_LDAC",
                        use_FWHM=fwhm_sci_pix,
                        crowded=False,
                        use_for_matching=True,
                        mdir=str(science_aligned_dir),
                        return_raw=True,
                        detect_thresh=0.5,
                        analysis_thresh=0.3,
                        detect_minarea=1,
                        back_size=_sci_back,
                    )
                    if sci_catalog3_raw is not None and len(sci_catalog3_raw) > _n_sci2:
                        self.logger.info(
                            "Sparse-field retry: science %d → %d sources",
                            _n_sci2, len(sci_catalog3_raw),
                        )
                        sci_fwhm2 = sci_fwhm3
                        sci_catalog2_raw = sci_catalog3_raw
                        sci_catalog2 = sci_catalog3_raw
                        fwhm_sci_pix = float(sci_fwhm3) if sci_fwhm3 and sci_fwhm3 > 0 else fwhm_sci_pix

                if _n_ref2 < _sparse_thresh:
                    ref_fwhm3, ref_catalog3_raw, _ = self.sextractor.run(
                        fits_path=str(ref_image_copy),
                        pixel_scale=ref_pix_scale,
                        seeing_fwhm=fwhm_ref_pix * ref_pix_scale,
                        catalog_type="FITS_LDAC",
                        use_FWHM=fwhm_ref_pix,
                        crowded=False,
                        use_for_matching=True,
                        mdir=str(reference_aligned_dir),
                        return_raw=True,
                        detect_thresh=0.5,
                        analysis_thresh=0.3,
                        detect_minarea=1,
                        back_size=_ref_back,
                    )
                    if ref_catalog3_raw is not None and len(ref_catalog3_raw) > _n_ref2:
                        self.logger.info(
                            "Sparse-field retry: reference %d → %d sources",
                            _n_ref2, len(ref_catalog3_raw),
                        )
                        ref_fwhm2 = ref_fwhm3
                        ref_catalog2_raw = ref_catalog3_raw
                        ref_catalog2 = ref_catalog3_raw
                        fwhm_ref_pix = float(ref_fwhm3) if ref_fwhm3 and ref_fwhm3 > 0 else fwhm_ref_pix

                # Update sex dicts with retry results
                sci_sex = {"fwhm": sci_fwhm2, "catalog": sci_catalog2, "catalog_path": sci_catalog_path}
                ref_sex = {"fwhm": ref_fwhm2, "catalog": ref_catalog2, "catalog_path": ref_catalog_path}

            # Save copies of the full catalogs for SCAMP BEFORE filter_matched_sources overwrites them
            # This must happen immediately after pass-2 SExtractor completes
            sci_catalog_scamp_backup = str(science_aligned_dir / "science_image_PYSEx_CAT_scamp.cat")
            ref_catalog_scamp_backup = str(reference_aligned_dir / "reference_image_PYSEx_CAT_scamp.cat")
            shutil.copy2(sci_catalog_path, sci_catalog_scamp_backup)
            shutil.copy2(ref_catalog_path, ref_catalog_scamp_backup)

            # Pre-compute SWarp resampling methods so Phase 1b can use them.
            # (Moved here from below so science-first GAIA SWarp has access.)
            # Use configurable undersampled threshold (default 2.5 px)
            undersampled_thresh = float(
                self.input_yaml.get("undersampled_fwhm_threshold", 2.5)
                if isinstance(self.input_yaml, dict)
                else 2.5
            )
            sci_is_undersampled = fwhm_sci_pix < undersampled_thresh
            ref_is_undersampled = fwhm_ref_pix < undersampled_thresh

            def _swarp_resampling_type(
                is_undersampled: bool, fwhm_pix: float, threshold: float = 2.5
            ) -> str:
                """
                Select SWarp resampling type based on FWHM for optimal PSF preservation.

                Three-tier strategy:
                - BILINEAR: FWHM < 1.5 px (best for very small/undersampled PSFs)
                - LANCZOS2: 1.5 <= FWHM < threshold (balanced quality/preservation)
                - LANCZOS3: FWHM >= threshold (full quality for well-sampled)
                """
                if is_undersampled is None:
                    is_undersampled = False
                if fwhm_pix is None or not np.isfinite(fwhm_pix):
                    self.logger.warning(
                        "Invalid FWHM (%s), defaulting to LANCZOS3", fwhm_pix
                    )
                    return "LANCZOS3"
                # Three-tier selection based on FWHM
                if fwhm_pix < 1.5:
                    method = "BILINEAR"
                    reason = "FWHM < 1.5 px (very undersampled)"
                elif fwhm_pix < threshold:
                    method = "LANCZOS2"
                    reason = f"FWHM < {threshold} px (undersampled)"
                else:
                    method = "LANCZOS3"
                    reason = f"FWHM >= {threshold} px (well-sampled)"
                self.logger.debug(
                    "SWarp resampling: %s selected (%s, FWHM=%.2f px)",
                    method, reason, fwhm_pix
                )
                return method

            sci_resampling_method = _swarp_resampling_type(
                sci_is_undersampled,
                sci_hdr_fwhm if (sci_hdr_fwhm and sci_hdr_fwhm > 0) else fwhm_sci_pix,
                undersampled_thresh,
            )
            ref_resampling_method = _swarp_resampling_type(
                ref_is_undersampled,
                ref_hdr_fwhm if (ref_hdr_fwhm and ref_hdr_fwhm > 0) else fwhm_ref_pix,
                undersampled_thresh,
            )

            # ------------------------------------------------------------------
            # Phase 1+2: Science-first GAIA astrometry (Option B)
            # ------------------------------------------------------------------
            science_first_astrometry = bool(
                isinstance(ts, dict) and ts.get("science_first_astrometry", False)
            )
            if science_first_astrometry:
                # SCAMP needs at least ~5 detections to match against GAIA.
                # Skip the GAIA phase entirely for very sparse fields to save time
                # and avoid a confusing "Not enough matched detections" warning.
                n_sci_for_gaia = len(sci_sex.get("catalog", []) or [])
                if n_sci_for_gaia < 8:
                    self.logger.info(
                        "Science-first GAIA astrometry: skipping (only %d science "
                        "detections, need >= 8 for reliable GAIA cross-match).",
                        n_sci_for_gaia,
                    )
                    science_first_astrometry = False
                else:
                    self.logger.info("Science-first GAIA astrometry: Phase 1a (SCAMP)")

                    # Compute feasible DISTORT_DEGREES for science GAIA SCAMP.
                    # Same logic as reference SCAMP: ensure enough sources for the
                    # polynomial degree.  Using n_sci_for_gaia as a proxy — SCAMP's
                    # internal GAIA matching typically matches 50-80% of sources.
                    _sci_min_sources_for_degree = {1: 12, 2: 24, 3: 40, 4: 60}
                    _sci_feasible_degree = 1
                    for _deg in range(sci_distort_order, 0, -1):
                        if n_sci_for_gaia >= _sci_min_sources_for_degree.get(_deg, 999):
                            _sci_feasible_degree = _deg
                            break

                    if _sci_feasible_degree < sci_distort_order:
                        self.logger.warning(
                            "Science has distortion order %d but only %d sources "
                            "(need %d for degree %d). GAIA SCAMP will use degree %d. "
                            "If alignment is poor, consider disabling science-first astrometry.",
                            sci_distort_order, n_sci_for_gaia,
                            _sci_min_sources_for_degree.get(sci_distort_order, 999),
                            sci_distort_order, _sci_feasible_degree,
                        )
                    self.logger.info(
                        "Science GAIA SCAMP: DISTORT_DEGREES=%d (sci order=%d, %d sources)",
                        _sci_feasible_degree, sci_distort_order, n_sci_for_gaia,
                    )

                    # Phase 1a: SCAMP on science catalog vs GAIA-DR3
                    # SCAMP writes .head next to the catalog, so output_dir must be
                    # the same directory as the catalog for the glob to find it.
                    sci_gaia_dir = science_aligned_dir
                    sci_gaia_scamp = self.run_scamp(
                        catalog_paths=sci_catalog_scamp_backup,
                        reference_cat=None,  # triggers GAIA-DR3
                        output_dir=str(sci_gaia_dir),
                        config={
                            "DISTORT_DEGREES": _sci_feasible_degree,
                            "CROSSID_RADIUS": "5.0",
                            "POSITION_MAXERR": "1.0",
                            "SN_THRESHOLDS": "3.0,100000.0",
                            "MATCH_RESOL": "0.0",
                            "ASTREF_WEIGHT": "1",
                            "VERBOSE_TYPE": "FULL"
                            if self.verbose_level >= 2
                            else "NORMAL",
                        },
                    )
                    if sci_gaia_scamp is None:
                        self.logger.warning(
                            "Science GAIA SCAMP failed. Disabling science-first astrometry."
                        )
                        science_first_astrometry = False
                    else:
                        sci_head_src = sci_gaia_scamp.get("head_file")
                        if not (sci_head_src and Path(sci_head_src).exists()):
                            self.logger.warning(
                                "No GAIA SCAMP .head produced; disabling science-first astrometry."
                            )
                            science_first_astrometry = False

                if science_first_astrometry:
                    # Copy .head next to sci_image_copy for SWarp
                    sci_head_dst = Path(sci_image_copy).with_suffix(".head")
                    if Path(sci_head_src).resolve() != sci_head_dst.resolve():
                        shutil.copy2(sci_head_src, sci_head_dst)
                    _normalize_head_file(sci_head_dst)
                    self.logger.info(
                        "Copied GAIA SCAMP .head to %s", sci_head_dst
                    )

                    # Phase 1b: SWarp science to GAIA-corrected grid
                    self.logger.info("Phase 1b: SWarp science to GAIA grid")
                    sci_gaia_swarp_dir = science_aligned_dir / "swarp_gaia"
                    sci_gaia_swarp_dir.mkdir(parents=True, exist_ok=True)

                    sci_gaia_swarp_cfg = {
                        "CENTER_TYPE": "MANUAL",
                        "CENTER": f"{center_ra:.8f},{center_dec:.8f}",
                        "PIXELSCALE_TYPE": "MANUAL",
                        "PIXEL_SCALE": sci_pix_scale,
                        "IMAGE_SIZE": f"{output_width},{output_height}",
                        "RESAMPLING_TYPE": sci_resampling_method,
                        "COMBINE": "Y",
                        "COMBINE_TYPE": "MEDIAN",
                        "OVERSAMPLING": 0,
                    }

                    sci_gaia_swarp_res = self.run_swarp(
                        [str(sci_image_copy)],
                        scamp_results=sci_gaia_scamp,
                        output_dir=str(sci_gaia_swarp_dir),
                        config=sci_gaia_swarp_cfg,
                        no_weight_maps=False,
                    )
                    if sci_gaia_swarp_res is None:
                        self.logger.warning(
                            "Science GAIA SWarp failed. Disabling science-first astrometry."
                        )
                        science_first_astrometry = False
                    else:
                        science_corrected_raw = Path(
                            sci_gaia_swarp_res["corrected_image"]
                        )
                        if not science_corrected_raw.exists():
                            self.logger.warning(
                                "Science GAIA SWarp output missing. Disabling science-first astrometry."
                            )
                            science_first_astrometry = False
                        else:
                            # Rename to a known stem for predictable catalog paths
                            science_corrected_fits = (
                                science_aligned_dir / "science_corrected.fits"
                            )
                            shutil.copy2(
                                str(science_corrected_raw),
                                str(science_corrected_fits),
                            )

                            # Update science image paths for downstream
                            sci_image_copy = science_corrected_fits
                            sci_image_for_swarp = science_corrected_fits

                            # Re-read WCS and grid from corrected science
                            with fits.open(sci_image_copy) as hdul:
                                sci_head = hdul[0].header
                                sci_data = hdul[0].data
                                sci_wcs = get_wcs(sci_head)
                                sci_shape = sci_data.shape
                                from astropy.wcs.utils import (
                                    proj_plane_pixel_scales,
                                )

                                _scales = proj_plane_pixel_scales(sci_wcs)
                                sci_pix_scale = (
                                    float(np.sqrt(_scales[0] * _scales[1])) * 3600.0
                                )
                                output_width = sci_shape[1]
                                output_height = sci_shape[0]
                                # Do NOT recompute center_ra/center_dec here.
                                # Phase 1 SWarp used the original center (from
                                # pre-GAIA WCS).  Recomputing from the GAIA-corrected
                                # WCS would shift the grid center by the GAIA
                                # correction, putting science and reference on
                                # different grids (off by up to several pixels).

                            self.logger.info(
                                "Science corrected: shape=%s scale=%.4f center=(%.6f,%.6f)",
                                sci_shape,
                                sci_pix_scale,
                                center_ra,
                                center_dec,
                            )

                            # Phase 2: Re-detect sources on corrected science
                            self.logger.info(
                                "Phase 2: Re-detecting sources on corrected science..."
                            )
                            (
                                sci_fwhm_gaia,
                                sci_catalog_gaia_raw,
                                sci_scale_gaia,
                            ) = self.sextractor.run(
                                fits_path=str(sci_image_copy),
                                pixel_scale=sci_pix_scale,
                                seeing_fwhm=fwhm_sci_pix * sci_pix_scale,
                                catalog_type="FITS_LDAC",
                                use_FWHM=fwhm_sci_pix,
                                crowded=sextractor_crowded,
                                use_for_matching=True,
                                mdir=str(science_aligned_dir),
                                return_raw=True,
                            )

                            # Predictable catalog path based on known stem
                            sci_catalog_path = str(
                                science_aligned_dir
                                / "science_corrected_PYSEx_CAT.cat"
                            )

                            # Update the sci_sex dict for downstream
                            sci_sex = {
                                "fwhm": sci_fwhm_gaia,
                                "catalog": sci_catalog_gaia_raw,
                                "catalog_path": sci_catalog_path,
                            }
                            fwhm_sci_pix = (
                                float(sci_fwhm_gaia)
                                if sci_fwhm_gaia and float(sci_fwhm_gaia) > 0
                                else fwhm_sci_pix
                            )

                            # Backup corrected science catalog for reference SCAMP
                            sci_catalog_scamp_backup = str(
                                science_aligned_dir
                                / "science_corrected_PYSEx_CAT_scamp.cat"
                            )
                            shutil.copy2(sci_catalog_path, sci_catalog_scamp_backup)

                            self.logger.info(
                                "Corrected science catalog: %s sources, FWHM=%.2f px",
                                len(sci_catalog_gaia_raw)
                                if sci_catalog_gaia_raw is not None
                                else 0,
                                fwhm_sci_pix,
                            )

            n_sci = len(sci_sex.get("catalog", []))
            n_ref = len(ref_sex.get("catalog", []))

            fwhm_sci_pix = float(sci_sex["fwhm"]) if "fwhm" in sci_sex else 2.5
            fwhm_ref_pix = float(ref_sex["fwhm"]) if "fwhm" in ref_sex else 2.5

            # Cap pass-2 FWHM using header values (same logic as pass-1 cap).
            # Pass-2 values are used for SCAMP CROSSID_RADIUS, so inflation here
            # directly causes SCAMP to use a too-large search radius, accepting
            # poor WCS solutions with ~1 px systematic offset.
            if sci_hdr_fwhm and np.isfinite(sci_hdr_fwhm) and sci_hdr_fwhm > 0:
                if fwhm_sci_pix > FWHM_INFLATION_FACTOR * sci_hdr_fwhm:
                    self.logger.warning(
                        "Pass-2 science FWHM %.1f px inflated (> %.1f x header %.1f px); capping.",
                        fwhm_sci_pix, FWHM_INFLATION_FACTOR, sci_hdr_fwhm,
                    )
                    fwhm_sci_pix = sci_hdr_fwhm
            if ref_hdr_fwhm and np.isfinite(ref_hdr_fwhm) and ref_hdr_fwhm > 0:
                if fwhm_ref_pix > FWHM_INFLATION_FACTOR * ref_hdr_fwhm:
                    self.logger.warning(
                        "Pass-2 reference FWHM %.1f px inflated (> %.1f x header %.1f px); capping.",
                        fwhm_ref_pix, FWHM_INFLATION_FACTOR, ref_hdr_fwhm,
                    )
                    fwhm_ref_pix = ref_hdr_fwhm

            # Recompute undersampling flags using the header FWHM when available.
            # The SExtractor-measured FWHM can be inflated even after point-source
            # quality cuts (e.g. ZTF science FWHM=2.03 px but SExtractor reports
            # 2.78 px).  Using the inflated value causes LANCZOS3 to be selected
            # for undersampled data, introducing ringing artifacts that distort
            # the PSF and produce dipoles in the subtraction.
            sci_fwhm_for_undersample = (
                sci_hdr_fwhm if (sci_hdr_fwhm and sci_hdr_fwhm > 0) else fwhm_sci_pix
            )
            ref_fwhm_for_undersample = (
                ref_hdr_fwhm if (ref_hdr_fwhm and ref_hdr_fwhm > 0) else fwhm_ref_pix
            )
            sci_is_undersampled = sci_fwhm_for_undersample < undersampled_thresh
            ref_is_undersampled = ref_fwhm_for_undersample < undersampled_thresh

            fwhm_sci_arcsec = fwhm_sci_pix * sci_pix_scale
            fwhm_ref_arcsec = fwhm_ref_pix * ref_pix_scale
            self.logger.info(
                "FWHM: sci=%.2f px (%.2f\") ref=%.2f px (%.2f\")  "
                "[header: sci=%.2f ref=%.2f  undersampled: sci=%s ref=%s]",
                fwhm_sci_pix, fwhm_sci_arcsec,
                fwhm_ref_pix, fwhm_ref_arcsec,
                sci_hdr_fwhm or -1, ref_hdr_fwhm or -1,
                sci_is_undersampled, ref_is_undersampled,
            )

            crossid_radius = max(2.0 * max(fwhm_sci_arcsec, fwhm_ref_arcsec), 3.0)
            self.logger.info("Cross-match radius: %.2f arcsec (2*FWHM)", crossid_radius)

            # Configurable minimum matched sources for a reliable SCAMP solution.
            # SCAMP needs at least ~6 matched sources for a linear WCS fit
            # (shift, scale, rotation = 4 params, plus 2 for robustness).
            align_cfg = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
            min_matched = int(align_cfg.get("alignment_min_matched_sources", 6))
            max_match_radius = float(align_cfg.get("alignment_max_match_radius_arcsec", 30.0))

            # Adaptive minimum for sparse fields: if the science catalog has
            # few sources, requiring 6 matches may be impossible.  Lower the
            # gate so SCAMP can still run (a linear fit needs only 3-4 matches).
            n_sci_sources = len(sci_sex.get("catalog", []) or [])
            if n_sci_sources > 0 and n_sci_sources <= 25:
                adaptive_min = max(3, int(np.ceil(0.5 * n_sci_sources)))
                if adaptive_min < min_matched:
                    self.logger.info(
                        "Sparse field: science has only %d sources; lowering min_matched "
                        "from %d to %d (50%% of science catalog).",
                        n_sci_sources, min_matched, adaptive_min,
                    )
                    min_matched = adaptive_min

            def _do_match(radius_arcsec: float):
                return self.filter_matched_sources(
                    sci_cat_path=sci_sex["catalog_path"],
                    ref_cat_path=ref_sex["catalog_path"],
                    match_radius_arcsec=radius_arcsec,
                    sci_image_path=str(sci_image_copy),
                    ref_image_path=str(ref_image_copy),
                )

            try:
                _num_matched, _ = _do_match(crossid_radius)
                # Retry with larger radius if too few matches
                if _num_matched < min_matched and crossid_radius < max_match_radius:
                    retry_radius = min(crossid_radius * 2.0, max_match_radius)
                    self.logger.warning(
                        "Only %d matched sources (< %d minimum). Retrying with %.1f\" radius (was %.1f\").",
                        _num_matched, min_matched, retry_radius, crossid_radius,
                    )
                    _num_matched, _ = _do_match(retry_radius)
                    crossid_radius = retry_radius

                if _num_matched >= 3 and _num_matched < min_matched:
                    # We have enough matches for SCAMP to solve a linear WCS
                    # (4 parameters: shift+scale/rotation).  Don't inflate the
                    # radius — the WCS is good, we just have a sparse field.
                    self.logger.info(
                        "Sparse field: %d matched sources at %.1f\" radius (< %d default minimum). "
                        "Proceeding with SCAMP — a linear fit needs only 3-4 matches.",
                        _num_matched, crossid_radius, min_matched,
                    )
                    min_matched = _num_matched

                elif _num_matched < 3:
                    # Wide-offset fallback: estimate global WCS offset from headers
                    # and retry with a radius that encompasses it. This handles cases
                    # where independently plate-solved images have large WCS discrepancies.
                    self.logger.info(
                        "Only %d matched sources after retry (< 3 minimum). "
                        "Attempting wide-offset estimation...",
                        _num_matched,
                    )
                    try:
                        # Restore original catalogs from backup before re-matching
                        shutil.copy2(sci_catalog_scamp_backup, sci_catalog_path)
                        shutil.copy2(ref_catalog_scamp_backup, ref_catalog_path)

                        dra, ddec = self._estimate_wcs_offset_from_headers(
                            str(sci_image_copy), str(ref_image_copy),
                        )
                        offset_magnitude = float(np.sqrt(dra**2 + ddec**2))

                        if offset_magnitude > crossid_radius:
                            # Use a radius that encompasses the offset plus scatter
                            wide_radius = min(
                                offset_magnitude + 3.0 * crossid_radius,
                                max_match_radius * 3.0,
                            )
                            self.logger.info(
                                "Wide-offset match (from WCS headers): offset=%.1f\" "
                                "(dRA=%.1f\" dDec=%.1f\"), using %.1f\" match radius",
                                offset_magnitude, dra, ddec, wide_radius,
                            )
                            _num_matched, _ = _do_match(wide_radius)
                            crossid_radius = wide_radius

                            if _num_matched >= min_matched:
                                self.logger.info(
                                    "Wide-offset matching succeeded: %d matched sources",
                                    _num_matched,
                                )
                            elif _num_matched >= 3:
                                self.logger.info(
                                    "Wide-offset matching found %d sources (< %d minimum). "
                                    "Proceeding with SCAMP using large CROSSID_RADIUS.",
                                    _num_matched, min_matched,
                                )
                                min_matched = _num_matched
                            else:
                                # Try catalog-based estimation as secondary
                                dra2, ddec2, n_consistent = self._estimate_wcs_offset(
                                    sci_catalog_scamp_backup, ref_catalog_scamp_backup,
                                )
                                cat_offset = float(np.sqrt(dra2**2 + ddec2**2))
                                if cat_offset > crossid_radius and n_consistent >= 3:
                                    wide_radius2 = min(
                                        cat_offset + 3.0 * crossid_radius,
                                        max_match_radius * 3.0,
                                    )
                                    if wide_radius2 > crossid_radius:
                                        shutil.copy2(sci_catalog_scamp_backup, sci_catalog_path)
                                        shutil.copy2(ref_catalog_scamp_backup, ref_catalog_path)
                                        _num_matched, _ = _do_match(wide_radius2)
                                        crossid_radius = wide_radius2
                                        if _num_matched >= 3:
                                            self.logger.info(
                                                "Catalog-based wide match: %d sources, proceeding",
                                                _num_matched,
                                            )
                                            min_matched = _num_matched
                        else:
                            self.logger.info(
                                "WCS header offset too small (%.1f\").",
                                offset_magnitude,
                            )
                    except Exception as offset_err:
                        self.logger.warning(
                            "Wide-offset estimation failed: %s", offset_err,
                        )

                if _num_matched < min_matched:
                    self.logger.warning(
                        "Only %d matched sources after all retries (< %d minimum). "
                        "Falling back to reproject/AstroAlign.",
                        _num_matched, min_matched,
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
            except Exception as e:
                self.logger.warning(
                    "Source matching failed: %s. Falling back to reproject/AstroAlign.",
                    e,
                )
                return self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )


            try:
                self.plot_matched_sources_side_by_side(
                    sci_image_path=str(sci_image_copy),
                    ref_image_path=str(ref_image_copy),
                    sci_cat_path=sci_sex["catalog_path"],
                    ref_cat_path=ref_sex["catalog_path"],
                    output_plot_path=output_dir / f"matched_sources_{Path(sci_image_copy).stem}.png",
                    label_color="#FF0000",
                    label_fontsize=7,
                    circle_radius_sci=fwhm_sci_pix,
                    circle_radius_ref=fwhm_ref_pix,
                    matched_circle_color="#FF0000",
                    unmatched_circle_color="blue",
                )
            except Exception as e:
                self.logger.debug("Matched-sources plot failed (non-fatal): %s", e)

            self.logger.info(
                "SCAMP+SWarp alignment: %d matched sources (sci=%d ref=%d)",
                _num_matched, n_sci, n_ref,
            )
            pix_scale = sci_pix_scale

            # SCAMP: derive parameters from FWHM and pixel scale
            crossid_arcsec = max(
                1.5 * max(fwhm_sci_arcsec, fwhm_ref_arcsec), 1.5 * pix_scale, 2.0
            )
            # Use at least the cross-match radius that succeeded
            if crossid_radius > crossid_arcsec:
                crossid_arcsec = crossid_radius
            # POSITION_MAXERR: maximum positional uncertainty for sources
            # accepted in the astrometric fit.  Too tight → SCAMP rejects
            # sources with any WCS imprecision.  Too loose → contamination.
            # Floor at 1.0" to handle typical plate-solve uncertainties.
            is_sparse_field = _num_matched < 50
            position_maxerr_arcsec = max(
                1.0,
                2.0 * pix_scale,
                0.5 * fwhm_ref_arcsec,
            )
            # Adaptive SN threshold: lower minimum for sparse fields to retain
            # faint sources that passed our SNR_WIN pre-filter.  SCAMP's
            # SN_THRESHOLDS uses FLUX_AUTO/FLUXERR_AUTO which can differ from
            # SNR_WIN (windowed), so we set it below our pre-filter threshold
            # to avoid double-filtering. SCAMP weights each source by its
            # positional uncertainty (ERRAWIN/ERRBWIN) regardless.
            sn_thresholds = "1.5,100000.0" if is_sparse_field else "3.0,100000.0"
            # Adaptive DISTORT_DEGREES: must be high enough to model the
            # reference image's existing distortion (SIP/PV), because SCAMP's
            # .head file REPLACES the reference WCS.  If DISTORT_DEGREES is
            # lower than the reference's distortion order, the .head degrades
            # the WCS — high-order SIP terms are lost and replaced with a
            # lower-order polynomial, introducing systematic offsets of several
            # pixels across the field.
            #
            # Polynomial degree N has (N+1)(N+2)/2 parameters per axis.
            # For sparse fields, degree 1 (linear: shift/rotation/scale) is
            # stable with as few as 5 sources.  Higher degrees require more
            # sources to avoid overfitting.
            #   Degree 1:  3 params/axis,  6 total, need >= 5 sources
            #   Degree 2:  6 params/axis, 12 total, need >= 12 sources
            #   Degree 3: 10 params/axis, 20 total, need >= 20 sources
            #   Degree 4: 15 params/axis, 30 total, need >= 30 sources
            # Minimum matched sources for each polynomial degree.
            # Degree 1 (linear WCS: shift/rotation/scale) needs only ~5 sources
            # to be stable for sparse fields.  SCAMP does not accept degree 0.
            _min_sources_for_degree = {0: 3, 1: 5, 2: 12, 3: 20, 4: 30}

            # Start with the reference's detected distortion order
            required_degree = ref_distort_order

            # Find the highest degree we can actually fit with available sources
            feasible_degree = 0
            for deg in range(required_degree, -1, -1):
                if _num_matched >= _min_sources_for_degree.get(deg, 999):
                    feasible_degree = deg
                    break

            # Track whether SCAMP's .head needs SIP preservation
            preserve_sip_in_head = False
            if feasible_degree < required_degree:
                if _num_matched < _min_sources_for_degree[0]:
                    self.logger.warning(
                        "Only %d matched sources — too few for any WCS correction "
                        "(need %d). Falling back to reproject/AstroAlign.",
                        _num_matched, _min_sources_for_degree[0],
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                # SCAMP will run at a lower degree than the reference's SIP order.
                # SCAMP's degree-2 solution captures the dominant distortion and
                # gives good alignment (0.35px centroid). Preserving the reference's
                # original SIP on top of SCAMP's PV distortion creates a double
                # correction, so we use SCAMP's solution as-is.
                preserve_sip_in_head = False
                self.logger.info(
                    "Reference has distortion order %d but only %d matched sources "
                    "(need %d for degree %d). Running SCAMP at degree %d "
                    "(SCAMP PV distortion replaces reference SIP).",
                    required_degree, _num_matched,
                    _min_sources_for_degree.get(required_degree, 999),
                    required_degree, feasible_degree,
                )

            distort_degrees = feasible_degree
            scamp_config_base = {
                "CROSSID_RADIUS": crossid_arcsec,
                "POSITION_MAXERR": position_maxerr_arcsec,
                "SN_THRESHOLDS": sn_thresholds,
                "MATCH_RESOL": "0.0",  # Auto-select best matching resolution
                "ASTREF_WEIGHT": "1",  # Weight by magnitude (prioritize bright, well-measured)
                "VERBOSE_TYPE": "FULL" if self.verbose_level >= 2 else "NORMAL",
                "WRITE_XML": "Y",
                # Relax ellipticity cut for sparse fields to include more sources,
                # but not so permissive that galaxies with poor centroids enter.
                "ELLIPTICITY_MAX": 0.7 if is_sparse_field else 0.5,
            }
            scamp_config_ref = {
                **scamp_config_base,
                # Upper FWHM threshold: reject galaxies while allowing moderate
                # extension.  Sparse fields need a wider window (5x vs 3x) since
                # SExtractor FWHM can be inflated by blending/noise.
                "FWHM_THRESHOLDS": f"{0.3*fwhm_ref_pix:.2f},{(5 if is_sparse_field else 3)*fwhm_ref_pix:.2f}",
            }
            # SCAMP does not accept DISTORT_DEGREES=0 ("keyword out of range").
            # Use 1 as the minimum — degree 1 is a linear WCS (shift/rotation/scale)
            # which is always valid and far better than falling back to reproject.
            scamp_config_ref["DISTORT_DEGREES"] = max(1, distort_degrees)
            sparse_note = " (sparse)" if is_sparse_field else ""
            distort_label = max(1, distort_degrees)
            self.logger.info(
                'SCAMP: crossid=%.1f" maxerr=%.1f" distort=%d%s (ref SIP/PV order=%d, %d matched)',
                crossid_arcsec, position_maxerr_arcsec, distort_label, sparse_note,
                ref_distort_order, _num_matched,
            )

            # Allow YAML to override the automatic SWarp resampling choice for expert tuning.
            # templates.swarp_resampling_science / templates.swarp_resampling_reference can be
            # set to any SWarp-supported RESAMPLING_TYPE (e.g. 'BILINEAR', 'LANCZOS2', 'LANCZOS3').
            override_sci = templates_cfg.get("swarp_resampling_science")
            override_ref = templates_cfg.get("swarp_resampling_reference")
            if isinstance(override_sci, str) and override_sci.strip():
                self.logger.info(
                    "Overriding science SWarp RESAMPLING_TYPE from config: %s -> %s",
                    sci_resampling_method,
                    override_sci.strip().upper(),
                )
                sci_resampling_method = override_sci.strip().upper()
            if isinstance(override_ref, str) and override_ref.strip():
                self.logger.info(
                    "Overriding reference SWarp RESAMPLING_TYPE from config: %s -> %s",
                    ref_resampling_method,
                    override_ref.strip().upper(),
                )
                ref_resampling_method = override_ref.strip().upper()

            swarp_config = {
                "CENTER_TYPE": "MANUAL",
                "CENTER": f"{center_ra:.8f},{center_dec:.8f}",
                "PIXELSCALE_TYPE": "MANUAL",
                "PIXEL_SCALE": pix_scale,
                "IMAGE_SIZE": f"{output_width},{output_height}",
                "RESAMPLING_TYPE": sci_resampling_method,
                # OVERSAMPLING>0 causes SWarp to compute grid extents at N× sub-pixel
                # resolution internally then round back, producing off-by-one shape
                # mismatches when two images are resampled onto the same grid.
                # OVERSAMPLING=0 disables this entirely.
                "OVERSAMPLING": 0,
            }

            self.logger.info(
                "SWarp: sci=%s ref=%s",
                sci_resampling_method, ref_resampling_method,
            )

            self.logger.info(
                "SWarp grid: center=(%.4f, %.4f) size=%dx%d scale=%.3f\"/px",
                center_ra, center_dec, output_width, output_height, pix_scale,
            )

            # Run SCAMP on reference catalog only, using science catalog as reference.
            # Science image is distortion-corrected first (no SCAMP for science).
            # Reference image is aligned to science using SCAMP, then .head file is used for SWarp.
            # Science .head is NOT used for alignment (only reference .head is included).

            ref_cat_tmp_stem = None  # will be set below if SCAMP runs with temp catalog

            if reference_already_scamp:
                self.logger.info(
                    "Reference header has SCAMP HISTORY; skipping SCAMP, using existing WCS."
                )
                scamp_result = {}
            else:
                # Run SCAMP on reference catalog only, using science catalog as reference.
                # The reference .head corrects the reference WCS to match the science WCS.
                # Only the reference .head is placed next to the reference image before SWarp.
                # Use the backup paths (full catalogs) for SCAMP
                ref_cat_path = Path(ref_catalog_scamp_backup)
                sci_cat_path = Path(sci_catalog_scamp_backup)
                ref_cat_tmp = reference_aligned_dir / f"{ref_cat_path.stem}_ref.cat"
                ref_cat_tmp_stem = ref_cat_tmp.stem

                try:
                    # Create temporary reference catalog with distinct FILTER value
                    # so SCAMP produces a .head file for it.
                    with fits.open(ref_cat_path, memmap=False) as hdul:
                        if len(hdul) > 1 and "FILTER" in hdul[1].header:
                            hdul[1].header["FILTER"] = "w_ref"
                        elif len(hdul) > 0 and "FILTER" in hdul[0].header:
                            hdul[0].header["FILTER"] = "w_ref"
                        hdul.writeto(ref_cat_tmp, overwrite=True)
                    self.logger.debug(
                        "Created temporary reference catalog with FILTER=w_ref: %s", ref_cat_tmp
                    )

                    self.logger.info(
                        "Running SCAMP on reference catalog (science catalog as astrometric anchor)..."
                    )
                    self.logger.info(
                        "Science catalog: %d sources (anchor), Reference catalog: %d sources (to be aligned)",
                        len(fits.getdata(sci_catalog_scamp_backup, ext=2)),
                        len(fits.getdata(ref_catalog_scamp_backup, ext=2)),
                    )
                    scamp_result = self.run_scamp(
                        str(ref_cat_tmp),
                        reference_cat=sci_catalog_scamp_backup,
                        output_dir=str(reference_aligned_dir),
                        config=scamp_config_ref,
                    )

                except Exception as e:
                    self.logger.error("Failed to create temporary catalog for SCAMP: %s", e)
                    self.logger.info("Falling back to single-catalog SCAMP on reference...")
                    scamp_result = self.run_scamp(
                        ref_catalog_scamp_backup,
                        reference_cat=sci_catalog_scamp_backup,
                        output_dir=str(reference_aligned_dir),
                        config=scamp_config_ref,
                    )

                if scamp_result is None:
                    # Retry SCAMP with degree 1 and larger cross-match radius
                    # before giving up on SCAMP+SWarp entirely.  SCAMP can fail
                    # due to too-strict matching parameters even when enough
                    # sources exist for a linear WCS correction.
                    _retry_config = dict(scamp_config_ref)
                    _retry_config["DISTORT_DEGREES"] = 1
                    _retry_config["CROSSID_RADIUS"] = max(float(crossid_arcsec), 5.0)
                    _retry_config["POSITION_MAXERR"] = max(float(position_maxerr_arcsec), 2.0)
                    self.logger.warning(
                        "SCAMP failed. Retrying with DISTORT_DEGREES=1, "
                        "CROSSID_RADIUS=%.1f\", POSITION_MAXERR=%.1f\".",
                        float(_retry_config["CROSSID_RADIUS"]),
                        float(_retry_config["POSITION_MAXERR"]),
                    )
                    try:
                        scamp_result = self.run_scamp(
                            str(ref_cat_tmp),
                            reference_cat=sci_catalog_scamp_backup,
                            output_dir=str(reference_aligned_dir),
                            config=_retry_config,
                        )
                    except Exception as retry_err:
                        self.logger.warning("SCAMP retry also failed: %s", retry_err)
                        scamp_result = None

                if scamp_result is None:
                    self.logger.warning(
                        "SCAMP failed (both initial and retry). "
                        "Falling back to reproject/AstroAlign."
                    )
                    try:
                        ref_cat_tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

                # Clean up temporary reference catalog now that SCAMP is done
                try:
                    ref_cat_tmp.unlink(missing_ok=True)
                except Exception:
                    pass

                # Quality gate: check SCAMP astrometric residual
                scamp_distortion = scamp_result.get("distortion") if isinstance(scamp_result, dict) else None
                max_scamp_residual = float(align_cfg.get("alignment_max_scamp_residual_arcsec", 2.0))
                if isinstance(scamp_distortion, dict):
                    scamp_rms = scamp_distortion.get("astrometric_rms_arcsec")
                    scamp_nstars = scamp_distortion.get("n_matched_stars")
                    if scamp_rms is not None and scamp_rms > max_scamp_residual:
                        self.logger.warning(
                            "SCAMP astrometric residual large (%.3f\" > %.1f\"). "
                            "Proceeding with SWarp — post-SWarp verification gate will decide.",
                            scamp_rms, max_scamp_residual,
                        )
                    if scamp_nstars is not None and scamp_nstars < min_matched:
                        self.logger.warning(
                            "SCAMP matched only %d stars (< %d minimum). "
                            "Proceeding with SWarp — post-SWarp verification gate will decide.",
                            scamp_nstars, min_matched,
                        )
                    if scamp_rms is None and scamp_nstars is None:
                        self.logger.warning(
                            "SCAMP XML parsing failed — quality gate bypassed. "
                            "Post-SWarp verification will be the only alignment check."
                        )
                else:
                    self.logger.warning(
                        "SCAMP distortion info unavailable — quality gate bypassed. "
                        "Post-SWarp verification will be the only alignment check."
                    )

            # Copy reference .head file next to reference image only.
            # Reference .head corrects reference WCS to match science WCS.
            # Both images are resampled onto the same fixed grid (CENTER/IMAGE_SIZE/PIXEL_SCALE).
            head_by_stem = (
                scamp_result.get("head_files_by_stem", {})
                if isinstance(scamp_result, dict)
                else {}
            )
            self.logger.debug(
                "SCAMP produced .head files for stems: %s", list(head_by_stem.keys())
            )

            # Only copy reference .head file
            label = "reference"
            cat_tmp_stem = ref_cat_tmp_stem
            orig_cat_path = ref_sex["catalog_path"]
            fits_copy = ref_image_copy

            # Try temp-catalog stem first (most common path), then original stem
            head_src = None
            if cat_tmp_stem:
                head_src = head_by_stem.get(cat_tmp_stem)
            if not head_src:
                orig_stem = Path(orig_cat_path).stem
                head_src = head_by_stem.get(orig_stem)
            # Fallback: single .head key (non-temp-catalog SCAMP path)
            if not head_src:
                head_src = scamp_result.get("head_file") if isinstance(scamp_result, dict) else None

            if head_src and Path(head_src).exists():
                head_dst = fits_copy.with_suffix(".head")
                if Path(head_src).resolve() != head_dst.resolve():
                    try:
                        shutil.copy2(head_src, head_dst)
                        self.logger.info("Copied .head for %s SWarp", label)
                    except Exception as e:
                        log_warning_from_exception(
                            self.logger, f"Could not copy .head for {label}", e
                        )
                # Normalize CTYPE codes so SWarp recognizes PV/SIP distortion.
                if _normalize_head_file(head_dst):
                    self.logger.info("Normalized .head projection codes for %s SWarp", label)
                # SIP preservation experiment: disabled. SCAMP produces PV (TPV)
                # distortion keywords, and adding the reference's SIP keywords on
                # top creates a double distortion correction (both PV and SIP
                # applied simultaneously). SCAMP at the feasible degree already
                # captures the dominant distortion well (0.351px centroid residual).
                # if preserve_sip_in_head:
                #     self._preserve_reference_sip_in_head(
                #         head_dst, ref_image_copy
                #     )
            else:
                self.logger.warning(
                    "No SCAMP .head found for %s image (searched stems: %s, available: %s)",
                    label, [cat_tmp_stem, Path(orig_cat_path).stem], list(head_by_stem.keys()),
                )

            resample_dir = science_aligned_dir / "resampled_output"
            resample_dir.mkdir(parents=True, exist_ok=True)

            # Pre-SWarp precaution: warn if pixel scale ratio is large.
            # When the reference has a much finer pixel scale than the science,
            # resampling onto the coarser science grid degrades the reference PSF.
            # This is unavoidable (both images must share the same grid), but the
            # warning helps diagnose PSF mismatch issues in SFFT.
            if sci_pix_scale > 0 and ref_pix_scale > 0:
                _ps_ratio = max(sci_pix_scale, ref_pix_scale) / min(sci_pix_scale, ref_pix_scale)
                if _ps_ratio > 1.5:
                    self.logger.warning(
                        "Large pixel scale ratio (%.2fx): sci=%.4f\"/px ref=%.4f\"/px. "
                        "Resampling to the coarser grid will degrade the PSF of the "
                        "finer-scale image. SFFT kernel should compensate, but "
                        "subtraction quality may be reduced.",
                        _ps_ratio, sci_pix_scale, ref_pix_scale,
                    )

            # Run SWarp with COMBINE=Y (one image per call) to guarantee full IMAGE_SIZE output.
            # With COMBINE=N, SWarp clips each .resamp.fits to the image's sky footprint.
            # Science has no .head file (it defines the output grid via its WCS + CRPIX at center).
            # Reference has its SCAMP .head placed next to it so SWarp applies WCS correction.
            # Determine combined resampling method for simultaneous SWarp
            # Use the more conservative (lower quality) of the two images to preserve PSFs
            sci_undersampled = (
                sci_is_undersampled if sci_is_undersampled is not None else False
            )
            ref_undersampled = (
                ref_is_undersampled if ref_is_undersampled is not None else False
            )
            # Use header-based FWHM for threshold checks — SExtractor FWHM can be
            # inflated by extended sources, causing undersampled images to be
            # treated as well-sampled (see BUG 92).
            sci_fwhm_resample = (
                sci_hdr_fwhm if (sci_hdr_fwhm and sci_hdr_fwhm > 0) else fwhm_sci_pix
            )
            ref_fwhm_resample = (
                ref_hdr_fwhm if (ref_hdr_fwhm and ref_hdr_fwhm > 0) else fwhm_ref_pix
            )
            sci_fwhm_valid = sci_fwhm_resample is not None and np.isfinite(sci_fwhm_resample)
            ref_fwhm_valid = ref_fwhm_resample is not None and np.isfinite(ref_fwhm_resample)

            # Use three-tier strategy for combined resampling
            # Default to most conservative method if either image needs it
            if sci_fwhm_valid and sci_fwhm_resample < 1.5:
                combined_resampling_method = "BILINEAR"
                reason = f"science FWHM < 1.5 px ({sci_fwhm_resample:.2f})"
            elif ref_fwhm_valid and ref_fwhm_resample < 1.5:
                combined_resampling_method = "BILINEAR"
                reason = f"reference FWHM < 1.5 px ({ref_fwhm_resample:.2f})"
            elif sci_undersampled or ref_undersampled:
                combined_resampling_method = "LANCZOS2"
                reason = "undersampled image(s) present"
            elif sci_fwhm_valid and sci_fwhm_resample < undersampled_thresh:
                combined_resampling_method = "LANCZOS2"
                reason = f"science FWHM < {undersampled_thresh} px"
            elif ref_fwhm_valid and ref_fwhm_resample < undersampled_thresh:
                combined_resampling_method = "LANCZOS2"
                reason = f"reference FWHM < {undersampled_thresh} px"
            else:
                combined_resampling_method = "LANCZOS3"
                reason = "both images well-sampled"

            swarp_config_combined = {
                **swarp_config,
                "RESAMPLING_TYPE": combined_resampling_method,
            }

            self.logger.info(
                "SWarp resampling: %s (%s)", combined_resampling_method, reason
            )

            # Run SWarp separately on each image with COMBINE=Y.
            # With COMBINE=N, SWarp clips each .resamp.fits to the input image sky footprint,
            # ignoring IMAGE_SIZE. With COMBINE=Y (single input per call), SWarp writes the
            # co-added output at exactly IMAGE_SIZE on the requested grid. This is the only
            # way to guarantee both outputs have the exact same shape.
            swarp_config_each = {
                **swarp_config_combined,
                "COMBINE": "Y",  # Force full-size output at IMAGE_SIZE
                "COMBINE_TYPE": "MEDIAN",
            }

            if resample_mode == "wcs_only":
                # Skip SWarp resampling - apply SCAMP WCS correction to header only
                self.logger.info("Skipping SWarp resampling (WCS-only alignment for template subtraction)")

                # Copy SCAMP-corrected reference image to aligned location
                aligned_ref = output_dir / f"aligned_ref_{Path(reference_image).stem}.fits"
                # Science image is never modified — use the copy as-is
                aligned_sci = Path(sci_image_for_swarp)

                # Copy reference image, then apply SCAMP .head WCS to its header
                shutil.copy2(ref_image_copy, aligned_ref)

                head_file = Path(ref_image_copy).with_suffix(".head")
                if head_file.exists():
                    self.logger.info("Applying SCAMP .head WCS to reference image header")
                    try:
                        from wcs import _normalize_projection_codes
                        scamp_header = fits.Header.fromtextfile(str(head_file))
                        scamp_header = _normalize_projection_codes(scamp_header, inplace=False)
                        with fits.open(aligned_ref, mode="update", memmap=False) as hdul:
                            hdr = hdul[0].header
                            hdr = remove_wcs_from_header(hdr)
                            _wcs_prefixes = (
                                "CRPIX", "CRVAL", "CTYPE", "CD", "PC", "CDELT",
                                "CROTA", "PV", "LONPOLE", "LATPOLE", "EQUINOX",
                                "WCSNAME", "CUNIT", "WCSAXES", "PROJP", "LTV",
                                "LTM", "RADECSYS", "RADESYS", "RADYSYS",
                                "LONGPOLE", "TNX", "SIP_",
                            )
                            _wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
                            for key in scamp_header:
                                if key in ("NAXIS", "NAXIS1", "NAXIS2"):
                                    continue
                                is_wcs = any(key.startswith(p) for p in _wcs_prefixes)
                                if not is_wcs and "_" in key:
                                    stem = key.split("_")[0] + "_"
                                    is_wcs = stem in _wcs_stems and key.startswith(stem.rstrip("_"))
                                if is_wcs:
                                    hdr[key] = scamp_header[key]
                            hdr["NAXIS1"] = hdul[0].data.shape[1]
                            hdr["NAXIS2"] = hdul[0].data.shape[0]
                            hdul.flush()
                        self.logger.info("SCAMP WCS applied to reference image: %s", aligned_ref)
                    except Exception as e:
                        self.logger.warning(
                            "Failed to apply SCAMP .head to reference image: %s. "
                            "Reference retains original WCS.", e
                        )
                else:
                    self.logger.warning(
                        "No SCAMP .head file found at %s; reference retains original WCS.",
                        head_file,
                    )

                self.logger.info(f"WCS-only alignment complete: sci={aligned_sci}, ref={aligned_ref}")
                
            elif resample_mode == "native_scale":
                # Each image resampled independently keeping its native pixel scale,
                # but aligned to the same sky center. Output shapes differ if pixel
                # scales differ, but sky coverage is approximately matched.
                self.logger.info(
                    "Using native-scale resampling: science=%.4f\"/px ref=%.4f\"/px",
                    sci_pix_scale, ref_pix_scale,
                )
                
                # Compute reference IMAGE_SIZE to match science sky coverage
                ref_output_width = int(round(sci_shape[1] * sci_pix_scale / ref_pix_scale))
                ref_output_height = int(round(sci_shape[0] * sci_pix_scale / ref_pix_scale))
                
                self.logger.info(
                    "Native-scale shapes: science=%dx%d reference=%dx%d",
                    sci_shape[1], sci_shape[0], ref_output_width, ref_output_height,
                )
                
                resample_dir_sci = resample_dir / "sci"
                resample_dir_ref = resample_dir / "ref"
                resample_dir_sci.mkdir(parents=True, exist_ok=True)
                resample_dir_ref.mkdir(parents=True, exist_ok=True)
                
                # Science image: native pixel scale and original shape
                swarp_config_sci = {
                    **swarp_config_combined,
                    "COMBINE": "Y",
                    "COMBINE_TYPE": "MEDIAN",
                    "PIXEL_SCALE": sci_pix_scale,
                    "IMAGE_SIZE": f"{sci_shape[1]},{sci_shape[0]}",
                }
                
                # Reference image: native pixel scale, shape scaled to match sky coverage
                # Use ref_resampling_method (not combined) since science is not resampled
                # in native_scale mode — the reference should use its own optimal method.
                swarp_config_ref = {
                    **swarp_config,
                    "RESAMPLING_TYPE": ref_resampling_method,
                    "COMBINE": "Y",
                    "COMBINE_TYPE": "MEDIAN",
                    "PIXEL_SCALE": ref_pix_scale,
                    "IMAGE_SIZE": f"{ref_output_width},{ref_output_height}",
                }
                
                # Never resample the science image — it defines the target grid.
                # SWarp resampling degrades the PSF and can introduce sub-pixel
                # shifts that produce dipoles in the subtracted image.
                aligned_sci = Path(sci_image_for_swarp)
                swarp_res_sci = {"corrected_image": str(aligned_sci)}
                self.logger.info("Native-scale: science image used as-is (no SWarp)")

                self.logger.debug("Running SWarp on reference image (native scale)...")
                swarp_res_ref = self.run_swarp(
                    [str(ref_image_copy)],
                    scamp_results=None,
                    output_dir=str(resample_dir_ref),
                    config=swarp_config_ref,
                    no_weight_maps=False,
                )

                if swarp_res_ref is None:
                    self.logger.info("SWarp failed. Falling back to AstroAlign.")
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

                aligned_ref = Path(swarp_res_ref["corrected_image"])
                
                if not aligned_sci.exists() or not aligned_ref.exists():
                    self.logger.info(
                        "Could not find SWarp output images. Falling back to AstroAlign."
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                    
            else:
                # "common_grid" (default): Both images resampled by SWarp onto
                # the same common grid (CENTER, PIXEL_SCALE, IMAGE_SIZE).
                # This ensures consistent resampling — both images go through
                # the same SWarp interpolation kernel, producing matched PSFs
                # for subtraction.  The science image's WCS defines the grid;
                # the reference's SCAMP .head corrects its WCS before SWarp.
                resample_dir_sci = resample_dir / "sci"
                resample_dir_ref = resample_dir / "ref"
                resample_dir_sci.mkdir(parents=True, exist_ok=True)
                resample_dir_ref.mkdir(parents=True, exist_ok=True)

                # Run SWarp on science image (no .head — uses its own WCS).
                # SWarp reads the full WCS (including SIP distortion) from the
                # FITS header and resamples onto the output TAN grid.
                self.logger.info("Common-grid: resampling science image with SWarp")
                swarp_res_sci = self.run_swarp(
                    [str(sci_image_for_swarp)],
                    scamp_results=None,
                    output_dir=str(resample_dir_sci),
                    config=swarp_config_each,
                    no_weight_maps=False,
                )
                if swarp_res_sci is None:
                    self.logger.warning(
                        "SWarp failed for science image. Falling back to AstroAlign."
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                aligned_sci = Path(swarp_res_sci["corrected_image"])

                # Run SWarp on reference image (SCAMP .head already placed
                # next to ref_image_copy by the .head copy logic above).
                self.logger.info("Common-grid: resampling reference image with SWarp")
                swarp_res_ref = self.run_swarp(
                    [str(ref_image_copy)],
                    scamp_results=None,
                    output_dir=str(resample_dir_ref),
                    config=swarp_config_each,
                    no_weight_maps=False,
                )
                if swarp_res_ref is None:
                    self.logger.warning(
                        "SWarp failed for reference image. Falling back to AstroAlign."
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                aligned_ref = Path(swarp_res_ref["corrected_image"])

                if not aligned_sci.exists() or not aligned_ref.exists():
                    self.logger.info(
                        "Could not find SWarp output images. Falling back to AstroAlign."
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

                try:
                    _sci_out = fits.getdata(aligned_sci)
                    _ref_out = fits.getdata(aligned_ref)
                    _sci_nan_frac = float(np.count_nonzero(~np.isfinite(_sci_out))) / _sci_out.size
                    _ref_nan_frac = float(np.count_nonzero(~np.isfinite(_ref_out))) / _ref_out.size
                    self.logger.info(
                        "SWarp output coverage: sci=%.1f%% ref=%.1f%%",
                        (1.0 - _sci_nan_frac) * 100,
                        (1.0 - _ref_nan_frac) * 100,
                    )
                except Exception as _diag_e:
                    self.logger.debug("Post-SWarp coverage diagnostic failed: %s", _diag_e)

            # Copy to the expected resampled_dir location for downstream logic
            resampled_dir = resample_dir
            sci_target = resampled_dir / "science_image.resamp.fits"
            ref_target = resampled_dir / "reference_image.resamp.fits"
            shutil.copy2(aligned_sci, sci_target)
            shutil.copy2(aligned_ref, ref_target)
            aligned_sci = sci_target
            aligned_ref = ref_target

            # Diagnostic: measure post-SWarp alignment residual via centroid
            # cross-match. Log-only — no pixel-level corrections are applied;
            # SCAMP+SWarp is trusted as the definitive alignment.
            #
            # BUG 97: Pixel-space centroid matching is only valid for common_grid
            # mode where both images share the same pixel grid. For wcs_only and
            # native_scale, images are on different pixel grids — pixel coordinates
            # don't correspond, so matching produces meaningless offsets.
            alignment_metadata = {}
            _post_swarp_verify = bool(
                align_cfg.get("post_swarp_verify", False)
            )
            if not _post_swarp_verify:
                self.logger.info(
                    "Post-SWarp verification disabled (post_swarp_verify=False); "
                    "skipping SExtractor cross-match."
                )
            elif resample_mode in ("wcs_only", "native_scale"):
                self.logger.info(
                    "Skipping pixel-space alignment verification for resample_mode=%s "
                    "(images on different pixel grids).",
                    resample_mode,
                )
            else:
              try:
                self.logger.info("Verifying post-SWarp alignment via SExtractor cross-match...")
                _verify_dir = str(Path(output_dir) / "post_swarp_verify")
                _det_fwhm = min(max(max(fwhm_sci_pix, fwhm_ref_pix), 2.5), 4.0)

                # Sparse-field retry: if the first SExtractor pass detects
                # very few sources, retry with an even lower detection
                # threshold.  Sparse fields (e.g. high-Galactic-latitude) can
                # have only 3-10 detectable sources; the standard alignment
                # config (DETECT_THRESH=0.8) may still miss the faintest ones.
                _SPARSE_RETRY_THRESH = 5
                _SPARSE_SEX_OVERRIDE = {
                    "DETECT_THRESH": 0.5,
                    "ANALYSIS_THRESH": 0.3,
                    "DETECT_MINAREA": 1,
                    "BACK_SIZE": 16,
                }

                sci_sex_result = self.run_sextractor(
                    str(aligned_sci),
                    output_dir=str(Path(_verify_dir) / "sci"),
                    for_alignment=True,
                    fwhm_pixels=_det_fwhm,
                )
                ref_sex_result = self.run_sextractor(
                    str(aligned_ref),
                    output_dir=str(Path(_verify_dir) / "ref"),
                    for_alignment=True,
                    fwhm_pixels=_det_fwhm,
                )

                sci_cat_verify = sci_sex_result.get("catalog")
                ref_cat_verify = ref_sex_result.get("catalog")
                n_sci_det = len(sci_cat_verify) if sci_cat_verify is not None else 0
                n_ref_det = len(ref_cat_verify) if ref_cat_verify is not None else 0

                # Sparse-field retry with even lower threshold
                if n_sci_det < _SPARSE_RETRY_THRESH or n_ref_det < _SPARSE_RETRY_THRESH:
                    self.logger.info(
                        "Sparse field detected (%d sci / %d ref sources); "
                        "retrying SExtractor with DETECT_THRESH=0.5",
                        n_sci_det, n_ref_det,
                    )
                    if n_sci_det < _SPARSE_RETRY_THRESH:
                        sci_sex_result = self.run_sextractor(
                            str(aligned_sci),
                            output_dir=str(Path(_verify_dir) / "sci_sparse"),
                            for_alignment=True,
                            fwhm_pixels=_det_fwhm,
                            config=_SPARSE_SEX_OVERRIDE,
                        )
                        sci_cat_verify = sci_sex_result.get("catalog")
                        n_sci_det = len(sci_cat_verify) if sci_cat_verify is not None else 0
                    if n_ref_det < _SPARSE_RETRY_THRESH:
                        ref_sex_result = self.run_sextractor(
                            str(aligned_ref),
                            output_dir=str(Path(_verify_dir) / "ref_sparse"),
                            for_alignment=True,
                            fwhm_pixels=_det_fwhm,
                            config=_SPARSE_SEX_OVERRIDE,
                        )
                        ref_cat_verify = ref_sex_result.get("catalog")
                        n_ref_det = len(ref_cat_verify) if ref_cat_verify is not None else 0

                self.logger.info(
                    "Alignment verification: detected %d sci / %d ref sources",
                    n_sci_det, n_ref_det,
                )

                # Sparse-field: allow verification with as few as 2 sources
                # on each side.  With 2 mutual matches we still get a
                # meaningful median offset.
                _min_det_for_verify = 2 if min(n_sci_det, n_ref_det) < 5 else 3

                if sci_cat_verify is not None and ref_cat_verify is not None and n_sci_det >= _min_det_for_verify and n_ref_det >= _min_det_for_verify:
                    from scipy.spatial import cKDTree
                    _x_col = "XWIN_IMAGE" if "XWIN_IMAGE" in sci_cat_verify.colnames else "X_IMAGE"
                    _y_col = "YWIN_IMAGE" if "YWIN_IMAGE" in sci_cat_verify.colnames else "Y_IMAGE"
                    sci_xy = np.column_stack([
                        np.asarray(sci_cat_verify[_x_col], float),
                        np.asarray(sci_cat_verify[_y_col], float),
                    ])
                    ref_xy = np.column_stack([
                        np.asarray(ref_cat_verify[_x_col], float),
                        np.asarray(ref_cat_verify[_y_col], float),
                    ])

                    # Mutual nearest-neighbor matching: a sci source
                    # matches a ref source only if each is the other's
                    # closest neighbor.  This eliminates false matches
                    # that artificially lower the measured offset.
                    tree_ref = cKDTree(ref_xy)
                    tree_sci = cKDTree(sci_xy)

                    d_sr, i_sr = tree_ref.query(sci_xy, k=1)
                    d_rs, i_rs = tree_sci.query(ref_xy, k=1)

                    idx_s = np.arange(len(sci_xy), dtype=int)
                    mutual = (i_rs[i_sr] == idx_s) & np.isfinite(d_sr)

                    # Adaptive match tolerance: in sparse fields the
                    # SWarp resampling kernel can shift centroids by
                    # up to ~1 px.  Use 2*FWHM normally, but widen to
                    # 3*FWHM when very few sources are available.
                    _n_total = len(sci_xy) + len(ref_xy)
                    if _n_total < 10:
                        match_tol = 3.0 * _det_fwhm
                    else:
                        match_tol = 2.0 * _det_fwhm
                    good = mutual & (d_sr < match_tol)
                    n_matched_verify = int(good.sum())

                    # Sparse-field: accept 2 matches (median of 2 is
                    # still a robust estimator for a constant offset).
                    _min_matches = 2 if _n_total < 10 else 3

                    if n_matched_verify >= _min_matches:
                        dx = sci_xy[good, 0] - ref_xy[i_sr[good], 0]
                        dy = sci_xy[good, 1] - ref_xy[i_sr[good], 1]

                        # Skip sigma clipping for small samples —
                        # MAD-based sigma estimates are unstable for
                        # N < 8 and can discard valid matches.
                        if n_matched_verify >= 8:
                            from astropy.stats import sigma_clip as _sc_verify
                            dx_clipped = _sc_verify(dx, sigma=2.5, maxiters=3)
                            dy_clipped = _sc_verify(dy, sigma=2.5, maxiters=3)
                            both_ok = ~dx_clipped.mask & ~dy_clipped.mask
                            if np.sum(both_ok) >= _min_matches:
                                _dx_use = dx[both_ok]
                                _dy_use = dy[both_ok]
                            else:
                                _dx_use = dx
                                _dy_use = dy
                        else:
                            _dx_use = dx
                            _dy_use = dy

                        med_dx = float(np.median(_dx_use))
                        med_dy = float(np.median(_dy_use))
                        rms_dx = float(np.std(_dx_use))
                        rms_dy = float(np.std(_dy_use))
                        _indiv_offsets = np.sqrt(_dx_use**2 + _dy_use**2)
                        _p95_offset = float(np.percentile(_indiv_offsets, 95))
                        _max_offset = float(np.max(_indiv_offsets))

                        total_offset = np.sqrt(med_dx**2 + med_dy**2)
                        self.logger.info(
                            "Alignment verification: offset=(%.3f, %.3f) px, "
                            "RMS=(%.3f, %.3f) px, total=%.3f px, "
                            "P95=%.3f px, max=%.3f px (%d matches)",
                            med_dx, med_dy, rms_dx, rms_dy, total_offset,
                            _p95_offset, _max_offset, n_matched_verify,
                        )

                        alignment_metadata = {
                            "offset_x": med_dx,
                            "offset_y": med_dy,
                            "rms_x": rms_dx,
                            "rms_y": rms_dy,
                            "n_matched": n_matched_verify,
                            "p95_offset": _p95_offset,
                            "max_offset": _max_offset,
                        }

                    else:
                        self.logger.info(
                            "Alignment verification: only %d matches (< %d); skipping.",
                            n_matched_verify, _min_matches,
                        )
                else:
                    self.logger.info(
                        "Alignment verification: SExtractor detected insufficient sources "
                        "(%d sci / %d ref; need >= %d each).",
                        n_sci_det, n_ref_det, _min_det_for_verify,
                    )
              except Exception as e:
                self.logger.warning("Alignment verification failed (non-fatal): %s", e)
                alignment_metadata = {}

            # Post-SWarp quality gate: reject alignment if systematic offset
            # or RMS is too large.  A small median offset with large RMS
            # indicates widespread misalignment (e.g., bad SCAMP solution or
            # WCS distortion mismatch) that the median hides.
            #
            max_acceptable_offset = float(
                align_cfg.get("alignment_max_offset_px", 0.5)
            )
            max_acceptable_rms = float(
                align_cfg.get("alignment_max_rms_px", 0.75)
            )
            max_acceptable_p95 = float(
                align_cfg.get("alignment_max_p95_px", 1.5)
            )
            _min_n_for_percentile = int(
                align_cfg.get("alignment_min_sources_for_field_gate", 20)
            )
            # Scale ALL thresholds by FWHM: centroid uncertainty is
            # proportional to PSF size.  Reference FWHM = 3 px (typical good
            # seeing); clamp scaling to [0.5, 3.0] to avoid extremes.
            _gate_fwhm = max(fwhm_sci_pix, fwhm_ref_pix)
            _fwhm_scale = max(0.5, min(3.0, _gate_fwhm / 3.0))
            max_acceptable_offset *= _fwhm_scale
            max_acceptable_rms *= _fwhm_scale
            max_acceptable_p95 *= _fwhm_scale
            if alignment_metadata and "offset_x" in alignment_metadata:
                _off = np.sqrt(
                    alignment_metadata["offset_x"] ** 2
                    + alignment_metadata["offset_y"] ** 2
                )
                _rms = np.sqrt(
                    alignment_metadata.get("rms_x", 0) ** 2
                    + alignment_metadata.get("rms_y", 0) ** 2
                )
                _p95 = alignment_metadata.get("p95_offset", 0.0)
                _n_match = alignment_metadata.get("n_matched", 0)

                # Sparse-field: use adaptive minimum for offset gate.
                # With 2 matches, only check offset (RMS/P95 are meaningless
                # for N < _min_n_for_percentile anyway).
                _gate_min_matches = 2 if _n_match < 8 else 3
                _reject = _n_match >= _gate_min_matches and _off > max_acceptable_offset
                if _n_match >= _min_n_for_percentile:
                    _reject = _reject or _rms > max_acceptable_rms or _p95 > max_acceptable_p95

                if _reject:
                    _reasons = []
                    if _off > max_acceptable_offset:
                        _reasons.append("offset=%.2f px (> %.2f px)" % (_off, max_acceptable_offset))
                    if _n_match >= _min_n_for_percentile:
                        if _rms > max_acceptable_rms:
                            _reasons.append("RMS=%.2f px (> %.2f px)" % (_rms, max_acceptable_rms))
                        if _p95 > max_acceptable_p95:
                            _reasons.append("P95=%.2f px (> %.2f px)" % (_p95, max_acceptable_p95))
                    self.logger.warning(
                        "Post-SWarp alignment rejected: %s (%d matches, FWHM=%.1f px). "
                        "Falling back to reproject/AstroAlign.",
                        "; ".join(_reasons) if _reasons else "unknown", _n_match, _gate_fwhm,
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                else:
                    self.logger.info(
                        "Post-SWarp alignment accepted: offset=%.2f px, RMS=%.2f px, "
                        "P95=%.2f px (%d matches, FWHM=%.1f px).",
                        _off, _rms, _p95, _n_match, _gate_fwhm,
                    )
            elif _post_swarp_verify and resample_mode == "common_grid" and bool(
                align_cfg.get("alignment_require_post_swarp_verification", True)
            ):
                # Sparse-field fallback: when post-SWarp verification cannot
                # measure offsets (too few sources detected/matched), check
                # whether SCAMP itself reported good astrometric residuals.
                # If SCAMP succeeded with a reasonable RMS and matched enough
                # stars, the SCAMP+SWarp alignment is trustworthy even without
                # independent post-SWarp source matching.  Only fall back to
                # reproject/AstroAlign if SCAMP residuals were also poor or
                # unavailable.
                _scamp_distortion = (
                    scamp_result.get("distortion")
                    if isinstance(scamp_result, dict)
                    else None
                )
                _scamp_rms_val = None
                _scamp_nstars_val = None
                if isinstance(_scamp_distortion, dict):
                    _scamp_rms_val = _scamp_distortion.get("astrometric_rms_arcsec")
                    _scamp_nstars_val = _scamp_distortion.get("n_matched_stars")

                _scamp_max_rms = float(
                    align_cfg.get("alignment_max_scamp_residual_arcsec", 2.0)
                )
                # Convert SCAMP arcsec residual to pixels for comparison
                _pixel_scale_arcsec = abs(
                    float(fits.getheader(str(aligned_sci)).get("CDELT2", 0))
                ) * 3600.0
                if _pixel_scale_arcsec > 0 and _scamp_rms_val is not None:
                    _scamp_rms_pix = _scamp_rms_val / _pixel_scale_arcsec
                else:
                    _scamp_rms_pix = None

                _scamp_trustworthy = (
                    _scamp_rms_val is not None
                    and _scamp_rms_val <= _scamp_max_rms
                    and _scamp_nstars_val is not None
                    and _scamp_nstars_val >= 3
                )

                if _scamp_trustworthy:
                    self.logger.info(
                        "Post-SWarp verification could not measure offsets (sparse field), "
                        "but SCAMP residuals are good (RMS=%.3f\" / %.2f px, %d stars). "
                        "Accepting SCAMP+SWarp alignment.",
                        _scamp_rms_val, _scamp_rms_pix or -1, _scamp_nstars_val,
                    )
                else:
                    self.logger.warning(
                        "Post-SWarp alignment could not be measured (sparse field) "
                        "and SCAMP residuals are poor or unavailable "
                        "(SCAMP RMS=%s, stars=%s). Falling back to reproject/AstroAlign.",
                        _scamp_rms_val, _scamp_nstars_val,
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

            # Log SWarp output WCS for both images so alignment can be verified.
            # Since CENTER = overlap sky midpoint and IMAGE_SIZE = overlap region size,
            # both outputs should have identical shapes.
            sci_wcs_info = None
            ref_wcs_info = None
            for _label, _path in [("science", aligned_sci), ("reference", aligned_ref)]:
                try:
                    with fits.open(_path) as _hdul:
                        _hdr = _hdul[0].header
                        _ny, _nx = _hdul[0].data.shape
                    wcs_info = {
                        "label": _label,
                        "shape": (_ny, _nx),
                        "crpix1": float(_hdr.get("CRPIX1", 0)),
                        "crpix2": float(_hdr.get("CRPIX2", 0)),
                        "crval1": float(_hdr.get("CRVAL1", 0)),
                        "crval2": float(_hdr.get("CRVAL2", 0)),
                        "ctype1": _hdr.get("CTYPE1", "?"),
                        "ctype2": _hdr.get("CTYPE2", "?"),
                    }
                    if _label == "science":
                        sci_wcs_info = wcs_info
                    else:
                        ref_wcs_info = wcs_info
                    self.logger.info(
                        "SWarp output [%s]: shape=%dx%d CRPIX=(%.1f,%.1f) CTYPE=%s/%s",
                        _label, _ny, _nx,
                        wcs_info["crpix1"], wcs_info["crpix2"],
                        wcs_info["ctype1"], wcs_info["ctype2"],
                    )
                except Exception as _e:
                    self.logger.debug("Could not read WCS from SWarp output [%s]: %s", _label, _e)

            # Verify CRPIX consistency between science and reference outputs
            if sci_wcs_info and ref_wcs_info:
                if sci_wcs_info["shape"] != ref_wcs_info["shape"]:
                    self.logger.warning(
                        "SWarp shapes differ: sci=%s ref=%s",
                        sci_wcs_info["shape"], ref_wcs_info["shape"]
                    )
                crpix_diff_x = abs(sci_wcs_info["crpix1"] - ref_wcs_info["crpix1"])
                crpix_diff_y = abs(sci_wcs_info["crpix2"] - ref_wcs_info["crpix2"])
                if crpix_diff_x > 0.01 or crpix_diff_y > 0.01:
                    self.logger.warning(
                        "CRPIX mismatch: sci=(%.1f,%.1f) ref=(%.1f,%.1f)",
                        sci_wcs_info["crpix1"], sci_wcs_info["crpix2"],
                        ref_wcs_info["crpix1"], ref_wcs_info["crpix2"]
                    )
                else:
                    self.logger.debug(
                        "CRPIX consistent: (%.1f,%.1f)",
                        sci_wcs_info["crpix1"], sci_wcs_info["crpix2"]
                    )

            # Verify output shapes match. Only relevant for common_grid mode where
            # both images are resampled to the same grid. For wcs_only and
            # native_scale, different shapes are expected and should not trigger
            # trim/pad or fallback.
            if resample_mode in ("wcs_only", "native_scale"):
                self.logger.debug(
                    "Skipping shape verification for resample_mode=%s", resample_mode
                )
            else:
                try:
                    with fits.open(aligned_sci) as _h:
                        _sci_data = _h[0].data
                        _sci_header = _h[0].header
                        _sci_shape = _sci_data.shape
                    with fits.open(aligned_ref) as _h:
                        _ref_data = _h[0].data
                        _ref_header = _h[0].header
                        _ref_wcs = get_wcs(_ref_header)
                        _ref_shape = _ref_data.shape

                    if _sci_shape != _ref_shape:
                        dy = abs(_sci_shape[0] - _ref_shape[0])
                        dx = abs(_sci_shape[1] - _ref_shape[1])
                        self.logger.warning(
                            "SWarp shape mismatch: sci=%s ref=%s (delta %d,%d)",
                            _sci_shape, _ref_shape, dy, dx,
                        )

                        # Large shape mismatch (>500 px) means SCAMP shifted the reference
                        # WCS enough that it no longer covers the output grid — padding with
                        # NaN would produce a spatially misaligned image.  Trigger fallback.
                        # Increased threshold to handle SWarp clipping when CRPIX is offset.
                        if dy > 500 or dx > 500:
                            self.logger.warning(
                                "Shape mismatch too large (%d,%d); falling back to reproject/AstroAlign.",
                                dy, dx,
                            )
                            return self._align_fallback_reproject_then_astroalign(
                                science_image, reference_image, output_dir
                            )

                        # Small mismatch (<=20 px): SWarp rounding artefact — safe to trim/pad.
                        # Center the cutout on the same sky position used as the SWarp CENTER
                        # (center_ra, center_dec) projected into the reference pixel frame, so
                        # science and reference share the same celestial anchor point.
                        ny_target, nx_target = _sci_shape

                        from astropy.nddata.utils import Cutout2D

                        # Find the pixel position of the SWarp CENTER in the reference image
                        try:
                            ref_cx_arr, ref_cy_arr = _ref_wcs.all_world2pix(
                                [center_ra], [center_dec], 0
                            )
                            ref_cx = float(ref_cx_arr[0])
                            ref_cy = float(ref_cy_arr[0])
                            if not (np.isfinite(ref_cx) and np.isfinite(ref_cy)):
                                raise ValueError(f"non-finite ref center: ({ref_cx}, {ref_cy})")
                        except Exception as _ce:
                            self.logger.debug(
                                "Could not project SWarp CENTER into reference WCS (%s); using pixel center instead.", _ce
                            )
                            # Use correct numpy 0-based center for Cutout2D
                            ref_cx = (_ref_shape[1] - 1) / 2.0
                            ref_cy = (_ref_shape[0] - 1) / 2.0

                        ref_cutout = Cutout2D(
                            _ref_data,
                            (ref_cx, ref_cy),
                            (ny_target, nx_target),  # Cutout2D expects (height, width)
                            wcs=_ref_wcs,
                            mode='partial',
                        )
                        _ref_data_adjusted = ref_cutout.data
                        from functions import update_header_from_wcs
                        _ref_header = _ref_header.copy()
                        update_header_from_wcs(_ref_header, ref_cutout.wcs)
                        _ref_header['NAXIS1'] = nx_target
                        _ref_header['NAXIS2'] = ny_target

                        # Synchronize CRPIX and CRVAL between science and reference so both
                        # images share the same celestial anchor at the image center.
                        # CRPIX is FITS 1-based, so the center is at (nx+1)/2, (ny+1)/2.
                        crpix1_center = (nx_target + 1) / 2.0
                        crpix2_center = (ny_target + 1) / 2.0
                        crval1 = _sci_header.get('CRVAL1')
                        crval2 = _sci_header.get('CRVAL2')

                        _sci_header['CRPIX1'] = crpix1_center
                        _sci_header['CRPIX2'] = crpix2_center
                        _ref_header['CRPIX1'] = crpix1_center
                        _ref_header['CRPIX2'] = crpix2_center
                        _ref_header['CRVAL1'] = crval1
                        _ref_header['CRVAL2'] = crval2

                        fits.writeto(aligned_sci, _sci_data, _sci_header, overwrite=True)
                        fits.writeto(aligned_ref, _ref_data_adjusted, _ref_header, overwrite=True)

                        self.logger.info(
                            "Reference adjusted to science shape=%s (centered on SWarp sky CENTER, pad/trim <=20 px).",
                            (ny_target, nx_target),
                        )
                    else:
                        self.logger.info(
                            "SWarp outputs match: both images shape=%s.", _sci_shape,
                        )
                except Exception as _e:
                    self.logger.error(
                        "Could not verify or fix SWarp output shapes: %s. Falling back.", _e
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

            # Overwrite the reference_image path in-place with the aligned version.
            # reference_image is already a per-science-image copy so overwriting it
            # does not affect other science images using the same original template.
            aligned_science_fpath = science_image
            aligned_reference_fpath = reference_image

            # SWarp reads the SCAMP .head file next to the input FITS during resampling
            # and writes the correct aligned WCS into the .resamp.fits output.
            # No post-SWarp WCS injection is needed or correct: the SCAMP .head has
            # CRPIX/CRVAL from the original (pre-resample) image space, so injecting
            # it after SWarp would overwrite the valid resampled-grid WCS with
            # coordinates from the wrong pixel space.

            # Validate WCS of aligned images before overwriting originals
            # Test both pix2world and world2pix to ensure WCS is fully invertible
            try:
                with fits.open(aligned_ref) as hdul:
                    ref_wcs = get_wcs(hdul[0].header)
                    nx, ny = hdul[0].data.shape[1], hdul[0].data.shape[0]
                    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
                    ref_ra, ref_dec = ref_wcs.all_pix2world([cx], [cy], 0)
                    if not (np.isfinite(ref_ra[0]) and np.isfinite(ref_dec[0])):
                        self.logger.error(
                            f"SWarp-aligned reference has invalid WCS pix2world: RA={ref_ra[0]}, Dec={ref_dec[0]}. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
                    # Test world2pix (critical for Cutout2D)
                    test_px, test_py = _safe_world2pix(ref_wcs, ref_ra[0], ref_dec[0], 0)
                    test_px, test_py = float(np.atleast_1d(test_px)[0]), float(np.atleast_1d(test_py)[0])
                    if not (np.isfinite(test_px) and np.isfinite(test_py)):
                        self.logger.error(
                            f"SWarp-aligned reference has non-invertible WCS: world2pix returned NaN/Inf. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
            except Exception as e:
                self.logger.error(f"Failed to validate WCS of SWarp-aligned reference: {e}")
                return self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )

            try:
                with fits.open(aligned_sci) as hdul:
                    sci_wcs = get_wcs(hdul[0].header)
                    nx, ny = hdul[0].data.shape[1], hdul[0].data.shape[0]
                    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
                    sci_ra, sci_dec = sci_wcs.all_pix2world([cx], [cy], 0)
                    if not (np.isfinite(sci_ra[0]) and np.isfinite(sci_dec[0])):
                        self.logger.error(
                            f"SWarp-aligned science has invalid WCS pix2world: RA={sci_ra[0]}, Dec={sci_dec[0]}. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
                    # Test world2pix (critical for Cutout2D)
                    test_px, test_py = _safe_world2pix(sci_wcs, sci_ra[0], sci_dec[0], 0)
                    test_px, test_py = float(np.atleast_1d(test_px)[0]), float(np.atleast_1d(test_py)[0])
                    if not (np.isfinite(test_px) and np.isfinite(test_py)):
                        self.logger.error(
                            f"SWarp-aligned science has non-invertible WCS: world2pix returned NaN/Inf. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
            except Exception as e:
                self.logger.error(f"Failed to validate WCS of SWarp-aligned science: {e}")
                return self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )

            try:
                shutil.copyfile(aligned_ref, aligned_reference_fpath)
                self.logger.debug(
                    "Saved aligned reference image to: %s",
                    aligned_reference_fpath,
                )
            except Exception as e:
                self.logger.warning(
                    "Could not save aligned reference image %s: %s",
                    aligned_reference_fpath,
                    e,
                )

            try:
                shutil.copyfile(aligned_sci, aligned_science_fpath)
                self.logger.debug(
                    "Overwrote science image with aligned version: %s",
                    aligned_science_fpath,
                )
            except Exception as e:
                self.logger.warning(
                    "Could not overwrite science image %s: %s",
                    aligned_science_fpath,
                    e,
                )
            # Keep SIP/TPV distortion coefficients in header to preserve distortion correction
            # SWarp outputs TAN projection but the distortion model is still needed for accurate WCS
            # Removing SIP coefficients causes systematic WCS errors across the field
            # SWarp resampling does not inherently correct distortion - it just applies linear transforms
            self.logger.debug("Preserving SIP/TPV distortion coefficients in aligned science image header")

            # Remove the aligned working directory after copying aligned images over the originals.
            if science_aligned_dir.exists():
                shutil.rmtree(science_aligned_dir, ignore_errors=True)
                self.logger.debug(
                    "Removed aligned working dir: %s", science_aligned_dir
                )
            if (
                reference_aligned_dir.exists()
                and reference_aligned_dir != science_aligned_dir
            ):
                shutil.rmtree(reference_aligned_dir, ignore_errors=True)
                self.logger.debug(
                    "Removed aligned working dir: %s", reference_aligned_dir
                )

            return {
                "science_aligned": str(aligned_science_fpath),
                "reference_aligned": str(aligned_reference_fpath),
                "science_resampling_method": sci_resampling_method,
                "reference_resampling_method": ref_resampling_method,
                "science_undersampled": sci_is_undersampled,
                "reference_undersampled": ref_is_undersampled,
                "science_fwhm_pixels": fwhm_sci_pix,
                "reference_fwhm_pixels": fwhm_ref_pix,
                "alignment_method": "scamp_swarp",
                "alignment_verification": alignment_metadata,
            }

        except Exception as e:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            self.logger.error(
                f"Alignment failed: Type: {exc_type.__name__} File: {fname} Line: {lineno} Message: {e}"
            )
            return None

    # ------------------------------ Reproject (WCS-based) path ------------------------------

    def align_with_reproject(
        self,
        science_image: str,
        reference_image: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict]:
        """
        Align the reference image onto the science image pixel grid using WCS reprojection.
        Requires valid WCS in both FITS headers. Returns same dict as align_with_astroalign, or None on failure.

        Uses ReprojectConfig from input_yaml to select the reproject method (adaptive,
        interp, exact) with automatic fallback.  This respects user-configured options
        such as reproject_method, reproject_interp_order, and reproject_parallel.
        """
        if not HAS_REPROJECT:
            self.logger.debug(
                "reproject package not available; skipping reproject alignment."
            )
            return None
        try:
            output_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(science_image).parent
            )
            with fits.open(science_image, mode="readonly") as h_sci:
                sci_data = h_sci[0].data
                sci_header = h_sci[0].header
                if sci_data is None or sci_data.size == 0:
                    return None
                if sci_data.dtype.kind != 'f':
                    sci_data = sci_data.astype(np.float32)
            with fits.open(reference_image, mode="readonly") as h_ref:
                ref_data = h_ref[0].data
                ref_header = h_ref[0].header
                if ref_data is None or ref_data.size == 0:
                    return None
                if ref_data.dtype.kind != 'f':
                    ref_data = ref_data.astype(np.float32)
            sci_wcs = get_wcs(sci_header)
            ref_wcs = get_wcs(ref_header)
            if sci_wcs is None or ref_wcs is None:
                self.logger.debug(
                    "Reproject: missing WCS in science or reference header."
                )
                return None

            # Pre-mask NaN pixels in reference data
            n_nan = int(np.sum(~np.isfinite(ref_data)))
            if n_nan > 0:
                self.logger.info(
                    "Reference has %d NaN pixels (%.2f%%) — replacing with 0 before reprojection.",
                    n_nan, 100.0 * n_nan / ref_data.size,
                )
                ref_data = np.where(np.isfinite(ref_data), ref_data, 0.0)

            # Build reproject config from input_yaml (same as templates.py)
            iy = getattr(self, "input_yaml", None) or {}
            align_cfg = iy.get("alignment", {}) if isinstance(iy, dict) else {}
            quality_cfg = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
            req_method = str(align_cfg.get("reproject_method", "adaptive")).lower().strip()
            interp_order = str(align_cfg.get("reproject_interp_order", "bicubic")).lower().strip()
            use_parallel = bool(align_cfg.get("reproject_parallel", False))
            roundtrip = bool(align_cfg.get("reproject_roundtrip_coords", True))
            conserve_flux = bool(align_cfg.get("reproject_adaptive_conserve_flux", False))
            center_jacobian = bool(align_cfg.get("reproject_adaptive_center_jacobian", False))

            # Auto-enable center_jacobian for large rotation differences
            try:
                sci_rot = np.degrees(np.arctan2(
                    sci_wcs.wcs.cd[0, 1], sci_wcs.wcs.cd[0, 0]
                ))
                ref_rot = np.degrees(np.arctan2(
                    ref_wcs.wcs.cd[0, 1], ref_wcs.wcs.cd[0, 0]
                ))
                _rot_diff = abs(sci_rot - ref_rot)
                if _rot_diff > 180:
                    _rot_diff = 360 - _rot_diff
                if not center_jacobian and _rot_diff > 30.0:
                    center_jacobian = True
                    self.logger.info(
                        "Large rotation (%.1f deg) — enabling center_jacobian for "
                        "more accurate reproject_adaptive resampling.",
                        _rot_diff,
                    )
            except Exception:
                pass

            # Normalise interp_order string for reproject_interp
            _interp_map = {
                "nearest": "nearest-neighbor", "nearest-neighbor": "nearest-neighbor",
                "nn": "nearest-neighbor",
                "bilinear": "bilinear", "linear": "bilinear", "1": "bilinear",
                "biquadratic": "biquadratic", "2": "biquadratic",
                "bicubic": "bicubic", "cubic": "bicubic", "3": "bicubic",
            }
            interp_order_norm = _interp_map.get(interp_order, "bilinear")

            # Auto-downgrade interpolation for undersampled images
            _fwhm = float(iy.get("fwhm", 0.0)) if isinstance(iy, dict) else 0.0
            if _fwhm > 0 and _fwhm < 2.5 and interp_order_norm in ("bicubic", "biquadratic"):
                self.logger.info(
                    "Undersampled image (FWHM=%.2f px < 2.5) — downgrading %s to bilinear "
                    "to avoid ringing artifacts.",
                    _fwhm, interp_order_norm,
                )
                interp_order_norm = "bilinear"

            # Method fallback chain: exact-first for best geometric accuracy
            all_methods = ("exact", "adaptive", "interp")
            if req_method in all_methods:
                fallbacks = [req_method] + [m for m in all_methods if m != req_method]
            else:
                fallbacks = ["exact", "adaptive", "interp"]

            aligned_ref = None
            footprint = None
            used_method = None
            last_exc = None

            for m in fallbacks:
                try:
                    if m == "adaptive":
                        if reproject_adaptive is None:
                            continue
                        aligned_ref, footprint = reproject_adaptive(
                            (ref_data, ref_wcs),
                            sci_wcs,
                            shape_out=sci_data.shape,
                            roundtrip_coords=roundtrip,
                            parallel=use_parallel,
                            conserve_flux=conserve_flux,
                            center_jacobian=center_jacobian,
                            **({"despike_jacobian": True} if "despike_jacobian" in _ADAPTIVE_PARAMS else {}),
                        )
                    elif m == "interp":
                        aligned_ref, footprint = reproject_interp(
                            (ref_data, ref_wcs),
                            sci_wcs,
                            shape_out=sci_data.shape,
                            order=interp_order_norm,
                            roundtrip_coords=roundtrip,
                        )
                    else:  # exact
                        try:
                            from reproject import reproject_exact
                        except ImportError:
                            continue
                        aligned_ref, footprint = reproject_exact(
                            (ref_data, ref_wcs),
                            sci_wcs,
                            shape_out=sci_data.shape,
                            parallel=use_parallel,
                        )
                    used_method = m
                    break
                except Exception as _e:
                    last_exc = _e
                    self.logger.debug("Reproject (%s) failed: %s", m, _e)

            if used_method is None or aligned_ref is None or footprint is None:
                self.logger.info("Reproject alignment failed (all methods): %s", last_exc)
                return None

            # Footprint coverage check — zero overlap means WCS mismatch
            fp_mask = footprint.astype(bool)
            n_footprint = int(np.sum(fp_mask))
            n_total = int(fp_mask.size)
            if n_footprint == 0:
                self.logger.warning(
                    "Reproject has zero footprint coverage (%d/%d pixels). "
                    "Template and science WCS do not overlap.",
                    n_footprint, n_total,
                )
                return None
            if n_footprint < n_total * 0.5:
                self.logger.warning(
                    "Reproject footprint coverage is low: %d/%d pixels (%.1f%%). "
                    "Large unmasked borders may affect subtraction quality.",
                    n_footprint, n_total, 100.0 * n_footprint / n_total,
                )

            aligned_ref = np.asarray(aligned_ref, dtype=float)
            # Mask non-footprint pixels with NaN
            aligned_ref[~fp_mask] = np.nan

            out_header = ref_header.copy()
            out_header = remove_wcs_from_header(out_header)
            from functions import copy_wcs_from_header
            copy_wcs_from_header(sci_header, out_header)
            base_ref = Path(reference_image).stem
            ext_ref = Path(reference_image).suffix
            aligned_reference_fpath = output_dir / f"{base_ref}{ext_ref}"
            from functions import safe_fits_write
            safe_fits_write(str(aligned_reference_fpath), aligned_ref, out_header)
            self.logger.info(
                "Alignment via WCS reproject succeeded (method=%s, coverage=%.1f%%).",
                used_method, 100.0 * n_footprint / n_total,
            )

            # Post-reproject alignment verification via centroid cross-match.
            # Reproject trusts both WCS blindly — independently plate-solved
            # images can disagree by several pixels.  Log the residual so
            # downstream stages know the alignment quality.
            reproject_metadata = {}
            try:
                from scipy.spatial import cKDTree

                _det_fwhm = min(max(
                    float(self.input_yaml.get("fwhm", 3.0)), 2.5
                ), 4.0) if hasattr(self, "input_yaml") else 3.0

                _reproj_verify_dir = str(Path(output_dir) / "reproject_verify")
                _sci_sex = self.run_sextractor(
                    science_image,
                    output_dir=str(Path(_reproj_verify_dir) / "sci"),
                    for_alignment=True,
                    fwhm_pixels=_det_fwhm,
                )
                _ref_sex = self.run_sextractor(
                    str(aligned_reference_fpath),
                    output_dir=str(Path(_reproj_verify_dir) / "ref"),
                    for_alignment=True,
                    fwhm_pixels=_det_fwhm,
                )
                _sci_cat_v = _sci_sex.get("catalog")
                _ref_cat_v = _ref_sex.get("catalog")
                _n_sci_reproj = len(_sci_cat_v) if _sci_cat_v is not None else 0
                _n_ref_reproj = len(_ref_cat_v) if _ref_cat_v is not None else 0

                # Sparse-field retry with lower threshold
                if _n_sci_reproj < 5 or _n_ref_reproj < 5:
                    self.logger.info(
                        "Sparse field in reproject verify (%d sci / %d ref); "
                        "retrying SExtractor with DETECT_THRESH=0.5",
                        _n_sci_reproj, _n_ref_reproj,
                    )
                    _SPARSE_REPROJ_OVERRIDE = {
                        "DETECT_THRESH": 0.5,
                        "ANALYSIS_THRESH": 0.3,
                        "DETECT_MINAREA": 1,
                        "BACK_SIZE": 16,
                    }
                    if _n_sci_reproj < 5:
                        _sci_sex = self.run_sextractor(
                            science_image,
                            output_dir=str(Path(_reproj_verify_dir) / "sci_sparse"),
                            for_alignment=True,
                            fwhm_pixels=_det_fwhm,
                            config=_SPARSE_REPROJ_OVERRIDE,
                        )
                        _sci_cat_v = _sci_sex.get("catalog")
                        _n_sci_reproj = len(_sci_cat_v) if _sci_cat_v is not None else 0
                    if _n_ref_reproj < 5:
                        _ref_sex = self.run_sextractor(
                            str(aligned_reference_fpath),
                            output_dir=str(Path(_reproj_verify_dir) / "ref_sparse"),
                            for_alignment=True,
                            fwhm_pixels=_det_fwhm,
                            config=_SPARSE_REPROJ_OVERRIDE,
                        )
                        _ref_cat_v = _ref_sex.get("catalog")
                        _n_ref_reproj = len(_ref_cat_v) if _ref_cat_v is not None else 0

                _reproj_min_det = 2 if min(_n_sci_reproj, _n_ref_reproj) < 5 else 3
                if _sci_cat_v is not None and _ref_cat_v is not None and _n_sci_reproj >= _reproj_min_det and _n_ref_reproj >= _reproj_min_det:
                    _x_col = "XWIN_IMAGE" if "XWIN_IMAGE" in _sci_cat_v.colnames else "X_IMAGE"
                    _y_col = "YWIN_IMAGE" if "YWIN_IMAGE" in _sci_cat_v.colnames else "Y_IMAGE"
                    _sci_xy = np.column_stack([
                        np.asarray(_sci_cat_v[_x_col], float),
                        np.asarray(_sci_cat_v[_y_col], float),
                    ])
                    _ref_xy = np.column_stack([
                        np.asarray(_ref_cat_v[_x_col], float),
                        np.asarray(_ref_cat_v[_y_col], float),
                    ])

                    # Mutual nearest-neighbor matching for robust verification
                    _tree_ref = cKDTree(_ref_xy)
                    _tree_sci = cKDTree(_sci_xy)
                    _d_sr, _i_sr = _tree_ref.query(_sci_xy, k=1)
                    _d_rs, _i_rs = _tree_sci.query(_ref_xy, k=1)
                    _idx_s = np.arange(len(_sci_xy), dtype=int)
                    _mutual = (_i_rs[_i_sr] == _idx_s) & np.isfinite(_d_sr)

                    # Adaptive match tolerance for sparse fields
                    _n_total_reproj = len(_sci_xy) + len(_ref_xy)
                    _reproj_match_tol = (3.0 if _n_total_reproj < 10 else 2.0) * _det_fwhm
                    _good = _mutual & (_d_sr < _reproj_match_tol)
                    _reproj_min_matches = 2 if _n_total_reproj < 10 else 3
                    if _good.sum() >= _reproj_min_matches:
                        _dx = _sci_xy[_good, 0] - _ref_xy[_i_sr[_good], 0]
                        _dy = _sci_xy[_good, 1] - _ref_xy[_i_sr[_good], 1]
                        _med_dx = float(np.median(_dx))
                        _med_dy = float(np.median(_dy))
                        _rms_dx = float(np.std(_dx))
                        _rms_dy = float(np.std(_dy))
                        _total = float(np.sqrt(_med_dx**2 + _med_dy**2))
                        _rms = float(np.sqrt(_rms_dx**2 + _rms_dy**2))
                        _n_match_reproj = int(_good.sum())
                        _indiv = np.sqrt(_dx**2 + _dy**2)
                        _p95_reproj = float(np.percentile(_indiv, 95))
                        self.logger.info(
                            "Post-reproject alignment: offset=(%.3f, %.3f) px, "
                            "RMS=(%.3f, %.3f) px, total=%.3f px, rms=%.3f px, "
                            "P95=%.3f px (%d matches)",
                            _med_dx, _med_dy, _rms_dx, _rms_dy, _total, _rms,
                            _p95_reproj, _n_match_reproj,
                        )

                        _max_off = float(
                            quality_cfg.get("alignment_max_offset_px", 0.5)
                        )
                        _max_rms = float(
                            quality_cfg.get("alignment_max_rms_px", 0.75)
                        )
                        _max_p95 = float(
                            quality_cfg.get("alignment_max_p95_px", 1.5)
                        )
                        _reproj_min_n = int(
                            quality_cfg.get("alignment_min_sources_for_field_gate", 20)
                        )
                        # FWHM-adaptive thresholds for fallback gates
                        _reproj_fwhm = float(self.input_yaml.get("fwhm", 3.0))
                        _reproj_scale = max(0.5, min(3.0, _reproj_fwhm / 3.0))
                        _max_off *= _reproj_scale
                        _max_rms *= _reproj_scale
                        _max_p95 *= _reproj_scale
                        _reproj_reject = _n_match_reproj >= _reproj_min_matches and _total > _max_off
                        if _n_match_reproj >= _reproj_min_n:
                            _reproj_reject = _reproj_reject or _rms > _max_rms or _p95_reproj > _max_p95
                        if _reproj_reject:
                            _reasons = []
                            if _total > _max_off:
                                _reasons.append("offset=%.2f px (> %.2f px)" % (_total, _max_off))
                            if _n_match_reproj >= _reproj_min_n:
                                if _rms > _max_rms:
                                    _reasons.append("RMS=%.2f px (> %.2f px)" % (_rms, _max_rms))
                                if _p95_reproj > _max_p95:
                                    _reasons.append("P95=%.2f px (> %.2f px)" % (_p95_reproj, _max_p95))
                            self.logger.warning(
                                "Reproject alignment rejected: %s (%d matches). "
                                "Falling through to next alignment method.",
                                "; ".join(_reasons) if _reasons else "unknown", _n_match_reproj,
                            )
                            return None
                        else:
                            self.logger.info(
                                "Reproject alignment accepted: offset=%.2f px, "
                                "RMS=%.2f px, P95=%.2f px (%d matches).",
                                _total, _rms, _p95_reproj, _n_match_reproj,
                            )
                            reproject_metadata = {
                                "offset_x": _med_dx,
                                "offset_y": _med_dy,
                                "rms_x": _rms_dx,
                                "rms_y": _rms_dy,
                                "n_matched": _n_match_reproj,
                                "p95_offset": _p95_reproj,
                            }
            except Exception as _ve:
                self.logger.debug("Post-reproject verification failed (non-fatal): %s", _ve)

            if not reproject_metadata:
                self.logger.warning(
                    "Reproject alignment could not be verified (insufficient sources); "
                    "accepting as best available fallback."
                )

            return {
                "science_aligned": science_image,
                "reference_aligned": str(aligned_reference_fpath),
                "alignment_method": f"reproject/{used_method}",
                **(reproject_metadata if reproject_metadata else {}),
            }
        except Exception as e:
            self.logger.info("Reproject alignment failed: %s", e)
            return None

    def _compute_optimal_output_shape(
        self,
        sci_data: np.ndarray,
        sci_wcs,
        ref_data: np.ndarray,
        ref_wcs,
        pix_scale: float,
        target_ra: Optional[float] = None,
        target_dec: Optional[float] = None,
    ) -> Tuple[int, int, float, float]:
        """Compute optimal output shape as intersection of valid (non-NaN) regions.
        
        This reduces output size when images have different coverage or significant
        masked edges, avoiding computation on regions where only one image has data.
        
        Parameters
        ----------
        sci_data, ref_data : ndarray
            Image data arrays (may contain NaN for masked regions)
        sci_wcs, ref_wcs : WCS
            World coordinate systems for each image
        pix_scale : float
            Output pixel scale in arcsec/pixel
        target_ra, target_dec : float, optional
            Target position in degrees. If provided, the output shape is guaranteed
            to include this position (expanded if necessary, up to science image bounds).
            
        Returns
        -------
        (width, height, center_ra, center_dec) : tuple
            Optimal output dimensions in pixels and sky centre of the overlap
            region (degrees).  Callers should use center_ra/center_dec as the
            SWarp CENTER so the output grid lies within both images' footprints.
        """
        try:
            # Find valid (finite) pixel regions in each image
            sci_valid = np.isfinite(sci_data)
            ref_valid = np.isfinite(ref_data)
            
            # Get bounding boxes of valid regions
            def _valid_bbox(valid_mask):
                rows = np.any(valid_mask, axis=1)
                cols = np.any(valid_mask, axis=0)
                if not np.any(rows) or not np.any(cols):
                    return None
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return (cmin, rmin, cmax, rmax)  # x1,y1,x2,y2 in pixel coords
            
            sci_bbox = _valid_bbox(sci_valid)
            ref_bbox = _valid_bbox(ref_valid)
            
            if sci_bbox is None or ref_bbox is None:
                # Fall back to science shape if no valid data found
                _scx = (sci_data.shape[1] - 1) / 2.0
                _scy = (sci_data.shape[0] - 1) / 2.0
                _fb_ra, _fb_dec = sci_wcs.all_pix2world([_scx], [_scy], 0)
                return sci_data.shape[1], sci_data.shape[0], float(_fb_ra[0]), float(_fb_dec[0])
            
            # Convert bounding box corners to world coordinates.
            # _valid_bbox gives the first/last 0-based pixel indices that contain
            # valid data.  The actual footprint extends half a pixel beyond those
            # centers, so offset by +/-0.5 to get the outer corners.
            def _bbox_corners(bbox):
                x1, y1, x2, y2 = bbox
                return np.array(
                    [[x1 - 0.5, y1 - 0.5],
                     [x2 + 0.5, y1 - 0.5],
                     [x2 + 0.5, y2 + 0.5],
                     [x1 - 0.5, y2 + 0.5]]
                )
            
            sci_corners = _bbox_corners(sci_bbox)
            ref_corners = _bbox_corners(ref_bbox)
            
            # Convert to RA/Dec
            sci_world = sci_wcs.all_pix2world(sci_corners, 0)
            ref_world = ref_wcs.all_pix2world(ref_corners, 0)
            
            # Find overlapping RA/Dec region
            ra_min = max(np.min(sci_world[:, 0]), np.min(ref_world[:, 0]))
            ra_max = min(np.max(sci_world[:, 0]), np.max(ref_world[:, 0]))
            dec_min = max(np.min(sci_world[:, 1]), np.min(ref_world[:, 1]))
            dec_max = min(np.max(sci_world[:, 1]), np.max(ref_world[:, 1]))
            
            # Check if there is any overlap
            if ra_min >= ra_max or dec_min >= dec_max:
                self.logger.warning(
                    "No overlap in valid regions; using science image shape."
                )
                _scx = (sci_data.shape[1] - 1) / 2.0
                _scy = (sci_data.shape[0] - 1) / 2.0
                _fb_ra, _fb_dec = sci_wcs.all_pix2world([_scx], [_scy], 0)
                return sci_data.shape[1], sci_data.shape[0], float(_fb_ra[0]), float(_fb_dec[0])
            
            # Convert overlap size to pixels at output pixel scale
            # cos(dec) factor for RA separation
            overlap_center_ra = (ra_min + ra_max) / 2
            overlap_center_dec = (dec_min + dec_max) / 2
            cos_dec = np.cos(np.radians(overlap_center_dec))
            ra_sep = (ra_max - ra_min) * cos_dec * 3600  # arcsec
            dec_sep = (dec_max - dec_min) * 3600  # arcsec
            
            width = int(ra_sep / pix_scale)
            height = int(dec_sep / pix_scale)
            
            # Ensure minimum size and add small margin
            width = max(width, 100)
            height = max(height, 100)
            
            # Ensure target is included in output (if target position provided)
            if target_ra is not None and target_dec is not None:
                # Convert target world coords to pixel coords in output grid
                # Output grid center is at (width/2, height/2) in pixel coords
                # with pix_scale arcsec/pixel
                cos_dec = np.cos(np.radians(target_dec))
                
                # Target offset from center in arcsec
                dra_arcsec = (target_ra - overlap_center_ra) * cos_dec * 3600
                ddec_arcsec = (target_dec - overlap_center_dec) * 3600
                
                # Convert to pixels
                dx_pix = dra_arcsec / pix_scale
                dy_pix = ddec_arcsec / pix_scale
                
                # Check if target falls within current output bounds
                half_w = width / 2
                half_h = height / 2
                
                # Expand width if needed
                if abs(dx_pix) >= half_w - 5:  # 5 pixel margin
                    new_half_w = abs(dx_pix) + 10  # Add margin
                    width = int(2 * new_half_w)
                    self.logger.debug(
                        "Expanded output width to include target: %d -> %d", 
                        int(2 * half_w), width
                    )
                
                # Expand height if needed
                if abs(dy_pix) >= half_h - 5:  # 5 pixel margin
                    new_half_h = abs(dy_pix) + 10  # Add margin
                    height = int(2 * new_half_h)
                    self.logger.debug(
                        "Expanded output height to include target: %d -> %d",
                        int(2 * half_h), height
                    )
            
            # Cap at original science size (don't expand beyond input)
            # But ensure target is included even if it means using full science size
            width = min(width, sci_data.shape[1])
            height = min(height, sci_data.shape[0])
            
            # Final check: if target is provided, verify it's within the output bounds
            # If not, use full science image size to ensure target is included
            if target_ra is not None and target_dec is not None:
                # Convert target to pixel coords in science image
                sci_target_x, sci_target_y = sci_wcs.all_world2pix([target_ra], [target_dec], 0)
                sci_target_x, sci_target_y = float(sci_target_x[0]), float(sci_target_y[0])
                
                # Check if target is within output bounds (with margin)
                margin = 50  # pixels
                if not (margin <= sci_target_x < width - margin and margin <= sci_target_y < height - margin):
                    self.logger.warning(
                        "Target (%.1f, %.1f) is outside output bounds (%dx%d). Using full science image size (%dx%d) to ensure target is included.",
                        sci_target_x, sci_target_y, width, height,
                        sci_data.shape[1], sci_data.shape[0]
                    )
                    width = sci_data.shape[1]
                    height = sci_data.shape[0]
            
            return width, height, overlap_center_ra, overlap_center_dec
            
        except Exception as e:
            self.logger.warning(
                "Could not compute optimal output shape: %s. Using science shape.", e
            )
            _scx = (sci_data.shape[1] - 1) / 2.0
            _scy = (sci_data.shape[0] - 1) / 2.0
            try:
                _fb_ra, _fb_dec = sci_wcs.all_pix2world([_scx], [_scy], 0)
                return sci_data.shape[1], sci_data.shape[0], float(_fb_ra[0]), float(_fb_dec[0])
            except Exception:
                return sci_data.shape[1], sci_data.shape[0], float(sci_wcs.wcs.crval[0]), float(sci_wcs.wcs.crval[1])

    def _compute_relative_wcs_correction(
        self,
        sci_image_path: str,
        ref_image_path: str,
        sci_cat_path: str,
        ref_cat_path: str,
        output_head_path: str,
    ) -> Optional[dict]:
        """
        Compute a relative WCS correction from matched sources, preserving
        the reference's SIP distortion.

        Instead of SCAMP replacing the entire WCS (which loses high-order SIP
        when insufficient sources are available), this computes a constant
        CRVAL + CD matrix correction from matched source positions and writes
        a .head file with the full corrected WCS (including original SIP
        coefficients).

        The correction is computed by:
        1. Converting reference source pixels -> sky via reference WCS
        2. Converting sky -> science pixels via science WCS
        3. Measuring the residual (science_actual - science_predicted)
        4. Fitting an affine model to the residual field (dx, dy vs rx, ry)
        5. Converting the affine correction to WCS terms:
           - Constant part -> CRVAL shift
           - Linear part -> CD matrix adjustment
        6. Applying both corrections to the reference's original WCS header,
           preserving all SIP distortion coefficients.

        This is equivalent to SCAMP degree 1 (shift + rotation + scale) but
        preserves the reference's high-order SIP distortion, which SCAMP
        would replace entirely.
        """
        try:
            from wcs import get_wcs as _get_wcs

            # Load matched catalogs
            with fits.open(sci_cat_path) as hs, fits.open(ref_cat_path) as hr:
                sci_tab = Table(hs[2].data)
                ref_tab = Table(hr[2].data)

            if (
                len(sci_tab) == 0
                or len(ref_tab) == 0
                or len(sci_tab) != len(ref_tab)
            ):
                self.logger.warning(
                    "Relative WCS correction: no matched sources available"
                )
                return None

            # Extract pixel positions (SExtractor is 1-indexed, WCS is 0-indexed)
            def _get_xy(tbl):
                xcol = "XWIN_IMAGE" if "XWIN_IMAGE" in tbl.colnames else "X_IMAGE"
                ycol = "YWIN_IMAGE" if "YWIN_IMAGE" in tbl.colnames else "Y_IMAGE"
                return (
                    np.asarray(tbl[xcol], float) - 1.0,
                    np.asarray(tbl[ycol], float) - 1.0,
                )

            sx, sy = _get_xy(sci_tab)
            rx, ry = _get_xy(ref_tab)

            # Load WCS from both images
            with fits.open(sci_image_path, memmap=False) as h_sci:
                sci_wcs = _get_wcs(h_sci[0].header)
            with fits.open(ref_image_path, memmap=False) as h_ref:
                ref_wcs = _get_wcs(h_ref[0].header)
                ref_header = h_ref[0].header.copy()

            if sci_wcs is None or ref_wcs is None:
                self.logger.warning(
                    "Relative WCS correction: invalid WCS in science or reference"
                )
                return None

            # Compute pixel-space residuals:
            # ref pixel -> sky (via ref WCS) -> science pixel (via sci WCS)
            ra, dec = ref_wcs.all_pix2world(rx, ry, 0)
            x_pred, y_pred = sci_wcs.all_world2pix(ra, dec, 0)

            dx = sx - x_pred
            dy = sy - y_pred

            # Robust statistics
            median_dx = float(np.median(dx))
            median_dy = float(np.median(dy))
            rms_before = float(np.sqrt(np.median(dx**2 + dy**2)))
            n_sources = len(dx)

            self.logger.info(
                "Relative WCS correction: %d sources, "
                "median offset=(%.3f, %.3f) px, RMS=%.3f px",
                n_sources, median_dx, median_dy, rms_before,
            )

            # Fit affine model: dx = a0 + a1*rx + a2*ry, dy = b0 + b1*rx + b2*ry
            # This captures shift (a0, b0) + rotation/scale (a1, a2, b1, b2)
            cd_sci = sci_wcs.wcs.cd
            cd_ref = ref_wcs.wcs.cd
            crpix_ref = ref_wcs.wcs.crpix

            if n_sources >= 6:
                # Enough sources for a full affine fit (6 parameters)
                # Design matrix: [1, rx, ry] for each source
                A_design = np.column_stack([np.ones(n_sources), rx, ry])
                # Solve for dx coefficients
                dx_coeffs, _, _, _ = np.linalg.lstsq(A_design, dx, rcond=None)
                # Solve for dy coefficients
                dy_coeffs, _, _, _ = np.linalg.lstsq(A_design, dy, rcond=None)

                a0, a1, a2 = dx_coeffs
                b0, b1, b2 = dy_coeffs

                self.logger.info(
                    "Affine fit: dx=%.4f + %.6f*rx + %.6f*ry, "
                    "dy=%.4f + %.6f*rx + %.6f*ry",
                    a0, a1, a2, b0, b1, b2,
                )

                # Convert affine correction to WCS terms:
                # The residual in science pixel space is:
                #   delta_pix = [a0 + a1*rx + a2*ry, b0 + b1*rx + b2*ry]
                #
                # We want the corrected reference WCS to produce sky positions
                # that, when converted to science pixels, shift by delta_pix.
                #
                # sky_shift = CD_sci @ delta_pix  (in degrees)
                #
                # The sky shift has a constant part and a linear part:
                #   sky_shift = CD_sci @ [a0, b0] + CD_sci @ M @ [rx, ry]
                # where M = [[a1, a2], [b1, b2]]
                #
                # Constant part -> CRVAL shift
                # Linear part -> CD matrix correction:
                #   We want: CD_new @ x = CD_old @ x + CD_sci @ M @ x
                #   (where x = pix - CRPIX, and rx = x + CRPIX_x)
                #   But the linear part also has a constant contribution from
                #   M @ CRPIX, which we absorb into CRVAL.
                #
                # So:
                #   CD_new = CD_old + CD_sci @ M
                #   delta_CRVAL = CD_sci @ [a0, b0] + CD_sci @ M @ CRPIX

                M = np.array([[a1, a2], [b1, b2]])
                constant_pix = np.array([a0, b0])
                crpix_arr = np.array([crpix_ref[0], crpix_ref[1]])

                delta_cd = cd_sci @ M
                delta_crval = cd_sci @ constant_pix + cd_sci @ M @ crpix_arr

                # Apply corrections to reference header
                corrected_header = ref_header.copy()
                corrected_header["CRVAL1"] = float(ref_header["CRVAL1"]) + float(delta_crval[0])
                corrected_header["CRVAL2"] = float(ref_header["CRVAL2"]) + float(delta_crval[1])

                # Adjust CD matrix
                cd_old = np.array([
                    [float(ref_header.get("CD1_1", cd_ref[0, 0])),
                     float(ref_header.get("CD1_2", cd_ref[0, 1]))],
                    [float(ref_header.get("CD2_1", cd_ref[1, 0])),
                     float(ref_header.get("CD2_2", cd_ref[1, 1]))],
                ])
                cd_new = cd_old + delta_cd
                corrected_header["CD1_1"] = float(cd_new[0, 0])
                corrected_header["CD1_2"] = float(cd_new[0, 1])
                corrected_header["CD2_1"] = float(cd_new[1, 0])
                corrected_header["CD2_2"] = float(cd_new[1, 1])

                self.logger.info(
                    "Affine WCS correction: dCRVAL=(%.6f, %.6f) deg (%.3f\", %.3f\"), "
                    "dCD=(%.2e, %.2e, %.2e, %.2e)",
                    delta_crval[0], delta_crval[1],
                    delta_crval[0] * 3600.0, delta_crval[1] * 3600.0,
                    delta_cd[0, 0], delta_cd[0, 1],
                    delta_cd[1, 0], delta_cd[1, 1],
                )

            else:
                # Too few sources for affine fit — use constant CRVAL shift only
                self.logger.info(
                    "Only %d sources — using constant CRVAL shift (no affine)",
                    n_sources,
                )
                delta_ra = float(cd_sci[0, 0] * median_dx + cd_sci[0, 1] * median_dy)
                delta_dec = float(cd_sci[1, 0] * median_dx + cd_sci[1, 1] * median_dy)

                corrected_header = ref_header.copy()
                corrected_header["CRVAL1"] = float(ref_header["CRVAL1"]) + delta_ra
                corrected_header["CRVAL2"] = float(ref_header["CRVAL2"]) + delta_dec

                self.logger.info(
                    "CRVAL correction: dRA=%.6f deg (%.3f\"), dDec=%.6f deg (%.3f\")",
                    delta_ra, delta_ra * 3600.0,
                    delta_dec, delta_dec * 3600.0,
                )

            # Verify correction: re-compute residuals with corrected WCS
            corrected_wcs = _get_wcs(corrected_header)
            if corrected_wcs is not None:
                ra_c, dec_c = corrected_wcs.all_pix2world(rx, ry, 0)
                x_pred_c, y_pred_c = sci_wcs.all_world2pix(ra_c, dec_c, 0)
                dx_c = sx - x_pred_c
                dy_c = sy - y_pred_c
                rms_after = float(np.sqrt(np.median(dx_c**2 + dy_c**2)))
                median_after = float(np.sqrt(np.median(dx_c)**2 + np.median(dy_c)**2))
                self.logger.info(
                    "Relative WCS correction verified: median %.3f -> %.3f px, "
                    "RMS %.3f -> %.3f px",
                    float(np.sqrt(median_dx**2 + median_dy**2)),
                    median_after, rms_before, rms_after,
                )
                rms_before = rms_after  # Use post-correction RMS for quality gate

            # Write corrected header as .head file (WCS keys only)
            wcs_prefixes = (
                "CRPIX", "CRVAL", "CTYPE", "CD", "PC", "CDELT",
                "CROTA", "PV", "LONPOLE", "LATPOLE", "EQUINOX",
                "WCSNAME", "CUNIT", "WCSAXES", "PROJP", "LTV",
                "LTM", "RADECSYS", "RADESYS", "RADYSYS",
                "LONGPOLE", "TNX", "SIP_",
            )
            wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")

            head_header = fits.Header()
            for key in corrected_header:
                if key in ("NAXIS", "NAXIS1", "NAXIS2", "SIMPLE", "BITPIX", "EXTEND"):
                    continue
                is_wcs = any(key.startswith(p) for p in wcs_prefixes)
                if not is_wcs and "_" in key:
                    stem = key.split("_")[0] + "_"
                    is_wcs = stem in wcs_stems and key.startswith(stem.rstrip("_"))
                if is_wcs:
                    head_header[key] = corrected_header[key]

            head_header.totextfile(output_head_path)

            return {
                "method": "relative_wcs_correction",
                "median_offset_px": (median_dx, median_dy),
                "rms_px": rms_before,
                "n_sources": n_sources,
                "distortion": {
                    "astrometric_rms_arcsec": rms_before * abs(cd_sci[0, 0]) * 3600.0,
                    "n_matched_stars": n_sources,
                },
            }

        except Exception as e:
            self.logger.warning("Relative WCS correction failed: %s", e)
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _preserve_reference_sip_in_head(
        self,
        head_path: Path,
        ref_image_path: Path,
    ) -> None:
        """
        Modify a SCAMP .head file to preserve the reference image's original
        SIP/PV distortion coefficients.

        SCAMP's .head replaces the entire WCS with a lower-degree solution,
        losing high-order distortion. This method:
        1. Reads the SCAMP .head (linear WCS: CRVAL, CRPIX, CD, CTYPE)
        2. Reads the reference image's original header (SIP/PV coefficients)
        3. Restores SIP/PV keywords and CTYPE from the reference into the .head
        4. Writes the modified .head back

        The result: SCAMP's linear correction (shift/rotation/scale) is preserved,
        but the reference's high-order SIP distortion is also applied.
        """
        try:
            from wcs import _normalize_projection_codes

            # Read SCAMP .head
            scamp_header = fits.Header.fromtextfile(str(head_path))

            # Read reference image header
            with fits.open(str(ref_image_path), memmap=False) as h_ref:
                ref_header = h_ref[0].header.copy()

            # Identify SIP/PV distortion keywords in the reference header
            # These are the high-order terms we want to preserve
            sip_prefixes = ("A_", "B_", "AP_", "BP_")
            pv_prefix = "PV"
            sip_order_keys = ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")

            # Collect SIP/PV keywords from reference
            sip_keys_restored = 0
            for key in ref_header:
                is_sip = (
                    any(key.startswith(p) for p in sip_prefixes)
                    or key in sip_order_keys
                )
                is_pv = key.startswith(pv_prefix) and key not in ("PV1_0", "PV1_1")  # Keep SCAMP's PV1_0/PV1_1 if present
                # Also catch PV_ prefix
                if not is_pv and key.startswith("PV_"):
                    is_pv = True
                if is_sip or is_pv:
                    scamp_header[key] = ref_header[key]
                    sip_keys_restored += 1

            # Restore CTYPE to include -SIP suffix if reference had SIP
            # SCAMP may have written CTYPE1='RA---TAN' (no -SIP)
            ref_ctype1 = ref_header.get("CTYPE1", "")
            ref_ctype2 = ref_header.get("CTYPE2", "")
            if "-SIP" in ref_ctype1 or "-SIP" in ref_ctype2:
                scamp_header["CTYPE1"] = ref_ctype1
                scamp_header["CTYPE2"] = ref_ctype2

            # Normalize projection codes to ensure consistency
            scamp_header = _normalize_projection_codes(scamp_header, inplace=False)

            # Write modified .head back
            scamp_header.totextfile(str(head_path), overwrite=True)

            self.logger.info(
                "Preserved %d SIP/PV distortion keywords from reference in .head "
                "(SCAMP linear WCS + reference high-order SIP)",
                sip_keys_restored,
            )

        except Exception as e:
            self.logger.warning(
                "Failed to preserve reference SIP in .head (%s). "
                "Using SCAMP .head as-is (may lose high-order distortion).",
                e,
            )
            import traceback
            self.logger.debug(traceback.format_exc())

    def _scamp_reproject_reference_to_science(
        self,
        ref_image_copy: Path,
        sci_image_path: str,
        output_dir: str,
        reproject_cfg: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Apply SCAMP .head WCS correction to reference, then reproject onto
        the science image's exact WCS pixel grid.

        This replaces SWarp for common_grid alignment.  Unlike SWarp which
        creates a new TAN grid (different CRPIX, no rotation, scalar pixel
        scale), reproject maps directly from the SCAMP-corrected reference
        WCS to the science WCS — preserving rotation, CD matrix, non-square
        pixels, and SIP distortion exactly.

        Science image is never touched.  Only the reference is resampled.

        Parameters
        ----------
        ref_image_copy : Path
            Path to the reference image copy (with .head file next to it).
        sci_image_path : str
            Path to the science image (defines target WCS grid).
        output_dir : str
            Directory for the reprojected reference output.
        reproject_cfg : dict, optional
            Alignment config from input_yaml (reproject_method, interp_order, etc.).

        Returns
        -------
        Optional[Path]
            Path to reprojected reference, or None on failure.
        """
        if not HAS_REPROJECT:
            self.logger.warning(
                "reproject package not available; cannot use SCAMP+reproject path."
            )
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "reference_reprojected.fits"

        try:
            # --- Read science WCS (the target grid) ---
            with fits.open(sci_image_path, memmap=False) as h_sci:
                sci_header = h_sci[0].header
                sci_data = h_sci[0].data
                if sci_data is None or sci_data.size == 0:
                    self.logger.error("Science image has no data for reproject target.")
                    return None
                sci_shape = sci_data.shape
                sci_wcs = get_wcs(sci_header)
                if sci_wcs is None:
                    self.logger.error("Science image has no valid WCS for reproject target.")
                    return None

            # --- Read reference data and header ---
            with fits.open(str(ref_image_copy), memmap=False) as h_ref:
                ref_data = np.asarray(h_ref[0].data, dtype=float)
                ref_header = h_ref[0].header.copy()

            if ref_data is None or ref_data.size == 0:
                self.logger.error("Reference image has no data for reproject.")
                return None

            # --- Pre-mask NaN pixels in reference data ---
            # reproject_adaptive/interp treat NaN as 0, which creates artifacts.
            # Replace NaNs with 0 and track the mask via footprint afterwards.
            n_nan_ref = int(np.sum(~np.isfinite(ref_data)))
            if n_nan_ref > 0:
                self.logger.info(
                    "Reference has %d NaN pixels (%.2f%%) — replacing with 0 before reprojection.",
                    n_nan_ref, 100.0 * n_nan_ref / ref_data.size,
                )
                ref_data = np.where(np.isfinite(ref_data), ref_data, 0.0)

            # --- Log WCS diagnostics for both images ---
            try:
                sci_ps = np.sqrt(abs(sci_wcs.wcs.cd[0,0]*sci_wcs.wcs.cd[1,1] - sci_wcs.wcs.cd[0,1]*sci_wcs.wcs.cd[1,0])) * 3600.0
                sci_rot = np.degrees(np.arctan2(sci_wcs.wcs.cd[0,1], sci_wcs.wcs.cd[0,0]))
                self.logger.info(
                    "Reproject diagnostics — science: shape=%s, pixscale=%.3f arcsec/px, "
                    "rotation=%.2f deg, CRPIX=(%.1f, %.1f)",
                    sci_shape, sci_ps, sci_rot, sci_wcs.wcs.crpix[0], sci_wcs.wcs.crpix[1],
                )
            except Exception:
                pass

            # --- Apply SCAMP .head to reference header (in memory) ---
            head_file = Path(ref_image_copy).with_suffix(".head")
            if head_file.exists():
                self.logger.info("Applying SCAMP .head WCS to reference for reproject")
                try:
                    from wcs import _normalize_projection_codes
                    scamp_header = fits.Header.fromtextfile(str(head_file))
                    scamp_header = _normalize_projection_codes(scamp_header, inplace=False)

                    # Remove old WCS from reference header, then copy SCAMP WCS in
                    ref_header = remove_wcs_from_header(ref_header)
                    _wcs_prefixes = (
                        "CRPIX", "CRVAL", "CTYPE", "CD", "PC", "CDELT",
                        "CROTA", "PV", "LONPOLE", "LATPOLE", "EQUINOX",
                        "WCSNAME", "CUNIT", "WCSAXES", "PROJP", "LTV",
                        "LTM", "RADECSYS", "RADESYS", "RADYSYS",
                        "LONGPOLE", "TNX", "SIP_",
                    )
                    _wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
                    for key in scamp_header:
                        if key in ("NAXIS", "NAXIS1", "NAXIS2"):
                            continue
                        is_wcs = any(key.startswith(p) for p in _wcs_prefixes)
                        if not is_wcs and "_" in key:
                            stem = key.split("_")[0] + "_"
                            is_wcs = stem in _wcs_stems and key.startswith(stem.rstrip("_"))
                        if is_wcs:
                            ref_header[key] = scamp_header[key]
                except Exception as e:
                    self.logger.warning(
                        "Failed to apply SCAMP .head to reference header (%s). "
                        "Reproject will use reference's original WCS.", e
                    )
            else:
                self.logger.warning(
                    "No SCAMP .head found at %s; reproject will use reference's original WCS.",
                    head_file,
                )

            # Build WCS from the SCAMP-corrected reference header
            ref_wcs_corrected = get_wcs(ref_header)
            if ref_wcs_corrected is None:
                self.logger.error("Could not build WCS from SCAMP-corrected reference header.")
                return None

            # --- Log reference WCS diagnostics ---
            try:
                ref_ps = np.sqrt(abs(ref_wcs_corrected.wcs.cd[0,0]*ref_wcs_corrected.wcs.cd[1,1] - ref_wcs_corrected.wcs.cd[0,1]*ref_wcs_corrected.wcs.cd[1,0])) * 3600.0
                ref_rot = np.degrees(np.arctan2(ref_wcs_corrected.wcs.cd[0,1], ref_wcs_corrected.wcs.cd[0,0]))
                self.logger.info(
                    "Reproject diagnostics — reference: shape=%s, pixscale=%.3f arcsec/px, "
                    "rotation=%.2f deg, CRPIX=(%.1f, %.1f)",
                    ref_data.shape, ref_ps, ref_rot,
                    ref_wcs_corrected.wcs.crpix[0], ref_wcs_corrected.wcs.crpix[1],
                )
                ps_ratio = ref_ps / sci_ps if sci_ps > 0 else 0.0
                rot_diff = abs(ref_rot - sci_rot)
                if rot_diff > 180:
                    rot_diff = 360 - rot_diff
                self.logger.info(
                    "Reproject diagnostics — pixel scale ratio=%.3f, rotation difference=%.2f deg",
                    ps_ratio, rot_diff,
                )
            except Exception:
                pass

            # --- Read reproject config ---
            cfg = reproject_cfg or {}
            req_method = str(cfg.get("reproject_method", "adaptive")).lower().strip()
            interp_order = str(cfg.get("reproject_interp_order", "bicubic")).lower().strip()
            use_parallel = bool(cfg.get("reproject_parallel", False))
            roundtrip = bool(cfg.get("reproject_roundtrip_coords", True))
            conserve_flux = bool(cfg.get("reproject_adaptive_conserve_flux", False))
            center_jacobian = bool(cfg.get("reproject_adaptive_center_jacobian", False))

            # --- Auto-enable center_jacobian for large rotation differences ---
            # The default Jacobian approximation (center_jacobian=False) is
            # inaccurate when the rotation between input and output is large
            # (>30 deg), causing sub-pixel misregistration that varies across
            # the field.  This is especially problematic for ~180 deg flips.
            try:
                if not center_jacobian and rot_diff > 30.0:
                    center_jacobian = True
                    self.logger.info(
                        "Large rotation (%.1f deg) — enabling center_jacobian for "
                        "more accurate reproject_adaptive resampling.",
                        rot_diff,
                    )
            except Exception:
                pass

            # --- Warn about TPV/SIP projection mismatch ---
            try:
                sci_ctype1 = str(sci_wcs.wcs.ctype[0]).upper()
                ref_ctype1 = str(ref_wcs_corrected.wcs.ctype[0]).upper()
                if "SIP" in sci_ctype1 and "TPV" in ref_ctype1:
                    self.logger.warning(
                        "Projection mismatch: science uses SIP distortion but SCAMP "
                        "produced TPV for reference. Reproject will convert between "
                        "models — residual distortion errors of ~0.5 px are possible."
                    )
            except Exception:
                pass

            _interp_map = {
                "nearest": "nearest-neighbor", "nearest-neighbor": "nearest-neighbor",
                "nn": "nearest-neighbor",
                "bilinear": "bilinear", "linear": "bilinear", "1": "bilinear",
                "biquadratic": "biquadratic", "2": "biquadratic",
                "bicubic": "bicubic", "cubic": "bicubic", "3": "bicubic",
            }
            interp_order_norm = _interp_map.get(interp_order, "bilinear")

            # --- Auto-downgrade interpolation for undersampled images ---
            # Bicubic/biquadratic introduce ringing artifacts when the PSF is
            # undersampled (FWHM < 2.5 px).  Bilinear is safer in that regime.
            iy = getattr(self, "input_yaml", None) or {}
            _fwhm = float(iy.get("fwhm", 0.0)) if isinstance(iy, dict) else 0.0
            if _fwhm > 0 and _fwhm < 2.5 and interp_order_norm in ("bicubic", "biquadratic"):
                self.logger.info(
                    "Undersampled image (FWHM=%.2f px < 2.5) — downgrading %s to bilinear "
                    "to avoid ringing artifacts.",
                    _fwhm, interp_order_norm,
                )
                interp_order_norm = "bilinear"

            # --- Run reproject: reference → science grid ---
            # exact is most geometrically accurate (exact pixel-area resampling);
            # adaptive is faster fallback; interp is last resort.
            all_methods = ("exact", "adaptive", "interp")
            if req_method in all_methods:
                fallbacks = [req_method] + [m for m in all_methods if m != req_method]
            else:
                fallbacks = ["exact", "adaptive", "interp"]

            aligned_ref = None
            footprint = None
            used_method = None
            last_exc = None

            for m in fallbacks:
                try:
                    if m == "adaptive":
                        if reproject_adaptive is None:
                            continue
                        aligned_ref, footprint = reproject_adaptive(
                            (ref_data, ref_wcs_corrected),
                            sci_wcs,
                            shape_out=sci_shape,
                            roundtrip_coords=roundtrip,
                            parallel=use_parallel,
                            conserve_flux=conserve_flux,
                            center_jacobian=center_jacobian,
                            **({"despike_jacobian": True} if "despike_jacobian" in _ADAPTIVE_PARAMS else {}),
                        )
                    elif m == "interp":
                        aligned_ref, footprint = reproject_interp(
                            (ref_data, ref_wcs_corrected),
                            sci_wcs,
                            shape_out=sci_shape,
                            order=interp_order_norm,
                            roundtrip_coords=roundtrip,
                        )
                    else:  # exact
                        try:
                            from reproject import reproject_exact
                        except ImportError:
                            continue
                        aligned_ref, footprint = reproject_exact(
                            (ref_data, ref_wcs_corrected),
                            sci_wcs,
                            shape_out=sci_shape,
                            parallel=use_parallel,
                        )
                    used_method = m
                    break
                except Exception as _e:
                    last_exc = _e
                    self.logger.debug("Reproject (%s) failed: %s", m, _e)

            if used_method is None or aligned_ref is None or footprint is None:
                self.logger.warning(
                    "All reproject methods failed for reference: %s", last_exc
                )
                return None

            # --- Post-process: mask non-footprint pixels, write output ---
            aligned_ref = np.asarray(aligned_ref, dtype=float)
            fp_mask = footprint.astype(bool)
            n_footprint = int(np.sum(fp_mask))
            n_total = int(fp_mask.size)
            if n_footprint == 0:
                self.logger.warning(
                    "Reproject footprint has zero coverage — WCS mismatch."
                )
                return None
            if n_footprint < n_total * 0.5:
                self.logger.warning(
                    "Reproject footprint coverage low: %d/%d pixels (%.1f%%).",
                    n_footprint, n_total, 100.0 * n_footprint / n_total,
                )
            aligned_ref[~fp_mask] = np.nan

            # Build output header: science WCS keywords copied into ref header
            from functions import copy_wcs_from_header, safe_fits_write
            out_header = ref_header.copy()
            out_header = remove_wcs_from_header(out_header)
            copy_wcs_from_header(sci_header, out_header)
            out_header["NAXIS1"] = sci_shape[1]
            out_header["NAXIS2"] = sci_shape[0]
            out_header["REPROJ"] = (True, "Reprojected to science grid via SCAMP+reproject")

            safe_fits_write(str(output_path), aligned_ref, out_header)

            self.logger.info(
                "SCAMP+reproject succeeded (method=%s, coverage=%.1f%%): %s",
                used_method, 100.0 * n_footprint / n_total, output_path,
            )
            return output_path

        except Exception as e:
            self.logger.warning(
                "SCAMP+reproject failed: %s", e
            )
            return None

    def _reproject_to_match(
        self, source_image: str, target_image: str, output_path: str
    ) -> Path:
        """
        Reproject source image to match target image's WCS grid.

        Uses reproject_interp with the configured interpolation order from
        input_yaml (alignment.reproject_interp_order, default bicubic).

        Parameters
        ----------
        source_image : str
            Path to image to reproject (reference)
        target_image : str
            Path to image defining the target WCS grid (science)
        output_path : str
            Path for output reprojected image

        Returns
        -------
        Path
            Path to reprojected output image
        """
        from astropy.io import fits
        from astropy.wcs import WCS
        from reproject import reproject_exact, reproject_adaptive, reproject_interp

        # Read configured interpolation order from input_yaml
        iy = getattr(self, "input_yaml", None) or {}
        align_cfg = iy.get("alignment", {}) if isinstance(iy, dict) else {}
        interp_order = str(align_cfg.get("reproject_interp_order", "bicubic")).lower().strip()
        _interp_map = {
            "nearest": "nearest-neighbor", "nearest-neighbor": "nearest-neighbor",
            "nn": "nearest-neighbor",
            "bilinear": "bilinear", "linear": "bilinear", "1": "bilinear",
            "biquadratic": "biquadratic", "2": "biquadratic",
            "bicubic": "bicubic", "cubic": "bicubic", "3": "bicubic",
        }
        interp_order_norm = _interp_map.get(interp_order, "bilinear")
        use_parallel = bool(align_cfg.get("reproject_parallel", True))

        from wcs import get_wcs as _get_wcs_for_reproj
        with fits.open(target_image) as target_hdul:
            target_wcs = _get_wcs_for_reproj(target_hdul[0].header)
            if target_wcs is None:
                self.logger.error("_reproject_to_match: get_wcs failed for target image")
                return None
            target_shape = target_hdul[0].data.shape

        with fits.open(source_image) as source_hdul:
            source_data = source_hdul[0].data
            source_header = source_hdul[0].header.copy()
            source_wcs = _get_wcs_for_reproj(source_header)
            if source_wcs is None:
                self.logger.error("_reproject_to_match: get_wcs failed for source image")
                return None

        # Reproject source onto target grid — exact-first for best accuracy
        reprojected_data = None
        footprint = None
        for m in ("exact", "adaptive", "interp"):
            try:
                if m == "exact":
                    reprojected_data, footprint = reproject_exact(
                        (source_data, source_wcs),
                        target_wcs,
                        shape_out=target_shape,
                        parallel=use_parallel,
                    )
                elif m == "adaptive":
                    if reproject_adaptive is None:
                        continue
                    reprojected_data, footprint = reproject_adaptive(
                        (source_data, source_wcs),
                        target_wcs,
                        shape_out=target_shape,
                        roundtrip_coords=True,
                        parallel=use_parallel,
                        **({"despike_jacobian": True} if "despike_jacobian" in _ADAPTIVE_PARAMS else {}),
                    )
                else:
                    reprojected_data, footprint = reproject_interp(
                        (source_data, source_wcs),
                        target_wcs,
                        shape_out=target_shape,
                        order=interp_order_norm,
                    )
                break
            except Exception as _e:
                self.logger.debug("_reproject_to_match (%s) failed: %s", m, _e)

        if reprojected_data is None:
            self.logger.error("_reproject_to_match: all reproject methods failed")
            return None

        # Update header with target WCS
        from functions import update_header_from_wcs
        output_header = source_header.copy()
        update_header_from_wcs(output_header, target_wcs)
        output_header["REPROJ"] = (True, "Reprojected to match science grid")

        # Write output
        fits.PrimaryHDU(data=reprojected_data, header=output_header).writeto(
            output_path, overwrite=True
        )

        self.logger.debug(f"Reprojected {source_image} -> {output_path}")
        return Path(output_path)

    def _align_fallback_reproject_then_astroalign(
        self,
        science_image: str,
        reference_image: str,
        output_dir,
    ) -> Dict:
        """Try reproject first; if it fails or is unavailable, fall back to AstroAlign."""
        result = self.align_with_reproject(
            science_image, reference_image, output_dir=output_dir
        )
        if result is not None and result.get("reference_aligned"):
            return result
        self.logger.info("Reproject failed. Falling back to AstroAlign.")
        return self.align_with_astroalign(
            science_image, reference_image, output_dir
        )

    # ------------------------------ AstroAlign path ------------------------------

    def align_with_astroalign(
        self,
        science_image: str,
        reference_image: str,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Align the reference image onto the science image pixel grid using AstroAlign.
        Uses SExtractor catalogs and a 1<->1 filter to build control points, prioritizing extended sources.
        """
        try:
            # Define output directories
            output_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(science_image).parent
            )
            # Make aligned directories unique per science image to prevent crosstalk
            # when multiple images are processed in the same output directory
            science_base_name = Path(science_image).stem
            science_aligned_dir = output_dir / f"aligned_sci_{science_base_name}"
            reference_aligned_dir = output_dir / f"aligned_ref_{science_base_name}"

            # Delete existing directories if they exist
            if science_aligned_dir.exists():
                shutil.rmtree(science_aligned_dir)
            if reference_aligned_dir.exists():
                shutil.rmtree(reference_aligned_dir)

            # Create new, empty directories
            science_aligned_dir.mkdir(parents=True, exist_ok=True)
            reference_aligned_dir.mkdir(parents=True, exist_ok=True)

            sci_image_copy = science_aligned_dir / "science_image.fits"
            ref_image_copy = reference_aligned_dir / "reference_image.fits"
            shutil.copy2(science_image, sci_image_copy)
            shutil.copy2(reference_image, ref_image_copy)
            sci_wcs, sci_pix_scale, sci_head = self._extract_wcs_and_scale(
                sci_image_copy
            )
            ref_wcs, ref_pix_scale, ref_head = self._extract_wcs_and_scale(
                ref_image_copy
            )

            def _clean_image_data(fits_path):
                with fits.open(fits_path, mode="update") as hdul:
                    data = hdul[0].data
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    # Preserve NaNs (chip gaps) instead of replacing with sentinel
                    data[~np.isfinite(data)] = np.nan
                    hdul[0].data = data
                    hdul.flush()

            iy = getattr(self, "input_yaml", None) or {}
            ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
            templates_cfg = iy.get("templates", {}) if isinstance(iy, dict) else {}
            sextractor_crowded = ts.get(
                "sextractor_crowded", templates_cfg.get("crowded_field", False)
            )

            def _extract_sources(fits_path, output_dir, aperture_radius, weight_path=None):
                return self.run_sextractor(
                    str(fits_path),
                    output_dir=str(output_dir),
                    aperture_radius=aperture_radius,
                    weight_path=weight_path,
                    crowded=sextractor_crowded,
                    for_alignment=True,
                )

            sci_aperture_radius = sci_head.get("APER", 7)
            ref_aperture_radius = ref_head.get("APER", 7)
            self.logger.info(
                f"Extracting sources from science image with aperture radius - {sci_aperture_radius:.1f} [px]"
            )
            sci_w = self._guess_map_weight_path(str(science_image))
            ref_w = self._guess_map_weight_path(str(reference_image))
            if sci_w:
                self.logger.info("AstroAlign SExtractor: using science MAP_WEIGHT %s", sci_w)
            if ref_w:
                self.logger.info("AstroAlign SExtractor: using reference MAP_WEIGHT %s", ref_w)

            sci_sex = _extract_sources(
                sci_image_copy, science_aligned_dir, sci_aperture_radius, weight_path=sci_w
            )
            self.logger.info(
                f"Extracting sources from reference image with aperture radius - {ref_aperture_radius:.1f} [px]"
            )
            ref_sex = _extract_sources(
                ref_image_copy, reference_aligned_dir, ref_aperture_radius, weight_path=ref_w
            )
            fwhm_sci_pix = float(sci_sex.get("fwhm", 2.5))
            fwhm_ref_pix = float(ref_sex.get("fwhm", 2.5))
            fwhm_sci_arcsec = fwhm_sci_pix * sci_pix_scale
            fwhm_ref_arcsec = fwhm_ref_pix * ref_pix_scale
            crossid_radius = max(2.0 * max(fwhm_sci_arcsec, fwhm_ref_arcsec), 3.0)

            def _load_matched_catalogs(sci_cat_path, ref_cat_path):
                with fits.open(sci_cat_path) as hs, fits.open(ref_cat_path) as hr:
                    sci_tab = Table(hs[2].data)
                    ref_tab = Table(hr[2].data)
                if (
                    len(sci_tab) == 0
                    or len(ref_tab) == 0
                    or len(sci_tab) != len(ref_tab)
                ):
                    raise ValueError(
                        "No 1:1 matched sources after filter_matched_sources"
                    )
                return sci_tab, ref_tab

            _num_matched, _ = self.filter_matched_sources(
                sci_cat_path=sci_sex["catalog_path"],
                ref_cat_path=ref_sex["catalog_path"],
                match_radius_arcsec=crossid_radius,
                sci_image_path=str(sci_image_copy),
                ref_image_path=str(ref_image_copy),
            )
            try:
                self.plot_matched_sources_side_by_side(
                    sci_image_path=str(sci_image_copy),
                    ref_image_path=str(ref_image_copy),
                    sci_cat_path=sci_sex["catalog_path"],
                    ref_cat_path=ref_sex["catalog_path"],
                    output_plot_path=science_aligned_dir / f"matched_sources_{Path(sci_image_copy).stem}.png",
                    label_color="#FF0000",
                    label_fontsize=10,
                    circle_radius_sci=fwhm_sci_pix,
                    circle_radius_ref=fwhm_ref_pix,
                )
            except Exception as _plot_exc:
                self.logger.debug("Matched-sources plot failed (non-fatal): %s", _plot_exc)
            sci_tab, ref_tab = _load_matched_catalogs(
                sci_sex["catalog_path"], ref_sex["catalog_path"]
            )

            def _extract_xy_snr(tbl):
                xcol = "XWIN_IMAGE" if "XWIN_IMAGE" in tbl.colnames else "X_IMAGE"
                ycol = "YWIN_IMAGE" if "YWIN_IMAGE" in tbl.colnames else "Y_IMAGE"
                snrcol = "SNR_WIN" if "SNR_WIN" in tbl.colnames else None
                x = np.asarray(tbl[xcol], float) - 1.0
                y = np.asarray(tbl[ycol], float) - 1.0
                if snrcol:
                    snr = np.asarray(tbl[snrcol], float)
                else:
                    snr = np.asarray(
                        (
                            tbl["FLUX_WIN"]
                            if "FLUX_WIN" in tbl.colnames
                            else tbl["FLUX_AUTO"]
                        ),
                        float,
                    )
                return x, y, snr

            sx, sy, ssnr = _extract_xy_snr(sci_tab)
            rx, ry, rsnr = _extract_xy_snr(ref_tab)
            joint = np.minimum(ssnr, rsnr)
            order = np.argsort(joint)[::-1]
            pts_sci = np.vstack([sy[order], sx[order]]).T
            pts_ref = np.vstack([ry[order], rx[order]]).T

            def _load_and_clean_image(fits_path):
                with fits.open(fits_path) as hdul:
                    # Keep float32 to avoid doubling memory on full frames.
                    img = np.asarray(hdul[0].data, dtype=np.float32)
                img = np.nan_to_num(img, nan=np.nan, posinf=np.nan, neginf=np.nan)
                return img

            ref_img = _load_and_clean_image(ref_image_copy)
            sci_img = _load_and_clean_image(sci_image_copy)
            MAX_CONTROL_POINTS = 300
            use_aafitrans = False
            try:
                import aafitrans

                tform, (matched_src, matched_dst) = aafitrans.find_transform(
                    pts_ref,
                    pts_sci,
                    max_control_points=min(MAX_CONTROL_POINTS, len(pts_ref)),
                    ttype="affine",
                    pixel_tolerance=max(fwhm_sci_pix, fwhm_ref_pix),
                    min_matches=3,
                    num_nearest_neighbors=3,
                    kdtree_search_radius=0.05,
                    n_samples=1,
                    get_best_fit=True,
                    seed=None,
                )
                use_aafitrans = True
                self.logger.info("Using aafitrans for alignment")
            except Exception as e:
                self.logger.info(
                    f"aafitrans not available or failed ({e}), using astroalign"
                )
                tform, (matched_src, matched_dst) = aa.find_transform(
                    pts_ref,
                    pts_sci,
                    max_control_points=min(MAX_CONTROL_POINTS, len(pts_ref)),
                )
            if use_aafitrans:
                try:
                    from scipy import ndimage

                    matrix = tform.params if hasattr(tform, "params") else tform._matrix
                    inv_matrix = np.linalg.inv(matrix)
                    aligned_ref_img = ndimage.affine_transform(
                        ref_img,
                        inv_matrix[:2, :2],
                        offset=inv_matrix[:2, 2],
                        output_shape=sci_img.shape,
                        order=3,
                        cval=np.nan,
                    )
                    footprint = np.isfinite(aligned_ref_img)
                except Exception as e:
                    self.logger.info(
                        f"aafitrans apply failed: {e}, falling back to astroalign"
                    )
                    use_aafitrans = False
                    tform, (matched_src, matched_dst) = aa.find_transform(
                        pts_ref,
                        pts_sci,
                        max_control_points=min(MAX_CONTROL_POINTS, len(pts_ref)),
                    )
            if not use_aafitrans:
                aligned_ref_img, footprint = aa.apply_transform(tform, ref_img, sci_img)
            # Preserve NaNs (chip gaps) instead of replacing with sentinel
            aligned_ref_img = np.nan_to_num(
                aligned_ref_img, nan=np.nan, posinf=np.nan, neginf=np.nan
            )
            aligned_ref_img[~footprint.astype(bool)] = np.nan

            def _save_aligned_image(data, header, output_path):
                from functions import safe_fits_write
                safe_fits_write(output_path, data, header)

            base_ref = os.path.splitext(os.path.basename(reference_image))[0]
            ext_ref = os.path.splitext(reference_image)[1]
            aligned_reference_fpath = os.path.join(output_dir, f"{base_ref}{ext_ref}")
            aligned_science_fpath = science_image
            out_hdr = ref_head.copy()
            out_hdr = remove_wcs_from_header(out_hdr)
            from functions import copy_wcs_from_header as _copy_wcs2
            _copy_wcs2(sci_head, out_hdr)
            _save_aligned_image(aligned_ref_img, out_hdr, aligned_reference_fpath)
            alignment_verified = False
            try:
                if use_aafitrans:
                    try:
                        src_w = tform(matched_src)  # skimage Transform.__call__ works
                    except TypeError:
                        # Fallback for transforms without __call__
                        src_w = tform.inverse(matched_src)
                else:
                    src_w = aa.matrix_transform(matched_src, tform.params)
                resid = np.sqrt(np.sum((src_w - matched_dst) ** 2, axis=1))
                rms_pix = float(np.sqrt(np.mean(resid**2))) if resid.size else np.nan
                _med_resid = float(np.median(resid)) if resid.size else np.nan
                _p95_resid = float(np.percentile(resid, 95)) if resid.size else np.nan
                _n_aa = int(len(matched_dst))
                self.logger.info(
                    f"AstroAlign: {_n_aa} matches, med={_med_resid:.3f} px, "
                    f"RMS={rms_pix:.3f} px, P95={_p95_resid:.3f} px"
                )

                _aa_max_off = float(ts.get("alignment_max_offset_px", 0.5))
                _aa_max_rms = float(ts.get("alignment_max_rms_px", 0.75))
                _aa_max_p95 = float(ts.get("alignment_max_p95_px", 1.5))
                _aa_min_n = int(ts.get("alignment_min_sources_for_field_gate", 20))
                # FWHM-adaptive thresholds for fallback gates
                _aa_fwhm = float(self.input_yaml.get("fwhm", 3.0))
                _aa_scale = max(0.5, min(3.0, _aa_fwhm / 3.0))
                _aa_max_off *= _aa_scale
                _aa_max_rms *= _aa_scale
                _aa_max_p95 *= _aa_scale
                # Sparse-field: accept 2 matches for AstroAlign
                _aa_min_matches = 2 if _n_aa < 8 else 3
                _aa_reject = _n_aa >= _aa_min_matches and np.isfinite(_med_resid) and _med_resid > _aa_max_off
                if _n_aa >= _aa_min_n:
                    _aa_reject = _aa_reject or (
                        np.isfinite(rms_pix) and rms_pix > _aa_max_rms
                    ) or (
                        np.isfinite(_p95_resid) and _p95_resid > _aa_max_p95
                    )
                if _aa_reject:
                    _reasons = []
                    if np.isfinite(_med_resid) and _med_resid > _aa_max_off:
                        _reasons.append("med=%.2f px (> %.2f px)" % (_med_resid, _aa_max_off))
                    if _n_aa >= _aa_min_n:
                        if np.isfinite(rms_pix) and rms_pix > _aa_max_rms:
                            _reasons.append("RMS=%.2f px (> %.2f px)" % (rms_pix, _aa_max_rms))
                        if np.isfinite(_p95_resid) and _p95_resid > _aa_max_p95:
                            _reasons.append("P95=%.2f px (> %.2f px)" % (_p95_resid, _aa_max_p95))
                    self.logger.warning(
                        "AstroAlign alignment rejected: %s (%d matches). "
                        "No more alignment methods available.",
                        "; ".join(_reasons) if _reasons else "unknown", _n_aa,
                    )
                    return None
                else:
                    self.logger.info(
                        "AstroAlign alignment accepted: med=%.2f px, "
                        "RMS=%.2f px, P95=%.2f px (%d matches).",
                        _med_resid if np.isfinite(_med_resid) else float('nan'),
                        rms_pix if np.isfinite(rms_pix) else float('nan'),
                        _p95_resid if np.isfinite(_p95_resid) else float('nan'),
                        _n_aa,
                    )
                    alignment_verified = bool(
                        _n_aa >= 3
                        and np.isfinite(_med_resid)
                        and np.isfinite(rms_pix)
                        and np.isfinite(_p95_resid)
                    )
            except Exception as e:
                self.logger.info(f"Could not compute alignment metrics: {e}")

            if not alignment_verified:
                self.logger.warning(
                    "AstroAlign alignment could not be verified (insufficient sources); "
                    "accepting as best available fallback."
                )

            # Remove aligned working directories and their contents after successful alignment.
            if science_aligned_dir.exists():
                shutil.rmtree(science_aligned_dir, ignore_errors=True)
                self.logger.debug("Removed aligned working dir: %s", science_aligned_dir)
            if reference_aligned_dir.exists():
                shutil.rmtree(reference_aligned_dir, ignore_errors=True)
                self.logger.debug("Removed aligned working dir: %s", reference_aligned_dir)

            return {
                "science_aligned": aligned_science_fpath,
                "reference_aligned": aligned_reference_fpath,
                "alignment_method": "astroalign",
            }
        except Exception as e:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            self.logger.error(
                f"Type: {exc_type.__name__} File: {fname} Line: {lineno} Message: {e}"
            )
            return {
                "science_aligned": None,
                "reference_aligned": None,
                "alignment_offset_arcsec": None,
            }

    # -------------------------------- SCAMP + SWarp --------------------------------
    def _parse_scamp_xml(self, xml_file: str) -> Optional[Dict]:
        """Extract WCS keys, astrometric residual, and matched star count from SCAMP XML.

        SCAMP outputs VOTable format with <TABLE>/<FIELD>/<TR>/<TD> structure,
        not simple named XML elements.
        """
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Strip XML namespaces for easier searching
            def _localname(tag):
                return tag.split("}")[-1] if "}" in tag else tag

            wcs = {}
            # WCS keywords are stored as PARAM elements with name=CD1_1 etc.
            for param in root.iter():
                ln = _localname(param.tag)
                if ln == "PARAM":
                    name = param.get("name", "")
                    if name in ("CD1_1", "CD1_2", "CD2_1", "CD2_2",
                                "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"):
                        try:
                            wcs[name] = float(param.get("value"))
                        except (ValueError, TypeError):
                            pass

            astrometric_rms = None
            n_matched_stars = None

            # Parse the Fields TABLE: FIELD definitions give column names,
            # DATA/TABLEDATA/TR/TD gives the values.
            for table in root.iter():
                if _localname(table.tag) != "TABLE":
                    continue
                table_id = table.get("ID", "")
                if table_id != "Fields":
                    continue

                # Collect FIELD names in order
                field_names = []
                for field in table:
                    if _localname(field.tag) == "FIELD":
                        field_names.append(field.get("name", ""))

                # Find DATA/TABLEDATA/TR
                for tr in table.iter():
                    if _localname(tr.tag) != "TR":
                        continue
                    tds = [td for td in tr if _localname(td.tag) == "TD"]
                    if len(tds) != len(field_names):
                        continue

                    col_map = dict(zip(field_names, [td.text or "" for td in tds]))

                    # AstromSigma_Reference: arraysize=2, values in arcsec
                    sigma_ref = col_map.get("AstromSigma_Reference", "")
                    if sigma_ref:
                        parts = sigma_ref.split()
                        if len(parts) >= 2:
                            try:
                                rms1 = float(parts[0])
                                rms2 = float(parts[1])
                                astrometric_rms = float(np.sqrt(rms1**2 + rms2**2))
                            except (ValueError, TypeError):
                                pass

                    # nstars from the external astrometric stats
                    nstars_str = col_map.get("AstromNDets_Reference", "")
                    if not nstars_str:
                        # Fallback: scan columns for a small integer after Chi2_Reference
                        chi2_idx = None
                        for i, fn in enumerate(field_names):
                            if fn == "Chi2_Reference":
                                chi2_idx = i
                                break
                        if chi2_idx is not None and chi2_idx + 1 < len(tds):
                            try:
                                n_matched_stars = int(float(tds[chi2_idx + 1].text))
                            except (ValueError, TypeError):
                                pass
                    else:
                        try:
                            n_matched_stars = int(float(nstars_str))
                        except (ValueError, TypeError):
                            pass

                    break  # Only one row in Fields table
                break  # Only one Fields table

            if astrometric_rms is not None:
                self.logger.info(
                    "SCAMP astrometric residual: %.4f\" (%d matched stars)",
                    astrometric_rms, n_matched_stars or -1,
                )

            return {
                "distortion_coeffs": {},
                "wcs_params": wcs,
                "astrometric_rms_arcsec": astrometric_rms,
                "n_matched_stars": n_matched_stars,
            }
        except Exception as e:
            self.logger.info(f"Could not parse SCAMP XML: {e}")
            return None

    def run_swarp(
        self,
        input_images: Union[str, List[str]],
        scamp_results: Optional[Dict] = None,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
        head_path: Optional[str] = None,
        no_weight_maps: bool = False,
    ) -> Dict:
        """
        Resample images using SWarp onto a common grid.

        For alignment of two images, place a SCAMP-generated .head file next to each
        input FITS with the same base name (e.g. ref.fits -> ref.head) so SWarp uses
        the refined WCS. Callers (e.g. align_and_resample_both_images) are responsible
        for copying the .head to the correct path before invoking this method.

        Config should set CENTER_TYPE/MANUAL, PIXEL_SCALE, IMAGE_SIZE to define the
        output grid; RESAMPLING_TYPE can be set per use (e.g. LANCZOS3, BILINEAR for
        undersampled). Returns resampled product paths and metadata.
        """
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image = str(output_dir / "output.fits")
        output_weights = str(output_dir / "weights.fits")
        log_file = str(output_dir / "swarp.log")

        # --- Config setup ---
        final_config = self.DEFAULT_SWARP_CONFIG.copy()
        final_config.update(
            {
                "WEIGHTOUT_NAME": output_weights,
                "NTHREADS": self.default_threads,
            }
        )
        if config:
            final_config.update(config)

        # --- Check SWarp executable ---
        swarp_cmd = self._check_executable("swarp")

        # --- Write SWarp config file ---
        config_file = str(output_dir / "default.swarp")
        with open(config_file, "w") as f:
            for k, v in final_config.items():
                f.write(f"{k}\t{v}\n")

        # --- Prepare input_images ---
        if isinstance(input_images, str):
            input_images = [input_images]
        input_images = [str(p) for p in input_images]

        # ------------------------------------------------------------------
        # Preserve NaNs through SWarp resampling via weight maps.
        #
        # SWarp does not reliably propagate NaNs in resampling kernels; the
        # supported mechanism for "no data" is a weight map (0 = invalid).
        # We therefore generate a MAP_WEIGHT image for each input:
        #   weight = 1 where finite, 0 where NaN/Inf.
        # With BLANK_BADPIXELS=Y and FILL_VALUE=NAN, SWarp will blank regions
        # with zero weight in the resampled products as NaN instead of 0.
        # ------------------------------------------------------------------
        weight_images: List[str] = []
        if no_weight_maps:
            weight_images = []
        for img_path in ([] if no_weight_maps else input_images):
            try:
                with fits.open(img_path, memmap=False) as hdul:
                    data = np.asarray(hdul[0].data, dtype=float)
                # Base weight: 1 for finite pixels, 0 for NaN/Inf
                w = np.isfinite(data).astype(np.float32)

                w_path = str(output_dir / (Path(img_path).stem + ".weight.fits"))
                fits.writeto(w_path, w.astype(np.float32), overwrite=True)
                weight_images.append(w_path)
            except Exception as e:
                # If weight generation fails, proceed without weights (legacy behaviour)
                log_warning_from_exception(
                    self.logger, f"Could not build SWarp weight map for {img_path}", e
                )
                weight_images = []
                break

        # --- Prepare output paths ---
        xml_file = str(output_dir / "swarp.xml")
        resample_dir = str(output_dir / "resampled")
        Path(resample_dir).mkdir(parents=True, exist_ok=True)

        # # --- Move .head files to head_path if specified ---
        # if head_path:
        #     head_path = Path(head_path)
        #     head_path.mkdir(parents=True, exist_ok=True)
        #     for img in input_images:
        #         head_file = Path(img).with_suffix('.head')
        #         if head_file.exists():
        #             try:
        #                 shutil.move(str(head_file), str(head_path / head_file.name))
        #             except Exception:
        #                 pass

        # --- Build SWarp command ---
        cmd = [
            swarp_cmd,
            *input_images,
            "-c",
            config_file,
            "-IMAGEOUT_NAME",
            output_image,
            "-RESAMPLE_DIR",
            resample_dir,
            "-XML_NAME",
            xml_file,
            "-NTHREADS",
            str(final_config["NTHREADS"]),
        ]

        # Enable weight-map propagation to preserve NaN/no-data regions.
        if weight_images:
            cmd.extend(
                [
                    "-WEIGHT_TYPE",
                    "MAP_WEIGHT",
                    "-WEIGHT_IMAGE",
                    ",".join(weight_images),
                ]
            )

        # # --- Add HEAD_PATH to SWarp command if specified ---
        # if head_path:
        #     cmd.extend(['-HEAD_PATH', str(head_path)])

        # --- Logging ---
        if self.verbose_level >= 2:
            import shlex

            self.logger.debug("Output image: %s", output_image)
            self.logger.debug("Running SWarp...\n%s", shlex.join(cmd))

        def _cleanup_swarp_outputs():
            """Remove SWarp output files on failure so the directory is clean for fallback/retry."""
            to_remove = [
                output_image,
                output_weights,
                config_file,
                xml_file,
            ]
            for p in to_remove:
                try:
                    if p and os.path.isfile(p):
                        os.remove(p)
                        self.logger.debug("Removed SWarp output: %s", p)
                except OSError as e:
                    self.logger.debug("Could not remove %s: %s", p, e)
            try:
                resample_path = Path(resample_dir)
                if resample_path.is_dir():
                    for f in resample_path.iterdir():
                        try:
                            f.unlink() if f.is_file() else shutil.rmtree(f)
                        except OSError as e:
                            self.logger.debug("Could not remove %s: %s", f, e)
            except Exception as e:
                self.logger.debug(
                    "Could not clean resample dir %s: %s", resample_dir, e
                )

        # --- Execute SWarp ---
        result = None
        try:
            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=300,
                )
            self.clean_log(log_file)
        except subprocess.TimeoutExpired:
            with open(log_file, "a") as f:
                f.write("\nERROR: SWarp execution timed out after 300 seconds\n")
            self.logger.warning("SWarp timed out (300 s). See %s", log_file)
            _cleanup_swarp_outputs()
            return None
        except Exception as e:
            log_warning_from_exception(
                self.logger, f"SWarp execution failed (see {log_file})", e
            )
            _cleanup_swarp_outputs()
            return None

        # --- Handle single input case ---
        if len(input_images) == 1:
            resamp_files = list(Path(resample_dir).glob("*.resamp.fits"))
            if resamp_files:
                output_image = str(resamp_files[0])
                if self.verbose_level >= 2:
                    self.logger.info(f"Using resampled file: {output_image}")

        # --- Check for errors ---
        if result.returncode != 0:
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                tail = log_content.strip().splitlines()[-5:] if log_content else []
                self.logger.warning(
                    "SWarp failed (return code %s). Last lines of %s:\n%s",
                    result.returncode,
                    log_file,
                    "\n".join(tail),
                )
            except Exception:
                self.logger.warning(
                    "SWarp failed with return code %s. See %s",
                    result.returncode,
                    log_file,
                )
            _cleanup_swarp_outputs()
            return None

        # ------------------------------------------------------------------
        # Re-impose NaNs on uncovered regions using SWarp weights.
        #
        # Some SWarp builds still emit exact 0.0 in no-data regions even when
        # BLANK_BADPIXELS=Y and FILL_VALUE=NAN. The robust indicator of coverage
        # is the corresponding weight map: weight <= 0 (or non-finite) means the
        # output pixel has no valid contributing input data.
        # ------------------------------------------------------------------
        def _apply_weight_nan_mask(resamp_fits: Path, weight_fits: Path) -> None:
            try:
                if not resamp_fits.is_file() or not weight_fits.is_file():
                    return
                with fits.open(resamp_fits, mode="update", memmap=False) as hdul_img:
                    img = np.asarray(hdul_img[0].data, dtype=float)
                    if img.ndim != 2:
                        return
                    with fits.open(weight_fits, mode="readonly", memmap=False) as hdul_w:
                        w = np.asarray(hdul_w[0].data, dtype=float)
                    if w.shape != img.shape:
                        return
                    bad = (~np.isfinite(w)) | (w <= 0)
                    if not np.any(bad):
                        return
                    n_before = int(np.count_nonzero(~np.isfinite(img)))
                    img[bad] = np.nan
                    hdul_img[0].data = img.astype(np.float32, copy=False)
                    hdul_img.flush()
                    n_after = int(np.count_nonzero(~np.isfinite(img)))
                    if self.verbose_level >= 2:
                        self.logger.debug(
                            "SWarp NaN re-mask: %s using %s (NaN/inf %d -> %d; bad=%d)",
                            str(resamp_fits),
                            str(weight_fits),
                            n_before,
                            n_after,
                            int(np.count_nonzero(bad)),
                        )
            except Exception as e:
                log_warning_from_exception(
                    self.logger,
                    f"Could not re-mask no-data pixels in {resamp_fits.name}",
                    e,
                )

        try:
            resamp_files = list(Path(resample_dir).glob("*.resamp.fits"))
            # Find per-resampled weight files if present; else fall back to SWarp WEIGHTOUT_NAME.
            fallback_weight = Path(output_weights)
            for rf in resamp_files:
                # Common SWarp naming conventions:
                #   image.resamp.fits -> image.resamp.weight.fits
                #   image.resamp.fits -> image.weight.fits
                candidates = [
                    rf.with_name(rf.name.replace(".resamp.fits", ".resamp.weight.fits")),
                    rf.with_name(rf.name.replace(".resamp.fits", ".weight.fits")),
                ]
                # Also allow any weight file sharing the same stem prefix.
                try:
                    candidates += list(rf.parent.glob(rf.stem + "*.weight*.fits"))
                except Exception:
                    pass

                picked = None
                for c in candidates:
                    if c.is_file():
                        picked = c
                        break
                if picked is None and fallback_weight.is_file():
                    picked = fallback_weight
                if picked is not None:
                    _apply_weight_nan_mask(rf, picked)
        except Exception:
            pass

        return {
            "output_dir": str(output_dir),
            "corrected_image": output_image,
            "weight_image": output_weights,
            "resampled_dir": resample_dir,
            "log_file": log_file,
            "config": final_config,
        }

    def run_scamp(
        self,
        catalog_paths,
        reference_cat: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Run SCAMP on one or more LDAC catalogs in a single call.

        Parameters
        ----------
        catalog_paths : str or list of str
            One or more LDAC catalog paths to pass to SCAMP. When multiple
            catalogs are supplied, SCAMP solves them together on the same
            astrometric grid and produces one .head file per catalog.
        reference_cat : str, optional
            Path to the astrometric reference catalog (ASTREF_CATALOG=FILE).
            When None, GAIA-DR3 is used.
        """
        if isinstance(catalog_paths, (str, Path)):
            catalog_paths = [str(catalog_paths)]
        else:
            catalog_paths = [str(p) for p in catalog_paths]

        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        stem = Path(catalog_paths[0]).stem

        # Read scamp_distort_degrees from wcs config for consistency with main pipeline
        iy = getattr(self, "input_yaml", None) or {}
        wcs_cfg = iy.get("wcs", {}) if isinstance(iy, dict) else {}
        distort_degrees = int(wcs_cfg.get("scamp_distort_degrees", 4))

        # ------------------------------------------------------------------
        # GAIA-DR3 catalog caching
        # ------------------------------------------------------------------
        gaia_cache_key: Optional[str] = None
        gaia_cache_hit = False
        gaia_temp_dir: Optional[Path] = None
        cached_gaia_path: Optional[Path] = None

        if reference_cat is None:
            gaia_cache_key = self._compute_gaia_cache_key(catalog_paths[0])
            cached_gaia_path = self._find_cached_gaia_catalog(gaia_cache_key)
            if cached_gaia_path is not None:
                gaia_cache_hit = True
                reference_cat = str(cached_gaia_path)
                self.logger.info(
                    "GAIA cache hit: using cached catalog %s", cached_gaia_path
                )

        final_config = {
            **self.DEFAULT_SCAMP_CONFIG,
            "DISTORT_DEGREES": distort_degrees,
            "ASTREF_CATALOG": "FILE" if reference_cat else "GAIA-DR3",
            "ASTREFCAT_NAME": reference_cat,
            "NTHREADS": self.default_threads,
            **(config or {}),
        }
        # COMBINE is a SWarp keyword, not a SCAMP keyword. Do not pass it to SCAMP.
        final_config.pop("COMBINE", None)
        final_config = {k: v for k, v in final_config.items() if v is not None}

        # If downloading GAIA (cache miss), ask SCAMP to save it for next time.
        if reference_cat is None and gaia_cache_key is not None:
            gaia_temp_dir = Path(mkdtemp(prefix="scamp_gaia_"))
            final_config["SAVE_REFCATALOG"] = "Y"
            final_config["REFOUT_CATPATH"] = str(gaia_temp_dir)
            self.logger.info(
                "GAIA cache miss: downloading and caching with key %s", gaia_cache_key
            )

        config_file = (
            str(output_path / f"{stem}_default.scamp")
            if output_path
            else f"{stem}_default.scamp"
        )
        with open(config_file, "w") as f:
            for k, v in final_config.items():
                f.write(f"{k}\t{v}\n")

        xml_file = (
            str(output_path / f"{stem}_scamp.xml")
            if output_path
            else f"{stem}_scamp.xml"
        )
        log_file = (
            str(output_path / f"{stem}_scamp.log")
            if output_path
            else f"{stem}_scamp.log"
        )

        cmd = [
            self._check_executable("scamp"),
            *catalog_paths,
            "-c",
            config_file,
            "-XML_NAME",
            xml_file,
            "-NTHREADS",
            str(final_config["NTHREADS"]),
        ]

        if self.verbose_level >= 2:
            import shlex

            self.logger.info(f"Running SCAMP: {shlex.join(cmd)}")

        def _clean_scamp_outputs():
            if output_path:
                # When preserve_scamp_logs is True, keep the log file for debugging
                to_remove = [config_file, xml_file]
                if not getattr(self, "preserve_scamp_logs", False):
                    to_remove.append(log_file)
                for p in to_remove + list(Path(output_path).glob("*.head")):
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError as e:
                        self.logger.debug("Could not remove SCAMP output %s: %s", p, e)
            if gaia_temp_dir and gaia_temp_dir.exists():
                try:
                    shutil.rmtree(gaia_temp_dir)
                except OSError as e:
                    self.logger.debug("Could not remove GAIA temp dir %s: %s", gaia_temp_dir, e)

        try:
            with open(log_file, "w") as log_f:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=180,
                )
            self.clean_log(log_file)
        except subprocess.TimeoutExpired:
            self.logger.error("SCAMP execution timed out after 180 seconds")
            _clean_scamp_outputs()
            raise

        # Check if log file exists and read it
        if not os.path.isfile(log_file):
            self.logger.warning(
                f"SCAMP log file not created at {log_file}; catalog may be invalid"
            )
            _clean_scamp_outputs()
            return None

        with open(log_file) as log_f:
            log_content = log_f.read()

        # Log SCAMP return code and key messages for debugging
        self.logger.info(
            f"SCAMP return code: {result.returncode}, log file: {log_file}"
        )
        if "Not enough matched detections" in log_content:
            self.logger.warning("SCAMP: Not enough matched detections in catalog")
        if "FATAL ERROR" in log_content:
            self.logger.warning(f"SCAMP: FATAL ERROR in log: {log_content[:500]}")

        if result.returncode != 0 or "Not enough matched detections" in log_content:
            self.logger.warning(
                f"SCAMP failed (code {result.returncode}). See {log_file}"
            )
            return None

        # After successful SCAMP run, save downloaded GAIA catalog to cache.
        if gaia_temp_dir is not None and gaia_cache_key is not None:
            saved_cats = list(gaia_temp_dir.glob("*.cat"))
            if saved_cats:
                self._save_gaia_catalog_to_cache(str(saved_cats[0]), gaia_cache_key)
            else:
                self.logger.debug("SCAMP did not save a reference catalog to %s", gaia_temp_dir)
            # Clean up temp dir
            try:
                shutil.rmtree(gaia_temp_dir)
            except OSError as e:
                self.logger.debug("Could not remove GAIA temp dir %s: %s", gaia_temp_dir, e)
            gaia_temp_dir = None

        distortion = self._parse_scamp_xml(xml_file)

        # Fallback: parse SCAMP log for residual info if XML parsing failed
        if distortion is None or (
            distortion.get("astrometric_rms_arcsec") is None
            and distortion.get("n_matched_stars") is None
        ):
            try:
                import re
                _log_rms = None
                _log_nstars = None
                _in_external_stats = False
                for line in log_content.splitlines():
                    if "Astrometric stats (external)" in line:
                        _in_external_stats = True
                        continue
                    if _in_external_stats and line.strip().startswith("Group"):
                        # Parse: Group  1:  0.291" 0.00752"   0.12       8 ...
                        # Strip "Group" prefix and colon, then extract numbers
                        line_clean = line.replace("Group", "").replace(":", "")
                        parts = line_clean.split()
                        vals = []
                        for p in parts:
                            p_clean = p.rstrip('"').strip()
                            try:
                                vals.append(float(p_clean))
                            except ValueError:
                                pass
                        # Expect: [group_num, dAXIS1, dAXIS2, chi2, nstars, ...]
                        if len(vals) >= 5:
                            _log_rms = float(np.sqrt(vals[1]**2 + vals[2]**2))
                            _log_nstars = int(vals[4])
                        _in_external_stats = False
                if _log_rms is not None or _log_nstars is not None:
                    self.logger.info(
                        "SCAMP log fallback: residual RMS=%.4f\", matched stars=%d",
                        _log_rms if _log_rms is not None else -1,
                        _log_nstars if _log_nstars is not None else -1,
                    )
                    if distortion is None:
                        distortion = {
                            "distortion_coeffs": {},
                            "wcs_params": {},
                            "astrometric_rms_arcsec": _log_rms,
                            "n_matched_stars": _log_nstars,
                        }
                    else:
                        if distortion.get("astrometric_rms_arcsec") is None:
                            distortion["astrometric_rms_arcsec"] = _log_rms
                        if distortion.get("n_matched_stars") is None:
                            distortion["n_matched_stars"] = _log_nstars
            except Exception:
                pass

        # Map each input catalog stem to its output .head file.
        # SCAMP names .head files after the catalog stem (same as the FITS stem).
        head_files_found = list(Path(output_path).glob("*.head")) if output_path else []
        head_files_by_stem = {p.stem: str(p) for p in head_files_found}

        # Legacy single-catalog callers expect a "head_file" key.
        head_file = str(head_files_found[0]) if head_files_found else None

        return {
            "output_dir": str(output_path) if output_path else None,
            "xml_file": xml_file,
            "log_file": log_file,
            "head_file": head_file,
            "head_files_by_stem": head_files_by_stem,
            "distortion": distortion,
            "config": final_config,
        }

    def run_scamp_swarp(
        self,
        cat_path,
        ref_cat_path,
        image_path,
        out_dir,
        is_undersampled,
        fwhm_pixels,
        image_type,
        scamp_config,
        swarp_config,
        target_ra=None,
        target_dec=None,
    ):
        """Run SCAMP then SWarp to align an image to a reference.

        Args:
            cat_path: Path to SExtractor catalog for the image to align.
            ref_cat_path: Path to reference catalog.
            image_path: Path to the FITS image.
            out_dir: Output directory for alignment products.
            is_undersampled: Whether the image is undersampled.
            fwhm_pixels: FWHM in pixels.
            image_type: Type of image (e.g., 'science', 'reference').
            scamp_config: SCAMP configuration dict.
            swarp_config: SWarp configuration dict.
            target_ra: Optional target RA (degrees).
            target_dec: Optional target Dec (degrees).
        """
        # Run SCAMP to compute WCS solution for image_path aligned to ref_cat_path.
        # run_scamp signature: (catalog_path, reference_cat, output_dir, config)
        scamp_res = self.run_scamp(
            catalog_path=cat_path,
            reference_cat=ref_cat_path,
            output_dir=str(out_dir),
            config=scamp_config,
        )
        if scamp_res is None:
            self.logger.info("SCAMP failed. Check logs for details.")
            return None
        head_file = Path(scamp_res.get("head_file", ""))

        if not head_file.exists() or head_file.stat().st_size == 0:
            self.logger.info(f"SCAMP did not produce a valid .head file at {head_file}")
            return None

        self.logger.debug("SCAMP produced .head file: %s", head_file)

        # SWarp looks for <stem>.head next to the input FITS to pick up the SCAMP WCS.
        # Copy the .head file to sit next to image_path so SWarp finds it automatically.
        image_head_dst = Path(image_path).with_suffix(".head")
        try:
            if head_file.resolve() != image_head_dst.resolve():
                shutil.copy2(str(head_file), str(image_head_dst))
                self.logger.info(
                    "Copied SCAMP .head to %s for SWarp to use.", image_head_dst
                )
            _normalize_head_file(image_head_dst)
        except Exception as _he:
            log_warning_from_exception(
                self.logger, f"Could not copy SCAMP .head next to {image_path}", _he
            )

        self.logger.debug("Running SWarp on %s", image_path)
        swarp_res = self.run_swarp(
            [str(image_path)],
            scamp_results=scamp_res,
            output_dir=str(out_dir),
            config=swarp_config,
        )
        if swarp_res is None:
            self.logger.info("SWarp failed. Check logs for details.")
            return None
        aligned_image = swarp_res["corrected_image"]
        self.clean_image(Path(aligned_image))
        return aligned_image

    def clean_image(self, path: Path):
        """Replace non-finite values (NaN/inf) in a FITS image with NaN.

        Note: exact zeros are NOT replaced — zero is a valid pixel value on
        background-subtracted images and replacing it with NaN would create
        artificial holes in the data that corrupt downstream subtraction.
        """
        with fits.open(path, mode="update") as hdul:
            data = np.asarray(hdul[0].data, dtype=float)
            data[~np.isfinite(data)] = np.nan
            hdul[0].data = data
            hdul.flush()

    # ------------------------------ Lifecycle -------------------------------
    def cleanup(self) -> None:
        """Remove all temporary directories created by this instance."""
        for temp_dir in list(self._temp_dirs):
            try:
                p = Path(temp_dir)
                if p.exists():
                    import shutil as _shutil

                    _shutil.rmtree(p)
            except Exception as e:
                self.logger.info(
                    f"Failed to remove temporary directory {temp_dir}: {e}"
                )
            finally:
                self._temp_dirs.discard(temp_dir)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------- Matched sources (1<->1) -------------------------
    def filter_well_defined_positions(
        self, catalog: Table, max_position_error_arcsec: float = 3
    ) -> Table:
        """
        Filter SExtractor catalog to keep only sources with well-defined positions.

        Parameters:
        - catalog: SExtractor catalog (astropy Table).
        - max_position_error_arcsec: Maximum allowed positional error in arcseconds.

        Returns:
        - filtered_catalog: Table with only sources that have well-defined positions.
        """
        good = np.ones(len(catalog), dtype=bool)
        if "ERRA_WORLD" in catalog.colnames and "ERRB_WORLD" in catalog.colnames:
            positional_error = (
                np.sqrt(catalog["ERRA_WORLD"] ** 2 + catalog["ERRB_WORLD"] ** 2) * 3600
            )
            good &= positional_error <= max_position_error_arcsec
        return catalog[good]

    def plot_matched_sources_side_by_side(
        self,
        sci_image_path: str,
        ref_image_path: str,
        sci_cat_path: str,
        ref_cat_path: str,
        output_plot_path: str = "matched_sources_side_by_side.png",
        label_color: str = "#FF0000",
        label_fontsize: int = 8,
        figsize: Optional[tuple] = None,
        cmap: str = "gray",
        draw_lines: bool = True,
        line_color: str = "#FF0000",
        line_alpha: float = 0.5,
        line_width: float = 0.5,
        circle_radius_sci: float = 5.0,
        circle_radius_ref: float = 5.0,
        matched_circle_color: str = "#FF0000",
        unmatched_circle_color: str = "blue",
        circle_alpha: float = 0.7,
        circle_edge_color: str = "none",
        circle_color: str = "none",
        circle_edge_width: float = 0.5,
        max_sources: int = 100,
        selection_mode: str = "uniform",  # "uniform", "random", or "first"
        random_seed: Optional[int] = 42,  # Seed for reproducible random selection
        **imshow_kwargs,
    ):
        try:
            import gc

            gc.collect()
            if figsize is None:
                from functions import set_size

                # Original default was (12, 6): height/width = 0.5
                # set_size uses height/width = aspect / golden_ratio
                golden_ratio = (5**0.5 + 1) / 2
                figsize = set_size(540, aspect=0.5 * golden_ratio)

            with (
                fits.open(sci_cat_path) as sci_hdul,
                fits.open(ref_cat_path) as ref_hdul,
            ):
                sci_cat = Table(sci_hdul[2].data)
                ref_cat = Table(ref_hdul[2].data)
            with (
                fits.open(sci_image_path) as sci_hdul,
                fits.open(ref_image_path) as ref_hdul,
            ):
                sci_data = sci_hdul[0].data.astype(np.float32)
                ref_data = ref_hdul[0].data.astype(np.float32)
            # Track if we rebin and the scale factor
            rebin_scale = 1.0
            if sci_data.size > 5e6:
                from astropy.nddata import block_reduce

                sci_data = block_reduce(sci_data, block_size=(4, 4), func=np.mean)
                ref_data = block_reduce(ref_data, block_size=(4, 4), func=np.mean)
                rebin_scale = 0.25  # Coordinates must be scaled by 1/4
                logging.info(f"Rebinned images by 4x for display, coordinates will be scaled by {rebin_scale}")
            
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, constrained_layout=True
            )
            zscale_sci = ZScaleInterval()
            zscale_ref = ZScaleInterval()
            vmin_sci, vmax_sci = zscale_sci.get_limits(sci_data)
            vmin_ref, vmax_ref = zscale_ref.get_limits(ref_data)
            cmap_img = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
            try:
                cmap_img = cmap_img.copy()
            except Exception:
                pass
            try:
                cmap_img.set_bad(color="white")
            except Exception:
                pass
            im1 = ax1.imshow(
                sci_data,
                cmap=cmap_img,
                vmin=vmin_sci,
                vmax=vmax_sci,
                **imshow_kwargs,
                origin="lower",
            )
            ax1.set_title("Science Image")
            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Science counts", fontsize=7)
            cbar1.ax.tick_params(labelsize=6)
            ax1.set_xlabel("X [Pixel]")
            ax1.set_ylabel("Y [Pixel]")
            im2 = ax2.imshow(
                ref_data,
                cmap=cmap_img,
                vmin=vmin_ref,
                vmax=vmax_ref,
                **imshow_kwargs,
                origin="lower",
            )
            ax2.set_title("Reference Image")
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Reference counts", fontsize=7)
            cbar2.ax.tick_params(labelsize=6)
            ax2.set_xlabel("X [Pixel]")
            ax2.set_ylabel("Y [Pixel]")

            def int_to_label(i):
                if i < 26:
                    return chr(65 + i)
                else:
                    return f"{chr(65 + (i // 26) - 1)}{chr(65 + (i % 26))}"

            def select_sources_spatially(catalog, max_sources, image_shape, selection_mode="uniform", random_seed=42):
                """Select sources across the image using specified selection mode."""
                if len(catalog) <= max_sources:
                    return catalog
                
                # Set random seed for reproducibility
                if random_seed is not None:
                    np.random.seed(random_seed)
                
                # Get valid sources with positions
                valid_sources = []
                for row in catalog:
                    if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames:
                        x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                        if np.isfinite(x) and np.isfinite(y):
                            valid_sources.append(row)
                
                if len(valid_sources) <= max_sources:
                    return valid_sources
                
                if selection_mode == "first":
                    # Original behavior: take first N sources
                    return valid_sources[:max_sources]
                elif selection_mode == "random":
                    # Random selection across entire image
                    np.random.shuffle(valid_sources)
                    return valid_sources[:max_sources]
                elif selection_mode == "uniform":
                    # True uniform spatial grid sampling - ensure coverage across entire image
                    img_h, img_w = image_shape
                    
                    # Create a grid that covers the entire image
                    n_grid = int(np.ceil(np.sqrt(max_sources)))
                    grid_spacing_x = img_w / n_grid
                    grid_spacing_y = img_h / n_grid
                    
                    selected_sources = []
                    grid_cells = {}  # Dictionary to store sources per grid cell
                    
                    # Assign sources to grid cells
                    for source in valid_sources:
                        x, y = source["XWIN_IMAGE"], source["YWIN_IMAGE"]
                        grid_x = min(int(x / grid_spacing_x), n_grid - 1)
                        grid_y = min(int(y / grid_spacing_y), n_grid - 1)
                        cell_key = (grid_x, grid_y)
                        
                        if cell_key not in grid_cells:
                            grid_cells[cell_key] = []
                        grid_cells[cell_key].append(source)
                    
                    # Create all possible grid cells (even empty ones)
                    all_grid_cells = [(gx, gy) for gx in range(n_grid) for gy in range(n_grid)]
                    
                    # First, try to select one source from each grid cell that has sources
                    selected_count = 0
                    for cell_key in all_grid_cells:
                        if selected_count >= max_sources:
                            break
                        
                        if cell_key in grid_cells and len(grid_cells[cell_key]) > 0:
                            # Randomly select one source from this cell
                            np.random.shuffle(grid_cells[cell_key])
                            selected_sources.append(grid_cells[cell_key][0])
                            selected_count += 1
                    
                    # If we still need more sources, fill remaining slots from populated cells
                    if selected_count < max_sources:
                        remaining_needed = max_sources - selected_count
                        
                        # Create a list of all remaining sources
                        remaining_sources = []
                        for cell_key in all_grid_cells:
                            if cell_key in grid_cells:
                                cell_sources = grid_cells[cell_key]
                                if len(cell_sources) > 1:  # Skip the one we already took
                                    remaining_sources.extend(cell_sources[1:])
                        
                        # Randomly select from remaining sources
                        if remaining_sources:
                            np.random.shuffle(remaining_sources)
                            additional_needed = min(remaining_needed, len(remaining_sources))
                            selected_sources.extend(remaining_sources[:additional_needed])
                            selected_count += additional_needed
                    
                    # If still not enough (sparse catalog), add from any available sources
                    if selected_count < max_sources:
                        remaining_needed = max_sources - selected_count
                        used_sources = set(selected_sources)
                        available_sources = [s for s in valid_sources if s not in used_sources]
                        
                        if available_sources:
                            np.random.shuffle(available_sources)
                            additional_needed = min(remaining_needed, len(available_sources))
                            selected_sources.extend(available_sources[:additional_needed])
                    
                    return selected_sources[:max_sources]
                else:
                    # Default to uniform if invalid mode
                    return select_sources_spatially(catalog, max_sources, image_shape, "uniform", random_seed)

            sci_positions = []
            sci_h, sci_w = sci_data.shape
            
            # Select science sources using specified selection mode
            logging.info(f"Selecting {max_sources} science sources using '{selection_mode}' mode")
            sci_selected = select_sources_spatially(sci_cat, max_sources, (sci_h, sci_w), selection_mode, random_seed)
            for i, row in enumerate(sci_selected):
                if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames:
                    x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                    # Scale coordinates if image was rebinned, then convert to 0-based
                    x_scaled = x * rebin_scale
                    y_scaled = y * rebin_scale
                    x_0based, y_0based = x_scaled - 1, y_scaled - 1
                    
                    # Validate coordinates are within image bounds
                    if not (0 <= x_0based < sci_w and 0 <= y_0based < sci_h):
                        logging.debug(f"Science source {i} out of bounds: ({x_0based:.1f}, {y_0based:.1f}) vs image ({sci_w}, {sci_h})")
                        continue
                    
                    sci_positions.append((x_0based, y_0based))
                    color = (
                        matched_circle_color
                        if "MATCH_ID" in row.colnames
                        else unmatched_circle_color
                    )
                    circle = Circle(
                        (x_0based, y_0based),
                        circle_radius_sci * rebin_scale,  # Scale circle radius too
                        facecolor=circle_color,
                        alpha=circle_alpha,
                        edgecolor=color,
                        linewidth=circle_edge_width,
                    )
                    ax1.add_patch(circle)
                    ax1.text(
                        x_0based,
                        y_0based - circle_radius_sci * rebin_scale - 2,
                        int_to_label(i),
                        color=label_color,
                        fontsize=label_fontsize,
                        ha="center",
                        va="top",
                    )
            
            ref_positions = []
            ref_h, ref_w = ref_data.shape
            
            # Select reference sources using specified selection mode
            logging.info(f"Selecting {max_sources} reference sources using '{selection_mode}' mode")
            ref_selected = select_sources_spatially(ref_cat, max_sources, (ref_h, ref_w), selection_mode, random_seed)
            for i, row in enumerate(ref_selected):
                if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames:
                    x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                    # Scale coordinates if image was rebinned, then convert to 0-based
                    x_scaled = x * rebin_scale
                    y_scaled = y * rebin_scale
                    x_0based, y_0based = x_scaled - 1, y_scaled - 1
                    
                    # Validate coordinates are within image bounds
                    if not (0 <= x_0based < ref_w and 0 <= y_0based < ref_h):
                        logging.debug(f"Reference source {i} out of bounds: ({x_0based:.1f}, {y_0based:.1f}) vs image ({ref_w}, {ref_h})")
                        continue
                    
                    ref_positions.append((x_0based, y_0based))
                    color = (
                        matched_circle_color
                        if "MATCH_ID" in row.colnames
                        else unmatched_circle_color
                    )
                    circle = Circle(
                        (x_0based, y_0based),
                        circle_radius_ref * rebin_scale,  # Scale circle radius too
                        facecolor=circle_color,
                        alpha=circle_alpha,
                        edgecolor=color,
                        linewidth=circle_edge_width,
                    )
                    ax2.add_patch(circle)
                    ax2.text(
                        x_0based,
                        y_0based - circle_radius_ref * rebin_scale - 2,
                        int_to_label(i),
                        color=label_color,
                        fontsize=label_fontsize,
                        ha="center",
                        va="top",
                    )
            
            # Handle remaining sources that exceed the limit
            def plot_remaining_sources(ax, catalog, image_shape, rebin_scale, max_sources, 
                                     selected_count, catalog_type="sources"):
                """Plot remaining sources as small crosses and add message."""
                
                # Get all valid sources
                all_valid_sources = []
                for row in catalog:
                    if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames:
                        x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                        if np.isfinite(x) and np.isfinite(y):
                            all_valid_sources.append(row)
                
                remaining_sources = len(all_valid_sources) - selected_count
                
                if remaining_sources > 0:
                    # Add message in top right
                    ax.text(0.98, 0.98, f"+{remaining_sources} more {catalog_type}", 
                           transform=ax.transAxes, fontsize=8, 
                           ha='right', va='top', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgrey', alpha=0.7),
                           zorder=20)
                    
                    # Plot remaining sources as small crosses
                    img_h, img_w = image_shape
                    cross_size = 3 * rebin_scale  # Small cross size
                    
                    for row in all_valid_sources[max_sources:]:
                        x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                        # Scale coordinates if image was rebinned, then convert to 0-based
                        x_scaled = x * rebin_scale
                        y_scaled = y * rebin_scale
                        x_0based, y_0based = x_scaled - 1, y_scaled - 1
                        
                        # Validate coordinates are within image bounds
                        if not (0 <= x_0based < img_w and 0 <= y_0based < img_h):
                            continue
                        
                        # Draw small cross
                        ax.plot([x_0based - cross_size, x_0based + cross_size], 
                               [y_0based, y_0based], 'r-', linewidth=0.5, alpha=0.6, zorder=3)
                        ax.plot([x_0based, x_0based], 
                               [y_0based - cross_size, y_0based + cross_size], 
                               'r-', linewidth=0.5, alpha=0.6, zorder=3)
                
                return remaining_sources
            
            # Plot remaining sources for science image
            sci_remaining = plot_remaining_sources(ax1, sci_cat, (sci_h, sci_w), rebin_scale, 
                                                  max_sources, len(sci_selected), "science sources")
            
            # Plot remaining sources for reference image  
            ref_remaining = plot_remaining_sources(ax2, ref_cat, (ref_h, ref_w), rebin_scale,
                                                  max_sources, len(ref_selected), "reference sources")
            
            total_sources_sci = len([row for row in sci_cat 
                                    if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames
                                    and np.isfinite(row["XWIN_IMAGE"]) and np.isfinite(row["YWIN_IMAGE"])])
            total_sources_ref = len([row for row in ref_cat
                                    if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames  
                                    and np.isfinite(row["XWIN_IMAGE"]) and np.isfinite(row["YWIN_IMAGE"])])
            
            logging.info(f"Science: {len(sci_selected)} plotted + {sci_remaining} crosses = {total_sources_sci} total")
            logging.info(f"Reference: {len(ref_selected)} plotted + {ref_remaining} crosses = {total_sources_ref} total")
            logging.info(f"Plotted {len(sci_positions)} science and {len(ref_positions)} reference sources within image bounds")
            
            # `constrained_layout=True` keeps colorbars and labels from
            # overlapping, so no additional tight_layout is needed.
            plt.savefig(output_plot_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            gc.collect()
        except Exception as e:
            import traceback

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            logging.info(
                f"\n\n Issue creating side by side plot \n"
                f"\n"
                f"Type     : {exc_type.__name__}\n"
                f"File     : {fname}\n"
                f"Line     : {lineno}\n"
                f"Message  : {str(e)}\n"
                f"\n" + traceback.format_exc()
            )

    def filter_matched_sources(
        self,
        sci_cat_path: str,
        ref_cat_path: str,
        match_radius_arcsec: float = 5.0,
        nmax: int = 1e6,
        sci_image_path: str = None,
        ref_image_path: str = None,
    ) -> tuple[int, float]:
        """
        Build 1-to-1 matched catalogs using SNR_APER >= 3, adding MAG_APER, MAGERR_APER, and SNR_APER columns.
        Uses RANSAC for alignment matching and spatial downsampling if too many sources.
        Prioritizes extended sources for alignment.
        Returns the number of matched sources and the match radius used.
        """
        if not all([sci_cat_path, ref_cat_path, sci_image_path, ref_image_path]):
            raise ValueError(
                "All paths (sci_cat_path, ref_cat_path, sci_image_path, ref_image_path) are required."
            )
        nmax = int(nmax)
        if nmax <= 0:
            raise ValueError("nmax must be positive.")

        def read_ldac(path: str) -> tuple[Table, fits.Header]:
            """Read LDAC catalog and return table and header."""
            with fits.open(path) as hdul:
                return Table(hdul[2].data), hdul[2].header

        def write_ldac(path: str, table: Table, header: fits.Header) -> None:
            """Write LDAC catalog with updated table."""
            with fits.open(path, mode="update") as hdul:
                hdul[2].data = table.as_array()
                hdul.flush()

        class ConstrainedSlopeRegressor(BaseEstimator, RegressorMixin):
            """Linear regressor that penalises deviation from a target slope."""

            def __init__(
                self, slope_constraint: float = 1.0, slope_tolerance: float = 0.0
            ):
                self.slope_constraint = slope_constraint
                self.slope_tolerance = slope_tolerance

            def fit(self, X, y):
                X, y = check_X_y(X, y)

                def loss(params):
                    slope, intercept = params
                    pred = slope * X.flatten() + intercept
                    mse = np.mean((y - pred) ** 2)
                    penalty = 100 * max(
                        0, abs(slope - self.slope_constraint) - self.slope_tolerance
                    )
                    return mse + penalty

                init = [1.0, float(np.mean(y) - np.mean(X))]
                result = minimize(loss, init, method="L-BFGS-B")
                self.slope_, self.intercept_ = result.x
                return self

            def predict(self, X):
                check_is_fitted(self)
                X = check_array(X)
                return self.slope_ * X.flatten() + self.intercept_

        sci_cat, sci_header = read_ldac(sci_cat_path)
        ref_cat, ref_header = read_ldac(ref_cat_path)

        # Initialize intercept before conditional blocks to prevent NameError
        intercept = None

        def add_mag_snr_aper(tbl: Table, use_magauto: bool = True) -> Table:
            """
            Add magnitude, magnitude error, and SNR columns.
            Optionally use MAG_AUTO, MAGERR_AUTO, and SNR_WIN if use_magauto is True.
            """
            if use_magauto and "MAG_AUTO" in tbl.colnames:
                # Use MAG_AUTO, MAGERR_AUTO, and SNR_WIN if available and requested
                mag = np.array(tbl["MAG_AUTO"], float)
                magerr = (
                    np.array(tbl["MAGERR_AUTO"], float)
                    if "MAGERR_AUTO" in tbl.colnames
                    else 0.02
                )
                snr = (
                    np.array(tbl["SNR_WIN"], float)
                    if "SNR_WIN" in tbl.colnames
                    else 0.0
                )
                tbl["MAG_APER"], tbl["MAGERR_APER"], tbl["SNR_APER"] = mag, magerr, snr
            elif "FLUX_APER" in tbl.colnames:
                # Fall back to FLUX_APER and FLUXERR_APER
                flux = np.array(tbl["FLUX_APER"], float)
                flux_err = (
                    np.array(tbl["FLUXERR_APER"], float)
                    if "FLUXERR_APER" in tbl.colnames
                    else np.sqrt(np.abs(flux))
                )
                mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
                magerr = np.where(
                    (flux > 0) & (flux_err > 0),
                    2.5 / np.log(10) * (flux_err / flux),
                    0.02,
                )
                snr = np.where(flux_err > 0, np.abs(flux) / flux_err, 0.0)
                tbl["MAG_APER"], tbl["MAGERR_APER"], tbl["SNR_APER"] = mag, magerr, snr
            else:
                # If neither is available, fill with NaN and default values
                tbl["MAG_APER"], tbl["MAGERR_APER"], tbl["SNR_APER"] = np.nan, 0.02, 0.0
            return tbl

        sci_cat = add_mag_snr_aper(sci_cat)
        ref_cat = add_mag_snr_aper(ref_cat)
        # SNR cut for matching: use 2.0 for alignment (lower than photometry's 3.0)
        # to retain fainter sources in sparse fields. If very few sources pass,
        # fall back to 1.5 to maximize match count for SCAMP.
        _snr_match_thresh = 2.0
        sci_mask = sci_cat["SNR_APER"] >= _snr_match_thresh
        ref_mask = ref_cat["SNR_APER"] >= _snr_match_thresh
        sci_cat_filtered = sci_cat[sci_mask]
        ref_cat_filtered = ref_cat[ref_mask]
        # Sparse-field fallback: if too few sources pass SNR >= 2.0, lower to 1.5
        if len(sci_cat_filtered) < 10 or len(ref_cat_filtered) < 10:
            _snr_match_thresh = 1.5
            sci_mask = sci_cat["SNR_APER"] >= _snr_match_thresh
            ref_mask = ref_cat["SNR_APER"] >= _snr_match_thresh
            sci_cat_filtered = sci_cat[sci_mask]
            ref_cat_filtered = ref_cat[ref_mask]
            self.logger.info(
                "Sparse field: lowered SNR match threshold to %.1f "
                "(%d sci / %d ref sources).",
                _snr_match_thresh,
                len(sci_cat_filtered),
                len(ref_cat_filtered),
            )
        if len(sci_cat_filtered) == 0 or len(ref_cat_filtered) == 0:
            sci_cat_filtered["MATCH_ID"] = np.arange(len(sci_cat_filtered), dtype=int)
            ref_cat_filtered["MATCH_ID"] = np.arange(len(ref_cat_filtered), dtype=int)
            write_ldac(sci_cat_path, sci_cat_filtered, sci_header)
            write_ldac(ref_cat_path, ref_cat_filtered, ref_header)
            return 0, match_radius_arcsec

        def get_xy(
            tbl: Table,
            prefer_win: bool = True,
            input_origin: int = 1,
            output_origin: int = 1,
        ) -> tuple[np.ndarray, np.ndarray, str]:
            """Get coordinates and convert between origin systems."""
            if (
                prefer_win
                and "XWIN_IMAGE" in tbl.colnames
                and "YWIN_IMAGE" in tbl.colnames
            ):
                x, y = tbl["XWIN_IMAGE"], tbl["YWIN_IMAGE"]
                coord_type = "WIN"
            elif "X_IMAGE" in tbl.colnames and "Y_IMAGE" in tbl.colnames:
                x, y = tbl["X_IMAGE"], tbl["Y_IMAGE"]
                coord_type = "ISO"
            else:
                raise ValueError(
                    "Neither XWIN_IMAGE/YWIN_IMAGE nor X_IMAGE/Y_IMAGE found in catalog."
                )
            if input_origin != output_origin:
                offset = input_origin - output_origin
                x = x - offset
                y = y - offset
            return x, y, coord_type


        sci_x, sci_y, sci_coord_type = get_xy(
            sci_cat_filtered, prefer_win=True, input_origin=1, output_origin=1
        )
        ref_x, ref_y, ref_coord_type = get_xy(
            ref_cat_filtered, prefer_win=True, input_origin=1, output_origin=1
        )

        def get_ra_dec(tbl: Table) -> tuple:
            if "XWIN_WORLD" in tbl.colnames and "YWIN_WORLD" in tbl.colnames:
                return tbl["XWIN_WORLD"], tbl["YWIN_WORLD"]
            if "X_WORLD" in tbl.colnames and "Y_WORLD" in tbl.colnames:
                return tbl["X_WORLD"], tbl["Y_WORLD"]
            raise ValueError(
                "Neither XWIN_WORLD/YWIN_WORLD nor X_WORLD/Y_WORLD found in catalog."
            )
            

        sci_ra, sci_dec = get_ra_dec(sci_cat_filtered)
        ref_ra, ref_dec = get_ra_dec(ref_cat_filtered)
        sci_coords = SkyCoord(sci_ra * u.deg, sci_dec * u.deg)
        ref_coords = SkyCoord(ref_ra * u.deg, ref_dec * u.deg)
        idx_sci, idx_ref, _, _ = search_around_sky(
            sci_coords, ref_coords, match_radius_arcsec * u.arcsec
        )
        best_pairs, used_ref = {}, set()
        for si, ri in zip(idx_sci, idx_ref):
            dist = sci_coords[si].separation(ref_coords[ri]).arcsec
            if si not in best_pairs or dist < best_pairs[si][1]:
                best_pairs[si] = (ri, dist)
        final_pairs = []
        for si, (ri, dist) in best_pairs.items():
            if ri not in used_ref and dist <= match_radius_arcsec:
                final_pairs.append((si, ri))
                used_ref.add(ri)
        if final_pairs:
            sci_idx, ref_idx = map(np.array, zip(*final_pairs))
            sci_cat_matched = sci_cat_filtered[sci_idx]
            ref_cat_matched = ref_cat_filtered[ref_idx]
            match_id = np.arange(len(sci_cat_matched), dtype=int)
            sci_cat_matched["MATCH_ID"] = match_id
            ref_cat_matched["MATCH_ID"] = match_id
        else:
            sci_cat_matched = sci_cat_filtered[:0]
            ref_cat_matched = ref_cat_filtered[:0]
            sci_cat_matched["MATCH_ID"] = np.array([], dtype=int)
            ref_cat_matched["MATCH_ID"] = np.array([], dtype=int)
        # Only apply RANSAC mag filtering if we have many initial matches (>= 50)
        # For fewer matches, use all matched sources to avoid over-filtering
        if len(sci_cat_matched) >= 50:
            sci_mag = np.array(sci_cat_matched["MAG_APER"], dtype=float)
            ref_mag = np.array(ref_cat_matched["MAG_APER"], dtype=float)
            valid = np.isfinite(sci_mag) & np.isfinite(ref_mag)
            if np.sum(valid) >= 10:
                sci_mag_clean = sci_mag[valid]
                ref_mag_clean = ref_mag[valid]
                intercept = None
                try:
                    # Use RANSAC for alignment matching
                    # residual_threshold=2.0 mag is more permissive to handle different zero points
                    ransac = RANSACRegressor(
                        estimator=ConstrainedSlopeRegressor(
                            slope_constraint=1.0, slope_tolerance=0.0
                        ),
                        residual_threshold=2.0,
                        max_trials=500,
                        min_samples=2,
                    )
                    ransac.fit(ref_mag_clean.reshape(-1, 1), sci_mag_clean)
                    intercept = ransac.estimator_.intercept_
                    inlier_mask_clean = ransac.inlier_mask_

                    if inlier_mask_clean.any():
                        inliers = np.zeros(len(sci_cat_matched), dtype=bool)
                        inliers[valid] = inlier_mask_clean
                        sci_cat_matched = sci_cat_matched[inliers]
                        ref_cat_matched = ref_cat_matched[inliers]
                        match_id = np.arange(len(sci_cat_matched), dtype=int)
                        sci_cat_matched["MATCH_ID"] = match_id
                        ref_cat_matched["MATCH_ID"] = match_id
                        self.logger.info(
                            f"RANSAC mag filter kept {len(sci_cat_matched)} sources, zp={intercept:.3f} mag"
                        )
                except Exception as exc:
                    self.logger.warning(f"RANSAC regression failed: {exc}")
                    # Fall back to simple median filtering with more permissive threshold
                    median_diff = np.median(sci_mag_clean - ref_mag_clean)
                    intercept = median_diff
                    inlier_mask_clean = np.abs((sci_mag_clean - ref_mag_clean) - median_diff) < 1.0
                    if inlier_mask_clean.any():
                        inliers = np.zeros(len(sci_cat_matched), dtype=bool)
                        inliers[valid] = inlier_mask_clean
                        sci_cat_matched = sci_cat_matched[inliers]
                        ref_cat_matched = ref_cat_matched[inliers]
                        match_id = np.arange(len(sci_cat_matched), dtype=int)
                        sci_cat_matched["MATCH_ID"] = match_id
                        ref_cat_matched["MATCH_ID"] = match_id
                        self.logger.info(
                            f"Median fallback kept {len(sci_cat_matched)} sources"
                        )
        else:
            # For fewer than 50 initial matches, skip RANSAC to avoid over-filtering
            self.logger.info(
                f"Skipping RANSAC mag filter (only {len(sci_cat_matched)} initial matches < 50 threshold)"
            )
            # Calculate simple intercept for plotting
            sci_mag = np.array(sci_cat_matched["MAG_APER"], dtype=float)
            ref_mag = np.array(ref_cat_matched["MAG_APER"], dtype=float)
            valid = np.isfinite(sci_mag) & np.isfinite(ref_mag)
            if np.sum(valid) >= 2:
                intercept = np.median(sci_mag[valid] - ref_mag[valid])
            else:
                intercept = None
        if len(sci_cat_matched) >= 5 and intercept is not None:
            from functions import set_size
            from plotting_utils import (
                apply_autophot_mplstyle, get_ransac_color, get_marker_size,
                get_alpha, get_line_width, ransac_grid, ransac_savefig,
                ransac_legend_top_outside, set_mag_axes_inverted_xy,
            )

            apply_autophot_mplstyle()
            fig, ax = plt.subplots(figsize=set_size(540, 1))
            sci_mag = np.array(sci_cat_matched["MAG_APER"], dtype=float)
            ref_mag = np.array(ref_cat_matched["MAG_APER"], dtype=float)
            ax.errorbar(
                ref_mag,
                sci_mag,
                xerr=np.array(ref_cat_matched["MAGERR_APER"], dtype=float),
                yerr=np.array(sci_cat_matched["MAGERR_APER"], dtype=float),
                fmt="o",
                markersize=get_marker_size('medium'),
                color=get_ransac_color('alignment'),
                ecolor="lightgrey",
                elinewidth=0.4,
                capsize=0,
                alpha=get_alpha('dark'),
                label=f"Matched [{len(sci_cat_matched)}]",
            )
            x_fit = np.linspace(np.nanmin(ref_mag), np.nanmax(ref_mag), 100)
            y_fit = x_fit + intercept
            residuals = sci_mag - (ref_mag + intercept)
            intercept_error = np.std(residuals) / np.sqrt(len(sci_mag))
            ax.fill_between(
                x_fit, y_fit - intercept_error, y_fit + intercept_error,
                color=get_ransac_color('error_band'), alpha=get_alpha('very_light'),
            )
            ax.plot(
                x_fit, y_fit,
                color=get_ransac_color('fit'),
                linestyle="--",
                lw=get_line_width('medium'),
                label=f"Fit: slope=1, intercept={intercept:.3f}",
            )
            ax.set_xlabel(r"Reference $m$ [mag]")
            ax.set_ylabel(r"Science $m$ [mag]")
            ransac_legend_top_outside(ax, ncol=2)
            ransac_grid(ax)
            set_mag_axes_inverted_xy(ax)
            ransac_savefig(fig, str(Path(sci_cat_path).with_suffix(".png")).replace(".png", "_Mag_Fit.png"))
            plt.close(fig)

        if len(sci_cat_matched) > nmax:
            n_bins = int(np.sqrt(nmax))
            spp = max(1, nmax // (n_bins**2))
            sci_x_bin, sci_y_bin, _ = get_xy(
                sci_cat_matched, prefer_win=True, input_origin=1, output_origin=1
            )
            x_bins = np.linspace(np.min(sci_x_bin), np.max(sci_x_bin), n_bins + 1)
            y_bins = np.linspace(np.min(sci_y_bin), np.max(sci_y_bin), n_bins + 1)
            sel = []
            for i in range(n_bins):
                for j in range(n_bins):
                    m = (
                        (sci_x_bin >= x_bins[i])
                        & (sci_x_bin < x_bins[i + 1])
                        & (sci_y_bin >= y_bins[j])
                        & (sci_y_bin < y_bins[j + 1])
                    )
                    idx = np.where(m)[0]
                    if idx.size:
                        snr_vals = sci_cat_matched["SNR_APER"][idx]
                        idx_sorted = idx[np.argsort(snr_vals)[::-1]]
                        sel.extend(idx_sorted[:spp])
            sel = np.array(sorted(set(sel)), dtype=int)[:nmax]
            sci_cat_matched = sci_cat_matched[sel]
            ref_cat_matched = ref_cat_matched[sel]
            match_id = np.arange(len(sci_cat_matched), dtype=int)
            sci_cat_matched["MATCH_ID"] = match_id
            ref_cat_matched["MATCH_ID"] = match_id
        write_ldac(sci_cat_path, sci_cat_matched, sci_header)
        write_ldac(ref_cat_path, ref_cat_matched, ref_header)
        self.logger.info(
            f"Final matched sources: {len(sci_cat_matched)}, match radius={match_radius_arcsec:.1f} arcsec"
        )
        return len(sci_cat_matched), match_radius_arcsec

    def _estimate_wcs_offset_from_headers(
        self,
        sci_image_path: str,
        ref_image_path: str,
    ) -> tuple[float, float]:
        """
        Estimate the global WCS offset between two images by comparing their
        WCS solutions at the same pixel coordinates. If both images were
        independently plate-solved, their WCS may disagree by a constant
        offset even though they cover the same sky region.

        Returns:
            tuple of (dRA_arcsec, dDec_arcsec) representing the median offset
            between the science and reference WCS at the same pixel positions.
        """
        from wcs import get_wcs as _get_wcs_for_offset

        with fits.open(sci_image_path) as hdul:
            sci_wcs = _get_wcs_for_offset(hdul[0].header)
            if sci_wcs is None:
                return np.nan, np.nan
            sci_shape = hdul[0].data.shape

        with fits.open(ref_image_path) as hdul:
            ref_wcs = _get_wcs_for_offset(hdul[0].header)
            if ref_wcs is None:
                return np.nan, np.nan

        # Sample at multiple positions across the science image
        margin = 50
        positions = [
            (margin, margin),
            (sci_shape[1] - margin, margin),
            (margin, sci_shape[0] - margin),
            (sci_shape[1] - margin, sci_shape[0] - margin),
            ((sci_shape[1] - 1) / 2, (sci_shape[0] - 1) / 2),
        ]

        dras, ddecs = [], []
        for px, py in positions:
            sky_sci = sci_wcs.pixel_to_world(px, py)
            sky_ref = ref_wcs.pixel_to_world(px, py)
            dra = (sky_sci.ra.deg - sky_ref.ra.deg) * 3600.0 * np.cos(np.radians(sky_sci.dec.deg))
            ddec = (sky_sci.dec.deg - sky_ref.dec.deg) * 3600.0
            dras.append(dra)
            ddecs.append(ddec)

        median_dra = float(np.median(dras))
        median_ddec = float(np.median(ddecs))
        offset_mag = float(np.sqrt(median_dra**2 + median_ddec**2))

        # Check consistency
        dra_std = float(np.std(dras))
        ddec_std = float(np.std(ddecs))

        self.logger.info(
            "WCS header offset: dRA=%.1f\" dDec=%.1f\" (magnitude=%.1f\", "
            "scatter: dRA_std=%.1f\" dDec_std=%.1f\")",
            median_dra, median_ddec, offset_mag, dra_std, ddec_std,
        )

        return median_dra, median_ddec

    def _estimate_wcs_offset(
        self,
        sci_cat_path: str,
        ref_cat_path: str,
    ) -> tuple[float, float, int]:
        """
        Estimate the global RA/Dec offset between two SExtractor catalogs
        when normal cross-matching fails due to large WCS discrepancies.

        For each science source, finds the nearest reference source and computes
        the separation vector (dRA, dDec). The true global offset produces a
        cluster of similar separation vectors, while random mismatches produce
        scattered vectors. Uses sigma-clipping on the offset distribution to
        find the consensus offset.

        Returns:
            tuple of (dRA_arcsec, dDec_arcsec, n_consistent) where n_consistent
            is the number of science sources whose nearest-neighbor offset is
            consistent with the consensus (within 2 sigma).
        """
        from astropy.io import fits as _fits
        from astropy.table import Table as _Table
        from astropy.coordinates import SkyCoord as _SkyCoord
        from astropy.coordinates import match_coordinates_sky as _match_sky
        import astropy.units as _u
        from astropy.stats import sigma_clip as _sigma_clip

        def _read_ldac(path):
            with _fits.open(path) as hdul:
                return _Table(hdul[2].data)

        sci_cat = _read_ldac(sci_cat_path)
        ref_cat = _read_ldac(ref_cat_path)

        if len(sci_cat) == 0 or len(ref_cat) == 0:
            return 0.0, 0.0, 0

        # Get world coordinates
        if "XWIN_WORLD" in sci_cat.colnames:
            sci_ra = np.array(sci_cat["XWIN_WORLD"], float)
            sci_dec = np.array(sci_cat["YWIN_WORLD"], float)
        else:
            sci_ra = np.array(sci_cat["X_WORLD"], float)
            sci_dec = np.array(sci_cat["Y_WORLD"], float)

        if "XWIN_WORLD" in ref_cat.colnames:
            ref_ra = np.array(ref_cat["XWIN_WORLD"], float)
            ref_dec = np.array(ref_cat["YWIN_WORLD"], float)
        else:
            ref_ra = np.array(ref_cat["X_WORLD"], float)
            ref_dec = np.array(ref_cat["Y_WORLD"], float)

        sci_coords = _SkyCoord(sci_ra * _u.deg, sci_dec * _u.deg)
        ref_coords = _SkyCoord(ref_ra * _u.deg, ref_dec * _u.deg)

        # Find nearest reference source for each science source
        idx, sep, _ = _match_sky(sci_coords, ref_coords)
        sep_arcsec = sep.arcsec

        # Compute offset vectors (sci - ref) in arcsec
        # Use the matched reference positions
        matched_ref = ref_coords[idx]
        dra = (sci_ra - np.array(matched_ref.ra.deg)) * 3600.0 * np.cos(np.radians(sci_dec))
        ddec = (sci_dec - np.array(matched_ref.dec.deg)) * 3600.0

        # Sigma-clip to find consensus offset
        dra_clipped = _sigma_clip(dra, sigma=2.5, maxiters=3)
        ddec_clipped = _sigma_clip(ddec, sigma=2.5, maxiters=3)

        # Need both to be unmasked for a consistent pair
        both_ok = ~dra_clipped.mask & ~ddec_clipped.mask

        if np.sum(both_ok) < 3:
            # Not enough consistent pairs; use median of all
            median_dra = float(np.median(dra))
            median_ddec = float(np.median(ddec))
            n_consistent = int(np.sum(both_ok))
        else:
            median_dra = float(np.median(dra[both_ok]))
            median_ddec = float(np.median(ddec[both_ok]))
            n_consistent = int(np.sum(both_ok))

        # Also compute the median separation after applying the offset
        # to verify the offset is real
        shifted_sci = _SkyCoord(
            (sci_ra - median_dra / 3600.0 / np.cos(np.radians(sci_dec))) * _u.deg,
            (sci_dec - median_ddec / 3600.0) * _u.deg,
        )
        idx2, sep2, _ = _match_sky(shifted_sci, ref_coords)
        residual_sep = float(np.median(sep2.arcsec))

        self.logger.info(
            "WCS offset estimate: dRA=%.1f\" dDec=%.1f\" (n_consistent=%d/%d, residual=%.1f\")",
            median_dra, median_ddec, n_consistent, len(sci_cat), residual_sep,
        )

        return median_dra, median_ddec, n_consistent
