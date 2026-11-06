#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WCS solver and FITS header cleaner.
Handles faint sources, cosmic rays, and edge cases.
Uses SExtractor with a Gaussian convolution filter for source detection by
default, with fallback to astrometry.net-only mode.
"""

# --- Standard Library Imports ---
import os
import re
import glob
import shutil
import logging
import warnings
import subprocess
import numpy as np
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager

# --- Third-Party Imports ---
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# --- Local Imports (optional) ---
try:
    from functions import clean_subprocess_log, strip_subprocess_noise, log_step, log_warning_from_exception, safe_fits_write, remove_wcs_from_header, copy_wcs_from_header, cap_console_lines, STATUS  # type: ignore
except (ModuleNotFoundError, ImportError):
    # Minimal fallback for environments missing the full photometry stack.
    STATUS = logging.INFO  # degrade STATUS to INFO when functions is absent
    cap_console_lines = None

    def log_step(message: str, *args, **kwargs) -> str:
        m = str(message).strip()
        if not m:
            return ""
        return f"\n---- {m} ----"

    def log_warning_from_exception(logger, message, exc, *, exc_info=False):
        logger.warning("%s: %s", message, exc, exc_info=exc_info)

    def safe_fits_write(*args, **kwargs):
        raise RuntimeError("safe_fits_write not available: functions module not found")

    def remove_wcs_from_header(header):
        raise RuntimeError("remove_wcs_from_header not available: functions module not found")

    def copy_wcs_from_header(src_header, dst_header):
        raise RuntimeError("copy_wcs_from_header not available: functions module not found")

    def clean_subprocess_log(path):
        return None

    def strip_subprocess_noise(text):
        return str(text)


def _resolve_sextractor_exe():
    """Lazy resolver for SExtractor executable to avoid circular import."""
    try:
        from utils.run_sex import get_sextractor_executable
        return get_sextractor_executable()
    except (ModuleNotFoundError, ImportError):
        return shutil.which("sex") or shutil.which("sextractor")

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
if cap_console_lines is not None:
    cap_console_lines(logging.getLogger().handlers)
logger = logging.getLogger(__name__)

# --- Suppress Astropy Warnings ---
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

# --- ANSI escape regex for log cleaning (compiled once, reused) ---
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# =============================================================================
# Helper Functions
# =============================================================================


def _wcs_cfg_float(cfg: dict, key: str, default: float) -> float:
    """
    Numeric WCS/SExtractor/SCAMP config reader.

    ``dict.get(key, default)`` does not fall back when the YAML value is ``null``
    (key present, value ``None``), which otherwise breaks ``float(None)``.
    """
    v = cfg.get(key, default)
    if v is None:
        return float(default)
    return float(v)


def _wcs_cfg_int(cfg: dict, key: str, default: int) -> int:
    """Read *key* from *cfg* as int, falling back to *default* on None."""
    v = cfg.get(key, default)
    if v is None:
        return int(default)
    return int(v)


def _wcs_cfg_bool(cfg: dict, key: str, default: bool = False) -> bool:
    """Read *key* from *cfg* as a bool, tolerating YAML bools and "true/false"
    strings.  Returns *default* when the key is absent or None."""
    v = cfg.get(key, default)
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


class AstrometryIndexError(RuntimeError):
    """solve-field reports no usable index files (empty/missing data dir)."""


@contextmanager
def silence_astropy_wcs_info():
    """
    Suppress the FITSFixedWarning / INFO log that astropy emits when SIP
    distortion keywords are present but CTYPE lacks the '-SIP' suffix.

    Astropy emits this message via two independent channels:
      1. ``warnings.warn(FITSFixedWarning)``  - caught by warnings.catch_warnings.
      2. ``logging.getLogger('astropy.wcs.wcs').info(...)`` - caught here.

    Both the child logger (astropy.wcs.wcs) and the parent (astropy.wcs) are
    silenced to WARNING to cover all astropy version paths.
    """
    _loggers = [
        logging.getLogger("astropy.wcs.wcs"),
        logging.getLogger("astropy.wcs"),
    ]
    prev_levels = [lg.level for lg in _loggers]
    for lg in _loggers:
        lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for lg, lv in zip(_loggers, prev_levels):
            lg.setLevel(lv)


def _log_scamp_vizier_failure_hint(
    logger: logging.Logger, scamp_out: str, scamp_log_fpath: str
) -> None:
    """If SCAMP log shows Vizier/network failure, log an actionable YAML hint."""
    if not scamp_out:
        return
    low = scamp_out.lower()
    if "vizier" not in low:
        return
    if not any(
        tok in low for tok in ("timeout", "connection", "error", "failed", "refused")
    ):
        return
    logger.warning(
        "SCAMP astrometric catalog fetch failed (Vizier/network).\n"
        "    Raise `wcs.scamp_ref_timeout` (AutoPHoT default 60 s; SCAMP\n"
        "    built-in is 10 s) or set `wcs.scamp_ref_server` to another\n"
        "    mirror (default is vizier.cfa.harvard.edu; alternatives\n"
        "    include vizier.unistra.fr, vizier.ast.cam.ac.uk).\n"
        "    SCAMP log: %s",
        scamp_log_fpath,
    )


def update_wcs_center(
    fits_path: str, ra: float, dec: float, overwrite: bool = True
) -> fits.Header:
    """
    Center the WCS reference pixel and set the reference sky coordinates.
    FITS is 1-indexed, so the centre pixel is (NAXIS+1)/2.

    Args:
        fits_path (str): Path to the FITS file.
        ra (float): Right Ascension for the reference pixel (degrees).
        dec (float): Declination for the reference pixel (degrees).
        overwrite (bool): Overwrite the file if it exists.

    Returns:
        fits.Header: Updated header, or None on failure.
    """
    try:
        with fits.open(fits_path, mode="update") as hdul:
            header = hdul[0].header
            image = hdul[0].data
            naxis1 = header.get("NAXIS1", image.shape[1])
            naxis2 = header.get("NAXIS2", image.shape[0])
            # FITS center pixel (1-indexed) is (naxis+1)/2
            header["CRPIX1"] = (naxis1 + 1) / 2.0
            header["CRPIX2"] = (naxis2 + 1) / 2.0
            header["CRVAL1"] = ra
            header["CRVAL2"] = dec
            hdul.flush()
            logger.info(
                "WCS header updated: CRPIX1=%.2f CRPIX2=%.2f CRVAL1=%.6f CRVAL2=%.6f",
                header["CRPIX1"],
                header["CRPIX2"],
                ra,
                dec,
            )
            return header
    except Exception as e:
        logger.exception("Failed to update WCS header: %s", e)
        return None


def table_to_ldac(table, header=None, writeto=None) -> fits.HDUList:
    """
    Converts an Astropy table into the LDAC format.

    Args:
        table (astropy.table.Table): Table to convert.
        header (fits.Header): Header to include in the output.
        writeto (str): Path to save the output.

    Returns:
        fits.HDUList: HDUList in LDAC format.
    """
    primary_hdu = fits.PrimaryHDU()
    header_str = header.tostring(endcard=True)
    header_str += fits.Header().tostring(endcard=True)
    header_col = fits.Column(
        name="Field Header Card", format=f"{len(header_str)}A", array=[header_str]
    )
    header_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([header_col]))
    header_hdu.header["EXTNAME"] = "LDAC_IMHEAD"
    data_hdu = fits.table_to_hdu(table)
    data_hdu.header["EXTNAME"] = "LDAC_OBJECTS"
    hdulist = fits.HDUList([primary_hdu, header_hdu, data_hdu])
    if writeto is not None:
        from astropy.io import fits
        # safe_fits_write only handles single image + header; write the
        # multi-extension HDUList directly.
        hdulist.writeto(writeto, overwrite=True)
    return hdulist


# --- WCS cache -------------------------------------------------------------
# Keyed by every WCS-relevant card in the header so repeated get_wcs() calls
# on an unchanged header skip re-parsing SIP/TPV distortion coefficients and
# re-validating the WCS. A changed keyword yields a different key, so stale
# entries are simply never matched.
# Module-level WCS cache.  The pipeline parallelises with processes, not
# threads, so each worker holds its own copy; there is no locking here.
_WCS_CACHE = {}
_WCS_CACHE_MAX = 32

# Header-key prefixes that feed the WCS transform.  The key must include the
# distortion families (SIP A_/B_/AP_/BP_ polynomials and *_ORDER cards, TPV
# PVi_j, TNX/D_/DP_ polynomials) - a fixed list of linear terms would let a
# post-SCAMP header that differs only in distortion coefficients reuse a WCS
# built from the pre-refinement solution.
_WCS_KEY_PREFIXES = (
    "CRPIX", "CRVAL", "CTYPE", "CUNIT",
    "CDELT", "CROTA", "CD", "PC", "PV",
    "LONPOLE", "LATPOLE", "LONGPOLE",
    "EQUINOX", "RADESYS", "RADECSYS", "RADYSYS",
    "WCSAXES", "WCSNAME", "PROJP",
    "LTV", "LTM", "SIP", "TNX",
)

# SIP / polynomial distortion coefficient keys are stems (A_ORDER, A_0_0,
# B_1_2, AP_2_0, D_1_1, DP_0_3, ...).  startswith() covers every order/index.
_WCS_KEY_STEMS = ("A_", "B_", "AP_", "BP_", "D_", "DP_")


def _wcs_cache_key(header):
    """Build a hashable cache key from every WCS-relevant card in *header*."""
    try:
        cards = []
        for key in header.keys():
            key_str = str(key).upper()
            if key_str in ("", "COMMENT", "HISTORY"):
                continue
            if key_str.startswith(_WCS_KEY_PREFIXES) or key_str.startswith(
                _WCS_KEY_STEMS
            ):
                value = header.get(key)
                # repr() keeps array/record values hashable and unambiguous.
                cards.append((key_str, repr(value)))
        return tuple(sorted(cards))
    except Exception:
        return None


def invalidate_wcs_cache():
    """Clear the WCS cache (call after WCS solving or header modification)."""
    _WCS_CACHE.clear()


def get_wcs(header: fits.Header, silent: bool = True) -> WCS:
    """
    Create a WCS object from a FITS header, handling SIP if present.
    Returns a 2D celestial WCS so it is safe for reproject and other 2D image operations.

    Args:
        header (fits.Header): FITS header.
        silent (bool): If True, suppress warnings during WCS creation.

    Returns:
        WCS: 2D celestial WCS object, or None if invalid.
    """
    if header is None:
        logger.warning("get_wcs: header is None")
        return None

    # Cache key = every WCS-relevant card (including distortion terms); an
    # unchanged header skips re-parsing SIP/TPV coefficients.
    cache_key = _wcs_cache_key(header)
    if cache_key is not None:
        cached = _WCS_CACHE.get(cache_key)
        if cached is not None:
            # Callers can mutate a returned WCS in place; hand out a copy so
            # the cached object stays pristine for later callers.
            try:
                return cached.deepcopy()
            except Exception:
                return cached

    try:
        # Normalise CTYPE first so the WCS() constructor always receives a
        # self-consistent header (SIP coefficients with TAN-SIP CTYPE, PV with
        # TPV). Otherwise astropy emits the FITSFixedWarning "SIP coefficients
        # present but CTYPE missing -SIP suffix" via both warnings.warn() and
        # logger.info() on astropy.wcs.wcs.
        header = _normalize_projection_codes(header, inplace=False)

        required = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2']
        missing = [k for k in required if k not in header]
        if missing:
            logger.warning("get_wcs: missing keywords %s", missing)
            return None

        # Suppress FITSFixedWarning for the entire WCS build + test:
        # warnings.catch_warnings covers the warnings.warn() path;
        # silence_astropy_wcs_info() covers the astropy.wcs.wcs INFO logger
        # path (astropy also emits the message as a log record). Both must
        # stay active through pixel_to_world() because lazy WCS evaluation
        # can trigger the warning on the first coordinate call.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with silence_astropy_wcs_info():
                wcs = WCS(header, fix=True, relax=True)

                if not wcs.has_celestial:
                    logger.warning("get_wcs: WCS has no celestial component")
                    return None

                # Reduce to a 2D celestial WCS so reproject and callers get a
                # consistent pixel grid.
                naxis = getattr(wcs.wcs, "naxis", 2)
                if naxis > 2:
                    wcs = wcs.celestial

                # Test transformation (CRPIX is FITS 1-based; pixel_to_world uses 0-based)
                try:
                    test_x = float(header['CRPIX1']) - 1.0
                    test_y = float(header['CRPIX2']) - 1.0
                    test_world = wcs.pixel_to_world(test_x, test_y)
                    if test_world is None:
                        logger.warning("get_wcs: WCS transformation test failed - returned None")
                        return None
                except Exception as e:
                    logger.warning("get_wcs: WCS transformation test failed with exception: %s", e)
                    return None

        if cache_key is not None:
            if len(_WCS_CACHE) >= _WCS_CACHE_MAX:
                _WCS_CACHE.pop(next(iter(_WCS_CACHE)))
            # Store a pristine copy: the caller may mutate the returned WCS,
            # and cache hits must never observe that mutation.
            try:
                _WCS_CACHE[cache_key] = wcs.deepcopy()
            except Exception:
                _WCS_CACHE[cache_key] = wcs
        return wcs

    except Exception as e:
        logger.warning("get_wcs: WCS creation failed with exception: %s", e)
        return None


def pixel_area_scale_arcsec(wcs_obj, shape=None) -> float:
    """Area-preserving nominal pixel scale in arcsec/pixel from a WCS.

    photutils 3.1 ``compute_pixel_area_map`` returns |det J| in
    arcsec**2 per pixel evaluated with central finite differences, so
    ``sqrt(median area)`` is the geometric-mean scale in arcsec/pixel
    across the whole field.  It follows the distortion across the image
    and stays well defined at the celestial poles and across the RA
    wraparound, unlike the x-axis-only ``proj_plane_pixel_scales`` value
    used previously.  When the map cannot be built (no shape,
    non-celestial WCS) the reference-pixel area from
    ``compute_pixel_areas`` is used, and as a last resort the old
    x-axis scale.

    Args:
        wcs_obj: astropy shared-WCS-interface object (WCS or gwcs).
        shape (tuple, optional): ``(ny, nx)`` image shape for the full
            map; falls back to ``wcs_obj.array_shape``/``pixel_shape``.

    Returns:
        float: median geometric-mean pixel scale in arcsec/pixel, or
            NaN when nothing usable can be derived.
    """
    try:
        from photutils.utils import compute_pixel_area_map, compute_pixel_areas
    except Exception:
        compute_pixel_area_map = compute_pixel_areas = None

    if compute_pixel_area_map is not None:
        try:
            if shape is None:
                shape = getattr(wcs_obj, "array_shape", None) or getattr(
                    wcs_obj, "pixel_shape", None
                )
            if shape is not None and len(shape) >= 2:
                areas = compute_pixel_area_map(
                    wcs_obj, tuple(int(s) for s in shape[:2])
                )
                med = float(np.nanmedian(np.asarray(areas, dtype=float)))
                if np.isfinite(med) and med > 0:
                    return float(np.sqrt(med))
            crpix = np.asarray(
                getattr(getattr(wcs_obj, "wcs", wcs_obj), "crpix"), dtype=float
            )
            if crpix.size >= 2 and np.all(np.isfinite(crpix[:2])):
                # CRPIX is FITS 1-based; WCS pixel coords are 0-based.
                a = compute_pixel_areas(
                    wcs_obj, [float(crpix[0] - 1.0)], [float(crpix[1] - 1.0)]
                )
                a = float(np.asarray(a, dtype=float).ravel()[0])
                if np.isfinite(a) and a > 0:
                    return float(np.sqrt(a))
        except Exception:
            pass

    try:
        from astropy.wcs import utils as wcs_utils

        xy = wcs_utils.proj_plane_pixel_scales(wcs_obj)
        if xy is not None and len(xy) > 0:
            s = float(np.asarray(xy, dtype=float).ravel()[0])
            if np.isfinite(s) and s > 0:
                return float(s * 3600.0)
    except Exception:
        pass
    return float("nan")


def _normalize_projection_codes(
    header: fits.Header,
    *,
    inplace: bool = False,
) -> fits.Header:
    """
    Keep CTYPE projection codes consistent with distortion keyword family.

    Rules:
    - SIP keywords only   -> enforce TAN-SIP CTYPE.
    - PV keywords only    -> enforce TPV CTYPE.
    - Both/none present   -> keep existing CTYPE unchanged.
    """
    out = header if inplace else header.copy()

    sip_order_keys = ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")
    sip_coeff_stems = ("A_", "B_", "AP_", "BP_")
    has_sip = any(k in out for k in sip_order_keys)
    if not has_sip:
        for key in out.keys():
            ks = str(key)
            if any(ks.startswith(stem) for stem in sip_coeff_stems):
                has_sip = True
                break

    has_pv = any(str(k).startswith("PV") for k in out.keys())

    if has_sip and has_pv:
        logger.debug(
            "Projection normalization: both SIP and PV keywords found; keeping existing CTYPE."
        )
        return out
    if not has_sip and not has_pv:
        return out

    model = "sip" if has_sip else "tpv"
    defaults = (
        ("CTYPE1", "RA---TAN-SIP" if model == "sip" else "RA---TPV"),
        ("CTYPE2", "DEC--TAN-SIP" if model == "sip" else "DEC--TPV"),
    )

    for ctype_key, ctype_default in defaults:
        cval = str(out.get(ctype_key, ctype_default))
        cval_u = cval.upper()
        updated = cval_u
        if model == "sip":
            if "TAN-SIP" in cval_u:
                continue
            if "TPV" in cval_u:
                updated = cval_u.replace("TPV", "TAN-SIP")
            elif "TAN" in cval_u:
                updated = cval_u.replace("TAN", "TAN-SIP", 1)
            else:
                updated = ctype_default
        else:
            if "TPV" in cval_u:
                continue
            if "TAN-SIP" in cval_u:
                updated = cval_u.replace("TAN-SIP", "TPV")
            elif "TAN" in cval_u:
                updated = cval_u.replace("TAN", "TPV", 1)
            else:
                updated = ctype_default
        if updated != cval:
            out[ctype_key] = updated
            logger.info(
                "Projection normalization: %s '%s' -> '%s' (%s model).",
                ctype_key,
                cval,
                updated,
                model.upper(),
            )

    return out


def _is_non_linear_key(key: str) -> bool:
    """Heuristic: detect non-linear distortion keywords in FITS WCS headers."""
    if key.startswith("PV"):
        return True
    if key.startswith("SIP_"):
        return True
    parts = key.split("_", 1)
    if len(parts) >= 2:
        stem = parts[0] + "_"
        return stem in ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
    return False


def _parse_floats_from_line(line: str) -> list[float]:
    """Extract all float-like tokens from a whitespace-separated string."""
    floats: list[float] = []
    for tok in line.strip().split():
        try:
            floats.append(float(tok))
        except Exception:
            pass
    return floats


def _extract_match_points_from_solve_field(
    match_path: str,
    initial_wcs: WCS,
    nx: int,
    ny: int,
    *,
    max_points: int = 500,
    max_lines: int = 20000,
    min_sep_arcsec: float = 0.0,
    max_sep_arcsec: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Best-effort extraction of (x,y) pixel and (ra,dec) sky from a solve-field
    *.match file.

    Column order is not guaranteed, so we select columns by value ranges and
    validate them using angular separation between the input RA/Dec and the
    `initial_wcs` prediction at (x,y).
    """
    if not match_path or not os.path.isfile(match_path):
        return None

    pix_x: list[float] = []
    pix_y: list[float] = []
    ra_list: list[float] = []
    dec_list: list[float] = []

    # FITS convention: solve-field pixel coords are treated as 1-based.
    with open(match_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f):
            if ln >= max_lines:
                break
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = _parse_floats_from_line(line)
            if len(vals) < 4:
                continue

            ra_candidates = [i for i, v in enumerate(vals) if 0.0 <= v <= 360.0]
            dec_candidates = [i for i, v in enumerate(vals) if -90.0 <= v <= 90.0]
            x_candidates = [i for i, v in enumerate(vals) if -1.0 <= v <= (nx + 1.0)]
            y_candidates = [i for i, v in enumerate(vals) if -1.0 <= v <= (ny + 1.0)]

            if not ra_candidates or not dec_candidates or not x_candidates or not y_candidates:
                continue

            best_sep = None
            best = None  # (xi, yi, rai, decj)

            for rai in ra_candidates:
                ra_val = vals[rai]
                for decj in dec_candidates:
                    dec_val = vals[decj]
                    for xi in x_candidates:
                        x_val = vals[xi]
                        for yi in y_candidates:
                            y_val = vals[yi]
                            try:
                                ra_pred, dec_pred = initial_wcs.all_pix2world(
                                    [[x_val]], [[y_val]], 0
                                )
                            except Exception:
                                continue

                            try:
                                c_pred = SkyCoord(
                                    ra=float(ra_pred[0][0]) * u.deg,
                                    dec=float(dec_pred[0][0]) * u.deg,
                                    frame="icrs",
                                )
                                c_true = SkyCoord(
                                    ra=ra_val * u.deg,
                                    dec=dec_val * u.deg,
                                    frame="icrs",
                                )
                                sep_arcsec = c_pred.separation(c_true).to(u.arcsec).value
                            except Exception:
                                continue

                            if best_sep is None or sep_arcsec < best_sep:
                                best_sep = float(sep_arcsec)
                                best = (xi, yi, rai, decj)

            if best is None or best_sep is None:
                continue

            if best_sep < min_sep_arcsec or best_sep > max_sep_arcsec:
                continue

            xi, yi, rai, decj = best
            x_val = vals[xi]
            y_val = vals[yi]
            ra_val = vals[rai]
            dec_val = vals[decj]

            pix_x.append(float(x_val))
            pix_y.append(float(y_val))
            ra_list.append(float(ra_val))
            dec_list.append(float(dec_val))

            if len(pix_x) >= max_points:
                break

    # Sparse fields can legitimately have <10 matched stars; a coarse
    # median/p95 comparison is still useful for the keep-input decision.
    if len(pix_x) < 5:
        return None

    return (
        np.asarray(pix_x, dtype=float),
        np.asarray(pix_y, dtype=float),
        np.asarray(ra_list, dtype=float),
        np.asarray(dec_list, dtype=float),
    )


def _extract_corr_points_from_solve_field(
    corr_path: str,
    *,
    max_points: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Extract matched (x,y) <-> (ra,dec) pairs from solve-field *.corr FITS table.

    This is preferred over parsing *.match text because column names are explicit.
    Typical astrometry.net columns include field_x/field_y and index_ra/index_dec.
    """
    if not corr_path or not os.path.isfile(corr_path):
        return None

    try:
        tbl = Table.read(corr_path)
    except Exception:
        return None

    if len(tbl) == 0:
        return None

    cols = {c.lower(): c for c in tbl.colnames}

    def pick_name(priority: list[str], fallback_contains: list[str]) -> str | None:
        for p in priority:
            if p in cols:
                return cols[p]
        for c in tbl.colnames:
            cl = c.lower()
            if all(tok in cl for tok in fallback_contains):
                return c
        return None

    x_col = pick_name(["field_x", "x"], ["field", "x"])
    y_col = pick_name(["field_y", "y"], ["field", "y"])
    # Prefer astrometry.net index-sky columns explicitly.
    ra_col = pick_name(
        ["index_ra", "index_alpha", "match_ra", "ra"],
        ["index", "ra"],
    )
    dec_col = pick_name(
        ["index_dec", "index_delta", "match_dec", "dec"],
        ["index", "dec"],
    )

    if x_col is None or y_col is None or ra_col is None or dec_col is None:
        return None
    logger.debug(
        "solve-field .corr column mapping: x=%s y=%s ra=%s dec=%s",
        str(x_col),
        str(y_col),
        str(ra_col),
        str(dec_col),
    )
    if not str(ra_col).lower().startswith("index_") or not str(dec_col).lower().startswith("index_"):
        logger.warning(
            "solve-field .corr is not using index_* sky columns (ra=%s dec=%s); check .corr schema for this astrometry.net version.",
            str(ra_col),
            str(dec_col),
        )

    x = np.asarray(tbl[x_col], dtype=float)
    y = np.asarray(tbl[y_col], dtype=float)
    ra = np.asarray(tbl[ra_col], dtype=float)
    dec = np.asarray(tbl[dec_col], dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(ra) & np.isfinite(dec)
    finite &= (ra >= 0.0) & (ra <= 360.0) & (dec >= -90.0) & (dec <= 90.0)
    x = x[finite]
    y = y[finite]
    ra = ra[finite]
    dec = dec[finite]

    if len(x) < 5:
        return None

    # Cap the point count for the downstream fit; sample evenly by row order.
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x, y, ra, dec = x[idx], y[idx], ra[idx], dec[idx]

    return x, y, ra, dec


def _wcs_match_separation_stats_arcsec(
    wcs_obj: WCS,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> tuple[float, float]:
    """
    Angular-separation stats (median, p95) between predicted and catalog sky.
    """
    if wcs_obj is None:
        logger.debug("_wcs_match_separation_stats_arcsec: wcs_obj is None")
        return np.nan, np.nan
    sky_true = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ra_p, dec_p = wcs_obj.all_pix2world(x_pix, y_pix, 0)
    sky_pred = SkyCoord(
        ra=np.asarray(ra_p) * u.deg, dec=np.asarray(dec_p) * u.deg, frame="icrs"
    )
    sep = sky_pred.separation(sky_true).to(u.arcsec).value
    sep = np.asarray(sep, dtype=float)
    if sep.size == 0:
        return np.nan, np.nan
    return float(np.nanmedian(sep)), float(np.nanpercentile(sep, 95.0))


def _best_wcs_match_separation_stats_arcsec(
    wcs_obj: WCS,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> tuple[float, float, float]:
    """
    Separation stats under both pixel-origin hypotheses; returns the better.

    Returns:
        (median_arcsec, p95_arcsec, applied_shift_px)
    """
    if wcs_obj is None:
        logger.debug("_best_wcs_match_separation_stats_arcsec: wcs_obj is None")
        return np.nan, np.nan, 0.0
    med1, p951 = _wcs_match_separation_stats_arcsec(wcs_obj, x_pix, y_pix, ra_deg, dec_deg)
    x0 = np.asarray(x_pix, dtype=float) + 1.0
    y0 = np.asarray(y_pix, dtype=float) + 1.0
    med0, p950 = _wcs_match_separation_stats_arcsec(wcs_obj, x0, y0, ra_deg, dec_deg)
    if np.isfinite(med0) and np.isfinite(med1):
        if med0 + 1e-6 < med1:
            return float(med0), float(p950), 1.0
        return float(med1), float(p951), 0.0
    if np.isfinite(med1):
        return float(med1), float(p951), 0.0
    if np.isfinite(med0):
        return float(med0), float(p950), 1.0
    return np.nan, np.nan, 0.0


def wcs_world_to_pixel(
    wcs_obj: WCS, ra_deg: np.ndarray, dec_deg: np.ndarray, *, origin: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert sky coordinates to pixel coordinates with a consistent origin.

    Parameters:
    -----------
    wcs_obj : WCS
        WCS object to use for conversion
    ra_deg : np.ndarray
        Right ascension in degrees
    dec_deg : np.ndarray
        Declination in degrees
    origin : int, optional
        Origin convention: 0 for 0-based (numpy), 1 for 1-based (FITS). Default is 0.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        (x_pixel, y_pixel) coordinates in the specified origin convention

    Notes:
    ------
    - If vectorized inversion fails (common for some distorted points), falls back to
      per-point conversion and returns NaN for failed coordinates.
    - Shared entry point for all world-to-pixel conversions in this module.
    """
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)
    n = int(min(len(ra_arr), len(dec_arr)))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    ra_arr = ra_arr[:n]
    dec_arr = dec_arr[:n]
    try:
        x_pix, y_pix = wcs_obj.all_world2pix(ra_arr, dec_arr, origin)
        return np.asarray(x_pix, dtype=float), np.asarray(y_pix, dtype=float)
    except Exception:
        x_out = np.full(n, np.nan, dtype=float)
        y_out = np.full(n, np.nan, dtype=float)
        for i in range(n):
            try:
                x_i, y_i = wcs_obj.all_world2pix(
                    np.array([ra_arr[i]], dtype=float),
                    np.array([dec_arr[i]], dtype=float),
                    origin,
                )
                x_out[i] = float(np.asarray(x_i, dtype=float)[0])
                y_out[i] = float(np.asarray(y_i, dtype=float)[0])
            except Exception:
                continue
        return x_out, y_out


def _safe_world_to_pixel_values(
    wcs_obj: WCS, ra_deg: np.ndarray, dec_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy wrapper: sky -> pixel with origin=0.

    Deprecated: use wcs_world_to_pixel instead.
    """
    return wcs_world_to_pixel(wcs_obj, ra_deg, dec_deg, origin=0)


def _build_scamp_optimization_trials(
    wcs_cfg: dict,
) -> list[tuple[str, dict]]:
    """
    Build SCAMP trial configurations for WCS optimization.

    Trial 0 keeps user/base config. Extra trials progressively tighten matching
    tolerances and optionally lower distortion degree by one, then we choose the
    best solution from matched-point residuals.
    """
    base = dict(wcs_cfg or {})
    # Multi-trial SCAMP optimization is expensive; a single call is the default.
    if bool(base.get("scamp_single_call", True)):
        return [("base", dict(base))]
    trials: list[tuple[str, dict]] = [("base", dict(base))]

    try:
        max_trials = int(base.get("scamp_optimize_max_trials", 3))
    except Exception:
        max_trials = 3
    max_trials = int(np.clip(max_trials, 1, 5))
    if max_trials <= 1:
        return trials

    crossid = float(base.get("scamp_crossid_radius", 2.5))
    poserr = float(base.get("scamp_position_maxerr", 1.0))
    degree = int(base.get("scamp_distort_degrees", 4))

    tight = dict(base)
    tight["scamp_crossid_radius"] = float(np.clip(crossid * 0.8, 0.6, 5.0))
    tight["scamp_position_maxerr"] = float(np.clip(poserr * 0.8, 0.3, 3.0))
    trials.append(("tight_match", tight))
    if len(trials) >= max_trials:
        return trials[:max_trials]

    low_degree = dict(tight)
    low_degree["scamp_distort_degrees"] = int(np.clip(degree - 1, 2, 6))
    trials.append(("tight_match_low_degree", low_degree))
    if len(trials) >= max_trials:
        return trials[:max_trials]

    very_tight = dict(base)
    very_tight["scamp_crossid_radius"] = float(np.clip(crossid * 0.65, 0.5, 5.0))
    very_tight["scamp_position_maxerr"] = float(np.clip(poserr * 0.65, 0.25, 3.0))
    very_tight["scamp_distort_degrees"] = int(np.clip(degree, 2, 6))
    trials.append(("very_tight_match", very_tight))

    return trials[:max_trials]


def _sip_summary(wcs_obj: WCS) -> str:
    """
    Return compact summary of SIP distortion terms for logging.
    """
    sip = getattr(wcs_obj, "sip", None)
    if sip is None:
        return "none"
    try:
        a_order = int(getattr(sip, "a_order", 0) or 0)
        b_order = int(getattr(sip, "b_order", 0) or 0)
        ap_order = int(getattr(sip, "ap_order", 0) or 0)
        bp_order = int(getattr(sip, "bp_order", 0) or 0)
        a_nnz = int(np.count_nonzero(getattr(sip, "a", np.zeros((1, 1)))))
        b_nnz = int(np.count_nonzero(getattr(sip, "b", np.zeros((1, 1)))))
        ap_nnz = int(np.count_nonzero(getattr(sip, "ap", np.zeros((1, 1)))))
        bp_nnz = int(np.count_nonzero(getattr(sip, "bp", np.zeros((1, 1)))))
        return (
            f"a/b/ap/bp order={a_order}/{b_order}/{ap_order}/{bp_order}, "
            f"nnz={a_nnz}/{b_nnz}/{ap_nnz}/{bp_nnz}"
        )
    except Exception:
        return "present (summary unavailable)"


def _log_sip_coefficients(wcs_obj: WCS, prefix: str = "fit_wcs_from_points") -> None:
    """
    Log compact SIP summary (no per-coefficient dump).
    """
    sip = getattr(wcs_obj, "sip", None)
    if sip is None:
        logger.info("%s SIP coefficients: none", prefix)
        return
    logger.info("%s SIP coefficients: %s", prefix, _sip_summary(wcs_obj))


def _has_nonzero_sip(header: fits.Header) -> bool:
    """
    Check whether a FITS header contains non-zero SIP distortion coefficients.

    Returns True if any A_i_j, B_i_j, AP_i_j, or BP_i_j coefficient is non-zero.
    Returns False if A_ORDER is absent, or if all coefficients are zero.
    """
    a_order = header.get("A_ORDER", 0)
    b_order = header.get("B_ORDER", 0)
    if a_order == 0 and b_order == 0:
        return False
    sip_stems = ("A_", "B_", "AP_", "BP_")
    for key in header.keys():
        ks = str(key)
        if ks in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"):
            continue
        for stem in sip_stems:
            if ks.startswith(stem):
                try:
                    if float(header[ks]) != 0.0:
                        return True
                except (ValueError, TypeError):
                    pass
                break
    return False


def _has_nonzero_distortion(header: fits.Header) -> bool:
    """
    Check whether a FITS header carries any non-zero non-linear distortion
    coefficients, covering SIP (A_/B_/AP_/BP_) and TPV-style polynomial
    terms (PVi_j, D_i_j, DP_i_j).

    Used to decide whether a solved WCS downgrades the distortion model of
    the input WCS (e.g. a TAN-only solve replacing a TPV/SIP input).
    """
    if header is None:
        return False
    coeff_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_")
    for key in header.keys():
        ks = str(key)
        if ks.endswith("_ORDER"):
            continue
        is_coeff = ks.startswith(coeff_stems)
        if not is_coeff and ks.startswith("PV") and "_" in ks:
            # PVi_j distortion terms (TPV/SCAMP convention); plain "PV"
            # or keywords like PVERSION are not coefficients.
            is_coeff = True
        if not is_coeff:
            continue
        try:
            if float(header[ks]) != 0.0:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _log_header_distortion_coefficients(
    header: fits.Header, prefix: str = "astrometry.net solved"
) -> None:
    """
    Log compact non-linear distortion summary (no per-coefficient dump).
    """
    try:
        keys = []
        pv_keys = []
        sip_keys = []
        for k in header.keys():
            ks = str(k)
            if ks.startswith("PV"):
                keys.append(ks)
                pv_keys.append(ks)
                continue
            if "_" in ks:
                stem = ks.split("_", 1)[0] + "_"
                if stem in ("A_", "B_", "AP_", "BP_"):
                    keys.append(ks)
                    sip_keys.append(ks)
        keys = sorted(set(keys))
        if len(keys) == 0:
            logger.info("%s distortion coefficients: none (no SIP/PV keys)", prefix)
            return
        logger.info(
            "%s distortion coefficients: total=%d (PV=%d, SIP=%d)",
            prefix,
            len(keys),
            len(set(pv_keys)),
            len(set(sip_keys)),
        )
    except Exception as exc:
        log_warning_from_exception(
            logger, f"{prefix} distortion coefficient logging failed", exc
        )


def _filter_points_against_initial_wcs(
    initial_wcs: WCS,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    max_sep_arcsec: float = 3.0,
    clip_sigma: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """
    Reject outlier pixel<->sky matches using residuals to initial_wcs.
    Returns filtered arrays, number removed, and pre-filter med/p95 residuals.
    """
    sky_true = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ra_p, dec_p = initial_wcs.all_pix2world(x_pix, y_pix, 0)
    sky_pred = SkyCoord(
        ra=np.asarray(ra_p) * u.deg, dec=np.asarray(dec_p) * u.deg, frame="icrs"
    )
    sep = np.asarray(sky_pred.separation(sky_true).to(u.arcsec).value, dtype=float)
    if sep.size == 0:
        return x_pix, y_pix, ra_deg, dec_deg, 0, np.nan, np.nan

    med0 = float(np.nanmedian(sep))
    p950 = float(np.nanpercentile(sep, 95.0))
    good = np.isfinite(sep)
    if np.isfinite(max_sep_arcsec) and max_sep_arcsec > 0:
        good &= sep <= float(max_sep_arcsec)

    # MAD-based clip around the median to suppress residual outliers.
    if np.any(good):
        s = sep[good]
        med = float(np.nanmedian(s))
        mad = float(np.nanmedian(np.abs(s - med)))
        sigma_robust = 1.4826 * mad if np.isfinite(mad) and mad > 0 else 0.0
        if sigma_robust > 0 and np.isfinite(clip_sigma) and clip_sigma > 0:
            good &= sep <= (med + float(clip_sigma) * sigma_robust)

    removed = int(len(sep) - int(np.count_nonzero(good)))
    return (
        x_pix[good],
        y_pix[good],
        ra_deg[good],
        dec_deg[good],
        removed,
        med0,
        p950,
    )


def load_background_std(background_std) -> np.ndarray:
    """
    Load background standard deviation from a FITS file or use a 2D array.
    If None, return None.

    Args:
        background_std (str, np.ndarray, or None): Background standard deviation.

    Returns:
        np.ndarray or None: Background standard deviation.
    """
    if background_std is None:
        return None
    if isinstance(background_std, str):
        with fits.open(background_std) as hdul:
            data = hdul[0].data
            # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
            if data.dtype.kind != 'f':
                data = data.astype(np.float32)
            return data
    if isinstance(background_std, np.ndarray):
        return background_std
    raise TypeError(
        "background_std must be a string (FITS filepath), a 2D numpy array, or None."
    )


def create_conv_file(path: str, fwhm_pixels: float = 3.0, force: bool = False) -> None:
    """
    Create a convolution kernel for SExtractor, optimized for the given FWHM.
    Skips writing if the file already exists unless force=True.
    Args:
        path: Path to save the convolution file.
        fwhm_pixels: FWHM in pixels, used to optimize the kernel size.
        force: If True, overwrite even when the file exists.
    """
    if not force and os.path.isfile(path):
        return
    kernel_size = max(3, int(np.ceil(fwhm_pixels * 3)))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    kernel_size = min(kernel_size, 7)  # SExtractor hard limit; filename is gaussian_7x7.conv
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


def _resolve_sextractor_conv_fwhm_pixels(
    wcs_cfg: dict, header: fits.Header, default_input: dict
) -> float:
    """
    Resolve the FWHM (pixels) used to build the SExtractor convolution kernel.

    Priority:
      1) wcs.sextractor_conv_fwhm_pixels
      2) FITS header FWHM/fwhm
      3) runtime config fwhm
      4) fallback default of 3.0 px
    """
    candidate_values = [
        wcs_cfg.get("sextractor_conv_fwhm_pixels"),
        header.get("FWHM"),
        header.get("fwhm"),
        default_input.get("fwhm"),
    ]
    for raw in candidate_values:
        try:
            fwhm_pix = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fwhm_pix) and fwhm_pix > 0:
            # Clip to a workable range; the 1.0 px floor keeps the kernel
            # valid for undersampled images (e.g. ZTF FWHM ~1.8 px).
            return float(np.clip(fwhm_pix, 1.0, 20.0))
    return 3.0


def _build_scale_args(
    scale_low: float,
    scale_high: float,
    include_pos: bool,
    ra: float = None,
    dec: float = None,
    radius_deg: float = None,
    background_std: float = None,
) -> list:
    """Build solve-field scale-bounds args plus an optional position cone."""
    out = [
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        str(scale_low),
        "--scale-high",
        str(scale_high),
    ]
    if background_std is not None:
        out += ["--sigma", str(background_std)]
    if include_pos and ra is not None and dec is not None:
        out += [
            "--ra",
            str(ra),
            "--dec",
            str(dec),
            "--radius",
            str(radius_deg),
        ]
    return out


def _sky_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Small-angle separation in degrees between two sky positions."""
    dra = (float(ra1) - float(ra2) + 180.0) % 360.0 - 180.0
    ddec = float(dec1) - float(dec2)
    cosd = np.cos(np.deg2rad(0.5 * (float(dec1) + float(dec2))))
    return float(np.hypot(dra * cosd, ddec))


def _header_position_hint(header):
    """
    World position (ra_deg, dec_deg) of the header WCS reference point.

    Uses the reference pixel (CRPIX) projected through the header WCS so
    distorted models still report the field center they actually encode;
    falls back to CRVAL when the projection fails.  Returns None when the
    header carries no usable WCS.
    """
    wcs_obj = get_wcs(header)
    if wcs_obj is None:
        return None
    try:
        crpix = np.asarray(getattr(wcs_obj.wcs, "crpix", []), dtype=float)
        if crpix.size >= 2 and np.all(np.isfinite(crpix[:2])):
            # CRPIX is FITS 1-based; origin=1 evaluates it directly.
            ra_h, dec_h = wcs_obj.all_pix2world(crpix[0], crpix[1], 1)
            ra_h = float(np.asarray(ra_h, dtype=float).ravel()[0])
            dec_h = float(np.asarray(dec_h, dtype=float).ravel()[0])
        else:
            crval = np.asarray(getattr(wcs_obj.wcs, "crval", []), dtype=float)
            ra_h, dec_h = float(crval[0]), float(crval[1])
        if np.isfinite(ra_h) and np.isfinite(dec_h):
            return ra_h, dec_h
    except Exception:
        pass
    return None


def _choose_position_hints(
    header,
    target_ra,
    target_dec,
    prefer_header: bool = True,
    dedupe_arcsec: float = 1.0,
) -> list:
    """
    Ordered (ra_deg, dec_deg) hint centers for solve-field --ra/--dec.

    The preferred center comes first; the other center is kept second so
    the solver can make one extra hinted pass when it lies outside the
    first search cone.  Entries closer than ``dedupe_arcsec`` are dropped.
    """
    header_hint = _header_position_hint(header)
    target_hint = None
    try:
        if target_ra is not None and target_dec is not None:
            t_ra, t_dec = float(target_ra), float(target_dec)
            if np.isfinite(t_ra) and np.isfinite(t_dec):
                target_hint = (t_ra, t_dec)
    except (TypeError, ValueError):
        target_hint = None
    if prefer_header and header_hint is not None:
        ordered = [header_hint, target_hint]
    else:
        ordered = [target_hint, header_hint]
    hints = []
    tol_deg = float(dedupe_arcsec) / 3600.0
    for hint in ordered:
        if hint is None:
            continue
        if any(
            _sky_sep_deg(hint[0], hint[1], kept[0], kept[1]) <= tol_deg
            for kept in hints
        ):
            continue
        hints.append(hint)
    return hints


def _coerce_detected_xy(detected_xy):
    """
    Normalize detected positions to an (N, 2) float array of (x, y).

    Accepts an Nx2 array-like or a table/DataFrame-like object with x/y
    pixel columns (x_pix/y_pix, x/y, or SExtractor XWIN_IMAGE/YWIN_IMAGE,
    X_IMAGE/Y_IMAGE).  Rows with non-finite values are dropped.  Returns
    None when no usable positions are present.
    """
    if detected_xy is None:
        return None
    try:
        cols = getattr(detected_xy, "columns", None)
        if cols is not None:
            lower = {str(c).strip().lower(): c for c in cols}
            arr = None
            for x_name, y_name in (
                ("x_pix", "y_pix"),
                ("x", "y"),
                ("xwin_image", "ywin_image"),
                ("x_image", "y_image"),
            ):
                if x_name in lower and y_name in lower:
                    arr = np.column_stack(
                        [
                            np.asarray(detected_xy[lower[x_name]], dtype=float),
                            np.asarray(detected_xy[lower[y_name]], dtype=float),
                        ]
                    )
                    break
            if arr is None:
                return None
        else:
            arr = np.asarray(detected_xy, dtype=float)
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
            if arr.ndim != 2 or arr.shape[1] < 2:
                return None
            arr = arr[:, :2]
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite]
        if arr.shape[0] == 0:
            return None
        return arr
    except Exception:
        return None


def _coerce_detected_flux(detected_xy, n_expected):
    """
    Brightness proxy for detected positions, (N,) float array or None.

    solve-field consumes an xylist in file order and builds quads from the
    first rows, so the list must be written brightest-first or sparse fields
    feed it unusable subsets.  ``_coerce_detected_xy`` drops every column but
    x/y; this recovers a ranking column (snr is preferred because it reflects
    reliability as well as brightness) and applies the same finite-row filter
    so the values stay row-aligned with the coerced positions.  Returns None
    when no usable column exists.
    """
    cols = getattr(detected_xy, "columns", None)
    if cols is None:
        return None
    try:
        n_expected = int(n_expected)
    except (TypeError, ValueError):
        return None
    if n_expected <= 0:
        return None
    lower = {str(c).strip().lower(): c for c in cols}
    xy_arr = None
    for x_name, y_name in (
        ("x_pix", "y_pix"),
        ("x", "y"),
        ("xwin_image", "ywin_image"),
        ("x_image", "y_image"),
    ):
        if x_name in lower and y_name in lower:
            try:
                xy_arr = np.column_stack(
                    [
                        np.asarray(detected_xy[lower[x_name]], dtype=float),
                        np.asarray(detected_xy[lower[y_name]], dtype=float),
                    ]
                )
            except Exception:
                xy_arr = None
            break
    if xy_arr is None or xy_arr.shape[0] == 0:
        return None
    row_ok = np.isfinite(xy_arr).all(axis=1)
    if int(row_ok.sum()) != n_expected:
        return None
    for name in ("snr", "peak_flux", "flux_ap", "flux_auto", "flux"):
        if name not in lower:
            continue
        try:
            vals = np.asarray(detected_xy[lower[name]], dtype=float).ravel()
        except Exception:
            continue
        if vals.size == row_ok.size and np.isfinite(vals[row_ok]).any():
            return vals[row_ok]
    return None


def _solve_nsigma_for_count(n_det, cfg_val, sparse_nsigma):
    """
    Solve-field --nsigma given a known detection count.

    An explicit zero is treated as maximally sparse: our detection stage
    found nothing, so the solver gets ``sparse_nsigma`` to push its own
    detection as deep as configured for starved fields.  Positive counts
    keep the proactive caps (fewer than ~30 detections -> 3, fewer than
    ~50 -> 4).  ``n_det=None`` (count unknown) leaves the configured
    value alone.
    """
    cfg_val = int(cfg_val)
    if n_det is None:
        return cfg_val
    n_det = int(n_det)
    if n_det == 0:
        # Never raise the threshold above the configured value.
        return min(cfg_val, int(sparse_nsigma))
    if n_det < 30:
        return min(cfg_val, 3)
    if n_det < 50:
        return min(cfg_val, 4)
    return cfg_val


def _detect_solve_sources(image, nsigma, max_n=500):
    """
    Relaxed source detection used to feed solve-field on starved fields.

    Median-subtracts the image and runs a small-FWHM DAOStarFinder at
    ``nsigma * mad_std``.  This is deliberately looser than the main
    detection stage: the goal is only a usable (x, y) list, so the solver
    can skip its own extraction.

    Returns:
        (xy, flux): (N,2) float array of 0-based (x, y) centroids and a
        flux array, brightest first, capped at ``max_n`` rows.  Both are
        None on failure or when nothing is detected.
    """
    try:
        from astropy.stats import mad_std
        from photutils.detection import DAOStarFinder

        data = np.asarray(image, dtype=float)
        finite = np.isfinite(data)
        if not finite.any():
            return None, None
        median = np.median(data[finite])
        noise = mad_std(data[finite], ignore_nan=True)
        if not np.isfinite(noise) or noise <= 0:
            return None, None
        finder = DAOStarFinder(fwhm=3.0, threshold=float(nsigma) * noise)
        with warnings.catch_warnings():
            # NoDetectionsWarning is the common starved-field outcome.
            warnings.simplefilter("ignore")
            tbl = finder(data - median)
        if tbl is None or len(tbl) == 0:
            return None, None
        xy = np.column_stack(
            [
                np.asarray(tbl["xcentroid"], dtype=float),
                np.asarray(tbl["ycentroid"], dtype=float),
            ]
        )
        flux = None
        for name in ("flux", "peak"):
            if name in tbl.colnames:
                flux = np.asarray(tbl[name], dtype=float)
                break
        if flux is None:
            flux = np.ones(len(tbl), dtype=float)
        # Brightest first: the solver consumes the list in order.
        order = np.argsort(flux)[::-1][: max(1, int(max_n))]
        return xy[order], flux[order]
    except Exception:
        return None, None


def _write_solve_xylist(xy, flux, path, width=None, height=None):
    """
    Write detected positions as a FITS binary-table xylist for solve-field.

    astrometry.net xylist coordinates use the FITS 1-based pixel
    convention, so the 0-based numpy coordinates are shifted by +1.  A
    FLUX column (non-finite values replaced by 1.0) lets the solver
    prioritize bright sources; IMAGEW/IMAGEH record the field size since
    an xylist carries no pixel data of its own.

    Returns *path* on success, None on failure.
    """
    try:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] == 0:
            return None
        n_src = xy.shape[0]
        f_col = np.ones(n_src, dtype=float)
        sortable = False
        if flux is not None:
            _f = np.asarray(flux, dtype=float).ravel()
            if _f.size == n_src and np.isfinite(_f).any():
                f_col = _f
                sortable = True
        # solve-field builds quads from the first rows of the list, so rows
        # must be brightest-first.  A scan-order (e.g. SExtractor) list on a
        # sparse field starves every object window of a usable quad.
        if sortable:
            order = np.argsort(
                np.where(np.isfinite(f_col), f_col, -np.inf)
            )[::-1]
            xy = xy[order]
            f_col = f_col[order]
        x_col = xy[:, 0] + 1.0
        y_col = xy[:, 1] + 1.0
        f_col = np.where(np.isfinite(f_col), f_col, 1.0)
        cols = fits.ColDefs(
            [
                fits.Column(name="X", format="D", array=x_col),
                fits.Column(name="Y", format="D", array=y_col),
                fits.Column(name="FLUX", format="D", array=f_col),
            ]
        )
        hdu = fits.BinTableHDU.from_columns(cols)
        # Field bounds are required for xylist inputs; without pixels the
        # solver cannot guess them reliably.
        if width is not None:
            hdu.header["IMAGEW"] = int(width)
        if height is not None:
            hdu.header["IMAGEH"] = int(height)
        hdu.writeto(path, overwrite=True)
        return path
    except Exception:
        return None


def _downsample_levels_for(nx, ny, downsample, ds_max, min_pixels, small_image):
    """
    Ordered solve-field --downsample levels to try.

    Large images keep the original ladder: the configured level, then
    doubling up to ``ds_max``.  Small images (larger axis <=
    ``small_image``) can afford finer sampling, so levels 1..downsample-1
    are prepended.  Every level must leave the smaller axis at or above
    ``min_pixels``; a solve on a thumbnail cannot succeed and wastes a
    full cpulimit.
    """
    nx = int(nx)
    ny = int(ny)
    ds = max(1, int(downsample))
    ds_max = max(ds, int(ds_max))
    levels = []
    if max(nx, ny) <= int(small_image):
        levels.extend(range(1, ds))
    while ds <= ds_max:
        levels.append(ds)
        ds *= 2
    bound = max(1.0, float(min_pixels))
    levels = [lvl for lvl in levels if min(nx, ny) / lvl >= bound]
    # Deduplicate while preserving order (fine levels can overlap the
    # configured level when downsample already equals 1).
    seen = set()
    levels = [lvl for lvl in levels if not (lvl in seen or seen.add(lvl))]
    if not levels:
        # Level 1 is the only usable attempt on an undersized frame.
        levels = [1]
    return levels


def _solve_artifact_glob(dirname, tmp_tag, image_base, pattern):
    """
    Paths matching *pattern* that belong to this solve invocation.

    Solve temp names embed ``{image_base}_{pid}``; a bare directory glob on
    the extension would also remove artifacts a concurrent solve in the
    same directory is still using.  The solver names the augmented xylist
    after the input stem, so ``.axy`` must additionally match the image
    base (``{image_base}.axy`` for image input; the ``{tmp_tag}.axy`` name
    produced for xylist input is already covered by the tag).
    """
    hits = glob.glob(os.path.join(dirname, f"*{tmp_tag}*{pattern}"))
    if pattern == ".axy":
        hits += glob.glob(os.path.join(dirname, f"{image_base}{pattern}"))
    return hits


def _set_flag_value(args, flag, value):
    """
    Return args with the ``flag value`` pair replaced by ``flag str(value)``.

    Only the element pair owned by *flag* is dropped.  Filtering on the
    value string instead removes any unrelated argument that happens to
    equal it (``--downsample 3`` dies alongside ``--nsigma 3``), leaving a
    dangling flag that consumes the next argument.
    """
    out = []
    i = 0
    while i < len(args):
        if args[i] == flag and i + 1 < len(args):
            i += 2
            continue
        out.append(args[i])
        i += 1
    out += [flag, str(value)]
    return out


def _should_verify_input_wcs(
    redo_requested: bool,
    verify_flag: bool,
    has_wcs: bool,
    n_det,
    min_matches: int,
) -> bool:
    """
    Truth table for the verify-and-skip gate.

    redo_wcs=True always solves; verification only runs when it is False,
    verify_input_wcs is enabled, the header WCS parses, and enough
    detected sources exist to make a match meaningful.
    """
    return bool(
        verify_flag
        and not redo_requested
        and has_wcs
        and n_det is not None
        and int(n_det) >= int(min_matches)
    )


def verify_wcs_residuals(
    wcs_obj,
    x_pix,
    y_pix,
    ra_deg,
    dec_deg,
    match_radius_arcsec: float,
):
    """
    Residual stats between detected pixels and catalog stars on the WCS.

    Catalog (ra, dec) positions are projected to pixels through
    ``wcs_obj`` and matched to the detected (x, y) positions with a
    KD-tree (numpy brute force when scipy is absent).  Separations are
    reported in arcsec using the WCS pixel scale.

    Returns:
        (median_arcsec, p95_arcsec, n_matches); NaN, NaN, 0 when nothing
        can be matched or the WCS has no usable scale.
    """
    nan_result = (float("nan"), float("nan"), 0)
    if wcs_obj is None:
        return nan_result
    x_pix = np.asarray(x_pix, dtype=float).ravel()
    y_pix = np.asarray(y_pix, dtype=float).ravel()
    ra_deg = np.asarray(ra_deg, dtype=float).ravel()
    dec_deg = np.asarray(dec_deg, dtype=float).ravel()
    n_det = int(min(x_pix.size, y_pix.size))
    n_cat = int(min(ra_deg.size, dec_deg.size))
    if n_det == 0 or n_cat == 0:
        return nan_result
    det_xy = np.column_stack([x_pix[:n_det], y_pix[:n_det]])
    det_ok = np.isfinite(det_xy).all(axis=1)
    det_xy = det_xy[det_ok]
    cat_ok = np.isfinite(ra_deg[:n_cat]) & np.isfinite(dec_deg[:n_cat])
    ra_c = ra_deg[:n_cat][cat_ok]
    dec_c = dec_deg[:n_cat][cat_ok]
    if det_xy.shape[0] == 0 or ra_c.size == 0:
        return nan_result

    # Pixel scale converts the match radius to pixels and the pixel
    # separations back to arcsec; the CD determinant is the fallback.
    scale = pixel_area_scale_arcsec(wcs_obj)
    if not np.isfinite(scale) or scale <= 0:
        try:
            cd = np.asarray(wcs_obj.wcs.cd, dtype=float)
            scale = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
        except Exception:
            scale = float("nan")
    if not np.isfinite(scale) or scale <= 0:
        return nan_result

    try:
        cx, cy = wcs_obj.all_world2pix(ra_c, dec_c, 0)
    except Exception:
        return nan_result
    cx = np.asarray(cx, dtype=float).ravel()
    cy = np.asarray(cy, dtype=float).ravel()
    proj_ok = np.isfinite(cx) & np.isfinite(cy)
    cat_xy = np.column_stack([cx[proj_ok], cy[proj_ok]])
    if cat_xy.shape[0] == 0:
        return nan_result

    try:
        from scipy.spatial import cKDTree

        dist_px, _ = cKDTree(det_xy).query(cat_xy, k=1)
    except Exception:
        diff = cat_xy[:, None, :] - det_xy[None, :, :]
        dist_px = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)

    tol_px = float(match_radius_arcsec) / scale
    sep_arcsec = np.asarray(dist_px, dtype=float)
    sep_arcsec = sep_arcsec[sep_arcsec <= tol_px] * scale
    if sep_arcsec.size == 0:
        return nan_result
    return (
        float(np.nanmedian(sep_arcsec)),
        float(np.nanpercentile(sep_arcsec, 95.0)),
        int(sep_arcsec.size),
    )




# =============================================================================
# Main Class
# =============================================================================


class WCSSolver:
    """
    WCS solver and FITS header cleaner.
    Handles faint sources, cosmic rays, and edge cases.
    """

    def __init__(
        self, fpath: str, image: np.ndarray, header: fits.Header, default_input: dict
    ):
        """
        Initialize the WCS checker and solver.

        Args:
            fpath (str): Path to the FITS file.
            image (np.ndarray): Image data.
            header (fits.Header): FITS header.
            default_input (dict): Default input parameters.
        """
        self.fpath = fpath
        self.header = header
        self.image = image
        self.default_input = default_input

    # --- Remove WCS Keys ---
    def remove_wcs_keys(self, delete_keys: bool = True) -> fits.Header:
        """
        Removes or renames specific WCS-related keywords from the FITS header.

        Args:
            delete_keys (bool): If True, delete the keys; otherwise, rename them.

        Returns:
            fits.Header: Updated header.
        """
        logger.info("Removing any pre-existing WCS keys")
        keywords = {
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CUNIT1",
            "CUNIT2",
            "CTYPE1",
            "CTYPE2",
            "WCSAXES",
            "EQUINOX",
            "LONPOLE",
            "LATPOLE",
            "CDELT1",
            "CDELT2",
            "A_ORDER",
            "A_0_0",
            "A_0_1",
            "A_0_2",
            "A_1_0",
            "A_1_1",
            "A_2_0",
            "B_ORDER",
            "B_0_0",
            "B_0_1",
            "B_0_2",
            "B_1_0",
            "B_1_1",
            "B_2_0",
            "AP_ORDER",
            "AP_0_0",
            "AP_0_1",
            "AP_0_2",
            "AP_1_0",
            "AP_1_1",
            "AP_2_0",
            "BP_ORDER",
            "BP_0_0",
            "BP_0_1",
            "BP_0_2",
            "BP_1_0",
            "BP_1_1",
            "BP_2_0",
            "PROJP1",
            "PROJP3",
            "RADECSYS",
            "PV1_1",
            "PV1_2",
            "PV2_1",
            "PV2_2",
            "LTV1",
            "LTV2",
            "LTM1_1",
            "LTM2_2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
            "RADESYS",
            "PV1_0",
            "PV1_1",
            "PV1_2",
            "PV1_3",
            "PV1_4",
            "PV1_5",
            "PV1_6",
            "PV1_7",
            "PV1_8",
            "PV1_9",
            "PV1_10",
            "PV1_11",
            "PV1_12",
            "PV1_13",
            "PV1_14",
            "PV1_15",
            "PV1_16",
            "PV1_17",
            "PV1_18",
            "PV1_19",
            "PV1_20",
            "PV1_21",
            "PV1_22",
            "TNX_0_0",
            "TNX_1_0",
            "TNX_0_1",
            "TNX_2_0",
            "TNX_1_1",
            "TNX_0_2",
            "TNX_3_0",
            "TNX_2_1",
            "TNX_1_2",
            "TNX_0_3",
            "TNX_4_0",
            "TNX_3_1",
            "TNX_2_2",
            "TNX_1_3",
            "TNX_0_4",
            "TNX_5_0",
            "TNX_4_1",
            "TNX_3_2",
            "TNX_2_3",
            "TNX_1_4",
            "TNX_0_5",
            "TNX_6_0",
            "TNX_5_1",
            "TNX_4_2",
            "TNX_3_3",
            "TNX_2_4",
            "TNX_1_5",
            "TNX_0_6",
            "PC001001",
            "PC001002",
            "PC002001",
            "PC002002",
            "A_1_1",
            "A_1_2",
            "A_1_3",
            "A_2_0",
            "A_2_1",
            "A_2_2",
            "A_2_3",
            "A_3_0",
            "A_3_1",
            "A_3_2",
            "A_3_3",
            "B_1_1",
            "B_1_2",
            "B_1_3",
            "B_2_0",
            "B_2_1",
            "B_2_2",
            "B_2_3",
            "B_3_0",
            "B_3_1",
            "B_3_2",
            "B_3_3",
            "SIP_A",
            "SIP_B",
            "SIP_C",
            "SIP_D",
            "SIP_AP",
            "SIP_BP",
            "SIP_CP",
            "SIP_DP",
        }
        for key in keywords:
            try:
                if key in self.header:
                    if delete_keys:
                        del self.header[key]
                    else:
                        new_key = f"_{key[1:]}" if len(key) > 1 else f"_{key}"
                        self.header.rename_keyword(key, new_key)
            except Exception as e:
                logger.exception("Error handling key '%s': %s", key, e)
        return self.header

    # --- Clean Log ---
    def clean_log(self, input_file: str, output_file: str = None) -> str:
        """
        Clean ANSI escape sequences and extra lines from log files.

        Args:
            input_file (str): Path to the input log file.
            output_file (str): Path to save the cleaned log file.

        Returns:
            str: Path to the cleaned log file.
        """
        with open(input_file, "r") as f:
            content = f.read()
        clean_content = _ANSI_ESCAPE.sub("", content)
        clean_content = "\n".join(
            line for line in clean_content.split("\n") if line.strip()
        )
        output_file = output_file or input_file
        with open(output_file, "w") as f:
            f.write(clean_content)
        clean_subprocess_log(output_file)
        return output_file

    def _resolve_astrometry_data_dir(self, solvefield_exe: str) -> str | None:
        """
        Find the astrometry.net index-file directory for the given solve-field
        executable.  Linuxbrew/Homebrew installations keep indexes in a Cellar
        data directory that the binary does not discover on its own.
        """
        if not solvefield_exe:
            return None

        # Honour an existing, valid ASTROMETRY_DATA first.
        _existing = os.environ.get("ASTROMETRY_DATA")
        if _existing and os.path.isdir(_existing):
            return _existing

        # Derive from binary location: <prefix>/bin/solve-field -> <prefix>/data
        _exe = Path(solvefield_exe).resolve()
        _candidate = _exe.parent.parent / "data"
        if _candidate.is_dir() and any(_candidate.glob("index-*.fits")):
            return str(_candidate)

        _system_paths = [
            "/usr/share/astrometry",
            "/usr/share/astrometry/data",
            "/usr/local/share/astrometry",
            "/usr/local/share/astrometry/data",
            "/opt/astrometry/data",
            "/opt/local/share/astrometry",
            "/opt/local/share/astrometry/data",
        ]
        for _sp in _system_paths:
            if os.path.isdir(_sp) and any(glob.glob(os.path.join(_sp, "index-*.fits"))):
                return _sp

        # Search up the resolved binary tree for a data directory (handles nested
        # Linuxbrew/ Homebrew layouts).
        for _parent in _exe.parents:
            _data_dir = _parent / "data"
            if _data_dir.is_dir() and any(_data_dir.glob("index-*.fits")):
                return str(_data_dir)

        return None

    def _run_solve_field(
        self,
        args: list,
        wcs_file: str,
        timeout_sec: float,
        logpath: str,
    ) -> bool:
        """Run solve-field with given args; return True if wcs_file was created."""
        # Shared cascade budget: each attempt clamps to the time remaining
        # and the retries stop once it is spent, so an unsolvable field
        # gives up in bounded time instead of burning every per-attempt
        # cpulimit.  None means uncapped (budget disabled or unset).
        deadline = getattr(self, "_solve_deadline", None)
        if deadline is not None:
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0:
                if not getattr(self, "_solve_budget_warned", False):
                    logger.warning(
                        "solve-field total time budget exhausted; "
                        "skipping remaining solve attempts"
                    )
                    self._solve_budget_warned = True
                return False
            timeout_sec = min(float(timeout_sec), remaining)
        try:
            logf = open(logpath, "a", encoding="utf-8")
            logf.write(" ".join(map(str, args)) + "\n")
            kwargs = dict(shell=False, stdout=logf, stderr=subprocess.STDOUT)
            if os.name != "nt":
                kwargs["preexec_fn"] = os.setsid

            # solve-field needs its index dir; Linuxbrew/Homebrew put it in a
            # Cellar data dir the binary does not search by default.
            _env = os.environ.copy()
            _data_dir = self._resolve_astrometry_data_dir(
                getattr(self, "_solvefield_exe", "")
            )
            if _data_dir:
                _env["ASTROMETRY_DATA"] = _data_dir
                logf.write(f"# ASTROMETRY_DATA={_data_dir}\n\n")
            kwargs["env"] = _env

            try:
                pro = subprocess.Popen(args, **kwargs)
            except Exception:
                logf.close()
                raise
            timed_out = False
            try:
                pro.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name != "nt":
                    try:
                        os.killpg(os.getpgid(pro.pid), 9)
                    except (OSError, AttributeError):
                        pro.kill()
                else:
                    pro.kill()
            finally:
                logf.close()
                pro.wait()
                if timed_out:
                    logger.warning("solve-field exceeded timeout (%s s)", timeout_sec)
                    # SIGKILL mid-write can leave a truncated WCS file that
                    # isfile() would otherwise accept as a solved result.
                    if os.path.isfile(wcs_file):
                        try:
                            os.remove(wcs_file)
                        except OSError:
                            pass
            self.clean_log(logpath)
            # An unreadable or empty index directory means every remaining
            # attempt fails identically; stop the solve rather than burn
            # the rest of the cpulimit budget on guaranteed failures.
            try:
                with open(logpath, errors="replace") as _chk:
                    _sf_log = _chk.read()
            except OSError:
                _sf_log = ""
            if (
                "failed to open index directory" in _sf_log
                or "must list at least one index" in _sf_log
            ):
                raise AstrometryIndexError(
                    "astrometry.net index files are missing or unreadable "
                    f"(see {logpath}); solve-field cannot run. Restore the "
                    "index files (scripts/download_astrometry_indexes.py) "
                    "or fix add_path in astrometry.cfg."
                )
            return os.path.isfile(wcs_file)
        except AstrometryIndexError:
            raise
        except Exception as e:
            log_warning_from_exception(logger, "solve-field run failed", e)
            return False

    def _plate_solve_scamp(
        self, wcs_cfg: dict, cpulimit: float | None = None
    ) -> fits.Header | float:
        """
        Attempt WCS solution using SExtractor + SCAMP.

        Returns a FITS header with updated WCS on success, or np.nan on failure.
        """
        logger.info(log_step("WCS: SCAMP"))

        scamp_exe = wcs_cfg.get("scamp_exe_loc") or shutil.which("scamp")
        sex_exe = wcs_cfg.get("sextractor_exe_loc") or _resolve_sextractor_exe()
        if not scamp_exe and shutil.which("scamp") is None:
            logger.warning("SCAMP executable not found; cannot use SCAMP solver.")
            return np.nan
        if not sex_exe:
            logger.warning("SExtractor executable not found; cannot use SCAMP solver.")
            return np.nan
        scamp_exe = str(
            scamp_exe
            if scamp_exe and os.path.isfile(str(scamp_exe))
            else shutil.which("scamp")
        )
        sex_exe = str(
            sex_exe if sex_exe and os.path.isfile(str(sex_exe)) else _resolve_sextractor_exe()
        )

        if not os.path.isfile(self.fpath):
            logger.error("Image file not found: %s", self.fpath)
            return np.nan

        timeout_sec = (
            float(cpulimit)
            if cpulimit is not None
            else _wcs_cfg_float(wcs_cfg, "cpulimit", 60.0)
        )

        dirname = os.path.dirname(self.fpath)
        base = os.path.splitext(os.path.basename(self.fpath))[0]
        scamp_log = os.path.join(dirname, f"SCAMP_{base}.log")

        with tempfile.TemporaryDirectory() as temp_dir:
            param_file = os.path.join(temp_dir, "scamp.param")
            params = [
                "XWIN_IMAGE",
                "YWIN_IMAGE",
                "ERRAWIN_IMAGE",
                "ERRBWIN_IMAGE",
                "ERRTHETAWIN_IMAGE",
                "FLUX_AUTO",
                "FLUXERR_AUTO",
                "MAG_AUTO",
                "MAGERR_AUTO",
                "FLUX_RADIUS",
                "FWHM_IMAGE",
                "ELLIPTICITY",
                "BACKGROUND",
                "THRESHOLD",
                "FLAGS",
                "SNR_WIN",
                "XPEAK_IMAGE",
                "YPEAK_IMAGE",
                "XWIN_WORLD",
                "YWIN_WORLD",
                "X_WORLD",
                "Y_WORLD",
                "ERRA_WORLD",
                "ERRB_WORLD",
                "ERRTHETA_WORLD",
                "ALPHA_J2000",
                "DELTA_J2000",
            ]
            Path(param_file).write_text("\n".join(params))

            # SExtractor's FILTER_NAME must point to a writable convolution
            # file; generating it in temp_dir lets us omit repository `.conv`
            # files.
            conv_filter_path = os.path.join(temp_dir, "gaussian_7x7.conv")
            conv_fwhm_pix = _resolve_sextractor_conv_fwhm_pixels(
                wcs_cfg, self.header, self.default_input
            )
            create_conv_file(conv_filter_path, fwhm_pixels=conv_fwhm_pix)
            logger.info(
                "SCAMP/SExtractor convolution kernel built from FWHM=%.2f px",
                conv_fwhm_pix,
            )

            config_file = os.path.join(temp_dir, "scamp.sex")

            pixel_scale = self.default_input.get("pixel_scale") or 0
            if not pixel_scale and self.header.get("CDELT1") is not None:
                try:
                    pixel_scale = abs(float(self.header["CDELT1"])) * 3600.0
                except (TypeError, KeyError):
                    pass
            if not pixel_scale:
                logger.warning(
                    "SCAMP: no pixel_scale in config or header; using 0.1--5 arcsec/pix"
                )

            gain_val = (
                self.header.get("GAIN")
                or self.header.get("gain")
                or self.default_input.get("gain")
            )
            gain_str = str(float(gain_val)) if gain_val is not None else "0"
            satur_val = (
                self.header.get("SATURATE")
                or self.header.get("saturate")
                or self.default_input.get("saturate")
            )
            try:
                satur_float = float(satur_val)
                # SExtractor rejects inf/NaN SATUR_LEVEL values
                if not np.isfinite(satur_float) or satur_float <= 0:
                    satur_float = 1e7
                # Sentinel-scale placeholders (1e10/1e30 "no limit" markers)
                # are not detector levels; use the same no-limit stand-in as a
                # missing key instead of capping to a bogus 65535.
                elif satur_float >= 1e8:
                    satur_float = 1e7
                # Cap at 65535 (16-bit unsigned max); header values like 1e30
                # trigger "SATUR_LEVEL keyword out of range" in SExtractor.
                max_saturation = 65535
                if satur_float > max_saturation:
                    satur_float = max_saturation
                # Format as integer if whole number (SExtractor prefers this)
                if float(satur_float).is_integer():
                    satur_str = str(int(satur_float))
                else:
                    satur_str = str(satur_float)
            except Exception:
                satur_str = "65535"

            detect_thresh = str(_wcs_cfg_float(wcs_cfg, "sextractor_detect_thresh", 1.5))
            analysis_thresh = str(
                _wcs_cfg_float(wcs_cfg, "sextractor_analysis_thresh", 1.2)
            )
            detect_minarea = str(_wcs_cfg_int(wcs_cfg, "sextractor_detect_minarea", 5))
            detect_maxarea = str(_wcs_cfg_int(wcs_cfg, "sextractor_detect_maxarea", 0))
            deblend_nthresh = str(
                _wcs_cfg_int(wcs_cfg, "sextractor_deblend_nthresh", 32)
            )
            deblend_mincont = str(
                _wcs_cfg_float(wcs_cfg, "sextractor_deblend_mincont", 0.005)
            )
            back_type = str(wcs_cfg.get("sextractor_back_type", "AUTO"))
            back_value = str(_wcs_cfg_float(wcs_cfg, "sextractor_back_value", 0.0))
            back_size = str(_wcs_cfg_int(wcs_cfg, "sextractor_back_size", 64))
            back_filtersize = str(
                _wcs_cfg_int(wcs_cfg, "sextractor_back_filtersize", 3)
            )
            backphoto_type = str(wcs_cfg.get("sextractor_backphoto_type", "GLOBAL"))
            backphoto_thick = str(
                _wcs_cfg_int(wcs_cfg, "sextractor_backphoto_thick", 24)
            )
            back_filtthresh = str(
                _wcs_cfg_float(wcs_cfg, "sextractor_back_filtthresh", 0.0)
            )
            seeing_fwhm_arcsec = (
                float(conv_fwhm_pix) * float(pixel_scale)
                if pixel_scale and float(pixel_scale) > 0
                else 1.0
            )
            phot_apertures = str(
                _wcs_cfg_float(
                    wcs_cfg,
                    "sextractor_phot_apertures",
                    float(1.7 * conv_fwhm_pix),
                )
            )
            final_config = {
                "DETECT_TYPE": "CCD",
                "DETECT_MINAREA": detect_minarea,
                "DETECT_MAXAREA": detect_maxarea,
                "DETECT_THRESH": detect_thresh,
                "ANALYSIS_THRESH": analysis_thresh,
                "DEBLEND_NTHRESH": deblend_nthresh,
                "DEBLEND_MINCONT": deblend_mincont,
                "FILTER": "Y",
                "FILTER_NAME": conv_filter_path,
                "BACK_TYPE": back_type,
                "BACK_VALUE": back_value,
                "BACK_SIZE": back_size,
                "BACK_FILTERSIZE": back_filtersize,
                "BACKPHOTO_TYPE": backphoto_type,
                "BACKPHOTO_THICK": backphoto_thick,
                "BACK_FILTTHRESH": back_filtthresh,
                "CLEAN": "Y",
                "CLEAN_PARAM": "1.0",
                "PHOT_AUTOPARAMS": "2.5,3.5",
                "PHOT_APERTURES": phot_apertures,
                "SEEING_FWHM": str(seeing_fwhm_arcsec),
                "GAIN": gain_str,
                "SATUR_LEVEL": satur_str,
                "PIXEL_SCALE": str(pixel_scale),
                "VERBOSE_TYPE": "NORMAL",
                "CATALOG_TYPE": "FITS_LDAC",
                "PARAMETERS_NAME": param_file,
            }
            with open(config_file, "w") as f:
                for k, v in final_config.items():
                    f.write(f"{k}\t{v}\n")

            cat_path = os.path.join(temp_dir, f"{base}.ldac")
            # Run SExtractor/SCAMP with cwd=temp_dir so SCAMP writes *.head next to the
            # LDAC catalog (SCAMP defaults to the current working directory).
            image_fpath = os.path.abspath(self.fpath)
            sex_cmd = [
                sex_exe,
                image_fpath,
                "-c",
                config_file,
                "-CATALOG_NAME",
                cat_path,
            ]
            try:
                logger.info("Running SExtractor for SCAMP catalog")
                result = subprocess.run(
                    sex_cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_sec,
                    cwd=temp_dir,
                )
                if result.stdout:
                    logger.debug(
                        "SExtractor stdout: %s", result.stdout.decode("utf-8", errors="replace")
                    )
                if result.stderr:
                    _stderr = strip_subprocess_noise(
                        result.stderr.decode("utf-8", errors="replace")
                    )
                    if _stderr:
                        logger.warning("SExtractor stderr: %s", _stderr)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "SExtractor for SCAMP exceeded timeout (%.1f s)", timeout_sec
                )
                return np.nan
            if not os.path.isfile(cat_path):
                logger.warning(
                    "SCAMP: SExtractor catalog not created at %s; aborting SCAMP solve.",
                    cat_path
                )
                try:
                    temp_files = os.listdir(temp_dir)
                    logger.warning(
                        "SCAMP: Temp directory contents: %s", ", ".join(temp_files)
                    )
                except Exception:
                    pass
                return np.nan

            astref_cat = str(wcs_cfg.get("scamp_astref_catalog", "GAIA-DR3"))
            scamp_cfg_path = os.path.join(temp_dir, "wcs_refine.scamp")
            _snth = wcs_cfg.get("scamp_nthreads")
            if _snth is None:
                _snth = 4
            try:
                scamp_threads = int(_snth)
            except Exception:
                scamp_threads = 4
            # Keep the pipeline "serial across images" (nCPU=1) but allow SCAMP to
            # use a small amount of internal threading by default.
            if scamp_threads < 4:
                scamp_threads = 4
            logger.info("SCAMP: using %d threads", scamp_threads)
            # SCAMP's built-in REF_TIMEOUT is only 10 s; often too short for
            # slow Vizier links.
            scamp_cfg = {
                "SOLVE_ASTROM": "Y",
                "SOLVE_PHOTOM": "N",
                "REF_TIMEOUT": _wcs_cfg_int(wcs_cfg, "scamp_ref_timeout", 60),
                "DISTORT_DEGREES": _wcs_cfg_int(wcs_cfg, "scamp_distort_degrees", 4),
                "MATCH": "Y",
                "MATCH_RESOL": 0,
                "MATCH_FLIPPED": "Y",
                "WRITE_XML": "N",
                "VERBOSE_TYPE": str(wcs_cfg.get("scamp_verbose_type", "LOG")),
                "ASTREF_CATALOG": astref_cat,
                "ASTREF_WEIGHT": _wcs_cfg_int(wcs_cfg, "scamp_astref_weight", 1),
                "ASTREFMAG_KEY": str(wcs_cfg.get("scamp_astrefmag_key", "MAG_AUTO")),
                "ASTREFMAGERR_KEY": str(
                    wcs_cfg.get("scamp_astrefmagerr_key", "MAGERR_AUTO")
                ),
                "ASTREFCENT_KEYS": str(
                    wcs_cfg.get("scamp_astrefcent_keys", "XWIN_WORLD,YWIN_WORLD")
                ),
                "ASTREFERR_KEYS": str(
                    wcs_cfg.get("scamp_astreferr_keys", "ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD")
                ),
                "DISTORT_KEYS": str(wcs_cfg.get("scamp_distort_keys", "XWIN_IMAGE,YWIN_IMAGE")),
                "SN_THRESHOLDS": str(wcs_cfg.get("scamp_sn_thresholds", "5.0,100000.0")),
                "CROSSID_RADIUS": _wcs_cfg_float(wcs_cfg, "scamp_crossid_radius", 2.5),
                "POSITION_MAXERR": _wcs_cfg_float(
                    wcs_cfg, "scamp_position_maxerr", 1.0
                ),
                "FWHM_THRESHOLDS": str(wcs_cfg.get("scamp_fwhm_thresholds", "1.0,15.0")),
                "MOSAIC_TYPE": str(wcs_cfg.get("scamp_mosaic_type", "UNCHANGED")),
                "STABILITY_TYPE": str(
                    wcs_cfg.get("scamp_stability_type", "EXPOSURE")
                ),
            }
            _ref_srv = wcs_cfg.get("scamp_ref_server") or "vizier.cfa.harvard.edu"
            scamp_cfg["REF_SERVER"] = str(_ref_srv).strip()
            with open(scamp_cfg_path, "w") as f:
                for k, v in scamp_cfg.items():
                    f.write(f"{k}\t{v}\n")
            scamp_cmd = [
                scamp_exe,
                cat_path,
                "-c",
                scamp_cfg_path,
                "-NTHREADS",
                str(scamp_threads),
            ]
            try:
                logger.info("Running SCAMP: %s", " ".join(map(str, scamp_cmd)))
                with open(scamp_log, "w", encoding="utf-8") as logf:
                    subprocess.run(
                        scamp_cmd,
                        check=False,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        timeout=timeout_sec,
                        cwd=temp_dir,
                    )
            except subprocess.TimeoutExpired:
                logger.warning("SCAMP exceeded timeout (%.1f s)", timeout_sec)
                return np.nan

            head_candidates = sorted(glob.glob(os.path.join(temp_dir, "*.head")))
            if not head_candidates:
                try:
                    with open(scamp_log, "r", encoding="utf-8", errors="ignore") as f:
                        scamp_out = f.read().strip()
                    if scamp_out:
                        logger.warning("SCAMP output:\n%s", scamp_out[-2000:])
                        _log_scamp_vizier_failure_hint(
                            logger, scamp_out, scamp_log
                        )
                except Exception:
                    pass
                logger.warning("SCAMP did not produce a .head file; WCS not updated.")
                return np.nan
            head_path = head_candidates[0]
            logger.debug("SCAMP produced .head file: %s", head_path)

            try:
                wcs_header = fits.Header.fromtextfile(head_path)
                wcs_header = _normalize_projection_codes(wcs_header, inplace=False)

                self.header = remove_wcs_from_header(self.header)
                _wcs_prefixes = (
                    "CRPIX",
                    "CRVAL",
                    "CTYPE",
                    "CD",
                    "PC",
                    "CDELT",
                    "CROTA",
                    "PV",
                    "LONPOLE",
                    "LATPOLE",
                    "EQUINOX",
                    "WCSNAME",
                    "CUNIT",
                    "WCSAXES",
                    "PROJP",
                    "LTV",
                    "LTM",
                    "RADECSYS",
                    "RADESYS",
                    "RADYSYS",
                    "LONGPOLE",
                    "TNX",
                    "SIP_",
                )
                _wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
                for key in wcs_header:
                    if key in ("NAXIS", "NAXIS1", "NAXIS2"):
                        continue
                    is_wcs = any(key.startswith(p) for p in _wcs_prefixes)
                    if not is_wcs and "_" in key:
                        stem = key.split("_")[0] + "_"
                        is_wcs = stem in _wcs_stems and key.startswith(stem.rstrip("_"))
                    if is_wcs:
                        self.header[key] = wcs_header[key]
                self.header["NAXIS1"] = self.image.shape[1]
                self.header["NAXIS2"] = self.image.shape[0]
                logger.info("SCAMP WCS solution applied to header.")
                return self.header
            except Exception as exc:
                log_warning_from_exception(
                    logger, "Failed to apply SCAMP WCS header", exc
                )
                return np.nan

    def _refine_distortion_with_scamp_from_rough_wcs(
        self,
        rough_wcs_header: fits.Header,
        wcs_cfg: dict,
        cpulimit: float | None = None,
    ) -> fits.Header | float:
        """
        Two-stage solve:
          1) solve-field provides rough WCS (passed in as ``rough_wcs_header``)
          2) SCAMP refines distortion (prefer TPV/PV) starting from that rough WCS

        Returns refined header on success, else np.nan.
        """
        try:
            logger.info(
                "Refining distortion with SCAMP using solve-field rough WCS seed."
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                seed_fpath = os.path.join(tmpdir, "scamp_seed.fits")
                seed_header = self.header.copy()
                seed_header = remove_wcs_from_header(seed_header)

                # Carry over WCS terms from solve-field rough solution.
                for k in rough_wcs_header:
                    if k in ("NAXIS", "NAXIS1", "NAXIS2"):
                        continue
                    try:
                        seed_header[k] = rough_wcs_header[k]
                    except Exception:
                        pass
                seed_header["NAXIS1"] = self.image.shape[1]
                seed_header["NAXIS2"] = self.image.shape[0]
                safe_fits_write(seed_fpath, self.image, seed_header, output_verify="ignore")

                seeded_solver = WCSSolver(
                    fpath=seed_fpath,
                    image=self.image,
                    header=seed_header,
                    default_input=self.default_input,
                )
                scamp_header = seeded_solver._plate_solve_scamp(
                    wcs_cfg=wcs_cfg, cpulimit=cpulimit
                )
                if isinstance(scamp_header, fits.Header):
                    logger.info(
                        "SCAMP distortion refinement succeeded from solve-field seed."
                    )
                    return scamp_header
                logger.warning(
                    "SCAMP distortion refinement unavailable/failed; keeping solve-field WCS."
                )
                return np.nan
        except Exception as exc:
            log_warning_from_exception(
                logger,
                "SCAMP distortion refinement from solve-field seed failed",
                exc,
            )
            return np.nan

    def _select_best_scamp_solution_after_solve(
        self,
        rough_wcs_header: fits.Header,
        wcs_cfg: dict,
        cpulimit: float | None,
        matched_points: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> fits.Header | float:
        """
        Run one or more SCAMP refinement trials and return the best header.

        If matched points are available, select by lowest median residual and then
        lowest p95 residual (arcsec). Otherwise return the first successful trial.
        """
        optimize_scamp = bool(wcs_cfg.get("scamp_optimize_after_solve", False))
        trials = [("base", dict(wcs_cfg))]
        if optimize_scamp and matched_points is not None:
            trials = _build_scamp_optimization_trials(wcs_cfg)

        best_trial_label = None
        best_trial_header = None
        best_trial_med = np.inf
        best_trial_p95 = np.inf
        x_m = y_m = ra_m = dec_m = None
        if matched_points is not None:
            x_m, y_m, ra_m, dec_m = matched_points

        for trial_label, trial_cfg in trials:
            scamp_header = self._refine_distortion_with_scamp_from_rough_wcs(
                rough_wcs_header=rough_wcs_header,
                wcs_cfg=trial_cfg,
                cpulimit=cpulimit,
            )
            if not isinstance(scamp_header, fits.Header):
                continue

            if x_m is not None and y_m is not None and ra_m is not None and dec_m is not None:
                scamp_wcs = get_wcs(scamp_header)
                if scamp_wcs is not None:
                    med_arcsec, p95_arcsec, _ = _best_wcs_match_separation_stats_arcsec(
                        scamp_wcs, x_m, y_m, ra_m, dec_m
                    )
                else:
                    logger.debug("SCAMP trial: get_wcs returned None, skipping residual calculation")
                    med_arcsec, p95_arcsec = np.nan, np.nan
                logger.info(
                    "SCAMP trial '%s': matched-star residual med/p95=%.3f/%.3f arcsec",
                    trial_label,
                    float(med_arcsec),
                    float(p95_arcsec),
                )
                if np.isfinite(med_arcsec) and (
                    med_arcsec < best_trial_med
                    or (
                        np.isclose(med_arcsec, best_trial_med)
                        and p95_arcsec < best_trial_p95
                    )
                ):
                    best_trial_label = str(trial_label)
                    best_trial_header = scamp_header
                    best_trial_med = float(med_arcsec)
                    best_trial_p95 = float(p95_arcsec)
            elif best_trial_header is None:
                best_trial_label = str(trial_label)
                best_trial_header = scamp_header

        if isinstance(best_trial_header, fits.Header):
            if best_trial_label is None:
                best_trial_label = "base"
            # Validate that the distortion model is invertible (catches extreme SCAMP solutions)
            try:
                test_wcs = get_wcs(best_trial_header)
                if test_wcs is not None:
                    # Round-trip at the image centre catches non-invertible,
                    # extreme distortions. Numpy 0-based centre is
                    # (nx-1)/2, (ny-1)/2.
                    cx = (best_trial_header.get("NAXIS1", 1000) - 1) / 2
                    cy = (best_trial_header.get("NAXIS2", 1000) - 1) / 2
                    ra_test, dec_test = test_wcs.all_pix2world(cx, cy, 0)
                    # world -> pix must reproduce the centre pixel
                    x_back, y_back = test_wcs.all_world2pix(ra_test, dec_test, 0, maxiter=50)
                    if not (np.isfinite(x_back) and np.isfinite(y_back)):
                        raise ValueError("WCS round-trip failed")
                    logger.debug("SCAMP WCS passed invertibility test at image center")
            except Exception as inv_exc:
                logger.warning(
                    "SCAMP trial '%s' distortion model failed invertibility test: %s. Rejecting extreme distortion solution.",
                    best_trial_label, inv_exc
                )
                return np.nan  # Reject this SCAMP solution, will fall back to solve-field SIP

            logger.info(
                "Using SCAMP WCS solution (trial='%s', preferred TPV/PV distortion model).",
                best_trial_label,
            )
            _log_header_distortion_coefficients(
                best_trial_header, prefix="SCAMP solved WCS"
            )
            return best_trial_header
        logger.warning("SCAMP unavailable/failed after solve-field; keeping solve-field WCS.")
        return np.nan

    # --- Robust linear plate-model refinement from matched (pixel, sky) points ---
    def _fit_wcs_from_points(
        self,
        x_pix: np.ndarray,
        y_pix: np.ndarray,
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
    ) -> dict | None:
        """
        Fit a linear tangent-plane WCS (CD + CRPIX + CRVAL) from matched
        (pixel, sky) pairs using a sigma-clipped least-squares plate model.

        astropy 8 removed ``WCS.fit_wcs_from_points``, so a full SIP
        polynomial refit is unavailable here.  This method robustly refits
        the *linear* terms (center, scale, rotation) which is the dominant
        source of residual offset when solve-field's pixel-scale hint or a
        small zero-point offset biases the solution.  Distortion (SIP/PV)
        terms already present in the solved header are left untouched.

        Returns a dict with ``keywords`` (CD1_1, CD1_2, CD2_1, CD2_2,
        CRPIX1, CRPIX2, CRVAL1, CRVAL2), ``n_used``, and ``rms_arcsec``, or
        ``None`` when there are too few points or the fit fails.
        """
        x = np.asarray(x_pix, dtype=float)
        y = np.asarray(y_pix, dtype=float)
        ra = np.asarray(ra_deg, dtype=float)
        dec = np.asarray(dec_deg, dtype=float)
        try:
            keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(ra) & np.isfinite(dec)
            keep &= (ra >= 0.0) & (ra <= 360.0) & (dec >= -90.0) & (dec <= 90.0)
            x, y, ra, dec = x[keep], y[keep], ra[keep], dec[keep]
            if x.size < 3:
                return None

            xc, yc = float(np.median(x)), float(np.median(y))
            ra0, dec0 = float(np.median(ra)), float(np.median(dec))
            cos_dec0 = float(np.cos(np.deg2rad(dec0)))
            if cos_dec0 <= 1e-6:
                cos_dec0 = 1e-6

            # Tangent-plane sky offsets (degrees) about (ra0, dec0).
            xi = (ra - ra0) * cos_dec0   # xi ~ dRA * cos(dec)
            eta = dec - dec0             # eta ~ dDec
            dx = x - xc
            dy = y - yc

            idx = np.arange(x.size)
            for _iteration in range(3):
                if idx.size < 3:
                    return None
                # Design matrix [1, dx, dy] -> [xi, eta].
                A = np.column_stack([np.ones(idx.size), dx[idx], dy[idx]])
                try:
                    coef_xi, _, _, _ = np.linalg.lstsq(A, xi[idx], rcond=None)
                    coef_eta, _, _, _ = np.linalg.lstsq(A, eta[idx], rcond=None)
                except np.linalg.LinAlgError:
                    return None
                pred_xi = A @ coef_xi
                pred_eta = A @ coef_eta
                # Residual arcsec on the sky (projected by cos(dec0) for RA).
                res_ra = (pred_xi - xi[idx]) / cos_dec0
                res_dec = pred_eta - eta[idx]
                res_arcsec = np.hypot(res_ra * cos_dec0, res_dec) * 3600.0
                # Robust sigma from MAD: sigma = 1.4826 * MAD.  Dividing by
                # 0.6745 (= 1/1.4826) AND multiplying by 1.4826 applied the
                # conversion twice, inflating the clip threshold from ~4.4*MAD
                # to ~6.6*MAD and letting astrometric outliers into the fit.
                sigma = float(
                    1.4826
                    * np.nanmedian(np.abs(res_arcsec - np.nanmedian(res_arcsec)))
                )
                if not np.isfinite(sigma) or sigma <= 0:
                    break
                good = res_arcsec <= 3.0 * sigma
                if good.all():
                    idx = np.arange(x.size)
                    break
                idx = idx[good]
            if idx.size < 3:
                return None

            # Final fit on the retained inliers.
            A = np.column_stack([np.ones(idx.size), dx[idx], dy[idx]])
            coef_xi, _, _, _ = np.linalg.lstsq(A, xi[idx], rcond=None)
            coef_eta, _, _, _ = np.linalg.lstsq(A, eta[idx], rcond=None)
            pred_xi = A @ coef_xi
            pred_eta = A @ coef_eta
            res_ra = (pred_xi - xi[idx]) / cos_dec0
            res_dec = pred_eta - eta[idx]
            rms_arcsec = float(
                np.sqrt(np.mean((res_ra * cos_dec0) ** 2 + res_dec**2)) * 3600.0
            )

            # CD maps pixel -> tangent-plane degrees (xi, eta).
            cd11, cd12 = float(coef_xi[1]), float(coef_xi[2])
            cd21, cd22 = float(coef_eta[1]), float(coef_eta[2])
            # CRVAL is the sky at CRPIX=(xc, yc) where dx=dy=0.
            crval1 = ra0 + float(coef_xi[0]) / cos_dec0
            crval2 = dec0 + float(coef_eta[0])

            return {
                "keywords": {
                    "CD1_1": cd11,
                    "CD1_2": cd12,
                    "CD2_1": cd21,
                    "CD2_2": cd22,
                    "CRPIX1": xc,
                    "CRPIX2": yc,
                    "CRVAL1": crval1,
                    "CRVAL2": crval2,
                },
                "n_used": int(idx.size),
                "rms_arcsec": rms_arcsec,
            }
        except Exception as exc:
            log_warning_from_exception(
                logger, "_fit_wcs_from_points failed", exc
            )
            return None

    def _fetch_verify_catalog(self, wcs_cfg: dict):
        """
        Fetch Gaia stars covering this frame for input-WCS verification.

        Cone search on the header CRVAL with radius = half the field
        diagonal (capped at 0.5 deg), keeping sources brighter than
        ``wcs.verify_gmag_limit``.  Returns (ra_deg, dec_deg) arrays, or
        None on any failure - every failure path logs a warning so the
        caller can fall through to the normal solve.
        """
        try:
            wcs_obj = get_wcs(self.header)
            if wcs_obj is None:
                logger.warning(
                    "verify_input_wcs: no usable header WCS for the Gaia "
                    "cone center; proceeding to solve."
                )
                return None
            ra0 = float(self.header.get("CRVAL1", np.nan))
            dec0 = float(self.header.get("CRVAL2", np.nan))
            if not (np.isfinite(ra0) and np.isfinite(dec0)):
                logger.warning(
                    "verify_input_wcs: header CRVAL not finite; proceeding "
                    "to solve."
                )
                return None
            ny, nx = self.image.shape[0], self.image.shape[1]
            scale = pixel_area_scale_arcsec(wcs_obj, (ny, nx))
            if np.isfinite(scale) and scale > 0:
                radius_deg = 0.5 * scale * float(np.hypot(nx, ny)) / 3600.0
            else:
                radius_deg = 0.25
            radius_deg = float(np.clip(radius_deg, 0.02, 0.5))
            gmag = _wcs_cfg_float(wcs_cfg, "verify_gmag_limit", 19.0)
            from astroquery.gaia import Gaia

            query = (
                "SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
                "WHERE CONTAINS(POINT('ICRS', ra, dec), "
                f"CIRCLE('ICRS', {ra0:.6f}, {dec0:.6f}, {radius_deg:.6f})) = 1 "
                f"AND phot_g_mean_mag <= {gmag:.2f}"
            )
            job = Gaia.launch_job(query, dump_to_file=False)
            tbl = job.get_results()
            if tbl is None or len(tbl) == 0:
                logger.warning(
                    "verify_input_wcs: Gaia cone query returned no rows; "
                    "proceeding to solve."
                )
                return None
            ra_cat = np.asarray(tbl["ra"], dtype=float)
            dec_cat = np.asarray(tbl["dec"], dtype=float)
            finite = np.isfinite(ra_cat) & np.isfinite(dec_cat)
            ra_cat, dec_cat = ra_cat[finite], dec_cat[finite]
            if ra_cat.size == 0:
                logger.warning(
                    "verify_input_wcs: Gaia catalog has no finite "
                    "coordinates; proceeding to solve."
                )
                return None
            logger.info(
                "verify_input_wcs: fetched %d Gaia stars (G<=%.1f, "
                "r=%.3f deg around %.5f,%.5f).",
                int(ra_cat.size),
                gmag,
                radius_deg,
                ra0,
                dec0,
            )
            return ra_cat, dec_cat
        except Exception as exc:
            log_warning_from_exception(
                logger,
                "verify_input_wcs: Gaia query failed; proceeding to solve",
                exc,
            )
            return None

    def _verify_input_wcs_against_catalog(
        self, wcs_cfg: dict, detected_xy: np.ndarray
    ) -> bool:
        """
        Cross-match detected sources to Gaia through the input WCS.

        Returns True only when enough pairs match and both the median and
        p95 residuals stay within the configured tolerances; every other
        outcome (no catalog, too few matches, residuals too large,
        unexpected failure) returns False so the caller solves normally.
        """
        try:
            wcs_obj = get_wcs(self.header)
            if wcs_obj is None:
                return False
            cat = self._fetch_verify_catalog(wcs_cfg)
            if cat is None:
                return False
            ra_cat, dec_cat = cat
            med, p95, n_matches = verify_wcs_residuals(
                wcs_obj,
                detected_xy[:, 0],
                detected_xy[:, 1],
                ra_cat,
                dec_cat,
                _wcs_cfg_float(wcs_cfg, "verify_match_radius_arcsec", 2.0),
            )
            min_matches = _wcs_cfg_int(wcs_cfg, "verify_min_matches", 8)
            if not np.isfinite(med) or n_matches < min_matches:
                logger.info(
                    "Input WCS verification inconclusive (n=%d matched "
                    "pairs, need %d) - proceeding to solve.",
                    int(n_matches),
                    int(min_matches),
                )
                return False
            max_med = _wcs_cfg_float(wcs_cfg, "verify_max_median_arcsec", 0.5)
            max_p95 = _wcs_cfg_float(wcs_cfg, "verify_max_p95_arcsec", 1.5)
            if med <= max_med and p95 <= max_p95:
                logger.log(
                    STATUS,
                    'Input WCS verified (n=%d, med=%.3f", p95=%.3f") - '
                    "skipping solve",
                    int(n_matches),
                    float(med),
                    float(p95),
                )
                return True
            logger.info(
                "Input WCS failed verification (n=%d, med=%.3f\", "
                "p95=%.3f\"; limits %.3f/%.3f arcsec) - proceeding to solve.",
                int(n_matches),
                float(med),
                float(p95),
                max_med,
                max_p95,
            )
            return False
        except Exception as exc:
            log_warning_from_exception(
                logger,
                "Input WCS verification failed; proceeding to solve",
                exc,
            )
            return False

    # --- Plate Solve ---
    def plate_solve(
        self,
        solvefield_exe: str = None,
        downsample: int = None,
        cpulimit: int = None,
        background_std: str = None,
        skip_verify: bool = True,
        n_detected_sources: int = None,
        detected_xy=None,
    ) -> fits.Header:
        """
        Run a WCS solver for this image.

        When wcs.solver is "scamp" and SCAMP/sex are available, use SCAMP
        (with SExtractor) to derive the WCS. Otherwise, fall back to
        Astrometry.net solve-field.

        ``detected_xy`` optionally carries the detected source positions
        (Nx2 array or table with x_pix/y_pix columns) used to verify an
        existing header WCS against Gaia before solving.
        """
        wcs_cfg = self.default_input.get("wcs") or {}
        redo_requested = bool(wcs_cfg.get("redo_wcs", True))
        projection_type = str(wcs_cfg.get("projection_type", "SIP")).strip().upper()
        if projection_type not in {"TPV", "SIP"}:
            logger.warning(
                "Unknown wcs.projection_type=%r; defaulting to SIP flow.",
                projection_type,
            )
            projection_type = "SIP"
        use_tpv_flow = projection_type == "TPV"
        logger.info(
            "WCS projection_type=%s (%s)",
            projection_type,
            "solve-field + SCAMP" if use_tpv_flow else "solve-field only",
        )
        # Set when a newly solved WCS is adopted (as opposed to the input
        # WCS surviving a verify/keep-input/fallback path); the caller uses
        # it for provenance.
        self.solved_new_wcs = False
        if redo_requested and not skip_verify:
            logger.info(
                "redo_wcs=True: forcing fresh solve (skip verify of input WCS)."
            )
            skip_verify = True
        solver = str(wcs_cfg.get("solver", "astrometry")).strip().lower()

        # --- Optional early exit: verify the input WCS against Gaia ---
        # redo_wcs=True always solves; this gate only applies when
        # redo_wcs is False and the opt-in verify_input_wcs is set.
        _det_xy = _coerce_detected_xy(detected_xy)
        if _should_verify_input_wcs(
            redo_requested,
            _wcs_cfg_bool(wcs_cfg, "verify_input_wcs", False),
            get_wcs(self.header) is not None,
            len(_det_xy) if _det_xy is not None else None,
            _wcs_cfg_int(wcs_cfg, "verify_min_matches", 8),
        ) and self._verify_input_wcs_against_catalog(wcs_cfg, _det_xy):
            return self.header
        # Inconclusive or failed verification is already logged; the
        # normal solve continues below.

        # Optional SCAMP path
        if solver == "scamp" and use_tpv_flow:
            try:
                header = self._plate_solve_scamp(wcs_cfg=wcs_cfg, cpulimit=cpulimit)
                if isinstance(header, fits.Header):
                    self.solved_new_wcs = True
                    return header
                logger.warning(
                    "SCAMP solve failed or unavailable; falling back to Astrometry.net."
                )
            except Exception as exc:
                logger.warning(
                    "SCAMP solve raised an exception (%s); falling back to Astrometry.net.",
                    exc,
                )
        elif solver == "scamp" and not use_tpv_flow:
            logger.warning(
                "solver=scamp requires projection_type=TPV; falling back to "
                "solve-field only (SIP)."
            )

        logger.info(
            log_step(
                f"WCS: Astrometry.net - {os.path.basename(self.fpath)}"
            )
        )
        # Resolve solve-field executable: honour explicit config, then PATH,
        # then common installation locations outside the active conda env.
        if solvefield_exe:
            _candidate = str(solvefield_exe)
            if os.path.isfile(_candidate) and os.access(_candidate, os.X_OK):
                solvefield_exe = _candidate
            else:
                solvefield_exe = shutil.which(_candidate)
        else:
            solvefield_exe = shutil.which("solve-field")

        if not solvefield_exe or not os.path.isfile(solvefield_exe):
            _common_paths = [
                "/home/linuxbrew/.linuxbrew/bin/solve-field",
                "/opt/homebrew/bin/solve-field",
                "/usr/local/bin/solve-field",
                "/usr/bin/solve-field",
                "/opt/astrometry/bin/solve-field",
                os.path.expanduser("~/.linuxbrew/bin/solve-field"),
                os.path.expanduser("~/homebrew/bin/solve-field"),
            ]
            # Also search Linuxbrew/ Homebrew Cellar versions
            _cellar_paths = sorted(
                glob.glob("/home/linuxbrew/.linuxbrew/Cellar/astrometry-net/*/bin/solve-field")
                + glob.glob("/opt/homebrew/Cellar/astrometry-net/*/bin/solve-field")
                + glob.glob(os.path.expanduser("~/.linuxbrew/Cellar/astrometry-net/*/bin/solve-field"))
                + glob.glob(os.path.expanduser("~/homebrew/Cellar/astrometry-net/*/bin/solve-field")),
                reverse=True,
            )
            for _p in _common_paths + _cellar_paths:
                if os.path.isfile(_p) and os.access(_p, os.X_OK):
                    solvefield_exe = _p
                    break

        if not solvefield_exe or not os.path.isfile(str(solvefield_exe)):
            logger.warning(
                "Astrometry.net 'solve-field' executable not found;\n"
                "    skipping solve-field WCS step. To install with conda:\n"
                "    conda install -c conda-forge astrometry\n"
                "    Then ensure 'solve-field' is on PATH or set\n"
                "    wcs.solve_field_exe_loc in your YAML."
            )
            if get_wcs(self.header) is not None:
                logger.warning(
                    "Falling back to existing header WCS (no solver available)."
                )
                return self.header
            return np.nan
        else:
            logger.debug("Using solve-field executable: %s", solvefield_exe)
            self._solvefield_exe = solvefield_exe
        if not os.path.isfile(self.fpath):
            logger.error("Image file not found: %s", self.fpath)
            return np.nan

        # Re-fetch in case SCAMP path modified config
        wcs_cfg = self.default_input.get("wcs") or {}
        # Optional profile-based overrides (useful for crowded fields).
        # Set in YAML as:
        #   wcs:
        #     profile: crowded
        #     crowded: { ... overrides ... }
        try:
            profile = str(wcs_cfg.get("profile", "default")).strip().lower()
        except Exception:
            profile = "default"
        if profile == "crowded":
            overrides = wcs_cfg.get("crowded") or {}
            if isinstance(overrides, dict) and overrides:
                merged = dict(wcs_cfg)
                merged.update(overrides)
                wcs_cfg = merged
                logger.info("WCS profile=crowded: applying crowded-field overrides")
        timeout_sec = (
            float(cpulimit)
            if cpulimit is not None
            else _wcs_cfg_float(wcs_cfg, "cpulimit", 45.0)
        )
        # Later retries get a bigger budget: a hinted solve answers fast or
        # not at all, while wide-scale and blind sweeps need real CPU time
        # before they can be called a genuine failure.
        late_timeout_sec = max(
            timeout_sec, _wcs_cfg_float(wcs_cfg, "late_cpulimit", 120.0)
        )
        blind_timeout_sec = max(
            timeout_sec, _wcs_cfg_float(wcs_cfg, "blind_cpulimit", 300.0)
        )
        # Shared wall-clock budget for every solve-field attempt in this
        # call (all retries, tweak orders, and downsample levels).  Without
        # it a field that fails everywhere burns each per-attempt cpulimit
        # in sequence - tens of minutes to hours.  null or <=0 disables.
        try:
            _total_budget_sec = float(
                wcs_cfg.get("solve_field_total_budget", 1800.0) or 0.0
            )
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid wcs.solve_field_total_budget=%r.",
                wcs_cfg.get("solve_field_total_budget"),
            )
            _total_budget_sec = 0.0
        self._solve_deadline = (
            time.monotonic() + _total_budget_sec
            if _total_budget_sec > 0
            else None
        )
        self._solve_budget_warned = False
        downsample = (
            int(downsample)
            if downsample is not None
            else _wcs_cfg_int(wcs_cfg, "downsample", 2)
        )

        # --- Prepare paths ---
        # The output WCS file must be unique per invocation: a fixed name is
        # shared by every image in the directory, so a concurrent solve
        # (nCPU>1) or a leftover file from an interrupted run would otherwise
        # be adopted as this image's WCS.
        dirname = os.path.dirname(self.fpath)
        base = os.path.splitext(os.path.basename(self.fpath))[0]
        _tmp_tag = f"{base}_{os.getpid()}"
        wcs_file = os.path.join(dirname, f"astrometry_{_tmp_tag}.wcs.fits")
        if os.path.isfile(wcs_file):
            os.remove(wcs_file)
        astrometry_log_fpath = os.path.join(dirname, f"Astrometry_{base}.log")
        sextractor_exe = (
            wcs_cfg.get("sextractor_exe_loc")
            or _resolve_sextractor_exe()
        )
        use_sextractor = sextractor_exe is not None
        # --- Create a temporary SExtractor config file ---
        with tempfile.TemporaryDirectory() as temp_dir:
            param_file = os.path.join(temp_dir, "default.param")
            params = [
                "XWIN_IMAGE",
                "YWIN_IMAGE",
                "FLUX_AUTO",
                "FLUXERR_AUTO",
                "MAG_AUTO",
                "MAGERR_AUTO",
                "FLUX_RADIUS",
                "FWHM_IMAGE",
                "ELLIPTICITY",
                "BACKGROUND",
                "THRESHOLD",
                "FLAGS",
                "SNR_WIN",
                "XPEAK_IMAGE",
                "YPEAK_IMAGE",
            ]
            Path(param_file).write_text("\n".join(params))
            config_file = os.path.join(temp_dir, "default.sex")

            # SExtractor needs a writable FILTER_NAME .conv file; generate it
            # in temp_dir.
            conv_filter_path = os.path.join(temp_dir, "gaussian_7x7.conv")
            conv_fwhm_pix = _resolve_sextractor_conv_fwhm_pixels(
                wcs_cfg, self.header, self.default_input
            )
            create_conv_file(conv_filter_path, fwhm_pixels=conv_fwhm_pix)
            logger.debug(
                "solve-field/SExtractor convolution kernel built from FWHM=%.2f px",
                conv_fwhm_pix,
            )

            pixel_scale = self.default_input.get("pixel_scale") or 0
            if not pixel_scale and self.header.get("CDELT1") is not None:
                try:
                    pixel_scale = (
                        abs(float(self.header["CDELT1"])) * 3600.0
                    )  # deg to arcsec/pix
                except (TypeError, KeyError):
                    pass
            if not pixel_scale:
                logger.warning(
                    "No pixel_scale in config or header; using wide scale "
                    "bounds (wcs.wide_scale_low/high_arcsec)"
                )
            # SExtractor: use header gain when available for better centroid weighting
            gain_val = (
                self.header.get("GAIN")
                or self.header.get("gain")
                or self.default_input.get("gain")
            )
            gain_str = str(float(gain_val)) if gain_val is not None else "0"
            detect_thresh = str(
                _wcs_cfg_float(wcs_cfg, "sextractor_detect_thresh", 1.5)
            )
            final_config = {
                "DETECT_TYPE": "CCD",
                "DETECT_MINAREA": str(
                    _wcs_cfg_int(wcs_cfg, "sextractor_detect_minarea", 5)
                ),
                "DETECT_THRESH": detect_thresh,
                "ANALYSIS_THRESH": str(
                    _wcs_cfg_float(wcs_cfg, "sextractor_analysis_thresh", 1.2)
                ),
                # Deblending helps split overlapping sources in crowded fields.
                "DEBLEND_NTHRESH": str(
                    _wcs_cfg_int(wcs_cfg, "sextractor_deblend_nthresh", 32)
                ),
                "DEBLEND_MINCONT": str(
                    _wcs_cfg_float(wcs_cfg, "sextractor_deblend_mincont", 0.005)
                ),
                "FILTER": "Y",
                "FILTER_NAME": conv_filter_path,
                "CLEAN": "Y",
                "CLEAN_PARAM": "1",
                "PHOT_AUTOPARAMS": "2.5,3.5",
                "GAIN": gain_str,
                "PIXEL_SCALE": str(pixel_scale),
                "VERBOSE_TYPE": "NORMAL",
                "CATALOG_TYPE": "FITS_LDAC",
                "PARAMETERS_NAME": param_file,
            }
            with open(config_file, "w") as f:
                for k, v in final_config.items():
                    f.write(f"{k}\t{v}\n")

            # --- Prepare SExtractor command ---
            sextractor_cmd = None
            if use_sextractor:
                logger.debug(
                    "Using SExtractor with Gaussian convolution filter for source detection."
                )
                sextractor_cmd = str(sextractor_exe)
            else:
                logger.warning("SExtractor not found. Proceeding without.")

            # --- Scale bounds ---
            # A wrong pixel_scale can prevent convergence, so build two sets:
            #   - constrained: around the hint (if present)
            #   - wide: wcs.wide_scale_{low,high}_arcsec (default 0.1--5.0)
            ra, dec = self.default_input.get("target_ra"), self.default_input.get(
                "target_dec"
            )
            radius_deg = _wcs_cfg_float(wcs_cfg, "search_radius", 0.5)
            scale_low_wide = _wcs_cfg_float(
                wcs_cfg, "wide_scale_low_arcsec", 0.1
            )
            scale_high_wide = _wcs_cfg_float(
                wcs_cfg, "wide_scale_high_arcsec", 5.0
            )

            # Ordered --ra/--dec hint centers.  When the header already
            # carries a usable WCS its reference position usually sits closer
            # to the true field than the configured target, so it goes first
            # under prefer_header_position_hint; the second center gets one
            # extra hinted pass per downsample level when it lies well
            # outside the first search cone.
            prefer_header_hint = _wcs_cfg_bool(
                wcs_cfg, "prefer_header_position_hint", True
            )
            _pos_hints = _choose_position_hints(
                self.header, ra, dec, prefer_header_hint
            )
            if _pos_hints:
                logger.info(
                    "solve-field position hint(s): %s",
                    ", ".join(
                        "(%.5f, %.5f)" % (h[0], h[1]) for h in _pos_hints
                    ),
                )

            def _scale_args_for_hint(
                scale_low: float, scale_high: float, include_pos: bool, hint=None
            ) -> list:
                h_ra, h_dec = (None, None) if hint is None else hint
                return _build_scale_args(
                    scale_low,
                    scale_high,
                    include_pos=include_pos,
                    ra=h_ra,
                    dec=h_dec,
                    radius_deg=radius_deg,
                    background_std=background_std,
                )

            if pixel_scale and pixel_scale > 0:
                scale_low_hint = pixel_scale * 0.6
                scale_high_hint = pixel_scale * 1.5
            else:
                scale_low_hint, scale_high_hint = scale_low_wide, scale_high_wide

            # Scale-only args carry no position and are shared by every hint
            # center; position args are rebuilt per center inside _try_level.
            scale_only_args = _scale_args_for_hint(
                scale_low_hint, scale_high_hint, include_pos=False
            )
            scale_only_args_wide = _scale_args_for_hint(
                scale_low_wide, scale_high_wide, include_pos=False
            )
            scale_args = _scale_args_for_hint(
                scale_low_hint,
                scale_high_hint,
                include_pos=True,
                hint=_pos_hints[0] if _pos_hints else None,
            )

            # --- Downsample: lighten astrometry.net load for large images ---
            ny, nx = self.image.shape[0], self.image.shape[1]
            if nx * ny < 1500 * 1500:
                downsample = max(1, downsample - 1)
            downsample = max(1, int(downsample))

            if os.path.exists(astrometry_log_fpath):
                os.remove(astrometry_log_fpath)
            # Per-invocation temp names (see wcs_file above): avoid collisions
            # between concurrent solves sharing an image directory.
            scamp_temp = os.path.join(dirname, f"astrometry_{_tmp_tag}_scamp")
            corr_temp = os.path.join(dirname, f"astrometry_{_tmp_tag}.corr")
            rdl_temp = os.path.join(dirname, f"astrometry_{_tmp_tag}.rdls")
            match_temp = os.path.join(dirname, f"astrometry_{_tmp_tag}.match")
            solved_temp = os.path.join(dirname, f"astrometry_{_tmp_tag}.solved")
            xyls_temp = os.path.join(dirname, f"{_tmp_tag}.xyls")
            common_args = [
                str(solvefield_exe),
                "--no-remove-lines",
                "--overwrite",
                "--downsample",
                str(downsample),
                "--wcs",
                str(wcs_file),
                # --scamp output is never consumed by the pipeline and its
                # writer raises a harmless but alarming "Column FLUX not
                # found" fitstable error on SExtractor-derived .axy tables.
                "--corr",
                corr_temp,
                "--rdls",
                rdl_temp,
                "--match",
                match_temp,
                "--solved",
                solved_temp,
                "--no-plots",
                "--cpulimit",
                str(max(1, int(timeout_sec))),
                str(self.fpath),
            ]
            # Optional: request more detected objects for solve-field
            # indexing/matching. Larger values help sparse fields at modest
            # runtime cost.
            objs = wcs_cfg.get("objs", 1200)
            if objs is not None:
                try:
                    n_objs = int(objs)
                    if n_objs > 0:
                        common_args.insert(-1, "--objs")
                        common_args.insert(-1, str(n_objs))
                        logger.debug(
                            "solve-field object budget: --objs %d", n_objs
                        )
                except Exception:
                    logger.warning(
                        "Ignoring invalid wcs.objs=%r (expected positive integer).",
                        objs,
                    )
            # --crpix-center forces CRPIX to the image center. That is valid for
            # astrometry.net but can disagree with instrument/subarray headers where
            # CRPIX is elsewhere; users often get better agreement keeping raw WCS
            # when this flag was always on. Default: omit (False); set True for
            # legacy behavior matching older AutoPHoT.
            use_crpix_center = bool(wcs_cfg.get("solve_field_crpix_center", False))
            
            # Sigma level for sources used by solve-field.
            # Lower values detect more sources (useful for sparse fields).
            solve_nsigma_val = int(wcs_cfg.get("solve_field_nsigma", 5))
            # Proactive nsigma reduction for sparse fields: if we know the
            # source count is low (or exactly zero, meaning the detection
            # stage found nothing), lower nsigma to help solve-field detect
            # more sources.  This avoids the reactive nsigma=3 retry after
            # zero-SIP failure.
            _sparse_nsigma = _wcs_cfg_int(
                wcs_cfg, "solve_field_sparse_nsigma", 3
            )
            _reduced_nsigma = _solve_nsigma_for_count(
                n_detected_sources, solve_nsigma_val, _sparse_nsigma
            )
            if _reduced_nsigma != solve_nsigma_val:
                logger.info(
                    "Proactive nsigma reduction: %s detected sources -> "
                    "nsigma %d->%d (detect more sources for sparse field)",
                    n_detected_sources,
                    solve_nsigma_val,
                    _reduced_nsigma,
                )
                solve_nsigma_val = _reduced_nsigma
            solve_nsigma = str(solve_nsigma_val)
            
            # fits2fits rewrites the input image to sanitize it on every
            # invocation; skip it when the input is known-standard FITS.
            if _wcs_cfg_bool(wcs_cfg, "solve_field_no_fits2fits", False):
                common_args.insert(-1, "--no-fits2fits")
            common_args.insert(-1, "--nsigma")
            common_args.insert(-1, solve_nsigma)
            if use_crpix_center:
                common_args.insert(-1, "--crpix-center")
                logger.info(
                    "solve-field: using --crpix-center (reference pixel at image center)"
                )
            else:
                logger.debug(
                    "solve-field: omitting --crpix-center (solver reference pixel; often closer to instrument WCS - set wcs.solve_field_crpix_center: true for legacy)"
                )
            # Tweak order(s) for solve-field. Single int or list, e.g. YAML:
            #   solve_field_tweak_order: 5
            #   solve_field_tweak_orders: [5, 3, 2, 1, 0]
            #
            # With n_detected_sources known, cap the initial order to avoid
            # wasted solve-field calls. astrometry.net minimum correspondences:
            # order 4->15, 3->10, 2->6, 1->3. We add a ~1.5x margin since not
            # all detections match index stars; a post-solve SIP magnitude
            # check catches residual overfitting (wild SIP at corners with too
            # few constraint stars).
            _min_sources_for_order = {0: 0, 1: 5, 2: 9, 3: 15, 4: 23}
            tweak_orders = [0]
            try:
                to_list = wcs_cfg.get("solve_field_tweak_orders", None)
                to_single = wcs_cfg.get("solve_field_tweak_order", 3)
                if isinstance(to_list, (list, tuple)) and len(to_list) > 0:
                    parsed = [int(v) for v in to_list]
                    tweak_orders = [v for v in parsed if v >= 0]
                elif to_single is not None:
                    v = int(to_single)
                    if v >= 0:
                        tweak_orders = [v]
            except Exception:
                tweak_orders = [0]
            if len(tweak_orders) == 0:
                tweak_orders = [0]

            # Proactive cap: skip tweak orders astrometry.net cannot satisfy
            # with the detected source count. Avoids the reactive retry chain
            # (up to 6 invocations) on sparse fields.
            if n_detected_sources is not None and n_detected_sources > 0:
                _max_feasible = 0
                for _ord, _min_src in sorted(_min_sources_for_order.items(), reverse=True):
                    if n_detected_sources >= _min_src:
                        _max_feasible = _ord
                        break
                _capped = [o for o in tweak_orders if o <= _max_feasible]
                if _capped:
                    if _capped != tweak_orders:
                        logger.info(
                            "Proactive tweak order cap: %d detected sources -> "
                            "max feasible order %d -> sequence %s (was %s)",
                            n_detected_sources, _max_feasible, _capped, tweak_orders,
                        )
                    tweak_orders = _capped
                elif _max_feasible > 0:
                    logger.info(
                        "Proactive tweak order cap: %d detected sources -> "
                        "requested order(s) %s exceed max feasible %d -> using %d",
                        n_detected_sources, tweak_orders, _max_feasible, _max_feasible,
                    )
                    tweak_orders = [_max_feasible]
                else:
                    logger.info(
                        "Proactive tweak order cap: %d detected sources -> "
                        "using order 0 only (too few for SIP)",
                        n_detected_sources,
                    )
                    tweak_orders = [0]

            logger.debug("solve-field tweak order sequence: %s", tweak_orders)

            # (ra, dec) hint carried by the args that produced wcs_file, or
            # None for the position-free (blind) path.  The post-solve
            # position check is anchored to this, not to the configured
            # target, so a correct blind solve far from the target is not
            # wrongly rejected.
            _solved_pos_hint = None

            # --- Detected-source xylist for starved fields ---
            # solve-field accepts a FITS xylist of source positions instead
            # of an image, which skips its own extraction entirely.  On
            # fields where our detection stage starved, this is the only
            # path that gives the solver anything to work with.
            _solve_xy = _det_xy
            _min_xyls_src = _wcs_cfg_int(
                wcs_cfg, "solve_field_xylist_min_sources", 8
            )
            _xyls_xy = _solve_xy
            _xyls_flux = (
                _coerce_detected_flux(detected_xy, len(_xyls_xy))
                if _xyls_xy is not None
                else None
            )
            if _xyls_xy is not None and len(_xyls_xy) < _min_xyls_src:
                _xyls_xy = None
            _relaxed_det_empty = False
            if _xyls_xy is None and _wcs_cfg_bool(
                wcs_cfg, "solve_field_use_xylist", True
            ):
                _det_xy2, _det_flux2 = _detect_solve_sources(
                    self.image,
                    _wcs_cfg_float(
                        wcs_cfg, "solve_field_sparse_detect_sigma", 2.5
                    ),
                )
                if _det_xy2 is not None and len(_det_xy2) >= _min_xyls_src:
                    _xyls_xy = _det_xy2
                    _xyls_flux = _det_flux2
                _relaxed_det_empty = _det_xy2 is None or len(_det_xy2) == 0
            _xyls_path = None
            if _xyls_xy is not None:
                _xyls_path = _write_solve_xylist(
                    _xyls_xy, _xyls_flux, xyls_temp, width=nx, height=ny
                )
            # A zero-source catalog makes every --use-source-extractor
            # attempt a no-op, so the first hinted pass drops it when both
            # counters agree the field is empty.
            _skip_sex_starved = bool(
                n_detected_sources is not None
                and int(n_detected_sources) == 0
                and _relaxed_det_empty
            )

            # --- Step 1: Optional verify existing WCS ---
            # Reserved for a future fast-verify path (solve-field --verify
            # against the input WCS); callers currently always pass
            # skip_verify=True and redo_wcs forces it True anyway.
            if not skip_verify:
                logger.info("Attempting to verify existing WCS")
                # scale_args carries the primary hint when present.
                _solved_pos_hint = _pos_hints[0] if _pos_hints else None
                args_verify = common_args + scale_args
                if use_sextractor and sextractor_cmd:
                    args_verify += [
                        "--use-source-extractor",
                        "--source-extractor-path",
                        sextractor_cmd,
                        "--source-extractor-config",
                        str(config_file),
                        "--x-column",
                        "XWIN_IMAGE",
                        "--y-column",
                        "YWIN_IMAGE",
                        "--sort-column",
                        "MAG_AUTO",
                        "--sort-ascending",
                    ]
                if self._run_solve_field(
                    args_verify, wcs_file, timeout_sec, astrometry_log_fpath
                ):
                    logger.info("WCS verified successfully")
                else:
                    logger.warning("Could not verify WCS - proceeding to blind solve")

            def _with_solve_cpulimit(args_list, budget_sec):
                # common_args bakes in the hinted-attempt cpulimit; late
                # strategies swap in their own budget instead of rebuilding
                # the whole arg list.
                out = list(args_list)
                try:
                    _i = out.index("--cpulimit")
                    out[_i + 1] = str(max(1, int(budget_sec)))
                except ValueError:
                    out += ["--cpulimit", str(max(1, int(budget_sec)))]
                return out

            def _attempt_solve_with_args(
                scale_args_use: list,
                label: str,
                pos_hint=None,
                input_path: str = None,
                allow_sextractor: bool = True,
                timeout_override: float | None = None,
            ) -> None:
                nonlocal _solved_pos_hint
                _solved_pos_hint = pos_hint
                # Late strategies carry a bigger CPU budget; a distinct
                # name keeps timeout_sec a closure read - assigning it here
                # would make it function-local and break the hinted path.
                _attempt_timeout = (
                    float(timeout_override)
                    if timeout_override is not None
                    else timeout_sec
                )
                # An xylist input replaces the image path (the last common
                # arg); the solver then skips extraction entirely.
                _base_args = common_args
                if input_path is not None:
                    _base_args = _base_args[:-1] + [str(input_path)]
                # Starved fields make SExtractor produce a zero-source
                # catalog, so the caller can drop the --use-source-extractor
                # args entirely rather than spend a cpulimit on no-ops.
                _use_sex_here = bool(
                    allow_sextractor and use_sextractor and sextractor_cmd
                )
                # Step 2: Solve with (optional) SExtractor, tweak orders from config
                for tweak_order in tweak_orders:
                    if os.path.isfile(wcs_file):
                        break
                    logger.info(
                        "Attempting WCS solve (%s) with tweak order %s",
                        label,
                        tweak_order,
                    )
                    args = (
                        _base_args
                        + ["--tweak-order", str(tweak_order), "--no-verify"]
                        + scale_args_use
                    )
                    args = _with_solve_cpulimit(args, _attempt_timeout)
                    if _use_sex_here:
                        # XWIN_IMAGE/YWIN_IMAGE = weighted centroid, more accurate than X_IMAGE/Y_IMAGE
                        args += [
                            "--use-source-extractor",
                            "--source-extractor-path",
                            sextractor_cmd,
                            "--source-extractor-config",
                            str(config_file),
                            "--x-column",
                            "XWIN_IMAGE",
                            "--y-column",
                            "YWIN_IMAGE",
                            "--sort-column",
                            "MAG_AUTO",
                            "--sort-ascending",
                        ]
                    if self._run_solve_field(
                        args, wcs_file, _attempt_timeout, astrometry_log_fpath
                    ):
                        logger.log(
                            STATUS,
                            "WCS solved (%s) with tweak order %s", label, tweak_order,
                        )
                        # solve-field may return no usable SIP even when asked:
                        # too few matched stars for the requested order yields
                        # A_ORDER>0 with all-zero coefficients, or a plain TAN
                        # WCS with no SIP keywords. Retry with lower tweak
                        # orders (down to 0) for a stable distortion model.
                        if tweak_order > 0:
                            try:
                                with fits.open(wcs_file) as _wh:
                                    _whdr = _wh[0].header
                                if not _has_nonzero_sip(_whdr):
                                    logger.warning(
                                        "solve-field produced no usable SIP with "
                                        "tweak order %d (too few matched stars). "
                                        "Retrying with lower tweak orders.",
                                        tweak_order,
                                    )
                                    # Keep the working solve aside: when every
                                    # lower order fails the file can be put back
                                    # by rename instead of another solve-field
                                    # run that only reproduces what we had.
                                    _wcs_backup = wcs_file + ".orig"
                                    try:
                                        shutil.copy2(wcs_file, _wcs_backup)
                                    except OSError:
                                        _wcs_backup = None
                                    for _lower in range(tweak_order - 1, -1, -1):
                                        if os.path.isfile(wcs_file):
                                            os.remove(wcs_file)
                                        _lower_args = (
                                            _base_args
                                            + ["--tweak-order", str(_lower), "--no-verify"]
                                            + scale_args_use
                                        )
                                        _lower_args = _with_solve_cpulimit(
                                            _lower_args, _attempt_timeout
                                        )
                                        if _use_sex_here:
                                            _lower_args += [
                                                "--use-source-extractor",
                                                "--source-extractor-path",
                                                sextractor_cmd,
                                                "--source-extractor-config",
                                                str(config_file),
                                                "--x-column",
                                                "XWIN_IMAGE",
                                                "--y-column",
                                                "YWIN_IMAGE",
                                                "--sort-column",
                                                "MAG_AUTO",
                                                "--sort-ascending",
                                            ]
                                        logger.info(
                                            "Retrying solve with tweak order %d "
                                            "to get non-zero SIP",
                                            _lower,
                                        )
                                        if self._run_solve_field(
                                            _lower_args, wcs_file,
                                            _attempt_timeout, astrometry_log_fpath
                                        ):
                                            try:
                                                with fits.open(wcs_file) as _lh:
                                                    _lhdr = _lh[0].header
                                                if _has_nonzero_sip(_lhdr):
                                                    logger.info(
                                                        "WCS solved with non-zero SIP "
                                                        "at tweak order %d",
                                                        _lower,
                                                    )
                                                    break
                                                elif _lower == 0:
                                                    logger.info(
                                                        "WCS solved with linear TAN "
                                                        "at tweak order 0 (no SIP "
                                                        "could be fit)",
                                                    )
                                                    break
                                                else:
                                                    logger.info(
                                                        "Tweak order %d still produced "
                                                        "zero SIP",
                                                        _lower,
                                                    )
                                            except Exception as _sip_hdr_exc:
                                                logger.debug(
                                                    "Could not inspect WCS header "
                                                    "after tweak-order retry: %s",
                                                    _sip_hdr_exc,
                                                )
                                        else:
                                            logger.warning(
                                                "No solution with tweak order %d",
                                                _lower,
                                            )
                                    # No lower order produced non-zero SIP:
                                    # restore the original solve (order 0 = TAN)
                                    if not os.path.isfile(wcs_file):
                                        if _wcs_backup and os.path.isfile(
                                            _wcs_backup
                                        ):
                                            logger.warning(
                                                "Could not solve with any lower tweak "
                                                "order; restoring the order-%d solve.",
                                                tweak_order,
                                            )
                                            shutil.move(_wcs_backup, wcs_file)
                                        else:
                                            logger.warning(
                                                "Could not solve with any lower tweak "
                                                "order; retrying original order %d.",
                                                tweak_order,
                                            )
                                            self._run_solve_field(
                                                args, wcs_file, _attempt_timeout,
                                                astrometry_log_fpath
                                            )
                                    if _wcs_backup and os.path.isfile(_wcs_backup):
                                        try:
                                            os.remove(_wcs_backup)
                                        except OSError:
                                            pass
                                    # Still zero SIP: retry once with lower
                                    # nsigma to detect more sources
                                    if os.path.isfile(wcs_file):
                                        try:
                                            with fits.open(wcs_file) as _fh:
                                                _fhdr = _fh[0].header
                                            if _fhdr.get("A_ORDER", 0) > 0 and not _has_nonzero_sip(_fhdr):
                                                _orig_nsigma = solve_nsigma
                                                _lower_nsigma = "3"
                                                logger.info(
                                                    "Retrying solve with nsigma=%s "
                                                    "(was %s) to detect more sources "
                                                    "for SIP fitting",
                                                    _lower_nsigma, _orig_nsigma,
                                                )
                                                if os.path.isfile(wcs_file):
                                                    os.remove(wcs_file)
                                                _nsigma_args = _set_flag_value(
                                                    args, "--nsigma", _lower_nsigma
                                                )
                                                if self._run_solve_field(
                                                    _nsigma_args, wcs_file,
                                                    _attempt_timeout, astrometry_log_fpath
                                                ):
                                                    try:
                                                        with fits.open(wcs_file) as _nh:
                                                            _nhdr = _nh[0].header
                                                        if _has_nonzero_sip(_nhdr):
                                                            logger.info(
                                                                "WCS solved with non-zero SIP "
                                                                "after lowering nsigma to %s",
                                                                _lower_nsigma,
                                                            )
                                                        else:
                                                            logger.warning(
                                                                "Still zero SIP with nsigma=%s; "
                                                                "field may be too sparse for "
                                                                "distortion fitting",
                                                                _lower_nsigma,
                                                            )
                                                    except Exception as _ns_hdr_exc:
                                                        logger.debug(
                                                            "Could not inspect WCS header "
                                                            "after nsigma retry: %s",
                                                            _ns_hdr_exc,
                                                        )
                                                else:
                                                    logger.warning(
                                                        "No solution with nsigma=%s",
                                                        _lower_nsigma,
                                                    )
                                        except Exception as _ns_solve_exc:
                                            logger.debug(
                                                "nsigma retry block failed: %s",
                                                _ns_solve_exc,
                                            )
                            except Exception as _sip_err:
                                logger.debug(
                                    "SIP zero-check failed (non-fatal): %s", _sip_err
                                )

                        # BUG 103c: SIP magnitude validation. Even with non-zero
                        # SIP, under-determined fits produce wild distortion at
                        # image corners where there are no constraint stars.
                        # Measure the max SIP correction at the 4 corners; if it
                        # exceeds 5% of the image size, retry with a lower tweak
                        # order.
                        if tweak_order > 0 and os.path.isfile(wcs_file):
                            try:
                                from astropy.wcs import WCS as _WCS
                                with fits.open(wcs_file) as _mh:
                                    _mhdr = _mh[0].header
                                if _mhdr.get("A_ORDER", 0) > 0 and _has_nonzero_sip(_mhdr):
                                    _mw = _WCS(_mhdr)
                                    _ny = _mhdr.get("NAXIS2", 2048)
                                    _nx = _mhdr.get("NAXIS1", 2048)
                                    _max_sip_px = 0.0
                                    # True pixel scale = sqrt(|det(CD)|) for a
                                    # rotated CD matrix; CD1_1 alone
                                    # under-estimates it when the matrix has a
                                    # rotation cross-term, which inflates the
                                    # measured SIP correction in pixels and
                                    # falsely rejects a valid solution.  Fall
                                    # back to CD1_1/CDELT only when the
                                    # determinant is unusable.
                                    _cd11 = _mhdr.get("CD1_1")
                                    _cd12 = _mhdr.get("CD1_2", 0.0)
                                    _cd21 = _mhdr.get("CD2_1", 0.0)
                                    _cd22 = _mhdr.get("CD2_2")
                                    _det_cd = None
                                    if (
                                        _cd11 is not None
                                        and _cd22 is not None
                                    ):
                                        _det_cd = abs(
                                            float(_cd11) * float(_cd22)
                                            - float(_cd12) * float(_cd21)
                                        )
                                    if (
                                        _det_cd is not None
                                        and np.isfinite(_det_cd)
                                        and _det_cd > 0
                                    ):
                                        _pixscale = float(np.sqrt(_det_cd)) * 3600
                                    else:
                                        _pixscale = abs(
                                            float(
                                                _mhdr.get(
                                                    "CD1_1",
                                                    _mhdr.get("CDELT1", 1e-4),
                                                )
                                            )
                                        ) * 3600
                                    _sip_thresh_px = 0.05 * max(_nx, _ny)
                                    for _cx, _cy in [(0,0), (_nx-1,0), (0,_ny-1), (_nx-1,_ny-1)]:
                                        _ra_s, _dec_s = _mw.all_pix2world(_cx, _cy, 0)
                                        _ra_t, _dec_t = _mw.wcs_pix2world(_cx, _cy, 0)
                                        # Small-angle separation: wrap dRA
                                        # and project by cos(dec), else a
                                        # high-dec or RA=0 field reports a
                                        # spuriously large correction.
                                        _dra = (
                                            _ra_s - _ra_t + 180.0
                                        ) % 360.0 - 180.0
                                        _ddec = _dec_s - _dec_t
                                        _cosd = np.cos(
                                            np.deg2rad(
                                                0.5 * (_dec_s + _dec_t)
                                            )
                                        )
                                        _delta = np.sqrt(
                                            (_dra * _cosd) ** 2 + _ddec**2
                                        ) * 3600
                                        _delta_px = _delta / max(_pixscale, 0.01)
                                        _max_sip_px = max(_max_sip_px, _delta_px)
                                    if _max_sip_px > _sip_thresh_px:
                                        logger.warning(
                                            "SIP overfitting detected: max corner "
                                            "correction = %.1f px (> %.1f px = 5%% of "
                                            "image) at tweak order %d. Retrying with "
                                            "lower orders.",
                                            _max_sip_px, _sip_thresh_px, tweak_order,
                                        )
                                        # Same backup-and-rename restore as the
                                        # zero-SIP descent above: re-solving the
                                        # original order just to get the file
                                        # back burns a full cpulimit.
                                        _wcs_backup = wcs_file + ".orig"
                                        try:
                                            shutil.copy2(wcs_file, _wcs_backup)
                                        except OSError:
                                            _wcs_backup = None
                                        for _lower in range(tweak_order - 1, -1, -1):
                                            if os.path.isfile(wcs_file):
                                                os.remove(wcs_file)
                                            _lower_args = (
                                                _base_args
                                                + ["--tweak-order", str(_lower), "--no-verify"]
                                                + scale_args_use
                                            )
                                            _lower_args = _with_solve_cpulimit(
                                                _lower_args, _attempt_timeout
                                            )
                                            if _use_sex_here:
                                                _lower_args += [
                                                    "--use-source-extractor",
                                                    "--source-extractor-path",
                                                    sextractor_cmd,
                                                    "--source-extractor-config",
                                                    str(config_file),
                                                    "--x-column", "XWIN_IMAGE",
                                                    "--y-column", "YWIN_IMAGE",
                                                    "--sort-column", "MAG_AUTO",
                                                    "--sort-ascending",
                                                ]
                                            logger.info(
                                                "Retrying solve with tweak order %d "
                                                "(SIP overfitting check)",
                                                _lower,
                                            )
                                            if self._run_solve_field(
                                                _lower_args, wcs_file,
                                                _attempt_timeout, astrometry_log_fpath
                                            ):
                                                try:
                                                    with fits.open(wcs_file) as _lh:
                                                        _lhdr = _lh[0].header
                                                    if _lhdr.get("A_ORDER", 0) > 0 and _has_nonzero_sip(_lhdr):
                                                        _lw = _WCS(_lhdr)
                                                        _lmax = 0.0
                                                        for _cx, _cy in [(0,0), (_nx-1,0), (0,_ny-1), (_nx-1,_ny-1)]:
                                                            _ra_s, _dec_s = _lw.all_pix2world(_cx, _cy, 0)
                                                            _ra_t, _dec_t = _lw.wcs_pix2world(_cx, _cy, 0)
                                                            _dra = (
                                                                _ra_s - _ra_t
                                                                + 180.0
                                                            ) % 360.0 - 180.0
                                                            _ddec = _dec_s - _dec_t
                                                            _cosd = np.cos(
                                                                np.deg2rad(
                                                                    0.5
                                                                    * (_dec_s + _dec_t)
                                                                )
                                                            )
                                                            _delta = np.sqrt(
                                                                (_dra * _cosd) ** 2
                                                                + _ddec**2
                                                            ) * 3600
                                                            _delta_px = _delta / max(
                                                                _pixscale, 0.01
                                                            )
                                                            _lmax = max(_lmax, _delta_px)
                                                        if _lmax <= _sip_thresh_px:
                                                            logger.info(
                                                                "SIP overfitting resolved at "
                                                                "tweak order %d (max corner "
                                                                "correction = %.1f px)",
                                                                _lower, _lmax,
                                                            )
                                                            break
                                                        else:
                                                            logger.info(
                                                                "Tweak order %d still has "
                                                                "large SIP (%.1f px)",
                                                                _lower, _lmax,
                                                            )
                                                    else:
                                                        # Zero SIP or order 0 - acceptable
                                                        logger.info(
                                                            "Tweak order %d produced "
                                                            "zero/none SIP - acceptable",
                                                            _lower,
                                                        )
                                                        break
                                                except Exception as _sip_hdr_exc:
                                                    logger.debug(
                                                        "Could not inspect WCS header "
                                                        "after SIP-overfit retry: %s",
                                                        _sip_hdr_exc,
                                                    )
                                            else:
                                                logger.warning(
                                                    "No solution with tweak order %d",
                                                    _lower,
                                                )
                                        # If all lower orders failed, restore original
                                        if not os.path.isfile(wcs_file):
                                            if _wcs_backup and os.path.isfile(
                                                _wcs_backup
                                            ):
                                                logger.warning(
                                                    "Could not solve with lower tweak "
                                                    "orders after SIP overfitting; "
                                                    "restoring the order-%d solve.",
                                                    tweak_order,
                                                )
                                                shutil.move(_wcs_backup, wcs_file)
                                            else:
                                                logger.warning(
                                                    "Could not solve with lower tweak "
                                                    "orders after SIP overfitting; "
                                                    "retrying original order %d",
                                                    tweak_order,
                                                )
                                                self._run_solve_field(
                                                    args, wcs_file, _attempt_timeout,
                                                    astrometry_log_fpath
                                                )
                                        if _wcs_backup and os.path.isfile(
                                            _wcs_backup
                                        ):
                                            try:
                                                os.remove(_wcs_backup)
                                            except OSError:
                                                pass
                                    else:
                                        logger.info(
                                            "SIP magnitude OK: max corner correction "
                                            "= %.1f px (threshold %.1f px) at tweak "
                                            "order %d",
                                            _max_sip_px, _sip_thresh_px, tweak_order,
                                        )
                            except Exception as _sip_mag_err:
                                logger.debug(
                                    "SIP magnitude check failed (non-fatal): %s",
                                    _sip_mag_err,
                                )

                        break
                    logger.warning(
                        "No solution (%s) with tweak order %s", label, tweak_order
                    )

            # --- Progressive downsample retry: rescue large images that time
            # out at the configured downsample by retrying with coarser
            # sampling (each level solves a smaller image).  Levels are
            # bounded below so the smaller axis never drops under
            # solve_field_downsample_min_pixels; small images also get finer
            # levels prepended since they solve quickly anyway.
            _downsample_levels = _downsample_levels_for(
                nx,
                ny,
                downsample,
                ds_max=_wcs_cfg_int(
                    wcs_cfg, "solve_field_downsample_max", 8
                ),
                min_pixels=_wcs_cfg_int(
                    wcs_cfg, "solve_field_downsample_min_pixels", 256
                ),
                small_image=_wcs_cfg_int(
                    wcs_cfg, "solve_field_fine_downsample_small_image", 1024
                ),
            )
            # Split common_args around its --downsample value so we can rebuild
            # it with a different downsample level on each retry.
            _ds_marker = common_args.index("--downsample")
            _common_args_prefix = common_args[: _ds_marker + 1]
            _common_args_suffix = common_args[_ds_marker + 2 :]

            _solved = False

            def _try_level(_level: int, starved: bool = False) -> bool:
                """Run the full solve-field attempt sequence at one downsample.

                Rebuilds ``common_args`` with the requested --downsample so the
                verify / blind / no-SExtractor retries all run at that level.
                ``starved`` marks a confirmed empty field: SExtractor gets a
                zero-source catalog either way, so the hinted passes drop the
                extraction args instead of burning a cpulimit on no-ops.
                """
                nonlocal common_args, _solved_pos_hint
                _saved_common_args = common_args
                common_args = (
                    _common_args_prefix + [str(_level)] + _common_args_suffix
                )
                _sa = scale_args
                _soa = scale_only_args
                try:
                    # Step 1: hinted solve (optional SExtractor, tweak
                    # orders).  One pass per position center; a second
                    # center gets a single extra pass and only when it
                    # lies outside the first search cone.
                    _hi = 0
                    while True:
                        _hint = (
                            _pos_hints[_hi] if _hi < len(_pos_hints) else None
                        )
                        if _hi > 0:
                            _sa = _scale_args_for_hint(
                                scale_low_hint,
                                scale_high_hint,
                                include_pos=True,
                                hint=_hint,
                            )
                            _soa = scale_only_args
                        _label = (
                            "with pixel-scale hint"
                            if (pixel_scale and pixel_scale > 0)
                            else "wide scale"
                        )
                        if _hi > 0:
                            _label += " (alternate position)"
                        _attempt_solve_with_args(
                            _sa,
                            label=_label,
                            pos_hint=_hint,
                            allow_sextractor=not starved,
                            # No pixel scale means the first pass is already
                            # the wide search, so it gets the late budget.
                            timeout_override=None
                            if (pixel_scale and pixel_scale > 0)
                            else late_timeout_sec,
                        )
                        if os.path.isfile(wcs_file):
                            return True
                        # Step 1b: drop the pixel-scale hint and retry wide.
                        if not os.path.isfile(wcs_file) and (
                            pixel_scale and pixel_scale > 0
                        ):
                            logger.warning(
                                "WCS not solved with pixel-scale hint "
                                "(%.4g arcsec/pix). Dropping pixel scale "
                                "constraint and retrying with wide scale "
                                "bounds (%.4g--%.4g arcsec/pix).",
                                float(pixel_scale),
                                scale_low_wide,
                                scale_high_wide,
                            )
                            _sa = _scale_args_for_hint(
                                scale_low_wide,
                                scale_high_wide,
                                include_pos=True,
                                hint=_hint,
                            )
                            _soa = scale_only_args_wide
                            _attempt_solve_with_args(
                                _sa,
                                label="wide scale (dropped pixel-scale hint)",
                                pos_hint=_hint,
                                allow_sextractor=not starved,
                                timeout_override=late_timeout_sec,
                            )
                            if os.path.isfile(wcs_file):
                                return True
                        # Advance to the next center only when it sits
                        # outside 0.25*radius_deg of the one just tried.
                        if _hint is None or _hi + 1 >= len(_pos_hints):
                            break
                        _sep_to_next = _sky_sep_deg(
                            _hint[0],
                            _hint[1],
                            _pos_hints[_hi + 1][0],
                            _pos_hints[_hi + 1][1],
                        )
                        if _sep_to_next <= 0.25 * radius_deg:
                            break
                        logger.info(
                            "Retrying hinted solve at second position "
                            "center (%.4f deg from the first hint).",
                            _sep_to_next,
                        )
                        _hi += 1
                    # Step 3: retry without SExtractor.  A single tweak
                    # order is enough: match acceptance is decided before
                    # the SIP refinement, so a field with no verified match
                    # runs the same index sweep at every order and the
                    # ladder only multiplies the cpulimit burn.
                    if not os.path.isfile(wcs_file):
                        logger.warning("Retrying without SExtractor")
                        tweak_order = tweak_orders[0]
                        logger.info(
                            "Attempting WCS solve without SExtractor "
                            "(tweak order %s)", tweak_order
                        )
                        args_no_sex = (
                            common_args
                            + [
                                "--tweak-order",
                                str(tweak_order),
                                "--no-verify",
                            ]
                            + _sa
                        )
                        args_no_sex = _with_solve_cpulimit(
                            args_no_sex, late_timeout_sec
                        )
                        if self._run_solve_field(
                            args_no_sex,
                            wcs_file,
                            late_timeout_sec,
                            astrometry_log_fpath,
                        ):
                            logger.log(
                                STATUS,
                                "WCS solved without SExtractor "
                                "(tweak order %s)", tweak_order,
                            )
                            return True
                        logger.warning(
                            "No solution without SExtractor "
                            "(tweak order %s)", tweak_order
                        )
                    # Step 4: blind solve (no position constraint).  Only
                    # run when a position cone was tried above; with no
                    # hint the first pass already used scale-only args, so
                    # a blind retry would be an exact duplicate.
                    if not os.path.isfile(wcs_file) and _pos_hints:
                        # Winning args carry no --ra/--dec, so the solved
                        # CRVAL must not be checked against any hint.
                        _solved_pos_hint = None
                        logger.warning(
                            "Retrying with blind solve (no position constraint)."
                        )
                        # Same single-order rationale as the no-SExtractor
                        # pass: a failed blind match is identical at every
                        # tweak order, so only the configured first order
                        # is worth the blind cpulimit.
                        tweak_order = tweak_orders[0]
                        logger.info(
                            "Blind solve with tweak order %s", tweak_order
                        )
                        args_blind = (
                            common_args
                            + [
                                "--tweak-order",
                                str(tweak_order),
                                "--no-verify",
                            ]
                            + _soa
                        )
                        args_blind = _with_solve_cpulimit(
                            args_blind, blind_timeout_sec
                        )
                        # A confirmed-starved field only produces a
                        # zero-source SExtractor catalog; skip the wasted
                        # extraction here as in the hinted passes.
                        if use_sextractor and sextractor_cmd and not starved:
                            args_blind += [
                                "--use-source-extractor",
                                "--source-extractor-path",
                                sextractor_cmd,
                                "--source-extractor-config",
                                str(config_file),
                                "--x-column",
                                "XWIN_IMAGE",
                                "--y-column",
                                "YWIN_IMAGE",
                                "--sort-column",
                                "MAG_AUTO",
                                "--sort-ascending",
                            ]
                        if self._run_solve_field(
                            args_blind,
                            wcs_file,
                            blind_timeout_sec,
                            astrometry_log_fpath,
                        ):
                            logger.log(
                                STATUS,
                                "WCS solved with blind solve "
                                "(tweak order %s)", tweak_order,
                            )
                            return True
                        logger.warning(
                            "No solution with blind solve "
                            "(tweak order %s)", tweak_order
                        )
                    return False
                finally:
                    common_args = _saved_common_args

            def _try_xyls_hints(xyls_input: str) -> bool:
                """Hinted solve attempts fed by a pre-detected xylist.

                Mirrors the Step-1 hinted passes of _try_level but feeds
                solve-field the xylist instead of the image, so the solver
                skips its own source extraction (and the SExtractor args are
                omitted).  --downsample does not apply to a position list,
                so this runs once on the base args rather than per level.
                """
                _hi = 0
                while True:
                    _hint = (
                        _pos_hints[_hi] if _hi < len(_pos_hints) else None
                    )
                    _sa = scale_args
                    if _hi > 0:
                        _sa = _scale_args_for_hint(
                            scale_low_hint,
                            scale_high_hint,
                            include_pos=True,
                            hint=_hint,
                        )
                    _label = (
                        "xylist with pixel-scale hint"
                        if (pixel_scale and pixel_scale > 0)
                        else "xylist wide scale"
                    )
                    if _hi > 0:
                        _label += " (alternate position)"
                    _attempt_solve_with_args(
                        _sa,
                        label=_label,
                        pos_hint=_hint,
                        input_path=xyls_input,
                        allow_sextractor=False,
                        timeout_override=None
                        if (pixel_scale and pixel_scale > 0)
                        else late_timeout_sec,
                    )
                    if os.path.isfile(wcs_file):
                        return True
                    # Same dropped-scale retry as the image path.
                    if not os.path.isfile(wcs_file) and (
                        pixel_scale and pixel_scale > 0
                    ):
                        _sa = _scale_args_for_hint(
                            scale_low_wide,
                            scale_high_wide,
                            include_pos=True,
                            hint=_hint,
                        )
                        _attempt_solve_with_args(
                            _sa,
                            label="xylist wide scale "
                            "(dropped pixel-scale hint)",
                            pos_hint=_hint,
                            input_path=xyls_input,
                            allow_sextractor=False,
                            timeout_override=late_timeout_sec,
                        )
                        if os.path.isfile(wcs_file):
                            return True
                    # Advance to the next center only when it sits outside
                    # 0.25*radius_deg of the one just tried.
                    if _hint is None or _hi + 1 >= len(_pos_hints):
                        break
                    _sep_to_next = _sky_sep_deg(
                        _hint[0],
                        _hint[1],
                        _pos_hints[_hi + 1][0],
                        _pos_hints[_hi + 1][1],
                    )
                    if _sep_to_next <= 0.25 * radius_deg:
                        break
                    logger.info(
                        "Retrying hinted xylist solve at second position "
                        "center (%.4f deg from the first hint).",
                        _sep_to_next,
                    )
                    _hi += 1
                return False

            if _xyls_path is not None:
                logger.info(
                    "solve-field: trying detected-source xylist first "
                    "(%d sources; solver-side extraction skipped)",
                    len(_xyls_xy),
                )
                if _try_xyls_hints(_xyls_path):
                    _solved = True

            for _li, _level in enumerate(_downsample_levels):
                if _solved:
                    break
                if _level != downsample:
                    logger.warning(
                        "WCS not solved at downsample %s; retrying with "
                        "downsample %s.",
                        downsample if downsample else _level // 2,
                        _level,
                    )
                if _try_level(
                    _level, starved=(_li == 0 and _skip_sex_starved)
                ):
                    _solved = True
                    break

            # The attempt cascade is over; release the shared budget so a
            # later reuse of this instance starts fresh.
            self._solve_deadline = None
            self._solve_budget_warned = False

            # --- If WCS still not solved, try SCAMP as fallback (e.g. crowded or IR fields) ---
            if not os.path.isfile(wcs_file):
                if use_tpv_flow:
                    logger.warning(
                        "Astrometry.net did not solve WCS; trying SCAMP as fallback."
                    )
                    try:
                        scamp_header = self._plate_solve_scamp(
                            wcs_cfg=wcs_cfg, cpulimit=cpulimit
                        )
                        if isinstance(scamp_header, fits.Header):
                            logger.info("SCAMP fallback succeeded.")
                            self.solved_new_wcs = True
                            return scamp_header
                    except Exception as exc:
                        log_warning_from_exception(logger, "SCAMP fallback failed", exc)
                else:
                    logger.warning(
                        "Astrometry.net did not solve WCS; projection_type=SIP disables SCAMP fallback."
                    )

            # --- If still not solved, return NaN ---
            if not os.path.isfile(wcs_file):
                # Final fallback: if the input header already contains a usable WCS,
                # keep and return it when solve-field/scamp solving fails.
                try:
                    _ = get_wcs(self.header)
                    logger.warning(
                        "Could not solve a new WCS; falling back to original header WCS."
                    )
                    for f in [scamp_temp, xyls_temp, os.path.join(dirname, "none")]:
                        if os.path.isfile(f):
                            os.remove(f)
                    return self.header
                except Exception:
                    pass
                for f in [scamp_temp, xyls_temp, os.path.join(dirname, "none")]:
                    if os.path.isfile(f):
                        os.remove(f)
                logger.warning("\tCould not solve WCS - returning NaN")
                return np.nan

            # --- Solved-WCS sanity validation: reject catastrophic mis-solves ---
            # A solve-field exit code of 0 is not sufficient: the solver can
            # converge to the wrong scale or the wrong sky position.  Validate
            # the solved WCS against the expected pixel scale and, when a
            # position constraint was used, the requested sky position before
            # trusting it.
            _solved_scale_ok = True
            _solved_pos_ok = True
            try:
                with fits.open(wcs_file) as _vh:
                    _solved_hdr = _vh[0].header
                _solved_wcs = get_wcs(_solved_hdr)
                if _solved_wcs is None:
                    # An unparseable solve would be adopted, then rejected at
                    # merge anyway; fail it here so the rejection is explicit.
                    _solved_scale_ok = False
                    logger.warning(
                        "Solved WCS file has no usable WCS keywords; "
                        "rejecting the solve."
                    )
                else:
                    _scale = pixel_area_scale_arcsec(
                        _solved_wcs,
                        (
                            int(_solved_hdr.get("NAXIS2", 0) or 0),
                            int(_solved_hdr.get("NAXIS1", 0) or 0),
                        ),
                    )
                    _scale_tol = float(
                        wcs_cfg.get("solved_scale_tolerance", 2.0) or 2.0
                    )
                    _expected_scale = float(pixel_scale) if pixel_scale else 0.0
                    if (
                        _scale_tol > 0
                        and np.isfinite(_scale)
                        and _scale > 0
                        and _expected_scale
                        and _expected_scale > 0
                    ):
                        _ratio = _scale / _expected_scale
                        if not (1.0 / _scale_tol <= _ratio <= _scale_tol):
                            _solved_scale_ok = False
                            logger.warning(
                                "Solved WCS pixel scale (%.4g arcsec/pix) "
                                "differs from expected (%.4g arcsec/pix) by "
                                "ratio %.2f (tolerance %.1f); rejecting solve.",
                                _scale,
                                _expected_scale,
                                _ratio,
                                _scale_tol,
                            )
                    # Position check: only meaningful when the winning args
                    # carried a position cone (--ra/--dec/--radius); blind
                    # solves have no constraint to validate against.  The
                    # check is anchored to the hint that was actually used,
                    # which may be the header reference position rather
                    # than the configured target.
                    _pos_tol = wcs_cfg.get("solved_position_tolerance_arcsec", None)
                    if _pos_tol is None:
                        _pos_tol = (
                            float(radius_deg) * 3600.0 if radius_deg else 0.0
                        )
                    _pos_tol = float(_pos_tol or 0.0)
                    if _pos_tol > 0 and _solved_pos_hint is not None:
                        _hint_ra, _hint_dec = (
                            float(_solved_pos_hint[0]),
                            float(_solved_pos_hint[1]),
                        )
                        _solved_crval1 = float(
                            _solved_hdr.get("CRVAL1", np.nan) or np.nan
                        )
                        _solved_crval2 = float(
                            _solved_hdr.get("CRVAL2", np.nan) or np.nan
                        )
                        if np.isfinite(_solved_crval1) and np.isfinite(
                            _solved_crval2
                        ):
                            _dra = (
                                (_solved_crval1 - _hint_ra + 180.0) % 360.0
                            ) - 180.0
                            _ddec = _solved_crval2 - _hint_dec
                            _pos_off = np.hypot(
                                _dra * np.cos(np.deg2rad(_hint_dec)), _ddec
                            ) * 3600.0
                            if _pos_off > _pos_tol:
                                _solved_pos_ok = False
                                logger.warning(
                                    "Solved WCS position is %.1f arcsec from "
                                    "the position hint used (ra=%.4f dec=%.4f; "
                                    "tolerance %.1f arcsec); rejecting solve.",
                                    _pos_off,
                                    _hint_ra,
                                    _hint_dec,
                                    _pos_tol,
                                )
            except Exception as _val_exc:
                log_warning_from_exception(
                    logger, "Solved-WCS sanity validation skipped", _val_exc
                )
            if not (_solved_scale_ok and _solved_pos_ok):
                logger.warning(
                    "Solved WCS failed sanity validation; treating solve as failed."
                )
                try:
                    _ = get_wcs(self.header)
                    logger.warning(
                        "Solved WCS rejected; falling back to original header WCS."
                    )
                    for f in [
                        scamp_temp,
                        xyls_temp,
                        wcs_file,
                        os.path.join(dirname, "none"),
                    ]:
                        if os.path.isfile(f):
                            os.remove(f)
                    return self.header
                except Exception:
                    pass
                for f in [
                    scamp_temp,
                    xyls_temp,
                    wcs_file,
                    os.path.join(dirname, "none"),
                ]:
                    if os.path.isfile(f):
                        os.remove(f)
                return np.nan

            # --- Clean up temporary files ---
            fit_sip_degree = wcs_cfg.get("fit_wcs_from_points_sip_degree")
            try:
                fit_sip_degree = int(fit_sip_degree) if fit_sip_degree else None
            except Exception:
                fit_sip_degree = None
            keep_match_and_axy = fit_sip_degree is not None and fit_sip_degree > 0
            # Keep solve-field match artifacts when we want input-vs-solved WCS
            # diagnostics, even if fit_wcs_from_points is disabled.
            compare_input_vs_solved = bool(
                wcs_cfg.get("compare_input_vs_solved_wcs", True)
            )
            keep_match_for_compare = bool(compare_input_vs_solved)
            keep_match_and_axy = keep_match_and_axy or keep_match_for_compare

            patterns_to_remove = [".rdls", ".solved", ".xyls", ".orig"]
            if not keep_match_and_axy:
                patterns_to_remove += [".axy", ".match", ".corr"]
            for pattern in patterns_to_remove:
                for f in _solve_artifact_glob(dirname, _tmp_tag, base, pattern):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            for f in [scamp_temp, xyls_temp, os.path.join(dirname, "none")]:
                if os.path.isfile(f):
                    os.remove(f)

            # --- Update FITS header with WCS from the .wcs.fits file ---
            # Snapshot the input header so a failed or degraded merge can
            # restore the original WCS instead of returning NaN.
            input_header_backup = self.header.copy()
            _input_header_wcs_valid = get_wcs(input_header_backup) is not None

            def _restore_input_wcs_inplace() -> None:
                # Restore WCS keywords on the SAME header object: callers
                # hold a reference to self.header, so rebinding it would
                # leave them pointing at a WCS-stripped header.
                remove_wcs_from_header(self.header)
                copy_wcs_from_header(input_header_backup, self.header)

            try:
                with fits.open(wcs_file) as wcs_hdul:
                    wcs_header = wcs_hdul[0].header
                wcs_header = _normalize_projection_codes(wcs_header, inplace=False)
                _log_header_distortion_coefficients(
                    wcs_header, prefix="astrometry.net solved WCS"
                )
                if wcs_header.get("A_ORDER", 0) > 0 and not _has_nonzero_sip(wcs_header):
                    logger.warning(
                        "astrometry.net solved WCS has A_ORDER=%d but all SIP\n"
                        "    coefficients are zero (too few matched stars for\n"
                        "    the requested tweak order). WCS is effectively\n"
                        "    TAN (linear). Consider lowering\n"
                        "    solve_field_tweak_order or increasing source\n"
                        "    detection sensitivity.",
                        int(wcs_header.get("A_ORDER", 0)),
                    )
                force_preserve_input_distortion = False
                keep_input_wcs_full = False
                matched_points_for_refine = None
                try:
                    input_wcs_for_compare = get_wcs(self.header)
                    solved_wcs_for_compare = get_wcs(wcs_header)
                    nx = int(self.image.shape[1])
                    ny = int(self.image.shape[0])
                    max_pts_diag = int(
                        wcs_cfg.get("fit_wcs_from_points_max_points", 1000)
                    )
                    diag_points = _extract_corr_points_from_solve_field(
                        corr_temp, max_points=max_pts_diag
                    )
                    if diag_points is None:
                        diag_points = _extract_match_points_from_solve_field(
                            match_temp,
                            solved_wcs_for_compare,
                            nx,
                            ny,
                            max_points=max_pts_diag,
                            max_lines=int(
                                wcs_cfg.get("fit_wcs_from_points_max_match_lines", 20000)
                            ),
                            max_sep_arcsec=float(
                                wcs_cfg.get("fit_wcs_from_points_max_sep_arcsec", 3.0)
                            ),
                        )
                    keep_if_not_better = bool(
                        wcs_cfg.get(
                            "keep_input_wcs_if_solved_not_better", True
                        )
                    )
                    clear_gain = False
                    if diag_points is not None:
                        x_d, y_d, ra_d, dec_d = diag_points
                        matched_points_for_refine = (
                            np.asarray(x_d, dtype=float),
                            np.asarray(y_d, dtype=float),
                            np.asarray(ra_d, dtype=float),
                            np.asarray(dec_d, dtype=float),
                        )
                        if input_wcs_for_compare is not None:
                            med_in, p95_in, sh_in = _best_wcs_match_separation_stats_arcsec(
                                input_wcs_for_compare, x_d, y_d, ra_d, dec_d
                            )
                        else:
                            logger.debug("input_wcs_for_compare is None, skipping comparison")
                            med_in, p95_in, sh_in = np.nan, np.nan, 0.0
                        if solved_wcs_for_compare is not None:
                            med_sv, p95_sv, sh_sv = _best_wcs_match_separation_stats_arcsec(
                                solved_wcs_for_compare, x_d, y_d, ra_d, dec_d
                            )
                        else:
                            logger.debug("solved_wcs_for_compare is None, skipping comparison")
                            med_sv, p95_sv, sh_sv = np.nan, np.nan, 0.0
                        logger.log(
                            STATUS,
                            "WCS compare:\tn=%d | input med/p95=%.3f/%.3f\" (shift=%+.1f px) | solved med/p95=%.3f/%.3f\" (shift=%+.1f px)",
                            int(len(x_d)),
                            float(med_in),
                            float(p95_in),
                            float(sh_in),
                            float(med_sv),
                            float(p95_sv),
                            float(sh_sv),
                        )
                        # If solve-field switches distortion model family
                        # (e.g. TPV -> TAN-SIP) without clear improvement,
                        # preserve the input distortion model and update
                        # linear terms only.
                        ctype1_in_u = str(self.header.get("CTYPE1", "")).upper()
                        ctype2_in_u = str(self.header.get("CTYPE2", "")).upper()
                        ctype1_sv_u = str(wcs_header.get("CTYPE1", "")).upper()
                        ctype2_sv_u = str(wcs_header.get("CTYPE2", "")).upper()
                        model_switch = (
                            ("TPV" in ctype1_in_u or "TPV" in ctype2_in_u)
                            and ("TAN-SIP" in ctype1_sv_u or "TAN-SIP" in ctype2_sv_u)
                        )
                        clear_gain = bool(
                            np.isfinite(med_in)
                            and np.isfinite(med_sv)
                            and np.isfinite(p95_in)
                            and np.isfinite(p95_sv)
                            and (med_sv <= med_in * 0.90)
                            and (p95_sv <= p95_in * 1.02)
                        )
                        if model_switch and not clear_gain:
                            if redo_requested:
                                logger.info(
                                    "redo_wcs=True: model-switch safeguard disabled (TPV->TAN-SIP check); continuing with solved/refined WCS."
                                )
                            else:
                                force_preserve_input_distortion = True
                                logger.warning(
                                    "WCS model switch TPV->TAN-SIP without clear gain (input med/p95=%.3f/%.3f, solved med/p95=%.3f/%.3f arcsec): "
                                    "preserving input distortion model and updating linear terms only.",
                                    float(med_in),
                                    float(p95_in),
                                    float(med_sv),
                                    float(p95_sv),
                                )
                        solved_not_better = bool(
                            np.isfinite(med_in)
                            and np.isfinite(med_sv)
                            and np.isfinite(p95_in)
                            and np.isfinite(p95_sv)
                            and ((med_sv > med_in * 1.02) or (p95_sv > p95_in * 1.02))
                        )
                        if keep_if_not_better and solved_not_better:
                            keep_input_wcs_full = True
                            logger.warning(
                                "Solved WCS is not better than input on matched points (input med/p95=%.3f/%.3f, solved med/p95=%.3f/%.3f arcsec): "
                                "keeping original full input WCS for this frame.",
                                float(med_in),
                                float(p95_in),
                                float(med_sv),
                                float(p95_sv),
                            )
                    # Distortion downgrade: a degraded solve (e.g. TAN-only
                    # because too few stars matched for the requested SIP
                    # order) must not strip a usable input distortion model.
                    # This fires even when too few matched points exist for
                    # the input-vs-solved comparison to run.
                    _input_has_distortion = _has_nonzero_distortion(self.header)
                    _solved_has_distortion = _has_nonzero_distortion(wcs_header)
                    if (
                        keep_if_not_better
                        and not keep_input_wcs_full
                        and not force_preserve_input_distortion
                        and input_wcs_for_compare is not None
                        and _input_has_distortion
                        and not _solved_has_distortion
                    ):
                        if clear_gain:
                            force_preserve_input_distortion = True
                            logger.info(
                                "Solved WCS lost the input distortion model but "
                                "shows a clear matched-point gain: preserving "
                                "input distortion with solved linear terms."
                            )
                        else:
                            keep_input_wcs_full = True
                            logger.warning(
                                "Solved WCS has no usable distortion model but "
                                "the input WCS does (too few matched stars for "
                                "the requested tweak order): keeping original "
                                "input WCS for this frame."
                            )
                    # --- Robust linear WCS refinement from matched stars ---
                    # Refit the solved WCS linear terms (center, scale, rotation)
                    # from the solve-field matched (pixel, sky) pairs with
                    # sigma-clipping.  astropy 8 removed WCS.fit_wcs_from_points,
                    # so a full SIP polynomial refit is unavailable here; we
                    # refit linear terms only and leave solve-field's distortion
                    # (SIP/PV) terms in place.  This corrects small zero-point /
                    # scale / rotation biases in the solved WCS.
                    if (
                        matched_points_for_refine is not None
                        and not keep_input_wcs_full
                        and _wcs_cfg_bool(
                            wcs_cfg, "refit_wcs_from_points", True
                        )
                    ):
                        _fit = self._fit_wcs_from_points(
                            matched_points_for_refine[0],
                            matched_points_for_refine[1],
                            matched_points_for_refine[2],
                            matched_points_for_refine[3],
                        )
                        if _fit is not None:
                            _before = np.nan
                            _after = np.nan
                            try:
                                _before_wcs = get_wcs(wcs_header)
                                if _before_wcs is not None:
                                    _before = (
                                        _best_wcs_match_separation_stats_arcsec(
                                            _before_wcs,
                                            matched_points_for_refine[0],
                                            matched_points_for_refine[1],
                                            matched_points_for_refine[2],
                                            matched_points_for_refine[3],
                                        )[0]
                                    )
                            except Exception:
                                _before = np.nan
                            for _fk, _fv in _fit["keywords"].items():
                                wcs_header[_fk] = _fv
                            wcs_header["CTYPE1"] = "RA---TAN"
                            wcs_header["CTYPE2"] = "DEC--TAN"
                            _normalize_projection_codes(
                                wcs_header, inplace=True
                            )
                            try:
                                _after_wcs = get_wcs(wcs_header)
                                if _after_wcs is not None:
                                    _after = (
                                        _best_wcs_match_separation_stats_arcsec(
                                            _after_wcs,
                                            matched_points_for_refine[0],
                                            matched_points_for_refine[1],
                                            matched_points_for_refine[2],
                                            matched_points_for_refine[3],
                                        )[0]
                                    )
                            except Exception:
                                _after = np.nan
                            logger.info(
                                "WCS linear refinement from %d matched stars: "
                                "rms %.3f -> %.3f arcsec.",
                                _fit["n_used"],
                                float(_before),
                                float(_after),
                            )
                    ctype1_in = str(self.header.get("CTYPE1", ""))
                    ctype2_in = str(self.header.get("CTYPE2", ""))
                    ctype1_sv = str(wcs_header.get("CTYPE1", ""))
                    ctype2_sv = str(wcs_header.get("CTYPE2", ""))
                    crpix1_in = float(self.header.get("CRPIX1", np.nan))
                    crpix2_in = float(self.header.get("CRPIX2", np.nan))
                    crpix1_sv = float(wcs_header.get("CRPIX1", np.nan))
                    crpix2_sv = float(wcs_header.get("CRPIX2", np.nan))
                    crval1_in = float(self.header.get("CRVAL1", np.nan))
                    crval2_in = float(self.header.get("CRVAL2", np.nan))
                    crval1_sv = float(wcs_header.get("CRVAL1", np.nan))
                    crval2_sv = float(wcs_header.get("CRVAL2", np.nan))
                    dra_arcsec = (
                        (crval1_sv - crval1_in + 180.0) % 360.0 - 180.0
                    ) * 3600.0
                    ddec_arcsec = (crval2_sv - crval2_in) * 3600.0
                    logger.info(
                        "WCS header delta: CTYPE (%s, %s) -> (%s, %s) | "
                        "dCRPIX=(%g, %g) px | dCRVAL=(%g, %g) arcsec",
                        ctype1_in, ctype2_in, ctype1_sv, ctype2_sv,
                        float(crpix1_sv - crpix1_in), float(crpix2_sv - crpix2_in),
                        float(dra_arcsec), float(ddec_arcsec),
                    )
                except Exception as exc:
                    log_warning_from_exception(
                        logger, "WCS input-vs-solved comparison skipped", exc
                    )

                if keep_input_wcs_full:
                    try:
                        self.header["NAXIS1"] = self.image.shape[1]
                        self.header["NAXIS2"] = self.image.shape[0]
                        safe_fits_write(self.fpath, self.image, self.header, output_verify="ignore")
                        # Cleanup solve artifacts even when keeping input WCS.
                        for pattern in [".rdls", ".solved", ".xyls", ".orig", ".axy", ".match", ".corr"]:
                            for f in _solve_artifact_glob(dirname, _tmp_tag, base, pattern):
                                try:
                                    os.remove(f)
                                except OSError:
                                    pass
                        for f in [scamp_temp, wcs_file, os.path.join(dirname, "none")]:
                            if f and os.path.isfile(f):
                                try:
                                    os.remove(f)
                                except OSError:
                                    pass
                        logger.info(
                            "Kept original input WCS (solved WCS not better on matched points)."
                        )
                        return self.header
                    except Exception as exc:
                        logger.warning(
                            "Failed while keeping original input WCS after solve comparison: %s",
                            exc,
                        )

                # Prefer SCAMP after solve-field when available to preserve TPV/PV-style
                # distortion solutions; fall back to solve-field WCS otherwise.
                trust_input_distortion = bool(
                    wcs_cfg.get("trust_input_distortion_model", False)
                )
                if redo_requested and trust_input_distortion:
                    logger.info(
                        "redo_wcs=True: ignoring input distortion trust and solving for a fresh distortion model."
                    )
                    trust_input_distortion = False
                prefer_scamp_tpv = bool(use_tpv_flow) and bool(
                    wcs_cfg.get("prefer_scamp_tpv_after_solve", True)
                )
                if use_tpv_flow and not prefer_scamp_tpv:
                    logger.info(
                        "prefer_scamp_tpv_after_solve=False: keeping solve-field "
                        "WCS without SCAMP refinement (projection_type=TPV)."
                    )
                if prefer_scamp_tpv:
                    try:
                        scamp_header = self._select_best_scamp_solution_after_solve(
                            rough_wcs_header=wcs_header,
                            wcs_cfg=wcs_cfg,
                            cpulimit=cpulimit,
                            matched_points=matched_points_for_refine,
                        )
                        if isinstance(scamp_header, fits.Header):
                            wcs_header = scamp_header
                    except Exception as exc:
                        logger.warning(
                            "SCAMP preference after solve-field failed: %s. Keeping solve-field WCS.",
                            exc,
                        )
                elif not use_tpv_flow:
                    logger.info(
                        "projection_type=SIP: skipping SCAMP refinement and using solve-field WCS."
                    )

                # Remove all previous WCS from science header, then add only WCS keywords from solver.
                # Optional behavior (option 1): preserve the *input* non-linear distortion model
                # (e.g. instrument TPV/PV polynomials) and only update linear terms from the solver.
                preserve_distortion = bool(
                    wcs_cfg.get("preserve_distortion_on_redo", False)
                )
                if redo_requested and preserve_distortion:
                    logger.info(
                        "redo_wcs=True: not preserving input distortion keywords; using freshly solved distortion model."
                    )
                    preserve_distortion = False
                if trust_input_distortion:
                    preserve_distortion = True
                    logger.info(
                        "trust_input_distortion_model=True: preserving input distortion keywords on redo_wcs"
                    )
                if force_preserve_input_distortion:
                    preserve_distortion = True
                    logger.info(
                        "Applying forced preserve-input-distortion mode due to weak TPV->TAN-SIP gain."
                    )
                # Heuristic: treat presence of PV or SIP/A/B polynomial coefficients as "non-linear distortion".
                _non_linear_key_prefixes = (
                    "PV",
                    "SIP_",
                    "A_",
                    "B_",
                    "AP_",
                    "BP_",
                    "D_",
                    "DP_",
                )
                had_non_linear_distortion = any(
                    k.startswith(_non_linear_key_prefixes)
                    for k in list(self.header.keys())
                )

                preserved_distortion_keys = {}
                preserved_ctype1 = self.header.get("CTYPE1")
                preserved_ctype2 = self.header.get("CTYPE2")
                if preserve_distortion and had_non_linear_distortion:
                    # Snapshot the non-linear/projection-specific keywords from the input WCS.
                    for k in list(self.header.keys()):
                        if k.startswith(_non_linear_key_prefixes):
                            preserved_distortion_keys[k] = self.header[k]

                self.header = remove_wcs_from_header(self.header)
                _wcs_prefixes = (
                    "CRPIX",
                    "CRVAL",
                    "CTYPE",
                    "CD",
                    "PC",
                    "CDELT",
                    "CROTA",
                    "PV",
                    "LONPOLE",
                    "LATPOLE",
                    "EQUINOX",
                    "WCSNAME",
                    "CUNIT",
                    "WCSAXES",
                    "PROJP",
                    "LTV",
                    "LTM",
                    "RADECSYS",
                    "RADESYS",
                    "RADYSYS",
                    "LONGPOLE",
                    "TNX",
                    "SIP_",
                )
                _wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
                for key in wcs_header:
                    if key in ("NAXIS", "NAXIS1", "NAXIS2"):
                        continue
                    is_wcs = any(key.startswith(p) for p in _wcs_prefixes)
                    if not is_wcs and "_" in key:
                        stem = key.split("_")[0] + "_"
                        is_wcs = stem in _wcs_stems and key.startswith(stem.rstrip("_"))
                    if is_wcs:
                        if preserve_distortion and had_non_linear_distortion:
                            # Skip non-linear distortion + projection identifiers.
                            if key in ("CTYPE1", "CTYPE2"):
                                continue
                            if key.startswith(
                                (
                                    "SIP_",
                                    "A_",
                                    "B_",
                                    "AP_",
                                    "BP_",
                                    "D_",
                                    "DP_",
                                    "PV",
                                )
                            ):
                                continue
                        self.header[key] = wcs_header[key]

                if preserve_distortion and had_non_linear_distortion:
                    # Restore input distortion model + projection identifiers.
                    if preserved_ctype1 is not None:
                        self.header["CTYPE1"] = preserved_ctype1
                    if preserved_ctype2 is not None:
                        self.header["CTYPE2"] = preserved_ctype2
                    for k, v in preserved_distortion_keys.items():
                        self.header[k] = v
                self.header = _normalize_projection_codes(self.header, inplace=True)
                self.header["NAXIS1"] = self.image.shape[1]
                self.header["NAXIS2"] = self.image.shape[0]

                # Astrometry.net solve-field outputs WCS in standard FITS 1-based convention,
                # so no CRPIX offset is needed by default. Optional crpix_offset (e.g. 0.5 or -0.5)
                # only if your solver uses a different pixel convention.
                crpix_offset = wcs_cfg.get("crpix_offset")
                if crpix_offset is None and wcs_cfg.get("crpix_fits_convention", False):
                    crpix_offset = 0.5  # legacy: old configs that applied +0.5
                if crpix_offset is None:
                    crpix_offset = 0.0
                if crpix_offset != 0.0:
                    if preserve_distortion and had_non_linear_distortion:
                        # When preserving distortion, we also preserve the
                        # distortion's associated linear terms. Applying an
                        # extra CRPIX offset would break that consistency.
                        allow = bool(wcs_cfg.get("allow_crpix_offset_with_preserved_distortion", False))
                        if not allow:
                            logger.info(
                                "Preserving distortion on redo: skipping crpix_offset=%.3f to keep PV/SIP consistency",
                                float(crpix_offset),
                            )
                            crpix_offset = 0.0
                if crpix_offset != 0.0:
                    with silence_astropy_wcs_info():
                        try:
                            tmp_wcs = get_wcs(self.header)
                            crpix1 = float(self.header["CRPIX1"])
                            crpix2 = float(self.header["CRPIX2"])
                            # CRPIX is defined in FITS 1-based pixel convention;
                            # use origin=1 so astropy does not apply an extra +1 offset.
                            rd = tmp_wcs.all_pix2world([[crpix1]], [[crpix2]], 1)
                            self.header["CRPIX1"] = crpix1 + crpix_offset
                            self.header["CRPIX2"] = crpix2 + crpix_offset
                            self.header["CRVAL1"] = float(rd[0].flat[0])
                            self.header["CRVAL2"] = float(rd[1].flat[0])
                            logger.debug(
                                "Applied CRPIX offset %.2f for solver convention",
                                crpix_offset,
                            )
                        except Exception as e:
                            log_warning_from_exception(
                                logger, "CRPIX offset correction skipped", e
                            )

                with silence_astropy_wcs_info():
                    merged_wcs = get_wcs(self.header)
                    if merged_wcs is None:
                        logger.warning("Merged WCS is None after get_wcs - cannot validate pixel scale")
                        if _input_header_wcs_valid:
                            logger.warning(
                                "Restoring original input WCS (merged solve unusable)."
                            )
                            try:
                                _restore_input_wcs_inplace()
                            except Exception:
                                return np.nan
                            return self.header
                    else:
                        try:
                            scale = pixel_area_scale_arcsec(
                                merged_wcs,
                                getattr(merged_wcs, "array_shape", None)
                                or (
                                    int(self.header["NAXIS2"]),
                                    int(self.header["NAXIS1"]),
                                )
                                if self.header is not None
                                else None,
                            )
                            if not np.isfinite(scale) or scale <= 0 or scale > 3600:
                                logger.warning(
                                    "Merged WCS has invalid pixel scale (%.6f); rejecting",
                                    scale,
                                )
                                if _input_header_wcs_valid:
                                    logger.warning(
                                        "Restoring original input WCS "
                                        "(merged solve invalid)."
                                    )
                                    try:
                                        _restore_input_wcs_inplace()
                                    except Exception:
                                        return np.nan
                                    return self.header
                                return np.nan
                        except Exception as ev:
                            log_warning_from_exception(
                                logger, "Could not validate merged WCS", ev
                            )

                # Adopted solutions always hit the disk - redo_wcs or a
                # missing WCS means the on-file astrometry is untrusted.
                safe_fits_write(self.fpath, self.image, self.header,
                    output_verify="ignore",
                )
                self.solved_new_wcs = True
                # Final cleanup of heavy solve-field artifacts once diagnostics
                # and header updates are complete.
                if keep_match_for_compare:
                    for pattern in [".axy", ".match", ".corr"]:
                        for f in _solve_artifact_glob(dirname, _tmp_tag, base, pattern):
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                if os.path.isfile(wcs_file):
                    os.remove(wcs_file)
                return self.header
            except Exception as e:
                for f in [
                    scamp_temp,
                    wcs_file,
                    os.path.join(dirname, "none"),
                ]:
                    if f and os.path.isfile(f):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                logger.exception("Failed to update header from WCS file: %s", e)
                if _input_header_wcs_valid:
                    # The merge may have stripped WCS keywords from
                    # self.header before failing; restore the input WCS so
                    # the image keeps its original astrometry.
                    logger.warning(
                        "Restoring original input WCS after merge failure."
                    )
                    try:
                        _restore_input_wcs_inplace()
                    except Exception:
                        return np.nan
                    try:
                        # A partial merge may already be on disk; rewrite so
                        # the file matches the restored header.
                        safe_fits_write(
                            self.fpath, self.image, self.header,
                            output_verify="ignore",
                        )
                    except Exception:
                        pass
                    return self.header
                return np.nan
