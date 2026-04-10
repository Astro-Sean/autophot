#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized and robust WCS solver and FITS header cleaner.
Handles faint sources, cosmic rays, and edge cases gracefully.
Uses SExtractor with a standard Gaussian convolution filter for source detection by default,
with fallback to astrometry.net-only mode.
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
from pathlib import Path
from contextlib import contextmanager

# --- Third-Party Imports ---
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from astropy.table import Table
from functions import safe_fits_write

# --- Local Imports (optional) ---
try:
    from functions import border_msg, log_warning_from_exception  # type: ignore
except (ModuleNotFoundError, ImportError):
    # Minimal fallback for environments missing the full photometry stack.
    def border_msg(message: str, *args, **kwargs) -> str:
        return str(message)

    def log_warning_from_exception(logger, message, exc, *, exc_info=False):
        logger.warning("%s: %s", message, exc, exc_info=exc_info)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Suppress Astropy Warnings ---
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

# --- Compiled regex for log cleaning (reuse) ---
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
    v = cfg.get(key, default)
    if v is None:
        return int(default)
    return int(v)


@contextmanager
def silence_astropy_wcs_info():
    """
    Context manager to silence Astropy WCS info-level logs.
    Useful for reducing log clutter during WCS operations.
    """
    wcs_logger = logging.getLogger("astropy.wcs.wcs")
    prev_level = wcs_logger.level
    wcs_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        wcs_logger.setLevel(prev_level)


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
        "SCAMP astrometric catalog fetch failed (Vizier/network). "
        "Raise `wcs.scamp_ref_timeout` (AutoPHoT default 60 s; SCAMP built-in is 10 s) "
        "or set `wcs.scamp_ref_server` to another mirror (default is vizier.cfa.harvard.edu; "
        "alternatives include vizier.unistra.fr, vizier.ast.cam.ac.uk). SCAMP log: %s",
        scamp_log_fpath,
    )


def update_wcs_center(
    fits_path: str, ra: float, dec: float, overwrite: bool = True
) -> fits.Header:
    """
    Updates the WCS header of a FITS file to center the reference pixel and set the reference sky coordinates.
    Uses FITS 1-indexed convention: reference pixel at image center is (NAXIS+1)/2.

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
        logger.exception(f"Failed to update WCS header: {e}")
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
        # This is catalog data, not image data, so NaN preservation is not critical
        # But use safe_fits_write for consistency
        from astropy.io import fits
        # For multi-extension HDULists, we need to use hdulist.writeto directly
        # safe_fits_write is for single image + header
        hdulist.writeto(writeto, overwrite=True)
    return hdulist


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
        return None
        
    try:
        header = _normalize_projection_codes(header, inplace=False)
        
        # Check for basic WCS keywords
        required = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2']
        missing = [k for k in required if k not in header]
        if missing:
            logger.debug(f"WCS creation failed: missing keywords {missing}")
            return None
            
        with warnings.catch_warnings():
            if silent:
                warnings.simplefilter("ignore")
            with silence_astropy_wcs_info():
                wcs = WCS(header, fix=True, relax=True)
                
        # Validate WCS has celestial component
        if not wcs.has_celestial:
            logger.debug("WCS has no celestial component")
            return None
            
        # Ensure 2D celestial WCS so reproject and callers get consistent pixel grid
        naxis = getattr(wcs.wcs, "naxis", 2)
        if naxis > 2:
            wcs = wcs.celestial
            
        # Test transformation
        try:
            test_x, test_y = float(header['CRPIX1']), float(header['CRPIX2'])
            test_world = wcs.pixel_to_world(test_x, test_y)
            if test_world is None:
                logger.debug("WCS transformation test failed")
                return None
        except Exception as e:
            logger.debug(f"WCS transformation test failed: {e}")
            return None
            
        return wcs
        
    except Exception as e:
        logger.debug(f"WCS creation failed: {e}")
        return None


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

    if len(pix_x) < 10:
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
    logger.info(
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

    if len(x) < 10:
        return None

    # Keep a manageable number of points (evenly sampled by row order).
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x, y, ra, dec = x[idx], y[idx], ra[idx], dec[idx]

    return x, y, ra, dec


def _choose_xy_shift_for_fit_wcs(
    initial_wcs: WCS,
    x_m: np.ndarray,
    y_m: np.ndarray,
    ra_m: np.ndarray,
    dec_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Determine whether matched x/y are likely 0-based or FITS 1-based.

    `fit_wcs_from_points` expects FITS convention (1-based). We compare angular
    residuals to the initial solve-field WCS under two hypotheses:
      H1: x/y already FITS-like      -> use (x, y)
      H0: x/y are 0-based centroids  -> use (x+1, y+1)
    and choose the lower-median-separation case.
    """
    try:
        sky_true = SkyCoord(ra=ra_m * u.deg, dec=dec_m * u.deg, frame="icrs")

        # Assume x/y are FITS-like (1-based) for all_pix2world origin=0
        ra1, dec1 = initial_wcs.all_pix2world(x_m, y_m, 0)
        sky1 = SkyCoord(ra=np.asarray(ra1) * u.deg, dec=np.asarray(dec1) * u.deg, frame="icrs")
        med_sep_1based = float(np.nanmedian(sky1.separation(sky_true).to(u.arcsec).value))

        # Assume x/y are 0-based -> convert to FITS-like by +1
        x0 = x_m + 1.0
        y0 = y_m + 1.0
        ra0, dec0 = initial_wcs.all_pix2world(x0, y0, 0)
        sky0 = SkyCoord(ra=np.asarray(ra0) * u.deg, dec=np.asarray(dec0) * u.deg, frame="icrs")
        med_sep_0based = float(np.nanmedian(sky0.separation(sky_true).to(u.arcsec).value))

        if np.isfinite(med_sep_0based) and np.isfinite(med_sep_1based):
            if med_sep_0based + 1e-6 < med_sep_1based:
                return x0, y0, 1.0
            return x_m, y_m, 0.0
    except Exception:
        pass
    # Conservative fallback: do not shift
    return x_m, y_m, 0.0


def _wcs_match_separation_stats_arcsec(
    wcs_obj: WCS,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> tuple[float, float]:
    """
    Compute robust angular-separation stats (median, p95) for matched points.
    """
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
    Compute robust separation stats for both pixel-origin hypotheses and return
    the best one.

    Returns:
        (median_arcsec, p95_arcsec, applied_shift_px)
    """
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
    Convert sky coordinates to pixel coordinates robustly with consistent origin.

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
    - This is the unified function for all WCS world-to-pixel conversions.
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
    Convert sky coordinates to pixel coordinates robustly (legacy wrapper).

    Deprecated: Use wcs_world_to_pixel instead.
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
    # Default behaviour: run SCAMP once per image.
    # Multi-trial SCAMP optimization can be expensive and is opt-in.
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


def _estimate_crpix_linear_nudge_from_matches(
    wcs_obj: WCS,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    min_points: int = 30,
) -> dict | None:
    """
    Estimate a small global CRPIX nudge (dx, dy) from matched star residuals.

    The estimate is based on robust medians of (x_obs - x_pred, y_obs - y_pred),
    trying both pixel-origin hypotheses (raw and +1 px) for match-table columns.
    """
    x_obs = np.asarray(x_pix, dtype=float)
    y_obs = np.asarray(y_pix, dtype=float)
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)
    n = int(min(len(x_obs), len(y_obs), len(ra_arr), len(dec_arr)))
    if n < int(max(3, min_points)):
        return None

    x_obs = x_obs[:n]
    y_obs = y_obs[:n]
    ra_arr = ra_arr[:n]
    dec_arr = dec_arr[:n]

    x_pred, y_pred = wcs_world_to_pixel(wcs_obj, ra_arr, dec_arr, origin=0)
    if len(x_pred) != n or len(y_pred) != n:
        return None

    best = None
    for shift_px in (0.0, 1.0):
        xo = x_obs + shift_px
        yo = y_obs + shift_px
        valid = (
            np.isfinite(xo)
            & np.isfinite(yo)
            & np.isfinite(x_pred)
            & np.isfinite(y_pred)
        )
        if int(np.count_nonzero(valid)) < int(max(3, min_points)):
            continue

        dx = xo[valid] - x_pred[valid]
        dy = yo[valid] - y_pred[valid]
        dx_med = float(np.nanmedian(dx))
        dy_med = float(np.nanmedian(dy))
        r_pre = np.hypot(dx, dy)
        r_post = np.hypot(dx - dx_med, dy - dy_med)
        pre_med = float(np.nanmedian(r_pre))
        post_med = float(np.nanmedian(r_post))

        candidate = {
            "origin_shift_px": float(shift_px),
            "dx_px": dx_med,
            "dy_px": dy_med,
            "pre_med_resid_px": pre_med,
            "post_med_resid_px": post_med,
            "n_valid": int(np.count_nonzero(valid)),
        }
        if best is None or (
            np.isfinite(candidate["pre_med_resid_px"])
            and candidate["pre_med_resid_px"] < best["pre_med_resid_px"]
        ):
            best = candidate

    return best


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

    # Robust clip around median to suppress residual outliers.
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
            # Keep kernel practical and stable for SExtractor.
            return float(np.clip(fwhm_pix, 1.0, 20.0))
    return 3.0


def create_nnw_file(path: str) -> None:
    """
    Create a default SExtractor neural network weights file for star/galaxy classification.
    """
    nnw_text = """
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
    nnw_text = "\n".join(line.strip() for line in nnw_text.split("\n") if line.strip())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(nnw_text)


# =============================================================================
# Main Class
# =============================================================================


class WCSSolver:
    """
    WCS solver and FITS header cleaner.
    Handles faint sources, cosmic rays, and edge cases gracefully.
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
                logger.exception(f"Error handling key '{key}': {e}")
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
        return output_file

    def _run_solve_field(
        self,
        args: list,
        wcs_file: str,
        timeout_sec: float,
        logpath: str,
    ) -> bool:
        """Run solve-field with given args; return True if wcs_file was created."""
        try:
            with open(logpath, "a", encoding="utf-8") as logf:
                logf.write(" ".join(map(str, args)) + "\n")
                kwargs = dict(shell=False, stdout=logf, stderr=subprocess.STDOUT)
                if os.name != "nt":
                    kwargs["preexec_fn"] = os.setsid
                pro = subprocess.Popen(args, **kwargs)
            try:
                pro.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(os.getpgid(pro.pid), 9)
                    except (OSError, AttributeError):
                        pro.kill()
                else:
                    pro.kill()
                pro.wait()
                logger.warning("solve-field exceeded timeout (%s s)", timeout_sec)
            self.clean_log(logpath)
            return os.path.isfile(wcs_file)
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
        logger.info(border_msg("Solving for WCS with SCAMP"))

        scamp_exe = wcs_cfg.get("scamp_exe_loc") or shutil.which("scamp")
        sex_exe = wcs_cfg.get("sextractor_exe_loc") or shutil.which("sex")
        if not scamp_exe and shutil.which("scamp") is None:
            logger.warning("SCAMP executable not found; cannot use SCAMP solver.")
            return np.nan
        if not sex_exe and shutil.which("sex") is None:
            logger.warning("SExtractor executable not found; cannot use SCAMP solver.")
            return np.nan
        scamp_exe = str(
            scamp_exe
            if scamp_exe and os.path.isfile(str(scamp_exe))
            else shutil.which("scamp")
        )
        sex_exe = str(
            sex_exe if sex_exe and os.path.isfile(str(sex_exe)) else shutil.which("sex")
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
        scamp_log = os.path.join(dirname, f"scamp_{base}.log")

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
                "CLASS_STAR",
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

            # SExtractor's FILTER_NAME must point to a writable convolution file.
            # We generate a temporary Gaussian kernel at runtime so we can omit
            # repository/distribution `.conv` files.
            conv_filter_path = os.path.join(temp_dir, "gaussian_7x7.conv")
            conv_fwhm_pix = _resolve_sextractor_conv_fwhm_pixels(
                wcs_cfg, self.header, self.default_input
            )
            create_conv_file(conv_filter_path, fwhm_pixels=conv_fwhm_pix)
            logger.info(
                "SCAMP/SExtractor convolution kernel built from FWHM=%.2f px",
                conv_fwhm_pix,
            )

            nnw_file = os.path.join(temp_dir, "default.nnw")
            create_nnw_file(nnw_file)
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
                satur_str = str(float(satur_val))
            except Exception:
                satur_str = "1e7"

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
                "STARNNW_NAME": nnw_file,
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
                logger.info(
                    "Running SExtractor for SCAMP: %s", " ".join(map(str, sex_cmd))
                )
                subprocess.run(
                    sex_cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_sec,
                    cwd=temp_dir,
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "SExtractor for SCAMP exceeded timeout (%.1f s)", timeout_sec
                )
                return np.nan
            if not os.path.isfile(cat_path):
                logger.warning(
                    "SCAMP: SExtractor catalog not created; aborting SCAMP solve."
                )
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
                logger.info(
                    "SCAMP threads requested=%r -> using 4 (minimum default).",
                    _snth,
                )
                scamp_threads = 4
            logger.info(
                "SCAMP threads: %d (wcs.scamp_nthreads=%r)",
                scamp_threads,
                _snth,
            )
            # SCAMP defaults REF_TIMEOUT=10s against vizier.unistra.fr; that often fails on slow links.
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
                # SCAMP did not emit any HEAD file in the working directory.
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
                from functions import remove_wcs_from_header

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
            from functions import remove_wcs_from_header

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

                # Run SCAMP on the seeded temporary image.
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
                med_arcsec, p95_arcsec, _ = _best_wcs_match_separation_stats_arcsec(
                    get_wcs(scamp_header), x_m, y_m, ra_m, dec_m
                )
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
            logger.info(
                "Using SCAMP WCS solution (trial='%s', preferred TPV/PV distortion model).",
                best_trial_label,
            )
            _log_header_distortion_coefficients(
                best_trial_header, prefix="SCAMP solved WCS"
            )
            return best_trial_header
        logger.info("SCAMP unavailable/failed after solve-field; keeping solve-field WCS.")
        return np.nan

    def _apply_post_scamp_linear_refinement(
        self,
        wcs_header: fits.Header,
        wcs_cfg: dict,
        matched_points: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> fits.Header:
        """
        Optionally apply a small CRPIX nudge using matched points.

        This preserves distortion terms and adjusts only the linear reference pixel.
        """
        try:
            enable_nudge = bool(wcs_cfg.get("post_scamp_linear_refine", True))
            if not enable_nudge or matched_points is None:
                return wcs_header

            x_m, y_m, ra_m, dec_m = matched_points
            min_points = int(
                max(8, wcs_cfg.get("post_scamp_linear_refine_min_points", 30))
            )
            max_shift_px = float(
                max(0.0, wcs_cfg.get("post_scamp_linear_refine_max_shift_px", 2.0))
            )
            min_improve_px = float(
                max(0.0, wcs_cfg.get("post_scamp_linear_refine_min_improve_px", 0.03))
            )
            min_improve_frac = float(
                max(
                    0.0,
                    wcs_cfg.get("post_scamp_linear_refine_min_improve_frac", 0.02),
                )
            )

            nudge = _estimate_crpix_linear_nudge_from_matches(
                get_wcs(wcs_header),
                x_m,
                y_m,
                ra_m,
                dec_m,
                min_points=min_points,
            )
            if nudge is None:
                return wcs_header

            dx_px = float(nudge["dx_px"])
            dy_px = float(nudge["dy_px"])
            shift_norm = float(np.hypot(dx_px, dy_px))
            pre_med_px = float(nudge["pre_med_resid_px"])
            post_med_px = float(nudge["post_med_resid_px"])
            need_improve = max(min_improve_px, pre_med_px * min_improve_frac)
            improve = pre_med_px - post_med_px
            if not (
                np.isfinite(dx_px)
                and np.isfinite(dy_px)
                and np.isfinite(pre_med_px)
                and np.isfinite(post_med_px)
                and shift_norm <= max_shift_px
                and improve >= need_improve
            ):
                logger.info(
                    "Skipped post-solve CRPIX nudge "
                    "(n=%d, |shift|=%.3f px, improve=%.4f px, required>=%.4f px, max_shift=%.3f px).",
                    int(nudge.get("n_valid", 0)),
                    float(shift_norm),
                    float(improve),
                    float(need_improve),
                    float(max_shift_px),
                )
                return wcs_header

            old_crpix1 = float(wcs_header.get("CRPIX1", np.nan))
            old_crpix2 = float(wcs_header.get("CRPIX2", np.nan))
            if not (np.isfinite(old_crpix1) and np.isfinite(old_crpix2)):
                return wcs_header

            wcs_before_nudge = get_wcs(wcs_header)
            wcs_header["CRPIX1"] = old_crpix1 + dx_px
            wcs_header["CRPIX2"] = old_crpix2 + dy_px
            med_before, p95_before, _ = _best_wcs_match_separation_stats_arcsec(
                wcs_before_nudge, x_m, y_m, ra_m, dec_m
            )
            med_after, p95_after, _ = _best_wcs_match_separation_stats_arcsec(
                get_wcs(wcs_header), x_m, y_m, ra_m, dec_m
            )
            logger.info(
                "Applied post-solve CRPIX nudge: dCRPIX=(%+.3f,%+.3f) px "
                "(n=%d, origin_shift=%+.1f px, residual med %.3f->%.3f px; "
                "sky med/p95 %.3f/%.3f -> %.3f/%.3f arcsec).",
                dx_px,
                dy_px,
                int(nudge["n_valid"]),
                float(nudge["origin_shift_px"]),
                pre_med_px,
                post_med_px,
                float(med_before),
                float(p95_before),
                float(med_after),
                float(p95_after),
            )
            return wcs_header
        except Exception as exc:
            log_warning_from_exception(
                logger, "Post-solve linear refinement skipped", exc
            )
            return wcs_header

    # --- Plate Solve ---
    def plate_solve(
        self,
        solvefield_exe: str = None,
        downsample: int = None,
        cpulimit: int = None,
        background_std: str = None,
        skip_verify: bool = True,
    ) -> fits.Header:
        """
        Run a WCS solver for this image.

        When wcs.solver is "scamp" and SCAMP/sex are available, use SCAMP
        (with SExtractor) to derive the WCS. Otherwise, fall back to
        Astrometry.net solve-field.
        """
        wcs_cfg = self.default_input.get("wcs") or {}
        redo_requested = bool(wcs_cfg.get("redo_wcs", True))
        projection_type = str(wcs_cfg.get("projection_type", "TPV")).strip().upper()
        if projection_type not in {"TPV", "SIP"}:
            logger.warning(
                "Unknown wcs.projection_type=%r; defaulting to TPV flow.",
                projection_type,
            )
            projection_type = "TPV"
        use_tpv_flow = projection_type == "TPV"
        logger.info(
            "WCS projection_type=%s (%s)",
            projection_type,
            "solve-field + SCAMP" if use_tpv_flow else "solve-field only",
        )
        if redo_requested and not skip_verify:
            logger.info(
                "redo_wcs=True: forcing fresh solve (skip verify of input WCS)."
            )
            skip_verify = True
        solver = str(wcs_cfg.get("solver", "astrometry")).strip().lower()

        # Optional SCAMP path
        if solver == "scamp" and use_tpv_flow:
            try:
                header = self._plate_solve_scamp(wcs_cfg=wcs_cfg, cpulimit=cpulimit)
                if isinstance(header, fits.Header):
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
            logger.info(
                "projection_type=SIP: ignoring solver=scamp and using solve-field only."
            )

        logger.info(
            border_msg(
                f"Solving for WCS with Astrometry.net ({os.path.basename(self.fpath)})"
            )
        )
        if not solvefield_exe:
            logger.warning(
                "Astrometry.net 'solve-field' executable not found; skipping solve-field WCS step. "
                "To install with conda:\n"
                "  conda install -c conda-forge astrometry\n"
                "Then ensure 'solve-field' is on PATH or set wcs.solve_field_exe_loc in your YAML."
            )
            return np.nan
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
        downsample = (
            int(downsample)
            if downsample is not None
            else _wcs_cfg_int(wcs_cfg, "downsample", 2)
        )

        # --- Prepare paths (fixed temp names to avoid collisions) ---
        dirname = os.path.dirname(self.fpath)
        base = os.path.splitext(os.path.basename(self.fpath))[0]
        wcs_file = os.path.join(dirname, "astrometry_temp.wcs.fits")
        astrometry_log_fpath = os.path.join(dirname, f"astrometry_{base}.log")
        sextractor_exe = (
            wcs_cfg.get("sextractor_exe_loc")
            or shutil.which("sex")
            or shutil.which("sextractor")
        )
        use_sextractor = sextractor_exe is not None
        # use_sextractor = False
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
                "CLASS_STAR",
                "ELLIPTICITY",
                "BACKGROUND",
                "THRESHOLD",
                "FLAGS",
                "SNR_WIN",
                "XPEAK_IMAGE",
                "YPEAK_IMAGE",
            ]
            Path(param_file).write_text("\n".join(params))
            nnw_file = os.path.join(temp_dir, "default.nnw")
            create_nnw_file(nnw_file)
            config_file = os.path.join(temp_dir, "default.sex")

            # Generate a temporary Gaussian convolution filter for SExtractor.
            conv_filter_path = os.path.join(temp_dir, "gaussian_7x7.conv")
            conv_fwhm_pix = _resolve_sextractor_conv_fwhm_pixels(
                wcs_cfg, self.header, self.default_input
            )
            create_conv_file(conv_filter_path, fwhm_pixels=conv_fwhm_pix)
            logger.info(
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
                    "No pixel_scale in config or header; using 0.1--5 arcsec/pix"
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
                "STARNNW_NAME": nnw_file,
            }
            with open(config_file, "w") as f:
                for k, v in final_config.items():
                    f.write(f"{k}\t{v}\n")

            # --- Prepare SExtractor command ---
            sextractor_cmd = None
            if use_sextractor:
                logger.info(
                    "Using SExtractor with Gaussian convolution filter for source detection."
                )
                sextractor_cmd = str(sextractor_exe)
            else:
                logger.warning("SExtractor not found. Proceeding without.")

            # --- Scale bounds: use range around known pixel scale when available ---
            # A wrong pixel_scale can prevent convergence. We therefore build two sets:
            #   - constrained: around the hint (if present)
            #   - wide: 0.1--5.0 arcsec/pix (no hint)
            ra, dec = self.default_input.get("target_ra"), self.default_input.get(
                "target_dec"
            )
            radius_deg = _wcs_cfg_float(wcs_cfg, "search_radius", 0.5)

            def _build_scale_args(
                scale_low: float, scale_high: float, include_pos: bool
            ) -> list[str]:
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

            if pixel_scale and pixel_scale > 0:
                scale_low_hint = pixel_scale * 0.6
                scale_high_hint = pixel_scale * 1.5
            else:
                scale_low_hint, scale_high_hint = 0.1, 5.0

            scale_low_wide, scale_high_wide = 0.1, 5.0

            scale_args_constrained = _build_scale_args(
                scale_low_hint, scale_high_hint, include_pos=True
            )
            scale_only_args_constrained = _build_scale_args(
                scale_low_hint, scale_high_hint, include_pos=False
            )
            scale_args_wide = _build_scale_args(
                scale_low_wide, scale_high_wide, include_pos=True
            )
            scale_only_args_wide = _build_scale_args(
                scale_low_wide, scale_high_wide, include_pos=False
            )

            # Start with constrained args (if pixel_scale provided); we will drop the hint on retry.
            scale_args = scale_args_constrained
            scale_only_args = scale_only_args_constrained

            # # --- solve-field matching and depth ---
            # code_tolerance = float(wcs_cfg.get("code_tolerance", 0.01))
            # scale_args += ["--code-tolerance", str(code_tolerance)]

            # --- Downsample: lighten astrometry.net load for large images ---
            ny, nx = self.image.shape[0], self.image.shape[1]
            if nx * ny < 1500 * 1500:
                downsample = max(1, downsample - 1)
            downsample = max(1, int(downsample))

            if os.path.exists(astrometry_log_fpath):
                os.remove(astrometry_log_fpath)
            new_fits_temp = os.path.join(dirname, "astrometry_newfits_temp.fits")
            scamp_temp = os.path.join(dirname, "astrometry_temp_scamp")
            corr_temp = os.path.join(dirname, "astrometry_temp.corr")
            rdl_temp = os.path.join(dirname, "astrometry_temp.rdls")
            match_temp = os.path.join(dirname, "astrometry_temp.match")
            solved_temp = os.path.join(dirname, "astrometry_temp.solved")
            common_args = [
                str(solvefield_exe),
                "--no-remove-lines",
                "--overwrite",
                "--downsample",
                str(downsample),
                "--new-fits",
                new_fits_temp,
                "--wcs",
                str(wcs_file),
                "--scamp",
                scamp_temp,
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
                str(int(timeout_sec)),
                str(self.fpath),
            ]
            # Optional: request more detected objects for solve-field indexing/matching.
            # Larger values can improve robustness in sparse fields at modest runtime cost.
            objs = wcs_cfg.get("objs", 1200)
            if objs is not None:
                try:
                    n_objs = int(objs)
                    if n_objs > 0:
                        common_args.insert(-1, "--objs")
                        common_args.insert(-1, str(n_objs))
                        logger.info(
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
            if use_crpix_center:
                common_args.insert(-1, "--crpix-center")
                logger.info(
                    "solve-field: using --crpix-center (reference pixel at image center)"
                )
            else:
                logger.info(
                    "solve-field: omitting --crpix-center (solver reference pixel; "
                    "often closer to instrument WCS - set wcs.solve_field_crpix_center: true for legacy)"
                )
            # Tweak order(s) for solve-field. Can be a single int or a list.
            # Example YAML:
            #   solve_field_tweak_order: 5
            #   solve_field_tweak_orders: [5, 3, 2, 1, 0]
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
            logger.info("solve-field tweak order sequence: %s", tweak_orders)

            # --- Step 1: Optional verify existing WCS ---
            if not skip_verify:
                logger.info("Attempting to verify WCS with --no-verify")
                args_verify = common_args + ["--no-verify"] + scale_args
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
                    ]
                if self._run_solve_field(
                    args_verify, wcs_file, timeout_sec, astrometry_log_fpath
                ):
                    logger.info("WCS verified successfully")
                else:
                    logger.warning("Could not verify WCS - proceeding to blind solve")

            def _attempt_solve_with_args(scale_args_use: list[str], label: str) -> None:
                # Step 2: Solve with (optional) SExtractor, tweak orders 2,1,0
                for tweak_order in tweak_orders:
                    if os.path.isfile(wcs_file):
                        break
                    logger.info(
                        "Attempting WCS solve (%s) with tweak order %s",
                        label,
                        tweak_order,
                    )
                    args = (
                        common_args
                        + ["--tweak-order", str(tweak_order), "--no-verify"]
                        + scale_args_use
                    )
                    if use_sextractor and sextractor_cmd:
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
                        args, wcs_file, timeout_sec, astrometry_log_fpath
                    ):
                        logger.info(
                            "WCS solved (%s) with tweak order %s", label, tweak_order
                        )
                        break
                    logger.warning(
                        "No solution (%s) with tweak order %s", label, tweak_order
                    )

            # First attempt: use the pixel-scale hint when available
            _attempt_solve_with_args(
                scale_args,
                label=(
                    "with pixel-scale hint"
                    if (pixel_scale and pixel_scale > 0)
                    else "wide scale"
                ),
            )

            # If still not solved and we had a pixel-scale hint, drop it and retry once.
            if not os.path.isfile(wcs_file) and (pixel_scale and pixel_scale > 0):
                logger.warning(
                    "WCS not solved with pixel-scale hint (%.4g arcsec/pix). Dropping pixel scale constraint and retrying with 0.1--5 arcsec/pix.",
                    float(pixel_scale),
                )
                scale_args = scale_args_wide
                scale_only_args = scale_only_args_wide
                _attempt_solve_with_args(
                    scale_args, label="wide scale (dropped pixel-scale hint)"
                )

            # --- Step 3: Retry without SExtractor ---
            if not os.path.isfile(wcs_file):
                logger.warning("Retrying without SExtractor")
                for tweak_order in tweak_orders:
                    if os.path.isfile(wcs_file):
                        break
                    logger.info(
                        "Attempting WCS solve without SExtractor (tweak order %s)",
                        tweak_order,
                    )
                    args_no_sex = (
                        common_args
                        + ["--tweak-order", str(tweak_order), "--no-verify"]
                        + scale_args
                    )
                    if self._run_solve_field(
                        args_no_sex, wcs_file, timeout_sec, astrometry_log_fpath
                    ):
                        logger.info(
                            "WCS solved without SExtractor (tweak order %s)",
                            tweak_order,
                        )
                        break
                    logger.warning(
                        "No solution without SExtractor (tweak order %s)", tweak_order
                    )

            # --- Step 4: If position-constrained solve failed, retry blind (no RA/Dec/radius) ---
            if not os.path.isfile(wcs_file) and ra is not None and dec is not None:
                logger.warning("Retrying with blind solve (no position constraint).")
                for tweak_order in tweak_orders:
                    if os.path.isfile(wcs_file):
                        break
                    logger.info("Blind solve with tweak order %s", tweak_order)
                    args_blind = (
                        common_args
                        + ["--tweak-order", str(tweak_order), "--no-verify"]
                        + scale_only_args
                    )
                    if use_sextractor and sextractor_cmd:
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
                        args_blind, wcs_file, timeout_sec, astrometry_log_fpath
                    ):
                        logger.info(
                            "WCS solved with blind solve (tweak order %s)", tweak_order
                        )
                        break
                    logger.warning(
                        "No solution with blind solve (tweak order %s)", tweak_order
                    )

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
                    for f in [new_fits_temp, scamp_temp, os.path.join(dirname, "none")]:
                        if os.path.isfile(f):
                            os.remove(f)
                    return self.header
                except Exception:
                    pass
                for f in [new_fits_temp, scamp_temp, os.path.join(dirname, "none")]:
                    if os.path.isfile(f):
                        os.remove(f)
                logger.warning("\tCould not solve WCS - returning NaN")
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

            patterns_to_remove = [".rdls", ".solved", ".xyls"]
            if not keep_match_and_axy:
                patterns_to_remove += [".axy", ".match", ".corr"]
            for pattern in patterns_to_remove:
                for f in glob.glob(os.path.join(dirname, f"*{pattern}")):
                    os.remove(f)
            for f in [new_fits_temp, scamp_temp, os.path.join(dirname, "none")]:
                if os.path.isfile(f):
                    os.remove(f)

            # --- Update FITS header with WCS from the .wcs.fits file ---
            try:
                with fits.open(wcs_file) as wcs_hdul:
                    wcs_header = wcs_hdul[0].header
                wcs_header = _normalize_projection_codes(wcs_header, inplace=False)
                _log_header_distortion_coefficients(
                    wcs_header, prefix="astrometry.net solved WCS"
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
                    if diag_points is not None:
                        x_d, y_d, ra_d, dec_d = diag_points
                        matched_points_for_refine = (
                            np.asarray(x_d, dtype=float),
                            np.asarray(y_d, dtype=float),
                            np.asarray(ra_d, dtype=float),
                            np.asarray(dec_d, dtype=float),
                        )
                        med_in, p95_in, sh_in = _best_wcs_match_separation_stats_arcsec(
                            input_wcs_for_compare, x_d, y_d, ra_d, dec_d
                        )
                        med_sv, p95_sv, sh_sv = _best_wcs_match_separation_stats_arcsec(
                            solved_wcs_for_compare, x_d, y_d, ra_d, dec_d
                        )
                        logger.info(
                            "WCS compare on solve-field matches (n=%d): input med/p95=%.3f/%.3f arcsec (shift=%+.1f px) | solved med/p95=%.3f/%.3f arcsec (shift=%+.1f px)",
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
                                    "redo_wcs=True: model-switch safeguard disabled "
                                    "(TPV->TAN-SIP check); continuing with solved/refined WCS."
                                )
                            else:
                                force_preserve_input_distortion = True
                                logger.warning(
                                    "WCS model switch TPV->TAN-SIP without clear gain "
                                    "(input med/p95=%.3f/%.3f, solved med/p95=%.3f/%.3f arcsec): "
                                    "preserving input distortion model and updating linear terms only.",
                                    float(med_in),
                                    float(p95_in),
                                    float(med_sv),
                                    float(p95_sv),
                                )
                        keep_if_not_better = bool(
                            wcs_cfg.get(
                                "keep_input_wcs_if_solved_not_better", True
                            )
                        )
                        if redo_requested:
                            keep_if_not_better = False
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
                                "Solved WCS is not better than input on matched points "
                                "(input med/p95=%.3f/%.3f, solved med/p95=%.3f/%.3f arcsec): "
                                "keeping original full input WCS for this frame.",
                                float(med_in),
                                float(p95_in),
                                float(med_sv),
                                float(p95_sv),
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
                    dra_arcsec = (crval1_sv - crval1_in) * 3600.0
                    ddec_arcsec = (crval2_sv - crval2_in) * 3600.0
                    logger.info(
                        "WCS header delta:\n"
                        "\tCTYPE: (%s, %s) -> (%s, %s)\n"
                        "\tdCRPIX=(%g, %g) px\n"
                        "\tdCRVAL=(%g, %g) arcsec",
                        ctype1_in,
                        ctype2_in,
                        ctype1_sv,
                        ctype2_sv,
                        float(crpix1_sv - crpix1_in),
                        float(crpix2_sv - crpix2_in),
                        float(dra_arcsec),
                        float(ddec_arcsec),
                    )
                except Exception as exc:
                    log_warning_from_exception(
                        logger, "WCS input-vs-solved comparison skipped", exc
                    )

                if keep_input_wcs_full and not redo_requested:
                    try:
                        self.header["NAXIS1"] = self.image.shape[1]
                        self.header["NAXIS2"] = self.image.shape[0]
                        safe_fits_write(self.fpath, self.image, self.header, output_verify="ignore")
                        # Cleanup solve artifacts even when keeping input WCS.
                        for pattern in [".rdls", ".solved", ".xyls", ".axy", ".match", ".corr"]:
                            for f in glob.glob(os.path.join(dirname, f"*{pattern}")):
                                try:
                                    os.remove(f)
                                except OSError:
                                    pass
                        for f in [new_fits_temp, scamp_temp, os.path.join(dirname, "none"), wcs_file]:
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
                used_refined_wcs = False
                fit_sip_degree_int = None
                trust_input_distortion = bool(
                    wcs_cfg.get("trust_input_distortion_model", False)
                )
                if redo_requested and trust_input_distortion:
                    logger.info(
                        "redo_wcs=True: ignoring input distortion trust and solving for a fresh distortion model."
                    )
                    trust_input_distortion = False
                prefer_scamp_tpv = bool(use_tpv_flow)
                if prefer_scamp_tpv and not bool(
                    wcs_cfg.get("prefer_scamp_tpv_after_solve", True)
                ):
                    logger.info(
                        "projection_type=TPV overrides prefer_scamp_tpv_after_solve=False; using solve-field -> SCAMP."
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
                else:
                    logger.info(
                        "projection_type=SIP: skipping SCAMP refinement and using solve-field WCS."
                    )

                wcs_header = self._apply_post_scamp_linear_refinement(
                    wcs_header=wcs_header,
                    wcs_cfg=wcs_cfg,
                    matched_points=matched_points_for_refine,
                )
                from functions import remove_wcs_from_header

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
                # If we explicitly refit a new TAN-SIP solution from matched points,
                # do NOT preserve old distortion keywords from the input header.
                # This avoids mixing old PV/TPV terms with newly-fitted SIP terms.
                if used_refined_wcs:
                    preserve_distortion = False
                    logger.info(
                        "Accepted fit_wcs_from_points refinement (sip_degree=%d): not preserving old distortion keywords on redo_wcs",
                        int(fit_sip_degree_int or 0),
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
                            rd = tmp_wcs.all_pix2world([[crpix1]], [[crpix2]], 0)
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

                # Validate merged WCS before writing
                with silence_astropy_wcs_info():
                    merged_wcs = get_wcs(self.header)
                    try:
                        from astropy.wcs import utils as wcs_utils

                        scale_val = wcs_utils.proj_plane_pixel_scales(merged_wcs)[0]
                        scale = (
                            float(scale_val * 3600)
                            if hasattr(scale_val, "value")
                            else float(scale_val * 3600)
                        )  # arcsec/pix
                        if not np.isfinite(scale) or scale <= 0 or scale > 3600:
                            logger.warning(
                                "Merged WCS has invalid pixel scale (%.6f); rejecting",
                                scale,
                            )
                            return np.nan
                    except Exception as ev:
                        log_warning_from_exception(
                            logger, "Could not validate merged WCS", ev
                        )

                safe_fits_write(self.fpath, self.image, self.header,
                    output_verify="ignore",
                )
                # Final cleanup of heavy solve-field artifacts once diagnostics
                # and header updates are complete.
                if keep_match_for_compare:
                    for pattern in [".axy", ".match", ".corr"]:
                        for f in glob.glob(os.path.join(dirname, f"*{pattern}")):
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                if os.path.isfile(wcs_file):
                    os.remove(wcs_file)
                logger.info("WCS information updated in the FITS header")
                return self.header
            except Exception as e:
                for f in [
                    new_fits_temp,
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
                return np.nan
