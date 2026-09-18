"""Light-curve plotting and photometry table generation for AutoPHOT.

Provides functions to:

* Plot publication-ready light curves with detections and upper limits
  (:func:`plot_lightcurve`).
* Plot a reference-star differential variability check
  (:func:`plot_variability_check`).
* Generate structured photometry tables with MJD, magnitudes, errors, and
  optional colour terms (:func:`generate_photometry_table`).
* Sort detection plots into detection/non-detection folders
  (:func:`check_detection_plots`).

Band colour mappings, effective wavelengths, and canonical band-label
resolution helpers are also defined here.
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import glob
import shutil
from functions import set_size, get_distance_modulus
from plotting_utils import get_marker_size, apply_autophot_mplstyle
from astropy.time import Time
from collections import Counter
from pathlib import Path
from matplotlib.ticker import MaxNLocator

# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================

# Colour pairs: filter -> (b1, b2) for colour-term (zeropoint) calibration.
color_map = {
    "u": ("u", "g"),
    "g": ("g", "r"),
    "r": ("g", "r"),
    "i": ("r", "i"),
    "z": ("i", "z"),
    "J": ("J", "H"),
    "H": ("J", "H"),
    "K": ("H", "K"),
    "U": ("U", "B"),
    "B": ("B", "V"),
    "V": ("B", "V"),
    "R": ("V", "R"),
    "I": ("R", "I"),
}
# Pairs to plot in the colour panel (order and subset of color_map).
_COLOR_PAIRS = [("g", "r"), ("r", "i"), ("i", "z")]

# Effective (pivot) wavelengths in Angstroms for common photometric bands.
BAND_WAVELENGTHS = {
    "u": 3543,
    "g": 4770,
    "r": 6231,
    "i": 7625,
    "z": 9134,
    "U": 3600,
    "B": 4400,
    "V": 5500,
    "R": 6580,
    "I": 8060,
    "J": 12350,
    "H": 16620,
    "K": 21590,
    "w": 6579,
    "W": 6579,
}

# Per-band plot colours; keep in sync with the user-supplied palette.
cols = {
    "u": "dodgerblue",
    "g": "g",
    "r": "r",
    "i": "goldenrod",
    "z": "k",
    "y": "0.5",
    "w": "firebrick",
    "Y": "0.5",
    "U": "slateblue",
    "B": "b",
    "V": "yellowgreen",
    "R": "crimson",
    "I": "chocolate",
    "G": "salmon",
    "E": "salmon",
    "J": "darkred",
    "H": "orangered",
    "K": "saddlebrown",
    "S": "mediumorchid",
    "D": "purple",
    "A": "midnightblue",
    "F": "#8E4585",
    "N": "#CC79A7",
    "o": "darkorange",
    "c": "#17A2B8",
    "W": "forestgreen",
    "Q": "peru",
}

def _load_filter_colors_from_db() -> dict:
    """
    Load `filter_colors` from `databases/filters.yml` (project palette).

    Returns {} on any failure so we can safely fall back to defaults.
    """
    try:
        db_path = Path(__file__).resolve().parent / "databases" / "filters.yml"
        if not db_path.is_file():
            return {}
        import yaml  # type: ignore

        with open(db_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fc = data.get("filter_colors", {}) if isinstance(data, dict) else {}
        if not isinstance(fc, dict):
            return {}
        out = {}
        for k, v in fc.items():
            ks = str(k).strip()
            vs = str(v).strip()
            if ks and vs:
                out[ks] = vs
        return out
    except Exception:
        return {}


# Prefer the project palette from databases/filters.yml; fall back to built-ins above.
_db_cols = _load_filter_colors_from_db()
BAND_COLORS = {**cols, **_db_cols}

# Colour indices -> plotting colour (distinct, hue between constituent bands).
COLOR_INDEX_COLORS = {
    "u-g": "#6A0DAD",
    "g-r": "#20B2AA",
    "r-i": "#E55B3C",
    "i-z": "#A52A2A",
    "g-i": "#4682B4",
    "r-z": "#CD5C5C",
    "u-r": "#8A2BE2",
    "u-i": "#9370DB",
    "u-z": "#7B68EE",
    "U-B": "#9932CC",
    "B-V": "#3CB371",
    "V-R": "#FFD700",
    "V-I": "#FF8C00",
    "R-I": "#FF6347",
    "B-R": "#3B9AB2",
    "B-I": "#1E90FF",
    "V-J": "#BDB76B",
    "V-K": "#6B8E23",
    "J-H": "#D2691E",
    "H-K": "#8B8000",
    "J-K": "#BC8F8F",
    "i-J": "#C71585",
    "i-H": "#DB7093",
    "z-J": "#8B4513",
}


def _normalize_photometry_columns(df):
    """
    Normalise *non-band* photometry table column names to lowercase.

    IMPORTANT: do NOT lowercase band-specific columns like `r_PSF` vs `R_PSF` since
    that collapses distinct filters (e.g. SDSS r vs Cousins R) onto the same name.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    norm_cols = []
    for c in df.columns:
        s = str(c).strip()
        # Preserve band columns and their ZP columns with original case.
        # Examples to preserve: "r_PSF", "R_PSF", "zp_r_PSF", "zp_R_PSF".
        if (
            len(s) >= 3
            and s[0].isalpha()
            and s[1] == "_"
            and s.split("_", 1)[1].upper() in {"PSF", "AP", "PSF_ERR", "AP_ERR"}
        ):
            norm_cols.append(s)
        elif (
            s.lower().startswith("zp_")
            and len(s) >= 6
            and s[3].isalpha()
            and s[4] == "_"
        ):
            norm_cols.append(s)
        else:
            norm_cols.append(s.lower())
    df.columns = norm_cols
    return df


def filter_value_matches_band(band_char: str, raw_filter) -> bool:
    """
    True if a photometry-table ``filter`` cell corresponds to band ``band_char``.

    Maps survey-style names (e.g. ``gp`` -> ``g``) using ``main._heuristic_filter_mapping``
    so long-form CSV rows match the band order used in plots.
    """
    if raw_filter is None or pd.isna(raw_filter):
        return False
    bc = str(band_char).strip().lower()
    if not bc:
        return False
    rs = str(raw_filter).strip()
    if rs.lower() == bc:
        return True
    try:
        from main import _heuristic_filter_mapping

        canon = str(_heuristic_filter_mapping(rs)).strip().lower()
        return canon == bc
    except Exception:
        return False


def _filter_series_matches_band(band_char: str, fser: pd.Series) -> pd.Series:
    """Vectorized-friendly version of filter_value_matches_band for a Series.

    Pre-computes the canonical mapping for *unique* filter values only (typically
    1-5) instead of calling _heuristic_filter_mapping per row (hundreds+).
    Returns a boolean Series aligned to *fser*.
    """
    bc = str(band_char).strip().lower()
    if not bc:
        return pd.Series(False, index=fser.index)

    # Fast path: direct string match (no import needed)
    rs_lower = fser.astype(str).str.strip().str.lower()
    direct_match = rs_lower == bc

    # Slow path: canonical form only for rows that missed the direct match.
    # Pre-compute per unique value to minimize _heuristic_filter_mapping calls.
    needs_canon = ~direct_match & fser.notna()
    if not needs_canon.any():
        return direct_match.fillna(False)

    try:
        from main import _heuristic_filter_mapping
    except Exception:
        return direct_match.fillna(False)

    unique_vals = fser[needs_canon].unique()
    canon_map = {}
    for uv in unique_vals:
        try:
            canon_map[uv] = str(_heuristic_filter_mapping(str(uv).strip())).strip().lower()
        except Exception:
            canon_map[uv] = ""

    canon_match = fser[needs_canon].map(canon_map).eq(bc)
    result = direct_match.copy()
    result[needs_canon] = canon_match
    return result.fillna(False)


def photometry_filter_series(df: pd.DataFrame):
    """Return the per-row filter column if present (``filter``, ``imagefilter``, ...)."""
    if df is None or df.empty:
        return None
    preferred = ("filter", "imagefilter", "band", "image_filter")
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_map:
            return df[lower_map[name]]
    return None


def canonical_bands_from_filter_series(fser: pd.Series):
    """
    Unique canonical band codes present in a filter column, sorted by pivot wavelength.

    Used for long-form CSVs (shared ``mag_psf`` / ``zp_psf``) so plots iterate real
    bands (e.g. ``w``) instead of the full bandlist (which would mis-label as ``F``
    if the filter column was not detected).
    """
    raw_unique = [
        str(x).strip() for x in fser.dropna().unique() if str(x).strip()
    ]
    canon_list = []
    seen = set()
    for raw in raw_unique:
        try:
            from main import _heuristic_filter_mapping

            c = str(_heuristic_filter_mapping(raw)).strip()
        except Exception:
            c = raw
        ck = str(c).lower()
        if ck not in seen:
            seen.add(ck)
            canon_list.append(c)
    return sorted(
        canon_list,
        key=lambda bb: BAND_WAVELENGTHS.get(
            bb, BAND_WAVELENGTHS.get(str(bb).lower(), 1e12)
        ),
    )


def canonical_band_label_map_from_filter_series(fser: pd.Series) -> dict[str, str]:
    """
    Map canonical band -> preferred display label taken from the `filter` column.

    This keeps legend labels faithful to the input data (e.g. if the CSV filter
    column contains "wp", the plotted label is "wp"), while still grouping rows
    by canonical band (e.g. wp->w) for long-form photometry tables.
    """
    if fser is None or len(fser) == 0:
        return {}
    raw = fser.dropna().astype(str).map(str.strip)
    raw = raw[raw.astype(bool)]
    if raw.empty:
        return {}

    def _canon(s: str) -> str:
        try:
            from main import _heuristic_filter_mapping

            return str(_heuristic_filter_mapping(s)).strip().lower()
        except Exception:
            return str(s).strip().lower()

    tmp = pd.DataFrame({"raw": raw})
    tmp["canon"] = tmp["raw"].map(_canon)
    # Most common raw label per canonical band; ties broken by first occurrence.
    out: dict[str, str] = {}
    for canon, g in tmp.groupby("canon", sort=False):
        counts = g["raw"].value_counts()
        if counts.empty:
            continue
        top_n = int(counts.iloc[0])
        top_vals = set(counts[counts == top_n].index.tolist())
        chosen = next((v for v in g["raw"].tolist() if v in top_vals), g["raw"].iloc[0])
        out[str(canon).strip().lower()] = str(chosen).strip()
    return out


def _inverted_fit_series_to_bool(series: pd.Series) -> pd.Series:
    """Parse ``_inverted_fit`` column to real booleans (CSV / mixed types safe).

    Do not use ``Series.astype(bool)`` on object columns: non-empty strings
    including the literal ``\"False\"`` would become True.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=bool)
    out = np.zeros(len(series), dtype=bool)
    idx = series.index
    for i, v in enumerate(series.to_numpy()):
        if pd.isna(v):
            continue
        if isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
        elif isinstance(v, (np.integer, int)):
            out[i] = v != 0
        elif isinstance(v, (np.floating, float)):
            fv = float(v)
            out[i] = np.isfinite(fv) and fv != 0.0
        elif isinstance(v, str):
            t = v.strip().lower()
            out[i] = t in ("1", "true", "t", "yes", "y")
        else:
            out[i] = bool(v)
    return pd.Series(out, index=idx)


def _resolve_band_triplet(cols, band: str, method: str):
    """
    Resolve the (mag, err, zp) column names for a band/method.
    Supports both exact-case columns (e.g. g_PSF) and legacy lowercase (e.g. g_psf)
    without conflating distinct filters that differ by case (e.g. r vs R).
    """
    # Prefer exact-case first, then lowercase legacy.
    m_low = str(method).strip().lower()
    cand = [
        # New uniform schema (preferred)
        (f"mag_{m_low}", f"mag_{m_low}_err", f"zp_{band}_{method}"),
        (f"mag_{m_low}", f"mag_{m_low}_err", f"zp_{m_low}"),
        (f"{band}_{method}", f"{band}_{method}_err", f"zp_{band}_{method}"),
        (
            f"{band}_{method}".lower(),
            f"{band}_{method}_err".lower(),
            f"zp_{band}_{method}".lower(),
        ),
    ]
    for trip in cand:
        if all(c in cols for c in trip):
            return trip
    return None


def _lmag_to_apparent(df: pd.DataFrame, zp_col: str) -> pd.Series:
    """Convert pipeline limiting mag (instrumental) to apparent using band zeropoint.
    Returns NaN when limiting magnitudes are invalid."""
    col = (
        "limiting_inst_mag"
        if "limiting_inst_mag" in df.columns
        else ("lmag" if "lmag" in df.columns else None)
    )
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    
    li = pd.to_numeric(df[col], errors="coerce")
    zp = pd.to_numeric(df[zp_col], errors="coerce")

    limiting_mag = li + zp

    # Mask non-finite and implausible limiting mags (<15 or >30).
    invalid_mask = (~np.isfinite(limiting_mag)) | (limiting_mag < 15) | (limiting_mag > 30)
    limiting_mag.loc[invalid_mask] = np.nan

    return limiting_mag


def _lmag_to_apparent_multi_snr(df: pd.DataFrame, zp_col: str, snr_thresholds: list = None, input_yaml: dict = None) -> dict:
    """Convert limiting magnitudes for multiple S/N thresholds to apparent magnitudes.

    Returns a dictionary with keys like 'Limit_3p0S2N', 'Limit_5p0S2N' containing apparent magnitudes.
    Note: Decimal points in column names are replaced with 'p' for compatibility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing limiting magnitude data
    zp_col : str
        Column name for zeropoint values
    snr_thresholds : list, optional
        List of S/N thresholds (e.g., [3.0, 5.0]). If None, reads from input_yaml config.
    input_yaml : dict, optional
        Configuration dictionary. If provided and snr_thresholds is None, 
        reads snr_thresholds from limiting_magnitude.snr_thresholds config.
    
    Returns:
    --------
    dict
        Dictionary with keys like 'Limit_3p0S2N', 'Limit_5p0S2N' containing apparent magnitudes.
    """
    if snr_thresholds is None:
        if input_yaml is not None:
            lim_cfg = input_yaml.get("limiting_magnitude") or {}
            snr_thresholds = lim_cfg.get("snr_thresholds", [3.0, 5.0])
        else:
            snr_thresholds = [3.0, 5.0]
    
    if not isinstance(snr_thresholds, list):
        snr_thresholds = [snr_thresholds]

    result = {}

    # Multi-S/N limiting magnitudes only (no general Limit column)
    for snr in snr_thresholds:
        # main.py writes apparent columns as limiting_mag_{snr:.0f}s2n; the
        # instrumental/legacy names below are fallbacks for older tables.
        apparent_col = f"limiting_mag_{snr:.0f}s2n"
        inst_col = f"limiting_inst_mag_snr_{snr}"
        fallback_col = f"lmag_snr_{snr}"

        col = None
        if apparent_col in df.columns:
            # Already apparent magnitude (from main.py)
            col = apparent_col
            is_apparent = True
        elif inst_col in df.columns:
            # Instrumental magnitude, needs conversion
            col = inst_col
            is_apparent = False
        elif fallback_col in df.columns:
            # Legacy apparent magnitude column
            col = fallback_col
            is_apparent = True
        elif "limiting_inst_mag" in df.columns:
            # Fallback to single limiting magnitude if S/N-specific not available
            col = "limiting_inst_mag"
            is_apparent = False
        elif "lmag" in df.columns:
            col = "lmag"
            is_apparent = True

        # 'p' stands in for the decimal point in output column names
        col_name = f'Limit_{snr:.1f}S2N'.replace('.', 'p')
        
        if col is None:
            result[col_name] = pd.Series(np.nan, index=df.index, dtype=float)
            continue

        li = pd.to_numeric(df[col], errors="coerce")
        zp = pd.to_numeric(df[zp_col], errors="coerce")

        if is_apparent:
            limiting_mag = li
        else:
            limiting_mag = li + zp

        # Mask non-finite and implausible limiting mags (<15 or >30).
        invalid_mask = (~np.isfinite(limiting_mag)) | (limiting_mag < 15) | (limiting_mag > 30)
        limiting_mag.loc[invalid_mask] = np.nan

        result[col_name] = limiting_mag.round(3)

    return result


def _format_reference_datetime(mjd) -> str:
    """Format an MJD as e.g. ``'9th August 9:00pm'`` (UTC, ordinal day)."""
    dt = Time(float(mjd), format="mjd", scale="utc").to_datetime()
    day = dt.day
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    hour12 = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{day}{suffix} {dt.strftime('%B')} {hour12}:{dt.minute:02d}{ampm}"


def _time_axis_transform(mjd_values, reference_epoch=0):
    """Decide the lightcurve x-axis transform for the given MJD values.

    Returns ``(transform, xlabel, unit)`` where ``transform`` maps an array of
    MJDs to x-axis coordinates.  When the data span less than one day the axis
    is switched to minutes (span < 3 hours) or hours since the first
    observation, labelled ``'Time since <date> UTC [<unit>]'``; otherwise the
    axis is phase (days since ``reference_epoch``) or raw MJD.  ``unit`` is
    ``'min'``, ``'hr'`` or ``None``.
    """
    mjd_arr = np.asarray(
        pd.to_numeric(pd.Series(np.ravel(np.asarray(mjd_values))), errors="coerce"),
        dtype=float,
    )
    mjd_arr = mjd_arr[np.isfinite(mjd_arr)]

    def _delta(ref):
        return lambda m: np.asarray(m, dtype=float) - ref

    if mjd_arr.size > 1:
        t0 = float(mjd_arr.min())
        span_days = float(mjd_arr.max() - t0)
        if 0.0 < span_days < 1.0:
            if span_days * 24.0 < 3.0:
                factor, unit = 1440.0, "min"
            else:
                factor, unit = 24.0, "hr"
            xlabel = (
                f"Time since {_format_reference_datetime(t0)} UTC [{unit}]"
            )
            return (
                lambda m: (np.asarray(m, dtype=float) - t0) * factor,
                xlabel,
                unit,
            )

    if reference_epoch:
        return (
            _delta(reference_epoch),
            f"Phase (days since {reference_epoch})",
            None,
        )
    return _delta(0.0), "Time [MJD]", None


def _compute_detection_mask(
    df: pd.DataFrame,
    mag_col: str,
    err_col: str,
    method: str,
    *,
    snr_limit: float,
    use_SNR_limit: bool,
) -> np.ndarray:
    """
    Compute detection mask used consistently across lightcurve products.

    Detection rule:
    - Prefer an explicit is_detection/detected flag when present.
    - Otherwise: SNR cut (if use_SNR_limit) or mag < lmag (otherwise).
    """
    # Prefer explicit detection flags from upstream (e.g. main.py), falling
    # back to SNR/magnitude-based inference when absent. This keeps downstream
    # classification in sync with the detection decision made by the pipeline.
    for flag_col in ("is_detection", "detected", "is_detected"):
        if flag_col in df.columns:
            flags = df[flag_col]
            try:
                is_bool_like = flags.dtype == bool or flags.isin(
                    [0, 1, True, False, np.nan]
                ).all()
            except Exception:
                is_bool_like = False
            if is_bool_like:
                try:
                    return np.asarray(flags.fillna(False).astype(bool), dtype=bool)
                except Exception:
                    pass

    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float, copy=False)
    err = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float, copy=False)
    lim_col = (
        "limiting_inst_mag"
        if "limiting_inst_mag" in df.columns
        else ("lmag" if "lmag" in df.columns else None)
    )
    lmag = (
        pd.to_numeric(df[lim_col], errors="coerce").to_numpy(dtype=float, copy=False)
        if lim_col is not None
        else np.full(len(df), np.nan, dtype=float)
    )

    def _snr_from_magerr(mag_err: np.ndarray) -> np.ndarray:
        # For magnitudes m = -2.5 log10(F) + const, error propagation gives:
        # sigma_m ~= 1.0857 / SNR  ->  SNR ~= 1.0857 / sigma_m
        c = 2.5 / np.log(10.0)
        mag_err = np.asarray(mag_err, dtype=float)
        return np.divide(
            c,
            mag_err,
            out=np.full_like(mag_err, np.nan, dtype=float),
            where=(mag_err > 0) & np.isfinite(mag_err),
        )

    method_u = str(method).upper()
    snr_source = "unknown"
    col_map = {c.lower(): c for c in df.columns}
    snr_psf_col = col_map.get("snr_psf")
    snr_ap_col = col_map.get("snr_ap")
    snr_col = col_map.get("snr")
    
    if use_SNR_limit:
        # Priority 1: Use explicit SNR columns if available
        if method_u == "PSF" and snr_psf_col:
            snr = pd.to_numeric(df[snr_psf_col], errors="coerce").to_numpy(dtype=float, copy=False)
            snr_source = "snr_psf"
        elif method_u == "AP" and snr_ap_col:
            snr = pd.to_numeric(df[snr_ap_col], errors="coerce").to_numpy(dtype=float, copy=False)
            snr_source = "snr_ap"
        elif snr_col:
            snr = pd.to_numeric(df[snr_col], errors="coerce").to_numpy(dtype=float, copy=False)
            snr_source = "snr"
        # Priority 2: Compute SNR from flux columns (avoids systematic errors in mag_err)
        elif method_u == "PSF" and col_map.get("flux_psf") and col_map.get("flux_psf_err"):
            flux = pd.to_numeric(df[col_map.get("flux_psf")], errors="coerce").to_numpy(dtype=float, copy=False)
            flux_err = pd.to_numeric(df[col_map.get("flux_psf_err")], errors="coerce").to_numpy(dtype=float, copy=False)
            snr = np.divide(
                np.abs(flux),  # Use absolute flux to handle negative PSF fits
                flux_err,
                out=np.full(len(df), np.nan, dtype=float),
                where=(flux_err > 0) & np.isfinite(flux_err),
            )
            snr_source = "flux_psf"
        elif method_u == "AP" and col_map.get("flux_ap") and col_map.get("flux_ap_err"):
            flux = pd.to_numeric(df[col_map.get("flux_ap")], errors="coerce").to_numpy(dtype=float, copy=False)
            flux_err = pd.to_numeric(df[col_map.get("flux_ap_err")], errors="coerce").to_numpy(dtype=float, copy=False)
            snr = np.divide(
                np.abs(flux),  # Use absolute flux for consistency
                flux_err,
                out=np.full(len(df), np.nan, dtype=float),
                where=(flux_err > 0) & np.isfinite(flux_err),
            )
            snr_source = "flux_ap"
        # Priority 3: Last resort - infer SNR from magnitude uncertainty (includes systematic errors)
        else:
            snr = _snr_from_magerr(err)
            snr_source = "mag_err"
        if len(snr) > 0 and not np.all(np.isnan(snr)):
            logging.debug("_compute_detection_mask: method=%s, snr_source=%s, snr_mean=%.3f, snr_max=%.3f", method, snr_source, np.nanmean(snr), np.nanmax(snr))
    else:
        # Still compute SNR for completeness, but detection uses lmag branch below.
        snr = _snr_from_magerr(err)

    if use_SNR_limit:
        detected = (
            np.isfinite(mag)
            & np.isfinite(err)
            & np.isfinite(snr)
            & (snr >= float(snr_limit))
        )
        if len(snr) > 0 and len(detected) > 0:
            logging.info("_compute_detection_mask: method=%s, snr_source=%s, snr=%.3f, limit=%s, detected=%s", method, snr_source, snr[0], snr_limit, detected[0])
    else:
        # Prefer a magnitude-vs-lmag comparison when a usable limiting mag
        # exists; otherwise fall back to SNR inferred from magnitude errors.
        MIN_REASONABLE_LIMITING_MAG = 10.0  # lmag below 10 is implausible here
        valid_lmag = np.isfinite(lmag) & (lmag > MIN_REASONABLE_LIMITING_MAG)
        if np.any(valid_lmag):
            detected = np.isfinite(mag) & np.isfinite(err) & (mag < lmag)
        else:
            snr_from_err = _snr_from_magerr(err)
            detected = (
                np.isfinite(mag)
                & np.isfinite(err)
                & np.isfinite(snr_from_err)
                & (snr_from_err >= float(snr_limit))
            )

    return np.asarray(detected, dtype=bool)


def plot_lightcurve(
    output_file,
    snr_limit=3,
    beta_limit=0.5,
    fwhm=3,
    method="PSF",
    reference_epoch=0,
    offset=0,
    redshift=0,
    show_limits=True,
    show_details=False,
    size=None,
    return_detections=True,
    format="png",
    show=False,
    single_plot=True,
    use_SNR_limit=True,
    mark_today=False,
    target_name=None,
    dpi=150,
    plot_color=False,
    color_match_days=0.5,
    ls="",
    max_plot_err=0.5,
    chi2_marginal_threshold=1000.0,
):
    """Plot a publication-ready lightcurve with detections and limits.

    Produces a scientifically formatted plot with clear axis labels (with units),
    readable fonts, optional title, legend for filters and limit symbols,
    and optional absolute-magnitude axis. Default figure size uses journal-friendly
    single-column width (set_size). Use format='pdf' for vector output.

    Parameters
    ----------
    output_file : str
        Path to the photometry CSV.
    snr_limit, beta_limit : float
        Detection thresholds.
    method : str
        Photometry method (e.g. 'PSF', 'AP').
    reference_epoch : float
        MJD reference for phase; 0 means x-axis is MJD.  When the plotted
        data span less than one day the axis instead switches to minutes
        (span < 3 h) or hours since the first observation, labelled
        'Time since <date> UTC [<unit>]'.
    offset : float
        Magnitude offset per band for stacking (visual).
    redshift : float
        If set, show absolute magnitude on twin y-axis.
    show_limits : bool
        Plot upper limits (downward triangles).
    show_details : bool
        Show detection/non-detection counts in corner.
    size : tuple or None
        (width_inch, height_inch). If None, uses set_size(505, aspect=0.6) for single.
    return_detections : bool
        If True, return path to detections CSV.
    format : str
        Output format: 'png', 'pdf', etc.
    show : bool
        If True, call plt.show().
    single_plot : bool
        One panel (all bands) vs one panel per band.
    use_SNR_limit : bool
        If True, use SNR >= snr_limit for detection; else use mag < limit.
    mark_today : bool
        If True, plot vertical line at current MJD.
    target_name : str or None
        Optional title (e.g. object name) for the plot.
    dpi : int
        DPI for raster formats (ignored for pdf).
    plot_color : bool
        If True, plot colour evolution (e.g. g-r, r-i) below the lightcurve using
        same-night pairs only (within color_match_days).
    color_match_days : float
        Max separation in days for pairing two filters as "same night" when
        computing colours (default 0.5).
    ls : str
        Line style for connecting detection points (default "" for no line).
        Set to "-" for a solid line connecting points.
    max_plot_err : float
        Maximum magnitude error for a detection to be plotted (default 0.5 mag).
        Detections with errors exceeding this threshold are excluded from the
        plot and a warning is logged. This prevents poorly constrained
        measurements from dominating the plot scale or obscuring real trends.
    chi2_marginal_threshold : float
        Reduced chi-squared threshold above which a detection is plotted as
        "marginal".  Detections with ``reduced_chi2`` exceeding this value are
        drawn with a white face and faded edge colour to visually flag poor
        PSF-fit quality (e.g. non-Gaussian PSF, sparse field).  Set to ``0``
        or ``None`` to disable.

    Returns
    -------
    str or None
        Path to detections CSV if return_detections and detections exist; else None.
    """
    # Switch to an interactive backend before any pyplot figure creation
    # so that plt.show() actually displays the window when show=True.
    # plt.switch_backend() works after pyplot is already imported, unlike
    # matplotlib.use() which silently fails once pyplot is loaded.
    apply_autophot_mplstyle()
    if show:
        import matplotlib

        current_backend = str(plt.get_backend()).lower()
        if "agg" in current_backend:
            for backend in ("QtAgg", "TkAgg"):
                try:
                    matplotlib.use(backend, force=True)
                    break
                except Exception:
                    continue

    today_mjd = None
    if mark_today:
        today = Time.now()
        today_mjd = today.mjd

    dm = get_distance_modulus(redshift) if redshift else 0
    base_cols = BAND_COLORS
    # Band plotting order (exclude Gaia G so it is not conflated with SDSS g).
    band_order = "FSDNAuUBgcVwrRoEiIzyYJHKWQ"
    cols = {b: base_cols.get(b, BAND_COLORS.get(b, "k")) for b in band_order}
    # Unknown filters: assign distinct generic colors from Matplotlib's cycle.
    _generic_cycle = (
        plt.rcParams.get("axes.prop_cycle", None).by_key().get("color", [])
        if plt.rcParams.get("axes.prop_cycle", None) is not None
        else []
    )
    if not _generic_cycle:
        _generic_cycle = ["k"]
    _unknown_color_map: dict[str, str] = {}

    def _color_for_band(band_label: str) -> str:
        if band_label in base_cols:
            return base_cols[band_label]
        bl = str(band_label).strip().lower()
        if bl in base_cols:
            return base_cols[bl]
        if bl not in _unknown_color_map:
            _unknown_color_map[bl] = _generic_cycle[
                len(_unknown_color_map) % len(_generic_cycle)
            ]
        return _unknown_color_map[bl]
    data = pd.read_csv(output_file)
    data = _normalize_photometry_columns(data)
    # If the CSV has duplicate column names (can happen after concatenation or
    # normalization), pandas returns a DataFrame for `df["col"]`, which breaks
    # numeric coercion and boolean logic. Keep the first occurrence.
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()].copy()
    save_path = os.path.dirname(output_file)
    base = os.path.splitext(os.path.basename(output_file))[0]

    # Decide the x-axis mapping up front. For intra-night data (< 1 day span)
    # this switches the axis to minutes/hours since the first observation with
    # a 'Time since <date>' label instead of raw MJD.
    x_transform, xlabel, subday_unit = _time_axis_transform(
        data["mjd"].values if "mjd" in data.columns else np.array([]),
        reference_epoch,
    )

    def _resolve_band_triplet(cols, band: str, method: str):
        """
        Resolve the (mag, err, zp) column names for a band/method.
        Supports both legacy lowercased columns (e.g. g_psf) and case-preserving
        columns (e.g. g_PSF / R_PSF) without conflating r and R when both exist.
        Also supports inverted counterparts (e.g. g_psf_inverted).
        """
        # Prefer exact-case first, then lowercase legacy.
        m_low = str(method).strip().lower()
        cand = [
            # New uniform schema (preferred)
            (f"mag_{m_low}", f"mag_{m_low}_err", f"zp_{band}_{method}"),
            (f"mag_{m_low}", f"mag_{m_low}_err", f"zp_{m_low}"),
            (f"{band}_{method}", f"{band}_{method}_err", f"zp_{band}_{method}"),
            (
                f"{band}_{method}".lower(),
                f"{band}_{method}_err".lower(),
                f"zp_{band}_{method}".lower(),
            ),
        ]
        # Inverted-image counterparts, used only if no normal triplet matches.
        cand_inv = [
            (f"{band}_{method}_inverted", f"{band}_{method}_err_inverted", f"zp_{band}_{method}"),
            (
                f"{band}_{method}_inverted".lower(),
                f"{band}_{method}_err_inverted".lower(),
                f"zp_{band}_{method}".lower(),
            ),
        ]
        for trip in cand:
            if all(c in cols for c in trip):
                return trip
        for trip in cand_inv:
            if all(c in cols for c in trip):
                return trip
        return None

    # Discover which bands are actually present. For long-form CSVs (shared
    # mag_psf / zp_psf + per-row ``filter``), every band letter would otherwise
    # resolve to the same triplet and only the first letter in band_order (``F``)
    # would be used - wrong for e.g. Pan-STARRS/ZTF ``w``.
    m_low_plot = str(method).strip().lower()
    long_form_uniform = (
        f"mag_{m_low_plot}" in data.columns
        and f"mag_{m_low_plot}_err" in data.columns
        and f"zp_{m_low_plot}" in data.columns
    )
    fser_disc = photometry_filter_series(data)
    bands_in_data = []
    used_triplets = set()
    if long_form_uniform and fser_disc is not None and fser_disc.notna().any():
        bands_in_data = canonical_bands_from_filter_series(fser_disc)
    if not bands_in_data:
        for b in band_order:
            triplet = _resolve_band_triplet(set(data.columns), b, method)
            if triplet is not None and triplet not in used_triplets:
                bands_in_data.append(b)
                used_triplets.add(triplet)
    if not bands_in_data:
        logging.getLogger(__name__).info(
            "No valid photometric bands found in '%s' for method '%s'.",
            output_file,
            method,
        )
        return None

    # No color plot with a single image (no same-night pairs possible)
    if plot_color and len(data) <= 1:
        plot_color = False

    # Publication-style sizing via set_size (journal single-column width).
    if size is None:
        if single_plot:
            figsize = set_size(540, aspect=1)
        else:
            figsize = (
                set_size(540)[0],
                set_size(505, aspect=1)[1] * len(bands_in_data),
            )
    else:
        figsize = size

    n_curve = 1 if single_plot else len(bands_in_data)
    n_total = n_curve + (1 if plot_color else 0)
    if plot_color and size is None:
        figsize = (figsize[0], figsize[1] * (1 + 0.45))

    if n_total == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = np.array([ax])
    else:
        from matplotlib.gridspec import GridSpec

        ratios = [1.5] * n_curve + ([1] if plot_color else [])
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_total, 1, figure=fig, height_ratios=ratios, hspace=0.08)
        axes = np.array([fig.add_subplot(gs[i]) for i in range(n_total)])
        for i in range(1, n_total):
            axes[i].sharex(axes[0])
    curve_axes = axes[:n_curve]
    color_ax = axes[-1] if plot_color else None

    # Tick styling suited to a single-column journal figure.
    for ax in curve_axes:
        ax.tick_params(axis="both", which="major")
        ax.tick_params(axis="both", which="minor", length=2.5)

    num_detect, num_nondetect = 0, 0
    detections_list = [] if return_detections else None
    nondetections_list = [] if return_detections else None
    bands_in_data = bands_in_data[::-1]
    has_limits_plotted = False
    plotted_inverted_hatch = False
    plotted_positive_flux_marker = False
    plotted_marginal_chi2_marker = False
    mid_idx = (
        len(bands_in_data) - 1
    ) // 2  # reference band (offset 0) is the middle one

    for idx, band in enumerate(bands_in_data):
        ax = curve_axes[0] if single_plot else curve_axes[idx]
        triplet = _resolve_band_triplet(set(data.columns), band, method)
        if triplet is None:
            continue
        col, err_col, zp_col = triplet
        # For new long-form output tables (shared mag_psf/zp_psf + per-row `filter`),
        # we must subset rows by the filter column. Otherwise each band loop plots
        # the full table and the last-drawn band's color dominates (appears "all red").
        if long_form_uniform and fser_disc is not None and fser_disc.notna().any():
            try:
                m = _filter_series_matches_band(band, fser_disc)
                df = data[m & np.isfinite(data[zp_col])].copy()
            except Exception:
                df = data[np.isfinite(data[zp_col])].copy()
        else:
            df = data[np.isfinite(data[zp_col])].copy()
        if df.empty:
            continue
        df.sort_values(by="mjd", inplace=True)
        # 'inst_*' columns are instrumental and need the zeropoint added;
        # mag_{method} columns (e.g. mag_psf) from main.py are already
        # calibrated apparent magnitudes.
        if col.startswith('inst_'):
            df["apparent_mag"] = df[col] + df[zp_col]
        else:
            df["apparent_mag"] = df[col]
        df["apparent_mag_err"] = df[err_col]

        band_offset = (idx - mid_idx) * offset
        df["apparent_mag"] = df["apparent_mag"] + band_offset
        # limiting_inst_mag in the photometry CSV is instrumental (see
        # main.py); convert to apparent so upper-limit points share the
        # detection magnitude scale. band_offset is NOT applied to lmag: it is
        # a detection threshold, not a plotted position.
        df["lmag"] = _lmag_to_apparent(df, zp_col)

        # ZTF-style SNU upper limit: prefer the 5sigma limit, else the 50%
        # completeness limit (matches the ZTF forced-photometry service).
        upper_limit_snr = 5.0  # ZTF SNU default
        upper_limit_col = f"limiting_mag_{upper_limit_snr:.0f}s2n"
        if upper_limit_col in df.columns:
            # main.py outputs apparent magnitude columns directly
            df["lmag_upper"] = pd.to_numeric(df[upper_limit_col], errors="coerce")
            # Use 5sigma limit where valid, otherwise fall back to 50% completeness limit
            valid_upper = np.isfinite(df["lmag_upper"]) & (df["lmag_upper"] > 10) & (df["lmag_upper"] < 30)
            df.loc[~valid_upper, "lmag_upper"] = df.loc[~valid_upper, "lmag"]
        else:
            df["lmag_upper"] = df["lmag"]

        df["plot_mag"] = df["apparent_mag"]
        df["plot_err"] = df["apparent_mag_err"]

        detected = _compute_detection_mask(
            df,
            "apparent_mag",
            "apparent_mag_err",
            method,
            snr_limit=float(snr_limit),
            use_SNR_limit=bool(use_SNR_limit),
        )
        detected_s = pd.Series(detected, index=df.index, dtype=bool)
        logging.info("plot_lightcurve: detected=%s", detected_s.iloc[0] if len(detected_s) > 0 else 'N/A')

        # Inverted-only detections come from the _inverted_fit flag or a
        # finite inst_inverted column.
        has_inverted = False
        inverted_col = None

        if "_inverted_fit" in df.columns:
            inv_flag = _inverted_fit_series_to_bool(df["_inverted_fit"])
            has_inverted = bool(inv_flag.any())
        else:
            inv_flag = pd.Series(False, index=df.index)

        if not has_inverted:
            inverted_col = "inst_inverted" if "inst_inverted" in df.columns else None
            has_inverted = inverted_col is not None and np.any(
                np.isfinite(pd.to_numeric(df[inverted_col], errors="coerce"))
            )

        if has_inverted:
            if "_inverted_fit" not in df.columns:
                inv_flag = pd.Series(False, index=df.index)
            if inverted_col is None and "inst_inverted" in df.columns:
                inverted_col = "inst_inverted"
            if inverted_col and inverted_col in df.columns:
                inv_finite = np.isfinite(
                    pd.to_numeric(df[inverted_col], errors="coerce").to_numpy(dtype=float)
                )
            else:
                inv_finite = np.zeros(len(df), dtype=bool)
            inv_finite = pd.Series(inv_finite, index=df.index)
            # Rows recovered on the inverted image: pipeline flag, or finite inverted mag
            # without a normal detection (oversubtraction / transient in template).
            inverted_only = inv_flag | (inv_finite & ~detected_s)
            normal_detected = detected_s & ~inverted_only
        else:
            inverted_only = pd.Series(False, index=df.index)
            normal_detected = detected_s

        # ------------------------------------------------------------------
        # Assemble detections and non-detections for this band
        # ------------------------------------------------------------------
        all_detects = df[normal_detected | inverted_only]
        nondetects = df[~detected_s & ~inverted_only]

        num_detect += len(all_detects)
        num_nondetect += len(nondetects)

        if return_detections and not all_detects.empty:
            detections_list.append(all_detects)
        if return_detections and not nondetects.empty:
            nondetections_list.append(nondetects)

        # Band colour and legend label (label carries the magnitude offset).
        c = _color_for_band(band)
        if offset != 0:
            leg_label = (
                band
                if band_offset == 0
                else (
                    f"{band}{band_offset:+.0f}"
                    if band_offset == int(band_offset)
                    else f"{band}{band_offset:+.1f}"
                )
            )
        else:
            leg_label = band

        # ------------------------------------------------------------------
        # Assign plot_mag / plot_err for each detection category
        # ------------------------------------------------------------------
        # Normal detections: use the calibrated apparent magnitude directly.
        df.loc[normal_detected, "plot_mag"] = df.loc[normal_detected, "apparent_mag"]
        df.loc[normal_detected, "plot_err"] = df.loc[normal_detected, "apparent_mag_err"]

        # Inverted-only detections: recovered on the inverted difference image.
        # These need their own magnitude / error columns, which may be
        # band-specific (apparent) or generic instrumental (needs ZP).
        if has_inverted and np.any(inverted_only):
            band_inv_mag_col = f"{band}_{method}_inverted" if f"{band}_{method}_inverted" in df.columns else None
            band_inv_err_col = f"{band}_{method}_err_inverted" if f"{band}_{method}_err_inverted" in df.columns else None

            inv_mag_col = band_inv_mag_col if band_inv_mag_col else ("inst_inverted" if "inst_inverted" in df.columns else inverted_col)
            inv_err_col = band_inv_err_col if band_inv_err_col else ("inst_inverted_err" if "inst_inverted_err" in df.columns else None)

            if inv_mag_col and inv_mag_col in df.columns:
                is_apparent = band_inv_mag_col is not None
                if not is_apparent and zp_col in df.columns:
                    # Instrumental: convert to apparent by adding the zeropoint.
                    df.loc[inverted_only, "inv_apparent_mag"] = df.loc[inverted_only, inv_mag_col] + df.loc[inverted_only, zp_col]
                    if inv_err_col and inv_err_col in df.columns:
                        df.loc[inverted_only, "inv_apparent_mag_err"] = df.loc[inverted_only, inv_err_col]
                    else:
                        df.loc[inverted_only, "inv_apparent_mag_err"] = np.nan
                    df.loc[inverted_only, "plot_mag"] = df.loc[inverted_only, "inv_apparent_mag"] + band_offset
                    df.loc[inverted_only, "plot_err"] = df.loc[inverted_only, "inv_apparent_mag_err"]
                else:
                    # Already apparent magnitude (band-specific inverted column).
                    df.loc[inverted_only, "plot_mag"] = df.loc[inverted_only, inv_mag_col] + band_offset
                    df.loc[inverted_only, "plot_err"] = df.loc[inverted_only, inv_err_col]

        # ------------------------------------------------------------------
        # Filter out poorly constrained detections (large error bars)
        # ------------------------------------------------------------------
        # Detections with magnitude errors exceeding max_plot_err are excluded
        # from the plot to prevent them from dominating the y-axis scale or
        # obscuring scientifically meaningful trends.  The removed points are
        # still counted in num_detect and included in the returned CSV; only
        # their visual representation is suppressed.
        if not all_detects.empty and max_plot_err is not None and max_plot_err > 0:
            plot_err_vals = pd.to_numeric(all_detects["plot_err"], errors="coerce")
            good_err = plot_err_vals.notna() & (plot_err_vals <= max_plot_err)
            n_removed = int((~good_err).sum())
            if n_removed > 0:
                removed_mjds = all_detects.loc[~good_err, "mjd"].tolist()
                logging.warning(
                    "plot_lightcurve: %d detection(s) in band %s excluded from plot "
                    "due to magnitude error > %.2f mag. MJDs: %s",
                    n_removed, band, max_plot_err,
                    ", ".join(f"{m:.5f}" for m in removed_mjds),
                )
                all_detects = all_detects[good_err].copy()

        # ------------------------------------------------------------------
        # Plot error bars and optional connecting line
        # ------------------------------------------------------------------
        if not all_detects.empty:
            ax.errorbar(
                x_transform(all_detects.mjd),
                all_detects["plot_mag"],
                yerr=all_detects["plot_err"],
                fmt='none',
                ecolor=c,
                capsize=get_marker_size('medium') / 4,
                capthick=0.8,
                elinewidth=0.5,
                zorder=2,
            )

            if ls:
                sorted_detects = all_detects.sort_values("mjd")
                ax.plot(
                    x_transform(sorted_detects.mjd),
                    sorted_detects["plot_mag"],
                    color=c,
                    linestyle=ls,
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=1,
                )

        # ------------------------------------------------------------------
        # Plot markers: normal (circle) vs inverted (hatched square)
        # ------------------------------------------------------------------
        if not all_detects.empty:
            if "_inverted_fit" in all_detects.columns:
                inv_fit_mask = _inverted_fit_series_to_bool(all_detects["_inverted_fit"])
            else:
                inv_fit_mask = pd.Series(False, index=all_detects.index)
            # Align inverted_only (full-band df index) to plotted rows.
            inv_row_mask = inverted_only.reindex(all_detects.index, fill_value=False)
            inv_row_mask = inv_row_mask.fillna(False).astype(bool)
            inv_fit_mask = inv_fit_mask | inv_row_mask

            # Normal detections: filled circles with black edge.
            # Detections with reduced_chi2 > chi2_marginal_threshold are
            # plotted as marginal (white face, faded edge) to flag poor
            # PSF-fit quality visually.
            normal_detects = all_detects[~inv_fit_mask]
            if not normal_detects.empty:
                # Split by chi2 quality if the column is available
                chi2_marginal_enabled = (
                    chi2_marginal_threshold is not None
                    and float(chi2_marginal_threshold) > 0
                    and "reduced_chi2" in normal_detects.columns
                )
                if chi2_marginal_enabled:
                    _chi2_vals = pd.to_numeric(
                        normal_detects["reduced_chi2"], errors="coerce"
                    )
                    marginal_mask = _chi2_vals > float(chi2_marginal_threshold)
                else:
                    marginal_mask = pd.Series(
                        False, index=normal_detects.index
                    )

                good_detects = normal_detects[~marginal_mask]
                marginal_detects = normal_detects[marginal_mask]

                if not good_detects.empty:
                    plotted_positive_flux_marker = True
                    ax.scatter(
                        x_transform(good_detects.mjd),
                        good_detects["plot_mag"],
                        s=get_marker_size('medium'),
                        c=c,
                        marker='o',
                        edgecolors='black',
                        linewidth=0.8,
                        zorder=3,
                        label=leg_label if leg_label else "",
                    )

                if not marginal_detects.empty:
                    plotted_marginal_chi2_marker = True
                    ax.scatter(
                        x_transform(marginal_detects.mjd),
                        marginal_detects["plot_mag"],
                        s=get_marker_size('medium'),
                        facecolors='white',
                        edgecolors=c,
                        linewidth=0.8,
                        alpha=0.5,
                        marker='o',
                        zorder=3,
                        # Only add the band label if no good detections
                        # claimed it already.
                        label=leg_label if good_detects.empty and leg_label else "",
                    )

            # Inverted detections: hatched squares with white diagonal stripes.
            inv_detects = all_detects[inv_fit_mask]
            if not inv_detects.empty:
                plotted_inverted_hatch = True
                sc = ax.scatter(
                    x_transform(inv_detects.mjd),
                    inv_detects["plot_mag"],
                    s=get_marker_size('medium'),
                    c=c,
                    marker='s',
                    edgecolors='black',
                    linewidth=0.8,
                    zorder=3,
                )
                sc.set_hatch('////')
                sc.set_edgecolor('white')
                # If only inverted detections exist for this band, add a
                # placeholder circle to the legend so the band is still listed.
                if normal_detects.empty:
                    ax.scatter([], [], s=get_marker_size('medium'), c=c, marker='o', edgecolors='black',
                             linewidth=0.8, label=leg_label if leg_label else "")

        if show_limits and not nondetects.empty:
            has_limits_plotted = True
            # Use ZTF-style SNU (5sigma) upper limit for non-detections where available
            limit_col = "lmag_upper" if "lmag_upper" in nondetects.columns else "lmag"
            ax.errorbar(
                x_transform(nondetects.mjd),
                nondetects[limit_col] + band_offset,
                color=c,
                ecolor=c,
                markeredgecolor=c,
                markerfacecolor="none",
                markeredgewidth=0.5,
                ls="",
                marker="v",
                markersize=get_marker_size('medium'),
                capsize=get_marker_size('medium') / 4,
                elinewidth=0.5,
                alpha=0.85,
                zorder=1,
            )
            # If this filter has only non-detections, still include it in the legend
            # as a standard detection marker (circle) so users can see the filter list.
            if all_detects.empty and leg_label:
                ax.scatter(
                    [],
                    [],
                    s=get_marker_size('medium'),
                    c=c,
                    marker="o",
                    edgecolors="black",
                    linewidth=0.8,
                    label=leg_label,
                )

        if not single_plot:
            ax.set_ylabel(f"{band} magnitude [mag]")
            ax.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.5)
            ax.minorticks_on()

    curve_axes[0].set_ylabel("Apparent Magnitude [mag]")
    (color_ax if plot_color else curve_axes[-1]).set_xlabel(xlabel)
    if subday_unit == "min":
        # Integer minute ticks for intra-night axes
        for _ax in list(curve_axes) + ([color_ax] if color_ax is not None else []):
            _ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    if plot_color:
        for ax in curve_axes:
            ax.tick_params(axis="x", labelbottom=False, length=0)

    for ax in curve_axes:
        ax.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.5)
        ax.minorticks_on()

    if mark_today and today_mjd is not None:
        today_rel = float(np.asarray(x_transform(today_mjd)))
        for ax in curve_axes:
            ax.axvline(
                x=today_rel,
                color="#C97B74",
                linestyle="--",
                alpha=0.7,
                linewidth=1.2,
                zorder=10,
                label="Today" if ax is curve_axes[0] else "",
            )

    if redshift and dm != 0:
        ax2 = curve_axes[0].twinx()
        ax2.set_xlim(curve_axes[0].get_xlim())
        ymin, ymax = curve_axes[0].get_ylim()
        ax2.set_ylim(ymax - dm, ymin - dm)
        ax2.set_ylabel("Absolute Magnitude [mag]")
        ax2.tick_params(axis="y")

    if show_details:
        curve_axes[0].text(
            0.02,
            0.97,
            f"Detections: {num_detect}\nNon-detections: {num_nondetect}",
            transform=curve_axes[0].transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="black", linewidth=0.5),
        )

    # Legend: filter symbols; optionally add limit legend entry once
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, labels = curve_axes[0].get_legend_handles_labels()
    # Inverted-only runs: explain hatched marker on the main legend. When both
    # normal and inverted points exist, the separate flux legend replaces this.
    if (
        plotted_inverted_hatch
        and not (plotted_positive_flux_marker and plotted_inverted_hatch)
        and "Inverted-image PSF" not in labels
    ):
        inv_patch = Patch(
            facecolor="0.55",
            edgecolor="white",
            hatch="////",
            linewidth=0.8,
            label="Inverted-image PSF",
        )
        handles.append(inv_patch)
        labels.append("Inverted-image PSF")
    if show_limits and has_limits_plotted and "Upper limit" not in labels:
        limit_handle = Line2D(
            [0],
            [0],
            color="black",
            marker="v",
            markersize=get_marker_size('medium'),
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.5,
            ls="",
            label="Upper limit",
        )
        handles.append(limit_handle)
        labels.append("Upper limit")
    if plotted_marginal_chi2_marker and "Marginal (high chi^2)" not in labels:
        marginal_handle = Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            markersize=get_marker_size('medium'),
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            alpha=0.5,
            ls="",
            label=f"Marginal (high chi^2, >{chi2_marginal_threshold:g})",
        )
        handles.append(marginal_handle)
        labels.append(f"Marginal (high chi^2, >{chi2_marginal_threshold:g})")

    # Choose number of legend columns so that the legend is taller than wide.
    n_labels = len(labels)
    if n_labels <= 4:
        ncol = 2
    elif n_labels <= 8:
        ncol = 2
    else:
        ncol = 3

    if n_labels == 3:
        # One row of three so the third label is on the top row (upper) and centered
        ncol = 3
    elif n_labels % 2 == 1 and n_labels > 1:
        # Odd > 3: add invisible placeholders so the last label is centered in its row
        inv = Line2D([], [], marker="none", linestyle="none", label="")
        handles = list(handles[:-1]) + [inv, handles[-1], inv]
        labels = list(labels[:-1]) + ["", labels[-1], ""]
        ncol = 3

    ax0 = curve_axes[0]
    leg_main = ax0.legend(
        handles,
        labels,
        loc="best",
        frameon=False,
        ncol=ncol,
    )
    ax0.add_artist(leg_main)

    # Second legend: flux sign only, same marker geometry as the lightcurve points.
    if plotted_positive_flux_marker and plotted_inverted_hatch:
        _fc = "0.45"
        h_pos = ax0.scatter(
            [],
            [],
            s=get_marker_size('medium'),
            marker="o",
            facecolors=_fc,
            edgecolors="black",
            linewidths=0.8,
        )
        h_neg = ax0.scatter(
            [],
            [],
            s=get_marker_size('medium'),
            marker="s",
            facecolors=_fc,
            edgecolors="white",
            linewidths=0.8,
        )
        h_neg.set_hatch("////")
        ax0.legend(
            [h_pos, h_neg],
            ["Positive flux", "Negative flux"],
            loc="lower right",
            frameon=False,
        )

    # No plot titles by default; but if target_name is supplied, use it as
    # a suptitle so multi-target lightcurves are distinguishable.
    if target_name:
        fig.suptitle(str(target_name), fontsize=11, y=0.98)

    ax0.invert_yaxis()

    # ---------- Colour evolution panel (same-night pairs only) ----------
    if plot_color and color_ax is not None and photometry_filter_series(data) is not None:
        color_ax.tick_params(axis="both", which="major")
        color_ax.tick_params(axis="both", which="minor", length=2.5)
        color_ax.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.5)
        color_ax.minorticks_on()
        color_ax.set_ylabel("Colour [mag]")
        for b1, b2 in _COLOR_PAIRS:
            if b1 not in bands_in_data or b2 not in bands_in_data:
                continue
            t1 = _resolve_band_triplet(set(data.columns), b1, method)
            t2 = _resolve_band_triplet(set(data.columns), b2, method)
            if t1 is None or t2 is None:
                continue
            col1, err1, zp1 = t1
            col2, err2, zp2 = t2
            fser = photometry_filter_series(data)
            if fser is not None:
                m1 = _filter_series_matches_band(b1, fser)
                m2 = _filter_series_matches_band(b2, fser)
                d1 = data[m1 & np.isfinite(data[zp1])].copy()
                d2 = data[m2 & np.isfinite(data[zp2])].copy()
            else:
                d1 = pd.DataFrame()
                d2 = pd.DataFrame()
            d1["mag"] = d1[col1] + d1[zp1] if col1.startswith('inst_') else d1[col1]
            d1["err"] = d1[err1]
            d2["mag"] = d2[col2] + d2[zp2] if col2.startswith('inst_') else d2[col2]
            d2["err"] = d2[err2]
            d1["lmag"] = _lmag_to_apparent(d1, zp1)
            d2["lmag"] = _lmag_to_apparent(d2, zp2)

            # ZTF-style SNU (upper limit): prefer 5sigma limit if available
            upper_limit_snr = 5.0
            upper_limit_col = f"limiting_mag_{upper_limit_snr:.0f}s2n"
            if upper_limit_col in d1.columns:
                # main.py outputs apparent magnitude columns directly
                d1["lmag_upper"] = pd.to_numeric(d1[upper_limit_col], errors="coerce")
                valid_upper = np.isfinite(d1["lmag_upper"]) & (d1["lmag_upper"] > 10) & (d1["lmag_upper"] < 30)
                d1.loc[~valid_upper, "lmag_upper"] = d1.loc[~valid_upper, "lmag"]
            else:
                d1["lmag_upper"] = d1["lmag"]
            if upper_limit_col in d2.columns:
                # main.py outputs apparent magnitude columns directly
                d2["lmag_upper"] = pd.to_numeric(d2[upper_limit_col], errors="coerce")
                valid_upper = np.isfinite(d2["lmag_upper"]) & (d2["lmag_upper"] > 10) & (d2["lmag_upper"] < 30)
                d2.loc[~valid_upper, "lmag_upper"] = d2.loc[~valid_upper, "lmag"]
            else:
                d2["lmag_upper"] = d2["lmag"]
            if use_SNR_limit:
                if method == "PSF" and "snr_psf" in d1.columns:
                    snr1 = np.asarray(d1["snr_psf"], dtype=float)
                    snr2 = np.asarray(d2["snr_psf"], dtype=float)
                elif (
                    method == "PSF"
                    and "flux_psf" in d1.columns
                    and "flux_psf_err" in d1.columns
                ):
                    snr1 = np.divide(
                        np.abs(d1["flux_psf"]),  # Use absolute flux to handle negative PSF fits
                        d1["flux_psf_err"],
                        out=np.full(len(d1), np.nan),
                        where=(np.asarray(d1["flux_psf_err"]) > 0)
                        & np.isfinite(d1["flux_psf_err"]),
                    )
                    snr2 = np.divide(
                        np.abs(d2["flux_psf"]),  # Use absolute flux to handle negative PSF fits
                        d2["flux_psf_err"],
                        out=np.full(len(d2), np.nan),
                        where=(np.asarray(d2["flux_psf_err"]) > 0)
                        & np.isfinite(d2["flux_psf_err"]),
                    )
                elif method == "AP" and "snr_ap" in d1.columns:
                    snr1 = np.asarray(d1["snr_ap"], dtype=float)
                    snr2 = np.asarray(d2["snr_ap"], dtype=float)
                elif "snr" in d1.columns:
                    snr1 = np.asarray(d1["snr"], dtype=float)
                    snr2 = np.asarray(d2["snr"], dtype=float)
                else:
                    snr1 = np.divide(
                        d1["mag"],
                        d1["err"],
                        out=np.zeros_like(d1["mag"]),
                        where=d1["err"] > 0,
                    )
                    snr2 = np.divide(
                        d2["mag"],
                        d2["err"],
                        out=np.zeros_like(d2["mag"]),
                        where=d2["err"] > 0,
                    )
            else:
                snr1 = np.divide(
                    d1["mag"],
                    d1["err"],
                    out=np.zeros_like(d1["mag"]),
                    where=d1["err"] > 0,
                )
                snr2 = np.divide(
                    d2["mag"],
                    d2["err"],
                    out=np.zeros_like(d2["mag"]),
                    where=d2["err"] > 0,
                )
            if use_SNR_limit:
                d1["det"] = (
                    np.isfinite(d1["mag"])
                    & np.isfinite(d1["err"])
                    & np.isfinite(snr1)
                    & (snr1 >= snr_limit)
                )
                d2["det"] = (
                    np.isfinite(d2["mag"])
                    & np.isfinite(d2["err"])
                    & np.isfinite(snr2)
                    & (snr2 >= snr_limit)
                )
            else:
                d1["det"] = (
                    np.isfinite(d1["mag"])
                    & np.isfinite(d1["err"])
                    & (d1["mag"] < d1["lmag"])
                )
                d2["det"] = (
                    np.isfinite(d2["mag"])
                    & np.isfinite(d2["err"])
                    & (d2["mag"] < d2["lmag"])
                )
            d1 = d1.sort_values("mjd")
            d2 = d2.sort_values("mjd")
            if d1.empty or d2.empty:
                continue
            mjd2 = d2["mjd"].values
            mag2 = d2["mag"].values
            err2_arr = d2["err"].values
            det2 = d2["det"].values
            lmag2 = d2["lmag"].values
            lmag1_arr = d1["lmag"].values
            lmag2_upper = d2["lmag_upper"].values if "lmag_upper" in d2.columns else lmag2
            lmag1_upper = d1["lmag_upper"].values if "lmag_upper" in d1.columns else lmag1_arr
            # Color = b1 - b2. Limit direction: b1 det + b2 limit -> true color <= color_value (upper limit, v);
            # b1 limit + b2 det -> true color >= color_value (lower limit, ^).
            # Use ZTF-style 5sigma upper limits (lmag_upper) where available.
            phase_pts, color_pts, err_pts = [], [], []
            phase_ul, color_ul = [], []  # upper limit on color (downward triangle v)
            phase_ll, color_ll = [], []  # lower limit on color (upward triangle ^)
            mjd1_arr = d1["mjd"].values
            mag1_arr = d1["mag"].values
            err1_arr = d1["err"].values
            det1_arr = d1["det"].values
            # Vectorised nearest-epoch match via searchsorted on sorted mjd2
            ins = np.searchsorted(mjd2, mjd1_arr)
            # searchsorted can return len(mjd2) when all mjd1 values are after
            # all mjd2 values. np.where evaluates all arguments eagerly, so we
            # need a clipped version for safe indexing into mjd2.
            ins_safe = np.clip(ins, 0, len(mjd2) - 1)
            j_arr = np.where(
                ins >= len(mjd2), len(mjd2) - 1,
                np.where(
                    ins == 0, 0,
                    np.where(
                        np.abs(mjd2[ins_safe] - mjd1_arr) <= np.abs(mjd2[ins_safe - 1] - mjd1_arr),
                        ins_safe, ins_safe - 1
                    )
                )
            )
            dt_arr = np.abs(mjd2[j_arr] - mjd1_arr)
            for i in range(len(mjd1_arr)):
                if dt_arr[i] > color_match_days:
                    continue
                j = j_arr[i]
                phase = float(np.asarray(x_transform((mjd1_arr[i] + mjd2[j]) / 2)))
                det1 = det1_arr[i]
                if det1 and det2[j]:
                    phase_pts.append(phase)
                    color_pts.append(mag1_arr[i] - mag2[j])
                    err_pts.append(np.sqrt(err1_arr[i] ** 2 + err2_arr[j] ** 2))
                elif det1 and not det2[j]:
                    phase_ul.append(phase)
                    color_ul.append(mag1_arr[i] - lmag2_upper[j])
                elif not det1 and det2[j]:
                    phase_ll.append(phase)
                    color_ll.append(lmag1_upper[i] - mag2[j])
            label = f"{b1}-{b2}"
            c = COLOR_INDEX_COLORS.get(label, cols.get(b1, "k"))
            if phase_pts:
                phase_arr = np.array(phase_pts)
                color_arr = np.array(color_pts)
                err_arr = np.array(err_pts)
                color_ax.errorbar(
                    phase_arr,
                    color_arr,
                    yerr=err_arr,
                    color=c,
                    ecolor=c,
                    marker="s",
                    markersize=get_marker_size('medium'),
                    ls="",
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label=label,
                    zorder=2,
                )
            if phase_ul:
                color_ax.errorbar(
                    np.array(phase_ul),
                    np.array(color_ul),
                    color=c,
                    markeredgecolor=c,
                    markerfacecolor="none",
                    markeredgewidth=0.5,
                    marker="v",
                    markersize=get_marker_size('medium'),
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    ls="",
                    zorder=2,
                )
            if phase_ll:
                color_ax.errorbar(
                    np.array(phase_ll),
                    np.array(color_ll),
                    color=c,
                    markeredgecolor=c,
                    markerfacecolor="none",
                    markeredgewidth=0.5,
                    marker="^",
                    markersize=get_marker_size('medium'),
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    ls="",
                    zorder=2,
                )
        color_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                        frameon=False)
        color_ax.invert_yaxis()

    # Include target name in the output filename so additional-target
    # lightcurves don't overwrite the primary target's plot.
    _plot_tag = f'LightCurve_{method}_{"single" if single_plot else "subplots"}'
    if target_name:
        _safe_tn = (
            str(target_name)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        _plot_tag = f'{_plot_tag}_{_safe_tn}'
    outname = f'{_plot_tag}.{format}'
    outpath = os.path.join(save_path, outname)
    save_kw = dict(dpi=dpi) if format.lower() != "pdf" else {}
    plt.savefig(outpath, **save_kw, bbox_inches="tight", facecolor="white")

    if show:
        import matplotlib

        # Force interactive backend right before show, in case anything
        # (e.g. imports from main.py) reset it to Agg during the function body.
        matplotlib.use('QtAgg', force=True)
        plt.show()
    else:
        plt.close(fig)

    det_file = None
    if return_detections and detections_list:
        valid_detections = [df for df in detections_list if not df.empty]
        if valid_detections:
            det_file = os.path.join(save_path, f"Detections_{base}_{method}.csv")
            pd.concat(valid_detections, ignore_index=True).to_csv(det_file, index=False, float_format="%.6f")
    if return_detections and nondetections_list:
        valid_nondetections = [df for df in nondetections_list if not df.empty]
        if valid_nondetections:
            nondet_file = os.path.join(save_path, f"Nondetections_{base}_{method}.csv")
            pd.concat(valid_nondetections, ignore_index=True).to_csv(
                nondet_file, index=False, float_format="%.6f"
            )

    return det_file


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================


def generate_photometry_table(
    output_file,
    snr_limit=3,
    beta_limit=0.5,
    method="PSF",
    reference_epoch=0,
    use_SNR_limit=False,
    include_color_table=True,
    color_match_days=0.5,
    input_yaml=None,
    target_name=None,
):
    """Generate a photometry table with MJD, ISO date, magnitude, error, filter, and limit value.
    Optionally build a same-night colour evolution table (e.g. g-r, r-i).

    Args:
        output_file: Path to the photometry data CSV file.
        snr_limit: Minimum SNR for a detection (if use_SNR_limit=True).
        beta_limit: Minimum beta for a detection.
        method: Photometry method (e.g., 'PSF').
        reference_epoch: Reference epoch for phase calculation.
        use_SNR_limit: If True, use SNR for detection; else use lmag.
        include_color_table: If True and data has a 'filter' column and >1 row, write a colour table.
        color_match_days: Max separation (days) for same-night colour pairs.
        input_yaml: Configuration dictionary for reading S/N thresholds from config.
    """
    complete_data = pd.read_csv(output_file)
    complete_data = _normalize_photometry_columns(complete_data)
    if complete_data.columns.duplicated().any():
        complete_data = complete_data.loc[:, ~complete_data.columns.duplicated()].copy()
    phot_table = []
    # Prefer using the per-row filter column if present. Map raw names (gp, Sloan_g, ...)
    # to canonical bands so `_resolve_band_triplet` matches the uniform mag/zp schema.
    fser_all = photometry_filter_series(complete_data)
    if fser_all is not None and fser_all.notna().any():
        bands = canonical_bands_from_filter_series(fser_all)
        label_map = canonical_band_label_map_from_filter_series(fser_all)
    else:
        # Fallback when no filter column exists.
        bands = list("FSDNAuUBgcVwrRoEiIzyYJHKWQ")
        label_map = {}

    used_triplets = set()
    for band in bands:
        trip = _resolve_band_triplet(set(complete_data.columns), band, method)
        if trip is None:
            continue
        col, err_col, zp_col = trip
        # Only skip if the same triplet was used AND there's no filter column to distinguish bands
        # If there's a filter column, we can process multiple bands with the same triplet
        fcol_check = photometry_filter_series(complete_data)
        if trip in used_triplets and fcol_check is None:
            continue
        used_triplets.add(trip)

        data = complete_data[np.isfinite(complete_data[zp_col])].copy()
        fcol = photometry_filter_series(data)
        if fcol is not None:
            mask = fcol.map(lambda rv: filter_value_matches_band(band, rv))
            data = data[mask.fillna(False)].copy()
            if data.empty:
                continue
            # Re-read the filter column on the band-filtered rows so the
            # reported Filter value comes from this band's own rows.
            fcol_filtered = photometry_filter_series(data)
        else:
            fcol_filtered = None
        
        # Use the actual filter value from the data instead of mapped label
        # This ensures the Filter column in the output matches the input data
        if fcol_filtered is not None and not fcol_filtered.empty:
            filter_value = fcol_filtered.mode()[0] if len(fcol_filtered.mode()) > 0 else str(band)
            band_label = str(filter_value).strip()
        else:
            band_label = label_map.get(str(band).strip().lower(), band)

        import logging
        logger = logging.getLogger(__name__)
        logger.info("Processing band: %s, band_label: %s, data rows: %s", band, band_label, len(data))
        # Coerce numerics: CSV concatenation can yield strings like "nan".
        if "beta" in data.columns:
            data["beta"] = pd.to_numeric(data["beta"], errors="coerce")
        lim_col = (
            "limiting_inst_mag"
            if "limiting_inst_mag" in data.columns
            else ("lmag" if "lmag" in data.columns else None)
        )
        if lim_col is None:
            data["lmag"] = np.nan
            lmag_inst = pd.to_numeric(data["lmag"], errors="coerce")
        else:
            lmag_inst = pd.to_numeric(data[lim_col], errors="coerce")
        zp_num = pd.to_numeric(data[zp_col], errors="coerce")
        # Pipeline stores limiting mag in instrumental system; detection cut uses apparent mags.
        data["lmag"] = lmag_inst + zp_num

        # ZTF-style SNU upper limit: prefer the 5sigma limit, else the 50%
        # completeness limit.
        upper_limit_snr = 5.0  # ZTF SNU default
        upper_limit_col = f"limiting_mag_{upper_limit_snr:.0f}s2n"
        if upper_limit_col in data.columns:
            # main.py outputs apparent magnitude columns directly
            data["lmag_upper"] = pd.to_numeric(data[upper_limit_col], errors="coerce")
            # Use 5sigma limit where valid, otherwise fall back to 50% completeness limit
            valid_upper = np.isfinite(data["lmag_upper"]) & (data["lmag_upper"] > 10) & (data["lmag_upper"] < 30)
            data.loc[~valid_upper, "lmag_upper"] = data.loc[~valid_upper, "lmag"]
        else:
            data["lmag_upper"] = data["lmag"]
        if col.startswith("inst_"):
            data["__app_mag_det__"] = pd.to_numeric(data[col], errors="coerce") + zp_num
            _det_mag_col = "__app_mag_det__"
        else:
            _det_mag_col = col
        detected = _compute_detection_mask(
            data,
            _det_mag_col,
            err_col,
            method,
            snr_limit=float(snr_limit),
            use_SNR_limit=bool(use_SNR_limit),
        )
        detected_s = pd.Series(detected, index=data.index, dtype=bool)

        # Inverted-only detections come from the _inverted_fit flag or a
        # finite inst_inverted column.
        has_inverted = False
        inverted_col = None

        if "_inverted_fit" in data.columns:
            inv_flag = _inverted_fit_series_to_bool(data["_inverted_fit"])
            has_inverted = bool(inv_flag.any())
        else:
            inv_flag = pd.Series(False, index=data.index)

        if not has_inverted:
            inverted_col = "inst_inverted" if "inst_inverted" in data.columns else None
            has_inverted = inverted_col is not None and np.any(
                np.isfinite(pd.to_numeric(data[inverted_col], errors="coerce"))
            )

        if has_inverted:
            if inverted_col is None and "inst_inverted" in data.columns:
                inverted_col = "inst_inverted"
            if inverted_col and inverted_col in data.columns:
                inv_finite = np.isfinite(
                    pd.to_numeric(data[inverted_col], errors="coerce").to_numpy(dtype=float)
                )
            else:
                inv_finite = np.zeros(len(data), dtype=bool)
            inv_finite = pd.Series(inv_finite, index=data.index)
            inverted_only = inv_flag | (inv_finite & ~detected_s)
            normal_detected = detected_s & ~inverted_only
        else:
            inverted_only = pd.Series(False, index=data.index)
            normal_detected = detected_s

        detects = data[normal_detected].copy()
        inv_detects = data[inverted_only].copy() if has_inverted else pd.DataFrame()
        nondetects = data[~detected_s & ~inverted_only].copy()

        if not detects.empty:
            # 'inst_*' columns are instrumental; others are already apparent.
            if col.startswith('inst_'):
                mag_values = detects[col] + detects[zp_col]
            else:
                mag_values = detects[col]

            limiting_mags = _lmag_to_apparent_multi_snr(detects, zp_col, input_yaml=input_yaml)

            row_data = {
                'Filter': band_label,
                'MJD': detects["mjd"].round(3),
                'Date': Time(detects["mjd"], format="mjd").iso,
                'Mag': mag_values.round(3),
                'Error': detects[err_col].round(3),
            }

            for limit_key, limit_values in limiting_mags.items():
                row_data[limit_key] = limit_values

            detects = detects.assign(**row_data)

            # Column order: base columns first, then all Limit_* columns.
            base_cols = ["MJD", "Date", "Mag", "Error", "Filter"]
            limit_cols = sorted([k for k in limiting_mags.keys() if k.startswith('Limit')])
            all_cols = base_cols + limit_cols

            detects = detects[all_cols]
            phot_table.append(detects)

        if not inv_detects.empty:
            inv_err_col = err_col if err_col in inv_detects.columns else None
            # Band-specific inverted apparent magnitude wins; otherwise fall
            # back to the generic instrumental column and convert with the ZP.
            band_inv_mag_col = f"{band}_{method}_inverted" if f"{band}_{method}_inverted" in inv_detects.columns else None
            band_inv_err_col = f"{band}_{method}_err_inverted" if f"{band}_{method}_err_inverted" in inv_detects.columns else None

            if band_inv_mag_col and band_inv_mag_col in inv_detects.columns:
                inv_mag_col = band_inv_mag_col
                inv_err_col = band_inv_err_col
                is_apparent = True
            else:
                inv_mag_col = "inst_inverted" if "inst_inverted" in inv_detects.columns else inverted_col
                inv_err_col = "inst_inverted_err" if "inst_inverted_err" in inv_detects.columns else None
                is_apparent = False
            
            if inv_mag_col and inv_mag_col in inv_detects.columns:
                # Use the actual filter value from the data instead of mapped label
                fcol_inv = photometry_filter_series(inv_detects)
                if fcol_inv is not None and not fcol_inv.empty:
                    filter_value_inv = fcol_inv.mode()[0] if len(fcol_inv.mode()) > 0 else str(band)
                    band_label = str(filter_value_inv).strip()
                else:
                    band_label = label_map.get(str(band).strip().lower(), band)

                if is_apparent:
                    mag_value = inv_detects[inv_mag_col]
                else:
                    mag_value = inv_detects[inv_mag_col] + inv_detects[zp_col]

                _err_s = (
                    pd.to_numeric(inv_detects[inv_err_col], errors="coerce").round(3)
                    if inv_err_col and inv_err_col in inv_detects.columns
                    else pd.Series(np.nan, index=inv_detects.index, dtype=float)
                )
                limiting_mags_inv = _lmag_to_apparent_multi_snr(inv_detects, zp_col, input_yaml=input_yaml)

                inv_row_data = {
                    'Filter': band_label,
                    'MJD': inv_detects["mjd"].round(3),
                    'Date': Time(inv_detects["mjd"], format="mjd").iso,
                    'Mag': mag_value.round(3),
                    'Error': _err_s,
                }

                for limit_key, limit_values in limiting_mags_inv.items():
                    inv_row_data[limit_key] = limit_values

                inv_detects = inv_detects.assign(**inv_row_data)

                # Column order: base columns first, then all Limit_* columns.
                base_cols = ["MJD", "Date", "Mag", "Error", "Filter"]
                limit_cols = sorted([k for k in limiting_mags_inv.keys() if k.startswith('Limit')])
                all_cols = base_cols + limit_cols

                inv_detects = inv_detects[all_cols]
                phot_table.append(inv_detects)

        if not nondetects.empty:
            limiting_mags_nd = _lmag_to_apparent_multi_snr(nondetects, zp_col, input_yaml=input_yaml)

            # Use the actual filter value from the data instead of mapped label
            fcol_nd = photometry_filter_series(nondetects)
            if fcol_nd is not None and not fcol_nd.empty:
                filter_value_nd = fcol_nd.mode()[0] if len(fcol_nd.mode()) > 0 else str(band)
                band_label = str(filter_value_nd).strip()
            else:
                band_label = label_map.get(str(band).strip().lower(), band)

            nd_row_data = {
                'Filter': band_label,
                'MJD': nondetects["mjd"].round(3),
                'Date': Time(nondetects["mjd"], format="mjd").iso,
                'Mag': pd.Series(np.nan, index=nondetects.index, dtype=float),
                'Error': pd.Series(np.nan, index=nondetects.index, dtype=float),
            }

            for limit_key, limit_values in limiting_mags_nd.items():
                nd_row_data[limit_key] = limit_values

            nondetects = nondetects.assign(**nd_row_data)

            # Column order: base columns first, then all Limit_* columns.
            base_cols = ["MJD", "Date", "Mag", "Error", "Filter"]
            limit_cols = sorted([k for k in limiting_mags_nd.keys() if k.startswith('Limit')])
            all_cols = base_cols + limit_cols

            nondetects = nondetects[all_cols]
            phot_table.append(nondetects)

    if not phot_table:
        # Empty table still needs the configured Limit_* columns.
        if input_yaml is not None:
            lim_cfg = input_yaml.get("limiting_magnitude") or {}
            snr_thresholds = lim_cfg.get("snr_thresholds", [3.0, 5.0])
        else:
            snr_thresholds = [3.0, 5.0]

        limit_cols = [f'Limit_{snr:.1f}S2N'.replace('.', 'p') for snr in snr_thresholds]
        default_cols = ["MJD", "Date", "Mag", "Error", "Filter"] + limit_cols
        out_phot = pd.DataFrame(columns=default_cols)
    else:
        out_phot = pd.concat(phot_table, ignore_index=True)
        if reference_epoch:
            out_phot.insert(2, "Phase", (out_phot["MJD"] - reference_epoch).round(3))
        out_phot.sort_values(["MJD", "Filter"], inplace=True)
        out_phot.reset_index(drop=True, inplace=True)

    save_path = os.path.dirname(output_file)
    _table_tag = f"LightCurve_{method}"
    if target_name:
        _safe_tn = (
            str(target_name)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        _table_tag = f"{_table_tag}_{_safe_tn}"
    fname = os.path.join(save_path, f"{_table_tag}.dat")
    # Uniform numeric formatting; non-detections use NaN for Mag/Error (written as empty or "-")
    out_phot.to_csv(fname, index=False, float_format="%.3f", na_rep="-")

    # ---------- Colour evolution table (same-night pairs) ----------
    color_table_path = None
    if (
        include_color_table
        and photometry_filter_series(complete_data) is not None
        and len(complete_data) > 1
    ):
        color_rows = []
        fser_cd = photometry_filter_series(complete_data)
        for b1, b2 in _COLOR_PAIRS:
            t1 = _resolve_band_triplet(set(complete_data.columns), b1, method)
            t2 = _resolve_band_triplet(set(complete_data.columns), b2, method)
            if t1 is None or t2 is None:
                continue
            col1, err1, zp1 = t1
            col2, err2, zp2 = t2
            m1 = fser_cd.map(lambda rv: filter_value_matches_band(b1, rv)).fillna(False)
            m2 = fser_cd.map(lambda rv: filter_value_matches_band(b2, rv)).fillna(False)
            d1 = complete_data[m1 & np.isfinite(complete_data[zp1])].copy()
            d2 = complete_data[m2 & np.isfinite(complete_data[zp2])].copy()
            if d1.empty or d2.empty:
                continue
            d1["mag"] = d1[col1] + d1[zp1] if col1.startswith("inst_") else d1[col1]
            d1["err"] = d1[err1]
            d1["lmag"] = _lmag_to_apparent(d1, zp1)
            d2["mag"] = d2[col2] + d2[zp2] if col2.startswith("inst_") else d2[col2]
            d2["err"] = d2[err2]
            d2["lmag"] = _lmag_to_apparent(d2, zp2)
            if use_SNR_limit:
                if method == "PSF" and "snr_psf" in d1.columns:
                    snr1 = np.asarray(d1["snr_psf"], dtype=float)
                    snr2 = np.asarray(d2["snr_psf"], dtype=float)
                elif (
                    method == "PSF"
                    and "flux_psf" in d1.columns
                    and "flux_psf_err" in d1.columns
                ):
                    snr1 = np.divide(
                        np.abs(d1["flux_psf"]),  # Use absolute flux to handle negative PSF fits
                        d1["flux_psf_err"],
                        out=np.full(len(d1), np.nan),
                        where=(np.asarray(d1["flux_psf_err"]) > 0)
                        & np.isfinite(d1["flux_psf_err"]),
                    )
                    snr2 = np.divide(
                        np.abs(d2["flux_psf"]),  # Use absolute flux to handle negative PSF fits
                        d2["flux_psf_err"],
                        out=np.full(len(d2), np.nan),
                        where=(np.asarray(d2["flux_psf_err"]) > 0)
                        & np.isfinite(d2["flux_psf_err"]),
                    )
                elif method == "AP" and "snr_ap" in d1.columns:
                    snr1 = np.asarray(d1["snr_ap"], dtype=float)
                    snr2 = np.asarray(d2["snr_ap"], dtype=float)
                elif "snr" in d1.columns:
                    snr1 = np.asarray(d1["snr"], dtype=float)
                    snr2 = np.asarray(d2["snr"], dtype=float)
                else:
                    snr1 = np.divide(
                        d1["mag"],
                        d1["err"],
                        out=np.zeros_like(d1["mag"]),
                        where=d1["err"] > 0,
                    )
                    snr2 = np.divide(
                        d2["mag"],
                        d2["err"],
                        out=np.zeros_like(d2["mag"]),
                        where=d2["err"] > 0,
                    )
            else:
                snr1 = np.divide(
                    d1["mag"],
                    d1["err"],
                    out=np.zeros_like(d1["mag"]),
                    where=d1["err"] > 0,
                )
                snr2 = np.divide(
                    d2["mag"],
                    d2["err"],
                    out=np.zeros_like(d2["mag"]),
                    where=d2["err"] > 0,
                )
            if use_SNR_limit:
                d1["det"] = (
                    np.isfinite(d1["mag"])
                    & np.isfinite(d1["err"])
                    & np.isfinite(snr1)
                    & (snr1 >= snr_limit)
                )
                d2["det"] = (
                    np.isfinite(d2["mag"])
                    & np.isfinite(d2["err"])
                    & np.isfinite(snr2)
                    & (snr2 >= snr_limit)
                )
            else:
                d1["det"] = (
                    np.isfinite(d1["mag"])
                    & np.isfinite(d1["err"])
                    & (d1["mag"] < d1["lmag"])
                )
                d2["det"] = (
                    np.isfinite(d2["mag"])
                    & np.isfinite(d2["err"])
                    & (d2["mag"] < d2["lmag"])
                )
            d1 = d1.sort_values("mjd")
            d2 = d2.sort_values("mjd")
            mjd2 = d2["mjd"].values
            mag2 = d2["mag"].values
            err2_arr = d2["err"].values
            det2 = d2["det"].values
            lmag2 = d2["lmag"].values
            lmag2_upper = d2["lmag_upper"].values if "lmag_upper" in d2.columns else lmag2
            mjd1_arr = d1["mjd"].values
            mag1_arr = d1["mag"].values
            err1_arr_tbl = d1["err"].values
            det1_arr_tbl = d1["det"].values
            lmag1_arr_tbl = d1["lmag"].values
            lmag1_upper = d1["lmag_upper"].values if "lmag_upper" in d1.columns else lmag1_arr_tbl
            ins = np.searchsorted(mjd2, mjd1_arr)
            # searchsorted can return len(mjd2) when all mjd1 values are after all mjd2 values.
            # np.where evaluates all arguments eagerly, so we need a clipped version
            # for safe indexing into mjd2.
            ins_safe = np.clip(ins, 0, len(mjd2) - 1)
            j_arr = np.where(
                ins >= len(mjd2), len(mjd2) - 1,
                np.where(
                    ins == 0, 0,
                    np.where(
                        np.abs(mjd2[ins_safe] - mjd1_arr) <= np.abs(mjd2[ins_safe - 1] - mjd1_arr),
                        ins, ins - 1
                    )
                )
            )
            dt_arr = np.abs(mjd2[j_arr] - mjd1_arr)
            for i in range(len(mjd1_arr)):
                if dt_arr[i] > color_match_days:
                    continue
                j = j_arr[i]
                mjd_mid = (mjd1_arr[i] + mjd2[j]) / 2
                date_str = Time(mjd_mid, format="mjd").iso
                phase = (mjd_mid - reference_epoch) if reference_epoch else np.nan
                color_name = f"{b1}-{b2}"
                if det1_arr_tbl[i] and det2[j]:
                    color_value = mag1_arr[i] - mag2[j]
                    err_val = np.sqrt(err1_arr_tbl[i] ** 2 + err2_arr[j] ** 2)
                    color_rows.append(
                        {
                            "MJD": round(mjd_mid, 3),
                            "Date": date_str,
                            "Phase": round(phase, 3) if np.isfinite(phase) else "-",
                            "Color": color_name,
                            "Value": round(color_value, 3),
                            "Error": round(err_val, 3),
                            "Limit": "-",
                        }
                    )
                elif det1_arr_tbl[i] and not det2[j]:
                    color_value = mag1_arr[i] - lmag2_upper[j]
                    color_rows.append(
                        {
                            "MJD": round(mjd_mid, 3),
                            "Date": date_str,
                            "Phase": round(phase, 3) if np.isfinite(phase) else "-",
                            "Color": color_name,
                            "Value": round(color_value, 3),
                            "Error": "-",
                            "Limit": "upper",
                        }
                    )
                elif not det1_arr_tbl[i] and det2[j]:
                    color_value = lmag1_upper[i] - mag2[j]
                    color_rows.append(
                        {
                            "MJD": round(mjd_mid, 3),
                            "Date": date_str,
                            "Phase": round(phase, 3) if np.isfinite(phase) else "-",
                            "Color": color_name,
                            "Value": round(color_value, 3),
                            "Error": "-",
                            "Limit": "lower",
                        }
                    )
        if color_rows:
            color_df = pd.DataFrame(color_rows)
            color_df.sort_values(["MJD", "Color"], inplace=True)
            color_table_path = os.path.join(
                save_path, f"{_table_tag}_Colors.dat"
            )
            color_df.to_csv(
                color_table_path, index=False, float_format="%.3f", na_rep="-"
            )

    return out_phot


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================


def _parse_calib_header(filepath) -> dict:
    """Parse the ``# key: value`` comment header of a per-image ``Calib_*.csv``."""
    info = {}
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") and ":" in line:
                    parts = line[1:].split(":", 1)
                    if len(parts) == 2:
                        info[parts[0].strip()] = parts[1].strip()
                elif line and not line.startswith("#"):
                    break
    except OSError:
        pass
    return info


def _parse_calib_catalog(filepath) -> pd.DataFrame:
    """Read the sequence-star catalog table from a ``Calib_*.csv`` file.

    The catalog is the CSV block following the ``#`` comment header (the first
    non-comment line is the column header, e.g. ``RA,DEC,...``).
    """
    try:
        return pd.read_csv(filepath, comment="#")
    except Exception:
        return pd.DataFrame()


def _nearest_epoch(query, epochs, tol):
    """Map each query MJD to the nearest catalog epoch within ``tol`` days.

    Returns a float array aligned to ``query`` holding the matched epoch value
    (or NaN when no epoch is within tolerance).  Used because target MJD and
    ``Calib_*.csv`` header MJD can differ at the float-rounding level.
    """
    epochs = np.sort(np.asarray(epochs, dtype=float))
    q = np.asarray(query, dtype=float)
    out = np.full(len(q), np.nan)
    if epochs.size == 0:
        return out
    ins = np.searchsorted(epochs, q)
    ins_c = np.clip(ins, 0, epochs.size - 1)
    cand = np.where(
        ins >= epochs.size,
        epochs.size - 1,
        np.where(
            ins == 0,
            0,
            np.where(
                np.abs(epochs[ins_c] - q) <= np.abs(epochs[ins_c - 1] - q),
                ins_c,
                ins_c - 1,
            ),
        ),
    )
    dt = np.abs(epochs[cand] - q)
    ok = np.isfinite(q) & (dt <= tol)
    out[ok] = epochs[cand[ok]]
    return out


def plot_variability_check(
    output_file,
    method="PSF",
    snr_min=25.0,
    flux_min=100.0,
    flux_max=None,
    min_epoch_frac=0.8,
    n_ensemble=20,
    n_ref_plot=25,
    max_plot_err=0.5,
    show_drift_panel=True,
    calib_dir=None,
    format="png",
    dpi=150,
    show=False,
    target_name=None,
):
    """Plot a differential variability check: target vs reference-star ensemble.

    For each band present in the lightcurve output, this reads the per-image
    ``Calib_*.csv`` sequence-star catalogs (searched recursively under the
    directory containing ``output_file``), builds the per-epoch mean
    instrumental magnitude of a bright reference-star ensemble, and subtracts
    that common-mode signal from every reference star and from the target.
    Ensemble members are first vetted for intrinsic variability (median/MAD
    outlier rejection on each member's mean-centred residual RMS) so that a
    variable star cannot contaminate the common-mode mean.

    The figure has, per band:

    * a top panel (when ``show_drift_panel``) showing the ensemble-mean
      instrumental magnitude and the raw target instrumental magnitude, each
      mean-centred -- curves that track each other indicate the variability is
      instrumental/atmospheric;
    * a bottom panel with per-star differential residuals
      ``inst - <ensemble mean> - <star mean>`` for reference stars (grey cloud
      plus per-epoch mean +/- std), and the same quantity for the target.

    If the target's residuals are flat and comparable to the reference-star
    scatter, the observed variability was instrumental; excess structure
    indicates intrinsic variability.  No model is fit to the transient light
    curve.  For data spanning < 1 day the time axis switches to
    minutes/hours since the first observation (as in :func:`plot_lightcurve`).

    Parameters
    ----------
    output_file : str
        Path to the concatenated photometry CSV (e.g. ``LightCurve_Output.csv``).
    method : str
        Photometry method whose instrumental columns are used ('PSF' or 'AP').
    snr_min, flux_min : float
        Minimum mean S/N and mean flux for a catalog star to join the
        reference ensemble (relaxed automatically if too few stars qualify).
    flux_max : float or None
        Optional maximum flux cut (saturation guard); None disables it.
    min_epoch_frac : float
        Fraction of epochs a star must be present in to qualify (default 0.8;
        progressively relaxed if fewer than 3 stars qualify).
    n_ensemble : int
        Number of brightest qualifying stars used for the ensemble mean.
    n_ref_plot : int
        Number of additional (non-ensemble) stars drawn as the grey cloud.
    max_plot_err : float
        Maximum target residual error plotted (mag).
    show_drift_panel : bool
        Show the per-band instrumental-drift comparison panel.
    calib_dir : str or None
        Directory searched recursively for ``Calib_*.csv`` catalogs.
        Defaults to the directory containing ``output_file``; point this at
        the main reduced directory when ``output_file`` lives elsewhere
        (e.g. additional-target CSVs in ``sub_targets/``).
    format, dpi, show : plotting controls
        Same conventions as :func:`plot_lightcurve`.
    target_name : str or None
        Optional label; appended to the output filename.

    Returns
    -------
    str or None
        Path to the saved figure, or None when insufficient data.
    """
    log = logging.getLogger(__name__)
    apply_autophot_mplstyle()
    if show:
        import matplotlib

        current_backend = str(plt.get_backend()).lower()
        if "agg" in current_backend:
            for backend in ("QtAgg", "TkAgg"):
                try:
                    matplotlib.use(backend, force=True)
                    break
                except Exception:
                    continue

    try:
        data = pd.read_csv(output_file)
    except Exception as exc:
        log.error("plot_variability_check: cannot read '%s': %s", output_file, exc)
        return None
    data = _normalize_photometry_columns(data)
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()].copy()
    if "mjd" not in data.columns:
        log.warning("plot_variability_check: no 'mjd' column in '%s'", output_file)
        return None

    save_path = os.path.dirname(os.path.abspath(output_file))
    m_low = str(method).strip().lower()
    method_u = str(method).strip().upper()

    # Discover bands, mirroring plot_lightcurve (long-form via `filter`
    # column, else wide-format per-band triplets).
    long_form = (
        f"mag_{m_low}" in data.columns and f"mag_{m_low}_err" in data.columns
    )
    fser = photometry_filter_series(data)
    bands = []
    if long_form and fser is not None and fser.notna().any():
        bands = canonical_bands_from_filter_series(fser)
    if not bands:
        used_triplets = set()
        for b in "FSDNAuUBgcVwrRoEiIzyYJHKWQ":
            trip = _resolve_band_triplet(set(data.columns), b, method)
            if trip is not None and trip not in used_triplets:
                bands.append(b)
                used_triplets.add(trip)
    if not bands:
        log.info(
            "plot_variability_check: no photometric bands found in '%s'.",
            output_file,
        )
        return None

    # Locate per-image Calib catalogs under the reduced output tree.
    search_dir = os.path.abspath(calib_dir) if calib_dir else save_path
    calib_files = sorted(
        glob.glob(os.path.join(search_dir, "**", "Calib_*.csv"), recursive=True)
    )
    if not calib_files:
        log.info(
            "plot_variability_check: no Calib_*.csv files found under '%s'.",
            search_dir,
        )
        return None

    calib_meta = []
    for fpath in calib_files:
        info = _parse_calib_header(fpath)
        try:
            mjd = float(info.get("mjd", "nan"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(mjd):
            continue
        calib_meta.append(
            {"path": fpath, "mjd": mjd, "filter": str(info.get("filter", "")).strip()}
        )
    if not calib_meta:
        log.info("plot_variability_check: no usable Calib headers found.")
        return None

    band_payloads = []
    all_mjds = []

    for band in bands:
        band_files = [
            m for m in calib_meta if filter_value_matches_band(band, m["filter"])
        ]
        if len(band_files) < 2:
            continue

        frames = []
        for meta in band_files:
            cat = _parse_calib_catalog(meta["path"])
            if cat.empty or "RA" not in cat.columns or "DEC" not in cat.columns:
                continue
            raw_filt = meta["filter"]
            cmap = {str(c).lower(): c for c in cat.columns}
            inst_col = cmap.get(f"inst_{raw_filt}_{method_u}".lower())
            if inst_col is None:
                continue
            inst_err_col = cmap.get(f"{inst_col}_err".lower())
            snr_col = cmap.get(f"snr_{m_low}", cmap.get("snr"))
            flux_col = cmap.get(f"flux_{m_low}")
            sub = pd.DataFrame(
                {
                    "ra": pd.to_numeric(cat["RA"], errors="coerce"),
                    "dec": pd.to_numeric(cat["DEC"], errors="coerce"),
                    "mjd": float(meta["mjd"]),
                    "inst": pd.to_numeric(cat[inst_col], errors="coerce"),
                    "inst_err": (
                        pd.to_numeric(cat[inst_err_col], errors="coerce")
                        if inst_err_col
                        else np.nan
                    ),
                    "snr": (
                        pd.to_numeric(cat[snr_col], errors="coerce")
                        if snr_col
                        else np.nan
                    ),
                    "flux": (
                        pd.to_numeric(cat[flux_col], errors="coerce")
                        if flux_col
                        else np.nan
                    ),
                }
            )
            sub = sub[
                np.isfinite(sub["inst"])
                & np.isfinite(sub["ra"])
                & np.isfinite(sub["dec"])
            ]
            if not sub.empty:
                frames.append(sub)
        if not frames:
            continue

        cat_all = pd.concat(frames, ignore_index=True)
        # Star identity key: sequence stars share catalog coordinates, so a
        # rounded (ra, dec) pair is a stable identifier across epochs.
        cat_all["star_id"] = list(
            zip(cat_all["ra"].round(4), cat_all["dec"].round(4))
        )
        n_epochs = int(cat_all["mjd"].nunique())
        if n_epochs < 2:
            continue

        # Per-star coverage/quality stats; invariant across the relaxation
        # ladders below, so computed once.
        stats = (
            cat_all.groupby("star_id")
            .agg(
                n_mjd=("mjd", "nunique"),
                mean_snr=("snr", "mean"),
                mean_flux=("flux", "mean"),
                max_flux=("flux", "max"),
            )
            .reset_index()
        )
        if not cat_all["snr"].notna().any():
            log.debug(
                "plot_variability_check: band %s - no snr values in Calib "
                "catalogs; snr_min cut inactive.",
                band,
            )
        if not cat_all["flux"].notna().any():
            log.debug(
                "plot_variability_check: band %s - no flux values in Calib "
                "catalogs; flux_min/flux_max cuts inactive.",
                band,
            )

        # Reference-ensemble selection with progressive relaxation so that
        # sparse/noisy fields still produce a diagnostic.
        sel = None
        frac_ladder = [min_epoch_frac] + [
            f for f in (0.6, 0.4, 0.0) if f < min_epoch_frac
        ]
        for frac in frac_ladder:
            for snr_cut in (snr_min, snr_min / 2.0, 0.0):
                need = max(2, int(np.ceil(frac * n_epochs)))
                good = stats[
                    (stats["n_mjd"] >= need)
                    & (stats["mean_snr"].fillna(np.inf) > snr_cut)
                    & (stats["mean_flux"].fillna(np.inf) > flux_min)
                ]
                if flux_max is not None:
                    good = good[good["max_flux"] < flux_max]
                if len(good) >= 3:
                    sel = good.sort_values("mean_flux", ascending=False)
                    break
            if sel is not None:
                break
        if sel is None or sel.empty:
            log.info(
                "plot_variability_check: band %s - fewer than 3 usable "
                "reference stars; skipping.",
                band,
            )
            continue

        ensemble_ids = set(sel.head(n_ensemble)["star_id"])

        def _ens_epoch_stats(ids):
            return (
                cat_all[cat_all["star_id"].isin(ids)]
                .groupby("mjd")["inst"]
                .agg(["mean", "std", "count"])
                .rename(
                    columns={
                        "mean": "ens_mean",
                        "std": "ens_std",
                        "count": "ens_n",
                    }
                )
                .reset_index()
            )

        # Veto variable ensemble members: a genuinely variable (or
        # saturated/blended) star in the ensemble leaks its signal into the
        # common-mode mean and distorts every residual. Iteratively reject
        # members whose mean-centred residual RMS is a MAD-based outlier,
        # keeping at least 3 stars.
        for _ in range(2):
            ens_epoch = _ens_epoch_stats(ensemble_ids)
            ens = cat_all[cat_all["star_id"].isin(ensemble_ids)].copy()
            ens["ens_mean"] = ens["mjd"].map(
                dict(zip(ens_epoch["mjd"], ens_epoch["ens_mean"]))
            )
            d = ens["inst"] - ens["ens_mean"]
            d = d - d.groupby(ens["star_id"]).transform("mean")
            rms = d.groupby(ens["star_id"]).std()
            rms = rms[np.isfinite(rms)]
            if len(rms) < 4:
                break
            rms_med = float(rms.median())
            rms_mad = float((rms - rms_med).abs().median())
            # 3-sigma-style cut with a floor: at least 1.5x the median RMS or
            # 15 mmag above it, whichever is larger, so only genuinely
            # discrepant members are rejected.
            rms_cut = rms_med + 3.0 * max(
                1.4826 * rms_mad, 0.5 * rms_med, 0.005
            )
            bad = set(rms[rms > rms_cut].index)
            if not bad or len(ensemble_ids) - len(bad) < 3:
                break
            ensemble_ids -= bad
            log.info(
                "plot_variability_check: band %s - rejected %d variable "
                "ensemble member(s) (residual RMS > %.4f mag).",
                band,
                len(bad),
                rms_cut,
            )
        ens_epoch = _ens_epoch_stats(ensemble_ids)
        ens_mean_map = dict(zip(ens_epoch["mjd"], ens_epoch["ens_mean"]))
        ens_std_map = dict(zip(ens_epoch["mjd"], ens_epoch["ens_std"]))
        ens_n_map = dict(zip(ens_epoch["mjd"], ens_epoch["ens_n"]))

        # Brightest qualifying stars not in the ensemble form the plotted
        # reference cloud (independent of the common-mode correction).
        plot_ids = set(
            sel[~sel["star_id"].isin(ensemble_ids)]
            .head(n_ref_plot)["star_id"]
        )
        if not plot_ids:
            # Fall back to plotting ensemble members themselves: their
            # residuals are then computed against a mean containing
            # themselves, so the plotted scatter understates the true
            # reference scatter (variance biased low by ~1-1/N).
            plot_ids = set(ensemble_ids)
            log.info(
                "plot_variability_check: band %s - reference cloud contains "
                "ensemble members; its scatter is not independent of the "
                "correction.",
                band,
            )

        # Per-star differential residuals: inst - ensemble_mean - <star mean>.
        ref = cat_all[cat_all["star_id"].isin(plot_ids)].copy()
        ref["ens_mean"] = ref["mjd"].map(ens_mean_map)
        ref = ref[np.isfinite(ref["ens_mean"])]
        if ref.empty:
            continue
        ref["diff"] = ref["inst"] - ref["ens_mean"]
        ref["delta"] = ref["diff"] - ref.groupby("star_id")["diff"].transform(
            "mean"
        )
        ref_epoch = (
            ref.groupby("mjd")["delta"].agg(["mean", "std"]).reset_index()
        )

        # Target instrumental magnitudes for this band.
        if long_form and fser is not None and fser.notna().any():
            tmask = _filter_series_matches_band(band, fser)
            tgt = data[tmask.fillna(False)].copy()
        else:
            tgt = data.copy()
        cmap_t = {str(c).lower(): c for c in tgt.columns}
        inst_t_col = None
        for cand in (
            f"inst_mag_{m_low}",
            f"inst_{band}_{method_u}",
            f"inst_{band}_{m_low}",
        ):
            if cand in tgt.columns:
                inst_t_col = cand
                break
            if cand.lower() in cmap_t:
                inst_t_col = cmap_t[cand.lower()]
                break
        if inst_t_col is not None:
            tgt["inst"] = pd.to_numeric(tgt[inst_t_col], errors="coerce")
            err_cand = f"{inst_t_col}_err"
            tgt["inst_err"] = (
                pd.to_numeric(tgt[err_cand], errors="coerce")
                if err_cand in tgt.columns
                else np.nan
            )
        else:
            # Derive instrumental mag = calibrated mag - zeropoint.
            trip = _resolve_band_triplet(set(tgt.columns), band, method)
            if trip is None:
                continue
            mcol, ecol, zcol = trip
            tgt["inst"] = pd.to_numeric(tgt[mcol], errors="coerce") - pd.to_numeric(
                tgt[zcol], errors="coerce"
            )
            tgt["inst_err"] = pd.to_numeric(tgt[ecol], errors="coerce")
        tgt = tgt[
            np.isfinite(tgt["inst"]) & np.isfinite(pd.to_numeric(tgt["mjd"], errors="coerce"))
        ]
        if tgt.empty:
            continue
        tgt["mjd"] = pd.to_numeric(tgt["mjd"], errors="coerce")
        # Map target rows to catalog epochs by nearest MJD (headers and the
        # lightcurve CSV may differ at the float-rounding level). Tolerance is
        # half the minimum epoch gap, capped at ~86 s.
        epoch_vals = np.sort(ens_epoch["mjd"].to_numpy(dtype=float))
        if epoch_vals.size > 1:
            min_gap = float(np.min(np.diff(epoch_vals)))
            epoch_tol = min(1e-3, min_gap / 2.0)
        else:
            epoch_tol = 1e-3
        tgt["epoch"] = _nearest_epoch(tgt["mjd"], epoch_vals, epoch_tol)
        tgt = tgt[np.isfinite(tgt["epoch"])]
        tgt["ens_mean"] = tgt["epoch"].map(ens_mean_map)
        tgt = tgt[np.isfinite(tgt["ens_mean"])]
        if tgt.empty:
            continue
        tgt["ens_std"] = tgt["epoch"].map(ens_std_map)
        tgt["ens_n"] = tgt["epoch"].map(ens_n_map)
        tgt["diff"] = tgt["inst"] - tgt["ens_mean"]
        tgt["delta_err"] = np.sqrt(
            tgt["inst_err"].fillna(0.0) ** 2
            + (
                tgt["ens_std"].fillna(0.0)
                / np.sqrt(tgt["ens_n"].clip(lower=1))
            )
            ** 2
        )
        if max_plot_err is not None and max_plot_err > 0:
            n_before = len(tgt)
            tgt = tgt[tgt["delta_err"].fillna(0.0) <= max_plot_err]
            if len(tgt) < n_before:
                log.info(
                    "plot_variability_check: band %s - %d target point(s) "
                    "excluded (delta_err > %.2f mag).",
                    band,
                    n_before - len(tgt),
                    max_plot_err,
                )
        if tgt.empty:
            continue
        # Centre on the plotted subset so the residual and drift panels share
        # the same effective zero point.
        tgt["delta"] = tgt["diff"] - tgt["diff"].mean()

        ref_rms = float(np.nanstd(ref["delta"])) if len(ref) else np.nan
        tgt_rms = float(np.nanstd(tgt["delta"]))
        var_ratio = (
            tgt_rms / ref_rms
            if np.isfinite(ref_rms) and ref_rms > 0
            else np.nan
        )
        n_ref_plotted = int(ref["star_id"].nunique())
        log.info(
            "plot_variability_check: band %s - %d ensemble stars, %d plotted "
            "refs, %d epochs; target RMS %.4f mag, ref RMS %.4f mag, "
            "ratio %.2f.",
            band,
            len(ensemble_ids),
            n_ref_plotted,
            n_epochs,
            tgt_rms,
            ref_rms,
            var_ratio,
        )

        band_payloads.append(
            {
                "band": band,
                "ens_epoch": ens_epoch,
                "ref": ref,
                "ref_epoch": ref_epoch,
                "tgt": tgt.sort_values("mjd"),
                "n_ensemble": len(ensemble_ids),
                "n_ref": n_ref_plotted,
                "var_ratio": var_ratio,
            }
        )
        all_mjds.extend(tgt["mjd"].tolist())
        all_mjds.extend(ens_epoch["mjd"].tolist())

    if not band_payloads:
        log.info(
            "plot_variability_check: insufficient catalog/target overlap; "
            "no figure produced."
        )
        return None

    x_transform, xlabel, subday_unit = _time_axis_transform(all_mjds, 0)

    panels_per_band = 2 if show_drift_panel else 1
    n_rows = len(band_payloads) * panels_per_band
    width_in = set_size(540, aspect=1)[0]
    height_in = set_size(505, aspect=1)[1] * 0.62 * n_rows
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(width_in, height_in))
    gs = GridSpec(n_rows, 1, figure=fig, height_ratios=[1.0] * n_rows, hspace=0.12)
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]
    for a in axes[1:]:
        a.sharex(axes[0])

    for i, payload in enumerate(band_payloads):
        band = payload["band"]
        c = BAND_COLORS.get(band, BAND_COLORS.get(str(band).lower(), "k"))
        tgt = payload["tgt"]
        ens_epoch = payload["ens_epoch"]
        ref = payload["ref"]
        ref_epoch = payload["ref_epoch"]

        row = i * panels_per_band
        if show_drift_panel:
            ax_drift = axes[row]
            row += 1
            # Centre the ensemble curve over the epochs the target actually
            # covers so the two curves are directly comparable.
            _tgt_ep = ens_epoch["mjd"].isin(set(tgt["epoch"]))
            _ens_ref = ens_epoch.loc[_tgt_ep, "ens_mean"]
            _ens_c0 = (
                float(_ens_ref.mean())
                if len(_ens_ref)
                else float(ens_epoch["ens_mean"].mean())
            )
            ens_c = ens_epoch["ens_mean"] - _ens_c0
            ax_drift.plot(
                x_transform(ens_epoch["mjd"]),
                ens_c,
                "D-",
                color="dimgrey",
                markersize=3,
                lw=0.8,
                label=(
                    f"Ensemble mean ({payload['n_ensemble']} stars)"
                ),
                zorder=2,
            )
            tgt_c = tgt["inst"] - tgt["inst"].mean()
            ax_drift.errorbar(
                x_transform(tgt["mjd"]),
                tgt_c,
                yerr=tgt["inst_err"],
                fmt="o",
                color=c,
                ecolor=c,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=get_marker_size("medium"),
                capsize=get_marker_size('medium') / 4,
                lw=0.5,
                label=f"{band} target",
                zorder=3,
            )
            ax_drift.set_ylabel("Delta Instrumental Magnitude [mag]")
            ax_drift.invert_yaxis()
            ax_drift.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.5)
            ax_drift.minorticks_on()
            ax_drift.legend(loc="best", frameon=False, fontsize=7)

        ax = axes[row]
        ax.axhline(0, color="black", lw=0.5, ls="--", zorder=1)
        if not ref.empty:
            ax.scatter(
                x_transform(ref["mjd"]),
                ref["delta"],
                s=6,
                c="lightgrey",
                alpha=0.4,
                edgecolors="none",
                label=f"Reference stars ({payload['n_ref']})",
                zorder=0,
            )
        if not ref_epoch.empty:
            ax.errorbar(
                x_transform(ref_epoch["mjd"]),
                ref_epoch["mean"],
                yerr=ref_epoch["std"],
                fmt="D",
                color="whitesmoke",
                ecolor="dimgrey",
                markersize=2,
                capsize=2 / 4,
                lw=0.5,
                markeredgecolor="dimgrey",
                markeredgewidth=0.5,
                label="Per-epoch mean +/- std",
                zorder=2,
            )
        ax.errorbar(
            x_transform(tgt["mjd"]),
            tgt["delta"],
            yerr=tgt["delta_err"],
            fmt="o",
            color=c,
            ecolor=c,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=get_marker_size("medium"),
            capsize=get_marker_size("medium") / 4,
            lw=0.5,
            label=f"{band} target",
            zorder=5,
        )
        ax.set_ylabel("Residual Instrumental Magnitude [mag]")
        # Fit the y-limits to the target residuals (+/- errors) with a small
        # margin, extended to zero so the residual reference line stays in
        # view; reference-star outliers cannot stretch the axis.
        _tgt_d = np.asarray(tgt["delta"], dtype=float)
        _tgt_e = np.asarray(tgt["delta_err"], dtype=float)
        _y_lo = np.nanmin(_tgt_d - _tgt_e)
        _y_hi = np.nanmax(_tgt_d + _tgt_e)
        if not (np.isfinite(_y_lo) and np.isfinite(_y_hi)):
            _y_lo, _y_hi = -0.05, 0.05
        _y_lo = min(_y_lo, 0.0)
        _y_hi = max(_y_hi, 0.0)
        _pad = max(0.05 * (_y_hi - _y_lo), 0.01)
        ax.set_ylim(_y_hi + _pad, _y_lo - _pad)
        ax.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.5)
        ax.minorticks_on()
        if np.isfinite(payload["var_ratio"]):
            ax.text(
                0.02,
                0.03,
                f"target/ref RMS = {payload['var_ratio']:.2f}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=7,
                bbox=dict(
                    facecolor="white", alpha=0.9, edgecolor="black", linewidth=0.5
                ),
            )
        ax.legend(loc="best", frameon=False, fontsize=7)

    axes[-1].set_xlabel(xlabel)
    for a in axes[:-1]:
        a.tick_params(axis="x", labelbottom=False)
    if subday_unit == "min":
        for a in axes:
            a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    if target_name:
        fig.suptitle(str(target_name), fontsize=11, y=0.995)

    _tag = f"Variability_Check_{method_u}"
    if target_name:
        _safe_tn = (
            str(target_name)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        _tag = f"{_tag}_{_safe_tn}"
    outpath = os.path.join(save_path, f"{_tag}.{format}")
    save_kw = dict(dpi=dpi) if format.lower() != "pdf" else {}
    fig.savefig(outpath, **save_kw, bbox_inches="tight", facecolor="white")
    log.info("plot_variability_check: saved '%s'", outpath)

    if show:
        plt.show()

    return outpath


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================


def check_detection_plots(output_file, method="PSF", *, snr_limit: float = 3.0, beta_limit: float = 0.5):
    """Copy target plots into detections/nondetections folders by filter.

    Args:
        output_file (str): Path to a CSV file containing photometry rows.
        method (str): Detection method, either 'PSF' or 'AP'. Defaults to 'PSF'.

    Returns:
        str: Path to the directory where plots are saved, or None if no output file.
    """
    if output_file is None:
        return None

    try:
        data = pd.read_csv(output_file)
        data = _normalize_photometry_columns(data)
        data = data.copy()
        data["is_detection"] = True
        # If called with Detections_*.csv and a matching Nondetections_*.csv exists,
        # include those rows so folder split is fully populated.
        base_name = os.path.basename(output_file)
        if base_name.startswith("Detections_"):
            nondet_name = "Nondetections_" + base_name[len("Detections_") :]
            nondet_path = os.path.join(os.path.dirname(output_file), nondet_name)
            if os.path.isfile(nondet_path):
                nd = pd.read_csv(nondet_path)
                nd = _normalize_photometry_columns(nd)
                nd = nd.copy()
                nd["is_detection"] = False
                data = pd.concat([data, nd], ignore_index=True)
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Failed to read detections table '%s': %s", output_file, exc, exc_info=True
        )
        return None

    save_path = os.path.join(os.path.dirname(output_file), f"Detections_{method}")
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    det_root = os.path.join(save_path, "Detections")
    nondet_root = os.path.join(save_path, "Nondetections")
    pathlib.Path(det_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(nondet_root).mkdir(parents=True, exist_ok=True)

    prefix_map = {"AP": "Aperture_", "PSF": "PSF_Target_"}
    prefix = prefix_map.get(method, "")

    band_counter = Counter()
    log = logging.getLogger(__name__)

    def _row_detection_state(row_obj) -> bool:
        """
        Best-effort detection classification for one photometry row.

        Prefer an explicit is_detection/detected flag when present, falling back
        to SNR-based inference only when the flag is absent. This keeps the
        classification in sync with the per-image detection decision made by
        the pipeline.
        """
        # Prefer explicit detection flags from upstream (e.g. main.py)
        for k in ("is_detection", "detected", "is_detected"):
            if k in row_obj and pd.notna(row_obj.get(k)):
                v = row_obj.get(k)
                if isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in {"true", "t", "1", "yes", "y", "det"}:
                        return True
                    if vv in {"false", "f", "0", "no", "n", "nondet", "limit"}:
                        return False
                try:
                    return bool(int(v))
                except Exception:
                    return bool(v)

        # Fallback: recompute from SNR if explicit flag is absent
        try:
            method_u = str(method).upper()
            snr_val = np.nan
            if method_u == "PSF":
                for k in ("snr_psf", "SNR_PSF"):
                    if k in row_obj and pd.notna(row_obj.get(k)):
                        snr_val = float(row_obj.get(k))
                        break
                if not np.isfinite(snr_val):
                    # flux-based fallback
                    fp = row_obj.get("flux_psf", row_obj.get("flux_PSF", np.nan))
                    fe = row_obj.get("flux_psf_err", row_obj.get("flux_PSF_err", np.nan))
                    fp = float(fp) if pd.notna(fp) else np.nan
                    fe = float(fe) if pd.notna(fe) else np.nan
                    if np.isfinite(fp) and np.isfinite(fe) and fe > 0:
                        snr_val = np.abs(fp) / fe  # Use absolute for negative fits
            else:
                for k in ("snr_ap", "SNR_AP", "snr", "SNR"):
                    if k in row_obj and pd.notna(row_obj.get(k)):
                        snr_val = float(row_obj.get(k))
                        break
                if not np.isfinite(snr_val):
                    fa = row_obj.get("flux_ap", row_obj.get("flux_AP", np.nan))
                    fe = row_obj.get("flux_ap_err", row_obj.get("flux_AP_err", np.nan))
                    fa = float(fa) if pd.notna(fa) else np.nan
                    fe = float(fe) if pd.notna(fe) else np.nan
                    if np.isfinite(fa) and np.isfinite(fe) and fe > 0:
                        snr_val = np.abs(fa) / fe

            if np.isfinite(snr_val):
                return bool(snr_val >= float(snr_limit))
        except Exception:
            pass

        # If this function is called with detections_<...>.csv, rows are detections by construction.
        return True

    # Process detections first so ambiguous files are preferentially assigned
    # to detections and never duplicated into nondetections.
    if "is_detection" in data.columns:
        data = data.sort_values(by="is_detection", ascending=False).reset_index(drop=True)

    assigned_group_by_src = {}
    copied_pairs = set()

    for row in data.to_dict("records"):
        try:
            # `filename` may be either a base stem (new) or a full path (legacy).
            # Prefer `filename_path` when present so we can locate plot files.
            filename_path = row.get("filename_path", None)
            if (
                isinstance(filename_path, str)
                and filename_path
                and filename_path.strip().lower() not in {"nan", "none"}
            ):
                loc = os.path.dirname(filename_path)
            else:
                fn = row.get("filename", "")
                if isinstance(fn, str) and fn.strip().lower() not in {"nan", "none"}:
                    loc = os.path.dirname(fn)
                else:
                    loc = ""

            # Prefer raster/vector plot outputs over other artifacts.
            prefixes = [prefix]
            if method == "PSF":
                # Support both legacy (`targetPSF_`) and current (`PSF_Target_`) plot prefixes.
                prefixes = ["PSF_Target_", "targetPSF_"]

            candidates = []
            for pfx in prefixes:
                search = os.path.join(loc, f"{pfx}*")
                candidates.extend(glob.glob(search))

            pdfs = [f for f in candidates if f.lower().endswith((".png", ".svg"))]
            files = sorted(pdfs)
            if not files:
                continue

            date = "".join(str(row.get("date", "")).split("-"))
            band = str(row.get("filter", "unknown")).strip() or "unknown"
            is_detection = _row_detection_state(row)
            split_root = det_root if is_detection else nondet_root
            band_dir = os.path.join(split_root, band)
            pathlib.Path(band_dir).mkdir(parents=True, exist_ok=True)

            # Pick the best-matching plot for this row to avoid accidental
            # cross-copying between rows in the same directory.
            row_stem = None
            fn_path = row.get("filename_path", None)
            if isinstance(fn_path, str) and fn_path.strip() and fn_path.strip().lower() not in {"nan", "none"}:
                row_stem = os.path.splitext(os.path.basename(fn_path.strip()))[0]
            else:
                fn = row.get("filename", None)
                if isinstance(fn, str) and fn.strip() and fn.strip().lower() not in {"nan", "none"}:
                    row_stem = os.path.splitext(os.path.basename(fn.strip()))[0]

            date_token = date if len(date) == 8 else None
            band_token = str(band).lower()

            def _score_candidate(path):
                base = os.path.basename(path).lower()
                score = 0
                if row_stem:
                    rs = row_stem.lower()
                    if rs in base:
                        score += 4
                if date_token and date_token in base:
                    score += 2
                if band_token and (
                    f"_{band_token}_" in base
                    or base.startswith(f"{band_token}_")
                    or f"{band_token}band_" in base
                ):
                    score += 1
                return score

            scored = sorted(((_score_candidate(f), f) for f in files), key=lambda t: (t[0], t[1]), reverse=True)
            best_score, src_file = scored[0]
            if len(files) > 1 and best_score <= 0:
                # Ambiguous: multiple candidates and no match signal -> skip.
                continue

            current_group = "detections" if is_detection else "nondetections"
            existing_group = assigned_group_by_src.get(src_file)
            if existing_group is not None and existing_group != current_group:
                # Never allow the same source plot into both trees.
                # Prefer detections when conflict occurs.
                if existing_group == "detections" and current_group == "nondetections":
                    continue
            assigned_group_by_src[src_file] = current_group

            dedupe_key = (src_file, current_group, band)
            if dedupe_key in copied_pairs:
                continue

            dest_file = os.path.join(
                band_dir, f"{band}band_{date}_{os.path.basename(src_file)}"
            )
            shutil.copyfile(src_file, dest_file)
            copied_pairs.add(dedupe_key)
            key = ("detections" if is_detection else "nondetections", band)
            band_counter[key] += 1
        except (IndexError, KeyError, AttributeError) as exc:
            log.warning(
                "Skipping detections row due to missing or malformed fields: %s",
                exc,
            )
            continue

    for (group_name, band), count in sorted(band_counter.items()):
        log.info("check_detection_plots: %s/%s -> %d file(s)", group_name, band, count)

    return save_path
