import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import glob
import shutil
from functions import set_size, get_distance_modulus
from astropy.time import Time
from collections import Counter

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
}

# Colours for plots (per-band). Keep these in sync with the palette user-supplied.
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
    "F": "hotpink",
    "N": "magenta",
    "o": "darkorange",
    "c": "cyan",
    "W": "forestgreen",
    "Q": "peru",
}

BAND_COLORS = cols

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
    "B-R": "#00CED1",
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
    cand = [
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
    """Convert pipeline ``lmag`` (instrumental) to apparent using band zeropoint."""
    if "lmag" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    li = pd.to_numeric(df["lmag"], errors="coerce")
    zp = pd.to_numeric(df[zp_col], errors="coerce")
    return li + zp


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
    - SNR cut (if use_SNR_limit) or mag < lmag (otherwise).
    """
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float, copy=False)
    err = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float, copy=False)
    lmag = (
        pd.to_numeric(df["lmag"], errors="coerce").to_numpy(dtype=float, copy=False)
        if "lmag" in df.columns
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
    if use_SNR_limit:
        if method_u == "PSF" and "snr_psf" in df.columns:
            snr = pd.to_numeric(df["snr_psf"], errors="coerce").to_numpy(dtype=float, copy=False)
        elif method_u == "AP" and "snr_ap" in df.columns:
            snr = pd.to_numeric(df["snr_ap"], errors="coerce").to_numpy(dtype=float, copy=False)
        elif "snr" in df.columns:
            snr = pd.to_numeric(df["snr"], errors="coerce").to_numpy(dtype=float, copy=False)
        elif method_u == "PSF" and "flux_psf" in df.columns and "flux_psf_err" in df.columns:
            flux = pd.to_numeric(df["flux_psf"], errors="coerce").to_numpy(dtype=float, copy=False)
            flux_err = pd.to_numeric(df["flux_psf_err"], errors="coerce").to_numpy(dtype=float, copy=False)
            snr = np.divide(
                flux,
                flux_err,
                out=np.full(len(df), np.nan, dtype=float),
                where=(flux_err > 0) & np.isfinite(flux_err),
            )
        elif method_u == "AP" and "flux_ap" in df.columns and "flux_ap_err" in df.columns:
            flux = pd.to_numeric(df["flux_ap"], errors="coerce").to_numpy(dtype=float, copy=False)
            flux_err = pd.to_numeric(df["flux_ap_err"], errors="coerce").to_numpy(dtype=float, copy=False)
            snr = np.divide(
                flux,
                flux_err,
                out=np.full(len(df), np.nan, dtype=float),
                where=(flux_err > 0) & np.isfinite(flux_err),
            )
        else:
            # Last resort: infer SNR from magnitude uncertainty.
            snr = _snr_from_magerr(err)
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
    else:
        detected = np.isfinite(mag) & np.isfinite(err) & (mag < lmag)

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
        MJD reference for phase; 0 means x-axis is MJD.
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

    Returns
    -------
    str or None
        Path to detections CSV if return_detections and detections exist; else None.
    """
    today_mjd = None
    if mark_today:
        today = Time.now()
        today_mjd = today.mjd

    dm = get_distance_modulus(redshift) if redshift else 0
    # Use the shared per-band palette.
    base_cols = BAND_COLORS
    # Band plotting order (exclude Gaia G so it is not conflated with SDSS g).
    band_order = "FSDNAuUBgcVwrRoEiIzyYJHKWQ"
    cols = {b: base_cols.get(b, BAND_COLORS.get(b, "k")) for b in band_order}
    data = pd.read_csv(output_file)
    data = _normalize_photometry_columns(data)
    # If the CSV has duplicate column names (can happen after concatenation or
    # normalization), pandas returns a DataFrame for `df["col"]`, which breaks
    # numeric coercion and boolean logic. Keep the first occurrence.
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()].copy()
    save_path = os.path.dirname(output_file)
    base = os.path.splitext(os.path.basename(output_file))[0]

    def _resolve_band_triplet(cols, band: str, method: str):
        """
        Resolve the (mag, err, zp) column names for a band/method.
        Supports both legacy lowercased columns (e.g. g_psf) and case-preserving
        columns (e.g. g_PSF / R_PSF) without conflating r and R when both exist.
        Also supports inverted counterparts (e.g. g_psf_inverted).
        """
        # Prefer exact-case first, then lowercase legacy.
        cand = [
            (f"{band}_{method}", f"{band}_{method}_err", f"zp_{band}_{method}"),
            (
                f"{band}_{method}".lower(),
                f"{band}_{method}_err".lower(),
                f"zp_{band}_{method}".lower(),
            ),
        ]
        # Also check for inverted versions
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
        # Check for inverted version if normal not found
        for trip in cand_inv:
            if all(c in cols for c in trip):
                return trip
        return None

    # Discover which bands are actually present in the table, ensuring we don't
    # accidentally map multiple band labels (e.g. r and R) onto the same triplet.
    bands_in_data = []
    used_triplets = set()
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

    # Publication-ready figure size: single-column width (505 pt), aspect ~0.6
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

    # Font sizes suitable for single-column journal figure
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
    mid_idx = (
        len(bands_in_data) - 1
    ) // 2  # reference band (offset 0) is the middle one

    for idx, band in enumerate(bands_in_data):
        ax = curve_axes[0] if single_plot else curve_axes[idx]
        triplet = _resolve_band_triplet(set(data.columns), band, method)
        if triplet is None:
            continue
        col, err_col, zp_col = triplet
        df = data[np.isfinite(data[zp_col])].copy()
        df.sort_values(by="mjd", inplace=True)
        # Check if col is already an apparent magnitude (has zeropoint applied)
        # If it starts with 'inst_' it's instrumental, otherwise it's apparent
        if col.startswith('inst_'):
            # Convert instrumental magnitude to apparent magnitude
            df["apparent_mag"] = df[col] + df[zp_col]
        else:
            # Already apparent magnitude, just use it
            df["apparent_mag"] = df[col]
        df["apparent_mag_err"] = df[err_col]

        band_offset = (idx - mid_idx) * offset
        df["apparent_mag"] = df["apparent_mag"] + band_offset
        # lmag in photometry CSV is instrumental (see main.py); convert to apparent
        # so upper-limit points match the detection magnitude scale on the plot.
        # Note: band_offset is NOT applied to lmag as it's used for detection thresholds,
        # not for visual plot positioning.
        if "lmag" not in df.columns:
            df["lmag"] = np.nan
        df["lmag"] = _lmag_to_apparent(df, zp_col)
        
        # Initialize plot_mag and plot_err for this band's df
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

        # Check for inverted-only detections using _inverted_fit flag or inst_inverted column
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

        # Combine normal detections and inverted-only detections for plotting
        all_detects = df[normal_detected | inverted_only]
        nondetects = df[~detected_s & ~inverted_only]
        
        num_detect += len(all_detects)  # Count all detections
        num_nondetect += len(nondetects)
        
        if return_detections and not all_detects.empty:
            detections_list.append(all_detects)
        if return_detections and not nondetects.empty:
            nondetections_list.append(nondetects)

        c = cols.get(band, "k")
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
        
        # For normal detections, use apparent_mag
        df.loc[normal_detected, "plot_mag"] = df.loc[normal_detected, "apparent_mag"]
        df.loc[normal_detected, "plot_err"] = df.loc[normal_detected, "apparent_mag_err"]
        
        # Prepare magnitude columns for inverted-only detections
        if has_inverted and np.any(inverted_only):
            # Check for band-specific inverted apparent magnitude first
            band_inv_mag_col = f"{band}_{method}_inverted" if f"{band}_{method}_inverted" in df.columns else None
            band_inv_err_col = f"{band}_{method}_err_inverted" if f"{band}_{method}_err_inverted" in df.columns else None
            
            # Fallback to generic inst_inverted if band-specific not available
            inv_mag_col = band_inv_mag_col if band_inv_mag_col else ("inst_inverted" if "inst_inverted" in df.columns else inverted_col)
            inv_err_col = band_inv_err_col if band_inv_err_col else ("inst_inverted_err" if "inst_inverted_err" in df.columns else None)
            
            # For inverted-only detections, we need to use inverted magnitudes
            if inv_mag_col and inv_mag_col in df.columns:
                # Check if this is already an apparent magnitude (band-specific inverted column)
                is_apparent = band_inv_mag_col is not None
                if not is_apparent and zp_col in df.columns:
                    # Convert instrumental to apparent
                    df.loc[inverted_only, "inv_apparent_mag"] = df.loc[inverted_only, inv_mag_col] + df.loc[inverted_only, zp_col] + band_offset
                    if inv_err_col and inv_err_col in df.columns:
                        df.loc[inverted_only, "inv_apparent_mag_err"] = df.loc[inverted_only, inv_err_col]
                    else:
                        df.loc[inverted_only, "inv_apparent_mag_err"] = np.nan
                    df.loc[inverted_only, "plot_mag"] = df.loc[inverted_only, "inv_apparent_mag"]
                    df.loc[inverted_only, "plot_err"] = df.loc[inverted_only, "inv_apparent_mag_err"]
                else:
                    # Already apparent magnitude
                    df.loc[inverted_only, "plot_mag"] = df.loc[inverted_only, inv_mag_col] + band_offset
                    df.loc[inverted_only, "plot_err"] = df.loc[inverted_only, inv_err_col]
        
        # First, plot error bars for all detections
        if not all_detects.empty:
            ax.errorbar(
                all_detects.mjd - reference_epoch,
                all_detects["plot_mag"],
                yerr=all_detects["plot_err"],
                fmt='none',  # no markers here
                ecolor=c,
                capsize=2,
                capthick=0.8,
                elinewidth=1,
                zorder=2,
            )

            # Optional: draw line connecting detection points
            if ls:
                # Sort by MJD for proper line connection
                sorted_detects = all_detects.sort_values("mjd")
                ax.plot(
                    sorted_detects.mjd - reference_epoch,
                    sorted_detects["plot_mag"],
                    color=c,
                    linestyle=ls,
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=1,
                )

        # Now plot markers, applying hatch style row by row for inverted fits
        if not all_detects.empty:
            if "_inverted_fit" in all_detects.columns:
                inv_fit_mask = _inverted_fit_series_to_bool(all_detects["_inverted_fit"])
            else:
                inv_fit_mask = pd.Series(False, index=all_detects.index)
            # Align inverted_only (full-band df index) to plotted rows — required for
            # correct hatch assignment (avoids index misalignment with | on Series).
            inv_row_mask = inverted_only.reindex(all_detects.index, fill_value=False)
            inv_row_mask = inv_row_mask.fillna(False).astype(bool)
            inv_fit_mask = inv_fit_mask | inv_row_mask

            # Plot normal detections (no hatch)
            normal_detects = all_detects[~inv_fit_mask]
            if not normal_detects.empty:
                plotted_positive_flux_marker = True
                ax.scatter(
                    normal_detects.mjd - reference_epoch,
                    normal_detects["plot_mag"],
                    s=80,
                    c=c,
                    marker='o',  # normal circular marker
                    edgecolors='black',
                    linewidth=0.8,
                    zorder=3,
                    label=leg_label if leg_label else "",
                )
            
            # Plot inverted detections with hatch
            inv_detects = all_detects[inv_fit_mask]
            if not inv_detects.empty:
                plotted_inverted_hatch = True
                sc = ax.scatter(
                    inv_detects.mjd - reference_epoch,
                    inv_detects["plot_mag"],
                    s=100,  # slightly larger marker size for visibility
                    c=c,   # face color
                    marker='s',  # square marker (patch) so hatch works
                    edgecolors='black',
                    linewidth=0.8,
                    zorder=3,  # markers on top of error bars
                )
                # Add diagonal stripes - only for inverted PSF fits (white hatch)
                sc.set_hatch('////')
                sc.set_edgecolor('white')
                # Add label manually without marker in legend (only if not already added)
                if normal_detects.empty:
                    ax.scatter([], [], s=100, c=c, marker='s', edgecolors='white', 
                             linewidth=0.8, label=leg_label if leg_label else "")

        if show_limits and not nondetects.empty:
            has_limits_plotted = True
            ax.errorbar(
                nondetects.mjd - reference_epoch,
                nondetects["lmag"] + band_offset,
                color=c,
                markeredgecolor=c,
                markerfacecolor="none",
                markeredgewidth=0.5,
                ls="",
                marker="v",
                markersize=5,
                alpha=0.85,
                zorder=1,
            )

        if not single_plot:
            ax.set_ylabel(f"{band} (mag)")
            ax.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.5)
            ax.minorticks_on()

    xlabel = (
        f"Phase (days since {reference_epoch})" if reference_epoch else "Time [MJD]"
    )
    curve_axes[0].set_ylabel("Apparent brightness [mag]")
    (color_ax if plot_color else curve_axes[-1]).set_xlabel(xlabel)
    if plot_color:
        for ax in curve_axes:
            ax.tick_params(axis="x", labelbottom=False, length=0)

    for ax in curve_axes:
        ax.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.5)
        ax.minorticks_on()

    if mark_today and today_mjd is not None:
        today_rel = today_mjd - reference_epoch
        for ax in curve_axes:
            ax.axvline(
                x=today_rel,
                color="#FF0000",
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
        ax2.set_ylabel("Absolute brightness [mag]")
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
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.5,
            ls="",
            label="Upper limit",
        )
        handles.append(limit_handle)
        labels.append("Upper limit")

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
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        ncol=ncol,
    )
    ax0.add_artist(leg_main)

    # Second legend: flux sign only, same marker geometry as the lightcurve points.
    if plotted_positive_flux_marker and plotted_inverted_hatch:
        _fc = "0.45"
        h_pos = ax0.scatter(
            [],
            [],
            s=80,
            marker="o",
            facecolors=_fc,
            edgecolors="black",
            linewidths=0.8,
        )
        h_neg = ax0.scatter(
            [],
            [],
            s=100,
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
            frameon=True,
            framealpha=0.95,
            edgecolor="black",
        )

    # No plot titles; keep the legend/axis labels only.

    ax0.invert_yaxis()

    # ---------- Colour evolution panel (same-night pairs only) ----------
    if plot_color and color_ax is not None and "filter" in data.columns:
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
            fcol = (
                data["filter"].astype(str).str.lower()
                if "filter" in data.columns
                else None
            )
            d1 = (
                data[(fcol == b1.lower()) & np.isfinite(data[zp1])].copy()
                if fcol is not None
                else pd.DataFrame()
            )
            d2 = (
                data[(fcol == b2.lower()) & np.isfinite(data[zp2])].copy()
                if fcol is not None
                else pd.DataFrame()
            )
            d1["mag"] = d1[col1] + d1[zp1] if col1.startswith('inst_') else d1[col1]
            d1["err"] = d1[err1]
            d2["mag"] = d2[col2] + d2[zp2] if col2.startswith('inst_') else d2[col2]
            d2["err"] = d2[err2]
            if "lmag" not in d1.columns:
                d1["lmag"] = np.nan
            if "lmag" not in d2.columns:
                d2["lmag"] = np.nan
            d1["lmag"] = _lmag_to_apparent(d1, zp1)
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
                        d1["flux_psf"],
                        d1["flux_psf_err"],
                        out=np.full(len(d1), np.nan),
                        where=(np.asarray(d1["flux_psf_err"]) > 0)
                        & np.isfinite(d1["flux_psf_err"]),
                    )
                    snr2 = np.divide(
                        d2["flux_psf"],
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
            # Color = b1 - b2. Limit direction: b1 det + b2 limit -> true color <= color_value (upper limit, v);
            # b1 limit + b2 det -> true color >= color_value (lower limit, ^).
            phase_pts, color_pts, err_pts = [], [], []
            phase_ul, color_ul = [], []  # upper limit on color (downward triangle v)
            phase_ll, color_ll = [], []  # lower limit on color (upward triangle ^)
            for _, row in d1.iterrows():
                mjd1 = row["mjd"]
                dt = np.abs(mjd2 - mjd1)
                if np.min(dt) > color_match_days:
                    continue
                j = np.argmin(dt)
                phase = (mjd1 + mjd2[j]) / 2 - reference_epoch
                det1 = row["det"]
                if det1 and det2[j]:
                    phase_pts.append(phase)
                    color_pts.append(row["mag"] - mag2[j])
                    err_pts.append(np.sqrt(row["err"] ** 2 + err2_arr[j] ** 2))
                elif det1 and not det2[j]:
                    # b1 detected, b2 limit: true b2 >= lmag2 => color <= mag1 - lmag2 -> upper limit (v)
                    phase_ul.append(phase)
                    color_ul.append(row["mag"] - lmag2[j])
                elif not det1 and det2[j]:
                    # b1 limit, b2 detected: true b1 >= lmag1 => color >= lmag1 - mag2 -> lower limit (^)
                    phase_ll.append(phase)
                    color_ll.append(row["lmag"] - mag2[j])
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
                    markersize=4,
                    ls="",
                    capsize=1.5,
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
                    markersize=4,
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
                    markersize=4,
                    ls="",
                    zorder=2,
                )
        color_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        color_ax.invert_yaxis()
    # fig.tight_layout()

    outname = f'LightCurve_{method}_{"single" if single_plot else "subplots"}.{format}'
    outpath = os.path.join(save_path, outname)
    save_kw = dict(dpi=dpi) if format.lower() != "pdf" else {}
    plt.savefig(outpath, **save_kw, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")

    det_file = None
    if return_detections and detections_list:
        valid_detections = [df for df in detections_list if not df.empty]
        if valid_detections:
            det_file = os.path.join(save_path, f"detections_{base}_{method}.csv")
            pd.concat(valid_detections, ignore_index=True).to_csv(det_file, index=False)
    if return_detections and nondetections_list:
        valid_nondetections = [df for df in nondetections_list if not df.empty]
        if valid_nondetections:
            nondet_file = os.path.join(save_path, f"nondetections_{base}_{method}.csv")
            pd.concat(valid_nondetections, ignore_index=True).to_csv(
                nondet_file, index=False
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
    """
    complete_data = pd.read_csv(output_file)
    complete_data = _normalize_photometry_columns(complete_data)
    if complete_data.columns.duplicated().any():
        complete_data = complete_data.loc[:, ~complete_data.columns.duplicated()].copy()
    phot_table = []
    # Prefer using the per-row filter column if present, so we don't duplicate
    # bands (e.g. 'g' and 'G') and we keep the dataset's actual filter names.
    if "filter" in complete_data.columns:
        _filters = complete_data["filter"].dropna().astype(str).map(str.strip)
        bands = [b for b in _filters.unique().tolist() if b]
        # Order by effective wavelength when known; otherwise keep input order.
        bands = sorted(
            bands,
            key=lambda b: BAND_WAVELENGTHS.get(
                b, BAND_WAVELENGTHS.get(b.lower(), 1e12)
            ),
        )
    else:
        # Fallback when no filter column exists.
        bands = list("FSDNAuUBgcVwrRoEiIzyYJHKWQ")

    used_triplets = set()
    for band in bands:
        trip = _resolve_band_triplet(set(complete_data.columns), band, method)
        if trip is None:
            continue
        col, err_col, zp_col = trip
        if trip in used_triplets:
            continue
        used_triplets.add(trip)

        data = complete_data[np.isfinite(complete_data[zp_col])].copy()
        if "filter" in data.columns:
            f = data["filter"].astype(str).str.strip()
            # Match filter name case-insensitively but preserve original label in output.
            data = data[f.str.lower() == str(band).lower()].copy()
            if data.empty:
                continue
        # Robust numeric coercion: CSV concatenation can yield strings like "nan".
        if "beta" in data.columns:
            data["beta"] = pd.to_numeric(data["beta"], errors="coerce")
        if "lmag" not in data.columns:
            data["lmag"] = np.nan
        lmag_inst = pd.to_numeric(data["lmag"], errors="coerce")
        zp_num = pd.to_numeric(data[zp_col], errors="coerce")
        # Pipeline stores lmag in instrumental system; detection cut uses apparent mags.
        data["lmag"] = lmag_inst + zp_num
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

        # Check for inverted-only detections using _inverted_fit flag or inst_inverted column
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
            # Check if col is already an apparent magnitude (has zeropoint applied)
            # If it starts with 'inst_' it's instrumental, otherwise it's apparent
            if col.startswith('inst_'):
                # Convert instrumental magnitude to apparent magnitude for the table
                mag_values = detects[col] + detects[zp_col]
            else:
                # Already apparent magnitude
                mag_values = detects[col]
            
            detects = detects.assign(
                Filter=band,
                Limit="-",
                MJD=detects["mjd"].round(3),
                Date=Time(detects["mjd"], format="mjd").iso,
                Mag=mag_values.round(3),
                Error=detects[err_col].round(3),
            )[["MJD", "Date", "Mag", "Error", "Filter", "Limit"]]
            phot_table.append(detects)

        if not inv_detects.empty:
            inv_err_col = err_col if err_col in inv_detects.columns else None
            # Check for band-specific inverted apparent magnitude first
            band_inv_mag_col = f"{band}_{method}_inverted" if f"{band}_{method}_inverted" in inv_detects.columns else None
            band_inv_err_col = f"{band}_{method}_err_inverted" if f"{band}_{method}_err_inverted" in inv_detects.columns else None
            
            # Use band-specific inverted apparent magnitude if available, otherwise convert instrumental
            if band_inv_mag_col and band_inv_mag_col in inv_detects.columns:
                inv_mag_col = band_inv_mag_col
                inv_err_col = band_inv_err_col
                is_apparent = True
            else:
                inv_mag_col = "inst_inverted" if "inst_inverted" in inv_detects.columns else inverted_col
                inv_err_col = "inst_inverted_err" if "inst_inverted_err" in inv_detects.columns else None
                is_apparent = False
            
            if inv_mag_col and inv_mag_col in inv_detects.columns:
                if is_apparent:
                    # Already apparent magnitude
                    mag_value = inv_detects[inv_mag_col]
                else:
                    # Convert instrumental to apparent
                    mag_value = inv_detects[inv_mag_col] + inv_detects[zp_col]
                
                _err_s = (
                    pd.to_numeric(inv_detects[inv_err_col], errors="coerce").round(3)
                    if inv_err_col and inv_err_col in inv_detects.columns
                    else pd.Series(np.nan, index=inv_detects.index, dtype=float)
                )
                inv_detects = inv_detects.assign(
                    Filter=band,
                    Limit=pd.to_numeric(inv_detects["lmag"], errors="coerce").round(
                        3
                    ),
                    MJD=inv_detects["mjd"].round(3),
                    Date=Time(inv_detects["mjd"], format="mjd").iso,
                    Mag=mag_value.round(3),
                    Error=_err_s,
                )[["MJD", "Date", "Mag", "Error", "Filter", "Limit"]]
                phot_table.append(inv_detects)

        if not nondetects.empty:
            nondetects = nondetects.assign(
                Filter=band,
                MJD=nondetects["mjd"].round(3),
                Date=Time(nondetects["mjd"], format="mjd").iso,
                Mag=pd.Series(np.nan, index=nondetects.index, dtype=float),
                Error=pd.Series(np.nan, index=nondetects.index, dtype=float),
                Limit=pd.to_numeric(nondetects["lmag"], errors="coerce").round(3),
            )[["MJD", "Date", "Mag", "Error", "Filter", "Limit"]]
            phot_table.append(nondetects)

    if not phot_table:
        out_phot = pd.DataFrame(
            columns=["MJD", "Date", "Mag", "Error", "Filter", "Limit"]
        )
    else:
        out_phot = pd.concat(phot_table, ignore_index=True)
        if reference_epoch:
            out_phot.insert(2, "Phase", (out_phot["MJD"] - reference_epoch).round(3))
        out_phot.sort_values(["MJD", "Filter"], inplace=True)
        out_phot.reset_index(drop=True, inplace=True)

    save_path = os.path.dirname(output_file)
    fname = os.path.join(save_path, f"lightcurve_{method}.dat")
    # Uniform numeric formatting; non-detections use NaN for Mag/Error (written as empty or "-")
    out_phot.to_csv(fname, index=False, float_format="%.3f", na_rep="-")

    # ---------- Colour evolution table (same-night pairs) ----------
    color_table_path = None
    if (
        include_color_table
        and "filter" in complete_data.columns
        and len(complete_data) > 1
    ):
        color_rows = []
        for b1, b2 in _COLOR_PAIRS:
            t1 = _resolve_band_triplet(set(complete_data.columns), b1, method)
            t2 = _resolve_band_triplet(set(complete_data.columns), b2, method)
            if t1 is None or t2 is None:
                continue
            col1, err1, zp1 = t1
            col2, err2, zp2 = t2
            fcol = complete_data["filter"].astype(str).str.lower()
            d1 = complete_data[
                (fcol == b1.lower()) & np.isfinite(complete_data[zp1])
            ].copy()
            d2 = complete_data[
                (fcol == b2.lower()) & np.isfinite(complete_data[zp2])
            ].copy()
            if d1.empty or d2.empty:
                continue
            if "lmag" not in d1.columns:
                d1["lmag"] = np.nan
            if "lmag" not in d2.columns:
                d2["lmag"] = np.nan
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
                        d1["flux_psf"],
                        d1["flux_psf_err"],
                        out=np.full(len(d1), np.nan),
                        where=(np.asarray(d1["flux_psf_err"]) > 0)
                        & np.isfinite(d1["flux_psf_err"]),
                    )
                    snr2 = np.divide(
                        d2["flux_psf"],
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
            for _, row in d1.iterrows():
                dt = np.abs(mjd2 - row["mjd"])
                if np.min(dt) > color_match_days:
                    continue
                j = np.argmin(dt)
                mjd_mid = (row["mjd"] + mjd2[j]) / 2
                date_str = Time(mjd_mid, format="mjd").iso
                phase = (mjd_mid - reference_epoch) if reference_epoch else np.nan
                color_name = f"{b1}-{b2}"
                if row["det"] and det2[j]:
                    color_value = row["mag"] - mag2[j]
                    err_val = np.sqrt(row["err"] ** 2 + err2_arr[j] ** 2)
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
                elif row["det"] and not det2[j]:
                    color_value = row["mag"] - lmag2[j]
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
                elif not row["det"] and det2[j]:
                    color_value = row["lmag"] - mag2[j]
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
                save_path, f"lightcurve_{method}_colors.dat"
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
        # If called with detections_*.csv and a matching nondetections_*.csv exists,
        # include those rows so folder split is fully populated.
        base_name = os.path.basename(output_file)
        if base_name.startswith("detections_"):
            nondet_name = "nondetections_" + base_name[len("detections_") :]
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

    save_path = os.path.join(os.path.dirname(output_file), f"detections_{method}")
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    det_root = os.path.join(save_path, "detections")
    nondet_root = os.path.join(save_path, "nondetections")
    pathlib.Path(det_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(nondet_root).mkdir(parents=True, exist_ok=True)

    prefix_map = {"AP": "aperture_", "PSF": "targetPSF_"}
    prefix = prefix_map.get(method, "")

    band_counter = Counter()
    log = logging.getLogger(__name__)

    def _row_detection_state(row_obj) -> bool:
        """
        Best-effort detection classification for one photometry row.

        Prefer an explicit is_detection/detected flag when present, but override
        it when the row contains enough photometric fields to recompute the
        detection rule (guards against stale/missing columns in large runs).
        """
        # If we have enough information, recompute detection state using the
        # same logic as plot_lightcurve() (SNR gate + optional beta).
        try:
            # beta: do not let missing/low beta veto detections with good SNR
            beta = row_obj.get("beta", np.nan)
            try:
                beta = float(beta)
            except Exception:
                beta = np.nan

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
                        snr_val = fp / fe
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
                        snr_val = fa / fe

            # Now compute beta_ok after snr_val is known
            beta_available = np.isfinite(beta)
            # Beta is informational for detection classification. For actual
            # photometric detection, SNR is the primary criterion. A source with
            # SNR >= limit is a detection regardless of beta value.
            beta_ok = (not beta_available) or (beta > float(beta_limit)) or (snr_val >= float(snr_limit))

            if np.isfinite(snr_val):
                detected_by_snr = snr_val >= float(snr_limit)
                return bool(detected_by_snr and beta_ok)
        except Exception:
            pass

        # Fall back to explicit flags when present.
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
        # If this function is called with detections_<...>.csv, rows are detections by construction.
        return True

    # Process detections first so ambiguous files are preferentially assigned
    # to detections and never duplicated into nondetections.
    if "is_detection" in data.columns:
        data = data.sort_values(by="is_detection", ascending=False).reset_index(drop=True)

    assigned_group_by_src = {}
    copied_pairs = set()

    for _, row in data.iterrows():
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

            # Prefer new PNG names when both PNG and PDF exist.
            prefixes = [prefix]
            if method == "PSF":
                # Support both legacy (`targetPSF_`) and new (`PSF_Target_`) plot prefixes.
                prefixes = ["PSF_Target_", "targetPSF_"]

            candidates = []
            for pfx in prefixes:
                search = os.path.join(loc, f"{pfx}*")
                candidates.extend(glob.glob(search))

            pngs = [f for f in candidates if f.lower().endswith(".png")]
            files = sorted(pngs)
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
