"""
Centralized plotting utilities for consistent figure formatting across autophot.
Color scheme follows Plante & Cushman (2020), "Choosing color palettes for
scientific figures" (Res Pract Thromb Haemost, doi:10.1002/rth2.12308): a
small designed palette (the RPTH palette) reused consistently across all
figures, extended with tints/shades of the core hues where extra categories
are needed, and high lightness contrast for colorblind accessibility.
"""

import numpy as np
from typing import Optional, Union

# RPTH palette (Plante & Cushman 2020, Table 1) plus derived swatches.
# The paper recommends pairing a base color with swatches of that color
# (e.g. navy + sky blue) rather than unrelated hues, so the extra named
# shades below are tints/shades of the six core colors.
RPTH = {
    'blue':          '#003366',  # RPTH Blue (navy)
    'medium_blue':   '#005CAB',  # RPTH Medium Blue
    'light_blue':    '#DCEEF3',  # RPTH Light Blue
    'red':           '#E31B23',  # RPTH Red
    'accent_yellow': '#FFC325',  # RPTH Accent Yellow
    'cool_gray':     '#E6F1EE',  # RPTH Cool Gray
    # Derived swatches
    'sky_blue':      '#5B9BD5',  # mid swatch between medium and light blue
    'steel':         '#8FA9BD',  # light steel swatch (shading / error bands)
    'deep_yellow':   '#D9A020',  # shaded accent yellow, readable on white
    'sage':          '#4E857B',  # deep shade of the cool-gray family
    'deep_red':      '#8F1D22',  # shaded red (secondary rejection/limit roles)
    'slate':         '#4E5B7A',  # desaturated navy swatch
    'deep_gray':     '#8C9B97',  # shaded cool gray (error bars, neutrals)
    'faint_gray':    '#B9CCC7',  # light cool-gray swatch (background data)
}

# Named-hue lookup (replaces the old Okabe-Ito palette).  All hues are RPTH
# colors or swatches; 'green'/'red' aliases included because limits.py
# requests them.
OKABE_ITO = {
    'orange':          '#D9A020',
    'sky_blue':        '#5B9BD5',
    'bluish_green':    '#4E857B',
    'green':           '#4E857B',
    'yellow':          '#FFC325',
    'blue':            '#005CAB',
    'navy':            '#003366',
    'vermilion':       '#E31B23',
    'red':             '#8F1D22',
    'reddish_purple':  '#4E5B7A',
    'gray':            '#8C9B97',
}

# Simplified scientific palette for main data categories
SCIENTIFIC_PALETTE = {
    'all_sources': '#B9CCC7',  # Faint cool gray for background/all data
    'inliers': '#005CAB',  # RPTH medium blue for good data points
    'outliers': '#E31B23',  # RPTH red for rejected points
    'fit': '#000000',  # Black for fit lines
    'robust': '#4E857B',  # Sage swatch for vetted/kept candidates
    'error_region': '#000000',  # Black with alpha for error regions
}

# Per-plot inlier primary colors - each RANSAC plot type gets a distinct color
# drawn from the RPTH palette/swatches.  Dark-mid lightness inliers contrast
# strongly with the red outliers per the paper's accessibility guidance.
RANSAC_PLOT_COLORS = {
    'zeropoint_ap':      '#005CAB',  # Medium blue   - Zeropoint scatter, aperture
    'zeropoint_psf':     '#009E73',  # Green         - Zeropoint scatter, PSF
    'zeropoint_hist_ap': '#005CAB',  # Medium blue   - Zeropoint histogram, aperture
    'zeropoint_hist_psf':'#009E73',  # Green         - Zeropoint histogram, PSF
    'color_term':        '#D9A020',  # Deep yellow   - Color-term polynomial plot
    'color_term_piece':  '#5B9BD5',  # Sky blue      - Piecewise color-term plot
    'linearity':         '#4E857B',  # Sage          - Linearity check plot
    'flux_comparison':   '#5B9BD5',  # Sky blue      - Template flux comparison plot
    'alignment':         '#5B9BD5',  # Sky blue      - Pre-SCAMP alignment match plot
    'outliers':          '#E31B23',  # RPTH red      - Outliers (all plots)
    'fit':               '#000000',  # Black         - Fit lines (all plots)
    'error_band':        '#8FA9BD',  # Steel swatch  - Error shading (all plots)
}

# Divergent color palette for source check plots
# Blue-white-red divergent scheme (both endpoints from the RPTH palette):
# blue = negative deviation, white = neutral, red = positive deviation
DIVERGENT_PALETTE = {
    'negative': '#005CAB',  # RPTH medium blue for negative deviations
    'neutral': '#FFFFFF',  # White for neutral/center
    'positive': '#E31B23',  # RPTH red for positive deviations
    # Source-check marker colors - RPTH hues for readability on images
    'target': '#E31B23',    # RPTH red
    'psf': '#4E857B',       # Sage
    'reference': '#005CAB', # RPTH medium blue (matches PLOT_COLORS)
    'fwhm_low': '#005CAB',  # RPTH medium blue for low FWHM
    'fwhm_mid': '#F7F7F7',  # Light gray for mid FWHM
    'fwhm_high': '#E31B23',  # RPTH red for high FWHM
    'cross': '#D9A020',  # Deep yellow for cross markers and labels (distinct from red target)
}

# =============================================================================
# Unified semantic palette for plot.py
# =============================================================================
# Every plotting function in plot.py should look up colors here instead of
# hardcoding hex strings or matplotlib named colors.  This ensures that the
# same semantic role (e.g. "target marker", "zero line", "error bar") always
# uses the same color across all diagnostic plots.
PLOT_COLORS = {
    # --- Source markers (overlays on images) ---
    'target':          '#E31B23',   # RPTH red - target aperture / target marker
    'target_secondary':'#005CAB',   # RPTH medium blue - non-primary targets
    'psf':             '#4E857B',   # Sage - PSF source markers
    'reference':       '#005CAB',   # RPTH medium blue - catalog / reference source markers
    'matched':         '#005CAB',   # RPTH medium blue - matched source circles (same as reference)
    'unmatched':       '#E31B23',   # RPTH red - unmatched source crosses
    'variable':        '#FFC325',   # RPTH accent yellow - variable source crosses and labels
    'variable_label':  '#D9A020',   # Deep yellow - variable source text labels
    'fwhm_sources':    'Oranges',   # Colormap name for FWHM-scaled circles
    'epsf_aperture':   '#D9A020',   # Deep yellow - ePSF aperture / input-position circles (distinct from blue 'reference' bounds in the same figure)
    'scamp_matched':   '#4E857B',   # Sage - SCAMP matched markers
    'matched_box':     '#00CFFF',   # Cyan - matched-source boxes in subtraction_check (high contrast on gray)

    # --- Scatter / offset plots ---
    'scatter_primary':  '#005CAB',  # RPTH medium blue - fallback scatter when no color-coding
    'scatter_cmap':     'viridis',  # Colormap for distance-colored scatter
    'error_bar':        '#8C9B97',  # Deep cool gray - error bars on offset plots
    'zero_line':        '#E31B23',  # RPTH red - zero reference lines on offset plots
    'median_line':      '#D9A020',  # Deep yellow - median offset lines
    'positive':         '#E31B23',  # RPTH red - positive deviation markers
    'negative':         '#005CAB',  # RPTH medium blue - negative deviation markers

    # --- Per-plot single-colour markers (distinct RPTH-family hues so each
    #     diagnostic figure has its own recognisable accent) ---
    'fwhm_scatter':     '#5B9BD5',  # Sky blue - FWHM vs instrumental mag
    'fwhm_rejected':    '#E31B23',  # RPTH red - FWHM-rejected markers
    'sfft_scatter':     '#4E857B',  # Sage - SFFT_Matching scatter
    'hist_primary':     '#005CAB',  # Medium blue - single-colour histograms
    'snr_mag_scatter':  '#4E857B',  # Sage - S/N vs magnitude scatter
    'injection_site':   '#5B9BD5',  # Sky blue - injection-site markers on images
    'epsf_recovery':    '#003366',  # Navy - ePSF injection-recovery markers
    'threshold_line':   '#D9A020',  # Deep yellow - median/threshold guide lines

    # --- Image display ---
    'image_cmap':       'gray',     # Default colormap for image display
    'image_cmap_alt':   'viridis',  # Alternate colormap (crowding plot)
    'nan_color':        'magenta',  # Color for NaN/masked pixels in image display
    'mask_overlay':     'magenta',  # Overlay for defect masks (matches NaN 'nan_color')
    'mask_overlay_alt': 'magenta',  # Overlay for source masks (source_check)

    # --- Figure chrome ---
    'figure_facecolor': 'white',    # Figure background
    'legend_facecolor': 'white',    # Legend background
    'legend_edgecolor': '#999999',  # Legend border (grey)
    'stats_bbox':       'white',    # Stats text box background
    'spine_color':      '#000000',  # Inset spine color

    # --- Segmentation contours ---
    'segmentation':     '#4E857B',  # Sage - segmentation contours
}


def get_plot_color(name: str, default: str = '#000000') -> str:
    """Get a semantic color from the unified PLOT_COLORS palette."""
    return PLOT_COLORS.get(name, default)


# Consistent marker sizes
MARKER_SIZES = {
    'small': 4,
    'medium': 6,
    'large': 10,
}

# Consistent line widths
LINE_WIDTHS = {
    'thin': 0.5,
    'medium': 1.0,
    'thick': 1.5,
}
# Consistent alpha values
ALPHA_VALUES = {
    'very_light': 0.15,
    'light': 0.25,
    'medium': 0.5,
    'dark': 0.85,
    'very_dark': 0.95,
}


def get_color(palette_name):
    """Get color from scientific palette."""
    return SCIENTIFIC_PALETTE.get(palette_name, '#000000')


def get_ransac_color(plot_type):
    """Get the designated inlier color for a named RANSAC plot type."""
    return RANSAC_PLOT_COLORS.get(plot_type, SCIENTIFIC_PALETTE['inliers'])


def get_okabe_color(color_name):
    """Get a named hue from the RPTH-derived palette (legacy accessor name)."""
    return OKABE_ITO.get(color_name, '#000000')


def get_rpth_color(color_name):
    """Get a color from the RPTH palette (Plante & Cushman 2020) incl. swatches."""
    return RPTH.get(color_name, '#000000')


def get_marker_size(size_name):
    """Get marker size from predefined sizes."""
    return MARKER_SIZES.get(size_name, 3)


def get_line_width(width_name):
    """Get line width from predefined widths."""
    return LINE_WIDTHS.get(width_name, 0.5)


def get_alpha(alpha_name):
    """Get alpha value from predefined values."""
    return ALPHA_VALUES.get(alpha_name, 0.75)


def get_divergent_color(color_name):
    """Get color from divergent palette."""
    return DIVERGENT_PALETTE.get(color_name, '#000000')


def apply_autophot_mplstyle():
    """
    Use ``autophot.mplstyle`` when present (repo root), else matplotlib defaults.
    Call at the start of figure construction for consistent RANSAC / calibration plots.
    """
    import os
    import matplotlib.pyplot as plt

    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "autophot.mplstyle")
    if os.path.exists(p):
        plt.style.use(p)


def safe_tight_layout(fig=None, **kwargs):
    """
    ``fig.tight_layout()`` without the "Axes not compatible" UserWarning.

    Any GridSpec built with explicit subplot params (hspace, wspace, ...)
    is reported as locally modified, so matplotlib marks every Axes on it
    as incompatible: tight_layout warns and adjusts nothing. The warning
    is expected noise for this codebase, so it is filtered here while any
    genuinely compatible axes still get adjusted. With fig=None the call
    applies to the current figure (``plt.tight_layout`` equivalent).
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible",
            category=UserWarning,
        )
        if fig is None:
            import matplotlib.pyplot as plt
            plt.tight_layout(**kwargs)
        else:
            fig.tight_layout(**kwargs)


def ransac_legend_top_outside(ax, *, ncol: int = 2, fontsize: Optional[Union[int, str]] = 8):
    """Shared legend placement for RANSAC / photometry comparison figures."""
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=ncol,
        fontsize=fontsize,
    )


def ransac_grid(ax):
    """Consistent grid style for all RANSAC / calibration plots."""
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)


def format_log_colorbar_ticks(cb, vmin, vmax, max_ticks=6):
    """Give a LogNorm colorbar plain-number ticks (10, 100, 1000, ...).

    Picks 'nice' tick values inside [vmin, vmax] and formats them with %g
    so labels read as normal numbers rather than 10^n mathtext or
    scientific notation.
    """
    from matplotlib.ticker import FuncFormatter, NullLocator

    candidates = np.array(
        [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500,
         750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000,
         30000, 50000, 100000],
        dtype=float,
    )
    ticks = candidates[(candidates >= vmin) & (candidates <= vmax)]
    if ticks.size == 0:
        ticks = np.array([vmin, vmax], dtype=float)
    elif ticks.size > max_ticks:
        step = int(np.ceil(ticks.size / max_ticks))
        ticks = ticks[::step]
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    cb.ax.yaxis.set_minor_locator(NullLocator())


PLOT_FORMATS = ("png", "svg")


def get_plot_format(input_yaml=None) -> str:
    """Return the configured diagnostic-plot format: 'png' or 'svg'.

    Reads the top-level ``plot_format`` key from the pipeline input yaml.
    Unset or unrecognised values fall back to 'png'.
    """
    try:
        fmt = str((input_yaml or {}).get("plot_format", "png") or "png")
    except Exception:
        fmt = "png"
    fmt = fmt.strip().lower().lstrip(".")
    return fmt if fmt in PLOT_FORMATS else "png"


def get_plot_ext(input_yaml=None) -> str:
    """Return the configured plot filename extension, e.g. '.png'."""
    return "." + get_plot_format(input_yaml)


def ransac_savefig(fig, path):
    """Consistent save settings for all RANSAC / calibration plots."""
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")


def set_mag_axes_inverted_xy(ax):
    """Standard magnitude axis orientation (brighter up/left) for x and y."""
    ax.invert_xaxis()
    ax.invert_yaxis()


def add_clean_legend(ax, labels=None, *, loc="best", bbox_to_anchor=None, **kwargs):
    """Place a legend with automatic ``ncol`` and a non-overlapping layout.

    Picks the number of columns from the handle count so wide legends stay
    compact (>=8 handles -> 3 cols, >=5 -> 2 cols, else 1 col), and defaults
    to a frameless legend.  Any caller-supplied ``kwargs`` (e.g. ``fontsize``,
    ``ncol``, ``frameon``) override the defaults.  Use ``bbox_to_anchor`` to
    move the legend outside the axes when it would otherwise cover titles,
    labels, or key features.

    Returns the created ``Legend`` (or ``None`` if there are no handles).
    """
    handles, leg_labels = ax.get_legend_handles_labels()
    if labels is not None:
        leg_labels = labels
    if not handles:
        return None
    n = len(handles)
    ncol = 3 if n >= 8 else (2 if n >= 5 else 1)
    defaults = dict(
        loc=loc,
        frameon=False,
        fontsize=8,
        ncol=ncol,
    )
    if bbox_to_anchor is not None:
        defaults["bbox_to_anchor"] = bbox_to_anchor
    defaults.update(kwargs)
    return ax.legend(handles, leg_labels, **defaults)
