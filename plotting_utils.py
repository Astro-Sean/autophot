"""
Centralized plotting utilities for consistent figure formatting across autophot.
Uses colorblind-friendly color palettes based on scientific visualization best practices.
"""

import numpy as np
from typing import Optional, Union

# Muted/pastel colorblind-friendly palette (neutral scientific tones derived
# from Okabe-Ito hues, desaturated for a calmer publication look)
OKABE_ITO = {
    'orange': '#D9A05B',
    'sky_blue': '#7FB8D9',
    'bluish_green': '#6FA88F',
    'yellow': '#D8C87A',
    'blue': '#5C8FB8',
    'vermilion': '#C97C6B',
    'reddish_purple': '#B08FB8',
    'gray': '#A3A3A3',
}

# Simplified scientific palette for main data categories
SCIENTIFIC_PALETTE = {
    'all_sources': '#D3D3D3',  # Light gray for background/all data
    'inliers': '#8FB8D4',  # Muted blue for good data points
    'outliers': '#C97B74',  # Muted brick/rose for rejected points
    'fit': '#000000',  # Black for fit lines
    'robust': '#6FA88F',  # Muted sage green for robust candidates
    'error_region': '#000000',  # Black with alpha for error regions
}

# Per-plot inlier primary colors - each RANSAC plot type gets a distinct color
# drawn from the muted palette for colorblind safety.
RANSAC_PLOT_COLORS = {
    'zeropoint_ap':      '#5C8FB8',  # Steel blue        - Zeropoint scatter, aperture
    'zeropoint_psf':     '#6FA88F',  # Sage green        - Zeropoint scatter, PSF
    'zeropoint_hist_ap': '#5C8FB8',  # Steel blue        - Zeropoint histogram, aperture
    'zeropoint_hist_psf':'#6FA88F',  # Sage green        - Zeropoint histogram, PSF
    'color_term':        '#D9A05B',  # Muted tan-orange  - Color-term polynomial plot
    'color_term_piece':  '#B08FB8',  # Muted lilac       - Piecewise color-term plot
    'linearity':         '#C97C6B',  # Muted terracotta  - Linearity check plot
    'flux_comparison':   '#7FB8D9',  # Muted sky blue    - Template flux comparison plot
    'alignment':         '#7FB8D9',  # Muted sky blue    - Pre-SCAMP alignment match plot
    'outliers':          '#C97B74',  # Muted brick       - Outliers (all plots)
    'fit':               '#000000',  # Black             - Fit lines (all plots)
    'error_band':        '#A3A3A3',  # Gray              - Error shading (all plots)
}

# Divergent color palette for source check plots
# Using a blue-white-red divergent scheme where blue = negative deviation, white = neutral, red = positive deviation
DIVERGENT_PALETTE = {
    'negative': '#5E6FB8',  # Muted slate blue for negative deviations
    'neutral': '#FFFFFF',  # White for neutral/center
    'positive': '#D9A05B',  # Muted tan-orange for positive deviations
    # Source-check marker colors - distinct muted hues for readability on images
    'target': '#D94F4F',    # Muted red
    'psf': '#5FA86F',       # Muted green
    'reference': '#5C8FB8', # Muted steel blue (matches PLOT_COLORS)
    'fwhm_low': '#5E6FB8',  # Muted slate blue for low FWHM
    'fwhm_mid': '#F7F7F7',  # Light gray for mid FWHM
    'fwhm_high': '#B05252',  # Muted brick red for high FWHM
    'cross': '#C97C6B',  # Muted terracotta for cross markers and labels (avoid black)
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
    'target':          '#D94F4F',   # Muted red – target aperture / target marker
    'target_secondary':'#6FAFB8',   # Muted teal – non-primary targets
    'psf':             '#5FA86F',   # Muted green – PSF source markers
    'reference':       '#5C8FB8',   # Muted steel blue – catalog / reference source markers
    'matched':         '#5C8FB8',   # Muted steel blue – matched source circles (same as reference)
    'unmatched':       '#D94F4F',   # Muted red – unmatched source crosses
    'variable':        '#D8B85C',   # Muted gold – variable source crosses and labels
    'variable_label':  '#A8893B',   # Muted dark gold – variable source text labels
    'fwhm_sources':    'Oranges',   # Colormap name for FWHM-scaled circles
    'epsf_aperture':   '#B08FB8',   # Muted lilac – ePSF aperture / input-position circles
    'scamp_matched':   '#6FA88F',   # Muted sage green – SCAMP matched markers

    # --- Scatter / offset plots ---
    'scatter_primary':  '#5C8FB8',  # Muted steel blue – fallback scatter when no color-coding
    'scatter_cmap':     'viridis',  # Colormap for distance-colored scatter
    'error_bar':        '#A3A3A3',  # Gray – error bars on offset plots
    'zero_line':        '#C97B74',  # Muted brick – zero reference lines on offset plots
    'median_line':      '#D9A05B',  # Muted tan-orange – median offset lines

    # --- Image display ---
    'image_cmap':       'gray',     # Default colormap for image display
    'image_cmap_alt':   'viridis',  # Alternate colormap (crowding plot)
    'nan_color':        'white',    # Color for NaN pixels in image display
    'mask_overlay':     '#D94F4F',  # Muted red overlay for defect masks
    'mask_overlay_alt': 'white',    # White overlay for source masks (source_check)

    # --- Figure chrome ---
    'figure_facecolor': 'white',    # Figure background
    'legend_facecolor': 'white',    # Legend background
    'legend_edgecolor': '#999999',  # Legend border (grey)
    'stats_bbox':       'white',    # Stats text box background
    'spine_color':      '#000000',  # Inset spine color

    # --- Segmentation contours ---
    'segmentation':     '#5FA86F',  # Muted green – segmentation contours
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
    """Get color from Okabe-Ito palette."""
    return OKABE_ITO.get(color_name, '#000000')


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
    to an opaque white frame so the legend stays readable over data.  Any
    caller-supplied ``kwargs`` (e.g. ``fontsize``, ``ncol``) override the
    defaults.  Use ``bbox_to_anchor`` to move the legend outside the axes
    when it would otherwise cover titles, labels, or key features.

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
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        fontsize=8,
        ncol=ncol,
    )
    if bbox_to_anchor is not None:
        defaults["bbox_to_anchor"] = bbox_to_anchor
    defaults.update(kwargs)
    return ax.legend(handles, leg_labels, **defaults)
