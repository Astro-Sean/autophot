"""
Centralized plotting utilities for consistent figure formatting across autophot.
Uses colorblind-friendly color palettes based on scientific visualization best practices.
"""

import numpy as np
from typing import Optional, Union

# Okabe-Ito colorblind-friendly palette (widely used in scientific publications)
OKABE_ITO = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermilion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'gray': '#999999',
}

# Simplified scientific palette for main data categories
SCIENTIFIC_PALETTE = {
    'all_sources': '#D3D3D3',  # Light gray for background/all data
    'inliers': '#8ec1da',  # Blue for good data points
    'outliers': '#e02b25',  # Vermilion/Orange-red for rejected points
    'fit': '#000000',  # Black for fit lines
    'robust': '#009E73',  # Bluish green for robust candidates
    'error_region': '#000000',  # Black with alpha for error regions
}

# Per-plot inlier primary colors - each RANSAC plot type gets a distinct color
# drawn from the Okabe-Ito palette for colorblind safety.
RANSAC_PLOT_COLORS = {
    'zeropoint_ap':      '#0072B2',  # Okabe blue        - Zeropoint scatter, aperture
    'zeropoint_psf':     '#009E73',  # Okabe green       - Zeropoint scatter, PSF
    'zeropoint_hist_ap': '#0072B2',  # Okabe blue        - Zeropoint histogram, aperture
    'zeropoint_hist_psf':'#009E73',  # Okabe green       - Zeropoint histogram, PSF
    'color_term':        '#E69F00',  # Okabe orange      - Color-term polynomial plot
    'color_term_piece':  '#CC79A7',  # Okabe reddish-purple - Piecewise color-term plot
    'linearity':         '#D55E00',  # Okabe vermilion   - Linearity check plot
    'flux_comparison':   '#56B4E9',  # Okabe sky blue    - Template flux comparison plot
    'alignment':         '#56B4E9',  # Okabe sky blue    - Pre-SCAMP alignment match plot
    'outliers':          '#e02b25',  # Red               - Outliers (all plots)
    'fit':               '#000000',  # Black             - Fit lines (all plots)
    'error_band':        '#999999',  # Gray              - Error shading (all plots)
}

# Divergent color palette for source check plots
# Using a blue-white-red divergent scheme where blue = negative deviation, white = neutral, red = positive deviation
DIVERGENT_PALETTE = {
    'negative': '#313695',  # Dark blue for negative deviations
    'neutral': '#FFFFFF',  # White for neutral/center
    'positive': '#FF8C00',  # Orange for positive deviations
    # Source-check marker colors - distinct primary colors for readability
    'target': '#FF0000',    # Red
    'psf': '#00AA00',       # Green
    'reference': '#0072B2', # Okabe blue (matches PLOT_COLORS)
    'fwhm_low': '#313695',  # Dark blue for low FWHM
    'fwhm_mid': '#F7F7F7',  # Light gray for mid FWHM
    'fwhm_high': '#B30000',  # Dark red for high FWHM
    'cross': '#D55E00',  # Okabe vermilion for cross markers and labels (avoid black)
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
    'target':          '#FF0000',   # Red – target aperture / target marker
    'psf':             '#00AA00',   # Green – PSF source markers
    'reference':       '#0072B2',   # Okabe blue – catalog / reference source markers
    'matched':         '#0072B2',   # Okabe blue – matched source circles (same as reference)
    'unmatched':       '#FF0000',   # Red – unmatched source crosses
    'variable':        '#FFD700',   # Gold – variable source crosses and labels
    'variable_label':  '#B8860B',   # Dark goldenrod – variable source text labels
    'fwhm_sources':    'Oranges',   # Colormap name for FWHM-scaled circles

    # --- Scatter / offset plots ---
    'scatter_primary':  '#0072B2',  # Okabe blue – fallback scatter when no color-coding
    'scatter_cmap':     'viridis',  # Colormap for distance-colored scatter
    'error_bar':        '#999999',  # Gray – error bars on offset plots
    'zero_line':        '#FF0000',  # Red – zero reference lines on offset plots
    'median_line':      '#E69F00',  # Okabe orange – median offset lines

    # --- Image display ---
    'image_cmap':       'gray',     # Default colormap for image display
    'image_cmap_alt':   'viridis',  # Alternate colormap (crowding plot)
    'nan_color':        'white',    # Color for NaN pixels in image display
    'mask_overlay':     '#FF0000',  # Red overlay for defect masks
    'mask_overlay_alt': 'white',    # White overlay for source masks (source_check)

    # --- Figure chrome ---
    'figure_facecolor': 'white',    # Figure background
    'legend_facecolor': 'white',    # Legend background
    'legend_edgecolor': '#999999',  # Legend border (grey)
    'stats_bbox':       'white',    # Stats text box background
    'spine_color':      '#000000',  # Inset spine color

    # --- Segmentation contours ---
    'segmentation':     '#00AA00',  # Green – segmentation contours
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
