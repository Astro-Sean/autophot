"""
Centralized plotting utilities for consistent figure formatting across autophot.
Uses colorblind-friendly color palettes based on scientific visualization best practices.
"""

import numpy as np

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

# Divergent color palette for source check plots
# Using a blue-white-red divergent scheme where blue = negative deviation, white = neutral, red = positive deviation
DIVERGENT_PALETTE = {
    'negative': '#313695',  # Dark blue for negative deviations
    'neutral': '#FFFFFF',  # White for neutral/center
    'positive': '#B30000',  # Dark red for positive deviations
    'target': '#59a89c',  # Vermilion for target source
    'psf': '#009E73',  # Bluish green for PSF sources
    'reference': '#e02b35',  # Yellow for reference sources
    'fwhm_low': '#313695',  # Dark blue for low FWHM
    'fwhm_mid': '#F7F7F7',  # Light gray for mid FWHM
    'fwhm_high': '#B30000',  # Dark red for high FWHM
    'cross': '#CC79A7',  # Reddish purple for cross markers
}
# /home/sbrennan/Desktop/SN2024pba/photometry/SN_A/SN2024pba/images_REDUCED/DEEP_SN2024pba_2025-02-11_ZTF_g/LOG_DEEP_SN2024pba_2025-02-11_ZTF_g.log
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
