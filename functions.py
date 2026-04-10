#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:58:29 2022

@author: seanbrennan
"""

# Import all required modules at the top
import numpy as np

import os
import sys
import warnings
import pandas as pd
import yaml
import logging
import matplotlib.pyplot as plt
import traceback
import textwrap
import inspect

import copy
from astropy.io import fits
from astropy.time import Time
from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.cosmology import FlatLambdaCDM
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.ndimage import distance_transform_edt
from scipy.special import erf, erfinv

from skimage.measure import moments

from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog, make_2dgaussian_kernel
from photutils.aperture import RectangularAperture


class ColoredLevelFormatter(logging.Formatter):
    """
    Logging formatter with ANSI color highlights based on log level.

    Uses colors only when stdout is a TTY; otherwise it behaves like a normal
    logging.Formatter (no ANSI escape sequences).
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[34m"
    RED = "\033[31m"

    def __init__(self, *args, use_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg_raw = record.getMessage()
        msg_clean = normalize_log_message(msg_raw)
        old_msg, old_args = record.msg, record.args
        if msg_clean != msg_raw:
            record.msg = msg_clean
            record.args = ()
        base = super().format(record)
        record.msg, record.args = old_msg, old_args
        if not self._use_color:
            return base

        try:
            use_color = sys.stdout.isatty()
        except Exception:
            use_color = False
        if not use_color:
            return base

        if record.levelno >= logging.CRITICAL:
            return f"{self.BOLD}{self.RED}{base}{self.RESET}"
        if record.levelno >= logging.ERROR:
            return f"{self.BOLD}{self.RED}{base}{self.RESET}"
        if record.levelno >= logging.WARNING:
            # Keep warnings high-contrast without extra color coding.
            return f"{self.BOLD}{base}{self.RESET}"
        if record.levelno >= logging.DEBUG:
            # Debug output: blue for quick visual scanning.
            return f"{self.BLUE}{base}{self.RESET}"

        # INFO/default: plain text (terminal default color, typically black/white).
        return base


def normalize_log_message(message: str, width: int = 150) -> str:
    """
    Normalize log message formatting for readability and consistency.

    - Converts tabs to spaces.
    - Trims trailing whitespace.
    - Collapses repeated blank lines.
    - Soft-wraps long lines to a fixed width with indentation preserved.
    """
    text = str(message).replace("\t", "    ")
    lines = [ln.rstrip() for ln in text.splitlines()]
    if text.endswith("\n"):
        # Preserve intentional trailing spacer lines from banner-style messages.
        lines.append("")

    compact: list[str] = []
    blank_seen = False
    for ln in lines:
        if ln.strip() == "":
            if not blank_seen:
                compact.append("")
            blank_seen = True
            continue
        blank_seen = False
        compact.append(ln)

    wrapped: list[str] = []
    for ln in compact:
        if not ln:
            wrapped.append("")
            continue
        if len(ln) <= width:
            wrapped.append(ln)
            continue
        indent_len = len(ln) - len(ln.lstrip(" "))
        indent = " " * indent_len
        wrapped_ln = textwrap.fill(
            ln.strip(),
            width=width,
            initial_indent=indent,
            subsequent_indent=indent + "  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped.extend(wrapped_ln.splitlines())

    # Keep intentional leading/trailing spacing (e.g. border banners),
    # but collapse internal blank-line runs via the logic above.
    return "\n".join(wrapped)


class LogMessageNormalizeFilter(logging.Filter):
    """Filter that normalizes message text before emission."""

    def __init__(self, width: int = 150):
        super().__init__()
        self.width = int(width)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            normalized = normalize_log_message(record.getMessage(), width=self.width)
            record.msg = normalized
            record.args = ()
        except Exception:
            pass
        return True


def configure_console_logging(
    *,
    level: int = logging.INFO,
    use_color: bool = True,
    formatter: logging.Formatter | None = None,
) -> logging.Handler:
    """
    Create a StreamHandler for stdout with optional ANSI color formatting.
    Caller is responsible for attaching the handler to the root logger.
    """

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.addFilter(LogMessageNormalizeFilter(width=150))
    if formatter is None:
        formatter = ColoredLevelFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            use_color=use_color,
        )
    handler.setFormatter(formatter)
    return handler


SUPPORTED_FILTER_GROUPS = {
    "UBVRI": tuple("UBVRI"),
    "ugriz": tuple("ugriz"),
    "JHK": tuple("JHK"),
    "extended": tuple("Yw"),  # Extended filters (Y-band, w-band)
}

SUPPORTED_PHOTOMETRIC_FILTERS = tuple(
    band for group in SUPPORTED_FILTER_GROUPS.values() for band in group
)

NON_PHOTOMETRIC_FILTER_KEYS = {
    "RA",
    "DEC",
    "name",
    "objname",
    "name_prefix",
    "RA_err",
    "DEC_err",
}


def get_supported_filter_groups() -> dict:
    """Return accepted filter-group keys for per-filter mapping."""
    return dict(SUPPORTED_FILTER_GROUPS)


def get_supported_photometric_filters() -> tuple:
    """Return all accepted photometric band names."""
    return tuple(SUPPORTED_PHOTOMETRIC_FILTERS)


# Composite YAML keys (optical + near-IR) that are not one single family letter-set.
_COMPOSITE_FILTER_GROUP_KEYS = {
    "grizjhk": tuple("grizJHK"),  # g,r,i,z,J,H,K (matches common refcat / Pan-STARRS+2MASS style maps)
    "ugrizjhk": tuple("ugrizJHK"),  # u,g,r,i,z,J,H,K
    "grizjhkYw": tuple("grizJHKYw"),  # g,r,i,z,J,H,K,Y,w (extended filter set)
    "ugrizjhkYw": tuple("ugrizJHKYw"),  # u,g,r,i,z,J,H,K,Y,w (extended filter set)
}


def parse_supported_filter_group_key(group_key):
    """
    Parse a mapping-group key into explicit supported bands.

    Accepted examples:
      - Full groups: "UBVRI", "ugriz", "JHK"
      - Valid subsets/singletons of one family: "griz", "u", "BV", "HK"
      - Composite keys: "grizJHK", "ugrizJHK" (optical + JHK for catalog.use_catalog maps)
    Rejected:
      - Mixed-family tokens: "uBV", "rJ"
      - Unsupported tokens/bands.
    """
    if group_key is None:
        return None
    key = str(group_key).strip()
    if key == "":
        return None

    # Fast path for canonical full-group keys.
    if key in SUPPORTED_FILTER_GROUPS:
        return tuple(SUPPORTED_FILTER_GROUPS[key])

    comp = _COMPOSITE_FILTER_GROUP_KEYS.get(key.lower())
    if comp is not None:
        return comp

    # Support subsets of exactly one canonical family.
    for family_bands in SUPPORTED_FILTER_GROUPS.values():
        fam_set = set(family_bands)
        if all(ch in fam_set for ch in key):
            ordered = []
            seen = set()
            for ch in key:
                if ch not in seen:
                    ordered.append(ch)
                    seen.add(ch)
            return tuple(ordered)
    return None


def normalize_photometric_filter_name(filter_name, available_filters=None):
    """
    Normalize a filter token to a supported photometric band.
    
    This function is now dynamic and accepts any filter name that is either:
    1. A standard photometric filter (UBVRI, ugriz, JHK families)
    2. Present in the available_filters list (from custom catalogs)
    3. A common alias of standard filters
    
    Non-photometric fields (RA/DEC/name and *_err columns) return None.
    
    Parameters
    ----------
    filter_name : str
        The filter name to normalize
    available_filters : list or tuple, optional
        List of available filters from catalogs. If provided, allows any filter
        name present in this list, enabling completely dynamic filter support.
    
    Returns
    -------
    str or None
        Normalized filter name or None for non-photometric fields
    """
    if filter_name is None:
        return None

    token = str(filter_name).strip()
    if token == "":
        return None

    if token in NON_PHOTOMETRIC_FILTER_KEYS or token.lower().endswith("_err"):
        return None

    # If available_filters is provided, accept any filter present there
    if available_filters is not None:
        available_set = set(str(f).strip() for f in available_filters)
        # Exact match first
        if token in available_set:
            return token
        # Case-insensitive match
        if token.lower() in {f.lower() for f in available_set}:
            # Return the exact case from available_filters
            for f in available_set:
                if f.lower() == token.lower():
                    return f
    
    # Standard photometric filters
    if token in SUPPORTED_PHOTOMETRIC_FILTERS:
        return token

    # Common aliases for standard filters
    token_l = token.lower()
    aliases = {
        "up": "u",
        "gp": "g",
        "rp": "r",
        "ip": "i",
        "zp": "z",
        "b": "B",
        "v": "V",
        "ks": "K",
        "j": "J",
        "h": "H",
        "k": "K",
        # Additional common variations
        "clear": None,
        "open": None,
        "luminance": None,
        "white": None,
    }
    
    result = aliases.get(token_l)
    if result is not None:
        return result
    
    # If available_filters is provided, try to find close matches
    if available_filters is not None:
        import difflib
        close_matches = difflib.get_close_matches(token, list(available_set), n=1, cutoff=0.8)
        if close_matches:
            return close_matches[0]
        
        # For custom catalogs, accept the original token if it's in the available filters
        # This is the key change that allows arbitrary filter names
        if token in available_set:
            return token
    
    return None


def sanitize_photometric_filters(filters, available_filters=None):
    """
    Keep only supported photometric filters while preserving order.

    Parameters
    ----------
    filters : list
        List of filter names to sanitize
    available_filters : list, optional
        List of available filters from catalogs. If provided, allows
        arbitrary filter names present in this list.

    Returns
    -------
    (cleaned, dropped)
        cleaned : list[str]
            Deduplicated list of supported filters.
        dropped : list[str]
            Raw filter tokens that were rejected.
    """
    cleaned = []
    seen = set()
    dropped = []
    for raw in filters or []:
        norm = normalize_photometric_filter_name(raw, available_filters=available_filters)
        if norm is None:
            dropped.append(str(raw))
            continue
        if norm in seen:
            continue
        cleaned.append(norm)
        seen.add(norm)
    return cleaned, dropped


def odd(n: int) -> int:
    """Return n if odd, else n+1."""
    n = int(n)
    return n + (n % 2 == 0)


def format_exception_origin(exc: BaseException) -> str:
    """
    Return ``path:lineno`` for the stack frame where *exc* was raised.

    Used in warning logs when the active ``sys.exc_info()`` stack may not apply.
    """
    tb = getattr(exc, "__traceback__", None)
    if tb is None:
        return "<no traceback>"
    while tb.tb_next:
        tb = tb.tb_next
    try:
        co = tb.tb_frame.f_code
        return f"{co.co_filename}:{tb.tb_lineno}"
    except Exception:
        return "<unknown>"


def log_warning_from_exception(
    logger: logging.Logger,
    message: str,
    exc: BaseException,
    *,
    exc_info: bool = False,
) -> None:
    """
    Log a WARNING for *exc* with explicit file:line for the raise site and log site.

    Prefer this over ``logger.warning("...%%s", e)`` inside ``except`` blocks so
    debugging information is consistent. Set ``exc_info=True`` to append a full
    traceback to the log record.
    """
    exc_origin = format_exception_origin(exc)
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        if caller is not None:
            co = caller.f_code
            log_origin = f"{co.co_filename}:{caller.f_lineno}"
        else:
            log_origin = "?"
    finally:
        del frame

    einfo = False
    if exc_info:
        einfo = (type(exc), exc, exc.__traceback__)

    logger.warning(
        "%s | exc at %s | logged at %s | %s: %s",
        message,
        exc_origin,
        log_origin,
        type(exc).__name__,
        exc,
        exc_info=einfo,
    )


def log_exception(e: Exception, msg: str = None):
    """
    Logs detailed exception information.

    Parameters
    ----------
    e : Exception
        The exception instance to log.
    msg : str, optional
        An optional message to display at the top of the log.
    """
    logger = logging.getLogger(__name__)

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    lineno = exc_tb.tb_lineno if exc_tb else -1

    log_message = ""
    if msg:
        log_message += f"{msg}\n\n"

    log_message += (
        f"Type     : {exc_type.__name__ if exc_type else type(e).__name__}\n"
        f"File     : {fname}\n"
        f"Line     : {lineno}\n"
        f"Message  : {str(e)}\n"
        f"\n" + traceback.format_exc()
    )

    logger.error(log_message)


def pad_ones(mask, padding):
    """
    Expands regions of 1s in a 2D mask to their nearest neighbors by a given amount.

    Parameters:
    mask (np.ndarray): A 2D binary array where 1s represent masked pixels.
    padding (int): The number of pixels to expand the regions by.

    Returns:
    np.ndarray: A new 2D mask with expanded regions.
    """
    if padding <= 0:
        return mask

    # Compute the distance transform from the zero regions
    distance = distance_transform_edt(1 - mask)

    # Expand the mask where the distance is within the padding radius
    expanded_mask = (distance <= padding).astype(np.uint8)

    return expanded_mask


def set_size(width, aspect=1, fraction=1):
    """
     Function to generate size of figures produced by AutoPhot. To specify the dimensions of a figure in matplotlib we use the figsize argument. However, the figsize argument takes inputs in inches and we have the width of our document in pts. To set the figure size we construct a function to convert from pts to inches and to determine an aesthetic figure height using the golden ratio. The golden ratio is given by:

     .. math ::

        \\phi = (5^{0.5} + 1) / 2 \\approx 1.618

    The ratio of the given width and height is set to the golden ratio


    Credit: `jwalton.info <https://jwalton.info/Embed-Publication-Matplotlib-Latex/>`_

    :param width: Width of figure in pts. 1pt == 1/72 inches
    :type width: float
    :param aspect: Aspect of image i.e. :math:`height  = width / \\phi \\times \\mathit{aspect}`, default  = 1
    :type aspect: float
    :return: Returns tuple of width, height in inches ready for use.
    :rtype: Tuple

    """

    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 + 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / golden_ratio

    fig_dim = (fig_width_in, fig_height_in * aspect)

    return fig_dim


def convert_to_mjd_astropy(date_string):

    try:
        # Try parsing with 'T' separator
        t = Time(date_string, format="isot", scale="utc")
    except ValueError:
        # If parsing fails, try without 'T' separator
        t = Time(date_string, format="fits", scale="utc")

    # Access the Modified Julian Date (MJD)
    mjd = t.mjd

    return mjd


def get_image_stats(image, sigma=3, maxiters=None):

    # Perform sigma clipping and calculate mean, median, and MAD in one step
    mean_value, median_value, std_value = sigma_clipped_stats(
        image,
        sigma=sigma,
        maxiters=maxiters,
        # background=sigma,
        cenfunc=np.nanmedian,  # Use nanmedian for the center function
        stdfunc=mad_std,  # Use mad_std for the standard deviation function
    )

    return mean_value, median_value, std_value


def calculate_bins(x, percentiles=[25, 75]):

    try:

        if not np.any(np.isfinite(x)):
            return "auto"  # Return default bins if no finite data
        # Your normal bin calculation here
        """
        Calculate the number of bins for a histogram using the Freedman-Diaconis rule.
    
        Parameters:
        x (array-like): Input data array or list of data values.
    
        Returns:
        int: Number of bins to use for the histogram.
        
        The Freedman-Diaconis rule is used to determine an optimal number of bins
        by considering the interquartile range (IQR) and the number of data points.
        """
        # Compute the 25th and 75th percentiles of the data
        q25, q75 = np.nanpercentile(x, percentiles)

        # Calculate the interquartile range (IQR)
        iqr = q75 - q25

        # Calculate the bin width using the Freedman-Diaconis rule
        bin_width = 2 * iqr * len(x) ** (-1 / 3)

        # Determine the number of bins
        data_range = np.nanmax(x) - np.nanmin(x)
        bins = round(data_range / bin_width)

        return bins

    except Exception:
        return "auto"


def save_to_fits(data, output_filename):
    try:
        # Use float32 to preserve NaNs (chip gaps) - integer dtypes cannot represent NaN
        data_to_write = data.astype(np.float32) if data.dtype.kind != 'f' else data
        # Create a PrimaryHDU object with the data
        hdu = fits.PrimaryHDU(data_to_write)

        # Create an HDU list and append the PrimaryHDU
        hdulist = fits.HDUList([hdu])

        # Write the HDU list to a FITS file
        hdulist.writeto(
            output_filename, overwrite=True, output_verify="silentfix+ignore"
        )
    except Exception as exc:
        logger = logging.getLogger(__name__)
        exc_type, _, exc_tb = sys.exc_info()
        fname = (
            os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            if exc_tb
            else "unknown"
        )
        line = exc_tb.tb_lineno if exc_tb else -1
        logger.error(
            "Failed to write FITS file '%s': %s in %s:%d",
            output_filename,
            exc_type.__name__,
            fname,
            line,
            exc_info=True,
        )
        return 0, 0

    return None


def get_distance_modulus(redshift, H0=70, omega=0.3):

    cosmo = FlatLambdaCDM(H0=H0, Om0=omega)
    d = cosmo.luminosity_distance(redshift).value * 1e6
    dm = 5 * np.log10(d / 10)

    return dm


class SuppressStdout:

    def __enter__(self):
        import sys, os

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):

        sys.stdout.close()
        sys.stdout = self._original_stdout


def beta_aperture(n, flux_aperture, npix, sigma, noise=0):
    """
    Detection confidence (beta) for a source using aperture flux.

    Beta is the probability that the measured flux is above the n-sigma
    detection threshold (higher = more confident detection). Used for
    detection criteria and limiting magnitude calculations.

    Parameters
    ----------
    n : float
        Detection threshold in sigma (e.g., 3 for 3*sigma).
    flux_aperture : float or np.ndarray
        Total flux measured in the aperture.
    npix : int or float
        Number of pixels in the aperture.
    sigma : float
        Background standard deviation per pixel.
    noise : float, optional
        Mean background offset if not background-subtracted (default 0).

    Returns
    -------
    beta : float or np.ndarray
        Detection confidence in [0, 1]. Higher values indicate a more
        confident detection above the threshold.
    """
    # Ensure non-negative flux after background subtraction
    source_flux = np.maximum(flux_aperture - noise * npix, 0.0)

    # Total noise in the aperture
    sigma_aperture = sigma * np.sqrt(npix)
    # Avoid division by zero
    sigma_aperture = np.maximum(sigma_aperture, np.finfo(float).tiny)

    # z-score: how far the measured flux is above the n-sigma threshold
    z = ((n * sigma_aperture) - source_flux) / (np.sqrt(2) * sigma_aperture)

    # Detection confidence; clip to [0, 1] for a proper probability
    beta = np.clip(0.5 * (1 - erf(z)), 0.0, 1.0)
    return beta


def beta_psf(n, flux_psf, flux_psf_err):
    """
    Detection confidence (beta) for a source using PSF flux and its uncertainty.

    Uses the fitted flux and its error to compute how confidently the source
    is above the n-sigma detection threshold. Prefer this over aperture-based
    beta when PSF photometry is available, as it uses the actual measurement
    uncertainty.

    Parameters
    ----------
    n : float
        Detection threshold in sigma (e.g., 3 for 3*sigma).
    flux_psf : float or np.ndarray
        PSF-fitted flux.
    flux_psf_err : float or np.ndarray
        Uncertainty on the PSF flux (1-sigma).

    Returns
    -------
    beta : float or np.ndarray
        Detection confidence in [0, 1]. Higher values indicate a more
        confident detection above the threshold.
    """
    flux_psf = np.maximum(np.asarray(flux_psf, dtype=float), 0.0)
    err = np.maximum(np.asarray(flux_psf_err, dtype=float), np.finfo(float).tiny)
    # Threshold flux = n * (1-sigma error); z-score for "flux above threshold"
    z = (n * err - flux_psf) / (np.sqrt(2) * err)
    beta = np.clip(0.5 * (1 - erf(z)), 0.0, 1.0)
    return beta


def border_msg(msg: str, body: str = "-", corner: str = "+") -> str:
    """
    Generate a simple, readable banner string for logging.

    Example:
        >>> logging.info(border_msg("Filter check"))
        # blank line above, then:
        # ----------------
        #  Filter check
        # ----------------
    """
    text = str(msg).strip()
    if not text:
        return ""

    line = text
    border = body * len(line)

    # Bold the title line on interactive terminals.
    # (ANSI sequences are ignored by non-TTY log capture.)
    try:
        use_bold = sys.stdout.isatty()
    except Exception:
        use_bold = False
    if use_bold:
        bold_prefix = "\033[1m"
        reset_suffix = "\033[0m"
        line = f"{bold_prefix}{line}{reset_suffix}"

    # Avoid leading newlines: most loggers prefix the first line with
    # "<timestamp> - INFO - " and then print message. If the message starts
    # with "\n", the "INFO -" portion appears blank, which looks messy.
    # Instead, put separators at the end so the banner still reads cleanly.
    return f"\n\n\n{border}\n{line}\n{border}\n\n"


# Telescope/instrument config: images must have FITS header keywords TELESCOP and INSTRUME.
# telescope.yml in the working dir lists all telescopes; each entry follows the structure below.
# Top level: TELESCOP value (e.g. "1m0-01", "ESO-NTT", "Palomar 48-inch"). Under each:
#   INSTRUME: instrument name -> instrument config dict.
#   location (optional): {name, lon, lat, alt} for the site.
#
# Instrument config entry structure (build telescope.yml entries from this):
INSTRUMENT_ENTRY_STRUCTURE = {
    "Name": "",  # Human-readable label, e.g. "TELESCOP+INSTRUME"
    "filter_key_0": "FILTER",
    "mjd": "MJD-OBS",
    "date": "DATE-OBS",
    "gain": "GAIN",
    "saturate": "SATURATE",
    "readnoise": "RDNOISE",
    "airmass": "AIRMASS",
    "exptime": "EXPTIME",
    "pixel_scale": 0.4,  # float, arcsec/pixel
    # Filter mappings: header value -> catalog band (e.g. "gp": "g", "rp": "r", "ip": "i")
}

# Built-in fallback when telescope.yml is missing (one example; add others to telescope.yml).
BUILTIN_TELESCOPE_DEFAULTS = {
    "Palomar 48-inch": {
        "INSTRUME": {
            "ZTF/MOSAIC": {
                "Name": "Palomar 48-inch+ZTF/MOSAIC",
                "filter_key_0": "FILTER",
                "ZTF_g": "g",
                "ZTF_r": "r",
                "ZTF_i": "i",
                "ztf_g": "g",
                "ztf_r": "r",
                "ztf_i": "i",
                "mjd": "OBSMJD",
                "date": "DEC_RATE",
                "gain": "GAIN",
                "saturate": "SATURATE",
                "readnoise": "RDNOISE",
                "airmass": "AIRMASS",
                "exptime": "EXPTIME",
                "pixel_scale": 0.4,
            },
            "ZTF": {
                "Name": "Palomar 48-inch+ZTF",
                "filter_key_0": "FILTER",
                "ZTF_g": "g",
                "ZTF_r": "r",
                "ZTF_i": "i",
                "ztf_g": "g",
                "ztf_r": "r",
                "ztf_i": "i",
                "mjd": "OBSMJD",
                "date": "DEC_RATE",
                "gain": "GAIN",
                "saturate": "SATURATE",
                "readnoise": "RDNOISE",
                "airmass": "AIRMASS",
                "exptime": "EXPTIME",
                "pixel_scale": 0.4,
            },
        },
        "location": {"name": None, "lon": None, "lat": None, "alt": None},
    },
    "SDSS": {
        "INSTRUME": {
            "SDSS": {
                "Name": "SDSS",
                "filter_key_0": "FILTER",
                "g": "g",
                "r": "r",
                "i": "i",
                "z": "z",
                "u": "u",
                "mjd": "MJD-OBS",
                "date": "DATE-OBS",
                "gain": "GAIN",
                "exptime": "EXPTIME",
                "pixel_scale": 0.4,
            },
        },
        "location": {"name": None, "lon": None, "lat": None, "alt": None},
    },
    "2MASS": {
        "INSTRUME": {
            "2MASS": {
                "Name": "2MASS",
                "filter_key_0": "FILTER",
                "J": "J",
                "H": "H",
                "K": "K",
                "Ks": "K",
                "K": "K",
                "mjd": "MJD-OBS",
                "date": "DATE-OBS",
                "gain": "GAIN",
                "exptime": "EXPTIME",
                "pixel_scale": 1.0,
            },
        },
        "location": {"name": None, "lon": None, "lat": None, "alt": None},
    },
    "ESO-VST": {
        "INSTRUME": {
            "OMEGACAM": {
                "Name": "ESO-VST/OMEGACAM",
                "filter_key_0": "FILTER",
                "g": "g",
                "r": "r",
                "i": "i",
                "z": "z",
                "u": "u",
                "g_SDSS": "g",
                "r_SDSS": "r",
                "i_SDSS": "i",
                "u_SDSS": "u",
                "mjd": "MJD-OBS",
                "date": "DATE-OBS",
                "gain": "GAIN",
                "exptime": "EXPTIME",
                "pixel_scale": 0.21,
            },
        },
        "location": {"name": None, "lon": None, "lat": None, "alt": None},
    },
    "ESO-VISTA": {
        "INSTRUME": {
            "VIRCAM": {
                "Name": "ESO-VISTA/VIRCAM",
                "filter_key_0": "FILTER",
                "J": "J",
                "H": "H",
                "K": "K",
                "Ks": "K",
                "mjd": "MJD-OBS",
                "date": "DATE-OBS",
                "gain": "GAIN",
                "exptime": "EXPTIME",
                "pixel_scale": 0.34,
            },
        },
        "location": {"name": None, "lon": None, "lat": None, "alt": None},
    },
}

# Instrument block keys in telescope.yml (only INSTRUME is supported).
INSTRUMENT_BLOCK_KEYS = ("INSTRUME",)


def get_instrument_config(telescope_data, telescope, instrument):
    """
    Resolve instrument config from telescope_data using INSTRUME only.
    Returns (block_key, config) e.g. ("INSTRUME", {...}) or (None, None) if not found.
    """
    tele_block = telescope_data.get(telescope) or {}
    for block_key in INSTRUMENT_BLOCK_KEYS:
        inst_block = tele_block.get(block_key) or {}
        if instrument in inst_block:
            return block_key, inst_block[instrument]
    return None, None


def load_telescope_config(wdir):
    """
    Load telescope.yml from wdir and merge with built-in defaults.
    Images must have TELESCOP and INSTRUME header keywords; telescope.yml lists
    all supported telescopes/instruments. User config overrides built-in.
    Returns merged dict keyed by TELESCOP then INSTRUME.
    """
    logger = logging.getLogger(__name__)
    out = copy.deepcopy(BUILTIN_TELESCOPE_DEFAULTS)

    # Only load from wdir telescope.yml (wdir-specific configuration)
    user_path = os.path.join(wdir, "telescope.yml")
    loaded_sources = []

    def _safe_load_yaml(path):
        try:
            if path and os.path.isfile(path):
                with open(path, "r") as stream:
                    data = yaml.safe_load(stream) or {}
                if data:
                    loaded_sources.append(os.path.abspath(path))
                return data
        except Exception:
            pass
        return {}

    def _deep_merge_into(base_out, loaded):
        # Deep-merge config into base: per-telescope and per-instrument entries are merged,
        # so an override of e.g. pixel_scale does not wipe existing filter mappings.
        if not isinstance(loaded, dict):
            return base_out
        for tele, block in loaded.items():
            base = base_out.get(tele, {})
            if not isinstance(base, dict):
                base = {}
            if not isinstance(block, dict):
                base_out[tele] = base
                continue
            for key, value in block.items():
                if key == "INSTRUME" and isinstance(value, dict):
                    inst_block = base.get("INSTRUME", {})
                    if not isinstance(inst_block, dict):
                        inst_block = {}
                    for inst_name, inst_cfg in value.items():
                        base_inst_cfg = inst_block.get(inst_name, {})
                        if not isinstance(base_inst_cfg, dict):
                            base_inst_cfg = {}
                        base_inst_cfg.update(inst_cfg or {})
                        inst_block[inst_name] = base_inst_cfg
                    base["INSTRUME"] = inst_block
                else:
                    base[key] = value
            base_out[tele] = base
        return base_out

    out = _deep_merge_into(out, _safe_load_yaml(user_path))
    if loaded_sources:
        logger.info("telescope.yml loaded from: %s", " (merged) ".join(loaded_sources))
    else:
        logger.info(
            "telescope.yml: using built-in defaults only (no file found at %r)",
            user_path,
        )
    return out


def compute_target_crowding(
    image,
    center_xy,
    box_half_size=50,
    nsigma=3.0,
    npixels=5,
    deblend=True,
    deblend_nlevels=32,
    deblend_contrast=0.001,
    neighbor_radius_pix=30.0,
    mask_dilate_pix=2,
    max_neighbors=50,
):
    """
    Compute a simple crowded-field diagnostic around a target position.

    Returns a dict with:
      - ok: bool
      - crowding_radius_pix, box_half_size
      - n_sources_total (in cutout)
      - n_neighbors_within_radius
      - nearest_neighbor_sep_pix (or None)
      - segmentation (2D int array, cutout coordinates; 0=background)
      - neighbor_mask (2D bool array, cutout coordinates; True=neighbor pixels)
      - neighbors (list of dicts with x_pix,y_pix,sep_pix in full-image pixels)
    """
    import numpy as np

    cx, cy = center_xy
    if cx is None or cy is None:
        return {"ok": False, "reason": "center_xy is None"}

    try:
        cx = float(cx)
        cy = float(cy)
    except Exception:
        return {"ok": False, "reason": "center_xy not numeric"}

    ny, nx = image.shape[:2]
    x0 = int(max(0, np.floor(cx - box_half_size)))
    x1 = int(min(nx, np.ceil(cx + box_half_size + 1)))
    y0 = int(max(0, np.floor(cy - box_half_size)))
    y1 = int(min(ny, np.ceil(cy + box_half_size + 1)))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return {"ok": False, "reason": "cutout too small"}

    cut = np.asarray(image[y0:y1, x0:x1], dtype=float)
    if not np.any(np.isfinite(cut)):
        return {"ok": False, "reason": "cutout all non-finite"}

    # Background / noise estimate for thresholding (robust to outliers)
    finite = cut[np.isfinite(cut)]
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.std(finite)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(finite)
    if not np.isfinite(sigma) or sigma <= 0:
        return {"ok": False, "reason": "cannot estimate noise"}

    # Segmentation/deblending (photutils) with safe fallbacks
    try:
        from photutils.segmentation import (
            detect_sources,
            detect_threshold,
            deblend_sources,
            SourceCatalog,
        )
    except Exception as exc:
        return {"ok": False, "reason": f"photutils.segmentation unavailable: {exc}"}

    threshold = detect_threshold(cut, nsigma=nsigma, background=med, error=sigma)
    segm = detect_sources(cut, threshold, npixels=npixels)
    if segm is None:
        return {
            "ok": True,
            "crowding_radius_pix": float(neighbor_radius_pix),
            "box_half_size": int(box_half_size),
            "n_sources_total": 0,
            "n_neighbors_within_radius": 0,
            "nearest_neighbor_sep_pix": None,
            "segmentation": np.zeros_like(cut, dtype=int),
            "neighbor_mask": np.zeros_like(cut, dtype=bool),
            "neighbors": [],
        }

    if deblend:
        try:
            segm = deblend_sources(
                cut,
                segm,
                npixels=npixels,
                nlevels=int(deblend_nlevels),
                contrast=float(deblend_contrast),
                progress_bar=False,
            )
        except Exception:
            # If deblending fails, keep the original segmentation.
            pass

    seg_data = np.asarray(segm.data, dtype=int)
    n_sources_total = int(np.nanmax(seg_data))

    # Catalog for centroids
    try:
        cat = SourceCatalog(cut, segm)
        xcen = np.asarray(cat.xcentroid)
        ycen = np.asarray(cat.ycentroid)
    except Exception:
        xcen = np.array([])
        ycen = np.array([])

    # Convert centroids to full-image pixels
    neighbors = []
    if xcen.size and ycen.size:
        dx = (x0 + xcen) - cx
        dy = (y0 + ycen) - cy
        sep = np.hypot(dx, dy)

        # Identify the segment containing the target pixel (cutout coords)
        tx = int(np.clip(round(cx) - x0, 0, cut.shape[1] - 1))
        ty = int(np.clip(round(cy) - y0, 0, cut.shape[0] - 1))
        target_label = int(seg_data[ty, tx])

        order = np.argsort(sep)
        for idx in order[: max_neighbors + 1]:
            if not np.isfinite(sep[idx]):
                continue
            label = int(
                idx + 1
            )  # SourceCatalog order corresponds to labels for simple cases
            # Prefer excluding by target_label if possible; otherwise exclude by sep ~ 0.
            if target_label > 0 and label == target_label:
                continue
            if sep[idx] < 1e-6:
                continue
            neighbors.append(
                {
                    "x_pix": float(x0 + xcen[idx]),
                    "y_pix": float(y0 + ycen[idx]),
                    "sep_pix": float(sep[idx]),
                }
            )
        # Keep only within neighbor radius for metrics/list
        neighbors_within = [n for n in neighbors if n["sep_pix"] <= neighbor_radius_pix]
    else:
        # If we couldn't build a catalog, still provide segmentation outputs
        tx = int(np.clip(round(cx) - x0, 0, cut.shape[1] - 1))
        ty = int(np.clip(round(cy) - y0, 0, cut.shape[0] - 1))
        target_label = int(seg_data[ty, tx])
        neighbors_within = []

    # Neighbor mask: all segments except the target segment
    neighbor_mask = seg_data > 0
    if target_label > 0:
        neighbor_mask &= seg_data != target_label

    if mask_dilate_pix and mask_dilate_pix > 0 and np.any(neighbor_mask):
        try:
            from scipy.ndimage import binary_dilation

            neighbor_mask = binary_dilation(
                neighbor_mask, iterations=int(mask_dilate_pix)
            )
        except Exception:
            pass

    n_neighbors_within_radius = int(len(neighbors_within))
    nearest_neighbor_sep_pix = (
        float(min(n["sep_pix"] for n in neighbors_within)) if neighbors_within else None
    )

    return {
        "ok": True,
        "crowding_radius_pix": float(neighbor_radius_pix),
        "box_half_size": int(box_half_size),
        "n_sources_total": int(n_sources_total),
        "n_neighbors_within_radius": n_neighbors_within_radius,
        "nearest_neighbor_sep_pix": nearest_neighbor_sep_pix,
        "segmentation": seg_data,
        "neighbor_mask": np.asarray(neighbor_mask, dtype=bool),
        "neighbors": neighbors_within[:max_neighbors],
        "cutout_bbox_xyxy": (x0, x1, y0, y1),
        "cutout_center_xy": (cx - x0, cy - y0),
    }


class AutophotYaml:

    def __init__(self, filepath=None, dict_name=None, wdir=None):

        self.filepath = filepath
        self.dict_name = dict_name
        self.wdir = wdir

    def load(self):

        if self.wdir != None:
            file_path = os.path.join(self.wdir, self.filepath)
        else:
            file_path = self.filepath

        with open(file_path, "r") as stream:
            var = yaml.safe_load(stream)

        if self.dict_name != None:
            data = var[self.dict_name]
        else:
            data = var

        return data

    def update(self, tele, inst_key, inst, key, new_val):

        doc = {key: new_val}

        with open(self.filepath, "r") as yamlfile:

            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml_backup = copy.deepcopy(cur_yaml)

            try:

                cur_yaml[tele][inst_key][inst].update(doc)
            except Exception:
                cur_yaml = cur_yaml_backup

        with open(self.filepath, "w+") as yamlfile:

            yaml.safe_dump(cur_yaml, yamlfile, default_flow_style=False)

    def create(fname, data):

        import yaml
        import os

        target_name = fname

        if ".yml" not in fname:
            fname += ".yml"

        data_new = {os.path.basename(target_name.replace(".yml", "")): data}
        with open(fname, "w") as outfile:
            yaml.dump(data_new, outfile, default_flow_style=False)


def get_header(fpath):
    from astropy.io.fits import getheader
    from astropy.io import fits

    """
    Robust function to get header from a FITS image for use in AutoPHOT. This function aims to find the correct
    telescope header information based on the "Telescop" header key. FITS images may contain multiple headers, 
    so the function combines them if necessary.

    :param fpath: Path to the FITS file.
    :type fpath: str
    :return: Combined header information.
    :rtype: Header object
    """
    try:
        # Attempt to open FITS file with 'ignore_missing_end' to handle incomplete files
        with fits.open(fpath, ignore_missing_end=True) as hdul:
            hdul.verify("silentfix+ignore")  # Try to fix any issues with the file

            # FITS keywords are typically uppercase; check case-insensitively for TELESCOP
            def has_telescop(header):
                return any(k.upper() == "TELESCOP" for k in header.keys())

            if has_telescop(hdul[0].header):
                headinfo = hdul[0].header.copy()
            else:
                # If not in primary, use first HDU that has TELESCOP (e.g. extension with image)
                for i in range(1, len(hdul)):
                    if has_telescop(hdul[i].header):
                        headinfo = hdul[i].header.copy()
                        break
                else:
                    headinfo = hdul[0].header.copy()
    except KeyError as e:
        # Handle missing or incorrect header keys (e.g., 'Telescop')
        raise Exception(f"KeyError: The required header keyword was not found: {e}")

    except Exception as e:
        # General exception handling, including file issues or unexpected errors
        raise Exception(f"An error occurred while reading the FITS file: {e}")

    # If the header is a list (indicating multiple HDUs), combine them
    if isinstance(headinfo, list):
        combined_header = headinfo[0].header

        for ext in headinfo[1:]:
            combined_header.update(ext.header)

        headinfo = combined_header

    return headinfo


def get_image_and_header(fpath):
    """
    Load FITS image and header with a single file open. Same semantics as
    get_image(fpath) and get_header(fpath), but avoids opening the file twice.

    :param fpath: Path to the FITS file.
    :return: (image, header) where image is a 2D numpy array copy and header is a copy.
    """
    import os
    from astropy.io import fits

    try:
        with fits.open(fpath, ignore_missing_end=True) as hdul:
            hdul.verify("silentfix+ignore")
            # Enhanced HDU selection for better FITS file support
            image = None
            best_hdu_idx = None
            
            # Strategy 1: Try 'sci' extension first (Hubble convention)
            try:
                sci_data = hdul["sci"].data
                if sci_data is not None and hasattr(sci_data, 'shape') and len(sci_data.shape) >= 2:
                    image = np.asarray(sci_data).copy()
                    # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
                    if image.dtype.kind != 'f':
                        image = image.astype(np.float32)
                    best_hdu_idx = hdul.index_of("sci")
                    print(f"Using 'sci' extension (HDU {best_hdu_idx}) with shape {image.shape}")
            except (KeyError, TypeError):
                pass
            
            # Strategy 2: If no 'sci' extension, find best HDU with 2D+ data
            if image is None:
                candidates = []
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None:
                        try:
                            test_image = np.asarray(hdu.data)
                            if hasattr(test_image, 'shape') and len(test_image.shape) >= 2:
                                # Prefer larger images and Primary/Image HDUs
                                score = 0
                                if isinstance(hdu, fits.PrimaryHDU):
                                    score += 10
                                elif isinstance(hdu, fits.ImageHDU):
                                    score += 5
                                score += np.log10(test_image.size) if test_image.size > 0 else 0
                                candidates.append((score, i, test_image))
                        except Exception:
                            continue
                
                if candidates:
                    # Sort by score and take the best candidate
                    candidates.sort(reverse=True)
                    best_score, best_idx, best_image = candidates[0]
                    image = best_image.copy()
                    # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
                    if image.dtype.kind != 'f':
                        image = image.astype(np.float32)
                    best_hdu_idx = best_idx
                    print(f"Selected HDU {best_idx} (score={best_score:.1f}) with shape {image.shape}")
            
            # Strategy 3: Last resort - try primary HDU
            if image is None and len(hdul) > 0:
                primary_data = hdul[0].data
                if primary_data is not None:
                    try:
                        test_image = np.asarray(primary_data)
                        if hasattr(test_image, 'shape'):
                            print(f"Using primary HDU as fallback with shape {getattr(test_image, 'shape', 'no shape')}")
                            image = test_image.copy()
                            # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
                            if image.dtype.kind != 'f':
                                image = image.astype(np.float32)
                            best_hdu_idx = 0
                    except Exception as e:
                        print(f"Error with primary HDU: {e}")
            
            # Final validation and error handling
            if image is None:
                # Print detailed HDU information for debugging
                print("HDU structure analysis:")
                for i, hdu in enumerate(hdul):
                    data_info = f"shape={getattr(hdu.data, 'shape', 'None')}" if hdu.data is not None else "None"
                    print(f"  HDU {i}: {hdu.__class__.__name__}, name='{hdu.name}', data={data_info}")
                raise Exception(f"No valid 2D+ image data found in FITS file: {os.path.basename(fpath)}")
            
            # Handle multi-dimensional data by taking first 2D slice
            if hasattr(image, 'shape') and len(image.shape) > 2:
                base = os.path.basename(fpath)
                print(f"Warning: {base} has {len(image.shape)}D data, taking first 2D slice")
                original_shape = image.shape
                while len(image.shape) > 2:
                    image = image[0]
                image = image.copy()
                print(f"  Reshaped from {original_shape} to {image.shape}")
            elif not hasattr(image, 'shape') or len(image.shape) < 2:
                base = os.path.basename(fpath)
                raise Exception(f"Warning: {base} is not a 2D array (found {getattr(image, 'shape', 'no shape')} data).")

            # Header: use the header from the same HDU that contains the image data
            # to ensure WCS keywords are preserved. Merge with TELESCOP header if different.
            def has_telescop(header):
                return any(k.upper() == "TELESCOP" for k in header.keys())
            
            # Start with header from the HDU containing the image data
            headinfo = hdul[best_hdu_idx].header.copy()
            
            # If the image HDU doesn't have TELESCOP, try to find it in other HDUs
            if not has_telescop(headinfo):
                for i in range(len(hdul)):
                    if has_telescop(hdul[i].header):
                        # Copy TELESCOP and related instrument keywords
                        telescop_header = hdul[i].header
                        for key in telescop_header.keys():
                            if key not in headinfo and (
                                key.upper() in ['TELESCOP', 'INSTRUME', 'FILTER', 
                                               'EXPTIME', 'MJD-OBS', 'DATE-OBS',
                                               'GAIN', 'RDNOISE', 'SATURATE']
                            ):
                                headinfo[key] = telescop_header[key]
                        break
            
            return image, headinfo
    except KeyError as e:
        raise Exception(f"KeyError: The required header keyword was not found: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the FITS file: {e}")


def get_image(fpath):

    import os
    from astropy.io import fits

    """
    Given a FITS file, this function attempts to retrieve the 2D image data using the "sci" extension. 
    If the image is not 2D or if the file is a FITS cube, an error will be raised.

    :param fpath: Path to the FITS file.
    :type fpath: str
    :return: 2D image data array.
    :rtype: numpy.ndarray
    :raises Exception: If the image is not 2D or an error occurs during reading.
    """
    try:
        # Try to get 2D image from 'sci' extension
        image = fits.getdata(fpath, extname="sci")

    except Exception:
        # If 'sci' extension fails, try getting image from the primary HDU
        image = fits.getdata(fpath)

    # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
    # Integer dtypes cannot represent NaN values, so we convert to float32
    if image.dtype.kind != 'f':
        image = image.astype(np.float32)

    # Check if the image data is a 2D array
    if len(image.shape) != 2:
        base = os.path.basename(fpath)
        raise Exception(f"Warning: {base} is not a 2D array.")

    return image


def concatenate_csv_files(folder_path, output_filename, loc_file="output.csv"):
    """
    Concatenate multiple CSV files into a single output file, ensuring empty cells are treated as NaN.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing CSV files to concatenate
    output_filename : str
        Name of the output concatenated CSV file
    loc_file : str, optional
        Name of the CSV files to look for in subdirectories (default: 'output.csv')
    """

    # Initialize an empty list to hold DataFrames
    concatenated_data = []

    from fnmatch import fnmatch

    # Traverse the folder using os.walk
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Support wildcard patterns, e.g. loc_file="OUTPUT_*.csv"
            if ("*" in loc_file) and fnmatch(file, loc_file):
                file_path = os.path.join(root, file)

                # Read CSV with empty cells treated as NaN
                df = pd.read_csv(
                    file_path,
                    keep_default_na=True,
                    na_values=["", " ", "NA", "N/A", "NaN", "null"],
                    dtype=str,  # Read all columns as string to preserve blanks before NaN replacement
                )

                # Replace all empty strings or whitespace-only strings with np.nan
                df = df.map(
                    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
                )

                concatenated_data.append(df)
            elif file == loc_file:
                file_path = os.path.join(root, file)

                # Read CSV with empty cells treated as NaN
                df = pd.read_csv(
                    file_path,
                    keep_default_na=True,
                    na_values=["", " ", "NA", "N/A", "NaN", "null"],
                    dtype=str,  # Read all columns as string to preserve blanks before NaN replacement
                )

                # Replace all empty strings or whitespace-only strings with np.nan
                df = df.map(
                    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
                )

                concatenated_data.append(df)

    if not concatenated_data:
        logging.getLogger(__name__).info(
            "No CSV files matched the requested pattern; nothing to concatenate."
        )
        return None

    # Concatenate all DataFrames
    concatenated_data = pd.concat(concatenated_data, ignore_index=True)

    # Write the concatenated data to the output file
    concatenated_data.to_csv(
        output_filename, index=False, na_rep="NaN"
    )  # Explicit NaN representation

    logging.getLogger(__name__).info(
        "Concatenated %d rows of tabular data into '%s'.",
        len(concatenated_data),
        output_filename,
    )

    return output_filename


def pix_dist(x1, x2, y1, y2):
    """
    Find the linear distance between two sets of points (x1,y1) -> (x2,y2)
    given by:

    .. math ::

       d = \\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}

    :param x1: x position of point 1
    :type x1: float
    :param x2: x position of point 2
    :type x2: float
    :param y1: y position of point 1
    :type y1: float
    :param y2: y position of point 2
    :type y2: float
    :return: Distance between to points
    :rtype: float

    """

    z1 = (x1 - x2) ** 2
    z2 = (y1 - y2) ** 2

    r = np.sqrt(z1 + z2)

    return r


def gauss_1d(x, A, x0, sigma):
    """
     1D gaussian function given by:

    .. math::

    G = A \\times e^{-\\frac{x-x_o}{2\\times \\sigma^2}}

     where *G* is the 1D gaussian function, *A* is the amplitude, *x* is the linear
     range of the function, :math:`x_0` is the center of the function, and
     :math:`\\sigma` is the standard deviation.


     :param x: Linear range of gaussian function
     :type x: 1D array
     :param A: Amplitude of gaussian function
     :type A: float
     :param x0: Center/maximum of gaussian function
     :type x0: float
     :param sigma: sigma/width of gaussian function
     :type sigma: float
     :return: Returns 1 dimensional function with length equal to length of input x
     array
     :rtype: 1D array

    """

    G = A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    return G


def snr(maxPixel, noiseBkg):

    with warnings.catch_warnings():

        snr_value = maxPixel / noiseBkg

    return snr_value


def snr_err(snr_value):
    """
     Error associated with signal to noise ratio (S/N). Equation  taken from `here <https://www.ucolick.org/~bolte/AY257/s_n.pdf>`_. Whe can associate the error on the instrumental magnitude of a source as:


    .. math ::

        m \\pm \\delta m = -2.5 \\times \\log_{10} ( S \\pm N)

        m \\pm \\delta m = -2.5 \\times \\log_{10} ( S  (1 \\pm N / S ) )

        m \\pm \\delta m = -2.5 \\times \\log_{10} ( S )   - 2.5 \\times \\log_{10}(1 \\pm N / S ) )

        \\delta m = \\mp 2.5\\times \\log_{10} (1 + \\frac{1}{S/N}) \\approx \\mp 1.0875 (N / S)

     :param snr_value: Signal-to-noise ratio of a point-like source.
     :type snr_value: float
     :return: Error associated with that source's S / N
     :rtype: float

    """

    from numpy import log10, errstate

    with errstate(divide="ignore", invalid="ignore"):
        snr_err_value = 2.5 * log10(1 + (1 / snr_value))

    return snr_err_value


def quadrature_add(values):
    from numpy import sqrt

    return sqrt(sum([i**2 for i in values]))


def moffat_2d(image, x0, y0, sky, A, image_params):
    """
     Returns 2D moffat function which is given by:


    .. math::

    M = A\\times (1+\\frac{(x-x_o)^2 + (y-y_0)^2}{\\sigma^2})^{-\\beta} +
    sky


     `Credit: ltam
     <https://www.ltam.lu/physique/astronomy/projects/star_prof/star_prof.html>`_


     :param image: 2 dimensions grid to map Moffat on
     :type image: 2D  array
     :param x0: x-center of Moffat function
     :type x0: float
     :param y0: y-center of Moffat function
     :type y0: float
     :param sky: sky/offset of Moffat function
     :type sky: float
     :param A: Amplitude of Moffat function
     :type A: float
     :param image_params: Dictionary containing the keys "alpha" and "beta" with
     their corresponding values
     :type image_params: dict
     :return: 2D Moffat function with the same shape as image input
     :rtype: 2D  array

    """
    x, y = image

    alpha = image_params["alpha"]
    beta = image_params["beta"]

    a = (x - x0) ** 2

    b = (y - y0) ** 2

    c = (a + b) / (alpha**2)

    d = (1 + c) ** -beta

    e = (A * d) + sky

    return e.flatten()


def mag(flux):
    """
    Calculate magnitude of a point source (instrumental: -2.5 * log10(flux)).

    Does not mutate the input. Non-positive flux values yield NaN in the output.

    :param flux: Flux in counts per second measured from source
    :type flux: float or array
    :return: Instrumental magnitude; NaN where flux <= 0
    :rtype: float or array
    """
    import pandas as pd

    if isinstance(flux, (int, float)):
        if flux <= 0:
            return np.nan
        return -2.5 * np.log10(float(flux))
    if isinstance(flux, (pd.core.series.Series, np.ndarray)):
        # Use a copy for the log so we do not mutate the caller's catalog/flux array
        flux_safe = np.asarray(flux, dtype=float).copy()
        flux_safe[flux_safe <= 0] = np.nan
        return -2.5 * np.log10(flux_safe)
    return np.nan


def rebin(arr, new_shape):
    """
     Rebin an array into a specific 2D shape

    :param arr: Array of values
    :type arr: array
    :param new_shape: New shape with which to rebin array into
    :type new_shape: tuple
    :return: rebinned array
    :rtype: array

    """
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(shape).mean(-1).mean(1)


def scale_roll(x, xc, m):
    """
    Used in building PSF function. When shiting and aligning residual tables this
    functions translates pixel shifts between different images cutouts.


    :param x: pixel position
    :type x: gloat
    :param xc: pixel position to which we want to move to
    :type xc: float
    :param m: scale multiplier
    :type m: int
    :return: DESCRIPTION
    :rtype: TYPE

    """

    dx = x - xc

    if m != 1:

        shift = int(round(dx * m))

    else:

        shift = int(dx * m)

    return shift


def remove_wcs_from_header(header):
    """
    Remove all WCS information from the header of a FITS file so that
    new WCS keywords can be merged without conflicts (e.g. after plate solving).

    Parameters:
    -----------
    header : fits.Header
        FITS header to modify in place.

    Returns:
    --------
    fits.Header
        The same header with WCS keywords removed.
    """
    # Prefixes: remove any key that starts with one of these
    wcs_prefixes = [
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
        "RADYSYS",  # RADYSYS typo in some headers
        "LONGPOLE",
        "TNX",
        "SIP_",
    ]
    # For SIP / polynomial distortion: key starts with this stem and contains '_'
    # (e.g. A_ORDER, A_0_0, B_1_2, AP_0_0, BP_2_1, D_*, DP_*)
    wcs_stem_underscore = ["A_", "B_", "AP_", "BP_", "D_", "DP_", "PV"]

    keys = list(header.keys())
    for key in keys:
        if key in ("NAXIS", "NAXIS1", "NAXIS2", "COMMENT", "HISTORY"):
            continue
        remove = False
        for prefix in wcs_prefixes:
            if key.startswith(prefix):
                remove = True
                break
        if not remove:
            stem = key.split("_")[0] + "_" if "_" in key else ""
            if stem in wcs_stem_underscore and key.startswith(stem.rstrip("_")):
                remove = True
        if remove:
            try:
                del header[key]
            except KeyError:
                pass

    # Add a comment to indicate WCS was removed
    header["COMMENT"] = "WCS information removed from this header"
    return header


def convert_ra_dec_to_hms_dms(ra_deg, dec_deg):
    # Create a SkyCoord object using RA and DEC in degrees
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk5", equinox="J2000")
    # Convert to the required format
    ra_str = coord.ra.to_string(unit=u.hour, sep=":", precision=1)
    dec_str = coord.dec.to_string(sep=":", precision=1, alwayssign=True)
    return f"{ra_str}, {dec_str}"


def gaussian(gridx, gridy, x0, y0, sky, A, sigma):
    """
    2D gaussian function given by:

    .. math::

    G = A \\times e^{-\\frac{(x-x_o)^2 - (y-y_0)^2}{2\\times \\sigma^2}} + sky

    where *G* is the 2D gaussian function, *A* is the amplitude, *x* and *y* are the linear
    range of the function, :math:`x_0` and :math:`y_0` are the centers of the function,
    :math:`\\sigma` is the standard deviation, and *sky* is the amplitude offset of the function

    :param image: 2 dimensional grid to map Gaussian onto
    :type image: 2D array
    :param x0: x-center of gaussian function
    :type x0: float
    :param y0: y-center of gaussian function
    :type y0: float
    :param sky: sky/offset of gaussian function
    :type sky: float
    :param A: Amplitude of gaussian function
    :type A: float
    :param image_params: Dictionary containing the key *sigma* with corresponding value
    :type image_params: dict
    :return: 2D gaussian function with the same shape as image input
    :rtype: 2D array
    """
    from numpy import exp, array

    x = gridx
    y = gridy

    a = array(x - x0) ** 2

    b = array(y - y0) ** 2

    c = 2 * sigma**2

    d = A * exp(-1 * (a + b) / c)

    e = d + sky

    return e


def moffat(gridx, gridy, x0, y0, sky, A, alpha, beta=4.675):
    """
     Returns 2D moffat function which is given by:


    .. math::

    M = A\\times (1+\\frac{(x-x_o)^2 + (y-y_0)^2}{\\sigma^2})^{-\\beta} +
    sky


     `Credit: ltam
     <https://www.ltam.lu/physique/astronomy/projects/star_prof/star_prof.html>`_


     :param image: 2 dimensions grid to map Moffat on
     :type image: 2D  array
     :param x0: x-center of Moffat function
     :type x0: float
     :param y0: y-center of Moffat function
     :type y0: float
     :param sky: sky/offset of Moffat function
     :type sky: float
     :param A: Amplitude of Moffat function
     :type A: float
     :param image_params: Dictionary containing the keys "alpha" and "beta" with
     their corresponding values
     :type image_params: dict
     :return: 2D Moffat function with the same shape as image input
     :rtype: 2D  array

    """
    x = gridx
    y = gridy

    a = (x - x0) ** 2

    b = (y - y0) ** 2

    c = (a + b) / (alpha**2)

    d = (1 + c) ** -beta

    e = (A * d) + sky

    return e


def fwhm_moffat(alpha, beta):
    """

    Calculate FWHM from Moffat function using:

    .. math::

       FWHM = 2 \\times \alpha \\times \\sqrt{2^{\\frac{1}{\\beta}}-1}

    where :math:`\alpha` corresponds to the width of moffat function and :math:`\\beta` describes the wings

    :param image_params: Dictionary containing 2 keys: *alpha* corresponding to the fitted width of the moffat function and *beta* describing the wings.
    :type image_params: dict
    :return: Full width half maximum of moffat function
    :rtype: float

    """

    from numpy import sqrt

    fwhm = 2 * alpha * sqrt((2 ** (1 / beta)) - 1)

    return fwhm


def fwhm_gaussian(sigma):

    from numpy import sqrt, log

    fwhm = 2 * sqrt(2 * log(2)) * sigma

    return fwhm


def sigma_gaussian(fwhm):

    from numpy import sqrt, log

    sigma = fwhm / (2 * sqrt(2 * log(2)))

    return sigma


def alpha_moffat(fwhm, beta=4.675):
    """

    Calculate FWHM from Moffat function using:

    .. math::

       FWHM = 2 \\times \alpha \\times \\sqrt{2^{\\frac{1}{\\beta}}-1}

    where :math:`\alpha` corresponds to the width of moffat function and :math:`\\beta` describes the wings

    :param image_params: Dictionary containing 2 keys: *alpha* corresponding to the fitted width of the moffat function and *beta* describing the wings.
    :type image_params: dict
    :return: Full width half maximum of moffat function
    :rtype: float

    """

    from numpy import sqrt

    alpha = 0.5 * fwhm * 1 / (sqrt(2 ** (1 / beta) - 1))

    return alpha


def trim_zeros_slices(arr):
    """

    TriM a 2D array of horizontal or vertical rows completely filled with zeroes. This is useful when aligning two images When  doing so if there isn't significant overlap between the two images, the resultant images may have vertical and horizontal lines completely filled with zeroes. This function will accept an image with said zeroed columns/row and return a smaller image with those arrays removed. This function will not exclude partially filled columns or rows.

    Credit: `Stackoverflow <https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d>`_


    :param arr: 2D array with horizontal or vertical rows/columns filled with zeroes.
    :type arr: 2D array.
    :return: 2D array which has been cleaned of zero columns and index map for original array.
    :rtype: tuple

    """

    boolean_array = np.zeros(arr.shape).astype(bool)

    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))

    boolean_array[slices] = True

    return arr[slices], boolean_array


def distance_to_uniform_row_col(image, x, y):

    # Ensure the input is a numpy array for easier manipulation
    image = np.array(image)
    rows, cols = image.shape

    # Find rows with all same values
    uniform_rows = [i for i in range(rows) if np.all(image[i] == image[i, 0])]

    # Find columns with all same values
    uniform_cols = [j for j in range(cols) if np.all(image[:, j] == image[0, j])]

    # Calculate the Manhattan distance to the nearest uniform row
    if uniform_rows:
        row_distances = [abs(x - row) for row in uniform_rows]
        min_row_distance = min(row_distances)
    else:
        min_row_distance = float("inf")  # If no uniform rows are found

    # Calculate the Manhattan distance to the nearest uniform column
    if uniform_cols:
        col_distances = [abs(y - col) for col in uniform_cols]
        min_col_distance = min(col_distances)
    else:
        min_col_distance = float("inf")  # If no uniform columns are found

    # Return the minimum distance to a uniform row or column
    return min(min_row_distance, min_col_distance)


def points_in_circum(r, center, n=8):
    """
    Generate series of x,y coordinates a distance r from the specified center,
    rounded to the nearest pixel.

    :param r: Distance from center
    :type r: float

    :param center: (x_center, y_center) pixel coordinates
    :type center: Tuple[float, float]

    :param n: Number of points, defaults to 8
    :type n: int, optional

    :return: List of x,y coordinates placed around the center at angles 2*pi/n * i
             and rounded to nearest pixel
    :rtype: List[Tuple[int, int]]
    """

    x_center, y_center = center
    return [
        (
            round(np.cos(2 * np.pi / n * i) * r + x_center),
            round(np.sin(2 * np.pi / n * i) * r + y_center),
        )
        for i in range(n)
    ]


def flux_upper_limit(n, sigma, beta_p):

    peak_flux = (n + (np.sqrt(2) * erfinv((2 * beta_p) - 1))) * sigma

    return peak_flux


def create_ds9_region_file(
    x_list,
    y_list,
    radius,
    filename="ds9_region.reg",
    color="green",
    text="",
    overwrite=False,
    correct_position=True,
):
    """
    Create a DS9 region file containing circular regions for multiple points.

    Parameters:
    - x_list (list): List of x positions for the centers of the circles.
    - y_list (list): List of y positions for the centers of the circles.
    - radius (float): Radius of the circles.
    - filename (str): Name of the DS9 region file to be created.

    Returns:
    - None
    """

    cor = 0
    if correct_position:
        cor = 1
    if len(x_list) != len(y_list):
        raise ValueError("Number of x and y positions must be the same.")

    region_content = ""
    for x, y in zip(x_list, y_list):
        region_content += (
            f"circle({x+cor}, {y+cor}, {radius}) # color={color} text={text}\n"
        )

    if overwrite:
        n = "w"
    else:
        n = "a+"

    with open(filename, n) as file:

        file.write(region_content)
    return 1


def write_position_2_ascii(dataframe, output_file):
    """
    Write x_pix and y_pix columns from a Pandas DataFrame to an ASCII file.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame with x_pix and y_pix columns.
    - output_file (str): Output ASCII file name.

    Returns:
    - None
    """

    # Extract x_pix and y_pix columns
    if "x_pix" not in dataframe.columns or "y_pix" not in dataframe.columns:
        raise ValueError("DataFrame must have 'x_pix' and 'y_pix' columns.")

    x_pix_column = dataframe["x_pix"].values
    y_pix_column = dataframe["y_pix"].values

    # Combine columns into a new DataFrame
    output_dataframe = pd.DataFrame({"x_pix": x_pix_column, "y_pix": y_pix_column})

    # Write DataFrame to ASCII file with "X Y" header
    with open(output_file, "w") as file:
        file.write("x y\n")  # Header line
        output_dataframe.to_csv(
            file, sep=" ", header=None, index=False, float_format="%.3f"
        )


def print_progress_bar(
    iterable, total=None, prefix="", length=30, fill="#", title=None
):
    """
    Print a progress bar in the terminal for a loop.
    Parameters:
        iterable (iterable): The iterable object (e.g., list, range) that you're iterating over.
        total (int, optional): Total number of iterations. If None, the length of the iterable will be used.
        prefix (str, optional): Prefix to display before the progress bar.
        length (int, optional): Length of the progress bar in characters.
        fill (str, optional): Character used to fill the progress bar.
        title (str, optional): Title to be displayed above the progress bar.
    Example usage:
        for i in print_progress_bar(range(100), title="Processing", prefix='Progress', length=40):
            # Your loop code here
    """
    if total is None:
        total = len(iterable)

    logger = logging.getLogger(__name__)

    def format_bar(iteration):
        percent = 100 * (iteration / float(total))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        return f"{prefix} [{bar}] {percent:5.1f}%"

    if title:
        logger.info(border_msg(title))

    last_logged = -1
    try:
        for i, item in enumerate(iterable, start=1):
            # Log progress at DEBUG to avoid cluttering INFO logs.
            current = int(100 * (i / float(total)))
            if current // 5 != last_logged // 5:
                logger.debug(format_bar(i))
                last_logged = current
            yield item
    finally:
        if last_logged < 100:
            logger.debug(format_bar(total))
        # Always emit a single INFO summary line for the completed loop.
        loop_label = None
        if title:
            # Use the first non-empty line of the title as a label.
            for line in str(title).splitlines():
                if line.strip():
                    loop_label = line.strip()
                    break
        if not loop_label:
            loop_label = prefix or "Progress"
        logger.info("%s completed: %d/%d item(s).", loop_label, int(total), int(total))


def get_normalized_histogram(data, bins="auto"):

    # Create the histogram
    data = data[~np.isnan(data)]
    if bins == "auto":
        bins = calculate_bins(data)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    # Calculate the normalization factor
    normalization_factor = np.nanmax(hist)

    # Normalize the histogram
    normalized_hist = hist / normalization_factor

    return normalized_hist, bin_edges


def dict_to_string_with_hashtag(dictionary, float_format="%.3f"):
    result = ""
    for key, value in dictionary.items():

        if isinstance(value, list):
            if len(value) == 1:
                value = value[0]

        if isinstance(value, float):
            value = float_format % value
        result += f"#{key}: {value}\n"
    return result


def safe_fits_write(fpath: str, image: np.ndarray, header: fits.Header, overwrite: bool = True, output_verify: str = "silentfix+ignore") -> None:
    """
    Write image and header to FITS file, preserving NaNs (chip gaps) by using float32 dtype.

    Integer dtypes cannot represent NaN values, so this function converts integer images
    to float32 before writing to ensure chip gaps and corrupted data regions are preserved.

    Parameters
    ----------
    fpath : str
        Path to output FITS file.
    image : np.ndarray
        Image data array.
    header : fits.Header
        FITS header.
    overwrite : bool, optional
        Overwrite existing file (default: True).
    output_verify : str, optional
        astropy.io.fits output verification mode (default: "silentfix+ignore").
    """
    # Use float32 to preserve NaNs (chip gaps) - integer dtypes cannot represent NaN
    image_to_write = image.astype(np.float32) if image.dtype.kind != 'f' else image
    fits.writeto(fpath, image_to_write, header, overwrite=overwrite, output_verify=output_verify)
