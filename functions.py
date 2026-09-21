#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utility functions for the AutoPHOT pipeline.

This module provides:

* Logging formatters (``PlainFormatter``, ``ColoredLevelFormatter``) and
  helpers (``log_step``, ``border_msg``, ``configure_console_logging``).
* FITS I/O helpers (``get_header``, ``get_image``, ``get_image_and_header``,
  ``save_to_fits``).
* Photometric utilities (``beta_aperture``, ``beta_psf``, ``snr``,
  ``snr_err``, ``mag``).
* WCS header helpers (``remove_wcs_from_header``, ``copy_wcs_from_header``,
  ``update_header_from_wcs``).
* Filter-name normalisation (``normalize_photometric_filter_name``,
  ``sanitize_photometric_filters``).
* YAML configuration loader (``AutophotYaml``).
* Miscellaneous helpers (``set_size``, ``pix_dist``, ``gauss_1d``,
  ``moffat_2d``, ``quadrature_add``).
"""

import numpy as np

import os
import re
import sys
import warnings
import pandas as pd
import yaml
import logging
import matplotlib.pyplot as plt
import traceback
import textwrap
import inspect
import unicodedata

import copy
from contextlib import contextmanager
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

logger = logging.getLogger(__name__)

# --- FITS I/O cache ---------------------------------------------------------
# mtime+size keyed cache so repeated get_header / get_image_and_header calls
# on the same (unmodified) file avoid re-opening and re-parsing the FITS file.
# Entries are invalidated when the file changes on disk (mtime/size mismatch)
# or explicitly via invalidate_fits_cache().
import threading as _threading

_FITS_CACHE_LOCK = _threading.Lock()
_FITS_HEADER_CACHE = {}   # key -> header
_FITS_IMAGE_CACHE = {}    # key -> (image, header)
_FITS_CACHE_MAX = 64      # max entries per cache


def _fits_cache_key(fpath):
    """Return a cache key tuple (path, mtime, size) or None if stat fails."""
    try:
        st = os.stat(fpath)
        return (os.path.abspath(fpath), st.st_mtime, st.st_size)
    except OSError:
        return None


def invalidate_fits_cache(fpath=None):
    """Invalidate cached FITS data for *fpath* (or the entire cache if None).

    Call this after an external tool (SWarp, SFFT, SCAMP, ...) modifies a
    FITS file on disk so the next read picks up the new content.
    """
    with _FITS_CACHE_LOCK:
        if fpath is None:
            _FITS_HEADER_CACHE.clear()
            _FITS_IMAGE_CACHE.clear()
            return
        abspath = os.path.abspath(fpath)
        # Drop all keys for this path (any mtime/size variant)
        for cache in (_FITS_HEADER_CACHE, _FITS_IMAGE_CACHE):
            stale = [k for k in cache if k[0] == abspath]
            for k in stale:
                del cache[k]


def _cache_evict(cache):
    """Evict oldest entries if cache exceeds the max size (FIFO)."""
    while len(cache) > _FITS_CACHE_MAX:
        cache.pop(next(iter(cache)))


class PlainFormatter(logging.Formatter):
    """
    Plain text formatter for log files - strips ANSI escape codes.
    
    Ensures log files contain clean, readable text without terminal
    formatting codes (bold, color, etc.) that are added by border_msg
    and other formatting functions.  Multi-line messages (e.g. border
    banners) are indented so continuation lines align with the first
    line's prefix.
    """
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        msg = self.ANSI_ESCAPE.sub('', msg)
        # Indent continuation lines so border banners and other multi-line
        # output align under the first line's timestamp/level prefix.
        if '\n' in msg:
            lines = msg.split('\n')
            dummy = copy.copy(record)
            dummy.msg, dummy.args = '', ()
            prefix = super().format(dummy)
            prefix_len = len(prefix)
            lines = [lines[0]] + [' ' * prefix_len + ln for ln in lines[1:]]
            msg = '\n'.join(lines)
        return msg


class ColoredLevelFormatter(logging.Formatter):
    """
    Logging formatter with consistent, minimal color coding for terminal output.

    Principle: Only warnings and errors get color. Routine output (INFO, DEBUG)
    remains plain black for clean, professional appearance.

    Layout:
        - No timestamp column: wall-clock time is logged once as a "Started:"
          line at the top of each run and again at the end with the elapsed
          seconds, so per-line timestamps only add noise.
        - A bordered banner opens a section; a blank line precedes each banner
          and each ``[Step]`` sub-section marker. Routine messages print flush
          left so lines stay short.
        - Non-INFO messages carry a ``[LEVEL]`` tag; wrapped continuation
          lines align under the message text.

    Color scheme:
        - INFO/DEBUG:     Plain black (no color)
        - WARNING:        Yellow (attention needed, not critical)
        - ERROR:          Red (action required)
        - CRITICAL:       Bold red (urgent failure)
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    YELLOW = "\033[33m"

    def __init__(self, *args, use_color: bool = True, compact: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_color = use_color
        self._compact = compact
        self._msg_count = 0
        self._in_section = False   # a banner has opened a pipeline section
        self._last_kind = None     # "banner" | "step" | "msg"

    @staticmethod
    def _kind(msg: str) -> str:
        first = msg.lstrip().split("\n")[0] if msg else ""
        if first[:1] in ("-", "+", "="):
            return "banner"
        if (
            first.startswith("[")
            and first.rstrip().endswith("]")
            and "\n" not in msg.strip()
        ):
            return "step"
        return "msg"

    @staticmethod
    def _align_continuation(msg: str, col: int) -> str:
        """Indent continuation lines of a multi-line message to column *col*."""
        if "\n" not in msg:
            return msg
        head, *tail = msg.split("\n")
        pad = " " * col
        return "\n".join([head] + [(pad + ln) if ln.strip() else ln for ln in tail])

    def _format_info(self, record: logging.LogRecord, msg_clean: str) -> str:
        self._msg_count += 1
        first_ever = self._msg_count == 1
        kind = self._kind(msg_clean)

        if kind == "banner":
            self._in_section = True
            lead = "" if first_ever else "\n"
        elif kind == "step" and self._last_kind != "banner":
            # Sub-step markers get a blank line unless a banner just opened.
            lead = "" if first_ever else "\n"
        else:
            lead = ""
        base = f"{lead}{msg_clean}"

        self._last_kind = kind
        return base

    def _format_leveled(self, record: logging.LogRecord, msg_clean: str) -> str:
        label = f"[{record.levelname}]"
        msg_clean = self._align_continuation(msg_clean, len(label) + 1)
        base = f"{label} {msg_clean}"
        self._msg_count += 1
        # Warnings and errors get a leading blank line so they stand out from
        # the surrounding routine output; DEBUG lines stay dense.
        if self._msg_count > 1 and record.levelno >= logging.WARNING:
            base = f"\n{base}"
        self._last_kind = "msg"
        return base

    def format(self, record: logging.LogRecord) -> str:
        msg_raw = record.getMessage()
        msg_clean = normalize_log_message(msg_raw)
        old_msg, old_args = record.msg, record.args
        if msg_clean != msg_raw:
            record.msg = msg_clean
            record.args = ()

        if self._compact and record.levelno == logging.INFO:
            base = self._format_info(record, msg_clean)
        else:
            base = self._format_leveled(record, msg_clean)

        record.msg, record.args = old_msg, old_args

        if not self._use_color:
            return base

        try:
            use_color = sys.stdout.isatty()
        except Exception:
            use_color = False
        if not use_color:
            return base

        # Only warnings and errors get color - everything else is plain
        if record.levelno >= logging.CRITICAL:
            return f"{self.BOLD}{self.RED}{base}{self.RESET}"
        if record.levelno >= logging.ERROR:
            return f"{self.RED}{base}{self.RESET}"
        if record.levelno >= logging.WARNING:
            return f"{self.YELLOW}{base}{self.RESET}"

        return base


# Common non-ASCII symbols mapped to plain-ASCII equivalents so log output
# stays readable in any terminal and grep-friendly in log files.
_ASCII_TRANSLATE = {
    ord("\u00b1"): "+/-",   # plus-minus
    ord("\u00d7"): "x",     # multiplication sign
    ord("\u00b0"): "deg",   # degree sign
    ord("\u00b5"): "u",     # micro sign
    ord("\u03bc"): "u",     # greek mu
    ord("\u03c3"): "sigma",
    ord("\u0394"): "Delta",
    ord("\u03b4"): "delta",
    ord("\u03b1"): "alpha",
    ord("\u03b2"): "beta",
    ord("\u03b3"): "gamma",
    ord("\u03bb"): "lambda",
    ord("\u03bd"): "nu",
    ord("\u03b8"): "theta",
    ord("\u03c6"): "phi",
    ord("\u03c7"): "chi",
    ord("\u03c0"): "pi",
    ord("\u03a3"): "Sigma",
    ord("\u2192"): "->",    # rightwards arrow
    ord("\u2190"): "<-",    # leftwards arrow
    ord("\u2194"): "<->",   # left right arrow
    ord("\u2212"): "-",     # minus sign
    ord("\u2265"): ">=",
    ord("\u2264"): "<=",
    ord("\u2260"): "!=",
    ord("\u2248"): "~=",
    ord("\u221e"): "inf",
    ord("\u221a"): "sqrt",
    ord("\u2013"): "-",     # en dash
    ord("\u2014"): "-",     # em dash
    ord("\u2026"): "...",   # ellipsis
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u201c"): '"',
    ord("\u201d"): '"',
    ord("\u2022"): "-",     # bullet
    ord("\u00b7"): ".",     # middle dot
}


def _to_ascii(text: str) -> str:
    """Replace known symbols with ASCII and drop anything else non-ASCII."""
    text = text.translate(_ASCII_TRANSLATE)
    # NFKD folds accented letters (e.g. e-acute -> e) before the strip.
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_log_message(message: str, width: int = 150) -> str:
    """
    Normalize log message formatting for readability and consistency.

    - Transliterates non-ASCII symbols to plain ASCII.
    - Converts tabs to spaces.
    - Trims trailing whitespace.
    - Collapses repeated blank lines.
    - Soft-wraps long lines to a fixed width with indentation preserved.
    """
    text = _to_ascii(str(message)).replace("\t", "    ")
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


# Lines that external binaries print on every run but carry no actionable
# information for this pipeline (single-threaded execution is enforced by
# the subprocess environment anyway).
_SUBPROCESS_LOG_NOISE = (
    "compiled using a version of the ATLAS library without support for multithreading",
)


def strip_subprocess_noise(text: str) -> str:
    """Remove known-benign noise lines from captured subprocess output."""
    kept = [
        ln for ln in str(text).splitlines()
        if ln.strip() and not any(noise in ln for noise in _SUBPROCESS_LOG_NOISE)
    ]
    return "\n".join(kept)


def clean_subprocess_log(path) -> None:
    """Drop known-benign noise lines from a captured subprocess log file."""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return
    kept = [
        ln for ln in lines
        if not any(noise in ln for noise in _SUBPROCESS_LOG_NOISE)
    ]
    if len(kept) != len(lines):
        with open(path, "w") as f:
            f.write("\n".join(kept) + "\n")


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
        formatter = ColoredLevelFormatter(use_color=use_color)
    handler.setFormatter(formatter)
    return handler


# Verbosity names accepted by ``global_verbose_level`` (YAML) and the
# ``--verbose-level`` CLI option.  Numeric values pass straight through.
VERBOSE_LEVELS = {
    "quiet": 0,
    "warning": 0,
    "error": 0,
    "normal": 1,
    "info": 1,
    "verbose": 2,
    "debug": 2,
}


def resolve_verbose_level(value) -> int:
    """
    Normalize a verbosity setting to the 0/1/2 integer convention.

    Accepts ints (clamped to 0-2) or names: ``quiet``/``warning``/``error``
    -> 0, ``normal``/``info`` -> 1, ``verbose``/``debug`` -> 2.
    Unrecognised values fall back to 1 (normal).
    """
    if isinstance(value, str):
        named = VERBOSE_LEVELS.get(value.strip().lower())
        if named is not None:
            return named
        try:
            value = int(value)
        except ValueError:
            return 1
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 1
    return 0 if v <= 0 else (2 if v >= 2 else 1)


def verbose_to_log_level(value) -> int:
    """Map a verbosity value (0/1/2 or name) to a ``logging`` level."""
    return {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }[resolve_verbose_level(value)]


def set_verbose_level(value) -> int:
    """
    Apply a verbosity setting to the root logger and its handlers.

    Use for runtime changes (e.g. a ``--verbose`` CLI flag read after the
    logging handlers are already configured).  Returns the normalized
    0/1/2 verbosity level.
    """
    level = verbose_to_log_level(value)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)
    return resolve_verbose_level(value)


@contextmanager
def quiet_root_logger(level: int = logging.WARNING):
    """
    Temporarily raise the root logger's level.

    Third-party helpers that emit per-step chatter straight through the
    root logger (e.g. spalipy's "Processing source entry 0") can be wrapped
    in this context so only warnings and errors reach the console.  The
    level is left untouched on DEBUG runs (so verbose output is preserved)
    and when it is already at or above *level*.
    """
    root = logging.getLogger()
    prev = root.level
    if prev <= logging.DEBUG or prev >= level:
        yield
        return
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(prev)


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
# Kept for backward compat: the same strings also parse via the generic
# per-character path below, but the lowercased lookup additionally accepts
# case variants such as "GRIZJHK" or "grizjhk".
_COMPOSITE_FILTER_GROUP_KEYS = {
    "grizjhk": tuple("grizJHK"),  # g,r,i,z,J,H,K (matches common refcat / Pan-STARRS+2MASS style maps)
    "ugrizjhk": tuple("ugrizJHK"),  # u,g,r,i,z,J,H,K
    "grizjhkYw": tuple("grizJHKYw"),  # g,r,i,z,J,H,K,Y,w (extended filter set)
    "ugrizjhkYw": tuple("ugrizJHKYw"),  # u,g,r,i,z,J,H,K,Y,w (extended filter set)
}

# Band letters are unique across families (u vs U are different bands), so a
# mixed key like "uRI" parses unambiguously one char at a time.
_ALL_BAND_CHARS = frozenset(SUPPORTED_PHOTOMETRIC_FILTERS)

_GROUP_KEY_SEPARATORS = re.compile(r"[,;|+\-/\s]+")


def _parse_band_token(token):
    """Parse one separator-free token into bands, or None if any char is unknown.

    Case is significant: 'r' is ugriz r and 'R' is UBVRI R, so band
    letters are always used exactly as written. The only case fallback
    fires when NO character is a valid band ("jhk" -> JHK, "bv" -> BV);
    uppercasing can then only turn non-band letters into bands, never
    swap one band for another. A token mixing real band letters with
    unknown ones fails ("bvri", "uXQ") rather than guessing at case.
    """
    if not token:
        return None
    if all(ch in _ALL_BAND_CHARS for ch in token):
        return tuple(dict.fromkeys(token))
    if not any(ch in _ALL_BAND_CHARS for ch in token):
        upper = token.upper()
        if all(ch in _ALL_BAND_CHARS for ch in upper):
            return tuple(dict.fromkeys(upper))
    return None


def parse_supported_filter_group_key(group_key):
    """
    Parse a mapping-group key into explicit supported bands.

    Accepted examples:
      - Full groups: "UBVRI", "ugriz", "JHK", "extended"
      - Any mix of band letters, families may mix: "griz", "u", "BV",
        "uRI", "rJ", "gBVw"
      - Separators as token boundaries: "u, RI", "u|JHK"
      - Lowercase words with no valid band letters get an uppercase
        rescue: "jhk" -> JHK, "bv" -> BV. Case is otherwise exact:
        "bvri" fails because r and i are real band letters - the rescue
        never swaps one band for another (no r->R or u->U guessing).
      - Composite keys: "grizJHK", "ugrizJHK" (optical + JHK for
        catalog.use_catalog maps), incl. case variants via the composite
        table ("GRIZJHK", "grizjhk")
    Rejected:
      - Keys containing unknown band letters: "xyz", "uXQ"
    """
    if group_key is None:
        return None
    key = str(group_key).strip()
    if key == "":
        return None

    # Canonical full-group keys take the fast path.
    if key in SUPPORTED_FILTER_GROUPS:
        return tuple(SUPPORTED_FILTER_GROUPS[key])

    comp = _COMPOSITE_FILTER_GROUP_KEYS.get(key.lower())
    if comp is not None:
        return comp

    bands = []
    for token in _GROUP_KEY_SEPARATORS.split(key):
        if not token:
            continue
        parsed = _parse_band_token(token)
        if parsed is None:
            # Reject the whole key: a partially valid key silently mapping
            # fewer bands than the user wrote is worse than no match.
            return None
        for b in parsed:
            if b not in bands:
                bands.append(b)
    return tuple(bands) if bands else None


def invalid_use_catalog_keys(use_catalog):
    """
    Return ``use_catalog`` mapping keys that can never match a filter band.

    A key is usable if it normalizes to a single supported band (the
    exact-match path) or parses to a supported band group. Keys containing
    unknown band letters (e.g. "uXQ") satisfy neither, so images for
    those bands silently fall back to the "default" entry.
    """
    bad = []
    if not isinstance(use_catalog, dict):
        return bad
    for key, value in use_catalog.items():
        if value is None:
            continue
        key_s = str(key).strip()
        if key_s.lower() in {"default", "*", "all"}:
            continue
        if (
            normalize_photometric_filter_name(key_s) is None
            and parse_supported_filter_group_key(key_s) is None
        ):
            bad.append(key_s)
    return bad


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
        if token in available_set:
            return token
        # Case-insensitive match returns the catalog's own casing.
        if token.lower() in {f.lower() for f in available_set}:
            for f in available_set:
                if f.lower() == token.lower():
                    return f
    
    if token in SUPPORTED_PHOTOMETRIC_FILTERS:
        return token

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
        # Non-photometric header values map to None.
        "clear": None,
        "open": None,
        "luminance": None,
        "white": None,
    }
    
    result = aliases.get(token_l)
    if result is not None:
        return result
    
    # available_filters also enables fuzzy matching, then a final exact
    # check - the exact check is what lets custom catalogs use arbitrary
    # filter names.
    if available_filters is not None:
        import difflib
        close_matches = difflib.get_close_matches(token, list(available_set), n=1, cutoff=0.8)
        if close_matches:
            return close_matches[0]

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


def normalize_target_name(name):
    """Strip transient-name prefixes (SN, AT) so catalog cache keys are stable.

    Both autophot.py (pre-fetch) and main.py (per-image) must use the same
    normalized name, otherwise the cached CSV is saved under one directory
    (e.g. ``sn2026yos/``) but looked up under another (``2026yos/``),
    causing redundant Gaia archive downloads.
    """
    if name and isinstance(name, str):
        return name.replace("SN", "").replace("AT", "")
    return name


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

    # Distance is measured from the zero regions, so masked pixels within
    # `padding` px of an existing 1 get set.
    distance = distance_transform_edt(1 - mask)

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

    fig_width_pt = width * fraction

    # pt -> inches (1 pt = 1/72.27 in)
    inches_per_pt = 1 / 72.27

    # Golden ratio for aesthetic figure height.
    golden_ratio = (5**0.5 + 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in / golden_ratio

    fig_dim = (fig_width_in, fig_height_in * aspect)

    return fig_dim


def convert_to_mjd_astropy(date_string):
    """Parse a date string (ISOT or FITS format) and return MJD."""
    try:
        # ISOT has a 'T' separator.
        t = Time(date_string, format="isot", scale="utc")
    except ValueError:
        # FITS format has no 'T'.
        t = Time(date_string, format="fits", scale="utc")

    mjd = t.mjd

    return mjd


def get_image_stats(image, sigma=3, maxiters=None):
    """Return sigma-clipped mean, median, and std for *image*."""
    mean_value, median_value, std_value = sigma_clipped_stats(
        image,
        sigma=sigma,
        maxiters=maxiters,
        # background=sigma,
        cenfunc=np.nanmedian,
        stdfunc=mad_std,
    )

    return mean_value, median_value, std_value


def calculate_bins(x, percentiles=[25, 75]):
    """Compute a Freedman-Diaconis-style bin width for *x*."""
    try:

        if not np.any(np.isfinite(x)):
            return "auto"  # default bins when there is no finite data
        q25, q75 = np.nanpercentile(x, percentiles)

        iqr = q75 - q25

        # Freedman-Diaconis bin width.
        bin_width = 2 * iqr * len(x) ** (-1 / 3)

        data_range = np.nanmax(x) - np.nanmin(x)
        bins = round(data_range / bin_width)

        return bins

    except Exception:
        return "auto"


def save_to_fits(data, output_filename):
    """Write *data* to a FITS file as float32 (preserves NaNs)."""
    try:
        # float32 preserves NaNs (chip gaps); integer dtypes cannot.
        data_to_write = data.astype(np.float32) if data.dtype.kind != 'f' else data
        hdu = fits.PrimaryHDU(data_to_write)

        hdulist = fits.HDUList([hdu])

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
    """Return the distance modulus for *redshift* using a flat LambdaCDM cosmology."""
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega)
    d = cosmo.luminosity_distance(redshift).value * 1e6
    dm = 5 * np.log10(d / 10)

    return dm


class SuppressStdout:
    """Context manager that temporarily redirects stdout to /dev/null."""

    def __enter__(self):
        import sys, os

        self._original_stdout = sys.stdout
        self._opened_devnull = True
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Only close stdout if we opened it (i.e., os.devnull)
        # Don't close permanently if it was already open
        if self._opened_devnull:
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
        Background-subtracted aperture flux in **per-second** units (e-/s),
        matching ``Aperture.measure``'s ``flux_AP``.
    npix : int or float
        Geometric aperture area in pixels (same ``area`` as ``Aperture`` uses
        for background subtraction), *not* the integer pixel count of a mask.
    sigma : float
        Per-pixel background RMS in **per-second** units (e-/s per pixel),
        matching ``noiseSky`` from ``Aperture.measure``.
    noise : float, optional
        Mean background offset if not background-subtracted (default 0).

    Returns
    -------
    beta : float or np.ndarray
        Detection confidence in [0, 1]. Higher values indicate a more
        confident detection above the threshold.
    """
    # abs() keeps detection confidence symmetric: a -5 sigma dip on a
    # difference image is as significant as a +5 sigma peak (consistent
    # with the beta_psf fix, BUG-5).  The sign is checked separately in
    # the detection logic (main.py is_detection), so this only affects
    # the confidence value, not the detection decision.
    source_flux = np.abs(flux_aperture - noise * npix)

    sigma_aperture = sigma * np.sqrt(npix)
    # Guard against division by zero.
    sigma_aperture = np.maximum(sigma_aperture, np.finfo(float).tiny)

    # z-score: how far the measured flux is above the n-sigma threshold
    z = ((n * sigma_aperture) - source_flux) / (np.sqrt(2) * sigma_aperture)

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
    # |flux| keeps detection confidence symmetric (a -5 sigma dip is as
    # significant as a +5 sigma peak); needed for inverted fits where
    # flux_PSF is negative.
    flux_abs = np.abs(np.asarray(flux_psf, dtype=float))
    err = np.maximum(np.asarray(flux_psf_err, dtype=float), np.finfo(float).tiny)
    # Threshold flux = n * (1-sigma error); z-score for "flux above threshold".
    # z may overflow to +/-inf for extreme flux/error values; erf saturates
    # there, so the result is still correct and the fp warning is noise.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = (n * err - flux_abs) / (np.sqrt(2) * err)
    beta = np.clip(0.5 * (1 - erf(z)), 0.0, 1.0)
    return beta


def log_step(msg: str) -> str:
    """
    Compact one-line marker for routine pipeline steps.

    Uses minimal decoration to reduce visual clutter while maintaining
    clear section markers. Prefer for frequently called stages.
    """
    m = str(msg).strip()
    if not m:
        return ""
    return f"[{m}]"


def _rule_line(title: str, char: str, width: int = 70) -> str:
    """Centered title embedded in a full-width rule, e.g. ``=== X ===``."""
    t = str(title).strip()
    inner = width
    if len(t) > inner - 2:
        t = t[: inner - 5] + "..."
    side = (inner - len(t) - 2) // 2
    return f"{char * side} {t} {char * (inner - side - len(t) - 2)}"


def border_msg(msg: str, body: str = "=", corner: str = "+",
               metadata: str | None = None, width: int = 70, use_ansi: bool | None = None) -> str:
    """
    Section banner for major log sections: ``=== Title ===``.

    Parameters
    ----------
    msg : str
        Main section title (centered in a full-width '=' rule)
    body : str
        Rule character (default '=')
    corner : str
        Unused; kept for backward compatibility with older call sites.
    metadata : str | None
        Optional second line, rendered as ``--- metadata ---``
    width : int
        Total banner width in characters
    use_ansi : bool | None
        Unused; kept for backward compatibility.

    Example:
        logging.info(border_msg("Template Preparation", metadata="align=SWarp catalog=Gaia"))
    """
    text = str(msg).strip()
    if not text:
        return ""
    rule = str(body or "=")[0]
    lines = [_rule_line(text, rule, width)]
    if metadata:
        lines.append(_rule_line(str(metadata).strip(), "-", width))
    return "\n".join(lines)


def ascii_kv(title: str, pairs, width: int = 70, framed: bool = False) -> str:
    """
    Key/value block with dotted leaders.

    pairs : iterable of (label, value) tuples; values are pre-formatted
    strings.  With ``framed=False`` the block is a lightweight sub-section:

        * Title
          ------------------  ----------------
          label ........... value

    With ``framed=True`` it is wrapped in '=' rules for major blocks:

        ============================ Title ============================
          label ........... value
        ================================================================
    """
    rows = [(str(k), str(v)) for k, v in pairs]
    if not rows:
        return ""
    label_w = max(len(k) for k, _ in rows)
    label_w = max(label_w, 8)
    val_x = label_w + 8  # '  ' + label + ' ... ' gap
    val_w = width - val_x - 2
    lines = []
    if framed:
        lines.append(_rule_line(str(title), "=", width))
    else:
        lines.append(f"* {str(title).strip()}")
        lines.append("  " + "-" * label_w + "  " + "-" * max(8, min(val_w, 24)))
    for k, v in rows:
        dots = "." * max(3, val_x - 2 - len(k) - 4)
        line = f"  {k} {dots} {v}"
        lines.append(line[:width] if len(line) > width else line)
    if framed:
        lines.append("=" * width)
    return "\n".join(lines)


def ascii_table(title: str, headers, rows, width: int = 70) -> str:
    """
    Column table framed in '-' rules:

        --------------------------- Title ----------------------------
           Method     ZP      err      N
           ------  -------  ------  ----
           AP      24.547   0.003   159
        -----------------------------------------------------------------
    """
    headers = [str(h) for h in headers]
    rows = [[str(c) for c in r] for r in rows]
    if not headers or not rows:
        return ""
    ncols = len(headers)
    col_w = [len(h) for h in headers]
    for r in rows:
        for i in range(min(ncols, len(r))):
            col_w[i] = max(col_w[i], len(r[i]))
    body_w = 4 + sum(col_w) + 2 * ncols  # leading indent + columns + gaps
    rule_w = min(max(width, body_w), max(width, 40))
    lines = []
    if title:
        lines.append(_rule_line(str(title), "-", rule_w))
    else:
        lines.append("-" * rule_w)
    lines.append(
        "   " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)).rstrip()
    )
    lines.append("   " + "  ".join("-" * col_w[i] for i in range(ncols)))
    for r in rows:
        cells = [r[i] if i < len(r) else "" for i in range(ncols)]
        lines.append(
            "   " + "  ".join(c.ljust(col_w[i]) for i, c in enumerate(cells)).rstrip()
        )
    lines.append("-" * rule_w)
    return "\n".join(lines)


def metrics_table(metrics: dict[str, tuple], title: str | None = None, width: int = 70) -> str:
    """
    Format a compact two-column metrics table.

    Parameters
    ----------
    metrics : dict[str, tuple]
        Dictionary of label -> (value, unit_or_note)
        Example: {"Seeing FWHM": (3.5, "px"), "Zeropoint": (25.34, "mag")}
    title : str | None
        Optional title line printed above the table
    width : int
        Total width of the formatted block

    Returns
    -------
    str
        Formatted multi-line string ready for logging.info()
    """
    if not metrics:
        return ""
    pairs = [
        (label, f"{value} {unit}".strip())
        for label, (value, unit) in metrics.items()
    ]
    return ascii_kv(title or "Metrics", pairs, width=width, framed=True)


def compact_status(filename: str, results: dict) -> str:
    """
    One-line status summary for completed image processing.

    Parameters
    ----------
    filename : str
        Base filename (will be truncated if too long)
    results : dict
        Must contain keys: 'mag', 'mag_err', 'snr', 'detected' (bool),
        optionally 'zp', 'n_cal', 'template'

    Example output:
        OK SN2024pba_ZTF_r.fits  r=18.34+/-0.03  S/N=12.5  OK Detection
    """
    base = os.path.basename(filename)
    if len(base) > 30:
        base = "..." + base[-27:]

    mag = results.get('mag', float('nan'))
    mag_err = results.get('mag_err', float('nan'))
    snr = results.get('snr', float('nan'))
    detected = results.get('detected', False)
    zp = results.get('zp')
    n_cal = results.get('n_cal')

    parts = [f"{'OK' if detected else 'o'} {base:>30}"]

    if np.isfinite(mag) and np.isfinite(mag_err):
        parts.append(f"m={mag:.2f}+/-{mag_err:.2f}")

    if np.isfinite(snr):
        parts.append(f"S/N={snr:.1f}")

    if zp is not None and np.isfinite(zp):
        parts.append(f"zp={zp:.2f}")

    if n_cal is not None:
        parts.append(f"cal={n_cal}")

    parts.append("OK Detection" if detected else "o Limit")

    return "  ".join(parts)


# Telescope/instrument config: images must have FITS header keywords TELESCOP and INSTRUME.
# telescope.yml in the working dir lists all telescopes; each entry follows the structure below.
# Top level: TELESCOP value (e.g. "1m0-01", "ESO-NTT", "Palomar 48-inch"). Under each:
#   INSTRUME: instrument name -> instrument config dict.
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
        logger.debug("telescope.yml loaded from: %s", " (merged) ".join(loaded_sources))
    else:
        logger.info(
            "telescope.yml: using built-in defaults only (no file found at %r)",
            user_path,
        )
    return out


def download_zogy(wdir, update=False, repo_url="https://github.com/pmvreeswijk/ZOGY"):
    """Download the ZOGY package from GitHub and extract it next to telescope.yml.

    The repository is downloaded as a zip archive and extracted into
    ``<wdir>/ZOGY/``.  If the directory already exists and *update* is False,
    the existing copy is used.  When *update* is True, the directory is
    removed and re-downloaded.

    After extraction, ``<wdir>`` is prepended to ``sys.path`` so that
    ``import zogy`` works from the local copy.

    Parameters
    ----------
    wdir : str
        Working directory where ``telescope.yml`` lives.  The ZOGY repo
        will be extracted into ``<wdir>/ZOGY/``.
    update : bool
        If True, overwrite an existing ``ZOGY/`` directory.
    repo_url : str
        GitHub repository URL (default: pmvreeswijk/ZOGY).

    Returns
    -------
    str or None
        Path to the extracted ``zogy.py`` file, or None on failure.
    """
    import io
    import zipfile
    import tempfile
    import shutil

    logger = logging.getLogger(__name__)

    zogy_dir = os.path.join(wdir, "ZOGY")
    zogy_py = os.path.join(zogy_dir, "zogy.py")

    if os.path.isfile(zogy_py) and not update:
        logger.info("ZOGY already present at %s (use update=True to re-download).", zogy_dir)
        if wdir not in sys.path:
            sys.path.insert(0, wdir)
        if zogy_dir not in sys.path:
            sys.path.insert(0, zogy_dir)
        return zogy_py

    if os.path.isdir(zogy_dir) and update:
        logger.info("Removing existing ZOGY directory at %s for update.", zogy_dir)
        shutil.rmtree(zogy_dir, ignore_errors=True)

    archive_url = f"{repo_url}/archive/refs/heads/main.zip"
    logger.info("Downloading ZOGY from %s ...", archive_url)

    try:
        import requests as _requests

        resp = _requests.get(archive_url, stream=True, timeout=120)
        resp.raise_for_status()
        zip_bytes = io.BytesIO(resp.content)
    except Exception as req_err:
        logger.warning("requests-based download failed (%s); trying urllib fallback.", req_err)
        try:
            from urllib.request import urlretrieve

            tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            tmp_zip.close()
            urlretrieve(archive_url, tmp_zip.name)
            with open(tmp_zip.name, "rb") as f:
                zip_bytes = io.BytesIO(f.read())
            os.unlink(tmp_zip.name)
        except Exception as url_err:
            logger.error("Failed to download ZOGY: %s", url_err)
            return None

    try:
        with zipfile.ZipFile(zip_bytes) as zf:
            zf.extractall(zogy_dir)
    except Exception as extract_err:
        logger.error("Failed to extract ZOGY archive: %s", extract_err)
        return None

    # GitHub zip extracts into a subfolder like "ZOGY-main/"
    extracted_items = os.listdir(zogy_dir)
    if len(extracted_items) == 1 and extracted_items[0].startswith("ZOGY-"):
        nested = os.path.join(zogy_dir, extracted_items[0])
        # Move contents up one level
        for item in os.listdir(nested):
            shutil.move(os.path.join(nested, item), os.path.join(zogy_dir, item))
        os.rmdir(nested)

    if not os.path.isfile(zogy_py):
        logger.error(
            "ZOGY download completed but zogy.py not found at %s. "
            "Check repo structure.", zogy_py
        )
        return None

    if wdir not in sys.path:
        sys.path.insert(0, wdir)
    if zogy_dir not in sys.path:
        sys.path.insert(0, zogy_dir)

    logger.info("ZOGY downloaded and extracted to %s", zogy_dir)
    return zogy_py


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

    # MAD-based background / noise estimate (resistant to outliers).
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

    try:
        cat = SourceCatalog(cut, segm)
        xcen = np.asarray(cat.x_centroid if hasattr(cat, 'x_centroid') else cat.xcentroid)
        ycen = np.asarray(cat.y_centroid if hasattr(cat, 'y_centroid') else cat.ycentroid)
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
    """Load and update YAML configuration files for AutoPHOT."""

    def __init__(self, filepath=None, dict_name=None, wdir=None):
        """Store path, optional sub-key, and working directory for later load/update."""
        self.filepath = filepath
        self.dict_name = dict_name
        self.wdir = wdir

    def load(self):
        """Load YAML from *filepath* and return the full dict or a named sub-key."""
        if self.wdir is not None:
            file_path = os.path.join(self.wdir, self.filepath)
        else:
            file_path = self.filepath

        with open(file_path, "r") as stream:
            var = yaml.safe_load(stream)

        if self.dict_name is not None:
            data = var[self.dict_name]
        else:
            data = var

        return data

    def update(self, tele, inst_key, inst, key, new_val):
        """Update a nested key in the YAML file under ``tele[inst_key][inst]``."""
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
    """Read a FITS header, tolerating missing extensions.

    Looks for the ``TELESCOP`` keyword across HDUs and returns the first
    matching header, merging with the primary header if needed.

    Results are cached by (path, mtime, size); call ``invalidate_fits_cache``
    after external tools modify the file.
    """
    key = _fits_cache_key(fpath)
    if key is not None:
        with _FITS_CACHE_LOCK:
            cached = _FITS_HEADER_CACHE.get(key)
            if cached is not None:
                return cached.copy()

    from astropy.io.fits import getheader
    from astropy.io import fits
    try:
        # 'ignore_missing_end' tolerates truncated files.
        with fits.open(fpath, ignore_missing_end=True) as hdul:
            hdul.verify("silentfix+ignore")

            # FITS keywords are typically uppercase; check case-insensitively for TELESCOP
            def has_telescop(header):
                return any(k.upper() == "TELESCOP" for k in header.keys())

            if has_telescop(hdul[0].header):
                headinfo = hdul[0].header.copy()
            else:
                # TELESCOP may live in an extension HDU (e.g. the image ext).
                for i in range(1, len(hdul)):
                    if has_telescop(hdul[i].header):
                        headinfo = hdul[i].header.copy()
                        break
                else:
                    headinfo = hdul[0].header.copy()
    except KeyError as e:
        # Missing or mis-cased header keys (e.g., 'Telescop').
        raise Exception(f"KeyError: The required header keyword was not found: {e}")

    except Exception as e:
        raise Exception(f"An error occurred while reading the FITS file: {e}")

    # A list header means multiple HDUs; combine them.
    if isinstance(headinfo, list):
        combined_header = headinfo[0].header

        for ext in headinfo[1:]:
            combined_header.update(ext.header)

        headinfo = combined_header

    if key is not None:
        with _FITS_CACHE_LOCK:
            _FITS_HEADER_CACHE[key] = headinfo
            _cache_evict(_FITS_HEADER_CACHE)

    return headinfo


def get_image_and_header(fpath):
    """
    Load FITS image and header with a single file open. Same semantics as
    get_image(fpath) and get_header(fpath), but avoids opening the file twice.

    Results are cached by (path, mtime, size); call ``invalidate_fits_cache``
    after external tools modify the file.

    :param fpath: Path to the FITS file.
    :return: (image, header) where image is a 2D numpy array copy and header is a copy.
    """
    key = _fits_cache_key(fpath)
    if key is not None:
        with _FITS_CACHE_LOCK:
            cached = _FITS_IMAGE_CACHE.get(key)
            if cached is not None:
                img, hdr = cached
                return img.copy(), hdr.copy()

    import os
    from astropy.io import fits

    try:
        with fits.open(fpath, ignore_missing_end=True) as hdul:
            hdul.verify("silentfix+ignore")
            # Pick the best image HDU across FITS layout conventions.
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
                    logger.info("Using 'sci' extension (HDU %s) with shape %s", best_hdu_idx, image.shape)
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
                    logger.debug("Selected HDU %s (score=%.1f) with shape %s", best_idx, best_score, image.shape)
            
            # Strategy 3: Last resort - try primary HDU
            if image is None and len(hdul) > 0:
                primary_data = hdul[0].data
                if primary_data is not None:
                    try:
                        test_image = np.asarray(primary_data)
                        if hasattr(test_image, 'shape'):
                            logger.debug("Using primary HDU as fallback with shape %s", getattr(test_image, 'shape', 'no shape'))
                            image = test_image.copy()
                            # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
                            if image.dtype.kind != 'f':
                                image = image.astype(np.float32)
                            best_hdu_idx = 0
                    except Exception as e:
                        logger.debug("Error with primary HDU: %s", e)
            
            if image is None:
                # Dump HDU structure to help debug exotic FITS layouts.
                logger.debug("HDU structure analysis:")
                for i, hdu in enumerate(hdul):
                    data_info = f"shape={getattr(hdu.data, 'shape', 'None')}" if hdu.data is not None else "None"
                    logger.debug("  HDU %s: %s, name='%s', data=%s", i, hdu.__class__.__name__, hdu.name, data_info)
                raise Exception(f"No valid 2D+ image data found in FITS file: {os.path.basename(fpath)}")
            
            # Cube data (e.g. data+error planes) collapses to its first 2D slice.
            if hasattr(image, 'shape') and len(image.shape) > 2:
                base = os.path.basename(fpath)
                logger.warning("%s has %sD data, taking first 2D slice", base, len(image.shape))
                original_shape = image.shape
                while len(image.shape) > 2:
                    image = image[0]
                image = image.copy()
                logger.info("  Reshaped from %s to %s", original_shape, image.shape)
            elif not hasattr(image, 'shape') or len(image.shape) < 2:
                base = os.path.basename(fpath)
                raise Exception(f"Warning: {base} is not a 2D array (found {getattr(image, 'shape', 'no shape')} data).")

            # Use the header from the same HDU as the image data so WCS
            # keywords stay paired with the pixels they describe; merge
            # TELESCOP metadata in from another HDU if needed.
            def has_telescop(header):
                return any(k.upper() == "TELESCOP" for k in header.keys())

            headinfo = hdul[best_hdu_idx].header.copy()

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
            
            if key is not None:
                with _FITS_CACHE_LOCK:
                    _FITS_IMAGE_CACHE[key] = (image, headinfo)
                    _cache_evict(_FITS_IMAGE_CACHE)
            return image, headinfo
    except KeyError as e:
        raise Exception(f"KeyError: The required header keyword was not found: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the FITS file: {e}")


def get_image(fpath):
    """Load a 2-D image array from a FITS file.

    Tries the ``sci`` extension first, then falls back to the primary HDU.
    Raises if the data is not 2-D.
    """
    import os
    from astropy.io import fits
    try:
        image = fits.getdata(fpath, extname="sci")

    except Exception:
        image = fits.getdata(fpath)

    # float32 preserves NaNs (chip gaps); integer dtypes cannot.
    if image.dtype.kind != 'f':
        image = image.astype(np.float32)

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

    concatenated_data = []

    from fnmatch import fnmatch

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Support wildcard patterns, e.g. loc_file="Output_*.csv"
            if ("*" in loc_file) and fnmatch(file, loc_file):
                file_path = os.path.join(root, file)

                # dtype=str preserves blank cells so they can be mapped
                # to NaN below rather than parsed as 0 or ''.
                df = pd.read_csv(
                    file_path,
                    keep_default_na=True,
                    na_values=["", " ", "NA", "N/A", "NaN", "null"],
                    dtype=str,
                )

                df = df.map(
                    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
                )

                concatenated_data.append(df)
            elif file == loc_file:
                file_path = os.path.join(root, file)

                df = pd.read_csv(
                    file_path,
                    keep_default_na=True,
                    na_values=["", " ", "NA", "N/A", "NaN", "null"],
                    dtype=str,
                )

                df = df.map(
                    lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
                )

                concatenated_data.append(df)

    if not concatenated_data:
        logging.getLogger(__name__).info(
            "No CSV files matched the requested pattern; nothing to concatenate."
        )
        return None

    concatenated_data = pd.concat(concatenated_data, ignore_index=True)

    # If concatenation produced duplicate column names (common when mixing legacy
    # wide-format outputs across versions), keep the first occurrence.
    if concatenated_data.columns.duplicated().any():
        concatenated_data = concatenated_data.loc[
            :, ~concatenated_data.columns.duplicated()
        ].copy()

    concatenated_data.to_csv(
        output_filename, index=False, na_rep="NaN"
    )  # write NaN explicitly

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
    """Return the simple signal-to-noise ratio ``maxPixel / noiseBkg``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        snr_value = maxPixel / noiseBkg

    return snr_value


def snr_err(snr_value):
    """
     Error associated with signal to noise ratio (S/N). Equation taken from `here <https://www.ucolick.org/~bolte/AY257/s_n.png>`_. The error on the instrumental magnitude of a source is:


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

    with np.errstate(divide="ignore", invalid="ignore"):
        # Standard linear approximation: dm = (2.5/ln(10)) / SNR = 1.0857/SNR.
        # This is the upper bound of the exact 2.5*log10(1 + 1/SNR) formula and
        # is the convention used by aperture.py and zeropoint._compute_delta_mag.
        snr_err_value = (2.5 / np.log(10.0)) / snr_value

    return snr_err_value


def quadrature_add(values):
    """Return the quadrature sum of *values* (sqrt of sum of squares).

    NaN terms are **skipped**, not propagated.  This prevents a single
    missing error term (e.g. NaN zeropoint_error) from silently zeroing
    out the entire calibrated magnitude error budget.  If *all* terms
    are NaN, returns NaN.
    """
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return np.nan
    return float(np.sqrt(sum(v ** 2 for v in finite)))


# ---------------------------------------------------------------------------
# Sampling regime classification (Howell 1989 sampling parameter)
# ---------------------------------------------------------------------------
# FWHM in pixels determines how well the PSF is sampled:
#   < 2 px : undersampled  - PSF core spans < 2 pixels, flux concentrated
#            in 1-2 pixels.  Need supersampled PSF, broader detection cuts,
#            larger fixed apertures.
#   2-3 px : critically sampled - PSF core barely resolved.  Moffat PSF
#            with free beta, moderate apertures.
#   3-5 px : well-sampled - standard PSF fitting, curve-of-growth apertures.
#   > 5 px : oversampled - SNR inefficient, warn user, larger apertures.
#
# These thresholds are configurable via input_yaml["photometry"]:
#   undersampled_fwhm_threshold (default 2.5) - already used by psf.py
#   critical_fwhm_threshold (default 3.0)
#   oversampled_fwhm_threshold (default 5.0)

SAMPLING_REGIME_UNDERSAMPLED = "undersampled"
SAMPLING_REGIME_CRITICAL = "critical"
SAMPLING_REGIME_WELL_SAMPLED = "well_sampled"
SAMPLING_REGIME_OVERSAMPLED = "oversampled"


def classify_sampling_regime(
    fwhm_px: float,
    input_yaml: dict = None,
) -> str:
    """Classify the PSF sampling regime from FWHM in pixels.

    Parameters
    ----------
    fwhm_px : float
        FWHM in pixels.
    input_yaml : dict, optional
        Configuration dict with optional thresholds under
        ``photometry``: ``undersampled_fwhm_threshold`` (default 2.5),
        ``critical_fwhm_threshold`` (default 3.0),
        ``oversampled_fwhm_threshold`` (default 5.0).

    Returns
    -------
    str
        One of ``SAMPLING_REGIME_UNDERSAMPLED``,
        ``SAMPLING_REGIME_CRITICAL``, ``SAMPLING_REGIME_WELL_SAMPLED``,
        ``SAMPLING_REGIME_OVERSAMPLED``.
    """
    if not np.isfinite(fwhm_px) or fwhm_px <= 0:
        return SAMPLING_REGIME_WELL_SAMPLED  # safe default

    phot_cfg = (input_yaml or {}).get("photometry", {}) or {}
    under_thr = float(phot_cfg.get("undersampled_fwhm_threshold", 2.5))
    crit_thr = float(phot_cfg.get("critical_fwhm_threshold", 3.0))
    over_thr = float(phot_cfg.get("oversampled_fwhm_threshold", 5.0))

    if fwhm_px <= under_thr:
        return SAMPLING_REGIME_UNDERSAMPLED
    elif fwhm_px <= crit_thr:
        return SAMPLING_REGIME_CRITICAL
    elif fwhm_px <= over_thr:
        return SAMPLING_REGIME_WELL_SAMPLED
    else:
        return SAMPLING_REGIME_OVERSAMPLED


def adaptive_aperture_radius(
    fwhm_px: float,
    input_yaml: dict = None,
) -> float:
    """Compute FWHM-regime-aware aperture radius in pixels.

    Undersampled data needs larger fixed apertures (flux concentrated
    in 1-2 pixels, aperture corrections are large and unstable for
    small radii).  Well-sampled data uses the standard ~1.5*FWHM.

    Parameters
    ----------
    fwhm_px : float
        FWHM in pixels.
    input_yaml : dict, optional
        Configuration dict (for threshold lookup).

    Returns
    -------
    float
        Aperture radius in pixels.
    """
    if not np.isfinite(fwhm_px) or fwhm_px <= 0:
        fwhm_px = 3.0  # safe default

    regime = classify_sampling_regime(fwhm_px, input_yaml)
    if regime == SAMPLING_REGIME_UNDERSAMPLED:
        # Minimum 4 px; 2.5*FWHM captures most flux for undersampled PSFs
        return max(4.0, 2.5 * fwhm_px)
    elif regime == SAMPLING_REGIME_CRITICAL:
        return 2.0 * fwhm_px
    elif regime == SAMPLING_REGIME_OVERSAMPLED:
        # Larger apertures proportional to FWHM
        return 1.8 * fwhm_px
    else:
        # Well-sampled: standard
        return 1.5 * fwhm_px


def adaptive_annulus_radii(
    fwhm_px: float,
    input_yaml: dict = None,
) -> tuple:
    """Compute FWHM-regime-aware sky annulus (inner, outer) radii in pixels.

    Returns
    -------
    tuple(float, float)
        (inner_radius, outer_radius) in pixels.
    """
    if not np.isfinite(fwhm_px) or fwhm_px <= 0:
        fwhm_px = 3.0

    regime = classify_sampling_regime(fwhm_px, input_yaml)
    if regime == SAMPLING_REGIME_UNDERSAMPLED:
        return 6.0, 9.0  # fixed, wide annulus
    elif regime == SAMPLING_REGIME_CRITICAL:
        return 3.0 * fwhm_px, 4.0 * fwhm_px
    elif regime == SAMPLING_REGIME_OVERSAMPLED:
        return 2.5 * fwhm_px, 4.0 * fwhm_px
    else:
        return 2.5 * fwhm_px, 4.0 * fwhm_px


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

    :param flux: Brightness in *per-second* units, matching ``Aperture.flux_AP`` and
        ``psf`` ``flux_PSF`` (e-/s when photometry uses ``image * gain`` in
        :mod:`aperture`).
    :type flux: float or array
    :return: Instrumental magnitude; NaN where flux <= 0
    :rtype: float or array
    """
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
    # Add divisibility check to prevent ValueError
    if arr.shape[0] % new_shape[0] != 0 or arr.shape[1] % new_shape[1] != 0:
        raise ValueError(
            f"Array dimensions {arr.shape} are not evenly divisible by new_shape {new_shape}"
        )
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
    wcs_stem_underscore = ["A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_"]

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


def copy_wcs_from_header(src_header, dst_header):
    """
    Copy all WCS keywords from src_header into dst_header.

    This is the inverse of remove_wcs_from_header.  It preserves the
    exact WCS representation (CD matrix, SIP, PV, etc.) from the source
    header, unlike WCS.to_header() which may drop the CD matrix and
    write CDELT=1.0.

    Parameters
    ----------
    src_header : fits.Header
        Source FITS header containing the WCS to copy.
    dst_header : fits.Header
        Destination FITS header (modified in place).

    Returns
    -------
    fits.Header
        The modified dst_header.
    """
    wcs_prefixes = [
        "CRPIX", "CRVAL", "CTYPE", "CD", "PC", "CDELT", "CROTA",
        "PV", "LONPOLE", "LATPOLE", "EQUINOX", "WCSNAME", "CUNIT",
        "WCSAXES", "PROJP", "LTV", "LTM", "RADECSYS", "RADESYS",
        "RADYSYS", "LONGPOLE", "TNX", "SIP_",
    ]
    wcs_stem_underscore = ["A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_"]

    for key in src_header.keys():
        if key in ("NAXIS", "NAXIS1", "NAXIS2", "COMMENT", "HISTORY"):
            continue
        is_wcs = False
        for prefix in wcs_prefixes:
            if key.startswith(prefix):
                is_wcs = True
                break
        if not is_wcs:
            stem = key.split("_")[0] + "_" if "_" in key else ""
            if stem in wcs_stem_underscore and key.startswith(stem.rstrip("_")):
                is_wcs = True
        if is_wcs:
            dst_header[key] = src_header[key]

    return dst_header


def update_header_from_wcs(header, wcs_obj):
    """
    Update a FITS header with WCS from a WCS object (e.g. Cutout2D.wcs).

    This is the correct way to update a header from a WCS object, unlike
    ``header.update(wcs.to_header(relax=True))`` which can:
    - Drop the CD matrix and write CDELT=1.0
    - Leave stale WCS keywords from the old header

    Steps:
    1. Remove all old WCS keywords from the header.
    2. Convert WCS to header with relax=True (preserves SIP/PV keywords).
    3. Restore CD matrix if astropy dropped it.
    4. Update the header in place.

    Parameters
    ----------
    header : fits.Header
        Destination FITS header (modified in place).
    wcs_obj : astropy.wcs.WCS
        WCS object to extract keywords from.

    Returns
    -------
    fits.Header
        The modified header.
    """
    header = remove_wcs_from_header(header)
    wcs_hdr = wcs_obj.to_header(relax=True)
    header.update(wcs_hdr)
    # Preserve CD matrix from the WCS if to_header dropped it
    if not any(k.startswith('CD') for k in wcs_hdr):
        cd = getattr(wcs_obj.wcs, 'cd', None)
        if cd is not None and wcs_obj.wcs.has_cd():
            header['CD1_1'] = cd[0, 0]
            header['CD1_2'] = cd[0, 1]
            header['CD2_1'] = cd[1, 0]
            header['CD2_2'] = cd[1, 1]
            for k in ['CDELT1', 'CDELT2']:
                header.pop(k, None)
    return header


def nan_crop(data, header, cx, cy, ny, nx):
    """
    Crop an image to (ny, nx) centred on (cx, cy), padding with NaN if the
    region extends beyond the array.  Only CRPIX1/CRPIX2 are updated in the
    header - all other WCS keywords (CD, SIP, PV, CTYPE, etc.) are left
    untouched, avoiding the distortion-dropping problems that Cutout2D's
    WCS round-trip can introduce.

    Parameters
    ----------
    data : np.ndarray
        2-D image array.
    header : fits.Header
        FITS header (modified in place).
    cx, cy : float
        Desired centre in 0-based pixel coordinates.
    ny, nx : int
        Desired output shape (rows, cols).

    Returns
    -------
    cropped : np.ndarray  shape (ny, nx)
    header : fits.Header   (same object, CRPIX updated)
    """
    src_ny, src_nx = data.shape

    # Output array, filled with NaN
    out = np.full((ny, nx), np.nan, dtype=data.dtype)

    # Source slice that maps into the output.
    # Use (nx-1)/2 (0-based pixel centre) not nx/2 (geometric centre between
    # pixels for even nx).  floor(x + 0.5) gives deterministic half-up
    # rounding; np.round uses round-half-to-even, which can choose different
    # integer slice origins for half-integer offsets depending on parity and
    # introduce ~0.5px shifts between science/template cutouts.
    src_y0 = int(np.floor(cy - (ny - 1) / 2.0 + 0.5))
    src_y1 = src_y0 + ny
    src_x0 = int(np.floor(cx - (nx - 1) / 2.0 + 0.5))
    src_x1 = src_x0 + nx

    # Clamp to source bounds
    sy0 = max(0, src_y0)
    sy1 = min(src_ny, src_y1)
    sx0 = max(0, src_x0)
    sx1 = min(src_nx, src_x1)

    # Corresponding destination indices
    dy0 = sy0 - src_y0
    dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - src_x0
    dx1 = dx0 + (sx1 - sx0)

    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy1, dx0:dx1] = data[sy0:sy1, sx0:sx1]

    # Update only CRPIX - the pixel that was at (cx, cy) in the source
    # should be at (nx/2, ny/2) in the output (0-based -> FITS 1-based).
    crpix1 = header.get("CRPIX1", None)
    crpix2 = header.get("CRPIX2", None)
    if crpix1 is not None:
        header["CRPIX1"] = float(crpix1) - src_x0
    if crpix2 is not None:
        header["CRPIX2"] = float(crpix2) - src_y0

    header["NAXIS1"] = nx
    header["NAXIS2"] = ny

    return out, header


def convert_ra_dec_to_hms_dms(ra_deg, dec_deg):
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
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

    image = np.array(image)
    rows, cols = image.shape

    # Uniform (constant-value) rows/columns mark padded edges.
    uniform_rows = [i for i in range(rows) if np.all(image[i] == image[i, 0])]
    uniform_cols = [j for j in range(cols) if np.all(image[:, j] == image[0, j])]

    if uniform_rows:
        row_distances = [abs(x - row) for row in uniform_rows]
        min_row_distance = min(row_distances)
    else:
        min_row_distance = float("inf")  # no uniform rows found

    if uniform_cols:
        col_distances = [abs(y - col) for col in uniform_cols]
        min_col_distance = min(col_distances)
    else:
        min_col_distance = float("inf")  # no uniform columns found

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

    if "x_pix" not in dataframe.columns or "y_pix" not in dataframe.columns:
        raise ValueError("DataFrame must have 'x_pix' and 'y_pix' columns.")

    x_pix_column = dataframe["x_pix"].values
    y_pix_column = dataframe["y_pix"].values

    output_dataframe = pd.DataFrame({"x_pix": x_pix_column, "y_pix": y_pix_column})

    with open(output_file, "w") as file:
        file.write("x y\n")
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

    data = data[~np.isnan(data)]
    if bins == "auto":
        bins = calculate_bins(data)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    normalization_factor = np.nanmax(hist)

    normalized_hist = hist / normalization_factor

    return normalized_hist, bin_edges


def dict_to_string_with_hashtag(dictionary, float_format="%.6f"):
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
    # Sanitize header to remove non-ASCII characters
    sanitized_header = fits.Header()
    for key, value in header.items():
        try:
            if isinstance(value, str):
                # Remove non-ASCII characters from string values
                sanitized_value = ''.join(char if ord(char) < 128 else '?' for char in str(value))
                sanitized_header[key] = sanitized_value
            elif isinstance(value, (list, tuple)):
                # Handle list values (e.g., HISTORY comments)
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        sanitized_list.append(''.join(char if ord(char) < 128 else '?' for char in item))
                    else:
                        sanitized_list.append(item)
                sanitized_header[key] = sanitized_list
            else:
                sanitized_header[key] = value
        except (UnicodeEncodeError, ValueError):
            # Skip problematic header cards
            continue

    # Use float32 to preserve NaNs (chip gaps) - integer dtypes cannot represent NaN
    image_to_write = image.astype(np.float32) if image.dtype.kind != 'f' else image

    # Try writing with sanitized header, fall back to more lenient mode if it fails
    try:
        fits.writeto(fpath, image_to_write, sanitized_header, overwrite=overwrite, output_verify=output_verify)
    except (UnicodeEncodeError, ValueError) as e:
        # If sanitization failed, try with even more lenient verification
        fits.writeto(fpath, image_to_write, sanitized_header, overwrite=overwrite, output_verify="fix")

    # Invalidate any cached entry for this path so subsequent reads see the new file
    invalidate_fits_cache(fpath)
