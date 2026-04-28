#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized and human-readable version of the automated photometry pipeline.
Features:
- Clear, descriptive comments and docstrings.
- Progress print statements for user feedback.
- Modular, reusable functions and classes.
- Consistent logging and error handling.
- British English spelling and style.
"""

# Force BLAS/OpenMP to 1 thread before any scientific imports (avoids exhausting
# process/thread limits on HPC when using multiprocessing; OpenBLAS defaults to 128).
import os

for _env in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env] = "1"

import sys
import time
import copy
import gc
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import get_close_matches

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import re

# Optional heavy dependencies.
# These are only required for parts of the pipeline (e.g. SIMBAD/TNS helpers).
# Keeping them optional allows lightweight utilities (like list_parameters) to be
# imported without requiring the full astronomy stack.
try:
    from astroquery.simbad import Simbad  # type: ignore
except Exception:  # pragma: no cover
    Simbad = None  # type: ignore

try:
    from astropy.coordinates import SkyCoord  # type: ignore
    from astropy import units as u  # type: ignore
except Exception:  # pragma: no cover
    SkyCoord = None  # type: ignore
    u = None  # type: ignore

# Module-level flag to optionally silence user-facing logging
QUIET_MODE = False

# Cache the parsed default schema so config validation is fast.
_DEFAULT_INPUT_SCHEMA_CACHE: Optional[dict] = None


def _get_default_input_schema() -> dict:
    """
    Load and cache the default configuration schema from `databases/default_input.yml`.

    This is used to validate that driver scripts (e.g. `run_autophot.py`) only set
    keys that exist in the pipeline's declared configuration.
    """
    global _DEFAULT_INPUT_SCHEMA_CACHE, QUIET_MODE
    if _DEFAULT_INPUT_SCHEMA_CACHE is not None:
        return _DEFAULT_INPUT_SCHEMA_CACHE

    prior_quiet = QUIET_MODE
    QUIET_MODE = True  # suppress log spam while loading the schema
    try:
        # Prefer the pipeline loader when available; otherwise fall back to
        # reading the default YAML directly (useful for minimal envs).
        try:
            _DEFAULT_INPUT_SCHEMA_CACHE = AutomatedPhotometry.load()
        except Exception:
            here = os.path.dirname(os.path.abspath(__file__))
            default_input_path = os.path.join(here, "databases", "default_input.yml")
            with open(default_input_path, "r") as f:
                data = yaml.safe_load(f) or {}
            _DEFAULT_INPUT_SCHEMA_CACHE = data.get("default_input", data)
    finally:
        QUIET_MODE = prior_quiet
    return _DEFAULT_INPUT_SCHEMA_CACHE


def list_parameters(
    *,
    contains: str | None = None,
    max_rows: int | None = None,
    include_null: bool = False,
    show_internal_params: bool = False,
    format: str = "cli",
    width: int | None = None,
    wrap: bool = False,
) -> list[dict]:
    """
    Print (and return) the configuration parameters accepted by AutoPhOT.

    This reads `databases/default_input.yml` as text to preserve line numbers and
    presents a flattened key list with defaults.

    Examples
    --------
    >>> from autophot import list_parameters
    >>> list_parameters(contains="aperture_size")
    >>> list_parameters(show_internal_params=True)
    """
    import os
    import yaml

    here = os.path.dirname(os.path.abspath(__file__))
    yml_path = os.path.join(here, "databases", "default_input.yml")
    with open(yml_path, "r") as f:
        lines = f.read().splitlines()

    # Parse the YAML for defaults.
    parsed = yaml.safe_load("\n".join(lines)) or {}
    root = parsed.get("default_input", parsed)

    def _format_default(v) -> str:
        if v is None:
            return "null"
        if isinstance(v, str):
            return repr(v)
        return str(v)

    def _is_set_internally(desc: str) -> bool:
        """
        Heuristic: treat 'Runtime:' and '(set internally)' markers as internal-only.
        """
        d = (desc or "").lower()
        return ("runtime:" in d) or ("set internally" in d)

    def _is_per_image(path: str, desc: str) -> bool:
        """
        Best-effort flag for values that are populated/updated per FITS image.

        In AutoPhOT, these are typically the 'Runtime:' fields and header-derived
        fallbacks (exposure_time, gain, fwhm, etc.).
        """
        p = str(path or "")
        d = (desc or "").lower()
        if ("runtime:" in d) or ("set internally" in d) or ("runtime fallback" in d):
            return True
        # Known per-image fields even when the YAML comment is terse.
        per_image_keys = {
            "fpath",
            "write_dir",
            "imageFilter",
            "exposure_time",
            "gain",
            "read_noise",
            "pixel_scale",
            "fwhm",
            "science_fwhm",
            "saturate",
            "scale",
        }
        return p in per_image_keys

    def _is_required(path: str) -> bool:
        """
        Best-effort required-key flag for a typical AutoPhOT run.

        These are the keys most workflows must provide (directly or indirectly)
        for the pipeline to do meaningful work.
        """
        required = {
            "fits_dir",
            "wdir",
            "catalog.use_catalog",
            "target_ra",
            "target_dec",
        }
        return str(path or "") in required

    # Walk text lines to map key-path -> line number and extract inline docs.
    entries: list[dict] = []
    stack: list[tuple[int, str]] = []  # (indent_spaces, key)

    # YAML keys in our config are simple tokens (letters/digits/_). Avoid fancy
    # regex character classes here; we only need a robust mapping of key->line.
    key_re = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+)\s*:\s*(?P<rest>.*)$")
    for idx0, raw in enumerate(lines):
        m = key_re.match(raw)
        if not m:
            continue
        indent = len(m.group("indent").replace("\\t", "    "))
        key = str(m.group("key")).strip()
        rest = str(m.group("rest")).strip()
        # Split "value # comment" while preserving the value part.
        value_part, comment_part = rest, ""
        if "#" in rest:
            value_part, comment_part = rest.split("#", 1)
            value_part = value_part.strip()
            comment_part = comment_part.strip()
        type_hint = ""
        desc = ""
        if comment_part:
            # Prefer the human description after '---' when present.
            # Example: "float --- Aperture radius ..." -> "Aperture radius ..."
            if "---" in comment_part:
                lhs, rhs = comment_part.split("---", 1)
                type_hint = lhs.strip()
                desc = rhs.strip()
            else:
                # Some entries only include a type hint (eg: "# bool").
                type_hint = comment_part.strip()
                desc = ""

        # Maintain indentation stack.
        while stack and indent <= stack[-1][0]:
            stack.pop()
        stack.append((indent, key))

        # Only record scalar-ish leaves (skip pure dict headers like "photometry:").
        if value_part == "" or value_part.endswith(":"):
            continue
        if value_part.startswith("#"):
            # No explicit value present (comment-only line).
            continue

        # Build path, stripping the top-level "default_input" if present.
        path_parts = [k for _, k in stack]
        if path_parts and path_parts[0] == "default_input":
            path_parts = path_parts[1:]
        path = ".".join(path_parts)

        # Look up parsed default value by traversing the YAML object.
        cur = root
        for p in path_parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                cur = None
                break

        if cur is None and not include_null:
            # Still include if the YAML line explicitly sets null? (cur is None
            # for both missing and null; we disambiguate by checking "null" token).
            if not re.search(r"\bnull\b", rest, flags=re.IGNORECASE):
                continue

        start_line = idx0 + 1
        end_line = start_line + 1 if (idx0 + 1 < len(lines) and lines[idx0 + 1].strip() == "") else start_line

        entries.append(
            {
                "path": path,
                "default": cur,
                "default_str": _format_default(cur),
                "description": desc,
                "type_hint": type_hint,
                "set_internally": _is_set_internally(desc),
                "per_image": _is_per_image(path, desc),
                "required": _is_required(path),
                "file": yml_path,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    # Filter.
    if contains:
        needle = str(contains).strip().lower()
        entries = [e for e in entries if needle in e["path"].lower()]

    if not bool(show_internal_params):
        entries = [e for e in entries if not bool(e.get("set_internally"))]

    # Sort for stable output (group runtime-set parameters together).
    def _sort_key(e: dict) -> tuple:
        p = str(e.get("path") or "")
        sec = p.split(".", 1)[0] if "." in p else "top"
        runtime_or_top = bool(e.get("set_internally")) or sec == "top"
        runtime_bucket = 0 if runtime_or_top else 1
        # Keep runtime/top items together and first.
        sec_sort = "runtime/top" if runtime_or_top else sec
        # Within each section, list required keys first for readability.
        req_rank = 0 if bool(e.get("required")) else 1
        return (runtime_bucket, sec_sort, req_rank, p)

    entries.sort(key=_sort_key)
    if max_rows is not None:
        entries = entries[: max(0, int(max_rows))]

    # Print.
    fmt = str(format or "cli").strip().lower()
    if fmt in {"md", "markdown"}:
        print("AutoPhOT accepted parameters (from default_input.yml)")
        for e in entries:
            rel_file = os.path.relpath(e["file"], here)
            desc = (e.get("description") or "").strip()
            t = (e.get("type_hint") or "").strip()
            info = desc
            if t and desc:
                info = f"{t} - {desc}"
            elif t and not desc:
                info = t
            desc_str = f" - {desc}" if desc else ""
            print(
                f"- **{e['path']}** `@{rel_file}` ({e['start_line']}-{e['end_line']})  "
                f"[default: {e['default_str']}]{desc_str}"
            )
        return entries

    # Default: CLI-style aligned output for terminal use.
    import shutil
    import textwrap

    # Table width:
    # - If `width` is set: use it exactly (can be wider than the terminal).
    # - Else: use detected terminal width.
    term_w = int(getattr(shutil.get_terminal_size(fallback=(120, 24)), "columns", 120))
    if width is not None:
        term_w = max(80, int(width))
    else:
        term_w = max(80, term_w)

    rel_file = os.path.relpath(yml_path, here)
    title = f"AutoPhOT parameters (from {rel_file})"
    print(title)
    print("-" * min(len(title), term_w))
    if not entries:
        print("(no matches)")
        return entries

    # Add explicit TYPE + PER-IMAGE + REQUIRED columns so users don't have to infer these.
    type_w = 12
    per_w = 8
    req_w = 8

    # Column widths (leave room for wrapped description).
    #
    # Two modes:
    # - wrap=True  : keep columns bounded so DESCRIPTION remains visible.
    # - wrap=False : use "natural" widths (longest entry) so rows do not wrap;
    #               this is intended for sideways scrolling.
    sep = "  "
    nat_key_w = max(3, max(len(str(e.get("path") or "")) for e in entries))
    nat_def_w = max(7, max(len(str(e.get("default_str") or "")) for e in entries))
    nat_type_w = max(4, max(len(str(e.get("type_hint") or "")) for e in entries))

    if not wrap:
        key_w = nat_key_w
        def_w = nat_def_w
        type_w = max(type_w, nat_type_w)
        min_desc_w = 0
        desc_w = max(20, term_w - (key_w + def_w + type_w + per_w + len(sep) * 4))
    else:
        def_w = min(22, nat_def_w)
        def_w = max(10, def_w)
        min_desc_w = 28
        max_key_w_by_term = term_w - def_w - type_w - per_w - len(sep) * 4 - min_desc_w
        key_w = min(60, nat_key_w, max(18, max_key_w_by_term))
        key_w = max(18, key_w)
        desc_w = max(min_desc_w, term_w - (key_w + def_w + type_w + per_w + len(sep) * 4))
    header = (
        f"{'KEY'.ljust(key_w)}{sep}"
        f"{'DEFAULT'.ljust(def_w)}{sep}"
        f"{'TYPE'.ljust(type_w)}{sep}"
        f"{'PER-IMG'.ljust(per_w)}{sep}"
        f"{'REQ'.ljust(req_w)}{sep}"
        f"{'DESCRIPTION'}"
    )
    # Do not truncate to `term_w`; allow very wide lines so terminals that
    # support horizontal scrolling can scroll sideways.
    print(header)
    print("-" * len(header))

    def _section_of(path: str) -> str:
        s = str(path or "").strip()
        if not s:
            return "other"
        return s.split(".", 1)[0] if "." in s else "top"

    last_section = None
    for e in entries:
        section = _section_of(e.get("path", ""))
        if bool(e.get("set_internally")) or section == "top":
            section = "runtime/top"
        if section != last_section:
            if last_section is not None:
                print("-" * len(header))
            # Make section headings visually obvious in plain terminals.
            # Use asterisk "box" with blank lines around it.
            section_title = str(section).replace("_", " ").upper()
            title_line = f"*** {section_title} ***"
            # Match the banner width to the title exactly.
            star_line = "*" * len(title_line)
            print("")
            print(star_line)
            print(title_line)
            print(star_line)
            print("")
            last_section = section

        desc = (e.get("description") or "").strip()
        t = (e.get("type_hint") or "").strip()
        # If the YAML doesn't include a type hint, infer a best-effort type from
        # the default value.
        if not t:
            dv = e.get("default", None)
            if dv is None:
                t = "null"
            elif isinstance(dv, bool):
                t = "bool"
            elif isinstance(dv, int) and not isinstance(dv, bool):
                t = "int"
            elif isinstance(dv, float):
                t = "float"
            elif isinstance(dv, (list, tuple)):
                t = "list"
            elif isinstance(dv, dict):
                t = "dict"
            elif isinstance(dv, str):
                t = "str"
            else:
                t = type(dv).__name__
        type_str = t
        per_flag = "Y" if bool(e.get("per_image")) else "N"
        req_flag = "Y" if bool(e.get("required")) else "N"
        # Keep description text clean; per-image status is in the PER-IMG column.
        info = desc
        if wrap:
            key_lines = (
                textwrap.wrap(
                    e["path"],
                    width=key_w,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [""]
            )
            desc_lines = (
                textwrap.wrap(
                    info,
                    width=desc_w,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                if info
                else [""]
            )
        else:
            key_lines = [e["path"]]
            desc_lines = [info]

        n_lines = max(len(key_lines), len(desc_lines))
        for i in range(n_lines):
            k = key_lines[i] if i < len(key_lines) else ""
            d = desc_lines[i] if i < len(desc_lines) else ""
            if i == 0:
                line = (
                    f"{k.ljust(key_w)}{sep}"
                    f"{e['default_str'][:def_w].ljust(def_w)}{sep}"
                    f"{type_str[:type_w].ljust(type_w)}{sep}"
                    f"{per_flag.ljust(per_w)}{sep}"
                    f"{req_flag.ljust(req_w)}{sep}"
                    f"{d}"
                )
            else:
                line = (
                    f"{k.ljust(key_w)}{sep}"
                    f"{'':{def_w}}{sep}"
                    f"{'':{type_w}}{sep}"
                    f"{'':{per_w}}{sep}"
                    f"{'':{req_w}}{sep}"
                    f"{d}"
                )
            print(line)
    return entries


# Backward/typo-friendly alias (requested): from autophot import list_paramters
def list_paramters(**kwargs):  # noqa: N802
    return list_parameters(**kwargs)


def _find_unknown_config_paths(
    config: dict, schema: dict, prefix: str = ""
) -> list[str]:
    """
    Return a list of unknown key paths present in `config` but not in `schema`.

    Notes
    -----
    - This validates keys only (not value types), because YAML defaults use
      `null`/None in several places where the expected type depends on later
      pipeline logic.
    - Validation is recursive for nested dicts.
    """
    unknown: list[str] = []
    if not isinstance(config, dict) or not isinstance(schema, dict):
        return unknown

    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            unknown.append(path)
            continue
        schema_value = schema[key]
        if isinstance(value, dict) and isinstance(schema_value, dict):
            unknown.extend(
                _find_unknown_config_paths(value, schema_value, prefix=path)
            )

    return unknown

# Project-specific helpers (assumed to be in your codebase).
#
# These imports pull in the full scientific stack (e.g. scikit-image). Keep them
# optional so lightweight utilities (like list_parameters) can be imported in
# minimal environments.
_IMPORT_ERROR_AUTOPHOT_DEPS: Exception | None = None
try:
    from functions import (  # type: ignore
        AutophotYaml,
        concatenate_csv_files,
        log_step,
        log_exception,
        print_progress_bar,
        sanitize_photometric_filters,
    )
    from prepare import Prepare  # type: ignore
except Exception as _exc:  # pragma: no cover
    _IMPORT_ERROR_AUTOPHOT_DEPS = _exc
    log_step = None  # type: ignore
    AutophotYaml = None  # type: ignore
    concatenate_csv_files = None  # type: ignore
    print_progress_bar = None  # type: ignore
    log_exception = None  # type: ignore
    sanitize_photometric_filters = None  # type: ignore
    Prepare = None  # type: ignore


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================
# --- Logging / subprocess helpers ---
def _log(message: str) -> None:
    """
    Logs a message using the logging module if handlers are configured,
    otherwise prints to stdout. When QUIET_MODE is True, messages are
    suppressed to avoid very noisy output during multi-file processing.
    """
    if QUIET_MODE:
        return
    logger = logging.getLogger(__name__)
    if logger.handlers:
        logger.info(message)
    else:
        # Fallback for environments without configured logging handlers.
        # In QUIET_MODE we still suppress output entirely.
        if not QUIET_MODE:
            print(message)


def _run_main_subprocess(
    python_executable: str,
    autophot_exe: str,
    filename: str,
    input_file: str,
    is_template: bool,
) -> Tuple[str, int]:
    """
    Run the low-level photometry pipeline (main.py) on a single FITS file.

    Returns (filename, return_code) so callers can detect failures.
    """
    if not os.path.exists(input_file):
        _log(
            f"[ERROR] Input YAML snapshot is missing: {input_file}\n"
            "It looks like it was deleted mid-run. Re-run the pipeline to regenerate it."
        )
        return filename, 2
    args = [python_executable, autophot_exe, "-f", filename, "-c", input_file]
    if is_template:
        args.append("-temp")

    # When QUIET_MODE is enabled (multi-file, batch use), completely suppress
    # child-process stdout/stderr so that nothing reaches the terminal. Logs
    # can still be written to per-process files if configured inside main.py.
    from subprocess import DEVNULL

    global QUIET_MODE
    kwargs = {"check": False, "text": True}
    if bool(QUIET_MODE):
        kwargs.update({"stdout": DEVNULL, "stderr": DEVNULL})

    try:
        result = subprocess.run(args, **kwargs)
        return filename, result.returncode
    except Exception:
        return filename, 1


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================
def find_variable_sources(
    ra_deg: float, dec_deg: float, radius_arcmin: int = 10
) -> pd.DataFrame:
    """
    Query SIMBAD for variable/interesting sources in sky region.

    Args:
        ra_deg (float): Right ascension (J2000 degrees)
        dec_deg (float): Declination (J2000 degrees)
        radius_arcmin (int): Search radius (arcminutes)

    Returns:
        pd.DataFrame: Sources with RA, DEC, OTYPE, separation, size info
    """
    t0 = time.perf_counter()

    # SIMBAD CONFIGURATION - TIMEOUT PROTECTION
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields(
        "otype",
        "ra",
        "dec",
        "main_id",
        "galdim_majaxis",
        "galdim_minaxis",
        "galdim_angle",
    )

    # Store and override settings
    original_timeout = getattr(Simbad, "TIMEOUT", 10)
    original_row_limit = getattr(Simbad, "ROW_LIMIT", 1000)

    Simbad.TIMEOUT = 30
    Simbad.ROW_LIMIT = 500

    centre_coord = SkyCoord(
        ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk5", equinox="J2000"
    )
    _log(
        f"Querying SIMBAD within {radius_arcmin}' of "
        f"{centre_coord.to_string('hmsdms')}."
    )

    # SOURCE TYPE DEFINITIONS
    object_types = {
        "V*": "Variable Star",
        "Pu*": "Pulsating",
        "Er*": "Eruptive",
        "Ir*": "Irregular",
        "BY*": "BY Dra",
        "RS*": "RS CVn",
        "Fl*": "Flare",
        "Ro*": "Rotating",
        "Ce*": "Cepheid",
        "RR*": "RR Lyr",
        "dS*": "delta Sct",
        "LP*": "Long Period",
        "Mi*": "Mira",
        "TT*": "T Tauri",
        "Or*": "Orion Var",
        "CV*": "Cataclysmic",
        "DN*": "Dwarf Nova",
        "NL*": "Nova-like",
        "No*": "Nova",
        "SN*": "Supernova",
        "AM*": "Polar",
        "DQ*": "Int Polar",
        "G": "Galaxy",
        "GiG": "Int Galaxy",
        "SBG": "Starburst",
        "AGN": "AGN",
        "SyG": "Seyfert",
        "LIN": "LINER",
        "QSO": "Quasar",
        "BLL": "BL Lac",
        "Em*": "Emission*",
        "Be*": "Be Star",
        "XB*": "X-ray Bin",
        "X": "X-ray",
        "PN": "P Nebula",
        "HII": "HII Region",
    }

    # TIMED SIMBAD QUERY WITH FAILSAFE
    query_start = time.perf_counter()
    result = None

    try:
        result = Simbad.query_region(centre_coord, radius=radius_arcmin * u.arcmin)
        query_time = time.perf_counter() - query_start

        if query_time > 25:
            _log(
                f"SIMBAD query completed in {query_time:.1f}s; "
                "consider a smaller search radius for faster performance."
            )

    except Exception as exc:
        _log(
            f"SIMBAD query failed after {time.perf_counter() - query_start:.1f}s: {exc}"
        )
        result = None

    # Restore SIMBAD settings
    Simbad.TIMEOUT = original_timeout
    Simbad.ROW_LIMIT = original_row_limit

    if result is None or len(result) == 0:
        total_time = time.perf_counter() - t0
        _log(f"No SIMBAD sources found in search region ({total_time:.1f}s).")
        return pd.DataFrame(
            columns=[
                "RA",
                "DEC",
                "OTYPE",
                "MAIN_ID",
                "OTYPE_LABEL",
                "OTYPE_opt",
                "separation_arcmin",
                "size_arcmin",
            ]
        )

    # DATA PROCESSING AND CLEANUP
    df = result.to_pandas()

    # Standardize coordinate columns
    ra_col = next((col for col in ["RA_d", "ra"] if col in df.columns), None)
    dec_col = next((col for col in ["DEC_d", "dec"] if col in df.columns), None)

    if ra_col and dec_col:
        df = df.rename(columns={ra_col: "RA", dec_col: "DEC"})
    else:
        _log("SIMBAD result does not contain usable RA/DEC columns; aborting.")
        return pd.DataFrame()

    # Numeric conversion with error handling
    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df["DEC"] = pd.to_numeric(df["DEC"], errors="coerce")
    df = df.dropna(subset=["RA", "DEC"])

    if df.empty:
        return pd.DataFrame()

    # Standardize OTYPE column
    otype_col = next((col for col in ["otype", "OTYPE"] if col in df.columns), None)
    if otype_col:
        df["OTYPE"] = df[otype_col]
    else:
        df["OTYPE"] = np.nan

    # Filter to interesting object types
    df["OTYPE_opt"] = df["OTYPE"]
    df = df[df["OTYPE_opt"].isin(object_types)].copy()

    if df.empty:
        _log(
            f"No SIMBAD sources matched the configured object types "
            f"({time.perf_counter() - t0:.1f}s)."
        )
        return pd.DataFrame()

    # ENRICHMENT: SEPARATION, SIZE, LABELS
    df["OTYPE_LABEL"] = df["OTYPE_opt"].map(object_types)

    # FIX: Create Quantity arrays explicitly for SkyCoord
    ra_qty = df["RA"].astype(float) * u.deg
    dec_qty = df["DEC"].astype(float) * u.deg

    # source_coords = SkyCoord(ra=ra_qty, dec=dec_qty, frame="fk5", equinox="J2000")
    # separations = centre_coord.separation(source_coords).to(u.arcmin).value
    # df["separation_arcmin"] = separations

    # Parse angular size (major axis in arcmin)
    def parse_size(dim):
        if pd.isna(dim) or not dim:
            return np.nan
        try:
            return float(re.search(r"[\d.]+", str(dim)).group())
        except Exception:
            return np.nan

    size_col = next(
        (col for col in ["galdim_majaxis", "dim"] if col in df.columns), None
    )
    if size_col:
        df["size_arcmin"] = df[size_col].apply(parse_size)
    else:
        df["size_arcmin"] = np.nan

    # FINAL CLEANUP AND SORTING
    out_cols = [
        "RA",
        "DEC",
        "OTYPE",
        "MAIN_ID",
        "OTYPE_LABEL",
        "OTYPE_opt",
        "separation_arcmin",
        "size_arcmin",
    ]
    available_cols = [col for col in out_cols if col in df.columns]

    if "MAIN_ID" not in df.columns and "main_id" in df.columns:
        df["MAIN_ID"] = df["main_id"]
    elif "MAIN_ID" not in df.columns:
        df["MAIN_ID"] = ""

    # result_df = df[available_cols].sort_values("separation_arcmin").reset_index(drop=True)

    result_df = df[available_cols].reset_index(drop=True)

    # SUMMARY STATISTICS
    total_time = time.perf_counter() - t0
    _log(
        f"SIMBAD variable-source query finished in {total_time:.1f}s "
        f"(server time: {query_time:.1f}s)."
    )

    if not result_df.empty:
        top_types = result_df["OTYPE_LABEL"].value_counts().head()
        summary = ", ".join(f"{otype} ({count})" for otype, count in top_types.items())
        _log(f"Most common SIMBAD object types in field: {summary}")

    return result_df


def prepare_template_directory(
    fits_dir: str,
    filters: Optional[Iterable[str]] = None,
    include_legacy_p_folders: bool = False,
    confirm_before_continue: bool = True,
) -> Dict[str, List[str]]:
    """
    Create the expected template folder structure under ``<fits_dir>/templates``.

    Parameters
    ----------
    fits_dir : str
        Root science-image directory used by AutoPHoT.
    filters : iterable[str], optional
        Filter list to build. If None, defaults to the supported families:
        UBVRI, ugriz, and JHK.
    include_legacy_p_folders : bool, optional
        If True, also create legacy Pan-STARRS style folders for ugriz
        (e.g. ``rp_template``) for backward compatibility. Defaults to False.
    confirm_before_continue : bool, optional
        If True (default), prompt the user after creating directories:
        "template folder created, do you want to continue?"
        This gives the user a chance to place template FITS files first.

    Returns
    -------
    dict
        Summary with keys:
        - ``templates_root``: absolute path to templates directory
        - ``created``: directories created during this call
        - ``existing``: directories already present
        - ``requested_filters``: normalized filters used
    """
    if fits_dir is None or str(fits_dir).strip() == "":
        raise ValueError("fits_dir must be a valid directory path.")

    fits_root = Path(str(fits_dir)).expanduser().resolve()
    templates_root = fits_root / "templates"
    templates_root.mkdir(parents=True, exist_ok=True)

    default_filters = ["U", "B", "V", "R", "I", "u", "g", "r", "i", "z", "J", "H", "K"]
    use_filters = list(filters) if filters is not None else default_filters
    normalized_filters, dropped_filters = sanitize_photometric_filters(use_filters)
    if dropped_filters:
        _log(
            "Ignoring unsupported/non-filter values while preparing templates: "
            + ", ".join(sorted(set(str(v) for v in dropped_filters)))
        )
    if not normalized_filters:
        raise ValueError(
            "No supported filters provided. Supported families are UBVRI, ugriz, and JHK."
        )

    created_dirs: List[str] = []
    existing_dirs: List[str] = []

    for band in normalized_filters:
        preferred_dir = templates_root / f"{band}_template"
        if preferred_dir.exists():
            existing_dirs.append(str(preferred_dir))
        else:
            preferred_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(preferred_dir))

        if include_legacy_p_folders and band in {"u", "g", "r", "i", "z"}:
            legacy_dir = templates_root / f"{band}p_template"
            if legacy_dir.exists():
                existing_dirs.append(str(legacy_dir))
            else:
                legacy_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(legacy_dir))

    _log(log_step(f"Template directory prepared: {templates_root}"))
    _log(
        "Template folder summary: "
        f"created={len(created_dirs)} existing={len(existing_dirs)} "
        f"filters={','.join(normalized_filters)}"
    )

    if confirm_before_continue:
        prompt = (
            f"\nTemplate folder created at: {templates_root}\n"
            "Please place your template FITS files in the appropriate subfolders.\n"
            "Do you want to continue? [y/N]: "
        )
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = "y"
        if answer not in {"y", "yes"}:
            raise RuntimeError(
                "Stopped by user after template directory creation. "
                "Add your templates, then rerun when ready."
            )

    return {
        "templates_root": str(templates_root),
        "created": created_dirs,
        "existing": existing_dirs,
        "requested_filters": normalized_filters,
    }


def __getattr__(name: str):
    """
    Resolve common misspellings for public module symbols.

    This allows typo-tolerant imports like:
        from autophot import prepapre_template_directory
    by dynamically mapping to the closest public callable/class.
    """
    if not isinstance(name, str) or not name or name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    public_symbols = {
        k: v
        for k, v in globals().items()
        if not k.startswith("_") and (callable(v) or isinstance(v, type))
    }
    if name in public_symbols:
        return public_symbols[name]

    matches = get_close_matches(name, list(public_symbols.keys()), n=1, cutoff=0.72)
    if matches:
        target = matches[0]
        logging.getLogger(__name__).warning(
            "Interpreting unknown symbol '%s' as '%s'.",
            name,
            target,
        )
        return public_symbols[target]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================
# --- Automated Photometry Pipeline ---
class AutomatedPhotometry:
    """
    Orchestrates the photometry pipeline for a target dataset.
    Handles template download, reduction, and light curve generation.
    """

    @staticmethod
    def load() -> dict:
        """
        Loads default input settings from the project YAML configuration.

        Returns:
            dict: Parsed configuration dictionary.
        """
        t0 = time.perf_counter()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_input_path = os.path.join(script_dir, "databases", "default_input.yml")
        default_input = AutophotYaml(default_input_path, "default_input").load()

        # Ensure expected nested sections exist so driver scripts can safely override
        # settings without needing repetitive setdefault() calls.
        for _k in (
            "preprocessing",
            "photometry",
            "templates",
            "wcs",
            "catalog",
            "cosmic_rays",
            "fitting",
            "source_detection",
            "limiting_magnitude",
            "alignment",
            "template_subtraction",
            "background",
            "zeropoint",
            "error",
            "psf",
            "target_photometry",
        ):
            if default_input.get(_k) is None:
                default_input[_k] = {}

        # Common preprocessing defaults used by main.py
        default_input["preprocessing"].setdefault("trim_image", 0)

        _log(log_step(f"Default input: {default_input_path}"))
        _log(f"Configuration loaded in {time.perf_counter() - t0:.3f} seconds.")
        return default_input

    @staticmethod
    def run_photometry(default_input: dict, do_photometry: bool = True) -> str:
        """
        Executes the photometry pipeline based on the provided configuration.

        Args:
            default_input (dict): Configuration dictionary for the pipeline.
            do_photometry (bool): If True, runs reductions before concatenation.

        Returns:
            str: Path to the aggregated light curve CSV.
        """
        t0 = time.perf_counter()
        gc.collect()  # Clean up memory early

        # Determine how many CPU workers to use for *image-level* parallelism.
        # Priority: explicit config key, then environment override, then 1.
        env_ncpu = os.environ.get("AUTOPHOT_NCPU")
        cfg_ncpu = default_input.get("nCPU")
        try:
            n_cpu = (
                int(env_ncpu)
                if env_ncpu is not None
                else int(cfg_ncpu) if cfg_ncpu is not None else 1
            )
        except (TypeError, ValueError):
            n_cpu = 1
        if n_cpu < 1:
            n_cpu = 1
        # SFFT subprocess uses libgomp/OpenMP and can leak semaphores; avoid nested
        # parallelism (ProcessPoolExecutor + SFFT) and HPC thread limits by forcing 1 worker
        # unless the user explicitly set AUTOPHOT_NCPU.
        tsub = default_input.get("template_subtraction") or {}
        if (
            env_ncpu is None
            and tsub.get("do_subtraction")
            and (tsub.get("method") or "").strip().lower() == "sfft"
        ):
            if n_cpu > 1:
                logging.getLogger(__name__).info(
                    "SFFT template subtraction enabled: forcing nCPU=1 to avoid thread/semaphore issues."
                )
            n_cpu = 1
        parallel_files = n_cpu > 1

        global QUIET_MODE
        if parallel_files:
            # Suppress this orchestrator's chatter in multi-file mode; the per-image
            # main.py processes still handle their own detailed logging.
            QUIET_MODE = True

            # Additionally, raise the level of any stream handlers attached to the
            # root logger so that INFO/DEBUG messages are not printed to screen.
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                try:
                    # Only adjust stream-like handlers (stdout / stderr); leave file
                    # handlers untouched so logs can still be written to disk.
                    if getattr(handler, "stream", None) is not None:
                        handler.setLevel(logging.WARNING)
                except Exception:
                    continue

        # Normalise paths and close open figures
        fits_dir = default_input.get("fits_dir") or ""
        if isinstance(fits_dir, str) and fits_dir.endswith("/"):
            fits_dir = fits_dir[:-1]
            default_input["fits_dir"] = fits_dir

        plt.close("all")

        # ------------------------------------------------------------------
        # Config validation: fail fast on typos/unknown keys from driver scripts
        # ------------------------------------------------------------------
        strict_validation_raw = os.environ.get("AUTOPHOT_STRICT_CONFIG", "1")
        strict_validation = str(strict_validation_raw).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if strict_validation:
            schema = _get_default_input_schema()
            unknown_paths = _find_unknown_config_paths(default_input, schema)
            if unknown_paths:
                unique_sorted = sorted(set(unknown_paths))
                raise KeyError(
                    "Unknown AutoPHOT config keys detected. "
                    "These were found in the driver-provided configuration but "
                    "do not exist in `databases/default_input.yml`:\n  - "
                    + "\n  - ".join(unique_sorted)
                )

        # ------------------------------------------------------------------
        # Optional Gaia custom catalog build from user transmission curves.
        #
        # New convention:
        # - Users may set `catalog.use_catalog` (scalar or mapping) to "gaia_custom"
        #   for the filter(s) they want calibrated using a Gaia curve-map catalog.
        # - When "gaia_custom" is requested, AutoPhOT requires
        #   `catalog.transmission_curve_map` (and/or `catalog.curve_map_svo`) and will
        #   build/reuse a single custom
        #   catalog CSV, then transparently route "gaia_custom" -> "custom" using
        #   `catalog.catalog_custom_fpath`.
        # - Plain "gaia" continues to use the standard Gaia photometry path with
        #   no curve-map build.
        # ------------------------------------------------------------------
        catalog_cfg = default_input.setdefault("catalog", {}) or {}

        def _is_gaia_custom_token(v) -> bool:
            if v is None:
                return False
            return str(v).strip().lower() in {"gaia_custom", "gaia-custom", "gaiacustom"}

        use_catalog_cfg = catalog_cfg.get("use_catalog", None)
        gaia_custom_requested = False
        gaia_custom_groups = []
        if _is_gaia_custom_token(use_catalog_cfg):
            gaia_custom_requested = True
            gaia_custom_groups = ["*"]
        elif isinstance(use_catalog_cfg, dict):
            for k, v in use_catalog_cfg.items():
                if _is_gaia_custom_token(v):
                    gaia_custom_requested = True
                    gaia_custom_groups.append(str(k))

        if gaia_custom_requested and do_photometry:
            from functions import parse_supported_filter_group_key, normalize_photometric_filter_name

            curve_map = catalog_cfg.get("transmission_curve_map", None)
            if curve_map is None or curve_map == {}:
                raise ValueError(
                    "catalog.use_catalog requests 'gaia_custom', but catalog.transmission_curve_map is not set. "
                    "Provide a band->curve-path mapping, e.g. {'g':'/path/g.dat','r':'/path/r.dat','i':'/path/i.dat'}."
                )
            if not isinstance(curve_map, dict):
                raise TypeError(
                    "catalog.transmission_curve_map must be a dictionary of band->curve path, "
                    "e.g. {'g': '/path/g.dat', 'r': '/path/r.dat'}."
                )

            # Determine which bands must be present in curve_map based on the mapping keys.
            required_bands = set()
            if gaia_custom_groups == ["*"]:
                # Global gaia_custom: require all curve_map keys (user-defined).
                required_bands = {str(k) for k in curve_map.keys()}
            else:
                for g in gaia_custom_groups:
                    bands = parse_supported_filter_group_key(g)
                    if bands:
                        required_bands.update(bands)
                    else:
                        # allow single-band keys (e.g. "g")
                        nb = normalize_photometric_filter_name(g)
                        if nb is not None:
                            required_bands.add(nb)
                        else:
                            # If it's a non-standard token, we can't infer required curves.
                            # Continue; the build will still use whatever is in curve_map.
                            pass

            missing = sorted([b for b in required_bands if b not in curve_map])
            if missing:
                raise ValueError(
                    "catalog.use_catalog requests 'gaia_custom' for bands requiring transmission_curve_map entries, "
                    f"but these are missing from catalog.transmission_curve_map: {missing}"
                )

            work_dir = default_input.get("fits_dir") or ""
            out_dir_name = "_" + str(default_input.get("outdir_name", "REDUCED"))
            out_dir_base = os.path.basename(work_dir) + out_dir_name
            out_dir = os.path.join(os.path.dirname(work_dir), out_dir_base)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            curve_catalog_csv = os.path.join(out_dir, "GAIA_CurveMap_Catalog.csv")

            # Store the custom catalog path; downstream uses it when resolving "custom".
            catalog_cfg["catalog_custom_fpath"] = curve_catalog_csv

            # Rewrite gaia_custom -> custom (scalar or mapping) so the rest of the
            # pipeline doesn't need to special-case gaia_custom.
            if _is_gaia_custom_token(use_catalog_cfg):
                catalog_cfg["use_catalog"] = "custom"
            elif isinstance(use_catalog_cfg, dict):
                new_map = {}
                for k, v in use_catalog_cfg.items():
                    new_map[k] = ("custom" if _is_gaia_custom_token(v) else v)
                catalog_cfg["use_catalog"] = new_map

            default_input["catalog"] = catalog_cfg

            if os.path.exists(curve_catalog_csv):
                _log(
                    f"Using existing Gaia curve-map catalog: {curve_catalog_csv} "
                    "(skipping rebuild)."
                )
            else:
                target_ra = default_input.get("target_ra", None)
                target_dec = default_input.get("target_dec", None)
                if target_ra is None or target_dec is None:
                    # If coordinates are not set explicitly, try TNS lookup
                    # using target_name before failing curve-map catalog build.
                    # Create a temporary Prepare instance for TNS lookup
                    try:
                        temp_prepare = Prepare(default_input=default_input)
                        tns_coords = temp_prepare.check_tns()
                        default_input.update(
                            {
                                "target_ra": tns_coords["radeg"],
                                "target_dec": tns_coords["decdeg"],
                                "name_prefix": tns_coords.get("name_prefix"),
                                "objname": tns_coords.get("objname"),
                            }
                        )
                        target_ra = default_input.get("target_ra", None)
                        target_dec = default_input.get("target_dec", None)
                        _log(
                            "Resolved target coordinates from TNS for gaia_custom build: "
                            f"RA={float(target_ra):.6f}, Dec={float(target_dec):.6f}"
                        )
                    except Exception as e:
                        _log(f"TNS lookup failed: {e}")
                        pass
                if target_ra is None or target_dec is None:
                    raise ValueError(
                        "catalog.use_catalog requests 'gaia_custom', but target_ra/target_dec are missing. "
                        "Please set both coordinates in degrees, or provide target_name "
                        "with working TNS credentials."
                    )

                radius_deg = float(
                    catalog_cfg.get(
                        "gaia_xp_radius_deg",
                        catalog_cfg.get("catalog_radius", 0.25),
                    )
                )
                max_sources = int(
                    catalog_cfg.get(
                        "gaia_curve_map_max_sources",
                        catalog_cfg.get("gaia_xp_max_sources", 100),
                    )
                )
                curve_order = str(
                    catalog_cfg.get(
                        "gaia_curve_map_order_by",
                        catalog_cfg.get("gaia_xp_order_by", "distance"),
                    )
                ).strip().lower()
                svo_curve_map = catalog_cfg.get("curve_map_svo", None)

                _log(
                    log_step("Gaia custom catalog (gaia_custom) curve map")
                )
                _log(
                    f"RA={float(target_ra):.6f}, Dec={float(target_dec):.6f}, "
                    f"radius={radius_deg:.4f} deg, max_sources={max_sources}, "
                    f"order_by={curve_order}, bands={sorted(required_bands) if required_bands else 'curve_map keys'}"
                )
                from autophot_gaia_curves import build_custom_catalog

                build_custom_catalog(
                    ra_deg=float(target_ra),
                    dec_deg=float(target_dec),
                    radius_deg=radius_deg,
                    max_sources=max_sources,
                    curves={str(k): str(v) for k, v in curve_map.items()},
                    out_csv=curve_catalog_csv,
                    svo_filters=(
                        {str(k): str(v) for k, v in svo_curve_map.items()}
                        if isinstance(svo_curve_map, dict) and len(svo_curve_map) > 0
                        else None
                    ),
                    gaia_query_pause_before_sec=float(
                        catalog_cfg.get("gaia_archive_query_pause_before_sec", 1.0)
                    ),
                    gaia_query_pause_after_sec=float(
                        catalog_cfg.get("gaia_archive_query_pause_after_sec", 1.0)
                    ),
                    gaia_xp_batch_size=int(catalog_cfg.get("gaia_xp_batch_size", 200)),
                    gaia_xp_batch_pause_sec=float(
                        catalog_cfg.get("gaia_xp_batch_pause_sec", 1.0)
                    ),
                    gaia_archive_max_retries=int(
                        catalog_cfg.get("gaia_archive_max_retries", 3)
                    ),
                    gaia_archive_retry_base_delay_sec=float(
                        catalog_cfg.get("gaia_archive_retry_base_delay_sec", 2.0)
                    ),
                    gaia_xp_order_by=curve_order,
                    gaia_xp_show_progress=bool(
                        catalog_cfg.get("gaia_xp_show_progress", True)
                    ),
                    gaia_nearest_prefetch_factor=int(
                        catalog_cfg.get("gaia_nearest_prefetch_factor", 50)
                    ),
                    gaia_nearest_prefetch_min=int(
                        catalog_cfg.get("gaia_nearest_prefetch_min", 500)
                    ),
                    gaia_nearest_prefetch_max=int(
                        catalog_cfg.get("gaia_nearest_prefetch_max", 10000)
                    ),
                )
                logging.getLogger(__name__).debug(
                    "Gaia curve-map catalog written: %s",
                    curve_catalog_csv,
                )

        # Initialise preparation helper
        prepare_db = Prepare(default_input=default_input)

        # Validate catalog configuration only if doing photometry
        # (skip checks when just recovering files and creating output table)
        available_filters = []
        if do_photometry:
            available_filters = prepare_db.check_catalog()

        # ------------------------------------------------------------------
        # Credentials: prefer environment variables via autophot_tokens, then YAML.
        # This keeps secrets out of YAML and allows HPC/CI configuration.
        # ------------------------------------------------------------------
        try:
            # Optional local override file (gitignored) for convenience.
            # If absent, fall back to environment-variable resolution in autophot_tokens.
            try:
                import autophot_tokens_local as autophot_tokens  # type: ignore
            except Exception:
                import autophot_tokens  # type: ignore

            default_input.setdefault("wcs", {})
            default_input.setdefault("catalog", {})
            wcs_cfg = default_input.get("wcs") or {}
            cat_cfg = default_input.get("catalog") or {}

            def _env_first(*names):
                for n in names:
                    v = os.getenv(n)
                    if v is not None and str(v).strip() != "":
                        return str(v).strip()
                return None

            def _env_int_first(*names):
                v = _env_first(*names)
                if v is None:
                    return None
                try:
                    return int(v)
                except Exception:
                    return None

            # TNS credentials
            # Treat empty-string values as "missing" so user overrides don't
            # accidentally block token resolution.
            for k in ("TNS_BOT_ID", "TNS_BOT_NAME", "TNS_BOT_API"):
                cur = wcs_cfg.get(k)
                is_missing = cur is None or (isinstance(cur, str) and cur.strip() == "")
                if is_missing:
                    v = getattr(autophot_tokens, k, None)
                    if v is None:
                        # Fallback to environment variables directly in case
                        # autophot_tokens module values are stale/cached.
                        if k == "TNS_BOT_ID":
                            v = _env_int_first("TNS_BOT_ID")
                        elif k == "TNS_BOT_NAME":
                            v = _env_first("TNS_BOT_NAME")
                        elif k == "TNS_BOT_API":
                            v = _env_first("TNS_BOT_API", "TNS_BOT_API_KEY")
                    if v is not None:
                        if isinstance(v, str):
                            v = v.strip()
                        if not (isinstance(v, str) and v == ""):
                            wcs_cfg[k] = v
            default_input["wcs"] = wcs_cfg

            # MAST CasJobs credentials (Refcat)
            # Treat empty-string values as "missing" and normalize types so
            # downstream libraries always receive clean strings.
            wsid_cur = cat_cfg.get("MASTcasjobs_wsid")
            if wsid_cur is None or (
                isinstance(wsid_cur, str) and wsid_cur.strip() == ""
            ):
                v = getattr(autophot_tokens, "MASTcasjobs_wsid", None)
                if v is None:
                    v = _env_int_first("MASTCASJOBS_WSID")
                if v is not None:
                    v = str(v).strip()
                    if v != "":
                        cat_cfg["MASTcasjobs_wsid"] = v

            pwd_cur = cat_cfg.get("MASTcasjobs_pwd")
            if pwd_cur is None or (isinstance(pwd_cur, str) and pwd_cur.strip() == ""):
                v = getattr(autophot_tokens, "MASTcasjobs_pwd", None)
                if v is None:
                    v = _env_first("MASTCASJOBS_PWD")
                if v is not None:
                    v = str(v).strip()
                    if v != "":
                        cat_cfg["MASTcasjobs_pwd"] = v
            default_input["catalog"] = cat_cfg
        except Exception:
            # Tokens module is optional; leave config as-is.
            pass

        # Resolve paths and interpreter
        current_dir = os.path.dirname(os.path.abspath(__file__))
        autophot_exe = os.path.join(current_dir, "main.py")
        python_executable = sys.executable

        if do_photometry:
            _log(log_step("AutoPhOT photometry run"))

            # List of available filters (excluding error columns)
            filt_list = [
                f
                for f in available_filters
                if "_err" not in f and f not in ["RA", "DEC"]
            ]
            _log(f"Available filters: {filt_list}")

            # Optional: Enrich target metadata from TNS
            try:
                _log(log_step("TNS check"))
                tns_coords = prepare_db.check_tns()
                default_input.update(
                    {
                        "target_ra": tns_coords["radeg"],
                        "target_dec": tns_coords["decdeg"],
                        "name_prefix": tns_coords["name_prefix"],
                        "objname": tns_coords["objname"],
                    }
                )
            except Exception:
                # Provide a clean, user-friendly message instead of exposing
                # low-level exceptions (e.g. missing 'radeg', network errors).
                if (
                    default_input.get("target_ra") is not None
                    and default_input.get("target_dec") is not None
                ):
                    _log(
                        "[WARNING] TNS lookup skipped; using target_ra/target_dec "
                        "values from the configuration."
                    )
                else:
                    _log(
                        "[ERROR] Cannot proceed: no TNS access, no target_name, "
                        "and no fallback RA/Dec provided in the configuration."
                    )
                    print(
                        f"\n{'='*60}\n"
                        f"ERROR: Cannot proceed without target coordinates.\n\n"
                        f"You must provide one of:\n"
                        f"  1. target_name + TNS API credentials (in wcs: TNS_BOT_ID, etc.)\n"
                        f"  2. target_ra and target_dec coordinates manually\n"
                        f"  3. target_name with RA/Dec fallback (if TNS fails)\n\n"
                        f"Add to your input:\n"
                        f"   autophot_input['target_ra'] = <RA in degrees>\n"
                        f"   autophot_input['target_dec'] = <Dec in degrees>\n"
                        f"{'='*60}\n"
                    )
                    sys.exit("Stopped: No target coordinates provided.")

            # Clean and validate input files
            file_list = prepare_db.clean()
            if not default_input.get("skip_file_check", False):
                file_list = prepare_db.check_files(flist=file_list)

                if len(file_list) == 0:
                    _log(
                        log_step("No images left after validation; skipping photometry")
                    )
                    do_photometry = False
                else:
                    file_list, required_filters = prepare_db.check_filters(
                        flist=file_list, available_filters=available_filters
                    )
            else:
                required_filters = available_filters

            if do_photometry:
                required_filters = list(set(required_filters))

                # Template handling and optional downloads
                template_folder = os.path.join(default_input["fits_dir"], "templates")
                ts_cfg = default_input.get("template_subtraction", {})
                # Backward compatibility: older configs used `get_PS1_template: True`.
                # Prefer the newer `download_templates: <kind>` selector.
                if not ts_cfg.get("download_templates", False) and ts_cfg.get(
                    "get_PS1_template", False
                ):
                    ts_cfg["download_templates"] = "panstarrs"
                if ts_cfg.get("download_templates", False) and ts_cfg.get(
                    "do_subtraction", False
                ):
                    download_kind = ts_cfg["download_templates"]
                    size_default = ts_cfg.get("templates_size", 10)
                    if download_kind is True:
                        download_kind = "panstarrs"

                    _log(
                        "Downloading templates for science image filters: %s",
                        ", ".join(sorted(required_filters)) if required_filters else "None"
                    )

                    if download_kind == "panstarrs":
                        _log(log_step("Download templates: Pan-STARRS"))
                        from templates import download_panstarrs_template

                        for f in required_filters:
                            download_panstarrs_template(
                                ra=default_input["target_ra"],
                                dec=default_input["target_dec"],
                                size=size_default,
                                template_folder=template_folder,
                                band=f,
                            )
                    elif download_kind == "sdss":
                        _log(log_step("Download templates: SDSS"))
                        from templates import download_sdss_template

                        for f in required_filters:
                            download_sdss_template(
                                ra=default_input["target_ra"],
                                dec=default_input["target_dec"],
                                size=size_default,
                                template_folder=template_folder,
                                f=f,
                            )
                    elif download_kind == "legacy":
                        _log(
                            log_step("Download templates: Legacy Surveys")
                        )
                        from templates import download_legacy_template

                        for f in required_filters:
                            download_legacy_template(
                                ra=default_input["target_ra"],
                                dec=default_input["target_dec"],
                                size=size_default,
                                template_folder=template_folder,
                                band=f,
                            )
                    elif download_kind == "2mass":
                        _log(log_step("Download templates: 2MASS"))
                        from templates import download_2mass_template

                        for f in required_filters:
                            download_2mass_template(
                                ra=default_input["target_ra"],
                                dec=default_input["target_dec"],
                                size=ts_cfg.get("templates_size", 10),
                                template_folder=template_folder,
                                band=f,
                            )

                # Discover and validate template files if subtraction is enabled
                template_file_list = []
                if ts_cfg.get("do_subtraction", False):
                    # Only prepare templates for filters actually needed by science images
                    # This avoids unnecessary template preparation for unused filters
                    template_required_filters = required_filters
                    logger = logging.getLogger(__name__)
                    logger.info(
                        "Looking for templates for science image filters: %s",
                        ", ".join(sorted(template_required_filters)) if template_required_filters else "None"
                    )
                    template_file_list = prepare_db.find_templates(
                        required_filters=template_required_filters
                    )
                    template_file_list = prepare_db.check_files(
                        flist=template_file_list, template_files=True
                    )
                    template_file_list, _ = prepare_db.check_filters(
                        flist=template_file_list, available_filters=available_filters
                    )
                    # Re-run telescope header check on science + templates so template images
                    # with TELESCOP/INSTRUME are included in the telescope database
                    if template_file_list and not default_input.get(
                        "skip_file_check", False
                    ):
                        combined = list(file_list) + list(template_file_list)
                        combined_checked = prepare_db.check_files(
                            flist=combined, template_files=True
                        )
                        template_folder_norm = os.path.normpath(template_folder)
                        file_list = [
                            f
                            for f in combined_checked
                            if os.path.normpath(f).startswith(template_folder_norm)
                            is False
                        ]
                        template_file_list = [
                            f
                            for f in combined_checked
                            if os.path.normpath(f).startswith(template_folder_norm)
                        ]

                # Prepare output directory and save input snapshot
                backup_yaml = copy.deepcopy(default_input)
                # If template subtraction was requested but no valid template
                # files were found, fall back to non-template photometry.
                if ts_cfg.get("do_subtraction", False) and not template_file_list:
                    _log(
                        "[WARNING] template_subtraction.do_subtraction=True "
                        "but no template files were found or passed checks. "
                        "Proceeding with non-template photometry only."
                    )
                    backup_yaml.setdefault("template_subtraction", {})[
                        "do_subtraction"
                    ] = False
                work_dir = backup_yaml["fits_dir"]
                new_dir = "_" + backup_yaml["outdir_name"]
                base_dir = os.path.basename(work_dir)
                work_loc = base_dir + new_dir
                new_output_dir = os.path.join(os.path.dirname(work_dir), work_loc)
                Path(new_output_dir).mkdir(parents=True, exist_ok=True)

                # At this point, main.py will handle catalog building/downloading.
                # If a catalog CSV already exists in the expected location, its
                # own logic will re-use that file without further network access.

                # SIMBAD variable sources near target for context

                if ts_cfg.get("get_simbad_sources", True):
                    try:
                        vars_df = find_variable_sources(
                            ra_deg=default_input["target_ra"],
                            dec_deg=default_input["target_dec"],
                        )
                        variable_sources = [r.to_dict() for _, r in vars_df.iterrows()]

                        backup_yaml["variable_sources"] = variable_sources
                    except Exception as e:
                        _log(f"[WARNING] SIMBAD variable-source enrichment skipped:")

                else:
                    backup_yaml["variable_sources"] = {}

                # Write input snapshot
                input_file = os.path.join(new_output_dir, "input.yaml")
                with open(input_file, "w") as fh:
                    yaml.dump(backup_yaml, fh, default_flow_style=False)

                # Reduce templates first
                if template_file_list:
                    if parallel_files and len(template_file_list) > 1:
                        # Parallelise template reductions when multiple workers requested.
                        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
                            futures = {
                                executor.submit(
                                    _run_main_subprocess,
                                    python_executable,
                                    autophot_exe,
                                    template,
                                    input_file,
                                    True,
                                ): template
                                for template in template_file_list
                            }
                            total_t = len(futures)
                            done_t = 0
                            for fut in as_completed(futures):
                                fname, rc = fut.result()
                                done_t += 1
                                if rc != 0:
                                    # Minimal reporting; main.py already logged details.
                                    _log(
                                        f"[{done_t}/{total_t}] [TEMPLATE FAIL] {fname} (exit code {rc})"
                                    )
                                else:
                                    _log(f"[{done_t}/{total_t}] [TEMPLATE OK]   {fname}")
                        gc.collect()
                    else:
                        _log(log_step("Reduce/calibrate template files"))
                        for template in print_progress_bar(
                            template_file_list, title="Template files calibrated"
                        ):
                            _run_main_subprocess(
                                python_executable,
                                autophot_exe,
                                template,
                                input_file,
                                True,
                            )
                            gc.collect()

                # Reduce science frames
                _log(log_step("Reduce/calibrate science files"))

                counter = 0
                if file_list:
                    file_list = np.sort(file_list)[
                        ::-1
                    ]  # Sort newest first if filenames encode time

                    if parallel_files and len(file_list) > 1:
                        # Parallel image-level execution: each worker runs main.py on one file.
                        _log(
                            f"Running {len(file_list)} science files with nCPU={n_cpu} (parallel)."
                        )
                        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
                            futures = {
                                executor.submit(
                                    _run_main_subprocess,
                                    python_executable,
                                    autophot_exe,
                                    str(file),
                                    input_file,
                                    False,
                                ): str(file)
                                for file in file_list
                            }
                            total = len(futures)
                            for fut in as_completed(futures):
                                fname, rc = fut.result()
                                counter += 1
                                if rc == 0:
                                    _log(f"[{counter}/{total}] [OK]    {fname}")
                                else:
                                    _log(f"[{counter}/{total}] [FAIL]  {fname} (exit code {rc})")
                                    if rc == 2 and (not os.path.exists(input_file)):
                                        _log(
                                            "[ERROR] Input YAML snapshot disappeared during processing. "
                                            "Stopping early; re-run required."
                                        )
                                        break
                        gc.collect()
                    else:
                        from tqdm import tqdm
                        for file in tqdm(file_list, desc="Processing", unit="file", total=len(file_list)):
                            try:
                                fname, rc = _run_main_subprocess(
                                    python_executable,
                                    autophot_exe,
                                    str(file),
                                    input_file,
                                    False,
                                )
                                if rc == 2 and (not os.path.exists(input_file)):
                                    _log(
                                        "[ERROR] Input YAML snapshot disappeared during processing. "
                                        "Stopping early; re-run required."
                                    )
                                    break
                                gc.collect()
                                counter += 1
                                _log(f"[{counter}/{len(file_list)}] [OK]    {str(file)}")
                            except Exception as e:
                                import traceback

                                tb = traceback.format_exc(limit=1)
                                _log(
                                    f"[{counter + 1}/{len(file_list)}] [ERROR] Problem with file: {file}: {e} | {tb}"
                                )

            # Concatenate per-image outputs into one light curve CSV
            reduced_loc = f"{default_input['fits_dir']}_{default_input['outdir_name']}"
            _log(log_step(f"Collect photometry: {reduced_loc}"))
            output_loc = os.path.join(reduced_loc, "lightcurve_output.csv")
            concatenate_csv_files(
                folder_path=reduced_loc,
                output_filename=output_loc,
                loc_file="OUTPUT_*.csv",
            )
            _log(
                f"Photometry pipeline completed in {time.perf_counter() - t0:.3f} seconds."
            )
            output_photometry = output_loc
        else:
            # do_photometry=False: recover existing files and create output table
            _log(log_step("Recover photometry (skip reductions)"))

            # Set up output directory path (same logic as when do_photometry=True)
            work_dir = default_input.get("fits_dir") or ""
            out_dir_name = "_" + str(default_input.get("outdir_name", "REDUCED"))
            reduced_loc = f"{work_dir}{out_dir_name}"

            # Concatenate existing per-image outputs into one light curve CSV
            _log(log_step(f"Collect photometry: {reduced_loc}"))
            output_photometry = os.path.join(reduced_loc, "lightcurve_output.csv")

            if os.path.exists(reduced_loc):
                concatenate_csv_files(
                    folder_path=reduced_loc,
                    output_filename=output_photometry,
                    loc_file="OUTPUT_*.csv",
                )
                _log(f"Output light curve: {output_photometry}")
            else:
                _log(f"[WARNING] Reduced directory not found: {reduced_loc}")
                output_photometry = ""

            _log(
                f"Recovery completed in {time.perf_counter() - t0:.3f} seconds."
            )

        return output_photometry


def main(argv: Optional[List[str]] = None) -> int:
    """
    Simple CLI wrapper for the high-level driver.

    Usage:
        autophot-driver run
        python autophot.py run
    """
    if argv is None:
        argv = sys.argv[1:]

    ap = AutomatedPhotometry()
    config = ap.load()

    # Global verbosity control (0=warnings/errors, 1=info, 2=debug).
    try:
        vlevel = int(config.get("global_verbose_level", 1))
    except Exception:
        vlevel = 1
    if vlevel <= 0:
        log_level = logging.WARNING
    elif vlevel == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    do_run = len(argv) > 0 and str(argv[0]).lower() == "run"
    output = ap.run_photometry(config, do_photometry=do_run)
    _log(f"Output light curve: {output}")
    return 0


# --- Main Execution ---
if __name__ == "__main__":
    raise SystemExit(main())
