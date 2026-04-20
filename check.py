#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FITS Header Inspector - Original API Compatible
===============================================

Original interface: FitsInfo(input_yaml, flist)
Analyzes FITS headers for telescope/instrument/filter keywords.
Builds telescope.yml database with user interaction.
Supports keyword aliases (DATE-OBS, FILTERS, etc.) and fuzzy matching.
"""

import os
import sys
import re
import yaml
import logging  # Project-specific helpers (assumed to be in your codebase)
from functions import (
    border_msg,
    AutophotYaml,
    concatenate_csv_files,
    print_progress_bar,
    sanitize_photometric_filters,
)
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import get_close_matches
from tqdm import tqdm
from collections import defaultdict

# -----------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------

# Astropy fallback for FITS header reading
try:
    from astropy.io import fits

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


# Class constants for keyword mapping and validation
DEFAULT_KEYS = {"TELESCOP": "TELESCOP", "INSTRUME": "INSTRUME", "FILTER": "FILTER"}


OPTIONAL_KEYWORDS = {
    "GAIN": {"label": "gain", "units": "e/ADU"},
    "RDNOISE": {"label": "readnoise", "units": "e/pixel"},
    "SATURATE": {"label": "saturate", "units": "ADU"},
    "AIRMASS": {"label": "airmass", "units": ""},
    "MJD": {"label": "mjd", "units": "Modified Julian date"},
    "Date": {"label": "date", "units": "Date of observations in ISO standard"},
    "EXPTIME": {"label": "exptime", "units": "seconds"},
}
AVOID_FILTERS = {"open", "clear", "open ", " clear"}

# Logical keyword (same names as OPTIONAL_KEYWORDS / ask_for_keyword) -> list of
# FITS key names in priority order. First key present in the header (case-insensitive)
# is chosen with no interactive prompt. Example:
#   auto_accept = {"MJD": ["MJD-OBS"], "Date": ["DATE-OBS"], "EXPTIME": ["EXPTIME"]}
auto_accept = {
    "MJD": ["MJD-OBS", "MJD_OBS", "MJD", "MJDSTART", "OBSMJD"],
    "Date": ["DATE-OBS", "DATEOBS", "UTC-OBS", "OBS-DATE"],
    "EXPTIME": ["EXPTIME", "EXPOSURE", "TEXP", "EXPTIME0"],
}


# Common FITS keyword aliases compiled from astronomy standards
KEYWORD_ALIASES = {
    "TELESCOP": ["TELESCOP"],
    # Instrument must always be carried in the INSTRUME keyword .
    "INSTRUME": ["INSTRUME"],
    "FILTER": ["FILTERS", "FLTRNAM", "FILNAM1", "FILTNAM1", "FILTER1", "FILTER"],
    "Date": ["DATE-OBS", "DATEOBS", "UTC-OBS", "OBS-DATE", "DATE", "UTSTART"],
    "EXPTIME": ["EXPOSURE", "TEXP", "EXPTIME", "EXPTIME0"],
    "GAIN": ["GAIN", "EGAIN", "READGAIN", "GAIN0"],
    "RDNOISE": ["RDNOISE", "RNOISE", "READNOIS", "RDNOISE0"],
    "SATURATE": ["SATURATE", "SATLEVEL", "FULLWELL", "SATURATE0"],
    "AIRMASS": ["AIRMASS", "AMASS", "AIRMASS0"],
    "MJD": ["MJD-OBS", "MJD_OBS", "MJD", "MJDSTART", "OBSMJD"],
}

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


def _header_key_ci(header: dict, canonical: str) -> Optional[str]:
    """Return the actual header key matching ``canonical`` (case-insensitive), or None."""
    if not header:
        return None
    want = canonical.upper()
    for k in header:
        if str(k).upper() == want:
            return str(k)
    return None


def auto_accept_header_key(header: dict, logical_keyword: str) -> Optional[str]:
    """
    If ``logical_keyword`` is listed in ``auto_accept``, return the first matching
    FITS key present in ``header`` (case-insensitive), else None.
    """
    candidates = auto_accept.get(logical_keyword)
    if not candidates:
        return None
    for canon in candidates:
        found = _header_key_ci(header, canon)
        if found is not None:
            return found
    return None


def get_header(filename):
    """
    Extract primary HDU header from FITS file as dictionary.

    Prioritizes astropy.io.fits for robust parsing.
    Falls back to custom 'functions.get_header' if available.

    Args:
        filename (str): Path to FITS file

    Returns:
        dict: Header keywords {KEY: value}

    Raises:
        ImportError: No FITS reader available
        Exception: File read errors logged
    """
    try:
        if ASTROPY_AVAILABLE:
            with fits.open(filename) as hdul:
                return dict(hdul[0].header)
        # Custom fallback for original codebase compatibility
        try:
            from functions import get_header as custom_get_header

            return custom_get_header(filename)
        except ImportError:
            raise ImportError("No FITS reader. Install astropy.")
    except Exception as e:
        logging.error(f"Cannot read {filename}: {e}")
        return {}


# -----------------------------------------------------------------------------
# MAIN CLASS
# -----------------------------------------------------------------------------


class FitsInfo:
    """
    FITS header analysis and telescope database builder.

    Original interface preserved: __init__(input_yaml, flist)
    Scans files for TELESCOP/INSTRUME/FILTER keywords using aliases + fuzzy matching.
    Interactively builds telescope.yml with instrument parameters.
    """

    def __init__(self, input_yaml, flist=None, template_files=False):
        """
        Initialize with original API parameters.

        Args:
            input_yaml (dict): Config with 'wdir', 'fits_dir' keys
            flist (list, optional): Explicit FITS files list. Auto-discovers if None
            template_files (bool): If True, do not reject files missing TELESCOP/INSTRUME
                (used when validating template images, which may lack those headers).

        Initializes:
            - File list from fits_dir or provided flist
            - Working directory for telescope.yml output
            - Logging and filter database
        """
        self.input_yaml = input_yaml
        self.wdir = Path(input_yaml.get("wdir", "."))
        fits_dir = input_yaml.get("fits_dir", None)
        if fits_dir is None or str(fits_dir).strip() == "":
            raise ValueError("fits_dir is required (set default_input.fits_dir).")
        self.fits_dir = Path(str(fits_dir))
        self.template_files = bool(template_files)

        # Build file list
        if flist:
            self.flist = flist
        else:
            self.flist = [str(p) for p in self.fits_dir.glob("*.[fi][t]s*")]

        # Only use working directory telescope.yml (wdir-specific configuration)
        self.telescope_file = self.wdir / "telescope.yml"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Load available filters database
        filters_path = Path(__file__).parent / "databases" / "filters.yml"
        loaded_filters = list(AutophotYaml(filters_path).load().get("W_eff", {}).keys())
        self.available_filters, dropped_filters = sanitize_photometric_filters(
            loaded_filters
        )
        if dropped_filters:
            self.logger.info(
                "Ignoring unsupported filter definitions: %s",
                ", ".join(sorted(set(dropped_filters))),
            )

        self.logger.info(f"Initialized: {len(self.flist)} files")
        # Global (cross-instrument) filter mapping cache. This prevents repetitive
        # prompts when scanning large heterogeneous datasets.
        self._global_filter_map: dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Filter mapping helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _norm_filter_token(token: str) -> str:
        """Normalize a FITS filter token for robust matching."""
        t = str(token).strip()
        t = t.replace(" ", "").replace("-", "_")
        return t.lower()

    def _resolve_standard_band(self, token: str) -> Optional[str]:
        """
        Heuristic mapping for common nonstandard filter tokens.

        Returns a standard band name (as present in self.available_filters) or None.
        """
        t = self._norm_filter_token(token)
        if not t:
            return None

        # Common survey-style suffixes (ZTF: gp/rp/ip; also generic *_p).
        if t in {"gp", "rp", "ip", "zp", "up", "wp"}:
            base = t[0]
            return base if base in self.available_filters else None

        # Tokens with underscores like gp_astrodon_2018, rp_cousins, etc.
        # Check first part for band indicators (exact case only)
        if "_" in t:
            first_part = t.split("_")[0]
            # Direct band match in first part (lowercase only - standard bands)
            if first_part in {"u", "g", "r", "i", "z", "y", "w", "b", "v", "j", "h", "k"}:
                return first_part if first_part in self.available_filters else None
            # Check for trailing p (e.g., gp -> g) - lowercase only
            if len(first_part) == 2 and first_part[1] == "p" and first_part[0] in "ugrizyw":
                return first_part[0] if first_part[0] in self.available_filters else None

        # Tokens like g782/r784/i705/z623 -> leading band letter (lowercase only).
        if len(t) >= 2 and t[0] in "ugrizy" and any(ch.isdigit() for ch in t[1:]):
            return t[0] if t[0] in self.available_filters else None

        # Things like V_HIGH, B_HIGH, I_BESS, Z_SPECIAL -> leading alpha run.
        # NOTE: Case-sensitive matching only - UBVRI vs ugriz are different filter systems
        m = re.match(r"^([a-zA-Z]+)", t)
        if m:
            lead = m.group(1)
            # Prefer exact case-sensitive match only
            for b in self.available_filters:
                if b == lead:
                    return b
            # Single-letter UBVRI style - exact case only
            if len(lead) == 1:
                for b in self.available_filters:
                    if b == lead:
                        return b
        return None

    def _get_global_filter_map(self, db: dict) -> dict:
        """Return the mutable global filter map stored in the telescope DB."""
        if not isinstance(db, dict):
            return {}
        gm = db.get("_global_filter_map", None)
        if not isinstance(gm, dict):
            db["_global_filter_map"] = {}
            gm = db["_global_filter_map"]
        return gm

    def _remember_mapping(self, db: dict, header_value: str, std_filter: str) -> None:
        """Store mapping in global cache (normalized + raw)."""
        gm = self._get_global_filter_map(db)
        raw = self._norm_filter_token(header_value)
        if raw:
            gm[raw] = str(std_filter)
        # Also store a simplified token (strip digits/underscores) to catch families.
        simp = re.sub(r"[^a-z]", "", raw)
        if simp:
            gm[simp] = str(std_filter)

    def ask_question(
        self,
        question,
        default_answer="n",
        expect_answer_type=str,
        options=None,
        ignore_word="skip",
    ):
        """
        Interactive user input with type validation and options.

        Loops until valid response. Supports defaults, skip, and option lists.

        Args:
            question (str): Prompt text
            default_answer (str): Default if Enter pressed
            expect_answer_type (type): Expected return type (str/float/bool)
            options (list): Valid choices list
            ignore_word (str): Special value to return immediately

        Returns:
            Validated answer in expected type
        """
        while True:
            # Format default nicely (e.g. 0.400 for floats, plain for strings)
            if isinstance(default_answer, (float, int)):
                default_str = f"{float(default_answer):.3f}"
            else:
                default_str = str(default_answer)

            lines = [
                "",
                f"{question}",
            ]
            if options:
                lines.append(f"  Options: {', '.join(options)}")
            lines.append(f"  (press Enter for default = {default_str})")
            lines.append("  > ")
            prompt = "\n".join(lines)

            raw = input(prompt)
            ans = raw.strip() or default_str
            print(f"  Selected: {ans}")

            if isinstance(ans, str) and ans == ignore_word:
                return ignore_word

            try:
                if isinstance(ans, str) and ans.lower() in ("true", "false"):
                    return ans.lower() == "true"
                # For numeric answers, cast via float; for str, just return.
                if expect_answer_type is str:
                    return str(ans)
                return expect_answer_type(float(ans))
            except Exception:
                if expect_answer_type == str:
                    return ans
                print("Invalid format. Enter one of the options.")

    def find_similar_keywords(self, keywords, search_term, cutoff=0.6):
        """
        Fuzzy keyword matching using difflib.get_close_matches.
        re missing for template images only

        Args:
            keywords (list): All available header keys
            search_term (str): Target keyword to match
            cutoff (float): Minimum similarity score (0.0-1.0)

        Returns:
            list: Sorted matching keywords
        """
        matches = get_close_matches(
            search_term.upper(), [k.upper() for k in keywords], n=10, cutoff=cutoff
        )
        return sorted(
            [keywords[[k.upper() for k in keywords].index(m)] for m in matches]
        )

    def ask_for_keyword(self, keyword, header, fname="", expected_units=None):
        """
        Non-interactive keyword selection with smart matching.

        Strategy:
        1. Exact alias matches (fastest)
        2. Fuzzy matching backup
        3. Auto-select first reasonable option

        Args:
            keyword (str): Target keyword name (ex: 'Date')
            header (dict): FITS header dictionary
            fname (str): Filename for context
            expected_units (str): Units hint for user

        Returns:
            str: Selected header keyword name
        """
        keys = list(header.keys())

        auto_k = auto_accept_header_key(header, keyword)
        if auto_k is not None:
            self.logger.info(
                "Using header key %r for %s (auto_accept)", auto_k, keyword
            )
            return auto_k

        # Step 1: Exact aliases (most reliable)
        aliases = KEYWORD_ALIASES.get(keyword, [])
        exact = list(dict.fromkeys([k for k in aliases if k in keys]))
        if len(exact) == 1:
            self.logger.info(f"Auto-selected exact match for {keyword}: {exact[0]}")
            return exact[0]

        # Step 2: Fuzzy matching
        fuzzy = self.find_similar_keywords(keys, keyword)
        candidates = list(set(exact + fuzzy))[:8]

        if candidates:
            # Auto-select first candidate
            selected = candidates[0]
            self.logger.info(f"Auto-selected keyword for {keyword}: {selected}")
            return selected

        # Fallback: search for keyword containing the target name
        for key in keys:
            if keyword.lower() in key.lower():
                self.logger.info(f"Fallback auto-selected keyword for {keyword}: {key}")
                return key

        # Last resort: return None
        self.logger.warning(f"Could not find suitable keyword for {keyword}")
        return None

    def check(self):
        """
        Main analysis pipeline.

        Three phases:
        1. Validate TELESCOP/INSTRUME presence -->classify files
        2. Setup telescope database entries interactively
        3. Extract filter keywords from all valid files

        Returns:
            list: Filenames with complete headers
        """
        self.logger.info(border_msg(f"Checking {len(self.flist)} files"))

        # PHASE 1: CLASSIFY FILES BY HEADER COMPLETENESS
        incorrect_files, correct_files = [], []
        tele_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        self.logger.info(border_msg("Basic headers"))
        for fname in tqdm(self.flist):
            header = get_header(fname)
            if not header:
                continue

            # Check required keys using aliases
            tele_key = next(
                (k for k in KEYWORD_ALIASES["TELESCOP"] if k in header), None
            )
            inst_key = next(
                (k for k in KEYWORD_ALIASES["INSTRUME"] if k in header), None
            )

            if tele_key and inst_key:
                tele = header[tele_key].strip()
                inst = header[inst_key].strip()
                correct_files.append(fname)
                # Always use 'INSTRUME' as the block key (not the raw header keyword like 'CAMERA')
                tele_dict[tele]["INSTRUME"][inst] = {}
            elif self.template_files:
                # Template images (e.g. from archives) may lack TELESCOP/INSTRUME; keep for calibration
                correct_files.append(fname)
            else:
                incorrect_files.append([fname, bool(tele_key), bool(inst_key)])

        # Report/save problematic files (skip when template_files: we already kept them)
        if incorrect_files and not self.template_files:
            outfile = self.fits_dir / "IncorrectFiles.txt"
            with open(outfile, "w") as f:
                for fname, has_t, has_i in incorrect_files:
                    missing = []
                    if not has_t:
                        missing.append("TELESCOP")
                    if not has_i:
                        missing.append("INSTRUME")
                    f.write(f"{fname}: missing {', '.join(missing)}\n")
            self.logger.info(
                "%d files with missing TELESCOP/INSTRUME headers -> %s",
                len(incorrect_files),
                outfile,
            )

        # PHASE 2: BUILD TELESCOPE DATABASE (Telescope-blind)
        db = self._load_db()

        self.logger.info(
            "Telescopes discovered in headers: %s", ", ".join(tele_dict.keys())
        )
        for tele, inst_groups in tqdm(tele_dict.items()):
            db.setdefault(tele, {})
            
            # Always use INSTRUME as the block key
            if "INSTRUME" not in db[tele]:
                db[tele]["INSTRUME"] = {}
            
            for inst_key, insts in inst_groups.items():
                for inst_name in insts:
                    entry = db[tele]["INSTRUME"].get(inst_name)
                    if not entry:  # New instrument - auto-setup with defaults
                        entry = {}
                        db[tele]["INSTRUME"][inst_name] = entry

                        # Auto-generate telescope/instrument label
                        entry["Name"] = f"{tele}+{inst_name}"
                        self.logger.info(f"Auto-created instrument entry: {entry['Name']}")

                        # Try to derive pixel scale from WCS
                        ps_default = 0.4
                        sample_file = None
                        for fname in correct_files:
                            h = get_header(fname)
                            tele_key_hdr = next(
                                (k for k in KEYWORD_ALIASES["TELESCOP"] if k in h), None
                            )
                            inst_key_hdr = next(
                                (k for k in KEYWORD_ALIASES["INSTRUME"] if k in h), None
                            )
                            if not tele_key_hdr or not inst_key_hdr:
                                continue
                            if (
                                h[tele_key_hdr].strip() == tele
                                and h[inst_key_hdr].strip() == inst_name
                            ):
                                sample_file = fname
                                break

                        if sample_file is not None:
                            try:
                                from functions import get_header as ap_get_header
                                from wcs import get_wcs as ap_get_wcs
                                import astropy.wcs as WCS_mod
                                import numpy as np_mod

                                hdr0 = ap_get_header(sample_file)
                                wcs_obj = ap_get_wcs(hdr0)
                                xy_scales = WCS_mod.utils.proj_plane_pixel_scales(wcs_obj)
                                if xy_scales is not None and len(xy_scales) > 0:
                                    cand = float(xy_scales[0]) * 3600.0
                                    if np_mod.isfinite(cand) and 0 < cand <= 5:
                                        ps_default = cand
                                        self.logger.info(f"Auto-derived pixel scale from WCS: {ps_default:.3f} arcsec/pix")
                            except Exception:
                                ps_default = 0.4

                        entry["pixel_scale"] = ps_default
                        entry["filter_key_0"] = "FILTER"

                        # Skip optional keyword extraction (non-interactive)
                        self.logger.info("Skipping optional keyword extraction (non-interactive mode)")

        # PHASE 3: FILTER KEYWORDS FOR ALL VALID FILES (skip files without TELESCOP/INSTRUME, e.g. templates)
        self.logger.info(border_msg(f"Filters ({len(correct_files)} files)"))
        for fname in tqdm(correct_files):
            header = get_header(fname)
            tele_key = next(
                (k for k in KEYWORD_ALIASES["TELESCOP"] if k in header), None
            )
            inst_key = next(
                (k for k in KEYWORD_ALIASES["INSTRUME"] if k in header), None
            )
            if not tele_key or not inst_key:
                continue
            tele, inst = header[tele_key].strip(), header[inst_key].strip()
            
            # Telescope-blind: create or get entry for any telescope/instrument combination
            # No telescope.yml entry required - works generically for any telescope
            if tele not in db:
                db[tele] = {}
            if "INSTRUME" not in db[tele]:
                db[tele]["INSTRUME"] = {}
            if inst not in db[tele]["INSTRUME"]:
                # Auto-create entry with generic defaults
                db[tele]["INSTRUME"][inst] = {
                    "Name": f"{tele}+{inst}",
                    "filter_key_0": "FILTER",
                }
                self.logger.info(
                    f"Auto-created telescope entry for {tele}+{inst} with generic defaults."
                )
            
            entry = db[tele]["INSTRUME"][inst]
            fkey = self._find_filter_key(header, entry)
            # Preserve original case for proper filter system distinction (e.g., r vs R)
            fval_raw = str(header[fkey]).strip().replace(" ", "")
            fval = fval_raw.lower()  # For AVOID_FILTERS check
            if fval and fval not in AVOID_FILTERS and fval_raw not in entry:
                if fval_raw in self.available_filters:
                    entry[fval_raw] = fval_raw
                else:
                    # Non-interactive: use heuristic to map filter
                    std_filter = self._resolve_standard_band(fval_raw)
                    if std_filter is not None:
                        entry[fval_raw] = std_filter
                        self.logger.info(f"Auto-mapped filter {fval_raw} -> {std_filter}")
                    else:
                        # Fallback: use the raw value as-is
                        entry[fval_raw] = fval_raw
                        self.logger.warning(f"Could not auto-map filter {fval_raw}, using as-is")

        # SAVE DATABASE
        self._save_db(db)
        self.logger.info(
            "Header check complete: %d/%d files OK.",
            len(correct_files),
            len(self.flist),
        )
        return correct_files

    def _setup_telescope(self, tele_entry, tele_name):
        """
        Telescope setup hook.

        Location information is no longer used, so this is a no-op that simply
        preserves any existing values in the database without prompting.
        """
        return

    def _ask_filter_mapping(self, header_value):
        """
        Ask user to map a FITS header filter value to a standard band name.
        Uses a clear, multi-line prompt and validates against catalog bands.
        """
        opts = sorted(
            self.available_filters, key=lambda x: (x not in "ugriz", x.lower())
        )
        opts_str = ", ".join(opts)
        default = "no_filter"
        prompt = (
            "\n"
            "  Filter name mapping\n"
            "  ------------------\n"
            f"  Header value from FITS:  {header_value!r}\n"
            "  Not in catalog; choose a standard band name for photometry,\n"
            f"  or enter 'no_filter' to skip.\n"
            f"\n  Standard bands:  {opts_str}\n"
            f"\n  Map {header_value!r} to [{default}]: "
        )
        # First, check global cache and heuristics to avoid repeated prompting.
        try:
            db = self._load_db()
        except Exception:
            db = {}
        gm = self._get_global_filter_map(db)
        hv_norm = self._norm_filter_token(header_value)
        # Use case-sensitive lookup for exact match
        if hv_norm in gm:
            mapped = str(gm[hv_norm]).strip()
            # If mapping is identical (case-sensitive), accept without confirmation
            if mapped == header_value:
                self.logger.info("Auto-mapped filter %r -> %r (global cache; identical).", header_value, mapped)
                return mapped
            # Different mapping - ask for confirmation
            self.logger.info("Global cache suggests: %r -> %r", header_value, mapped)
            confirm = self.ask_question(
                f"Confirm cached mapping '{header_value}' -> '{mapped}'?",
                default_answer="y",
                expect_answer_type=str,
                options=["y", "n"],
            )
            if confirm.lower() in ("y", "yes", ""):
                self.logger.info("Confirmed cached: %r -> %r", header_value, mapped)
                return mapped
            else:
                self.logger.info("Rejected cached mapping for %r", header_value)
                # Continue to heuristic or manual prompt
        simp = re.sub(r"[^a-z]", "", hv_norm)
        if simp in gm:
            mapped = str(gm[simp]).strip()
            # If mapping is identical (case-sensitive), accept without confirmation
            if mapped == header_value:
                self.logger.info(
                    "Auto-mapped filter %r -> %r (global cache; simplified token=%r; identical).",
                    header_value,
                    mapped,
                    simp,
                )
                return mapped
            # Different mapping - ask for confirmation
            self.logger.info("Global cache suggests: %r -> %r (simplified)", header_value, mapped)
            confirm = self.ask_question(
                f"Confirm cached mapping '{header_value}' -> '{mapped}'?",
                default_answer="y",
                expect_answer_type=str,
                options=["y", "n"],
            )
            if confirm.lower() in ("y", "yes", ""):
                self.logger.info("Confirmed cached: %r -> %r", header_value, mapped)
                return mapped
            else:
                self.logger.info("Rejected cached mapping for %r", header_value)
                # Continue to heuristic or manual prompt
        guessed = self._resolve_standard_band(header_value)
        if guessed is not None:
            # If mapping is identical (case-sensitive), accept without confirmation
            if guessed == header_value:
                self.logger.info("Auto-mapped filter %r -> %r (identical, no confirmation needed).", header_value, guessed)
                try:
                    self._remember_mapping(db, header_value, guessed)
                    self._save_db(db)
                except Exception:
                    pass
                return guessed
            
            # Mapping is different - ask for confirmation
            self.logger.info("Heuristic suggests: %r -> %r", header_value, guessed)
            confirm = self.ask_question(
                f"Confirm mapping '{header_value}' -> '{guessed}'?",
                default_answer="y",
                expect_answer_type=str,
                options=["y", "n"],
            )
            if confirm.lower() in ("y", "yes", ""):
                self.logger.info("Confirmed: %r -> %r", header_value, guessed)
                try:
                    self._remember_mapping(db, header_value, guessed)
                    self._save_db(db)
                except Exception:
                    pass
                return guessed
            else:
                self.logger.info("Rejected heuristic mapping for %r", header_value)
                # Fall through to manual prompt below

        while True:
            ans = input(prompt).strip().lower() or default
            print(f"  Selected: {ans}")
            if ans == default:
                try:
                    self._remember_mapping(db, header_value, default)
                    self._save_db(db)
                except Exception:
                    pass
                return default
            if ans in self.available_filters:
                try:
                    self._remember_mapping(db, header_value, ans)
                    self._save_db(db)
                except Exception:
                    pass
                return ans
            # Allow exact match ignoring case
            for b in self.available_filters:
                if b.lower() == ans:
                    try:
                        self._remember_mapping(db, header_value, b)
                        self._save_db(db)
                    except Exception:
                        pass
                    return b
            print(f"  Invalid. Choose one of {opts_str}, or '{default}'.")

    def _find_filter_key(self, header, entry):
        """
        Select best filter keyword from header.

        Priority: FILTER -->existing filter_key_N -->auto-search
        Skips 'open'/'clear' filters.
        Dynamically adds new filter_key_{N+1} entries.
        Non-interactive: auto-selects first reasonable filter key.

        Args:
            header (dict): FITS header
            entry (dict): Instrument database entry

        Returns:
            str: Selected filter keyword name
        """
        fkeys = ["FILTER"] + [k for k in entry if k.startswith("filter_key_")]
        for fk in fkeys:
            if fk in header:
                fval = str(header[fk]).strip().lower().replace(" ", "")
                if fval not in AVOID_FILTERS:
                    return fk

        # Auto-search for filter keyword (non-interactive)
        # Look for keywords containing "filter" or "FPA.FILTER"
        for key in header.keys():
            key_lower = key.lower()
            if "filter" in key_lower and key not in fkeys:
                fval = str(header[key]).strip().lower().replace(" ", "")
                if fval not in AVOID_FILTERS:
                    # Register this new filter key
                    next_n = (
                        max(
                            [
                                int(re.search(r"filter_key_(\d+)", k).group(1))
                                for k in entry
                                if "filter_key_" in k
                            ]
                            + [0]
                        )
                        + 1
                    )
                    new_fk = f"filter_key_{next_n}"
                    entry[new_fk] = key
                    self.logger.info(f"Auto-selected filter key: {key} -> {new_fk}")
                    return key

        # Fallback: use first key that looks like a filter
        for key in header.keys():
            fval = str(header[key]).strip().lower().replace(" ", "")
            if fval and fval not in AVOID_FILTERS and len(fval) <= 10:
                # Register this new filter key
                next_n = (
                    max(
                        [
                            int(re.search(r"filter_key_(\d+)", k).group(1))
                            for k in entry
                            if "filter_key_" in k
                        ]
                        + [0]
                    )
                    + 1
                )
                new_fk = f"filter_key_{next_n}"
                entry[new_fk] = key
                self.logger.info(f"Fallback auto-selected filter key: {key} -> {new_fk}")
                return key

        # Last resort: use FILTER as default
        self.logger.warning("No suitable filter key found, using 'FILTER' as default")
        return "FILTER"

    def _load_db(self):
        """Load existing telescope.yml from wdir or return empty dict.
        
        Auto-converts legacy CAMERA blocks to INSTRUME.
        """
        db = {}
        if self.telescope_file.exists():
            with open(self.telescope_file) as f:
                db = yaml.safe_load(f) or {}
        
        # Auto-convert legacy CAMERA blocks to INSTRUME
        for tele, tele_data in db.items():
            if isinstance(tele_data, dict) and "CAMERA" in tele_data:
                if "INSTRUME" not in tele_data:
                    self.logger.warning(
                        f"Converting legacy CAMERA block to INSTRUME for telescope {tele}"
                    )
                    tele_data["INSTRUME"] = tele_data.pop("CAMERA")
                else:
                    # Both exist - merge CAMERA into INSTRUME
                    self.logger.warning(
                        f"Merging legacy CAMERA block into INSTRUME for telescope {tele}"
                    )
                    camera_data = tele_data.pop("CAMERA")
                    for inst, inst_data in camera_data.items():
                        if inst not in tele_data["INSTRUME"]:
                            tele_data["INSTRUME"][inst] = inst_data
        
        return db

    def _save_db(self, db):
        """
        Save telescope database as YAML to wdir.

        Uses safe_dump with readable formatting.
        """
        os.makedirs(self.telescope_file.parent, exist_ok=True)
        with open(self.telescope_file, "w") as f:
            yaml.safe_dump(db, f, default_flow_style=False, sort_keys=False)


# -----------------------------------------------------------------------------
# SUPPORT CLASSES
# -----------------------------------------------------------------------------


class AutophotYaml:
    """Simple YAML loader for filters database."""

    def __init__(self, path):
        self.path = path

    def load(self):
        try:
            with open(self.path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {"W_eff": {}}


# -----------------------------------------------------------------------------
# USAGE EXAMPLE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    input_yaml = {"wdir": "./working_dir/", "fits_dir": "./reduced_fits/"}
    flist = None  # Auto-discover

    fi = FitsInfo(input_yaml, flist)
    correct_files = fi.check()
