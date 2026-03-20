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


# Common FITS keyword aliases compiled from astronomy standards
KEYWORD_ALIASES = {
    "TELESCOP": ["TELESCOPE", "TELNAME", "OBSERVAT", "OBSERVATORY", "TELESCOP"],
    # Instrument must always be carried in the INSTRUME keyword (no DETECTOR alias).
    "INSTRUME": ["INSTRUMENT", "CAMERA", "INSTRUMEN", "INSTRUME"],
    "FILTER": ["FILTERS", "FLTRNAM", "FILNAM1", "FILTNAM1", "FILTER1", "FILTER"],
    "Date": ["DATE-OBS", "DATEOBS", "UTC-OBS", "OBS-DATE", "DATE", "UTSTART"],
    "EXPTIME": ["EXPOSURE", "TEXP", "EXPTIME", "EXPTIME0"],
    "GAIN": ["GAIN", "EGAIN", "READGAIN", "GAIN0"],
    "RDNOISE": ["RDNOISE", "RNOISE", "READNOIS", "RDNOISE0"],
    "SATURATE": ["SATURATE", "SATLEVEL", "FULLWELL", "SATURATE0"],
    "AIRMASS": ["AIRMASS", "AMASS", "AIRMASS0"],
    "MJD": ["MJD-OBS", "MJD", "MJDSTART", "MJD-OBS"],
}

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


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
        self.fits_dir = Path(input_yaml.get("fits_dir", "."))
        self.template_files = bool(template_files)

        # Build file list
        if flist:
            self.flist = flist
        else:
            self.flist = [str(p) for p in self.fits_dir.glob("*.[fi][t]s*")]

        self.telescope_file = self.wdir / "telescope.yml"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Load available filters database
        filters_path = Path(__file__).parent / "databases" / "filters.yml"
        self.available_filters = list(
            AutophotYaml(filters_path).load().get("W_eff", {}).keys()
        )

        self.logger.info(f"Initialized: {len(self.flist)} files")

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
        Interactive keyword selection with smart matching.

        Strategy:
        1. Exact alias matches (fastest)
        2. Fuzzy matching backup
        3. Manual entry fallback

        Args:
            keyword (str): Target keyword name (ex: 'Date')
            header (dict): FITS header dictionary
            fname (str): Filename for context
            expected_units (str): Units hint for user

        Returns:
            str: Selected header keyword name
        """
        keys = list(header.keys())
        print(f"\nKeyword '{keyword}' (file: {fname})")

        # Step 1: Exact aliases (most reliable)
        aliases = KEYWORD_ALIASES.get(keyword, [])
        exact = [k for k in aliases if k in keys]
        if exact:
            print(f"Exact match: {', '.join(exact[:3])}")
        if len(exact) == 1:
            print(f"Using: {exact[0]}")
            return exact[0]

        # Step 2: Fuzzy matching
        fuzzy = self.find_similar_keywords(keys, keyword)
        candidates = list(set(exact + fuzzy))[:8]

        if candidates:
            print("  # | KEY            | VALUE")
            print("  " + "-" * 33)
            kw_dict = {i + 1: k for i, k in enumerate(candidates)}
            for i, k in kw_dict.items():
                header_value = (
                    str(header[k])[:20] + "..."
                    if len(str(header[k])) > 20
                    else str(header[k])
                )
                print(f"  [{i}] {k:<15} | {header_value}")

            sel = self.ask_question(f"Select {keyword}", "1", float, ignore_word="skip")
            if sel != "skip" and sel in kw_dict:
                return kw_dict[sel]

        # Step 3: Manual entry
        print(f"  Aliases: {', '.join(aliases[:4])}")
        key = self.ask_question(f"Enter '{keyword}' key", "ignored", str)
        return "not_given_by_user" if key == "ignored" else key

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
                tele_dict[tele][inst_key][inst] = {}
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

        # PHASE 2: BUILD TELESCOPE DATABASE
        db = self._load_db()

        self.logger.info(
            "Telescopes discovered in headers: %s", ", ".join(tele_dict.keys())
        )
        for tele, inst_groups in tqdm(tele_dict.items()):
            db.setdefault(tele, {})
            self._setup_telescope(db[tele], tele)

            for inst_key, insts in inst_groups.items():
                for inst_name in insts:
                    entry = db[tele].setdefault(inst_key, {}).setdefault(inst_name, {})
                    if not entry:  # New instrument
                        entry["Name"] = self.ask_question(
                            f"Label '{tele}+{inst_name}'", f"{tele}+{inst_name}"
                        )
                        # Try to derive a default pixel scale from WCS of a file
                        # that actually belongs to this (tele, inst_name) pair.
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
                                xy_scales = WCS_mod.utils.proj_plane_pixel_scales(
                                    wcs_obj
                                )
                                if xy_scales is not None and len(xy_scales) > 0:
                                    cand = float(xy_scales[0]) * 3600.0
                                    if np_mod.isfinite(cand) and 0 < cand <= 5:
                                        ps_default = cand
                            except Exception:
                                ps_default = 0.4
                        ps = self.ask_question(
                            "Pixel scale (arcsec/pix)",
                            ps_default,
                            float,
                            ignore_word="skip",
                        )
                        entry["pixel_scale"] = None if ps == "skip" else float(ps)
                        entry["filter_key_0"] = "FILTER"

                    # Extract optional keywords using first valid file
                    if correct_files:
                        h = get_header(correct_files[0])
                        for opt, info in OPTIONAL_KEYWORDS.items():
                            label = info["label"]
                            if label not in entry:
                                entry[label] = self.ask_for_keyword(
                                    opt,
                                    h,
                                    os.path.basename(correct_files[0]),
                                    info["units"],
                                )

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
            if (
                tele not in db
                or inst_key not in db[tele]
                or inst not in db[tele][inst_key]
            ):
                continue
            entry = db[tele][inst_key][inst]
            fkey = self._find_filter_key(header, entry)
            fval = str(header[fkey]).strip().lower().replace(" ", "")
            if fval and fval not in AVOID_FILTERS and fval not in entry:
                if fval in self.available_filters:
                    entry[fval] = fval
                else:
                    std_filter = self._ask_filter_mapping(fval)
                    entry[fval] = std_filter

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
        while True:
            ans = input(prompt).strip().lower() or default
            print(f"  Selected: {ans}")
            if ans == default:
                return default
            if ans in self.available_filters:
                return ans
            # Allow exact match ignoring case
            for b in self.available_filters:
                if b.lower() == ans:
                    return b
            print(f"  Invalid. Choose one of {opts_str}, or '{default}'.")

    def _find_filter_key(self, header, entry):
        """
        Select best filter keyword from header.

        Priority: FILTER -->existing filter_key_N -->new search
        Skips 'open'/'clear' filters.
        Dynamically adds new filter_key_{N+1} entries.

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

        # Search and register new filter key
        new_key = self.ask_for_keyword("filter", header)
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
        entry[new_fk] = new_key
        return new_key

    def _load_db(self):
        """Load existing telescope.yml or return empty dict."""
        if self.telescope_file.exists():
            with open(self.telescope_file) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_db(self, db):
        """
        Save telescope database as YAML.

        Ensures working directory exists.
        Uses safe_dump with readable formatting.
        """
        os.makedirs(self.wdir, exist_ok=True)
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
