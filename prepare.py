#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized and human-readable version of the `prepare` class for AutoPHOT.
This class handles:
- Loading and validating configuration
- Cleaning and filtering image files
- Checking catalogs, TNS, and filters
- Finding and validating template files

All function names and core logic are preserved.
"""

import os
import sys
import pathlib
import logging
import glob
import shutil
import numpy as np
import pandas as pd
import yaml
from functools import reduce
from typing import List, Dict, Optional, Tuple, Union

import astropy.wcs as WCS

# Project-specific helpers (assumed to be in your codebase)
from functions import (
    AutophotYaml,
    border_msg,
    get_header,
    get_instrument_config,
    load_telescope_config,
    ColoredLevelFormatter,
    print_progress_bar,
    normalize_photometric_filter_name,
    sanitize_photometric_filters,
    parse_supported_filter_group_key,
    log_warning_from_exception,
)
from check import FitsInfo
from tns import get_coords
from wcs import get_wcs


class Prepare:
    """
    Prepares and validates input data for the AutoPHOT pipeline.
    Handles configuration, file cleaning, catalog checks, TNS lookups, filter validation, and template discovery.
    """

    def __init__(self, default_input: Dict):
        """
        Initialize the prepare class with the input configuration.

        Args:
            default_input (Dict): Configuration dictionary for the AutoPHOT pipeline.
        """
        self.input_yaml = default_input
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        # Ensure console messages have unique color highlights.
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.setFormatter(
                ColoredLevelFormatter(
                    fmt="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S",
                    use_color=True,
                )
            )
        self.logger = logging.getLogger(__name__)

    # --- Load Configuration ---
    @staticmethod
    def load() -> Dict:
        """
        Loads the default input YAML configuration file.

        Returns:
            Dict: Parsed YAML configuration.
        """
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the default input YAML
        default_input_path = os.path.join(script_dir, "databases", "default_input.yml")
        # Load and return the YAML configuration
        default_input_yaml = AutophotYaml(default_input_path, "default_input").load()
        logging.info(
            border_msg(
                f"Default input loaded from: {default_input_path}", body="-", corner="+"
            )
        )
        return default_input_yaml

    def clean(self) -> List[str]:
        """
        Prepares a list of valid FITS files for processing, excluding templates,
        subtractions, and invalid files. Creates the output directory structure
        if it does not exist.

        Restart behaviour
        -----------------
        This respects the ``restart`` flag in ``input_yaml`` (default True):

        - ``restart = True`` (default) - include all valid FITS files and
          reprocess them (redo even if ``output.csv`` already exists).
        - ``restart = False`` - skip files that already have an ``output.csv``
          in the corresponding ``_REDUCED`` tree (only process new/unprocessed files).

        Returns
        -------
        List[str]
            List of valid FITS file paths to be processed.
        """
        files_removed = 0
        total_candidates = 0
        valid_files = []
        # Normalize the FITS directory path
        self.input_yaml["fits_dir"] = self.input_yaml["fits_dir"].rstrip("/")
        # Set up the root output directory, mirroring main.py's layout:
        # new_output_dir = os.path.join(os.path.dirname(fits_dir),
        #                               os.path.basename(fits_dir) + f"_{outdir_name}")
        fits_dir = self.input_yaml["fits_dir"]
        new_dir = f"_{self.input_yaml['outdir_name']}"
        base_dir = os.path.basename(fits_dir).replace(new_dir, "")
        work_loc = f"{base_dir}{new_dir}"
        work_root = os.path.join(os.path.dirname(fits_dir), work_loc)
        pathlib.Path(work_root).mkdir(parents=True, exist_ok=True)
        # Define forbidden substrings and valid extensions
        forbidden_substrings = [
            "subtraction",
            "template",
            ".wcs",
            "PSF_model",
            "footprint",
            "sources_",
        ]
        valid_extensions = (".fits", ".fit", ".fts", "fits.fz")

        def _is_valid_file(filename: str, root: str) -> bool:
            """
            Checks if a file is valid for processing.
            Args:
                filename (str): The filename to check.
                root (str): The root directory of the file.
            Returns:
                bool: True if the file is valid, False otherwise.
            """
            return (
                filename.endswith(valid_extensions)
                and not any(substring in filename for substring in forbidden_substrings)
                and not any(substring in root for substring in forbidden_substrings)
            )

        # Determine the root directory to search (templates or main FITS directory).
        # `template_subtraction.prepare_templates` was removed from the public config;
        # keep the behavior behind a safe lookup so older configs won't crash.
        ts_cfg = self.input_yaml.get("template_subtraction") or {}
        prepare_templates = bool(ts_cfg.get("prepare_templates", False))
        search_root = (
            os.path.join(self.input_yaml["fits_dir"], "templates")
            if prepare_templates
            else self.input_yaml["fits_dir"]
        )

        # Populate the list of valid files
        for root, _, files in os.walk(search_root):
            for filename in files:
                if _is_valid_file(filename, root):
                    filepath = os.path.join(root, filename)
                    total_candidates += 1
                    # Construct the expected output directory for this file.
                    # IMPORTANT: mirror main.py's directory logic so that restart
                    # checks are consistent and we don't re-run files that already
                    # have OUTPUT_{base}.csv.
                    base = (
                        os.path.splitext(filename)[0]
                        .replace(" ", "_")
                        .replace(".", "_")
                        .replace("_APT", "")
                        .replace("_ERROR", "")
                    )

                    if prepare_templates:
                        # Template mode: outputs live directly under work_root/base
                        cur_dir = os.path.join(work_root, base)
                    else:
                        # Science mode: reproduce the nested subdirectory layout
                        wdir = self.input_yaml["fits_dir"]
                        root_dir = os.path.dirname(filepath)
                        rel = root_dir.replace(wdir, "")
                        sub_dirs = [
                            part.replace("_APT", "").replace(" ", "_")
                            for part in rel.split("/")
                            if part
                        ]
                        cur_dir = work_root
                        for sub in sub_dirs:
                            cur_dir = os.path.join(cur_dir, f"{sub}_APT")
                        cur_dir = os.path.join(cur_dir, base)

                    if prepare_templates:
                        # Template preparation doesn't always write a per-image output
                        # CSV; use the generated template catalog as the completion
                        # marker.
                        output_csv_path = os.path.join(
                            cur_dir, f"imageCalib_template_{base}.csv"
                        )
                    else:
                        output_csv_path = os.path.join(
                            cur_dir, f"OUTPUT_{base}.csv"
                        )

                    # Honour the restart flag (default True = redo all):
                    # - restart=True  -> include file (reprocess even if output exists)
                    # - restart=False -> skip files that already have OUTPUT_{base}.csv
                    if os.path.exists(output_csv_path) and not self.input_yaml.get(
                        "restart", True
                    ):
                        files_removed += 1
                        continue

                    valid_files.append(filepath)
        restart = self.input_yaml.get("restart", True)
        self.logger.info(
            "Restart = %s -> %s",
            restart,
            (
                "reprocess all files (ignore existing OUTPUT_{base}.csv)"
                if restart
                else "skip files that already have OUTPUT_{base}.csv (only process new/unprocessed)"
            ),
        )
        self.logger.info(
            "Scanned %d FITS file(s): %d completed (OUTPUT present), %d pending.",
            total_candidates,
            files_removed,
            len(valid_files),
        )

        return valid_files

    # --- Check Files ---
    def check_files(self, flist: List[str], template_files: bool = False) -> List[str]:
        """
        Validates the list of FITS files using the `FitsInfo` function.

        Args:
            flist (List[str]): List of file paths to validate.
            template_files (bool): If True, do not reject files missing TELESCOP/INSTRUME
                (for template images that may lack those headers).

        Returns:
            List[str]: List of files that meet the validation criteria.
        """
        return FitsInfo(
            input_yaml=self.input_yaml, flist=flist, template_files=template_files
        ).check()

    # --- Check Catalog ---
    def check_catalog(self) -> List[str]:
        """
        Determines the available filters from the catalog, including custom catalogs and IR sequence data.

        Returns:
            List[str]: List of available filter names.
        """
        selected_catalog = self.input_yaml["catalog"]["use_catalog"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        catalog_yml_path = os.path.join(script_dir, "databases", "catalog.yml")

        def _resolve_catalog_for_filter(catalog_choice, image_filter=None):
            if not isinstance(catalog_choice, dict):
                return catalog_choice
            use_filter = str(image_filter or "").strip()
            use_filter_norm = normalize_photometric_filter_name(use_filter)
            if use_filter:
                membership_matches = []
                for k, v in catalog_choice.items():
                    if v is None:
                        continue
                    key_s = str(k).strip()
                    key_l = key_s.lower()
                    if key_l in {"default", "*", "all"}:
                        continue
                    key_bands = parse_supported_filter_group_key(key_s)
                    if key_bands and use_filter in key_bands:
                        membership_matches.append(str(k))
                if len(membership_matches) > 1:
                    self.logger.warning(
                        "catalog.use_catalog mapping is ambiguous for filter '%s': matched keys=%s. "
                        "Using precedence: exact key > first membership key > default.",
                        image_filter,
                        membership_matches,
                    )
            if use_filter:
                for k, v in catalog_choice.items():
                    key_s = str(k).strip()
                    if v is None or key_s.lower() in {"default", "*", "all"}:
                        continue
                    key_norm = normalize_photometric_filter_name(key_s)
                    if key_s == use_filter and key_norm is not None:
                        return v
                # Backward-compatible fallback: normalized exact match.
                for k, v in catalog_choice.items():
                    key_s = str(k).strip()
                    if v is None or key_s.lower() in {"default", "*", "all"}:
                        continue
                    key_norm = normalize_photometric_filter_name(key_s)
                    if (
                        key_norm is not None
                        and use_filter_norm is not None
                        and key_norm == use_filter_norm
                    ):
                        return v
                for k, v in catalog_choice.items():
                    key_s = str(k).strip()
                    key_l = key_s.lower()
                    if key_l in {"default", "*", "all"}:
                        continue
                    key_bands = parse_supported_filter_group_key(key_s)
                    if v is not None and key_bands and use_filter in key_bands:
                        return v
            for dkey in ("default", "*", "all"):
                if dkey in catalog_choice and catalog_choice[dkey] is not None:
                    return catalog_choice[dkey]
            for v in catalog_choice.values():
                if v is not None:
                    return v
            return None

        selected_catalog_resolved = _resolve_catalog_for_filter(
            selected_catalog, self.input_yaml.get("imageFilter")
        )
        catalog_input = AutophotYaml(catalog_yml_path, selected_catalog_resolved).load()

        if self.input_yaml["catalog"].get("build_catalog", False):
            available_filters = list(catalog_input.keys())
        elif selected_catalog_resolved == "custom":
            target = self.input_yaml["target_name"]
            fname = (
                f"{target}_RAD_{float(self.input_yaml['catalog']['catalog_radius'])}"
            )
            if not self.input_yaml["catalog"]["catalog_custom_fpath"]:
                self.logger.error(
                    "Custom catalog selected but 'catalog_custom_fpath' is not defined in the configuration."
                )
                sys.exit(1)
            else:
                fname = self.input_yaml["catalog"]["catalog_custom_fpath"]
            try:
                custom_table = pd.read_csv(fname)
            except pd.errors.EmptyDataError:
                self.logger.error(
                    "Custom catalog CSV is empty or has no parseable header. "
                    "Remove the file to force a rebuild, or fix the path: %s",
                    fname,
                )
                sys.exit(1)
            if custom_table.empty:
                self.logger.error(
                    "Custom catalog contains no sources (0 data rows): %s",
                    fname,
                )
                sys.exit(1)
            # Dynamic filter discovery: accept any filter that exists in custom catalog
            # This enables completely arbitrary filter names
            available_filters = []
            
            # First, add standard filters from catalog_input
            for f in catalog_input.keys():
                if f in custom_table.columns:
                    available_filters.append(f)
            
            # Then, discover additional filters from custom catalog columns
            # Look for columns that might be photometric (have corresponding _err columns)
            for col in custom_table.columns:
                col_str = str(col).strip()
                # Skip non-photometric columns
                if col_str in ['ra', 'dec', 'RA', 'DEC', 'name', 'objname', 'id', 'ID']:
                    continue
                # Skip error columns
                if col_str.endswith('_err') or col_str.endswith('err'):
                    continue
                # Skip if already added
                if col_str in available_filters:
                    continue
                # Check if this looks like a photometric filter (has corresponding error column)
                err_col_candidates = [f"{col_str}_err", f"{col_str}err", f"{col_str}_ERROR", f"{col_str}ERROR"]
                has_error_col = any(err_col in custom_table.columns for err_col in err_col_candidates)
                
                if has_error_col:
                    available_filters.append(col_str)
                    self.logger.info(
                        "Discovered dynamic filter '%s' from custom catalog (has error column)",
                        col_str
                    )
                else:
                    # Even without error column, include if it has numeric data
                    try:
                        # Check if column contains numeric data
                        numeric_data = pd.to_numeric(custom_table[col_str], errors='coerce')
                        if numeric_data.notna().sum() > 0:  # Has some valid numeric data
                            available_filters.append(col_str)
                            self.logger.info(
                                "Discovered dynamic filter '%s' from custom catalog (numeric data)",
                                col_str
                            )
                    except Exception:
                        pass
        else:
            if isinstance(selected_catalog, dict):
                available_filters = []
                seen_filters = set()
                cat_names = []
                for v in selected_catalog.values():
                    if v is None:
                        continue
                    vv = str(v).strip()
                    if vv and vv not in cat_names:
                        cat_names.append(vv)
                for cat_name in cat_names:
                    try:
                        c_input = AutophotYaml(catalog_yml_path, cat_name).load()
                        for f in c_input.keys():
                            if f not in seen_filters:
                                seen_filters.add(f)
                                available_filters.append(f)
                    except Exception:
                        continue
            else:
                available_filters = list(catalog_input.keys())

        # Include IR sequence data if specified
        if self.input_yaml["catalog"].get("include_IR_sequence_data", False):
            available_filters += ["J", "H", "K"]
        available_filters, dropped_filters = sanitize_photometric_filters(
            available_filters, available_filters=available_filters
        )
        if dropped_filters:
            self.logger.info(
                "Ignoring unsupported/non-photometric catalog keys: %s",
                ", ".join(sorted(set(dropped_filters))),
            )
        return available_filters

    # --- Check TNS ---
    def check_tns(self) -> Dict:
        """
        Retrieves or caches transient object information from the TNS API.
        Falls back to user-provided coordinates if TNS is unavailable.

        Returns:
            Dict: Dictionary containing RA, DEC, and object information.
        """
        # If the user already provided coordinates, do NOT query TNS.
        # Treat target_ra/target_dec as authoritative (degrees, FK5/J2000).
        if (
            self.input_yaml.get("target_ra") is not None
            and self.input_yaml.get("target_dec") is not None
        ):
            self.logger.info(
                "Skipping TNS query because target_ra/target_dec are already provided (%.6f, %.6f).",
                float(self.input_yaml["target_ra"]),
                float(self.input_yaml["target_dec"]),
            )
            return {
                "ra": float(self.input_yaml["target_ra"]),
                "dec": float(self.input_yaml["target_dec"]),
            }

        target_name = self.input_yaml.get("target_name", None)
        tns_dir = pathlib.Path(self.input_yaml["wdir"]) / "tns_objects"
        tns_dir.mkdir(parents=True, exist_ok=True)
        if target_name is None or (isinstance(target_name, str) and target_name.strip() == ""):
            # No name and no coordinates; caller decides whether to continue.
            return {}

        transient_path = tns_dir / f"{target_name}.yml"

        def _log_tns_summary(tns_data: Dict, source: str) -> None:
            """Log a concise TNS summary each time coordinates are used."""
            try:
                objname = tns_data.get("objname", target_name)
                prefix = str(tns_data.get("name_prefix", "") or "").strip()
                display_name = f"{prefix}{objname}" if prefix else str(objname)
                ra_val = tns_data.get("radeg", tns_data.get("ra"))
                dec_val = tns_data.get("decdeg", tns_data.get("dec"))
                obj_type = tns_data.get("type", tns_data.get("object_type", "unknown"))
                if ra_val is not None and dec_val is not None:
                    self.logger.info(
                        "TNS info (%s): %s  RA=%.6f  Dec=%.6f  Type=%s",
                        source,
                        display_name,
                        float(ra_val),
                        float(dec_val),
                        obj_type,
                    )
                else:
                    self.logger.info(
                        "TNS info (%s): %s  Type=%s",
                        source,
                        display_name,
                        obj_type,
                    )
            except Exception:
                self.logger.info("TNS info (%s): %s", source, str(target_name))

        # Early return for user-provided coordinates
        if target_name is None:
            if (
                self.input_yaml["target_ra"] is not None
                and self.input_yaml["target_dec"] is not None
            ):
                return {
                    "ra": self.input_yaml["target_ra"],
                    "dec": self.input_yaml["target_dec"],
                }
            response = (
                input(
                    "\nNo target_name and no RA/Dec provided in the configuration.\n"
                    "Continue without target information? [y/N]: "
                )
                .strip()
                .lower()
            )
            if response not in ("y", "yes"):
                raise Exception(
                    "No target information provided and user declined to continue."
                )
            return {}

        # Check for cached TNS data
        if transient_path.is_file():
            tns_response = AutophotYaml(str(transient_path), target_name).load()
            self.logger.info(
                "Using cached TNS information for %s.", tns_response["objname"]
            )
            _log_tns_summary(tns_response, source="cache")
            return tns_response

        # Fetch new TNS data
        tns_bot_id = self.input_yaml["wcs"].get("TNS_BOT_ID")
        if tns_bot_id is None or (
            isinstance(tns_bot_id, str) and tns_bot_id.strip() == ""
        ):
            self.logger.warning(
                "No TNS Bot ID configured for target '%s'. Falling back to manual coordinates.",
                target_name,
            )
            if (
                self.input_yaml["target_ra"] is not None
                and self.input_yaml["target_dec"] is not None
            ):
                return {
                    "ra": self.input_yaml["target_ra"],
                    "dec": self.input_yaml["target_dec"],
                }
            raise Exception("No TNS Bot ID and no manual RA/DEC coordinates provided.")

        try:
            self.logger.info("Querying TNS for target '%s'.", target_name)
            tns_response = get_coords(
                objname=target_name,
                TNS_BOT_ID=self.input_yaml["wcs"]["TNS_BOT_ID"],
                TNS_BOT_NAME=self.input_yaml["wcs"]["TNS_BOT_NAME"],
                TNS_BOT_API=self.input_yaml["wcs"]["TNS_BOT_API"],
            )
            
            # If TNS returned None (object not found) and no manual coordinates provided,
            # stop and ask user for RA/Dec
            if tns_response is None:
                if (
                    self.input_yaml.get("target_ra") is not None
                    and self.input_yaml.get("target_dec") is not None
                ):
                    self.logger.warning(
                        "TNS lookup failed for '%s', but manual RA/Dec provided. Using manual coordinates.",
                        target_name,
                    )
                    return {
                        "ra": float(self.input_yaml["target_ra"]),
                        "dec": float(self.input_yaml["target_dec"]),
                    }
                else:
                    # No TNS response and no manual coordinates - cannot proceed with target_name
                    self.logger.error(
                        "TNS lookup failed for target '%s' and no RA/Dec coordinates provided.",
                        target_name,
                    )
                    print(
                        f"\n{'='*60}\n"
                        f"ERROR: Target '{target_name}' not found on TNS.\n\n"
                        f"You provided a target_name but:\n"
                        f"  1. TNS lookup failed (no API credentials or object not found), AND\n"
                        f"  2. No manual RA/Dec coordinates were provided.\n\n"
                        f"To fix this, either:\n"
                        f"  a) Provide TNS API credentials in your input YAML:\n"
                        f"       wcs:\n"
                        f"         TNS_BOT_ID: <your_bot_id>\n"
                        f"         TNS_BOT_NAME: <your_bot_name>\n"
                        f"         TNS_BOT_API: <your_api_key>\n\n"
                        f"  b) Provide RA/Dec coordinates manually:\n"
                        f"       autophot_input['target_ra'] = <RA in degrees>\n"
                        f"       autophot_input['target_dec'] = <Dec in degrees>\n\n"
                        f"  c) Remove target_name to run without a specific target\n"
                        f"{'='*60}\n"
                    )
                    sys.exit("Stopped: TNS lookup failed and no RA/Dec provided for target_name.")
            
            AutophotYaml.create(str(transient_path), tns_response)
            obj_name = tns_response.get("name_prefix", "") + tns_response["objname"]
            self.logger.info("Retrieved TNS information for %s.", obj_name)
            _log_tns_summary(tns_response, source="query")
            return tns_response
        except Exception as exc:
            exc_type, _, exc_tb = sys.exc_info()
            fname = (
                os.path.basename(exc_tb.tb_frame.f_code.co_filename)
                if exc_tb
                else "unknown"
            )
            line = exc_tb.tb_lineno if exc_tb else -1
            self.logger.error(
                "TNS API error: %s in %s:%d - %s",
                exc_type.__name__,
                fname,
                line,
                exc,
            )
            sys.exit(
                "Cannot reach TNS server - check internet connection and API credentials."
            )

    def _update_telescope_yml(
        self,
        wdir: str,
        telescope: str,
        block_key: str,
        instrument: str,
        header_filter_value: str,
        catalog_band: str,
    ) -> None:
        """
        Add or update a filter mapping in wdir/telescope.yml so that
        header_filter_value (e.g. 'K') maps to catalog_band (e.g. 'K') for this telescope/instrument.
        Creates telescope.yml or the telescope/instrument block if missing (with filter_key_0: FILTER).
        """
        path = os.path.join(wdir, "telescope.yml")
        try:
            if os.path.isfile(path):
                with open(path, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
        except Exception as e:
            log_warning_from_exception(
                self.logger, "Could not load telescope.yml for update", e
            )
            return
        if telescope not in data:
            data[telescope] = {}
        if block_key not in data[telescope]:
            data[telescope][block_key] = {}
        if instrument not in data[telescope][block_key]:
            data[telescope][block_key][instrument] = {"filter_key_0": "FILTER"}
        inst_block = data[telescope][block_key][instrument]
        if (
            str(header_filter_value) not in inst_block
            or inst_block[str(header_filter_value)] != catalog_band
        ):
            inst_block[str(header_filter_value)] = catalog_band
            try:
                os.makedirs(wdir, exist_ok=True)
                with open(path, "w") as f:
                    yaml.dump(
                        data,
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                        allow_unicode=True,
                    )
                self.logger.info(
                    "Updated telescope.yml: %s / %s filter %s -> %s",
                    telescope,
                    instrument,
                    header_filter_value,
                    catalog_band,
                )
            except Exception as e:
                log_warning_from_exception(
                    self.logger, "Could not write telescope.yml", e
                )
        #
        # Pixel scale updates use a separate helper so they can be added without
        # touching user-defined values if already present.

    def _maybe_update_pixel_scale_yml(
        self,
        wdir: str,
        telescope: str,
        block_key: str,
        instrument: str,
        header,
    ) -> None:
        """
        Compute pixel scale from the image WCS and, if a sensible value is
        found, store it in wdir/telescope.yml for this telescope/instrument.

        This only writes when either:
        - the telescope/instrument entry does not yet exist in telescope.yml, or
        - it exists but has no valid numeric ``pixel_scale``.

        Existing user-provided ``pixel_scale`` values are left unchanged.
        """
        # Derive candidate pixel scale from WCS (arcsec/pixel)
        try:
            with np.errstate(all="ignore"):
                wcs_obj = get_wcs(header)
                xy_scales = WCS.utils.proj_plane_pixel_scales(wcs_obj)
                if xy_scales is None or len(xy_scales) == 0:
                    return
                pixel_scale_candidate = float(xy_scales[0]) * 3600.0
        except Exception as exc:
            # WCS is missing or invalid; nothing to do.
            self.logger.debug(
                "Could not derive pixel scale from WCS for %s/%s: %s",
                telescope,
                instrument,
                exc,
            )
            return

        if not (np.isfinite(pixel_scale_candidate) and 0 < pixel_scale_candidate <= 5):
            # Discard clearly unreasonable scales.
            return

        path = os.path.join(wdir, "telescope.yml")
        try:
            if os.path.isfile(path):
                with open(path, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
        except Exception as e:
            self.logger.warning(
                "Could not load telescope.yml for pixel_scale update: %s", e
            )
            return

        # Ensure nested structure exists
        if telescope not in data:
            data[telescope] = {}
        if block_key not in data[telescope]:
            data[telescope][block_key] = {}
        if instrument not in data[telescope][block_key]:
            data[telescope][block_key][instrument] = {}

        inst_block = data[telescope][block_key][instrument]
        existing_ps = inst_block.get("pixel_scale")
        try:
            existing_val = float(existing_ps)
        except (TypeError, ValueError):
            existing_val = None

        # If a valid pixel_scale is already present, do not override it.
        if existing_val is not None and np.isfinite(existing_val) and existing_val > 0:
            return

        inst_block["pixel_scale"] = float(pixel_scale_candidate)
        try:
            os.makedirs(wdir, exist_ok=True)
            with open(path, "w") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            self.logger.info(
                "Updated telescope.yml pixel_scale: %s / %s -> %.3f arcsec/pixel",
                telescope,
                instrument,
                pixel_scale_candidate,
            )
        except Exception as e:
            self.logger.warning(
                "Could not write telescope.yml for pixel_scale update: %s", e
            )

    # --- Check Filters ---
    def check_filters(
        self, flist: List[str], available_filters: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Validates the list of image files based on their filter information.

        The logic is robust to missing or incomplete ``telescope.yml`` entries:
        if a telescope/instrument combination is not found, the code falls back
        to using common header names (e.g. ``FILTER``) and the raw filter value.

        Args
        ----
        flist : list of str
            File paths to check.
        available_filters : list of str
            List of available filter names from the catalog.

        Returns
        -------
        (list of str, list of str)
            Updated list of file paths and the list of filters that passed.
        """
        filter_unavailable: List[str] = []
        filter_available: List[str] = []
        out_flist: List[str] = []
        files_removed = 0
        filters_removed = 0
        filters_not_selected = 0
        available_filters, dropped_available = sanitize_photometric_filters(
            available_filters, available_filters=available_filters
        )
        if dropped_available:
            self.logger.info(
                "Filter check: dropped unsupported catalog keys: %s",
                ", ".join(sorted(set(dropped_available))),
            )
        selected_filters_cfg = self.input_yaml.get("selected_filters", [])
        selected_filters = []
        if selected_filters_cfg and selected_filters_cfg[0] is not None:
            selected_filters, dropped_selected = sanitize_photometric_filters(
                selected_filters_cfg, available_filters=available_filters
            )
            if dropped_selected:
                self.logger.warning(
                    "selected_filters contains unsupported values; ignoring: %s",
                    ", ".join(sorted(set(dropped_selected))),
                )
            if not selected_filters:
                self.logger.warning(
                    "selected_filters has no supported bands; no files will pass this constraint."
                )
                selected_filters = []

        # Cache of header filter -> catalog band mappings seen in this run,
        # keyed by (telescope, instrument, raw_header_value). This prevents us
        # from repeatedly updating telescope.yml with the same mapping (e.g.
        # many exposures with FILTER='rp' all mapping to 'r').
        seen_filter_mappings: Dict[Tuple[str, str, str], str] = {}

        tele_key = "TELESCOP"
        inst_key = "INSTRUME"
        avoid_keys = ["clear", "open"]

        # Load telescope configuration (telescope.yml + built-in). Images must have TELESCOP and INSTRUME.
        tele_autophot_input = load_telescope_config(self.input_yaml["wdir"])

        self.logger.info(
            border_msg(
                "Filter check",
                body="-",
                corner="+",
            )
        )
        self.logger.info(
            "Checking %d image(s). Catalog bands: %s\n",
            len(flist),
            ", ".join(sorted(available_filters)),
        )

        # Fail fast: the pipeline requires an explicit catalog choice to map
        # instrument filter names onto supported catalog bands.
        use_catalog = (self.input_yaml.get("catalog") or {}).get("use_catalog", None)
        if (
            use_catalog is None
            or str(use_catalog).strip() == ""
            or str(use_catalog).lower() == "none"
        ):
            msg = (
                "catalog.use_catalog is not set (null/None). "
                "AutoPHOT requires a catalog to map filters and perform calibration. "
                "Set `default_input.catalog.use_catalog` (e.g. gaia, pan_starrs, sdss, legacy, apass, 2mass, refcat, custom)."
            )
            self.logger.warning(msg)
            raise ValueError(msg)

        # If no files were passed in, return immediately to avoid zero-division
        # in the progress-bar helper and keep template-only runs robust.
        if not flist:
            return out_flist, filter_available

        for name in print_progress_bar(flist):
            is_template = "templates" in os.path.normpath(name)
            ts_cfg = self.input_yaml.get("template_subtraction") or {}
            prepare_templates = bool(ts_cfg.get("prepare_templates", False))
            if (
                prepare_templates
                and "PSF_model" in name
            ):
                # Skip pre-computed PSF model products when validating templates
                continue

            headinfo = get_header(name)
            tele = headinfo.get(tele_key)
            inst = headinfo.get(inst_key)

            mapping = None
            block_key = None
            if tele in tele_autophot_input:
                block_key, mapping = get_instrument_config(
                    tele_autophot_input, tele, inst
                )
                if mapping is None and not is_template:
                    self.logger.warning(
                        "Instrument '%s' for telescope '%s' not found in telescope.yml; "
                        "falling back to raw FILTER header for %s.",
                        inst,
                        tele,
                        name,
                    )
                else:
                    # Optionally update telescope.yml with pixel_scale derived from WCS
                    # when a valid telescope/instrument mapping exists.
                    if tele and inst and block_key is not None:
                        self._maybe_update_pixel_scale_yml(
                            self.input_yaml["wdir"],
                            tele,
                            block_key,
                            inst,
                            headinfo,
                        )
            else:
                # For template files with missing TELESCOP/INSTRUME, quietly fall back
                # to raw FILTER header without noisy warnings.
                if not is_template:
                    self.logger.warning(
                        "Telescope '%s' not found in telescope.yml; falling back to raw FILTER header for %s.",
                        tele,
                        name,
                    )

            fits_filter = "no_filter"
            filter_name = "no_filter"

            if mapping is not None:
                # Use telescope.yml mapping (or header value if not yet in mapping)
                filter_keys = [k for k in mapping if k.startswith("filter_key_")]
                header_key = None
                for filter_header_key in filter_keys:
                    candidate = mapping[filter_header_key]
                    if candidate not in headinfo:
                        continue
                    value = str(headinfo[candidate]).lower()
                    if value in avoid_keys:
                        continue
                    header_key = candidate
                    break

                if header_key is None:
                    # Could not find a suitable filter header; fall back
                    self.logger.warning(
                        "No valid filter header found for '%s' with telescope '%s' / instrument '%s'; "
                        "falling back to raw FILTER header.",
                        name,
                        tele,
                        inst,
                    )
                else:
                    fits_filter = headinfo.get(header_key, "no_filter")
                    # Case-insensitive lookup for filter mapping
                    fits_filter_str = str(fits_filter)
                    filter_name = fits_filter_str  # Default to raw value
                    for key, value in mapping.items():
                        if key.startswith("filter_key_"):
                            continue  # Skip metadata keys
                        if str(key).lower() == fits_filter_str.lower():
                            filter_name = str(value)
                            break
            else:
                # Fallback: try common header names directly
                candidate_keys = ["FILTER", "FILTER1", "FILTER2"]
                for key in candidate_keys:
                    if key in headinfo and str(headinfo[key]).lower() not in avoid_keys:
                        fits_filter = headinfo[key]
                        filter_name = str(fits_filter)
                        break

            # If still no filter (e.g. template with no FILTER/TELESCOP), infer from path
            # e.g. .../gp_template/... -> g, .../K_template/... -> K, .../zp_template/... -> z, .../B_template/... -> B
            if filter_name == "no_filter" and "templates" in os.path.normpath(name):
                norm = os.path.normpath(name)
                # Standard template patterns
                standard_patterns = (
                    "u_template", "g_template", "r_template", "i_template", "z_template",
                    "gp_template", "rp_template", "ip_template", "up_template", "zp_template",
                    "J_template", "H_template", "K_template",
                    "B_template", "V_template", "R_template", "I_template",
                    "Y_template", "w_template", "y_template",
                )
                
                # Check standard patterns first
                for folder_band in standard_patterns:
                    if folder_band in norm:
                        band = folder_band.split("_")[0].replace("p", "")
                        if band in available_filters:
                            filter_name = band
                            fits_filter = band
                        break
                
                # If no standard pattern matched, try dynamic patterns
                if filter_name == "no_filter" and available_filters:
                    # Create dynamic template patterns from available filters
                    for avail_filter in available_filters:
                        if f"{avail_filter}_template" in norm:
                            filter_name = avail_filter
                            fits_filter = avail_filter
                            self.logger.info(
                                "Discovered dynamic template pattern '%s_template' for %s",
                                avail_filter, os.path.basename(name)
                            )
                            break

            # Apply catalog and user filter constraints
            # Dynamic filter validation: accept any filter that can be normalized
            # using the available_filters list, enabling completely custom filter names
            normalized_filter = None
            if filter_name != "no_filter":
                normalized_filter = normalize_photometric_filter_name(
                    filter_name, 
                    available_filters=available_filters
                )
            
            if (
                normalized_filter is None 
                and filter_name != "no_filter"
                and not prepare_templates
            ):
                # Filter could not be normalized with available filters
                self.logger.info(
                    "Filter %s could not be matched to catalog filters (available: %s) for %s",
                    filter_name,
                    ", ".join(sorted(available_filters)),
                    os.path.basename(name),
                )
                self.logger.info(
                    "Solutions: 1) Add filter to custom catalog, 2) Use Gaia custom catalog with transmission curves, 3) Check telescope.yml mapping"
                )
                
                files_removed += 1
                filters_removed += 1
                filter_unavailable.append(filter_name)
                continue
            
            # Update filter_name to the normalized version
            if normalized_filter is not None:
                filter_name = normalized_filter
                # Only log normalization for significant changes, not for common cases
                if (filter_name != str(fits_filter) and 
                    not (str(fits_filter).endswith('.00000') and filter_name == str(fits_filter).split('.')[0])):
                    self.logger.debug(
                        "Auto-normalized filter %s -> %s for %s",
                        str(fits_filter),
                        filter_name,
                        os.path.basename(name),
                    )

            if (
                self.input_yaml["select_filter"]
                and not prepare_templates
            ):
                # When using select_filter, require the raw header value to be in do_filter
                if str(fits_filter) not in self.input_yaml["do_filter"]:
                    files_removed += 1
                    filters_removed += 1
                    filter_unavailable.append(filter_name)
                    continue

            if selected_filters and selected_filters[0] is not None:
                if filter_name not in selected_filters:
                    files_removed += 1
                    filters_not_selected += 1
                    continue

            filter_available.append(filter_name)
            out_flist.append(name)

            # If this telescope/instrument had a header filter value not in telescope.yml,
            # add it once per run and remember the mapping so repeated occurrences of the
            # same convention (e.g. 'rp' -> 'r') are not re-inserted.
            key = (str(tele), str(inst), str(fits_filter))
            if (
                mapping is not None
                and block_key is not None
                and fits_filter != "no_filter"
                and key not in seen_filter_mappings
                and str(fits_filter) not in mapping
            ):
                self._update_telescope_yml(
                    self.input_yaml["wdir"],
                    tele,
                    block_key,
                    inst,
                    fits_filter,
                    filter_name,
                )
                seen_filter_mappings[key] = filter_name

        if len(filter_unavailable) > 0:
            self.logger.info(
                "  %d file(s) removed: filter not in catalog (%s).",
                len(filter_unavailable),
                ", ".join(sorted(set(filter_unavailable))),
            )
        if filters_not_selected > 0:
            self.logger.info(
                "  %d file(s) removed: filter not in selected_filters.",
                filters_not_selected,
            )
        self.logger.info(
            "  Retained: %d file(s) with filters: %s",
            len(out_flist),
            ", ".join(sorted(set(filter_available))) if filter_available else "none",
        )

        return out_flist, filter_available

    # --- Find Templates ---
    def find_templates(self, required_filters: Optional[List[str]] = None) -> List[str]:
        """
        Searches for and validates template files in the specified directory.

        Args:
            required_filters (Optional[List[str]]): List of required filter names. If None, uses default filters.

        Returns:
            List[str]: List of valid template file paths.
        """
        self.logger.info(border_msg("Searching for template files"))
        template_list = []

        # Normalize the FITS directory path
        if self.input_yaml["fits_dir"].endswith("/"):
            self.input_yaml["fits_dir"] = self.input_yaml["fits_dir"][:-1]

        template_dir = os.path.join(self.input_yaml["fits_dir"], "templates")
        self.logger.info("Looking for templates in directory: %s", template_dir)
        template_status = {}

        # Load default filters if none provided
        if not required_filters:
            # For dynamic filter system, use available filters from catalog
            # This ensures templates work with transmission curves
            try:
                available_filters = self.check_catalog()
                if available_filters:
                    required_filters = available_filters
                    self.logger.info(
                        "Using available catalog filters for template search: %s",
                        ", ".join(sorted(required_filters))
                    )
                else:
                    # Fallback to default filters
                    base_filepath = os.path.dirname(os.path.abspath(__file__))
                    base_database = os.path.join(base_filepath, "databases")
                    filters_yml = "filters.yml"
                    required_filters = (
                        AutophotYaml(os.path.join(base_database, filters_yml))
                        .load()["default_dmag"]
                        .keys()
                    )
            except Exception:
                # Fallback to default filters if catalog check fails
                base_filepath = os.path.dirname(os.path.abspath(__file__))
                base_database = os.path.join(base_filepath, "databases")
                filters_yml = "filters.yml"
                required_filters = (
                    AutophotYaml(os.path.join(base_database, filters_yml))
                    .load()["default_dmag"]
                    .keys()
                )

        raw_filters = [str(f).strip() for f in required_filters if str(f).strip()]
        # For template lookup, use available_filters if available to support dynamic filters
        try:
            template_available_filters = self.check_catalog()
            required_filters, dropped_required = sanitize_photometric_filters(raw_filters, available_filters=template_available_filters)
        except Exception:
            # Fallback to default behavior if catalog check fails
            required_filters, dropped_required = sanitize_photometric_filters(raw_filters)
        if dropped_required:
            self.logger.info(
                "Template lookup: ignored %d unsupported/non-filter keys.",
                len(dropped_required),
            )

        if not os.path.isdir(template_dir):
            self.logger.warning(
                "No template folder found at expected location; skipping templates. "
                "Expected position: %s",
                template_dir,
            )
            return []

        incorrect_setup = False

        for filter_x in required_filters:
            canonical_filter = str(filter_x).strip()
            # Support both modern (r_template) and legacy (rp_template) folder names
            # for ugriz bands. Prefer modern naming when both are present.
            dir_candidates = [f"{canonical_filter}_template"]
            if canonical_filter in {"u", "g", "r", "i", "z"}:
                dir_candidates.append(f"{canonical_filter}p_template")

            existing_dirs = [
                os.path.join(template_dir, d) for d in dir_candidates if os.path.isdir(os.path.join(template_dir, d))
            ]
            template_status[canonical_filter] = {}

            if len(existing_dirs) == 0:
                template_status[canonical_filter]["status"] = "directory does not exist"
                template_status[canonical_filter]["fpath"] = None
                template_status[canonical_filter]["expected_dirs"] = [
                    os.path.join(template_dir, d) for d in dir_candidates
                ]
                continue

            expected_template_dir = existing_dirs[0]
            template_status[canonical_filter]["expected_dirs"] = [
                os.path.join(template_dir, d) for d in dir_candidates
            ]
            template_status[canonical_filter]["dir_used"] = expected_template_dir
            if os.path.basename(expected_template_dir).endswith("p_template"):
                self.logger.info(
                    "[%s template] Using legacy folder naming: %s",
                    canonical_filter,
                    os.path.basename(expected_template_dir),
                )

            template_files = glob.glob(os.path.join(expected_template_dir, "*.fits"))
            psf_model_files = glob.glob(
                os.path.join(expected_template_dir, "PSF_model_*")
            )
            template_files = list(set(template_files) - set(psf_model_files))
            original_files = glob.glob(
                os.path.join(expected_template_dir, "*.fits.original")
            )

            if len(original_files) == 1:
                template_status[canonical_filter]["status"] = "found"
                shutil.copyfile(
                    original_files[0], original_files[0].replace(".original", "")
                )
                template_status[canonical_filter]["fpath"] = original_files[0].replace(
                    ".original", ""
                )
            elif len(template_files) == 0:
                template_status[canonical_filter]["status"] = "not found"
                template_status[canonical_filter]["fpath"] = None
                incorrect_setup = True
            elif len(template_files) > 1:
                self.logger.warning(
                    "Multiple (%d) template files found for %s-band; choosing the largest.",
                    len(template_files),
                    canonical_filter,
                )
                incorrect_setup = True
                template_file_sizes = {f: os.path.getsize(f) for f in template_files}
                best_guess = max(template_file_sizes, key=template_file_sizes.get)
                template_status[canonical_filter]["status"] = "multiple found"
                template_status[canonical_filter]["fpath"] = best_guess
            else:
                template_status[canonical_filter]["status"] = "found"
                template_status[canonical_filter]["fpath"] = template_files[0]

        # Log template status
        for key, info in template_status.items():
            status = info.get("status")
            if status == "found":
                self.logger.info("[%s template] Using file: %s", key, info.get("fpath"))
            elif status in ("not found", "directory does not exist"):
                expected_dirs = info.get("expected_dirs") or [
                    os.path.join(template_dir, f"{key}_template")
                ]
                self.logger.warning(
                    "[%s template] Missing; expected directory one of: %s",
                    key,
                    ", ".join(expected_dirs),
                )
            elif status == "multiple found":
                self.logger.warning(
                    "[%s template] Multiple candidates; using best guess: %s",
                    key,
                    info.get("fpath"),
                )

        # Prompt user if issues detected
        if incorrect_setup:
            quit_answer = (
                input(
                    "\nTemplate directory appears incomplete or ambiguous (missing and/or multiple templates).\n"
                    "Continue anyway and skip problematic filters? [y/N]: "
                )
                .strip()
                .lower()
                or "n"
            )
            if quit_answer not in ("y", "yes"):
                raise Exception(
                    "Please check template directory and subdirectories and try again."
                )

        template_list = [
            template_status[f]["fpath"]
            for f in template_status
            if template_status[f].get("fpath") is not None
        ]
        return template_list
