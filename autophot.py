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
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
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

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
import re

# Module-level flag to optionally silence user-facing logging
QUIET_MODE = False

# Project-specific helpers (assumed to be in your codebase)
from functions import (
    border_msg,
    autophot_yaml,
    concatenate_csv_files,
    print_progress_bar,
    log_exception
)
from prepare import prepare
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
def find_variable_sources(ra_deg: float, dec_deg: float, radius_arcmin: int = 10) -> pd.DataFrame:
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
    Simbad.add_votable_fields("otype", "ra", "dec", "main_id", 
                            "galdim_majaxis", "galdim_minaxis", "galdim_angle")
    
    # Store and override settings
    original_timeout = getattr(Simbad, 'TIMEOUT', 10)
    original_row_limit = getattr(Simbad, 'ROW_LIMIT', 1000)
    
    Simbad.TIMEOUT = 30
    Simbad.ROW_LIMIT = 500
    
    centre_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk5", equinox="J2000")
    _log(
        f"Querying SIMBAD within {radius_arcmin}' of "
        f"{centre_coord.to_string('hmsdms')}."
    )
    
    # SOURCE TYPE DEFINITIONS
    object_types = {
        "V*": "Variable Star", "Pu*": "Pulsating", "Er*": "Eruptive", 
        "Ir*": "Irregular", "BY*": "BY Dra", "RS*": "RS CVn", "Fl*": "Flare",
        "Ro*": "Rotating", "Ce*": "Cepheid", "RR*": "RR Lyr", "dS*": "δ Sct",
        "LP*": "Long Period", "Mi*": "Mira", "TT*": "T Tauri", "Or*": "Orion Var",
        "CV*": "Cataclysmic", "DN*": "Dwarf Nova", "NL*": "Nova-like", 
        "No*": "Nova", "SN*": "Supernova", "AM*": "Polar", "DQ*": "Int Polar",
        "G": "Galaxy", "GiG": "Int Galaxy", "SBG": "Starburst", 
        "AGN": "AGN", "SyG": "Seyfert", "LIN": "LINER", "QSO": "Quasar", "BLL": "BL Lac",
        "Em*": "Emission*", "Be*": "Be Star", "XB*": "X-ray Bin", "X": "X-ray",
        "PN": "P Nebula", "HII": "HII Region"
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
        return pd.DataFrame(columns=[
            "RA", "DEC", "OTYPE", "MAIN_ID", "OTYPE_LABEL", "OTYPE_opt", 
            "separation_arcmin", "size_arcmin"
        ])
    
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
    df["RA"] = pd.to_numeric(df["RA"], errors='coerce')
    df["DEC"] = pd.to_numeric(df["DEC"], errors='coerce')
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
            return float(re.search(r'[\d.]+', str(dim)).group())
        except:
            return np.nan
    
    size_col = next((col for col in ["galdim_majaxis", "dim"] if col in df.columns), None)
    if size_col:
        df["size_arcmin"] = df[size_col].apply(parse_size)
    else:
        df["size_arcmin"] = np.nan
    
    # FINAL CLEANUP AND SORTING
    out_cols = ["RA", "DEC", "OTYPE", "MAIN_ID", "OTYPE_LABEL", "OTYPE_opt", 
               "separation_arcmin", "size_arcmin"]
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
        default_input = autophot_yaml(default_input_path, "default_input").load()

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

        _log(border_msg(f"Default input loaded from: {default_input_path}", body="-", corner="+"))
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
        cfg_ncpu = default_input.get("nCPU") or default_input.get("nCpu") or default_input.get("ncpu")
        try:
            n_cpu = int(env_ncpu) if env_ncpu is not None else int(cfg_ncpu) if cfg_ncpu is not None else 1
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
        fits_dir = default_input.get("fits_dir", "")
        if fits_dir.endswith("/"):
            fits_dir = fits_dir[:-1]
            default_input["fits_dir"] = fits_dir
            
            
        plt.close("all")

        # Initialise preparation helper and validate catalog configuration
        prepare_db = prepare(default_input=default_input)
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

            # TNS credentials
            for k in ("TNS_BOT_ID", "TNS_BOT_NAME", "TNS_BOT_API"):
                if wcs_cfg.get(k) is None:
                    v = getattr(autophot_tokens, k, None)
                    if v is not None:
                        wcs_cfg[k] = v
            default_input["wcs"] = wcs_cfg

            # MAST CasJobs credentials (Refcat)
            if cat_cfg.get("MASTcasjobs_wsid") is None:
                v = getattr(autophot_tokens, "MASTcasjobs_wsid", None)
                if v is not None:
                    cat_cfg["MASTcasjobs_wsid"] = v
            if cat_cfg.get("MASTcasjobs_pwd") is None:
                v = getattr(autophot_tokens, "MASTcasjobs_pwd", None)
                if v is not None:
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
            _log(border_msg("Performing photometry with AutoPhOT"))

            # List of available filters (excluding error columns)
            filt_list = [f for f in available_filters if "_err" not in f and f not in ["RA", "DEC"]]
            _log(f"Available filters: {filt_list}")

            # Optional: Enrich target metadata from TNS
            try:
                _log(border_msg("Checking TNS for transient information"))
                tns_coords = prepare_db.check_TNS()
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
                if default_input.get("target_ra") is not None and default_input.get("target_dec") is not None:
                    _log(
                        "[WARNING] TNS lookup skipped; using target_ra/target_dec "
                        "values from the configuration."
                    )
                else:
                    _log(
                        "[WARNING] TNS lookup skipped; no TNS access or target_name, "
                        "and no fallback RA/Dec provided in the configuration."
                    )

            # Clean and validate input files
            file_list = prepare_db.clean()
            if not default_input.get("skip_file_check", False):
                file_list = prepare_db.check_files(flist=file_list)

                if len(file_list) == 0:
                    _log(border_msg("No images left after validation.",body = '!',corner = '!'))
                    do_photometry = False
                else:
                    file_list, required_filters = prepare_db.check_filters(
                        flist=file_list, available_filters=available_filters)
            else:
                required_filters = available_filters

            if do_photometry:
                required_filters = list(set(required_filters))

                # Template handling and optional downloads
                template_folder = os.path.join(default_input["fits_dir"], "templates")
                ts_cfg = default_input.get("template_subtraction", {})
                if ts_cfg.get("download_templates", False) and ts_cfg.get("do_subtraction", False):
                    download_kind = ts_cfg["download_templates"]
                    size_default = ts_cfg.get("templates_size", 10)

                    if download_kind == "panstarrs":
                        _log(border_msg("Downloading template images from Pan-STARRS"))
                        from templates import download_panstarrs_template

                        for f in required_filters:
                            download_panstarrs_template(
                                ra=default_input["target_ra"],
                                dec=default_input["target_dec"],
                                size=size_default,
                                template_folder=template_folder,
                                f=f,
                            )
                    elif download_kind == "sdss":
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
                    template_file_list = prepare_db.find_templates(required_filters=required_filters)
                    template_file_list = prepare_db.check_files(
                        flist=template_file_list, template_files=True
                    )
                    template_file_list, _ = prepare_db.check_filters(
                        flist=template_file_list, available_filters=available_filters
                    )
                    # Re-run telescope header check on science + templates so template images
                    # with TELESCOP/INSTRUME are included in the telescope database
                    if template_file_list and not default_input.get("skip_file_check", False):
                        combined = list(file_list) + list(template_file_list)
                        combined_checked = prepare_db.check_files(
                            flist=combined, template_files=True
                        )
                        template_folder_norm = os.path.normpath(template_folder)
                        file_list = [f for f in combined_checked if os.path.normpath(f).startswith(template_folder_norm) is False]
                        template_file_list = [f for f in combined_checked if os.path.normpath(f).startswith(template_folder_norm)]

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
                    backup_yaml.setdefault("template_subtraction", {})["do_subtraction"] = False
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
                            ra_deg=default_input["target_ra"], dec_deg=default_input["target_dec"]
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
                            for fut in as_completed(futures):
                                fname, rc = fut.result()
                                if rc != 0:
                                    # Minimal reporting; main.py already logged details.
                                    _log(f"[TEMPLATE FAIL] {fname} (exit code {rc})")
                        gc.collect()
                    else:
                        _log(border_msg("Reducing and calibrating template files"))
                        for template in print_progress_bar(template_file_list, title="Template files calibrated"):
                            _run_main_subprocess(
                                python_executable,
                                autophot_exe,
                                template,
                                input_file,
                                True,
                            )
                            gc.collect()

                # Reduce science frames
                _log(border_msg("Reducing and calibrating science files"))
            
                counter = 0
                if file_list:
                    file_list = np.sort(file_list)[::-1]  # Sort newest first if filenames encode time

                    if parallel_files and len(file_list) > 1:
                        # Parallel image-level execution: each worker runs main.py on one file.
                        _log(f"Running {len(file_list)} science files with nCPU={n_cpu} (parallel).")
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
                            for fut in as_completed(futures):
                                fname, rc = fut.result()
                                if rc == 0:
                                    _log(f"[OK]    {fname}")
                                else:
                                    _log(f"[FAIL]  {fname} (exit code {rc})")
                                counter += 1
                        gc.collect()
                    else:
                        for file in print_progress_bar(file_list, title="Science files calibrated\n"):
                            try:
                                _log(f"\n{counter} / {len(file_list)}")
                                _run_main_subprocess(
                                    python_executable,
                                    autophot_exe,
                                    str(file),
                                    input_file,
                                    False,
                                )
                                gc.collect()
                                counter += 1
                            except Exception as e:
                                import traceback

                                tb = traceback.format_exc(limit=1)
                                _log(f"[ERROR] Problem with file: {file}: {e} | {tb}")

        # Concatenate per-image outputs into one light curve CSV
        reduced_loc = f"{default_input['fits_dir']}_{default_input['outdir_name']}"
        _log(border_msg(f"Collecting reduced photometry in {reduced_loc}"))
        output_loc = os.path.join(reduced_loc, "lightcurve_output.csv")
        concatenate_csv_files(
            folder_path=reduced_loc, output_filename=output_loc, loc_file="output.csv"
        )
        _log(f"Photometry pipeline completed in {time.perf_counter() - t0:.3f} seconds.")
        return output_loc

def main(argv: Optional[List[str]] = None) -> int:
    """
    Simple CLI wrapper for the high-level driver.

    Usage:
        autophot-driver run
        python autophot.py run
    """
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    ap = AutomatedPhotometry()
    config = ap.load()
    do_run = len(argv) > 0 and str(argv[0]).lower() == "run"
    output = ap.run_photometry(config, do_photometry=do_run)
    _log(f"Output light curve: {output}")
    return 0


# --- Main Execution ---
if __name__ == "__main__":
    raise SystemExit(main())
