#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIPER/HPC driver for the ZTF -> AutoPhOT pipeline.

This script does **two phases**:

1. On the login/head node:
   - Reads a BTS-style catalog of ZTF transients.
   - For each transient:
       * Queries Fritz for RA/Dec.
       * Downloads ZTF science images and reference templates.
       * Stacks images into an `images/` directory laid out for AutoPhOT.
       * Copies templates into that directory.
       * Compresses raw FITS into `raw.zip` to save space.
   - Records, for every successfully prepared transient, the information
     needed to run AutoPhOT (fits_dir, RA/Dec, name).

2. When **all** transients are prepared:
   - Submits a *chain* of Slurm jobs, one per transient, using
     `run_autophot_VIPER.sbatch`.
   - Each job depends on the previous (`--dependency=afterok:<jobid>`),
     so at most **one AutoPhOT job runs at a time**.

Assumptions for the Slurm job script:
-------------------------------------
- There is a `run_autophot_VIPER.sbatch` in the same directory as this file.
- It honours environment variables supplied via `sbatch --export`, e.g.:
    AUTOPHOT_DIR   : path to this repository on the HPC filesystem
    AUTOPHOT_FITS_DIR : path to the transient's `images` directory
    AUTOPHOT_TARGET_RA / AUTOPHOT_TARGET_DEC : target coordinates (deg)
    AUTOPHOT_NAME  : transient name (e.g. SN2020abc)

  You can adapt your existing `run_autophot_VIPER.sbatch` to read these
  variables and call `python -m autophot` (or a small wrapper) with a
  suitable configuration.
"""

import os
import sys
import shutil
import zipfile
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from ztfquery import fritz

# ---------------------------------------------------------------------------
# Imports from local/companion projects
# ---------------------------------------------------------------------------

# AutoPhOT repo (this project)
AUTOPHOT_DIR_DEFAULT = "/home/sbrennan/Documents/autophot_object"
if AUTOPHOT_DIR_DEFAULT not in sys.path:
    sys.path.append(AUTOPHOT_DIR_DEFAULT)
from autophot import AutomatedPhotometry  # noqa: F401 (import side-effects/config schema)

# ZTF helper modules (user project)
ZTF_HELPER_DIR = "/home/sbrennan/Documents/ZTF"
if ZTF_HELPER_DIR not in sys.path:
    sys.path.append(ZTF_HELPER_DIR)
from download_science_images import download_science_images  # type: ignore
from download_reference_images import download_reference_images  # type: ignore
from download_deepreference_images import download_deepreference_images  # type: ignore

# Local helper for templates and stacking (from the Precursor/sample repo)
SAMPLE_HELPER_DIR = "/home/sbrennan/Desktop/Precursor/sample"
if SAMPLE_HELPER_DIR not in sys.path:
    sys.path.append(SAMPLE_HELPER_DIR)
from download_IPAC_images import (  # type: ignore
    copy_directory_with_structure,
)
from stack import stack_ZTF_IRSA_images  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ztf_viper_pipeline")


def log(msg: str, level: int = logging.INFO) -> None:
    logger.log(level, msg)


# ---------------------------------------------------------------------------
# Configuration (paths & time cuts) - edit for HPC deployment
# ---------------------------------------------------------------------------

# Working directory on the HPC filesystem: one subdirectory per transient.
WLOC = "/home/sbrennan/Desktop/Precursor/sample/"

# Input BTS/target catalog (must be readable on the head node).
TLOC = "/home/sbrennan/Desktop/Precursor/data/BTS_interacting_within_316Mpc.txt"

# Time cutoffs for template and peak selection (in JD)
TEMPLATE_CUTOFF = Time("2021-01-01").mjd + 2400000.5
PEAK_CUTOFF_LO = Time("2019-01-01").mjd + 2400000.5
PEAK_CUTOFF_HI = Time("2025-01-01").mjd + 2400000.5

# Epoch limits for image selection (days around peak)
EPOCH_MIN = 365
EPOCH_MAX = 31

# Filters and data-selection flags
FILTERS = ["zr", "zg"]
USE_DEEPREF = False

# Transient selection controls
IGNORE: List[str] = []  # Names (lowercase) to skip
ACCEPT: List[str] = []  # If non-empty, only these are processed
REDO: int = 0           # 0: skip if reduced output exists; 1: always redo

# TEST_MODE: when True, prepare and submit **only one** transient.
# This is useful for quick end-to-end tests on a new HPC setup.
TEST_MODE: bool = bool(int(os.environ.get("RUN_SAMPLE_TEST_MODE", "0")))


# ---------------------------------------------------------------------------
# Secrets helper (for AutoPhOT catalog/WCS credentials)
# ---------------------------------------------------------------------------

def load_secret(env_var: str, fallback_module_attr: str | None = None) -> str:
    """
    Load a secret from environment variables or a fallback module attribute.
    """
    value = os.environ.get(env_var)
    if value:
        return value
    if fallback_module_attr:
        import autophot_tokens  # type: ignore

        return getattr(autophot_tokens, fallback_module_attr)
    raise RuntimeError(f"Secret '{env_var}' not set in environment or fallback module.")


# ---------------------------------------------------------------------------
# Utility helpers for file management
# ---------------------------------------------------------------------------

def collect_compress_and_delete_fits(directory: str, output_zip: str) -> None:
    """
    Collect, compress, and delete FITS files in `directory` into `output_zip`.
    """
    fits_files = [f for f in os.listdir(directory) if f.endswith(".fits")]
    if not fits_files:
        log(f"No FITS files found in {directory} to compress.", logging.WARNING)
        return

    with zipfile.ZipFile(output_zip, "w") as zipf:
        for file in fits_files:
            file_path = os.path.join(directory, file)
            zipf.write(file_path, arcname=file)

    for file in fits_files:
        os.remove(os.path.join(directory, file))

    log(f"Compressed and deleted {len(fits_files)} .fits files into {output_zip}")


def unzip_and_delete(raw_loc: str, extract_folder: str) -> None:
    """
    Unzip `raw_loc` into `extract_folder` and delete the original zip.
    """
    if not os.path.exists(raw_loc):
        log(f"The file {raw_loc} does not exist.", logging.WARNING)
        return
    try:
        with zipfile.ZipFile(raw_loc, "r") as zip_ref:
            os.makedirs(extract_folder, exist_ok=True)
            zip_ref.extractall(extract_folder)
            log(f"Files extracted to {extract_folder}")
        os.remove(raw_loc)
        log(f"{raw_loc} has been deleted.")
    except zipfile.BadZipFile:
        log(f"The file {raw_loc} is not a valid zip file.", logging.ERROR)
    except Exception as e:
        log(f"An error occurred during unzip: {e}", logging.ERROR)


def delete_all_files_in_directory(odir: str) -> None:
    """
    Delete all files and subdirectories in a directory.
    """
    if not os.path.exists(odir):
        log(f"The directory {odir} does not exist.", logging.WARNING)
        return
    if not os.path.isdir(odir):
        log(f"{odir} is not a valid directory.", logging.ERROR)
        return
    try:
        shutil.rmtree(odir)
        log(f"Directory {odir} and all its contents have been deleted.")
    except Exception as e:
        log(f"Failed to delete {odir}: {e}", logging.ERROR)


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the BTS catalog.
    """
    data = pd.read_csv(filepath)
    if "ZTFID" not in data.columns or "IAUID" not in data.columns:
        log("ZTFID or IAUID column missing from input file.", logging.ERROR)
        sys.exit(1)

    data = data[data["ZTFID"].notna()]
    data.sort_values(by="ZTFID", inplace=True)
    return data


def format_name_from_row(row: pd.Series) -> str:
    """
    Normalise the transient name based on the IAUID.
    """
    name = str(row["IAUID"]).replace(" ", "")
    if name.startswith("20"):
        name = "SN" + name
    return name


# ---------------------------------------------------------------------------
# Data structure: one prepared transient ready for AutoPhOT
# ---------------------------------------------------------------------------

@dataclass
class PreparedTransient:
    name: str
    fits_dir: str
    ra: float
    dec: float


# ---------------------------------------------------------------------------
# Phase 1: Download and prepare all transients (no Slurm here)
# ---------------------------------------------------------------------------

def prepare_transients(data: pd.DataFrame) -> List[PreparedTransient]:
    """
    For each transient in `data`, download science + templates, stack, and
    produce an `images` directory that AutoPhOT can consume.

    Returns a list of `PreparedTransient` for which preparation succeeded.
    """
    prepared: List[PreparedTransient] = []

    data = data.sort_values(by="IAUID")
    SIZE = 500

    for idx, row in data.iterrows():
        try:
            transient = row["ZTFID"]
            try:
                rise = float(row["rise"])
            except Exception:
                rise = 21
            peak = float(row["peakt"])
            name = format_name_from_row(row)

            log(f"\n\n === Preparing transient {name} (ZTFID={transient}) with peak {peak} === ")

            # Selection: ignore/accept lists and peak-time window.
            if name.lower() in [x.lower() for x in IGNORE]:
                log(f"{name} is in the ignore list. Skipping.")
                continue
            if ACCEPT and name not in ACCEPT:
                log(f"{name} is not in the accept list. Skipping.")
                continue
            if peak < PEAK_CUTOFF_LO or peak > PEAK_CUTOFF_HI:
                log(f"{name} peaks outside of the given time range. Skipping.")
                continue

            # Directories for this transient
            newloc = os.path.join(WLOC, name)
            os.makedirs(newloc, exist_ok=True)
            raw_loc = os.path.join(newloc, "raw.zip")
            odir = os.path.join(newloc, "images")
            odir_reduced = os.path.join(newloc, "images_REDUCED")
            expected_output = os.path.join(odir_reduced, "lightcurve_output.csv")

            if os.path.exists(expected_output) and not REDO:
                log(f"Reduced output already exists for {name}. Skipping.")
                continue

            # If we have a previous archive, unpack it and clean old reductions.
            if os.path.exists(raw_loc):
                delete_all_files_in_directory(odir)
                unzip_and_delete(raw_loc, extract_folder=newloc)
                try:
                    delete_all_files_in_directory(odir_reduced)
                except Exception:
                    log(f"Failed to delete reduced directory for {name}.")

            # Coordinates from Fritz
            source = fritz.FritzSource.from_name(transient)
            ra, dec = source.get_coordinates()
            log(f"Coordinates for {name}: RA={ra}, DEC={dec}.")

            jd = peak
            jd_min = jd - EPOCH_MIN - rise
            jd_max = jd + EPOCH_MAX
            mjd_min = jd_min - 2400000.5
            mjd_max = jd_max - 2400000.5

            # Download science images
            log("[INFO] Downloading science images from ZTF...")
            _ = download_science_images(
                ra,
                dec,
                saveloc=newloc,
                size=SIZE,
                mjd_min=mjd_min,
                mjd_max=mjd_max,
                filters=FILTERS,
                mag_limit_min=19.0,
                seeing_max=3.0,
                max_workers=8,
            )

            # Download templates
            template_loc = os.path.join(newloc, "templates")
            os.makedirs(template_loc, exist_ok=True)

            if USE_DEEPREF:
                status = download_deepreference_images(
                    ra,
                    dec,
                    saveloc=template_loc,
                    size=SIZE,
                    filters=FILTERS,
                )
                # If download_deepreference_images returns a status dict, pick
                # out filters that failed and re-download via standard refs.
                get_templates = FILTERS
                try:
                    failed = [
                        f
                        for f, st in status.get("filter_status", {}).items()
                        if st != "ok"
                    ]
                    if failed:
                        get_templates = failed
                except Exception:
                    pass
            else:
                get_templates = FILTERS

            if get_templates:
                log(f"[INFO] Downloading reference images for filters: {get_templates}")
                download_reference_images(
                    ra,
                    dec,
                    saveloc=template_loc,
                    size=SIZE,
                    filters=get_templates,
                )

            # Stack images and arrange for AutoPhOT
            log("[INFO] Stacking ZTF images into nightly bins (combining same-night exposures)...")
            stack_ZTF_IRSA_images(
                rdir=newloc,
                transient=name,
                odir=odir,
                delete_original=False,
                nightly_stack=True,
            )

            log("[INFO] Copying template directory into images directory...")
            copy_directory_with_structure(
                src_dir=os.path.join(newloc, "templates"),
                dest_dir=odir,
            )

            log("[INFO] Compressing and cleaning up raw FITS files...")
            collect_compress_and_delete_fits(directory=newloc, output_zip=raw_loc)

            prepared.append(
                PreparedTransient(
                    name=name,
                    fits_dir=odir,
                    ra=float(ra),
                    dec=float(dec),
                )
            )
            log(f"[SUCCESS] Preparation completed for {name}.\n")

            # In TEST_MODE, stop after the first successfully prepared object.
            if TEST_MODE:
                log("TEST_MODE is enabled: stopping after first prepared transient.")
                break

        except Exception as e:
            import traceback

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename) if exc_tb else "unknown"
            lineno = exc_tb.tb_lineno if exc_tb else -1
            log(
                f"\nException preparing {name} in {fname} at line {lineno} "
                f"({exc_type.__name__ if exc_type else type(e).__name__}): {str(e)}\n{traceback.format_exc()}",
                logging.ERROR,
            )

    return prepared


# ---------------------------------------------------------------------------
# Phase 2: Submit chained Slurm jobs (one AutoPhOT job at a time)
# ---------------------------------------------------------------------------

def submit_autophot_jobs(
    prepared: List[PreparedTransient],
    sbatch_script: Path,
    autophot_dir_hpc: Path | None = None,
) -> None:
    """
    Submit one Slurm job per prepared transient, with dependencies so that
    only **one** job runs at a time:

        job_1  (no dependency)
        job_2  --dependency=afterok:job_1
        job_3  --dependency=afterok:job_2
        ...
    """
    if not prepared:
        log("No prepared transients to submit; nothing to do.", logging.WARNING)
        return

    if autophot_dir_hpc is None:
        autophot_dir_hpc = Path(AUTOPHOT_DIR_DEFAULT)

    if not sbatch_script.exists():
        log(f"Slurm script not found: {sbatch_script}", logging.ERROR)
        return

    # In TEST_MODE, only submit the first prepared transient.
    to_submit = prepared[:1] if TEST_MODE else prepared

    prev_job_id: str | None = None

    for pt in to_submit:
        env = os.environ.copy()
        env["AUTOPHOT_DIR"] = str(autophot_dir_hpc)
        env["AUTOPHOT_FITS_DIR"] = pt.fits_dir
        env["AUTOPHOT_TARGET_RA"] = f"{pt.ra:.8f}"
        env["AUTOPHOT_TARGET_DEC"] = f"{pt.dec:.8f}"
        env["AUTOPHOT_NAME"] = pt.name

        cmd: List[str] = ["sbatch"]
        if prev_job_id is not None:
            cmd.append(f"--dependency=afterok:{prev_job_id}")
        cmd.append(str(sbatch_script))

        log(f"Submitting AutoPhOT job for {pt.name} with command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            log(
                f"Failed to submit Slurm job for {pt.name}: {e.stderr or e.stdout}",
                logging.ERROR,
            )
            # Do not break the chain; try to continue with the next transient.
            continue

        # Parse job ID from "Submitted batch job <id>"
        job_id = None
        for line in (result.stdout or "").splitlines():
            if "Submitted batch job" in line:
                parts = line.strip().split()
                job_id = parts[-1]
                break
        if not job_id:
            log(
                f"Could not parse Slurm job ID for {pt.name} "
                f"(stdout: {result.stdout!r})",
                logging.WARNING,
            )
            # If we do not know the job ID, stop chaining dependencies.
            prev_job_id = None
        else:
            log(f"Submitted AutoPhOT job for {pt.name} with JobID={job_id}")
            prev_job_id = job_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log(f"Working directory: {WLOC}")
    log(f"Target file location: {TLOC}")

    data = load_and_preprocess_data(TLOC)
    log(f"Loaded {len(data)} transients from {TLOC}.")

    # Phase 1: prepare all transients (downloads, stacking, templates).
    prepared = prepare_transients(data)
    log(f"Prepared {len(prepared)} transients for AutoPhOT processing.")

    # Phase 2: submit serial AutoPhOT jobs via Slurm.
    this_dir = Path(__file__).resolve().parent
    sbatch_script = this_dir / "run_autophot_VIPER.sbatch"
    submit_autophot_jobs(prepared, sbatch_script=sbatch_script)


if __name__ == "__main__":
    main()

