#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:55:18 2022
@author: seanbrennan
"""

"""
Photometry pipeline for CCD/NIR images.

Inputs:
    - FITS (-f): The FITS file containing the image data.
    - YAML (-c): Configuration file specifying parameters for the pipeline.
    - Optional: -temp flag to prepare a template image.

Setup:
    - Working directory: Where all output files will be saved.
    - Logging: Logs all operations and errors for debugging and tracking.
    - Metadata: Extracts and sets up metadata such as Modified Julian Date (MJD), gain, read noise, saturation level, exposure time, filter, and pixel scale.

WCS (World Coordinate System):
    - Validates and solves the WCS information in the FITS header.
    - Refines the WCS solution for better accuracy.
    - Persists the updated WCS information back to the FITS header.

Preprocessing:
    - Cosmic-ray removal: Identifies and removes cosmic rays from the image.
    - Background modeling: Estimates and subtracts the background noise.
    - Trim/recrop: Trims or recrops the image to focus on the region of interest.
    - Optional north-up reprojection: Aligns the image so that north is up and east is left.

Sources:
    - Builds or downloads a catalog of reference sources.
    - Measures the Full Width at Half Maximum (FWHM) of sources.
    - Determines the optimum aperture for photometry.
    - Performs aperture or ePSF+PSF photometry on detected sources.

Calibration:
    - Fits zeropoints to calibrate the photometry.
    - Writes the calibration information to the FITS headers.

Template Subtraction:
    - Aligns the science image with a template image.
    - Optionally uses the ZOGY algorithm for subtraction.
    - Applies background correction to the difference image.

Target:
    - Measures the target at the expected coordinates.
    - Calculates Signal-to-Noise Ratio (SNR) and detection limits.
    -     Saves the updated FITS files and CSV outputs.
"""

# Safeguard: force BLAS/OpenMP to 1 thread before any scientific imports (avoids exhausting
# process/thread limits when using multiprocessing on HPC; OpenBLAS often defaults to 128).
import os

for _env in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env] = "1"


# TODO: add in a trimming if the template image is smaller than the sciene image


# =============================================================================
# Main Function: run_photometry
# =============================================================================
def run_photometry():

    # ---------------------------------------------------------------------
    # CLI parsing (must happen before heavy imports)
    # ---------------------------------------------------------------------
    import argparse
    import re
    from pathlib import Path

    def _print_default_yaml_help() -> None:
        """
        Print available YAML options and their inline explanations.

        This reads `databases/default_input.yml` and uses indentation to infer
        nested key paths. Explanations are taken from trailing `# ...` comments.
        """
        yml_path = Path(__file__).resolve().parent / "databases" / "default_input.yml"
        if not yml_path.exists():
            print(f"[ERROR] Default YAML not found at: {yml_path}")
            return

        lines = yml_path.read_text(encoding="utf-8", errors="replace").splitlines()

        # Parse mapping keys using indentation; keep comments as help text.
        key_stack: list[tuple[int, str]] = []
        out: list[tuple[str, str]] = []
        key_re = re.compile(
            r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z0-9_\-]+)\s*:\s*(?P<rest>.*)$"
        )

        for raw in lines:
            if not raw.strip():
                continue
            if raw.lstrip().startswith("#"):
                continue

            m = key_re.match(raw)
            if not m:
                continue

            indent = len(m.group("indent").replace("\t", "    "))
            key = m.group("key")
            rest = m.group("rest") or ""

            # maintain stack
            while key_stack and indent <= key_stack[-1][0]:
                key_stack.pop()
            key_stack.append((indent, key))

            # Extract explanation from inline comment
            expl = ""
            if "#" in rest:
                expl = rest.split("#", 1)[1].strip()
            else:
                if "#" in raw:
                    expl = raw.split("#", 1)[1].strip()

            path = ".".join(k for _, k in key_stack)
            out.append((path, expl))

        if not out:
            print(f"[WARN] No keys parsed from: {yml_path}")
            return

        key_w = max(len(k) for k, _ in out)
        try:
            print(f"\nAutoPHOT YAML options from: {yml_path}\n")
            for k, expl in out:
                if expl:
                    print(f"{k:<{key_w}}  {expl}")
                else:
                    print(f"{k:<{key_w}}")
            print("")
        except BrokenPipeError:
            # Allows piping to `head`/`less` without noisy tracebacks.
            return

    parser = argparse.ArgumentParser(description="Perform photometry operations.")
    parser.add_argument(
        "--config-help",
        dest="config_help",
        action="store_true",
        help="Print all keys in databases/default_input.yml with inline explanations, then exit.",
        default=False,
    )
    parser.add_argument("-f", dest="filepath", type=str, help="Filepath of FITS file")
    parser.add_argument(
        "-c", dest="input_yaml", type=str, help="Path to the input YAML file."
    )
    parser.add_argument(
        "-temp",
        dest="prepare_template",
        action="store_true",
        help="Flag to prepare a template.",
        default=False,
    )
    args = parser.parse_args()

    if bool(getattr(args, "config_help", False)):
        _print_default_yaml_help()
        return None

    #  Access Parsed Arguments
    science_file = args.filepath  # Path to the science FITS file
    input_yaml_loc = args.input_yaml  # Path to the input YAML file
    prepare_template = (
        args.prepare_template
    )  # If True, run in template-preparation mode

    import time

    import datetime
    import logging
    import shutil

    import uuid
    import warnings
    from collections import OrderedDict
    from pathlib import Path

    # Scientific libraries for astronomical data processing.
    import numpy as np
    import pandas as pd
    import yaml
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata.utils import Cutout2D
    from astropy.stats import sigma_clipped_stats
    from astropy.time import Time
    import astropy.wcs as WCS
    from astropy import units as u
    from astropy.utils.exceptions import AstropyWarning
    from photutils.centroids import centroid_com, centroid_2dg, centroid_sources
    from scipy.spatial import cKDTree

    # Custom modules for specific photometry tasks.
    from aperture import Aperture
    from catalog import Catalog
    from fwhm import Find_FWHM
    from functions import (
        AutophotYaml,
        border_msg,
        convert_to_mjd_astropy,
        get_instrument_config,
        load_telescope_config,
        dict_to_string_with_hashtag,
        get_header,
        get_image,
        get_image_and_header,
        pix_dist,
        quadrature_add,
        SuppressStdout,
        beta_aperture,
        beta_psf,
        log_exception,
        odd,
        ColoredLevelFormatter,
    )
    from limits import Limits
    from plot import Plot
    from psf import PSF
    from templates import Templates
    from wcs import WCSSolver, get_wcs
    from zeropoint import Zeropoint
    from background import BackgroundSubtractor
    from utils import run_IDC

    from cosmic import RemoveCosmicRays
    from utils.run_sex import SExtractorWrapper

    from scipy.cluster.hierarchy import fclusterdata

    """
    Perform photometry operations.

    Args:
        -f (str): Filepath of FITS file.
        -c (str): Path to the input YAML file.
        -temp (bool): Flag to prepare a template. Default is False.

    Returns:
        None
    """

    # Experiment with this - WCS -> pixel index
    index = 0
    start = time.time()

    #  Filter Out Astropy Warnings
    # Suppresses Astropy warnings to keep the log clean.
    warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

    #  Load Input YAML File
    # Loads the YAML configuration file which contains parameters for the pipeline.
    with open(input_yaml_loc, "r") as file:
        input_yaml = yaml.safe_load(file)

    # Worker count for per-image parallelism (aperture sources, limits). When nCPU is 1,
    # use 1 worker so we do one source at a time; otherwise respect nCPU or leave default.
    _ncpu = input_yaml.get("nCPU") or input_yaml.get("nCpu") or input_yaml.get("ncpu")
    n_jobs = max(1, int(_ncpu)) if _ncpu is not None else None

    #  Helper Function: Update Target Pixel Coordinates
    # Updates the target's pixel coordinates after any changes to the WCS.
    # Uses origin=0 (0-based) so pixel coords match numpy array indexing everywhere.
    def update_target_pixel_coords(input_yaml, imageWCS, index=0):
        """Update target pixel coordinates after WCS changes. index is WCS origin (0=0-based)."""
        target_x_pix, target_y_pix = imageWCS.all_world2pix(
            input_yaml["target_ra"],
            input_yaml["target_dec"],
            index,
        )
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix
        return target_x_pix, target_y_pix

    try:
        #  Set Up Working Directory and Output
        # Retrieves the working directory and output directory name from the YAML configuration.
        wdir = input_yaml["fits_dir"]
        output_dir_suffix = "_" + input_yaml["outdir_name"]

        # Creates the new output directory path.
        fits_basename = os.path.basename(wdir)
        reduced_dir_name = fits_basename + output_dir_suffix
        new_output_dir = os.path.join(os.path.dirname(wdir), reduced_dir_name)

        # Creates the new output directory if it does not exist.
        Path(new_output_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(
            new_output_dir
        )  # Change the current working directory to the new output directory.

        #  Check for Existing Output
        # Checks if the output file already exists to avoid reprocessing.
        filename_with_ext = os.path.basename(science_file)
        base, file_extension = os.path.splitext(filename_with_ext)
        base = (
            base.replace(" ", "_")
            .replace(".", "_")
            .replace("_APT", "")
            .replace("_ERROR", "")
        )
        wdir = input_yaml["fits_dir"]
        output_dir_suffix = "_" + input_yaml["outdir_name"]
        fits_basename = os.path.basename(wdir)
        reduced_dir_name = fits_basename + output_dir_suffix
        new_output_dir = os.path.join(os.path.dirname(wdir), reduced_dir_name)
        cur_dir = os.path.join(new_output_dir, base)
        output_csv_path = os.path.join(cur_dir, "output.csv")
        calibration_file = os.path.join(cur_dir, "calib.csv")

        #  Store Base Filename in YAML
        # Stores the base filename without any extension in the YAML configuration.
        input_yaml["base"] = base

        #  Skip if output exists and we are not restarting (resume mode).
        # restart=True (default): reprocess all files (redo even if output.csv exists).
        # restart=False: skip files that already have output.csv (only process new/unprocessed).
        if (
            os.path.exists(output_csv_path)
            and not input_yaml.get("restart", True)
            and not prepare_template
        ):
            logging.info(
                border_msg(
                    f"Skipping {science_file}:\nOutput.csv already exists in {cur_dir}\nSet 'restart' to True to reprocess all.",
                    body="~",
                    corner="~",
                )
            )
            return None

        #  Set Current Directory Based on Template Flag
        # Sets the current directory based on whether a template is being prepared.
        if prepare_template:
            cur_dir = os.path.dirname(science_file)
        else:
            # Creates a subdirectory system based on the input directory structure.
            root = os.path.dirname(science_file)
            sub_dirs = root.replace(wdir, "").split("/")
            sub_dirs = [i.replace("_APT", "").replace(" ", "_") for i in sub_dirs]
            cur_dir = new_output_dir
            for i in range(len(sub_dirs)):
                if i:  # If the directory is not blank
                    subdir_path = os.path.join(cur_dir, sub_dirs[i] + "_APT")
                    Path(subdir_path).mkdir(parents=True, exist_ok=True)
                    cur_dir = subdir_path
            # Finally, creates a folder with the filename as its name.
            cur_dir = os.path.join(cur_dir, input_yaml["base"])
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

        #  Set Up Logging
        # Closes any existing logging handlers to avoid duplicate logs.
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)  # Explicitly remove the handler

        # Sets up logging to file with improved configuration.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=os.path.join(cur_dir, f"LOG_{input_yaml['base']}.log"),
            filemode="w",
            encoding="utf-8",  # Explicit encoding for better compatibility
            force=True,  # Ensures this overrides any existing config
        )

        # Creates a console handler with improved settings.
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # Creates a formatter with more detailed information.
        formatter = ColoredLevelFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            use_color=True,
        )
        console.setFormatter(formatter)

        # Adds the handler to the root logger.
        logging.getLogger("").addHandler(console)

        # Optionally sets the root logger level explicitly.
        logging.getLogger("").setLevel(logging.INFO)

        # Optionally adds exception handling.
        logging.raiseExceptions = (
            False  # Prevents logging errors from crashing the program.
        )

        # =============================================================================
        # Helper Function: Shorten Filename if Needed
        # =============================================================================

        def shorten_filename_if_needed(original_path, max_length=255):
            """
            Checks if the full file path exceeds max_length.
            If so, creates a copy with a short random name in the same folder.

            Returns:
                short_path (Path): Path to the shortened file (or original if no change)
                was_shortened (bool): Whether the file was shortened
            """
            original_path = Path(original_path).resolve()
            if len(str(original_path)) <= max_length:
                return original_path, False

            # Generates a new name in the same directory.
            parent_dir = original_path.parent
            short_name = f"image_{uuid.uuid4().hex[:8]}{original_path.suffix}"
            short_path = parent_dir / short_name

            # Copies the original file to the new short-named version.
            shutil.copy2(original_path, short_path)
            return short_path, True

        #  Handle Template or Science File
        # Handles the template or science file based on the prepare_template flag.
        replaced = False
        if prepare_template:
            fpath = science_file
            if os.path.exists(fpath + ".original"):
                # Replaces the template file with the original.
                replaced = True
                os.remove(fpath)
                shutil.copyfile(fpath + ".original", fpath)
            elif not os.path.exists(fpath.replace(".original", "")):
                # Only pre-reduced file found - copy and save.
                logging.info("Pre-reduced file found; copying and saving.")
                shutil.copyfile(fpath, fpath.replace(".original", ""))
                fpath = fpath.replace(".original", "")
            elif ".original" in fpath:
                # Copies the template for recovery.
                shutil.copyfile(fpath, fpath.replace(".original", ""))
                fpath = fpath.replace(".original", "")
            else:
                logging.info("Copying template for recovery.")
                shutil.copyfile(fpath, fpath + ".original")
        else:
            # Copies the new file to the new directory.
            fpath = os.path.join(cur_dir, base + "_APT" + file_extension).replace(
                " ", "_"
            )
            shutil.copyfile(science_file, fpath)

        # Shortens the filename if it is too long.
        fpath, was_shortened = shorten_filename_if_needed(fpath)

        #  Set Up File Paths and Logging
        # Updates the input YAML with the file path and sets up logging.
        input_yaml["fpath"] = fpath
        base_filename = os.path.basename(fpath)
        write_dir = (cur_dir + "/").replace(" ", "_")
        input_yaml["write_dir"] = write_dir
        logging.info(border_msg(f"Processing file: {base_filename}"))
        logging.info("Full path: %s", fpath)
        logging.info("Start time: %s", datetime.datetime.now())
        if was_shortened:
            logging.info(
                "Original filename was too long; copied to a temporary location."
            )
        if replaced:
            logging.info(
                "Pre-reduced file found; replacing template file with original."
            )

        # When processing a template, ensure TELESCOP/INSTRUME/FILTER exist (e.g. after restore from .original)
        if prepare_template and "templates" in os.path.normpath(fpath):
            try:
                with fits.open(fpath, mode="update") as hdul:
                    h = hdul[0].header
                    if not h.get("TELESCOP") or not str(h.get("TELESCOP", "")).strip():
                        norm = os.path.normpath(fpath)
                        if (
                            "gp_template" in norm
                            or "rp_template" in norm
                            or "ip_template" in norm
                            or "up_template" in norm
                        ):
                            tele, inst = "SDSS", "SDSS"
                            band = "g"
                            if "up_template" in norm:
                                band = "u"
                            elif "gp_template" in norm:
                                band = "g"
                            elif "rp_template" in norm:
                                band = "r"
                            elif "ip_template" in norm:
                                band = "i"
                            h["TELESCOP"], h["INSTRUME"], h["FILTER"] = tele, inst, band
                        else:
                            for folder, (tele, inst, band) in [
                                ("J_template", "2MASS", "2MASS", "J"),
                                ("H_template", "2MASS", "2MASS", "H"),
                                ("K_template", "2MASS", "2MASS", "K"),
                            ]:
                                if folder in norm:
                                    h["TELESCOP"], h["INSTRUME"], h["FILTER"] = (
                                        tele,
                                        inst,
                                        band,
                                    )
                                    break
                        hdul.flush()
            except Exception as e:
                logging.debug("Could not ensure template headers: %s", e)

        # =============================================================================
        # Image and Header Processing
        # =============================================================================
        #  Load Image and Header Data (single open to avoid repeated file I/O)
        image, header = get_image_and_header(fpath)

        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(float)
            fits.writeto(fpath, image, header, overwrite=True)

        #  Extract Instrument, Telescope, and Filter metadata
        telescope_key = "TELESCOP"
        instrument_key = "INSTRUME"
        filter_key = "FILTER"
        telescope = header.get(telescope_key)
        instrument = header.get(instrument_key)
        filt = header.get(filter_key)

        # For template images that may lack TELESCOP/INSTRUME/FILTER, set a
        # generic telescope/instrument and infer the filter from the template
        # folder name (e.g. .../V_template/ -> FILTER='V'), then write them into
        # the FITS header before enforcing the requirement.
        is_template = "templates" in os.path.normpath(fpath)
        if is_template and (
            not telescope
            or str(telescope).strip() == ""
            or not instrument
            or str(instrument).strip() == ""
            or not filt
            or str(filt).strip() == ""
        ):
            # Generic placeholders for templates without explicit telescope/instrument.
            if not telescope or str(telescope).strip() == "":
                header[telescope_key] = "Unknown_reference"
                telescope = "Unknown_reference"
            if not instrument or str(instrument).strip() == "":
                header[instrument_key] = "Unknown_instrument"
                instrument = "Unknown_instrument"

            # Infer filter from the template folder name, e.g. V_template, gp_template.
            if not filt or str(filt).strip() == "":
                norm_path = os.path.normpath(fpath)
                filt_candidate = None
                for part in norm_path.split(os.sep):
                    if part.endswith("_template"):
                        base = part.split("_")[0]
                        # gp_template -> g, zp_template -> z, otherwise use as-is.
                        if base.endswith("p") and len(base) > 1:
                            filt_candidate = base[:-1]
                        else:
                            filt_candidate = base
                        break
                if filt_candidate:
                    header[filter_key] = filt_candidate
                    filt = filt_candidate

            # Persist inferred keywords back into the template file
            fits.writeto(fpath, image, header, overwrite=True)

        if not telescope or str(telescope).strip() == "":
            raise ValueError(
                f"FITS header missing or empty TELESCOP keyword. Image: {fpath}. "
                "All images must have TELESCOP and INSTRUME header keywords."
            )
        if not instrument or str(instrument).strip() == "":
            raise ValueError(
                f"FITS header missing or empty INSTRUME keyword. Image: {fpath}. "
                "All images must have TELESCOP and INSTRUME header keywords."
            )
        telescope = str(telescope).strip()
        instrument = str(instrument).strip()

        #  Load Telescope Configuration from YAML (built-in merged if telescope.yml missing; supports INSTRUME only)
        telescope_data = load_telescope_config(input_yaml["wdir"])
        instrument_key, telescope_config = get_instrument_config(
            telescope_data, telescope, instrument
        )
        if telescope_config is None:
            if is_template:
                # Templates may not have a corresponding entry in telescope.yml.
                # Use generic defaults and rely primarily on WCS for pixel scale.
                logging.warning(
                    "Template image %s has no telescope.yml entry for (%s, %s); "
                    "using generic defaults.",
                    fpath,
                    telescope,
                    instrument,
                )
                telescope_config = {
                    "Name": f"{telescope}+{instrument}",
                    "gain": "GAIN",
                    "saturate": "SATURATE",
                    "readnoise": "RDNOISE",
                    "airmass": "AIRMASS",
                    "mjd": "MJD-OBS",
                    "date": "DATE-OBS",
                    "exptime": "EXPTIME",
                }
            else:
                logging.error(
                    "No configuration found for telescope %s, instrument %s.",
                    telescope,
                    instrument,
                )
                raise ValueError(
                    f"No configuration found for telescope: {telescope}, instrument: {instrument}"
                )

        #  Handle Modified Julian Date (MJD)
        # Try telescope.yml mjd keyword, then common alternates, then date conversion.
        mjd_key = telescope_config.get("mjd", "MJD-OBS")
        date_key = telescope_config.get("date", "DATE-OBS")
        date_mjd = np.nan
        try:
            if mjd_key and mjd_key != "not_given_by_user" and mjd_key in header:
                mjd_value = header[mjd_key]
                date_mjd = float(mjd_value) if mjd_value is not None else np.nan
        except (TypeError, ValueError):
            pass
        if not np.isfinite(date_mjd) or date_mjd == 0:
            for alt in ("MJD", "OBSMJD", "MJDSTART", "MJD-OBS", "MJD_OBS"):
                if alt != mjd_key and alt in header:
                    try:
                        mjd_value = header[alt]
                        date_mjd = float(mjd_value) if mjd_value is not None else np.nan
                        if np.isfinite(date_mjd) and date_mjd != 0:
                            logging.info("MJD from keyword %r: %.3f", alt, date_mjd)
                            break
                    except (TypeError, ValueError):
                        pass
        if not np.isfinite(date_mjd) or date_mjd == 0:
            try:
                if date_key and date_key != "not_given_by_user" and date_key in header:
                    date_iso = header[date_key]
                    date_mjd = convert_to_mjd_astropy(date_iso)
            except Exception:
                pass
        if telescope == "MPI-2.2" and "TDP-MID" in header:
            date_mjd = header["TDP-MID"]
            logging.info("MPI-2.2MM detected (TDP-MID); MJD: %.3f", date_mjd)
        if not np.isfinite(date_mjd) or date_mjd == 0:
            date_mjd = Time.now().mjd
            logging.info(
                "Invalid MJD detected; setting MJD to today's value: %.1f", date_mjd
            )

        #  Handle Gain
        gain_key = telescope_config.get("gain", "GAIN")
        gain = header.get(gain_key, 1)

        try:
            gain = float(gain)
        except (ValueError, TypeError):
            gain = 1
            logging.warning("Invalid gain in header; defaulting to 1.")

        if gain == 0:
            gain = 1
            logging.warning("Gain is 0; defaulting to 1.")

        image_median = np.median(image)
        low_median_threshold = 1e-6
        high_gain_threshold = 1e6

        if image_median < low_median_threshold and gain > high_gain_threshold:
            logging.warning(
                "Applying gain %.1e to image (low median %.1e).", gain, image_median
            )
            image = image * gain
            fits.writeto(fpath, image, header, overwrite=True)
            gain = 1
            logging.warning("Gain reset to 1 after application.")

        #  Handle Saturation (telescope.yml may use saturate: not_given_by_user)
        # Priority:
        #   1) telescope.yml explicit keyword if valid and present in header
        #   2) fallback to standard 'SATURATE' header key if present
        #   3) otherwise, treat as effectively no hard saturation (inf)
        saturate_key = telescope_config.get("saturate", "SATURATE")
        if (
            saturate_key
            and saturate_key != "not_given_by_user"
            and saturate_key in header
        ):
            saturate = header[saturate_key]
        elif "SATURATE" in header:
            saturate = header["SATURATE"]
        else:
            # No usable saturation keyword provided; assume effectively no hard
            # saturation so downstream masks do not classify the entire frame
            # as saturated.
            saturate = np.inf

        input_yaml["saturate"] = saturate
        # FITS headers cannot store inf; only write saturate when it is finite.
        if np.isfinite(saturate):
            header["saturate"] = float(saturate)

        #  Handle Read Noise and Airmass (telescope.yml may use not_given_by_user)
        rn_key = telescope_config.get("readnoise", "RDNOISE")
        readnoise = (
            header.get(rn_key, 0) if rn_key and rn_key != "not_given_by_user" else 0
        )
        am_key = telescope_config.get("airmass", "AIRMASS")
        airmass = (
            header.get(am_key, 1) if am_key and am_key != "not_given_by_user" else 1
        )

        input_yaml["readnoise"] = readnoise
        header["RDNOISE"] = readnoise

        #  Handle Exposure Time
        exposure_time = header.get(telescope_config.get("exptime", "EXPTIME"), 30)
        if exposure_time == 0:
            logging.info("Exposure time is zero; defaulting to 30 s.")
            exposure_time = 30
        header["exptime"] = exposure_time

        # Placeholder
        ImageFWHM = None

        # =============================================================================
        #          Populate Output Dictionary
        # =============================================================================

        # Populates the output dictionary with basic metadata.
        output = OrderedDict(
            {
                "fname": fpath,
                "telescope": telescope_config["Name"],
                "TELESCOP": telescope,
                "INSTRUME": instrument_key,
                "instrument": instrument,
                "mjd": date_mjd,
                "airmass": airmass,
            }
        )

        #  Update Input YAML with Instrument Metadata
        # Updates the input YAML with instrument metadata.
        input_yaml.update(
            {
                "tele": telescope,
                "inst": instrument,
                "instrument_key": instrument_key,
            }
        )

        # =============================================================================
        # Filter and Pixel Scale
        # =============================================================================
        #  Find Correct Filter Key
        # Attempts to find the correct filter key from the header.

        avoid_keys = [""]

        if is_template:
            # For templates, trust the FILTER keyword (or common variants) directly.
            for fk in ("FILTER", "FILTER1", "FILTER2"):
                if fk in header and str(header[fk]).strip().lower() not in avoid_keys:
                    input_yaml["filter_key"] = fk
                    raw_filter = str(header[fk]).strip()
                    imageFilter = raw_filter
                    break
            else:
                raise Exception(
                    f"Template image {fpath} is missing a usable FILTER keyword."
                )
            # If we have a telescope.yml mapping for this telescope/instrument, apply it for templates too.
            try:
                mapping = telescope_data[telescope][instrument_key][instrument]
                if isinstance(mapping, dict):
                    _meta = {
                        "Name",
                        "filter_key_0",
                        "mjd",
                        "date",
                        "gain",
                        "saturate",
                        "readnoise",
                        "airmass",
                        "exptime",
                        "pixel_scale",
                    }
                    m = mapping.get(imageFilter)
                    if m is None:
                        for k, v in mapping.items():
                            if (
                                k not in _meta
                                and isinstance(v, str)
                                and k.strip().upper() == imageFilter.strip().upper()
                            ):
                                m = v
                                break
                    if m is not None:
                        imageFilter = m
            except Exception:
                pass
        else:
            open_filter = False
            found_correct_key = False
            filter_keys = [
                i
                for i in list(telescope_data[telescope][instrument_key][instrument])
                if i.startswith("filter_key_")
            ]
            _inst = telescope_data[telescope][instrument_key][instrument]
            _filter_meta = {
                "Name",
                "filter_key_0",
                "mjd",
                "date",
                "gain",
                "saturate",
                "readnoise",
                "airmass",
                "exptime",
                "pixel_scale",
            }
            for filter_header_key in filter_keys:
                header_key_candidate = _inst[filter_header_key]
                if header_key_candidate not in header:
                    continue
                header_val = str(header[header_key_candidate]).strip()
                if header_val.lower() in avoid_keys:
                    open_filter = True
                    continue
                # If the header value is a known filter mapping (exact or case-insensitive), we found our key
                if header_val in _inst:
                    found_correct_key = True
                    break
                if any(
                    k not in _filter_meta
                    and isinstance(v, str)
                    and k.strip().upper() == header_val.strip().upper()
                    for k, v in _inst.items()
                ):
                    found_correct_key = True
                    break

            if not found_correct_key and open_filter:
                raise Exception("Cannot find correct filter keyword")

            #  Get Image Filter
            # Retrieves the image filter from the header.
            if found_correct_key:
                input_yaml["filter_key"] = telescope_data[telescope][instrument_key][
                    instrument
                ][filter_header_key]
                raw_filter = str(header[input_yaml["filter_key"]]).strip()
                # Use telescope.yml mapping (e.g. Z_SPECIAL -> z); try exact, then case variants
                mapping = telescope_data[telescope][instrument_key][instrument]
                _meta = {
                    "Name",
                    "filter_key_0",
                    "mjd",
                    "date",
                    "gain",
                    "saturate",
                    "readnoise",
                    "airmass",
                    "exptime",
                    "pixel_scale",
                }
                imageFilter = mapping.get(raw_filter)
                if imageFilter is None:
                    for k, v in mapping.items():
                        if (
                            k not in _meta
                            and isinstance(v, str)
                            and k.strip().upper() == raw_filter.strip().upper()
                        ):
                            imageFilter = v
                            break
                if imageFilter is None:
                    imageFilter = raw_filter
            else:
                # Fallback: no mapping found; use a common header key directly when present
                for fk in ("FILTER", "FILTER1", "FILTER2"):
                    if (
                        fk in header
                        and str(header[fk]).strip().lower() not in avoid_keys
                    ):
                        input_yaml["filter_key"] = fk
                        imageFilter = str(header[fk]).strip()
                        break
                else:
                    raise Exception("Cannot determine image filter from header.")

        # =============================================================================
        # Pixel Scale
        # =============================================================================
        # Prefer pixel scale from existing WCS when available; otherwise fall back
        # to the telescope.yml value (science images) or leave None (templates).

        pixel_scale = None
        try:
            with SuppressStdout():
                imageWCS = get_wcs(header)  # WCS values, may raise if no valid WCS
                xy_pixel_scales = WCS.utils.proj_plane_pixel_scales(imageWCS)
                if xy_pixel_scales is not None and len(xy_pixel_scales) > 0:
                    pixel_scale_candidate = (
                        float(xy_pixel_scales[0]) * 3600.0
                    )  # arcsec/pixel
                else:
                    pixel_scale_candidate = np.nan
        except Exception as e:
            log_exception(e)
            pixel_scale_candidate = np.nan

        # Accept WCS-derived value only if it looks sensible
        if np.isfinite(pixel_scale_candidate) and 0 < pixel_scale_candidate <= 5:
            pixel_scale = pixel_scale_candidate
            logging.info("Pixel scale from WCS: %.3f arcsec/pixel", pixel_scale)
        else:
            if not is_template:
                # Fallback: use telescope.yml pixel_scale if defined
                ps = telescope_data[telescope][instrument_key][instrument].get(
                    "pixel_scale"
                )
                if ps is not None:
                    try:
                        pixel_scale = float(ps)
                        logging.info(
                            "Pixel scale from telescope.yml: %.3f arcsec/pixel",
                            pixel_scale,
                        )
                    except Exception:
                        pixel_scale = None
            if pixel_scale is None:
                logging.warning(
                    "Could not determine pixel scale from WCS or telescope.yml; leaving as None."
                )

        #  Special Case for MPI+GROND in the IR
        # Handles special cases for MPI+GROND in the infrared, without overriding pixel scale.
        if telescope == "MPI-2.2":
            if "BACKMEAN" in header:
                backmean_val = float(header["BACKMEAN"])
                image += backmean_val
                logging.info(
                    "MPI-2.2 / GROND: added BACKMEAN = %.3e to image (restoring raw level before our background subtraction).",
                    backmean_val,
                )
            if imageFilter in ["J", "H", "K"]:
                IR_gain_key = f"{imageFilter}_GAIN"
                if IR_gain_key in header:
                    logging.info(
                        "Detected GROND IR; setting GAIN key to %s", IR_gain_key
                    )
                    header["gain"] = header[IR_gain_key]

        #  Update WCS Pixel Scale
        # Updates the pixel scale in the input YAML.
        input_yaml["wcs"]["pixel_scale"] = pixel_scale

        #  Update Target Coordinates
        # Updates the target coordinates if provided.
        if input_yaml["target_name"] is not None:
            target_ra = input_yaml["target_ra"]
            target_dec = input_yaml["target_dec"]
            target_coords = SkyCoord(
                target_ra, target_dec, unit=(u.deg, u.deg), frame="fk5", equinox="J2000"
            )
            input_yaml["target_ra"] = target_coords.ra.degree
            input_yaml["target_dec"] = target_coords.dec.degree
        elif (
            input_yaml["target_ra"] is not None and input_yaml["target_dec"] is not None
        ):
            target_coords = SkyCoord(
                input_yaml["target_ra"],
                input_yaml["target_dec"],
                unit=(u.deg, u.deg),
                frame="fk5",
                equinox="J2000",
            )
            input_yaml["target_ra"] = target_coords.ra.degree
            input_yaml["target_dec"] = target_coords.dec.degree
        else:
            try:
                if input_yaml.get("use_header_radec", False):
                    # TODO: Address this in a future version
                    raise Exception("Not available in current version")
                    target_coords = SkyCoord(
                        header["CAT-RA"],
                        header["CAT-DEC"],
                        unit=(u.hourangle, u.deg),
                        frame="fk5",
                        equinox="J2000",
                    )
                    input_yaml["target_ra"] = target_coords.ra.degree
                    input_yaml["target_dec"] = target_coords.dec.degree
            except Exception:
                logging.info("No RA/DEC keywords found.")

        #  Set Target Name
        # Sets the target name based on available information.
        if not (input_yaml["target_name"] is None):
            input_yaml["target_name"] = (
                input_yaml["target_name"].replace("SN", "").replace("AT", "")
            )
        elif not (input_yaml["target_ra"] is None) and not (
            input_yaml["target_dec"] is None
        ):
            input_yaml["target_name"] = "Transient"
        else:
            input_yaml["target_name"] = "Center of Field"

        #  Update Input YAML with Image and Filter Metadata
        # Updates the input YAML with image and filter metadata.
        input_yaml["imageFilter"] = imageFilter
        input_yaml["pixel_scale"] = pixel_scale
        input_yaml["exposure_time"] = float(exposure_time)
        input_yaml["saturate"] = saturate
        input_yaml["gain"] = gain

        #  Log Telescope and Instrument Metadata
        # Logs the telescope and instrument metadata.

        logging.info("Telescope: %s", telescope)
        logging.info("Instrument: %s", instrument)
        logging.info("Filter: %s", imageFilter)
        logging.info("MJD: %.3f", date_mjd)
        logging.info("Gain: %.3f e/ADU", gain)
        logging.info("Read noise: %.3f e/pixel", readnoise)
        logging.info("Saturation level: %.3f ADU", saturate)
        logging.info("Exposure time: %.1f s", float(exposure_time))

        if pixel_scale:
            logging.info("Pixel scale: %.3f arcsec/pixel", pixel_scale)

            input_yaml["pixel_scale"] = pixel_scale

        date = Time([date_mjd], format="mjd", scale="utc")
        date = date.iso[0].split(" ")[0]
        logging.info("Date of observation: %s", date)

        header["gain"] = gain
        # header["saturate"] = saturate
        header["RDNOISE"] = readnoise

        # =============================================================================
        # Image Preprocessing
        # =============================================================================
        #  Replace Non-Finite Values
        # Replaces non-finite values in the image with a very small number.
        # image[~np.isfinite(image)] = 1e-30
        fits.writeto(fpath, image, header, overwrite=True)

        #  Image Trimming
        # Trims the image to a specified size centered on the target.
        trim_image = input_yaml["preprocessing"].get("trim_image", 0)
        if trim_image > 0:
            try:
                logging.info(
                    border_msg(
                        f"Trimming {base_filename} to {trim_image} arcmin box centered on target"
                    )
                )
                # Use in-memory image and header (already loaded; no need to re-open file).
                imageWCS = get_wcs(header)
                # Creates a cutout of the image centered on the target coordinates.
                cutout = Cutout2D(
                    image.astype(float),
                    target_coords,
                    (trim_image * u.arcmin * 2),  # Convert to arcmin (box size)
                    wcs=imageWCS,
                    mode="partial",
                    fill_value=1e-30,
                )
                # Updates the image and header with the trimmed values.
                image = cutout.data
                header.update(cutout.wcs.to_header())
                # Updates the WCS values after trimming.
                imageWCS = get_wcs(header)
                # Writes the modified image and header back to the FITS file.
                fits.writeto(fpath, image, header, overwrite=True)
                logging.info(f"New image shape after trimming: {image.shape}")
            except Exception as e:
                log_exception(e)
                logging.info("Could not trim image; ignoring the operation.")

        # =============================================================================
        # Image Recropping (if needed)
        # =============================================================================
        #  Image Recropping
        # Recrops the image to exclude uniform rows/columns at the boundaries.
        # Uses in-memory image and header from previous step (no re-open).
        try:
            imageWCS = get_wcs(header)

            # Finds the center and boundaries of the non-uniform region in the image.
            center_y, center_x, top_row, bottom_row, left_col, right_col = Templates(
                input_yaml=input_yaml
            ).find_non_uniform_center(image)

            # Calculates the height and width of the cropped image.
            height = bottom_row - top_row
            width = right_col - left_col

            # Checks if cropping is actually needed.
            if (height < image.shape[0] - 1) or (width < image.shape[1] - 1):
                logging.info(
                    border_msg(
                        f"Recropping {base_filename} to exclude uniform rows/columns at image boundaries"
                    )
                )
                position = (center_x, center_y)  # Cutout2D expects (x, y) position
                size = (height, width)

                # Creates a cutout of the non-uniform region in the image.
                imageCutout = Cutout2D(
                    image,
                    position,
                    size,
                    wcs=imageWCS,
                    mode="partial",
                    fill_value=1e-30,
                )

                # Updates the image and WCS with the cutout values.
                image = imageCutout.data
                imageWCS = get_wcs(imageCutout.wcs.to_header())
                header.update(imageWCS.to_header())

                # Writes the modified image and header back to the FITS file.
                fits.writeto(fpath, image, header, overwrite=True)
                logging.info(f"New image shape after recropping: {image.shape}")

            # Use the in-memory image/header (already updated) to refresh YAML
            # dimensions instead of re-opening the FITS file.
            input_yaml["NAXIS1"] = image.shape[1]
            input_yaml["NAXIS2"] = image.shape[0]
        except Exception as e:
            log_exception(e)
            logging.info("Could not recrop the image; operation ignored.")

        # =============================================================================
        #   Run SExtractor
        # =============================================================================
        # Runs SExtractor to measure the image FWHM and detect sources.
        try:
            ImageFWHM, FWHMSources, scale = SExtractorWrapper(config=input_yaml).run(
                fpath,
                crowded=input_yaml.get("photometry", {}).get("crowded_field", False),
            )
        except Exception as e:
            log_exception(e, "SEXtractor failed - trying pythonic source detection")

            # Measures the image FWHM, isolated sources, and scale.
            ImageFWHM, FWHMSources, scale = Find_FWHM(
                input_yaml=input_yaml
            ).measure_image(
                image=image,
            )

        # =============================================================================
        # Measuring the background statitiscs (don't remove it)
        # =============================================================================
        # Creates a background remover instance.

        bg_remover = BackgroundSubtractor(input_yaml)

        # Removes the background (without plotting).
        result = bg_remover.remove(image, plot=False, fwhm=ImageFWHM)

        # Accesses the results.
        background_surface = result["background"]
        background_rms = result["background_rms"]
        defects_mask = result["defects_mask"]

        logging.info(f"Preliminary FWHM: {ImageFWHM:.1f} pixels")

        header["fwhm"] = ImageFWHM

        # Set WCS profile to "crowded" from initial FWHM/background when crowded_auto is True.
        # Uses source count and density from the same run that produced ImageFWHM/FWHMSources.
        wcs_cfg = input_yaml.get("wcs") or {}
        if wcs_cfg.get("crowded_auto", False):
            profile = str(wcs_cfg.get("profile", "default")).strip().lower()
            if profile in ("auto", "default"):
                n_src = 0
                if FWHMSources is not None and hasattr(FWHMSources, "__len__"):
                    n_src = len(FWHMSources)
                pixel_scale_arcsec = input_yaml.get("pixel_scale") or 0
                if not pixel_scale_arcsec and header.get("CDELT1") is not None:
                    try:
                        pixel_scale_arcsec = abs(float(header["CDELT1"])) * 3600.0
                    except (TypeError, KeyError):
                        pass
                area_sq_arcmin = 0.0
                if pixel_scale_arcsec and pixel_scale_arcsec > 0:
                    ny, nx = image.shape[0], image.shape[1]
                    area_sq_arcmin = (nx * ny) * (pixel_scale_arcsec / 60.0) ** 2
                # Extremely conservative defaults: only the very most crowded
                # images should switch WCS to crowded mode automatically.
                n_min = wcs_cfg.get("crowded_auto_n_sources_min", 1500)
                density_min = wcs_cfg.get("crowded_auto_sources_per_arcmin2_min", 200.0)
                density = (n_src / area_sq_arcmin) if area_sq_arcmin > 0 else 0.0
                is_crowded = n_src >= n_min or (
                    area_sq_arcmin > 0 and density >= density_min
                )
                # Optional: high background RMS (e.g. nebulosity, dense field) can also trigger crowded.
                bg_rms_min = wcs_cfg.get("crowded_auto_background_rms_min")
                if bg_rms_min is not None and background_rms is not None:
                    try:
                        med_rms = float(np.nanmedian(background_rms))
                        if np.isfinite(med_rms) and med_rms >= float(bg_rms_min):
                            is_crowded = True
                    except Exception:
                        pass
                if is_crowded:
                    input_yaml.setdefault("wcs", {})["profile"] = "crowded"
                    logging.info(
                        "WCS crowded_auto: using crowded profile (initial FWHM check: %d sources, %.2f per sq arcmin).",
                        n_src,
                        density if area_sq_arcmin > 0 else 0.0,
                    )
                elif profile == "auto":
                    input_yaml.setdefault("wcs", {})["profile"] = "default"

        # =============================================================================
        #          Cosmic Ray Removal
        # =============================================================================
        # Removes cosmic rays from the image if enabled. Skipped if header has
        # CRAY_RMD or CRSTATUS indicating prior cleaning (see default_input.yml).
        cosmic_rays_mask = np.zeros(image.shape)
        _cr_status = header.get("CRSTATUS")
        _cr_status = (
            _cr_status[0] if isinstance(_cr_status, tuple) else _cr_status
        ) or ""
        already_cleaned = (
            header.get("CRAY_RMD", False) is not False
            or str(_cr_status).strip().lower() == "success"
        )
        if (
            input_yaml["cosmic_rays"].get("remove_cmrays", False)
            and not already_cleaned
        ):
            if telescope == "PS1":
                pass
            else:
                logging.info(
                    border_msg(f"Removing cosmic rays and streaks from {base_filename}")
                )
                use_lacosmic = input_yaml["cosmic_rays"].get("use_lacosmic", False)
                image, cosmic_rays_mask = RemoveCosmicRays(
                    input_yaml=input_yaml,
                    fpath=fpath,
                    image=image,
                    header=header,
                    use_lacosmic=use_lacosmic,
                ).remove(
                    bkg=background_surface,
                    bkg_rms=background_rms,
                    gain=gain,
                    psf_fwhm=ImageFWHM,
                )

        # =============================================================================
        #          Check for Existing WCS
        # =============================================================================
        # Checks if there is an existing WCS in the header.

        existingWCS = False
        updated_header = None

        # Tries to read the existing WCS.
        try:
            with SuppressStdout():
                WCSvalues_old = get_wcs(header)
            existingWCS = True
            logging.info("Pre-existing WCS found in header")
        except Exception as e:
            log_exception(e, "No pre-existing WCS found")

        # Initializes the WCSSolver object.
        with SuppressStdout():
            imageWCS_obj = WCSSolver(
                fpath=fpath,
                image=image,
                header=header,
                default_input=input_yaml,
            )

        # =============================================================================
        #         Attempts WCS redo if requested.
        # =============================================================================
        # When apply_solved_to_fits is False, run the solver (e.g. for scale) but do NOT
        # write the solved WCS into the science FITS. Keeps original WCS for alignment/
        # subtraction (often better when input is in tmp and output in tmp_reduced).
        apply_solved_to_fits = input_yaml["wcs"].get("apply_solved_to_fits", True)
        wcs_updated = False
        while input_yaml["wcs"].get("redo_wcs", False):
            with SuppressStdout():
                updated_header = imageWCS_obj.plate_solve(
                    solvefield_exe=input_yaml["wcs"].get("solve_field_exe_loc"),
                )
            if updated_header is None or (
                isinstance(updated_header, float) and np.isnan(updated_header)
            ):
                logging.info("Plate solve returned NaN or None")
                if existingWCS:
                    logging.info("Falling back to pre-existing WCS")
                    header.update(WCSvalues_old.to_header())
                    break
                else:
                    raise Exception(
                        "No fallback WCS available; header left without WCS"
                    )
            else:
                logging.info("Plate solve successful")
                if apply_solved_to_fits:
                    header = updated_header
                    fits.writeto(fpath, image, header, overwrite=True)
                    logging.info("Updated header written to file after WCS update")
                    wcs_updated = True
                else:
                    # Use solved WCS only to update pixel_scale in YAML; leave FITS header unchanged for better subtraction
                    try:
                        _solved_wcs = get_wcs(updated_header)
                        _xy = WCS.utils.proj_plane_pixel_scales(_solved_wcs)
                        input_yaml["pixel_scale"] = float(_xy[0] * 3600)
                        logging.info(
                            "apply_solved_to_fits=False: keeping original WCS in FITS; updated pixel_scale in config only"
                        )
                    except Exception:
                        pass
                break  # Exits after one successful attempt.
        # Use current header (solved or original) for rest of pipeline
        imageWCS = get_wcs(header)

        # Gets the pixel scale in arcseconds.
        xy_pixel_scales = WCS.utils.proj_plane_pixel_scales(imageWCS)
        pixel_scale = xy_pixel_scales[0] * 3600

        # Sets the range for which the PSF model can move around.
        input_yaml["dx"] = np.ceil(ImageFWHM)
        input_yaml["dy"] = np.ceil(ImageFWHM)

        # Updates the pixel scale in the input YAML.
        input_yaml["pixel_scale"] = pixel_scale

        # =============================================================================
        # WCS Refinement (optional: enable via wcs.refine_after_solve if needed)
        # =============================================================================
        if input_yaml.get("wcs", {}).get("refine_after_solve", False) and wcs_updated:
            try:
                fpath = run_IDC.ImageDistortionCorrector(
                    input_yaml=input_yaml
                ).refine_image(fpath, reference_catalog="GAIA-EDR3")
            except Exception as e:
                log_exception(e, "Issue with template alignment")

        # =============================================================================
        # Variable Sources
        # =============================================================================
        imageWCS = get_wcs(header)

        # Loads variable sources from the input YAML.
        if "variable_sources" in input_yaml:
            variable_sources_lst = input_yaml["variable_sources"]
            variable_sources = pd.DataFrame(
                variable_sources_lst,
                columns=[
                    "RA",
                    "DEC",
                    "OTYPE",
                    "MAIN_ID",
                    "OTYPE_LABEL",
                    "OTYPE_opt",
                    "separation_arcmin",
                    "galdim_majaxis",
                    "galdim_minaxis",
                    "galdim_angle",
                ],
            )
            logging.info(
                f"Loaded {len(variable_sources)} variable sources from input_yaml."
            )
        else:
            variable_sources = pd.DataFrame([])
            logging.info("No variable sources found in input_yaml.")
        # Filters variable sources to only include those within the image boundaries.
        if not variable_sources.empty:
            coords = SkyCoord(
                ra=variable_sources["RA"].values * u.deg,
                dec=variable_sources["DEC"].values * u.deg,
                frame="fk5",
                equinox="J2000",
            )
            # Converts sky coordinates to pixel coordinates using WCS.
            x_pix, y_pix = imageWCS.world_to_pixel(coords)
            # Gets image size.
            height, width = image.shape
            # Creates a mask of sources inside the image boundaries.
            inside_mask = (
                (x_pix >= 0) & (x_pix < width) & (y_pix >= 0) & (y_pix < height)
            )
            # Filters sources inside the image.
            variable_sources = variable_sources.loc[inside_mask].copy()
            variable_sources["x_pix"] = x_pix[inside_mask]
            variable_sources["y_pix"] = y_pix[inside_mask]
            if len(variable_sources) == 0:
                logging.info("No variable sources found within image boundaries.")
            else:
                logging.info(
                    f"Variable sources within image boundaries ({len(variable_sources)}):"
                )
        else:
            logging.info(
                "Variable sources DataFrame is empty; no sources to check for image boundaries."
            )

        # =============================================================================
        #          TNS Position Check
        # =============================================================================
        # Checks the target position using TNS coordinates.

        logging.info(
            border_msg(
                f"TNS position check for {input_yaml.get('target_name', 'Transient')}"
            )
        )
        target_x_expected, target_y_expected = imageWCS.all_world2pix(
            input_yaml["target_ra"], input_yaml["target_dec"], index
        )
        logging.info(
            f"TNS RA/Dec: {input_yaml['target_ra']:.6f}, {input_yaml['target_dec']:.6f}"
        )
        logging.info(
            f"Expected pixel position: ({target_x_expected:.2f}, {target_y_expected:.2f})"
        )

        # Checks the reverse transform.
        ra_back, dec_back = imageWCS.all_pix2world(
            target_x_expected, target_y_expected, index
        )
        offset_arcsec = np.sqrt(
            ((ra_back - input_yaml["target_ra"]) * 3600) ** 2
            + ((dec_back - input_yaml["target_dec"]) * 3600) ** 2
        )
        logging.info(f"WCS round-trip error: {offset_arcsec:.3f} arcsec")

        # =============================================================================
        # Remove the background
        # =============================================================================

        result = bg_remover.remove(
            image,
            header=header,
            plot=True,
            fwhm=ImageFWHM,
            galaxies=variable_sources,
            mask_simbad_galaxies=True,
        )

        background_surface = result["background"]
        background_rms = result["background_rms"]
        defects_mask = result["defects_mask"]

        # Subtracts the background surface from the image.
        image -= background_surface

        # Update saturation to match background-subtracted image units so
        # downstream masks and SExtractor use a consistent threshold.
        saturate_sub = saturate - np.nanmedian(background_surface)
        input_yaml["saturate"] = saturate_sub
        # FITS headers cannot store inf; only write when finite.
        if np.isfinite(saturate_sub):
            header["saturate"] = float(saturate_sub)

        # Writes the modified image and header back to the file.
        fits.writeto(fpath, image, header, overwrite=True)
        # Save the background_rms array with '.weight' inserted before the suffix
        base, ext = os.path.splitext(fpath)
        weight_fpath = f"{base}.weight{ext}"
        fits.writeto(weight_fpath, background_rms, header, overwrite=True)
        # Keep using in-memory image, header (no reload needed)

        # =============================================================================
        # Target Position
        # =============================================================================

        #  Get Target Pixel Location
        # Gets the target pixel location using the WCS.
        imageWCS = get_wcs(header)  # WCS values
        xy_pixel_scales = WCS.utils.proj_plane_pixel_scales(imageWCS)
        pixel_scale = xy_pixel_scales[0] * 3600

        # Set Target Coordinates
        # Determines target (RA, Dec, pixel) from user/TNS/header or image center; logs once at end.
        tname = input_yaml.get("target_name") or "Transient"
        section_title = (
            f"Target position ({tname})"
            if (tname and str(tname).strip() != "Transient")
            else "Target position"
        )
        logging.info(border_msg(section_title))

        if (input_yaml["target_name"] is None) or (
            input_yaml["target_name"] == "Transient"
        ):
            if input_yaml["target_ra"] is None and input_yaml["target_dec"] is None:
                # Use image center when no target information is provided.
                center_pix = (image.shape[1] / 2, image.shape[0] / 2)
                center = imageWCS.all_pix2world([center_pix[0]], [center_pix[1]], index)
                target_coords = SkyCoord(
                    center[0][0],
                    center[1][0],
                    unit=(u.deg, u.deg),
                    frame="fk5",
                    equinox="J2000",
                )
                input_yaml["target_ra"] = target_coords.ra.degree
                input_yaml["target_dec"] = target_coords.dec.degree
                input_yaml["target_x_pix"] = center_pix[0]
                input_yaml["target_y_pix"] = center_pix[1]
                target_x_pix, target_y_pix = center_pix[0], center_pix[1]
            else:
                # User-provided RA/Dec.
                target_ra = input_yaml["target_ra"]
                target_dec = input_yaml["target_dec"]
                target_coords = SkyCoord(
                    target_ra,
                    target_dec,
                    unit=(u.deg, u.deg),
                    frame="fk5",
                    equinox="J2000",
                )
                target_x_pix, target_y_pix = update_target_pixel_coords(
                    input_yaml, imageWCS, index
                )
                if not (
                    (0 <= target_x_pix < image.shape[1])
                    and (0 <= target_y_pix < image.shape[0])
                ):
                    logging.warning(
                        "Target outside image bounds (0,0)–(%d,%d)",
                        image.shape[1],
                        image.shape[0],
                    )
        elif input_yaml["target_name"] is not None:
            try:
                target_ra = input_yaml["target_ra"]
                target_dec = input_yaml["target_dec"]
                target_coords = SkyCoord(
                    target_ra,
                    target_dec,
                    unit=(u.deg, u.deg),
                    frame="fk5",
                    equinox="J2000",
                )
                target_x_pix, target_y_pix = imageWCS.all_world2pix(
                    target_ra, target_dec, index
                )
                input_yaml["target_ra"] = target_ra
                input_yaml["target_dec"] = target_dec
                input_yaml["target_x_pix"] = target_x_pix
                input_yaml["target_y_pix"] = target_y_pix
                if not (
                    (0 <= target_x_pix < image.shape[1])
                    and (0 <= target_y_pix < image.shape[0])
                ):
                    logging.error("Target is OUTSIDE image bounds!")
                    raise Exception(
                        f"Target {input_yaml['target_name']} is outside image boundaries "
                        f"at pixel ({target_x_pix:.1f}, {target_y_pix:.1f})"
                    )
            except Exception as e:
                log_exception(
                    e,
                    f"FAILED to determine target position for {input_yaml['target_name']}",
                )
                raise Exception(
                    f"{e}\nFailed to converge on target position!\n"
                    f"Are you sure {input_yaml['target_name']} is in this image?"
                )
        else:
            try:
                if "RA" not in header or "DEC" not in header:
                    raise KeyError("RA/DEC not found in header")
                target_coords = SkyCoord(
                    header["RA"],
                    header["DEC"],
                    unit=(u.deg, u.deg),
                    frame="fk5",
                    equinox="J2000",
                )
                input_yaml["target_ra"] = target_coords.ra.degree
                input_yaml["target_dec"] = target_coords.dec.degree
                target_x_pix, target_y_pix = update_target_pixel_coords(
                    input_yaml, imageWCS, index
                )
            except Exception as e:
                logging.error("FAILED to get coordinates from FITS header: %s", e)
                logging.exception("NO RA/DEC keywords found")
                raise

        # Single summary line: image size, target, sky coords, pixel, round-trip, bounds.
        nx, ny = image.shape[1], image.shape[0]
        ra = input_yaml["target_ra"]
        dec = input_yaml["target_dec"]
        x = input_yaml["target_x_pix"]
        y = input_yaml["target_y_pix"]
        ra_back, dec_back = imageWCS.all_pix2world(x, y, index)
        offset_arcsec = np.sqrt(
            ((ra_back - ra) * 3600) ** 2 + ((dec_back - dec) * 3600) ** 2
        )
        in_bounds = (0 <= x < nx) and (0 <= y < ny)
        logging.info(
            "Image %d x %d px | Target %s | RA %.6f deg Dec %.6f deg | "
            "Pixel (%.2f, %.2f) | Round-trip %.4f arcsec | %s",
            nx,
            ny,
            input_yaml.get("target_name", "Transient"),
            ra,
            dec,
            x,
            y,
            offset_arcsec,
            "within bounds" if in_bounds else "OUTSIDE bounds",
        )

        # =============================================================================
        # Source Masking
        # =============================================================================
        # Ensure target pixel coordinates are consistent with the final WCS.
        target_x_pix, target_y_pix = update_target_pixel_coords(
            input_yaml, imageWCS, index
        )
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix

        # =============================================================================
        # Template Subtraction
        # =============================================================================
        # Sets up template subtraction if enabled.
        # Creates an instance of the template class.
        template_functions = Templates(input_yaml)

        templateFpath = None
        template_available = False
        science_path_original = fpath
        scienceFpath_cutout = fpath
        if (
            input_yaml["template_subtraction"].get("do_subtraction", False)
            and not prepare_template
        ):
            try:
                # Gets the correct template and puts it in the right place.
                templateFpath = template_functions.get_template()
                try:
                    if templateFpath is None:
                        template_available = False
                        raise Exception("Template file not found")
                    try:
                        templateDir = os.path.dirname(templateFpath)
                    except Exception:
                        raise Exception("Failed to copy template PSF")

                    # Creates the destination directory if it does not exist.
                    dest_dir = os.path.dirname(fpath)

                    # Constructs the full destination path for the template.
                    dest_path = os.path.join(dest_dir, os.path.basename(templateFpath))

                    # Copies the template file.
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    shutil.copyfile(templateFpath, dest_path)

                    # Verifies the copy was successful.
                    if not os.path.exists(dest_path):
                        raise FileNotFoundError(
                            f"Failed to copy template to {dest_path}"
                        )

                    # Updates the templateFpath to point to the new location.
                    templateFpath = dest_path

                    #  Handle weight map
                    # Constructs the weight map path for the template.
                    base, ext = os.path.splitext(templateFpath)
                    template_weight_path = f"{base}.weight{ext}"

                    # Copies the weight map if it exists.
                    if os.path.exists(template_weight_path):
                        # Constructs the destination path for the weight map.
                        dest_weight_path = os.path.join(
                            dest_dir, os.path.basename(template_weight_path)
                        )

                        # Copies the weight map file.
                        if os.path.exists(dest_weight_path):
                            os.remove(dest_weight_path)
                        shutil.copyfile(template_weight_path, dest_weight_path)

                        # Verifies the copy was successful.
                        if not os.path.exists(dest_weight_path):
                            raise FileNotFoundError(
                                f"Failed to copy weight map to {dest_weight_path}"
                            )

                    if not templateFpath:
                        input_yaml["template_subtraction"]["do_subtraction"] = False
                        template_available = False
                        logging.info(
                            border_msg("!!! No template images found - skipping !!!")
                        )
                    else:
                        fpath, templateFpath = template_functions.align(
                            scienceFpath=fpath,
                            templateFpath=templateFpath,
                            method=input_yaml["template_subtraction"][
                                "alignment_method"
                            ],
                        )
                        template_available = True

                except Exception:
                    template_available = False
                    input_yaml["template_subtraction"]["do_subtraction"] = False
                    pass

            except Exception as e:
                log_exception(e, "Template subtraction failed")
                template_available = False

        # =============================================================================
        # Get background statistics after subtraction
        # =============================================================================
        # Reload in case template alignment overwrote fpath (single open for both).
        image, header = get_image_and_header(fpath)
        imageWCS = get_wcs(header)

        result = bg_remover.remove(
            image,
            header=header,
            plot=False,
            fwhm=ImageFWHM,
            galaxies=variable_sources,
            mask_simbad_galaxies=True,
        )

        background_surface = result["background"]
        background_rms = result["background_rms"]
        defects_mask = result["defects_mask"]

        # Save the background_rms array with '.weight' inserted before the suffix
        base, ext = os.path.splitext(fpath)
        weight_fpath = f"{base}.weight{ext}"
        fits.writeto(weight_fpath, background_rms, header, overwrite=True)

        target_x_pix, target_y_pix = update_target_pixel_coords(
            input_yaml, imageWCS, index
        )
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix

        # =============================================================================
        #     Get a Reference catalog
        # =============================================================================
        # Creates an instance of the catalog class.
        Calibrate_Catalog = Catalog(input_yaml=input_yaml)

        # Builds or downloads the catalog of reference sources.
        if input_yaml["catalog"].get("build_catalog", False):
            unCatalogSources = Calibrate_Catalog.build_complete_catalog(
                target_coords=target_coords,
                catalog_list=["refcat", "sdss", "pan_starrs", "apass", "2mass"],
                max_separation=5,
            )
        else:
            unCatalogSources = Calibrate_Catalog.download(
                target_coords=target_coords,
                target_name=input_yaml["target_name"],
                catalogName=input_yaml["catalog"]["use_catalog"],
                catalog_custom_fpath=input_yaml["catalog"].get(
                    "catalog_custom_fpath", None
                ),
            )

        #  Clean Catalog
        # Cleans the catalog by removing sources outside the image borders.
        width = image.shape[1]
        height = image.shape[0]
        border = 11

        CatalogSources = Calibrate_Catalog.clean(
            selectedCatalog=unCatalogSources,
            image_wcs=imageWCS,
            catalogName=input_yaml["catalog"]["use_catalog"],
            get_local_sources=False,
            border=border,
        )
        if CatalogSources is None or len(CatalogSources) == 0:
            logging.warning(
                "No catalog sources available after cleaning; skipping catalog-based calibration for this image."
            )
            CatalogSources = None
        else:
            mask_x = (CatalogSources["x_pix"].values >= border) & (
                CatalogSources["x_pix"].values < width - border
            )
            mask_y = (CatalogSources["y_pix"].values >= border) & (
                CatalogSources["y_pix"].values < height - border
            )
            mask = (mask_x) & (mask_y)
            CatalogSources = CatalogSources[mask]

        # =============================================================================
        # Run source detection on final calibrated image
        # =============================================================================
        # Runs SExtractor to measure the image FWHM and detect sources.

        try:
            sex_crowded = input_yaml.get("photometry", {}).get("crowded_field", False)
            ImageFWHM, FWHMSources, scale = SExtractorWrapper(config=input_yaml).run(
                fpath,
                pixel_scale=pixel_scale,
                masked_sources=variable_sources,
                weight_path=weight_fpath,
                crowded=sex_crowded,
            )
            ImageFWHM, FWHMSources, scale = SExtractorWrapper(config=input_yaml).run(
                fpath,
                pixel_scale=pixel_scale,
                masked_sources=variable_sources,
                weight_path=weight_fpath,
                use_FWHM=ImageFWHM,
                crowded=sex_crowded,
            )
        except Exception as e:
            log_exception(e, "Issue with SExtractor")

            # Measures the image FWHM, isolated sources, and scale.
            ImageFWHM, FWHMSources, scale = Find_FWHM(
                input_yaml=input_yaml
            ).measure_image(
                image=image,
                mask=defects_mask,
            )

        input_yaml["fwhm"] = ImageFWHM
        input_yaml["scale"] = scale

        header["fwhm"] = ImageFWHM

        # Adaptive crowded-field detection (source density + background coverage)
        try:
            ny, nx = image.shape[0], image.shape[1]
            pixel_scale_arcsec = float(
                WCS.utils.proj_plane_pixel_scales(imageWCS)[0] * 3600.0
            )
        except Exception:
            pixel_scale_arcsec = 0.3
        area_sq_arcmin = (
            (ny * nx) * (pixel_scale_arcsec / 60.0) ** 2 if pixel_scale_arcsec else 0.0
        )
        # Use the raw SExtractor detection count (before filtering/downsampling)
        # when estimating how much of the image is covered by stars. The
        # filtered FWHMSources table is optimised for PSF/FWHM work and can
        # significantly undercount in very crowded fields.
        phot_cfg = input_yaml.get("photometry", {}) or {}
        raw_detect_count = int(phot_cfg.get("last_source_detection_raw_count", 0))
        n_src = (
            len(FWHMSources)
            if FWHMSources is not None and hasattr(FWHMSources, "__len__")
            else 0
        )
        # Estimate how much of the image is covered by source profiles using
        # the measured FWHM and image size. This is a crude proxy for the
        # fraction of pixels that remain as "empty" background.
        try:
            fwhm_pix = float(ImageFWHM)
        except Exception:
            fwhm_pix = np.nan
        coverage_est = 0.0
        if np.isfinite(fwhm_pix) and fwhm_pix > 0 and ny > 0 and nx > 0:
            # Assume each source effectively occupies a disk of radius 1.0 x FWHM.
            # For coverage, use the *raw* detection count; this better reflects
            # true stellar packing and makes the crowded-field trigger respond
            # correctly when the image is effectively filled with stars, while
            # being less aggressive on sparse images.
            eff_radius = 1.0 * fwhm_pix
            area_per_source = np.pi * eff_radius**2
            n_cov = raw_detect_count if raw_detect_count > 0 else n_src
            coverage_est = min(1.0, (n_cov * area_per_source) / float(ny * nx))
        background_frac_est = max(0.0, 1.0 - coverage_est)
        max_background_frac = float(
            input_yaml["photometry"].get("crowded_max_background_fraction", 0.7)
        )

        # Crowded mode is intended only for fields where the usable background
        # is heavily suppressed (image essentially filled with stars). Use only
        # the estimated background fraction as the trigger to avoid applying
        # crowded behaviour to sparse images.
        is_crowded = background_frac_est <= max_background_frac
        input_yaml["photometry"]["crowded_field"] = is_crowded
        if is_crowded:
            # Use crowded options everywhere: WCS, background, limits, aperture, etc.
            input_yaml.setdefault("wcs", {})["profile"] = "crowded"
            density = n_src / area_sq_arcmin if area_sq_arcmin > 0 else 0.0
            logging.info(
                "Crowded field metrics: n_src=%d, density=%.2f per sq arcmin, "
                "coverage_est=%.2f, background_frac_est=%.2f",
                n_src,
                density,
                coverage_est,
                background_frac_est,
            )
            logging.info(
                "Crowded field detected (%d sources, %.2f per sq arcmin) -> using crowded options (WCS, aperture, limits, background).",
                n_src,
                density,
            )
            try:
                ImageFWHM, FWHMSources, scale = SExtractorWrapper(
                    config=input_yaml
                ).run(
                    fpath,
                    pixel_scale=pixel_scale,
                    masked_sources=variable_sources,
                    weight_path=weight_fpath,
                    use_FWHM=ImageFWHM,
                    crowded=True,
                )
                input_yaml["fwhm"] = ImageFWHM
                input_yaml["scale"] = scale
                header["fwhm"] = ImageFWHM
                logging.info(
                    "Re-ran SExtractor with crowded-field parameters for full source detection."
                )
            except Exception as e:
                log_exception(
                    e,
                    "SExtractor crowded-field re-run failed; continuing with existing catalog.",
                )

        # =============================================================================
        #  Exclude sources near cosmic rays (if any) and find well-isolated sources
        # =============================================================================
        # If we have no FWHM sources at all (e.g. SExtractor filtering removed
        # everything), skip cosmic-ray-based exclusion and isolation steps.
        if FWHMSources is None or len(FWHMSources) == 0:
            logging.warning(
                "No point sources available after SExtractor; skipping cosmic-ray "
                "exclusion and isolation."
            )
            excluded_sources = pd.DataFrame()
            IsolatedSources = pd.DataFrame(columns=["x_pix", "y_pix"])
        else:
            # Constants
            DISTANCE_THRESHOLD_FACTOR = 2
            distance_threshold = DISTANCE_THRESHOLD_FACTOR * ImageFWHM

            # Build a KDTree for masked pixels for efficient nearest-neighbor search
            masked_pixels = np.argwhere(cosmic_rays_mask)

            if len(masked_pixels) == 0:
                logging.info("No cosmic ray pixels found. Skipping source exclusion.")
                excluded_sources = FWHMSources.iloc[
                    []
                ]  # Empty DataFrame with same columns
            else:
                tree = cKDTree(masked_pixels)
                source_coords = FWHMSources[["x_pix", "y_pix"]].values
                # Query the tree for the minimum distance to any masked pixel for each source
                min_distances, _ = tree.query(
                    source_coords, k=1, distance_upper_bound=distance_threshold
                )

                # Filter sources
                excluded_sources = FWHMSources[min_distances <= distance_threshold]
                FWHMSources = FWHMSources[min_distances > distance_threshold]

                if not excluded_sources.empty:
                    logging.info(
                        "Excluded %d sources due to proximity to removed cosmic rays "
                        "(threshold: %.2f pixels).",
                        len(excluded_sources),
                        distance_threshold,
                    )

            # Find well isolated sources (only reduce when necessary; keep more in crowded fields)
            crowded_field = input_yaml["photometry"].get("crowded_field", False)
            isolation_dist = (scale * 0.5) if crowded_field else scale
            if crowded_field:
                logging.info(
                    "Crowded field: using relaxed isolation (min_distance=%.1f px) "
                    "to retain more sources.",
                    isolation_dist,
                )
            IsolatedSources = Find_FWHM(input_yaml=input_yaml).filter_isolated_sources(
                FWHMSources, min_distance=isolation_dist
            )

        IsolatedSources = Catalog(input_yaml=input_yaml).recenter(
            IsolatedSources, image, boxsize=scale, error=background_rms
        )

        # Converts pixel coordinates of isolated sources to world coordinates.
        IsolatedSources["RA"], IsolatedSources["DEC"] = imageWCS.all_pix2world(
            IsolatedSources.x_pix.values,
            IsolatedSources.y_pix.values,
            index,
        )

        # Creates SkyCoord objects for sources and target.
        target_skycoord = SkyCoord(
            ra=target_coords.ra,
            dec=target_coords.dec,
            unit="deg",
            frame="fk5",
            equinox="J2000",
        )

        source_coords = SkyCoord(
            ra=IsolatedSources["RA"] * u.degree,
            dec=IsolatedSources["DEC"] * u.degree,
            unit="deg",
            frame="fk5",
            equinox="J2000",
        )

        # Calculates separations.
        separations = source_coords.separation(target_skycoord)

        # Filters sources that are at least the specified distance away.
        min_separation = input_yaml["catalog"].get("max_distance", 10) * u.arcmin
        initial_source_count = len(IsolatedSources)
        IsolatedSources = IsolatedSources[separations < min_separation]
        final_source_count = len(IsolatedSources)

        if initial_source_count != final_source_count:
            logging.info(
                f"Using {final_source_count} sources within {min_separation} arcminutes from target"
            )

        # Updates the input YAML with the FWHM and scale.
        imageWCS = get_wcs(header)
        target_x_pix, target_y_pix = update_target_pixel_coords(
            input_yaml, imageWCS, index
        )
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix

        # =============================================================================
        # PSF Model Building
        # =============================================================================

        # Determines if only aperture photometry should be performed.
        if input_yaml["photometry"].get("do_AperturePhotometry", False):
            do_aperture_ONLY = True
        else:
            do_aperture_ONLY = False

        # =============================================================================
        # Aperture Photometry
        # =============================================================================

        #  Measure Aperture Photometry
        # Measures initial aperture photometry for isolated sources.
        aperture_photometry = Aperture(
            input_yaml=input_yaml,
            image=image,
        )
        IsolatedSources = aperture_photometry.measure(
            sources=IsolatedSources,
            plot=False,
            ap_size=1.7 * input_yaml["fwhm"],
            background_rms=background_rms,
            n_jobs=n_jobs,
        )

        IsolatedSources, fit_params, saturation_range = Find_FWHM(
            input_yaml=input_yaml
        ).check_linearity(IsolatedSources)
        IsolatedSources = IsolatedSources.reset_index()

        # Calculates the box size with optimized odd conversion.
        boxsize = odd(min(int(np.ceil(input_yaml["fwhm"])) * 2, 5))
        # boxsize += boxsize % 2 == 0  # Makes odd if even

        # Removes sources near the image edges.
        width = image.shape[1]
        height = image.shape[0]
        mask_x = (IsolatedSources["x_pix"].values >= border) & (
            IsolatedSources["x_pix"].values < width - border
        )
        mask_y = (IsolatedSources["y_pix"].values >= border) & (
            IsolatedSources["y_pix"].values < height - border
        )
        mask = (
            (mask_x)
            & (mask_y)
            & (np.isfinite(IsolatedSources["x_pix"]))
            & (np.isfinite(IsolatedSources["y_pix"]))
        )
        IsolatedSources = IsolatedSources[mask]
        logging.info(f"Number of sources: {len(IsolatedSources)}")

        # Too few isolated stars: cannot build PSF or measure optimum radius; continue with aperture-only.
        min_sources_for_psf = 3
        crowded_field = input_yaml["photometry"].get("crowded_field", False)
        # Crowded fields: use a robust aperture radius of ~1.5 FWHM to reduce blending.
        optimum_radius_crowded = input_yaml["photometry"].get(
            "crowded_optimum_radius_fwhm", 1.5
        )
        if len(IsolatedSources) < min_sources_for_psf:
            logging.warning(
                "Too few isolated sources (%d) to build PSF or measure optimum radius; continuing with aperture-only photometry.",
                len(IsolatedSources),
            )
            do_aperture_ONLY = True
            optimum_radius = optimum_radius_crowded if crowded_field else 1.7
        # Checks if finding the optimum aperture radius is enabled.
        elif input_yaml["photometry"].get("find_optimum_radius", True):
            # Measures the optimum aperture radius.
            try:
                IsolatedSources, optimum_radius, scale = (
                    aperture_photometry.measure_optimum_radius(
                        sources=IsolatedSources,
                        plot=True,
                        background_rms=background_rms,
                        n_jobs=n_jobs,
                        crowded=crowded_field,
                    )
                )
                # Checks if the optimum aperture radius is less than 5 x FWHM.
                if optimum_radius < 5:
                    input_yaml["scale"] = odd(scale)
                else:
                    optimum_radius = optimum_radius_crowded if crowded_field else 1.7
                if crowded_field:
                    optimum_radius = min(optimum_radius, optimum_radius_crowded)
            except Exception as e:
                log_exception(
                    e, "measure_optimum_radius failed; using default aperture radius."
                )
                optimum_radius = optimum_radius_crowded if crowded_field else 1.7
        else:
            optimum_radius = optimum_radius_crowded if crowded_field else 1.7

        input_yaml["photometry"]["aperture_radius"] = odd(
            int(np.ceil(optimum_radius * input_yaml["fwhm"]))
        )
        aperture_radius = float(input_yaml["photometry"]["aperture_radius"])
        logging.info(
            "Aperture radius: %.2f [pixels] (optimum_radius=%.2f FWHM%s)",
            aperture_radius,
            optimum_radius,
            ", crowded" if crowded_field else "",
        )

        # =============================================================================
        # Aperture correction (AP -> total flux)
        # =============================================================================
        # Use isolated, well-behaved stars to measure the aperture correction from the
        # science aperture radius to effectively infinite radius via a curve of growth.
        # The correction is stored in input_yaml so zeropoint calibration can place
        # aperture-based zeropoints on the same total-flux scale as PSF photometry.
        input_yaml["aperture_correction"] = 0.0
        input_yaml["aperture_correction_err"] = 0.0
        try:
            if IsolatedSources is not None and len(IsolatedSources) >= 5:
                logging.info(
                    "Computing aperture correction using %d isolated sources "
                    "at radius %.2f pixels",
                    len(IsolatedSources),
                    aperture_radius,
                )
                ap_corr, ap_corr_err = Aperture(
                    input_yaml=input_yaml,
                    image=image,
                ).compute_aperture_correction(
                    image=image,
                    sources=IsolatedSources,
                    fwhm=ImageFWHM,
                    ap_size=aperture_radius,
                    background_rms=background_rms,
                    plot=input_yaml.get("zeropoint", {}).get("zp_plot", False),
                )
                if np.isfinite(ap_corr):
                    input_yaml["aperture_correction"] = float(ap_corr)
                    input_yaml["aperture_correction_err"] = (
                        float(ap_corr_err) if np.isfinite(ap_corr_err) else 0.0
                    )
                    logging.info(
                        "Aperture correction (AP to total): %.3f +/- %.3f mag (stored for later; "
                        "set apply_aperture_correction: true to apply in pipeline)",
                        input_yaml["aperture_correction"],
                        input_yaml["aperture_correction_err"],
                    )
                else:
                    logging.info(
                        "Aperture correction not reliable; leaving as 0.0 mag."
                    )
            else:
                logging.info(
                    "Too few isolated sources (%s) for aperture correction; "
                    "skipping and leaving correction at 0.0 mag.",
                    "none" if IsolatedSources is None else len(IsolatedSources),
                )
        except Exception as e:
            log_exception(
                e,
                "Aperture correction computation failed; proceeding without correction.",
            )

        # =============================================================================
        # Catalog Sources
        # =============================================================================
        #  Clean and Measure Catalog Sources
        # Cleans and measures the catalog sources.
        border = 1 * scale
        width = image.shape[1]
        height = image.shape[0]
        if CatalogSources is not None and len(CatalogSources) > 0:
            logging.info(f"Found {len(CatalogSources)} sources in field")
            CatalogSources = Calibrate_Catalog.recenter(
                CatalogSources, image, boxsize=scale / 2
            )
            CatalogSources = Calibrate_Catalog.measure(
                selectedCatalog=CatalogSources,
                image=image,
            )
            mask_x = (CatalogSources["x_pix"].values >= border) & (
                CatalogSources["x_pix"].values < width - border
            )
            mask_y = (CatalogSources["y_pix"].values >= border) & (
                CatalogSources["y_pix"].values < height - border
            )
            mask_nans = (
                np.isnan(CatalogSources["x_pix"])
                | np.isnan(CatalogSources["y_pix"])
                | np.isnan(CatalogSources["flux_AP"])
            )
            mask = (mask_x) & (mask_y) & (~mask_nans)
            CatalogSources = CatalogSources[mask]
            CatalogSources = Calibrate_Catalog.downsample_sources_by_position(
                CatalogSources
            )
        else:
            logging.warning(
                "No catalog sources available for photometric calibration; proceeding without catalog-based calibration."
            )

        # =============================================================================
        # PSF Photometry
        # =============================================================================
        #  Convert Pixel Coordinates to World Coordinates
        # Converts pixel coordinates of isolated sources to world coordinates.
        ra_IsolatedSources, dec_IsolatedSources = imageWCS.all_pix2world(
            IsolatedSources.x_pix.values,
            IsolatedSources.y_pix.values,
            index,
        )

        # Calculates the distance of each isolated source from the target.
        dist = pix_dist(
            IsolatedSources.x_pix.values,
            target_x_pix,
            IsolatedSources.y_pix.values,
            target_y_pix,
        )

        # Adds the RA, DEC, and distance columns to the IsolatedSources DataFrame.
        IsolatedSources["RA"] = ra_IsolatedSources
        IsolatedSources["DEC"] = dec_IsolatedSources
        IsolatedSources["dist"] = dist

        # =============================================================================
        # Build PSF Model
        # =============================================================================
        # When template subtraction was used, build the ePSF from the *original* (pre-alignment)
        # science image so that star cutouts are not degraded by resampling (SWarp/AstroAlign).
        epsf_model = None
        PSFSources = None
        if (
            template_available
            and science_path_original != fpath
            and os.path.exists(science_path_original)
            and (not do_aperture_ONLY or prepare_template)
        ):
            try:
                image_orig = get_image(science_path_original)
                header_orig = get_header(science_path_original)
                result_orig = bg_remover.remove(
                    image_orig,
                    header=header_orig,
                    plot=False,
                    fwhm=ImageFWHM,
                    galaxies=variable_sources,
                    mask_simbad_galaxies=True,
                )
                # Use sources that passed linearity and optimum-aperture checks on the current image.
                wcs_orig = get_wcs(header_orig)
                x_orig, y_orig = wcs_orig.all_world2pix(
                    IsolatedSources["RA"].values,
                    IsolatedSources["DEC"].values,
                    index,
                )
                psf_sources_orig = IsolatedSources.copy()
                psf_sources_orig["x_pix"] = x_orig
                psf_sources_orig["y_pix"] = y_orig
                h_orig, w_orig = image_orig.shape
                border_orig = int(np.ceil(scale)) if scale else 10
                in_bounds = (
                    (psf_sources_orig["x_pix"] >= border_orig)
                    & (psf_sources_orig["x_pix"] < w_orig - border_orig)
                    & (psf_sources_orig["y_pix"] >= border_orig)
                    & (psf_sources_orig["y_pix"] < h_orig - border_orig)
                )
                psf_sources_orig = psf_sources_orig.loc[
                    in_bounds
                    & np.isfinite(psf_sources_orig["x_pix"])
                    & np.isfinite(psf_sources_orig["y_pix"])
                ].reset_index(drop=True)
                if len(psf_sources_orig) == 0:
                    logging.warning(
                        "No linearity/optimum-checked sources fall on original image; skipping PSF build from original."
                    )
                else:
                    logging.info(
                        f"Building PSF from original image using {len(psf_sources_orig)} sources that passed linearity and optimum-aperture checks."
                    )
                    epsf_model, PSFSources = PSF(
                        image=image_orig,
                        input_yaml=input_yaml,
                    ).build(
                        psfSources=psf_sources_orig,
                        mask=result_orig["defects_mask"],
                        background_rms=result_orig["background_rms"],
                    )
                    if epsf_model is not None:
                        logging.info(
                            "PSF built from original (pre-alignment) science image to avoid resampling degradation."
                        )
            except Exception as e:
                log_exception(
                    e,
                    "PSF build from original image failed; falling back to aligned image.",
                )
                epsf_model, PSFSources = None, None

        # Builds the PSF model from the current (possibly aligned) image when not already built from original.
        if (
            (not do_aperture_ONLY or prepare_template)
            and epsf_model is None
            and len(IsolatedSources) >= min_sources_for_psf
        ):
            try:
                Calibrate_Catalog = Catalog(input_yaml=input_yaml)
                epsf_model, PSFSources = PSF(
                    image=image,
                    input_yaml=input_yaml,
                ).build(
                    psfSources=IsolatedSources,
                    mask=defects_mask,
                    background_rms=background_rms,
                )
                if epsf_model is None:
                    logging.warning(
                        "PSF build returned no model (e.g. insufficient isolated stars); continuing with aperture-only."
                    )
                    do_aperture_ONLY = True
                    PSFSources = None
            except Exception as e:
                log_exception(
                    e,
                    "PSF build failed (e.g. lack of isolated stars); continuing with aperture-only photometry.",
                )
                epsf_model = None
                PSFSources = None
                do_aperture_ONLY = True

        # Log PSF roundness and run PSF fit on catalog when we have an ePSF (from original or current image).
        if not do_aperture_ONLY or prepare_template:
            if PSFSources is not None:
                roundness = PSFSources["roundness"]
                mean_roundness, median_roundness, std_roundness = sigma_clipped_stats(
                    roundness,
                    sigma=3.0,
                    cenfunc=np.nanmedian,
                    stdfunc="mad_std",
                )
                logging.info(
                    f"PSF sources roundness: {mean_roundness:.2f} +/- {std_roundness:.2f}"
                )
            if not epsf_model:
                logging.info("ePSF not created")
                do_aperture_ONLY = True
                PSFSources = None
            else:
                CatalogSources = PSF(
                    image=image,
                    input_yaml=input_yaml,
                ).fit(
                    epsf_model=epsf_model,
                    sources=CatalogSources,
                    plotTarget=False,
                    ignore_sources=variable_sources,
                    background_rms=background_rms,
                    iterative=False,
                )

        # =============================================================================
        # Zeropoint Calculation
        # =============================================================================
        # Calculates the zeropoint for the image.

        Calibrate_Catalog = Catalog(input_yaml=input_yaml)
        GetZeropoint = Zeropoint(input_yaml=input_yaml)
        CatalogSources = GetZeropoint.clean(sources=CatalogSources)

        ImageColorTerm, ImageColorTermError = GetZeropoint.fit_color_term(
            catalog=CatalogSources
        )

        # Gets the zeropoint and plots the histogram.
        # CatalogSources, image_zeropoint = GetZeropoint.fit_zeropoint(catalog=CatalogSources,
        #                                                    fixed_color_slope= ImageColorTerm,
        #                                                    fixed_color_slope_err= ImageColorTermError,
        #                                                    )

        # Gets the zeropoint and plots the histogram.
        CatalogSources, image_zeropoint = GetZeropoint.estimate_zeropoint(
            catalog=CatalogSources,
            fixed_color_slope=ImageColorTerm,
            fixed_color_slope_err=ImageColorTermError,
        )

        # Updates the header with the zeropoint values.
        for m in image_zeropoint.keys():
            try:
                header[f"ZP_{m}"] = image_zeropoint[m]["zeropoint"]
                header[f"ZP_{m}_e"] = image_zeropoint[m]["zeropoint_error"]
            except Exception as e:
                log_exception(e, f"Issue with {m} zeropoint")
                header[f"ZP_{m}"] = "unknown"
                header[f"ZP_{m}_e"] = "unknown"

        if len(variable_sources) > 0:
            xpix_variable_sources, ypix_variable_sources = imageWCS.all_world2pix(
                variable_sources["RA"].values,
                variable_sources["DEC"].values,
                index,
            )
            variable_sources["x_pix"] = xpix_variable_sources
            variable_sources["y_pix"] = ypix_variable_sources
        # Plots the source check.
        Plot(input_yaml=input_yaml).source_check(
            image=image,
            psfSources=PSFSources,
            catalogSources=CatalogSources,
            FWHMSources=FWHMSources,
            mask=defects_mask,
            variable_sources=variable_sources,
        )
        image_sources = None

        header["aper"] = int(np.ceil(optimum_radius * input_yaml["fwhm"]))
        header["RDNOISE"] = readnoise

        # Writes the modified image and header back to the FITS file.
        fits.writeto(fpath, image, header, overwrite=True)

        # =============================================================================
        # Template Preparation
        # =============================================================================
        #  Prepare Template
        # Prepares a template if the prepare_template flag is set.

        if prepare_template:
            # Checks if the image filter is in ['u', 'g', 'r', 'i', 'z'].
            if imageFilter in ["u", "g", "r", "i", "z"]:
                imageFilter += "p"
            # Creates a new basename for the template file.
            newBasename = imageFilter + "_template.fits"
            newWeightBasename = imageFilter + "_template.weight.fits"

            # Logs the renaming of the template filename.
            logging.info(
                f"Renaming template filename: {fpath} -> {os.path.join(cur_dir, newBasename)}"
            )

            # If the weight map exists, save it with the new basename
            if os.path.exists(weight_fpath):
                fits.writeto(
                    os.path.join(cur_dir, newWeightBasename),
                    get_image(weight_fpath),
                    get_header(weight_fpath),
                    overwrite=True,
                )
                logging.info(
                    f"Saved weight map: {os.path.join(cur_dir, newWeightBasename)}"
                )

            # Saves the cleaned catalog and PSF sources to CSV files.
            CatalogSources.to_csv(os.path.join(cur_dir, "imageCalib_template.csv"))
            IsolatedSources.to_csv(os.path.join(cur_dir, "PSFSources_template.csv"))

            # Writes the modified image and header to the new FITS file.
            fits.writeto(
                os.path.join(cur_dir, newBasename),
                image,
                header,
                overwrite=True,
            )
            logging.info(f"\n\nEnd of {imageFilter} template calibration\n\n")
            return 1

        # =============================================================================
        # Template Subtraction
        # =============================================================================

        # Performs template subtraction if enabled and a template is available.
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix

        MatchingSources = None
        PreformSubtraction = False
        ConsistentSources = None

        if (
            input_yaml["template_subtraction"].get("do_subtraction", False)
            and template_available
            and not prepare_template
        ):
            science_image = get_image(fpath)
            template_image = get_image(templateFpath)
            if science_image.shape != template_image.shape:
                logging.info(
                    border_msg(
                        f"Cropping science and reference image "
                        f"({os.path.basename(fpath)} vs {os.path.basename(templateFpath)})"
                    )
                )
                if 0:
                    # TODO: This can mess up with the WCS alignment.
                    fpath, templateFpath = Templates(input_yaml=input_yaml).crop(
                        scienceFpath=fpath,
                        templateFpath=templateFpath,
                    )

                else:

                    # Loads the science image.
                    science_image = get_image(fpath)
                    science_header = get_header(fpath)
                    science_wcs = get_wcs(science_header)
                    # Gets the image shape and center.
                    ny, nx = science_image.shape
                    science_center_pix = (nx / 2, ny / 2)
                    # Converts the center pixel to sky coordinates.
                    science_center_world = science_wcs.all_pix2world(
                        *science_center_pix, index
                    )
                    center_coord = SkyCoord(
                        ra=science_center_world[0],
                        dec=science_center_world[1],
                        unit="deg",
                        frame="fk5",
                        equinox="J2000",
                    )
                    # Stores the target pixel coordinates.
                    target_x_pix, target_y_pix = update_target_pixel_coords(
                        input_yaml, science_wcs, index
                    )
                    # Creates a cutout for the science image.
                    science_cutout = Cutout2D(
                        data=science_image,
                        position=center_coord,
                        size=(ny, nx),
                        wcs=science_wcs,
                        mode="partial",
                        fill_value=1e-30,
                    )
                    science_header.update(science_cutout.wcs.to_header())
                    # Loads the template image.
                    template_image = get_image(templateFpath)
                    template_header = get_header(templateFpath)
                    template_wcs = get_wcs(template_header)
                    # Creates a cutout for the template image.
                    template_cutout = Cutout2D(
                        data=template_image,
                        position=center_coord,
                        size=(ny, nx),
                        wcs=template_wcs,
                        mode="partial",
                        fill_value=1e-30,
                    )
                    template_header.update(template_cutout.wcs.to_header())
                    # Saves the results.
                    fits.writeto(
                        fpath,
                        science_cutout.data.astype(science_image.dtype),
                        header=science_header,
                        overwrite=True,
                    )
                    fits.writeto(
                        templateFpath,
                        template_cutout.data.astype(template_image.dtype),
                        header=template_header,
                        overwrite=True,
                    )
                    # Logs the cutout information.
                    logging.info(
                        f"Cutout aligned using sky center: RA={center_coord.ra.deg:.3f}, Dec={center_coord.dec.deg:.3f}"
                    )
                    logging.info(f"Science cutout shape: {science_cutout.data.shape}")
                    logging.info(f"Template cutout shape: {template_cutout.data.shape}")
                    logging.info(
                        f"Target pixel coordinates (science): x={target_x_pix:.2f}, y={target_y_pix:.2f}"
                    )

            # Reloads the image and header.
            image = get_image(fpath)
            header = get_header(fpath)
            imageWCS = get_wcs(header)
            height, width = image.shape

            # Converts the target coordinates to pixel coordinates and stores in input_yaml.
            target_x, target_y = imageWCS.all_world2pix(
                input_yaml["target_ra"],
                input_yaml["target_dec"],
                index,
            )
            input_yaml["target_x_pix"] = target_x
            input_yaml["target_y_pix"] = target_y

            # Use both catalog and detected sources for matching: covers poor catalog coverage (detection fills in)
            # and detection failures or missed sources (catalog fills in).
            logging.info(
                border_msg(
                    f"Finding matching sources in science and reference images "
                    f"({os.path.basename(fpath)} vs {os.path.basename(templateFpath)})"
                )
            )
            _, detected_sources, _ = SExtractorWrapper(config=input_yaml).run(
                fpath,
                pixel_scale=pixel_scale,
                masked_sources=variable_sources,
                weight_path=weight_fpath,
                use_FWHM=ImageFWHM,
                crowded=input_yaml.get("photometry", {}).get("crowded_field", False),
                use_for_matching=True,
            )
            use_catalog = input_yaml.get("template_subtraction", {}).get(
                "use_catalog_for_matching", True
            )
            has_catalog = (
                CatalogSources is not None
                and len(CatalogSources) > 0
                and "x_pix" in CatalogSources.columns
                and "y_pix" in CatalogSources.columns
            )
            has_detected = detected_sources is not None and len(detected_sources) > 0

            if has_detected:
                MatchingSources = detected_sources.copy()
                if use_catalog and has_catalog:
                    common = [
                        c
                        for c in MatchingSources.columns
                        if c in CatalogSources.columns
                    ]
                    catalog_sub = CatalogSources[common].copy()
                    for c in MatchingSources.columns:
                        if c not in catalog_sub.columns:
                            catalog_sub[c] = np.nan
                    catalog_sub = catalog_sub[MatchingSources.columns]
                    n_before = len(MatchingSources)
                    MatchingSources = pd.concat(
                        [MatchingSources, catalog_sub], ignore_index=True
                    )
                    logging.info(
                        f"Matching: {n_before} detected + {len(CatalogSources)} catalog -> {len(MatchingSources)} total"
                    )
                else:
                    logging.info(f"Matching: {len(MatchingSources)} detected sources")
            elif use_catalog and has_catalog:
                # Detection failed or returned no sources; use catalog only.
                MatchingSources = CatalogSources[["x_pix", "y_pix"]].copy()
                logging.info(
                    f"Matching: 0 detected; using {len(MatchingSources)} catalog sources"
                )
            else:
                MatchingSources = pd.DataFrame(columns=["x_pix", "y_pix"])
                logging.warning(
                    "Matching: no detected sources and no catalog; matching list is empty"
                )

            # Builds final per-row coordinates.
            if "x_fit" in MatchingSources.columns:
                MatchingSources["x_coord"] = MatchingSources["x_fit"].fillna(
                    MatchingSources["x_pix"]
                )
            else:
                MatchingSources["x_coord"] = MatchingSources["x_pix"]
            if "y_fit" in MatchingSources.columns:
                MatchingSources["y_coord"] = MatchingSources["y_fit"].fillna(
                    MatchingSources["y_pix"]
                )
            else:
                MatchingSources["y_coord"] = MatchingSources["y_pix"]

            # Coordinates array for distance/cluster operations.
            coords = MatchingSources[["x_coord", "y_coord"]].values

            # Clusters the sources.
            cluster_labels = fclusterdata(coords, t=ImageFWHM, criterion="distance")

            # Groups and collapses clusters.
            numeric_cols = MatchingSources.select_dtypes(include=np.number).columns
            agg_funcs = {
                col: "median" if col in numeric_cols else "first"
                for col in MatchingSources.columns
            }
            labels_series = pd.Series(cluster_labels, index=MatchingSources.index)
            merged_sources = MatchingSources.groupby(labels_series).agg(agg_funcs)
            logging.info(
                f"Collapsed {len(MatchingSources)} -> {len(merged_sources)} median sources"
            )

            # Prepares cluster member counts.
            cluster_counts = labels_series.value_counts().to_dict()

            # Image shape fallback.
            _height, _width = image.shape if image is not None else (height, width)
            centroid_adjusted = 0
            centroid_func = centroid_2dg if ImageFWHM < 3.0 else centroid_com
            # Iterates clusters with >1 members and attempts centroiding using photutils.
            for label in merged_sources.index:
                if cluster_counts.get(label, 1) <= 1:
                    continue
                members = MatchingSources[labels_series == label]
                # Center guess (float).
                try:
                    center_x = float(
                        members["x_coord"].median()
                        if "x_coord" in members.columns
                        else members["x_pix"].median()
                    )
                    center_y = float(
                        members["y_coord"].median()
                        if "y_coord" in members.columns
                        else members["y_pix"].median()
                    )
                except Exception:
                    continue
                # Chooses odd box size from FWHM, minimum 7 to have enough pixels for fit.
                box = max(int(np.ceil(ImageFWHM)) * 2 + 1, 7)
                # Ensures box fits inside image, else reduces.
                half = box // 2
                if (
                    (center_x - half < 0)
                    or (center_x + half >= _width)
                    or (center_y - half < 0)
                    or (center_y + half >= _height)
                ):
                    # Crops box to available area; centroid_sources will still handle small boxes.
                    box = min(
                        box,
                        2
                        * int(
                            min(
                                center_x,
                                _width - center_x - 1,
                                center_y,
                                _height - center_y - 1,
                            )
                        )
                        + 1,
                    )
                    if box < 3:
                        continue
                # Runs centroid_sources with centroid_2dg first, then falls back to centroid_com.
                try:
                    x_c, y_c = centroid_sources(
                        image,
                        [center_x],
                        [center_y],
                        box_size=box,
                        centroid_func=centroid_func,
                    )
                    if np.isfinite(x_c[0]) and np.isfinite(y_c[0]):
                        merged_sources.at[label, "x_pix"] = float(x_c[0])
                        merged_sources.at[label, "y_pix"] = float(y_c[0])
                        centroid_adjusted += 1
                        continue
                except Exception:
                    pass
                # Falls back to center-of-mass centroid.
                try:
                    x_c, y_c = centroid_sources(
                        image,
                        [center_x],
                        [center_y],
                        box_size=box,
                        centroid_func=centroid_func,
                    )
                    if np.isfinite(x_c[0]) and np.isfinite(y_c[0]):
                        merged_sources.at[label, "x_pix"] = float(x_c[0])
                        merged_sources.at[label, "y_pix"] = float(y_c[0])
                        centroid_adjusted += 1
                        continue
                except Exception:
                    # Leaves aggregated coordinates if centroiding fails.
                    continue

            if centroid_adjusted:
                logging.info(
                    f"Centroid-refined positions for {centroid_adjusted} clusters (multi-member)"
                )

            # Recalculates pixel coordinates from RA/DEC returned by the aggregation if present.
            ra_vals = merged_sources.get("RA")
            dec_vals = merged_sources.get("DEC")
            if (ra_vals is not None) and (dec_vals is not None):
                x_pix, y_pix = imageWCS.all_world2pix(
                    ra_vals.values, dec_vals.values, index
                )
                merged_sources["x_pix"] = x_pix
                merged_sources["y_pix"] = y_pix
            else:
                merged_sources["x_pix"] = merged_sources.get(
                    "x_pix", merged_sources.get("x_coord")
                )
                merged_sources["y_pix"] = merged_sources.get(
                    "y_pix", merged_sources.get("y_coord")
                )

            # If not enough sources, bails out early.
            matched_df = merged_sources.copy()
            ConsistentSources = None
            stamp_loc = None

            template_image = get_image(templateFpath)
            template_header = get_header(templateFpath)

            # Build a KDTree for masked pixels for efficient nearest-neighbor search
            masked_image = (defects_mask) | (image == 0) | (template_image == 0)
            masked_pixels = np.argwhere(masked_image)

            if len(masked_pixels) == 0:
                logging.info("Image contains no nan regions ")
                excluded_sources = FWHMSources.iloc[[]]  # Empty DataFrame
            else:
                tree = cKDTree(masked_pixels)
                source_coords = matched_df[["x_pix", "y_pix"]].values
                # Query the tree for the minimum distance to any masked pixel for each source
                min_distances, _ = tree.query(
                    source_coords, k=1, distance_upper_bound=scale
                )

                matched_df = matched_df[min_distances > scale]

                if not excluded_sources.empty:
                    logging.info(
                        f"Excluded {len(excluded_sources)} sources due to proximity to  a nan region "
                        f"(threshold: {distance_threshold:.2f} pixels)."
                    )

            df_zogy_science = None
            df_zogy_template = None
            if len(matched_df) > 3:
                logging.info(f"Sufficient sources for processing: {len(matched_df)}")
                # Prepares source tables for image and template photometry.
                image_sources = matched_df.copy()
                template_sources = matched_df.copy()

                science_large_aperture = header.get("aper", 10)
                logging.info(
                    f"Using large aperture for science image: {science_large_aperture:.1f} pixels"
                )

                # Photometry on the science image.
                aperture_photometry = Aperture(
                    input_yaml=input_yaml,
                    image=image,
                )
                image_sources = aperture_photometry.measure(
                    sources=image_sources[["x_pix", "y_pix"]],
                    exposure_time=exposure_time,
                    ap_size=science_large_aperture,
                )
                # Loads the template image and header.
                template_fwhm = template_header.get("FWHM", 3)

                templatelarge_aperture = template_header.get("aper", 10)
                logging.info(
                    f"Using large aperture for reference image: {templatelarge_aperture:.1f} pixels"
                )

                # Photometry on the template image.
                template_aperture = Aperture(
                    input_yaml=input_yaml, image=template_image
                )
                template_sources = template_aperture.measure(
                    sources=template_sources[["x_pix", "y_pix"]],
                    exposure_time=template_header.get("exposure_time"),
                    ap_size=templatelarge_aperture,
                    gain=template_header.get("GAIN"),
                    n_jobs=n_jobs,
                )

                #  NEW: Centroid Check for Each Source
                # Define a tolerance for positional alignment (e.g., 1 pixels)
                POSITION_TOLERANCE = 1

                # Centroid each source in the template image
                template_sources["x_centroid"] = np.nan
                template_sources["y_centroid"] = np.nan
                for idx, row in image_sources.iterrows():
                    center_x, center_y = row["x_pix"], row["y_pix"]
                    box = max(
                        int(np.ceil(ImageFWHM)) * 2 + 1, 7
                    )  # Ensure box size is odd and large enough
                    try:
                        x_c, y_c = centroid_sources(
                            template_image,
                            [center_x],
                            [center_y],
                            box_size=box,
                            centroid_func=centroid_2dg,
                        )
                        if np.isfinite(x_c[0]) and np.isfinite(y_c[0]):
                            template_sources.at[idx, "x_centroid"] = float(x_c[0])
                            template_sources.at[idx, "y_centroid"] = float(y_c[0])
                    except Exception:
                        pass

                ## Calculate the distance between the original pixel position and the centroid
                distance = np.sqrt(
                    (template_sources["x_pix"] - template_sources["x_centroid"]) ** 2
                    + (template_sources["y_pix"] - template_sources["y_centroid"]) ** 2
                )

                # Filter sources where the centroid is well-aligned with the original pixel position
                well_aligned_mask = distance < POSITION_TOLERANCE

                # Apply the mask to both image_sources and template_sources
                image_sources = image_sources[well_aligned_mask]
                template_sources = template_sources[well_aligned_mask]

                # Log the number of sources removed or the mean offset
                n_removed = len(well_aligned_mask) - sum(well_aligned_mask)
                if n_removed > 0:
                    logging.info(
                        f"Removed {n_removed} sources due to centroid misalignment "
                        f"(tolerance: {POSITION_TOLERANCE} pixels)"
                    )
                else:
                    mean_offset = (
                        np.nanmean(distance[well_aligned_mask])
                        if np.any(well_aligned_mask)
                        else 0.0
                    )
                    logging.info(
                        f"Selected matching sources have mean centroid offset of {mean_offset:.3f} pixels"
                    )

                logging.info(
                    f"Well-detected sources in both images: {len(image_sources)}"
                )

                # Finds flux-consistent sources between image and template.
                if len(image_sources) > 5:
                    template_obj = Templates(input_yaml=input_yaml)
                    MatchingSources, offset_params = (
                        template_obj.find_flux_consistent_sources(
                            image_sources,
                            template_sources,
                        )
                    )
                else:
                    MatchingSources = image_sources

                # Converts pixel coordinates back to world coordinates and attaches to the table.
                if not MatchingSources.empty:
                    ra_vals, dec_vals = imageWCS.all_pix2world(
                        MatchingSources["x_pix"].values,
                        MatchingSources["y_pix"].values,
                        index,
                    )
                    MatchingSources["RA"] = ra_vals
                    MatchingSources["DEC"] = dec_vals
                    # Fits the PSF model to the matched sources.
                    MatchingSources = PSF(
                        image=image,
                        input_yaml=input_yaml,
                    ).fit(
                        epsf_model=epsf_model,
                        sources=MatchingSources,
                        background_rms=background_rms,
                    )

                    # ------------------------------------------------------------------
                    # Refine SFFT / HOTPANTS priors: keep only isolated, PSF-like stars
                    # (remove extended objects / galaxies via size outliers and reject
                    # crowded stars with close neighbours in either image).
                    # ------------------------------------------------------------------
                    ms = MatchingSources.copy()
                    try:
                        # Size-based outlier rejection (robust sigma-clipping on FWHM)
                        size_col = None
                        for c in ("fwhm", "fwhm_psf", "fwhm_model"):
                            if c in ms.columns:
                                size_col = c
                                break
                        if size_col is not None:
                            size = np.asarray(ms[size_col], dtype=float)
                            finite = np.isfinite(size) & (size > 0)
                            if finite.any():
                                med = np.nanmedian(size[finite])
                                mad = 1.4826 * np.nanmedian(np.abs(size[finite] - med))
                                if not np.isfinite(mad) or mad == 0:
                                    mad = med * 0.1 if med > 0 else 1.0
                                n_sigma = 3.0
                                good_size = finite & (
                                    np.abs(size - med) <= n_sigma * mad
                                )
                                ms = ms[good_size]

                        # Crowding rejection: require each prior star to be relatively isolated
                        # in both the science and template images within a radius ~2.5*FWHM.
                        if (
                            len(ms) > 0
                            and {"x_pix", "y_pix"}.issubset(image_sources.columns)
                            and {"x_pix", "y_pix"}.issubset(template_sources.columns)
                        ):
                            from scipy.spatial import cKDTree

                            sci_xy_all = np.vstack(
                                [
                                    image_sources["x_pix"].values,
                                    image_sources["y_pix"].values,
                                ]
                            ).T
                            ref_xy_all = np.vstack(
                                [
                                    template_sources["x_pix"].values,
                                    template_sources["y_pix"].values,
                                ]
                            ).T
                            sci_tree = cKDTree(sci_xy_all)
                            ref_tree = cKDTree(ref_xy_all)

                            ms_xy = np.vstack(
                                [ms["x_pix"].values, ms["y_pix"].values]
                            ).T
                            fwhm_pix = float(input_yaml.get("science_fwhm", ImageFWHM))
                            crowd_r = 2.5 * max(fwhm_pix, 1.0)
                            max_nei = 1
                            sci_counts = np.array(
                                [
                                    len(sci_tree.query_ball_point(pt, crowd_r)) - 1
                                    for pt in ms_xy
                                ]
                            )
                            ref_counts = np.array(
                                [
                                    len(ref_tree.query_ball_point(pt, crowd_r)) - 1
                                    for pt in ms_xy
                                ]
                            )
                            isolated = (sci_counts <= max_nei) & (ref_counts <= max_nei)
                            ms = ms[isolated]

                        MatchingSources = ms
                    except Exception as e:
                        logging.getLogger(__name__).debug(
                            "Refining matching sources for SFFT/HOTPANTS failed (non-fatal): %s",
                            e,
                        )

                    # For ZOGY: same stars for both PSFs; keep native pixel convention
                    if (
                        "zogy" in input_yaml["template_subtraction"]["method"]
                        and not MatchingSources.empty
                    ):
                        df_zogy_science = MatchingSources[["x_pix", "y_pix"]].copy()
                        try:
                            tz = template_sources.loc[MatchingSources.index].copy()
                            if (
                                "x_centroid" in tz.columns
                                and "y_centroid" in tz.columns
                                and tz["x_centroid"].notna().all()
                                and tz["y_centroid"].notna().all()
                            ):
                                df_zogy_template = tz[
                                    ["x_centroid", "y_centroid"]
                                ].rename(
                                    columns={
                                        "x_centroid": "x_pix",
                                        "y_centroid": "y_pix",
                                    }
                                )
                            else:
                                df_zogy_template = tz[["x_pix", "y_pix"]].copy()
                        except Exception:
                            df_zogy_template = df_zogy_science.copy()
                    else:
                        df_zogy_science = None
                        df_zogy_template = None

                    ConsistentSources = MatchingSources[
                        ["x_pix", "y_pix"]
                    ].values.tolist()

                    stamp_loc = os.path.join(write_dir, "stamps_positions.txt")
                    # Write coordinates to a stamp positions text file for HOTPANTS or related use.
                    # HOTPANTS expects 1-based FITS-style pixel coordinates, so add +1 here once.
                    with open(stamp_loc, "w") as f:
                        for x, y in ConsistentSources:
                            f.write(f"{x+1} {y+1}\n")

                else:
                    ConsistentSources = []
                    MatchingSources = pd.DataFrame(columns=["x_pix", "y_pix"])

                logging.info(f"{len(ConsistentSources)} consistent sources found.")
            else:
                ConsistentSources = []
                MatchingSources = pd.DataFrame(columns=["x_pix", "y_pix"])
                logging.info("Insufficient sources for matching ")

            #  Variable Sources
            # Handles variable sources if present.
            # stamp_loc = None
            if len(variable_sources) > 0:
                xpix_variable_sources, ypix_variable_sources = imageWCS.all_world2pix(
                    variable_sources["RA"].values,
                    variable_sources["DEC"].values,
                    index,
                )
                variable_sources["x_pix"] = xpix_variable_sources
                variable_sources["y_pix"] = ypix_variable_sources
                # Applies the border mask.
                border = 1.5 * ImageFWHM
                height, width = image.shape
                mask_x = (variable_sources["x_pix"] >= border) & (
                    variable_sources["x_pix"] < width - border
                )
                mask_y = (variable_sources["y_pix"] >= border) & (
                    variable_sources["y_pix"] < height - border
                )
                variable_sources = variable_sources[mask_x & mask_y]

                variable_sources["x_pix"] += 1
                variable_sources["y_pix"] += 1
                masked_sources = variable_sources[["x_pix", "y_pix"]].values.tolist()
            else:
                masked_sources = []

            #  ZOGY Method: build science and template PSF.
            # Science PSF from science image (matched stars). Template PSF: either from
            # independently selected stars on the reference (better) or same matched stars.
            if (
                "zogy" in input_yaml["template_subtraction"]["method"]
                and df_zogy_science is not None
                and df_zogy_template is not None
                and len(df_zogy_science) >= 5
            ):
                template_image = get_image(templateFpath)
                template_header = get_header(templateFpath)
                use_independent_template_psf = input_yaml["template_subtraction"].get(
                    "zogy_template_psf_independent", True
                )
                df_zogy_template_build = df_zogy_template
                if use_independent_template_psf:
                    try:
                        (
                            template_fwhm,
                            template_fwhm_sources,
                            template_scale,
                        ) = Find_FWHM(input_yaml=input_yaml).measure_image(
                            image=template_image,
                        )
                        template_isolated = Find_FWHM(
                            input_yaml=input_yaml
                        ).filter_isolated_sources(
                            template_fwhm_sources, min_distance=template_scale
                        )
                        if len(template_isolated) >= 5:
                            template_isolated = Catalog(input_yaml=input_yaml).recenter(
                                template_isolated,
                                template_image,
                                boxsize=template_scale,
                                error=None,
                            )
                            xcol = (
                                "x_pix" if "x_pix" in template_isolated.columns else "x"
                            )
                            ycol = (
                                "y_pix" if "y_pix" in template_isolated.columns else "y"
                            )
                            df_zogy_template_build = template_isolated[
                                [xcol, ycol]
                            ].copy()
                            df_zogy_template_build.columns = ["x_pix", "y_pix"]
                            logging.info(
                                "Building ZOGY template PSF from %d stars selected on reference (zogy_template_psf_independent=True).",
                                len(df_zogy_template_build),
                            )
                        else:
                            logging.info(
                                "Only %d isolated stars on reference; using matched stars for template PSF.",
                                len(template_isolated),
                            )
                    except Exception as e:
                        logging.warning(
                            "Independent template PSF star selection failed (%s); using matched stars.",
                            e,
                        )
                if not use_independent_template_psf:
                    logging.info(
                        "Building ZOGY PSFs from %d same stars (science + template).",
                        len(df_zogy_science),
                    )
                # Science PSF from science image (matched stars)
                PSF(
                    image=image,
                    input_yaml=input_yaml,
                    header=header,
                ).build(
                    psfSources=df_zogy_science,
                    mask=defects_mask,
                    background_rms=background_rms,
                    filename_prefix="PSF_model_image",
                )
                # Template PSF from template image (independent or matched stars)
                PSF(
                    image=template_image,
                    input_yaml=input_yaml,
                    header=template_header,
                ).build(
                    psfSources=df_zogy_template_build,
                    mask=None,
                    make_template_psf=True,
                    filename_prefix="PSF_model_template",
                )

            # =============================================================================
            #          Perform Subtraction
            # =============================================================================
            # Performs the subtraction.

            try:
                sfft_matched_sources = os.path.join(
                    write_dir, "sfft_matching_sources.csv"
                )
                fpath_nosub = fpath

                if os.path.exists(sfft_matched_sources):
                    os.remove(sfft_matched_sources)

                fpath, subtraction_mask, _ = Templates(input_yaml=input_yaml).subtract(
                    scienceFpath=fpath,
                    templateFpath=templateFpath,
                    method=input_yaml["template_subtraction"]["method"],
                    matching_sources=ConsistentSources,
                    masked_sources=masked_sources,
                    stamp_loc=stamp_loc,
                    scienceNoise=weight_fpath,
                    templateNoise=template_weight_path,
                )
                if fpath is None:
                    logging.warning(
                        "Template subtraction failed or produced invalid difference image; "
                        "using original science image for photometry."
                    )
                    fpath = science_path_original
                    PreformSubtraction = False
                else:
                    PreformSubtraction = True

                if os.path.exists(sfft_matched_sources):
                    MatchingSources = pd.read_csv(sfft_matched_sources)
                    MatchingSources["x_pix"] = MatchingSources["X_IMAGE_REF_SCI_MEAN"]
                    MatchingSources["y_pix"] = MatchingSources["Y_IMAGE_REF_SCI_MEAN"]
                    os.remove(sfft_matched_sources)

            except Exception as e:
                log_exception(e)
                PreformSubtraction = False

            # Reloads the image.
            image = get_image(fpath)
            if np.sum(image) == 0 and not PreformSubtraction:
                logging.info(
                    "TEMPLATE SUBTRACTION RETURNED ZERO IMAGE - Attempting on original image"
                )
                fpath = science_path_original
                PreformSubtraction = False
                image = get_image(fpath)

            elif PreformSubtraction:
                logging.info("Measuring background from difference image")
                bg_remover = BackgroundSubtractor(input_yaml)
                result = bg_remover.remove(image, plot=False, fwhm=ImageFWHM)
                background_surface = result["background"]
                background_rms = result["background_rms"]

        # Gets the header of the image.
        header = get_header(fpath)

        # Gets the WCS information from the header.
        imageWCS = get_wcs(header)

        # Converts the target coordinates to pixel coordinates.
        target_x_pix, target_y_pix = imageWCS.all_world2pix(
            input_yaml["target_ra"],
            input_yaml["target_dec"],
            index,
        )
        target_x_pix, target_y_pix = update_target_pixel_coords(
            input_yaml, imageWCS, index
        )

        # Updates the input YAML with the target pixel coordinates.
        input_yaml["target_ra"] = target_coords.ra.degree
        input_yaml["target_dec"] = target_coords.dec.degree
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix

        if input_yaml["photometry"].get("remove_local_surface", 3):
            bg_remover = BackgroundSubtractor(input_yaml)
            # Use a slightly larger exclusion radius around the target so that
            # extended host light is not pulled into the local background model.
            image, bkg_map, background_rms = bg_remover.remove_local_surface(
                image,
                x0=target_x_pix,
                y0=target_y_pix,
                box_half_size=int(25 * ImageFWHM),
                fwhm_pixels=ImageFWHM,
                exclude_inner_radius=2.5 * ImageFWHM,
            )
        # remove_local_surface signature:
        # image, x0, y0, box_half_size=100, fwhm_pixels=None,
        # exclude_inner_radius=8, dilate_factor=2.0

        # =============================================================================
        # Targeted Photometry
        # =============================================================================

        #  Log Start of Targeted Photometry
        # Logs the start of targeted photometry.
        logging.info(border_msg(f"Targeted photometry on {input_yaml['target_name']}"))

        # Sets the detection limit.
        detection_limit = input_yaml["photometry"].get("detection_limit", 3)

        # Prepares initial target coordinates.
        TargetPosition = pd.DataFrame(
            {
                "x_pix": [target_x_pix],
                "y_pix": [target_y_pix],
            }
        )
        logging.info(
            f"Transient's expected location: x = {target_x_pix:.3f} pixels, y = {target_y_pix:.3f} pixels"
        )

        # Refines the centroid with COM inside ~1xFWHM box (odd box size).
        boxsize = int(np.ceil(input_yaml["fwhm"]))
        if boxsize % 2 == 0:
            boxsize += 1

        # Performs aperture photometry at the refined position.
        AperturePhotometry = Aperture(
            input_yaml=input_yaml,
            image=image,
        )
        TargetPosition = AperturePhotometry.measure(
            sources=TargetPosition,
            plot=True,
            saveTarget=True,
            background_rms=background_rms,
            n_jobs=n_jobs,
        )
        prelim_threshold = TargetPosition["threshold"].iloc[0]
        perform_ForcePhotometry = False

        # Sets up the target position for PSF fitting.
        TargetPosition["x_fit"] = [np.nan]
        TargetPosition["y_fit"] = [np.nan]
        TargetPosition["x_fit_err"] = [np.nan]
        TargetPosition["y_fit_err"] = [np.nan]

        # max_radius_arcsec = input_yaml["photometry"].get('fitting_xy_bounds',1) / 2
        max_radius_pix = ImageFWHM * 1.5 / 2
        logging.info(f"Fitting source within {max_radius_pix:.1f} pixels")

        # Performs PSF fitting on the target position if aperture photometry is not required.
        if not do_aperture_ONLY:
            TargetPosition = PSF(
                image=image,
                input_yaml=input_yaml,
            ).fit(
                epsf_model=epsf_model,
                sources=TargetPosition,
                plotTarget=True,
                forcePhotometry=perform_ForcePhotometry,
                is_target_fit=True,
                background_rms=background_rms,
                xy_bounds=max_radius_pix,
            )

            if "flags" in TargetPosition:
                from photutils.psf import decode_psf_flags

                if np.isfinite(TargetPosition["flags"].iloc[0]):
                    target_flags = int(TargetPosition["flags"].iloc[0])

                    # logging.info(f"Target Flags: {target_flags}")
                    issues = decode_psf_flags(target_flags)

                    # Check for fitting issues and log warnings
                    if issues:  # issues is non-empty list or list of lists
                        if isinstance(issues, list) and all(
                            isinstance(sub, list) for sub in issues
                        ):
                            # Handle array of sources (list of lists)
                            for idx, source_issues in enumerate(issues):
                                if source_issues:  # Non-empty issues for this source
                                    logging.info(
                                        f"Target fitting issues (source {idx}): {source_issues}"
                                    )
                        else:
                            # Single source (list of str)
                            logging.info(f"Target fitting issues: {issues}")

        # logging.info(TargetPosition.columns)
        # When MCMC is used, LSQ quality metrics (reduced_chi2, cfit, qfit) may be NaN.
        # Only log them when they are present and finite.
        if "reduced_chi2" in TargetPosition:
            reduced_chi2_value = TargetPosition["reduced_chi2"].iloc[0]
            if np.isfinite(reduced_chi2_value):
                logging.info(f"Target reduced chi2: {reduced_chi2_value:.1e}")

        if "cfit" in TargetPosition:
            cfit_value = TargetPosition["cfit"].iloc[0]
            if np.isfinite(cfit_value):
                logging.info(f"Target cfit: {cfit_value:.1e}")

        if "qfit" in TargetPosition:
            qfit_value = TargetPosition["qfit"].iloc[0]
            if np.isfinite(qfit_value):
                logging.info(
                    f"Target qfit: {qfit_value:.1e} (qfit of zero indicates a good fit)"
                )

        # =============================================================================
        # Limiting magnitdues
        # =============================================================================

        get_LimitingMagnitude = input_yaml["photometry"].get(
            "get_LimitingMagnitude", True
        )

        # Checks if the fitting converged.
        if np.isnan(TargetPosition["x_fit_err"].iloc[0]) or np.isnan(
            TargetPosition["y_fit_err"].iloc[0]
        ):
            logging.info("Fitting did not converge - getting limiting magnitudes")
            logging.info(
                f"Best fit transient location: x = {TargetPosition['x_fit'].iloc[0]:.3f}, y = {TargetPosition['y_fit'].iloc[0]:.3f}"
            )
            get_LimitingMagnitude = True
        else:
            logging.info(
                f"Transient fitted position: x = {TargetPosition['x_fit'].iloc[0]:.3f} +/- {TargetPosition['x_fit_err'].iloc[0]:.3f}, "
                f"y = {TargetPosition['y_fit'].iloc[0]:.3f} +/- {TargetPosition['y_fit_err'].iloc[0]:.3f}"
            )

        # Checks if template subtraction was performed.
        if PreformSubtraction:
            target_x_pix_expected, target_y_pix_expected = imageWCS.all_world2pix(
                input_yaml["target_ra"],
                input_yaml["target_dec"],
                index,
            )
            # Creates an instance of the plot class.
            makePlots = Plot(input_yaml=input_yaml)
            # Plots the template subtraction check.
            makePlots.subtraction_check(
                image=get_image(fpath_nosub),
                ref=get_image(templateFpath),
                diff=image,
                expected_location=[target_x_pix_expected, target_y_pix_expected],
                fitted_location=[
                    TargetPosition["x_fit"].iloc[0],
                    TargetPosition["y_fit"].iloc[0],
                ],
                inset_size=scale,
                aperture_size=header["aper"],
                mask=subtraction_mask,
                matching_sources=MatchingSources,
                masked_sources=variable_sources,
            )

        # FWHM: from PSF fitting when PSF was run (fwhm used to build ePSF), else from Gaussian fit to target.
        if (
            not do_aperture_ONLY
            and "fwhm_psf" in TargetPosition.columns
            and np.isfinite(TargetPosition["fwhm_psf"].iloc[0])
        ):
            target_fwhm = float(TargetPosition["fwhm_psf"].iloc[0])
        else:
            gaussian_fits = Find_FWHM(input_yaml=input_yaml).fit_gaussian(
                image,
                x=TargetPosition["x_fit"].iloc[0],
                y=TargetPosition["y_fit"].iloc[0],
                dx=max_radius_pix,
                dy=max_radius_pix,
                sigma=ImageFWHM / 2.335,
            )
            target_fwhm = gaussian_fits["fwhmx"]

        #  Position Offset Analysis
        # Analyzes the position offset of the target.
        target_coords = SkyCoord(
            target_ra,
            target_dec,
            unit=(u.deg, u.deg),
            frame="fk5",
            equinox="J2000",
        )

        # Converts pixel coordinates to world coordinates.
        extracted_position = imageWCS.all_pix2world(
            TargetPosition["x_fit"].iloc[0],
            TargetPosition["y_fit"].iloc[0],
            index,
        )

        # Creates a SkyCoord object for the extracted position.
        coords_science_i = SkyCoord(
            extracted_position[0],
            extracted_position[1],
            unit=(u.deg, u.deg),
            frame="fk5",
            equinox="J2000",
        )
        separation = coords_science_i.separation(target_coords).arcsecond
        try:
            # Prefer PSF-based beta when flux and error are available (better for detection criteria)
            if (
                not do_aperture_ONLY
                and "flux_PSF" in TargetPosition.columns
                and "flux_PSF_err" in TargetPosition.columns
            ):
                f_psf = float(TargetPosition["flux_PSF"].iloc[0])
                f_psf_err = float(TargetPosition["flux_PSF_err"].iloc[0])
                if np.isfinite(f_psf) and np.isfinite(f_psf_err) and f_psf_err > 0:
                    target_beta = float(beta_psf(detection_limit, f_psf, f_psf_err))
                else:
                    flux_col = "flux_AP"
                    target_beta = beta_aperture(
                        n=detection_limit,
                        flux_aperture=float(TargetPosition[flux_col].iloc[0]),
                        sigma=float(TargetPosition["noiseSky"].iloc[0]),
                        npix=float(TargetPosition["area"].iloc[0]),
                    )
            else:
                flux_col = (
                    "flux_PSF" if "flux_PSF" in TargetPosition.columns else "flux_AP"
                )
                target_beta = beta_aperture(
                    n=detection_limit,
                    flux_aperture=float(TargetPosition[flux_col].iloc[0]),
                    sigma=float(TargetPosition["noiseSky"].iloc[0]),
                    npix=float(TargetPosition["area"].iloc[0]),
                )
        except Exception:
            target_beta = np.nan

        # Logs the measured SNR and target detectability (aperture and PSF when available).
        snr_ap = float(TargetPosition["SNR"].iloc[0])
        logging.info(f"Target SNR (aperture): {snr_ap:.1f}")
        if (
            not do_aperture_ONLY
            and "flux_PSF" in TargetPosition.columns
            and "flux_PSF_err" in TargetPosition.columns
        ):
            flux_psf = float(TargetPosition["flux_PSF"].iloc[0])
            flux_psf_err = float(TargetPosition["flux_PSF_err"].iloc[0])
            snr_psf = (
                flux_psf / flux_psf_err
                if flux_psf_err > 0 and np.isfinite(flux_psf_err)
                else np.nan
            )
            if np.isfinite(snr_psf):
                logging.info(f"Target SNR (PSF): {snr_psf:.1f}")
        logging.info(
            f"Target threshold: {TargetPosition['threshold'].iloc[0]:.1f} x background standard deviation"
        )
        logging.info(f"Target detectability: {target_beta * 100:.1f} %")
        logging.info(f"Target location measured with FWHM: {target_fwhm:.1f} pixels")

        # Calculates pixel offsets.
        dx_pix = TargetPosition["x_fit"].iloc[0] - input_yaml["target_x_pix"]
        dy_pix = TargetPosition["y_fit"].iloc[0] - input_yaml["target_y_pix"]
        offset_pix = np.sqrt(dx_pix**2 + dy_pix**2)

        logging.info("POSITION OFFSET ANALYSIS:")
        logging.info(
            f"\tExpected pixel position: ({input_yaml['target_x_pix']:.3f}, {input_yaml['target_y_pix']:.3f})"
        )
        logging.info(
            f"\tFitted pixel position:   ({TargetPosition['x_fit'].iloc[0]:.3f}, {TargetPosition['y_fit'].iloc[0]:.3f})"
        )
        logging.info(f"\tPixel offset: dx = {dx_pix:+.3f}, dy = {dy_pix:+.3f}")
        logging.info(f"\tTotal pixel offset: {offset_pix:.3f} pixels")

        # Calculates RA/Dec error in arcseconds from pixel errors.
        if not np.isnan(TargetPosition["x_fit_err"].iloc[0]) and not np.isnan(
            TargetPosition["y_fit_err"].iloc[0]
        ):
            # Builds sky coordinates for (xpix, ypix).
            sky_center = SkyCoord(
                extracted_position[0],
                extracted_position[1],
                unit=(u.deg, u.deg),
                frame="fk5",
                equinox="J2000",
            )
            # Builds sky coordinates for (xpix + xpix_err, ypix) and (xpix, ypix + ypix_err).
            sky_dx = imageWCS.pixel_to_world(
                TargetPosition["x_fit"].iloc[0] + TargetPosition["x_fit_err"].iloc[0],
                TargetPosition["y_fit"].iloc[0],
            )
            sky_dy = imageWCS.pixel_to_world(
                TargetPosition["x_fit"].iloc[0],
                TargetPosition["y_fit"].iloc[0] + TargetPosition["y_fit_err"].iloc[0],
            )
            # Calculates separations in arcseconds.
            ra_err = sky_center.separation(sky_dx).arcsecond
            dec_err = sky_center.separation(sky_dy).arcsecond
            fitting_error_arcsec = np.sqrt(ra_err**2 + dec_err**2)
            logging.info(
                f"\tFitting uncertainty: {TargetPosition['x_fit_err'].iloc[0]:.3f}, {TargetPosition['y_fit_err'].iloc[0]:.3f} pixels"
            )
        else:
            ra_err = np.nan
            dec_err = np.nan
            fitting_error_arcsec = 0
            logging.info("\tFitting uncertainty: N/A (fit did not converge)")

        # Calculates the offset in arcseconds (including direction).
        expected_sky = imageWCS.pixel_to_world(
            input_yaml["target_x_pix"], input_yaml["target_y_pix"]
        )
        fitted_sky = imageWCS.pixel_to_world(
            TargetPosition["x_fit"].iloc[0], TargetPosition["y_fit"].iloc[0]
        )

        # Calculates RA and Dec offsets with proper cos(dec) correction.
        dra_arcsec = (
            (fitted_sky.ra.degree - expected_sky.ra.degree)
            * 3600
            * np.cos(np.radians(expected_sky.dec.degree))
        )
        ddec_arcsec = (fitted_sky.dec.degree - expected_sky.dec.degree) * 3600
        logging.info(
            f'\tSky offset: dRA = {dra_arcsec:+.3f}", dDec = {ddec_arcsec:+.3f}"'
        )
        logging.info(
            f"\tTotal separation: {separation:.3f} +/- {fitting_error_arcsec:.3f} arcseconds"
        )

        # =============================================================================
        # Calibration and Output
        # =============================================================================
        #  Calibrate Magnitudes
        # Calibrates the magnitudes for each method (AP, PSF)

        for method in ["AP", "PSF"]:
            if method not in image_zeropoint:
                logging.info(f"{method} zeropoint not available - skipping")
                continue
            idx = 0
            inst_col = f"inst_{input_yaml['imageFilter']}_{method}"
            if inst_col not in TargetPosition.columns:
                logging.info(
                    f"{method} column not in TargetPosition (fit may have failed) - setting to NaN"
                )
                TargetPosition[inst_col] = [np.nan]
                TargetPosition[f"{inst_col}_err"] = [np.nan]
                TargetPosition[f"{input_yaml['imageFilter']}_{method}"] = [np.nan]
                TargetPosition[f"{input_yaml['imageFilter']}_{method}_err"] = [np.nan]
                continue
            try:
                # Calibrated magnitude: inst_mag + ZP. For AP, optionally subtract
                # aperture correction (only when apply_aperture_correction is True;
                # otherwise aperture correction is stored for use later).
                ap_corr = float(input_yaml.get("aperture_correction", 0.0) or 0.0)
                ap_corr_err = float(
                    input_yaml.get("aperture_correction_err", 0.0) or 0.0
                )
                apply_ap_corr = bool(input_yaml.get("apply_aperture_correction", False))
                if method == "AP" and apply_ap_corr and np.isfinite(ap_corr):
                    cal_mag = (
                        TargetPosition.at[idx, inst_col]
                        + image_zeropoint[method]["zeropoint"]
                        - ap_corr
                    )
                    errorTerms = [
                        TargetPosition.at[idx, f"{inst_col}_err"],
                        image_zeropoint[method]["zeropoint_error"],
                        ap_corr_err,
                    ]
                else:
                    cal_mag = (
                        TargetPosition.at[idx, inst_col]
                        + image_zeropoint[method]["zeropoint"]
                    )
                    errorTerms = [
                        TargetPosition.at[idx, f"{inst_col}_err"],
                        image_zeropoint[method]["zeropoint_error"],
                    ]
                TargetPosition.at[idx, f"{input_yaml['imageFilter']}_{method}"] = (
                    cal_mag
                )
                TargetPosition.at[idx, f"{input_yaml['imageFilter']}_{method}_err"] = (
                    quadrature_add(errorTerms)
                )
                # Logs the calibrated magnitude and error.
                inst_err_col = f"{inst_col}_err"
                cal_mag_col = f"{input_yaml['imageFilter']}_{method}"
                cal_err_col = f"{input_yaml['imageFilter']}_{method}_err"

                logging.info(
                    "Instrumental %s %s-band magnitude: %.3f +/- %.3f [mag]",
                    method,
                    input_yaml["imageFilter"],
                    TargetPosition.at[idx, inst_col],
                    TargetPosition.at[idx, inst_err_col],
                )

                logging.info(
                    "Calibrated %s %s-band magnitude: %.3f +/- %.3f [mag]",
                    method,
                    input_yaml["imageFilter"],
                    TargetPosition.at[idx, cal_mag_col],
                    TargetPosition.at[idx, cal_err_col],
                )

            except Exception as e:
                log_exception(e)
                # Sets the calibrated magnitude and error to NaN.
                TargetPosition.at[idx, inst_col] = np.nan
                TargetPosition.at[idx, f"{inst_col}_err"] = np.nan
                TargetPosition.at[idx, f"{input_yaml['imageFilter']}_{method}"] = np.nan
                TargetPosition.at[idx, f"{input_yaml['imageFilter']}_{method}_err"] = (
                    np.nan
                )

        # =============================================================================
        #          Detection Limits
        # =============================================================================
        # Calculates the probable and injected detection limits if the target has low SNR.
        ProbableLimit = np.nan
        InjectedLimit = np.nan
        if (
            TargetPosition.at[idx, "threshold"] < 5
            or TargetPosition.at[idx, "SNR"] < 5
            or get_LimitingMagnitude
        ):
            if (
                TargetPosition.at[idx, "threshold"] < 5
                or TargetPosition.at[idx, "SNR"] < 5
                or get_LimitingMagnitude
            ):
                snr_val = TargetPosition.at[idx, "SNR"]
                logging.info(
                    border_msg("Getting detection limits for transient neighborhood")
                )
                # Creates an instance of the limits class.
                getDetectionLimits = Limits(input_yaml=input_yaml)
                try:
                    # Gets the expanded cutout of the image.
                    expandedCutout = getDetectionLimits.get_cutout(image=image)
                    if expandedCutout is None:
                        logging.warning(
                            "getCutout returned None; skipping detection limits."
                        )
                        ProbableLimit = np.nan
                        InjectedLimit = np.nan
                    else:
                        # Calculates the probable detection limit.
                        ProbableLimit = getDetectionLimits.get_probable_limit(
                            expandedCutout,
                            bkg_level=3,
                            detection_limit=detection_limit,
                            useBeta=True,
                            beta=0.5,
                            plot=True,
                            n_jobs=n_jobs,
                        )
                        # Calculates the injected detection limit.
                        if epsf_model:
                            if "AP" in image_zeropoint:
                                zeropoint = image_zeropoint["AP"]["zeropoint"]
                            else:
                                zeropoint = None
                            logging.info(
                                f"Performing artificial source injection with threshold [{detection_limit}] limit"
                            )
                            InjectedLimit = getDetectionLimits.get_injected_limit(
                                image,
                                initialGuess=ProbableLimit - 1,
                                detection_limit=detection_limit,
                                position=(
                                    TargetPosition["x_fit"].iloc[0],
                                    TargetPosition["y_fit"].iloc[0],
                                ),
                                epsf_model=epsf_model,
                                background_rms=background_rms,
                                zeropoint=zeropoint,
                                plot=True,
                                n_jobs=n_jobs,
                            )
                        else:
                            TargetPosition["flux_PSF"] = [np.nan]
                            TargetPosition["flux_PSF_err"] = [np.nan]
                            InjectedLimit = np.nan
                        # Logs the injected detection limit for each method.
                        for method in ["AP", "PSF"]:
                            if method not in image_zeropoint:
                                continue
                            logging.info(
                                f"\tInjected detection limit ({method}): {InjectedLimit + image_zeropoint[method]['zeropoint']:.3f}"
                            )
                except Exception as e:
                    log_exception(e)
                    ProbableLimit = np.nan
                    InjectedLimit = np.nan

        # =============================================================================
        # Save Output
        # =============================================================================

        #  Initialize Output Dictionary
        # Initializes the output dictionary with all values at once.
        output = {
            "filename": fpath,
            "date": date,
            "mjd": date_mjd,
            "telescope": telescope,
            "instrument": instrument,
            "image_fwhm": ImageFWHM,
            "airmass": airmass,
            "exposure_time": input_yaml["exposure_time"],
            "filter": input_yaml["imageFilter"],
            "xpix": TargetPosition.at[idx, "x_fit"],
            "ypix": TargetPosition.at[idx, "y_fit"],
            "xpix_err": (
                TargetPosition.at[idx, "x_fit_err"]
                if "x_fit_err" in TargetPosition.columns
                else np.nan
            ),
            "ypix_err": (
                TargetPosition.at[idx, "y_fit_err"]
                if "y_fit_err" in TargetPosition.columns
                else np.nan
            ),
            "snr": TargetPosition.at[idx, "SNR"],
            "SNR_AP": float(TargetPosition.at[idx, "SNR"]),
            "SNR_PSF": (
                float(
                    TargetPosition.at[idx, "flux_PSF"]
                    / TargetPosition.at[idx, "flux_PSF_err"]
                )
                if not do_aperture_ONLY
                and "flux_PSF" in TargetPosition.columns
                and "flux_PSF_err" in TargetPosition.columns
                and TargetPosition.at[idx, "flux_PSF_err"] > 0
                and np.isfinite(TargetPosition.at[idx, "flux_PSF_err"])
                else np.nan
            ),
            "threshold": TargetPosition["threshold"].iloc[0],
            "target_fwhm": target_fwhm,
            "fwhm_psf": (
                float(TargetPosition["fwhm_psf"].iloc[0])
                if not do_aperture_ONLY and "fwhm_psf" in TargetPosition.columns
                else np.nan
            ),
            "separation": separation if "separation" in locals() else np.nan,
            "beta": target_beta,
            "lmag": InjectedLimit,
            "PreformSubtractioned": (
                PreformSubtraction if "PreformSubtraction" in locals() else False
            ),
            "etime": time.time() - start,
        }

        # Converts pixel coordinates to world coordinates.
        output.update(
            {
                "ra": extracted_position[0],
                "dec": extracted_position[1],
            }
        )
        output.update(
            {
                "ra_err_arcsec": ra_err,
                "dec_err_arcsec": dec_err,
            }
        )

        # Adds zeropoint values.
        image_filter = input_yaml["imageFilter"]
        for method in image_zeropoint.keys():
            if method in image_zeropoint:
                try:
                    output.update(
                        {
                            f"zp_{image_filter}_{method}": image_zeropoint[method][
                                "zeropoint"
                            ],
                            f"zp_{image_filter}_{method}_err": image_zeropoint[method][
                                "zeropoint_error"
                            ],
                        }
                    )
                except Exception:
                    pass

        # Adds flux and magnitude values (only for methods present in TargetPosition).
        for method in image_zeropoint.keys():
            prefix = f"{image_filter}_{method}"
            inst_prefix = f"inst_{image_filter}_{method}"
            flux_col = f"flux_{method}"
            if (
                flux_col not in TargetPosition.columns
                or inst_prefix not in TargetPosition.columns
            ):
                output.update(
                    {
                        f"flux_{method}": np.nan,
                        f"flux_{method}_err": np.nan,
                        inst_prefix: np.nan,
                        f"{inst_prefix}_err": np.nan,
                        prefix: np.nan,
                        f"{prefix}_err": np.nan,
                    }
                )
            else:
                output.update(
                    {
                        f"flux_{method}": TargetPosition.at[idx, flux_col],
                        f"flux_{method}_err": TargetPosition.at[idx, f"{flux_col}_err"],
                        inst_prefix: TargetPosition.at[idx, inst_prefix],
                        f"{inst_prefix}_err": TargetPosition.at[
                            idx, f"{inst_prefix}_err"
                        ],
                        prefix: TargetPosition.at[idx, prefix],
                        f"{prefix}_err": TargetPosition.at[idx, f"{prefix}_err"],
                    }
                )

        # Normalise column headers to lowercase for consistent downstream use (lightcurve, etc.).
        output_normalised = {str(k).strip().lower(): v for k, v in output.items()}
        # Saves the output.
        pd.DataFrame(output_normalised, index=[0]).to_csv(
            os.path.join(write_dir, "output.csv"),
            index=False,
            float_format="%.6f",
        )

        # Converts the output dictionary to a string.
        output_str = dict_to_string_with_hashtag(output)

        # Opens the file in write mode to add the output string first.
        with open(calibration_file, "w") as file:
            file.write("# Output dictionary")
            file.write(output_str + "")

        # Redoes the sources if enabled.
        if input_yaml["photometry"].get("redo_sources", False):
            image = get_image(scienceFpath_cutout)
            header = get_header(scienceFpath_cutout)
            # Gets the WCS information from the header.
            imageWCS = get_wcs(header)
            # Converts world coordinates of isolated sources to pixel coordinates.
            x_pix_IsolatedSources, y_pix_IsolatedSources = imageWCS.all_world2pix(
                IsolatedSources["RA"].values,
                IsolatedSources["DEC"].values,
                index,
            )
            # Adds the pixel coordinates to the IsolatedSources DataFrame.
            IsolatedSources["x_pix"] = x_pix_IsolatedSources
            IsolatedSources["y_pix"] = y_pix_IsolatedSources
            # Defines border and image dimensions.
            border = 2 * scale
            width = image.shape[1]
            height = image.shape[0]
            # Applies the border mask to filter isolated sources.
            mask_x = (IsolatedSources["x_pix"] >= border) & (
                IsolatedSources["x_pix"] < width - border
            )
            mask_y = (IsolatedSources["y_pix"] >= border) & (
                IsolatedSources["y_pix"] < height - border
            )
            IsolatedSources = IsolatedSources[mask_x & mask_y]
            # Performs PSF fitting on the filtered sources.
            IsolatedSources = PSF(
                image=image,
                input_yaml=input_yaml,
            ).fit(
                epsf_model=epsf_model,
                sources=IsolatedSources,
                plotTarget=False,
                background_rms=background_rms,
            )
            # Converts pixel coordinates back to world coordinates.
            ra_IsolatedSources, dec_IsolatedSources = imageWCS.all_pix2world(
                IsolatedSources["x_pix"].values,
                IsolatedSources["y_pix"].values,
                index,
            )
            # Adds the RA and DEC columns to the IsolatedSources DataFrame.
            IsolatedSources["RA"] = ra_IsolatedSources
            IsolatedSources["DEC"] = dec_IsolatedSources
            # Opens the file in write mode to add the output string first.
            with open(
                os.path.join(write_dir, "SOURCES_" + input_yaml["base"] + ".csv"), "w"
            ) as file:
                file.write("# Output dictionary")
                file.write(output_str + "")
            # Opens the file again in append mode to add the clean catalog.
            with open(
                os.path.join(write_dir, "SOURCES_" + input_yaml["base"] + ".csv"), "a"
            ) as file:
                IsolatedSources.to_csv(file, index=False, float_format="%.6f")

        #  Global Photometry
        # Performs global photometry if enabled.
        if (
            image_sources is not None
            and input_yaml["photometry"].get("perform_global_photometry_sigma", None)
            is not None
        ):
            fname = f"sources_{input_yaml['photometry']['perform_global_photometry_sigma']:.0f}sigma_{base}.csv"
            # Writes both the output string and the catalog in one operation.
            with open(os.path.join(write_dir, fname), "w") as file:
                file.write(output_str)
                image_sources.to_csv(file, index=False, float_format="%.6f")

        # Logs the completion of photometric measurements.
        end = time.time() - start
        logging.info(
            border_msg(
                f"Photometric measurements done [{end:.1f}s]", body="*", corner="!"
            )
        )

    except Exception as e:
        log_exception(e)

    return None


if __name__ == "__main__":
    run_photometry()
