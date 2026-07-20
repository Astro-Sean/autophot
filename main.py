#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:55:18 2022
@author: seanbrennan

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
import contextlib

for _env in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env] = "1"

# =============================================================================
# Imports
# =============================================================================

# Standard Library
import argparse
import datetime
import logging
import re
import shutil
import time
import uuid
import warnings
from collections import OrderedDict
from importlib.metadata import version as _pkg_version
from pathlib import Path

# Set matplotlib to use non-interactive backend to prevent Wayland/Qt display issues
import matplotlib
matplotlib.use('Agg')

# Third-Party Libraries
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
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.centroids import centroid_com, centroid_2dg, centroid_sources
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fclusterdata

# SATURATE handling constants - used consistently across the codebase
SATURATE_INTERNAL_FALLBACK = np.inf  # Internal processing: no saturation limit
SATURATE_FITS_FALLBACK = 1e30  # FITS storage: finite value for "no saturation"

try:
    AUTOPHOT_VERSION = _pkg_version("autophot")
except Exception:
    AUTOPHOT_VERSION = "unknown"

# Local Modules
from aperture import (
    Aperture,
    exposure_seconds_from_header,
    gain_e_per_adu_from_header,
    resolve_gain_e_per_adu,
)
from catalog import Catalog, cross_match_sources
from cosmic import RemoveCosmicRays
from fwhm import Find_FWHM
from functions import (
    AutophotYaml,
    log_step,
    border_msg,
    PlainFormatter,
    ColoredLevelFormatter,
    metrics_table,
    compact_status,
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
    flux_upper_limit,
    log_exception,
    log_warning_from_exception,
    odd,
    LogMessageNormalizeFilter,
    safe_fits_write,
)
from limits import BETA_APERTURE_SIGMA_N, Limits
from plot import Plot
from psf import PSF
from templates import Templates
from utils import run_IDC
from utils.run_sex import SExtractorWrapper
from wcs import WCSSolver, get_wcs
from zeropoint import Zeropoint
from background import BackgroundSubtractor


# TODO: add trimming if the template image is smaller than the science image


# =============================================================================
# Utility Functions
# =============================================================================

def _safe_wcs_from_header(header, silent=True):
    """
    Safely extract WCS from a FITS header with validation.
    
    Uses get_wcs for consistency across the codebase.
    Returns None if WCS is missing, invalid, or has no celestial component.
    
    Parameters
    ----------
    header : astropy.io.fits.Header or dict-like
        FITS header to extract WCS from
    silent : bool
        If True, suppress warnings about invalid WCS
        
    Returns
    -------
    wcs : astropy.wcs.WCS or None
        Valid WCS object with celestial coordinates, or None if invalid
    """
    return get_wcs(header, silent=silent)


def _heuristic_filter_mapping(raw_filter: str) -> str:
    """
    Telescope-blind heuristic to map raw filter names to standard catalog bands.
    
    Examples:
        gp_astrodon_2018 -> g
        rp -> r
        ip -> i
        zp -> z
        Sloan_g -> g
    """
    f = str(raw_filter).strip().lower().replace(" ", "").replace("-", "_")
    
    # Standard survey-style suffixes (ZTF, Pan-STARRS style)
    if f in {"gp", "rp", "ip", "zp", "up", "wp"}:
        return f[0]  # gp -> g, rp -> r, etc.
    
    # Tokens with underscores like gp_astrodon_2018, rp_cousins, sloan_g, etc.
    # Check first part for band indicators
    if "_" in f:
        first_part = f.split("_")[0]
        # Direct band match in first part
        if first_part in {"u", "g", "r", "i", "z", "y", "w", "b", "v", "j", "h", "k"}:
            return first_part
        # Check for trailing p (e.g., gp -> g)
        if len(first_part) == 2 and first_part[1] == "p" and first_part[0] in "ugrizyw":
            return first_part[0]
    
    # Check for trailing p (e.g., gp -> g)
    if len(f) == 2 and f[1] == "p" and f[0] in "ugrizyw":
        return f[0]
    
    # Tokens like g782/r784/i705/z623 -> leading band letter + digits
    if len(f) >= 2 and f[0] in "ugrizy" and any(ch.isdigit() for ch in f[1:]):
        return f[0]
    
    # Direct match for single-letter bands
    if f in {"u", "g", "r", "i", "z", "y", "w", "b", "v", "j", "h", "k"}:
        return f
    
    # Check if it starts with a valid band letter
    if f and f[0] in "ugrizyw":
        return f[0]
    
    # Default: return the original filter name
    return raw_filter


def _trim_nan_boundaries(image_data, header, target_x=None, target_y=None, buffer_pixels=10):
    """
    Trim image to remove NaN boundary regions while ensuring target remains in image.

    Parameters
    ----------
    image_data : np.ndarray
        2D image array (can contain NaNs)
    header : fits.Header
        FITS header to update with new WCS after trimming
    target_x, target_y : float, optional
        Target position in pixels (1-indexed, typical FITS convention)
    buffer_pixels : int
        Minimum buffer around valid data region

    Returns
    -------
    trimmed_data : np.ndarray
        Image with NaN boundaries removed
    trimmed_header : fits.Header
        Updated header with corrected WCS
    trim_info : dict
        Information about trimming performed
    """
    nan_count = np.sum(np.isnan(image_data))
    total_pixels = image_data.size
    nan_pct = 100 * nan_count / total_pixels
    logging.info(f"NaN boundary trimming: {nan_count}/{total_pixels} pixels are NaN ({nan_pct:.1f}%)")

    # Find valid (non-NaN) pixels
    valid_mask = ~np.isnan(image_data)

    # If no NaNs or all NaNs, return as-is
    if not np.any(valid_mask) or np.all(valid_mask):
        return image_data, header, {"trimmed": False}

    # Find valid region bounds
    rows_with_valid = np.any(valid_mask, axis=1)
    cols_with_valid = np.any(valid_mask, axis=0)

    if not np.any(rows_with_valid) or not np.any(cols_with_valid):
        return image_data, header, {"trimmed": False}

    y_min, y_max = np.where(rows_with_valid)[0][[0, -1]]
    x_min, x_max = np.where(cols_with_valid)[0][[0, -1]]
    
    # Add buffer
    y_min = max(0, y_min - buffer_pixels)
    y_max = min(image_data.shape[0] - 1, y_max + buffer_pixels)
    x_min = max(0, x_min - buffer_pixels)
    x_max = min(image_data.shape[1] - 1, x_max + buffer_pixels)

    # Check if target is included (if provided)
    expanded = False
    if target_x is not None and target_y is not None:
        # Convert to 0-indexed for array checking
        tx_0idx, ty_0idx = target_x - 1, target_y - 1

        # Expand bounds to include target if needed
        if tx_0idx < x_min:
            x_min = max(0, int(tx_0idx) - buffer_pixels)
            expanded = True
        if tx_0idx > x_max:
            x_max = min(image_data.shape[1] - 1, int(tx_0idx) + buffer_pixels)
            expanded = True
        if ty_0idx < y_min:
            y_min = max(0, int(ty_0idx) - buffer_pixels)
            expanded = True
        if ty_0idx > y_max:
            y_max = min(image_data.shape[0] - 1, int(ty_0idx) + buffer_pixels)
            expanded = True

        # Final verification: ensure target is within bounds
        if not (x_min <= tx_0idx <= x_max and y_min <= ty_0idx <= y_max):
            logging.warning(
                f"NaN boundary trimming: target at ({tx_0idx:.1f}, {ty_0idx:.1f}) is outside final bounds x=[{x_min},{x_max}], y=[{y_min},{y_max}]"
            )
            # Force include target by expanding bounds
            x_min = min(x_min, int(tx_0idx) - buffer_pixels)
            x_max = max(x_max, int(tx_0idx) + buffer_pixels)
            y_min = min(y_min, int(ty_0idx) - buffer_pixels)
            y_max = max(y_max, int(ty_0idx) + buffer_pixels)
            # Clamp to image bounds
            x_min = max(0, x_min)
            x_max = min(image_data.shape[1] - 1, x_max)
            y_min = max(0, y_min)
            y_max = min(image_data.shape[0] - 1, y_max)
            expanded = True
            logging.warning(f"NaN boundary trimming: forced bounds to include target: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")
        elif expanded:
            logging.info(f"NaN boundary trimming: successfully expanded bounds to include target")

    logging.info(f"NaN boundary trimming: final bounds x=[{x_min},{x_max}] px, y=[{y_min},{y_max}] px")
    logging.info(f"NaN boundary trimming: original shape {image_data.shape}, will trim to ({y_max - y_min + 1}, {x_max - x_min + 1}) px")

    # Perform trim using nan_crop to preserve WCS distortion keywords
    # (CD matrix, SIP, PV) that WCS round-trip can drop.
    cutout_x = (x_min + x_max + 1) / 2.0
    cutout_y = (y_min + y_max + 1) / 2.0
    cutout_ny = y_max - y_min + 1
    cutout_nx = x_max - x_min + 1

    from functions import nan_crop
    trimmed_header = header.copy()
    trimmed_data, trimmed_header = nan_crop(
        image_data, trimmed_header, cutout_x, cutout_y, cutout_ny, cutout_nx
    )
    
    # Store trim info in history
    trimmed_header.add_history(f'Trimmed: removed NaN boundaries [{x_min}:{x_max+1},{y_min}:{y_max+1}]')
    
    trim_info = {
        "trimmed": True,
        "x_slice": (x_min, x_max + 1),
        "y_slice": (y_min, y_max + 1),
        "original_shape": image_data.shape,
        "trimmed_shape": trimmed_data.shape,
        "target_preserved": target_x is not None and target_y is not None
    }
    
    return trimmed_data, trimmed_header, trim_info


# =============================================================================
# Main Function: run_photometry
# =============================================================================
def run_photometry():
    """Main entry point for the AutoPHOT photometry pipeline.

    Checks for required Astromatic tools (SExtractor, SCAMP, SWarp), loads
    configuration, and orchestrates the full reduction workflow: header
    inspection, WCS solving, background subtraction, template handling,
    source detection, photometry, and light-curve generation.
    """

    # ---------------------------------------------------------------------
    # Check for optional Astromatic tools
    # ---------------------------------------------------------------------
    logger = logging.getLogger(__name__)

    # Check which tools are available
    sextractor_exe = shutil.which("sex")
    scamp_exe = shutil.which("scamp")
    swarp_exe = shutil.which("swarp")

    missing_tools = []
    if not sextractor_exe:
        missing_tools.append("SExtractor (sex)")
    if not scamp_exe:
        missing_tools.append("SCAMP")
    if not swarp_exe:
        missing_tools.append("SWarp")

    if missing_tools:
        logger.warning(
            "Optional Astromatic tools not found on PATH: %s. These are only required for advanced features (SCAMP for TPV distortion, "
            "SWarp for image resampling). The pipeline will fall back to alternative methods "
            "(astrometry.net for WCS, AstroAlign/Reproject for template alignment). "
            "To install: conda install -c conda-forge astromatic-source-extractor "
            "astromatic-scamp astromatic-swarp",
            ", ".join(missing_tools)
        )

    # ---------------------------------------------------------------------
    # CLI parsing (must happen before heavy imports)
    # ---------------------------------------------------------------------

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

    """
    Perform photometry operations.

    Args:
        -f (str): Filepath of FITS file.
        -c (str): Path to the input YAML file.
        -temp (bool): Flag to prepare a template. Default is False.

    Returns:
        None
    """

    # Use 0-based indexing to match numpy array convention
    wcs_origin = 0
    start = time.time()

    #  Filter Out Astropy Warnings
    # Suppresses Astropy warnings to keep the log clean.
    warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

    #  Load Input YAML File
    # Loads the YAML configuration file which contains parameters for the pipeline.
    try:
        with open(input_yaml_loc, "r") as file:
            input_yaml = yaml.safe_load(file)
    except FileNotFoundError:
        logging.getLogger(__name__).error(
            "Input YAML file is missing: %s\n"
            "It looks like this file was deleted mid-run. The pipeline must be re-run.",
            str(input_yaml_loc),
        )
        raise SystemExit(2)

    # Worker counts for *within-image* parallelism.
    #
    # Note: `nCPU` controls image-level multiprocessing in driver/batch modes.
    # These per-step worker counts control internal loops (sources, injection trials)
    # for a single image.
    ap_n_jobs = int((input_yaml.get("photometry") or {}).get("aperture_n_jobs", 1))
    ap_n_jobs = max(1, ap_n_jobs)
    lim_n_jobs = int((input_yaml.get("limiting_magnitude") or {}).get("n_jobs", 1))
    lim_n_jobs = max(1, lim_n_jobs)

    #  Clear module-level caches to ensure independence between image runs
    # This prevents state contamination when processing multiple images
    try:
        from limits import _flux_for_mag_cached
        _flux_for_mag_cached.cache_clear()
    except Exception:
        pass
    try:
        from background import _disk_structuring_element
        _disk_structuring_element.cache_clear()
    except Exception:
        pass

    #  Validate Target Information
    # Check that at least one of target_name, target_ra, or target_dec is provided.
    # If all are missing, raise a warning and stop the code.
    target_name = input_yaml.get("target_name")
    target_ra = input_yaml.get("target_ra")
    target_dec = input_yaml.get("target_dec")

    if target_name is None and target_ra is None and target_dec is None:
        logging.getLogger(__name__).warning(
            "No target information provided: target_name, target_ra, and target_dec are all missing. "
            "Please provide at least one of these parameters to proceed."
        )
        raise SystemExit(
            "ERROR: Missing target information. "
            "Please set at least one of: target_name, target_ra, or target_dec in your configuration."
        )

    #  Helper Function: Update Target Pixel Coordinates
    # Updates the target's pixel coordinates after any changes to the WCS.
    # Uses origin=0 (0-based) to match numpy array convention.
    def update_target_pixel_coords(input_yaml, imageWCS, wcs_origin=0):
        """Update target pixel coordinates after WCS changes. wcs_origin is WCS origin (0=0-based)."""
        target_x_pix, target_y_pix = imageWCS.all_world2pix(
            input_yaml["target_ra"],
            input_yaml["target_dec"],
            wcs_origin,
        )
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix
        return target_x_pix, target_y_pix

    def _remove_catalog_duplicates(catalog_df, method='pandas', sep_threshold=0.1):
        """
        Remove duplicate catalog entries using specified method.
        
        Parameters
        ----------
        catalog_df : pd.DataFrame
            Catalog with potential duplicates
        method : str
            'pandas' for RA/DEC based removal, 'astropy' for angular separation
        sep_threshold : float
            Separation threshold in arcsec for 'astropy' method
            
        Returns
        -------
        pd.DataFrame
            Deduplicated catalog
        """
        if catalog_df is None or len(catalog_df) == 0:
            return catalog_df
            
        n_pre = len(catalog_df)
        
        if method == 'pandas' and {"RA", "DEC"}.issubset(catalog_df.columns):
            # Fast pandas method for exact duplicates
            catalog_df = (
                catalog_df.sort_values(["RA", "DEC"])
                .drop_duplicates(subset=["RA", "DEC"], keep="first")
                .reset_index(drop=True)
            )
        elif method == 'pandas':
            # Pixel-space fallback
            if {"x_pix", "y_pix"}.issubset(catalog_df.columns):
                _rx = np.round(catalog_df["x_pix"].to_numpy(dtype=float), 2)
                _ry = np.round(catalog_df["y_pix"].to_numpy(dtype=float), 2)
                catalog_df = (
                    catalog_df.assign(_rx=_rx, _ry=_ry)
                    .sort_values(["_rx", "_ry"])
                    .drop_duplicates(subset=["_rx", "_ry"], keep="first")
                    .drop(columns=["_rx", "_ry"])
                    .reset_index(drop=True)
                )
        elif method == 'astropy' and {"RA", "DEC"}.issubset(catalog_df.columns):
            # Astropy method with angular separation.
            # nthneighbor=2 finds the nearest *other* source for each entry.
            # For each close pair (sep < threshold) we keep the lower-index member
            # and drop the higher-index one — this guarantees exactly one copy
            # survives rather than both being dropped.
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            coords = SkyCoord(
                ra=catalog_df["RA"].values * u.degree,
                dec=catalog_df["DEC"].values * u.degree
            )

            if len(coords) >= 2:
                idx_match, sep2, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
                thr = sep_threshold * u.arcsec
                # Vectorized: drop source i if its nearest neighbour j is closer
                # than the threshold AND has a lower index (j is the kept copy).
                drop = (sep2 <= thr) & (idx_match < np.arange(len(catalog_df)))
                catalog_df = catalog_df[~drop].reset_index(drop=True)
        else:
            # Fallback to pandas method if astropy method not applicable
            if {"RA", "DEC"}.issubset(catalog_df.columns):
                catalog_df = (
                    catalog_df.sort_values(["RA", "DEC"])
                    .drop_duplicates(subset=["RA", "DEC"], keep="first")
                    .reset_index(drop=True)
                )
            elif {"x_pix", "y_pix"}.issubset(catalog_df.columns):
                _rx = np.round(catalog_df["x_pix"].to_numpy(dtype=float), 2)
                _ry = np.round(catalog_df["y_pix"].to_numpy(dtype=float), 2)
                catalog_df = (
                    catalog_df.assign(_rx=_rx, _ry=_ry)
                    .sort_values(["_rx", "_ry"])
                    .drop_duplicates(subset=["_rx", "_ry"], keep="first")
                    .drop(columns=["_rx", "_ry"])
                    .reset_index(drop=True)
                )
        
        n_post = len(catalog_df)
        if n_post < n_pre:
            logging.info(
                f"Removed {n_pre - n_post} duplicate sources using {method} method"
            )
        
        return catalog_df

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
        cur_dir = os.path.join(new_output_dir, base)
        # Standardized per-image outputs (must include the FITS filename stem).
        # In template-preparation mode, the completion marker is the generated
        # template catalog rather than the normal per-image output CSV.
        if prepare_template:
            output_csv_path = os.path.join(
                os.path.dirname(science_file), f"imageCalib_template_{base}.csv"
            )
        else:
            output_csv_path = os.path.join(cur_dir, f"OUTPUT_{base}.csv")
            calibration_file = os.path.join(cur_dir, f"CALIB_{base}.csv")

        #  Store Base Filename in YAML
        # Stores the base filename without any extension in the YAML configuration.
        input_yaml["base"] = base

        #  Skip if output exists and we are not restarting (resume mode).
        # restart=True (default): reprocess all files (redo even if OUTPUT exists).
        # restart=False: skip files that already have OUTPUT_{base}.csv (only process new/unprocessed).
        if (
            os.path.exists(output_csv_path)
            and not input_yaml.get("restart", True)
            and not prepare_template
        ):
            _out = (
                f"imageCalib_template_{base}.csv in {os.path.dirname(science_file)}"
                if prepare_template
                else f"OUTPUT_{base}.csv in {cur_dir}"
            )
            logging.info(
                log_step(
                    f"Skip (restart=False): {os.path.basename(science_file)} — {_out}. "
                    "Set restart=True to reprocess."
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

        # Global verbosity control (0=warnings/errors, 1=info, 2=debug).
        vlevel = input_yaml.get("global_verbose_level", 1)
        try:
            vlevel = int(vlevel)
        except Exception:
            vlevel = 1
        if vlevel <= 0:
            log_level = logging.WARNING
        elif vlevel == 1:
            log_level = logging.INFO
        else:
            log_level = logging.DEBUG

        # Sets up logging to file with plain formatter (no ANSI codes).
        log_file = os.path.join(cur_dir, f"LOG_{input_yaml['base']}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        plain_formatter = PlainFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler.setFormatter(plain_formatter)
        
        # Add filter for message normalization
        normalize_filter = LogMessageNormalizeFilter(width=150)
        file_handler.addFilter(normalize_filter)
        
        # Get root logger and add file handler
        root_logger = logging.getLogger("")
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)

        # Creates a console handler with improved settings.
        console = logging.StreamHandler()
        console.setLevel(log_level)

        # Creates a formatter with more detailed information.
        formatter = ColoredLevelFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            use_color=True,
        )
        console.setFormatter(formatter)
        console.addFilter(normalize_filter)

        # Adds the handler to the root logger.
        logging.getLogger("").addHandler(console)

        # Optionally sets the root logger level explicitly.
        logging.getLogger("").setLevel(log_level)

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
        logging.info("")
        logging.info(f"Running AutoPhOT v{AUTOPHOT_VERSION} on {base_filename}")
        if was_shortened or replaced:
            logging.info("Using pre-processed file")

        # When processing a template, ensure TELESCOP/INSTRUME/FILTER exist (e.g. after restore from .original)
        # Create a copy in the science directory to avoid crosstalk when multiple images use the same template
        if prepare_template and "templates" in os.path.normpath(fpath):
            try:
                # Copy template to science directory to avoid modifying shared template file
                sci_dir = Path(fpath).parent
                template_name = Path(fpath).name
                template_copy = str(sci_dir / f"template_{template_name}")
                shutil.copy2(fpath, template_copy)
                fpath = template_copy
                
                with fits.open(fpath, mode="update") as hdul:
                    h = hdul[0].header
                    if not h.get("TELESCOP") or not str(h.get("TELESCOP", "")).strip():
                        norm = os.path.normpath(fpath)
                        # Support both modern (r_template) and legacy (rp_template)
                        # folder names for ugriz template directories.
                        folder_band = None
                        for part in norm.split(os.sep):
                            if not part.endswith("_template"):
                                continue
                            prefix = part.split("_")[0]
                            if prefix in {"u", "g", "r", "i", "z"}:
                                folder_band = prefix
                                break
                            if prefix in {"up", "gp", "rp", "ip", "zp"}:
                                folder_band = prefix[:-1]
                                break

                        if folder_band is not None:
                            tele, inst = "SDSS", "SDSS"
                            h["TELESCOP"], h["INSTRUME"], h["FILTER"] = (
                                tele,
                                inst,
                                folder_band,
                            )
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
            # Promote to float32 early so NaNs can be represented without
            # doubling memory (float64) for full-frame images.
            image = image.astype(np.float32)
            safe_fits_write(fpath, image, header)

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
            # Fix header card ordering and ensure NAXIS values are valid
            try:
                # Ensure NAXIS values are set correctly from image shape first
                header['NAXIS'] = 2
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
                
                # Remove any None or invalid NAXIS values that could cause issues
                for key in ['NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4']:
                    if key in header:
                        val = header[key]
                        if val is None or not isinstance(val, int):
                            if key == 'NAXIS':
                                header[key] = 2
                            elif key == 'NAXIS1':
                                header[key] = image.shape[1]
                            elif key == 'NAXIS2':
                                header[key] = image.shape[0]
                            else:
                                del header[key]
                
                safe_fits_write(fpath, image, header)
            except Exception as e:
                logging.warning(f"Header write failed: {e}, using minimal header")
                # Create minimal header with just essential info
                minimal_header = fits.Header()
                minimal_header['NAXIS'] = 2
                minimal_header['NAXIS1'] = image.shape[1]
                minimal_header['NAXIS2'] = image.shape[0]
                # Copy essential WCS keywords if they exist
                for wcs_key in ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2']:
                    if wcs_key in header:
                        minimal_header[wcs_key] = header[wcs_key]
                safe_fits_write(fpath, image, minimal_header)

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
            raise ValueError(
                f"No telescope.yml entry found for telescope '{telescope}' and instrument '{instrument}'. "
                f"Please add an entry to telescope.yml or use check.py to build the database. "
                f"Supported telescopes/instruments can be found in telescope.yml."
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
            except Exception as e:
                logging.warning(f"Failed to convert date from header key {date_key}: {e}")
        if telescope == "MPI-2.2" and "TDP-MID" in header:
            date_mjd = header["TDP-MID"]
            logging.info("MPI-2.2MM detected (TDP-MID); MJD: %.3f", date_mjd)
        if not np.isfinite(date_mjd) or date_mjd == 0:
            date_mjd = Time.now().mjd
            logging.info(
                "Invalid MJD detected; setting MJD to today's value: %.1f", date_mjd
            )

        image_median = np.median(image)
        low_median_threshold = 1e-6
        high_gain_threshold = 1e6

        # DISABLED: Gain correction removed for consistency
        # The image should remain in ADU and gain should only be applied
        # internally within photometry functions (e.g., aperture.py line 599)
        # This ensures consistent handling across all photometry operations
        # and avoids ambiguity about whether the image is in ADU or electrons.
        # if image_median < low_median_threshold and gain > high_gain_threshold:
        #     logging.warning(
        #         "Applying gain %.1e to image (low median %.1e).", gain, image_median
        #     )
        #     image = image * gain
        #     safe_fits_write(fpath, image, header)
        #     gain = 1
        #     logging.warning("Gain reset to 1 after application.")

        #  Keep NaN values as NaN - chip gaps handled by background estimator
        #  which properly ignores NaN regions during background modeling
        safe_fits_write(fpath, image, header)

        #  Handle Saturation (telescope.yml may use saturate: not_given_by_user)
        # Priority:
        #   1) telescope.yml explicit keyword if valid and present in header
        #   2) fallback to standard 'SATURATE' header key if present
        #   3) otherwise, treat as effectively no hard saturation (inf)
        # Note: Use module-level constants for consistent SATURATE handling
        
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
            saturate = SATURATE_INTERNAL_FALLBACK

        # Validate saturate is a valid numeric type
        try:
            saturate = float(saturate)
        except (TypeError, ValueError):
            logging.warning(
                f"SATURATE keyword has invalid value '{saturate}' (not a number); "
                f"falling back to no saturation limit."
            )
            saturate = SATURATE_INTERNAL_FALLBACK

        input_yaml["saturate"] = saturate
        # FITS headers cannot store inf; only write saturate when it is finite.
        if np.isfinite(saturate):
            header["saturate"] = float(saturate)
        else:
            # For FITS compatibility, use a large finite value when saturate is inf
            header["saturate"] = SATURATE_FITS_FALLBACK

        #  Handle Read Noise and Airmass (telescope.yml may use not_given_by_user)
        rn_key = telescope_config.get("readnoise", "RDNOISE")
        readnoise = (
            header.get(rn_key, 0) if rn_key and rn_key != "not_given_by_user" else 0
        )
        am_key = telescope_config.get("airmass", "AIRMASS")
        airmass = (
            header.get(am_key, 1) if am_key and am_key != "not_given_by_user" else 1
        )

        input_yaml["read_noise"] = readnoise
        input_yaml["airmass"] = airmass
        header["RDNOISE"] = readnoise

        #  Handle Exposure Time — science frames must have a valid header; ZTF reference
        # templates often omit EXPTIME, so use default_input exposure_time for template paths only.
        primary_key = telescope_config.get("exptime", "EXPTIME")
        pref = [] if primary_key == "not_given_by_user" else [primary_key]
        try:
            exposure_time, used_exptime_key = exposure_seconds_from_header(header, pref)
        except ValueError as exc:
            if is_template:
                fallback = float(input_yaml.get("exposure_time", 30.0) or 30.0)
                if not (np.isfinite(fallback) and fallback > 0):
                    fallback = 30.0
                exposure_time = fallback
                used_exptime_key = "default_input.exposure_time (template)"
                logging.warning(
                    "No usable EXPTIME in template header for %s; using %.3g s from "
                    "default_input.exposure_time. (%s)",
                    fpath,
                    float(exposure_time),
                    exc,
                )
                header["EXPTIME"] = float(exposure_time)
                if "EXPOSURE" not in header:
                    header["EXPOSURE"] = float(exposure_time)
            else:
                raise ValueError(
                    f"No valid exposure time in FITS header for {fpath!r}: {exc}"
                ) from exc
        header["exptime"] = float(exposure_time)
        logging.info(
            "Exposure time:\t%.5g s (header keyword %s)",
            float(exposure_time),
            used_exptime_key,
        )

        # Placeholder
        ImageFWHM = None

        # =============================================================================
        #          Populate Output Dictionary
        # =============================================================================

        # Removed dead code - first output = OrderedDict assignment was completely replaced later

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
            except Exception as e:
                logging.warning(f"Failed to apply telescope.yml filter mapping for template: {e}")
        else:
            # Science image filter detection
            # Use telescope.yml filter key mappings
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
                # Use telescope.yml mapping if available; otherwise use heuristics
                imageFilter = None
                if telescope in telescope_data and instrument_key in telescope_data.get(telescope, {}):
                    mapping = telescope_data[telescope][instrument_key].get(instrument, {})
                    _meta = {
                        "Name", "filter_key_0", "mjd", "date", "gain",
                        "saturate", "readnoise", "airmass", "exptime", "pixel_scale",
                    }
                    imageFilter = mapping.get(raw_filter)
                    if imageFilter is None:
                        for k, v in mapping.items():
                            if k not in _meta and isinstance(v, str) and k.strip().upper() == raw_filter.strip().upper():
                                imageFilter = v
                                break
                
                # Telescope-blind: if no mapping found, use heuristic to derive filter
                if imageFilter is None:
                    imageFilter = _heuristic_filter_mapping(raw_filter)
                    if imageFilter != raw_filter:
                        logging.info(
                            f"Filter '{raw_filter}' mapped to '{imageFilter}' via heuristic (no telescope.yml entry)"
                        )
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
                if imageWCS is None:
                    # Debug: show what WCS keywords are present
                    wcs_keys = [k for k in header.keys() if k.startswith(('CRPIX', 'CRVAL', 'CDELT', 'CTYPE', 'CD1_', 'CD2_', 'PC1_', 'PC2_', 'PV'))]
                    logging.debug(f"WCS keywords in header: {wcs_keys}")
                    logging.debug(f"CTYPE1/CTYPE2: {header.get('CTYPE1', 'N/A')}, {header.get('CTYPE2', 'N/A')}")
                    pixel_scale_candidate = np.nan
                else:
                    xy_pixel_scales = proj_plane_pixel_scales(imageWCS)
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
            logging.info("Pixel scale:\t%.3f arcsec/pixel", pixel_scale)
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
                            "Pixel scale:\t%.3f arcsec/pixel (telescope.yml)",
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

        #  Handle Gain — after instrument-specific header patches (e.g. GROND IR).
        # telescope.yml gain can be either a numeric value (e-/ADU) or a header keyword.
        primary_gain = telescope_config.get("gain", "GAIN")
        
        # Check if telescope.yml gain is a numeric value (direct gain in e-/ADU)
        if isinstance(primary_gain, (int, float)) and primary_gain > 0:
            gain = float(primary_gain)
            gain_header_key = f"telescope.yml_gain_{primary_gain}"
            logging.info(
                "Gain:\t\t%.5g e-/ADU (from telescope.yml)", gain
            )
        elif primary_gain == "not_given_by_user":
            # No gain specified in telescope.yml, use header lookup
            try:
                gain, gain_header_key = gain_e_per_adu_from_header(header, [])
                logging.info(
                    "Gain:\t\t%.5g e-/ADU (header keyword %s)", float(gain), gain_header_key
                )
            except ValueError as exc:
                img_type = "template" if is_template else "science"
                logging.warning(
                    "%s image has no usable GAIN in FITS header (%s). "
                    "Falling back to input_yaml['gain']=%.2f e-/ADU. "
                    "Set 'gain' in input YAML or telescope.yml if this is incorrect.",
                    img_type.capitalize(),
                    exc,
                    float(input_yaml.get("gain", 1.0)),
                )
                gain = float(resolve_gain_e_per_adu(None, input_yaml))
                gain_header_key = f"fallback_from_yaml_gain_{gain:.2f}"
                logging.info(
                    "Gain:\t\t%.5g e-/ADU (from input_yaml fallback)", float(gain)
                )
        else:
            # telescope.yml gain is a header keyword string
            pref_gain = [primary_gain]
            try:
                gain, gain_header_key = gain_e_per_adu_from_header(header, pref_gain)
                logging.info(
                    "Gain:\t\t%.5g e-/ADU (header keyword %s)", float(gain), gain_header_key
                )
            except ValueError as exc:
                img_type = "template" if is_template else "science"
                logging.warning(
                    "%s image has no usable GAIN in FITS header (%s). "
                    "Falling back to input_yaml['gain']=%.2f e-/ADU. "
                    "Set 'gain' in input YAML or telescope.yml if this is incorrect.",
                    img_type.capitalize(),
                    exc,
                    float(input_yaml.get("gain", 1.0)),
                )
                gain = float(resolve_gain_e_per_adu(None, input_yaml))
                gain_header_key = f"fallback_from_yaml_gain_{gain:.2f}"
                logging.info(
                    "Gain:\t\t%.5g e-/ADU (from input_yaml fallback)", float(gain)
                )

        #  Update WCS Pixel Scale
        # Updates the pixel scale in the input YAML.
        input_yaml["wcs"]["pixel_scale"] = pixel_scale

        #  Update Target Coordinates
        # Updates the target coordinates if provided.
        if input_yaml["target_name"] is not None:
            target_ra = input_yaml["target_ra"]
            target_dec = input_yaml["target_dec"]
            # Validate that coordinates are provided when target_name is set
            if target_ra is None or target_dec is None:
                logging.error(
                    f"Target name '{input_yaml['target_name']}' provided but "
                    f"target_ra={target_ra}, target_dec={target_dec}. "
                    "Please provide valid coordinates or remove target_name."
                )
                raise ValueError(
                    f"Missing coordinates for target '{input_yaml['target_name']}'. "
                    "Provide target_ra/target_dec or remove target_name."
                )
            target_coords = SkyCoord(
                target_ra, target_dec, unit=(u.deg, u.deg), frame="icrs"
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
                frame="icrs",
            )
            input_yaml["target_ra"] = target_coords.ra.degree
            input_yaml["target_dec"] = target_coords.dec.degree
        else:
            try:
                use_hdr = input_yaml.get("use_header_radec", False)
                if use_hdr:
                    # Allow True (use default keys) or an explicit (RA_KEY, DEC_KEY) mapping.
                    ra_key, dec_key = "CAT-RA", "CAT-DEC"
                    if isinstance(use_hdr, (list, tuple)) and len(use_hdr) >= 2:
                        ra_key, dec_key = str(use_hdr[0]).strip(), str(use_hdr[1]).strip()
                    elif isinstance(use_hdr, str) and "," in use_hdr:
                        parts = [p.strip() for p in use_hdr.split(",") if p.strip()]
                        if len(parts) >= 2:
                            ra_key, dec_key = parts[0], parts[1]

                    if ra_key in header and dec_key in header:
                        ra_val = header[ra_key]
                        dec_val = header[dec_key]

                        # Heuristic units:
                        # - sexagesimal RA strings -> hourangle
                        # - numeric values -> degrees
                        ra_is_str = isinstance(ra_val, str)
                        dec_is_str = isinstance(dec_val, str)
                        ra_unit = u.hourangle if (ra_is_str and ":" in ra_val) else u.deg
                        dec_unit = u.deg

                        target_coords = SkyCoord(
                            ra_val,
                            dec_val,
                            unit=(ra_unit, dec_unit),
                            frame="icrs",
                        )
                        input_yaml["target_ra"] = float(target_coords.ra.degree)
                        input_yaml["target_dec"] = float(target_coords.dec.degree)
                        logging.info(
                            "Using target RA/Dec from header keys %s/%s: RA=%.6f deg, Dec=%.6f deg",
                            ra_key,
                            dec_key,
                            float(target_coords.ra.degree),
                            float(target_coords.dec.degree),
                        )
                    else:
                        logging.info(
                            "use_header_radec enabled but header keys not found: RA_KEY=%s present=%s, DEC_KEY=%s present=%s",
                            ra_key,
                            ra_key in header,
                            dec_key,
                            dec_key in header,
                        )
            except Exception as e:
                logging.info("No RA/DEC keywords found (%s).", e)

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

        # Format observation date and time first (as requested)
        date = Time([date_mjd], format="mjd", scale="utc")
        date_str = date.iso[0].split(" ")[0]  # Date part
        time_str = date.iso[0].split(" ")[1]  # Time part
        
        # Format time as requested (e.g., "8:00pm")
        try:
            time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
            time_obj = datetime.datetime.strptime(time_obj.strftime("%H:%M:%S"), "%H:%M:%S")
            formatted_time = time_obj.strftime("%-I%p").lower()
        except Exception:
            formatted_time = time_str  # Fallback to original format if parsing fails
        
        # Format date as dd-mm-year
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%d-%m-%Y")
        except Exception:
            formatted_date = date_str  # Fallback to original format if parsing fails
        
        logging.info(
            "Observation:\t%s at %s | Telescope: %s, Instrument: %s, Filter: %s",
            formatted_date, formatted_time, telescope, instrument, imageFilter,
        )

        if pixel_scale:
            input_yaml["pixel_scale"] = pixel_scale

        if np.isfinite(saturate) and saturate != SATURATE_INTERNAL_FALLBACK:
            logging.info(f"Saturation:\t{saturate:.1f} ADU")
        else:
            logging.info("Saturation:\tnot available (no limit)")

        if readnoise > 0:
            logging.info(f"Read noise:\t{readnoise:.3f} e-")

        if "airmass" in input_yaml and input_yaml["airmass"]:
            logging.info(f"Airmass:\t{input_yaml['airmass']:.3f}")

        header["gain"] = gain
        # saturate already written to header at line 1130 (if finite)
        # RDNOISE already written to header at line 1143

        # =============================================================================
        # Image Preprocessing
        # =============================================================================
        logging.info(border_msg("Image preprocessing"))

        # -------------------------------------------------------------------------
        # Trim NaN boundaries (chip gaps / no-coverage edges)
        # -------------------------------------------------------------------------
        trim_nan_edges = input_yaml["preprocessing"].get("trim_nan_edges", True)
        if trim_nan_edges:
            try:
                # Get target pixel coordinates to ensure target stays in image.
                target_x, target_y = None, None
                if "target_ra" in input_yaml and "target_dec" in input_yaml:
                    try:
                        wcs = get_wcs(header, silent=True)
                        if wcs is not None and wcs.has_celestial:
                            target_x, target_y = wcs.all_world2pix(
                                float(input_yaml["target_ra"]),
                                float(input_yaml["target_dec"]),
                                0,
                            )
                            target_x, target_y = float(target_x), float(target_y)
                            logging.info(
                                "Target position for NaN trimming: (%.1f, %.1f) px",
                                target_x,
                                target_y,
                            )
                    except Exception as wcs_exc:
                        logging.warning(
                            "Could not get target pixel coords for NaN trimming: %s",
                            wcs_exc,
                        )

                buffer = input_yaml["preprocessing"].get("nan_trim_buffer", 10)
                image, header, trim_info = _trim_nan_boundaries(
                    image,
                    header,
                    target_x=target_x,
                    target_y=target_y,
                    buffer_pixels=buffer,
                )
                if trim_info.get("trimmed", False):
                    logging.info(
                        "Trimmed NaN boundaries: %s -> %s",
                        trim_info.get("original_shape"),
                        trim_info.get("trimmed_shape"),
                    )
                    safe_fits_write(fpath, image, header)

                    # Refresh WCS and recalculate target pixel coordinates.
                    try:
                        new_wcs = _safe_wcs_from_header(header, silent=True)
                        if (
                            new_wcs is not None
                            and "target_ra" in input_yaml
                            and "target_dec" in input_yaml
                        ):
                            new_target_x, new_target_y = new_wcs.all_world2pix(
                                float(input_yaml["target_ra"]),
                                float(input_yaml["target_dec"]),
                                0,
                            )
                            input_yaml["target_x_pix"] = float(new_target_x)
                            input_yaml["target_y_pix"] = float(new_target_y)
                            logging.info(
                                "Target pixel coordinates refreshed after trimming: (%.1f, %.1f) px",
                                float(new_target_x),
                                float(new_target_y),
                            )
                        elif new_wcs is None:
                            logging.warning(
                                "WCS not available after trimming; cannot refresh target coordinates"
                            )
                    except Exception as wcs_refresh_exc:
                        logging.warning(
                            "Could not refresh target coordinates after trimming: %s",
                            wcs_refresh_exc,
                        )
            except Exception as trim_exc:
                logging.warning("NaN boundary trimming failed: %s", trim_exc)

        #  Image Trimming
        # Trims the image to a specified size centered on the target.
        trim_image = input_yaml["preprocessing"].get("trim_image", 0)
        do_trim = trim_image > 0
        
        if do_trim:
            # Check if requested trim size is larger than image
            try:
                pixel_scale = float(input_yaml.get("pixel_scale", 1.0))  # arcsec/pixel
            except (ValueError, TypeError):
                pixel_scale = 1.0
            if not np.isfinite(pixel_scale) or pixel_scale <= 0:
                pixel_scale = 1.0
            
            # Estimate image size in arcmin
            image_width_arcmin = image.shape[1] * pixel_scale / 60.0
            image_height_arcmin = image.shape[0] * pixel_scale / 60.0
            min_image_dim_arcmin = min(image_width_arcmin, image_height_arcmin)
            
            # If trim_image is larger than image, skip trimming
            if trim_image >= min_image_dim_arcmin:
                logging.warning(
                    f"Requested trim size ({trim_image} arcmin) is larger than image "
                    f"({image_width_arcmin:.1f} x {image_height_arcmin:.1f} arcmin). "
                    f"Skipping trim operation."
                )
                do_trim = False
        
        if do_trim:
            try:
                logging.info(
                    log_step(
                        f"Trim: {base_filename} to {trim_image} arcmin (target center)"
                    )
                )
                imageWCS = get_wcs(header)
                
                # Check if WCS is valid for celestial coordinates
                if imageWCS is None or not hasattr(imageWCS, 'celestial') or imageWCS.celestial is None:
                    logging.warning("WCS is not suitable for celestial coordinates, using pixel-based trimming")
                    # Fall back to pixel-based trimming using target pixel coords
                    if 'target_x_pix' in input_yaml and 'target_y_pix' in input_yaml:
                        center_x = input_yaml['target_x_pix']
                        center_y = input_yaml['target_y_pix']
                    else:
                        # Use image center as fallback
                        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
                    
                    # Calculate trim size in pixels (assume 1 arcsec/pixel if no scale)
                    try:
                        pixel_scale = float(input_yaml.get("pixel_scale", 1.0))  # arcsec/pixel
                    except (ValueError, TypeError):
                        pixel_scale = 1.0
                    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
                        pixel_scale = 1.0
                    trim_pixels = int((trim_image * 60) / pixel_scale)  # Convert arcmin to pixels
                    
                    # Create pixel-based cutout using nan_crop to preserve WCS
                    from functions import nan_crop
                    image, header = nan_crop(
                        image.astype(float), header,
                        center_x, center_y,
                        trim_pixels, trim_pixels,
                    )
                else:
                    # Use WCS to find pixel center, then nan_crop
                    target_x_px, target_y_px = imageWCS.all_world2pix(
                        input_yaml["target_ra"], input_yaml["target_dec"], 0
                    )
                    trim_pixels = int((trim_image * 60) / pixel_scale)
                    from functions import nan_crop
                    image, header = nan_crop(
                        image.astype(float), header,
                        target_x_px, target_y_px,
                        trim_pixels, trim_pixels,
                    )

                # Writes the modified image and header back to the FITS file.
                safe_fits_write(fpath, image, header)
                logging.info(f"New image shape after trimming: {image.shape}")

                # Trim NaN boundaries created by nan_crop (fill_value=np.nan)
                # This must happen before background subtraction
                # Calculate center of trimmed image for target preservation
                # Correct numpy 0-based center is (nx-1)/2, (ny-1)/2.
                center_x = (image.shape[1] - 1) / 2.0
                center_y = (image.shape[0] - 1) / 2.0
                logging.info(f"Attempting NaN boundary trimming after 5 arcmin cutout: center=({center_x:.1f}, {center_y:.1f})")
                buffer = input_yaml["preprocessing"].get("nan_trim_buffer", 10)
                image, header, trim_info = _trim_nan_boundaries(
                    image, header, target_x=center_x, target_y=center_y, buffer_pixels=buffer
                )
                logging.info(f"NaN boundary trimming result: trimmed={trim_info['trimmed']}, trim_info={trim_info}")
                if trim_info["trimmed"]:
                    logging.info(
                        f"Trimmed NaN boundaries after 5 arcmin cutout: {trim_info['original_shape']} -> {trim_info['trimmed_shape']}"
                    )
                    safe_fits_write(fpath, image, header)
                else:
                    logging.warning("NaN boundary trimming returned trimmed=False - no NaNs removed")
            except Exception as e:
                logging.warning(f"Could not trim image: {e}; ignoring the operation.")

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
                    log_step(
                        f"Recrop: {base_filename} (remove uniform edge rows/cols)"
                    )
                )
                position = (center_x, center_y)  # (x, y) position
                # Use nan_crop to preserve WCS distortion keywords.
                from functions import nan_crop
                image, header = nan_crop(
                    image, header, center_x, center_y, height, width
                )

                # Writes the modified image and header back to the FITS file.
                safe_fits_write(fpath, image, header)
                logging.info(f"New image shape after recropping: {image.shape}")

            # Use the in-memory image/header (already updated) to refresh YAML
            # dimensions instead of re-opening the FITS file.
            input_yaml["NAXIS1"] = image.shape[1]
            input_yaml["NAXIS2"] = image.shape[0]
        except Exception as e:
            log_exception(e)
            logging.info("Could not recrop the image; operation ignored.")

        # Refresh WCS and target pixel coordinates after all trimming
        # This ensures catalog lookup uses correct coordinates for trimmed image
        try:
            imageWCS = get_wcs(header)
            if imageWCS is not None and hasattr(imageWCS, 'celestial') and imageWCS.celestial is not None:
                if "target_ra" in input_yaml and "target_dec" in input_yaml:
                    target_ra = float(input_yaml["target_ra"])
                    target_dec = float(input_yaml["target_dec"])
                    target_x_pix, target_y_pix = imageWCS.all_world2pix(
                        target_ra, target_dec, 0
                    )
                    input_yaml["target_x_pix"] = float(target_x_pix)
                    input_yaml["target_y_pix"] = float(target_y_pix)
                    logging.info(
                        f"Target pixel coordinates refreshed after trimming: "
                        f"({target_x_pix:.1f}, {target_y_pix:.1f}) px"
                    )
        except Exception as wcs_refresh_exc:
            logging.warning(f"Could not refresh target coordinates after trimming: {wcs_refresh_exc}")

        # =============================================================================
        #   Check if target is in NaN region (chip gap)
        # =============================================================================
        if "target_x_pix" in input_yaml and "target_y_pix" in input_yaml:
            try:
                # Reload image to check target pixel value
                check_image = get_image(fpath)
                tx = int(round(float(input_yaml["target_x_pix"])))
                ty = int(round(float(input_yaml["target_y_pix"])))
                
                # Check bounds and NaN
                if (0 <= ty < check_image.shape[0] and 0 <= tx < check_image.shape[1]):
                    if np.isnan(check_image[ty, tx]):
                        logging.error(
                            f"Target pixel ({tx}, {ty}) is NaN (chip gap). "
                            f"Skipping image {os.path.basename(fpath)}."
                        )
                        return
                else:
                    logging.error(
                        f"Target pixel ({tx}, {ty}) is outside image bounds "
                        f"({check_image.shape[1]}, {check_image.shape[0]}). "
                        f"Skipping image {os.path.basename(fpath)}."
                    )
                    return
            except Exception as check_exc:
                logging.warning(f"Could not check target pixel value: {check_exc}")

        # =============================================================================
        #   Run SExtractor
        # =============================================================================
        logging.info(border_msg("Source detection and FWHM"))
        # Runs SExtractor to measure the image FWHM and detect sources.
        try:
            ImageFWHM, FWHMSources, scale = SExtractorWrapper(config=input_yaml).run(
                fpath,
                crowded=input_yaml.get("photometry", {}).get("crowded_field", False),
            )
        except Exception as e:
            log_exception(e, "SEXtractor failed - trying pythonic source detection")

            # Load image for pythonic detection
            image = get_image(fpath)

            # Measures the image FWHM, isolated sources, and scale.
            ImageFWHM, FWHMSources, scale = Find_FWHM(
                input_yaml=input_yaml
            ).measure_image(
                image=image,
            )

        # Filter sources near NaN/masked regions to improve FWHM accuracy
        if FWHMSources is not None and len(FWHMSources) > 0:
            temp_image = get_image(fpath)
            nan_mask = np.isnan(temp_image)
            if np.any(nan_mask):
                from scipy.spatial import cKDTree
                masked_pixels = np.argwhere(nan_mask)
                tree = cKDTree(masked_pixels)
                source_coords = FWHMSources[["x_pix", "y_pix"]].values
                min_distances, _ = tree.query(source_coords, k=1, distance_upper_bound=2 * ImageFWHM)
                n_before = len(FWHMSources)
                FWHMSources = FWHMSources[min_distances > 2 * ImageFWHM]
                n_excluded = n_before - len(FWHMSources)
                if n_excluded > 0:
                    logging.info(
                        f"Excluded {n_excluded} sources within 2xFWHM of NaN/masked regions for FWHM calculation"
                    )

        # =============================================================================
        # Measuring the background statistics (don't remove it)
        # =============================================================================
        logging.info(border_msg("Background: initial pass"))
        # Creates a background remover instance.

        bg_remover = BackgroundSubtractor(input_yaml)

        # Removes the background (without plotting).
        result = bg_remover.remove(image, plot=False, fwhm=ImageFWHM)

        # Accesses the results.
        # Keep large background products in float32/bool to reduce memory footprint.
        background_surface = np.asarray(result["background"], dtype=np.float32)
        background_rms = np.asarray(result["background_rms"], dtype=np.float32)
        defects_mask = np.asarray(result["defects_mask"], dtype=bool)
        hardware_defects_mask = np.asarray(result["hardware_defects_mask"], dtype=bool)
        # Drop the result dict — it holds references to 4-6 full-image arrays
        # (background, rms, defects_mask, hardware_defects_mask, source_mask,
        # subtracted image) that are no longer needed here.  Without this the
        # dict keeps all of them alive until a new `result =` assignment.
        del result

        logging.info(f"Preliminary FWHM: {ImageFWHM:.1f} pixels")

        # Global undersampled-mode flag for consistent behavior across modules.
        phot_cfg = input_yaml.get("photometry", {}) or {}
        undersampled_thr = float(phot_cfg.get("undersampled_fwhm_threshold", 2.5))
        input_yaml["undersampled_mode"] = bool(
            np.isfinite(ImageFWHM) and float(ImageFWHM) <= undersampled_thr
        )
        logging.info(
            "Undersampled mode: %s (FWHM=%.2f px, threshold=%.2f px)",
            input_yaml["undersampled_mode"],
            float(ImageFWHM),
            undersampled_thr,
        )

        header["fwhm"] = ImageFWHM

        # Set WCS profile to "crowded" from initial FWHM/background when crowded_auto is True.
        # Uses source count and density from the same run that produced ImageFWHM/FWHMSources.
        wcs_cfg = input_yaml.get("wcs") or {}
        if wcs_cfg.get("crowded_auto", True):
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
                    except Exception as e:
                        logging.debug(f"Failed to check crowded field condition: {e}")
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
        cosmic_rays_mask = np.zeros(image.shape, dtype=bool)
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
            logging.info(
                log_step(
                    f"Remove cosmic rays / streaks: {base_filename}"
                )
            )
            use_lacosmic = input_yaml["cosmic_rays"].get("use_lacosmic", False)
            _cr_cfg = input_yaml.get("cosmic_rays", {})
            _cr_plot = bool(_cr_cfg.get("cr_plot", False))
            _cr_sigclip = float(_cr_cfg.get("cr_sigclip", 4.5))
            _cr_sigfrac = float(_cr_cfg.get("cr_sigfrac", 0.3))
            _cr_objlim = float(_cr_cfg.get("cr_objlim", 10.0))
            _cr_dilate_factor = float(_cr_cfg.get("cr_dilate_factor", 1.0))
            _cr_dilate_iters = int(_cr_cfg.get("cr_dilate_iterations", 2))
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
                readnoise=readnoise,
                satlevel=saturate,
                psf_fwhm=ImageFWHM,
                sigclip=_cr_sigclip,
                sigfrac=_cr_sigfrac,
                objlim=_cr_objlim,
                dilate_factor=_cr_dilate_factor,
                dilate_iterations=_cr_dilate_iters,
                plot=_cr_plot,
            )

        # =============================================================================
        #          Check for Existing WCS
        # =============================================================================
        logging.info(border_msg("WCS: check header and plate solve"))
        # Checks if there is an existing WCS in the header.

        existingWCS = False
        updated_header = None

        # Tries to read the existing WCS.
        try:
            with SuppressStdout():
                WCSvalues_old = get_wcs(header)
            if WCSvalues_old is None:
                raise ValueError("get_wcs returned None (missing required WCS keywords)")
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
                    n_detected_sources=len(FWHMSources) if FWHMSources is not None else None,
                )
            if updated_header is None or (
                isinstance(updated_header, float) and np.isnan(updated_header)
            ):
                logging.info("Plate solve returned NaN or None")
                allow_fallback_on_fail = bool(
                    input_yaml.get("wcs", {}).get(
                        "allow_fallback_to_existing_on_solve_fail", False
                    )
                )
                if existingWCS and allow_fallback_on_fail:
                    logging.info("Falling back to pre-existing WCS")
                    from functions import update_header_from_wcs
                    update_header_from_wcs(header, WCSvalues_old)
                    break
                else:
                    raise Exception(
                        "WCS solve failed and fallback to existing WCS is disabled."
                    )
            else:
                logging.info("Plate solve successful")
                if apply_solved_to_fits:
                    header = updated_header
                    safe_fits_write(fpath, image, header)
                    wcs_updated = True
                else:
                    # Use solved WCS only to update pixel_scale in YAML; leave FITS header unchanged for better subtraction
                    try:
                        _solved_wcs = get_wcs(updated_header)
                        _xy = proj_plane_pixel_scales(_solved_wcs)
                        input_yaml["pixel_scale"] = float(_xy[0] * 3600)
                        logging.info(
                            "apply_solved_to_fits=False: keeping original WCS in FITS; updated pixel_scale in config only"
                        )
                    except Exception as e:
                        logging.warning(f"Failed to update pixel_scale from solved WCS: {e}")
                break  # Exits after one successful attempt.
        # Use current header (solved or original) for rest of pipeline
        imageWCS = get_wcs(header)
        if imageWCS is None and updated_header is not None:
            # Try the solved header object if present.
            try:
                imageWCS = get_wcs(updated_header)
                if imageWCS is not None and apply_solved_to_fits:
                    header = updated_header
                    safe_fits_write(fpath, image, header)
                    logging.info(
                        "Recovered usable WCS from solved header and wrote it to FITS."
                    )
            except Exception:
                imageWCS = None

        # Gets the pixel scale in arcseconds.
        if imageWCS is None:
            # Graceful fallback: allow pipeline to continue with a configured pixel scale.
            fallback = input_yaml.get("pixel_scale", None)
            if fallback is None:
                fallback = (input_yaml.get("wcs", {}) or {}).get(
                    "pixel_scale_arcsec", None
                )
            try:
                pixel_scale = float(fallback) if fallback is not None else float("nan")
            except Exception:
                pixel_scale = float("nan")
            if not np.isfinite(pixel_scale) or pixel_scale <= 0:
                logging.error(
                    "Failed to create WCS from header after solve and no valid fallback "
                    "pixel_scale is configured. Missing keywords likely include CRPIX/CRVAL."
                )
                raise Exception("WCS is None after solve - cannot compute pixel scale")
            logging.warning(
                "WCS unavailable after solve (missing CRPIX/CRVAL etc). "
                "Falling back to configured pixel_scale=%.6g arcsec/px; "
                "WCS-dependent features may be degraded.",
                pixel_scale,
            )
        else:
            xy_pixel_scales = proj_plane_pixel_scales(imageWCS)
            pixel_scale = float(xy_pixel_scales[0] * 3600.0)

        # Sets the range for which the PSF model can move around.
        input_yaml["dx"] = np.ceil(ImageFWHM)
        input_yaml["dy"] = np.ceil(ImageFWHM)

        # Updates the pixel scale in the input YAML.
        input_yaml["pixel_scale"] = pixel_scale

        # =============================================================================
        # WCS Refinement (optional: enable via wcs.refine_after_solve if needed)
        # =============================================================================
        if input_yaml.get("wcs", {}).get("refine_after_solve", False) and wcs_updated:
            logging.warning(
                "wcs.refine_after_solve=True but refine_image is not implemented; "
                "skipping WCS refinement pass."
            )

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
                frame="icrs",
            )
            # Converts sky coordinates to pixel coordinates using WCS.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x_pix, y_pix = imageWCS.world_to_pixel(coords)
            except Exception as exc:
                log_warning_from_exception(
                    logging.getLogger(),
                    "Variable-source WCS transform failed for bulk conversion; "
                    "retrying per source and skipping non-convergent points",
                    exc,
                )
                # Try bulk conversion with quiet=True so WCS returns NaN for
                # non-convergent sources instead of raising (avoids per-source loop).
                ra_arr = variable_sources["RA"].to_numpy(dtype=float)
                dec_arr = variable_sources["DEC"].to_numpy(dtype=float)
                try:
                    xy = imageWCS.all_world2pix(
                        np.column_stack([ra_arr, dec_arr]), 0, quiet=True
                    )
                    x_pix = xy[:, 0].astype(float)
                    y_pix = xy[:, 1].astype(float)
                except Exception:
                    x_pix = np.full(len(variable_sources), np.nan, dtype=float)
                    y_pix = np.full(len(variable_sources), np.nan, dtype=float)
                n_failed = int(np.count_nonzero(~np.isfinite(x_pix) | ~np.isfinite(y_pix)))
                if n_failed > 0:
                    logging.warning(
                        "Skipped %d/%d variable sources due to non-convergent WCS distortion inversion.",
                        n_failed,
                        len(variable_sources),
                    )
            # Gets image size.
            height, width = image.shape
            # Creates a mask of sources inside the image boundaries.
            inside_mask = (
                np.isfinite(x_pix)
                & np.isfinite(y_pix)
                & (x_pix >= 0)
                & (x_pix < width)
                & (y_pix >= 0)
                & (y_pix < height)
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

        # Use origin=0 for consistent 0-based indexing (matching numpy arrays)
        target_x_expected, target_y_expected = imageWCS.all_world2pix(
            input_yaml["target_ra"], input_yaml["target_dec"], 0
        )
        logging.info(
            f"TNS check:\t{input_yaml.get('target_name', 'Transient')} | "
            f"RA/Dec: {input_yaml['target_ra']:.6f}, {input_yaml['target_dec']:.6f} | "
            f"Pixel: ({target_x_expected:.2f}, {target_y_expected:.2f})"
        )

        # =============================================================================
        # Remove the background
        # =============================================================================
        logging.info(
            border_msg("Background: masked pass (variable sources, SIMBAD)")
        )

        result = bg_remover.remove(
            image,
            header=header,
            plot=True,
            fwhm=ImageFWHM,
            galaxies=variable_sources,
            mask_simbad_galaxies=True,
        )

        background_surface = np.asarray(result["background"], dtype=np.float32)
        background_rms = np.asarray(result["background_rms"], dtype=np.float32)
        defects_mask = np.asarray(result["defects_mask"], dtype=bool)
        hardware_defects_mask = np.asarray(result["hardware_defects_mask"], dtype=bool)
        del result

        # Cache background results for potential reuse after template subtraction
        fpath_before_subtraction = fpath
        # These arrays can be full-frame (many MB). Keep a single shared copy
        # and avoid duplicating them unless we truly need to mutate them.
        background_surface_cached = background_surface
        background_rms_cached = background_rms
        defects_mask_cached = defects_mask
        hardware_defects_mask_cached = hardware_defects_mask

        # Background subtraction enabled (user request).
        # Only background surface is removed; no other flux scaling applied.
        image -= background_surface

        # Writes the modified image and header back to the file.
        safe_fits_write(fpath, image, header)
        # Save the background_rms array with '.weight' inserted before the suffix
        base, ext = os.path.splitext(fpath)
        weight_fpath = f"{base}.weight{ext}"
        safe_fits_write(weight_fpath, background_rms, header)
        del background_surface
        # Keep using in-memory image, header (no reload needed)

        # =============================================================================
        # Target Position
        # =============================================================================

        #  Get Target Pixel Location
        # Gets the target pixel location using the WCS.
        imageWCS = get_wcs(header)  # WCS values
        xy_pixel_scales = proj_plane_pixel_scales(imageWCS)
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
                # Correct numpy 0-based center is (nx-1)/2, (ny-1)/2.
                center_pix = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
                center = imageWCS.all_pix2world([center_pix[0]], [center_pix[1]], 0)
                target_coords = SkyCoord(
                    center[0][0],
                    center[1][0],
                    unit=(u.deg, u.deg),
                    frame="icrs",
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
                    frame="icrs",
                )
                target_x_pix, target_y_pix = update_target_pixel_coords(
                    input_yaml, imageWCS, wcs_origin
                )
                if not (
                    (0 <= target_x_pix < image.shape[1])
                    and (0 <= target_y_pix < image.shape[0])
                ):
                    logging.warning(
                        "Target outside image bounds (0,0)-(%d,%d)",
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
                    frame="icrs",
                )
                target_x_pix, target_y_pix = imageWCS.all_world2pix(
                    target_ra, target_dec, wcs_origin
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
                    frame="icrs",
                )
                input_yaml["target_ra"] = target_coords.ra.degree
                input_yaml["target_dec"] = target_coords.dec.degree
                target_x_pix, target_y_pix = update_target_pixel_coords(
                    input_yaml, imageWCS, wcs_origin
                )
            except Exception as e:
                logging.error("FAILED to get coordinates from FITS header: %s", e)
                logging.exception("NO RA/DEC keywords found")
                raise

        # Structured summary: easier to scan than a single long line.
        nx, ny = image.shape[1], image.shape[0]
        ra = input_yaml["target_ra"]
        dec = input_yaml["target_dec"]
        x = input_yaml["target_x_pix"]
        y = input_yaml["target_y_pix"]
        in_bounds = (0 <= x < nx) and (0 <= y < ny)
        target_label = str(input_yaml.get("target_name", "Transient"))
        x_center = 0.5 * (nx - 1)
        y_center = 0.5 * (ny - 1)
        dx_center = float(x - x_center)
        dy_center = float(y - y_center)
        border_margin_px = float(min(x, y, (nx - 1) - x, (ny - 1) - y))
        logging.info(
            "Target:\t\t%s | Sky: RA %.6f, Dec %.6f | Pixel: (%.2f, %.2f) | "
            "Image: %dx%d px | Offset: dx=%+.2f, dy=%+.2f px | "
            "Bounds: %s (margin %.2f px)",
            target_label, ra, dec, x, y, nx, ny, dx_center, dy_center,
            "within bounds" if in_bounds else "OUTSIDE bounds", border_margin_px,
        )

        # =============================================================================
        # Source Masking
        # =============================================================================
        # Ensure target pixel coordinates are consistent with the final WCS.
        target_x_pix, target_y_pix = update_target_pixel_coords(
            input_yaml, imageWCS, wcs_origin
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
            logging.info(border_msg("Template: locate, align, and match"))
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
                            log_step("No template images — skip subtraction")
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

                        # Diagnostic plot: source alignment after template alignment
                        try:
                            makePlots = Plot(input_yaml=input_yaml)
                            makePlots.plot_alignment_offset(
                                sci_fpath=fpath,
                                template_fpath=templateFpath,
                            )
                        except Exception as exc:
                            logging.warning(f"Alignment offset plot failed: {exc}")

                        try:
                            _wcs_apply = input_yaml.get("wcs", {}).get(
                                "apply_solved_to_fits", True
                            )
                            _am = str(
                                input_yaml.get("template_subtraction", {}).get(
                                    "alignment_method", ""
                                )
                            ).lower()
                            if (
                                not _wcs_apply
                                and _am == "reproject"
                            ):
                                logging.warning(
                                    "alignment_method=reproject but wcs.apply_solved_to_fits "
                                    "is False: reprojection uses the WCS **on disk**, which may "
                                    "differ from a newer plate solution computed in memory. "
                                    "Expect residual source misalignment / dipoles. Fix: set "
                                    "apply_solved_to_fits: True, or use alignment_method "
                                    "'astroalign' / 'swarp'."
                                )
                        except Exception as e:
                            logging.warning(f"Template alignment check failed: {e}")

                except Exception as e:
                    log_exception(e, "Template alignment failed")
                    template_available = False
                    input_yaml["template_subtraction"]["do_subtraction"] = False

            except Exception as e:
                log_exception(e, "Template subtraction failed")
                template_available = False
        
        
        # =============================================================================
        # Source Masking
        # =============================================================================
        # Ensure target pixel coordinates are consistent with the final WCS.
        
        # Reload in case template alignment overwrote fpath (single open for both).
        image, header = get_image_and_header(fpath)
        imageWCS = get_wcs(header)
        

        target_x_pix, target_y_pix = update_target_pixel_coords(
            input_yaml, imageWCS, wcs_origin
        )
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix



        # =============================================================================
        # Get background statistics after subtraction
        # =============================================================================
        logging.info(
            border_msg("Calibration stage: image, background, and weight")
        )


        # Reuse cached background results if fpath hasn't changed
        if fpath == fpath_before_subtraction:
            background_surface = background_surface_cached
            background_rms = background_rms_cached
            defects_mask = defects_mask_cached
            hardware_defects_mask = hardware_defects_mask_cached
            logging.info("Reusing cached background results (fpath unchanged)")
        else:
            result = bg_remover.remove(
                image,
                header=header,
                plot=False,
                fwhm=ImageFWHM,
                galaxies=variable_sources,
                mask_simbad_galaxies=True,
            )

            background_surface = np.asarray(result["background"], dtype=np.float32)
            background_rms = np.asarray(result["background_rms"], dtype=np.float32)
            defects_mask = np.asarray(result["defects_mask"], dtype=bool)
            hardware_defects_mask = np.asarray(result["hardware_defects_mask"], dtype=bool)
            del result

        # Save the background_rms array with '.weight' inserted before the suffix
        base, ext = os.path.splitext(fpath)
        weight_fpath = f"{base}.weight{ext}"
        safe_fits_write(weight_fpath, background_rms, header)

        # =============================================================================
        #     Get a Reference catalog
        # =============================================================================
        logging.info(border_msg("Reference photometric catalog"))
        # Creates an instance of the catalog class.
        Calibrate_Catalog = Catalog(input_yaml=input_yaml)

        # Builds or downloads the catalog of reference sources.
        selected_catalog_name = Calibrate_Catalog._require_catalog_selected(
            input_yaml["catalog"].get("use_catalog")
        )
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
                catalogName=selected_catalog_name,
                catalog_custom_fpath=input_yaml["catalog"].get(
                    "catalog_custom_fpath", None
                ),
            )

        #  Clean Catalog
        # Cleans the catalog by removing sources outside the image borders.
        width = image.shape[1]
        height = image.shape[0]
        border = 11

        # Recalculate catalog pixel coordinates using post-alignment WCS
        # The catalog was downloaded before alignment, so its pixel coordinates
        # are based on the pre-alignment WCS. After SCAMP+SWarp alignment,
        # the image is resampled and the WCS changes, so we must recalculate.
        if "RA" in unCatalogSources.columns and "DEC" in unCatalogSources.columns:
            try:
                coords = SkyCoord(
                    ra=unCatalogSources["RA"].values * u.deg,
                    dec=unCatalogSources["DEC"].values * u.deg,
                    frame="icrs"
                )
                x_pix, y_pix = imageWCS.world_to_pixel(coords)
                unCatalogSources["x_pix"] = np.asarray(x_pix, dtype=float).ravel()
                unCatalogSources["y_pix"] = np.asarray(y_pix, dtype=float).ravel()
                logging.info("Recalculated catalog pixel coordinates using post-alignment WCS")
            except Exception as e:
                logging.warning(f"Failed to recatalog pixel coordinates: {e}")

        CatalogSources = Calibrate_Catalog.clean(
            selectedCatalog=unCatalogSources,
            image_wcs=imageWCS,
            catalogName=selected_catalog_name,
            get_local_sources=False,
            border=border,
        )
        # Deduplicate immediately after cleaning to prevent duplicates from propagating
        if CatalogSources is not None and len(CatalogSources) > 0:
            n_pre = len(CatalogSources)
            CatalogSources = _remove_catalog_duplicates(CatalogSources, method='astropy', sep_threshold=0.1)
            n_post = len(CatalogSources)
            if n_post < n_pre:
                logging.info(f"Removed {n_pre - n_post} duplicates from cleaned catalog")
        if CatalogSources is None or len(CatalogSources) == 0:
            logging.warning(
                "No catalog sources available after cleaning; skipping catalog-based calibration for this image."
            )
            CatalogSources = None
        else:
            if ("x_pix" not in CatalogSources.columns) or (
                "y_pix" not in CatalogSources.columns
            ):
                logging.warning(
                    "Catalog sources missing x_pix/y_pix after cleaning; skipping catalog border filter."
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
                n_before = len(CatalogSources)
                CatalogSources_filtered = CatalogSources[mask]
                n_after = len(CatalogSources_filtered)
                removed_fraction = (n_before - n_after) / n_before if n_before > 0 else 0

                if n_after < n_before:
                    logging.info(
                        f"Border filtering: removed {n_before - n_after} sources outside border, {n_after} sources remaining"
                    )

                # Skip border filtering if it removes more than 50% of sources
                # This can happen when WCS alignment changes the image geometry significantly
                if removed_fraction > 0.5:
                    logging.warning(
                        f"Border filtering removed {removed_fraction*100:.1f}% of sources; skipping border filter to preserve catalog sources"
                    )
                    CatalogSources = CatalogSources  # Keep original unfiltered catalog
                elif n_after == 0:
                    logging.warning(
                        f"All catalog sources removed by border filter (border={border}, image={width}x{height}); skipping border filter"
                    )
                    CatalogSources = CatalogSources  # Keep original unfiltered catalog
                else:
                    CatalogSources = CatalogSources_filtered
                    # De-duplicate catalog entries using helper function
                    n_pre_dedup = len(CatalogSources)
                    CatalogSources = _remove_catalog_duplicates(CatalogSources)
                    n_post_dedup = len(CatalogSources)
                    if n_post_dedup < n_pre_dedup:
                        logging.info(
                            "Catalog de-duplication: dropped %d duplicate entries (%d -> %d).",
                            int(n_pre_dedup - n_post_dedup),
                            int(n_pre_dedup),
                            int(n_post_dedup),
                        )

        # =============================================================================
        # Run source detection on final calibrated image
        # =============================================================================
        logging.info(border_msg("Source detection on calibrated image"))
        # Runs SExtractor to measure the image FWHM and detect sources.

        def _run_sextractor_two_pass(config, fpath, **kwargs):
            """Run SExtractor with optional FWHM refinement pass."""
            fwhm, sources, scale = SExtractorWrapper(config=config).run(fpath, **kwargs)
            if np.isfinite(fwhm):
                fwhm, sources, scale = SExtractorWrapper(config=config).run(
                    fpath, use_FWHM=fwhm, **kwargs
                )
            return fwhm, sources, scale

        _pre_remeasure_fwhm = input_yaml.get("fwhm")  # save before overwrite
        try:
            sex_crowded = input_yaml.get("photometry", {}).get("crowded_field", False)
            ImageFWHM, FWHMSources, scale = _run_sextractor_two_pass(
                config=input_yaml,
                fpath=fpath,
                pixel_scale=pixel_scale,
                masked_sources=variable_sources,
                weight_path=weight_fpath,
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
        # Measured science-frame seeing FWHM (pixels); same quantity as output `image_fwhm`.
        input_yaml["science_fwhm"] = (
            float(ImageFWHM) if np.isfinite(ImageFWHM) else None
        )
        input_yaml["scale"] = scale
        phot_cfg = input_yaml.get("photometry", {}) or {}
        undersampled_thr = float(phot_cfg.get("undersampled_fwhm_threshold", 2.5))
        input_yaml["undersampled_mode"] = bool(
            np.isfinite(ImageFWHM) and float(ImageFWHM) <= undersampled_thr
        )

        # Preserve the initial FWHM if the post-alignment re-measurement is
        # inflated.  The initial "Source detection and FWHM" step (above)
        # runs on the original image with ~28 point sources and produces a
        # reliable FWHM (e.g., 3.99 px).  This post-alignment re-run on the
        # aligned image often finds far fewer sources (9-12) due to resampling
        # and masking changes, and galaxy contamination inflates the FWHM
        # (e.g., 7.85 px).  Using the inflated value causes SFFT to select
        # kernel_order=0 (constant kernel) when the true PSF difference is
        # large, leading to flux scaling mismatches and dipoles.
        if (
            _pre_remeasure_fwhm is not None
            and np.isfinite(_pre_remeasure_fwhm)
            and np.isfinite(ImageFWHM)
            and float(ImageFWHM) > 1.5 * float(_pre_remeasure_fwhm)
        ):
            logging.warning(
                "Post-alignment FWHM %.2f px is inflated (> 1.5 x initial %.2f px); "
                "preserving initial FWHM for subtraction kernel sizing.",
                float(ImageFWHM), float(_pre_remeasure_fwhm),
            )
            ImageFWHM = float(_pre_remeasure_fwhm)
            input_yaml["fwhm"] = ImageFWHM
            input_yaml["science_fwhm"] = ImageFWHM

        # Only write FWHM to header if it's finite (FITS headers reject NaN)
        if np.isfinite(ImageFWHM):
            header["fwhm"] = ImageFWHM

        # Adaptive crowded-field detection (source density + background coverage)
        try:
            ny, nx = image.shape[0], image.shape[1]
            pixel_scale_arcsec = float(
                proj_plane_pixel_scales(imageWCS)[0] * 3600.0
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
                input_yaml["science_fwhm"] = (
                    float(ImageFWHM) if np.isfinite(ImageFWHM) else None
                )
                input_yaml["scale"] = scale
                phot_cfg = input_yaml.get("photometry", {}) or {}
                undersampled_thr = float(
                    phot_cfg.get("undersampled_fwhm_threshold", 2.5)
                )
                input_yaml["undersampled_mode"] = bool(
                    np.isfinite(ImageFWHM) and float(ImageFWHM) <= undersampled_thr
                )
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
            DISTANCE_THRESHOLD_FACTOR = 1
            distance_threshold = DISTANCE_THRESHOLD_FACTOR * ImageFWHM

            # Avoid building a huge list of masked-pixel coordinates (np.argwhere can
            # dominate memory/time on large masks). Use a distance transform instead.
            if not np.any(cosmic_rays_mask):
                logging.info("No cosmic ray pixels found. Skipping source exclusion.")
                excluded_sources = FWHMSources.iloc[
                    []
                ]  # Empty DataFrame with same columns
            else:
                from scipy.ndimage import distance_transform_edt

                # distance to nearest masked pixel for every pixel (float64, but avoids
                # allocating an (N_mask, 2) coordinate array which can be much larger).
                distmap = distance_transform_edt(~cosmic_rays_mask)
                xy = FWHMSources[["x_pix", "y_pix"]].to_numpy(dtype=float)
                xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, distmap.shape[1] - 1)
                yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, distmap.shape[0] - 1)
                min_distances = distmap[yi, xi]
                del distmap

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
            n_fwhm_sources = len(FWHMSources)
            if crowded_field:
                isolation_dist = scale * 0.5
            elif n_fwhm_sources <= 20:
                isolation_dist = scale * 0.5
                logging.info(
                    "Sparse field (%d sources): using relaxed isolation (min_distance=%.1f px) "
                    "to retain more sources.",
                    n_fwhm_sources, isolation_dist,
                )
            else:
                isolation_dist = scale
            if crowded_field:
                logging.info(
                    "Crowded field: using relaxed isolation (min_distance=%.1f px) "
                    "to retain more sources.",
                    isolation_dist,
                )
            _raw_sex_cat = (input_yaml.get("photometry") or {}).get("last_raw_sex_catalog")
            IsolatedSources = Find_FWHM(input_yaml=input_yaml).filter_isolated_sources(
                FWHMSources, min_distance=isolation_dist, all_sources=_raw_sex_cat
            )

        IsolatedSources = Catalog(input_yaml=input_yaml).recenter(
            IsolatedSources, image, boxsize=scale, error=background_rms
        )

        # Converts pixel coordinates of isolated sources to world coordinates.
        # Use origin=0 for consistent 0-based indexing (matching numpy arrays)
        IsolatedSources["RA"], IsolatedSources["DEC"] = imageWCS.all_pix2world(
            IsolatedSources.x_pix.values,
            IsolatedSources.y_pix.values,
            0,
        )

        # Creates SkyCoord objects for sources and target.
        target_skycoord = SkyCoord(
            ra=target_coords.ra,
            dec=target_coords.dec,
            unit="deg",
            frame="icrs",
        )

        source_coords = SkyCoord(
            ra=IsolatedSources["RA"] * u.degree,
            dec=IsolatedSources["DEC"] * u.degree,
            unit="deg",
            frame="icrs",
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

        # =============================================================================
        # PSF Model Building
        # =============================================================================
        logging.info(
            border_msg("Aperture, optimum radius, and PSF on sequence stars")
        )

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
            ap_size=1.7 * ImageFWHM,
            background_rms=background_rms,
            n_jobs=ap_n_jobs,
        )

        IsolatedSources, fit_params, saturation_range = Find_FWHM(
            input_yaml=input_yaml
        ).check_linearity(IsolatedSources)
        IsolatedSources = IsolatedSources.reset_index()

        # Calculates the box size with optimized odd conversion.
        boxsize = odd(min(int(np.ceil(ImageFWHM)) * 2, 5))
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
        # Keep a broader pre-optimum pool for PSF building. Optimum-radius
        # selection is tuned for aperture stability and can become too strict
        # for robust ePSF construction in sparse/crowded epochs.
        psf_source_pool = IsolatedSources.copy()

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
                        n_jobs=input_yaml.get('n_jobs',1),
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

        # If optimum-radius filtering left a very small set, keep using the
        # broader pre-optimum pool for PSF modelling while retaining the
        # filtered set for aperture-radius estimation/corrections.
        min_psf_pool = max(4, int(input_yaml["photometry"].get("psf_min_candidates", 8)))
        if len(IsolatedSources) < min_psf_pool and len(psf_source_pool) > len(IsolatedSources):
            logging.info(
                "Optimum-radius selection returned %d sources; using broader pre-optimum pool "
                "(%d sources) for PSF building.",
                len(IsolatedSources),
                len(psf_source_pool),
            )
        else:
            psf_source_pool = IsolatedSources.copy()

        input_yaml["photometry"]["aperture_radius"] = odd(
            int(np.ceil(optimum_radius * ImageFWHM))
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
                    plot=bool(
                        (input_yaml.get("photometry") or {}).get(
                            "plot_aperture_correction", False
                        )
                    ),
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

            # Transfer per-source FWHM and peak_flux from SExtractor (FWHMSources) to catalog sources.
            # peak_flux (FLUX_MAX in ADU) is needed by zeropoint.clean() for saturation/non-linear rejection.
            if (
                FWHMSources is not None
                and len(FWHMSources) > 0
                and "fwhm" in FWHMSources.columns
                and len(CatalogSources) > 0
                and {"x_pix", "y_pix"}.issubset(CatalogSources.columns)
            ):
                try:
                    from scipy.spatial import cKDTree

                    fwhm_coords = FWHMSources[["x_pix", "y_pix"]].to_numpy(dtype=float)
                    cat_coords = CatalogSources[["x_pix", "y_pix"]].to_numpy(dtype=float)
                    tree = cKDTree(fwhm_coords)
                    distances, idxs = tree.query(cat_coords, k=1)
                    # Only assign if match is within 3 px (generous for catalog vs detection mismatch)
                    match_ok = distances <= 3.0
                    CatalogSources["fwhm"] = np.nan
                    CatalogSources.loc[match_ok, "fwhm"] = FWHMSources.iloc[idxs[match_ok]]["fwhm"].values
                    # Also transfer peak_flux (SExtractor FLUX_MAX in ADU) for saturation checks
                    if "peak_flux" in FWHMSources.columns:
                        CatalogSources["peak_flux"] = np.nan
                        CatalogSources.loc[match_ok, "peak_flux"] = FWHMSources.iloc[idxs[match_ok]]["peak_flux"].values
                    n_matched = int(match_ok.sum())
                    if n_matched > 0:
                        logging.info(
                            f"Matched {n_matched}/{len(CatalogSources)} catalog sources to SExtractor FWHM"
                        )
                except Exception as e:
                    logging.warning(f"Could not match catalog sources to SExtractor FWHM: {e}")
                    CatalogSources["fwhm"] = np.nan
                    if "peak_flux" not in CatalogSources.columns:
                        CatalogSources["peak_flux"] = np.nan
            else:
                CatalogSources["fwhm"] = np.nan
                if "peak_flux" not in CatalogSources.columns:
                    CatalogSources["peak_flux"] = np.nan
        else:
            logging.warning(
                "No catalog sources available for photometric calibration; proceeding without catalog-based calibration."
            )

        # =============================================================================
        # PSF Photometry
        # =============================================================================
        #  Convert Pixel Coordinates to World Coordinates
        # Converts pixel coordinates of isolated sources to world coordinates.
        # Use origin=0 for consistent 0-based indexing (matching numpy arrays)
        ra_IsolatedSources, dec_IsolatedSources = imageWCS.all_pix2world(
            IsolatedSources.x_pix.values,
            IsolatedSources.y_pix.values,
            0,
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
        # By default, build the ePSF from the aligned image to ensure the PSF model
        # matches the data it will be applied to. This avoids PSF shape mismatches when
        # resampling changes pixel scale or introduces distortion.
        # To build from the original (pre-alignment) image instead (to avoid interpolation
        # artifacts in the PSF model itself), set psf_build_from_aligned=False.
        epsf_model = None
        PSFSources = None
        build_from_aligned = bool(
            phot_cfg.get("psf_build_from_aligned", True)
        )
        if (
            not build_from_aligned
            and template_available
            and science_path_original != fpath
            and os.path.exists(science_path_original)
            and (not do_aperture_ONLY or prepare_template)
        ):
            try:
                image_orig, header_orig = get_image_and_header(science_path_original)
                result_orig = bg_remover.remove(
                    image_orig,
                    header=header_orig,
                    plot=False,
                    fwhm=ImageFWHM,
                    galaxies=variable_sources,
                    mask_simbad_galaxies=True,
                )
                # Map the PSF source pool from current image pixels -> sky -> original image pixels.
                # Do not mix IsolatedSources and psf_source_pool here because they can differ
                # in length after optimum-radius filtering.
                wcs_orig = get_wcs(header_orig)
                psf_sources_orig = psf_source_pool.copy()
                finite_xy = (
                    np.isfinite(psf_sources_orig["x_pix"].to_numpy(dtype=float))
                    & np.isfinite(psf_sources_orig["y_pix"].to_numpy(dtype=float))
                )
                psf_sources_orig = psf_sources_orig.loc[finite_xy].copy()
                # Use origin=0 for consistent 0-based indexing (matching numpy arrays)
                ra_pool, dec_pool = imageWCS.all_pix2world(
                    psf_sources_orig["x_pix"].to_numpy(dtype=float),
                    psf_sources_orig["y_pix"].to_numpy(dtype=float),
                    0,
                )
                x_orig, y_orig = wcs_orig.all_world2pix(ra_pool, dec_pool, 0)
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
                    # Release the original full-resolution image immediately after
                    # PSF build — it is no longer needed and can be ~64MB.
                    del image_orig, result_orig
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
            if build_from_aligned:
                logging.info(
                    "Building PSF from aligned image (psf_build_from_aligned=True)."
                )
            try:
                Calibrate_Catalog = Catalog(input_yaml=input_yaml)
                epsf_model, PSFSources = PSF(
                    image=image,
                    input_yaml=input_yaml,
                ).build(
                    psfSources=psf_source_pool,
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
                from astropy.stats import mad_std
                mean_roundness, median_roundness, std_roundness = sigma_clipped_stats(
                    roundness,
                    sigma=3.0,
                    cenfunc=np.nanmedian,
                    stdfunc=mad_std,
                )
                logging.info(
                    f"PSF sources roundness: {mean_roundness:.2f} +/- {std_roundness:.2f}"
                )
            if not epsf_model:
                logging.info("ePSF not created")
                do_aperture_ONLY = True
                PSFSources = None
            else:
                if CatalogSources is None or len(CatalogSources) == 0:
                    logging.warning(
                        "Skipping PSF photometry on catalog sources: no catalog sources available."
                    )
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
                    # Diagnostic plot: WCS catalog position vs PSF fitted position
                    try:
                        makePlots = Plot(input_yaml=input_yaml)
                        makePlots.plot_wcs_vs_psf_offset(CatalogSources, imageWCS=imageWCS)
                    except Exception as exc:
                        logging.warning(f"WCS vs PSF offset plot failed: {exc}")

        # =============================================================================
        # Zeropoint Calculation
        # =============================================================================
        logging.info(border_msg("Zeropoint and color terms"))
        # Calculates the zeropoint for the image.

        Calibrate_Catalog = Catalog(input_yaml=input_yaml)
        GetZeropoint = Zeropoint(input_yaml=input_yaml)
        # Preserve the full measured catalog for writing to the CALIB file.
        # GetZeropoint.clean() applies strict quality cuts that are appropriate
        # for ZP fitting but would silently drop many sources from the output.
        CatalogSources_for_calib = CatalogSources.copy() if CatalogSources is not None else None

        # Remove known variable sources from the calibration catalog before
        # fitting the zeropoint.  Variable stars (Cepheids, RR Lyr, CVs, etc.)
        # have brightness that changes with time, so their measured flux on
        # the science image will not match the reference catalog magnitude,
        # introducing scatter and bias into the ZP fit.
        if (
            CatalogSources is not None
            and len(CatalogSources) > 0
            and variable_sources is not None
            and len(variable_sources) > 0
            and {"x_pix", "y_pix"}.issubset(CatalogSources.columns)
            and {"x_pix", "y_pix"}.issubset(variable_sources.columns)
        ):
            zp_cfg = input_yaml.get("zeropoint", {}) or {}
            var_match_radius = float(zp_cfg.get("variable_source_match_radius_pix", 5.0))
            n_before = len(CatalogSources)
            CatalogSources = cross_match_sources(
                CatalogSources, variable_sources, match_radius_pix=var_match_radius
            )
            n_removed = n_before - len(CatalogSources)
            if n_removed > 0:
                logging.info(
                    "Removed %d catalog sources matching known variable sources "
                    "(match radius=%.1f px); %d sources remain for ZP calibration.",
                    n_removed, var_match_radius, len(CatalogSources),
                )
            else:
                logging.debug(
                    "No catalog sources matched known variable sources (radius=%.1f px).",
                    var_match_radius,
                )

        CatalogSources = GetZeropoint.clean(sources=CatalogSources)

        # Check linearity of catalog sources before fitting zeropoint
        # This filters out non-linear/saturated catalog sources
        if CatalogSources is not None and len(CatalogSources) > 0:
            try:
                CatalogSources, linearity_params, saturation_range = Calibrate_Catalog.check_saturation_range(
                    CatalogSources
                )
                intercept = linearity_params.get('intercept', 0)
                intercept_err = linearity_params.get('intercept_error', 0)
                intercept_str = f"{intercept:.3f}" if np.isfinite(intercept) else "N/A"
                intercept_err_str = f"{intercept_err:.3f}" if np.isfinite(intercept_err) else "N/A"
                logging.info(
                    f"Catalog linearity check: {linearity_params.get('n_inliers', 0)} linear sources, "
                    f"ZP={intercept_str} +/- {intercept_err_str}"
                )
            except Exception as linearity_exc:
                logging.warning(f"Catalog linearity check failed: {linearity_exc}, proceeding with all sources")

        # When using a Gaia custom catalog built from user transmission curves
        # (catalog.transmission_curve_map / custom throughputs), the catalog photometric system
        # already incorporates the effective bandpasses. In that case, a
        # separate empirical color-term correction is usually unnecessary and
        # can even add extra noise when the color coverage is limited.
        use_custom_throughputs = False
        try:
            cat_cfg = input_yaml.get("catalog") or {}
            curve_map_cfg = cat_cfg.get("transmission_curve_map")
            resolved_use_catalog = str(
                Calibrate_Catalog._resolve_catalog_for_filter(
                    cat_cfg.get("use_catalog", None)
                )
                or ""
            ).strip().lower()
            use_custom_throughputs = (
                bool(curve_map_cfg) and resolved_use_catalog == "custom"
            )
        except Exception:
            use_custom_throughputs = False

        # Check if color term correction is enabled
        photo_cfg = input_yaml.get("photometry", {})
        apply_colorterms = bool(photo_cfg.get("apply_colorterms", False))

        if use_custom_throughputs:
            logging.info(
                "Using Gaia custom catalog via catalog.transmission_curve_map (custom throughputs); "
                "disabling zeropoint color correction."
            )
            ImageColorTerm, ImageColorTermError = None, None
            color_coeffs, color_coeff_errors = None, None
            n_segments = 1
        elif not apply_colorterms:
            logging.info(
                "Color term correction disabled (photometry.apply_colorterms=False)."
            )
            ImageColorTerm, ImageColorTermError = None, None
            color_coeffs, color_coeff_errors = None, None
            n_segments = 1
        else:
            logging.info(
                "Color term correction enabled (photometry.apply_colorterms=True); "
                "measuring and applying color term corrections to catalog sources."
            )
            phot_cfg = input_yaml.get("photometry", {}) or {}
            n_segments = int(phot_cfg.get("color_term_n_segments", 1))

            color_coeffs, color_coeff_errors = GetZeropoint.fit_color_term(
                catalog=CatalogSources
            )
            # Treat non-finite coefficients as "no color correction".
            # Handle piecewise linear format: ((breakpoint,), (slope1, slope2), intercept)
            def has_non_finite(coeffs):
                """Check if any coefficient value is non-finite, handling nested tuples."""
                if coeffs is None:
                    return True
                # Flatten nested tuples and check each value
                def flatten(t):
                    if isinstance(t, tuple):
                        for item in t:
                            yield from flatten(item)
                    else:
                        yield t
                return any(not np.isfinite(v) for v in flatten(coeffs))

            if has_non_finite(color_coeffs):
                color_coeffs, color_coeff_errors = None, None
                n_segments = 1  # Fallback to no color term
            else:
                # Extract slope for backwards compatibility with existing code
                if n_segments > 1:
                    # Piecewise linear: slopes are in coefficients[1]
                    ImageColorTerm = color_coeffs[1][0]  # First slope
                    ImageColorTermError = color_coeff_errors[1][0] if color_coeff_errors is not None else None
                else:
                    # fit_color_term can return a scalar or wrong arity when very few stars remain.
                    if not isinstance(color_coeffs, (tuple, list)) or len(color_coeffs) < 2:
                        logging.warning(
                            "Color term fit returned unexpected coefficients %r; disabling color term.",
                            color_coeffs,
                        )
                        color_coeffs, color_coeff_errors = None, None
                        n_segments = 1
                        ImageColorTerm, ImageColorTermError = None, None
                    else:
                        ImageColorTerm = color_coeffs[1]
                        ImageColorTermError = (
                            color_coeff_errors[1] if color_coeff_errors is not None else None
                        )

        fit_mode = "piecewise" if n_segments > 1 else "polynomial"
        CatalogSources, image_zeropoint = GetZeropoint.fit_zeropoint(
            catalog=CatalogSources,
            fixed_color_coeffs=color_coeffs,
            fixed_color_coeff_errors=color_coeff_errors,
            fit_mode=fit_mode,
            n_segments=n_segments,
        )

        # Updates the header with the zeropoint values.
        for m in image_zeropoint.keys():
            try:
                zp_val = image_zeropoint[m].get("zeropoint", np.nan)
                zp_err = image_zeropoint[m].get("zeropoint_error", np.nan)

                # FITS headers cannot store NaN reliably for scalar values.
                if np.isfinite(zp_val):
                    header[f"ZP_{m}"] = float(zp_val)
                else:
                    logging.warning(
                        "%s zeropoint is not finite; writing 'unknown' to header.",
                        m,
                    )
                    header[f"ZP_{m}"] = "unknown"

                if np.isfinite(zp_err):
                    header[f"ZP_{m}_e"] = float(zp_err)
                else:
                    header[f"ZP_{m}_e"] = "unknown"
            except Exception as e:
                log_exception(e, f"Issue with {m} zeropoint")
                header[f"ZP_{m}"] = "unknown"
                header[f"ZP_{m}_e"] = "unknown"

        if len(variable_sources) > 0:
            try:
                xpix_variable_sources, ypix_variable_sources = imageWCS.all_world2pix(
                    variable_sources["RA"].values,
                    variable_sources["DEC"].values,
                    wcs_origin,
                )
                variable_sources["x_pix"] = xpix_variable_sources
                variable_sources["y_pix"] = ypix_variable_sources
            except Exception as e:
                logging.warning(
                    f"WCS coordinate conversion failed for variable sources: {e}. "
                    "Filtering out sources that failed conversion."
                )
                # Try converting sources one by one to identify which ones fail
                valid_indices = []
                xpix_list = []
                ypix_list = []
                for i, (ra, dec) in enumerate(zip(variable_sources["RA"].values, variable_sources["DEC"].values)):
                    try:
                        xpix, ypix = imageWCS.all_world2pix([ra], [dec], wcs_origin)
                        xpix_list.append(xpix[0])
                        ypix_list.append(ypix[0])
                        valid_indices.append(i)
                    except Exception:
                        continue
                if len(valid_indices) > 0:
                    variable_sources = variable_sources.iloc[valid_indices].copy()
                    variable_sources["x_pix"] = xpix_list
                    variable_sources["y_pix"] = ypix_list
                    logging.info(f"Retained {len(valid_indices)}/{len(variable_sources)} variable sources after WCS conversion")
                else:
                    logging.warning("All variable sources failed WCS conversion, using empty list")
                    variable_sources = pd.DataFrame(columns=variable_sources.columns)
        # Plots the source check.
        # Remove mask parameter to avoid masking sources in plots
        Plot(input_yaml=input_yaml).source_check(
            image=image,
            psfSources=PSFSources,
            catalogSources=CatalogSources,
            FWHMSources=FWHMSources,
            variable_sources=variable_sources,
        )
        image_sources = None

        header["aper"] = int(np.ceil(optimum_radius * ImageFWHM))
        # RDNOISE already written to header at line 1143

        # Writes the modified image and header back to the FITS file.
        safe_fits_write(fpath, image, header)

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
                weight_image, weight_header = get_image_and_header(weight_fpath)
                safe_fits_write(
                    os.path.join(cur_dir, newWeightBasename),
                    weight_image,
                    weight_header,
                )
                logging.getLogger(__name__).debug(
                    "Weight map written: %s",
                    os.path.join(cur_dir, newWeightBasename),
                )

            # Saves the cleaned catalog and PSF sources to CSV files.
            # Use the FITS filename stem for consistent naming.
            CatalogSources.to_csv(
                os.path.join(cur_dir, f"imageCalib_template_{input_yaml['base']}.csv"),
                index=False,
                float_format="%.6f",
            )
            IsolatedSources.to_csv(
                os.path.join(cur_dir, f"PSFSources_template_{input_yaml['base']}.csv"),
                index=False,
                float_format="%.6f",
            )

            # Writes the modified image and header to the new FITS file.
            safe_fits_write(os.path.join(cur_dir, newBasename), image, header)
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
            logging.info(border_msg("Difference image (template subtraction)"))
            science_image = get_image(fpath)
            template_image = get_image(templateFpath)
            if science_image.shape != template_image.shape:
                logging.info(
                    log_step(
                        f"Crop: {os.path.basename(fpath)} vs {os.path.basename(templateFpath)}"
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
                    science_image, science_header = get_image_and_header(fpath)
                    science_wcs = get_wcs(science_header)
                    # Gets the image shape.
                    ny, nx = science_image.shape
                    # Use the science image pixel center as the cutout center.
                    # SWarp is configured with CENTER = science pixel center and
                    # IMAGE_SIZE = science shape, so both resampled images are already
                    # registered to this grid and the pixel center is the natural anchor.
                    # Correct numpy 0-based center is (nx-1)/2, (ny-1)/2.
                    science_center_pix = ((nx - 1) / 2.0, (ny - 1) / 2.0)
                    science_center_world = science_wcs.all_pix2world(
                        science_center_pix[0], science_center_pix[1], 0
                    )
                    # Validate WCS transformation result
                    if not (np.isfinite(science_center_world[0]) and np.isfinite(science_center_world[1])):
                        logger.error(
                            f"WCS transformation returned invalid coordinates at pixel center {science_center_pix}. "
                            f"RA={science_center_world[0]}, Dec={science_center_world[1]}. "
                            f"This indicates an invalid WCS header. Check SCAMP alignment results."
                        )
                        raise ValueError(
                            f"Invalid WCS transformation: pixel center {science_center_pix} maps to "
                            f"RA={science_center_world[0]}, Dec={science_center_world[1]}"
                        )
                    
                    # Additional validation: test round-trip conversion
                    try:
                        test_px, test_py = science_wcs.all_world2pix(
                            science_center_world[0], science_center_world[1], 0
                        )
                        if not (np.isfinite(test_px) and np.isfinite(test_py)):
                            logger.error(
                                f"WCS round-trip failed: world2pix returned invalid pixel coordinates. "
                                f"Input RA/Dec={science_center_world}, output px,py={test_px},{test_py}. "
                                f"This indicates a corrupted WCS header from SWarp."
                            )
                            raise ValueError(
                                f"Invalid WCS round-trip: world2pix returned NaN/Inf for valid sky coordinates"
                            )
                    except Exception as e:
                        logger.error(
                            f"WCS round-trip validation failed: {e}. "
                            f"This indicates a corrupted WCS header from SWarp alignment."
                        )
                        raise ValueError(
                            f"WCS validation failed: cannot perform round-trip coordinate conversion. "
                            f"The WCS header from SWarp alignment is likely corrupted."
                        )
                    center_coord = SkyCoord(
                        ra=science_center_world[0],
                        dec=science_center_world[1],
                        unit="deg",
                        frame="icrs",
                    )
                    # Stores the target pixel coordinates.
                    target_x_pix, target_y_pix = update_target_pixel_coords(
                        input_yaml, science_wcs, wcs_origin
                    )
                    # Creates a cutout for the science image.
                    # Use nan_crop to avoid WCS round-trip issues
                    # (CD matrix / SIP distortion dropping).
                    from functions import nan_crop
                    science_image, science_header = nan_crop(
                        science_image, science_header,
                        science_center_pix[0], science_center_pix[1],
                        ny, nx,
                    )
                    # Loads the template image.
                    template_image, template_header = get_image_and_header(templateFpath)
                    template_wcs = get_wcs(template_header)
                    
                    # Validate template WCS before attempting crop
                    try:
                        # Test if template WCS can convert the center coordinate to pixel coordinates
                        test_tx, test_ty = template_wcs.all_world2pix(
                            science_center_world[0], science_center_world[1], 0
                        )
                        if not (np.isfinite(test_tx) and np.isfinite(test_ty)):
                            logger.error(
                                f"Template WCS cannot convert center coordinate to valid pixel coordinates. "
                                f"Input RA/Dec={science_center_world}, output px,py={test_tx},{test_ty}. "
                                f"Template WCS is likely corrupted from SWarp alignment."
                            )
                            raise ValueError(
                                f"Invalid template WCS: world2pix returned NaN/Inf for valid sky coordinates. "
                                f"SWarp alignment likely produced a corrupted WCS header."
                            )
                    except Exception as e:
                        logger.error(
                            f"Template WCS validation failed: {e}. "
                            f"The template WCS from SWarp alignment is likely corrupted."
                        )
                        raise ValueError(
                            f"Template WCS validation failed: cannot convert sky coordinates to pixels. "
                            f"SWarp alignment likely produced a corrupted WCS header."
                        )
                    
                    # Creates a cutout for the template image.
                    # Use nan_crop to preserve WCS distortion keywords.
                    tny, tnx = template_image.shape
                    template_center_pix = (tnx / 2, tny / 2)
                    template_image, template_header = nan_crop(
                        template_image, template_header,
                        template_center_pix[0], template_center_pix[1],
                        ny, nx,
                    )
                    # Saves the results.
                    safe_fits_write(fpath, science_image, science_header)
                    safe_fits_write(templateFpath, template_image, template_header)
                    # Logs the cutout information.
                    logging.info(
                        f"Cutout aligned using sky center: RA={center_coord.ra.deg:.3f}, Dec={center_coord.dec.deg:.3f}"
                    )
                    logging.info(f"Science cutout shape: {science_image.shape}")
                    logging.info(f"Template cutout shape: {template_image.shape}")
                    logging.info(
                        f"Target pixel coordinates (science): x={target_x_pix:.2f} px, y={target_y_pix:.2f} px"
                    )

            # Reloads the image and header.
            image, header = get_image_and_header(fpath)
            imageWCS = get_wcs(header)
            height, width = image.shape

            # Converts the target coordinates to pixel coordinates and stores in input_yaml.
            target_x, target_y = imageWCS.all_world2pix(
                input_yaml["target_ra"],
                input_yaml["target_dec"],
                wcs_origin,
            )
            input_yaml["target_x_pix"] = target_x
            input_yaml["target_y_pix"] = target_y

            # Use high-quality isolated sources from psf_source_pool for matching.
            # These sources were detected on the original image before resampling degradation,
            # then recentered and validated for PSF building. Using them avoids detecting
            # sources on the degraded resampled image (important for undersampled data like ZTF).
            logging.info(
                log_step(
                    f"Match sources: {os.path.basename(fpath)} vs {os.path.basename(templateFpath)}"
                )
            )

            # Use psf_source_pool if available (high-quality isolated sources)
            if psf_source_pool is not None and len(psf_source_pool) > 0:
                detected_sources = psf_source_pool.copy()
                logging.info(
                    f"Using {len(detected_sources)} high-quality isolated sources from PSF pool for subtraction matching "
                    f"(avoiding detection on resampled image for better PSF quality)"
                )
            else:
                # Fallback: detect sources on resampled image if psf_source_pool not available
                _, detected_sources, _ = SExtractorWrapper(config=input_yaml).run(
                    fpath,
                    pixel_scale=pixel_scale,
                    masked_sources=variable_sources,
                    weight_path=weight_fpath,
                    use_FWHM=ImageFWHM,
                    crowded=input_yaml.get("photometry", {}).get("crowded_field", False),
                    use_for_matching=True,
                )
                logging.info(
                    f"Detected {len(detected_sources)} sources on resampled image for subtraction matching "
                    f"(PSF source pool not available)"
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
                    logging.info(
                        f"Matching: {len(MatchingSources)} sources (catalog available but not concatenated to avoid duplication)"
                    )
                else:
                    logging.info(f"Matching: {len(MatchingSources)} sources")
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

            if len(coords) < 2:
                logging.info(
                    "Only %d source(s) for template matching; skipping clustering.",
                    len(coords),
                )
                cluster_labels = np.ones(len(coords), dtype=int)
            else:
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
            centroid_func = (
                centroid_2dg if bool(input_yaml.get("undersampled_mode", False)) else centroid_com
            )
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
                    ra_vals.values, dec_vals.values, wcs_origin
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

            template_image, template_header = get_image_and_header(templateFpath)

            # Build a KDTree for masked pixels for efficient nearest-neighbor search
            # Include NaN/Inf pixels explicitly since NaN == 0 is False
            masked_image = (
                (defects_mask)
                | (image == 0)
                | (template_image == 0)
                | ~np.isfinite(image)
                | ~np.isfinite(template_image)
            )
            masked_pixels = np.argwhere(masked_image)

            if len(masked_pixels) == 0:
                logging.info("Image contains no nan regions ")
                excluded_sources = FWHMSources.iloc[[]]  # Empty DataFrame
            else:
                from scipy.spatial import cKDTree
                tree = cKDTree(masked_pixels)
                source_coords = matched_df[["x_pix", "y_pix"]].values
                # Configurable proximity threshold for masked regions (default: FWHM * 1.5 instead of 3.0)
                # Can be configured via photometry.masked_region_proximity_fwhm_mult in input_yaml
                proximity_fwhm_mult = float(
                    (input_yaml.get("photometry", {}) or {}).get("masked_region_proximity_fwhm_mult", 1.5)
                )
                proximity_threshold = ImageFWHM * proximity_fwhm_mult

                # Adaptive exclusion: if excluding sources near masked regions leaves
                # too few sources, progressively relax the threshold.
                # Use a fixed minimum of 5 (not sfft_min_prior_sources) because:
                # (1) the NaN exclusion is about photometric reliability, not kernel fitting;
                # (2) sfft_min_prior_sources may be set high (e.g. 10) to let SFFT do its
                # own matching, but we still want to exclude sources with corrupted photometry.
                min_sources_needed = 5
                while True:
                    min_distances, _ = tree.query(
                        source_coords, k=1, distance_upper_bound=proximity_threshold
                    )
                    n_kept = int((min_distances > proximity_threshold).sum())
                    if n_kept >= min_sources_needed or proximity_fwhm_mult <= 0.1:
                        break
                    proximity_fwhm_mult /= 2.0
                    proximity_threshold = ImageFWHM * proximity_fwhm_mult
                    logging.info(
                        f"Relaxing masked-region proximity threshold to FWHM x {proximity_fwhm_mult:.2f} "
                        f"({proximity_threshold:.1f} px) — only {n_kept} sources survived at previous threshold."
                    )

                excluded_sources = matched_df[min_distances <= proximity_threshold]
                matched_df = matched_df[min_distances > proximity_threshold]

                if not excluded_sources.empty:
                    logging.info(
                        f"Excluded {len(excluded_sources)} sources due to proximity to a nan/masked region "
                        f"(threshold: {proximity_threshold:.2f} pixels = FWHM x {proximity_fwhm_mult:.1f})."
                    )

            df_zogy_science = None
            df_zogy_template = None
            if len(matched_df) >= 3:
                logging.info(f"Sufficient sources for processing: {len(matched_df)}")
                # Prepares source tables for image and template photometry.
                image_sources = matched_df.copy()
                template_sources = matched_df.copy()

                # Use standard aperture radius instead of large doubled aperture
                # Large apertures cause aperture_sum_invalid in highly masked images
                science_aperture = aperture_radius
                logging.info(
                    f"Using aperture for science image: {science_aperture:.1f} pixels"
                )

                # Photometry on the science image.
                aperture_photometry = Aperture(
                    input_yaml=input_yaml,
                    image=image,
                )
                image_sources = aperture_photometry.measure(
                    sources=image_sources[["x_pix", "y_pix"]],
                    exposure_time=exposure_time,
                    ap_size=science_aperture,
                )
                # Loads the template image and header.
                template_fwhm = template_header.get("FWHM", 3)

                # Use standard aperture radius for template as well
                template_aperture_size = aperture_radius
                logging.info(
                    f"Using aperture for reference image: {template_aperture_size:.1f} pixels"
                )

                # Photometry on the template image (use *template* exposure from its header).
                try:
                    template_exposure, tpl_exp_key = exposure_seconds_from_header(
                        template_header, None
                    )
                except ValueError:
                    template_exposure = float(
                        input_yaml.get("exposure_time", 30.0) or 30.0
                    )
                    tpl_exp_key = "default_input.exposure_time (template)"
                    logging.warning(
                        "Template header lacks EXPTIME; using %.3g s for aperture photometry.",
                        float(template_exposure),
                    )
                logging.info(
                    "Template exposure time: %.5g s (header %s)",
                    float(template_exposure),
                    tpl_exp_key,
                )
                template_gain, tpl_gain_key = gain_e_per_adu_from_header(
                    template_header, None
                )
                logging.info(
                    "Template gain: %.5g e-/ADU (header %s)",
                    float(template_gain),
                    tpl_gain_key,
                )
                template_aperture = Aperture(
                    input_yaml=input_yaml, image=template_image
                )
                template_sources = template_aperture.measure(
                    sources=template_sources[["x_pix", "y_pix"]],
                    exposure_time=float(template_exposure),
                    ap_size=template_aperture_size,
                    gain=float(template_gain),
                    n_jobs=ap_n_jobs,
                )

                #  NEW: Centroid Check for Each Source
                # Define a tolerance for positional alignment (increased from 3 to 5 for sparse fields)
                # Can be configured via photometry.centroid_tolerance in input_yaml
                POSITION_TOLERANCE = float(
                    (input_yaml.get("photometry", {}) or {}).get("centroid_tolerance", 5.0)
                )

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
                dx = template_sources["x_pix"] - template_sources["x_centroid"]
                dy = template_sources["y_pix"] - template_sources["y_centroid"]
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # A systematic alignment offset (e.g. 3px uniform shift from WCS
                # residual) affects ALL sources equally.  Filtering on absolute
                # distance would remove every source even though they are valid —
                # just uniformly shifted.  Instead, measure the median systematic
                # offset and filter only on residual deviations from it.
                if np.any(np.isfinite(distance)) and len(distance) >= 5:
                    med_dx = float(np.nanmedian(dx))
                    med_dy = float(np.nanmedian(dy))
                    systematic_offset = np.sqrt(med_dx ** 2 + med_dy ** 2)
                    # Residual distance after removing systematic component
                    residual_dx = dx - med_dx
                    residual_dy = dy - med_dy
                    residual_distance = np.sqrt(residual_dx ** 2 + residual_dy ** 2)
                else:
                    med_dx = 0.0
                    med_dy = 0.0
                    systematic_offset = 0.0
                    residual_distance = distance.copy()

                # Filter sources where the residual deviation is within tolerance.
                # This preserves sources that share a common systematic offset while
                # removing sources with truly bad centroids (blends, cosmic rays, etc.)
                well_aligned_mask = residual_distance < POSITION_TOLERANCE

                # Log diagnostic statistics before filtering
                if np.any(np.isfinite(distance)):
                    mean_offset = float(np.nanmean(distance))
                    median_offset = float(np.nanmedian(distance))
                    std_offset = float(np.nanstd(distance))
                    logging.info(
                        "Centroid alignment: mean=%.3f, median=%.3f, std=%.3f px, "
                        "systematic offset=(%.3f, %.3f) px (%.3f total), "
                        "residual mean=%.3f px (tolerance=%.1f px)",
                        mean_offset, median_offset, std_offset,
                        med_dx, med_dy, systematic_offset,
                        float(np.nanmean(residual_distance)), POSITION_TOLERANCE
                    )

                # If alignment is poor and we reject everything, adaptively relax
                # the tolerance to keep enough sources for flux-consistent matching.
                # This prevents template subtraction from cascading into failures
                # when the WCS/distortion model is slightly mismatched.
                if well_aligned_mask.sum() < 5 and np.any(np.isfinite(distance)):
                    relaxed = float(max(POSITION_TOLERANCE, 0.75 * float(ImageFWHM)))
                    well_aligned_mask = residual_distance < relaxed
                    logging.info(
                        "Centroid alignment yielded <5 sources; relaxing centroid tolerance to %.2f px",
                        relaxed,
                    )

                # Exclude sources that fall within masked regions (hardware defects)
                # to avoid using contaminated sources for flux matching
                if hardware_defects_mask_cached is not None:
                    x_int = template_sources["x_pix"].astype(int).values
                    y_int = template_sources["y_pix"].astype(int).values
                    # Clip to image bounds
                    x_int = np.clip(x_int, 0, hardware_defects_mask_cached.shape[1] - 1)
                    y_int = np.clip(y_int, 0, hardware_defects_mask_cached.shape[0] - 1)
                    not_masked = ~hardware_defects_mask_cached[y_int, x_int]
                    well_aligned_mask = well_aligned_mask & not_masked
                    n_masked = len(not_masked) - not_masked.sum()
                    if n_masked > 0:
                        logging.info(
                            f"Excluded {n_masked} sources falling within masked regions"
                        )

                # Apply the mask to both image_sources and template_sources
                image_sources = image_sources[well_aligned_mask]
                template_sources = template_sources[well_aligned_mask]

                # Log the number of sources removed or the mean offset
                n_removed = len(well_aligned_mask) - sum(well_aligned_mask)
                if n_removed > 0:
                    logging.info(
                        f"Removed {n_removed} sources due to centroid misalignment "
                        f"(residual tolerance: {POSITION_TOLERANCE} pixels, "
                        f"systematic offset: {systematic_offset:.3f} px)"
                    )
                else:
                    mean_offset = (
                        np.nanmean(residual_distance[well_aligned_mask])
                        if np.any(well_aligned_mask)
                        else 0.0
                    )
                    logging.info(
                        f"Selected matching sources have mean centroid offset of {mean_offset:.3f} pixels"
                    )

                logging.info(
                    f"Well-detected sources in both images: {len(image_sources)}"
                )

                # Cross-match science and template sources for subtraction
                if len(image_sources) > 5:
                    template_obj = Templates(input_yaml=input_yaml)
                    logging.info(f"Flux-consistent matching: input {len(image_sources)} sources")
                    MatchingSources, _ = template_obj.find_flux_consistent_sources(
                        image_sources,
                        template_sources,
                    )
                    logging.info(f"Flux-consistent matching: output {len(MatchingSources)} sources")
                else:
                    MatchingSources = image_sources

                # Converts pixel coordinates back to world coordinates and attaches to the table.
                if not MatchingSources.empty:
                    # Use origin=0 for consistent 0-based indexing (matching numpy arrays)
                    ra_vals, dec_vals = imageWCS.all_pix2world(
                        MatchingSources["x_pix"].values,
                        MatchingSources["y_pix"].values,
                        0,
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
                    n_before_refine = len(ms)
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
                                n_size_rejected = len(ms) - good_size.sum()
                                if n_size_rejected > 0:
                                    logging.info(
                                        f"Size-based rejection: removed {n_size_rejected} sources "
                                        f"(FWHM sigma-clipping, n_sigma={n_sigma})"
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
                            # query_ball_point accepts the full array at once —
                            # avoids a Python loop over every matched source.
                            sci_counts = np.array(
                                [len(nb) - 1 for nb in sci_tree.query_ball_point(ms_xy, crowd_r)]
                            )
                            ref_counts = np.array(
                                [len(nb) - 1 for nb in ref_tree.query_ball_point(ms_xy, crowd_r)]
                            )
                            isolated = (sci_counts <= max_nei) & (ref_counts <= max_nei)
                            n_crowd_rejected = len(ms) - isolated.sum()
                            if n_crowd_rejected > 0:
                                logging.info(
                                    f"Crowding rejection: removed {n_crowd_rejected} sources "
                                    f"(radius={crowd_r:.1f}px, max_nei={max_nei})"
                                )
                            ms = ms[isolated]

                        n_after_refine = len(ms)
                        if n_after_refine < n_before_refine:
                            logging.info(
                                f"Source refinement: {n_before_refine} -> {n_after_refine} sources "
                                f"({n_before_refine - n_after_refine} removed)"
                            )

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

                    sub_method = str(
                        input_yaml["template_subtraction"]["method"]
                    ).lower()
                    stamp_loc = (
                        os.path.join(write_dir, "stamps_positions.txt")
                        if sub_method == "hotpants"
                        else None
                    )
                    # Write coordinates to a stamp positions text file for HOTPANTS.
                    # HOTPANTS expects 1-based FITS-style pixel coordinates, so add +1.
                    if stamp_loc is not None:
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

            # Allow manual override of matching sources via YAML config
            manual_sources = input_yaml["template_subtraction"].get("sfft_manual_matching_sources")
            if manual_sources is not None and len(manual_sources) > 0:
                logging.info(
                    f"Using {len(manual_sources)} manual matching sources from config (sfft_manual_matching_sources)"
                )
                ConsistentSources = [[float(x), float(y)] for x, y in manual_sources]
                # Create MatchingSources DataFrame for compatibility
                if MatchingSources is None or len(MatchingSources) == 0:
                    MatchingSources = pd.DataFrame(ConsistentSources, columns=["x_pix", "y_pix"])

            #  Variable Sources
            # Handles variable sources if present.
            # stamp_loc = None
            if len(variable_sources) > 0:
                xpix_variable_sources, ypix_variable_sources = imageWCS.all_world2pix(
                    variable_sources["RA"].values,
                    variable_sources["DEC"].values,
                    wcs_origin,
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

                # Build 1-based (x, y) pairs for SFFT's XY_PriorBan without
                # mutating variable_sources: those coords stay 0-based for all
                # SExtractor masking calls that come before AND after this point.
                masked_sources = (
                    variable_sources[["x_pix", "y_pix"]].values + 1.0
                ).tolist()
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
                template_image, template_header = get_image_and_header(templateFpath)
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
                        log_warning_from_exception(
                            logging.getLogger(),
                            "Independent template PSF star selection failed; using matched stars",
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
                    write_dir, f"SFFT_Matching_Sources_{input_yaml['base']}.csv"
                )
                sfft_matched_sources_legacy = os.path.join(
                    write_dir, "sfft_matching_sources.csv"
                )
                if not os.path.exists(sfft_matched_sources) and os.path.exists(
                    sfft_matched_sources_legacy
                ):
                    sfft_matched_sources = sfft_matched_sources_legacy
                fpath_nosub = fpath

                if os.path.exists(sfft_matched_sources):
                    os.remove(sfft_matched_sources)

                # Compute combined pipeline scale (source-detection cutout size, typically
                # 5*FWHM).  This is passed to subtract() for HOTPANTS kernel sizing.
                # SFFT computes its own kernel half-width from FWHM inside subtract().
                from utils.run_sex import scale_multiplier_from_config, clamp_scale_from_config
                sci_scale = input_yaml.get("scale")
                template_fwhm = template_header.get("FWHM", 3.0)
                scale_mult = scale_multiplier_from_config(input_yaml)
                template_scale = int(clamp_scale_from_config(input_yaml, scale_mult * template_fwhm))
                combined_scale = max(sci_scale, template_scale) if sci_scale is not None else template_scale
                logging.info(
                    "Subtraction pipeline scale (HOTPANTS): science=%d template=%d -> combined=%d px",
                    sci_scale if sci_scale is not None else -1,
                    template_scale,
                    combined_scale,
                )

                fpath, subtraction_mask, masked_centers, kernel_half_width = Templates(input_yaml=input_yaml).subtract(
                    scienceFpath=fpath,
                    templateFpath=templateFpath,
                    method=input_yaml["template_subtraction"]["method"],
                    matching_sources=ConsistentSources,
                    masked_sources=masked_sources,
                    stamp_loc=stamp_loc,
                    scienceNoise=weight_fpath,
                    templateNoise=template_weight_path,
                    background_defects_mask=hardware_defects_mask,
                    scale=combined_scale,
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
                    if {
                        "X_IMAGE_REF_SCI_MEAN",
                        "Y_IMAGE_REF_SCI_MEAN",
                    }.issubset(MatchingSources.columns):
                        # X/Y_IMAGE_REF_SCI_MEAN come from SExtractor (1-based);
                        # convert to 0-based to match the pipeline convention.
                        MatchingSources["x_pix"] = MatchingSources["X_IMAGE_REF_SCI_MEAN"] - 1
                        MatchingSources["y_pix"] = MatchingSources["Y_IMAGE_REF_SCI_MEAN"] - 1
                    elif {"x_center", "y_center"}.issubset(MatchingSources.columns):
                        MatchingSources["x_pix"] = MatchingSources["x_center"]
                        MatchingSources["y_pix"] = MatchingSources["y_center"]
                    elif {"x_pix", "y_pix"}.issubset(MatchingSources.columns):
                        pass
                    else:
                        logging.warning(
                            "SFFT matched-sources file missing expected coordinate columns; available=%s",
                            list(MatchingSources.columns),
                        )
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

            # Difference image background is already zeroed (sigma-clipped median,
            # with target exclusion) inside templates.subtract() before writing the
            # FITS file.  Do not re-subtract here.

        # Gets the header of the image.
        header = get_header(fpath)

        # -----------------------------------------------------------------------
        # VSCALE error propagation (LSST-style variance rescaling)
        #
        # run_sfft.py may have rescaled the difference image by a factor of
        # 1/VSCALE (IQR-based noise calibration).  The background_rms array
        # that was computed from the science image is therefore stale: it
        # reflects the pre-subtraction noise level rather than the noise of the
        # written difference image.
        #
        # Applying the same scale factor to background_rms here ensures that all
        # downstream error models (aperture, PSF, limiting magnitude injection)
        # see a noise floor consistent with the difference image pixels.
        #
        # Note: if remove_local_surface runs below, it recomputes background_rms
        # from the diff image directly, which is already consistent with the
        # rescaled pixels — so the correction below is safe in both cases (it is
        # overwritten by the recomputed value when remove_local_surface is active).
        # -----------------------------------------------------------------------
        if PreformSubtraction:
            _vscale_header = float(header.get("VSCALE", 1.0))
            if _vscale_header != 1.0 and np.isfinite(_vscale_header) and _vscale_header > 0:
                if background_rms is not None:
                    background_rms = background_rms * _vscale_header
                if abs(_vscale_header - 1.0) > 0.1:
                    logging.warning(
                        "VSCALE=%.4f in difference image header: background_rms rescaled "
                        "by the same factor to keep error model consistent with image pixels. "
                        "A VSCALE far from 1.0 may indicate noise model assumptions need review.",
                        _vscale_header,
                    )
                else:
                    logging.info(
                        "VSCALE=%.4f applied to background_rms for consistent error model on difference image.",
                        _vscale_header,
                    )

        # Gets the WCS information from the header.
        imageWCS = get_wcs(header)

        # Converts the target coordinates to pixel coordinates.
        target_x_pix, target_y_pix = imageWCS.all_world2pix(
            input_yaml["target_ra"],
            input_yaml["target_dec"],
            wcs_origin,
        )

        # Prevent downstream local background failures when WCS maps the
        # target outside the trimmed image (this can happen when a plate
        # solution WCS is used but its projection/distortion model disagrees
        # with the pipeline's internal alignment expectations).
        # For local background fitting only, clamp x/y into image bounds so
        # `remove_local_surface()` does not hard-fail when the plate solution
        # maps the expected transient position outside the trimmed frame.
        # Keep `target_x_pix/y_pix` unchanged for the actual transient centroid
        # fitting (so we do not silently "move" the transient).
        ny, nx = image.shape[0], image.shape[1]
        bg_target_x_pix = float(target_x_pix) if np.isfinite(target_x_pix) else float(nx) / 2.0
        bg_target_y_pix = float(target_y_pix) if np.isfinite(target_y_pix) else float(ny) / 2.0
        if not (0 <= bg_target_x_pix < nx and 0 <= bg_target_y_pix < ny):
            margin = int(max(1, np.ceil(1.0 * float(ImageFWHM))))
            xlo = min(max(0, margin), nx - 1)
            ylo = min(max(0, margin), ny - 1)
            xhi = max(0, nx - 1 - margin)
            yhi = max(0, ny - 1 - margin)
            bg_target_x_pix = float(np.clip(bg_target_x_pix, xlo, xhi))
            bg_target_y_pix = float(np.clip(bg_target_y_pix, ylo, yhi))
            logging.warning(
                "Target pixel from WCS (%.1f, %.1f) outside image (%dx%d); clamping to (%.1f, %.1f) only for local background fit.",
                float(target_x_pix) if np.isfinite(target_x_pix) else float("nan"),
                float(target_y_pix) if np.isfinite(target_y_pix) else float("nan"),
                int(nx),
                int(ny),
                bg_target_x_pix,
                bg_target_y_pix,
            )

        # Updates the input YAML with the target pixel coordinates.
        input_yaml["target_ra"] = target_coords.ra.degree
        input_yaml["target_dec"] = target_coords.dec.degree
        input_yaml["target_x_pix"] = target_x_pix
        input_yaml["target_y_pix"] = target_y_pix

        # For subtraction diagnostics: keep a copy of the pre-local-background
        # image so the difference panel reflects the raw subtraction output.
        # Local background subtraction is a photometry aid and can visually
        # suppress real residual structure in the difference image.
        diff_image_for_plot = image if not PreformSubtraction else np.array(image, copy=True)

        # Optional: LPI-style local background infill under the transient only
        # (structured backgrounds). This is a small-stamp, target-only correction
        # inspired by Saydjari & Finkbeiner 2022 (ApJ 933:155).
        phot_cfg = input_yaml.get("photometry") or {}
        lpi_extra_flux_err = np.nan
        if bool(phot_cfg.get("lpi_background_for_target", False)):
            try:
                from lpi_background import (
                    predict_background_under_source,
                    save_lpi_diagnostic_plot,
                )

                fwhm_px = float(ImageFWHM)
                inner_r = float(phot_cfg.get("lpi_inner_radius_scale_fwhm", 1.5)) * fwhm_px
                outer_r = float(phot_cfg.get("lpi_outer_radius_scale_fwhm", 4.5)) * fwhm_px
                half = int(
                    np.ceil(float(phot_cfg.get("lpi_stamp_half_size_scale_fwhm", 6.0)) * fwhm_px)
                )
                raw_min_shift = phot_cfg.get("lpi_min_shift_px", None)
                min_shift_px = float(raw_min_shift) if raw_min_shift is not None else float(outer_r)
                # Snapshot the local stamp before applying LPI (for safety checks).
                stamp_before = np.array(
                    image[
                        max(0, int(np.rint(bg_target_y_pix)) - half) : min(
                            image.shape[0], int(np.rint(bg_target_y_pix)) + half + 1
                        ),
                        max(0, int(np.rint(bg_target_x_pix)) - half) : min(
                            image.shape[1], int(np.rint(bg_target_x_pix)) + half + 1
                        ),
                    ],
                    dtype=float,
                    copy=True,
                )
                bg_pred, bg_sig = predict_background_under_source(
                    image,
                    x0=float(bg_target_x_pix),
                    y0=float(bg_target_y_pix),
                    inner_radius_px=inner_r,
                    outer_radius_px=outer_r,
                    stamp_half_size_px=half,
                    n_samples=int(phot_cfg.get("lpi_n_samples", 250)),
                    sample_window_px=int(phot_cfg.get("lpi_sample_window_px", 30)),
                    min_shift_px=min_shift_px,
                    ridge_lambda=float(phot_cfg.get("lpi_ridge_lambda", 0.01)),
                    rng_seed=input_yaml.get("rng_seed", None),
                )
                # Apply only inside the hidden region (bg_pred is 0 elsewhere).
                y0i = int(np.rint(float(bg_target_y_pix)))
                x0i = int(np.rint(float(bg_target_x_pix)))
                y1 = max(0, y0i - half)
                y2 = min(image.shape[0], y0i + half + 1)
                x1 = max(0, x0i - half)
                x2 = min(image.shape[1], x0i + half + 1)
                sy1 = half - (y0i - y1)
                sx1 = half - (x0i - x1)
                sy2 = sy1 + (y2 - y1)
                sx2 = sx1 + (x2 - x1)
                image[y1:y2, x1:x2] = image[y1:y2, x1:x2] - bg_pred[sy1:sy2, sx1:sx2]

                # Safety: if the LPI correction would remove essentially all signal in the PSF core,
                # skip it (this indicates the regression learned the source, not the background).
                try:
                    stamp_after = np.array(image[y1:y2, x1:x2], dtype=np.float32, copy=False)
                except ValueError:
                    # NumPy 2.0+ raises ValueError if copy cannot be avoided
                    stamp_after = np.asarray(image[y1:y2, x1:x2], dtype=np.float32)
                yy0, xx0 = np.mgrid[0 : bg_pred.shape[0], 0 : bg_pred.shape[1]]
                rr0 = np.hypot(xx0 - half, yy0 - half)
                core = rr0 <= float(inner_r)
                core_before = float(np.nansum(stamp_before[core[sy1:sy2, sx1:sx2]]))
                core_after = float(np.nansum(stamp_after[core[sy1:sy2, sx1:sx2]]))
                if np.isfinite(core_before) and np.isfinite(core_after):
                    # If the pre-LPI core sum is <= 0 (possible in noisy/oversubtracted
                    # difference images), the ratio test is not meaningful. In that case,
                    # conservatively skip LPI and keep the original image.
                    if core_before <= 0:
                        image[y1:y2, x1:x2] = image[y1:y2, x1:x2] + bg_pred[sy1:sy2, sx1:sx2]
                        raise RuntimeError(
                            "LPI core-flux safety triggered (core_before<=0; ratio undefined); "
                            "skipping LPI to preserve transient flux."
                        )
                    frac = core_after / core_before
                    if frac < 0.2:
                        # Revert subtraction in the stamp region.
                        image[y1:y2, x1:x2] = image[y1:y2, x1:x2] + bg_pred[sy1:sy2, sx1:sx2]
                        raise RuntimeError(
                            f"LPI core-flux safety triggered (core_after/core_before={frac:.3f}); "
                            "skipping LPI to preserve transient flux."
                        )

                # Mark that LPI has already been applied to the target region in this image.
                # This prevents double-application in PSF.fit (which would suppress real flux).
                input_yaml["_lpi_target_applied_to_image"] = True

                # Optional diagnostic plot (no "saved plot" log line).
                try:
                    if bool(phot_cfg.get("lpi_save_diagnostic_plot", True)):
                        base0 = os.path.splitext(os.path.basename(fpath))[0]
                        write_dir0 = os.path.dirname(fpath)
                        save_png = os.path.join(write_dir0, f"LPI_Target_{base0}.png")
                        save_lpi_diagnostic_plot(
                            image=diff_image_for_plot if PreformSubtraction else image,
                            x0=float(bg_target_x_pix),
                            y0=float(bg_target_y_pix),
                            stamp_half_size_px=int(half),
                            inner_radius_px=float(inner_r),
                            outer_radius_px=float(outer_r),
                            bg_pred=np.asarray(bg_pred, float),
                            bg_sig=np.asarray(bg_sig, float),
                            save_path=str(save_png),
                            title=f"LPI target background infill: {base0}",
                        )
                except Exception:
                    pass

                # Convert per-pixel predicted background scatter (in image units)
                # into an additional flux uncertainty term inside the transient
                # aperture. This is a conservative proxy for correlated-background
                # uncertainty (Saydjari & Finkbeiner 2022).
                try:
                    phot_ap_rad = (input_yaml.get("photometry") or {}).get(
                        "aperture_radius", None
                    )
                    if phot_ap_rad is None:
                        phot_ap_rad = float(
                            (input_yaml.get("photometry") or {}).get("aperture_size", 1.7)
                        ) * float(ImageFWHM)
                    ap_rad = float(phot_ap_rad)
                    yy, xx = np.mgrid[0 : bg_sig.shape[0], 0 : bg_sig.shape[1]]
                    rr = np.hypot(xx - half, yy - half)
                    ap_mask = rr <= ap_rad
                    sig_ap = np.asarray(bg_sig, float)[ap_mask]
                    sig_ap = sig_ap[np.isfinite(sig_ap)]
                    if sig_ap.size > 0:
                        extra_counts_err = float(np.sqrt(np.sum(sig_ap**2)))
                        exptime = float(input_yaml["exposure_time"])
                        if np.isfinite(exptime) and exptime > 0:
                            lpi_extra_flux_err = extra_counts_err / exptime
                except Exception:
                    lpi_extra_flux_err = np.nan

                logging.info(
                    "Target LPI background infill applied (inner=%.1f px, outer=%.1f px, samples=%d).",
                    float(inner_r),
                    float(outer_r),
                    int(phot_cfg.get("lpi_n_samples", 250)),
                )
            except Exception as e:
                log_warning_from_exception(
                    logging.getLogger(__name__),
                    "Target LPI background infill failed; continuing without it",
                    e,
                )

        # Optional uniform DC lift in the local subtract box (see `background:` YAML).
        # We also reuse the *same* local-fit cutout for target AP, target PSF,
        # and injected limiting magnitude so these measurements are consistent.
        local_cutout_nonneg_lift = 0.0
        local_cutout_box = None
        if input_yaml["photometry"].get("remove_local_surface", 3):
            # Ensure the local-cutout DC lift is enabled (shifted-mean mode).
            try:
                if "background" not in input_yaml or input_yaml["background"] is None:
                    input_yaml["background"] = {}
                input_yaml["background"]["local_nonnegative_target_offset"] = True
            except Exception:
                pass
            bg_remover = BackgroundSubtractor(input_yaml)
            # Use a slightly larger exclusion radius around the target so that
            # extended host light is not pulled into the local background model.
            image, bkg_map, background_rms, nn_meta = bg_remover.remove_local_surface(
                image,
                x0=bg_target_x_pix,
                y0=bg_target_y_pix,
                box_half_size=int(25 * ImageFWHM),
                fwhm_pixels=ImageFWHM,
                exclude_inner_radius=None,
            )
            try:
                local_cutout_nonneg_lift = float(nn_meta.get("lift", 0.0) or 0.0)
                local_cutout_box = nn_meta.get("box")
            except Exception:
                local_cutout_nonneg_lift = 0.0
                local_cutout_box = None
        # remove_local_surface signature:
        # image, x0, y0, box_half_size=100, fwhm_pixels=None,
        # exclude_inner_radius=8, dilate_factor=2.0

        # -------------------------------------------------------------------------
        # Build a shared target cutout for AP / PSF / limiting magnitude
        # -------------------------------------------------------------------------
        target_cutout = None
        target_cutout_rms = None
        cutout_x0 = 0
        cutout_y0 = 0
        cutout_target_x = float(bg_target_x_pix)
        cutout_target_y = float(bg_target_y_pix)

        # Create Limits instance for get_cutout method
        from limits import Limits
        getDetectionLimits = Limits(input_yaml=input_yaml, catalog=CatalogSources)

        try:
            # Use get_cutout to create target cutout (ensures consistency with limiting magnitude test)
            # Pass the background-subtracted image
            # get_cutout returns (data, cx, cy) tuple
            cutout_result = getDetectionLimits.get_cutout(image=image)
            if cutout_result is not None:
                target_cutout, cutout_cx, cutout_cy = cutout_result
                # Calculate cutout origin in full-image coordinates
                # cutout_cx/cy are the target position in cutout-local coordinates
                # We need cutout_x0/y0 to convert fitted positions back to full-image
                cutout_x0 = float(bg_target_x_pix) - float(cutout_cx)
                cutout_y0 = float(bg_target_y_pix) - float(cutout_cy)
                # Extract corresponding background RMS region
                if background_rms is not None and np.ndim(background_rms) == 2:
                    # Use get_cutout on background_rms as well
                    rms_result = getDetectionLimits.get_cutout(image=background_rms)
                    if rms_result is not None:
                        target_cutout_rms, _, _ = rms_result
                        target_cutout_rms = np.abs(np.asarray(target_cutout_rms, dtype=float))
                # Target position in cutout-local coordinates, as returned by
                # Cutout2D.position_cutout.  Do NOT assume the geometric centre
                # of the array: near image edges the cutout is partial and the
                # target sits offset from shape/2.
                cutout_target_x = float(cutout_cx)
                cutout_target_y = float(cutout_cy)
        except Exception as e:
            logging.warning(f"get_cutout failed for target cutout: {e}, falling back to direct slicing")
            target_cutout = None
            target_cutout_rms = None

        # Fallback to direct slicing if get_cutout failed
        if target_cutout is None and local_cutout_box is not None:
            try:
                y0b, y1b, x0b, x1b = [int(v) for v in local_cutout_box]
                cutout_y0, cutout_x0 = int(y0b), int(x0b)
                target_cutout = np.asarray(image[y0b:y1b, x0b:x1b], dtype=np.float32)
                # Background RMS cutout (if available) so error models match.
                if background_rms is not None and np.ndim(background_rms) == 2:
                    target_cutout_rms = np.asarray(
                        background_rms[y0b:y1b, x0b:x1b], dtype=np.float32
                    )
                cutout_target_x = float(bg_target_x_pix) - float(x0b)
                cutout_target_y = float(bg_target_y_pix) - float(y0b)
            except Exception:
                target_cutout = None
                target_cutout_rms = None

        # If a uniform DC lift was applied to make the cutout background nonnegative,
        # keep it in place for measurements so the *mean level* is shifted positive.
        # Flux remains unbiased because both AP and PSF subtract a local background
        # estimated from an annulus (the constant cancels), but Poisson terms avoid
        # pathological negative-count regimes in some noise models.
        if (
            target_cutout is not None
            and np.isfinite(local_cutout_nonneg_lift)
            and float(local_cutout_nonneg_lift) != 0.0
        ):
            logging.info(
                "Target cutout: keeping uniform bias level %.6g in measurement image (shifted mean enabled).",
                float(local_cutout_nonneg_lift),
            )

        # Fallback: if local cutout wasn't built, use full image as before.
        image_for_target = target_cutout if target_cutout is not None else image
        background_rms_for_target = (
            target_cutout_rms if target_cutout_rms is not None else background_rms
        )

        # =============================================================================
        # Targeted Photometry
        # =============================================================================

        #  Log Start of Targeted Photometry
        # Logs the start of targeted photometry.
        logging.info(
            border_msg(f"Target photometry: {input_yaml['target_name']}")
        )

        # Sets the detection limit used for target detection decisions elsewhere
        # (this is distinct from limiting-magnitude recovery gating below).
        detection_limit = input_yaml["photometry"].get("detection_limit", 3)

        # Prepares initial target coordinates.
        # Run target AP/PSF on the shared cutout when available.
        # We'll shift fitted positions back to full-image pixels afterwards.
        TargetPosition = pd.DataFrame(
            {
                "x_pix": [cutout_target_x if target_cutout is not None else target_x_pix],
                "y_pix": [cutout_target_y if target_cutout is not None else target_y_pix],
            }
        )
        logging.info(
            f"Transient's expected location: x = {target_x_pix:.3f} pixels, y = {target_y_pix:.3f} pixels"
        )

        # Refines the centroid with COM inside ~1xFWHM box (odd box size).
        boxsize = int(np.ceil(ImageFWHM))
        if boxsize % 2 == 0:
            boxsize += 1

        # Performs aperture photometry at the refined position.
        AperturePhotometry = Aperture(
            input_yaml=input_yaml,
            image=image_for_target,
        )
        # IMPORTANT: on difference images (and after our local-surface correction/bias),
        # the local sky in the annulus can legitimately be negative. The default
        # aperture path floors negative annulus medians to 0 (to protect Poisson
        # noise models), which will overestimate flux and can disagree with PSF.
        # For the target cutout photometry we want to subtract the annulus median
        # as measured (even if negative) to stay consistent with the local model.
        _old_enforce_nn = None
        try:
            _old_enforce_nn = bool(
                (input_yaml.get("photometry") or {}).get(
                    "enforce_nonnegative_local_background", True
                )
            )
            if "photometry" not in input_yaml or input_yaml["photometry"] is None:
                input_yaml["photometry"] = {}
            input_yaml["photometry"]["enforce_nonnegative_local_background"] = False
            logging.info(
                "Target AP: subtracting annulus median as measured (allowing negative local background)."
            )
            TargetPosition = AperturePhotometry.measure(
                sources=TargetPosition,
                plot=True,
                saveTarget=True,
                background_rms=background_rms_for_target,
                n_jobs=input_yaml.get("n_jobs", 1),
            )
        finally:
            if _old_enforce_nn is not None:
                try:
                    input_yaml["photometry"]["enforce_nonnegative_local_background"] = bool(
                        _old_enforce_nn
                    )
                except Exception:
                    pass
        if np.isfinite(lpi_extra_flux_err) and "flux_AP_err" in TargetPosition.columns:
            try:
                old = float(TargetPosition["flux_AP_err"].iloc[0])
                TargetPosition.loc[TargetPosition.index[0], "flux_AP_err"] = float(
                    np.sqrt(old**2 + float(lpi_extra_flux_err) ** 2)
                )
                # Keep SNR consistent with updated uncertainty if possible.
                if "flux_AP" in TargetPosition.columns and "SNR" in TargetPosition.columns:
                    f = float(TargetPosition["flux_AP"].iloc[0])
                    ferr = float(TargetPosition["flux_AP_err"].iloc[0])
                    if np.isfinite(f) and np.isfinite(ferr) and ferr > 0:
                        TargetPosition.loc[TargetPosition.index[0], "SNR"] = f / ferr
            except Exception:
                pass
        prelim_threshold = TargetPosition["threshold"].iloc[0]
        perform_ForcePhotometry = False

        # Sets up the target position for PSF fitting.
        TargetPosition["x_fit"] = [np.nan]
        TargetPosition["y_fit"] = [np.nan]
        TargetPosition["x_fit_err"] = [np.nan]
        TargetPosition["y_fit_err"] = [np.nan]

        # Use the global photometry fitting bound configured in arcseconds.
        # Conversion to pixels is handled in PSF.fit; here we log the expected
        # value when pixel_scale is available.
        fit_bound_arcsec = float(
            (input_yaml.get("photometry", {}) or {}).get("fitting_xy_bounds", 1.0)
        )
        pix_scale = input_yaml.get("pixel_scale", np.nan)
        if np.isfinite(pix_scale) and float(pix_scale) > 0:
            fit_bound_pix = fit_bound_arcsec / float(pix_scale)
            logging.info(
                "Target PSF fitting bound: %.3f arcsec (%.2f px at %.3f arcsec/px)",
                fit_bound_arcsec,
                fit_bound_pix,
                float(pix_scale),
            )
        else:
            logging.info(
                "Target PSF fitting bound: %.3f arcsec (pixel_scale unavailable; "
                "PSF.fit will use robust fallback conversion).",
                fit_bound_arcsec,
            )

        # Stage 1: Create inverted image for negative PSF detection if enabled
        inverted_image = None
        phot_cfg = input_yaml.get("photometry", {}) or {}
        check_inverted = phot_cfg.get("check_inverted_image", False)
        if check_inverted:
            try:
                # Use the local background measured from aperture annulus
                # This is more accurate than global median for the target region
                if "local_bkg_raw" in TargetPosition.columns and np.isfinite(TargetPosition["local_bkg_raw"].iloc[0]):
                    bkg_median = float(TargetPosition["local_bkg_raw"].iloc[0])
                    logging.info(f"Using aperture annulus background for inversion: {bkg_median:.3f}")
                elif "local_bkg_used" in TargetPosition.columns and np.isfinite(TargetPosition["local_bkg_used"].iloc[0]):
                    bkg_median = float(TargetPosition["local_bkg_used"].iloc[0])
                    logging.info(f"Using aperture annulus background (used) for inversion: {bkg_median:.3f}")
                else:
                    # Fallback to global median if local background not available
                    if target_cutout is not None:
                        image_data = np.array(target_cutout, dtype=float, copy=True)
                    else:
                        image_data = np.array(image, dtype=float, copy=True)
                    bkg_median = float(np.nanmedian(image_data))
                    logging.info(f"Using global median background for inversion: {bkg_median:.3f}")
                
                # Get image data for inversion - convert to electrons like psf.py does
                gain = resolve_gain_e_per_adu(None, input_yaml)
                if target_cutout is not None:
                    image_data = np.array(target_cutout, dtype=float, copy=True) * gain
                else:
                    image_data = np.array(image, dtype=float, copy=True) * gain

                # Double background subtraction: subtract once to zero, again to negative, then flip
                # Formula: -(image_data - 2*bkg_e) = 2*bkg_e - image_data
                # Background stays at bkg_e level, negative dips become positive peaks
                bkg_e = bkg_median * gain
                bkg_sub = image_data - bkg_e
                bkg_sub2 = bkg_sub - bkg_e
                inv_data = -bkg_sub2  # No clipping, keep full range
                inverted_image = inv_data
                logging.info("Created inverted image for negative PSF detection (-(data - 2*bkg), no clip).")
            except Exception as exc:
                logging.warning(f"Failed to create inverted image: {exc}")
                inverted_image = None

        # Performs PSF fitting on the target position if aperture photometry is not required.
        if not do_aperture_ONLY:
            TargetPosition = PSF(
                image=image_for_target,
                input_yaml=input_yaml,
            ).fit(
                epsf_model=epsf_model,
                sources=TargetPosition,
                plotTarget=True,
                forcePhotometry=perform_ForcePhotometry,
                is_target_fit=True,
                background_rms=background_rms_for_target,
                inverted_image=inverted_image,
            )

        
        # If we used a cutout for the target fit, shift results back to full-image pixels
        # so downstream logging/output stays consistent.
        if target_cutout is not None:
            try:
                for col in ("x_pix", "y_pix", "x_fit", "y_fit", "x_fit_normal", "y_fit_normal"):
                    if col in TargetPosition.columns and np.isfinite(TargetPosition[col].iloc[0]):
                        old_val = float(TargetPosition[col].iloc[0])
                        if col.startswith("x"):
                            new_val = old_val + float(cutout_x0)
                            TargetPosition.loc[TargetPosition.index[0], col] = new_val
                        else:
                            new_val = old_val + float(cutout_y0)
                            TargetPosition.loc[TargetPosition.index[0], col] = new_val
            except Exception as e:
                logging.warning(f"Failed to convert cutout coordinates: {e}")

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
        # Check if inverted PSF fit was used
        inverted_tag = ""
        if "_inverted_fit" in TargetPosition.columns and TargetPosition["_inverted_fit"].iloc[0]:
            inverted_tag = " [inverted]"

        # Stage 2: If inverted fit was used, run aperture photometry on inverted image
        if "_inverted_fit" in TargetPosition.columns and TargetPosition["_inverted_fit"].iloc[0]:
            if inverted_image is not None:
                try:
                    logging.info("Running aperture photometry on inverted image for inverted detection.")
                    # Create aperture photometry instance for inverted image
                    AperturePhotometryInverted = Aperture(
                        input_yaml=input_yaml,
                        image=inverted_image,
                    )
                    # Measure on inverted image (no copy needed — .measure() returns a new DataFrame)
                    TargetPositionInverted = AperturePhotometryInverted.measure(
                        sources=TargetPositionInverted,
                        plot=True,
                        saveTarget=True,
                        background_rms=background_rms_for_target,
                        n_jobs=input_yaml.get("n_jobs", 1),
                    )
                    # Store inverted aperture results with _inverted suffix
                    if "flux_AP" in TargetPositionInverted.columns:
                        TargetPosition["flux_AP_inverted"] = TargetPositionInverted["flux_AP"]
                    if "flux_AP_err" in TargetPositionInverted.columns:
                        TargetPosition["flux_AP_err_inverted"] = TargetPositionInverted["flux_AP_err"]
                    if "SNR" in TargetPositionInverted.columns:
                        TargetPosition["SNR_AP_inverted"] = TargetPositionInverted["SNR"]
                    if "local_bkg_raw" in TargetPositionInverted.columns:
                        TargetPosition["local_bkg_raw_inverted"] = TargetPositionInverted["local_bkg_raw"]
                    if "local_bkg_used" in TargetPositionInverted.columns:
                        TargetPosition["local_bkg_used_inverted"] = TargetPositionInverted["local_bkg_used"]
                    if "sky_bkg_total" in TargetPositionInverted.columns:
                        TargetPosition["sky_bkg_total_inverted"] = TargetPositionInverted["sky_bkg_total"]
                    if "sky_bkg_total_flux" in TargetPositionInverted.columns:
                        TargetPosition["sky_bkg_total_flux_inverted"] = TargetPositionInverted["sky_bkg_total_flux"]
                    if "noiseSky" in TargetPositionInverted.columns:
                        TargetPosition["noiseSky_inverted"] = TargetPositionInverted["noiseSky"]
                    logging.info("Aperture photometry on inverted image completed.")
                except Exception as exc:
                    logging.warning(f"Aperture photometry on inverted image failed: {exc}")

        # When MCMC is used, LSQ quality metrics (reduced_chi2, cfit, qfit) may be NaN.
        # Only log them when they are present and finite.
        if "reduced_chi2" in TargetPosition:
            reduced_chi2_value = TargetPosition["reduced_chi2"].iloc[0]
            if np.isfinite(reduced_chi2_value):
                logging.info(f"Target reduced chi2{inverted_tag}:\t{reduced_chi2_value:.1e}")

        if "cfit" in TargetPosition:
            cfit_value = TargetPosition["cfit"].iloc[0]
            if np.isfinite(cfit_value):
                logging.info(f"Target cfit{inverted_tag}:\t\t{cfit_value:.1e}")

        if "qfit" in TargetPosition:
            qfit_value = TargetPosition["qfit"].iloc[0]
            if np.isfinite(qfit_value):
                logging.info(
                    f"Target qfit{inverted_tag}:\t\t{qfit_value:.1e} (qfit of zero indicates a good fit)"
                )

        # =============================================================================
        # Limiting magnitudes
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
                f"Transient fitted position{inverted_tag}:\tx = {TargetPosition['x_fit'].iloc[0]:.3f} +/- {TargetPosition['x_fit_err'].iloc[0]:.3f}, "
                f"y = {TargetPosition['y_fit'].iloc[0]:.3f} +/- {TargetPosition['y_fit_err'].iloc[0]:.3f}"
            )

        # Checks if template subtraction was performed.
        if PreformSubtraction:
            target_x_pix_expected, target_y_pix_expected = imageWCS.all_world2pix(
                input_yaml["target_ra"],
                input_yaml["target_dec"],
                wcs_origin,
            )
            # Ensure the header contains the aperture radius (pixels) used for plots.
            # Some instruments do not provide APER; derive it from the config.
            try:
                phot_cfg = input_yaml.get("photometry") or {}
                aper_rad_pix = phot_cfg.get("aperture_radius", None)
                if aper_rad_pix is None:
                    aper_size_fwhm = float(phot_cfg.get("aperture_size", 1.7))
                    aper_rad_pix = aper_size_fwhm * float(ImageFWHM)
                aper_rad_pix = float(aper_rad_pix)
                header["APER"] = aper_rad_pix
            except Exception:
                aper_rad_pix = float(1.7) * float(ImageFWHM)
            # Creates an instance of the plot class.
            makePlots = Plot(input_yaml=input_yaml)
            # Load weight maps if available
            weight_map_sci = None
            weight_map_ref = None
            try:
                write_dir = Path(input_yaml["fpath"]).parent
                # Try multiple possible locations for weight maps
                # Weight maps are now copied to aligned directory after SWarp
                possible_paths = [
                    write_dir / "aligned" / "science_image.weight.fits",
                    write_dir / "aligned" / "resampled" / "science_image.weight.fits",
                    write_dir / "resampled" / "science_image.weight.fits",
                    write_dir / "science_image.weight.fits",
                ]
                sci_weight_path = None
                for p in possible_paths:
                    if p.exists():
                        sci_weight_path = p
                        with fits.open(p) as hdul:
                            weight_map_sci = np.asarray(hdul[0].data, dtype=np.float32)
                        break
                if sci_weight_path is None:
                    pass  # Weight maps are optional, no warning needed

                possible_paths_ref = [
                    write_dir / "aligned" / "reference_image.weight.fits",
                    write_dir / "aligned" / "resampled" / "reference_image.weight.fits",
                    write_dir / "resampled" / "reference_image.weight.fits",
                    write_dir / "reference_image.weight.fits",
                ]
                ref_weight_path = None
                for p in possible_paths_ref:
                    if p.exists():
                        ref_weight_path = p
                        with fits.open(p) as hdul:
                            weight_map_ref = np.asarray(hdul[0].data, dtype=np.float32)
                        break
                if ref_weight_path is None:
                    pass  # Weight maps are optional, no warning needed
            except Exception as e:
                logging.warning(f"Could not load weight maps for subtraction check: {e}")
            
            # Get WCS information for marking target location
            wcs_sci = None
            wcs_ref = None
            try:
                from astropy.wcs import WCS
                with fits.open(fpath_nosub) as hdul:
                    wcs_sci = WCS(hdul[0].header)
                with fits.open(templateFpath) as hdul:
                    wcs_ref = WCS(hdul[0].header)
            except Exception as e:
                logging.warning(f"Could not load WCS for subtraction check: {e}")
            
            # Get target coordinates from input_yaml
            target_ra = input_yaml.get("target_ra")
            target_dec = input_yaml.get("target_dec")
            
            # Check for decorrelated difference image for additional panel
            diff_decorrelated = None
            try:
                # Construct the difference image path from the write directory and base filename
                write_dir = Path(input_yaml["fpath"]).parent
                base_name = os.path.splitext(os.path.basename(fpath))[0]
                diff_path = write_dir / f"diff_{base_name}.fits"
                
                # Check for both old and new naming conventions
                # New: diff_*_decorr.fits (decorrelated), diff.fits (non-decorrelated)
                # Old: diff_*_photometry.fits (non-decorrelated), diff.fits (decorrelated)
                decorr_diff_path_new = write_dir / f"diff_{base_name}_decorr.fits"
                photometry_diff_path_old = write_dir / f"diff_{base_name}_photometry.fits"
                
                # Try new naming convention first
                if decorr_diff_path_new.exists():
                    with fits.open(decorr_diff_path_new) as hdul:
                        diff_header_check = hdul[0].header
                        decorr_status = diff_header_check.get("DECORR", (False, ""))
                        logger.info(f"Decorrelated difference image (new naming) DECORR header: {decorr_status}")
                        
                        if decorr_status[0]:
                            diff_decorrelated = hdul[0].data
                            logger.info("Adding decorrelated difference image as additional panel in subtraction check")
                        else:
                            logger.info("Decorrelated image exists but DECORR=False, skipping additional panel")
                
                # Fall back to old naming convention
                elif photometry_diff_path_old.exists():
                    # Old convention: main diff.fits is decorrelated, _photometry.fits is non-decorrelated
                    # So we need to load the main diff.fits as the decorrelated version
                    if diff_path.exists():
                        with fits.open(diff_path) as hdul:
                            diff_header_check = hdul[0].header
                            decorr_status = diff_header_check.get("DECORR", (False, ""))
                            logger.info(f"Main difference image (old naming) DECORR header: {decorr_status}")
                            
                            if decorr_status[0]:
                                diff_decorrelated = hdul[0].data
                                logger.info("Adding decorrelated difference image (old naming) as additional panel in subtraction check")
                            else:
                                logger.info("Main difference image (old naming) has DECORR=False, skipping additional panel")
                    else:
                        logger.info(f"Main difference image not found at {diff_path}")
                else:
                    logger.debug("No decorrelated difference image found (checked both naming conventions)")
            except Exception as e:
                logger.warning(f"Could not load decorrelated difference image: {e}")
            
            # Plots the template subtraction check.
            makePlots.subtraction_check(
                image=get_image(fpath_nosub),
                ref=get_image(templateFpath),
                diff=diff_image_for_plot,
                expected_location=[target_x_pix_expected, target_y_pix_expected],
                fitted_location=[
                    TargetPosition["x_fit"].iloc[0],
                    TargetPosition["y_fit"].iloc[0],
                ],
                inset_size=scale,
                aperture_size=float(header.get("APER", aper_rad_pix)),
                mask=subtraction_mask,
                matching_sources=MatchingSources,
                masked_sources=variable_sources,
                weight_map_sci=weight_map_sci,
                weight_map_ref=weight_map_ref,
                wcs_sci=wcs_sci,
                wcs_ref=wcs_ref,
                target_ra=target_ra,
                target_dec=target_dec,
                masked_source_centers=masked_centers,
                kernel_half_width=kernel_half_width,
                diff_decorrelated=diff_decorrelated,
            )

        # Target FWHM should reflect the *measured target* width on this frame,
        # not the global image FWHM or the PSF model FWHM used to build the ePSF.
        #
        # `TargetPosition["fwhm_psf"]` is typically the PSF-model FWHM (often == ImageFWHM),
        # so we keep that separately in the output as `fwhm_psf` and prefer a direct
        # Gaussian-on-target measurement for `target_fwhm`.
        target_fwhm = np.nan
        target_fwhm_gauss = np.nan
        try:
            # Maximum allowed offset (pixels) for the target Gaussian centroid.
            # `Find_FWHM.fit_gaussian` caps dx/dy to 3 px internally for stability.
            max_radius_pix = 3.0
            # Use the difference image if available (PreformSubtraction=True),
            # otherwise fall back to the science image. The ConstantModel
            # background term in fit_gaussian handles the near-zero background.
            _sci = image
            _tx = float(TargetPosition["x_fit"].iloc[0])
            _ty = float(TargetPosition["y_fit"].iloc[0])
            _half = max(10, int(np.ceil(3 * ImageFWHM)))
            cutout = Cutout2D(_sci, (_tx, _ty), (2 * _half + 1, 2 * _half + 1), mode="partial", fill_value=np.nan)
            _cutout = np.asarray(cutout.data, dtype=np.float32)
            gaussian_fits = Find_FWHM(input_yaml=input_yaml).fit_gaussian(
                _cutout,
                x=cutout.input_position_cutout[0],
                y=cutout.input_position_cutout[1],
                dx=max_radius_pix,
                dy=max_radius_pix,
                sigma=ImageFWHM / 2.335,
            )
            target_fwhm_gauss = float(gaussian_fits.get("fwhmx", np.nan))
        except Exception:
            target_fwhm_gauss = np.nan

        if np.isfinite(target_fwhm_gauss) and target_fwhm_gauss > 0:
            target_fwhm = float(target_fwhm_gauss)
        elif (
            (not do_aperture_ONLY)
            and "fwhm_psf" in TargetPosition.columns
            and np.isfinite(TargetPosition["fwhm_psf"].iloc[0])
        ):
            # Fallback only if Gaussian fit failed.
            target_fwhm = float(TargetPosition["fwhm_psf"].iloc[0])
        else:
            target_fwhm = float(ImageFWHM) if np.isfinite(ImageFWHM) else np.nan

        # If the target PSF S/N is very low, any target-width estimate is
        # effectively unconstrained and can be misleading. In that case, do not
        # report a target FWHM.
        try:
            target_snr_psf = np.nan
            if (
                (not do_aperture_ONLY)
                and "flux_PSF" in TargetPosition.columns
                and "flux_PSF_err" in TargetPosition.columns
            ):
                _f = float(TargetPosition["flux_PSF"].iloc[0])
                _fe = float(TargetPosition["flux_PSF_err"].iloc[0])
                if np.isfinite(_fe) and _fe > 0:
                    target_snr_psf = float(np.abs(_f) / _fe)
            if np.isfinite(target_snr_psf) and target_snr_psf < 3.0:
                target_fwhm = np.nan
        except Exception:
            pass

        #  Position Offset Analysis
        # Analyzes the position offset of the target.
        target_coords = SkyCoord(
            target_ra,
            target_dec,
            unit=(u.deg, u.deg),
            frame="icrs",
        )

        # Converts pixel coordinates to world coordinates.
        # Use origin=0 for consistent 0-based indexing (matching numpy arrays)
        extracted_position = imageWCS.all_pix2world(
            TargetPosition["x_fit"].iloc[0],
            TargetPosition["y_fit"].iloc[0],
            0,
        )

        # Creates a SkyCoord object for the extracted position.
        coords_science_i = SkyCoord(
            extracted_position[0],
            extracted_position[1],
            unit=(u.deg, u.deg),
            frame="icrs",
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
        logging.info(f"Target SNR (aperture):\t{snr_ap:.1f}")
        
        # If inverted fit was used, also log the inverted SNR for aperture
        if "_inverted_fit" in TargetPosition.columns and TargetPosition["_inverted_fit"].iloc[0]:
            # Compute inverted aperture SNR from absolute flux
            if "flux_AP" in TargetPosition.columns and "flux_AP_err" in TargetPosition.columns:
                ap_flux = float(TargetPosition["flux_AP"].iloc[0])
                ap_err = float(TargetPosition["flux_AP_err"].iloc[0])
                if ap_err > 0 and np.isfinite(ap_err):
                    snr_ap_inverted = np.abs(ap_flux) / ap_err
                    logging.info(f"Target SNR (aperture) [inverted]:\t{snr_ap_inverted:.1f}")
        
        if (
            not do_aperture_ONLY
            and "flux_PSF" in TargetPosition.columns
            and "flux_PSF_err" in TargetPosition.columns
        ):
            flux_psf = float(TargetPosition["flux_PSF"].iloc[0])
            flux_psf_err = float(TargetPosition["flux_PSF_err"].iloc[0])
            snr_psf = (
                np.abs(flux_psf) / flux_psf_err  # Use absolute flux for SNR (significance is always positive)
                if flux_psf_err > 0 and np.isfinite(flux_psf_err)
                else np.nan
            )
            if np.isfinite(snr_psf):
                logging.info(f"Target SNR (PSF){inverted_tag}:\t{snr_psf:.1f}")
            # Difference-image PSF before inverted replacement (negative flux = oversubtraction dip)
            if (
                "_inverted_fit" in TargetPosition.columns
                and TargetPosition["_inverted_fit"].iloc[0]
                and "flux_PSF_normal" in TargetPosition.columns
            ):
                fn = float(TargetPosition["flux_PSF_normal"].iloc[0])
                fne = float(TargetPosition["flux_PSF_err_normal"].iloc[0])
                if np.isfinite(fn) and np.isfinite(fne) and fne > 0:
                    logging.info(
                        "Target PSF on difference image (pre-invert): flux=%.4g +/- %.4g e/s, SNR=%.1f",
                        fn,
                        fne,
                        np.abs(fn) / fne,
                    )
                elif np.isfinite(fn):
                    logging.info(
                        "Target PSF on difference image (pre-invert): flux=%.4g e/s",
                        fn,
                    )
        logging.info(
            f"Target threshold:\t{TargetPosition['threshold'].iloc[0]:.1f} x background standard deviation"
        )
        logging.info(f"Target detectability:\t{target_beta * 100:.1f} %")
        logging.info(f"Target FWHM:\t\t{target_fwhm:.1f} px")

        # Calculates pixel offsets.
        dx_pix = TargetPosition["x_fit"].iloc[0] - input_yaml["target_x_pix"]
        dy_pix = TargetPosition["y_fit"].iloc[0] - input_yaml["target_y_pix"]
        offset_pix = np.sqrt(dx_pix**2 + dy_pix**2)

        logging.info(
            "Position offset:\texpected (%.3f, %.3f) -> fitted (%.3f, %.3f) px | "
            "dx=%+.3f, dy=%+.3f | total=%.3f px",
            input_yaml['target_x_pix'], input_yaml['target_y_pix'],
            TargetPosition['x_fit'].iloc[0], TargetPosition['y_fit'].iloc[0],
            dx_pix, dy_pix, offset_pix,
        )

        # Calculates RA/Dec error in arcseconds from pixel errors.
        if not np.isnan(TargetPosition["x_fit_err"].iloc[0]) and not np.isnan(
            TargetPosition["y_fit_err"].iloc[0]
        ):
            # Builds sky coordinates for (xpix, ypix).
            sky_center = SkyCoord(
                extracted_position[0],
                extracted_position[1],
                unit=(u.deg, u.deg),
                frame="icrs",
            )
            # Builds sky coordinates for (xpix + xpix_err, ypix) and (xpix, ypix + ypix_err).
            # pixel_to_world uses 0-based indexing by default (matching numpy arrays)
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
                "Fitting uncertainty: %.3f, %.3f px",
                TargetPosition['x_fit_err'].iloc[0], TargetPosition['y_fit_err'].iloc[0],
            )
        else:
            ra_err = np.nan
            dec_err = np.nan
            fitting_error_arcsec = 0
            logging.info("Fitting uncertainty: N/A (fit did not converge)")

        # Calculates the offset in arcseconds (including direction).
        # pixel_to_world uses 0-based indexing by default (matching numpy arrays)
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
            "Sky offset: dRA=%+.3f\", dDec=%+.3f\" | total separation: %.3f +/- %.3f arcsec",
            dra_arcsec, ddec_arcsec, separation, fitting_error_arcsec,
        )

        # Store fitted RA/Dec for output
        fitted_ra_deg = fitted_sky.ra.degree
        fitted_dec_deg = fitted_sky.dec.degree

        # =============================================================================
        # Calibration and Output
        # =============================================================================
        logging.info(
            border_msg("Calibrated magnitudes (AP and PSF on target)")
        )
        #  Calibrate Magnitudes
        # Calibrates the magnitudes for each method (AP, PSF)

        for method in ["AP", "PSF"]:
            if method not in image_zeropoint or "zeropoint" not in image_zeropoint[method]:
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
                apply_ap_corr = bool(
                    (input_yaml.get("photometry") or {}).get(
                        "apply_aperture_correction", False
                    )
                )
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
                    "Instrumental %s %s%s-band magnitude: %.3f +/- %.3f [mag]",
                    method,
                    input_yaml["imageFilter"],
                    inverted_tag if method == "PSF" else "",
                    TargetPosition.at[idx, inst_col],
                    TargetPosition.at[idx, inst_err_col],
                )

                logging.info(
                    "Calibrated %s %s%s-band magnitude: %.3f +/- %.3f [mag]",
                    method,
                    input_yaml["imageFilter"],
                    inverted_tag if method == "PSF" else "",
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
        # Calculates the injected detection limit if the target has low SNR.
        InjectedLimit = np.nan
        _multi_snr_results = None
        if (
            TargetPosition.at[idx, "threshold"] < 5
            or TargetPosition.at[idx, "SNR"] < 5
            or get_LimitingMagnitude
        ):
            snr_val = TargetPosition.at[idx, "SNR"]
            logging.info(
                border_msg("Limiting magnitude (injection) near target")
            )
            # Limits instance already created earlier for get_cutout
            try:
                # Use the full image for PSF calibration to avoid coordinate conversion issues
                image_for_limiting = image  # Always use full image, not cutout
                # If we already have the local-fit cutout, don't re-cut it again
                # (Limits.get_cutout uses input_yaml target pixel coords which are
                # full-frame and will be out-of-bounds on the cutout array).
                if target_cutout is not None:
                    expandedCutout = np.asarray(target_cutout, dtype=float)
                else:
                    # Gets the expanded cutout of the image.
                    # get_cutout returns (data, cx, cy) tuple
                    cutout_result = getDetectionLimits.get_cutout(image=image_for_limiting)
                    if cutout_result is not None:
                        expandedCutout, _, _ = cutout_result
                    else:
                        expandedCutout = None
                if expandedCutout is None:
                    logging.warning(
                        "getCutout returned None; skipping detection limits."
                    )
                    InjectedLimit = np.nan
                else:
                    if "limiting_magnitude" not in input_yaml or not isinstance(
                        input_yaml.get("limiting_magnitude"), dict
                    ):
                        input_yaml["limiting_magnitude"] = {}
                    lim_cfg = input_yaml["limiting_magnitude"]
                    # Match injection/recovery to how the transient is measured:
                    # aperture-only -> AP; PSF path -> PSF (explicit EMCEE still allowed).
                    _rraw = lim_cfg.get("recovery_method", "auto")
                    _rlow = (
                        str(_rraw).strip().lower()
                        if _rraw is not None
                        else "auto"
                    )
                    if _rlow in (
                        "auto",
                        "default",
                        "match_transient",
                        "match_target",
                    ):
                        _res = "AP" if do_aperture_ONLY else "PSF"
                        lim_cfg["recovery_method"] = _res
                        logging.info(
                            "limiting_magnitude.recovery_method auto -> %s (transient path: %s)",
                            _res,
                            "aperture-only" if do_aperture_ONLY else "PSF",
                        )
                    beta_limit = float(lim_cfg.get("beta_limit", 0.5))
                    raw_initial_guess = lim_cfg.get("initial_guess", np.nan)
                    try:
                        initial_guess = (
                            np.nan
                            if raw_initial_guess is None
                            else float(raw_initial_guess)
                        )
                    except Exception:
                        initial_guess = np.nan
                    try:
                        beta_sigma = float(
                            flux_upper_limit(
                                n=3.0, sigma=1.0, beta_p=float(beta_limit)
                            )
                        )
                        beta_sigma_str = f"{beta_sigma:.2f}"
                    except Exception:
                        beta_sigma_str = "unknown"
                    logging.info(
                        "Limiting mag config:\tbeta_limit=%g (~%s sigma), "
                        "completeness_target=%.2f, recovery_method=%s",
                        float(beta_limit), beta_sigma_str,
                        float(lim_cfg.get("completeness_target", 0.5)),
                        str(lim_cfg.get("recovery_method", "auto")),
                    )
                    detection_snr_limit = lim_cfg.get("detection_limit", None)
                    # Calculates the injected detection limit.
                    if epsf_model:
                        # Use a zeropoint consistent with the recovery/photometry method.
                        # - AP recovery -> AP zeropoint (if available)
                        # - PSF/EMCEE recovery -> PSF zeropoint preferred, else fall back to AP
                        rec_method = str(lim_cfg.get("recovery_method", "PSF")).strip().upper()
                        if rec_method in {"MCMC", "EMCEE"}:
                            rec_method = "EMCEE"
                        if rec_method in {"PSF", "EMCEE"}:
                            if "PSF" in image_zeropoint and "zeropoint" in image_zeropoint["PSF"]:
                                zeropoint = image_zeropoint["PSF"]["zeropoint"]
                            elif "AP" in image_zeropoint and "zeropoint" in image_zeropoint["AP"]:
                                zeropoint = image_zeropoint["AP"]["zeropoint"]
                            else:
                                zeropoint = None
                        else:
                            if "AP" in image_zeropoint and "zeropoint" in image_zeropoint["AP"]:
                                zeropoint = image_zeropoint["AP"]["zeropoint"]
                            elif "PSF" in image_zeropoint and "zeropoint" in image_zeropoint["PSF"]:
                                zeropoint = image_zeropoint["PSF"]["zeropoint"]
                            else:
                                zeropoint = None
                        logging.info(
                            "Performing artificial source injection (beta_limit=%.3f, snr_gate=%s)",
                            float(beta_limit),
                            "off" if detection_snr_limit is None else str(detection_snr_limit),
                        )
                        # For consistency, run injection directly on the full image.
                        # Prefer the fitted position; fall back to the expected pixel
                        # position when the PSF fit did not converge (x_fit/y_fit NaN).
                        _lp_x = float(TargetPosition.at[idx, "x_fit"])
                        _lp_y = float(TargetPosition.at[idx, "y_fit"])
                        if not (np.isfinite(_lp_x) and np.isfinite(_lp_y)):
                            _lp_x = float(TargetPosition.at[idx, "x_pix"])
                            _lp_y = float(TargetPosition.at[idx, "y_pix"])
                            logging.info(
                                "Limiting magnitude: PSF fit position is NaN; "
                                "using expected pixel position (%.2f, %.2f) for injection.",
                                _lp_x, _lp_y,
                            )
                        lim_pos = (_lp_x, _lp_y)
                        lim_rms = background_rms
                        # Limiting-magnitude injection uses the same image used for target
                        # measurement by default (including any local DC bias/lift applied by
                        # background.remove_local_surface in shifted-mean mode).
                        #
                        # Optional expert toggle: revert that lift for injection/recovery to
                        # measure depth on the raw (unshifted) image scale.
                        # NaN-fill defects (saturated pixels, streaks, trails, detected sources)
                        # so that injection site filtering and recovery photometry treat them
                        # as invalid.  Use a copy to avoid mutating the shared image array.
                        # For difference images: do NOT mask detected sources (use hardware_defects_mask
                        # which excludes the source mask). Properly subtracted diff images should have
                        # no sources, so we want to inject in those regions too.
                        is_diff_image = "diff_" in os.path.basename(str(fpath))
                        if is_diff_image:
                            # Copy at float32 precision then mask in-place — avoids creating a
                            # temporary float64 intermediate that np.where().astype() would produce.
                            image_for_limits = np.asarray(image, dtype=np.float32)
                            image_for_limits[hardware_defects_mask] = np.nan
                            logging.info(
                                "Limiting magnitude: using hardware_defects_mask (no source masking) for difference image."
                            )
                        else:
                            image_for_limits = np.asarray(image, dtype=np.float32)
                            image_for_limits[defects_mask] = np.nan
                        try:
                            lim_cfg_local = input_yaml.get("limiting_magnitude") or {}
                            revert_lift = bool(lim_cfg_local.get("revert_target_dc_bias_for_injection", False))
                        except Exception:
                            revert_lift = False
                        if revert_lift:
                            try:
                                if (
                                    "local_cutout_nonneg_lift" in locals()
                                    and "local_cutout_box" in locals()
                                    and np.isfinite(local_cutout_nonneg_lift)
                                    and float(local_cutout_nonneg_lift) != 0.0
                                    and isinstance(local_cutout_box, (tuple, list))
                                    and len(local_cutout_box) == 4
                                ):
                                    y0b, y1b, x0b, x1b = [int(v) for v in local_cutout_box]
                                    if (
                                        0 <= y0b < y1b <= image.shape[0]
                                        and 0 <= x0b < x1b <= image.shape[1]
                                    ):
                                        # Start from the already defect-NaN-filled array so
                                        # the DC-bias revert doesn't lose the defects mask.
                                        image_for_limits = np.asarray(image_for_limits, dtype=np.float32).copy()
                                        image_for_limits[y0b:y1b, x0b:x1b] = (
                                            image_for_limits[y0b:y1b, x0b:x1b]
                                            - float(local_cutout_nonneg_lift)
                                        )
                                        logging.info(
                                            "Limiting magnitude: reverted local target DC bias (+%.6g) for injection box %s "
                                            "(limiting_magnitude.revert_target_dc_bias_for_injection=True).",
                                            float(local_cutout_nonneg_lift),
                                            str(local_cutout_box),
                                        )
                            except Exception:
                                image_for_limits = image

                        # Consistent with target photometry (see recovery_method auto above):
                        # allow negative annulus medians (do not floor to 0) during
                        # injection/recovery measurements.
                        _old_enforce_nn_lim = None
                        try:
                            _old_enforce_nn_lim = bool(
                                (input_yaml.get("photometry") or {}).get(
                                    "enforce_nonnegative_local_background", True
                                )
                            )
                            if "photometry" not in input_yaml or input_yaml["photometry"] is None:
                                input_yaml["photometry"] = {}
                            input_yaml["photometry"]["enforce_nonnegative_local_background"] = False
                            # Check if multi-SNR thresholds are configured
                            lim_cfg_local = input_yaml.get("limiting_magnitude") or {}
                            snr_thresholds = lim_cfg_local.get("snr_thresholds", [3.0])
                            
                            if len(snr_thresholds) > 1 and lim_cfg_local.get("report_all_limits", False):
                                # Use multi-SNR function when multiple thresholds are configured
                                multi_snr_results = getDetectionLimits.get_injected_limits_multi_snr(
                                    image_for_limits,
                                    initialGuess=initial_guess,
                                    snr_thresholds=snr_thresholds,
                                    detection_cutoff=beta_limit,
                                    position=lim_pos,
                                    epsf_model=epsf_model,
                                    background_rms=lim_rms,
                                    zeropoint=zeropoint,
                                    plot=True,
                                    n_jobs=lim_n_jobs,
                                    image_zeropoint=image_zeropoint,
                                )
                                
                                # Extract the primary limit for backward compatibility (use 3σ if available, otherwise first)
                                if 'snr_3.0' in multi_snr_results and multi_snr_results['snr_3.0'].get('valid', False):
                                    InjectedLimit = multi_snr_results['snr_3.0']['limiting_mag']
                                elif len(multi_snr_results) > 0:
                                    first_valid = next((v for k, v in multi_snr_results.items() 
                                                    if k.startswith('snr_') and v.get('valid', False)), None)
                                    InjectedLimit = first_valid['limiting_mag'] if first_valid else np.nan
                                else:
                                    InjectedLimit = np.nan
                                
                                # Store multi-SNR results for CALIB file output (deferred until output dict exists)
                                _multi_snr_results = multi_snr_results
                                
                            else:
                                # Use single-SNR function for backward compatibility
                                InjectedLimit = getDetectionLimits.get_injected_limit(
                                    image_for_limits,
                                    initialGuess=initial_guess,
                                    detection_limit=detection_snr_limit,
                                    detection_cutoff=beta_limit,
                                    position=lim_pos,
                                    epsf_model=epsf_model,
                                    background_rms=lim_rms,
                                    zeropoint=zeropoint,
                                    plot=True,
                                    n_jobs=lim_n_jobs,
                                    image_zeropoint=image_zeropoint,
                                )
                        finally:
                            if _old_enforce_nn_lim is not None:
                                try:
                                    input_yaml["photometry"][
                                        "enforce_nonnegative_local_background"
                                    ] = bool(_old_enforce_nn_lim)
                                except Exception:
                                    pass
                        # Recompute aperture beta with the same n used in injection trials
                        # (limits._injection_worker passes beta_n=3.0 into beta_aperture). Do not
                        # use photometry.detection_limit here: that is a separate pipeline gate and
                        # would make stored beta disagree with the limiting-magnitude experiment.
                        # Note: recovery_method PSF gates trials on fit SNR, but beta_aperture is
                        # still computed with n=3 for diagnostics/completeness plots.
                        if np.isfinite(InjectedLimit):
                            try:
                                target_beta = float(
                                    beta_aperture(
                                        n=float(BETA_APERTURE_SIGMA_N),
                                        flux_aperture=float(
                                            TargetPosition["flux_AP"].iloc[0]
                                        ),
                                        sigma=float(
                                            TargetPosition["noiseSky"].iloc[0]
                                        ),
                                        npix=float(
                                            TargetPosition["area"].iloc[0]
                                        ),
                                    )
                                )
                            except Exception:
                                pass
            except Exception as e:
                log_exception(e)
                InjectedLimit = np.nan

        # =============================================================================
        # Save Output
        # =============================================================================

        #  Initialize Output Dictionary
        # Initializes the output dictionary with all values at once.
        output = {
            # More descriptive than the full FITS path:
            # `filename` is the FITS stem used in our standardized output names.
            "filename": input_yaml["base"],
            # Keep full path for downstream code that needs to locate plot files.
            "filename_path": fpath,
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
            "ra": fitted_ra_deg,
            "dec": fitted_dec_deg,
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
                    np.abs(TargetPosition.at[idx, "flux_PSF"])  # Use absolute flux for SNR
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
            # Limiting magnitude in the instrumental-magnitude system.
            # Downstream plotting/lightcurve code converts to apparent magnitude by adding
            # the selected band/method zeropoint (single zeropoint application).
            "limiting_inst_mag": (
                float(InjectedLimit) if np.isfinite(InjectedLimit) else np.nan
            ),
            "PreformSubtractioned": (
                PreformSubtraction if "PreformSubtraction" in locals() else False
            ),
            "etime": time.time() - start,
        }
        output.update(
            {
                "mag_ap": np.nan,
                "mag_ap_err": np.nan,
                "mag_psf": np.nan,
                "mag_psf_err": np.nan,
                "inst_mag_ap": np.nan,
                "inst_mag_ap_err": np.nan,
                "inst_mag_psf": np.nan,
                "inst_mag_psf_err": np.nan,
                "zp_ap": np.nan,
                "zp_ap_err": np.nan,
                "zp_psf": np.nan,
                "zp_psf_err": np.nan,
            }
        )

        # Store multi-SNR limiting magnitude results (deferred from earlier computation)
        if _multi_snr_results is not None:
            output['multi_snr_limits'] = _multi_snr_results
            
            # Add individual limiting magnitude columns to OUTPUT CSV
            for key, result in _multi_snr_results.items():
                if key.startswith('snr_') and result.get('valid', False):
                    snr = result.get('snr_threshold', np.nan)
                    lim_mag = result.get('limiting_mag', np.nan)
                    if np.isfinite(snr) and np.isfinite(lim_mag):
                        # Convert to apparent magnitude if zeropoint is available
                        apparent_mag = lim_mag
                        if image_zeropoint and len(image_zeropoint) > 0:
                            first_method = list(image_zeropoint.keys())[0]
                            if "zeropoint" in image_zeropoint[first_method]:
                                zp = image_zeropoint[first_method]["zeropoint"]
                                if np.isfinite(zp):
                                    apparent_mag = lim_mag + zp
                        # Add individual limiting magnitude column
                        column_name = f"limiting_mag_{snr:.0f}s2n"
                        output[column_name] = apparent_mag

        # Provide lowercase aliases for downstream tools (e.g. lightcurve.py) that
        # expect snake_case columns.
        try:
            output["snr_ap"] = float(output.get("SNR_AP", np.nan))
        except Exception:
            output["snr_ap"] = np.nan
        try:
            output["snr_psf"] = float(output.get("SNR_PSF", np.nan))
        except Exception:
            output["snr_psf"] = np.nan

        # Update generic snr column to use the best available SNR (prefer PSF if higher)
        # This ensures the default snr column reflects the most reliable detection statistic
        # IMPORTANT: This must happen BEFORE the detection flag computation below
        try:
            snr_ap = output.get("snr_ap", np.nan)
            snr_psf = output.get("snr_psf", np.nan)
            if np.isfinite(snr_psf) and np.isfinite(snr_ap):
                # Use PSF SNR if it's higher (better detection)
                output["snr"] = max(snr_psf, snr_ap)
            elif np.isfinite(snr_psf):
                output["snr"] = snr_psf
            else:
                output["snr"] = snr_ap
        except Exception:
            pass

        # Explicit detection flag for downstream light-curve processing.
        # Forced photometry is always performed; this records whether the measured
        # S/N exceeds the configured detection threshold. Using the best
        # available SNR (max of PSF/AP) gives a single, clear detection state.
        # Note: We check SNR only (not magnitude finiteness) because magnitudes are
        # populated later in the code (after zeropoint application). The SNR is
        # sufficient for detection classification.
        try:
            best_snr = float(output.get("snr", np.nan))
            det_thresh = float(detection_limit)
            output["is_detection"] = bool(
                np.isfinite(best_snr) and best_snr >= det_thresh
            )
        except Exception:
            output["is_detection"] = False

        try:
            if "flux_AP" in TargetPosition.columns:
                output["flux_ap"] = float(TargetPosition.at[idx, "flux_AP"])
            if "flux_AP_err" in TargetPosition.columns:
                output["flux_ap_err"] = float(TargetPosition.at[idx, "flux_AP_err"])
            if "flux_PSF" in TargetPosition.columns:
                output["flux_psf"] = float(TargetPosition.at[idx, "flux_PSF"])
            if "flux_PSF_err" in TargetPosition.columns:
                output["flux_psf_err"] = float(TargetPosition.at[idx, "flux_PSF_err"])
            
            # Add inverted fit parameters if inverted fit was used
            if "_inverted_fit" in TargetPosition.columns and TargetPosition.at[idx, "_inverted_fit"]:
                # Add the flag itself to output so lightcurve can detect it
                output["_inverted_fit"] = True
                # Difference-image (normal) PSF fit preserved before inverted overwrite
                if "flux_PSF_normal" in TargetPosition.columns:
                    output["flux_psf_normal"] = float(
                        TargetPosition.at[idx, "flux_PSF_normal"]
                    )
                if "flux_PSF_err_normal" in TargetPosition.columns:
                    output["flux_psf_err_normal"] = float(
                        TargetPosition.at[idx, "flux_PSF_err_normal"]
                    )
                if "flux_PSF_normal" in TargetPosition.columns and "flux_PSF_err_normal" in TargetPosition.columns:
                    fn = float(TargetPosition.at[idx, "flux_PSF_normal"])
                    fne = float(TargetPosition.at[idx, "flux_PSF_err_normal"])
                    if fne > 0 and np.isfinite(fne):
                        output["snr_psf_normal"] = float(np.abs(fn) / fne)
                for _nk, _tk in (
                    ("x_fit_normal", "x_fit_normal"),
                    ("y_fit_normal", "y_fit_normal"),
                    ("x_fit_err_normal", "x_fit_err_normal"),
                    ("y_fit_err_normal", "y_fit_err_normal"),
                    ("cfit_normal", "cfit_normal"),
                    ("qfit_normal", "qfit_normal"),
                    ("reduced_chi2_normal", "reduced_chi2_normal"),
                    ("flags_normal", "flags_normal"),
                ):
                    if _tk in TargetPosition.columns:
                        v = TargetPosition.at[idx, _tk]
                        output[_nk] = int(v) if _nk == "flags_normal" and pd.notna(v) else float(v)
                _flt = input_yaml["imageFilter"]
                _inst_n = f"inst_{_flt}_PSF_normal"
                _inst_ne = f"inst_{_flt}_PSF_normal_err"
                if _inst_n in TargetPosition.columns:
                    output[_inst_n.lower()] = float(TargetPosition.at[idx, _inst_n])
                if _inst_ne in TargetPosition.columns:
                    output[_inst_ne.lower()] = float(TargetPosition.at[idx, _inst_ne])
                if "flux_PSF_inverted" in TargetPosition.columns:
                    output["flux_psf_inverted"] = float(TargetPosition.at[idx, "flux_PSF_inverted"])
                if "flux_PSF_err_inverted" in TargetPosition.columns:
                    output["flux_psf_err_inverted"] = float(TargetPosition.at[idx, "flux_PSF_err_inverted"])
                if "inst_inverted" in TargetPosition.columns:
                    output["inst_inverted"] = float(TargetPosition.at[idx, "inst_inverted"])
                if "inst_inverted_err" in TargetPosition.columns:
                    output["inst_inverted_err"] = float(TargetPosition.at[idx, "inst_inverted_err"])
                # Inverted fit SNR for PSF
                if "flux_PSF_inverted" in TargetPosition.columns and "flux_PSF_err_inverted" in TargetPosition.columns:
                    inv_flux = float(TargetPosition.at[idx, "flux_PSF_inverted"])
                    inv_err = float(TargetPosition.at[idx, "flux_PSF_err_inverted"])
                    if inv_err > 0 and np.isfinite(inv_err):
                        output["snr_psf_inverted"] = np.abs(inv_flux) / inv_err
                # Inverted fit SNR for aperture (from absolute flux)
                if "flux_AP" in TargetPosition.columns and "flux_AP_err" in TargetPosition.columns:
                    ap_flux = float(TargetPosition.at[idx, "flux_AP"])
                    ap_err = float(TargetPosition.at[idx, "flux_AP_err"])
                    if ap_err > 0 and np.isfinite(ap_err):
                        output["snr_ap_inverted"] = np.abs(ap_flux) / ap_err
                # Inverted aperture photometry parameters (from Stage 2)
                if "flux_AP_inverted" in TargetPosition.columns:
                    output["flux_ap_inverted"] = float(TargetPosition.at[idx, "flux_AP_inverted"])
                if "flux_AP_err_inverted" in TargetPosition.columns:
                    output["flux_ap_err_inverted"] = float(TargetPosition.at[idx, "flux_AP_err_inverted"])
                if "SNR_AP_inverted" in TargetPosition.columns:
                    output["snr_ap_inverted_stage2"] = float(TargetPosition.at[idx, "SNR_AP_inverted"])
                if "local_bkg_raw_inverted" in TargetPosition.columns:
                    output["local_bkg_raw_inverted"] = float(TargetPosition.at[idx, "local_bkg_raw_inverted"])
                if "local_bkg_used_inverted" in TargetPosition.columns:
                    output["local_bkg_used_inverted"] = float(TargetPosition.at[idx, "local_bkg_used_inverted"])
                # Inverted PSF fit parameters
                if "x_fit_inverted" in TargetPosition.columns:
                    output["x_fit_inverted"] = float(TargetPosition.at[idx, "x_fit_inverted"])
                if "y_fit_inverted" in TargetPosition.columns:
                    output["y_fit_inverted"] = float(TargetPosition.at[idx, "y_fit_inverted"])
                if "x_fit_err_inverted" in TargetPosition.columns:
                    output["x_fit_err_inverted"] = float(TargetPosition.at[idx, "x_fit_err_inverted"])
                if "y_fit_err_inverted" in TargetPosition.columns:
                    output["y_fit_err_inverted"] = float(TargetPosition.at[idx, "y_fit_err_inverted"])
                if "cfit_inverted" in TargetPosition.columns:
                    output["cfit_inverted"] = float(TargetPosition.at[idx, "cfit_inverted"])
                if "qfit_inverted" in TargetPosition.columns:
                    output["qfit_inverted"] = float(TargetPosition.at[idx, "qfit_inverted"])
                if "reduced_chi2_inverted" in TargetPosition.columns:
                    output["reduced_chi2_inverted"] = float(TargetPosition.at[idx, "reduced_chi2_inverted"])
                if "flags_inverted" in TargetPosition.columns:
                    output["flags_inverted"] = int(TargetPosition.at[idx, "flags_inverted"])
                if "fwhm_psf_inverted" in TargetPosition.columns:
                    output["fwhm_psf_inverted"] = float(TargetPosition.at[idx, "fwhm_psf_inverted"])
        except Exception:
            pass

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
                    m_low = str(method).strip().lower()
                    if m_low in ("ap", "psf"):
                        output[f"zp_{m_low}"] = image_zeropoint[method]["zeropoint"]
                        output[f"zp_{m_low}_err"] = image_zeropoint[method][
                            "zeropoint_error"
                        ]
                except Exception:
                    pass

        # -------------------------------------------------------------------------
        # Derived outputs: limiting magnitude in apparent system (limits.py-consistent)
        # -------------------------------------------------------------------------
        try:
            lim_inst = float(output.get("limiting_inst_mag", np.nan))
            if not np.isfinite(lim_inst):
                output["limiting_mag"] = np.nan
            else:
                lim_cfg = input_yaml.get("limiting_magnitude") or {}
                recovery_method = str(lim_cfg.get("recovery_method", "PSF")).strip().upper()
                # Keep consistent with limits.py's selected_zeropoint rules.
                if recovery_method in {"AUTO", "DEFAULT", "MATCH_TRANSIENT", "MATCH_TARGET"}:
                    recovery_method = "AP" if do_aperture_ONLY else "PSF"
                elif recovery_method in {"MCMC", "EMCEE"}:
                    recovery_method = "EMCEE"

                if recovery_method in {"PSF", "EMCEE"}:
                    zp_sel = output.get("zp_psf", np.nan)
                    zp_fallback = output.get("zp_ap", np.nan)
                else:
                    zp_sel = output.get("zp_ap", np.nan)
                    zp_fallback = output.get("zp_psf", np.nan)

                zp_sel = float(zp_sel) if np.isfinite(zp_sel) else np.nan
                if not np.isfinite(zp_sel):
                    zp_sel = float(zp_fallback) if np.isfinite(zp_fallback) else np.nan

                output["limiting_mag"] = lim_inst + zp_sel if np.isfinite(zp_sel) else np.nan
        except Exception:
            output["limiting_mag"] = np.nan

        # Adds uniform flux/magnitude values (filter stored in `filter` column).
        #
        # We intentionally avoid writing wide per-filter columns like:
        #   {filter}_{method}, inst_{filter}_{method}, zp_{filter}_{method}
        # because they produce confusing/duplicated headers after concatenation.
        for method in image_zeropoint.keys():
            m_low = str(method).strip().lower()
            if m_low not in ("ap", "psf"):
                continue

            prefix = f"{image_filter}_{method}"
            inst_prefix = f"inst_{image_filter}_{method}"
            if prefix in TargetPosition.columns:
                output[f"mag_{m_low}"] = TargetPosition.at[idx, prefix]
                if f"{prefix}_err" in TargetPosition.columns:
                    output[f"mag_{m_low}_err"] = TargetPosition.at[idx, f"{prefix}_err"]
            if inst_prefix in TargetPosition.columns:
                output[f"inst_mag_{m_low}"] = TargetPosition.at[idx, inst_prefix]
                if f"{inst_prefix}_err" in TargetPosition.columns:
                    output[f"inst_mag_{m_low}_err"] = TargetPosition.at[
                        idx, f"{inst_prefix}_err"
                    ]

        # Normalise column headers to lowercase for consistent downstream use (lightcurve, etc.).
        output_normalised = {str(k).strip().lower(): v for k, v in output.items()}

        # Order columns for readability (long-form schema; keep any extras at the end).
        preferred = [
            # identity / timing
            "filename",
            "filename_path",
            "date",
            "mjd",
            "filter",
            # core photometry
            "mag_psf",
            "mag_psf_err",
            "mag_ap",
            "mag_ap_err",
            "limiting_inst_mag",
            # quality
            "snr",
            "snr_psf",
            "snr_ap",
            "threshold",
            "beta",
            "is_detection",
            # fluxes
            "flux_psf",
            "flux_psf_err",
            "flux_ap",
            "flux_ap_err",
            # calibration
            "zp_psf",
            "zp_psf_err",
            "zp_ap",
            "zp_ap_err",
            # PSF / seeing
            "target_fwhm",
            "fwhm_psf",
            "image_fwhm",
            # pointing
            "xpix",
            "ypix",
            "xpix_err",
            "ypix_err",
            "ra",
            "dec",
            "ra_err_arcsec",
            "dec_err_arcsec",
            # meta
            "telescope",
            "instrument",
            "airmass",
            "exposure_time",
            "preformsubtractioned",
            "etime",
        ]
        ordered = {}
        for k in preferred:
            if k in output_normalised:
                ordered[k] = output_normalised[k]
        for k in output_normalised.keys():
            if k not in ordered:
                ordered[k] = output_normalised[k]
        output_normalised = ordered
        # Saves the standardized per-image output.
        output_df = pd.DataFrame(output_normalised, index=[0])
        output_df.to_csv(
            output_csv_path,
            index=False,
            float_format="%.6f",
        )

        # Remove verbose multi_snr_limits dict from CALIB output (individual columns remain)
        output.pop("multi_snr_limits", None)

        # Converts the output dictionary to a string.
        output_str = dict_to_string_with_hashtag(output)

        # Write calibration file with zeropoint and sequence star information
        # Build target information as comments
        target_lines = ["# Target information"]
        if "target_name" in input_yaml and input_yaml["target_name"]:
            target_lines.append(f"# target_name: {input_yaml['target_name']}")
        if "target_ra" in input_yaml and input_yaml["target_ra"] is not None:
            target_lines.append(f"# target_ra: {input_yaml['target_ra']:.6f} deg")
        if "target_dec" in input_yaml and input_yaml["target_dec"] is not None:
            target_lines.append(f"# target_dec: {input_yaml['target_dec']:.6f} deg")
        if "target_x_pix" in input_yaml and input_yaml["target_x_pix"] is not None:
            target_lines.append(f"# target_x_pix: {input_yaml['target_x_pix']:.6f} px")
        if "target_y_pix" in input_yaml and input_yaml["target_y_pix"] is not None:
            target_lines.append(f"# target_y_pix: {input_yaml['target_y_pix']:.6f} px")
        
        # Build zeropoint info as clean comments
        zp_lines = ["# Zeropoint and calibration information"]
        for method in image_zeropoint.keys():
            if method in image_zeropoint:
                zp = image_zeropoint[method].get("zeropoint", np.nan)
                zp_err = image_zeropoint[method].get("zeropoint_error", np.nan)
                zp_lines.append(f"# {method}_zeropoint: {zp:.6f}")
                zp_lines.append(f"# {method}_zeropoint_error: {zp_err:.6f}")
                # Add color term if available
                if "color_term" in image_zeropoint[method]:
                    ct = image_zeropoint[method].get("color_term")
                    ct_err = image_zeropoint[method].get("color_term_error")
                    zp_lines.append(f"# {method}_color_term: {ct:.6f}")
                    zp_lines.append(f"# {method}_color_term_error: {ct_err:.6f}")
        
        # Add multi-SNR limiting magnitude information if available
        if _multi_snr_results is not None:
            lim_lines = ["# Multi-S/N detection limits"]
            multi_snr_results = _multi_snr_results
            
            # Add individual S/N limits
            for key, result in multi_snr_results.items():
                if key.startswith('snr_') and result.get('valid', False):
                    snr = result.get('snr_threshold', np.nan)
                    lim_mag = result.get('limiting_mag', np.nan)
                    if np.isfinite(snr) and np.isfinite(lim_mag):
                        # Convert to apparent magnitude if zeropoint is available
                        apparent_mag = lim_mag
                        if method in image_zeropoint and "zeropoint" in image_zeropoint[method]:
                            zp = image_zeropoint[method]["zeropoint"]
                            if np.isfinite(zp):
                                apparent_mag = lim_mag + zp
                        lim_lines.append(f"# limiting_mag_{snr:.0f}S2N: {apparent_mag:.3f}")
            
            # Add comparison information
            if 'comparisons' in multi_snr_results:
                lim_lines.append("# S/N threshold comparisons:")
                for comp_key, comp_data in multi_snr_results['comparisons'].items():
                    snr_high = comp_data.get('snr_high', np.nan)
                    snr_low = comp_data.get('snr_low', np.nan)
                    delta_mag = comp_data.get('delta_mag', np.nan)
                    if np.isfinite(snr_high) and np.isfinite(snr_low) and np.isfinite(delta_mag):
                        lim_lines.append(f"# delta_mag_{snr_high:.0f}S2N_vs_{snr_low:.0f}S2N: {delta_mag:+.3f}")
            
            # Add adaptive threshold recommendation
            if 'adaptive_threshold' in multi_snr_results:
                adaptive_snr = multi_snr_results['adaptive_threshold']
                lim_lines.append(f"# adaptive_snr_threshold: {adaptive_snr:.0f}S2N")
            
            zp_lines.extend(lim_lines)
        elif np.isfinite(InjectedLimit):
            # Fallback for single S/N limiting magnitude (backward compatibility)
            lim_lines = ["# Detection limits"]
            apparent_mag = InjectedLimit
            # Try to convert to apparent magnitude using available zeropoint
            if image_zeropoint and len(image_zeropoint) > 0:
                first_method = list(image_zeropoint.keys())[0]
                if "zeropoint" in image_zeropoint[first_method]:
                    zp = image_zeropoint[first_method]["zeropoint"]
                    if np.isfinite(zp):
                        apparent_mag = InjectedLimit + zp
            lim_lines.append(f"# limiting_mag_3S2N: {apparent_mag:.3f}")
            zp_lines.extend(lim_lines)

        # Opens the file in write mode to add the output string and target/zeropoint info.
        with open(calibration_file, "w") as file:
            file.write("# Output dictionary\n")
            file.write(output_str)
            file.write("\n" + "\n".join(target_lines))
            file.write("\n" + "\n".join(zp_lines))

        # Append sequence star catalog to CALIB file.
        # Use CatalogSources_for_calib (full measured catalog, saved before
        # GetZeropoint.clean() trimmed it for ZP fitting) so that all measured
        # sequence stars are recorded, not just the strict ZP-fitting subset.
        _calib_df = CatalogSources_for_calib
        if _calib_df is not None and not _calib_df.empty:
            _calib_df = _calib_df.copy()

            # Ensure PSF columns exist (NaN if PSF fitting was skipped)
            for col in ["x_fit", "y_fit", "x_fit_err", "y_fit_err",
                        "flux_PSF", "flux_PSF_err", "fwhm", "fwhm_psf"]:
                if col not in _calib_df.columns:
                    _calib_df[col] = np.nan

            # Prefer per-source SExtractor fwhm over constant fwhm_psf for CALIB output
            if "fwhm" in _calib_df.columns and "fwhm_psf" in _calib_df.columns:
                _calib_df["fwhm"] = _calib_df["fwhm"].fillna(_calib_df["fwhm_psf"])
                _calib_df.drop(columns=["fwhm_psf"], inplace=True)

            # Disambiguate duplicate SNR columns:
            #   SNR  = aperture photometry SNR (aperture_sum / sqrt_var)
            #   snr  = maxPixel / noiseSky (from functions.snr)
            #   SNR_err is never populated — drop it
            if "SNR" in _calib_df.columns:
                _calib_df.rename(columns={"SNR": "snr_ap"}, inplace=True)
            if "snr" in _calib_df.columns:
                _calib_df.rename(columns={"snr": "snr_peak"}, inplace=True)
            if "SNR_err" in _calib_df.columns:
                _calib_df.drop(columns=["SNR_err"], inplace=True)

            # Drop columns that are entirely NaN / empty to keep CALIB tidy
            _all_nan = _calib_df.columns[
                _calib_df.isna().all() | (_calib_df.astype(str) == "").all()
            ].tolist()
            if _all_nan:
                logging.debug(f"Dropping {len(_all_nan)} all-NaN columns from CALIB: {_all_nan}")
                _calib_df = _calib_df.drop(columns=_all_nan)

            # Safety check: ensure catalog still has rows and columns after cleanup
            if _calib_df.empty or len(_calib_df.columns) == 0:
                logging.warning("CatalogSources became empty after column cleanup; skipping catalog write")
            else:
                # Remove exact duplicate coordinates
                _calib_df_dedup = _remove_catalog_duplicates(
                    _calib_df,
                    method='astropy',
                    sep_threshold=0.1,
                )
                if _calib_df_dedup is not None and not _calib_df_dedup.empty:
                    logging.info(f"Writing {len(_calib_df_dedup)} catalog sources to CALIB file")
                    with open(calibration_file, "a", newline='') as file:
                        file.write("\n# Sequence star catalog used for calibration\n")
                        _calib_df_dedup.to_csv(file, index=False, float_format="%.6f")
                        file.flush()
                else:
                    logging.warning("CatalogSources_dedup is empty after deduplication; skipping catalog write")
        elif _calib_df is not None:
            logging.warning("CatalogSources is empty - no catalog sources will be written")
        else:
            logging.debug("No catalog sources available to write to CALIB file")

        # Redoes the sources if enabled.
        if input_yaml["photometry"].get("redo_sources", False):
            image, header = get_image_and_header(scienceFpath_cutout)
            # Gets the WCS information from the header.
            imageWCS = get_wcs(header)
            # Converts world coordinates of isolated sources to pixel coordinates.
            x_pix_IsolatedSources, y_pix_IsolatedSources = imageWCS.all_world2pix(
                IsolatedSources["RA"].values,
                IsolatedSources["DEC"].values,
                wcs_origin,
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
            # Use origin=0 for consistent 0-based indexing (matching numpy arrays)
            ra_IsolatedSources, dec_IsolatedSources = imageWCS.all_pix2world(
                IsolatedSources["x_pix"].values,
                IsolatedSources["y_pix"].values,
                0,
            )
            # Adds the RA and DEC columns to the IsolatedSources DataFrame.
            IsolatedSources["RA"] = ra_IsolatedSources
            IsolatedSources["DEC"] = dec_IsolatedSources
            isolated_sources_legacy = os.path.join(
                write_dir, "SOURCES_" + input_yaml["base"] + ".csv"
            )
            isolated_sources_std = os.path.join(
                write_dir, f"ISOLATED_SOURCES_{input_yaml['base']}.csv"
            )

            for path in (isolated_sources_std, isolated_sources_legacy):
                # Write header comment and catalog data in a single open.
                with open(path, "w") as file:
                    file.write("# Output dictionary\n")
                    file.write(output_str)
                    IsolatedSources.to_csv(file, index=False, float_format="%.6f")

        #  Global Photometry
        # Performs global photometry if enabled.
        if (
            image_sources is not None
            and input_yaml["photometry"].get("perform_global_photometry_sigma", None)
            is not None
        ):
            sigma_val = float(input_yaml["photometry"]["perform_global_photometry_sigma"])
            fname = f"sources_{sigma_val:.0f}sigma_{base}.csv"
            fname_std = f"GLOBAL_PHOT_{sigma_val:.0f}SIGMA_{base}.csv"
            # Writes both the output string and the catalog in one operation.
            with open(os.path.join(write_dir, fname), "w") as file:
                file.write(output_str)
                image_sources.to_csv(file, index=False, float_format="%.6f")
            with open(os.path.join(write_dir, fname_std), "w") as file:
                file.write(output_str)
                image_sources.to_csv(file, index=False, float_format="%.6f")

        # Logs the completion of photometric measurements.
        end = time.time() - start
        logging.info(log_step(f"Photometry finished [{end:.1f}s]"))
        logging.info("")

    except Exception as e:
        log_exception(e)

    return None


if __name__ == "__main__":
    run_photometry()
