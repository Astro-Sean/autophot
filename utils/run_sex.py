#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized SExtractor Wrapper for Point Source Detection and FWHM Estimation
Author: Sean Brennan
Date: October 24, 2025

Features:
    - Supports custom background and RMS maps (FITS files or 2D arrays)
    - Robust FWHM estimation using iterative sigma clipping
    - Advanced filtering for point sources, saturated sources, and edge effects
    - Handles crowded and blended sources
    - Optimized for non-uniform backgrounds
    - Optimized convolution kernel if FWHM is provided
    - Drops sources instead of setting FWHM to NaN
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, List, Set

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from scipy.spatial import cKDTree

from functions import log_step, log_warning_from_exception

logger = logging.getLogger(__name__)

# Cached path to a working SExtractor binary (avoids repeated subprocess checks).
_SEXTRACTOR_EXE: Optional[str] = None
_SEXTRACTOR_PROBE_FAILED: bool = False

# =============================================================================
# Utility Functions
# =============================================================================


def get_sextractor_executable() -> Optional[str]:
    """
    Resolve and cache a working SExtractor binary (`sex` or `sextractor` on PATH).

    Returns:
        Absolute path to the executable, or None if not found / not working.
    """
    global _SEXTRACTOR_EXE, _SEXTRACTOR_PROBE_FAILED
    if _SEXTRACTOR_EXE is not None:
        return _SEXTRACTOR_EXE
    if _SEXTRACTOR_PROBE_FAILED:
        return None
    for name in ("sex", "sextractor"):
        path = shutil.which(name)
        if path is None:
            continue
        try:
            subprocess.run(
                [path, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=15,
            )
            _SEXTRACTOR_EXE = path
            logger.info("SExtractor executable detected: %s", path)
            return path
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    _SEXTRACTOR_PROBE_FAILED = True
    logger.error(
        "SExtractor not found. Install SExtractor and ensure `sex` or `sextractor` is on PATH."
    )
    return None


def reset_sextractor_executable_cache() -> None:
    """
    Reset the cached SExtractor executable path and probe-failed flag.
    Call this if SExtractor is installed after the pipeline has already started.
    """
    global _SEXTRACTOR_EXE, _SEXTRACTOR_PROBE_FAILED
    _SEXTRACTOR_EXE = None
    _SEXTRACTOR_PROBE_FAILED = False
    logger.info("SExtractor executable cache cleared.")


def is_sextractor_installed() -> bool:
    """
    Check if SExtractor is installed and accessible in the system path.

    Returns:
        bool: True if SExtractor is found, False otherwise.
    """
    return get_sextractor_executable() is not None


def scale_multiplier_from_config(config: dict) -> float:
    """
    Factor used to convert measured FWHM (px) into pipeline ``scale`` (cutout/box size).

    Reads ``source_detection.scale_multipler`` (historic YAML spelling) or
    ``source_detection.scale_multiplier``, then top-level ``scale_multiplier``.
    Default 4.0 matches legacy AutoPHoT when the YAML key was never wired up.
    """
    if not isinstance(config, dict):
        return 4.0
    sd = config.get("source_detection") or {}
    if isinstance(sd, dict):
        for key in ("scale_multipler", "scale_multiplier"):
            v = sd.get(key)
            if v is not None:
                try:
                    return max(0.5, float(v))
                except (TypeError, ValueError):
                    pass
    v = config.get("scale_multiplier")
    if v is not None:
        try:
            return max(0.5, float(v))
        except (TypeError, ValueError):
            pass
    return 4.0


def clamp_scale_from_config(config: dict, scale_px: float) -> int:
    """
    Clamp a computed runtime ``scale`` (pixels) to configured min/max bounds.

    Reads:
    - source_detection.scale_min_px
    - source_detection.scale_max_px

    Returns an integer scale >= 1.
    """
    try:
        sd = (config.get("source_detection") or {}) if isinstance(config, dict) else {}
        min_px = int(sd.get("scale_min_px", 11))
        max_px = int(sd.get("scale_max_px", 300))
        min_px = max(1, min_px)
        max_px = max(min_px, max_px)
        s = int(round(float(scale_px)))
        return max(min_px, min(max_px, s))
    except Exception:
        try:
            return max(1, int(round(float(scale_px))))
        except Exception:
            return 11


@contextmanager
def temp_directory_context(base_dir: Optional[str] = None):
    """
    Context manager for creating and cleaning up a temporary directory.

    Args:
        base_dir (str, optional): Base directory for the temporary directory. Defaults to system temp.

    Yields:
        Path: Path to the temporary directory.
    """
    temp_dir = tempfile.TemporaryDirectory(prefix="PYSEx_", dir=base_dir)
    try:
        yield Path(temp_dir.name)
    finally:
        temp_dir.cleanup()


def _write_fits_array_to_file(array: np.ndarray, path: Path) -> None:
    """
    Write a 2D numpy array to a FITS file.

    Args:
        array (np.ndarray): 2D array to write.
        path (Path): Destination path for the FITS file.
    """
    from functions import safe_fits_write
    safe_fits_write(str(path), array, fits.Header())


# =============================================================================
# Core Class: SExtractorWrapper
# =============================================================================


class SExtractorWrapper:
    """
    A wrapper class for running SExtractor, optimized for point source detection and FWHM estimation.
    Supports custom background and RMS maps (FITS files or 2D arrays) for improved source detection in crowded fields.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the SExtractorWrapper with a configuration dictionary.

        Args:
            config (dict): Configuration parameters for SExtractor.
        """
        self.config = config
        # Default number of threads for SExtractor; top-level or under wcs.* in YAML.
        _wcs_cfg = config.get("wcs") or {}
        try:
            cfg_threads = int(
                config.get(
                    "sextractor_nthreads",
                    _wcs_cfg.get("sextractor_nthreads", 4),
                )
            )
        except (TypeError, ValueError):
            cfg_threads = 4
        # Respect user thread config with appropriate logging
        if cfg_threads < 1:
            logger.warning(
                "sextractor_nthreads=%d is invalid; using 1.", cfg_threads
            )
            self.n_threads = 1
        elif cfg_threads < 4:
            logger.info(
                "sextractor_nthreads=%d requested; note minimum recommended is 4 for performance.",
                cfg_threads,
            )
            self.n_threads = cfg_threads
        else:
            self.n_threads = cfg_threads

    # --- Helper Methods ---

    def _create_conv_file(self, temp_dir: Path, fwhm_pixels: float = 5.0) -> Path:
        """
        Create a convolution file for SExtractor.
        If fwhm_pixels is provided and > 0, optimize the kernel for the given FWHM.
        Otherwise, use a default 3x3 kernel.

        Args:
            temp_dir (Path): Path to the temporary directory.
            fwhm_pixels (float): FWHM in pixels, used to optimize the kernel size.

        Returns:
            Path: Path to the created convolution file.
        """

        logger.info(
            "Creating convolution kernel with FWHM: %.1f pixels", fwhm_pixels
        )

        if fwhm_pixels > 0:
            fp = float(max(3.0, min(fwhm_pixels, 10.0)))
            kernel_size = max(3, int(np.ceil(fp * 3)))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd size
            center = kernel_size // 2
            sigma = fp / 2.355  # FWHM = 2.355 * sigma
            # ogrid returns two broadcastable arrays (not one ndarray); subtract center after unpack.
            yy, xx = np.ogrid[0:kernel_size, 0:kernel_size]
            kernel = np.exp(
                -0.5 * ((xx - center) ** 2 + (yy - center) ** 2) / (sigma**2)
            )
            header_lines = (
                "CONV NORM",
                f"# {kernel_size}x{kernel_size} convolution mask with FWHM = {fp:.1f} pixels",
            )
            row_strs = (" ".join(f"{v:.6f}" for v in row) for row in kernel)
            conv_text = "\n".join((*header_lines, *row_strs))
        else:
            conv_text = """CONV NORM
# 3x3 convolution mask with FWHM = 2 pixels
1 2 1
2 4 2
1 2 1"""
        conv_path = temp_dir / "point_source.conv"
        with open(conv_path, "w") as f:
            f.write(conv_text.strip() + "\n")
        return conv_path

    def _create_nnw_file(self, temp_dir: Path) -> Path:
        """
        Create a neural network weights file for SExtractor's star/galaxy classifier.

        Args:
            temp_dir (Path): Path to the temporary directory.

        Returns:
            Path: Path to the created neural network weights file.
        """
        nnw_text = """
            NNW
            # Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
            # inputs: 9 for profile parameters + 1 for seeing.
            # outputs: Stellarity index (0.0 to 1.0)
            # Seeing FWHM range: from 0.025 to 5.5'' (images must have 1.5 < FWHM < 5 pixels)
            # Optimized for Moffat profiles with 2<= beta <= 4.
            3 10 10  1
            -1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01
            3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00
            -3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00
            4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
            -2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00
            4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e+00 -1.01592e-02 -7.54052e+00
            1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
            -5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01
            2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01
            3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01
            -8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
            -3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00
            -1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00
            0.00000e+00
            1.00000e+00
        """
        nnw_path = temp_dir / "default.nnw"
        cleaned = "\n".join(line.strip() for line in nnw_text.splitlines() if line.strip())
        with open(nnw_path, "w") as f:
            f.write(cleaned + "\n")
        return nnw_path

    def _create_param_file(self, temp_dir: Path, params: list) -> Path:
        """
        Create a parameter file for SExtractor.

        Args:
            temp_dir (Path): Path to the temporary directory.
            params (list): List of parameters to include in the file.

        Returns:
            Path: Path to the created parameter file.
        """
        param_path = temp_dir / "PYSEx.param"
        with open(param_path, "w") as f:
            f.write("\n".join(params))
        return param_path

    def _create_config_file(self, temp_dir: Path, config_dict: dict) -> Path:
        """
        Create a configuration file for SExtractor.

        Args:
            temp_dir (Path): Path to the temporary directory.
            config_dict (dict): Dictionary of configuration key-value pairs.

        Returns:
            Path: Path to the created configuration file.
        """
        config_path = temp_dir / "default.sex"
        with open(config_path, "w") as f:
            for key, value in config_dict.items():
                f.write(f"{key} {value}\n")
        return config_path

    # --- Core Methods ---

    def calculate_robust_fwhm(
        self,
        fwhm_values: np.ndarray,
        n_iterations: Optional[int] = None,
    ) -> float:
        """
        Calculate the robust FWHM using iterative sigma clipping.

        Args:
            fwhm_values (np.ndarray): Array of FWHM values.
            n_iterations (int, optional): Sigma-clipping iterations; if None, uses
                ``photometry.fwhm_sigma_clip_maxiters`` from config (default 15).

        Returns:
            float: Median FWHM of the clipped values.
        """
        if n_iterations is None:
            phot = self.config.get("photometry") or {}
            try:
                n_iterations = int(phot.get("fwhm_sigma_clip_maxiters", 15))
            except (TypeError, ValueError):
                n_iterations = 15
        n_iterations = max(1, int(n_iterations))
        clipped = sigma_clip(fwhm_values, sigma=5.0, maxiters=n_iterations)
        fwhm_value = float(np.ma.median(clipped))
        # Guard against pathological estimates (all zeros, NaNs, etc.).
        if not np.isfinite(fwhm_value) or fwhm_value <= 0.0:
            fallback = float(self.config.get("fwhm_fallback_pixels", 3.0))
            logger.warning(
                "Robust FWHM estimate is non-positive or invalid (%.3f); "
                "falling back to %.2f pixels.",
                fwhm_value,
                fallback,
            )
            return fallback
        return fwhm_value

    # import numpy as np
    #  import pandas as pd
    #  from astropy.io import fits
    #  import time
    #  import logging

    #  logger = logging.getLogger(__name__)

    def filter_sextractor_sources(
        self,
        sources: pd.DataFrame,
        header: fits.Header,
        fwhm_est: Optional[float] = None,
        saturation: Optional[float] = None,
        flags: int = 2,
        masked_sources: Optional[pd.DataFrame] = None,
        snr_limit: float = 3.0,
        NMAX: Optional[int] = 1000,  # Maximum allowed number of sources
        n_grid: int = 10,  # Number of grid cells along each axis for spatial downsampling
        relaxed_cuts: bool = False,  # If True, keep more sources (wider FWHM range, for matching)
        bad_region_mask: Optional[
            np.ndarray
        ] = None,  # Mask for chip gaps / constant-value regions
    ) -> pd.DataFrame:
        """
        Filter SExtractor sources to exclude non-point sources, saturated sources, and edge sources.
        If the final number of sources exceeds NMAX, downsample spatially to ensure uniform coverage.
        When relaxed_cuts is True, use wider FWHM bounds to retain more sources (e.g. for matching).

        Args:
            sources (pd.DataFrame): SExtractor source catalog (with renamed columns).
            header (fits.Header): FITS header for image dimensions.
            fwhm_est (float, optional): Estimated FWHM in pixels. If None, calculated robustly.
            saturation (float, optional): Saturation limit in ADU.
            flags (int, optional): Maximum allowed SExtractor flag value. Defaults to 2.
            masked_sources (pd.DataFrame, optional): Sources to exclude by position.
            snr_limit (float, optional): Minimum SNR threshold. Defaults to 3.0.
            NMAX (int, optional): Maximum number of sources to return. If None, no downsampling.
            n_grid (int, optional): Number of grid cells for spatial downsampling. Defaults to 10.
            relaxed_cuts (bool, optional): If True, use wider FWHM bounds to retain more sources. Defaults to False.
            bad_region_mask (np.ndarray, optional): Boolean mask of pixels to reject
                (e.g. chip gaps, dead areas, constant-value borders). Sources whose
                positions fall inside the mask are dropped early.

        Returns:
            pd.DataFrame: Filtered (and optionally downsampled) source catalog.
        """

        start = time.time()
        if sources.empty:
            logger.warning("SExtractor catalog is empty. Skipping filtering.")
            return sources

        n0 = len(sources)
        logger.info(f"Initial source count: {n0}")
        # Expose the raw detection count back through the config so that callers
        # (e.g. crowded-field heuristics in main.py) can distinguish between the
        # total detections and the filtered/downsampled subset used for FWHM.
        try:
            phot_cfg = self.config.setdefault("photometry", {})
            phot_cfg["last_source_detection_raw_count"] = int(n0)
        except Exception:
            # Config is best-effort; never break filtering if this fails.
            pass

        # --- Step 1: Drop sources inside bad regions (chip gaps / no-flux borders) ---
        if (
            bad_region_mask is not None
            and "x_pix" in sources.columns
            and "y_pix" in sources.columns
        ):
            ny, nx = bad_region_mask.shape
            x_idx = np.clip(np.rint(sources["x_pix"].values).astype(int), 0, nx - 1)
            y_idx = np.clip(np.rint(sources["y_pix"].values).astype(int), 0, ny - 1)
            in_bad = bad_region_mask[y_idx, x_idx]
            n_bad = int(in_bad.sum())
            if n_bad > 0:
                sources = sources[~in_bad].copy()
                logger.info(
                    f"Rejected {n_bad} sources inside constant-value / chip-gap regions"
                )

        # --- Step 2: FWHM sanity cut (relaxed_cuts keeps more for matching) ---
        # Keep a copy so we can fall back if this cut is too aggressive.
        sources_before_fwhm = sources.copy()
        if relaxed_cuts:
            fwhm_lo, fwhm_hi = 0.5, 150.0
        else:
            fwhm_lo, fwhm_hi = 1.1, 100.0
        mask = (sources["fwhm"] <= fwhm_lo) | (sources["fwhm"] >= fwhm_hi)
        n_rejected_fwhm = int(mask.sum())
        sources = sources[~mask].copy()
        if n_rejected_fwhm > 0:
            logger.info(
                f"Rejected {n_rejected_fwhm} sources outside FWHM range [{fwhm_lo}, {fwhm_hi}]"
            )
        # If this removed everything (or nearly everything), disable the FWHM cut
        # for this image rather than returning an empty catalogue.
        if len(sources) == 0 and n0 > 0:
            logger.warning(
                "FWHM filter removed all sources (n=%d). Disabling FWHM cut for this image.",
                n0,
            )
            sources = sources_before_fwhm

        # --- Step 3: SNR cut ---
        snr_ok = sources["snr"].to_numpy() > snr_limit
        n_after_snr = int((~snr_ok).sum())
        sources = sources.loc[snr_ok].copy()
        if n_after_snr > 0:
            logger.info(f"Rejected {n_after_snr} low-SNR sources (SNR < {snr_limit:.1f})")

        # --- Step 4: Estimate FWHM if needed ---
        if fwhm_est is None and len(sources) > 0:
            fwhm_est = self.calculate_robust_fwhm(sources["fwhm"].values)
            fwhm_std = np.nanstd(sources["fwhm"].values)
            logger.info(
                f"Estimated FWHM: {fwhm_est:.2f} pixels (sigma = {fwhm_std:.2f} pixels)"
            )

        # --- Step 5: Drop sources near masked positions ---
        if (
            masked_sources is not None
            and fwhm_est is not None
        ):
            if isinstance(masked_sources, Table):
                ms_df = masked_sources.to_pandas()
            else:
                ms_df = masked_sources

            mask_coords = np.empty((0, 2), dtype=np.float64)
            if ms_df is not None and len(ms_df) > 0:
                match_radius = float(fwhm_est * 2)
                mask_coords = ms_df[["x_pix", "y_pix"]].to_numpy(dtype=np.float64)

            if len(mask_coords) > 0:
                coords = sources[["x_pix", "y_pix"]].to_numpy(dtype=np.float64)
                tree = cKDTree(mask_coords)
                # Inf distance if no neighbor within match_radius
                try:
                    dist, _ = tree.query(
                        coords,
                        k=1,
                        distance_upper_bound=match_radius,
                        workers=-1,
                    )
                except TypeError:
                    dist, _ = tree.query(
                        coords, k=1, distance_upper_bound=match_radius
                    )
                near_masked = np.isfinite(dist)
                removed_near_masked = int(near_masked.sum())
                sources = sources.loc[~near_masked].copy()
                if removed_near_masked > 0:
                    logger.info(
                        "Rejected %d sources within %.2f pixels of masked regions",
                        removed_near_masked,
                        match_radius,
                    )

        # --- Step 6: FLAGS cut ---
        if "flags" in sources.columns:
            n_flagged = int((sources["flags"] > flags).sum())
            sources = sources[sources["flags"] <= flags].copy()
            if n_flagged > 0:
                logger.info(
                    "Rejected %d sources with FLAGS > %d", n_flagged, flags
                )

        # --- Step 7: Sharpness cut (relaxed range when relaxed_cuts) ---
        if "sharpness" in sources.columns:
            sharp_lo, sharp_hi = (0.1, 1.2) if relaxed_cuts else (0.2, 1.0)
            mask = (sources["sharpness"] <= sharp_lo) | (
                sources["sharpness"] >= sharp_hi
            )
            rejected_sharp = mask.sum()
            sources = sources[~mask].copy()
            if rejected_sharp > 0:
                logger.info(
                    f"Rejected {rejected_sharp} sources based on sharpness (range [{sharp_lo}, {sharp_hi}])"
            )

        # --- Step 8: Saturation cut ---
        if saturation is not None and "peak_flux" in sources.columns:
            n_saturated = len(sources[sources["peak_flux"] >= 0.99 * saturation])
            sources = sources[sources["peak_flux"] < 0.99 * saturation].copy()
            if n_saturated > 0:
                logger.info(
                    f"Rejected {n_saturated} saturated sources (peak_flux >= 0.99 x saturation)"
                )

        # --- Step 9: Edge cut ---
        x_max, y_max = header.get("NAXIS1", 0), header.get("NAXIS2", 0)
        margin = 5
        mask = (
            (sources["x_pix"] <= margin)
            | (sources["x_pix"] >= x_max - margin)
            | (sources["y_pix"] <= margin)
            | (sources["y_pix"] >= y_max - margin)
        )
        edge_rejected = mask.sum()
        sources = sources[~mask].copy()
        if edge_rejected > 0:
            logger.info(
                f"Rejected {edge_rejected} sources within {margin} pixels of the image edge"
            )

        # --- Step 10: Spatial Downsampling if needed ---
        if NMAX is not None and len(sources) > NMAX:
            x_pix = sources["x_pix"].values
            y_pix = sources["y_pix"].values

            # Divide the image into a grid
            x_bins = np.linspace(0, x_max, n_grid + 1)
            y_bins = np.linspace(0, y_max, n_grid + 1)

            # Assign each source to a grid cell
            # Use left-inclusive bins so every source falls into exactly one cell.
            x_indices = np.clip(np.digitize(x_pix, x_bins[1:-1]), 0, n_grid - 1)
            y_indices = np.clip(np.digitize(y_pix, y_bins[1:-1]), 0, n_grid - 1)

            # Create a unique identifier for each grid cell
            grid_ids = y_indices * n_grid + x_indices

            # Group sources by grid cell and sort each cell by SNR (descending)
            groups = {
                gid: grp.sort_values(by="snr", ascending=False)
                for gid, grp in sources.groupby(grid_ids)
            }

            rng = np.random.default_rng(42)
            cells = np.array(list(groups.keys()), dtype=int)
            rng.shuffle(cells)

            selected_idx: List[int] = []
            selected_set: Set[int] = set()

            # First pass: take the best source from each non-empty cell to
            # guarantee at least one source per region, up to NMAX.
            for cell in cells:
                grp = groups[cell]
                if grp.empty:
                    continue
                ix = grp.index[0]
                selected_idx.append(ix)
                selected_set.add(ix)
                if len(selected_idx) >= NMAX:
                    break

            # Additional passes: take 2nd-best, 3rd-best, ... from each cell
            # (skipping already-chosen rows) until NMAX is reached or no more
            # candidates remain. This keeps the spatial distribution uniform
            # instead of over-sampling crowded regions.
            k = 1
            while len(selected_idx) < NMAX:
                added = 0
                for cell in cells:
                    grp = groups[cell]
                    if len(grp) <= k:
                        continue
                    idx = int(grp.index[k])
                    if idx in selected_set:
                        continue
                    selected_idx.append(idx)
                    selected_set.add(idx)
                    added += 1
                    if len(selected_idx) >= NMAX:
                        break
                if added == 0:
                    break

            # If we still need more sources (e.g. some cells are very sparse),
            # fill the remainder with globally highest-SNR sources that are not
            # already selected.
            if len(selected_idx) < NMAX:
                remaining = sources.drop(index=selected_idx).sort_values(
                    by="snr", ascending=False
                )
                need = max(0, NMAX - len(selected_idx))
                extra = list(remaining.index[:need])
                selected_idx.extend(extra)

            # Combine and sort the selected sources
            sources = (
                sources.loc[selected_idx]
                .sort_values(by="snr", ascending=False)
                .head(NMAX)
            )
            logger.info(
                f"Downsampled catalogue to {NMAX} sources using a {n_grid} x {n_grid} spatial grid"
            )

        # --- Wrap up ---
        # As a final safety net, if all sources have been removed by the
        # filtering steps but we started with a non-empty catalogue, fall back
        # to the pre-FWHM-cut catalogue so that downstream code always has
        # something to work with.
        if len(sources) == 0 and n0 > 0:
            logger.warning(
                "All sources removed by filtering pipeline; returning unfiltered "
                "catalogue prior to FWHM cut."
            )
            sources = sources_before_fwhm

        sources = sources.reset_index(drop=True)
        elapsed = time.time() - start
        logger.info(
            f"Filtering completed in {elapsed:.3f} s "
            f"(retained {len(sources)} of {n0} initial sources)"
        )
        return sources

    # --- Main Execution ---

    def run(
        self,
        fits_path: str,
        psf_path: Optional[str] = None,
        fits_ref: Optional[str] = None,
        catalog_type: str = "FITS_LDAC",
        gain_key: str = "GAIN",
        satur_key: str = "SATURATE",
        pixel_scale: float = 0,
        seeing_fwhm: float = 1.0,
        back_type: str = "AUTO",
        back_value: float = 0.0,
        back_size: int = 64,
        back_filtersize: int = 3,
        back_pearson: float = 3.5,
        backphoto_type: str = "LOCAL",
        backphoto_thick: int = 24,
        back_filtthresh: float = 0.0,
        use_filt: bool = True,
        detect_thresh: float = 2.0,
        analysis_thresh: float = 2.0,
        detect_minarea: int = 5,
        detect_maxarea: int = 0,
        deblend_nthresh: int = 64,
        deblend_mincont: float = 0.005,
        clean: str = "Y",
        phot_apertures: float = 5.0,
        negative_corr: bool = True,
        checkimage_type: str = "NONE",
        vignet: Optional[Tuple[int, int]] = None,
        stamp_imgsize: Optional[int] = None,
        flags: int = 2,
        mdir: Optional[str] = None,
        verbose_type: str = "QUIET",
        verbose_level: int = 2,
        default_scale: int = 15,
        masked_sources: Optional[Table] = None,
        weight_path: Optional[str] = None,
        use_FWHM: float = 0.0,
        crowded: Optional[bool] = None,
        use_for_matching: bool = False,
    ) -> Tuple[float, Optional[pd.DataFrame], int]:
        """
        Run SExtractor to detect point sources and estimate FWHM.
        If use_FWHM is provided as a positive float, use it to optimize the convolution kernel.
        If weight_path is provided, use it as a weight map for SExtractor.

        Args:
            fits_path (str): Path to the input FITS file.
            psf_path (str, optional): Path to the PSF file. Defaults to None.
            fits_ref (str, optional): Path to the reference FITS file. Defaults to None.
            catalog_type (str, optional): Output catalog type. Defaults to "FITS_LDAC".
            gain_key (str, optional): FITS header key for gain. Defaults to "GAIN".
            satur_key (str, optional): FITS header key for saturation. Defaults to "SATURATE".
            pixel_scale (float, optional): Pixel scale in arcsec/pixel. Defaults to 1.0.
            seeing_fwhm (float, optional): Estimated seeing FWHM in arcsec. Defaults to 2.0.
            back_type (str, optional): Background type (AUTO or MANUAL). Defaults to "AUTO".
            back_value (float, optional): Background value. Defaults to 0.0.
            back_size (int, optional): Background mesh size. Defaults to 64.
            back_filtersize (int, optional): Background filter size. Defaults to 3.
            back_pearson (float, optional): Pearson's factor for background estimation. Defaults to 3.5.
            backphoto_type (str, optional): Background photo type (GLOBAL or LOCAL). Defaults to "LOCAL".
            backphoto_thick (int, optional): Thickness of the background LOCAL annulus. Defaults to 24.
            back_filtthresh (float, optional): Threshold above which the background-map filter operates. Defaults to 0.0.
            use_filt (bool, optional): Use convolution filter. Defaults to True.
            detect_thresh (float, optional): Detection threshold. Defaults to 2.0.
            analysis_thresh (float, optional): Analysis threshold. Defaults to 2.0.
            detect_minarea (int, optional): Minimum detection area. Defaults to 5.
            detect_maxarea (int, optional): Maximum detection area. Defaults to 0.
            deblend_nthresh (int, optional): Deblending threshold. Defaults to 32.
            deblend_mincont (float, optional): Deblending minimum contrast. Defaults to 0.005.
            clean (str, optional): Cleaning option. Defaults to "Y".
            phot_apertures (float, optional): Photometric apertures. Defaults to 5.0.
            negative_corr (bool, optional): Apply negative correlation. Defaults to True.
            checkimage_type (str, optional): Check image type. Defaults to "NONE".
            vignet (tuple, optional): Vignet size. Defaults to None.
            stamp_imgsize (int, optional): Stamp image size. Defaults to None.
            flags (int, optional): Maximum allowed SExtractor flag value. Defaults to 2.
            mdir (str, optional): Directory for temporary files. Defaults to None.
            verbose_type (str, optional): Verbosity type. Defaults to "QUIET".
            verbose_level (int, optional): Verbosity level. Defaults to 2.
            default_scale (int, optional): Default scale for FWHM. Defaults to 15.
            masked_sources (Table, optional): Sources to mask. Defaults to None.
            weight_path (str, optional): Path to the weight map file. Defaults to None.
            use_FWHM (float, optional): FWHM in pixels to use for convolution kernel optimization. Defaults to 0.0.
            crowded (bool, optional): If True, use parameters tuned for crowded fields: tighter deblending,
                smaller back mesh, lower detection threshold, and force `BACKPHOTO_TYPE=GLOBAL`.
                If None, taken from config["photometry"]["crowded_field"] or config["template_subtraction"]["sextractor_crowded"].
            use_for_matching (bool, optional): If True, retain more sources (no NMAX, lower SNR, relaxed FWHM) for matching.

        Returns:
            tuple: (fwhm, sources, scale)
        """
        VALID_CATALOG_TYPES = {"FITS_1.0", "FITS_LDAC", "ASCII", "ASCII_HEAD", "ASCII_SKYCAT", "ASCII_VOTABLE"}
        if catalog_type not in VALID_CATALOG_TYPES:
            raise ValueError(
                f"Invalid catalog_type={catalog_type!r}. "
                f"Must be one of: {sorted(VALID_CATALOG_TYPES)}"
            )
        if get_sextractor_executable() is None:
            raise RuntimeError("SExtractor is not installed.")
        if crowded is None:
            # Crowded if photometry or template_subtraction request it (aligns with SFFT crowded).
            crowded = bool(
                self.config.get("photometry", {}).get("crowded_field", False)
            ) or bool(
                self.config.get("template_subtraction", {}).get(
                    "sextractor_crowded", False
                )
            )
        if crowded:
            logger.info(
                "Using SExtractor crowded-field parameters (forcing BACKPHOTO_TYPE=GLOBAL, "
                "tighter deblending, smaller back mesh)."
            )
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")
        if psf_path is not None:
            psf_path = Path(psf_path)
            if not psf_path.exists():
                raise FileNotFoundError(f"PSF file not found: {psf_path}")
        if weight_path is not None:
            weight_path = Path(weight_path)
            if not weight_path.exists():
                raise FileNotFoundError(f"Weight map file not found: {weight_path}")

        with temp_directory_context(base_dir=mdir) as temp_dir:
            # Resolve binary inside this block so the command always has a defined path
            # (cached by get_sextractor_executable; fail-fast already ran above).
            sextractor_bin = get_sextractor_executable()
            if sextractor_bin is None:
                raise RuntimeError("SExtractor is not installed.")
            mode_label = "crowded" if crowded else "default"
            logger.info(
                log_step(f"SExtractor: {fits_path.name} ({mode_label})")
            )
            # Single open: one mmap/read pass for header + data (bad-region mask).
            with fits.open(fits_path, memmap=True) as hdul:
                hdu0 = hdul[0]
                header = hdu0.header.copy()  # copy header before HDUList closes
                phot_cfg_early = self.config.get("photometry") or {}
                if phot_cfg_early.get("sextractor_reject_constant_regions", True):
                    raw_data = hdu0.data
                    image_data = raw_data.astype(np.float32) if raw_data.dtype.kind != 'f' else raw_data
                else:
                    image_data = None
            gain = float(header.get(gain_key, 1.0))
            saturation = float(header.get(satur_key, 1e7))

            fwhm_for_kernel = float(use_FWHM)
            if not np.isfinite(fwhm_for_kernel) or fwhm_for_kernel <= 0:
                # Use known image FWHM when available to avoid a generic kernel.
                for v in (
                    header.get("FWHM"),
                    header.get("fwhm"),
                    self.config.get("fwhm"),
                ):
                    try:
                        vv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(vv) and vv > 0:
                        fwhm_for_kernel = vv
                        break
            if fwhm_for_kernel > 0:
                logger.info(
                    "Using FWHM = %.1f pixels to construct the SExtractor convolution kernel",
                    fwhm_for_kernel,
                )
                phot_apertures = 1.7 * fwhm_for_kernel

            conv_path = (
                self._create_conv_file(temp_dir, fwhm_pixels=fwhm_for_kernel)
                if use_filt
                else None
            )
            nnw_path = self._create_nnw_file(temp_dir)

            # SExtractor parameters
            params = [
                "XWIN_IMAGE",
                "YWIN_IMAGE",
                "FLUX_AUTO",
                "FLUXERR_AUTO",
                "FWHM_IMAGE",
                "ELLIPTICITY",
                "SNR_WIN",
                "FLAGS",
                "CLASS_STAR",
                "FLUX_MAX",
                "A_IMAGE",
                "B_IMAGE",
                "THETA_IMAGE",
                "MU_MAX",
                "ISOAREA_IMAGE",
                "FLUX_RADIUS",
            ]
            if vignet is not None:
                params.append(f"VIGNET({vignet[0]},{vignet[1]})")
            param_path = self._create_param_file(temp_dir, params)

            # SExtractor configuration
            config_dict = {
                "CATALOG_TYPE": catalog_type,
                "VERBOSE_TYPE": verbose_type,
                "NTHREADS": str(self.n_threads),
                "GAIN": str(gain),
                "SATUR_LEVEL": str(saturation),
                "PIXEL_SCALE": str(pixel_scale),
                "BACK_TYPE": back_type,
                "BACK_VALUE": str(back_value),
                "BACK_SIZE": str(back_size),
                "BACK_FILTERSIZE": str(back_filtersize),
                "BACK_PEARSON": str(back_pearson),
                "BACKPHOTO_TYPE": backphoto_type,
                "BACKPHOTO_THICK": str(backphoto_thick),
                "BACK_FILTTHRESH": str(back_filtthresh),
                "DETECT_THRESH": str(detect_thresh),
                "ANALYSIS_THRESH": str(analysis_thresh),
                "DETECT_MINAREA": str(detect_minarea),
                "DETECT_MAXAREA": str(detect_maxarea),
                "DEBLEND_NTHRESH": str(deblend_nthresh),
                "DEBLEND_MINCONT": str(deblend_mincont),
                "CHECKIMAGE_TYPE": checkimage_type,
                "SEEING_FWHM": str(seeing_fwhm),
                "PHOT_APERTURES": str(phot_apertures),
                "FILTER": "Y" if use_filt else "N",
                "STARNNW_NAME": str(nnw_path),
                "CLEAN": clean,
                "CLEAN_PARAM": "1.0",
                "PHOT_AUTOPARAMS": "2.5,3.5",
                "MAG_ZEROPOINT": "0.0",
            }
            if crowded:
                # Crowded-field tune: force GLOBAL background photos type (per your requirement)
                # and align thresholds/deblending with SFFT defaults to reduce biased detections.
                config_dict.update(
                    {
                        "BACKPHOTO_TYPE": "GLOBAL",
                        "BACK_SIZE": "64",
                        "BACK_FILTERSIZE": "3",
                        "DETECT_THRESH": "5.0",
                        "ANALYSIS_THRESH": "5.0",
                        "DETECT_MINAREA": "5",
                        "DEBLEND_NTHRESH": "64",
                        "DEBLEND_MINCONT": "0.005",
                    }
                )

            # Add weight map configuration if provided
            if weight_path is not None:
                config_dict.update(
                    {
                        "WEIGHT_TYPE": "MAP_RMS",
                        "WEIGHT_IMAGE": str(weight_path),
                        # "WEIGHT_GAIN": "Y",
                        # "RESCALE_WEIGHTS": "Y",
                        # "WEIGHT_THRESH": "0.0",
                    }
                )

            if conv_path is not None:
                config_dict["FILTER_NAME"] = str(conv_path)
            if psf_path is not None:
                config_dict["PSF_NAME"] = str(psf_path)

            config_path = self._create_config_file(temp_dir, config_dict)

            # Output paths
            base_name = fits_path.stem
            catalog_path = temp_dir / f"{base_name}_PYSEx_CAT.fits"

            # Run SExtractor
            cmd = [
                sextractor_bin,
                str(fits_path),
                "-c",
                str(config_path),
                "-CATALOG_NAME",
                str(catalog_path),
                "-PARAMETERS_NAME",
                str(param_path),
            ]
            if fits_ref is not None:
                # SExtractor dual-image mode: first argument must be "ref.fits,sci.fits" as one token
                cmd[1] = f"{fits_ref},{fits_path}"
            if checkimage_type != "NONE":
                check_image_path = temp_dir / f"{base_name}_PYSEx_CHECK.fits"
                cmd.extend(["-CHECKIMAGE_NAME", str(check_image_path)])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"SExtractor failed: {result.stderr}")

            # Process output
            if not catalog_path.exists():
                raise FileNotFoundError("SExtractor did not produce an output catalog.")
            tbhdu = 2 if catalog_type == "FITS_LDAC" else 1
            sources = Table.read(catalog_path, hdu=tbhdu, format="fits")
            if len(sources) == 0:
                logger.warning("No sources detected by SExtractor.")
                return 0.0, None, default_scale

            # Rename columns for compatibility
            newcols = [
                "x_pix",
                "y_pix",
                "flux_AP",
                "flux_AP_err",
                "fwhm",
                "roundness",
                "snr",
                "flags",
                "class_star",
                "peak_flux",
                "a",
                "b",
                "theta",
                "mu_max",
                "area",
                "flux_radius",
            ]
            sources = sources.to_pandas()
            # Handle optional VIGNET column before renaming fixed columns
            if vignet is not None and len(sources.columns) == len(newcols) + 1:
                # VIGNET is always appended last by SExtractor
                vignet_data = sources.iloc[:, -1]
                sources = sources.iloc[:, :len(newcols)]
                sources.columns = newcols
                sources["vignet"] = vignet_data
            elif len(sources.columns) == len(newcols):
                sources.columns = newcols
            else:
                logger.warning(
                    "SExtractor output has %d columns but expected %d; "
                    "skipping column rename to avoid data corruption.",
                    len(sources.columns),
                    len(newcols),
                )

            # SExtractor uses 1-based pixel coordinates; convert to 0-based
            sources["x_pix"] -= 1
            sources["y_pix"] -= 1

            # Initial filtering (keep more sources when use_for_matching or crowded)
            initial_count = len(sources)
            if use_for_matching or crowded:
                nmax = None
                effective_snr_limit = self.config.get("photometry", {}).get(
                    "sextractor_snr_min_matching", 2.0
                )
                relaxed_cuts = use_for_matching
            else:
                nmax = self.config.get("photometry", {}).get("sextractor_nmax", 1000)
                effective_snr_limit = 3.0  # default SNR limit for standard mode
                relaxed_cuts = False
            # Optional mask for chip gaps / flat borders (can remove real stars on
            # very smooth backgrounds - disable via photometry config if needed).
            bad_region_mask = None
            phot_cfg_run = self.config.get("photometry") or {}
            if phot_cfg_run.get("sextractor_reject_constant_regions", True):
                # Note: _constant_region_mask function not implemented in background.py
                # This feature is disabled until the function is implemented.
                bad_region_mask = None
            else:
                logger.info(
                    "photometry.sextractor_reject_constant_regions=False: skipping "
                    "constant-region source rejection"
                )

            sources = self.filter_sextractor_sources(
                sources,
                header,
                masked_sources=masked_sources,
                NMAX=nmax,
                snr_limit=effective_snr_limit,
                relaxed_cuts=relaxed_cuts,
                bad_region_mask=bad_region_mask,
            )

            # Final count and validation
            final_count = len(sources)
            logger.info(f"Filtered from {initial_count} to {final_count} point sources")
            if final_count == 0:
                logger.warning("[ERROR] All sources filtered out")
                return 0.0, None, default_scale

            # Calculate FWHM and scale
            fwhm_values = sources["fwhm"].values
            fwhm = self.calculate_robust_fwhm(fwhm_values)
            scale_multiplier = scale_multiplier_from_config(self.config)
            # Legacy behavior: `default_scale` is a hard floor when the FWHM-based
            # scale is too small.
            raw_scale = max(float(scale_multiplier) * float(fwhm), float(default_scale))
            # Clamp to configured bounds (and ensure >= 1).
            scale = clamp_scale_from_config(self.config, raw_scale)  # returns int per annotation
            logger.info(
                "Found %d point sources, robust FWHM %.2f px; scale = %.1f "
                "(FWHM x %.2f from source_detection / config)",
                final_count,
                fwhm,
                scale,
                scale_multiplier,
            )
            return fwhm, sources, scale
