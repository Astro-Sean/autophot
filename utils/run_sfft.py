#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFFT image subtraction. Limits BLAS/OpenMP threads at startup to avoid
'Thread creation failed: Resource temporarily unavailable' when run as a
subprocess or on HPC with strict process/thread limits.

Note: The resource_tracker "leaked semaphore" warning at shutdown can come from
the SFFT/numpy/NumExpr stack (internal use of multiprocessing primitives). This
script does not use multiprocessing; run with OMP_NUM_THREADS=1 and the caller
should use nCPU=1 when SFFT is enabled to minimise leaks and thread exhaustion.
"""

import argparse
import os
import ast
import logging
import sys
import time
import warnings

# Add parent directory to path to import functions.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import ColoredLevelFormatter, LogMessageNormalizeFilter, set_size, safe_fits_write, border_msg

# Limit BLAS/OpenMP threads before any scientific imports (avoids libgomp
# "Resource temporarily unavailable" when this script is run as a subprocess
# or alongside other parallel jobs).
for _env in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, "1")
# Thread-count limits are applied via env vars above; affinity pinning is left
# to the scheduler / HPC job manager.

import numpy as np
import pandas as pd
from astropy.io import fits
import sfft  # Import the sfft package to check its version
from sfft.EasySparsePacket import Easy_SparsePacket
from sfft.EasyCrowdedPacket import Easy_CrowdedPacket
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from typing import Optional, Tuple

# Simple logging function for early use (before main logger setup)
def _early_log(msg: str) -> None:
    print(msg)

# NumPy 2.0 compatibility: handle removed functions
# numpy.in1d was removed in NumPy 2.0, but SFFT still uses it
if not hasattr(np, 'in1d'):
    # Add compatibility shim for numpy.in1d (removed in NumPy 2.0)
    def in1d(ar1, ar2, assume_unique=False, invert=False):
        """Compatibility shim for numpy.in1d (removed in NumPy 2.0)."""
        ar1 = np.asarray(ar1)
        ar2 = np.asarray(ar2)
        if assume_unique:
            ar2 = np.unique(ar2)
        mask = np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)
        return mask
    np.in1d = in1d
    _early_log("Added numpy.in1d compatibility shim for NumPy 2.x")

# NumPy 2.0 compatibility: handle numpy.char removal
# Most modern code doesn't use numpy.char, but we provide fallback if needed
try:
    import numpy.char as np_char
except (ImportError, AttributeError):
    np_char = None

# Import new SFFT features (v1.5.0+)
try:
    from sfft.BSplineSFFT import BSpline_Packet, BSpline_MatchingKernel, BSpline_DeCorrelation
    _HAS_BSPLINE = True
except ImportError:
    _HAS_BSPLINE = False

try:
    from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator as DeCorrelationCalculator
    _HAS_DECORRELATION = True
except ImportError:
    _HAS_DECORRELATION = False


def _odd(n: int) -> int:
    """Return n unchanged (as int).

    Historical note: this function used to round up to the next odd integer,
    based on the incorrect assumption that KerHW must be odd.  SFFT constructs
    the kernel as (2*KerHW + 1) × (2*KerHW + 1), so the *full kernel* is
    always odd-sized regardless of whether KerHW is even or odd.  Forcing KerHW
    to be odd (e.g. 4 → 5) inflates the kernel from 9×9 to 11×11 for no reason.
    The function is kept for call-site compatibility but now just casts to int.
    """
    return int(n)


def run_sfft() -> Optional[int]:
    """
    SFFT (Sparse Field Flux Transport) image subtraction pipeline.
    Performs image subtraction between a science image and a reference image using
    the sfft backends. Supports crowded-field mode, masking, and diagnostic plotting.
    Returns 1 on success, None on fatal error.
    """

    # --- Setup Logging ---
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        # Add ANSI highlighting for WARN/ERROR/DEBUG in the console.
        root_logger = logging.getLogger()
        _normalize_filter = LogMessageNormalizeFilter(width=120)
        for h in root_logger.handlers:
            h.addFilter(_normalize_filter)
            h.setFormatter(
                ColoredLevelFormatter(
                    fmt="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S",
                    use_color=True,
                )
            )
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")
    t0_total = time.time()

    def log_info(msg: str) -> None:
        try:
            logger.info(msg)
        except Exception:
            print(msg)  # Fallback if logging not configured

    def _to_dataframe(obj) -> pd.DataFrame:
        """Best-effort conversion of an SFFT table-like object to DataFrame."""
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if hasattr(obj, "to_pandas"):
            try:
                return obj.to_pandas()
            except Exception:
                pass
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()

    def _pick_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Find plausible x/y coordinate columns in an SFFT source catalog."""
        candidates = [
            ("X_IMAGE_REF_SCI_MEAN", "Y_IMAGE_REF_SCI_MEAN"),
            ("x_center", "y_center"),
            ("x", "y"),
            ("x_pix", "y_pix"),
            ("X_IMAGE", "Y_IMAGE"),
        ]
        for xcol, ycol in candidates:
            if xcol in df.columns and ycol in df.columns:
                return xcol, ycol
        return None, None

    def _flag_mask_from_columns(
        df: pd.DataFrame, include_tokens: Tuple[str, ...], exclude_tokens: Tuple[str, ...] = ()
    ) -> np.ndarray:
        """
        Build a boolean mask from likely flag columns.

        Picks the matching flag-like column with the largest positive count.
        """
        if df.empty:
            return np.zeros(0, dtype=bool)
        best_mask = np.zeros(len(df), dtype=bool)
        best_count = 0
        for col in df.columns:
            c_low = str(col).strip().lower()
            if not all(tok in c_low for tok in include_tokens):
                continue
            if any(tok in c_low for tok in exclude_tokens):
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().sum() == 0:
                continue
            vals = series.dropna().values
            uniq = np.unique(vals)
            # Keep to flag-like columns only (bool / 0-1 style).
            if len(uniq) > 6:
                continue
            if not np.all(np.isfinite(uniq)):
                continue
            col_mask = np.asarray(series.fillna(0).values > 0, bool)
            count = int(np.count_nonzero(col_mask))
            if count > best_count:
                best_count = count
                best_mask = col_mask
        return best_mask

    def _coords_from_prep_data(prep_data_obj) -> np.ndarray:
        """Extract post-anomaly coordinates directly from prep_data keys when available."""
        if not isinstance(prep_data_obj, dict):
            return np.empty((0, 2), float)
        for key, value in prep_data_obj.items():
            key_l = str(key).lower()
            if "post" not in key_l or "anom" not in key_l:
                continue
            try:
                arr = np.asarray(value, dtype=float)
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.size > 0:
                return arr[:, :2]
        return np.empty((0, 2), float)

    def _write_post_anomaly_sources_csv(
        prep_data_obj, matched_df: Optional[pd.DataFrame], out_dir_path: str, out_base_name: str
    ) -> int:
        """
        Export post-anomaly source coordinates for optional retry feedback.

        Output: SFFT_PostAnomaly_Sources_<base>.csv with x/y columns in SCI/REF mean frame.
        """
        out_csv = os.path.join(out_dir_path, f"SFFT_PostAnomaly_Sources_{out_base_name}.csv")
        anomaly_xy = _coords_from_prep_data(prep_data_obj)

        # Fallback: derive from SubSource catalog flags.
        if anomaly_xy.size == 0:
            cat_df = pd.DataFrame()
            if isinstance(prep_data_obj, dict) and "SExCatalog-SubSource" in prep_data_obj:
                cat_df = _to_dataframe(prep_data_obj.get("SExCatalog-SubSource"))
            if cat_df.empty and isinstance(matched_df, pd.DataFrame):
                cat_df = matched_df.copy()
            if not cat_df.empty:
                xcol, ycol = _pick_xy_columns(cat_df)
                if xcol and ycol:
                    pac_mask = _flag_mask_from_columns(
                        cat_df,
                        include_tokens=("post", "anom"),
                        exclude_tokens=("prior",),
                    )
                    if pac_mask.size == len(cat_df) and np.any(pac_mask):
                        xy = cat_df.loc[pac_mask, [xcol, ycol]].copy()
                        xy = xy.apply(pd.to_numeric, errors="coerce").dropna()
                        anomaly_xy = xy.to_numpy(float)

        if anomaly_xy.size == 0:
            # Keep state explicit: remove stale file from prior runs.
            try:
                if os.path.exists(out_csv):
                    os.remove(out_csv)
            except Exception:
                pass
            return 0

        # Deduplicate and write standardized columns.
        df_anom = pd.DataFrame(anomaly_xy, columns=["X_IMAGE_REF_SCI_MEAN", "Y_IMAGE_REF_SCI_MEAN"])
        df_anom = df_anom.replace([np.inf, -np.inf], np.nan).dropna()
        if df_anom.empty:
            return 0
        df_anom = df_anom.drop_duplicates().reset_index(drop=True)
        df_anom.to_csv(out_csv, index=False, float_format="%.6f")
        log_info(
            f"Exported {len(df_anom)} post-anomaly sources for feedback: {out_csv}"
        )
        return int(len(df_anom))

    # --- Print SFFT Version ---
    log_info(f"Using SFFT version: {sfft.__version__}")

    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="SFFT image subtraction.")
    parser.add_argument("-sci", type=str, required=True, help="Science FITS file path.")
    parser.add_argument(
        "-ref", type=str, required=True, help="Reference FITS file path."
    )
    parser.add_argument(
        "-diff", type=str, default=None, help="Output difference FITS path."
    )
    parser.add_argument("-mask", type=str, default=None, help="Boolean mask FITS path.")
    parser.add_argument(
        "-crowded", action="store_true", help="Use crowded-field solver."
    )

    # Match HOTPANTS: default 0 (constant kernel). Pipeline passes this from config.
    parser.add_argument(
        "-kernel_order",
        type=int,
        default=0,
        help="Kernel polynomial order (0 = constant, like HOTPANTS -ko).",
    )
    parser.add_argument(
        "-bg_order",
        type=int,
        default=0,
        help="Background spatial polynomial order (0=constant). Science and reference are assumed background-subtracted (pipeline subtracts both).",
    )

    parser.add_argument(
        "-kernel_half_width", type=float, default=0, help="Kernel half width."
    )
    parser.add_argument(
        "-kernel_hw_fwhm_multiplier",
        type=float,
        default=2.0,
        help="Multiplier for physics-based kernel sizing: kernel_hw = ceil(mult * sqrt(FWHM_broad^2 - FWHM_narrow^2)). "
             "Higher values give larger kernels (more fitting freedom, slower). Default 2.0.",
    )
    parser.add_argument(
        "-masked_sources",
        type=str,
        default="[]",
        help='List of [x, y] pairs to ban, e.g., "[[12.3, 45.6], [78.9, 10.2]]".',
    )
    parser.add_argument(
        "-matching_sources",
        type=str,
        default="[]",
        help="List of [x, y] pairs to preselect, same format as above.",
    )
    parser.add_argument(
        "-detect_thresh",
        type=float,
        default=None,
        help="SExtractor DETECT_THRESH (sigma). If None, uses 1.5 for sparse, 3.0 for crowded.",
    )
    parser.add_argument("-plot", action="store_true", help="Generate diagnostic plots.")
    parser.add_argument(
        "-forceconv",
        type=str,
        default="REF",
        help="Deprecated: always REF. Reference image is always convolved (DIFF = SCI - conv(REF)).",
    )

    # Pass gain/saturate values so SFFT does not depend on headers (avoids KeyError when SATURATE missing).
    parser.add_argument(
        "-gain_sci",
        type=float,
        default=None,
        help="Science image gain (e/ADU). If set, written to FITS before SFFT.",
    )
    parser.add_argument(
        "-gain_ref", type=float, default=None, help="Reference image gain (e/ADU)."
    )
    parser.add_argument(
        "-saturate_sci",
        type=float,
        default=None,
        help="Science saturation (ADU). Use a large value (e.g. 1e30) for no limit.",
    )
    parser.add_argument(
        "-saturate_ref", type=float, default=None, help="Reference saturation (ADU)."
    )

    # SExtractor background mesh: smaller BACK_SIZE => finer mesh (better for gradients/hosts),
    # but can overfit noise if too small. Typical values: 32, 64, 128.
    parser.add_argument(
        "-back_size",
        type=int,
        default=64,
        help="SExtractor BACK_SIZE (mesh cell size, px).",
    )
    parser.add_argument(
        "-back_filtersize",
        type=int,
        default=3,
        help="SExtractor BACK_FILTERSIZE (median filter size, mesh cells).",
    )
    parser.add_argument(
        "-backphototype",
        type=str,
        default="LOCAL",
        help="SExtractor BACKPHOTO_TYPE for source photometry background: LOCAL or GLOBAL.",
    )
    parser.add_argument(
        "-constphotratio",
        type=str,
        default="false",
        help="SFFT ConstPhotRatio: 'true' restricts kernel sum (default SFFT behaviour), 'false' fits flux scaling polynomial (default, like HOTPANTS).",
    )
    parser.add_argument(
        "-only_flags",
        type=str,
        default="0,1,2",
        help="Comma-separated SExtractor FLAGS values to keep (e.g. '0,1'); use 'none' to disable.",
    )
    parser.add_argument(
        "-cvrej_magd_thresh",
        type=float,
        default=0.12,
        help="SFFT coarse variable-rejection magnitude-difference threshold.",
    )
    parser.add_argument(
        "-evrej_ratio_thresh",
        type=float,
        default=6.0,
        help="SFFT elaborate variable-rejection ratio threshold.",
    )
    parser.add_argument(
        "-evrej_safe_magdev",
        type=float,
        default=0.04,
        help="SFFT elaborate variable-rejection safe magnitude-deviation threshold.",
    )
    parser.add_argument(
        "-pac_ratio_thresh",
        type=float,
        default=2.8,
        help="SFFT post-anomaly-check ratio threshold.",
    )
    parser.add_argument(
        "-allow_crowded_bg_order_override",
        type=str,
        default="false",
        help="If true and crowded mode with bg_order=0, override to bg_order=2.",
    )
    parser.add_argument(
        "-star_ext_iter",
        type=int,
        default=1,
        help="SFFT StarExt_iter (source extension iterations). If None, uses defaults (4 for sparse, 2 for crowded). Higher values (6-8) improve deblending but are slower.",
    )
    parser.add_argument(
        "-use_bspline_kernel",
        type=str,
        default="true",
        help="Use B-Spline kernel matching (SFFT v1.5.0+). 'true' enables B-Spline kernel for complex PSF variations.",
    )
    parser.add_argument(
        "-decorrelate_noise",
        type=str,
        default="true",
        help="Apply noise decorrelation to difference image (SFFT v1.5.0+). 'true' whitens correlated noise.",
    )
    parser.add_argument(
        "-save_original_diff",
        type=str,
        default="true",
        help="Save original (non-decorrelated) difference image for photometry. 'true' saves both decorrelated and original versions.",
    )
    parser.add_argument(
        "-kernel_hw_min",
        type=int,
        default=3,
        help="Minimum kernel half-width in pixels.",
    )
    parser.add_argument(
        "-kernel_hw_max",
        type=int,
        default=50,
        help="Maximum kernel half-width in pixels.",
    )
    parser.add_argument(
        "-min_prior_sources",
        type=int,
        default=3,
        help="Minimum number of prior sources required to use them for kernel fitting. If fewer sources are provided, SFFT will perform its own source matching.",
    )
    parser.add_argument(
        "-coarse_var_rejection",
        type=str,
        default="false",
        help="Enable SFFT coarse variable-star rejection (true/false).",
    )
    parser.add_argument(
        "-elabo_var_rejection",
        type=str,
        default="false",
        help="Enable SFFT elaborate variable-star rejection (true/false).",
    )
    args = parser.parse_args()

    # --- Parse Coordinate Lists ---
    def parse_xy_list(s: str) -> Optional[np.ndarray]:
        """
        Parse a string representing a list of [x, y] pairs.

        Accepted formats:
          - "[[x1, y1], [x2, y2]]"
          - "[x1, y1]" (single pair)
          - "[x1, y1, x2, y2]" (flat list, reshaped to Nx2)
        """
        s = (s or "").strip()
        if s in ("", "[]", "None"):
            return None
        try:
            obj = ast.literal_eval(s)
            coords_array = np.array(obj, dtype=float)
            if coords_array.ndim == 1:
                if coords_array.size == 2:
                    coords_array = coords_array.reshape(1, 2)
                elif coords_array.size % 2 == 0:
                    coords_array = coords_array.reshape(-1, 2)
                else:
                    raise ValueError(
                        "Must contain an even number of values to form [x, y] pairs."
                    )
            if coords_array.ndim != 2 or coords_array.shape[1] != 2:
                raise ValueError("Must be a list of [x, y] pairs.")
            if not np.all(np.isfinite(coords_array)):
                n_bad = int(np.sum(~np.isfinite(coords_array).all(axis=1)))
                log_info(
                    f"Warning: {n_bad} coordinate pairs contain non-finite values in '{s}'; "
                    "they will be removed during validation."
                )
                # Don't reject here; _sanitize_xy_sources will clean them
            return coords_array
        except Exception as e:
            log_info(f"Warning: Could not parse list '{s}': {e}. Ignoring.")
            return None

    masked_sources = parse_xy_list(args.masked_sources)
    matching_sources = parse_xy_list(args.matching_sources)
    # matching_sources = None
    if masked_sources is not None:
        log_info(f"Masked sources: {masked_sources.shape[0]}")
    else:
        log_info("No masked sources provided.")

    if matching_sources is not None and matching_sources.shape[0] > 0:
        log_info(f"Matching sources (prior): {matching_sources.shape[0]}")
    else:
        matching_sources = None
        log_info("No prior matching sources; SFFT will perform source matching.")

    # --- Validate Inputs ---
    FITS_SCI = args.sci
    FITS_REF = args.ref

    for path, name in [(FITS_SCI, "Science"), (FITS_REF, "Reference")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} FITS not found: {path}")
        try:
            with fits.open(path) as hdul:
                if len(hdul) == 0:
                    raise ValueError(f"{name} FITS has no HDUs.")
        except Exception as e:
            raise ValueError(f"Invalid {name} FITS: {e}")

    # --- Output Paths ---
    out_dir = os.path.dirname(os.path.abspath(FITS_SCI)) or "."
    # Standardize to the same "base" used by main.py (remove suffixes, normalize
    # punctuation), so output file names include the FITS filename stem.
    fits_sci_stem = os.path.splitext(os.path.basename(FITS_SCI))[0]
    out_base = (
        fits_sci_stem.replace(" ", "_")
        .replace(".", "_")
        .replace("_APT", "")
        .replace("_ERROR", "")
    )
    FITS_DIFF = args.diff or os.path.join(out_dir, f"diff_{os.path.basename(FITS_SCI)}")

    # --- Load Headers (Once) ---
    # Use float64 throughout: SFFT internals operate in float64 and mixing float32
    # would cause a silent precision downgrade when data is written back to FITS.
    def get_fits_info(fits_path: str) -> Tuple[fits.Header, np.ndarray]:
        with fits.open(fits_path, memmap=False) as hdul:
            header = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float64)
            return header, data

    hdr_sci, data_sci = get_fits_info(FITS_SCI)
    hdr_ref, data_ref = get_fits_info(FITS_REF)

    # Build a combined invalid-pixel mask from both inputs so that no-data regions
    # (NaN or exactly zero, e.g. from SWarp padding) in either image are masked in
    # BOTH images before passing to SFFT.  This prevents SFFT from fitting the
    # convolution kernel across image-boundary padding or chip gaps where only one
    # image has valid data.
    invalid_sci = ~np.isfinite(data_sci) | (data_sci == 0)
    invalid_ref = ~np.isfinite(data_ref) | (data_ref == 0)
    combined_invalid_mask = invalid_sci | invalid_ref
    if np.any(combined_invalid_mask):
        log_info(
            f"Combined invalid mask: {int(np.count_nonzero(combined_invalid_mask))} pixels "
            "will be forced to NaN in both images before SFFT."
        )

    # Create temporary FITS files with synchronized NaN masks for SFFT input.
    # We do NOT mutate the original science / template files (they may be reused on
    # retry paths such as post-anomaly feedback).
    _temp_sci_path = os.path.join(out_dir, f"._sfft_sci_sync_{os.getpid()}.fits")
    _temp_ref_path = os.path.join(out_dir, f"._sfft_ref_sync_{os.getpid()}.fits")
    _temp_paths_created = []
    try:
        if np.any(combined_invalid_mask):
            data_sci_sync = data_sci.copy()
            data_ref_sync = data_ref.copy()
            data_sci_sync[combined_invalid_mask] = np.nan
            data_ref_sync[combined_invalid_mask] = np.nan
            safe_fits_write(_temp_sci_path, data_sci_sync, hdr_sci, overwrite=True)
            safe_fits_write(_temp_ref_path, data_ref_sync, hdr_ref, overwrite=True)
            _temp_paths_created = [_temp_sci_path, _temp_ref_path]
            FITS_SCI = _temp_sci_path
            FITS_REF = _temp_ref_path
            log_info(f"Synchronized NaN masks written to temporary FITS for SFFT input.")
    except Exception as e:
        log_info(f"Warning: Failed to write synchronized-mask temp FITS: {e}. Using original files.")
        # If temp write fails, fall back to original paths (FITS_SCI/FITS_REF unchanged)

    def _sanitize_xy_sources(
        xy: Optional[np.ndarray], label: str, width: int, height: int
    ) -> Optional[np.ndarray]:
        """
        Keep source lists permissive but valid:
        - remove non-finite coordinates
        - remove out-of-image coordinates
        - de-duplicate nearly identical points
        """
        if xy is None:
            return None
        arr = np.asarray(xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            return None

        n_in = int(arr.shape[0])
        finite_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        arr = arr[finite_mask]

        if arr.shape[0] > 0:
            in_bounds = (
                (arr[:, 0] >= 0.0)
                & (arr[:, 0] < float(width))
                & (arr[:, 1] >= 0.0)
                & (arr[:, 1] < float(height))
            )
            arr = arr[in_bounds]

        if arr.shape[0] > 0:
            rounded = np.round(arr, decimals=3)
            _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
            arr = arr[np.sort(uniq_idx)]

        n_out = int(arr.shape[0])
        n_drop = n_in - n_out
        if n_drop > 0:
            log_info(
                f"{label}: kept {n_out}/{n_in} sources after basic validity checks "
                f"(dropped {n_drop} unusable entries)."
            )
        else:
            log_info(f"{label}: kept all {n_out} sources.")
        return arr if n_out > 0 else None

    ny, nx = data_sci.shape
    matching_sources = _sanitize_xy_sources(matching_sources, "Matching sources", nx, ny)
    masked_sources = _sanitize_xy_sources(masked_sources, "Masked sources", nx, ny)
    
    # Improve prior source validation: require minimum sources for reliable kernel fitting
    MIN_PRIOR_SOURCES = int(getattr(args, "min_prior_sources", 3) or 3)
    if matching_sources is not None and len(matching_sources) < MIN_PRIOR_SOURCES:
        log_info(
            f"Warning: Only {len(matching_sources)} prior sources provided "
            f"(minimum {MIN_PRIOR_SOURCES} required for reliable kernel fitting). "
            "Letting SFFT perform source matching instead."
        )
        matching_sources = None
    elif matching_sources is None:
        log_info("No valid prior matching sources after checks; SFFT will perform source matching.")

    # --- Ensure GAIN and SATURATE in FITS (pass values, not keywords) ---
    # Write values into headers so SFFT/MeLOn find the keywords (avoids KeyError when SATURATE missing).
    SATURATE_FALLBACK = 1e30  # finite value for "no saturation" (FITS cannot store inf)

    def _ensure_gain_saturate(
        fits_path: str, gain: Optional[float], saturate: Optional[float], label: str
    ) -> None:
        if gain is None and saturate is None:
            return
        try:
            with fits.open(fits_path, mode="update") as hdul:
                h = hdul[0].header
                if gain is not None:
                    gval = float(gain)
                    h["GAIN"] = gval
                    h.comments["GAIN"] = "Gain (e/ADU) provided by autophot for SFFT"
                    log_info(f"{label}: set GAIN = {gval}")
                if saturate is not None:
                    sval = float(saturate) if np.isfinite(saturate) else SATURATE_FALLBACK
                    h["SATURATE"] = sval
                    h.comments["SATURATE"] = (
                        "Effective saturation (ADU) after autophot background handling for SFFT"
                    )
                    log_info(f"{label}: set SATURATE = {sval}")
        except Exception as e:
            log_info(
                f"Warning: Could not write GAIN/SATURATE to {label} FITS header: {e}. "
                "SFFT will rely on header values already present."
            )

    def _float_or_default(input_value, default: float) -> float:
        if input_value is None:
            return default
        try:
            v = float(input_value)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    gain_sci = (
        args.gain_sci
        if args.gain_sci is not None
        else _float_or_default(hdr_sci.get("GAIN", hdr_sci.get("gain")), 1.0)
    )
    gain_ref = (
        args.gain_ref
        if args.gain_ref is not None
        else _float_or_default(hdr_ref.get("GAIN", hdr_ref.get("gain")), 1.0)
    )
    sat_sci_raw = (
        args.saturate_sci
        if args.saturate_sci is not None
        else hdr_sci.get("SATURATE", hdr_sci.get("saturate"))
    )
    sat_ref_raw = (
        args.saturate_ref
        if args.saturate_ref is not None
        else hdr_ref.get("SATURATE", hdr_ref.get("saturate"))
    )
    sat_sci = _float_or_default(sat_sci_raw, SATURATE_FALLBACK)
    sat_ref = _float_or_default(sat_ref_raw, SATURATE_FALLBACK)

    _ensure_gain_saturate(FITS_SCI, gain_sci, sat_sci, "Science")
    _ensure_gain_saturate(FITS_REF, gain_ref, sat_ref, "Reference")

    # --- FWHM and Gain ---
    template_fwhm = float(hdr_ref.get("FWHM", 3.0))
    science_fwhm = float(hdr_sci.get("FWHM", 3.0))
    log_info(
        f"Science FWHM: {science_fwhm:.1f} pixels | Template FWHM: {template_fwhm:.1f} pixels"
    )

    # --- Kernel and Detection Parameters ---
    fwhm_min = min(template_fwhm, science_fwhm)
    fwhm_max = max(template_fwhm, science_fwhm)
    FWHM = max(np.ceil(template_fwhm), np.ceil(science_fwhm))

    psf_area_min = np.pi * (fwhm_min) ** 2
    # SFFT defaults use DETECT_MINAREA=5; keep at least that to avoid spurious
    # source selection that can bias the scale/background fit.
    detect_minarea = max(3, int(np.ceil(psf_area_min * 0.5)))
    detect_maxarea = 0

    # ---------------------------------------------------------------------------
    # Kernel half-width sizing
    # ---------------------------------------------------------------------------
    # Literature basis:
    #
    #   SFFT (Hu et al. 2022, ApJ 936):
    #       KerHW = int(KerHWRatio * max(FWHM_REF, FWHM_SCI)), KerHWRatio in [1.5, 2.5].
    #       The kernel must enclose the broader PSF, not just the difference kernel.
    #       SFFT source comment: "Ratio of kernel half-width to FWHM (typically 1.5–2.5)."
    #
    #   Alard & Lupton 1998 / Astier (private comm. cited in Miller et al. 2008):
    #       sigma_kernel = 0.5 * |sigma_sci - sigma_ref|  (sigma = FWHM/2.355)
    #       L = 8 * sigma_kernel + 5  =>  KerHW = (L-1)/2 = 4*sigma_kernel + 2
    #       = 4*(FWHM_broad-FWHM_narrow)/(2*2.355) + 2  ≈ 0.85*FWHM_diff + 2
    #       This sets a *minimum* — the kernel must contain the PSF mismatch lobe.
    #
    #   LSST ip_diffim (PsfMatchConfigAL):
    #       kernel_size = kernelSizeFwhmScaling * sigma_largest_Gaussian_basis
    #       kernelSizeFwhmScaling = 6.0 (default); sigma_basis ≈ FWHM_broad/2.355
    #       => kernel_size ≈ 6 * FWHM_broad/2.355 ≈ 2.55 * FWHM_broad
    #       KerHW = (kernel_size - 1)/2 ≈ 1.28 * FWHM_broad  (min 10, max 17)
    #
    #   Israel 2007 (Astron. Nachr.):
    #       Kernel half-width must be >= 3*sigma_conv to enclose 99.7% of the
    #       matching Gaussian; undersizing causes structured residuals.
    #       sigma_conv = sqrt(sigma_broad^2 - sigma_narrow^2) = fwhm_conv/2.355
    #       => minimum contribution: ceil(3 * fwhm_conv / 2.355)
    #
    # Design decision — two independent lower bounds, take the larger:
    #
    #   hw_broad:  multiplier * FWHM_broad
    #              The kernel must contain the broader PSF's support. This is the
    #              primary term from SFFT (ratio 2.0) and LSST (effective ratio ~1.28).
    #              We use a user-configurable multiplier (default 2.0) applied to FWHM_broad.
    #
    #   hw_conv:   ceil(3 * sigma_conv) = ceil(3 * fwhm_conv / 2.355)
    #              Ensures the PSF-difference lobe is fully enclosed (Israel 2007).
    #              Active only when PSF sizes differ substantially; for identical PSFs
    #              fwhm_conv -> 0 so this term vanishes.
    #
    # Undersampled images (FWHM_broad < 2 px): PSF wings extend far relative to
    # pixel size. Boost the broad-PSF multiplier (not the difference term) by 1.5×
    # to capture ringing artefacts, capped at 4.0.
    # ---------------------------------------------------------------------------
    KER_HW_LIMIT_MIN = int(getattr(args, "kernel_hw_min", 3) or 3)
    KER_HW_LIMIT_MAX = int(getattr(args, "kernel_hw_max", 50) or 50)
    KerHWLimit = (KER_HW_LIMIT_MIN, KER_HW_LIMIT_MAX)

    if float(args.kernel_half_width) == 0:
        fwhm_broad = max(template_fwhm, science_fwhm)
        fwhm_narrow = min(template_fwhm, science_fwhm)

        # PSF-difference FWHM (quadrature subtraction of Gaussians).
        fwhm_conv = np.sqrt(max(fwhm_broad ** 2 - fwhm_narrow ** 2, 0.0))

        # --- multiplier on fwhm_broad (primary sizing term) ---
        _mult = float(getattr(args, "kernel_hw_fwhm_multiplier", 2.0) or 2.0)
        if not np.isfinite(_mult) or _mult <= 0:
            _mult = 2.0
        _mult = max(1.0, min(_mult, 5.0))

        # Boost for undersampled images: wings extend beyond pixel grid.
        # Apply to fwhm_broad term only (that's where the support deficit arises).
        _mult_effective = _mult
        if fwhm_broad < 2.0:
            _mult_effective = min(_mult * 1.5, 4.0)
            log_info(
                f"Undersampled PSF (FWHM_broad={fwhm_broad:.2f}px): "
                f"boosting broad-PSF multiplier {_mult:.2f} -> {_mult_effective:.2f}"
            )

        # hw_broad: must contain the broader PSF support (SFFT / LSST convention).
        hw_broad = int(np.ceil(_mult_effective * fwhm_broad))

        # hw_conv: must enclose the PSF-difference lobe to 3-sigma (Israel 2007).
        # sigma_conv = fwhm_conv / 2.355; 3*sigma_conv = 3*fwhm_conv/2.355 ≈ 1.274*fwhm_conv.
        SIGMA_FWHM = 2.3548200450309493   # 2*sqrt(2*ln2)
        hw_conv = int(np.ceil(3.0 * fwhm_conv / SIGMA_FWHM)) if fwhm_conv > 0 else 0

        if fwhm_conv > 10.0:
            log_info(
                f"WARNING: large PSF mismatch (FWHM_conv={fwhm_conv:.1f}px). "
                "Subtraction quality may be degraded; consider reselecting the template."
            )
        elif fwhm_conv > 5.0:
            log_info(
                f"INFO: moderate PSF mismatch (FWHM_conv={fwhm_conv:.1f}px). "
                "Kernel sizing will be adjusted accordingly."
            )

        kernel_half_width = max(KER_HW_LIMIT_MIN, hw_broad, hw_conv)
        log_info(
            f"Auto kernel half-width: mult={_mult:.2f} (eff={_mult_effective:.2f}) "
            f"FWHM_broad={fwhm_broad:.2f}px FWHM_conv={fwhm_conv:.2f}px "
            f"hw_broad={hw_broad} hw_conv={hw_conv} -> {kernel_half_width} px "
            f"(limits: {KER_HW_LIMIT_MIN}-{KER_HW_LIMIT_MAX})"
        )
    else:
        kernel_half_width = float(args.kernel_half_width)

    # Clamp auto/manual width to SFFT limits before any downstream use.
    k_lo, k_hi = KerHWLimit
    if kernel_half_width < k_lo:
        log_info(
            f"Kernel half width {kernel_half_width:.1f} px below minimum {k_lo}; clamping."
        )
        kernel_half_width = k_lo
    if kernel_half_width > k_hi:
        log_info(
            f"Kernel half width {kernel_half_width:.1f} px above maximum {k_hi}; clamping to prevent excessive computation."
        )
        kernel_half_width = k_hi

    # Ensure integer type throughout (SFFT expects int for GKerHW)
    kernel_half_width = int(_odd(kernel_half_width))
    log_info(f"Using kernel half width: {kernel_half_width} px")
    boundary = kernel_half_width

    # --- SFFT Parameters ---
    # (From SFFT source: thomasvrussell/sfft, EasySparsePacket.ESP / EasyCrowdedPacket.ECP)
    # - GKerHW: given kernel half-width (px). If None, SFFT uses KerHWRatio * Max(FWHM_REF,FWHM_SCI) then clip to KerHWLimit.
    # - KerHWRatio: default 2.0 in SFFT; KerHW = int(clip(KerHWRatio * FWHM_La, KerHWLimit[0], KerHWLimit[1])).
    # - KerHWLimit: (min, max) kernel half-width; SFFT default (2, 20). We use (3, 50) for large FWHM differences.
    # - ForceConv: 'REF'|'SCI'|'AUTO'. REF => DIFF=SCI-conv(REF) (transient keeps science PSF). AUTO picks by seeing.
    # ECP (crowded) expects 'Cupy' (capital C, lowercase py); ESP accepts same.
    # try:
    #     import cupy  # noqa: F401
    #     BACKEND_4SUBTRACT = 'Cupy'
    # except Exception:
    BACKEND_4SUBTRACT = "Numpy"

    CUDA_DEVICE_4SUBTRACT = "0"

    # Use 1 thread for SFFT internals to avoid libgomp/process limits when run
    # as a subprocess or on HPC (avoids "Thread creation failed" and semaphore leaks).
    NUM_CPU_THREADS_4SUBTRACT = 1

    # Honour the -forceconv CLI argument.
    # REF  => DIFF = SCI - conv(REF): transients keep the science PSF (default,
    #         recommended when science has broader PSF than the template).
    # SCI  => DIFF = conv(SCI) - REF: use when the template has broader PSF.
    # AUTO => SFFT chooses based on measured FWHMs.
    # Always convolve the reference image (ForceConv=REF).
    # DIFF = SCI - conv(REF): the transient keeps the science PSF.
    ForceConv = "REF"
    log_info("ForceConv=REF (reference always convolved). DIFF = SCI - conv(REF).")
    GAIN_KEY = "GAIN"
    SATUR_KEY = "SATURATE"

    def _parse_bool_str(name: str, value: str) -> bool:
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False
        raise ValueError(f"Invalid -{name}='{value}'; expected true/false.")

    kernel_poly_order = args.kernel_order
    bg_poly_order = max(0, int(args.bg_order))
    allow_crowded_bg_order_override = _parse_bool_str(
        "allow_crowded_bg_order_override",
        getattr(args, "allow_crowded_bg_order_override", "false"),
    )
    if args.crowded and bg_poly_order == 0 and allow_crowded_bg_order_override:
        # MultiEasyCrowdedPacket default favours a non-trivial BGPolyOrder.
        # Pipelines often pass 0 as "no explicit background model", but in crowded
        # (non-sky-subtracted) cases this can destabilize scaling.
        bg_poly_order = 2
        log_info("Crowded SFFT: bg_order=0 overridden to 2 for stability.")
    log_info(f"Background polynomial order: {bg_poly_order}")

    # --- SExtractor parameters (both sparse and crowded) ---
    # BACKPHOTO_TYPE controls the background model used for source-level photometry.
    # LOCAL uses a background evaluated around each source; GLOBAL uses the global background.
    is_crowded = args.crowded

    # SExtractor detection threshold for SFFT source selection.
    # 1.5 sigma for sparse fields (background-subtracted images may have low flux)
    # 3.0 sigma for crowded fields (avoids noise peaks)
    # The previous fixed value of 3.0 caused source detection failures on
    # SWarp-resampled, background-subtracted images with low flux levels.
    # User can override via -detect_thresh argument.
    if args.detect_thresh is not None and args.detect_thresh > 0:
        DETECT_THRESH = float(args.detect_thresh)
        log_info(f"Using user-specified DETECT_THRESH: {DETECT_THRESH:.1f}")
    else:
        DETECT_THRESH = 1.5 if not is_crowded else 3.0
    DEBLEND_MINCON = 0.005

    constant_phot_ratio = _parse_bool_str(
        "constphotratio", getattr(args, "constphotratio", "false")
    )

    # sfft defaults: StarExt_iter=2 for crowded; 4 for sparse.
    # Override with user-provided value if given (0 is sentinel for "use defaults").
    if args.star_ext_iter is not None and args.star_ext_iter > 0:
        StarExt_iter = int(args.star_ext_iter)
        log_info(f"Using user-specified StarExt_iter: {StarExt_iter}")
    else:
        StarExt_iter = 2 if is_crowded else 4

    BACKPHOTO_TYPE = str(getattr(args, "backphototype", "LOCAL")).upper().strip()
    if BACKPHOTO_TYPE not in ("LOCAL", "GLOBAL"):
        raise ValueError(
            f"Invalid -backphototype='{BACKPHOTO_TYPE}'; expected 'LOCAL' or 'GLOBAL'."
        )
    mode_label = "Crowded-field" if is_crowded else "Sparse-field"
    log_info(
        f"{mode_label} SFFT: BACKPHOTO_TYPE={BACKPHOTO_TYPE}, ConstPhotRatio={constant_phot_ratio}, "
        f"StarExt_iter={StarExt_iter}, DETECT_THRESH={DETECT_THRESH:.1f}, DEBLEND_MINCONT={DEBLEND_MINCON:.4f}"
    )

    BACK_SIZE = int(max(16, args.back_size))
    BACK_FILTERSIZE = int(max(1, args.back_filtersize))
    log_info(
        f"SExtractor background mesh:\n"
        f"  BACK_SIZE: {BACK_SIZE} px\n"
        f"  BACK_FILTERSIZE: {BACK_FILTERSIZE}"
    )

    def parse_only_flags(s: str):
        text = str(s or "").strip().lower()
        if text in ("", "none", "null", "false"):
            return None
        vals = []
        for tok in text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            vals.append(int(tok))
        return vals if len(vals) > 0 else None

    ONLY_FLAGS = parse_only_flags(args.only_flags)

    COARSE_VAR_REJECTION = _parse_bool_str(
        "coarse_var_rejection",
        getattr(args, "coarse_var_rejection", "false"),
    )
    CVREJ_MAGD_THRESH = float(args.cvrej_magd_thresh)
    ELABO_VAR_REJECTION = _parse_bool_str(
        "elabo_var_rejection",
        getattr(args, "elabo_var_rejection", "false"),
    )
    EVREJ_RATIO_THREH = float(args.evrej_ratio_thresh)
    EVREJ_SAFE_MAGDEV = float(args.evrej_safe_magdev)
    PAC_RATIO_THRESH = float(args.pac_ratio_thresh)
    log_info(
        "SFFT rejection params: "
        f"ONLY_FLAGS={ONLY_FLAGS} "
        f"CVREJ_MAGD_THRESH={CVREJ_MAGD_THRESH:.3f} "
        f"EVREJ_RATIO_THRESH={EVREJ_RATIO_THREH:.3f} "
        f"EVREJ_SAFE_MAGDEV={EVREJ_SAFE_MAGDEV:.3f} "
        f"PAC_RATIO_THRESH={PAC_RATIO_THRESH:.3f}"
    )

    # KerHWRatio: ratio between FWHM and kernel half-width when GKerHW is None.
    # SFFT source (EasySparsePacket.ESP): KerHW = int(clip(KerHWRatio * Max(FWHM_REF, FWHM_SCI), limit[0], limit[1])).
    # We pass GKerHW explicitly so this is only used as fallback; keep SFFT default 2.0.
    KerHWRatio = 2.0

    # --- New SFFT Features (v1.5.0+) ---
    use_bspline_kernel = _parse_bool_str("use_bspline_kernel", args.use_bspline_kernel)
    decorrelate_noise = _parse_bool_str("decorrelate_noise", args.decorrelate_noise)
    save_original_diff = _parse_bool_str("save_original_diff", args.save_original_diff)

    if use_bspline_kernel and not _HAS_BSPLINE:
        log_info("Warning: B-Spline kernel requested but not available (SFFT v1.5.0+ required). Using standard kernel.")
        use_bspline_kernel = False

    if decorrelate_noise and not _HAS_DECORRELATION:
        log_info("Warning: Noise decorrelation requested but not available (SFFT v1.5.0+ required). Skipping decorrelation.")
        decorrelate_noise = False

    if use_bspline_kernel:
        log_info("Using B-Spline kernel matching for complex PSF variations")
    if decorrelate_noise:
        log_info("Noise decorrelation enabled for difference image")

    # --- Load Mask (Optional) ---
    prior_ban_mask = None
    if args.mask:
        try:
            mask_raw = fits.getdata(args.mask)
            if mask_raw.dtype != bool:
                mask_raw = mask_raw.astype(bool)

            # FITS and numpy both use row-major (row=y, col=x) storage, so the
            # mask written by the pipeline should already be in the correct
            # orientation and no transposition is needed.  The previous heuristic
            # (pick the orientation whose unmasked science median is closest to 0)
            # was unreliable on background-subtracted images because both
            # orientations give median ≈ 0, making the choice effectively random.
            if mask_raw.shape == data_sci.shape:
                prior_ban_mask = mask_raw
                log_info(
                    f"Loaded mask: shape {mask_raw.shape}, "
                    f"{int(np.count_nonzero(mask_raw))} masked pixels."
                )
            else:
                raise ValueError(
                    f"Mask shape {mask_raw.shape} does not match science shape {data_sci.shape}. "
                    "Ensure the universal_mask.fits was written with the same spatial grid as "
                    "the science image (no transposition is applied)."
                )
        except Exception as e:
            log_info(f"Warning: Could not load mask '{args.mask}': {e}")

    # Both science and template are background-subtracted by the pipeline before
    # reaching this script, so BACK_VALUE=0.0 is correct.  Estimating a non-zero
    # offset here from the (already-zero-median) science image produced a
    # double-subtraction inside SFFT's internal SExtractor step, biasing source
    # photometry used for kernel fitting and corrupting the flux scale.
    BACK_VALUE = 0.0
    log_info("SFFT BACK_VALUE=0.0 (inputs are pipeline-background-subtracted).")

    # --- Run SFFT ---
    t0_sfft = time.time()
    try:
        matched_sources = pd.DataFrame()
        if args.crowded:
            # Crowded-field subtraction (ECP): no prior source list; uses SExtractor + masking.
            log_info("Running crowded-field subtraction (ECP).")
            fits_solution = (
                os.path.join(out_dir, "sfft_solution.fits") if FITS_DIFF else None
            )
            # GAIN and SATURATE have been written to both FITS headers above.
            result = Easy_CrowdedPacket.ECP(
                FITS_REF=FITS_REF,
                FITS_SCI=FITS_SCI,
                FITS_DIFF=FITS_DIFF,
                FITS_Solution=fits_solution,
                ForceConv=ForceConv,
                GKerHW=int(kernel_half_width),
                KerHWRatio=KerHWRatio,
                KerHWLimit=KerHWLimit,
                KerPolyOrder=kernel_poly_order,
                BGPolyOrder=bg_poly_order,
                ConstPhotRatio=constant_phot_ratio,
                MaskSatContam=False,
                GAIN_KEY=GAIN_KEY,
                SATUR_KEY=SATUR_KEY,
                BACK_TYPE="MANUAL",
                BACK_VALUE=BACK_VALUE,
                BACK_SIZE=BACK_SIZE,
                BACK_FILTERSIZE=BACK_FILTERSIZE,
                DETECT_THRESH=DETECT_THRESH,
                ANALYSIS_THRESH=DETECT_THRESH,
                DETECT_MINAREA=detect_minarea,
                DETECT_MAXAREA=detect_maxarea,
                DEBLEND_MINCONT=DEBLEND_MINCON,
                BACKPHOTO_TYPE=BACKPHOTO_TYPE,
                ONLY_FLAGS=ONLY_FLAGS,
                BoundarySIZE=boundary,
                BACK_SIZE_SUPER=128,
                StarExt_iter=StarExt_iter,
                PriorBanMask=prior_ban_mask,
                BACKEND_4SUBTRACT=BACKEND_4SUBTRACT,
                CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT,
                NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT,
                VERBOSE_LEVEL=2,
            )
            # ECP returns (PixA_DIFF, SFFTPrepDict, Solution, SFFT_FSCAL_MEAN, SFFT_FSCAL_SIG)
            if not result:
                raise RuntimeError("Easy_CrowdedPacket.ECP returned None or empty result.")
            diff_image = result[0]
            prep_data = result[1] if len(result) > 1 else {}
            # ECP may write FITS_DIFF itself; if we have diff_image and file missing, write it
            if diff_image is not None and FITS_DIFF and not os.path.isfile(FITS_DIFF):
                try:
                    with fits.open(FITS_SCI, mode="readonly") as hdl:
                        hdu = hdl[0].copy()
                        # SFFT internally uses (ny, nx); FITS data is also (ny, nx) row-major.
                        # Only transpose if diff_image is in (nx, ny) order.
                        sci_ny, sci_nx = hdu.data.shape
                        if diff_image.shape == (sci_nx, sci_ny):
                            hdu.data = diff_image.T
                        elif diff_image.shape == (sci_ny, sci_nx):
                            hdu.data = diff_image
                        else:
                            log_info(
                                f"Warning: diff_image shape {diff_image.shape} does not match "
                                f"science shape {(sci_ny, sci_nx)}; writing as-is."
                            )
                            hdu.data = diff_image
                        # Use safe_fits_write to preserve NaNs
                        safe_fits_write(FITS_DIFF, hdu.data, hdu.header)
                except Exception as e:
                    log_info(f"Warning: Could not write ECP diff to {FITS_DIFF}: {e}")
            # Optional: write a minimal matching-sources CSV from ECP catalog if present
            cat_key = "SExCatalog-SubSource"
            if cat_key in prep_data:
                catalog = prep_data[cat_key]
                matched_sources = _to_dataframe(catalog)
                log_info(
                    f"Number of sources used in crowded-field matching: {len(matched_sources)}"
                )
                xcol = (
                    "X_IMAGE_REF_SCI_MEAN"
                    if "X_IMAGE_REF_SCI_MEAN" in matched_sources.columns
                    else None
                )
                ycol = (
                    "Y_IMAGE_REF_SCI_MEAN"
                    if "Y_IMAGE_REF_SCI_MEAN" in matched_sources.columns
                    else None
                )
                if xcol is None:
                    for a, b in [
                        ("x_center", "y_center"),
                        ("X_IMAGE_REF_SCI_MEAN", "Y_IMAGE_REF_SCI_MEAN"),
                    ]:
                        if (
                            a in matched_sources.columns
                            and b in matched_sources.columns
                        ):
                            xcol, ycol = a, b
                            break
                out_csv = os.path.join(
                    out_dir, f"SFFT_Matching_Sources_{out_base}.csv"
                )
                if xcol and ycol:
                    df_out = matched_sources[[xcol, ycol]].rename(
                        columns={
                            xcol: "X_IMAGE_REF_SCI_MEAN",
                            ycol: "Y_IMAGE_REF_SCI_MEAN",
                        }
                    )
                    df_out.to_csv(out_csv, index=False, float_format="%.6f")
                else:
                    matched_sources.to_csv(out_csv, index=False, float_format="%.6f")
        else:
            log_info("Running sparse-field subtraction (ESP).")
            try:
                # First attempt: honour prior matching/ban lists from the pipeline.
                result = Easy_SparsePacket.ESP(
                    FITS_REF=FITS_REF,
                    FITS_SCI=FITS_SCI,
                    FITS_DIFF=FITS_DIFF,
                    GKerHW=int(kernel_half_width),
                    KerHWRatio=KerHWRatio,
                    # UPDATED: tighter kernel HW limits
                    KerHWLimit=KerHWLimit,
                    KerPolyOrder=kernel_poly_order,
                    ForceConv=ForceConv,
                    BACK_TYPE="MANUAL",
                    BACK_VALUE=BACK_VALUE,
                    BACK_SIZE=BACK_SIZE,
                    BACK_FILTERSIZE=BACK_FILTERSIZE,
                    BACKPHOTO_TYPE=BACKPHOTO_TYPE,
                    BGPolyOrder=bg_poly_order,
                    DETECT_THRESH=DETECT_THRESH,
                    DETECT_MINAREA=detect_minarea,
                    DETECT_MAXAREA=detect_maxarea,
                    DEBLEND_MINCONT=DEBLEND_MINCON,
                    MaskSatContam=False,
                    ConstPhotRatio=constant_phot_ratio,
                    GAIN_KEY=GAIN_KEY,
                    SATUR_KEY=SATUR_KEY,
                    XY_PriorSelect=matching_sources,
                    XY_PriorBan=masked_sources,
                    MatchTol=None,
                    # Matching tolerance: overly tight tolerances can lock onto a bad solution.
                    MatchTolFactor=1.0,
                    Hough_MINFR=0.1,
                    Hough_PeakClip=0.4,
                    BeltHW=0.2,
                    COARSE_VAR_REJECTION=COARSE_VAR_REJECTION,
                    CVREJ_MAGD_THRESH=CVREJ_MAGD_THRESH,
                    ELABO_VAR_REJECTION=ELABO_VAR_REJECTION,
                    EVREJ_RATIO_THREH=EVREJ_RATIO_THREH,
                    EVREJ_SAFE_MAGDEV=EVREJ_SAFE_MAGDEV,
                    StarExt_iter=StarExt_iter,
                    PostAnomalyCheck=True,
                    PAC_RATIO_THRESH=PAC_RATIO_THRESH,
                    BoundarySIZE=boundary,
                    ONLY_FLAGS=ONLY_FLAGS,
                    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT,
                    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT,
                    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT,
                )
            except np.linalg.LinAlgError as e:
                # Degenerate kernel design matrix (e.g. too few / collinear sources
                # after applying XY_PriorSelect / XY_PriorBan). Retry once letting
                # SFFT perform its own source matching with no priors.
                log_info(
                    f"SFFT ESP failed with singular matrix when using priors ({e}). "
                    "This typically occurs when:"
                    "  1. Too few prior sources for reliable kernel fitting (default minimum: 3)"
                    "  2. Prior sources are collinear or poorly distributed"
                    "  3. Prior sources have large positional errors"
                    "Retrying without prior-selected / prior-banned sources."
                )
                result = Easy_SparsePacket.ESP(
                    FITS_REF=FITS_REF,
                    FITS_SCI=FITS_SCI,
                    FITS_DIFF=FITS_DIFF,
                    GKerHW=int(kernel_half_width),
                    KerHWRatio=KerHWRatio,
                    KerHWLimit=KerHWLimit,
                    KerPolyOrder=kernel_poly_order,
                    ForceConv=ForceConv,
                    BACK_TYPE="MANUAL",
                    BACK_VALUE=BACK_VALUE,
                    BACK_SIZE=BACK_SIZE,
                    BACK_FILTERSIZE=BACK_FILTERSIZE,
                    BACKPHOTO_TYPE=BACKPHOTO_TYPE,
                    BGPolyOrder=bg_poly_order,
                    DETECT_THRESH=DETECT_THRESH,
                    DETECT_MINAREA=detect_minarea,
                    DETECT_MAXAREA=detect_maxarea,
                    DEBLEND_MINCONT=DEBLEND_MINCON,
                    MaskSatContam=False,
                    ConstPhotRatio=constant_phot_ratio,
                    GAIN_KEY=GAIN_KEY,
                    SATUR_KEY=SATUR_KEY,
                    XY_PriorSelect=None,
                    XY_PriorBan=None,
                    MatchTol=None,
                    MatchTolFactor=1,
                    Hough_MINFR=0.1,
                    Hough_PeakClip=0.4,
                    BeltHW=0.2,
                    COARSE_VAR_REJECTION=COARSE_VAR_REJECTION,
                    CVREJ_MAGD_THRESH=CVREJ_MAGD_THRESH,
                    ELABO_VAR_REJECTION=ELABO_VAR_REJECTION,
                    EVREJ_RATIO_THREH=EVREJ_RATIO_THREH,
                    EVREJ_SAFE_MAGDEV=EVREJ_SAFE_MAGDEV,
                    StarExt_iter=StarExt_iter,
                    PostAnomalyCheck=True,
                    PAC_RATIO_THRESH=PAC_RATIO_THRESH,
                    BoundarySIZE=boundary,
                    ONLY_FLAGS=ONLY_FLAGS,
                    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT,
                    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT,
                    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT,
                )
            # ESP returns (diff_image, prep_data, ...); support tuple or list.
            if len(result) < 2:
                raise ValueError(
                    f"SFFT ESP returned {len(result)} value(s), expected at least 2"
                )
            diff_image, prep_data = result[0], result[1]

            # Apply B-Spline kernel if requested (SFFT v1.5.0+)
            if use_bspline_kernel and _HAS_BSPLINE:
                try:
                    log_info("Applying B-Spline kernel refinement...")
                    # Re-run subtraction with B-Spline kernel
                    bspline_result = BSpline_Packet.BSP(
                        FITS_REF=FITS_REF,
                        FITS_SCI=FITS_SCI,
                        FITS_DIFF=FITS_DIFF,
                        GKerHW=int(kernel_half_width),
                        KerHWRatio=KerHWRatio,
                        KerHWLimit=KerHWLimit,
                        KerPolyOrder=kernel_poly_order,
                        ForceConv=ForceConv,
                        BACK_TYPE="MANUAL",
                        BACK_VALUE=BACK_VALUE,
                        BACK_SIZE=BACK_SIZE,
                        BACK_FILTERSIZE=BACK_FILTERSIZE,
                        BACKPHOTO_TYPE=BACKPHOTO_TYPE,
                        BGPolyOrder=bg_poly_order,
                        DETECT_THRESH=DETECT_THRESH,
                        DETECT_MINAREA=detect_minarea,
                        DETECT_MAXAREA=detect_maxarea,
                        DEBLEND_MINCONT=DEBLEND_MINCON,
                        MaskSatContam=False,
                        ConstPhotRatio=constant_phot_ratio,
                        GAIN_KEY=GAIN_KEY,
                        SATUR_KEY=SATUR_KEY,
                        XY_PriorSelect=matching_sources if matching_sources is not None else None,
                        XY_PriorBan=masked_sources if masked_sources is not None else None,
                        MatchTol=None,
                        MatchTolFactor=1,
                        Hough_MINFR=0.1,
                        Hough_PeakClip=0.4,
                        BeltHW=0.2,
                        COARSE_VAR_REJECTION=COARSE_VAR_REJECTION,
                        CVREJ_MAGD_THRESH=CVREJ_MAGD_THRESH,
                        ELABO_VAR_REJECTION=ELABO_VAR_REJECTION,
                        EVREJ_RATIO_THREH=EVREJ_RATIO_THREH,
                        EVREJ_SAFE_MAGDEV=EVREJ_SAFE_MAGDEV,
                        StarExt_iter=StarExt_iter,
                        PostAnomalyCheck=True,
                        PAC_RATIO_THRESH=PAC_RATIO_THRESH,
                        BoundarySIZE=boundary,
                        ONLY_FLAGS=ONLY_FLAGS,
                        BACKEND_4SUBTRACT=BACKEND_4SUBTRACT,
                        CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT,
                        NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT,
                    )
                    if bspline_result and len(bspline_result) >= 2:
                        diff_image, prep_data = bspline_result[0], bspline_result[1]
                        log_info("B-Spline kernel refinement completed successfully")
                except Exception as e:
                    log_info(f"Warning: B-Spline kernel refinement failed: {e}. Using standard kernel result.")

            cat_key = "SExCatalog-SubSource"
            if cat_key not in prep_data:
                raise KeyError(
                    f"SFFT prep_data missing '{cat_key}'; incompatible SFFT version?"
                )
            catalog = prep_data[cat_key]
            matched_sources = _to_dataframe(catalog)

            log_info(f"Number of sources used in matching: {len(matched_sources)}")

            # main.py expects columns X_IMAGE_REF_SCI_MEAN, Y_IMAGE_REF_SCI_MEAN
            xcol = (
                "X_IMAGE_REF_SCI_MEAN"
                if "X_IMAGE_REF_SCI_MEAN" in matched_sources.columns
                else None
            )
            ycol = (
                "Y_IMAGE_REF_SCI_MEAN"
                if "Y_IMAGE_REF_SCI_MEAN" in matched_sources.columns
                else None
            )
            if xcol is None or ycol is None:
                for a, b in [
                    ("x_center", "y_center"),
                    ("X_IMAGE_REF_SCI_MEAN", "Y_IMAGE_REF_SCI_MEAN"),
                ]:
                    if a in matched_sources.columns and b in matched_sources.columns:
                        xcol, ycol = a, b
                        break
            out_csv = os.path.join(
                out_dir, f"SFFT_Matching_Sources_{out_base}.csv"
            )
            if xcol and ycol:
                df_out = matched_sources[[xcol, ycol]].rename(
                    columns={xcol: "X_IMAGE_REF_SCI_MEAN", ycol: "Y_IMAGE_REF_SCI_MEAN"}
                )
                df_out.to_csv(out_csv, index=False, float_format="%.6f")
            else:
                matched_sources.to_csv(out_csv, index=False, float_format="%.6f")

        try:
            _write_post_anomaly_sources_csv(
                prep_data_obj=prep_data,
                matched_df=matched_sources,
                out_dir_path=out_dir,
                out_base_name=out_base,
            )
        except Exception as e:
            log_info(f"Warning: Could not export post-anomaly sources: {e}")

        try:
            convd = fits.getheader(FITS_DIFF).get("CONVD", "UNKNOWN")
            log_info(f"[{convd}] is convolved in subtraction.")
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Post-subtraction image quality improvements (LSST-inspired + SFFT v1.5.0+)
        #
        # 1. SFFT NOISE DECORRELATION (SFFT v1.5.0+)
        #    SFFT provides built-in noise decorrelation for difference images,
        #    particularly useful for coadded images or when convolution is involved.
        #    This whitens correlated noise in the difference image.
        #
        # 2. DECORRELATION KERNEL  (LSST DMTN-021, Reiss & Lupton 2016)
        #    The A&L PSF-matching kernel κ is convolved with the template,
        #    which introduces pixel-pixel covariance in the difference image D.
        #    Detected sources therefore appear correlated, inflating peak S/N
        #    at scales of ~KerHW px and causing a detection threshold that must
        #    be raised to ~5.5σ (rather than the canonical 5.0σ) to control false
        #    positives.  LSST corrects this by convolving D with a whitening
        #    (decorrelation) kernel ψ computed in Fourier space from κ and the
        #    mean variances of the two images (DMTN-021 Eq. 2):
        #
        #      ψ(k) = sqrt( (σ₁² + σ₂²) / (σ₁² + κ²(k)·σ₂²) )
        #      D′   = ψ ⊗ D
        #
        #    After decorrelation, pixel noise is spatially uncorrelated and
        #    matched-filter detection can be run at 5.0σ with no excess FPR.
        #    The decorrelation kernel is ~2×KerHW px in size and inexpensive.
        #
        # 3. VARIANCE SCALING  (LSST ScaleVarianceTask, DMTN-021 §4.1)
        #    Warping and co-adding introduce pixel covariance that causes the
        #    variance plane to underestimate the true noise.  LSST rescales the
        #    variance by a factor that brings IQR(D/sqrt(V)) to unity.
        #    Here we propagate the expected Gaussian variance of the difference
        #    image ( σ_diff² = σ_sci² + σ_ref² ) from the sigma-clipped image
        #    statistics, then rescale the difference image so its measured noise
        #    matches that expectation.  This is equivalent to the LSST
        #    pixel-based ScaleVarianceTask estimator.
        # ------------------------------------------------------------------
        log_info(border_msg("Post-subtraction quality improvements", metadata="LSST-inspired + SFFT v1.5.0+", use_ansi=False))
        log_info("  SFFT noise decorrelation (v1.5.0+) — whitens correlated noise")
        log_info("  Decorrelation kernel (DMTN-021) — whitens A&L convolution noise")
        log_info("  Variance scaling (ScaleVarianceTask) — IQR-based noise calibration")

        # Apply noise decorrelation if requested
        # Note: SFFT's DeCorrelation_Calculator requires kernel information from the SFFT solution
        # which is not easily accessible in the current pipeline architecture. The LSST decorrelation
        # (DMTN-021) provides equivalent noise whitening and is used as the primary method.
        if decorrelate_noise:
            if _HAS_DECORRELATION:
                log_info("SFFT decorrelation available but requires kernel information from SFFT solution.")
                log_info("Using LSST decorrelation (DMTN-021) as equivalent alternative.")
            else:
                log_info("SFFT decorrelation not available (SFFT v1.5.0+ required).")
                log_info("Using LSST decorrelation (DMTN-021) as equivalent alternative.")
            # LSST decorrelation will be applied below in the _decorrelate_diffim function

        def _decorrelate_diffim(
            diff: np.ndarray,
            kernel: np.ndarray,
            var_sci: float,
            var_ref: float,
            nan_mask: np.ndarray,
        ) -> np.ndarray:
            """Apply LSST DMTN-021 decorrelation to the A&L difference image.

            Parameters
            ----------
            diff : (H, W) float array — the raw difference image.
            kernel : (kH, kW) float array — the A&L matching kernel κ used to
                     convolve the template (already normalised to sum ≈ 1).
            var_sci : mean per-pixel variance of the science image (σ₁²).
            var_ref : mean per-pixel variance of the reference/template image (σ₂²).
            nan_mask : bool array, True where pixels are invalid; used to
                       temporarily fill NaN regions with zeros for FFTs.

            Returns
            -------
            Decorrelated difference image D′ (same shape as diff).

            Notes
            -----
            Algorithm (DMTN-021 §2.1):
              ψ(k) = sqrt( (σ₁² + σ₂²) / (σ₁² + |κ(k)|² · σ₂²) )

            Implementation:
              1. Embed κ in a zero-padded array of the full image size (H×W).
                 This makes ψ(k) defined at the same Fourier frequencies as D.
              2. Compute ψ(k) in Fourier space.
              3. IRFFT2 back to image size, FFT-shift to centre, and extract the
                 central (2·kH + 1) × (2·kW + 1) support of ψ.
                 ψ is compact — its real-space support is the same spatial scale
                 as κ itself.  Extracting at hw = kH captures >99% of the power
                 and avoids edge wrap-around artefacts.
              4. Convolve D with ψ via scipy.fftconvolve (real-space, so masked
                 pixels are handled gracefully without full-image FFT).
            """
            try:
                from scipy.signal import fftconvolve

                H, W = diff.shape
                kH, kW = kernel.shape

                # --- Build ψ in the full-image Fourier domain ---
                # Embed κ in a (H, W) zero-padded array, centred at (0,0) via
                # ifftshift so that the origin is at the top-left corner (numpy
                # rfft2 convention).
                ker_full = np.zeros((H, W), dtype=np.float64)
                ker_full[:kH, :kW] = kernel
                K_f = np.fft.rfft2(np.fft.ifftshift(ker_full))   # (H, W//2+1)
                K2  = np.real(K_f * np.conj(K_f))                 # |κ(k)|²

                denom = var_sci + K2 * var_ref
                # Guard against near-zero denominator (numerical safety)
                denom = np.where(
                    denom > 1e-30 * (var_sci + var_ref),
                    denom,
                    var_sci + var_ref,
                )
                psi_f = np.sqrt((var_sci + var_ref) / denom)      # (H, W//2+1)

                # --- Extract compact real-space ψ kernel ---
                # IRFFT2 back to image size then FFT-shift to centre.
                psi_full = np.real(np.fft.irfft2(psi_f, s=(H, W)))  # (H, W)
                psi_full = np.fft.fftshift(psi_full)

                # Extract central ±kH / ±kW support (ψ is compact at ~kernel scale)
                cH, cW = H // 2, W // 2
                hw_h, hw_w = max(kH, 1), max(kW, 1)
                psi_small = psi_full[
                    cH - hw_h : cH + hw_h + 1,
                    cW - hw_w : cW + hw_w + 1,
                ].copy()

                # Normalise so ψ preserves total flux (unit sum, not unit energy)
                psi_sum = float(psi_small.sum())
                if abs(psi_sum) > 1e-10:
                    psi_small /= psi_sum

                # --- Convolve D with ψ ---
                # Replace NaNs with 0 for convolution; restore afterwards.
                diff_filled = diff.copy()
                diff_filled[nan_mask] = 0.0

                diff_decorr = fftconvolve(diff_filled, psi_small, mode="same")

                # Restore NaN regions
                diff_decorr[nan_mask] = np.nan
                return diff_decorr.astype(diff.dtype)

            except Exception as _e:
                log_info(f"Warning: decorrelation kernel failed ({_e}); skipping.")
                return diff

        def _scale_diffim_variance(
            diff: np.ndarray,
            var_expected: float,
            nan_mask: np.ndarray,
        ) -> tuple:
            """Rescale the difference image so its measured noise matches expectation.

            Implements the LSST ScaleVarianceTask pixel-based estimator:
            compute SNR = D / sqrt(var_expected) for background pixels, measure
            its spread via IQR, and rescale so IQR/1.349 (the Gaussian sigma
            equivalent) equals 1.0.  The IQR is robust to bright sources and
            artifacts.

            Parameters
            ----------
            diff : (H, W) float array — the difference image.
            var_expected : expected Gaussian variance σ_diff² = σ_sci² + σ_ref².
            nan_mask : bool array, True where pixels are invalid.

            Returns
            -------
            (scaled_diff, scale_factor) : scaled image and the factor applied.
            A scale_factor of 1.0 means no change was applied.
            """
            try:
                valid = ~nan_mask & np.isfinite(diff)
                if valid.sum() < 1000:
                    return diff, 1.0

                sigma_expected = float(np.sqrt(max(var_expected, 1e-30)))
                snr_vals = diff[valid] / sigma_expected

                # IQR-based robust standard-deviation estimate (LSST convention)
                q25, q75 = np.percentile(snr_vals, [25.0, 75.0])
                iqr_sigma = (q75 - q25) / 1.3489795003921634   # 1/Φ⁻¹(0.75)

                if iqr_sigma < 0.1 or not np.isfinite(iqr_sigma):
                    return diff, 1.0   # pathological image, skip

                # Safety: only rescale if the discrepancy is non-trivial (>5%)
                # but not extreme (>3×, which suggests a bug rather than covariance).
                if 0.95 <= iqr_sigma <= 3.0:
                    if abs(iqr_sigma - 1.0) < 0.05:
                        return diff, 1.0   # already consistent, nothing to do
                    scale = 1.0 / iqr_sigma
                    return (diff * scale), scale
                return diff, 1.0

            except Exception as _e:
                log_info(f"Warning: variance scaling failed ({_e}); skipping.")
                return diff, 1.0

        # Apply decorrelation + variance scaling to the SFFT output.
        # Both steps operate on the FITS file in-place to keep downstream code
        # (which reads the file from disk) consistent.
        try:
            with fits.open(FITS_DIFF, mode="update", memmap=False) as hdul:
                diff_arr = np.asarray(hdul[0].data, dtype=np.float64)
                diff_hdr = hdul[0].header

                # Build an invalid-pixel mask for this stage (NaN or ±Inf).
                _nan_mask = ~np.isfinite(diff_arr)

                # Retrieve image variances from SFFT header (sigma-clipped means)
                _sci_var = None
                _ref_var = None
                try:
                    _fwhm_ref = float(diff_hdr.get("FWHM_REF", 0))
                    _fwhm_sci = float(diff_hdr.get("FWHM_SCI", 0))
                except Exception:
                    _fwhm_ref = _fwhm_sci = 0.0
                # Estimate per-image variances from sigma-clipped noise of each
                # input image (use the data already loaded).
                try:
                    _sci_vals = data_sci[np.isfinite(data_sci)].ravel()
                    _ref_vals = data_ref[np.isfinite(data_ref)].ravel()
                    if len(_sci_vals) > 1000 and len(_ref_vals) > 1000:
                        _q25s, _q75s = np.percentile(_sci_vals, [25.0, 75.0])
                        _q25r, _q75r = np.percentile(_ref_vals, [25.0, 75.0])
                        _sci_var = ((_q75s - _q25s) / 1.3489795003921634) ** 2
                        _ref_var = ((_q75r - _q25r) / 1.3489795003921634) ** 2
                except Exception:
                    pass

                # --- Step 1: Decorrelation kernel ---
                # Retrieve the A&L kernel from the SFFT solution (result index 2).
                _applied_decorr = False
                _diff_arr_original = diff_arr.copy()  # Save original before decorrelation
                try:
                    if len(result) >= 3 and _sci_var is not None and _ref_var is not None:
                        from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel
                        _solution = result[2]
                        _kerhw = int(diff_hdr.get("KERHW", 0))
                        if _kerhw > 0 and _solution is not None:
                            _L = 2 * _kerhw + 1
                            _N0, _N1 = data_sci.shape
                            _DK = int(diff_hdr.get("KERORDER", diff_hdr.get("KERPOLY", 0)))
                            _Fpq_raw = diff_hdr.get("BGORDER", diff_hdr.get("BGPOLY", 0))
                            _Fpq = int((_Fpq_raw + 1) * (_Fpq_raw + 2) // 2)
                            # Realize the kernel at the image centre.
                            # Realize_MatchingKernel takes XY_q (Fortran coord) in __init__
                            # and returns KerStack of shape (Num_request, L, L).
                            _cx = float(_N1) / 2.0  # Fortran X = column
                            _cy = float(_N0) / 2.0  # Fortran Y = row
                            _XY_q = np.array([[_cx, _cy]])
                            _ker_stack = Realize_MatchingKernel(_XY_q).FromArray(
                                Solution=_solution,
                                N0=_N0, N1=_N1,
                                L0=_L, L1=_L,
                                DK=_DK, Fpq=_Fpq,
                            )
                            # KerStack[0] is the kernel at the requested coordinate
                            _ker_2d = np.asarray(_ker_stack[0]).squeeze()
                            if _ker_2d.ndim == 2 and _ker_2d.shape[0] == _L:
                                diff_arr = _decorrelate_diffim(
                                    diff_arr, _ker_2d,
                                    float(_sci_var), float(_ref_var), _nan_mask
                                )
                                _applied_decorr = True
                                log_info(
                                    f"Decorrelation kernel applied: KerHW={_kerhw} px, "
                                    f"σ_sci={np.sqrt(_sci_var):.2f}, σ_ref={np.sqrt(_ref_var):.2f}"
                                )
                except Exception as _e:
                    log_info(f"Warning: could not apply decorrelation kernel: {_e}")

                # --- Step 2: Variance scaling ---
                _var_expected = (_sci_var or 0.0) + (_ref_var or 0.0)
                _applied_vscale = False
                if _var_expected > 0:
                    diff_arr, _vscale = _scale_diffim_variance(
                        diff_arr, _var_expected, _nan_mask
                    )
                    _applied_vscale = True
                    if abs(_vscale - 1.0) > 0.005:
                        log_info(
                            f"Variance scaling applied: IQR-sigma factor={1.0/_vscale:.4f} -> rescaled by {_vscale:.4f}"
                        )
                        diff_hdr["VSCALE"] = (round(float(_vscale), 6),
                                              "Variance rescale factor (LSST-style IQR)")
                    else:
                        log_info(
                            f"Variance scaling: noise consistent (IQR-sigma factor={1.0/_vscale:.4f}); no rescale"
                        )

                # Summary of which improvements were applied
                if _applied_decorr:
                    diff_hdr["DECORR"] = (True, "DMTN-021 A&L decorrelation applied")
                    # IMPORTANT: Decorrelation is applied for detection quality but
                    # photometry should be performed on the original (non-decorrelated)
                    # difference image to preserve proper noise characteristics for
                    # error estimation. The decorrelated image is saved for reference
                    # but photometry pipelines should use the original difference image.
                    log_info(
                        "NOTE: Decorrelation applied for improved detection quality. "
                        "For photometry, consider using the original (non-decorrelated) "
                        "difference image to preserve proper noise characteristics."
                    )
                    
                    # Save original (non-decorrelated) difference image if requested
                    if save_original_diff and FITS_DIFF:
                        _orig_diff_path = FITS_DIFF.replace(".fits", "_original.fits")
                        try:
                            _orig_hdr = diff_hdr.copy()
                            _orig_hdr["DECORR"] = (False, "Original (non-decorrelated) difference image")
                            _orig_hdr["COMMENT"] = "Use this image for photometry to preserve noise characteristics"
                            safe_fits_write(_orig_diff_path, _diff_arr_original.astype(np.float32), _orig_hdr, overwrite=True)
                            log_info(f"Saved original (non-decorrelated) difference image: {_orig_diff_path}")
                        except Exception as _save_e:
                            log_info(f"Warning: Could not save original difference image: {_save_e}")
                    
                decorr_status = "ON" if _applied_decorr else "OFF (kernel unavailable)"
                vscale_status = "ON" if _applied_vscale else "OFF (insufficient data)"
                log_info(f"Post-subtraction summary: decorrelation={decorr_status}, variance scaling={vscale_status}")

                hdul[0].data = diff_arr.astype(np.float32)
                hdul[0].header = diff_hdr

        except Exception as _post_e:
            log_info(f"Warning: post-subtraction processing failed ({_post_e}); skipping.")

        # ------------------------------------------------------------------
        # Re-impose invalid-pixel mask on output difference image
        #
        # Some subtraction / resampling steps can emit exact zeros in regions
        # where either input image had NaNs or zeros (chip gaps / no-data / SWarp
        # padding). Those pixels should remain "invalid" and propagate as NaNs,
        # otherwise downstream background/SNR/limits can be biased.
        # ------------------------------------------------------------------
        try:
            if np.any(combined_invalid_mask) and FITS_DIFF and os.path.isfile(FITS_DIFF):
                with fits.open(FITS_DIFF, mode="update", memmap=False) as hdul:
                    diff = np.asarray(hdul[0].data, dtype=float)
                    if diff.shape == combined_invalid_mask.shape:
                        n_before = int(np.count_nonzero(~np.isfinite(diff)))
                        diff[combined_invalid_mask] = np.nan
                        hdul[0].data = diff
                        hdul.flush()
                        n_after = int(np.count_nonzero(~np.isfinite(diff)))
                        n_mask = int(np.count_nonzero(combined_invalid_mask))
                        log_info(
                            f"Applied combined invalid mask to diff: NaN/inf {n_before} -> {n_after} "
                            f"(mask={n_mask} px)"
                        )
                    else:
                        log_info(
                            f"Warning: combined_invalid_mask shape {combined_invalid_mask.shape} "
                            f"!= diff shape {diff.shape}; cannot reapply invalid mask."
                        )
        except Exception as e:
            log_info(f"Warning: failed to reapply invalid mask to diff: {e}")

        t1_sfft = time.time()
        log_info(f"SFFT core elapsed: {t1_sfft - t0_sfft:.3f} s")

        # --- Save Fitted-Pixel Visuals (sequential to avoid extra processes / semaphore leaks) ---
        if args.plot:

            def save_fitted_pix(label: str, fits_path: str) -> None:
                try:
                    pix_a = prep_data.get(f"PixA_{label}")
                    active_mask = prep_data.get("Active-Mask")
                    if pix_a is None or active_mask is None:
                        return
                    sky, sky_sig = SkyLevel_Estimator.SLE(PixA_obj=pix_a)
                    noise = np.random.normal(sky, sky_sig, pix_a.shape)
                    pix_a_vis = pix_a.copy()
                    pix_a_vis[~active_mask] = noise[~active_mask]
                    out_path = os.path.join(
                        out_dir,
                        f"{os.path.basename(fits_path).replace('.fits', '')}.fittedPix.fits",
                    )
                    with fits.open(fits_path, mode="readonly") as hdl:
                        header_copy = hdl[0].header.copy()
                    safe_fits_write(out_path, pix_a_vis.T, header_copy)
                except Exception as e:
                    log_info(f"Warning: Failed to save fitted-pixel for {label}: {e}")

            for _label, _path in [("SCI", FITS_SCI), ("REF", FITS_REF)]:
                save_fitted_pix(_label, _path)

        # --- Diagnostic Plot (Parallel) ---
        if args.plot:

            def generate_plot() -> None:
                try:
                    import matplotlib

                    matplotlib.use("agg")
                    import matplotlib.pyplot as plt

                    ast_ss = prep_data.get("SExCatalog-SubSource")
                    if ast_ss is None or len(ast_ss) == 0:
                        return
                    required_cols = ["MAG_REF_REF", "MAGERR_REF_REF", "MAG_REF_SCI", "MAGERR_REF_SCI"]
                    # Handle both pandas DataFrame and astropy Table
                    col_names = ast_ss.columns if hasattr(ast_ss, 'columns') else ast_ss.dtype.names
                    missing_cols = [c for c in required_cols if c not in col_names]
                    if missing_cols:
                        log_info(
                            f"Warning: Diagnostic plot skipped - missing columns: {missing_cols}. "
                            "SFFT catalog schema may differ from expected version."
                        )
                        return
                    
                    # Get input sources count from matching_sources
                    n_input = len(matching_sources) if matching_sources is not None else 0
                    n_final = len(ast_ss)
                    
                    x_data = ast_ss["MAG_REF_REF"]
                    ex_data = ast_ss["MAGERR_REF_REF"]
                    y_data = ast_ss["MAG_REF_SCI"] - ast_ss["MAG_REF_REF"]
                    ey_data = ast_ss["MAGERR_REF_SCI"]
                    median = np.median(y_data)
                    lower, upper = (
                        median - CVREJ_MAGD_THRESH,
                        median + CVREJ_MAGD_THRESH,
                    )
                    fig, ax = plt.subplots(
                        figsize=set_size(540, aspect=1.01), dpi=150
                    )
                    ax.errorbar(
                        x_data,
                        y_data,
                        xerr=ex_data,
                        yerr=ey_data,
                        fmt="o",
                        markersize=3.5,
                        mfc="none",
                        capsize=2.5,
                        elinewidth=0.9,
                        markeredgewidth=0.9,
                        alpha=0.85,
                        label=f"Final sources ({n_final})",
                    )
                    ax.hlines(
                        [median, lower, upper],
                        xmin=np.nanmin(x_data),
                        xmax=np.nanmax(x_data),
                        linestyles=(0, (5, 2)),
                        linewidth=1.2,
                        colors="C1",
                        label="median / thresholds",
                    )
                    ax.set_xlabel("MAG_REF (REF)")
                    ax.set_ylabel("MAG_REF (SCI) - MAG_REF (REF)")
                    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
                    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
                    ax.set_title(f"SFFT source matching: {n_input} input -> {n_final} final sources", fontsize=10)
                    png_path = os.path.join(out_dir, f"VarCheck_{out_base}.png")
                    try:
                        plt.savefig(
                            png_path, bbox_inches="tight", dpi=150
                        )
                    except Exception:
                        pass
                    plt.close(fig)
                except Exception as e:
                    log_info(f"Warning: Failed to generate diagnostic plot: {e}")

            generate_plot()

    except Exception as e:
        exc_type, _, exc_tb = sys.exc_info()
        fname = (
            os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            if exc_tb
            else "unknown"
        )
        line = exc_tb.tb_lineno if exc_tb else -1
        log_info(f"Fatal Error: {exc_type} in {fname} at line {line}: {e}")
        return None
    finally:
        # Clean up temporary synchronized-mask FITS files if they were created.
        _tmp_paths_to_clean = locals().get("_temp_paths_created", [])
        for _tmp_path in _tmp_paths_to_clean:
            try:
                if os.path.isfile(_tmp_path):
                    os.remove(_tmp_path)
                    log_info(f"Cleaned up temp file: {_tmp_path}")
            except Exception:
                pass

    log_info(f"Total elapsed: {time.time() - t0_total:.3f} s")
    return 1


if __name__ == "__main__":
    # Ensure repo root is on sys.path when run as a standalone script.
    # This avoids `ModuleNotFoundError: No module named 'functions'` when the
    # current working directory is not the project root.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    sys.exit(0 if run_sfft() else 1)
