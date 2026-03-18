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

# Limit BLAS/OpenMP threads before any scientific imports (avoids libgomp
# "Resource temporarily unavailable" when this script is run as a subprocess
# or alongside other parallel jobs).
for _env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")
# Optional: pin this process to a single CPU (Linux) so SFFT uses one process, one core.
try:
    _pid = os.getpid()
    _aff = os.sched_getaffinity(_pid)
    if _aff:
        _one = {next(iter(_aff))}
        os.sched_setaffinity(_pid, _one)
except (AttributeError, OSError):
    pass

import numpy as np
import pandas as pd
from astropy.io import fits
import sfft  # Import the sfft package to check its version
from sfft.EasySparsePacket import Easy_SparsePacket
from sfft.EasyCrowdedPacket import Easy_CrowdedPacket
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from typing import Optional, Tuple


def _odd(n: int) -> int:
    """Return n if odd, else n+1."""
    n = int(n)
    return n + (n % 2 == 0)



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
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")
    t0_total = time.time()

    def log_info(msg: str) -> None:
        try:
            logger.info(msg)
        except Exception:
            print(msg)  # Fallback if logging not configured

    # --- Print SFFT Version ---
    log_info(f"Using SFFT version: {sfft.__version__}")

    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description='SFFT image subtraction.')
    parser.add_argument('-sci', type=str, required=True, help='Science FITS file path.')
    parser.add_argument('-ref', type=str, required=True, help='Reference FITS file path.')
    parser.add_argument('-diff', type=str, default=None, help='Output difference FITS path.')
    parser.add_argument('-mask', type=str, default=None, help='Boolean mask FITS path.')
    parser.add_argument('-crowded', action='store_true', help='Use crowded-field solver.')

    # Match HOTPANTS: default 0 (constant kernel). Pipeline passes this from config.
    parser.add_argument('-kernel_order', type=int, default=0, help='Kernel polynomial order (0 = constant, like HOTPANTS -ko).')
    parser.add_argument('-bg_order', type=int, default=0,
                        help='Background spatial polynomial order (0=constant). Science and reference are assumed background-subtracted (pipeline subtracts both).')

    parser.add_argument('-kernel_half_width', type=float, default=0, help='Kernel half width.')
    parser.add_argument('-masked_sources', type=str, default='[]',
                        help='List of [x, y] pairs to ban, e.g., "[[12.3, 45.6], [78.9, 10.2]]".')
    parser.add_argument('-matching_sources', type=str, default='[]',
                        help='List of [x, y] pairs to preselect, same format as above.')
    parser.add_argument('-plot', action='store_true', help='Generate diagnostic plots.')

    # Pass gain/saturate values so SFFT does not depend on headers (avoids KeyError when SATURATE missing).
    parser.add_argument('-gain_sci', type=float, default=None, help='Science image gain (e/ADU). If set, written to FITS before SFFT.')
    parser.add_argument('-gain_ref', type=float, default=None, help='Reference image gain (e/ADU).')
    parser.add_argument('-saturate_sci', type=float, default=None, help='Science saturation (ADU). Use a large value (e.g. 1e30) for no limit.')
    parser.add_argument('-saturate_ref', type=float, default=None, help='Reference saturation (ADU).')

    # SExtractor background mesh: smaller BACK_SIZE => finer mesh (better for gradients/hosts),
    # but can overfit noise if too small. Typical values: 32, 64, 128.
    parser.add_argument('-back_size', type=int, default=128, help='SExtractor BACK_SIZE (mesh cell size, px).')
    parser.add_argument('-back_filtersize', type=int, default=6, help='SExtractor BACK_FILTERSIZE (median filter size, mesh cells).')
    args = parser.parse_args()

    # --- Parse Coordinate Lists ---
    def parse_xy_list(s: str) -> Optional[np.ndarray]:
        """
        Parse a string representing a list of [x, y] pairs.

        Accepted formats:
          - "[[x1, y1], [x2, y2]]"
          - "[x1, y1]" (single pair)
          - "[x1, y1, x2, y2]" (flat list, reshaped to N×2)
        """
        s = (s or '').strip()
        if s in ('', '[]', 'None'):
            return None
        try:
            obj = ast.literal_eval(s)
            arr = np.array(obj, dtype=float)
            if arr.ndim == 1:
                if arr.size == 2:
                    arr = arr.reshape(1, 2)
                elif arr.size % 2 == 0:
                    arr = arr.reshape(-1, 2)
                else:
                    raise ValueError("Must contain an even number of values to form [x, y] pairs.")
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("Must be a list of [x, y] pairs.")
            return arr
        except Exception as e:
            log_info(f"Warning: Could not parse list '{s}': {e}. Ignoring.")
            return None

    masked_sources = parse_xy_list(args.masked_sources)
    matching_sources = parse_xy_list(args.matching_sources)

    if masked_sources is not None:
        log_info(f"Masked sources: {masked_sources.shape[0]}")
        
        
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
    out_dir = os.path.dirname(os.path.abspath(FITS_SCI)) or '.'
    FITS_DIFF = args.diff or os.path.join(out_dir, f"diff_{os.path.basename(FITS_SCI)}")

    # --- Load Headers (Once) ---
    def get_fits_info(fits_path: str) -> Tuple[fits.Header, np.ndarray]:
        with fits.open(fits_path) as hdul:
            return hdul[0].header, hdul[0].data

    hdr_sci, data_sci = get_fits_info(FITS_SCI)
    hdr_ref, data_ref = get_fits_info(FITS_REF)

    # --- Ensure GAIN and SATURATE in FITS (pass values, not keywords) ---
    # Write values into headers so SFFT/MeLOn find the keywords (avoids KeyError when SATURATE missing).
    SATURATE_FALLBACK = 1e30  # finite value for "no saturation" (FITS cannot store inf)

    def _ensure_gain_saturate(fits_path: str, gain: Optional[float], saturate: Optional[float], label: str) -> None:
        if gain is None and saturate is None:
            return
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

    def _float_or_default(val, default: float) -> float:
        if val is None:
            return default
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    gain_sci = args.gain_sci if args.gain_sci is not None else _float_or_default(hdr_sci.get("GAIN", hdr_sci.get("gain")), 1.0)
    gain_ref = args.gain_ref if args.gain_ref is not None else _float_or_default(hdr_ref.get("GAIN", hdr_ref.get("gain")), 1.0)
    sat_sci_raw = args.saturate_sci if args.saturate_sci is not None else hdr_sci.get("SATURATE", hdr_sci.get("saturate"))
    sat_ref_raw = args.saturate_ref if args.saturate_ref is not None else hdr_ref.get("SATURATE", hdr_ref.get("saturate"))
    sat_sci = _float_or_default(sat_sci_raw, SATURATE_FALLBACK)
    sat_ref = _float_or_default(sat_ref_raw, SATURATE_FALLBACK)

    _ensure_gain_saturate(FITS_SCI, gain_sci, sat_sci, "Science")
    _ensure_gain_saturate(FITS_REF, gain_ref, sat_ref, "Reference")

    # --- FWHM and Gain ---
    template_fwhm = float(hdr_ref.get('FWHM', 3.0))
    science_fwhm = float(hdr_sci.get('FWHM', 3.0))
    log_info(f"Science FWHM: {science_fwhm:.1f} pixels | Template FWHM: {template_fwhm:.1f} pixels")

    # --- Kernel and Detection Parameters ---
    fwhm_min = min(template_fwhm, science_fwhm)
    fwhm_max = max(template_fwhm, science_fwhm)
    FWHM = max(np.ceil(template_fwhm), np.ceil(science_fwhm))

    psf_area_min = np.pi * (fwhm_min) ** 2
    detect_minarea = min(3, int(np.ceil(psf_area_min * 0.5)))
    detect_maxarea = 0

    # Kernel half-width: user override, or auto. Use 2*FWHM so kernel is large enough
    # to fit both science and template PSFs (user requirement). HOTPANTS uses 1.5*FWHM;
    # slightly larger kernel here can improve SFFT stability.
    if float(args.kernel_half_width) == 0:
        kernel_half_width = int(_odd(max(5, np.ceil(FWHM * 2))))
        log_info(f'Auto kernel half-width: 2 * FWHM = {kernel_half_width} px')
    else:
        kernel_half_width = float(args.kernel_half_width)

    if kernel_half_width % 2 == 0:
        kernel_half_width += 1
        log_info(f'Adjusted kernel half width to odd: {kernel_half_width:.1f} px')
    else:
        log_info(f'Using kernel half width: {kernel_half_width:.1f} px')

    # Boundary: strip where convolution is invalid. Use kernel half-width (minimal)
    # so the output diff image is as large as possible; was 3*fwhm_max which made output narrow.
    boundary = int(kernel_half_width)

    # KerHWLimit (min, max): SFFT clamps kernel half-width to this range.
    # Keep max large enough that large sci/ref FWHM differences are not clamped.
    KER_HW_LIMIT_MIN = 3
    KER_HW_LIMIT_MAX = 50
    KerHWLimit = (KER_HW_LIMIT_MIN, KER_HW_LIMIT_MAX)

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
    BACKEND_4SUBTRACT = 'Numpy'

    CUDA_DEVICE_4SUBTRACT = '0'

    # Use 1 thread for SFFT internals to avoid libgomp/process limits when run
    # as a subprocess or on HPC (avoids "Thread creation failed" and semaphore leaks).
    NUM_CPU_THREADS_4SUBTRACT = 1

    # Convolve REF to match SCI so DIFF = SCI - conv(REF). Transient (science-only) then
    # keeps the science PSF. AUTO can convolve science when it has worse seeing, which
    # makes the transient look wrong (e.g. broader).
    ForceConv = 'REF'
    GAIN_KEY = 'GAIN'
    SATUR_KEY = 'SATURATE'

    kernel_poly_order = args.kernel_order
    bg_poly_order = max(0, int(args.bg_order))
    log_info(f"Background polynomial order: {bg_poly_order}")

    # --- SExtractor parameters (both sparse and crowded) ---
    # BACKPHOTO_TYPE: GLOBAL so SExtractor does not estimate local background at star positions.
    # LOCAL can bias local backgrounds and drive oversubtraction.
    # DETECT_THRESH: higher = fewer spurious sources (stabilizes matching/solution).
    is_crowded = args.crowded

    DETECT_THRESH = 2.8 if is_crowded else 2.5
    DEBLEND_MINCON = 0.0003 if is_crowded else 0.0005
    # Let SFFT solve the photometric scale ratio between REF and SCI.
    # Forcing a constant ratio can drive systematic over/under-subtraction when
    # throughput/zeropoint differs between images.
    constant_phot_ratio = False
    StarExt_iter = 4
    BACKPHOTO_TYPE = "GLOBAL"
    if is_crowded:
        log_info(
            f"Crowded-field SFFT: BACKPHOTO_TYPE={BACKPHOTO_TYPE}, "
            f"DETECT_THRESH={DETECT_THRESH:.1f}, DEBLEND_MINCONT={DEBLEND_MINCON:.4f}"
        )
    else:
        log_info(
            f"Sparse-field SFFT: BACKPHOTO_TYPE={BACKPHOTO_TYPE}, "
            f"DETECT_THRESH={DETECT_THRESH:.1f}, DEBLEND_MINCONT={DEBLEND_MINCON:.4f}"
        )

    BACK_SIZE = int(max(16, args.back_size))
    BACK_FILTERSIZE = int(max(1, args.back_filtersize))
    log_info(f"SExtractor background mesh: BACK_SIZE={BACK_SIZE} px, BACK_FILTERSIZE={BACK_FILTERSIZE}")

    COARSE_VAR_REJECTION = True
    # UPDATED: tighter photometric dev threshold
    CVREJ_MAGD_THRESH = 0.1
    ELABO_VAR_REJECTION = True

    # KerHWRatio: ratio between FWHM and kernel half-width when GKerHW is None.
    # SFFT source (EasySparsePacket.ESP): KerHW = int(clip(KerHWRatio * Max(FWHM_REF, FWHM_SCI), limit[0], limit[1])).
    # We pass GKerHW explicitly so this is only used as fallback; keep SFFT default 2.0.
    KerHWRatio = 2.0

    # --- Load Mask (Optional) ---
    prior_ban_mask = None
    if args.mask:
        try:
            prior_ban_mask = fits.getdata(args.mask).T
            if prior_ban_mask.dtype != bool:
                prior_ban_mask = prior_ban_mask.astype(bool)
        except Exception as e:
            log_info(f"Warning: Could not load mask '{args.mask}': {e}")

    # --- Run SFFT ---
    t0_sfft = time.time()
    try:
        if args.crowded:
            # Crowded-field subtraction (ECP): no prior source list; uses SExtractor + masking.
            log_info("Running crowded-field subtraction (ECP).")
            fits_solution = os.path.join(out_dir, "sfft_solution.fits") if FITS_DIFF else None
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
                MaskSatContam=True,
                GAIN_KEY=GAIN_KEY,
                SATUR_KEY=SATUR_KEY,
                BACK_TYPE='MANUAL',
                BACK_VALUE=0.0,
                BACK_SIZE=BACK_SIZE,
                BACK_FILTERSIZE=BACK_FILTERSIZE,
                DETECT_THRESH=DETECT_THRESH,
                ANALYSIS_THRESH=DETECT_THRESH,
                DETECT_MINAREA=detect_minarea,
                DETECT_MAXAREA=detect_maxarea,
                DEBLEND_MINCONT=DEBLEND_MINCON,
                BACKPHOTO_TYPE=BACKPHOTO_TYPE,
                ONLY_FLAGS=None,
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
            diff_image = result[0] if result else None
            prep_data = result[1] if len(result) > 1 else {}
            # ECP may write FITS_DIFF itself; if we have diff_image and file missing, write it
            if diff_image is not None and FITS_DIFF and not os.path.isfile(FITS_DIFF):
                try:
                    with fits.open(FITS_SCI, mode='readonly') as hdl:
                        hdu = hdl[0].copy()
                        # SFFT uses (ny, nx) row-major; FITS is (nx, ny) in header
                        hdu.data = diff_image.T if diff_image.shape[0] != hdu.data.T.shape[0] else diff_image
                        hdu.writeto(FITS_DIFF, overwrite=True)
                except Exception as e:
                    log_info(f"Warning: Could not write ECP diff to {FITS_DIFF}: {e}")
            # Optional: write a minimal matching-sources CSV from ECP catalog if present
            cat_key = 'SExCatalog-SubSource'
            if cat_key in prep_data:
                catalog = prep_data[cat_key]
                matched_sources = catalog.to_pandas() if hasattr(catalog, 'to_pandas') else pd.DataFrame(catalog)
                log_info(f'Number of sources used in crowded-field matching: {len(matched_sources)}')
                xcol = 'X_IMAGE_REF_SCI_MEAN' if 'X_IMAGE_REF_SCI_MEAN' in matched_sources.columns else None
                ycol = 'Y_IMAGE_REF_SCI_MEAN' if 'Y_IMAGE_REF_SCI_MEAN' in matched_sources.columns else None
                if xcol is None:
                    for a, b in [('x_center', 'y_center'), ('X_IMAGE_REF_SCI_MEAN', 'Y_IMAGE_REF_SCI_MEAN')]:
                        if a in matched_sources.columns and b in matched_sources.columns:
                            xcol, ycol = a, b
                            break
                out_csv = os.path.join(out_dir, 'sfft_matching_sources.csv')
                if xcol and ycol:
                    df_out = matched_sources[[xcol, ycol]].rename(
                        columns={xcol: 'X_IMAGE_REF_SCI_MEAN', ycol: 'Y_IMAGE_REF_SCI_MEAN'}
                    )
                    df_out.to_csv(out_csv, index=False)
                else:
                    matched_sources.to_csv(out_csv, index=False)
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
                    BACK_TYPE='MANUAL',
                    BACK_VALUE=0.0,
                    BACK_SIZE=BACK_SIZE,
                    BACK_FILTERSIZE=BACK_FILTERSIZE,
                    BACKPHOTO_TYPE=BACKPHOTO_TYPE,
                    BGPolyOrder=bg_poly_order,
                    DETECT_THRESH=DETECT_THRESH,
                    DETECT_MINAREA=detect_minarea,
                    DETECT_MAXAREA=detect_maxarea,
                    DEBLEND_MINCONT=DEBLEND_MINCON,
                    MaskSatContam=True,
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
                    EVREJ_RATIO_THREH=5.0,
                    # UPDATED: slightly tighter safety mag deviation
                    EVREJ_SAFE_MAGDEV=0.03,
                    StarExt_iter=StarExt_iter,
                    PostAnomalyCheck=True,
                    PAC_RATIO_THRESH=2.5,
                    BoundarySIZE=boundary,
                    ONLY_FLAGS=None,
                    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT,
                    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT,
                    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT
                )
            except np.linalg.LinAlgError as e:
                # Degenerate kernel design matrix (e.g. too few / collinear sources
                # after applying XY_PriorSelect / XY_PriorBan). Retry once letting
                # SFFT perform its own source matching with no priors.
                log_info(
                    f"SFFT ESP failed with singular matrix when using priors ({e}); "
                    "retrying without prior-selected / prior-banned sources."
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
                    BACK_TYPE='MANUAL',
                    BACK_VALUE=0.0,
                    BACK_SIZE=BACK_SIZE,
                    BACK_FILTERSIZE=BACK_FILTERSIZE,
                    BACKPHOTO_TYPE=BACKPHOTO_TYPE,
                    BGPolyOrder=bg_poly_order,
                    DETECT_THRESH=DETECT_THRESH,
                    DETECT_MINAREA=detect_minarea,
                    DETECT_MAXAREA=detect_maxarea,
                    DEBLEND_MINCONT=DEBLEND_MINCON,
                    MaskSatContam=True,
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
                    EVREJ_RATIO_THREH=5.0,
                    EVREJ_SAFE_MAGDEV=0.03,
                    StarExt_iter=StarExt_iter,
                    PostAnomalyCheck=True,
                    PAC_RATIO_THRESH=2.5,
                    BoundarySIZE=boundary,
                    ONLY_FLAGS=[0, 1],
                    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT,
                    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT,
                    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT
                )
            # ESP returns (diff_image, prep_data, ...); support tuple or list.
            if len(result) < 2:
                raise ValueError(f"SFFT ESP returned {len(result)} value(s), expected at least 2")
            diff_image, prep_data = result[0], result[1]
            cat_key = 'SExCatalog-SubSource'
            if cat_key not in prep_data:
                raise KeyError(f"SFFT prep_data missing '{cat_key}'; incompatible SFFT version?")
            catalog = prep_data[cat_key]
            matched_sources = catalog.to_pandas() if hasattr(catalog, 'to_pandas') else pd.DataFrame(catalog)

            log_info(f'Number of sources used in matching: {len(matched_sources)}')

            # main.py expects columns X_IMAGE_REF_SCI_MEAN, Y_IMAGE_REF_SCI_MEAN
            xcol = 'X_IMAGE_REF_SCI_MEAN' if 'X_IMAGE_REF_SCI_MEAN' in matched_sources.columns else None
            ycol = 'Y_IMAGE_REF_SCI_MEAN' if 'Y_IMAGE_REF_SCI_MEAN' in matched_sources.columns else None
            if xcol is None or ycol is None:
                for a, b in [('x_center', 'y_center'), ('X_IMAGE_REF_SCI_MEAN', 'Y_IMAGE_REF_SCI_MEAN')]:
                    if a in matched_sources.columns and b in matched_sources.columns:
                        xcol, ycol = a, b
                        break
            out_csv = os.path.join(out_dir, 'sfft_matching_sources.csv')
            if xcol and ycol:
                df_out = matched_sources[[xcol, ycol]].rename(
                    columns={xcol: 'X_IMAGE_REF_SCI_MEAN', ycol: 'Y_IMAGE_REF_SCI_MEAN'}
                )
                df_out.to_csv(out_csv, index=False)
            else:
                matched_sources.to_csv(out_csv, index=False)

        try:
            convd = fits.getheader(FITS_DIFF).get('CONVD', 'UNKNOWN')
            log_info(f"[{convd}] is convolved in subtraction.")
        except Exception:
            pass

        t1_sfft = time.time()
        log_info(f"SFFT core elapsed: {t1_sfft - t0_sfft:.3f} s")

        # --- Save Fitted-Pixel Visuals (sequential to avoid extra processes / semaphore leaks) ---
        if args.plot:
            def save_fitted_pix(label: str, fits_path: str) -> None:
                try:
                    pix_a = prep_data.get(f'PixA_{label}')
                    active_mask = prep_data.get('Active-Mask')
                    if pix_a is None or active_mask is None:
                        return
                    sky, sky_sig = SkyLevel_Estimator.SLE(PixA_obj=pix_a)
                    noise = np.random.normal(sky, sky_sig, pix_a.shape)
                    pix_a_vis = pix_a.copy()
                    pix_a_vis[~active_mask] = noise[~active_mask]
                    out_path = os.path.join(
                        out_dir,
                        f"{os.path.basename(fits_path).replace('.fits', '')}.fittedPix.fits"
                    )
                    with fits.open(fits_path, mode='readonly') as hdl:
                        hdl[0].data = pix_a_vis.T
                        hdl.writeto(out_path, overwrite=True)
                except Exception as e:
                    log_info(f"Warning: Failed to save fitted-pixel for {label}: {e}")

            for _label, _path in [('SCI', FITS_SCI), ('REF', FITS_REF)]:
                save_fitted_pix(_label, _path)

        # --- Diagnostic Plot (Parallel) ---
        if args.plot:
            def generate_plot() -> None:
                try:
                    import matplotlib
                    matplotlib.use('agg')
                    import matplotlib.pyplot as plt
                    ast_ss = prep_data.get('SExCatalog-SubSource')
                    if ast_ss is None or len(ast_ss) == 0:
                        return
                    x_data = ast_ss['MAG_REF_REF']
                    ex_data = ast_ss['MAGERR_REF_REF']
                    y_data = ast_ss['MAG_REF_SCI'] - ast_ss['MAG_REF_REF']
                    ey_data = ast_ss['MAGERR_REF_SCI']
                    median = np.median(y_data)
                    lower, upper = median - CVREJ_MAGD_THRESH, median + CVREJ_MAGD_THRESH
                    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
                    ax.errorbar(
                        x_data, y_data, xerr=ex_data, yerr=ey_data,
                        fmt='o', markersize=3.5, mfc='none', capsize=2.5,
                        elinewidth=0.9, markeredgewidth=0.9, alpha=0.85,
                        label='MAG(SCI) - MAG(REF)'
                    )
                    ax.hlines([median, lower, upper], xmin=np.nanmin(x_data), xmax=np.nanmax(x_data),
                              linestyles=(0, (5, 2)), linewidth=1.2, colors='C1',
                              label='median / thresholds')
                    ax.set_xlabel('MAG_REF (REF)')
                    ax.set_ylabel('MAG_REF (SCI) - MAG_REF (REF)')
                    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
                    ax.legend(fontsize=9, framealpha=1)
                    plt.savefig(os.path.join(out_dir, 'varcheck.pdf'), bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    log_info(f"Warning: Failed to generate diagnostic plot: {e}")

            generate_plot()

    except Exception as e:
        exc_type, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1] if exc_tb else 'unknown'
        line = exc_tb.tb_lineno if exc_tb else -1
        log_info(f"Fatal Error: {exc_type} in {fname} at line {line}: {e}")
        return None

    log_info(f"Total elapsed: {time.time() - t0_total:.3f} s")
    return 1


if __name__ == "__main__":
    sys.exit(0 if run_sfft() else 1)
