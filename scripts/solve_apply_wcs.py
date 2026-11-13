#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone example: run astrometry.net solve-field and merge the solved WCS
keywords into an existing FITS header without resampling the image.

Usage:
  python3 scripts/solve_apply_wcs.py --input /path/to/image.fits --output /path/to/image_wcs.fits

Notes:
  - This script expects astrometry.net installed and `solve-field` on PATH
    (or use `--solve-field-exe`).
  - It performs only a FITS-header merge (data pixels are unchanged).
  - Default merge mode preserves the input non-linear distortion model
    (PV/SIP/A/B coefficients) and only updates the linear sky mapping
    (CRVAL/coordinate origin). This helps avoid common projection/distortion
    mismatches with downstream reprojection.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils


WCS_PREFIXES_REMOVE = [
    "CRPIX",
    "CRVAL",
    "CTYPE",
    "CD",
    "PC",
    "CDELT",
    "CROTA",
    "PV",
    "LONPOLE",
    "LATPOLE",
    "EQUINOX",
    "WCSNAME",
    "CUNIT",
    "WCSAXES",
    "PROJP",
    "LTV",
    "LTM",
    "RADECSYS",
    "RADESYS",
    "RADYSYS",
    "LONGPOLE",
    "TNX",
    "SIP_",
]

WCS_STEMS_NONLINEAR = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")


def remove_wcs_from_header(header: fits.Header) -> fits.Header:
    """
    Remove common WCS keys from a FITS header so solver WCS can be merged
    without keyword conflicts.
    """
    # Modify in place but return for convenience
    keys = list(header.keys())
    for key in keys:
        if key in ("NAXIS", "NAXIS1", "NAXIS2", "COMMENT", "HISTORY"):
            continue
        remove = False
        for prefix in WCS_PREFIXES_REMOVE:
            if key.startswith(prefix):
                remove = True
                break
        if not remove and "_" in key:
            # Remove polynomial/SIP coefficients too.
            stem = key.split("_")[0] + "_" if "_" in key else ""
            if stem in WCS_STEMS_NONLINEAR and key.startswith(stem.rstrip("_")):
                remove = True
        if remove:
            try:
                del header[key]
            except Exception:
                pass
    return header


def is_non_linear_key(key: str) -> bool:
    if key.startswith("PV"):
        return True
    if key.startswith("SIP_"):
        return True
    # SIP/A/B/D/DP coefficient keys are of the form A_0_0, BP_1_2, etc.
    parts = key.split("_", 1)
    if len(parts) >= 2:
        stem = parts[0] + "_"
        return stem in ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
    return False


def is_linear_key(key: str) -> bool:
    if key in ("CRPIX1", "CRPIX2"):
        return True
    if key.startswith(("CD", "PC", "CDELT")):
        return True
    if key == "CROTA":
        return True
    return False


def is_wcs_key_from_solver(key: str) -> bool:
    # Mirrors the logic in wcs.py: accept standard WCS prefixes plus SIP/PV coefficient stems.
    wcs_prefixes = (
        "CRPIX",
        "CRVAL",
        "CTYPE",
        "CD",
        "PC",
        "CDELT",
        "CROTA",
        "PV",
        "LONPOLE",
        "LATPOLE",
        "EQUINOX",
        "WCSNAME",
        "CUNIT",
        "WCSAXES",
        "PROJP",
        "LTV",
        "LTM",
        "RADECSYS",
        "RADESYS",
        "RADYSYS",
        "LONGPOLE",
        "TNX",
        "SIP_",
    )
    wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
    if any(key.startswith(p) for p in wcs_prefixes):
        return True
    if "_" in key:
        stem = key.split("_")[0] + "_"
        return stem in wcs_stems and key.startswith(stem.rstrip("_"))
    return False


@dataclass
class MergeConfig:
    preserve_nonlinear_distortion: bool = True
    preserve_linear_terms: bool = True
    solve_field_crpix_center: bool = False


def run_solve_field(
    solve_field_exe: str,
    input_fits: Path,
    tmp_dir: Path,
    *,
    downsample: int = 3,
    cpulimit_sec: int = 45,
    tweak_order: int = 2,
    scale_low: float | None = None,
    scale_high: float | None = None,
    ra: float | None = None,
    dec: float | None = None,
    radius_deg: float | None = None,
    crpix_center: bool = False,
) -> Path:
    """
    Returns the path to the solver-produced .wcs.fits file.
    """
    wcs_file = tmp_dir / "astrometry_temp.wcs.fits"
    new_fits_temp = tmp_dir / "astrometry_newfits_temp.fits"
    scamp_temp = tmp_dir / "astrometry_temp_scamp"
    corr_temp = tmp_dir / "astrometry_temp.corr"
    rdl_temp = tmp_dir / "astrometry_temp.rdls"
    match_temp = tmp_dir / "astrometry_temp.match"
    solved_temp = tmp_dir / "astrometry_temp.solved"

    cmd = [
        str(solve_field_exe),
        "--no-remove-lines",
        "--overwrite",
        "--downsample",
        str(downsample),
        "--new-fits",
        str(new_fits_temp),
        "--wcs",
        str(wcs_file),
        "--scamp",
        str(scamp_temp),
        "--corr",
        str(corr_temp),
        "--rdls",
        str(rdl_temp),
        "--match",
        str(match_temp),
        "--solved",
        str(solved_temp),
        "--no-plots",
        "--cpulimit",
        str(int(cpulimit_sec)),
        str(input_fits),
        "--tweak-order",
        str(int(tweak_order)),
        "--no-verify",
    ]

    # Pixel-scale constraints (optional)
    if scale_low is not None and scale_high is not None:
        cmd += [
            "--scale-units",
            "arcsecperpix",
            "--scale-low",
            str(scale_low),
            "--scale-high",
            str(scale_high),
        ]

    # Position constrained solve (optional)
    if ra is not None and dec is not None and radius_deg is not None:
        cmd += ["--ra", str(ra), "--dec", str(dec), "--radius", str(radius_deg)]

    if crpix_center:
        cmd += ["--crpix-center"]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"solve-field failed:\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}")

    if not wcs_file.exists():
        raise FileNotFoundError(f"solve-field did not produce wcs file: {wcs_file}")

    return wcs_file


def merge_solver_wcs_into_header(
    input_fits: Path,
    output_fits: Path,
    solver_wcs_file: Path,
    merge_cfg: MergeConfig,
) -> None:
    print(f"Merging solver WCS from {solver_wcs_file.name} into {output_fits.name}")

    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

    input_keys = list(header.keys())
    had_non_linear = any(is_non_linear_key(k) for k in input_keys)

    preserve_nl = merge_cfg.preserve_nonlinear_distortion and had_non_linear
    preserve_lin = merge_cfg.preserve_linear_terms and had_non_linear

    preserved_nl_keys: Dict[str, object] = {}
    preserved_lin_keys: Dict[str, object] = {}
    preserved_ctype1 = header.get("CTYPE1")
    preserved_ctype2 = header.get("CTYPE2")

    if preserve_nl:
        for k in input_keys:
            if is_non_linear_key(k):
                preserved_nl_keys[k] = header[k]

        # Preserve the projection identifiers too (often important for SIP/PV handling).
        # If CTYPE is missing, restoration is a no-op.
        if preserved_ctype1 is not None:
            preserved_nl_keys["CTYPE1"] = preserved_ctype1
        if preserved_ctype2 is not None:
            preserved_nl_keys["CTYPE2"] = preserved_ctype2

    if preserve_lin:
        for k in input_keys:
            if is_linear_key(k):
                preserved_lin_keys[k] = header[k]

    with fits.open(solver_wcs_file) as wcs_hdul:
        solver_header = wcs_hdul[0].header

    remove_wcs_from_header(header)

    for key in solver_header.keys():
        if key in ("NAXIS", "NAXIS1", "NAXIS2"):
            continue
        if not is_wcs_key_from_solver(key):
            continue

        if preserve_nl and is_non_linear_key(key):
            continue
        if preserve_nl and key in ("CTYPE1", "CTYPE2"):
            continue
        if preserve_lin and is_linear_key(key):
            continue

        header[key] = solver_header[key]

    # Restore preserved distortion model and linear terms
    if preserve_nl:
        for k, v in preserved_nl_keys.items():
            # CTYPE keys may not exist after removal if they were removed by remove_wcs_from_header.
            header[k] = v

    if preserve_lin:
        for k, v in preserved_lin_keys.items():
            header[k] = v

    header["NAXIS1"] = int(data.shape[1])
    header["NAXIS2"] = int(data.shape[0])

    # Validate merged WCS pixel scale roughly
    try:
        w = WCS(header)
        scale_val = wcs_utils.proj_plane_pixel_scales(w)[0]
        # scale_val can be a Quantity or float; normalize to arcsec/pix
        arcsec_per_pix = float(scale_val * 3600.0) if hasattr(scale_val, "value") else float(scale_val * 3600.0)
        if not np.isfinite(arcsec_per_pix) or arcsec_per_pix <= 0 or arcsec_per_pix > 3600:
            print(f"WARNING: merged WCS has suspicious pixel scale: {arcsec_per_pix}")
        else:
            print(f"Merged WCS pixel scale approx: {arcsec_per_pix:.3f} arcsec/pix")
    except Exception as exc:
        print(f"WARNING: could not validate merged WCS: {exc}")

    fits.writeto(output_fits, data, header, overwrite=True, output_verify="ignore")
    print(f"Wrote: {output_fits}")


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description="Run solve-field and merge WCS into an existing FITS header.")
    ap.add_argument("--input", required=True, type=str, help="Input FITS path")
    ap.add_argument("--output", required=True, type=str, help="Output FITS path")
    ap.add_argument("--solve-field-exe", default="solve-field", type=str, help="solve-field executable")
    ap.add_argument("--timeout-sec", default=45, type=int, help="CPU limit for solve-field")
    ap.add_argument("--downsample", default=3, type=int, help="Downsampling for solve-field")
    ap.add_argument("--tweak-order", default=2, type=int, help="solve-field tweak order")

    # Optional sky constraint
    ap.add_argument("--ra", type=float, default=None, help="RA constraint (deg)")
    ap.add_argument("--dec", type=float, default=None, help="Dec constraint (deg)")
    ap.add_argument("--radius-deg", type=float, default=None, help="Search radius for constrained solve (deg)")

    # Optional scale constraints
    ap.add_argument("--scale-low", type=float, default=None, help="scale-low (arcsec/pix)")
    ap.add_argument("--scale-high", type=float, default=None, help="scale-high (arcsec/pix)")

    ap.add_argument(
        "--crpix-center",
        action="store_true",
        help="Pass --crpix-center to solve-field (legacy behavior; may disagree with instrument WCS).",
    )

    # Merge mode: preserve distortion (option 1 style) by default.
    ap.add_argument(
        "--overwrite-fully",
        action="store_true",
        help="Overwrite all WCS keywords from solver (disable preserve distortion defaults).",
    )
    args = ap.parse_args(list(argv))

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input FITS not found: {input_path}")

    solve_field_exe = args.solve_field_exe
    # Let PATH resolution happen if the user gave "solve-field".
    if shutil.which(solve_field_exe) is None and not Path(solve_field_exe).exists():
        raise FileNotFoundError(
            f"solve-field executable not found: {solve_field_exe}. Set --solve-field-exe or ensure it is on PATH."
        )

    merge_cfg = MergeConfig(
        preserve_nonlinear_distortion=not args.overwrite_fully,
        preserve_linear_terms=not args.overwrite_fully,
        solve_field_crpix_center=bool(args.crpix_center),
    )

    with tempfile.TemporaryDirectory(prefix="wcs_solve_apply_") as tmp:
        tmp_dir = Path(tmp)
        solver_wcs_file = run_solve_field(
            solve_field_exe=solve_field_exe,
            input_fits=input_path,
            tmp_dir=tmp_dir,
            downsample=int(args.downsample),
            cpulimit_sec=int(args.timeout_sec),
            tweak_order=int(args.tweak_order),
            scale_low=args.scale_low,
            scale_high=args.scale_high,
            ra=args.ra,
            dec=args.dec,
            radius_deg=args.radius_deg,
            crpix_center=bool(args.crpix_center),
        )

        merge_solver_wcs_into_header(
            input_fits=input_path,
            output_fits=output_path,
            solver_wcs_file=solver_wcs_file,
            merge_cfg=merge_cfg,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

