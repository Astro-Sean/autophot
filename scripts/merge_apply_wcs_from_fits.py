#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge WCS keywords from one FITS into another (data pixels unchanged).

This is useful when you solve astrometry in an external tool (e.g. PixInsight)
and want to apply the solved WCS to an existing FITS without resampling.

Typical workflow for PixInsight:
  1) Run PixInsight ImageSolver / distortion correction on `input.fits`.
  2) Export a FITS that contains the solved WCS (often the solver can write out
     a new FITS or you can save the solved WCS into a FITS).
  3) Run this script to copy the WCS keywords from that exported FITS into
     your original science FITS.

Usage:
  python3 scripts/merge_apply_wcs_from_fits.py \
    --input /path/to/original.fits \
    --wcs-source /path/to/pixinsight_solved.fits \
    --output /path/to/original_with_wcs.fits
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils


WCS_PREFIXES = (
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
WCS_STEMS_NONLINEAR = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")


def is_non_linear_key(key: str) -> bool:
    if key.startswith("PV"):
        return True
    if key.startswith("SIP_"):
        return True
    parts = key.split("_", 1)
    if len(parts) >= 2:
        stem = parts[0] + "_"
        return stem in WCS_STEMS_NONLINEAR and key.startswith(stem.rstrip("_"))
    return False


def is_wcs_key(key: str) -> bool:
    if key in ("NAXIS", "NAXIS1", "NAXIS2", "COMMENT", "HISTORY"):
        return False
    if any(key.startswith(p) for p in WCS_PREFIXES):
        return True
    if "_" in key:
        stem = key.split("_")[0] + "_"
        return stem in WCS_STEMS_NONLINEAR and key.startswith(stem.rstrip("_"))
    return False


def remove_wcs_from_header(header: fits.Header) -> fits.Header:
    """
    Remove all WCS-related keywords from a header so we can copy a solver WCS
    cleanly without conflicting keys.
    """
    keys = list(header.keys())
    for key in keys:
        if not is_wcs_key(key):
            continue
        try:
            del header[key]
        except Exception:
            pass
    return header


def header_has_non_linear_distortion(header: fits.Header) -> bool:
    for k in header.keys():
        if is_non_linear_key(k):
            return True
    return False


def copy_wcs_keys(src_header: fits.Header, dst_header: fits.Header) -> None:
    for key in src_header.keys():
        if key in ("NAXIS", "NAXIS1", "NAXIS2"):
            continue
        if is_wcs_key(key):
            dst_header[key] = src_header[key]


@dataclass
class MergeMode:
    mode: str = "overwrite"  # overwrite | preserve_input_distortion_linear_only


def merge_wcs(
    input_fits: Path,
    wcs_source_fits: Path,
    output_fits: Path,
    merge_mode: MergeMode,
) -> None:
    with fits.open(input_fits) as hdul:
        data = hdul[0].data
        in_header = hdul[0].header.copy()

    with fits.open(wcs_source_fits) as hdul:
        wcs_header = hdul[0].header

    preserve_input_nl = merge_mode.mode == "preserve_input_distortion_linear_only"
    had_input_nl = header_has_non_linear_distortion(in_header)

    preserved_nl_keys: Dict[str, object] = {}
    preserved_ctype1 = in_header.get("CTYPE1")
    preserved_ctype2 = in_header.get("CTYPE2")
    preserved_lin_keys: Dict[str, object] = {}

    if preserve_input_nl and had_input_nl:
        for k in in_header.keys():
            if is_non_linear_key(k):
                preserved_nl_keys[k] = in_header[k]
            if k in ("CRPIX1", "CRPIX2"):
                preserved_lin_keys[k] = in_header[k]
            if k.startswith(("CD", "PC", "CDELT")) or k == "CROTA":
                preserved_lin_keys[k] = in_header[k]

    remove_wcs_from_header(in_header)

    for key in wcs_header.keys():
        if key in ("NAXIS", "NAXIS1", "NAXIS2"):
            continue
        if not is_wcs_key(key):
            continue

        if preserve_input_nl and had_input_nl:
            # Skip non-linear and projection identifiers so preserved input remains.
            if is_non_linear_key(key):
                continue
            if key in ("CTYPE1", "CTYPE2"):
                continue
            # We already preserved CRPIX/CD/PC; allow linear-term overwrite otherwise.
            if key in preserved_lin_keys:
                continue

        in_header[key] = wcs_header[key]

    # Restore preserved input distortion + linear terms if requested
    if preserve_input_nl and had_input_nl:
        if preserved_ctype1 is not None:
            in_header["CTYPE1"] = preserved_ctype1
        if preserved_ctype2 is not None:
            in_header["CTYPE2"] = preserved_ctype2
        for k, v in preserved_nl_keys.items():
            in_header[k] = v
        for k, v in preserved_lin_keys.items():
            in_header[k] = v

    in_header["NAXIS1"] = int(data.shape[1])
    in_header["NAXIS2"] = int(data.shape[0])

    # Validate roughly
    try:
        w = WCS(in_header)
        scale_val = wcs_utils.proj_plane_pixel_scales(w)[0]
        arcsec_per_pix = float(scale_val * 3600.0) if hasattr(scale_val, "value") else float(scale_val * 3600.0)
        if not np.isfinite(arcsec_per_pix) or arcsec_per_pix <= 0 or arcsec_per_pix > 3600:
            print(f"WARNING: merged WCS suspicious pixel scale: {arcsec_per_pix}")
    except Exception as exc:
        print(f"WARNING: could not validate merged WCS: {exc}")

    fits.writeto(output_fits, data, in_header, overwrite=True, output_verify="ignore")
    print(f"Wrote: {output_fits}")


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description="Merge/copy WCS keywords between FITS files.")
    ap.add_argument("--input", required=True, help="Original FITS whose data/timing should be kept.")
    ap.add_argument("--wcs-source", required=True, help="FITS containing the desired solved WCS (PixInsight output).")
    ap.add_argument("--output", required=True, help="Output FITS path.")
    ap.add_argument(
        "--mode",
        choices=["overwrite", "preserve_input_distortion_linear_only"],
        default="overwrite",
        help="overwrite: copy all WCS keywords from wcs-source. preserve_input_distortion_linear_only: keep input PV/SIP/A/B/D/DP terms.",
    )
    args = ap.parse_args(list(argv))

    input_fits = Path(args.input).expanduser().resolve()
    wcs_source_fits = Path(args.wcs_source).expanduser().resolve()
    output_fits = Path(args.output).expanduser().resolve()

    if not input_fits.exists():
        raise FileNotFoundError(f"--input not found: {input_fits}")
    if not wcs_source_fits.exists():
        raise FileNotFoundError(f"--wcs-source not found: {wcs_source_fits}")

    merge_wcs(
        input_fits=input_fits,
        wcs_source_fits=wcs_source_fits,
        output_fits=output_fits,
        merge_mode=MergeMode(mode=args.mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

