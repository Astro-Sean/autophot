from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .builder import GaiaCurveCatalogBuilder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build custom AutoPHoT catalog from Gaia XP + transmission curves."
    )
    parser.add_argument("--ra", type=float, required=True, help="Target RA [deg]")
    parser.add_argument("--dec", type=float, required=True, help="Target Dec [deg]")
    parser.add_argument(
        "--radius-deg", type=float, default=0.1, help="Search radius [deg]"
    )
    parser.add_argument(
        "--max-sources", type=int, default=1000, help="Max Gaia sources to process"
    )
    parser.add_argument(
        "--curve",
        action="append",
        default=[],
        metavar="BAND=PATH",
        help="Local curve mapping. Repeat per band, e.g. --curve r=/tmp/r.dat",
    )
    parser.add_argument(
        "--svo-filter",
        action="append",
        default=[],
        metavar="BAND=SVO_ID",
        help="Fetch passband from SVO FPS, e.g. --svo-filter r=SDSS/SDSS.r",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path for AutoPHoT custom catalog",
    )
    parser.add_argument(
        "--curve-cache-dir",
        type=Path,
        default=Path("svo_curves"),
        help="Directory to cache SVO-downloaded curves",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    builder = GaiaCurveCatalogBuilder()
    local_curves = builder.parse_mapping(args.curve, "--curve")
    svo_curves = builder.parse_mapping(args.svo_filter, "--svo-filter")

    builder.build(
        ra_deg=args.ra,
        dec_deg=args.dec,
        radius_deg=args.radius_deg,
        max_sources=args.max_sources,
        curves=local_curves,
        out_csv=args.out_csv,
        svo_filters=svo_curves if svo_curves else None,
        curve_cache_dir=args.curve_cache_dir,
    )


if __name__ == "__main__":
    main()

