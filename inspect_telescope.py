#!/usr/bin/env python3
"""
inspect_telescope.py — CLI wrapper for autophot.inspect_telescope()

Usage
-----
    python inspect_telescope.py /path/to/fits/folder [options]

For the full API see:
    from autophot import inspect_telescope
    help(inspect_telescope)
"""

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan a FITS folder and build/update telescope.yml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("fits_dir", help="Folder containing FITS images.")
    p.add_argument("--wdir", default=None, help="Output directory for telescope.yml (default: fits_dir).")
    p.add_argument("--recursive", action="store_true", help="Also scan sub-folders.")
    p.add_argument("--dry-run", action="store_true", help="Print YAML, do not write.")
    p.add_argument("--verbose", action="store_true", help="Per-file diagnostics.")
    p.add_argument("--filter-key", default=None, metavar="KEY", help="Preferred FITS filter header key.")
    p.add_argument("--gain-key", default="GAIN", metavar="KEY")
    p.add_argument("--rn-key", default="RDNOISE", metavar="KEY")
    p.add_argument("--sat-key", default="SATURATE", metavar="KEY")
    p.add_argument("--exptime-key", default="EXPTIME", metavar="KEY")
    p.add_argument("--mjd-key", default="MJD-OBS", metavar="KEY")
    p.add_argument("--date-key", default="DATE-OBS", metavar="KEY")
    p.add_argument("--airmass-key", default="AIRMASS", metavar="KEY")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    from autophot import inspect_telescope

    inspect_telescope(
        fits_dir=args.fits_dir,
        wdir=args.wdir,
        recursive=args.recursive,
        dry_run=args.dry_run,
        verbose=args.verbose,
        filter_key=args.filter_key,
        gain_key=args.gain_key,
        rn_key=args.rn_key,
        sat_key=args.sat_key,
        exptime_key=args.exptime_key,
        mjd_key=args.mjd_key,
        date_key=args.date_key,
        airmass_key=args.airmass_key,
    )


if __name__ == "__main__":
    main()
