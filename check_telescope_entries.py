#!/usr/bin/env python3
"""
check_telescope_entries.py
===========================
Standalone script to check a directory for new telescope.yml entries.

Scans FITS images in a directory and compares against an existing
telescope.yml, reporting only what would be new (new telescopes,
new instruments, new filter mappings, or new metadata keys).

Usage
-----
    python check_telescope_entries.py /path/to/fits/dir [--wdir /path/to/wdir] [--verbose]

Options
-------
    fits_dir        Directory containing FITS images.
    --wdir          Working directory containing telescope.yml [default: fits_dir]
    --recursive     Also scan sub-folders.
    --verbose       Show per-file details.
"""

import argparse
import copy
import os
import sys
import logging

import numpy as np
import yaml
from astropy.io import fits as _fits

FITS_EXTENSIONS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")
FILTER_CANDIDATES = ["FILTER", "FILTER1", "FILTER2", "FILTERID", "INSFILTE", "FILTNAM1"]
AVOID_FILTERS = {"clear", "open", "none", "", "n/a"}

log = logging.getLogger("check_telescope_entries")


def _find_files(d, rec):
    out = []
    if rec:
        for root, _, names in os.walk(d):
            for n in names:
                if any(n.lower().endswith(e) for e in FITS_EXTENSIONS):
                    out.append(os.path.join(root, n))
    else:
        for n in os.listdir(d):
            if any(n.lower().endswith(e) for e in FITS_EXTENSIONS):
                out.append(os.path.join(d, n))
    return sorted(out)


def _read_header(fpath):
    with _fits.open(fpath, ignore_missing_end=True, memmap=False) as hdul:
        hdul.verify("silentfix+ignore")
        for hdu in hdul:
            if any(k.upper() == "TELESCOP" for k in hdu.header.keys()):
                return hdu.header.copy()
        return hdul[0].header.copy()


def _pixel_scale(header):
    try:
        from astropy.wcs import WCS, utils as wcsutils
        with np.errstate(all="ignore"):
            wcs_obj = WCS(header, naxis=2)
            scales = wcsutils.proj_plane_pixel_scales(wcs_obj)
            if scales is not None and len(scales) > 0:
                ps = float(scales[0]) * 3600.0
                if np.isfinite(ps) and 0 < ps <= 20:
                    return round(ps, 4)
    except Exception:
        pass
    return None


def _hval(header, key, default=None):
    try:
        if key and key in header:
            return header[key]
    except Exception:
        pass
    return default


def _read_filter(header, forced_key):
    candidates = ([forced_key] + FILTER_CANDIDATES) if forced_key else FILTER_CANDIDATES
    for key in candidates:
        val = _hval(header, key)
        if val is not None and str(val).strip().lower() not in AVOID_FILTERS:
            return str(val).strip(), key
    return None, None


def _load_yml(path):
    if os.path.isfile(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _diff_dicts(new, old, path=""):
    """Recursively find keys in new that are not in old."""
    missing = {}
    for k, v in new.items():
        full_path = f"{path}.{k}" if path else k
        if k not in old:
            missing[k] = v
        elif isinstance(v, dict) and isinstance(old[k], dict):
            sub_missing = _diff_dicts(v, old[k], full_path)
            if sub_missing:
                missing[k] = sub_missing
    return missing


def check_directory(
    fits_dir: str,
    wdir: str,
    recursive: bool = False,
    verbose: bool = False,
):
    fits_dir = os.path.abspath(str(fits_dir))
    wdir_path = os.path.abspath(str(wdir))
    yml_path = os.path.join(wdir_path, "telescope.yml")

    if not os.path.isdir(fits_dir):
        log.error("fits_dir does not exist: %s", fits_dir)
        sys.exit(1)

    existing = _load_yml(yml_path)
    if not existing:
        log.info("No existing telescope.yml found at %s — all entries would be new.", yml_path)

    files = _find_files(fits_dir, recursive)
    if not files:
        log.warning("No FITS files found in %s", fits_dir)
        return

    log.info("Scanning %d file(s) in %s", len(files), fits_dir)

    seen: dict = {}
    skipped = 0

    for fpath in files:
        try:
            header = _read_header(fpath)
        except Exception as exc:
            log.debug("Cannot read %s: %s", fpath, exc)
            skipped += 1
            continue

        tele = _hval(header, "TELESCOP")
        inst = _hval(header, "INSTRUME")
        if not tele or not inst:
            if verbose:
                log.info("  SKIP (no TELESCOP/INSTRUME): %s", os.path.basename(fpath))
            skipped += 1
            continue

        tele, inst = str(tele).strip(), str(inst).strip()
        fval, fhdr_key = _read_filter(header, None)
        ps = _pixel_scale(header)

        if (tele, inst) not in seen:
            seen[(tele, inst)] = {
                "filters": {}, "filter_header_key": None, "pixel_scales": [],
                "gain_keys": set(), "rn_keys": set(), "sat_keys": set(),
                "exptime_keys": set(), "mjd_keys": set(), "date_keys": set(),
                "airmass_keys": set(),
            }
        rec = seen[(tele, inst)]

        if fval:
            rec["filters"][fval] = rec["filters"].get(fval, 0) + 1
        if fhdr_key and rec["filter_header_key"] is None:
            rec["filter_header_key"] = fhdr_key
        if ps is not None:
            rec["pixel_scales"].append(ps)

        for attr, kw in [
            ("gain_keys", "GAIN"), ("rn_keys", "RDNOISE"), ("sat_keys", "SATURATE"),
            ("exptime_keys", "EXPTIME"), ("mjd_keys", "MJD-OBS"),
            ("date_keys", "DATE-OBS"), ("airmass_keys", "AIRMASS"),
        ]:
            if kw and kw in header:
                rec[attr].add(kw)

        if verbose:
            log.info(
                "  %-40s  TELESCOP=%-20s INSTRUME=%-15s FILTER=%-8s ps=%s",
                os.path.basename(fpath), tele, inst,
                fval or "?", f'{ps:.3f}"' if ps else "N/A",
            )

    if skipped:
        log.info("Skipped %d file(s).", skipped)

    if not seen:
        log.warning("No valid telescope/instrument entries found.")
        return

    # Build what would be generated
    generated: dict = {}
    for (tele, inst), rec in seen.items():
        filter_hdr = rec["filter_header_key"] or "FILTER"
        ps_list = rec["pixel_scales"]
        ps_med = float(np.median(ps_list)) if ps_list else None

        def _best_key(found_set, default):
            return next(iter(found_set), default)

        inst_block: dict = {
            "filter_key_0": filter_hdr,
            "mjd": _best_key(rec["mjd_keys"], "MJD-OBS"),
            "date": _best_key(rec["date_keys"], "DATE-OBS"),
            "gain": _best_key(rec["gain_keys"], "GAIN"),
            "saturate": _best_key(rec["sat_keys"], "SATURATE"),
            "readnoise": _best_key(rec["rn_keys"], "RDNOISE"),
            "airmass": _best_key(rec["airmass_keys"], "AIRMASS"),
            "exptime": _best_key(rec["exptime_keys"], "EXPTIME"),
        }
        if ps_med is not None:
            inst_block["pixel_scale"] = round(ps_med, 4)
        for fv in sorted(rec["filters"]):
            if fv not in inst_block:
                inst_block[fv] = fv

        tele_block = generated.setdefault(tele, {})
        ib = tele_block.setdefault("INSTRUME", {})
        ib[inst] = inst_block

    # Diff against existing
    new_entries = _diff_dicts(generated, existing)

    if not new_entries:
        print("\nNo new telescope.yml entries would be added.")
        return

    print("\n" + "=" * 60)
    print("NEW TELESCOPE.YML ENTRIES (would be added):")
    print("=" * 60)
    print(yaml.dump(new_entries, default_flow_style=False, sort_keys=False, allow_unicode=True))
    print("=" * 60)

    # Summary
    new_telescopes = len(new_entries)
    new_instruments = sum(len(v.get("INSTRUME", {})) for v in new_entries.values())
    print(f"\nSummary: {new_telescopes} new telescope(s), {new_instruments} new instrument(s)")


def _build_parser():
    p = argparse.ArgumentParser(
        description="Check a directory for new telescope.yml entries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("fits_dir", help="Directory containing FITS images.")
    p.add_argument("--wdir", default=None, help="Working dir with telescope.yml [default: fits_dir].")
    p.add_argument("--recursive", action="store_true", help="Also scan sub-folders.")
    p.add_argument("--verbose", action="store_true", help="Show per-file details.")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    fits_dir = os.path.abspath(args.fits_dir)
    wdir = os.path.abspath(args.wdir) if args.wdir else fits_dir

    check_directory(fits_dir, wdir, recursive=args.recursive, verbose=args.verbose)


if __name__ == "__main__":
    main()
