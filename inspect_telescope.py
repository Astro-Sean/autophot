#!/usr/bin/env python3
"""
inspect_telescope.py
====================
Standalone utility to scan a folder of FITS images and build / update
``telescope.yml`` in that folder's working directory.

For every image it discovers:
  - reads TELESCOP / INSTRUME headers
  - reads the filter name from candidate header keys
  - derives pixel scale from the WCS (arcsec/pixel)
  - reads gain, readnoise, saturate, exptime, mjd, date, airmass header keywords
  - generates a best-guess ``telescope.yml`` entry

The output file is written to ``<wdir>/telescope.yml`` (default: the input
folder itself).  If the file already exists it is deep-merged rather than
overwritten so that manual customisations are preserved.

Usage
-----
    python inspect_telescope.py /path/to/fits/folder [--wdir /path/to/output] [--dry-run] [--verbose]

Options
-------
    fits_dir        Folder containing FITS images (searched recursively
                    with --recursive, or top-level only by default).
    --wdir          Output directory for telescope.yml  [default: fits_dir]
    --recursive     Also scan sub-folders.
    --dry-run       Print the generated YAML but do not write anything.
    --verbose       Print per-file diagnostics.
    --filter-key    FITS header keyword to read filter from  [default: auto]
    --gain-key      FITS header keyword for gain             [default: GAIN]
    --rn-key        FITS header keyword for read noise       [default: RDNOISE]
    --sat-key       FITS header keyword for saturation       [default: SATURATE]
    --exptime-key   FITS header keyword for exposure time    [default: EXPTIME]
    --mjd-key       FITS header keyword for MJD              [default: MJD-OBS]
    --date-key      FITS header keyword for date             [default: DATE-OBS]
    --airmass-key   FITS header keyword for airmass          [default: AIRMASS]
"""

import argparse
import copy
import os
import sys
import glob
import logging

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FITS_EXTENSIONS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")
FILTER_HEADER_CANDIDATES = ["FILTER", "FILTER1", "FILTER2", "FILTERID", "INSFILTE", "FILTNAM1"]
AVOID_FILTER_VALUES = {"clear", "open", "none", "", "n/a"}

log = logging.getLogger("inspect_telescope")


def _find_fits_files(fits_dir: str, recursive: bool = False):
    files = []
    if recursive:
        for root, _dirs, names in os.walk(fits_dir):
            for n in names:
                if any(n.lower().endswith(ext) for ext in FITS_EXTENSIONS):
                    files.append(os.path.join(root, n))
    else:
        for n in os.listdir(fits_dir):
            if any(n.lower().endswith(ext) for ext in FITS_EXTENSIONS):
                files.append(os.path.join(fits_dir, n))
    return sorted(files)


def _open_header(fpath: str):
    from astropy.io import fits

    with fits.open(fpath, ignore_missing_end=True, memmap=False) as hdul:
        hdul.verify("silentfix+ignore")
        # prefer HDU that has TELESCOP
        for hdu in hdul:
            if any(k.upper() == "TELESCOP" for k in hdu.header.keys()):
                return hdu.header.copy()
        return hdul[0].header.copy()


def _pixel_scale_from_wcs(header) -> float | None:
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


def _header_val(header, key, default=None):
    try:
        if key and key in header:
            return header[key]
    except Exception:
        pass
    return default


def _read_filter(header, filter_key: str | None = None) -> str | None:
    if filter_key:
        candidates = [filter_key] + FILTER_HEADER_CANDIDATES
    else:
        candidates = FILTER_HEADER_CANDIDATES
    for key in candidates:
        val = _header_val(header, key)
        if val is not None and str(val).strip().lower() not in AVOID_FILTER_VALUES:
            return str(val).strip(), key
    return None, None


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base (modifies base in place, returns it)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_yml(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _write_yml(path: str, data: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

def scan_folder(
    fits_dir: str,
    wdir: str,
    recursive: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    filter_key: str | None = None,
    gain_key: str = "GAIN",
    rn_key: str = "RDNOISE",
    sat_key: str = "SATURATE",
    exptime_key: str = "EXPTIME",
    mjd_key: str = "MJD-OBS",
    date_key: str = "DATE-OBS",
    airmass_key: str = "AIRMASS",
):
    files = _find_fits_files(fits_dir, recursive=recursive)
    if not files:
        log.warning("No FITS files found in %s", fits_dir)
        return

    log.info("Found %d FITS file(s) in %s", len(files), fits_dir)

    yml_path = os.path.join(wdir, "telescope.yml")
    existing = _load_yml(yml_path)
    new_data: dict = copy.deepcopy(existing)

    # per-(telescope, instrument) aggregated info
    seen: dict = {}  # key -> {"filters": {raw: count}, "pixel_scales": [], "gain_keys": set, ...}

    skipped = 0
    for fpath in files:
        try:
            header = _open_header(fpath)
        except Exception as exc:
            log.debug("Cannot read %s: %s", fpath, exc)
            skipped += 1
            continue

        tele = _header_val(header, "TELESCOP")
        inst = _header_val(header, "INSTRUME")
        if not tele or not inst:
            if verbose:
                log.info("  SKIP (no TELESCOP/INSTRUME): %s", os.path.basename(fpath))
            skipped += 1
            continue

        tele = str(tele).strip()
        inst = str(inst).strip()
        key = (tele, inst)

        fval, fhdr_key = _read_filter(header, filter_key)
        ps = _pixel_scale_from_wcs(header)

        if key not in seen:
            seen[key] = {
                "filters": {},
                "filter_header_key": None,
                "pixel_scales": [],
                "gain_keys_found": set(),
                "rn_keys_found": set(),
                "sat_keys_found": set(),
                "exptime_keys_found": set(),
                "mjd_keys_found": set(),
                "date_keys_found": set(),
                "airmass_keys_found": set(),
                "gain_val": None,
                "rn_val": None,
            }
        rec = seen[key]

        if fval:
            rec["filters"][fval] = rec["filters"].get(fval, 0) + 1
        if fhdr_key and rec["filter_header_key"] is None:
            rec["filter_header_key"] = fhdr_key
        if ps is not None:
            rec["pixel_scales"].append(ps)

        # Collect keyword presence
        for attr, kw in [
            ("gain_keys_found", gain_key),
            ("rn_keys_found", rn_key),
            ("sat_keys_found", sat_key),
            ("exptime_keys_found", exptime_key),
            ("mjd_keys_found", mjd_key),
            ("date_keys_found", date_key),
            ("airmass_keys_found", airmass_key),
        ]:
            if kw and kw in header:
                rec[attr].add(kw)

        # Snapshot one gain / readnoise value for reference
        if rec["gain_val"] is None:
            gv = _header_val(header, gain_key)
            if gv is not None:
                try:
                    rec["gain_val"] = float(gv)
                except (TypeError, ValueError):
                    pass
        if rec["rn_val"] is None:
            rnv = _header_val(header, rn_key)
            if rnv is not None:
                try:
                    rec["rn_val"] = float(rnv)
                except (TypeError, ValueError):
                    pass

        if verbose:
            log.info(
                "  %-40s  TELESCOP=%-20s INSTRUME=%-15s FILTER=%-8s ps=%.3f\"",
                os.path.basename(fpath),
                tele,
                inst,
                fval or "?",
                ps or 0,
            )

    # Build / update YAML from aggregated records
    for (tele, inst), rec in seen.items():
        block_key = "INSTRUME"
        filter_hdr = rec["filter_header_key"] or "FILTER"

        # Median pixel scale
        ps_list = rec["pixel_scales"]
        ps_median = float(np.median(ps_list)) if ps_list else None

        # Build instrument block
        inst_block: dict = {"filter_key_0": filter_hdr}

        # Keyword fields — use the key if found in headers, else keep the standard default
        def _best_key(found_set, default):
            return next(iter(found_set), default)

        inst_block["mjd"] = _best_key(rec["mjd_keys_found"], mjd_key)
        inst_block["date"] = _best_key(rec["date_keys_found"], date_key)
        inst_block["gain"] = _best_key(rec["gain_keys_found"], gain_key)
        inst_block["saturate"] = _best_key(rec["sat_keys_found"], sat_key)
        inst_block["readnoise"] = _best_key(rec["rn_keys_found"], rn_key)
        inst_block["airmass"] = _best_key(rec["airmass_keys_found"], airmass_key)
        inst_block["exptime"] = _best_key(rec["exptime_keys_found"], exptime_key)

        if ps_median is not None:
            inst_block["pixel_scale"] = round(ps_median, 4)

        # Filter mappings: identity by default (user must verify)
        for fval in sorted(rec["filters"]):
            if fval not in inst_block:
                inst_block[fval] = fval  # placeholder: raw -> raw

        # Deep-merge into existing data (preserves user customisations)
        tele_block = new_data.setdefault(tele, {})
        ib_block = tele_block.setdefault(block_key, {})
        existing_inst = ib_block.get(inst, {})
        if not isinstance(existing_inst, dict):
            existing_inst = {}
        # Only add keys not already present (do not clobber user values)
        for k, v in inst_block.items():
            if k not in existing_inst:
                existing_inst[k] = v
        ib_block[inst] = existing_inst

        log.info(
            "Telescope: %-20s  Instrument: %-15s  Filters: %-30s  pixel_scale: %s",
            tele,
            inst,
            ", ".join(sorted(rec["filters"])) or "(none found)",
            f"{ps_median:.4f}\"" if ps_median else "N/A",
        )

    if skipped:
        log.info("Skipped %d file(s) (unreadable or missing TELESCOP/INSTRUME).", skipped)

    if not seen:
        log.warning("No valid telescope/instrument combinations found — telescope.yml not written.")
        return

    if dry_run:
        print("\n# ---- telescope.yml (dry-run, not written) ----")
        print(yaml.dump(new_data, default_flow_style=False, sort_keys=False, allow_unicode=True))
        print("# -----------------------------------------------\n")
    else:
        _write_yml(yml_path, new_data)
        log.info("telescope.yml written to: %s", yml_path)
        if existing:
            log.info("(Existing entries were preserved; new keys merged in.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan a FITS folder and build/update telescope.yml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("fits_dir", help="Folder containing FITS images.")
    p.add_argument(
        "--wdir",
        default=None,
        help="Output directory for telescope.yml (default: fits_dir).",
    )
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
    p = _build_parser()
    args = p.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    fits_dir = os.path.abspath(args.fits_dir)
    if not os.path.isdir(fits_dir):
        log.error("fits_dir does not exist: %s", fits_dir)
        sys.exit(1)

    wdir = os.path.abspath(args.wdir) if args.wdir else fits_dir

    scan_folder(
        fits_dir=fits_dir,
        wdir=wdir,
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
