#!/usr/bin/env python3
"""
make_skymapper_template.py
==========================
Build a SkyMapper subtraction template sized to cover a science dataset.

The SkyMapper DR4 cutout service limits each request to <10 arcmin per
side, so a single cutout cannot cover a typical wide-field science
frame.  This script:

  1. reads the WCS of every science FITS in a directory and computes the
     bounding box of the union of their footprints (centre + FOV),
  2. tiles that region into <=cell arcmin cells and queries the
     SkyMapper SIAP service for every image overlapping each cell,
  3. downloads a cell-sized cutout for each (image, cell) pair,
  4. stacks them with the same per-image sky-subtract + ZPAPPROX rescale
     + reproject + median logic used by build_skymapper_template.py.

Usage
-----
    # report only - print the union centre/FOV and download plan
    python make_skymapper_template.py /path/to/science_dir --dry-run

    # full build, u band
    python make_skymapper_template.py /path/to/science_dir --band u --clobber
"""

import argparse
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import requests
from astropy.io import fits
from astropy.io.votable import parse as parse_votable
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_skymapper_template import load_scaled

SIAP_URL = "https://api.skymapper.nci.org.au/public/siap/dr4/query"
FIT_EXTS = (".fits", ".fit", ".fz", ".fits.gz")
BLOCK = 1024  # output block side (px) - bounds stacking memory


def scan_footprints(sci_dir, science_band=None):
    """Union RA/Dec bounding box over all science FITS in *sci_dir*."""
    files = sorted(
        f for f in glob.glob(os.path.join(sci_dir, "*"))
        if f.lower().endswith(FIT_EXTS)
    )
    if not files:
        sys.exit(f"no FITS files found in {sci_dir}")

    ra_lo, ra_hi = np.inf, -np.inf
    dec_lo, dec_hi = np.inf, -np.inf
    used = 0
    for f in files:
        try:
            hdr = fits.getheader(f)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(f)}: {exc}")
            continue
        if science_band and science_band.lower() not in str(
            hdr.get("FILTER", "")
        ).lower():
            continue
        w = WCS(hdr)
        if not w.has_celestial:
            print(f"  [skip] {os.path.basename(f)}: no celestial WCS")
            continue
        nx, ny = hdr["NAXIS1"], hdr["NAXIS2"]
        corners = w.all_pix2world(
            [[-0.5, -0.5], [nx - 0.5, -0.5],
             [-0.5, ny - 0.5], [nx - 0.5, ny - 0.5]], 0
        )
        ra_lo = min(ra_lo, corners[:, 0].min())
        ra_hi = max(ra_hi, corners[:, 0].max())
        dec_lo = min(dec_lo, corners[:, 1].min())
        dec_hi = max(dec_hi, corners[:, 1].max())
        used += 1
    if not used:
        sys.exit("no usable science images (check --science-band)")

    ra_c = 0.5 * (ra_lo + ra_hi)
    dec_c = 0.5 * (dec_lo + dec_hi)
    # RA span on the sky shrinks by cos(dec); the bbox in RA degrees is
    # what the square-degree cutouts tile, so keep it in RA coords.
    width_deg = ra_hi - ra_lo
    height_deg = dec_hi - dec_lo
    return {
        "n": used,
        "ra": ra_c,
        "dec": dec_c,
        "width_deg": width_deg,
        "height_deg": height_deg,
    }


def tile_region(ra_c, dec_c, width_deg, height_deg, margin_deg, cell_deg):
    """Cell centres covering the padded bounding box.

    Returns (cells, pitch_deg) where cells is a list of (i, j, ra, dec).
    Pitch <= cell so neighbouring cutouts overlap.
    """
    w = width_deg + 2.0 * margin_deg
    h = height_deg + 2.0 * margin_deg
    nx = max(1, int(np.ceil(w / cell_deg)))
    ny = max(1, int(np.ceil(h / cell_deg)))
    pitch_x, pitch_y = w / nx, h / ny
    ra0 = ra_c - w / 2.0
    dec0 = dec_c - h / 2.0
    cells = [
        (i, j, ra0 + (i + 0.5) * pitch_x, dec0 + (j + 0.5) * pitch_y)
        for j in range(ny) for i in range(nx)
    ]
    return cells, (pitch_x, pitch_y)


def siap_query(ra, dec, size_deg, band, retries=3):
    """Rows of SkyMapper images overlapping a square cutout."""
    params = {
        "POS": f"{ra},{dec}",
        "SIZE": f"{size_deg},{size_deg}",
        "BAND": band,
        "FORMAT": "image/fits",
        "INTERSECT": "OVERLAPS",
        "VERB": "3",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(SIAP_URL, params=params, timeout=60)
            resp.raise_for_status()
            table = parse_votable(BytesIO(resp.content)).get_first_table()
            return table.array
        except Exception as exc:
            if attempt >= retries:
                print(f"  [warn] SIAP query failed at {ra},{dec}: {exc}")
                return []
            time.sleep(3 * attempt)


def _s(value):
    """VOTable char fields may arrive as bytes or str."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return str(value)


def fetch_cutout(url, path, retries=3):
    """Download one cutout; returns (path, ok)."""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(path, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            with open(path, "rb") as fh:
                if fh.read(6) != b"SIMPLE":
                    raise ValueError("not a FITS file")
            return path, True
        except Exception as exc:
            if attempt >= retries:
                print(f"  [warn] download failed {url}: {exc}")
                return path, False
            time.sleep(3 * attempt)


def block_subwcs(wcs, x0, y0):
    """WCS of the output block whose lower-left corner is (x0, y0)."""
    b = wcs.deepcopy()
    b.wcs.crpix = wcs.wcs.crpix - [x0, y0]
    return b


def stack_cutouts(files, ra_c, dec_c, width_deg, height_deg, pixscale):
    """ZP-rescale + reproject + median-combine, block-wise."""
    from reproject import reproject_interp

    nx = int(np.ceil(width_deg * 3600.0 / pixscale))
    ny = int(np.ceil(height_deg * 3600.0 / pixscale))
    out_wcs = WCS(naxis=2)
    out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out_wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    out_wcs.wcs.crval = [ra_c, dec_c]
    out_wcs.wcs.cdelt = [-pixscale / 3600.0, pixscale / 3600.0]

    zps, exps = [], []
    for f in files:
        h = fits.getheader(f)
        zps.append(float(h["ZPAPPROX"]))
        exps.append(float(h.get("EXPTIME", np.nan)))
    zp_ref = float(np.median(zps))
    print(f"stacking {len(files)} cutouts on {nx}x{ny} grid "
          f"@ {pixscale}\"/px, ZP ref {zp_ref:.3f}")

    nbx = int(np.ceil(nx / BLOCK))
    nby = int(np.ceil(ny / BLOCK))
    block_imgs = {(i, j): [] for j in range(nby) for i in range(nbx)}
    block_cov = {(i, j): [] for j in range(nby) for i in range(nbx)}

    for k, f in enumerate(files, 1):
        data, hdr, _ = load_scaled(f, zp_ref)
        img_wcs = WCS(hdr)
        # Bounding box of this input in output pixels, so only the
        # output blocks it actually overlaps get a reproject call.
        iny, inx = data.shape
        c = img_wcs.all_pix2world(
            [[0, 0], [inx, 0], [0, iny], [inx, iny]], 0
        )
        px = out_wcs.all_world2pix(c[:, 0], c[:, 1], 0)
        x_lo = max(0, int(np.floor(px[0].min())) - 1)
        x_hi = min(nx, int(np.ceil(px[0].max())) + 1)
        y_lo = max(0, int(np.floor(px[1].min())) - 1)
        y_hi = min(ny, int(np.ceil(px[1].max())) + 1)
        for j in range(y_lo // BLOCK, min(nby, y_hi // BLOCK + 1)):
            for i in range(x_lo // BLOCK, min(nbx, x_hi // BLOCK + 1)):
                bx0, by0 = i * BLOCK, j * BLOCK
                bnx = min(BLOCK, nx - bx0)
                bny = min(BLOCK, ny - by0)
                reproj, foot = reproject_interp(
                    (data, img_wcs), block_subwcs(out_wcs, bx0, by0),
                    shape_out=(bny, bnx),
                )
                good = (foot > 0) & np.isfinite(reproj)
                block_imgs[(i, j)].append(
                    np.where(good, reproj, np.nan).astype(np.float32)
                )
                block_cov[(i, j)].append(good)
        if k % 20 == 0 or k == len(files):
            print(f"  reprojected {k}/{len(files)}")

    out = np.full((ny, nx), np.nan, dtype=np.float32)
    ncov = np.zeros((ny, nx), dtype=np.int16)
    for (i, j), imgs in block_imgs.items():
        if not imgs:
            continue
        bx0, by0 = i * BLOCK, j * BLOCK
        bny, bnx = imgs[0].shape
        out[by0:by0 + bny, bx0:bx0 + bnx] = np.nanmedian(
            np.array(imgs), axis=0
        )
        ncov[by0:by0 + bny, bx0:bx0 + bnx] = np.sum(
            np.array(block_cov[(i, j)]), axis=0
        )
    out[ncov == 0] = np.nan
    return out, ncov, out_wcs, zp_ref, exps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("science_dir",
                    help="directory containing the science FITS frames")
    ap.add_argument("--band", default="u",
                    help="SkyMapper band to fetch (u,v,g,r,i,z); "
                         "default u")
    ap.add_argument("--science-band", default=None,
                    help="only use science frames whose FILTER contains "
                         "this string when computing the footprint "
                         "(default: use all)")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="padding around the union footprint, arcmin "
                         "(default 1.0)")
    ap.add_argument("--cell", type=float, default=9.0,
                    help="cutout cell side, arcmin; must be <10 "
                         "(default 9.0)")
    ap.add_argument("--pixscale", type=float, default=0.5,
                    help="output pixel scale, arcsec/px (default 0.5, "
                         "SkyMapper native)")
    ap.add_argument("--image-type", default=None,
                    help="restrict to SkyMapper image_type, e.g. 'main' "
                         "or 'main,short' (default: all)")
    ap.add_argument("--fwhm-max", type=float, default=0.0,
                    help="drop exposures with mean_fwhm above this, "
                         "arcsec (default 0 = keep all)")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel downloads (default 4)")
    ap.add_argument("--workdir", default=None,
                    help="download/scratch dir; default "
                         "<science_dir>/skymapper_work")
    ap.add_argument("--out", default=None,
                    help="output template path; default "
                         "<science_dir>/templates/<band>_template/"
                         "skymapper_<band>_band_template.fits")
    ap.add_argument("--coverage-out", default=None,
                    help="optional path for the exposure-count map")
    ap.add_argument("--dry-run", action="store_true",
                    help="print geometry + download plan, fetch nothing")
    ap.add_argument("--keep-cutouts", action="store_true",
                    help="keep downloaded cutouts after stacking")
    ap.add_argument("--clobber", action="store_true",
                    help="overwrite an existing output file")
    args = ap.parse_args()

    band = args.band.strip().lower()
    if band not in list("uvgriz"):
        sys.exit(f"SkyMapper has no band '{band}'")
    cell_deg = args.cell / 60.0
    if cell_deg >= 10.0 / 60.0:
        sys.exit("--cell must be <10 arcmin (SkyMapper cutout limit)")
    margin_deg = args.margin / 60.0
    workdir = args.workdir or os.path.join(args.science_dir, "skymapper_work")
    out_path = args.out or os.path.join(
        args.science_dir, "templates", f"{band}_template",
        f"skymapper_{band}_band_template.fits",
    )
    print(f"== Scanning {args.science_dir} ==")
    fp = scan_footprints(args.science_dir, args.science_band)
    w_arc = fp["width_deg"] * 60.0
    h_arc = fp["height_deg"] * 60.0
    print(f"{fp['n']} science frames | union footprint "
          f"{w_arc:.1f}' x {h_arc:.1f}' (RA x Dec)")
    print(f"optimum centre: RA {fp['ra']:.5f}  Dec {fp['dec']:.5f}")

    cells, (pitch_x, pitch_y) = tile_region(
        fp["ra"], fp["dec"], fp["width_deg"], fp["height_deg"],
        margin_deg, cell_deg,
    )
    print(f"padded FOV {(w_arc + 2 * args.margin):.1f}' x "
          f"{(h_arc + 2 * args.margin):.1f}' -> {len(cells)} cells of "
          f"{args.cell:.1f}' (pitch {pitch_x * 60:.1f}' x "
          f"{pitch_y * 60:.1f}')")

    print("== Querying SkyMapper SIAP ==")
    want_types = (
        {t.strip() for t in args.image_type.split(",")}
        if args.image_type else None
    )
    jobs = []
    seen = set()
    for i, j, ra, dec in cells:
        rows = siap_query(ra, dec, cell_deg, band)
        kept = 0
        for r in rows:
            if want_types and _s(r["image_type"]) not in want_types:
                continue
            if args.fwhm_max and float(r["mean_fwhm"]) > args.fwhm_max:
                continue
            url = _s(r["get_fits"])
            if url in seen:
                continue
            seen.add(url)
            img_id = re.search(r"image=([^&]+)", url)
            img_id = img_id.group(1) if img_id else _s(r["image_name"])
            fname = f"{img_id}_c{j:02d}r{i:02d}.fits"
            jobs.append((url, os.path.join(
                workdir, "cutouts", band, fname)))
            kept += 1
        print(f"  cell ({i},{j}) at {ra:.4f},{dec:.4f}: "
              f"{len(rows)} images, {kept} new cutouts")
    print(f"total: {len(jobs)} cutouts to download")
    if args.dry_run:
        return
    if os.path.exists(out_path) and not args.clobber:
        sys.exit(f"{out_path} exists - pass --clobber or --out")
    if not jobs:
        sys.exit("no SkyMapper images cover this region")

    print("== Downloading cutouts ==")
    os.makedirs(os.path.join(workdir, "cutouts", band), exist_ok=True)
    ok_files, n_done = [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(fetch_cutout, u, p): p for u, p in jobs}
        for fut in as_completed(futs):
            path, ok = fut.result()
            n_done += 1
            if ok:
                ok_files.append(path)
            if n_done % 20 == 0 or n_done == len(jobs):
                print(f"  {n_done}/{len(jobs)} downloaded "
                      f"({len(ok_files)} ok)")
    if len(ok_files) < 2:
        sys.exit("too few cutouts downloaded to stack")

    print("== Stacking ==")
    out, ncov, out_wcs, zp_ref, exps = stack_cutouts(
        ok_files, fp["ra"], fp["dec"],
        fp["width_deg"] + 2 * margin_deg,
        fp["height_deg"] + 2 * margin_deg,
        args.pixscale,
    )
    finite = np.isfinite(out)
    print(f"stack done | finite {100 * finite.mean():.1f}% | "
          f"coverage {int(ncov[finite].min())}-{int(ncov.max())} imgs/px")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hdr = fits.Header()
    hdr.update(out_wcs.to_header(relax=True))
    hdr["TELESCOP"] = "SkyMapper"
    hdr["INSTRUME"] = "SkyMapper"
    hdr["FILTER"] = band
    hdr["BUNIT"] = "adu"
    hdr["GAIN"] = 0.75  # e-/ADU, Bessell+2011 / DR4
    hdr["ZPAPPROX"] = zp_ref
    hdr["NCOMBINE"] = len(ok_files)
    hdr["EXPTIME"] = float(np.nanmedian(exps))
    hdr.add_comment(
        "Per-image ZP rescale + sky subtraction before median stack; "
        "pixel values are adu on the ZPAPPROX scale."
    )
    fits.writeto(out_path, out, hdr, overwrite=args.clobber)
    print(f"wrote {out_path}")

    if args.coverage_out:
        fits.writeto(args.coverage_out, ncov.astype(np.int16),
                     hdr, overwrite=True)
        print(f"wrote {args.coverage_out}")

    if not args.keep_cutouts:
        import shutil
        shutil.rmtree(os.path.join(workdir, "cutouts", band),
                      ignore_errors=True)
        print("removed cutout scratch files (use --keep-cutouts to keep)")


if __name__ == "__main__":
    main()
