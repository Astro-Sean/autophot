#!/usr/bin/env python3
"""Download Legacy Survey FITS cutouts for a fixed sky position.

Fetches one cutout per band from the legacysurvey.org viewer cutout
service and writes files compatible with AutoPhOT ingestion (FILTER and
BUNIT header keywords set). Band coverage is validated per file so
missing bands are skipped instead of saved as empty frames.

Example:
    python download_legacy.py --outdir ~/Desktop/tmp2
"""

import argparse
import io
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

CUTOUT_URL = "https://www.legacysurvey.org/viewer/fits-cutout"

# Viewer layers for each data release. g/r/i/z exist at this declination
# via DECam (DECaLS/DES), so all four are worth requesting.
ALL_BANDS = ("g", "r", "i", "z")


def parse_args():
    p = argparse.ArgumentParser(
        description="Download Legacy Survey FITS cutouts per band."
    )
    p.add_argument(
        "--ra",
        default="06:04:00.34",
        help="Right ascension (sexagesimal or degrees).",
    )
    p.add_argument(
        "--dec",
        default="-54:15:51.38",
        help="Declination (sexagesimal or degrees).",
    )
    p.add_argument(
        "--bands",
        default="griz",
        help="Bands to request, e.g. 'grz'. Default: griz.",
    )
    p.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Cutout size in pixels. 1024 px ~ 4.5 arcmin.",
    )
    p.add_argument(
        "--pixscale",
        type=float,
        default=0.262,
        help="Pixel scale in arcsec/pixel. 0.262 is native DECam.",
    )
    p.add_argument(
        "--layer",
        default="ls-dr11",
        help="Viewer layer (e.g. ls-dr11, ls-dr10).",
    )
    p.add_argument(
        "--outdir",
        default=str(Path.home() / "Desktop" / "tmp2"),
        help="Output directory.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download attempts per band.",
    )
    return p.parse_args()


def fetch_cutout(coord, band, size, pixscale, layer, retries):
    url = (
        f"{CUTOUT_URL}?ra={coord.ra.deg:.7f}&dec={coord.dec.deg:.7f}"
        f"&layer={layer}&pixscale={pixscale}&bands={band}&size={size}"
    )
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=120) as resp:
                if resp.status != 200:
                    raise HTTPError(url, resp.status, resp.reason, resp.headers, None)
                return resp.read()
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    raise last_err


def validate_and_save(data, band, layer, coord, outpath):
    # Reject frames with no survey coverage; the service can return an
    # all-NaN image at footprint edges instead of an error.
    try:
        hdu = fits.open(io.BytesIO(data))
    except Exception as e:
        print(f"  not a valid FITS file: {e}")
        return False
    try:
        img = hdu[0].data
        if img is None or img.ndim != 2:
            print("  no 2D image in primary HDU")
            return False
        frac_finite = np.isfinite(img).mean()
        if frac_finite < 0.5:
            print(f"  only {frac_finite:.0%} finite pixels; likely outside footprint")
            return False
        hdr = hdu[0].header
        hdr["FILTER"] = (band, "Bandpass (LS single-letter, AB system)")
        hdr["BUNIT"] = ("nanomaggy", "Flux unit; AB zeropoint is 22.5")
        hdr["TELESCOP"] = ("CTIO 4m", "Legacy Surveys coadd telescope")
        hdr["INSTRUME"] = ("DECam", "Legacy Surveys coadd instrument")
        hdr["SURVEY"] = (layer, "Legacy Surveys data release layer")
        hdr["RA_TARG"] = (coord.ra.deg, "[deg] Requested RA")
        hdr["DEC_TARG"] = (coord.dec.deg, "[deg] Requested Dec")
        hdu.writeto(outpath, overwrite=True)
    finally:
        hdu.close()
    return True


def main():
    args = parse_args()
    coord = SkyCoord(args.ra, args.dec, unit=("hourangle", "deg"), frame="icrs")
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    coord_str = coord.to_string("hmsdms", sep="", precision=2).replace(" ", "")
    print(f"Position : {coord.to_string('hmsdms', sep=':')} "
          f"({coord.ra.deg:.5f}, {coord.dec.deg:.5f} deg)")
    print(f"Layer    : {args.layer}")
    print(f"Cutout   : {args.size} x {args.size} px @ {args.pixscale} arcsec/px")
    print(f"Output   : {outdir}")

    saved, skipped = [], []
    for band in args.bands:
        if band not in ALL_BANDS:
            print(f"[{band}] unknown band, skipping (valid: {''.join(ALL_BANDS)})")
            skipped.append(band)
            continue
        print(f"[{band}] downloading...")
        try:
            data = fetch_cutout(
                coord, band, args.size, args.pixscale, args.layer, args.retries
            )
        except Exception as e:
            print(f"[{band}] download failed: {e}")
            skipped.append(band)
            continue
        outpath = outdir / f"{args.layer}_{coord_str}_{band}.fits"
        if validate_and_save(data, band, args.layer, coord, outpath):
            print(f"[{band}] saved {outpath.name} ({len(data)/1e6:.1f} MB)")
            saved.append(outpath)
        else:
            skipped.append(band)

    print(f"\nDone: {len(saved)} saved, {len(skipped)} skipped {skipped or ''}")
    if not saved:
        sys.exit(1)


if __name__ == "__main__":
    main()
