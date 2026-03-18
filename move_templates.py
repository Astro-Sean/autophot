#!/usr/bin/env python3
"""Move one template per band from archive(2) to EP250916a/images/templates.
   Reads filter from FITS header, decompresses .fits.fz to .fits, one file per folder."""
import os
import re
import shutil
from pathlib import Path

try:
    from astropy.io import fits
except ImportError:
    raise SystemExit("Need astropy: pip install astropy")

SRC = Path("/home/sbrennan/Downloads/archive(2)")
DEST_ROOT = Path("/home/sbrennan/Desktop/EP250916a/images/templates")

BAND_TO_DIR = {
    "g_SDSS": "gp_template",
    "r_SDSS": "rp_template",
    "i_SDSS": "ip_template",
    "u_SDSS": "up_template",
}

def get_filter(path):
    with fits.open(path) as hdul:
        h = hdul[0].header
        for k in h:
            if "FILT" in k and "NAME" in k:
                v = h[k]
                if isinstance(v, str):
                    return v.strip()
                return str(v).strip()
    return None

def main():
    # Collect (band, path) for each .fits.fz, one per band (first wins)
    band_to_path = {}
    for p in sorted(SRC.glob("*.fits.fz")):
        filt = get_filter(p)
        if not filt or filt not in BAND_TO_DIR:
            print(f"Skipping (unknown filter): {p.name} -> {filt}")
            continue
        band = filt
        if band not in band_to_path:
            band_to_path[band] = p
            print(f"Using for {BAND_TO_DIR[band]}: {p.name}")

    for band, src_path in band_to_path.items():
        folder = BAND_TO_DIR[band]
        dest_dir = DEST_ROOT / folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Remove any existing template files in this folder (ensure only one)
        for old in dest_dir.iterdir():
            if old.suffix.lower() in (".fits", ".fts", ".fit") and "_template" in old.name:
                old.unlink()
                print("Removed", old)
            elif old.suffix.lower() in (".fits", ".fts", ".fit") or old.name.endswith(".fits.fz"):
                # Remove any other FITS so only one remains
                old.unlink()
                print("Removed", old)

        # Output name: sdss_{g,r,i,u}_band_template.fits (pipeline expects _template in name)
        band_letter = band.split("_")[0]
        out_name = f"sdss_{band_letter}_band_template.fits"
        out_path = dest_dir / out_name

        # Decompress and write
        with fits.open(src_path) as hdul:
            hdul.writeto(out_path, overwrite=True)
        print(f"Wrote: {out_path}")

    print("Done. One template per folder.")

if __name__ == "__main__":
    main()
