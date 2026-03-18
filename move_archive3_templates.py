#!/usr/bin/env python3
"""
Move templates from archive(3) to EP250916a/images/templates.
- One image per band folder (g, r, i, u, J, H, K).
- Only use files where target (268.278669, -35.316751) is in FOV.
- Writes single-extension FITS (the extension that contains the target).
"""
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="astropy")
warnings.filterwarnings("ignore", message="RADECSYS")

from astropy.io import fits
from astropy.wcs import WCS

SRC = Path("/home/sbrennan/Downloads/archive(3)")
DEST_ROOT = Path("/home/sbrennan/Desktop/EP250916a/images/templates")
TARGET_RA = 268.278669
TARGET_DEC = -35.316751

# Header filter value -> (folder_name, output_band_letter for filename)
BAND_MAP = {
    "g_SDSS": ("gp_template", "g"),
    "r_SDSS": ("rp_template", "r"),
    "i_SDSS": ("ip_template", "i"),
    "u_SDSS": ("up_template", "u"),
    "J": ("J_template", "J"),
    "H": ("H_template", "H"),
    "K": ("K_template", "K"),
    "Ks": ("K_template", "K"),
}


def get_filter(hdul):
    """Get filter from primary or first extension (ESO FILT1 NAME or FILTER)."""
    for idx in [0, 1]:
        if idx >= len(hdul):
            continue
        h = hdul[idx].header
        for k in h:
            if "FILT" in k and "NAME" in k:
                v = h[k]
                if isinstance(v, str):
                    v = v.strip()
                else:
                    v = str(v).strip()
                if v:
                    return v
        if "FILTER" in h:
            v = h["FILTER"]
            if isinstance(v, str):
                return v.strip()
            return str(v).strip()
    return None


def find_extension_with_target(hdul, ra, dec):
    """Return (ext_index, hdu) for first extension with 2D data and target inside, else (None, None)."""
    for i in range(1, len(hdul)):
        hdu = hdul[i]
        if hdu.data is None or getattr(hdu.data, "ndim", 0) != 2:
            continue
        try:
            w = WCS(hdu.header)
            x, y = w.world_to_pixel_values(ra, dec)
            nx, ny = hdu.data.shape[1], hdu.data.shape[0]
            if 0 <= x < nx and 0 <= y < ny:
                return i, hdu
        except Exception:
            continue
    return None, None


def main():
    # Collect (band_key, path, ext_index) for files that have target in FOV; first per band wins
    band_to_path_ext = {}
    for p in sorted(SRC.glob("*.fits.fz")):
        try:
            with fits.open(p) as hdul:
                filt = get_filter(hdul)
                if not filt:
                    print(f"Skipping (no filter): {p.name}")
                    continue
                folder, band_letter = BAND_MAP.get(filt) or (None, None)
                if folder is None:
                    print(f"Skipping (unknown filter): {p.name} -> {repr(filt)}")
                    continue
                ext_i, _ = find_extension_with_target(hdul, TARGET_RA, TARGET_DEC)
                if ext_i is None:
                    print(f"Skipping (target not in FOV): {p.name} {filt}")
                    continue
                if folder not in band_to_path_ext:
                    band_to_path_ext[folder] = (p, ext_i, band_letter)
                    print(f"Using for {folder}: {p.name} (ext {ext_i})")
        except Exception as e:
            print(f"Error: {p.name}: {e}")

    for folder, (src_path, ext_index, band_letter) in band_to_path_ext.items():
        dest_dir = DEST_ROOT / folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing FITS in this folder
        for old in list(dest_dir.iterdir()):
            if old.suffix.lower() in (".fits", ".fts", ".fit") or old.name.endswith(".fits.fz"):
                old.unlink()
                print(f"Removed: {old}")

        # Optical: sdss_X_band_template.fits; NIR: 2MASS_X_band_template.fits
        if folder in ("gp_template", "rp_template", "ip_template", "up_template"):
            out_name = f"sdss_{band_letter}_band_template.fits"
        else:
            out_name = f"2MASS_{band_letter}_band_template.fits"
        out_path = dest_dir / out_name

        with fits.open(src_path) as hdul:
            hdu = hdul[ext_index]
            fits.HDUList([fits.PrimaryHDU(data=hdu.data, header=hdu.header)]).writeto(
                out_path, overwrite=True
            )
        print("Wrote", out_path)

    print("Done. One template per folder; target (%.4f, %.4f) in FOV." % (TARGET_RA, TARGET_DEC))


if __name__ == "__main__":
    main()
