#!/usr/bin/env python3
"""Add TELESCOP and INSTRUME (and FILTER if missing) to template FITS under templates_dir."""
from pathlib import Path
from astropy.io import fits

TEMPLATES_DIR = Path("/home/sbrennan/Desktop/EP250916a/images/templates")

# Infer (TELESCOP, INSTRUME) from path/filename for template images.
def infer_tele_inst(path: Path) -> tuple:
    name = path.name.lower()
    if "2mass" in name:
        return "2MASS", "2MASS"
    if "panstarrs" in name or "ps1" in name:
        return "Pan-STARRS", "PS1"
    # sdss_* or gp/rp/ip/up_template (optical)
    return "SDSS", "SDSS"

def band_from_path(path: Path) -> str:
    # gp_template -> g, K_template -> K
    folder = path.parent.name
    if folder.endswith("_template"):
        band = folder.replace("_template", "").replace("p", "")  # gp -> g
        if band == "g" or band == "r" or band == "i" or band == "u":
            return band
        return folder[0]  # J_template -> J, H_template -> H, K_template -> K
    return "g"

def main():
    for fpath in sorted(TEMPLATES_DIR.rglob("*.fits")) + sorted(TEMPLATES_DIR.rglob("*.fit")):
        if "PSF_model" in fpath.name:
            continue
        try:
            with fits.open(fpath, mode="update") as hdul:
                h = hdul[0].header
                need_update = False
                if not h.get("TELESCOP") or str(h.get("TELESCOP", "")).strip() == "":
                    tele, inst = infer_tele_inst(fpath)
                    h["TELESCOP"] = tele
                    need_update = True
                    print(f"  {fpath.name}: set TELESCOP={tele}")
                if not h.get("INSTRUME") or str(h.get("INSTRUME", "")).strip() == "":
                    tele, inst = infer_tele_inst(fpath)
                    h["INSTRUME"] = inst
                    need_update = True
                    print(f"  {fpath.name}: set INSTRUME={inst}")
                band = band_from_path(fpath)
                if not h.get("FILTER") or str(h.get("FILTER", "")).strip() == "":
                    h["FILTER"] = band
                    need_update = True
                    print(f"  {fpath.name}: set FILTER={band}")
                if need_update:
                    hdul.flush()
                    print(f"Updated: {fpath}")
        except Exception as e:
            print(f"Error: {fpath}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
