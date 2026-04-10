#!/usr/bin/env python3
"""
Patch Pan-STARRS FITS headers in-place so they work with autophot.
Combines primary and extension headers when two are present, then fills
TELESCOP, INSTRUME, FILTER, EXPTIME, GAIN, RDNOISE, OBSMJD from existing keywords.
"""

import sys
from pathlib import Path
from astropy.io import fits


def merge_headers(primary: fits.Header, secondary: fits.Header) -> fits.Header:
    """
    Combine two FITS headers. Primary is used first; any keyword from secondary
    that is not already in the result is appended (secondary fills gaps).
    """
    out = primary.copy()
    for card in secondary.cards:
        if card.keyword in ("", "COMMENT", "HISTORY"):
            out.append(card)
        elif card.keyword not in out:
            out.append(card)
    return out


def get_secondary_header(hdul: fits.HDUList) -> fits.Header | None:
    """
    If the second HDU is an LDAC-style table containing a header string,
    parse and return that header. Otherwise if it has a .header, return it.
    """
    if len(hdul) < 2:
        return None
    h1 = hdul[1]
    if h1.header.get("EXTNAME") == "LDAC_IMHEAD" and hasattr(h1, "columns"):
        try:
            # LDAC: header stored in table column (often 'Field Header Card')
            for col_name in ("Field Header Card", "FIELD HEADER CARD", "Header"):
                if col_name in h1.columns.names:
                    raw = h1.data[col_name][0]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    return fits.Header.fromstring(raw)
        except Exception:
            pass
    if hasattr(h1, "header"):
        return h1.header
    return None


def fix_panstarrs_header(h):
    """
    Patch a Pan-STARRS header in-place so it works smoothly with autophot.
    Tries to reuse existing values; only fills or renames when missing.
    """
    # --- Telescope / instrument ---
    h.setdefault("TELESCOP", "Pan-STARRS1")
    h.setdefault("INSTRUME", "GPC1")

    # --- Filter ---
    filt = (
        h.get("FILTER")
        or h.get("FLT")
        or h.get("HIERARCH FILTERID")
        or h.get("HIERARCH FPA.FILTERID")
    )
    if filt is not None:
        band = str(filt).strip().lower()
        if band in {"g", "r", "i", "z"}:
            h["FILTER"] = band
        else:
            h["FILTER"] = band
    else:
        h.setdefault("FILTER", "r")

    # --- Exposure time ---
    if "EXPTIME" not in h:
        exp = (
            h.get("EXPOSURE")
            or h.get("ELAPTIME")
            or h.get("HIERARCH CELL.EXPTIME")
            or 30.0
        )
        try:
            h["EXPTIME"] = float(exp)
        except Exception:
            h["EXPTIME"] = 30.0

    # --- Gain ---
    if "GAIN" not in h:
        gain = h.get("HIERARCH CELL.GAIN") or h.get("HIERARCH DET.GAIN")
        if gain is not None:
            try:
                h["GAIN"] = float(gain)
            except Exception:
                pass

    # --- Read noise ---
    if "RDNOISE" not in h:
        rd = h.get("READNOISE") or h.get("HIERARCH CELL.READNOISE")
        if rd is not None:
            try:
                h["RDNOISE"] = float(rd)
            except Exception:
                pass

    # --- Observation MJD ---
    if "OBSMJD" not in h:
        mjd = h.get("MJD-OBS") or h.get("MJD_OBS")
        if mjd is not None:
            h["OBSMJD"] = mjd


def main(root, recursive=False):
    root = Path(root).expanduser().resolve()
    if recursive:
        files = sorted(root.rglob("*.fits"))
    else:
        files = sorted(root.glob("*.fits"))
    if not files:
        print(f"No FITS files found in {root}.")
        return

    print(f"Updating {len(files)} FITS file(s) under {root}.")
    for f in files:
        try:
            with fits.open(f, mode="readonly", do_not_scale_image_data=True) as hdul:
                primary_header = hdul[0].header.copy()
                primary_data = hdul[0].data
                # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
                if primary_data is not None and primary_data.dtype.kind != 'f':
                    primary_data = primary_data.astype(np.float32)
                if primary_data is None and len(hdul) > 1:
                    for i in range(1, len(hdul)):
                        if (
                            hasattr(hdul[i], "data")
                            and getattr(hdul[i].data, "ndim", 0) == 2
                        ):
                            primary_data = hdul[i].data
                            break

                secondary = get_secondary_header(hdul)
                if secondary is not None:
                    combined = merge_headers(primary_header, secondary)
                else:
                    combined = primary_header

                fix_panstarrs_header(combined)

                tmp_path = f.with_suffix(f.suffix + ".tmp")
                from functions import safe_fits_write
                safe_fits_write(str(tmp_path), primary_data, combined, output_verify="ignore")
            tmp_path.replace(f)
            print(f"  OK: {f.name}.")
        except Exception as e:
            print(f"  Failed: {f.name} ({e})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fix Pan-STARRS FITS headers for autophot")
    p.add_argument("directory", help="Directory containing FITS files")
    p.add_argument(
        "-r", "--recursive", action="store_true", help="Search subdirectories for .fits"
    )
    args = p.parse_args()
    main(args.directory, recursive=args.recursive)
