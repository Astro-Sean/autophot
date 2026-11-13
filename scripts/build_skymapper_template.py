#!/usr/bin/env python3
"""
build_skymapper_template.py
===========================
Stack SkyMapper cutouts into a template for image subtraction.

Each SkyMapper SIAP cutout has its own photometric zeropoint (ZPAPPROX):
u-band throughput varies between detectors, so the values differ by up to
~0.8 mag even at equal EXPTIME.  Stacking raw ADU therefore mixes flux
scales and produces a subtly wrong coadd (the median lands between
image-population clusters instead of on the true sky).  This script

  1. subtracts a robust per-image sky level (median),
  2. rescales each image to a common reference ZP via ZPAPPROX,
  3. reprojects each scaled image onto a common grid,
  4. median-combines with NaN-aware coverage handling.

The absolute flux scale of a subtraction template is not critical - the
pipeline flux-matches against field sources empirically - but internal
consistency is, and that is what the ZP rescaling restores.

Usage
-----
    python build_skymapper_template.py -o out.fits image1.fits [image2.fits ...]
    python build_skymapper_template.py -d /tmp/skymapper_u -o out.fits
"""

import argparse
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


def sky_median(data):
    return float(np.nanmedian(data))


def load_scaled(fname, zp_ref):
    """Return (sky-subtracted, ZP-rescaled image, header, scale_factor)."""
    with fits.open(fname) as hdul:
        data = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header.copy()
    zp = hdr.get("ZPAPPROX")
    if zp is None or not np.isfinite(zp):
        raise ValueError(f"{fname}: missing/invalid ZPAPPROX")
    scale = 10.0 ** (-0.4 * (zp - zp_ref))
    data = (data - sky_median(data)) * scale
    return data, hdr, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*", help="input SkyMapper cutout FITS files")
    ap.add_argument("-d", "--dir", help="directory of cutouts (all *.fits)")
    ap.add_argument("-o", "--out", required=True, help="output coadd path")
    ap.add_argument("--pixscale", type=float, default=0.5,
                    help="output pixel scale in arcsec/pixel (default 0.5)")
    ap.add_argument("--size", type=float, default=10.0,
                    help="output field size in arcmin (default 10)")
    ap.add_argument("--center", default=None,
                    help="ra,dec of grid centre in degrees; default is the "
                         "WCS centre of the first input")
    args = ap.parse_args()

    import glob
    files = sorted(args.images) or sorted(glob.glob(args.dir + "/*.fits"))
    if len(files) < 2:
        sys.exit("need >=2 input images")

    # Reference ZP: median of inputs keeps the coadd on a plausible
    # exposure scale; absolute scale is reset by flux matching anyway.
    zps, exps, fwhms = [], [], []
    for f in files:
        h = fits.getheader(f)
        zps.append(float(h["ZPAPPROX"]))
        exps.append(float(h.get("EXPTIME", np.nan)))
        fwhms.append(float(h.get("QAFWHM", np.nan)))
    zp_ref = float(np.median(zps))
    print(f"{len(files)} inputs | ZP {min(zps):.3f}-{max(zps):.3f} "
          f"(ref {zp_ref:.3f}) | seeing {min(fwhms):.2f}-{max(fwhms):.2f} px")

    # Output grid: square box centred on --center (or the first input's
    # WCS centre when --center is not given).
    if args.center:
        ra_c, dec_c = (float(v) for v in args.center.split(","))
        hdr0 = fits.getheader(files[0])
    else:
        hdr0 = fits.getheader(files[0])
        wcs0 = WCS(hdr0)
        ny, nx = hdr0["NAXIS2"], hdr0["NAXIS1"]
        ra_c, dec_c = wcs0.pixel_to_world_values(nx / 2.0, ny / 2.0)

    npix = int(round(args.size * 60.0 / args.pixscale))
    out_wcs = WCS(naxis=2)
    out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out_wcs.wcs.crpix = [(npix + 1) / 2.0, (npix + 1) / 2.0]
    out_wcs.wcs.crval = [ra_c, dec_c]
    out_wcs.wcs.cdelt = [-args.pixscale / 3600.0, args.pixscale / 3600.0]
    out_shape = (npix, npix)
    print(f"grid {npix}x{npix} @ {args.pixscale}\"/px centred "
          f"({ra_c:.5f}, {dec_c:.5f})")

    stack, ncov = [], np.zeros(out_shape, dtype=int)
    for f in files:
        data, hdr, scale = load_scaled(f, zp_ref)
        img_wcs = WCS(hdr)
        reproj, footprint = reproject_interp(
            (data, img_wcs), out_wcs, shape_out=out_shape
        )
        good = footprint > 0
        stack.append(np.where(good, reproj, np.nan).astype(np.float32))
        ncov += good.astype(int)
        print(f"  {f.split('/')[-1]}: scale={scale:.3f} "
              f"sky+noise-normalised, {good.sum()} px")

    cube = np.nanmedian(np.array(stack), axis=0)
    cube[ncov == 0] = np.nan
    print(f"coverage: min={ncov.min()} max={ncov.max()} "
          f"nan_frac={np.mean(~np.isfinite(cube)):.4f}")

    # Header: ZP is now uniform (zp_ref); EXPTIME loses meaning after
    # scaling so the median exposure is stamped for provenance only.
    out_hdr = fits.Header()
    out_hdr.update(out_wcs.to_header(relax=True))
    out_hdr["TELESCOP"] = "SkyMapper"
    out_hdr["INSTRUME"] = "SkyMapper"
    out_hdr["FILTER"] = hdr0.get("FILTER", "u")
    out_hdr["BUNIT"] = "adu"
    out_hdr["GAIN"] = 0.75  # e-/ADU, Bessell+2011 / DR4
    out_hdr["ZPAPPROX"] = zp_ref
    out_hdr["NCOMBINE"] = len(files)
    out_hdr["EXPTIME"] = float(np.nanmedian(exps))
    out_hdr.add_comment(
        "Per-image ZP rescale + sky subtraction before median stack; "
        "pixel values are adu on the ZPAPPROX scale."
    )
    fits.writeto(args.out, cube.astype(np.float32), out_hdr, overwrite=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
