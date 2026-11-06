"""Synthetic contaminated CCD frame for exercising MaxiMask.

Builds a single image populated with the contaminant classes described
in Paillassa, Bertin & Bouy (2020, arXiv:1907.08298): cosmic rays,
hot and dead columns/pixels, a satellite trail, a saturated star with
diffraction spikes and a bleed trail, nebulosity, fringing and an
overscan strip. Every injected feature is recorded in a boolean truth
mask so MaxiMask detections can be scored for recall.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.synthetic.psf_models import fwhm_to_sigma, gaussian_psf_2d

# MaxiMask only processes HDUs larger than its 400 px tile size.
IMAGE_SHAPE = (1000, 1400)  # (ny, nx)
BACKGROUND = 800.0
BACKGROUND_RMS = 25.0
SATURATE = 60000.0
SEEING_FWHM = 4.0
OVERSCAN_WIDTH = 48

# Truth-mask keys, named after the MaxiMask class abbreviations.
TRUTH_CLASSES = [
    "CR",
    "HCL",
    "DCL",
    "HP",
    "DP",
    "TRL",
    "FR",
    "NEB",
    "SAT",
    "SP",
    "OV",
]


def _stamp_segment(image, mask, y0, x0, y1, x1, width, amplitude):
    """Rasterise a line segment into *image* and *mask*.

    Oversamples the segment so diagonal lines have no gaps, then stamps
    a square footprint of *width* pixels at each sample.
    """
    ny, nx = image.shape
    length = np.hypot(y1 - y0, x1 - x0)
    n = max(2, int(np.ceil(4.0 * length)))
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    half = max(0, round(width / 2.0 - 0.5))
    for y, x in zip(ys, xs):
        y, x = round(y), round(x)
        ylo, yhi = max(0, y - half), min(ny, y + half + 1)
        xlo, xhi = max(0, x - half), min(nx, x + half + 1)
        image[ylo:yhi, xlo:xhi] += amplitude
        mask[ylo:yhi, xlo:xhi] = True


def _stamp_profile_line(image, mask, yc, xc, dr, width, amp0, decay, amp_min):
    """Stamp a radial feature whose amplitude decays with distance.

    Used for diffraction spikes and charge-bleed trails around the
    saturated star. Only pixels above *amp_min* enter the truth mask:
    the far end fades into the noise and cannot be detected.
    """
    ny, nx = image.shape
    half = max(0, round(width / 2.0 - 0.5))
    r = 0.0
    while True:
        amp = amp0 * np.exp(-r / decay)
        if amp < amp_min:
            break
        y, x = round(yc + dr[0] * r), round(xc + dr[1] * r)
        if not (0 <= y < ny and 0 <= x < nx):
            break
        ylo, yhi = max(0, y - half), min(ny, y + half + 1)
        xlo, xhi = max(0, x - half), min(nx, x + half + 1)
        image[ylo:yhi, xlo:xhi] += amp
        mask[ylo:yhi, xlo:xhi] = True
        r += 0.5


def make_contaminated_image(seed: int = 7):
    """Create a synthetic CCD frame with injected contaminants.

    Returns
    -------
    dict with keys:
        image : float32 (ny, nx) - contaminated science image
        header : astropy.io.fits.Header
        truth : dict of bool (ny, nx) arrays per MaxiMask class
        clean : bool (ny, nx) - pixels free of sources and contaminants
        meta : dict - locations of injected features (for zoom plots)
    """
    rng = np.random.default_rng(seed)
    ny, nx = IMAGE_SHAPE
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    truth = {name: np.zeros((ny, nx), dtype=bool) for name in TRUTH_CLASSES}

    # Sky background with a gentle gradient, noise added at the end so
    # truth masks stay pixel-exact.
    image = np.full((ny, nx), BACKGROUND, dtype=np.float64)
    image += 60.0 * np.linspace(-1, 1, nx)[np.newaxis, :]
    sigma = fwhm_to_sigma(SEEING_FWHM)

    x_science_max = nx - OVERSCAN_WIDTH

    # Field stars, kept clear of the defect columns and the saturated
    # star so class truth masks do not overlap.
    star_flux_img = np.zeros((ny, nx), dtype=np.float64)
    sat_cy, sat_cx = 320.0, 420.0
    defect_cols = (260, 1150)
    star_xy = []
    for _ in range(18):
        x0 = rng.uniform(60, x_science_max - 60)
        y0 = rng.uniform(60, ny - 60)
        if min(abs(x0 - c) for c in defect_cols) < 15:
            continue
        if np.hypot(y0 - sat_cy, x0 - sat_cx) < 180:
            continue
        flux = np.exp(rng.uniform(np.log(4e3), np.log(1.2e5)))
        star_flux_img += gaussian_psf_2d(xx, yy, x0, y0, flux, sigma)
        star_xy.append((x0, y0, flux))
    image += star_flux_img

    # Nebulosity: a smooth low-surface-brightness blob, large enough
    # that the 200 px background mesh cannot absorb it entirely.
    neb_cy, neb_cx, neb_amp = 700.0, 950.0, 150.0
    neb = neb_amp * np.exp(-(((yy - neb_cy) / 90.0) ** 2 + ((xx - neb_cx) / 60.0) ** 2))
    image += neb
    truth["NEB"] = neb > 0.15 * neb_amp

    # Fringing: fixed-pattern ripple in the lower-left quadrant.
    fr_y0, fr_y1, fr_x0, fr_x1 = 60, 330, 60, 460
    fringe_amp, fringe_period = 20.0, 7.0
    fr = np.zeros((ny, nx))
    fr[fr_y0:fr_y1, fr_x0:fr_x1] = fringe_amp * np.sin(
        2.0
        * np.pi
        * (yy[fr_y0:fr_y1, fr_x0:fr_x1] + xx[fr_y0:fr_y1, fr_x0:fr_x1])
        / fringe_period
    )
    image += fr
    truth["FR"][fr_y0:fr_y1, fr_x0:fr_x1] = True

    # Satellite trails: one bright diagonal crossing the field and one
    # shorter fainter segment.
    _stamp_segment(image, truth["TRL"], 60, 0, 920, 1350, width=4, amplitude=700.0)
    _stamp_segment(image, truth["TRL"], 0, 700, 260, 480, width=2, amplitude=180.0)

    # Cosmic rays: thin jagged streaks a few tens of pixels long.
    cr_points = []
    for _ in range(8):
        x = rng.uniform(80, x_science_max - 80)
        y = rng.uniform(80, ny - 80)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(8, 45)
        amp = rng.uniform(1500, 20000)
        width = int(rng.integers(1, 3))
        steps = int(rng.integers(2, 5))
        pts = [(y, x)]
        for s in range(steps):
            angle += rng.uniform(-0.6, 0.6)
            y += (length / steps) * np.sin(angle)
            x += (length / steps) * np.cos(angle)
            pts.append((y, x))
        for a, b in pairwise(pts):
            _stamp_segment(image, truth["CR"], a[0], a[1], b[0], b[1], width, amp)
        cr_points.append(pts[0])

    # Hot and dead columns running the full detector height.
    hot_col, dead_col = defect_cols
    image[:, hot_col] += 350.0
    truth["HCL"][:, hot_col] = True
    image[:, dead_col] = 80.0
    truth["DCL"][:, dead_col] = True

    # Scattered hot and dead pixels inside the science area only.
    for _ in range(60):
        y = int(rng.integers(5, ny - 5))
        x = int(rng.integers(5, x_science_max - 5))
        image[y, x] += rng.uniform(2000, 30000)
        truth["HP"][y, x] = True
    for _ in range(60):
        y = int(rng.integers(5, ny - 5))
        x = int(rng.integers(5, x_science_max - 5))
        image[y, x] = rng.uniform(0, 50)
        truth["DP"][y, x] = True

    # Saturated star: flat full-well core over extended Gaussian wings,
    # plus charge bleed along its column and four diffraction spikes.
    wing_amp, wing_sigma = 3000.0, 8.0
    image += wing_amp * np.exp(
        -((yy - sat_cy) ** 2 + (xx - sat_cx) ** 2) / (2 * wing_sigma**2)
    )
    core = (yy - sat_cy) ** 2 + (xx - sat_cx) ** 2 <= 13.0**2
    image[core] = SATURATE + 2000.0
    truth["SAT"][core] = True
    for dr in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
        _stamp_profile_line(
            image,
            truth["SP"],
            sat_cy,
            sat_cx,
            dr,
            width=3,
            amp0=4000.0,
            decay=55.0,
            amp_min=30.0,
        )
    for y in range(int(sat_cy) - 60, int(sat_cy) - 14):
        image[y, int(sat_cx) - 1 : int(sat_cx) + 2] = SATURATE + 1000.0
        truth["SAT"][y, int(sat_cx) - 1 : int(sat_cx) + 2] = True

    # Overscan strip: flat bias level with very low noise.
    image[:, x_science_max:] = 1500.0
    truth["OV"][:, x_science_max:] = True

    # Gaussian read/skynoise; keep the overscan strip quiet.
    noise = rng.normal(0.0, BACKGROUND_RMS, size=(ny, nx))
    noise[:, x_science_max:] *= 0.15
    image += noise

    image[image > SATURATE + 2000.0] = SATURATE + 2000.0

    # Clean pixels: no injected contaminant and no significant star flux.
    occupied = np.zeros((ny, nx), dtype=bool)
    for m in truth.values():
        occupied |= m
    occupied |= star_flux_img > 3.0 * BACKGROUND_RMS
    clean = ~occupied

    header = fits.Header()
    header["TELESCOP"] = "Synthetic"
    header["INSTRUME"] = "MaxiMaskTestCam"
    header["FILTER"] = "r"
    header["EXPTIME"] = 120.0
    header["GAIN"] = 1.0
    header["RDNOISE"] = BACKGROUND_RMS
    header["SATURATE"] = SATURATE
    header["SEEING"] = SEEING_FWHM
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.cdelt = [-0.4 / 3600, 0.4 / 3600]
    wcs.wcs.crval = [150.0, 30.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header.update(wcs.to_header())

    meta = {
        "sat_center": (sat_cy, sat_cx),
        "defect_cols": defect_cols,
        "cr_points": cr_points,
        "neb_center": (neb_cy, neb_cx),
        "fringe_box": (fr_y0, fr_y1, fr_x0, fr_x1),
        "overscan_x0": x_science_max,
        "stars": star_xy,
    }

    return {
        "image": image.astype(np.float32),
        "header": header,
        "truth": truth,
        "clean": clean,
        "meta": meta,
    }
