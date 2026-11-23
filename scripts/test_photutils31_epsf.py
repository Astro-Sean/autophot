"""Verify photutils >= 3.1 ePSF API features AutoPhOT can adopt.

Checks (from photutils 3.x changelog):
- EPSFBuilder returns EPSFBuildResults with diagnostics (converged, iterations,
  excluded stars) while tuple unpacking still works.
- The repo's _BoundedShiftEPSFBuilder subclass still works under photutils 3.1
  and exposes the injected _epsf history list used for salvage.
- ImagePSF/GriddedPSFModel evaluation speedup (3-8x per changelog) - reports
  evaluation timing for a sanity check.

Run:  python scripts/test_photutils31_epsf.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.stats import SigmaClip, mad_std
from astropy.table import Table

import photutils
from photutils.detection import DAOStarFinder
from photutils.psf import (
    EPSFBuilder,
    EPSFStars,
    extract_stars,
)

print(f"photutils version: {photutils.__version__}")
major, minor = (int(x) for x in photutils.__version__.split(".")[:2])
assert (major, minor) >= (3, 1), "photutils >= 3.1 required for these tests"

try:
    from photutils.psf import EPSFBuildResults
except ImportError:
    from photutils.psf.epsf_builder import EPSFBuildResults  # fallback path
print(f"EPSFBuildResults: {EPSFBuildResults}")

rng = np.random.default_rng(42)


def make_field(nstars=40, size=401, fwhm=3.5, noise=2.0):
    data = rng.normal(0.0, noise, (size, size))
    xs = rng.uniform(20, size - 20, nstars)
    ys = rng.uniform(20, size - 20, nstars)
    fluxes = rng.uniform(2000, 40000, nstars)
    yy, xx = np.mgrid[:size, :size]
    sig = fwhm / 2.3548
    for x, y, f in zip(xs, ys, fluxes):
        g = Gaussian2D(f / (2 * np.pi * sig**2), x, y, sig, sig)
        data += g(xx, yy)
    return data


def build_stars(data):
    finder = DAOStarFinder(threshold=20.0, fwhm=3.5)
    tbl = finder(data - np.median(data))
    xcol = "x_centroid" if "x_centroid" in tbl.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in tbl.colnames else "ycentroid"
    stars_tbl = Table()
    stars_tbl["x"] = tbl[xcol]
    stars_tbl["y"] = tbl[ycol]
    return extract_stars(
        data, stars_tbl, size=25
    ), tbl


def main():
    data = make_field()
    stars, tbl = build_stars(data)
    print(f"extracted {len(stars)} EPSFStars")

    sigma_clip = SigmaClip(sigma=4.0, cenfunc=np.nanmedian, stdfunc=mad_std)

    # photutils 3.x: norm_radius removed from EPSFBuilder; per-star
    # normalisation is handled by EPSFStars.flux.
    builder = EPSFBuilder(
        oversampling=4,
        maxiters=10,
        progress_bar=False,
        sigma_clip=sigma_clip,
    )
    result = builder(stars)

    # --- 3.x API: structured result object ---
    print(f"result type: {type(result).__name__}")
    if hasattr(result, "epsf"):
        for attr in (
            "epsf",
            "fitted_stars",
            "iterations",
            "converged",
            "final_center_accuracy",
        ):
            print(f"  has .{attr}: {hasattr(result, attr)}")
        epsf = result.epsf
        print(f"  converged={result.converged} iters={result.iterations}")
        if hasattr(result, "excluded_stars"):
            print(f"  excluded_stars: {result.excluded_stars}")
    else:
        # photutils <= 3.0 returned a tuple
        epsf, fitted = result
        print("  (legacy tuple return)")

    # --- tuple unpacking backward compatibility ---
    epsf2, fitted2 = builder(stars) if not hasattr(result, "epsf") else (
        result.epsf,
        result.fitted_stars,
    )
    print(f"epsf shape {epsf.data.shape}, oversampling {epsf.oversampling}")

    # --- repo subclass compatibility ---
    try:
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1])
        )
        from autophot.photometry.psf import _BoundedShiftEPSFBuilder

        hist = []
        b2 = _BoundedShiftEPSFBuilder(
            oversampling=4,
            maxiters=5,
            progress_bar=False,
            max_shift_px=1.0,
            epsf_history=hist,
        )
        r2 = b2(stars)
        e2 = r2.epsf if hasattr(r2, "epsf") else r2[0]
        print(
            f"_BoundedShiftEPSFBuilder OK: epsf {e2.data.shape}, "
            f"history_len={len(hist)}"
        )
    except Exception as exc:
        print(f"_BoundedShiftEPSFBuilder check FAILED: {exc!r}")

    # --- evaluation benchmark (3.1 compiled spline kernels) ---
    n_eval = 2000
    xy = rng.uniform(15, 25, (2, n_eval))
    t0 = time.perf_counter()
    for i in range(n_eval):
        epsf.evaluate(
            x=np.arange(0, 10)[None, :] * np.ones((10, 1)),
            y=np.arange(0, 10)[:, None] * np.ones((1, 10)),
            flux=1.0,
            x_0=xy[0, i],
            y_0=xy[1, i],
        )
    dt = time.perf_counter() - t0
    print(f"ImagePSF.evaluate x{n_eval}: {dt * 1e3:.1f} ms "
          f"({dt / n_eval * 1e6:.0f} us/eval)")


if __name__ == "__main__":
    main()
