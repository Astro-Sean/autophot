"""Tests for PSF source-selection funnel relaxation.

Covers:
* Find_FWHM.check_linearity undersampled relaxation - the RANSAC inlier
  residual threshold adapts to subpixel-phase peak-flux scatter instead of
  rejecting most genuine stars, while truly saturated/non-linear sources
  are still rejected.
* The low-inlier-fraction guard that keeps all quality-cut sources when
  the linearity fit cannot vet sources reliably.
* Aperture.measure_optimum_radius undersampled tail-screen relaxation and
  stage-count logging (regression: 28 stable -> 18 was silently caused by
  the final tail screen on undersampled data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fwhm import Find_FWHM
from aperture import Aperture


def _linearity_catalog(n=70, offset_scatter=0.25, n_saturated=0, seed=1):
    """Catalog with flux_AP/maxPixel and a realistic m_peak - m_inst offset
    scatter (as produced by subpixel phase jitter on undersampled data)."""
    rng = np.random.default_rng(seed)
    flux = rng.uniform(2e3, 5e4, n)
    offset = 2.2 + rng.normal(0.0, offset_scatter, n)
    peak = flux * 10 ** (-offset / 2.5)
    if n_saturated:
        # Saturated stars: peak suppressed far below the linear locus,
        # inflating delta = m_peak - m_inst by ~1.5 mag.
        flux[:n_saturated] = 6e4
        peak[:n_saturated] = 6e4 * 10 ** (-(2.2 + 1.5) / 2.5)
    return pd.DataFrame(
        {
            "flux_AP": flux,
            "maxPixel": np.abs(peak),
            "flux_AP_err": flux * 0.02,
            "maxPixel_err": np.abs(peak) * 0.05,
            "threshold": np.full(n, 10.0),
            "flags": np.zeros(n),
            "ellipticity": np.full(n, 0.1),
            "fwhm": rng.normal(2.2, 0.2, n),
        }
    )


class TestCheckLinearityUndersampled:
    def test_undersampled_keeps_quality_sources(self):
        """Subpixel-phase offset scatter should not cull the catalog."""
        df = _linearity_catalog()
        out, fit_params, sat_range = Find_FWHM(
            {"photometry": {}, "fwhm": 2.2}
        ).check_linearity(df.copy())
        # Essentially all quality-cut sources survive on undersampled data.
        assert len(out) >= 0.9 * len(df)
        assert np.isfinite(sat_range[0])

    def test_well_sampled_uses_fixed_threshold(self):
        """Same catalog flagged well-sampled uses the 0.25-mag cut."""
        df = _linearity_catalog()
        out_us, _, _ = Find_FWHM(
            {"photometry": {}, "fwhm": 2.2}
        ).check_linearity(df.copy())
        out_ws, _, _ = Find_FWHM(
            {"photometry": {}, "fwhm": 5.0}
        ).check_linearity(df.copy())
        assert len(out_ws) < len(out_us)

    def test_undersampled_still_rejects_saturated(self):
        """The adaptive threshold is capped, so genuine non-linearity is
        still excluded."""
        df = _linearity_catalog(n_saturated=5)
        out, _, _ = Find_FWHM(
            {"photometry": {}, "fwhm": 2.2}
        ).check_linearity(df.copy())
        assert len(out) <= len(df) - 4
        assert len(out) >= 0.8 * len(df)

    def test_low_inlier_fraction_keeps_all(self):
        """When the fit cannot vet sources reliably, keep the quality-cut
        pool rather than returning a handful of survivors."""
        df = _linearity_catalog(offset_scatter=0.8)  # incoherent offsets
        out, fit_params, _ = Find_FWHM(
            {"photometry": {}, "fwhm": 5.0}
        ).check_linearity(df.copy())
        assert len(out) == len(df)


class TestOptimumRadiusUndersampled:
    def _field(self, seed=3):
        """Undersampled field (FWHM~2.2 px) of isolated stars with a range
        of brightnesses, plus per-source catalog rows."""
        rng = np.random.default_rng(seed)
        ny, nx = 400, 400
        fwhm = 2.2
        sigma = fwhm / 2.355
        img = rng.normal(100.0, 2.0, (ny, nx))
        rows = []
        half = 12
        for i in range(30):
            x = rng.uniform(60, nx - 60)
            y = rng.uniform(60, ny - 60)
            flux = rng.uniform(4e3, 4e4)
            xi, yi = int(round(x)), int(round(y))
            yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
            stamp = flux * np.exp(
                -((xx - (x - xi)) ** 2 + (yy - (y - yi)) ** 2) / (2 * sigma**2)
            )
            img[yi - half : yi + half + 1, xi - half : xi + half + 1] += stamp
            snr = flux / (2.0 * np.pi * np.count_nonzero(stamp > 1)) ** 0.5
            rows.append(dict(x_pix=x, y_pix=y, SNR=max(snr, 10.0)))
        return img, pd.DataFrame(rows)

    def _yaml(self, fwhm=2.2, relax=2.0):
        return {
            "fwhm": fwhm,
            "gain": 1.0,
            "photometry": {
                "undersampled_fwhm_threshold": 2.5,
                "optimum_radius_undersampled_tail_relax": relax,
            },
        }

    def test_undersampled_tail_relax_retains_more(self):
        img, sources = self._field()
        bkg_rms = np.full_like(img, 2.0)
        strict = Aperture(self._yaml(relax=1.0), img).measure_optimum_radius(
            sources=sources.copy(), plot=False, n_jobs=1,
            background_rms=bkg_rms,
        )[0]
        relaxed = Aperture(self._yaml(relax=2.0), img).measure_optimum_radius(
            sources=sources.copy(), plot=False, n_jobs=1,
            background_rms=bkg_rms,
        )[0]
        assert len(relaxed) >= len(strict)
        # The funnel must never collapse to a handful of stars here.
        assert len(relaxed) >= 8


class TestOptimumRadiusNormalization:
    """The per-source CoG normalization must not hinge on a single outer
    bin: a defect or neighbour in the last radial annulus used to move
    the profile maximum and bias every downstream statistic for that
    source.  Normalizing by the median of the outer ~0.5 FWHM of bins
    makes the denominator robust to single-bin outliers."""

    def _field(self, seed=5, contaminate=True):
        rng = np.random.default_rng(seed)
        ny = nx = 300
        fwhm = 4.0
        sigma = fwhm / 2.355
        img = rng.normal(0.0, 2.0, (ny, nx))
        yy, xx = np.mgrid[0:ny, 0:nx]
        rows = []
        pos = [(x, y) for x in np.linspace(60, 240, 5)
               for y in np.linspace(60, 240, 5)]
        for cx, cy in pos:
            img += 3e4 * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
            )
            rows.append(dict(x_pix=cx, y_pix=cy, SNR=400.0))
        if contaminate:
            # Compact spike at ~3.3 FWHM from the corner star - inside the
            # outermost normalization bins, small enough to evade the
            # contamination screens but large enough to move a max-based
            # denominator.
            img += 3e3 * np.exp(
                -((xx - 60 - 3.3 * fwhm) ** 2 + (yy - 60) ** 2)
                / (2 * sigma**2)
            )
        return img, pd.DataFrame(rows), fwhm

    def _yaml(self, fwhm):
        return {
            "fwhm": fwhm,
            "gain": 1.0,
            "write_dir": "/tmp",
            "base": "test",
            "photometry": {},
        }

    def test_sane_radius_on_clean_field(self):
        img, sources, fwhm = self._field(contaminate=False)
        out, r_fwhm, scale = Aperture(self._yaml(fwhm), img).measure_optimum_radius(
            sources=sources.copy(), plot=False, n_jobs=1,
            background_rms=np.full_like(img, 2.0),
        )
        # EE95% of a Gaussian sits at ~1.04 FWHM.
        assert 0.6 < r_fwhm < 2.0
        assert len(out) >= 8

    def test_outer_annulus_spike_does_not_inflate_radius(self):
        img, sources, fwhm = self._field(contaminate=True)
        out, r_fwhm, scale = Aperture(self._yaml(fwhm), img).measure_optimum_radius(
            sources=sources.copy(), plot=False, n_jobs=1,
            background_rms=np.full_like(img, 2.0),
        )
        # The spike-contaminated star may be screened out, but it must not
        # drag the ensemble optimum radius high.
        assert 0.6 < r_fwhm < 2.0
        assert len(out) >= 8


class TestTailScreenRescueExcludesOutliers:
    """GROND_g_TDP7 regression: a broad galaxy-like source survived
    optimum-radius selection despite the 10-sigma radius clip correctly
    rejecting it, because the min-keep rescue after the tail screen drew
    its candidates from the PRE-clip pool.  The rescue may only restore
    sources the noisy tail screen rejected - never clip outliers."""

    def _field(self, seed=3):
        rng = np.random.default_rng(seed)
        fwhm = 8.0
        sigma = fwhm / 2.355
        # Sky noise makes the normalised CoG tails wiggle so the outer
        # flatness screen rejects every source and the rescue path runs.
        img = rng.normal(50.0, 15.0, (400, 400))
        yy, xx = np.mgrid[0:400, 0:400]
        rows = []
        for cx, cy in [(80, 80), (80, 320), (320, 80), (320, 320)]:
            img += 8e3 * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
            )
            rows.append(dict(x_pix=float(cx), y_pix=float(cy), SNR=80.0))
        # One broad galaxy-like source: its EE80 radius is ~2x the stars'.
        img += 6e3 * np.exp(
            -((xx - 200) ** 2 + (yy - 200) ** 2) / (2 * (2 * sigma) ** 2)
        )
        rows.append(dict(x_pix=200.0, y_pix=200.0, SNR=50.0))
        return img, pd.DataFrame(rows), fwhm

    def test_rescue_does_not_readmit_clip_outlier(self):
        img, sources, fwhm = self._field()
        yaml_ = {
            "fwhm": fwhm,
            "gain": 1.0,
            "write_dir": "/tmp",
            "base": "t",
            "photometry": {},
        }
        out, r_fwhm, scale = Aperture(yaml_, img).measure_optimum_radius(
            sources=sources.copy(), plot=False, n_jobs=1,
            background_rms=np.full_like(img, 15.0),
        )
        outlier = np.isclose(out["x_pix"], 200.0) & np.isclose(
            out["y_pix"], 200.0
        )
        assert not outlier.any()
        # The four clean stars must be rescued so the field is not starved.
        assert len(out) >= 4
