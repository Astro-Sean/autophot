"""Regression tests for sparse-field matched-source recovery and SATURATE
sentinel handling.

Covers the failure modes seen on GROND_g_TDP6_cor_ana_7_MJD60881_34009:

- a stored ``SATURATE=1e10`` "no limit" placeholder must be recognised as
  "no saturation level recorded" everywhere it is read, not capped to a
  bogus 65535 detector level;
- the PSF fit-quality cuts (cfit/qfit/reduced_chi2/flags) each used to test
  their keep-floor against the un-cut count, so successive cuts combined to
  starve the SFFT prior pool (9 -> 3 observed);
- an unbounded centroid could wander off its seed onto a neighbour or mask
  edge, producing a spurious science-vs-template offset that rejected a
  usable prior (14 sources lost to "centroid misalignment" despite a 0.2 px
  systematic offset).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astropy.io import fits
from astropy.wcs import WCS

from functions import SATURATE_SENTINEL_MIN, _bounded_centroid, _keep_floor_cut
from utils.data_validation import (
    VAL_MISSING_SATURATE,
    VAL_SATURATED_PIXELS,
    validate_image,
)


def _gaussian_image(size=64, stars=((32.0, 32.0, 1e5),), fwhm=4.0, bg=100.0):
    """Image with Gaussian stars at given (x, y, flux) positions."""
    y, x = np.mgrid[0:size, 0:size]
    sigma = fwhm / 2.3548
    img = np.full((size, size), bg, dtype=float)
    for cx, cy, flux in stars:
        img += flux * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )
    return img


class TestKeepFloorCut:
    """Floor-aware partial application of PSF-quality cuts."""

    def test_removes_all_flagged_when_pool_survives(self):
        mask = np.ones(10, dtype=bool)
        bad = np.zeros(10, dtype=bool)
        bad[[0, 1, 2]] = True
        metric = np.arange(10.0)
        new_mask, n_removed, n_flagged = _keep_floor_cut(
            mask, bad, metric, min_keep=5
        )
        assert n_flagged == 3
        assert n_removed == 3
        assert new_mask.sum() == 7

    def test_partial_removal_keeps_floor_and_best_ranked(self):
        mask = np.ones(8, dtype=bool)
        bad = np.zeros(8, dtype=bool)
        # Flag rows 4..7 with increasing badness; worst is row 7.
        bad[4:] = True
        metric = np.arange(8.0)
        new_mask, n_removed, n_flagged = _keep_floor_cut(
            mask, bad, metric, min_keep=5
        )
        # room = 8 - 5 = 3 removals; row 4 (least-bad flagged) survives.
        assert n_flagged == 4
        assert n_removed == 3
        assert new_mask.sum() == 5
        assert new_mask[4]
        assert not new_mask[5] and not new_mask[6] and not new_mask[7]

    def test_pool_at_floor_removes_nothing(self):
        mask = np.ones(5, dtype=bool)
        bad = np.zeros(5, dtype=bool)
        bad[[0, 1]] = True
        new_mask, n_removed, n_flagged = _keep_floor_cut(
            mask, bad, np.arange(5.0), min_keep=5
        )
        assert n_flagged == 2
        assert n_removed == 0
        assert new_mask.sum() == 5

    def test_successive_cuts_cannot_starve_pool(self):
        """The GROND failure mode: independent cuts each saw enough rows to
        strip and together removed 9 -> 3.  With the floor evaluated on the
        accumulated mask the pool cannot drop below min_keep."""
        mask = np.ones(9, dtype=bool)
        qfit = np.full(9, 0.5)
        qfit[[6, 7, 8]] = 2.0  # three bad qfit rows
        chi2 = np.full(9, 1.0)
        chi2[[3, 4, 5]] = 20.0  # three bad chi2 rows
        mask, r1, _ = _keep_floor_cut(
            mask, np.isfinite(qfit) & (qfit > 1.0), qfit, 5
        )
        mask, r2, _ = _keep_floor_cut(
            mask, np.isfinite(chi2) & (chi2 > 10.0), chi2, 5
        )
        assert r1 == 3  # 9 -> 6, still >= floor
        assert r2 == 1  # only one more may go: 6 -> 5
        assert mask.sum() == 5

    def test_dead_rows_are_not_recut(self):
        mask = np.ones(6, dtype=bool)
        mask[[0, 1]] = False  # already removed by an earlier cut
        bad = np.zeros(6, dtype=bool)
        bad[[0, 1, 2]] = True
        _, n_removed, n_flagged = _keep_floor_cut(
            mask, bad, np.arange(6.0), min_keep=2
        )
        # Only row 2 is a live candidate.
        assert n_flagged == 1
        assert n_removed == 1

    def test_nan_metric_ranks_worst(self):
        mask = np.ones(5, dtype=bool)
        bad = np.zeros(5, dtype=bool)
        bad[[1, 2]] = True
        metric = np.array([0.0, np.nan, 5.0, 0.0, 0.0])
        new_mask, n_removed, _ = _keep_floor_cut(
            mask, bad, metric, min_keep=4
        )
        assert n_removed == 1
        assert not new_mask[1]  # NaN dropped before the 5.0 outlier
        assert new_mask[2]


class TestBoundedCentroid:
    def test_on_peak_seed_returns_finite(self):
        img = _gaussian_image()
        x_c, y_c = _bounded_centroid(img, 32.3, 31.8, box=13, wander_max=2.5)
        assert np.isfinite(x_c) and np.isfinite(y_c)
        assert abs(x_c - 32.0) < 1.0
        assert abs(y_c - 32.0) < 1.0

    def test_neighbour_grab_rejected(self):
        """A seed on star A whose centroid lands on blended neighbour B is a
        centroiding failure, not a position."""
        img = _gaussian_image(stars=((30.0, 32.0, 1e5), (40.0, 32.0, 1e5)))
        x_c, y_c = _bounded_centroid(img, 30.0, 32.0, box=25, wander_max=2.5)
        assert not np.isfinite(x_c) or abs(x_c - 30.0) <= 2.5

    def test_nan_patch_returns_nan(self):
        img = np.full((64, 64), np.nan)
        x_c, y_c = _bounded_centroid(img, 32.0, 32.0, box=13, wander_max=2.5)
        assert np.isnan(x_c) and np.isnan(y_c)

    def test_no_source_returns_near_seed_or_nan(self):
        """Noise-only box: com stays near the seed or fails; it must never
        return a position beyond wander_max."""
        rng = np.random.default_rng(0)
        img = rng.normal(100.0, 5.0, (64, 64))
        x_c, y_c = _bounded_centroid(img, 32.0, 32.0, box=13, wander_max=2.5)
        if np.isfinite(x_c):
            assert abs(x_c - 32.0) <= 2.5
            assert abs(y_c - 32.0) <= 2.5


class TestSaturateSentinel:
    """Stored no-limit placeholders must read as "no level recorded"."""

    def _header(self, saturate):
        h = fits.Header()
        h["EXPTIME"] = 30.0
        h["GAIN"] = 1.0
        h["RDNOISE"] = 5.0
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [128, 128]
        wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
        wcs.wcs.crval = [150.0, 30.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        h.update(wcs.to_header())
        if saturate is not None:
            h["SATURATE"] = saturate
        return h

    @pytest.mark.parametrize("sentinel", [1e10, 1e30, 1e8])
    def test_sentinel_warns_missing(self, sentinel):
        image = np.ones((64, 64), dtype=float) * 100
        report = validate_image(image, self._header(sentinel))
        assert report.flags & VAL_MISSING_SATURATE
        assert "saturate" not in report.info

    def test_valid_saturate_used(self):
        image = np.ones((64, 64), dtype=float) * 100
        report = validate_image(image, self._header(50000.0))
        assert not (report.flags & VAL_MISSING_SATURATE)
        assert report.info["saturate"] == 50000.0

    def test_non_numeric_saturate_does_not_crash(self):
        image = np.ones((64, 64), dtype=float) * 100
        report = validate_image(image, self._header("junk"))
        assert report.flags & VAL_MISSING_SATURATE

    def test_saturated_pixels_still_flagged(self):
        image = np.ones((64, 64), dtype=float) * 100
        image[:10, :10] = 60000.0
        report = validate_image(image, self._header(50000.0))
        assert report.flags & VAL_SATURATED_PIXELS


class TestStampRadialFwhm:
    """Stamp-morphology veto for SFFT priors: a source wider than
    ``sfft_stamp_fwhm_max_ratio`` x the frame PSF in either image is
    extended and must not anchor the kernel."""

    @staticmethod
    def _field(seed=7):
        rng = np.random.default_rng(seed)
        return rng.normal(100.0, 2.0, (200, 200))

    @staticmethod
    def _add_gauss(img, x, y, fwhm, flux):
        yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
        sig = fwhm / 2.3548
        return img + flux * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig**2)
        )

    def test_star_measures_true_fwhm(self):
        from main import _stamp_radial_fwhm

        img = self._add_gauss(self._field(), 100, 100, 3.0, 5000)
        got = _stamp_radial_fwhm(img, 100.4, 99.6, 3.0)
        assert np.isfinite(got)
        assert abs(got - 3.0) < 0.5

    def test_extended_source_flagged(self):
        from main import _stamp_radial_fwhm

        img = self._add_gauss(self._field(), 60, 60, 8.0, 5000)
        got = _stamp_radial_fwhm(img, 60, 60, 3.0)
        assert np.isfinite(got)
        assert got / 3.0 > 1.6

    def test_source_larger_than_stamp_is_extended(self):
        # A flat-top blob fills the stamp so the sky annulus sits
        # inside the source; the raw-level guard must still flag it
        # rather than hide the extent behind an overestimated sky.
        from main import _stamp_radial_fwhm

        rng = np.random.default_rng(0)
        img = np.full((200, 200), 100.0) + rng.normal(0, 2.0, (200, 200))
        yy, xx = np.mgrid[0:200, 0:200]
        img[np.hypot(xx - 100, yy - 100) < 10] += 5000
        got = _stamp_radial_fwhm(img, 100, 100, 3.0)
        assert np.isfinite(got)
        assert got / 3.0 > 1.6

    def test_unmeasurable_returns_nan(self):
        from main import _stamp_radial_fwhm

        img = self._field()
        # blank sky: no peak
        assert not np.isfinite(_stamp_radial_fwhm(img, 30, 150, 3.0))
        # bad position / bad fwhm
        assert not np.isfinite(_stamp_radial_fwhm(img, np.nan, 50, 3.0))
        assert not np.isfinite(_stamp_radial_fwhm(img, 50, 50, np.nan))


class TestSparsePriorSupplement:
    """Catalog-seeded prior supplementation: only positions showing a
    measurable, non-extended star in BOTH images may join the pool;
    positions covered by a science detection, duplicates, bans, and
    edge stamps are excluded."""

    @staticmethod
    def _images(seed=11):
        rng = np.random.default_rng(seed)
        sci = rng.normal(100.0, 2.0, (300, 300))
        tpl = rng.normal(50.0, 1.0, (300, 300))
        yy, xx = np.mgrid[0:300, 0:300]
        return sci, tpl, xx, yy

    @staticmethod
    def _add(im, x, y, fwhm, flux, xx, yy):
        s = fwhm / 2.3548
        return im + flux * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * s**2)
        )

    def _field(self):
        import pandas as pd

        sci, tpl, xx, yy = self._images()
        for x, y, f in [(50, 50, 3.0), (150, 50, 3.0), (250, 50, 3.0)]:
            sci = self._add(sci, x, y, 3.0, 4000, xx, yy)
            tpl = self._add(tpl, x, y, 5.0, 8000, xx, yy)
        sci = self._add(sci, 150, 150, 9.0, 6000, xx, yy)  # extended in sci
        tpl = self._add(tpl, 150, 150, 5.0, 8000, xx, yy)
        sci = self._add(sci, 60, 220, 3.0, 4000, xx, yy)  # sci-only star
        cat = pd.DataFrame(
            {
                "x_pix": [50.0, 150.0, 250.0, 150.0, 200.0, 60.0, 250.0,
                          40.0, 5.0],
                "y_pix": [50.0, 50.0, 50.0, 150.0, 200.0, 220.0, 250.0,
                          40.0, 5.0],
            }
        )
        return sci, tpl, cat

    def test_only_mutual_stars_kept(self):
        from main import _sparse_prior_supplement

        sci, tpl, cat = self._field()
        out = _sparse_prior_supplement(
            cat,
            existing_xy=np.array([[250.0, 50.0]]),
            covered_xy=np.array([[40.0, 40.0]]),
            ban_xy=np.array([[250.0, 250.0]]),
            image=sci,
            template_image=tpl,
            fwhm_sci=3.0,
            fwhm_tpl=5.0,
            max_ratio=1.6,
            ban_pix=10.0,
            max_add=10,
        )
        kept = {tuple(r) for r in out.tolist()}
        # (50,50) and (150,50) are stars in both; galaxy (150,150),
        # noise (200,200), sci-only (60,220), dup (250,50), covered
        # (40,40), banned (250,250), and edge (5,5) are all rejected.
        assert kept == {(50.0, 50.0), (150.0, 50.0)}

    def test_empty_inputs_and_max_add(self):
        import pandas as pd
        from main import _sparse_prior_supplement

        sci, tpl, cat = self._field()
        assert len(
            _sparse_prior_supplement(
                pd.DataFrame(columns=["x_pix", "y_pix"]),
                np.empty((0, 2)),
                np.empty((0, 2)),
                np.empty((0, 2)),
                sci,
                tpl,
                3.0,
                5.0,
                1.6,
            )
        ) == 0
        out = _sparse_prior_supplement(
            cat,
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty((0, 2)),
            sci,
            tpl,
            3.0,
            5.0,
            1.6,
            max_add=1,
        )
        assert len(out) <= 1


class TestStampRadialFwhmEmptyBins:
    """Pixelated annuli leave some radial bins empty; the crossing
    interpolation must not be poisoned by a NaN neighbour bin."""

    def test_broad_template_star_measures(self):
        from main import _stamp_radial_fwhm

        rng = np.random.default_rng(11)
        tpl = rng.normal(50.0, 1.0, (300, 300))
        yy, xx = np.mgrid[0:300, 0:300]
        s = 5.0 / 2.3548
        tpl += 8000 * np.exp(-((xx - 50.0) ** 2 + (yy - 50.0) ** 2) / (2 * s * s))
        got = _stamp_radial_fwhm(tpl, 50, 50, 5.0)
        assert np.isfinite(got)
        assert abs(got - 5.0) < 1.0
