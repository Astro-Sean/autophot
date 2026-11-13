"""Tests for the selectable point-source finder in Find_FWHM.measure_image.

``source_detection.finder`` selects DAOStarFinder ('dao', default) or
IRAFStarFinder ('iraf') for the detection pass that feeds the image-FWHM
median.  DAO convolves with an internal zero-sum Gaussian kernel, so
convolved noise excursions mimic stellar morphologies; its detection
significance therefore defaults to ~4 sigma on the convolved scale
(dao_nsigma) rather than the 3 sigma used on the pre-smoothed image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

photutils = pytest.importorskip("photutils")

from autophot.pipeline.fwhm import Find_FWHM


def _stellar_field(n_stars=8, fwhm=4.0, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, noise, (220, 220))
    yy, xx = np.mgrid[0:220, 0:220]
    sig = fwhm / 2.355
    # Deterministic 60-px spacing: random positions can land inside the
    # crowding filter's minimum separation.
    gx, gy = np.meshgrid(np.arange(40, 220, 60), np.arange(40, 220, 60))
    xs, ys = gx.ravel()[:n_stars], gy.ravel()[:n_stars]
    for cx, cy in zip(xs, ys):
        img += 2500.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig**2))
    return img


def _fwhm_obj(finder):
    return Find_FWHM(
        {
            "saturate": 60000.0,
            "fwhm": 4.0,
            "source_detection": {"finder": finder},
        }
    )


class TestFinderSelection:
    def test_dao_recovers_stellar_fwhm(self):
        img = _stellar_field()
        fw, df, _ = _fwhm_obj("dao").measure_image(img, fwhm_initial=4.0)
        assert np.isfinite(fw)
        assert fw == pytest.approx(4.0, abs=0.6)
        assert len(df) >= 6

    def test_iraf_still_works(self):
        img = _stellar_field()
        fw, df, _ = _fwhm_obj("iraf").measure_image(img, fwhm_initial=4.0)
        assert np.isfinite(fw)
        assert fw == pytest.approx(4.0, abs=0.6)
        assert len(df) >= 6

    def test_default_is_dao(self):
        f = Find_FWHM({"saturate": 60000.0, "fwhm": 4.0})
        img = _stellar_field()
        fw, df, _ = f.measure_image(img, fwhm_initial=4.0)
        assert fw == pytest.approx(4.0, abs=0.6)
        assert len(df) >= 6

    def test_unknown_finder_falls_back_to_dao(self, caplog):
        img = _stellar_field()
        with caplog.at_level("WARNING"):
            fw, df, _ = _fwhm_obj("bogus").measure_image(img, fwhm_initial=4.0)
        assert fw == pytest.approx(4.0, abs=0.6)
        assert len(df) >= 6
        assert any("finder" in r.getMessage().lower() for r in caplog.records)

    def test_dao_rejects_convolved_noise(self):
        # Pure noise: no real stars.  At 4 sigma on the convolved scale,
        # noise blobs must not yield a plausible FWHM median.
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 3.0, (220, 220))
        fw, df, _ = _fwhm_obj("dao").measure_image(img, fwhm_initial=4.0)
        # Either no sources or a handful of marginal detections whose
        # median does not mimic the assumed FWHM.
        if len(df) > 0:
            assert not (np.isfinite(fw) and abs(fw - 4.0) < 0.5)

    def test_roundness_column_synthesized(self):
        # DAOStarFinder reports roundness1/roundness2 only; the cleaning
        # path needs a unified 'roundness' column.
        img = _stellar_field()
        _, df, _ = _fwhm_obj("dao").measure_image(img, fwhm_initial=4.0)
        assert "roundness" in df.columns
        assert np.isfinite(df["roundness"]).all()


class TestKernelRecalibration:
    """A matched-filter finder needs the kernel FWHM near the true PSF;
    a bad initial guess must be corrected from the brightest
    detections, or compact junk dominates the measured median."""

    def test_undersized_kernel_recovers(self):
        # True PSF ~7 px, initial guess 3 - an undersized kernel gives
        # compact noise the best matched response.
        img = _stellar_field(n_stars=8, fwhm=7.0)
        fw, df, _ = _fwhm_obj("dao").measure_image(img, fwhm_initial=3.0)
        assert np.isfinite(fw)
        assert fw == pytest.approx(7.0, rel=0.2)
        assert len(df) >= 4

    def test_oversized_kernel_recovers(self):
        # True PSF ~4 px, initial guess 10 - an oversized kernel can
        # find nothing or only the broadest sources.
        img = _stellar_field(n_stars=8, fwhm=4.0)
        fw, df, _ = _fwhm_obj("dao").measure_image(img, fwhm_initial=10.0)
        assert np.isfinite(fw)
        assert fw == pytest.approx(4.0, rel=0.25)
        assert len(df) >= 4

    def test_correct_guess_unchanged(self):
        # A good guess must not be "refined" away - the band is
        # [0.75, 1.33] x the current kernel size.
        img = _stellar_field(n_stars=8, fwhm=4.0)
        fw, df, _ = _fwhm_obj("dao").measure_image(img, fwhm_initial=4.0)
        assert fw == pytest.approx(4.0, abs=0.6)
        assert len(df) >= 6
