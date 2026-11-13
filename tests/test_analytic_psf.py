"""Regression tests for the analytic (composite-Moffat) PSF fallback.

The fitted composite components and the canonical profile are specified by
effective (pixel-integrated) widths, so the delivered stamp must be
point-sampled: integrating the continuous model over the pixel a second
time widened the stamp ~10% (2.0 px truth rendered at 2.25 px).  These
tests build the truth independently with photutils' pixel-integrated
MoffatPRF and check the stamp against it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

psf = pytest.importorskip("autophot.photometry.psf")
pytest.importorskip("photutils.psf")
from photutils.psf import MoffatPRF

from autophot.analysis import limits
import autophot.pipeline.main as main_mod

TRUTH_BETA = 3.0


def _truth_prf(fwhm, n, dx=0.0, dy=0.0):
    """Unit-sum pixel-integrated Moffat at an optionally shifted centre."""
    gamma = float(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / TRUTH_BETA) - 1.0))
    ctr = (n - 1) / 2.0
    yy, xx = np.indices((n, n))
    m = MoffatPRF(
        flux=1.0,
        x_0=ctr + dx,
        y_0=ctr + dy,
        alpha=max(gamma, 1e-6),
        beta=TRUTH_BETA,
    )
    data = np.asarray(m(xx, yy), dtype=float)
    return data / np.nansum(data)


def _make_stars(fwhm, n_stars=8, flux=2e5, noise=5.0, rng=None):
    """Star cutouts whose truth is the pixel-integrated Moffat above."""
    rng = rng or np.random.default_rng(1)
    n = 2 * int(np.ceil(6.0 * fwhm)) + 1
    ctr = (n - 1) / 2.0
    stars = []
    for _ in range(n_stars):
        dx, dy = rng.uniform(-0.5, 0.5, size=2)
        data = flux * _truth_prf(fwhm, n, dx=dx, dy=dy)
        data = data + rng.normal(0.0, noise, size=data.shape)
        stars.append(
            SimpleNamespace(
                data=data,
                cutout_center=np.array([ctr + dx, ctr + dy]),
                weights=np.ones_like(data),
                mask=None,
                flux=np.nan,
            )
        )
    return stars


def _stamp_fwhm(stamp, osamp=1):
    return psf.measure_epsf_fwhm_native(np.asarray(stamp.data, float), osamp)


def _axis_ratio(data):
    """Second-moment minor/major axis ratio of a stamp."""
    yy, xx = np.indices(data.shape)
    w = np.clip(np.asarray(data, float), 0.0, None)
    w = w / np.nansum(w)
    cx = np.sum(xx * w)
    cy = np.sum(yy * w)
    mxx = np.sum((xx - cx) ** 2 * w)
    myy = np.sum((yy - cy) ** 2 * w)
    mxy = np.sum((xx - cx) * (yy - cy) * w)
    cov = np.array([[mxx, mxy], [mxy, myy]])
    vals = np.linalg.eigvalsh(cov)
    return float(np.sqrt(vals[0] / vals[1]))


class TestCanonicalStamp:
    @pytest.mark.parametrize("fwhm", [2.0, 3.0, 5.0])
    def test_fwhm_and_provenance(self, fwhm):
        n = 2 * int(np.ceil(6.0 * fwhm)) + 1
        stamp, _ = psf._analytic_psf_stamp(fwhm, 1, n, 4.765, stars=None)
        assert _stamp_fwhm(stamp) == pytest.approx(fwhm, rel=0.02)
        assert stamp._autophot_kind == "analytic-canonical"
        assert stamp._autophot_fitted is False
        assert stamp._autophot_integration == "effective"
        assert np.isclose(np.nansum(stamp.data), 1.0)


class TestFittedStamp:
    @pytest.mark.parametrize("fwhm", [2.0, 4.0])
    def test_matches_truth(self, fwhm):
        n = 2 * int(np.ceil(6.0 * fwhm)) + 1
        truth = _truth_prf(fwhm, n)
        stars = _make_stars(fwhm)
        stamp, _ = psf._analytic_psf_stamp(fwhm, 1, n, TRUTH_BETA, stars=stars)
        assert stamp._autophot_fitted is True
        assert stamp._autophot_kind.startswith("analytic-c")
        assert stamp._autophot_kind != "analytic-canonical"
        assert _stamp_fwhm(stamp) == pytest.approx(
            psf.measure_epsf_fwhm_native(truth, 1), rel=0.03
        )
        st = np.asarray(stamp.data, float)
        a = np.nansum(st * truth) / np.nansum(st**2)
        assert a == pytest.approx(1.0, rel=0.015)


class TestEllipticityPassthrough:
    def test_caller_ellipticity_applies(self):
        stamp, _ = psf._analytic_psf_stamp(
            4.0, 1, 49, 4.765, stars=None, ellipticity=(0.6, 0.3)
        )
        assert stamp._autophot_ellipticity == (0.6, 0.3)
        assert _axis_ratio(np.asarray(stamp.data, float)) < 0.8


class TestGain:
    def test_gain_accepted_and_sanitised(self):
        stars = _make_stars(4.0)
        comps, _ = psf._fit_moffat_composite_full(stars, 4.0, gain=2.0)
        assert comps is not None
        assert np.isfinite(np.asarray(comps, float)).all()
        comps_nan, _ = psf._fit_moffat_composite_full(stars, 4.0, gain=float("nan"))
        assert comps_nan is not None
        assert np.isfinite(np.asarray(comps_nan, float)).all()


class TestInjectionStamp:
    def test_fwhm_and_norm_native(self):
        model = limits._analytic_psf_for_injection(3.0, 1)
        assert model is not None
        assert _stamp_fwhm(model) == pytest.approx(3.0, rel=0.02)
        assert np.nansum(model.data) == pytest.approx(1.0)

    def test_norm_oversampled(self):
        model = limits._analytic_psf_for_injection(3.0, 2)
        assert model is not None
        assert np.nansum(model.data) == pytest.approx(4.0)


class TestFallbackFwhmHelper:
    def test_valid_value(self):
        assert main_mod._analytic_fallback_fwhm(3.2) == 3.2

    @pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf, 0.0, -1.0, "x"])
    def test_invalid_values(self, bad):
        assert main_mod._analytic_fallback_fwhm(bad) is None
