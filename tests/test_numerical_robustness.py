"""Regression tests for the numerical-robustness audit fixes.

Covers: the haversine separation in additional_targets.py, NaN-poisoned
MCMC diagnostics, and the median-SE convention in utils/psf_validation.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from autophot.services.additional_targets import _angular_sep_arcsec
from autophot.analysis.mcmc_diagnostics import (
    assess_convergence,
    MCMC_FLAG_NAN_DIAG,
)


class TestAngularSeparation:
    def test_high_dec_not_underestimated(self):
        # 1 arcmin true dDec at dec=60 must read ~60 arcsec; the old
        # code double-applied cos(dec) inside the haversine.
        sep = float(
            _angular_sep_arcsec(100.0, 60.0, 100.0, 60.0 + 1.0 / 60.0)[0]
        )
        assert abs(sep - 60.0) < 0.1

    def test_high_dec_ra_offset(self):
        # 1 arcmin of *angle* at dec=60 is 2 arcmin of RA.  With the
        # double-cos bug the returned separation was ~60 arcsec
        # (underestimate by ~2x).
        sep = float(
            _angular_sep_arcsec(100.0, 60.0, 100.0 + 2.0 / 60.0, 60.0)[0]
        )
        assert abs(sep - 60.0) < 0.5

    def test_ra_wraparound(self):
        # 0.2 deg of RA across the 0/360 boundary near the equator.
        sep = float(_angular_sep_arcsec(359.9, 0.0, 0.1, 0.0)[0])
        assert abs(sep - 0.2 * 3600.0) < 1.0


class TestNaNPoisonedMCMC:
    def _good_chain(self, seed=0):
        rng = np.random.default_rng(seed)
        # 8 walkers x 400 steps x 2 params, well-mixed.
        return rng.normal(size=(8, 400, 2)) + np.linspace(
            0, 0.05, 400
        )[None, :, None]

    def test_nan_param_flags_and_fails_convergence(self):
        chain = self._good_chain()
        chain[:, :, 1] = np.nan  # one fully poisoned parameter
        rep = assess_convergence(
            chain, acceptance_fraction=0.4, min_steps=100
        )
        assert rep.flags & MCMC_FLAG_NAN_DIAG
        assert not rep.converged

    def test_partial_nan_walker_flags(self):
        chain = self._good_chain()
        chain[3, 100:, 0] = np.nan  # one walker dies mid-run
        rep = assess_convergence(
            chain, acceptance_fraction=0.4, min_steps=100
        )
        assert rep.flags & MCMC_FLAG_NAN_DIAG
        assert not rep.converged

    def test_nan_acceptance_handled(self):
        chain = self._good_chain()
        acc = np.full(8, 0.4)
        acc[0] = np.nan
        rep = assess_convergence(
            chain, acceptance_fraction=acc, min_steps=100
        )
        assert np.isfinite(rep.acceptance_fraction)


class TestApertureWorkerNoise:
    """_measure_worker must carry the annulus-estimator noise term."""

    def _make_args(self, img, r_ap=5.0, r_in=8.0, r_out=10.0, pos=60.0):
        from photutils.aperture import (
            CircularAperture,
            CircularAnnulus,
            aperture_photometry,
        )

        ap = CircularAperture([(pos, pos)], r=r_ap)
        ann = CircularAnnulus([(pos, pos)], r_in=r_in, r_out=r_out)
        phot = aperture_photometry(img, ap).to_pandas()
        args = (
            0,
            [a.to_mask(method="center") for a in ap],
            [a.to_mask(method="center") for a in ann],
            img,                    # image_e
            None,                   # error -> fallback variance path
            0.0,                    # read_noise_sq
            1.0,                    # inv_exposure_time
            float(np.pi * r_ap**2),  # area
            phot,
            1.0,                    # gain
            False,                  # enforce_nonnegative_local_bkg
            False,                  # verbose
            None,                   # defects_mask
        )
        return args

    def test_annulus_noise_term_present(self):
        from autophot.photometry.aperture import _measure_worker

        rng = np.random.RandomState(7)
        img = rng.normal(100.0, 10.0, (120, 120))
        yy, xx = np.mgrid[:120, :120]
        img += 500.0 * np.exp(
            -((xx - 60.0) ** 2 + (yy - 60.0) ** 2) / (2 * 1.5**2)
        )
        res = _measure_worker(self._make_args(img))
        assert "fail_reason" not in res

        # Thin annulus (~113 px), wide aperture (~78 px): the
        # A^2 * SE(median)^2 term is comparable to the pixel noise, so the
        # error must exceed the no-annulus fallback by ~40%.
        aperture_sum = res["counts_AP"]
        fallback = np.sqrt(abs(aperture_sum) + res["area"] * 10.0**2)
        assert res["flux_AP_err"] > 1.2 * fallback

    def test_quantized_annulus_uses_biweight_fallback(self):
        from autophot.photometry.aperture import _measure_worker

        rng = np.random.RandomState(11)
        # sigma=0.4 ADU quantized to integers: >50% of annulus pixels
        # repeat the median value, so MAD collapses to 0 while a real
        # sub-ADU scatter exists - the biweight fallback keeps the source
        # measurable instead of failing with bkg_invalid.
        img = np.round(rng.normal(100.0, 0.4, (120, 120))).astype(float)
        yy, xx = np.mgrid[:120, :120]
        _star = 500.0 * np.exp(
            -((xx - 60.0) ** 2 + (yy - 60.0) ** 2) / (2 * 1.5**2)
        )
        # Keep the annulus exactly integer-valued so MAD == 0 strictly;
        # even a 1e-22 wing would leave a nonzero-but-meaningless MAD.
        _core = (xx - 60.0) ** 2 + (yy - 60.0) ** 2 < 8.0**2
        img[_core] += _star[_core]
        res = _measure_worker(self._make_args(img))
        assert "fail_reason" not in res
        assert np.isfinite(res["flux_AP_err"]) and res["flux_AP_err"] > 0
        assert res["noiseSky"] > 0
