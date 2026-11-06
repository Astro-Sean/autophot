"""Regression tests for the PSF-fitter bug sweep.

Covers:
* centroid_com_with_error: NaN-safe accumulation - a single bad pixel must
  not poison the centroid, masked pixels are skipped, and a fully invalid
  cutout returns NaN rather than raising.
* PoissonLikelihoodFitter: the analytic gradient must agree with a
  finite-difference of the Poisson log-likelihood (the data term n_i was
  previously omitted), and the reported covariance must come from the
  positive information matrix -H rather than the negative-definite
  Hessian.
* MCMCFitter: an undersized walker ensemble is raised to the minimum
  affine-invariant count instead of silently producing degenerate chains.
* _ConstantLocalBackground: the target-fit last-resort background must
  return the global median, not a tiny annulus estimate.
* get_quartic_kernel: the returned kernel is finite, normalized, peaked
  at the centre, and bounded in L1 norm.
* _PositiveFluxFitter: a negative-going least-squares solution must pin
  at the flux >= 0 bound for single and grouped (flat-model) fits -
  photutils drops parameter bounds when it builds the flat group model,
  so the bound has to be applied to the model the fitter receives.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from psf import (
    MCMCFitter,
    PoissonLikelihoodFitter,
    _connected_source_groups,
    _ConstantLocalBackground,
    _epsf_ee_radius_native,
    _positive_flux_policy,
    _PositiveFluxFitter,
    centroid_com_with_error,
    get_quartic_kernel,
)

ImagePSF = pytest.importorskip("photutils.psf", reason="photutils").ImagePSF


class TestConnectedSourceGroups:
    def test_nearby_targets_share_group(self):
        groups = _connected_source_groups(
            [100.0, 104.0],
            [100.0, 101.0],
            min_separation=5.0,
        )
        assert groups.tolist() == [0, 0]

    def test_distant_targets_are_separate(self):
        groups = _connected_source_groups(
            [100.0, 150.0],
            [100.0, 100.0],
            min_separation=5.0,
        )
        assert groups.tolist() == [0, 1]

    def test_grouping_is_transitive(self):
        groups = _connected_source_groups(
            [100.0, 104.0, 108.0, 150.0],
            [100.0, 100.0, 100.0, 100.0],
            min_separation=5.0,
        )
        assert groups.tolist() == [0, 0, 0, 1]


def _gaussian_stamp(shape=21, x0=10.0, y0=10.0, sigma=1.5):
    yy, xx = np.mgrid[0:shape, 0:shape]
    img = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
    return img / img.sum()


class TestCentroidNaNSafety:
    def test_single_nan_pixel(self):
        data = np.full((11, 11), 10.0)
        data += 500.0 * _gaussian_stamp(11, x0=5.0, y0=5.0, sigma=1.5)
        data[0, 0] = np.nan
        xc, yc, xe, ye = centroid_com_with_error(data)
        assert np.isfinite(xc) and np.isfinite(yc)
        assert xc == pytest.approx(5.0, abs=0.2)
        assert yc == pytest.approx(5.0, abs=0.2)

    def test_masked_outskirts(self):
        data = np.full((11, 11), 10.0)
        data += 500.0 * _gaussian_stamp(11, x0=5.0, y0=5.0, sigma=1.5)
        mask = np.zeros((11, 11), bool)
        mask[:2, :] = True  # two fully masked edge rows
        xc, yc, _, _ = centroid_com_with_error(data, mask=mask)
        assert np.isfinite(xc) and np.isfinite(yc)
        assert xc == pytest.approx(5.0, abs=0.2)

    def test_all_nan_returns_nan(self):
        xc, yc, xe, ye = centroid_com_with_error(np.full((11, 11), np.nan))
        assert np.isnan(xc) and np.isnan(yc) and np.isnan(xe) and np.isnan(ye)

    def test_all_negative_returns_nan(self):
        xc, yc, _, _ = centroid_com_with_error(np.full((11, 11), -3.0))
        assert np.isnan(xc) and np.isnan(yc)


class TestPoissonLikelihoodFitter:
    def _setup(self, flux_true=800.0, seed=42):
        rng = np.random.default_rng(seed)
        psf_img = _gaussian_stamp(21)
        data = rng.poisson(flux_true * psf_img).astype(float)
        yy, xx = np.mgrid[0:21, 0:21]
        return psf_img, data, xx, yy

    def test_gradient_vanishes_at_mle(self):
        # With data == model the true Poisson gradient is exactly zero.
        # The old derivative omitted the observed-count term
        # (dlnL_dnbar = -1 + 1/n_bar), leaving a gradient of order -N_pix
        # even at the optimum.
        psf_img, _, xx, yy = self._setup()
        data = 900.0 * psf_img
        model = ImagePSF(psf_img, flux=900.0, x_0=10.0, y_0=10.0)
        fitter = PoissonLikelihoodFitter()

        params = np.array(model.parameters, float)
        grad, _ = fitter._compute_derivatives(
            data, model, xx, yy, params, model.param_names
        )
        assert np.all(np.abs(grad) < 1e-3)

    def test_gradient_matches_finite_difference(self):
        # The fitter's own model derivative is a forward difference, so
        # allow its ~25% approximation noise; this check catches sign and
        # order-of-magnitude errors (the missing-n_i bug was ~100x off).
        psf_img, data, xx, yy = self._setup()
        model = ImagePSF(psf_img, flux=900.0, x_0=10.0, y_0=10.0)
        fitter = PoissonLikelihoodFitter()

        params = np.array(model.parameters, float)
        grad, _ = fitter._compute_derivatives(
            data, model, xx, yy, params, model.param_names
        )

        eps = 1e-5 * np.abs(params) + 1e-8
        fd = np.zeros_like(params)
        for i in range(len(params)):
            dp = params.copy()
            dp[i] += eps[i]
            model.parameters = dp
            lp = fitter._log_likelihood(data, model(xx, yy))
            dm = params.copy()
            dm[i] -= eps[i]
            model.parameters = dm
            lm = fitter._log_likelihood(data, model(xx, yy))
            fd[i] = (lp - lm) / (2 * eps[i])
        model.parameters = params

        assert grad == pytest.approx(fd, rel=0.35, abs=0.6)

    def test_fit_recovers_flux_and_positive_covariance(self):
        psf_img, data, xx, yy = self._setup(flux_true=800.0)
        model = ImagePSF(psf_img, flux=1000.0, x_0=10.3, y_0=9.8)
        fitter = PoissonLikelihoodFitter()

        res = fitter(model, xx, yy, data, maxiters=30)

        # Poisson MLE flux equals total counts; 800-count draw lands ~820.
        assert res.flux.value == pytest.approx(800.0, abs=80.0)
        assert res.x_0.value == pytest.approx(10.0, abs=0.2)
        assert res.y_0.value == pytest.approx(10.0, abs=0.2)

        cov = fitter.fit_info["param_cov"]
        assert np.all(np.isfinite(cov))
        assert np.all(np.diag(cov) > 0)
        errs = fitter.fit_info["param_errs"]
        assert np.all(np.isfinite(errs))
        # Poisson flux error ~ sqrt(N); sanity band around sqrt(800) ~ 28.
        assert errs[0] == pytest.approx(np.sqrt(800.0), rel=0.5)

    def test_weighted_fit_does_not_double_count_source_poisson(self):
        # The pipeline builds photutils weights from calc_total_error:
        #   sigma = sqrt(bkg_error^2 + data)   (source Poisson already inside)
        # while the Poisson Hessian adds n_bar for the source.  Passing those
        # weights must not count the source photons twice, which would inflate
        # the flux error toward sqrt(2 N) ~ 40 instead of sqrt(N) ~ 28.
        psf_img, data, xx, yy = self._setup(flux_true=800.0)
        model = ImagePSF(psf_img, flux=1000.0, x_0=10.3, y_0=9.8)
        fitter = PoissonLikelihoodFitter()

        bkg = 1.0
        total_error = np.sqrt(bkg**2 + np.clip(data, 0.0, None))
        weights = 1.0 / total_error

        fitter(model, xx, yy, data, weights=weights, maxiters=30)

        errs = fitter.fit_info["param_errs"]
        assert np.all(np.isfinite(errs))
        assert errs[0] == pytest.approx(np.sqrt(800.0), rel=0.35)
        # Explicitly guard against the double-counted curvature.
        assert errs[0] < 0.85 * np.sqrt(2.0 * 800.0)

    def test_position_shift_not_vetoed_by_edge_zeros(self):
        # An ePSF model evaluated a hair outside its interpolator domain
        # renders exact zeros at the border; the old `all(model > 0)` gate
        # vetoed every such step and froze the fit at iteration 0.
        psf_img, data, xx, yy = self._setup()
        model = ImagePSF(psf_img, flux=1000.0, x_0=9.5, y_0=10.0)
        fitter = PoissonLikelihoodFitter()
        res = fitter(model, xx, yy, data, maxiters=30)
        assert res.x_0.value == pytest.approx(10.0, abs=0.3)


class TestConstantLocalBackground:
    def test_returns_global_value_per_source(self):
        bkg = _ConstantLocalBackground(3.25)
        out = bkg(None, np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]))
        assert np.allclose(out, 3.25)
        assert len(out) == 3

    def test_scalar_position(self):
        bkg = _ConstantLocalBackground(-1.5)
        out = bkg(None, np.array([7.0]), np.array([9.0]))
        assert np.allclose(out, -1.5)


class TestQuarticKernelValidation:
    def test_kernel_is_valid_smoothing(self):
        k = get_quartic_kernel(4)
        assert np.all(np.isfinite(k))
        assert k.sum() == pytest.approx(1.0)
        half = k.shape[0] // 2
        assert k[half, half] > 0
        assert np.abs(k).sum() <= 3.0


class TestEpsfEERadiusNative:
    def test_compact_gaussian(self):
        stamp = _gaussian_stamp(31, x0=15.0, y0=15.0, sigma=1.5)
        r95 = _epsf_ee_radius_native(stamp, 1, 0.95)
        # sigma=1.5 Gaussian: 95% EE at r ~ 3.7 native px.
        assert r95 == pytest.approx(3.7, abs=1.0)

    def test_extended_wings_grow_radius(self):
        # Moffat-like stamp with slow-decaying wings (beta=2, alpha=2).
        yy, xx = np.mgrid[0:41, 0:41]
        r2 = (xx - 20.0) ** 2 + (yy - 20.0) ** 2
        stamp = (1.0 + r2 / 4.0) ** -2.0
        stamp /= stamp.sum()
        r95 = _epsf_ee_radius_native(stamp, 1, 0.95)
        compact = _epsf_ee_radius_native(
            _gaussian_stamp(41, x0=20.0, y0=20.0, sigma=2.0), 1, 0.95
        )
        assert r95 >= compact + 2.0
        assert r95 >= 7.0

    def test_noise_wings_ignored(self):
        # A compact PSF plus symmetric pixel noise should not fake wings:
        # ring medians cancel the noise, so the radius stays compact.
        rng = np.random.default_rng(1)
        stamp = _gaussian_stamp(41, x0=20.0, y0=20.0, sigma=1.2)
        stamp += rng.normal(0.0, 1e-3, stamp.shape) * stamp.max()
        r95 = _epsf_ee_radius_native(stamp, 1, 0.95)
        assert r95 == pytest.approx(3.0, abs=1.5)

    def test_positive_pedestal_does_not_fake_wings(self):
        # A few-star ePSF often carries a median-positive noise floor with
        # ring-to-ring fluctuations (not a uniform offset).  Neither a plain
        # ring median nor a pedestal subtraction alone rejects it; the
        # monotone envelope must keep the radius compact.
        rng = np.random.default_rng(2)
        stamp = _gaussian_stamp(31, x0=15.0, y0=15.0, sigma=1.3)
        stamp /= stamp.sum()
        pedestal = np.abs(rng.normal(0.0, 2e-4, stamp.shape)) * 0.0
        pedestal += np.clip(
            rng.normal(2e-4, 8e-5, stamp.shape), 0.0, None
        )
        stamp = stamp + pedestal
        stamp /= stamp.sum()
        r95 = _epsf_ee_radius_native(stamp, 1, 0.95)
        assert r95 < 7.0

    def test_oversampling_scales_radius(self):
        stamp = _gaussian_stamp(62, x0=30.5, y0=30.5, sigma=3.0)
        r1 = _epsf_ee_radius_native(stamp, 1, 0.95)
        r2 = _epsf_ee_radius_native(stamp, 2, 0.95)
        assert r2 == pytest.approx(r1 / 2.0, rel=0.15)

    def test_invalid_input_returns_nan(self):
        assert np.isnan(_epsf_ee_radius_native(np.zeros((9, 9)), 1, 0.95))
        assert np.isnan(
            _epsf_ee_radius_native(np.full((9, 9), np.nan), 1, 0.95)
        )


class TestPositiveFluxBound:
    """psf_fit_positive_flux must bound the amplitude inside the fitter,
    not just NaN the result afterwards - an unconstrained fit converged
    on a negative solution cannot be recovered downstream."""

    @staticmethod
    def _dip_data(seed=7):
        """Noise field with a -800-count PSF-shaped dip at (17, 17) of a
        70x70 frame plus a +1500-count companion inside the same group."""
        rng = np.random.default_rng(seed)
        stamp = _gaussian_stamp(25, x0=12.0, y0=12.0, sigma=2.0)
        img = rng.normal(0.0, 2.0, (70, 70))
        img[5:30, 5:30] -= 800.0 * stamp
        img[10:35, 22:47] += 1500.0 * stamp
        return stamp, img

    def test_trflsq_single_source_pins_at_zero(self):
        from astropy.modeling.fitting import TRFLSQFitter

        stamp, img = self._dip_data()
        yy, xx = np.mgrid[0:70, 0:70]
        model = ImagePSF(stamp, flux=200.0, x_0=17.0, y_0=17.0)

        unb = TRFLSQFitter()(
            model.copy(), xx.ravel(), yy.ravel(), img.ravel()
        )
        assert unb.flux.value < 0.0  # the unconstrained fit does go negative

        # Fix the centroid so only the amplitude bound is exercised; a
        # free position is legitimately allowed to wander to the best
        # positive noise bump when flux cannot go negative.
        held = model.copy()
        held.x_0.fixed = True
        held.y_0.fixed = True
        wrapped = _PositiveFluxFitter(TRFLSQFitter())
        res = wrapped(held, xx.ravel(), yy.ravel(), img.ravel())
        assert np.isfinite(res.flux.value)
        assert res.flux.value >= 0.0
        assert res.flux.value < 1.0  # pinned at the bound, not refit high

    def test_grouped_flat_model_bound_propagates(self):
        # photutils 3.x builds a flat model (flux_0, flux_1) for groups
        # and drops the template's parameter bounds; the wrapper must
        # apply them to the model the fitter actually sees.
        pytest.importorskip("photutils.psf", reason="photutils")
        from astropy.modeling.fitting import TRFLSQFitter
        from astropy.nddata import NDData
        from astropy.table import QTable
        import astropy.units as u
        from photutils.background import LocalBackground
        from photutils.psf import PSFPhotometry, SourceGrouper

        stamp, img = self._dip_data()
        nd = NDData(data=img, unit=u.electron)
        init = QTable(
            {
                "x": [17.0, 34.0],
                "y": [17.0, 22.0],
                "flux": [200.0, 200.0] * u.electron,
            }
        )

        def _run(fitter):
            ph = PSFPhotometry(
                psf_model=ImagePSF(stamp, flux=200.0, x_0=12.0, y_0=12.0),
                fitter=fitter,
                fit_shape=(15, 15),
                aperture_radius=5,
                local_bkg_estimator=LocalBackground(18, 22),
                grouper=SourceGrouper(min_separation=25.0),
                xy_bounds=3.0,
                progress_bar=False,
            )
            return ph(nd, init_params=init)

        res_unb = _run(TRFLSQFitter())
        assert float(res_unb["flux_fit"][0].value) < 0.0

        res_bnd = _run(_PositiveFluxFitter(TRFLSQFitter()))
        flux = np.asarray(
            [v.value for v in res_bnd["flux_fit"]], float
        )
        assert np.all(np.isfinite(flux))
        assert np.all(flux >= 0.0)
        # The real companion must still be measured, not suppressed.
        assert flux[1] == pytest.approx(1500.0, rel=0.15)

    def test_poisson_fitter_projects_steps_into_bounds(self):
        # All-zero counts pull the Newton step below zero; the bound
        # projection must pin the trial flux at 0 instead.
        stamp = _gaussian_stamp(21)
        data = np.zeros((21, 21))
        yy, xx = np.mgrid[0:21, 0:21]
        model = ImagePSF(stamp, flux=500.0, x_0=10.0, y_0=10.0)
        model.flux.bounds = (0.0, None)
        res = PoissonLikelihoodFitter()(model, xx, yy, data, maxiters=10)
        assert np.isfinite(res.flux.value)
        assert res.flux.value >= 0.0

    def test_wrapper_exposes_maxiter_and_delegates_fit_info(self):
        import inspect

        assert "maxiter" in inspect.signature(
            _PositiveFluxFitter.__call__
        ).parameters
        from astropy.modeling.fitting import TRFLSQFitter

        inner = TRFLSQFitter()
        wrapped = _PositiveFluxFitter(inner)
        stamp = _gaussian_stamp(21)
        yy, xx = np.mgrid[0:21, 0:21]
        wrapped(
            ImagePSF(stamp, flux=100.0, x_0=10.0, y_0=10.0),
            xx.ravel(),
            yy.ravel(),
            (800.0 * stamp).ravel(),
        )
        assert wrapped.fit_info is inner.fit_info

    def test_wrapper_translates_maxiter_to_maxiters(self):
        """photutils calls the fitter with the astropy spelling ``maxiter``;
        MCMCFitter/PoissonLikelihoodFitter spell it ``maxiters``.  The
        wrapper must translate, not forward verbatim."""

        class _Recorder:
            def __init__(self):
                self.seen = None

            def __call__(self, model, x, y, data, weights=None,
                         maxiters=None, **kwargs):
                self.seen = maxiters
                return model

        inner = _Recorder()
        wrapped = _PositiveFluxFitter(inner)
        stamp = _gaussian_stamp(21)
        yy, xx = np.mgrid[0:21, 0:21]
        model = ImagePSF(stamp, flux=100.0, x_0=10.0, y_0=10.0)
        wrapped(model, xx.ravel(), yy.ravel(), (800.0 * stamp).ravel(),
                maxiter=42)
        assert inner.seen == 42

    def test_wrapper_drops_maxiter_when_fitter_lacks_it(self):
        """A wrapped fitter with no iteration-cap parameter must not be
        called with a ``maxiter`` it cannot accept."""

        class _Recorder:
            def __init__(self):
                self.kwargs = None

            def __call__(self, model, x, y, data, weights=None, **kwargs):
                self.kwargs = kwargs
                return model

        inner = _Recorder()
        wrapped = _PositiveFluxFitter(inner)
        stamp = _gaussian_stamp(21)
        yy, xx = np.mgrid[0:21, 0:21]
        model = ImagePSF(stamp, flux=100.0, x_0=10.0, y_0=10.0)
        wrapped(model, xx.ravel(), yy.ravel(), (800.0 * stamp).ravel(),
                maxiter=42)
        assert "maxiter" not in inner.kwargs
        assert "maxiters" not in inner.kwargs


class TestPositiveFluxPolicy:
    """Default policy: fitted amplitudes are bounded at >=0 on every
    image, difference images included.  psf_diff_negative_flux is the
    explicit opt-out for signed residuals."""

    def test_science_image_default(self):
        assert _positive_flux_policy({}, is_difference_image=False)

    def test_difference_image_bounded_by_default(self):
        # The TDP8 regression: CONVD=REF must not silently allow a
        # negative primary amplitude.
        assert _positive_flux_policy({}, is_difference_image=True)

    def test_diff_opt_out(self):
        cfg = {"psf_diff_negative_flux": True}
        assert not _positive_flux_policy(cfg, is_difference_image=True)
        # The opt-out must not affect normal images.
        assert _positive_flux_policy(cfg, is_difference_image=False)

    def test_master_switch_off(self):
        cfg = {"psf_fit_positive_flux": False}
        assert not _positive_flux_policy(cfg, is_difference_image=False)
        assert not _positive_flux_policy(cfg, is_difference_image=True)


class TestAnalyticPsfPlot:
    """plot_analytic_psf renders the star/residual grid and per-component
    projections for analytic-model deliveries, and fails soft."""

    @staticmethod
    def _stamp_and_stars(seed=3):
        from types import SimpleNamespace

        from psf import _analytic_psf_stamp

        rng = np.random.default_rng(seed)
        fwhm = 6.0
        stars = []
        for _ in range(4):
            n = 41
            c = (n - 1) / 2 + rng.uniform(-0.3, 0.3, 2)
            yy, xx = np.indices((n, n))
            rr = np.hypot(xx - c[0], yy - c[1])
            prof = 0.02 * (1 + (rr / 2.4) ** 2) ** (-4.0) + 0.004 * (
                1 + (rr / 7.5) ** 2
            ) ** (-2.0)
            flux = 4000.0
            stars.append(
                SimpleNamespace(
                    data=prof * flux + rng.normal(0, 2.0, prof.shape),
                    cutout_center=c,
                    weights=np.ones_like(prof),
                    mask=None,
                    flux=flux,
                )
            )
        stamp, comps = _analytic_psf_stamp(fwhm, 1, 41, 4.765, stars=stars)
        return stamp, stars

    def test_renders_and_saves(self, tmp_path):
        from psf import plot_analytic_psf

        stamp, stars = self._stamp_and_stars()
        out = tmp_path / "analytic.png"
        assert (
            plot_analytic_psf(
                stamp, stars=stars, fwhm_native=6.0, save_path=str(out)
            )
            is None
        )
        assert out.exists() and out.stat().st_size > 10000

    def test_fails_soft_without_components(self, tmp_path):
        from types import SimpleNamespace

        from psf import plot_analytic_psf

        bad = SimpleNamespace(
            data=np.zeros((9, 9)), oversampling=1
        )
        assert (
            plot_analytic_psf(bad, save_path=str(tmp_path / "x.png"))
            is None
        )


class TestAnalyticCalibratorFit:
    def test_analytic_psf_recovers_catalog_positions_and_fluxes(self):
        import pandas as pd

        from psf import PSF, _analytic_psf_stamp

        gain = 2.0
        exposure_time = 30.0
        model, _ = _analytic_psf_stamp(
            3.0, 1, 31, 4.765, stars=None
        )
        yy, xx = np.mgrid[:128, :128]
        positions = np.array(
            [[30.3, 32.6], [64.4, 33.1], [45.6, 77.2]], dtype=float
        )
        fluxes_e = np.array([50000.0, 30000.0, 80000.0])
        image_e = np.zeros((128, 128), dtype=float)
        for (x, y), flux in zip(positions, fluxes_e):
            image_e += model.evaluate(
                xx, yy, flux=flux, x_0=x, y_0=y
            )
        image_adu = image_e / gain
        cfg = {
            "fwhm": 3.0,
            "gain": gain,
            "read_noise": 0.0,
            "exposure_time": exposure_time,
            "imageFilter": "r",
            "pixel_scale": 1.0,
            "scale": 15,
            "photometry": {
                "aperture_radius": 5.0,
                "fitting_xy_bounds": 2.0,
                "fitting_xy_bounds_detect_snr": 0.0,
                "psf_fit_shape_bright_scale_fwhm": 3.0,
                "psf_fit_shape_bright_min_arcsec": 0.0,
                "psf_localbkg_ee_min": 0.97,
                "psf_fit_positive_flux": True,
                "do_AperturePhotometry": False,
                "psf_target_fit_shape_boost": 1.0,
            },
        }
        sources = pd.DataFrame(
            {
                "x_pix": positions[:, 0] + 0.25,
                "y_pix": positions[:, 1] - 0.2,
                "counts_AP": fluxes_e,
            }
        )

        measured = PSF(cfg, image_adu, header={}).fit(
            epsf_model=model,
            sources=sources,
            background_rms=np.full_like(image_adu, 0.5),
            plot=False,
        )

        np.testing.assert_allclose(
            measured["x_fit"], positions[:, 0], atol=0.05
        )
        np.testing.assert_allclose(
            measured["y_fit"], positions[:, 1], atol=0.05
        )
        np.testing.assert_allclose(
            measured["flux_PSF"],
            fluxes_e / exposure_time,
            rtol=0.03,
        )


class TestMCMCWalkerFloor:
    def test_undersized_ensemble_is_raised(self, caplog):
        psf_img = _gaussian_stamp(11, x0=5.0, y0=5.0)
        rng = np.random.default_rng(0)
        data = 500.0 * psf_img + rng.normal(0, 5, psf_img.shape)
        yy, xx = np.mgrid[0:11, 0:11]

        model = ImagePSF(psf_img, flux=400.0, x_0=5.0, y_0=5.0)
        fitter = MCMCFitter(
            nwalkers=2,
            nsteps=10,
            batch_steps=5,
            thin=1,
            early_stop=False,
            min_autocorr_N=10**9,
        )
        with caplog.at_level(logging.WARNING):
            fitter(model, xx, yy, data, noise_variance=25.0)

        # ndim=3 -> minimum ensemble 2*3+2 = 8 walkers.
        assert fitter.nwalkers == 8
        assert any("too small" in r.getMessage() for r in caplog.records)


class TestMoffatCompositeStarCap:
    """_fit_moffat_composite must cap components by independent star count:
    one cutout contributes hundreds of correlated pixels but only one PSF
    realisation, so few stars cannot support a multi-component fit."""

    @staticmethod
    def _star(sigma=1.6, flux=5000.0, shape=21):
        from types import SimpleNamespace

        data = flux * _gaussian_stamp(shape, sigma=sigma) + 100.0
        return SimpleNamespace(
            data=data,
            cutout_center=np.array([(shape - 1) / 2.0, (shape - 1) / 2.0]),
            mask=None,
            weights=None,
            flux=flux,
        )

    def test_single_star_yields_single_component(self):
        from psf import _fit_moffat_composite

        comps = _fit_moffat_composite([self._star()], fwhm=4.0)
        assert comps is not None
        assert len(comps) == 1

    def test_three_stars_capped_below_component_budget(self):
        """The tried count is capped at n_stars - 1; three stars cannot
        support the full three-component model."""
        from psf import _fit_moffat_composite

        rng = np.random.default_rng(3)
        stars = []
        for _ in range(3):
            s = self._star()
            s.data = s.data + rng.normal(0, 2.0, s.data.shape)
            stars.append(s)
        comps = _fit_moffat_composite(stars, fwhm=4.0)
        assert comps is not None
        assert 1 <= len(comps) <= 2

    def test_many_stars_allow_multiple_components(self):
        from psf import _fit_moffat_composite

        rng = np.random.default_rng(7)
        stars = []
        for _ in range(8):
            s = self._star()
            s.data = s.data + rng.normal(0, 1.0, s.data.shape)
            stars.append(s)
        comps = _fit_moffat_composite(stars, fwhm=4.0)
        assert comps is not None
        assert 1 <= len(comps) <= 3

    def test_moffat_truth_stays_below_full_component_budget(self):
        # A single-Moffat profile must not consume the full component
        # budget - BIC only arbitrates between same-count variants in the
        # current fitter, so the guard exercised here is the collapse of
        # a component below the amplitude floor.
        from psf import _fit_moffat_composite

        def moffat(r, amp, gamma, beta):
            return amp * (1.0 + (r / gamma) ** 2) ** (-beta)

        rng = np.random.default_rng(11)
        stars = []
        for _ in range(6):
            s = self._star(flux=8000.0, shape=41)
            yy, xx = np.indices(s.data.shape)
            r = np.hypot(xx - 20.0, yy - 20.0)
            s.data = 8000.0 * moffat(r, 0.05, 2.0, 4.5) + rng.normal(
                0.0, 0.5, s.data.shape
            )
            s.cutout_center = np.array([20.0, 20.0])
            stars.append(s)
        comps = _fit_moffat_composite(stars, fwhm=5.0)
        assert comps is not None
        assert 1 <= len(comps) <= 2

    def test_bic_recovers_second_component_when_supported(self):
        # A genuine core+wing profile with enough stars must select the
        # second component - under-fitting is the failure mode the BIC
        # guards against.
        from psf import _fit_moffat_composite

        def moffat(r, amp, gamma, beta):
            return amp * (1.0 + (r / gamma) ** 2) ** (-beta)

        rng = np.random.default_rng(13)
        stars = []
        for _ in range(6):
            s = self._star(flux=8000.0, shape=41)
            yy, xx = np.indices(s.data.shape)
            r = np.hypot(xx - 20.0, yy - 20.0)
            s.data = 8000.0 * (
                moffat(r, 0.05, 2.0, 4.5) + moffat(r, 0.004, 8.0, 1.8)
            ) + rng.normal(0.0, 0.5, s.data.shape)
            s.cutout_center = np.array([20.0, 20.0])
            stars.append(s)
        comps = _fit_moffat_composite(stars, fwhm=5.0)
        assert comps is not None
        # At least the second (wing) component must be selected; BIC may
        # add a marginal third term (bound-pinned far-wing bump), which is
        # harmless to the profile - under-fitting to a single Moffat is
        # the failure this guards.
        assert len(comps) >= 2
        # The fitted profile - not the component values - must reproduce
        # the true wing level at large radius.
        from psf import _moffat_sum_profile

        truth = (
            moffat(12.0, 0.05, 2.0, 4.5) + moffat(12.0, 0.004, 8.0, 1.8)
        )
        pred = _moffat_sum_profile(np.array([12.0]), np.ravel(comps), len(comps))[0]
        assert pred == pytest.approx(truth, rel=0.5)
