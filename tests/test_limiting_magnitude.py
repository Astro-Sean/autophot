"""Regression tests for the limiting-magnitude machinery (limits.py).

Covers the post-search calibrated S/N gate, the flux<->magnitude injection
convention, and the per-trial detection-flag reconstruction used to re-solve
the completeness crossing without rerunning photometry.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from autophot.analysis.limits import (
    Limits,
    _analytic_psf_for_injection,
    _flux_for_mag_cached,
    _injection_worker,
    _recovered_detection_flags,
    _render_epsf_on_cutout,
)


def _lim_cfg(**overrides):
    cfg = {
        "fwhm": 5.0,
        "scale": 25.0,
        "gain": 1.0,
        "exposure_time": 60.0,
        "read_noise": 0.0,
        "imageFilter": "w",
        "target_x_pix": 80.0,
        "target_y_pix": 80.0,
        "rng_seed": 42,
        "photometry": {
            "aperture_radius": 8.0,
            "crowded_field": False,
            "enforce_nonnegative_local_background": False,
            "psf_fit_shape_vfaint_scale_fwhm": 2.0,
        },
        "limiting_magnitude": {
            "recovery_method": "PSF",
            "completeness_target": 0.5,
            "inject_source_location": 3,
            "inject_min_radius_fwhm": 2.0,
            "inject_max_radius_fwhm": 5.5,
            "inject_use_quiet_sites": False,
            "inject_max_variance_ratio": 0.0,
            "inject_max_abs_mean_sigma": 0.0,
            "inject_avoid_zero_pixels": True,
            "inject_final_n_sites": 20,
            "inject_candidate_n_sites": 0,
            "injection_jitter_repetitions": 1,
            "m50_bootstrap_n": 0,
            "completeness_solver": "bisect",
            "uncal_from_injections": True,
            "uncal_apply": True,
            "uncal_min_sources": 10,
            "uncal_factor_clip": [0.5, 2.0],
            "recovery_calibrate_snr": True,
            "plot_injection_recovery": False,
        },
    }
    lim = overrides.pop("limiting_magnitude", None)
    phot = overrides.pop("photometry", None)
    cfg.update(overrides)
    if lim:
        cfg["limiting_magnitude"].update(lim)
    if phot:
        cfg["photometry"].update(phot)
    return cfg


def _noise_image(sigma=20.0, shape=(160, 160), seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=shape)


class TestRecoveredDetectionFlags:
    """The rebuilt gate must reproduce _injection_worker's decision exactly."""

    def test_signed_snr_gate_and_error_scale(self):
        flux = np.array([100.0, 50.0, -50.0, np.nan, 40.0])
        err = np.array([25.0, 20.0, 10.0, 5.0, np.nan])
        cfg = {}
        # Raw gate at S/N >= 3: only flux=100/25=4 passes (50/20=2.5 fails,
        # negative flux fails, nans fail).
        got = _recovered_detection_flags(flux, err, 100.0, 3.0, cfg)
        assert got.tolist() == [True, False, False, False, False]
        # Errors inflated x1.5: 100/37.5=2.67 now fails too.
        got2 = _recovered_detection_flags(flux, err * 1.5, 100.0, 3.0, cfg)
        assert got2.tolist() == [False, False, False, False, False]

    def test_zero_or_negative_error_never_detects(self):
        flux = np.array([100.0, 100.0])
        err = np.array([0.0, -3.0])
        got = _recovered_detection_flags(flux, err, 100.0, 3.0, {})
        assert got.tolist() == [False, False]

    def test_require_positive_flux_toggle(self):
        flux = np.array([-100.0])
        err = np.array([10.0])
        on = _recovered_detection_flags(
            flux, err, 100.0, 3.0, {"recovery_require_positive_flux": True}
        )
        off = _recovered_detection_flags(
            flux,
            err,
            100.0,
            3.0,
            {
                "recovery_require_positive_flux": False,
                "recovery_use_absolute_snr": True,
            },
        )
        assert not on[0]
        assert off[0]

    def test_absolute_snr_still_requires_positive_by_default(self):
        # |S/N| >= 3 with a negative flux: passes the S/N half but not the
        # default positive-flux half -- matches the worker's separate gates.
        flux = np.array([-100.0])
        err = np.array([10.0])
        got = _recovered_detection_flags(
            flux, err, 100.0, 3.0, {"recovery_use_absolute_snr": True}
        )
        assert not got[0]

    def test_flux_ratio_gate(self):
        flux = np.array([500.0, 80.0])
        err = np.array([10.0, 10.0])
        cfg = {"recovery_max_flux_ratio": 2.0}
        got = _recovered_detection_flags(flux, err, 100.0, 3.0, cfg)
        # 500 > 2x injected 100 -> rejected; 80 passes.
        assert got.tolist() == [False, True]


class TestFluxForMag:
    def test_unit_flux_round_trip(self):
        counts_ref = 1000.0  # e- per unit flux
        exptime = 60.0
        f_rate = counts_ref / exptime
        m = -2.5 * np.log10(f_rate)
        assert _flux_for_mag_cached(m, counts_ref, exptime) == pytest.approx(1.0)

    def test_scales_tenfold_per_2p5_mag(self):
        f1 = _flux_for_mag_cached(-5.0, 1000.0, 60.0)
        f2 = _flux_for_mag_cached(-7.5, 1000.0, 60.0)
        assert f2 / f1 == pytest.approx(10.0)

    def test_bad_reference_returns_nan(self):
        assert np.isnan(_flux_for_mag_cached(-5.0, 0.0, 60.0))
        assert np.isnan(_flux_for_mag_cached(-5.0, np.nan, 60.0))


class TestWorkerFluxScale:
    """PSF recovery must round-trip the injected amplitude on the same scale
    the magnitude calibration uses (the model flux parameter)."""

    def test_psf_worker_recovers_injected_amplitude(self):
        rng = np.random.default_rng(3)
        fwhm, osamp = 5.0, 2
        model = _analytic_psf_for_injection(fwhm, oversampling=osamp)
        assert model is not None
        cutout = rng.normal(0.0, 10.0, size=(60, 60))
        yaml = _lim_cfg()
        rms = np.full((60, 60), 10.0)
        F_amp = 5.0e3
        det, beta, f_hat, f_err, err_str = _injection_worker(
            (
                30.0,
                30.0,
                F_amp,
                cutout,
                osamp,
                model,
                yaml,
                rms,
                3.0,
                3.0,
                "PSF",
            )
        )
        assert err_str is None
        assert det
        assert np.isfinite(f_hat) and np.isfinite(f_err) and f_err > 0
        # WLS recovers the injected amplitude within a few percent.
        assert f_hat == pytest.approx(F_amp, rel=0.05)

    def test_total_vs_aperture_reference_flux(self):
        # The PSF-mode reference must be the TOTAL unit-render integral, not
        # the aperture integral: for a finite PSF the two differ by the
        # encircled fraction, and picking the wrong one injects every source
        # ~1/EE too bright (spuriously deep limit).
        fwhm = 5.0
        model = _analytic_psf_for_injection(fwhm, oversampling=1)
        H = W = 101
        gridy, gridx = np.indices((H, W))
        unit = model.evaluate(x=gridx, y=gridy, flux=1.0, x_0=50.0, y_0=50.0)
        total = float(np.nansum(unit))
        r = 5.0
        yy, xx = np.mgrid[0:H, 0:W]
        ap = (xx - 50.0) ** 2 + (yy - 50.0) ** 2 <= r * r
        ap_int = float(np.nansum(unit[ap]))
        ee = ap_int / total
        assert 0.3 < ee < 0.95  # sanity: aperture misses real wing flux
        # flux_for_mag on the total scale puts 10^-0.4m*exptime in the whole
        # profile; on the aperture scale it would carry only EE of that.
        exptime = 60.0
        m = -5.0
        F_total = _flux_for_mag_cached(m, total, exptime)
        F_ap = _flux_for_mag_cached(m, ap_int, exptime)
        rendered_total = F_total * total
        assert rendered_total == pytest.approx(10.0 ** (-0.4 * m) * exptime)
        # aperture-scaled injection would over-brighten by 1/EE.
        assert F_ap * total == pytest.approx(rendered_total / ee)


class TestInjectedLimitEndToEnd:
    def _run(self, yaml, sigma=20.0, **kwargs):
        lim = Limits(yaml)
        img = kwargs.pop("image", _noise_image(sigma=sigma))
        return lim.get_injected_limit(
            full_image=img,
            position=(80.0, 80.0),
            epsf_model=kwargs.pop("epsf_model", None),
            initialGuess=kwargs.pop("initialGuess", -3.5),
            detection_limit=kwargs.pop("detection_limit", 3.0),
            plot=False,
            background_rms=kwargs.pop(
                "background_rms", np.full_like(img, sigma)
            ),
            n_jobs=1,
            _return_details=True,
            **kwargs,
        )

    def test_finite_limit_and_details(self):
        yaml = _lim_cfg()
        det = self._run(yaml)
        assert np.isfinite(det["inject_lmag"])
        # photutils dev1329 reports tighter flux_err than dev1232: the
        # empirical calibration lands ~1.33 (boxcar stamp) / ~1.36
        # (MoffatPRF stamp).  The band asserts "roughly calibrated",
        # not exact parity with the old build.
        assert det["snr_gate_factor"] == pytest.approx(1.0, abs=0.45)
        assert "inject_lmag_raw" in det

    def test_noise_scaling_shifts_limit(self):
        det_hi = self._run(_lim_cfg(), sigma=60.0)
        det_lo = self._run(_lim_cfg())
        assert np.isfinite(det_hi["inject_lmag"])
        # Higher sky noise -> shallower (brighter, smaller m) limit.
        assert det_hi["inject_lmag"] < det_lo["inject_lmag"]

    def test_calibrated_gate_shallows_limit(self, monkeypatch):
        """A factor-k error inflation must move the reported limit brighter
        by ~2.5*log10(k): the same trials no longer pass S/N >= threshold
        once errors carry the calibrated scale."""
        k_fake = 1.6
        fake_res = SimpleNamespace(
            calibration_factor=k_fake,
            n_sources=9999,
            z_mad_std=k_fake,
            bias_frac=0.0,
            fraction_within_1sigma=0.68,
            fraction_within_2sigma=0.95,
            to_dict=lambda: {},
        )
        import autophot.analysis.uncertainty_calibration as uc

        monkeypatch.setattr(uc, "calibrate_uncertainties", lambda *a, **k: fake_res)
        det = self._run(_lim_cfg())
        assert np.isfinite(det["inject_lmag"])
        assert det["snr_gate_factor"] == pytest.approx(k_fake)
        assert np.isfinite(det["inject_lmag_raw"])
        # Corrected limit must be shallower (brighter) than the raw crossing,
        # by roughly 2.5*log10(k) in the linear-noise regime.
        assert det["inject_lmag"] < det["inject_lmag_raw"]
        shift = det["inject_lmag_raw"] - det["inject_lmag"]
        assert 0.2 < shift < 1.2

    def test_calibrate_gate_disabled_keeps_raw(self, monkeypatch):
        k_fake = 1.6
        fake_res = SimpleNamespace(
            calibration_factor=k_fake,
            n_sources=9999,
            z_mad_std=k_fake,
            bias_frac=0.0,
            fraction_within_1sigma=0.68,
            fraction_within_2sigma=0.95,
            to_dict=lambda: {},
        )
        import autophot.analysis.uncertainty_calibration as uc

        monkeypatch.setattr(uc, "calibrate_uncertainties", lambda *a, **k: fake_res)
        yaml = _lim_cfg(limiting_magnitude={"recovery_calibrate_snr": False})
        det = self._run(yaml)
        assert np.isfinite(det["inject_lmag"])
        assert det["snr_gate_factor"] == pytest.approx(1.0)
        assert not np.isfinite(det["inject_lmag_raw"])

    def test_mask_covering_annulus_vetoes_all_sites(self):
        # A mask that blankets the injection annulus (e.g. a host-galaxy
        # source footprint inherited from the science frame) leaves no
        # valid sites, so the search must bail out instead of fitting an
        # empty completeness curve.
        img = _noise_image()
        mask = np.zeros_like(img, dtype=bool)
        yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
        mask[(xx - 80.0) ** 2 + (yy - 80.0) ** 2 <= 40.0**2] = True
        det = self._run(_lim_cfg(), image=img, mask=mask)
        assert not np.isfinite(det["inject_lmag"])

    def test_sparse_mask_keeps_sites(self):
        # Masked pixels elsewhere in the frame must not veto annulus
        # sites: masked_sites only rejects sites whose aperture disk
        # overlaps a masked pixel.
        img = _noise_image()
        mask = np.zeros_like(img, dtype=bool)
        yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
        mask[(xx - 15.0) ** 2 + (yy - 15.0) ** 2 <= 25.0**2] = True
        det = self._run(_lim_cfg(), image=img, mask=mask)
        assert np.isfinite(det["inject_lmag"])


class TestOversampledRenderNormalization:
    """photutils ePSF data sums to prod(oversampling); the dense-grid render
    must divide that back out or every injected source carries osamp^2 too
    much flux."""

    def test_detector_flux_conserved(self):
        model = _analytic_psf_for_injection(5.0, oversampling=4)
        assert model is not None
        img = _render_epsf_on_cutout(model, 41, 41, 20.0, 20.0, 1.0, 4)
        assert np.nansum(img) == pytest.approx(1.0, rel=1e-3)

    def test_flux_scales_linearly(self):
        model = _analytic_psf_for_injection(5.0, oversampling=4)
        img = _render_epsf_on_cutout(model, 41, 41, 20.0, 20.0, 2.5, 4)
        assert np.nansum(img) == pytest.approx(2.5, rel=1e-3)

    def test_subpixel_phase_preserved(self):
        model = _analytic_psf_for_injection(5.0, oversampling=4)
        x0, y0 = 20.37, 19.62
        img = _render_epsf_on_cutout(model, 41, 41, x0, y0, 1.0, 4)
        yy, xx = np.indices(img.shape)
        w = np.clip(img, 0.0, None)
        cx = np.sum(xx * w) / np.sum(w)
        cy = np.sum(yy * w) / np.sum(w)
        assert cx == pytest.approx(x0, abs=0.05)
        assert cy == pytest.approx(y0, abs=0.05)

    def test_matches_native_render(self):
        m1 = _analytic_psf_for_injection(5.0, oversampling=1)
        m4 = _analytic_psf_for_injection(5.0, oversampling=4)
        a = _render_epsf_on_cutout(m1, 41, 41, 20.0, 20.0, 1.0, 1)
        b = _render_epsf_on_cutout(m4, 41, 41, 20.0, 20.0, 1.0, 4)
        assert np.nansum(a) == pytest.approx(np.nansum(b), rel=0.05)
        np.testing.assert_allclose(a, b, rtol=0.05, atol=2e-3)


class TestExpectedFluxGate:
    """The worker's flux-ratio gate must compare the recovered quantity
    against the recovery-method scale (args[11]), not the model amplitude."""

    def _worker_args(self, expected_flux, seed=None):
        rng = np.random.default_rng(3)
        model = _analytic_psf_for_injection(5.0, oversampling=2)
        cutout = rng.normal(0.0, 10.0, size=(60, 60))
        yaml = _lim_cfg(
            limiting_magnitude={
                "recovery_max_flux_ratio": 2.0,
                "inject_source_poisson_noise": False,
            }
        )
        rms = np.full((60, 60), 10.0)
        return (
            30.0,
            30.0,
            1.0e3,
            cutout,
            2,
            model,
            yaml,
            rms,
            3.0,
            3.0,
            "PSF",
            expected_flux,
            seed,
        )

    def test_gate_uses_expected_flux_not_amp(self):
        # recovered ~= 1e3; expected 10 -> 100x over-ratio -> rejected even
        # though it matches the injected amplitude itself.
        det, _, f_hat, _, err_str = _injection_worker(self._worker_args(10.0))
        assert err_str is None
        assert np.isfinite(f_hat) and f_hat > 0
        assert not det
        det2, _, _, _, _ = _injection_worker(self._worker_args(1.0e3))
        assert det2


class TestSystematicErrBudget:
    """The injection S/N gate must carry the same systematic error terms the
    real target photometry does, otherwise the reported limit is deeper than
    a real source could reach (main.py folds the kernel-model fraction,
    flux-scale discrepancy, and LPI floor into the measured errors)."""

    def _worker_args(self, **yaml_overrides):
        rng = np.random.default_rng(3)
        model = _analytic_psf_for_injection(5.0, oversampling=2)
        cutout = rng.normal(0.0, 10.0, size=(60, 60))
        yaml = _lim_cfg(
            limiting_magnitude={"inject_source_poisson_noise": False}
        )
        yaml.update(yaml_overrides)
        rms = np.full((60, 60), 10.0)
        return (30.0, 30.0, 5.0e4, cutout, 2, model, yaml, rms, 3.0, 3.0, "PSF")

    def test_kernel_err_frac_demotes_trial(self):
        # Bright source with S/N >> 3 on photon noise alone; a 50% kernel
        # model term pushes the effective S/N to ~2, below the gate.
        det, _, f_hat, f_err, err_str = _injection_worker(
            self._worker_args(psf_kernel_model_err_frac=0.5)
        )
        assert err_str is None
        assert np.isfinite(f_hat) and f_hat > 0
        assert not det
        assert f_err == pytest.approx(abs(f_hat) * 0.5, rel=0.1)

    def test_flux_scale_frac_demotes_trial(self):
        det, _, f_hat, _, err_str = _injection_worker(
            self._worker_args(flux_scale_discrep_frac=0.5)
        )
        assert err_str is None
        assert np.isfinite(f_hat)
        assert not det

    def test_lpi_floor_demotes_trial(self):
        det, _, f_hat, f_err, err_str = _injection_worker(
            self._worker_args(_lpi_extra_flux_err_for_limits=3.0e4)
        )
        assert err_str is None
        assert np.isfinite(f_hat)
        assert not det
        assert f_err >= 3.0e4 - 1.0

    def test_zero_budget_detects(self):
        det, _, _, _, err_str = _injection_worker(self._worker_args())
        assert err_str is None
        assert det


class TestSourcePoissonInjection:
    """inject_source_poisson_noise adds shot noise on the injected source so
    the completeness transition is not artificially sharp; seeds make each
    trial a reproducible independent draw."""

    def _run(self, seed):
        rng = np.random.default_rng(11)
        model = _analytic_psf_for_injection(5.0, oversampling=2)
        cutout = rng.normal(0.0, 10.0, size=(60, 60))
        yaml = _lim_cfg(
            limiting_magnitude={"inject_source_poisson_noise": True}
        )
        rms = np.full((60, 60), 10.0)
        return _injection_worker(
            (
                30.0,
                30.0,
                5.0e4,
                cutout,
                2,
                model,
                yaml,
                rms,
                3.0,
                3.0,
                "PSF",
                5.0e4,
                seed,
            )
        )

    def test_same_seed_reproducible(self):
        _, _, f1, _, _ = self._run(123)
        _, _, f2, _, _ = self._run(123)
        assert f1 == f2

    def test_seeds_scatter_around_amplitude(self):
        f = np.array([self._run(s)[2] for s in range(10)])
        assert np.isfinite(f).all()
        assert np.std(f) > 0
        # Zero-mean increment: the recovered amplitude averages to the
        # injected value.
        assert np.mean(f) == pytest.approx(5.0e4, rel=0.1)

    def test_disabled_is_deterministic(self):
        rng = np.random.default_rng(11)
        model = _analytic_psf_for_injection(5.0, oversampling=2)
        cutout = rng.normal(0.0, 10.0, size=(60, 60))
        yaml = _lim_cfg(
            limiting_magnitude={"inject_source_poisson_noise": False}
        )
        rms = np.full((60, 60), 10.0)

        def run(seed):
            return _injection_worker(
                (
                    30.0,
                    30.0,
                    5.0e4,
                    cutout,
                    2,
                    model,
                    yaml,
                    rms,
                    3.0,
                    3.0,
                    "PSF",
                    5.0e4,
                    seed,
                )
            )[2]

        assert run(1) == run(2)


class TestJointSiteBootstrap:
    """Bootstrap resamples must share one site-index draw across magnitudes;
    hard sites stay hard at every magnitude."""

    def _flags(self, n_sites=20):
        rng = np.random.default_rng(0)
        mags = np.array([-3.0, -4.0, -5.0])
        site_skill = rng.random(n_sites)
        flags = {}
        for i, m in enumerate(mags):
            p = np.clip(0.9 - 0.35 * i + (site_skill - 0.5), 0.05, 0.95)
            flags[float(m)] = rng.random(n_sites) < p
        return flags

    def test_finite_on_smooth_transition(self):
        yaml = _lim_cfg(limiting_magnitude={"m50_bootstrap_n": 100})
        lim = Limits(yaml)
        err = lim._bootstrap_m50_uncertainty(self._flags(), 0.5)
        assert np.isfinite(err) and err >= 0.0

    def test_unequal_flag_counts_returns_nan(self):
        yaml = _lim_cfg(limiting_magnitude={"m50_bootstrap_n": 50})
        lim = Limits(yaml)
        flags = self._flags()
        flags[-5.0] = flags[-5.0][:-1]
        assert np.isnan(lim._bootstrap_m50_uncertainty(flags, 0.5))

    def test_deterministic_with_rng_seed(self):
        yaml = _lim_cfg(limiting_magnitude={"m50_bootstrap_n": 80})
        flags = self._flags()
        a = Limits(yaml)._bootstrap_m50_uncertainty(flags, 0.5)
        b = Limits(yaml)._bootstrap_m50_uncertainty(flags, 0.5)
        assert a == b


class TestRedoHonored:
    """injection_jitter_repetitions must change the number of trials per
    magnitude (previously accepted but ignored)."""

    def test_repetitions_multiply_trials(self, monkeypatch):
        import autophot.analysis.limits as limits_mod

        counts = {}
        real_worker = limits_mod._injection_worker

        def spy(args):
            counts.setdefault(round(float(args[2]), 3), 0)
            counts[round(float(args[2]), 3)] += 1
            return real_worker(args)

        monkeypatch.setattr(limits_mod, "_injection_worker", spy)
        n_sites = 10
        yaml = _lim_cfg(
            limiting_magnitude={
                "inject_final_n_sites": n_sites,
                "injection_jitter_repetitions": 4,
                "inject_source_poisson_noise": False,
            }
        )
        lim = Limits(yaml)
        det = lim.get_injected_limit(
            full_image=_noise_image(),
            position=(80.0, 80.0),
            initialGuess=-3.5,
            detection_limit=3.0,
            plot=False,
            background_rms=np.full((160, 160), 20.0),
            n_jobs=1,
            _return_details=True,
        )
        assert counts, "injection worker was never called"
        for f_amp, n_calls in counts.items():
            assert n_calls % 4 == 0, (
                f"mag with F={f_amp} ran {n_calls} trials, "
                "not a multiple of 4 repetitions"
            )
        assert np.isfinite(det["inject_lmag"])


class TestInjectionSiteFloor:
    """inject_final_n_sites is floored at 6: the completeness fit needs
    a handful of distinct sites to stay honest on spatially varying
    backgrounds - fewer measures a single spot rather than a field
    limit."""

    def test_requested_below_floor_still_gets_six(self):
        yaml = _lim_cfg(
            limiting_magnitude={
                "inject_final_n_sites": 2,
                "injection_jitter_repetitions": 1,
                "inject_source_poisson_noise": False,
            }
        )
        det = Limits(yaml).get_injected_limit(
            full_image=_noise_image(),
            position=(80.0, 80.0),
            initialGuess=-3.5,
            detection_limit=3.0,
            plot=False,
            background_rms=np.full((160, 160), 20.0),
            n_jobs=1,
            _return_details=True,
        )
        sites = det.get("injection_df")
        assert sites is not None and len(sites) >= 6

    def test_shipped_default_at_or_above_floor(self):
        import yaml as yaml_mod
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        cfg = yaml_mod.safe_load(
            (root / "autophot" / "data" / "default_input.yml").read_text()
        )
        n = cfg["default_input"]["limiting_magnitude"][
            "inject_final_n_sites"
        ]
        assert int(n) >= 6


class TestBrightNeighbourSiteVeto:
    """Sites near a bright source must be rejected: masked-source PSF
    wings (distance veto) and compact neighbours the mask missed (peak
    veto) both bias every recovered flux high, faking detections."""

    def _run(self, yaml, img, mask=None):
        return Limits(yaml).get_injected_limit(
            full_image=img,
            position=(80.0, 80.0),
            initialGuess=-3.5,
            detection_limit=3.0,
            plot=False,
            background_rms=np.full((160, 160), 20.0),
            n_jobs=1,
            mask=mask,
            _return_details=True,
        )

    def test_wing_veto_clears_masked_star(self):
        # Bright masked star inside the injection annulus at (100, 80):
        # sites must keep aperture edge + wing_fwhm*FWHM (= 8 + 3*5 = 23)
        # clear of its masked footprint.
        rng = np.random.default_rng(7)
        img = rng.normal(0.0, 20.0, size=(160, 160))
        yy, xx = np.mgrid[0:160, 0:160]
        img = img + 5e4 * np.exp(
            -((xx - 100.0) ** 2 + (yy - 80.0) ** 2) / (2 * (5.0 / 2.355) ** 2)
        )
        mask = (xx - 100.0) ** 2 + (yy - 80.0) ** 2 <= 8.0**2
        det = self._run(_lim_cfg(), img, mask=mask)
        sites = det["injection_df"]
        assert sites is not None and len(sites) > 0
        from scipy import ndimage

        dist_map = ndimage.distance_transform_edt(~mask)
        xs = np.round(sites["x_pix"].to_numpy()).astype(int)
        ys = np.round(sites["y_pix"].to_numpy()).astype(int)
        assert dist_map[ys, xs].min() >= 8.0 + 3.0 * 5.0

    def test_compact_peak_veto_clears_unmasked_source(self):
        # Same star but unmasked: the peak veto must reject sites whose
        # aperture + 1-FWHM guard ring (r = 8 + 5 = 13) contains it.
        rng = np.random.default_rng(7)
        img = rng.normal(0.0, 20.0, size=(160, 160))
        yy, xx = np.mgrid[0:160, 0:160]
        img = img + 3e3 * np.exp(
            -((xx - 100.0) ** 2 + (yy - 80.0) ** 2) / (2 * (5.0 / 2.355) ** 2)
        )
        det = self._run(_lim_cfg(), img, mask=None)
        sites = det["injection_df"]
        assert sites is not None and len(sites) > 0
        d = np.hypot(sites["x_pix"] - 100.0, sites["y_pix"] - 80.0)
        assert d.min() >= 13.0

    def test_veto_disabled_allows_wing_sites(self):
        # With both vetoes off, the same field must admit sites inside
        # what the wing veto would have rejected (dist to star < 31,
        # outside the masked-footprint zone at 16).
        rng = np.random.default_rng(7)
        img = rng.normal(0.0, 20.0, size=(160, 160))
        yy, xx = np.mgrid[0:160, 0:160]
        img = img + 5e4 * np.exp(
            -((xx - 100.0) ** 2 + (yy - 80.0) ** 2) / (2 * (5.0 / 2.355) ** 2)
        )
        mask = (xx - 100.0) ** 2 + (yy - 80.0) ** 2 <= 8.0**2
        yaml = _lim_cfg(
            limiting_magnitude={
                "inject_bright_wing_fwhm": 0.0,
                "inject_neighbour_peak_sigma": 0.0,
            }
        )
        det = self._run(yaml, img, mask=mask)
        sites = det["injection_df"]
        assert sites is not None and len(sites) > 0
        d = np.hypot(sites["x_pix"] - 100.0, sites["y_pix"] - 80.0)
        assert d.min() < 25.0

    def test_starvation_floor_restores_least_contaminated(self, caplog):
        # A large masked footprint in the annulus starves the vetoed
        # pool below inject_final_n_sites: the floor must restore the
        # least-contaminated sites rather than abort to NaN.
        rng = np.random.default_rng(7)
        img = rng.normal(0.0, 20.0, size=(160, 160))
        yy, xx = np.mgrid[0:160, 0:160]
        mask = (xx - 100.0) ** 2 + (yy - 80.0) ** 2 <= 25.0**2
        with caplog.at_level(logging.WARNING, logger="limits"):
            det = self._run(_lim_cfg(), img, mask=mask)
        sites = det["injection_df"]
        assert sites is not None and len(sites) >= 20
        assert np.isfinite(det["inject_lmag"])
        assert any(
            "restoring" in r.getMessage().lower() for r in caplog.records
        )


class TestCompletenessPlotLayout:
    """The completeness figure must keep its legends inside the axes:
    a figure-level legend above the upper panel overlapped the
    apparent-magnitude axis label (user-reported)."""

    def test_no_figure_level_legend(self, tmp_path, monkeypatch):
        pytest.importorskip("photutils")
        import matplotlib.figure

        cfg = _lim_cfg(
            fpath=str(tmp_path / "x.fits"),
            write_dir=str(tmp_path),
        )
        lim = Limits(cfg)
        bracket = [
            (-4.0, 1.0, 10.0, 1.0),
            (-3.0, 0.9, 10.0, 1.0),
            (-2.0, 0.5, 10.0, 1.0),
            (-1.0, 0.1, 10.0, 1.0),
        ]
        saved = []
        real_savefig = matplotlib.figure.Figure.savefig
        monkeypatch.setattr(
            matplotlib.figure.Figure,
            "savefig",
            lambda self, *a, **k: (
                saved.append(self),
                real_savefig(self, *a, **k),
            )[-1],
        )
        lim._plot_completeness(
            None,
            None,
            None,
            bracket,
            [],
            -2.0,
            0.5,
            3.0,
            "PSF",
            position=(80.0, 80.0),
        )
        assert saved, "completeness figure was never saved"
        fig = saved[0]
        assert not fig.legends, (
            "figure-level legend reappeared above the axes - it "
            "overlaps the apparent-magnitude axis label"
        )
        labels = [
            lbl
            for a in fig.axes
            for leg in [a.get_legend()]
            if leg is not None
            for lbl in [t.get_text() for t in leg.get_texts()]
        ]
        assert "Bracket search" in labels


class TestLogisticEmceeGuards:
    """_solve_m50_logistic_emcee must refuse degenerate/prior-dominated
    inputs instead of returning a plausible-looking m50."""

    @staticmethod
    def _lim(tmp_path):
        return Limits(
            {"rng_seed": 0, "write_dir": str(tmp_path), "fpath": "x.fits"}
        )

    @staticmethod
    def _trials_fn(func):
        # run_trials_at_mag(m, redo, pool, return_flags) -> 5-tuple, flags last
        def _fn(m, redo=1, pool=None, return_flags=True):
            flags = np.asarray(func(float(m), int(redo)), dtype=bool)
            return None, None, None, None, flags

        return _fn

    @staticmethod
    def _cfg():
        return {
            "logistic_emcee_nmags": 5,
            "logistic_emcee_nwalkers": 16,
            "logistic_emcee_nsteps": 2000,
            "logistic_emcee_span_mag": 1.5,
        }

    def test_all_detections_rejected(self, tmp_path):
        lim = self._lim(tmp_path)
        fn = self._trials_fn(lambda m, r: np.ones(r))
        with pytest.raises(RuntimeError, match="one-sided"):
            lim._solve_m50_logistic_emcee(
                run_trials_at_mag=fn,
                pool=None,
                m_guess=0.0,
                redo=8,
                completeness_target=0.5,
                lim_cfg=self._cfg(),
            )

    def test_all_misses_rejected(self, tmp_path):
        lim = self._lim(tmp_path)
        fn = self._trials_fn(lambda m, r: np.zeros(r))
        with pytest.raises(RuntimeError, match="one-sided"):
            lim._solve_m50_logistic_emcee(
                run_trials_at_mag=fn,
                pool=None,
                m_guess=0.0,
                redo=8,
                completeness_target=0.5,
                lim_cfg=self._cfg(),
            )

    def test_unbracketed_target_rejected(self, tmp_path):
        # Grid never reaches the 50% crossing: everything detects except a
        # token miss at the faintest point, so the fit is not degenerate
        # but the target is not bracketed by the grid ends.
        lim = self._lim(tmp_path)

        def _outcomes(m, r):
            f = np.ones(r)
            if m > 1.4:  # faintest grid point
                f[0] = 0
            return f

        with pytest.raises(RuntimeError, match="not bracketed"):
            lim._solve_m50_logistic_emcee(
                run_trials_at_mag=self._trials_fn(_outcomes),
                pool=None,
                m_guess=0.0,
                redo=8,
                completeness_target=0.5,
                lim_cfg=self._cfg(),
            )

    def test_clean_transition_succeeds(self, tmp_path):
        emcee = pytest.importorskip("emcee")
        lim = self._lim(tmp_path)
        rng = np.random.default_rng(1)
        m50_true, s_true = 0.0, 0.3

        def _outcomes(m, r):
            p = 1.0 / (1.0 + np.exp((m - m50_true) / s_true))
            return rng.random(r) < p

        m50, m50_err = lim._solve_m50_logistic_emcee(
            run_trials_at_mag=self._trials_fn(_outcomes),
            pool=None,
            m_guess=0.0,
            redo=10,
            completeness_target=0.5,
            lim_cfg=self._cfg(),
        )
        assert np.isfinite(m50) and np.isfinite(m50_err)
        assert m50 == pytest.approx(m50_true, abs=0.4)
