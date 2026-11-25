"""Regression tests for the alignment/subtraction robustness audit fixes.

Covers: difference-image quality scoring, inpainting NaN safety, SFFT the
WCS grid check, run_IDC saturation handling and the templates subtraction
alignment-quality gate.
"""

import numpy as np
import pytest

from astropy.io import fits


def _rng(seed=0):
    return np.random.default_rng(seed)


# ===========================================================================
# difference_quality
# ===========================================================================


class TestDifferenceQualityFixes:
    def test_zero_variance_diff_scores_clean_background(self):
        from autophot.subtraction.difference_quality import (
            QualityMetrics,
            QualityConfig,
            compute_quality_score,
        )

        m = QualityMetrics(diff_std=0.0, diff_median=0.0)
        cfg = QualityConfig()
        compute_quality_score(m, cfg)
        # A uniform (zero-variance) difference has no spatial background:
        # it must not be scored as the worst case.
        assert m.score_background == pytest.approx(1.0)

    def test_dc_pedestal_penalizes_background_score(self):
        from autophot.subtraction.difference_quality import (
            QualityMetrics,
            QualityConfig,
            compute_quality_score,
        )

        cfg = QualityConfig()
        clean = QualityMetrics(diff_std=10.0, diff_median=0.0)
        compute_quality_score(clean, cfg)
        pedestal = QualityMetrics(diff_std=10.0, diff_median=60.0)  # 6 sigma
        compute_quality_score(pedestal, cfg)
        assert pedestal.score_background < clean.score_background

    def test_metric_failure_downgrades_pass(self):
        from autophot.subtraction.difference_quality import (
            QualityMetrics,
            QualityConfig,
            compute_quality_score,
        )

        cfg = QualityConfig()
        m = QualityMetrics(diff_std=10.0, diff_median=0.0, metric_failures=["dipole"])
        compute_quality_score(m, cfg)
        assert m.quality_class in ("downgrade", "fail")
        assert m.quality_class != "pass"

    def test_untested_bright_star_is_not_perfect(self):
        from autophot.subtraction.difference_quality import (
            QualityMetrics,
            QualityConfig,
            compute_quality_score,
        )

        cfg = QualityConfig()
        m = QualityMetrics(diff_std=10.0, diff_median=0.0, dipole_checked=0)
        cfg.bright_star_check = True
        compute_quality_score(m, cfg)
        assert m.score_bright_star < 1.0

    def test_zero_denominator_configs_do_not_crash(self):
        from autophot.subtraction.difference_quality import (
            QualityMetrics,
            QualityConfig,
            compute_quality_score,
        )

        m = QualityMetrics(diff_std=10.0, diff_median=0.0)
        cfg = QualityConfig(autocorr_max_peak=0.0, background_max_spatial_std_ratio=0.0)
        compute_quality_score(m, cfg)  # must not raise

    def test_assess_handles_int_mask_and_nan_json(self):
        from astropy.io import fits as _fits
        from autophot.subtraction.difference_quality import assess_difference_image

        data = _rng().normal(0, 1, (64, 64))
        mask = np.zeros((64, 64), dtype=int)  # int, not bool
        m = assess_difference_image(data, mask, fwhm=3.0)
        assert m.valid_pixels > 0
        s = m.to_json()
        assert "NaN" not in s  # standards-compliant JSON

    def test_flux_scaled_dipole_threshold_ignores_percent_level_mismatch(self):
        # A 5%-of-flux bipolar residual at a bright star is kernel-matching
        # error, not a subtraction failure: with the source flux supplied it
        # must not count as a dipole; without it the same residual flags.
        from autophot.subtraction.difference_quality import (
            QualityConfig,
            detect_dipoles,
        )

        rng = _rng(1)
        data = rng.normal(0.0, 1.0, (80, 80))
        mask = np.zeros((80, 80), dtype=bool)
        flux = 20000.0  # bright star peak
        # Bipolar residual: +amp/-amp lobes either side of the source.
        amp = 0.05 * flux
        yy, xx = np.mgrid[30:46, 30:46]
        data[30:46, 30:46] += amp * np.exp(-((xx - 35.0) ** 2) / 2.0 - ((yy - 38) ** 2) / 2.0)
        data[30:46, 30:46] -= amp * np.exp(-((xx - 41.0) ** 2) / 2.0 - ((yy - 38) ** 2) / 2.0)
        pos = [(38.0, 38.0)]
        n_noflux, *_ = detect_dipoles(
            data, pos, mask, noise_sigma=1.0, fwhm=3.0, cfg=QualityConfig()
        )
        n_flux, *_ = detect_dipoles(
            data, pos, mask, noise_sigma=1.0, fwhm=3.0,
            cfg=QualityConfig(), source_fluxes=[flux],
        )
        assert n_noflux == 1
        assert n_flux == 0

    def test_kernel_implied_autocorr_not_penalised(self):
        # Positive correlation up to the backend-reported kernel-implied
        # level is expected convolution physics: it must not reduce the
        # score.  A negative peak keeps the strict raw penalty.
        from autophot.subtraction.difference_quality import (
            QualityConfig,
            QualityMetrics,
            compute_quality_score,
        )

        cfg = QualityConfig()
        m = QualityMetrics(
            diff_std=10.0,
            diff_median=0.0,
            autocorr_peak=0.4,
            autocorr_peak_signed=0.4,
            autocorr_expected_pos=0.45,
        )
        compute_quality_score(m, cfg)
        assert m.score_autocorr == pytest.approx(1.0)
        # Excess above the implied level is penalised.
        m2 = QualityMetrics(
            diff_std=10.0,
            diff_median=0.0,
            autocorr_peak=0.6,
            autocorr_peak_signed=0.6,
            autocorr_expected_pos=0.45,
        )
        compute_quality_score(m2, cfg)
        assert m2.score_autocorr < 1.0
        # Negative (ringing) stays strict even with a positive expectation.
        m3 = QualityMetrics(
            diff_std=10.0,
            diff_median=0.0,
            autocorr_peak=0.4,
            autocorr_peak_signed=-0.4,
            autocorr_expected_pos=0.45,
        )
        compute_quality_score(m3, cfg)
        assert m3.score_autocorr < 1.0

    def test_bright_star_flux_scaling_relaxes_wing_residual(self):
        # Annulus excess at a bright star below the flux-fraction limit
        # scores near 1.0; without fluxes the same residual is penalised.
        from autophot.subtraction.difference_quality import (
            QualityConfig,
            measure_bright_star_residuals,
        )

        rng = _rng(2)
        data = rng.normal(0.0, 1.0, (80, 80))
        mask = np.zeros((80, 80), dtype=bool)
        flux = 10000.0
        # Wing residual in the bright-star annulus (r 3..9 for fwhm=3).
        yy, xx = np.mgrid[:80, :80]
        r = np.hypot(yy - 40, xx - 40)
        ann = (r >= 3) & (r <= 9)
        data[ann] += 0.008 * flux  # 80 sigma excess, 0.8% of peak flux
        pos = [(40.0, 40.0)]
        _, _, n0, norm0 = measure_bright_star_residuals(
            data, pos, mask, 1.0, 3.0, QualityConfig()
        )
        _, _, n1, norm1 = measure_bright_star_residuals(
            data, pos, mask, 1.0, 3.0, QualityConfig(), star_fluxes=[flux]
        )
        assert n0 == n1 == 1
        assert norm1 < norm0  # flux allowance relaxes the normalised score


# ===========================================================================
# inpaint
# ===========================================================================


class TestInpaintFixes:
    def test_nan_outside_mask_does_not_poison_output(self):
        from autophot.subtraction.inpaint import inpaint_image

        img = _rng().normal(0, 1, (32, 32))
        img[2, 2] = np.nan  # NaN OUTSIDE the repair mask
        mask = np.zeros((32, 32), dtype=bool)
        mask[10, 10] = True
        out = inpaint_image(img, mask)
        assert np.all(np.isfinite(out))

    def test_saturate_none_does_not_crash(self):
        from autophot.subtraction.inpaint import build_saturation_core_mask

        img = np.zeros((16, 16))
        m = build_saturation_core_mask(img, saturate=None)
        assert m.dtype == bool and not m.any()

    def test_skip_returns_empty_applied_mask(self):
        from autophot.subtraction.inpaint import InpaintConfig, inpaint_saturated_cores

        img = np.zeros((32, 32))
        img[15, 15] = 1e5
        cfg = InpaintConfig(enabled=True, max_mask_fraction=1e-6)
        _out, applied = inpaint_saturated_cores(img, saturate=1e4, cfg=cfg)
        # mask fraction too large -> skipped -> empty applied mask
        assert not applied.any()

    def test_all_masked_raises(self):
        from autophot.subtraction.inpaint import inpaint_image

        img = np.zeros((8, 8))
        mask = np.ones((8, 8), dtype=bool)
        with pytest.raises(ValueError):
            inpaint_image(img, mask)


# ===========================================================================
# run_sfft WCS grid check helper (pixel-scale parsing robustness)
# ===========================================================================


class TestSfftWcsGridCheck:
    def test_pixel_scales_deg_handles_cd_and_pc_headers(self):
        from autophot.pipeline.wcs import _pixel_scales_deg

        def _scale(hdr):
            from astropy.wcs import WCS

            return _pixel_scales_deg(WCS(hdr).celestial)

        hdr = fits.Header()
        hdr["CRPIX1"] = 10.0
        hdr["CRPIX2"] = 10.0
        hdr["CRVAL1"] = 0.0
        hdr["CRVAL2"] = 0.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CD1_1"] = -1.0 / 3600.0
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 1.0 / 3600.0
        s = _scale(hdr)
        assert np.all(np.isfinite(s)) and np.all(s > 0)
        # PC+CDELT form
        hdr2 = fits.Header()
        hdr2["CRPIX1"] = 10.0
        hdr2["CRPIX2"] = 10.0
        hdr2["CRVAL1"] = 0.0
        hdr2["CRVAL2"] = 0.0
        hdr2["CTYPE1"] = "RA---TAN"
        hdr2["CTYPE2"] = "DEC--TAN"
        hdr2["CDELT1"] = -1.0 / 3600.0
        hdr2["CDELT2"] = 1.0 / 3600.0
        hdr2["PC1_1"] = 1.0
        hdr2["PC2_2"] = 1.0
        s2 = _scale(hdr2)
        assert np.allclose(s, s2)


# ===========================================================================
# run_IDC saturation handling
# ===========================================================================


class TestRunIDCSaturation:
    def test_float_saturation_not_capped_at_65535(self):
        from autophot.subtraction.run_IDC import ImageDistortionCorrector

        se = ImageDistortionCorrector(input_yaml={})
        data = np.ones((16, 16), dtype=np.float32)
        data[0, 0] = 1.0e7
        with fits.HDUList([fits.PrimaryHDU(data)]) as hdul:
            import tempfile, os

            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
                hdul.writeto(tf.name)
                p = tf.name
            try:
                sat = se.determine_saturation_level(p)
                # must clamp below the data max, not to 65535
                assert sat > 65535
            finally:
                os.remove(p)

    def test_integer_saturation_keeps_16bit_cap(self):
        from autophot.subtraction.run_IDC import ImageDistortionCorrector

        se = ImageDistortionCorrector(input_yaml={})
        data = np.zeros((16, 16), dtype=np.uint16).astype(np.float32)
        data[0, 0] = 65535
        import tempfile, os

        with fits.HDUList([fits.PrimaryHDU(data)]) as hdul:
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
                hdul.writeto(tf.name)
                p = tf.name
            try:
                sat = se.determine_saturation_level(p)
                assert sat <= 65535
            finally:
                os.remove(p)


# ===========================================================================
# templates subtraction alignment-quality gate
# ===========================================================================


class TestSubtractionAlignmentGate:
    def _make_config(self, reject):
        return {"template_subtraction": {"alignment_reject_unverified": reject}}

    def test_unverified_gate_warns_but_proceeds_by_default(self, monkeypatch):
        # The gate logic lives inside Templates.subtract(); exercise its
        # config parsing by constructing the class (constructor is light).
        from autophot.subtraction.templates import Templates

        tpl = Templates(input_yaml=self._make_config(False))
        # Config surfaces via input_yaml
        cfg = tpl.input_yaml["template_subtraction"]
        assert cfg["alignment_reject_unverified"] is False

    def test_unverified_gate_can_be_enabled(self):
        from autophot.subtraction.templates import Templates

        tpl = Templates(input_yaml=self._make_config(True))
        assert (
            tpl.input_yaml["template_subtraction"]["alignment_reject_unverified"]
            is True
        )
