"""Tests for the uncertainty calibration module (utils/uncertainty_calibration.py)."""

from __future__ import annotations

import numpy as np
import pytest

from utils.uncertainty_calibration import (
    calibrate_uncertainties,
    apply_calibration,
    UncertaintyCalibrationResult,
)


class TestCalibrateUncertainties:
    """Test the uncertainty calibration analysis."""

    def test_well_calibrated(self):
        """Well-calibrated errors: z ~ N(0, 1)."""
        rng = np.random.default_rng(42)
        n = 500
        flux_true = rng.uniform(1000, 10000, n)
        # True errors match reported errors
        flux_err = np.sqrt(flux_true)  # Poisson-like
        z = rng.normal(0, 1, n)
        flux_meas = flux_true + z * flux_err

        result = calibrate_uncertainties(flux_true, flux_meas, flux_err)

        assert result.n_sources == n
        assert abs(result.z_mean) < 0.2
        assert abs(result.z_std - 1.0) < 0.15
        assert abs(result.z_mad_std - 1.0) < 0.15
        assert abs(result.calibration_factor - 1.0) < 0.15
        assert result.coverage_1sigma_ok
        assert result.coverage_2sigma_ok
        assert result.passed

    def test_underestimated_errors(self):
        """Underestimated errors: z_std > 1."""
        rng = np.random.default_rng(42)
        n = 500
        flux_true = rng.uniform(1000, 10000, n)
        # True scatter is 2x the reported error
        true_err = np.sqrt(flux_true) * 2
        reported_err = np.sqrt(flux_true)  # Half the true error
        z = rng.normal(0, 1, n)
        flux_meas = flux_true + z * true_err

        result = calibrate_uncertainties(flux_true, flux_meas, reported_err)

        assert result.z_std > 1.5
        assert result.z_mad_std > 1.5
        assert result.errors_underestimated
        assert result.calibration_factor > 1.5
        assert not result.passed

    def test_overestimated_errors(self):
        """Overestimated errors: z_std < 1."""
        rng = np.random.default_rng(42)
        n = 500
        flux_true = rng.uniform(1000, 10000, n)
        # True scatter is 0.5x the reported error
        true_err = np.sqrt(flux_true) * 0.5
        reported_err = np.sqrt(flux_true)
        z = rng.normal(0, 1, n)
        flux_meas = flux_true + z * true_err

        result = calibrate_uncertainties(flux_true, flux_meas, reported_err)

        assert result.z_std < 0.8
        assert result.z_mad_std < 0.8
        assert result.errors_overestimated
        assert result.calibration_factor < 0.8

    def test_biased_measurements(self):
        """Biased measurements: z_mean != 0."""
        rng = np.random.default_rng(42)
        n = 500
        flux_true = rng.uniform(1000, 10000, n)
        flux_err = np.sqrt(flux_true)
        # 10% systematic bias
        z = rng.normal(0, 1, n)
        flux_meas = flux_true * 1.10 + z * flux_err

        result = calibrate_uncertainties(flux_true, flux_meas, flux_err)

        assert result.z_mean > 2.0
        assert abs(result.bias_frac - 0.10) < 0.02

    def test_too_few_sources(self):
        """Too few sources should return failure."""
        result = calibrate_uncertainties(
            np.array([100, 200, 300]),
            np.array([110, 190, 310]),
            np.array([10, 10, 10]),
            min_sources=10,
        )
        assert result.n_sources < 10
        assert not result.passed

    def test_non_finite_filtered(self):
        """Non-finite values should be filtered."""
        flux_true = np.array([1000, 2000, np.nan, 4000, 5000] * 20)
        flux_meas = np.array([1010, 1990, 3000, 4010, np.inf] * 20)
        flux_err = np.array([10, 10, 10, 10, 0] * 20)

        result = calibrate_uncertainties(flux_true, flux_meas, flux_err, min_sources=10)
        assert result.n_sources < 100
        assert result.n_sources > 0

    def test_to_dict(self):
        """to_dict should produce serializable output."""
        result = UncertaintyCalibrationResult(
            n_sources=100, z_mean=0.01, z_std=1.05, passed=True,
        )
        d = result.to_dict()
        assert d["uncal_n_sources"] == 100
        assert d["uncal_passed"] is True

    def test_summary(self):
        """summary should produce readable text."""
        result = UncertaintyCalibrationResult(
            n_sources=100, z_mean=0.01, z_std=1.05, passed=True,
            fraction_within_1sigma=0.68, fraction_within_2sigma=0.95,
        )
        text = result.summary()
        assert "PASS" in text
        assert "100 sources" in text


class TestApplyCalibration:
    """Test the apply_calibration function."""

    def test_basic_application(self):
        """Calibration factor multiplies errors."""
        errors = np.array([1.0, 2.0, 3.0])
        calibrated = apply_calibration(errors, 1.5)
        np.testing.assert_allclose(calibrated, [1.5, 3.0, 4.5])

    def test_scalar_input(self):
        """Scalar input works."""
        result = apply_calibration(10.0, 1.2)
        assert result == pytest.approx(12.0)

    def test_invalid_factor(self):
        """Invalid calibration factor returns original errors."""
        errors = np.array([1.0, 2.0, 3.0])
        calibrated = apply_calibration(errors, np.nan)
        np.testing.assert_array_equal(calibrated, errors)

    def test_zero_factor(self):
        """Zero calibration factor returns original errors."""
        errors = np.array([1.0, 2.0, 3.0])
        calibrated = apply_calibration(errors, 0.0)
        np.testing.assert_array_equal(calibrated, errors)

    def test_negative_factor(self):
        """Negative calibration factor returns original errors."""
        errors = np.array([1.0, 2.0, 3.0])
        calibrated = apply_calibration(errors, -1.0)
        np.testing.assert_array_equal(calibrated, errors)


class TestUncertaintyCalibrationIntegration:
    """Integration test: calibrate then apply."""

    def test_calibrate_and_apply(self):
        """Calibration factor should produce unit scatter when applied."""
        rng = np.random.default_rng(42)
        n = 1000
        flux_true = rng.uniform(1000, 10000, n)
        # True errors are 1.5x reported
        true_err = np.sqrt(flux_true) * 1.5
        reported_err = np.sqrt(flux_true)
        z = rng.normal(0, 1, n)
        flux_meas = flux_true + z * true_err

        result = calibrate_uncertainties(flux_true, flux_meas, reported_err)
        assert result.calibration_factor > 1.2

        calibrated_err = apply_calibration(reported_err, result.calibration_factor)

        # Residuals with calibrated errors should have unit scatter.
        z_calibrated = (flux_meas - flux_true) / calibrated_err
        z_cal_std = np.std(z_calibrated)

        assert abs(z_cal_std - 1.0) < 0.15


class TestScaleErrorColumns:
    """Test scale_error_columns: applying the factor to a photometry row."""

    def _row(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "flux_PSF_err": [10.0],
                "flux_AP_err": [12.0],
                "flux_PSF_err_normal": [9.0],
                "flux_PSF_err_inverted": [11.0],
                "flux_AP_err_inverted": [13.0],
                "inst_r_PSF_err": [0.10],
                "inst_r_AP_err": [0.12],
                "inst_r_PSF_normal_err": [0.09],
                "inst_inverted_err": [0.11],
                # calibrated mag err = sqrt(inst^2 + zp_err^2) with zp_err=0.03
                "r_PSF_err": [np.sqrt(0.10**2 + 0.03**2)],
                "r_AP_err": [np.sqrt(0.12**2 + 0.03**2)],
                "SNR": [50.0],
                "SNR_AP_inverted": [25.0],
            }
        )

    def test_psf_scales_psf_columns_only(self):
        from utils.uncertainty_calibration import scale_error_columns

        df = self._row()
        scaled = scale_error_columns(df, 1.5, "PSF", "r")

        assert df["flux_PSF_err"].iloc[0] == pytest.approx(15.0)
        assert df["flux_PSF_err_normal"].iloc[0] == pytest.approx(13.5)
        assert df["inst_r_PSF_err"].iloc[0] == pytest.approx(0.15)
        assert df["inst_inverted_err"].iloc[0] == pytest.approx(0.165)
        # AP-family columns untouched
        assert df["flux_AP_err"].iloc[0] == pytest.approx(12.0)
        assert df["inst_r_AP_err"].iloc[0] == pytest.approx(0.12)
        assert "flux_AP_err" not in scaled
        # Calibrated PSF mag err: quadrature delta on the inst term only
        expected = np.sqrt(0.10**2 * 1.5**2 + 0.03**2)
        assert df["r_PSF_err"].iloc[0] == pytest.approx(expected)
        assert df["r_AP_err"].iloc[0] == pytest.approx(
            np.sqrt(0.12**2 + 0.03**2)
        )

    def test_ap_scales_ap_columns_and_snr(self):
        from utils.uncertainty_calibration import scale_error_columns

        df = self._row()
        scale_error_columns(df, 2.0, "AP", "r")

        assert df["flux_AP_err"].iloc[0] == pytest.approx(24.0)
        assert df["flux_AP_err_inverted"].iloc[0] == pytest.approx(26.0)
        assert df["inst_r_AP_err"].iloc[0] == pytest.approx(0.24)
        assert df["flux_PSF_err"].iloc[0] == pytest.approx(10.0)
        # Aperture SNR shrinks as errors grow
        assert df["SNR"].iloc[0] == pytest.approx(25.0)
        assert df["SNR_AP_inverted"].iloc[0] == pytest.approx(12.5)
        expected = np.sqrt(0.12**2 * 4.0 + 0.03**2)
        assert df["r_AP_err"].iloc[0] == pytest.approx(expected)

    def test_factor_one_is_noop(self):
        from utils.uncertainty_calibration import scale_error_columns

        df = self._row()
        scaled = scale_error_columns(df, 1.0, "PSF", "r")
        assert scaled == []
        assert df["flux_PSF_err"].iloc[0] == pytest.approx(10.0)

    def test_invalid_factor_is_noop(self):
        from utils.uncertainty_calibration import scale_error_columns

        df = self._row()
        assert scale_error_columns(df, np.nan, "PSF", "r") == []
        assert scale_error_columns(df, -1.0, "PSF", "r") == []
        assert df["flux_PSF_err"].iloc[0] == pytest.approx(10.0)

    def test_missing_columns_tolerated(self):
        import pandas as pd

        from utils.uncertainty_calibration import scale_error_columns

        df = pd.DataFrame({"flux_PSF_err": [5.0]})
        scaled = scale_error_columns(df, 1.5, "PSF", "r")
        assert scaled == ["flux_PSF_err"]
        assert df["flux_PSF_err"].iloc[0] == pytest.approx(7.5)
