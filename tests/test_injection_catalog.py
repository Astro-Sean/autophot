"""Tests for the injection catalog module (utils/injection_catalog.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from autophot.analysis.injection_catalog import InjectionCatalog, InjectionRecord


class TestInjectionCatalog:
    """Test the InjectionCatalog class."""

    def test_empty_catalog(self):
        """Empty catalog should have zero length."""
        cat = InjectionCatalog(seed=42, recovery_method="AP")
        assert len(cat) == 0
        df = cat.to_dataframe()
        assert df.empty
        stats = cat.summary_statistics()
        assert stats["n_total"] == 0

    def test_add_record(self):
        """Adding a record should increase length."""
        cat = InjectionCatalog(seed=42, recovery_method="PSF")
        cat.add_record(
            magnitude=15.0, x_inj=100, y_inj=100,
            flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
            detected=True, detection_snr=20.0,
        )
        assert len(cat) == 1
        assert cat.records[0].trial_id == 0
        assert cat.records[0].flux_true == 1000.0

    def test_multiple_records(self):
        """Multiple records should get sequential trial IDs."""
        cat = InjectionCatalog(seed=42)
        for i in range(10):
            cat.add_record(
                magnitude=15.0 + i, x_inj=100, y_inj=100,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True,
            )
        assert len(cat) == 10
        assert cat.records[0].trial_id == 0
        assert cat.records[9].trial_id == 9

    def test_to_dataframe(self):
        """to_dataframe should produce correct columns."""
        cat = InjectionCatalog(seed=42, recovery_method="AP")
        cat.add_record(
            magnitude=15.0, x_inj=100, y_inj=100,
            flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
            detected=True, detection_snr=20.0,
        )
        df = cat.to_dataframe()
        assert len(df) == 1
        assert "trial_id" in df.columns
        assert "flux_true" in df.columns
        assert "flux_recovered" in df.columns
        assert "detected" in df.columns
        assert "quality_flags" in df.columns
        # pandas may return numpy.bool_, which fails `is True`
        assert df["detected"][0] is True or df["detected"][0] == True

    def test_save_csv(self, tmp_path):
        """save_csv should write a readable CSV file."""
        cat = InjectionCatalog(seed=42, recovery_method="PSF")
        for i in range(5):
            cat.add_record(
                magnitude=15.0 + i, x_inj=100 + i, y_inj=200 + i,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )
        path = tmp_path / "injection_catalog.csv"
        cat.save_csv(path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 5
        assert "flux_true" in df.columns

    def test_summary_statistics(self):
        """summary_statistics should compute correct values."""
        cat = InjectionCatalog(seed=42)
        # 10 detected, 5 not detected
        for i in range(10):
            cat.add_record(
                magnitude=15.0, x_inj=100, y_inj=100,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )
        for i in range(5):
            cat.add_record(
                magnitude=18.0, x_inj=100, y_inj=100,
                flux_true=100.0, flux_recovered=0.0, flux_err=50.0,
                detected=False, detection_snr=0.0,
            )
        stats = cat.summary_statistics()
        assert stats["n_total"] == 15
        assert stats["n_detected"] == 10
        assert stats["detection_rate"] == pytest.approx(10 / 15)
        assert abs(stats["flux_bias_frac"] - 0.01) < 0.01  # ~1% bias

    def test_completeness_curve(self):
        """completeness_curve should produce correct bins."""
        cat = InjectionCatalog(seed=42)
        # Bright sources: all detected
        for i in range(20):
            cat.add_record(
                magnitude=14.0, x_inj=100, y_inj=100,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )
        # Faint sources: half detected
        for i in range(20):
            cat.add_record(
                magnitude=18.0, x_inj=100, y_inj=100,
                flux_true=100.0, flux_recovered=101.0 if i < 10 else 0.0,
                flux_err=50.0,
                detected=(i < 10), detection_snr=2.0 if i < 10 else 0.0,
            )
        curve = cat.completeness_curve(n_bins=10)
        assert len(curve) == 10
        assert "completeness" in curve.columns
        # Bright bin should have ~100% completeness
        bright = curve[curve["mag_center"] < 15]
        bright_valid = bright["completeness"].dropna()
        assert all(bright_valid > 0.9)
        # Faint bin should have ~50% completeness
        faint = curve[curve["mag_center"] > 17]
        faint_valid = faint["completeness"].dropna()
        assert all(faint_valid < 0.6)

    def test_uncertainty_calibration_data(self):
        """uncertainty_calibration_data should return valid arrays."""
        cat = InjectionCatalog(seed=42)
        for i in range(20):
            cat.add_record(
                magnitude=15.0, x_inj=100, y_inj=100,
                flux_true=1000.0, flux_recovered=1010.0 + i, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )
        ft, fm, fe = cat.uncertainty_calibration_data()
        assert len(ft) == 20
        assert len(fm) == 20
        assert len(fe) == 20
        np.testing.assert_allclose(ft, 1000.0)

    def test_uncertainty_calibration_data_filters(self):
        """uncertainty_calibration_data should filter non-finite and undetected."""
        cat = InjectionCatalog(seed=42)
        cat.add_record(magnitude=15.0, x_inj=100, y_inj=100,
                       flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                       detected=True)
        cat.add_record(magnitude=15.0, x_inj=100, y_inj=100,
                       flux_true=np.nan, flux_recovered=1010.0, flux_err=50.0,
                       detected=True)
        cat.add_record(magnitude=15.0, x_inj=100, y_inj=100,
                       flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                       detected=False)
        cat.add_record(magnitude=15.0, x_inj=100, y_inj=100,
                       flux_true=1000.0, flux_recovered=1010.0, flux_err=0.0,
                       detected=True)
        ft, fm, fe = cat.uncertainty_calibration_data()
        assert len(ft) == 1  # Only the first record is fully valid

    def test_seed_recorded(self):
        """Seed should be recorded in each record."""
        cat = InjectionCatalog(seed=12345)
        cat.add_record(
            magnitude=15.0, x_inj=100, y_inj=100,
            flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
            detected=True,
        )
        assert cat.records[0].seed == 12345

    def test_quality_flags_recorded(self):
        """Quality flags should be recorded."""
        cat = InjectionCatalog(seed=42)
        cat.add_record(
            magnitude=15.0, x_inj=100, y_inj=100,
            flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
            detected=True, quality_flags=3,
        )
        assert cat.records[0].quality_flags == 3

    def test_repr(self):
        """repr should show trial count and method."""
        cat = InjectionCatalog(seed=42, recovery_method="PSF")
        cat.add_record(
            magnitude=15.0, x_inj=100, y_inj=100,
            flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
            detected=True,
        )
        text = repr(cat)
        assert "n_trials=1" in text
        assert "PSF" in text
