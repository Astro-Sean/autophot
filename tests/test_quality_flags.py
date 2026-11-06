"""Tests for the quality flag system (utils/quality_flags.py)."""

from __future__ import annotations

import numpy as np
import pytest

from utils.quality_flags import (
    QualityFlag,
    from_photutils_flags,
    from_snr,
    from_chi2,
    from_mcmc_diagnostics,
)


class TestQualityFlag:
    """Test the QualityFlag enum."""

    def test_ok_is_zero(self):
        assert QualityFlag.OK == 0

    def test_combinable(self):
        """Flags can be combined with |."""
        combined = QualityFlag.LOW_SN | QualityFlag.CROWDED
        assert QualityFlag.LOW_SN in combined
        assert QualityFlag.CROWDED in combined
        assert QualityFlag.SATURATED not in combined

    def test_is_ok(self):
        assert QualityFlag.OK.is_ok()
        assert not QualityFlag.LOW_SN.is_ok()

    def test_is_fatal(self):
        assert QualityFlag.SATURATED.is_fatal()
        assert QualityFlag.OPTIMIZER_NON_CONVERGENCE.is_fatal()
        assert QualityFlag.NEGATIVE_FLUX.is_fatal()
        assert not QualityFlag.LOW_SN.is_fatal()
        assert not QualityFlag.OK.is_fatal()

    def test_is_degraded(self):
        assert QualityFlag.LOW_SN.is_degraded()
        assert QualityFlag.CROWDED.is_degraded()
        assert not QualityFlag.SATURATED.is_degraded()  # fatal, not degraded
        assert not QualityFlag.OK.is_degraded()

    def test_combined_fatal_and_degraded(self):
        """Fatal + degraded = fatal (not degraded)."""
        combined = QualityFlag.SATURATED | QualityFlag.LOW_SN
        assert combined.is_fatal()
        assert not combined.is_degraded()

    def test_flag_names(self):
        """flag_names returns list of set flag names."""
        combined = QualityFlag.LOW_SN | QualityFlag.CROWDED
        names = combined.flag_names()
        assert "LOW_SN" in names
        assert "CROWDED" in names
        assert len(names) == 2

    def test_flag_names_empty(self):
        """OK flag has no names."""
        assert QualityFlag.OK.flag_names() == []

    def test_to_dict(self):
        """to_dict returns serializable representation."""
        qf = QualityFlag.LOW_SN | QualityFlag.CROWDED
        d = qf.to_dict()
        assert d["quality_flags"] == int(qf)
        assert d["quality_status"] == "degraded"
        assert "LOW_SN" in d["quality_flag_names"]
        assert "CROWDED" in d["quality_flag_names"]

    def test_to_dict_ok(self):
        d = QualityFlag.OK.to_dict()
        assert d["quality_flags"] == 0
        assert d["quality_status"] == "ok"
        assert d["quality_flag_names"] == []


class TestFromPhotutilsFlags:
    """Test mapping from photutils PSF flags."""

    def test_no_flags(self):
        assert from_photutils_flags(0) == QualityFlag.OK

    def test_non_convergence(self):
        """photutils 3.1 bit 3 (8) = no_convergence."""
        qf = from_photutils_flags(1 << 3)
        assert QualityFlag.OPTIMIZER_NON_CONVERGENCE in qf

    def test_at_bounds(self):
        """photutils 3.1 bit 5 (32) = parameter_near_bound."""
        qf = from_photutils_flags(1 << 5)
        assert QualityFlag.PARAMETER_AT_BOUND in qf

    def test_non_finite(self):
        """photutils 3.1 non-finite position/flux bits."""
        for bit in (1 << 9, 1 << 10):
            qf = from_photutils_flags(bit)
            assert QualityFlag.NON_FINITE_PIXELS in qf

    def test_non_finite_local_bkg(self):
        """photutils 3.1 bit 11 (2048) = non_finite_local_bkg."""
        qf = from_photutils_flags(1 << 11)
        assert QualityFlag.BAD_BACKGROUND in qf

    def test_negative_flux(self):
        """photutils 3.1 bit 2 (4) = non_positive_flux."""
        qf = from_photutils_flags(1 << 2)
        assert QualityFlag.NEGATIVE_FLUX in qf

    def test_fully_masked_too_few(self):
        """fully_masked (128) / too_few_pixels (256) -> mask fraction."""
        assert QualityFlag.HIGH_MASK_FRACTION in from_photutils_flags(128)
        assert QualityFlag.HIGH_MASK_FRACTION in from_photutils_flags(256)

    def test_outside_fit_region_crowded(self):
        """position_outside_fit_region (4096) usually means a neighbour."""
        assert QualityFlag.CROWDED in from_photutils_flags(4096)

    def test_combined(self):
        """Multiple bits map to multiple flags."""
        qf = from_photutils_flags((1 << 3) | (1 << 5))
        assert QualityFlag.OPTIMIZER_NON_CONVERGENCE in qf
        assert QualityFlag.PARAMETER_AT_BOUND in qf


class TestFromSNR:
    """Test S/N to quality flag mapping."""

    def test_high_snr(self):
        assert from_snr(50.0, detection_limit=3.0) == QualityFlag.OK

    def test_low_snr(self):
        assert QualityFlag.LOW_SN in from_snr(2.0, detection_limit=3.0)

    def test_at_threshold(self):
        """Exactly at threshold is OK (>= threshold)."""
        assert from_snr(3.0, detection_limit=3.0) == QualityFlag.OK


class TestFromChi2:
    """Test chi-squared to quality flag mapping."""

    def test_good_chi2(self):
        assert from_chi2(1.0) == QualityFlag.OK

    def test_bad_chi2(self):
        assert QualityFlag.SUSPICIOUS_RESIDUALS in from_chi2(10.0, threshold=5.0)

    def test_at_threshold(self):
        assert from_chi2(5.0, threshold=5.0) == QualityFlag.OK


class TestFromMCMCDiagnostics:
    """Test MCMC diagnostics to quality flag mapping."""

    def test_converged(self):
        qf = from_mcmc_diagnostics(rhat=1.01, n_eff=500, acc_frac=0.3, converged=True)
        assert qf == QualityFlag.OK

    def test_high_rhat(self):
        qf = from_mcmc_diagnostics(rhat=1.5)
        assert QualityFlag.MCMC_NON_CONVERGENCE in qf

    def test_low_ess(self):
        qf = from_mcmc_diagnostics(n_eff=50)
        assert QualityFlag.MCMC_LOW_ESS in qf

    def test_bad_acc_frac(self):
        qf = from_mcmc_diagnostics(acc_frac=0.05)
        assert QualityFlag.MCMC_NON_CONVERGENCE in qf

    def test_explicit_non_convergence(self):
        qf = from_mcmc_diagnostics(converged=False)
        assert QualityFlag.MCMC_NON_CONVERGENCE in qf

    def test_nan_rhat_ignored(self):
        qf = from_mcmc_diagnostics(rhat=np.nan)
        assert qf == QualityFlag.OK
