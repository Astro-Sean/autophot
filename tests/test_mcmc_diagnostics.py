"""Tests for MCMC convergence diagnostics (utils/mcmc_diagnostics.py)."""

from __future__ import annotations

import numpy as np
import pytest

from utils.mcmc_diagnostics import (
    compute_split_rhat,
    compute_rhat_all_params,
    compute_ess,
    compute_ess_all_params,
    assess_convergence,
    MCMCConvergenceReport,
    MCMC_FLAG_OK,
    MCMC_FLAG_HIGH_RHAT,
    MCMC_FLAG_LOW_ESS,
    MCMC_FLAG_BAD_ACCEPTANCE,
    MCMC_FLAG_TOO_FEW_STEPS,
)


class TestSplitRhat:
    """Test the split-chain R-hat computation."""

    def test_converged_chain(self):
        """A well-mixed chain should have R-hat ~ 1."""
        rng = np.random.default_rng(42)
        # 32 walkers, 2000 steps, all sampling from same distribution
        chain = rng.normal(0, 1, (32, 2000))
        rhat = compute_split_rhat(chain)
        assert rhat == pytest.approx(1.0, abs=0.1)

    def test_non_converged_chain(self):
        """Chains stuck in different modes should have R-hat > 1.1."""
        rng = np.random.default_rng(42)
        n_walkers = 32
        n_steps = 2000
        chain = np.zeros((n_walkers, n_steps))
        # First half of walkers sample from N(0, 1), second from N(5, 1)
        chain[:n_walkers // 2] = rng.normal(0, 1, (n_walkers // 2, n_steps))
        chain[n_walkers // 2:] = rng.normal(5, 1, (n_walkers // 2, n_steps))
        rhat = compute_split_rhat(chain)
        assert rhat > 1.5  # Clearly non-converged

    def test_too_short_chain(self):
        """Chains shorter than 8 steps should return NaN."""
        chain = np.random.normal(0, 1, (10, 4))
        rhat = compute_split_rhat(chain)
        assert np.isnan(rhat)

    def test_constant_chain(self):
        """A constant chain (zero variance) should return NaN."""
        chain = np.full((10, 100), 5.0)
        rhat = compute_split_rhat(chain)
        assert np.isnan(rhat)

    def test_2d_required(self):
        """Wrong dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            compute_split_rhat(np.random.normal(0, 1, 100))


class TestRhatAllParams:
    """Test multi-parameter R-hat computation."""

    def test_multi_param(self):
        """3-parameter chain should return 3 R-hat values."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (32, 2000, 3))
        rhats = compute_rhat_all_params(chain)
        assert len(rhats) == 3
        for r in rhats:
            assert r == pytest.approx(1.0, abs=0.1)

    def test_one_param_non_converged(self):
        """If one parameter is non-converged, its R-hat should be high."""
        rng = np.random.default_rng(42)
        n_walkers, n_steps = 32, 2000
        chain = np.zeros((n_walkers, n_steps, 2))
        chain[:, :, 0] = rng.normal(0, 1, (n_walkers, n_steps))
        # Make param 1 non-converged
        chain[:n_walkers // 2, :, 1] = rng.normal(0, 1, (n_walkers // 2, n_steps))
        chain[n_walkers // 2:, :, 1] = rng.normal(10, 1, (n_walkers // 2, n_steps))
        rhats = compute_rhat_all_params(chain)
        assert rhats[0] < 1.1
        assert rhats[1] > 1.5


class TestESS:
    """Test effective sample size computation."""

    def test_independent_samples(self):
        """Independent samples should have n_eff ~ n_total."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (32, 1000))
        n_eff = compute_ess(chain)
        # For independent samples, n_eff should be close to n_walkers * n_steps
        assert n_eff > 10000  # Should be a large fraction of 32000

    def test_correlated_samples(self):
        """Highly correlated samples should have low n_eff."""
        rng = np.random.default_rng(42)
        n_walkers, n_steps = 32, 1000
        chain = np.zeros((n_walkers, n_steps))
        for w in range(n_walkers):
            chain[w, 0] = rng.normal(0, 1)
            for t in range(1, n_steps):
                chain[w, t] = 0.99 * chain[w, t - 1] + 0.01 * rng.normal(0, 1)
        n_eff = compute_ess(chain)
        # Highly correlated: n_eff should be much less than n_total
        assert n_eff < 1000

    def test_constant_chain(self):
        """Constant chain should return n_total (no variance)."""
        chain = np.full((10, 100), 5.0)
        n_eff = compute_ess(chain)
        assert n_eff == 1000

    def test_short_chain(self):
        """Short chain should return n_total."""
        chain = np.random.normal(0, 1, (10, 2))
        n_eff = compute_ess(chain)
        assert n_eff == 20


class TestAssessConvergence:
    """Test the full convergence assessment."""

    def test_converged_run(self):
        """A well-mixed chain with good acceptance should converge."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (32, 2000, 3))
        report = assess_convergence(chain, acceptance_fraction=0.3)
        assert report.converged
        assert report.rhat < 1.1
        assert report.n_eff > 100
        assert report.flags == MCMC_FLAG_OK

    def test_non_converged_rhat(self):
        """High R-hat should flag non-convergence."""
        rng = np.random.default_rng(42)
        n_walkers, n_steps = 32, 2000
        chain = np.zeros((n_walkers, n_steps, 1))
        chain[:n_walkers // 2] = rng.normal(0, 1, (n_walkers // 2, n_steps, 1))
        chain[n_walkers // 2:] = rng.normal(10, 1, (n_walkers // 2, n_steps, 1))
        report = assess_convergence(chain, acceptance_fraction=0.3)
        assert not report.converged
        assert report.flags & MCMC_FLAG_HIGH_RHAT

    def test_low_ess(self):
        """Low ESS should flag non-convergence."""
        rng = np.random.default_rng(42)
        n_walkers, n_steps = 32, 1000
        chain = np.zeros((n_walkers, n_steps, 1))
        for w in range(n_walkers):
            chain[w, 0] = rng.normal(0, 1)
            for t in range(1, n_steps):
                chain[w, t] = 0.999 * chain[w, t - 1] + 0.001 * rng.normal(0, 1)
        report = assess_convergence(chain, acceptance_fraction=0.3)
        assert report.flags & MCMC_FLAG_LOW_ESS

    def test_bad_acceptance(self):
        """Bad acceptance fraction should flag."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (32, 2000, 1))
        report = assess_convergence(chain, acceptance_fraction=0.01)
        assert report.flags & MCMC_FLAG_BAD_ACCEPTANCE
        assert not report.converged

    def test_too_few_steps(self):
        """Too few steps should flag."""
        chain = np.random.normal(0, 1, (32, 50, 1))
        report = assess_convergence(chain, acceptance_fraction=0.3, min_steps=100)
        assert report.flags & MCMC_FLAG_TOO_FEW_STEPS

    def test_to_dict(self):
        """to_dict should produce serializable output."""
        report = MCMCConvergenceReport(
            rhat=1.02, n_eff=500, acceptance_fraction=0.3,
            total_steps=2000, n_walkers=32, converged=True,
        )
        d = report.to_dict()
        assert d["mcmc_rhat"] == 1.02
        assert d["mcmc_n_eff"] == 500
        assert d["mcmc_converged"] is True

    def test_summary(self):
        """summary should produce readable text."""
        report = MCMCConvergenceReport(
            rhat=1.02, n_eff=500, acceptance_fraction=0.3,
            total_steps=2000, n_walkers=32, converged=True,
        )
        text = report.summary()
        assert "CONVERGED" in text
        assert "1.020" in text

    def test_summary_non_converged(self):
        """summary for non-converged should show flags."""
        report = MCMCConvergenceReport(
            rhat=1.5, n_eff=50, acceptance_fraction=0.05,
            total_steps=500, n_walkers=32, converged=False,
            flags=MCMC_FLAG_HIGH_RHAT | MCMC_FLAG_LOW_ESS | MCMC_FLAG_BAD_ACCEPTANCE,
        )
        text = report.summary()
        assert "NOT CONVERGED" in text
        assert "HIGH_RHAT" in text
        assert "LOW_ESS" in text
        assert "BAD_ACCEPTANCE" in text

    def test_2d_chain_handled(self):
        """2-D chain (single param) should be handled."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (32, 2000))
        report = assess_convergence(chain, acceptance_fraction=0.3)
        assert report.n_params == 1
