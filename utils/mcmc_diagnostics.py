"""MCMC convergence diagnostics for AutoPhOT.

Provides:
- Gelman-Rubin R-hat from split chains (detects non-convergence)
- Effective sample size (n_eff) computation
- Acceptance fraction assessment
- Structured convergence report with quality flags

These diagnostics complement emcee's built-in autocorrelation time
estimation.  R-hat is particularly important because it detects
non-convergence that autocorrelation alone can miss (e.g., when chains
are stuck in different modes).

Scientific motivation:  MCMC posteriors are only valid if the chains
have converged.  Reporting a flux and uncertainty from non-converged
chains manufactures false precision.  The R-hat diagnostic catches
this by splitting each chain in half and checking that the two halves
agree.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MCMCConvergenceReport:
    """Convergence diagnostics for an MCMC run.

    Attributes
    ----------
    rhat : float
        Gelman-Rubin R-hat statistic (split-chain).  < 1.1 = converged.
    rhat_per_param : np.ndarray
        Per-parameter R-hat values.
    n_eff : float
        Effective sample size (minimum across parameters).
    n_eff_per_param : np.ndarray
        Per-parameter effective sample sizes.
    acceptance_fraction : float
        Mean acceptance fraction.  Should be in [0.15, 0.8] for emcee.
    total_steps : int
        Total number of MCMC steps.
    n_walkers : int
        Number of walkers.
    n_params : int
        Number of parameters.
    converged : bool
        True if R-hat < 1.1, n_eff > 100, and acceptance fraction is
        in the acceptable range.
    flags : int
        Quality flags (bitmask).
    """

    rhat: float = np.nan
    rhat_per_param: np.ndarray = field(default_factory=lambda: np.array([]))
    n_eff: float = np.nan
    n_eff_per_param: np.ndarray = field(default_factory=lambda: np.array([]))
    acceptance_fraction: float = np.nan
    total_steps: int = 0
    n_walkers: int = 0
    n_params: int = 0
    converged: bool = False
    flags: int = 0

    def to_dict(self) -> dict:
        return {
            "mcmc_rhat": float(self.rhat) if np.isfinite(self.rhat) else np.nan,
            "mcmc_n_eff": float(self.n_eff) if np.isfinite(self.n_eff) else np.nan,
            "mcmc_acc_frac": float(self.acceptance_fraction) if np.isfinite(self.acceptance_fraction) else np.nan,
            "mcmc_total_steps": int(self.total_steps),
            "mcmc_n_walkers": int(self.n_walkers),
            "mcmc_converged": bool(self.converged),
            "mcmc_flags": int(self.flags),
        }

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines = [
            f"MCMC Convergence: {status}",
            f"  R-hat: {self.rhat:.3f} (threshold: 1.1)",
            f"  n_eff: {self.n_eff:.0f} (minimum: 100)",
            f"  Acceptance: {self.acceptance_fraction:.3f} (range: 0.15-0.8)",
            f"  Steps: {self.total_steps}, Walkers: {self.n_walkers}",
        ]
        if self.flags:
            flag_names = []
            if self.flags & MCMC_FLAG_HIGH_RHAT:
                flag_names.append("HIGH_RHAT")
            if self.flags & MCMC_FLAG_LOW_ESS:
                flag_names.append("LOW_ESS")
            if self.flags & MCMC_FLAG_BAD_ACCEPTANCE:
                flag_names.append("BAD_ACCEPTANCE")
            if self.flags & MCMC_FLAG_TOO_FEW_STEPS:
                flag_names.append("TOO_FEW_STEPS")
            lines.append(f"  Flags: {', '.join(flag_names)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

MCMC_FLAG_OK = 0
MCMC_FLAG_HIGH_RHAT = 1 << 0       # R-hat > 1.1
MCMC_FLAG_LOW_ESS = 1 << 1         # n_eff < 100
MCMC_FLAG_BAD_ACCEPTANCE = 1 << 2  # acc_frac outside [0.15, 0.8]
MCMC_FLAG_TOO_FEW_STEPS = 1 << 3   # total_steps < 100


# ---------------------------------------------------------------------------
# R-hat (Gelman-Rubin) from split chains
# ---------------------------------------------------------------------------

def compute_split_rhat(chain: np.ndarray) -> float:
    """Compute split-chain Gelman-Rubin R-hat for a single parameter.

    The chain is split in half, treating each half as an independent
    chain.  R-hat compares between-chain variance to within-chain
    variance.

    Parameters
    ----------
    chain : np.ndarray, shape (n_walkers, n_steps)
        MCMC chain for a single parameter.

    Returns
    -------
    float
        R-hat statistic.  Values close to 1.0 indicate convergence.
        Values > 1.1 indicate non-convergence.
    """
    chain = np.asarray(chain, float)
    if chain.ndim != 2:
        raise ValueError(f"chain must be 2-D (n_walkers, n_steps), got {chain.ndim}-D")

    n_walkers, n_steps = chain.shape

    # Need at least 4 steps per half-chain
    if n_steps < 8:
        return np.nan

    half = n_steps // 2
    chain1 = chain[:, :half]
    chain2 = chain[:, half:2 * half]

    # Treat as 2*n_walkers chains of length half
    m = 2 * n_walkers  # number of split chains
    n = half           # length of each split chain

    # Chain means
    chain_means = np.concatenate([chain1.mean(axis=1), chain2.mean(axis=1)])

    # Overall mean
    overall_mean = np.mean(chain_means)

    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)

    # Within-chain variance
    chain_vars = np.concatenate([
        np.var(chain1, axis=1, ddof=1),
        np.var(chain2, axis=1, ddof=1),
    ])
    W = np.mean(chain_vars)

    if W <= 0:
        return np.nan

    # Marginal posterior variance estimate
    var_hat = (n - 1) / n * W + B / n

    if var_hat <= 0:
        return np.nan

    rhat = np.sqrt(var_hat / W)
    return float(rhat)


def compute_rhat_all_params(chain: np.ndarray) -> np.ndarray:
    """Compute R-hat for all parameters in a 3-D chain.

    Parameters
    ----------
    chain : np.ndarray, shape (n_walkers, n_steps, n_params)

    Returns
    -------
    np.ndarray, shape (n_params,)
        R-hat for each parameter.
    """
    chain = np.asarray(chain, float)
    if chain.ndim != 3:
        raise ValueError(f"chain must be 3-D (n_walkers, n_steps, n_params), got {chain.ndim}-D")

    n_params = chain.shape[2]
    rhats = np.empty(n_params)
    for i in range(n_params):
        rhats[i] = compute_split_rhat(chain[:, :, i])
    return rhats


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

def compute_ess(chain: np.ndarray) -> float:
    """Compute effective sample size for a single parameter.

    Uses the autocorrelation-based estimator:

        n_eff = n_walkers * n_steps / (1 + 2 * sum(rho_k))

    where rho_k is the autocorrelation at lag k.

    Parameters
    ----------
    chain : np.ndarray, shape (n_walkers, n_steps)

    Returns
    -------
    float
        Effective sample size.
    """
    chain = np.asarray(chain, float)
    if chain.ndim != 2:
        raise ValueError(f"chain must be 2-D, got {chain.ndim}-D")

    n_walkers, n_steps = chain.shape
    if n_steps < 4:
        return float(n_walkers * n_steps)

    # Compute autocorrelation for each walker, then average
    taus = []
    for w in range(n_walkers):
        c = chain[w] - chain[w].mean()
        if np.all(c == 0):
            continue
        n = len(c)
        # Pad to next power of 2 for efficient FFT autocorrelation
        nfft = 1
        while nfft < 2 * n:
            nfft *= 2
        f = np.fft.fft(c, n=nfft)
        acf = np.fft.ifft(f * np.conj(f)).real[:n]
        acf /= acf[0]

        # Find where ACF first drops below 0.05 (Geyer's initial positive sequence)
        # Use the initial monotone positive sequence estimator
        # Sum consecutive pairs; stop when the sum becomes negative
        # This is the Geyer (1992) initial positive sequence estimator
        tau = 1.0
        for k in range(1, n - 1, 2):
            if k + 1 >= n:
                break
            pair_sum = acf[k] + acf[k + 1]
            if pair_sum < 0:
                break
            tau += 2 * pair_sum

        taus.append(max(tau, 1.0))

    if not taus:
        return float(n_walkers * n_steps)

    mean_tau = np.mean(taus)
    n_eff = n_walkers * n_steps / mean_tau
    return float(n_eff)


def compute_ess_all_params(chain: np.ndarray) -> np.ndarray:
    """Compute ESS for all parameters in a 3-D chain."""
    chain = np.asarray(chain, float)
    if chain.ndim != 3:
        raise ValueError(f"chain must be 3-D, got {chain.ndim}-D")

    n_params = chain.shape[2]
    ess = np.empty(n_params)
    for i in range(n_params):
        ess[i] = compute_ess(chain[:, :, i])
    return ess


# ---------------------------------------------------------------------------
# Full convergence report
# ---------------------------------------------------------------------------

def assess_convergence(
    chain: np.ndarray,
    acceptance_fraction: float | np.ndarray,
    rhat_threshold: float = 1.1,
    ess_threshold: float = 100,
    acc_frac_range: tuple[float, float] = (0.15, 0.8),
    min_steps: int = 100,
) -> MCMCConvergenceReport:
    """Assess MCMC convergence from chain and acceptance fraction.

    Parameters
    ----------
    chain : np.ndarray, shape (n_walkers, n_steps, n_params)
        MCMC chain (after burn-in removal).
    acceptance_fraction : float or array
        Mean acceptance fraction (or per-walker array).
    rhat_threshold : float
        R-hat must be below this for convergence (default 1.1).
    ess_threshold : float
        Minimum effective sample size (default 100).
    acc_frac_range : (lo, hi)
        Acceptable acceptance fraction range.
    min_steps : int
        Minimum number of steps for reliable diagnostics.

    Returns
    -------
    MCMCConvergenceReport
    """
    report = MCMCConvergenceReport()

    chain = np.asarray(chain, float)
    if chain.ndim != 3:
        # Try to handle 2-D chain (single parameter)
        if chain.ndim == 2:
            chain = chain[:, :, np.newaxis]
        else:
            log.warning("MCMC convergence: chain has wrong dimensions (%d-D)", chain.ndim)
            report.flags |= MCMC_FLAG_TOO_FEW_STEPS
            return report

    n_walkers, n_steps, n_params = chain.shape
    report.n_walkers = n_walkers
    report.n_params = n_params
    report.total_steps = n_steps

    if n_steps < min_steps:
        report.flags |= MCMC_FLAG_TOO_FEW_STEPS
        log.warning("MCMC convergence: only %d steps (< %d minimum)", n_steps, min_steps)

    # R-hat
    try:
        rhats = compute_rhat_all_params(chain)
        report.rhat_per_param = rhats
        report.rhat = float(np.nanmax(rhats))
        if np.isfinite(report.rhat) and report.rhat > rhat_threshold:
            report.flags |= MCMC_FLAG_HIGH_RHAT
    except Exception as e:
        log.debug("R-hat computation failed: %s", e)
        report.rhat = np.nan

    # ESS
    try:
        ess = compute_ess_all_params(chain)
        report.n_eff_per_param = ess
        report.n_eff = float(np.nanmin(ess))
        if np.isfinite(report.n_eff) and report.n_eff < ess_threshold:
            report.flags |= MCMC_FLAG_LOW_ESS
    except Exception as e:
        log.debug("ESS computation failed: %s", e)
        report.n_eff = np.nan

    # Acceptance fraction
    if isinstance(acceptance_fraction, np.ndarray):
        acc = float(np.mean(acceptance_fraction))
    else:
        acc = float(acceptance_fraction)
    report.acceptance_fraction = acc

    if np.isfinite(acc):
        if acc < acc_frac_range[0] or acc > acc_frac_range[1]:
            report.flags |= MCMC_FLAG_BAD_ACCEPTANCE

    # Overall convergence
    report.converged = (
        report.flags == 0
        and np.isfinite(report.rhat)
        and report.rhat <= rhat_threshold
        and np.isfinite(report.n_eff)
        and report.n_eff >= ess_threshold
    )

    return report
