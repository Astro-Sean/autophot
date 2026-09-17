"""Unified quality flag system for AutoPhOT photometry.

Provides a composable bitmask that distinguishes all failure modes
identified in the photometry pipeline.  This does NOT replace existing
flags (photutils flags, diff_quality_class, is_detection); it provides
an additional structured field that unifies them.

Flag bits are organised into sections:
  0x0000_0001 - 0x0000_00FF: Data/validation flags
  0x0000_0100 - 0x0000_FFFF: PSF flags
  0x0001_0000 - 0x00FF_FFFF: Fitting flags
  0x0100_0000 - 0xFFFF_FFFF: Pipeline/calibration flags

Scientific motivation:  A single ``quality_flags`` integer lets downstream
consumers (light-curve code, automated pipelines, users) quickly assess
measurement quality without parsing multiple independent flag systems.
"""

from __future__ import annotations

from enum import IntFlag, auto


class QualityFlag(IntFlag):
    """Composable quality flags for photometric measurements.

    Use ``QualityFlag.OK`` for a clean measurement.  Multiple flags can
    be combined: ``QualityFlag.LOW_SN | QualityFlag.CROWDED``.

    The ``is_fatal()`` method returns True for flags that indicate the
    measurement should not be trusted at all.
    """

    OK = 0

    # ---- Data / validation (0x01 - 0xFF) ----
    NON_FINITE_PIXELS = auto()        # Non-finite pixels in fitting region
    HIGH_MASK_FRACTION = auto()       # > 20% of pixels masked
    EDGE_PROXIMITY = auto()           # Source near image boundary
    SATURATED = auto()                # Source is saturated or in non-linear regime
    BAD_BACKGROUND = auto()           # Background estimation unreliable

    # ---- PSF (0x100 - 0xFFFF) ----
    PSF_FEW_STARS = 1 << 8            # Too few PSF stars for reliable model
    PSF_POOR_FIT = 1 << 9             # PSF held-out validation failed
    PSF_SPATIAL_VARIATION = 1 << 10   # Significant PSF variation across field
    PSF_UNDERSAMPLED = 1 << 11        # PSF built from undersampled data
    PSF_MODEL_MISMATCH = 1 << 12      # Residuals indicate PSF model mismatch

    # ---- Fitting (0x10000 - 0xFFFFFF) ----
    LOW_SN = 1 << 16                  # S/N below detection threshold
    OPTIMIZER_NON_CONVERGENCE = 1 << 17  # Optimizer did not converge
    MCMC_NON_CONVERGENCE = 1 << 18    # MCMC did not converge (R-hat > 1.1)
    MCMC_LOW_ESS = 1 << 19            # MCMC effective sample size too low
    PARAMETER_AT_BOUND = 1 << 20      # Fitted parameter hit boundary
    SUSPICIOUS_RESIDUALS = 1 << 21    # Residuals show structure (high chi2)
    CROWDED = 1 << 22                 # Source is blended/crowded
    NEGATIVE_FLUX = 1 << 23           # Fitted flux is negative
    MCMC_FALLBACK = 1 << 24           # MCMC fell back to deterministic fit
    POSTERIOR_DEGENERACY = 1 << 25    # Strong parameter degeneracy detected

    # ---- Pipeline / calibration (0x1000000+) ----
    NO_ZEROPPOINT = 1 << 26           # No zeropoint available
    ZEROPPOINT_FEW_STARS = 1 << 27    # Too few stars for zeropoint
    SUBTRACTION_DOWNGRADE = 1 << 28   # Difference image quality: downgrade
    SUBTRACTION_FAIL = 1 << 29        # Difference image quality: fail
    ALIGNMENT_POOR = 1 << 30          # Template alignment quality poor
    # Bit 31 reserved (sign bit in some contexts)

    # ---- Convenience groupings ----
    @classmethod
    def fatal_flags(cls) -> "QualityFlag":
        """Flags that indicate the measurement should not be trusted."""
        return (
            cls.SATURATED
            | cls.PSF_POOR_FIT
            | cls.OPTIMIZER_NON_CONVERGENCE
            | cls.MCMC_NON_CONVERGENCE
            | cls.SUBTRACTION_FAIL
            | cls.NEGATIVE_FLUX
        )

    @classmethod
    def degraded_flags(cls) -> "QualityFlag":
        """Flags that indicate degraded but potentially usable quality."""
        return (
            cls.LOW_SN
            | cls.HIGH_MASK_FRACTION
            | cls.EDGE_PROXIMITY
            | cls.BAD_BACKGROUND
            | cls.PSF_FEW_STARS
            | cls.PSF_SPATIAL_VARIATION
            | cls.PSF_UNDERSAMPLED
            | cls.PSF_MODEL_MISMATCH
            | cls.MCMC_LOW_ESS
            | cls.PARAMETER_AT_BOUND
            | cls.SUSPICIOUS_RESIDUALS
            | cls.CROWDED
            | cls.MCMC_FALLBACK
            | cls.POSTERIOR_DEGENERACY
            | cls.NO_ZEROPPOINT
            | cls.ZEROPPOINT_FEW_STARS
            | cls.SUBTRACTION_DOWNGRADE
            | cls.ALIGNMENT_POOR
        )

    def is_fatal(self) -> bool:
        """True if any fatal flag is set."""
        return bool(self & self.fatal_flags())

    def is_degraded(self) -> bool:
        """True if any degraded flag is set (but no fatal flags)."""
        return bool(self & self.degraded_flags()) and not self.is_fatal()

    def is_ok(self) -> bool:
        """True if no flags are set (clean measurement)."""
        return self == QualityFlag.OK

    def flag_names(self) -> list[str]:
        """Return list of flag names that are set."""
        names = []
        for member in QualityFlag:
            if member == QualityFlag.OK:
                continue
            if self & member:
                names.append(member.name)
        return names

    def to_dict(self) -> dict:
        """Return a dictionary representation for serialization."""
        return {
            "quality_flags": int(self),
            "quality_status": (
                "fatal" if self.is_fatal()
                else "degraded" if self.is_degraded()
                else "ok"
            ),
            "quality_flag_names": self.flag_names(),
        }


# ---------------------------------------------------------------------------
# Mapping from existing pipeline flags to QualityFlag
# ---------------------------------------------------------------------------

def from_photutils_flags(flags: int) -> QualityFlag:
    """Map photutils PSF photometry flags to QualityFlag.

    photutils flag bits (from photutils.psf.utils.decode_psf_flags):
    - bit 0: fit did not converge
    - bit 1: fit at parameter bounds
    - bit 2: data contains non-finite values
    - bit 4: no covariance matrix
    - bit 8: negative flux
    - bit 16: NO_COVARIANCE (photutils 3.0+)
    """
    qf = QualityFlag.OK
    if flags & (1 << 0):
        qf |= QualityFlag.OPTIMIZER_NON_CONVERGENCE
    if flags & (1 << 1):
        qf |= QualityFlag.PARAMETER_AT_BOUND
    if flags & (1 << 2):
        qf |= QualityFlag.NON_FINITE_PIXELS
    if flags & (1 << 8):
        qf |= QualityFlag.NEGATIVE_FLUX
    return qf


def from_diff_quality(quality_class: str) -> QualityFlag:
    """Map difference-image quality classification to QualityFlag."""
    if quality_class is None:
        return QualityFlag.OK
    qc = str(quality_class).lower().strip()
    if qc == "fail":
        return QualityFlag.SUBTRACTION_FAIL
    elif qc == "downgrade":
        return QualityFlag.SUBTRACTION_DOWNGRADE
    else:
        return QualityFlag.OK


def from_snr(snr: float, detection_limit: float = 3.0) -> QualityFlag:
    """Map S/N to quality flag."""
    if snr < detection_limit:
        return QualityFlag.LOW_SN
    return QualityFlag.OK


def from_chi2(reduced_chi2: float, threshold: float = 5.0) -> QualityFlag:
    """Map reduced chi-squared to quality flag."""
    if reduced_chi2 > threshold:
        return QualityFlag.SUSPICIOUS_RESIDUALS
    return QualityFlag.OK


def from_mcmc_diagnostics(
    rhat: float | None = None,
    n_eff: float | None = None,
    acc_frac: float | None = None,
    converged: bool | None = None,
) -> QualityFlag:
    """Map MCMC convergence diagnostics to quality flags.

    Parameters
    ----------
    rhat : float
        Gelman-Rubin R-hat statistic.  Should be < 1.1 for convergence.
    n_eff : float
        Effective sample size.  Should be > 100 for reliable posteriors.
    acc_frac : float
        Acceptance fraction.  Should be in [0.15, 0.8] for emcee.
    converged : bool
        Explicit convergence flag from the sampler.
    """
    qf = QualityFlag.OK
    if converged is False:
        qf |= QualityFlag.MCMC_NON_CONVERGENCE
    if rhat is not None and np.isfinite(rhat) and rhat > 1.1:
        qf |= QualityFlag.MCMC_NON_CONVERGENCE
    if n_eff is not None and np.isfinite(n_eff) and n_eff < 100:
        qf |= QualityFlag.MCMC_LOW_ESS
    if acc_frac is not None and np.isfinite(acc_frac):
        if acc_frac < 0.15 or acc_frac > 0.8:
            qf |= QualityFlag.MCMC_NON_CONVERGENCE
    return qf


import numpy as np  # noqa: E402
