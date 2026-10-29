"""Empirical uncertainty calibration for AutoPhOT photometry.

Compares reported flux uncertainties to the empirical scatter observed
in injection/recovery experiments.  If the reported errors are systematically
underestimated or overestimated, a calibration factor is computed and
can be applied to correct future measurements.

The key diagnostic is the normalized residual:
    z_i = (F_measured_i - F_true_i) / sigma_F_i

For well-calibrated uncertainties, z should have:
    - mean ~ 0 (no systematic bias)
    - std ~ 1 (errors match empirical scatter)

If std(z) > 1: errors are underestimated (overconfident)
If std(z) < 1: errors are overestimated (conservative)

Scientific motivation:  Formal (optimizer-derived) errors assume the
model is perfect and the noise model is correct.  In practice, PSF
mismatch, correlated noise, and background errors inflate the true
scatter.  Empirical calibration catches these effects that formal
errors miss.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class UncertaintyCalibrationResult:
    """Results of uncertainty calibration analysis.

    Attributes
    ----------
    n_sources : int
        Number of sources used for calibration.
    z_mean : float
        Mean of normalized residuals (F_meas - F_true) / sigma_F.
        Should be ~0 for unbiased measurements.
    z_std : float
        Standard deviation of normalized residuals.
        Should be ~1 for well-calibrated uncertainties.
    z_median : float
        Median of normalized residuals (resistant to outliers).
    z_mad_std : float
        MAD-based standard deviation of normalized residuals.
    calibration_factor : float
        Factor by which to multiply reported errors.
        = z_mad_std (so that calibrated z has unit scatter).
    bias_frac : float
        Systematic flux bias fraction (median of (F_meas - F_true) / F_true).
    fraction_within_1sigma : float
        Fraction of sources with |z| < 1.  Should be ~68% for Gaussian.
    fraction_within_2sigma : float
        Fraction of sources with |z| < 2.  Should be ~95% for Gaussian.
    coverage_1sigma_ok : bool
        True if 1-sigma coverage is within [0.60, 0.76] (68% +/- 8%).
    coverage_2sigma_ok : bool
        True if 2-sigma coverage is within [0.90, 0.99] (95% +/- 4%).
    errors_underestimated : bool
        True if z_std > 1.2 (errors too small by > 20%).
    errors_overestimated : bool
        True if z_std < 0.8 (errors too large by > 20%).
    passed : bool
        True if uncertainty calibration is acceptable.
    """

    n_sources: int = 0
    z_mean: float = np.nan
    z_std: float = np.nan
    z_median: float = np.nan
    z_mad_std: float = np.nan
    calibration_factor: float = np.nan
    bias_frac: float = np.nan
    fraction_within_1sigma: float = np.nan
    fraction_within_2sigma: float = np.nan
    coverage_1sigma_ok: bool = False
    coverage_2sigma_ok: bool = False
    errors_underestimated: bool = False
    errors_overestimated: bool = False
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "uncal_n_sources": self.n_sources,
            "uncal_z_mean": self.z_mean,
            "uncal_z_std": self.z_std,
            "uncal_z_median": self.z_median,
            "uncal_z_mad_std": self.z_mad_std,
            "uncal_calibration_factor": self.calibration_factor,
            "uncal_bias_frac": self.bias_frac,
            "uncal_frac_1sigma": self.fraction_within_1sigma,
            "uncal_frac_2sigma": self.fraction_within_2sigma,
            "uncal_coverage_1sigma_ok": self.coverage_1sigma_ok,
            "uncal_coverage_2sigma_ok": self.coverage_2sigma_ok,
            "uncal_errors_underestimated": self.errors_underestimated,
            "uncal_errors_overestimated": self.errors_overestimated,
            "uncal_passed": self.passed,
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Uncertainty Calibration: {status} ({self.n_sources} sources)",
            f"  z mean: {self.z_mean:.3f} (target: ~0)",
            f"  z std: {self.z_std:.3f} (target: ~1)",
            f"  z MAD std: {self.z_mad_std:.3f}",
            f"  Calibration factor: {self.calibration_factor:.3f}",
            f"  Bias: {self.bias_frac*100:.2f}%",
            f"  1-sigma coverage: {self.fraction_within_1sigma*100:.0f}% (target: 68%)",
            f"  2-sigma coverage: {self.fraction_within_2sigma*100:.0f}% (target: 95%)",
        ]
        if self.errors_underestimated:
            lines.append("  WARNING: Errors underestimated (z_std > 1.2)")
        if self.errors_overestimated:
            lines.append("  NOTE: Errors overestimated (z_std < 0.8)")
        return "\n".join(lines)


def calibrate_uncertainties(
    flux_true: np.ndarray,
    flux_measured: np.ndarray,
    flux_error: np.ndarray,
    min_sources: int = 10,
    coverage_1sigma_range: tuple[float, float] = (0.60, 0.76),
    coverage_2sigma_range: tuple[float, float] = (0.90, 0.99),
    error_ratio_threshold: float = 0.2,
) -> UncertaintyCalibrationResult:
    """Analyse uncertainty calibration from injection/recovery results.

    Parameters
    ----------
    flux_true : array
        True injected fluxes.
    flux_measured : array
        Measured (recovered) fluxes.
    flux_error : array
        Reported flux uncertainties.
    min_sources : int
        Minimum number of sources for reliable calibration.
    coverage_1sigma_range : (lo, hi)
        Acceptable range for 1-sigma coverage fraction.
    coverage_2sigma_range : (lo, hi)
        Acceptable range for 2-sigma coverage fraction.
    error_ratio_threshold : float
        Threshold for flagging under/overestimated errors (fractional
        deviation from unit scatter).

    Returns
    -------
    UncertaintyCalibrationResult
    """
    result = UncertaintyCalibrationResult()

    ft = np.asarray(flux_true, float)
    fm = np.asarray(flux_measured, float)
    fe = np.asarray(flux_error, float)

    valid = (
        np.isfinite(ft) & np.isfinite(fm) & np.isfinite(fe)
        & (ft != 0) & (fe > 0)
    )
    n = int(valid.sum())
    result.n_sources = n

    if n < min_sources:
        log.warning(
            "Uncertainty calibration: only %d valid sources (< %d minimum)",
            n, min_sources,
        )
        result.passed = False
        return result

    ft_v = ft[valid]
    fm_v = fm[valid]
    fe_v = fe[valid]

    # Normalized residuals
    z = (fm_v - ft_v) / fe_v

    result.z_mean = float(np.mean(z))
    result.z_std = float(np.std(z))
    result.z_median = float(np.median(z))
    result.z_mad_std = float(1.4826 * np.median(np.abs(z - np.median(z))))
    result.calibration_factor = float(result.z_mad_std) if result.z_mad_std > 0 else 1.0

    result.bias_frac = float(np.median((fm_v - ft_v) / ft_v))

    result.fraction_within_1sigma = float(np.mean(np.abs(z) < 1.0))
    result.fraction_within_2sigma = float(np.mean(np.abs(z) < 2.0))

    result.coverage_1sigma_ok = (
        coverage_1sigma_range[0] <= result.fraction_within_1sigma <= coverage_1sigma_range[1]
    )
    result.coverage_2sigma_ok = (
        coverage_2sigma_range[0] <= result.fraction_within_2sigma <= coverage_2sigma_range[1]
    )

    result.errors_underestimated = result.z_mad_std > (1.0 + error_ratio_threshold)
    result.errors_overestimated = result.z_mad_std < (1.0 - error_ratio_threshold)

    result.passed = (
        result.coverage_1sigma_ok
        and result.coverage_2sigma_ok
        and abs(result.z_median) < 0.5  # No severe bias
        and not result.errors_underestimated
    )

    return result


def apply_calibration(
    flux_error: np.ndarray | float,
    calibration_factor: float,
) -> np.ndarray | float:
    """Apply uncertainty calibration factor to reported errors.

    Parameters
    ----------
    flux_error : array or float
        Reported flux uncertainties.
    calibration_factor : float
        Factor from ``UncertaintyCalibrationResult.calibration_factor``.
        Typically > 1 if errors were underestimated.

    Returns
    -------
    Calibrated uncertainties (flux_error * calibration_factor).
    """
    if not np.isfinite(calibration_factor) or calibration_factor <= 0:
        log.warning(
            "Invalid calibration factor %.3f; returning uncalibrated errors",
            calibration_factor,
        )
        return flux_error
    return flux_error * calibration_factor


def scale_error_columns(
    df,
    factor: float,
    method: str,
    filt: str,
) -> list[str]:
    """Scale a photometry row's error columns by an empirical factor in place.

    The injection/recovery calibration factor applies to the *measurement*
    error only, so flux and instrumental-magnitude errors scale linearly
    while calibrated magnitude errors grow by the quadrature delta

        mag_err^2  ->  mag_err^2 + (factor^2 - 1) * inst_err^2

    which leaves the zeropoint/systematic terms untouched (they are not
    exercised by same-image injections).

    Parameters
    ----------
    df : DataFrame
        Photometry table to update in place (typically the 1-row target table).
    factor : float
        Calibration factor (``UncertaintyCalibrationResult.calibration_factor``).
    method : str
        Recovery method the factor was measured with: "PSF"/"EMCEE"/"MCMC"
        scale the PSF-fit columns, "AP" scales the aperture columns.
    filt : str
        Image filter name used in the calibrated-magnitude column names
        (``{filt}_PSF_err``, ``inst_{filt}_PSF_err``, ...).

    Returns
    -------
    list of str
        Names of the columns that were modified.
    """
    if df is None or len(df) == 0:
        return []
    f = float(factor)
    if not np.isfinite(f) or f <= 0 or np.isclose(f, 1.0):
        return []

    psf_family = str(method).strip().upper() in ("PSF", "EMCEE", "MCMC")
    filt = str(filt)
    scaled: list[str] = []

    def _scale(col: str) -> None:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * f
            scaled.append(col)

    # Calibrated magnitude error: adjust only the instrumental-error term of
    # the quadrature sum (zp error and systematic floors are unchanged).
    inst_col = f"inst_{filt}_{'PSF' if psf_family else 'AP'}_err"
    cal_col = f"{filt}_{'PSF' if psf_family else 'AP'}_err"
    if inst_col in df.columns and cal_col in df.columns:
        inst_err = pd.to_numeric(df[inst_col], errors="coerce").to_numpy(float)
        cal_err = pd.to_numeric(df[cal_col], errors="coerce").to_numpy(float)
        new_err = np.sqrt(
            np.clip(cal_err**2 + (f**2 - 1.0) * inst_err**2, 0.0, None)
        )
        df[cal_col] = new_err
        scaled.append(cal_col)

    if psf_family:
        for col in (
            "flux_PSF_err",
            "flux_PSF_err_normal",
            "flux_PSF_err_inverted",
            inst_col,
            f"inst_{filt}_PSF_normal_err",
            "inst_inverted_err",
        ):
            _scale(col)
    else:
        for col in (
            "flux_AP_err",
            "flux_AP_err_inverted",
            inst_col,
        ):
            _scale(col)
        # Aperture SNR columns are flux/error ratios: scale inversely.
        for col in ("SNR", "SNR_AP_inverted"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / f
                scaled.append(col)

    return scaled
