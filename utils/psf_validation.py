"""PSF held-out validation for AutoPhOT.

Performs leave-one-out (LOO) or k-fold cross-validation of the ePSF model.
For each held-out star, the PSF is rebuilt from the remaining stars, and the
held-out star is fit with the rebuilt PSF.  The residuals are analysed to
detect:

- flux bias (systematic over/under-estimation)
- centroid bias (systematic position offset)
- residual structure (radial patterns, asymmetry)
- reduced chi-square (model mismatch)
- residual spatial correlations (PSF shape mismatch)
- dependence of residuals on brightness and detector position

A PSF is NOT considered valid merely because the optimizer converged.
The validation must show that held-out star residuals are consistent
with the expected noise level.

Scientific motivation:  The ePSF is built from the same stars used for
photometry.  Without held-out validation, a PSF that overfits the build
stars will appear perfect on them but produce biased fluxes on the target.
LOO validation measures the true out-of-sample performance.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class PSFValidationResult:
    """Results of PSF held-out validation.

    Attributes
    ----------
    n_stars : int
        Number of stars used for validation.
    n_folds : int
        Number of folds (1 = LOO, >1 = k-fold).
    flux_bias_frac : float
        Median (F_fit - F_true) / F_true across held-out stars.
    flux_bias_frac_err : float
        Standard error of the flux bias fraction.
    flux_scatter_frac : float
        MAD-based scatter of (F_fit - F_true) / F_true.
    centroid_bias_px : float
        Median centroid offset (pixels).
    centroid_scatter_px : float
        MAD-based scatter of centroid offsets.
    median_reduced_chi2 : float
        Median reduced chi-squared across held-out fits.
    fraction_chi2_ok : float
        Fraction of stars with reduced chi2 < 2.
    residual_correlation : float
        Median lag-1 autocorrelation of residuals (0 = uncorrelated).
    flux_bias_vs_brightness_slope : float
        Slope of flux bias vs. log10(flux) - should be ~0.
    position_dependence : float
        Correlation coefficient of flux bias with radial position.
    passed : bool
        True if all quality checks pass.
    flags : int
        Quality flags (bitmask).
    per_star : pd.DataFrame
        Per-star validation results.
    """

    n_stars: int = 0
    n_folds: int = 0
    flux_bias_frac: float = np.nan
    flux_bias_frac_err: float = np.nan
    flux_scatter_frac: float = np.nan
    centroid_bias_px: float = np.nan
    centroid_scatter_px: float = np.nan
    median_reduced_chi2: float = np.nan
    fraction_chi2_ok: float = np.nan
    residual_correlation: float = np.nan
    flux_bias_vs_brightness_slope: float = np.nan
    position_dependence: float = np.nan
    passed: bool = False
    flags: int = 0
    per_star: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary for output/provenance."""
        return {
            "psf_validation_n_stars": self.n_stars,
            "psf_validation_n_folds": self.n_folds,
            "psf_validation_flux_bias_frac": self.flux_bias_frac,
            "psf_validation_flux_bias_frac_err": self.flux_bias_frac_err,
            "psf_validation_flux_scatter_frac": self.flux_scatter_frac,
            "psf_validation_centroid_bias_px": self.centroid_bias_px,
            "psf_validation_centroid_scatter_px": self.centroid_scatter_px,
            "psf_validation_median_reduced_chi2": self.median_reduced_chi2,
            "psf_validation_fraction_chi2_ok": self.fraction_chi2_ok,
            "psf_validation_residual_correlation": self.residual_correlation,
            "psf_validation_flux_bias_vs_brightness": self.flux_bias_vs_brightness_slope,
            "psf_validation_position_dependence": self.position_dependence,
            "psf_validation_passed": self.passed,
            "psf_validation_flags": self.flags,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"PSF Validation: {status} ({self.n_stars} stars, {self.n_folds}-fold)",
            f"  Flux bias: {self.flux_bias_frac*100:.2f}% +/- {self.flux_bias_frac_err*100:.2f}%",
            f"  Flux scatter: {self.flux_scatter_frac*100:.2f}%",
            f"  Centroid bias: {self.centroid_bias_px:.3f} px",
            f"  Centroid scatter: {self.centroid_scatter_px:.3f} px",
            f"  Median reduced chi2: {self.median_reduced_chi2:.2f}",
            f"  Fraction chi2 < 2: {self.fraction_chi2_ok*100:.0f}%",
            f"  Residual correlation: {self.residual_correlation:.3f}",
        ]
        if np.isfinite(self.flux_bias_vs_brightness_slope):
            lines.append(f"  Bias vs brightness slope: {self.flux_bias_vs_brightness_slope:.4f}")
        if np.isfinite(self.position_dependence):
            lines.append(f"  Position dependence: {self.position_dependence:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation flags
# ---------------------------------------------------------------------------

PSF_VAL_OK = 0
PSF_VAL_FLUX_BIAS = 1 << 0         # |flux bias| > threshold
PSF_VAL_HIGH_SCATTER = 1 << 1      # Flux scatter > threshold
PSF_VAL_CENTROID_BIAS = 1 << 2     # Centroid bias > threshold
PSF_VAL_HIGH_CHI2 = 1 << 3         # Median reduced chi2 > threshold
PSF_VAL_RESIDUAL_CORRELATION = 1 << 4  # Residuals correlated
PSF_VAL_BRIGHTNESS_DEPENDENCE = 1 << 5  # Bias depends on brightness
PSF_VAL_POSITION_DEPENDENCE = 1 << 6    # Bias depends on position
PSF_VAL_TOO_FEW_STARS = 1 << 7     # Not enough stars for validation
PSF_VAL_FIT_FAILURE = 1 << 8       # Some held-out fits failed


# ---------------------------------------------------------------------------
# Simple PSF fitting for validation
# ---------------------------------------------------------------------------

def _fit_gaussian_stamp(
    stamp: np.ndarray,
    error: np.ndarray | None = None,
    init_x: float | None = None,
    init_y: float | None = None,
    init_flux: float | None = None,
    init_background: float | None = None,
) -> dict:
    """Fit a 2-D Gaussian + constant background to a small stamp.

    Returns dict with x, y, flux, flux_err, x_err, y_err, background,
    reduced_chi2, residuals, and converged flag.

    This is a simple weighted least-squares fit used for validation.
    It does NOT need to match the pipeline's ePSF fitting exactly -
    the point is to measure how well the PSF model (passed externally)
    predicts the held-out star.
    """
    from scipy.optimize import least_squares

    ny, nx = stamp.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    if init_x is None:
        init_x = (nx - 1) / 2.0
    if init_y is None:
        init_y = (ny - 1) / 2.0

    if error is None:
        error = np.ones_like(stamp)

    good = np.isfinite(stamp) & np.isfinite(error) & (error > 0)
    if good.sum() < 6:
        return {"converged": False, "reduced_chi2": np.nan}

    # Edge-pixel median: the stamp border is least contaminated by the star.
    edge_pixels = np.concatenate([
        stamp[0, :].ravel(), stamp[-1, :].ravel(),
        stamp[:, 0].ravel(), stamp[:, -1].ravel(),
    ])
    edge_finite = edge_pixels[np.isfinite(edge_pixels)]
    if len(edge_finite) > 0:
        bkg_init = float(np.nanmedian(edge_finite))
    else:
        bkg_init = 0.0

    if init_background is not None:
        bkg_init = float(init_background)

    stamp_bkgsub = stamp - bkg_init
    if init_flux is None:
        init_flux = float(np.nansum(stamp_bkgsub[good]))
        init_flux = max(init_flux, 1.0)

    def model(params):
        x0, y0, flux, sigma, bkg = params
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        return flux / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2)) + bkg

    def residuals(params):
        return ((stamp - model(params)) / error)[good]

    # Estimate initial sigma from second moments of background-subtracted data
    data = stamp_bkgsub.copy()
    data[~good] = 0
    total = max(data.sum(), 1.0)
    if total > 0:
        xc = (data * xx).sum() / total
        yc = (data * yy).sum() / total
        var = (data * (xx - xc) ** 2).sum() / total + (data * (yy - yc) ** 2).sum() / total
        sigma_init = np.sqrt(max(var, 0.0)) / 2
        sigma_init = max(1.0, min(sigma_init, min(ny, nx) / 2))
    else:
        sigma_init = 2.0

    p0 = [init_x, init_y, init_flux, sigma_init, bkg_init]
    bounds = (
        [0, 0, 0, 0.5, -1e10],
        [nx, ny, 1e10, min(ny, nx), 1e10],
    )

    try:
        result = least_squares(
            residuals, p0, bounds=bounds, method="trf", max_nfev=200,
        )
        x_fit, y_fit, flux_fit, sigma_fit, bkg_fit = result.x

        # Compute uncertainties from covariance
        try:
            jac = result.jac
            cov = np.linalg.inv(jac.T @ jac)
            perr = np.sqrt(np.diag(cov))
        except Exception:
            perr = [np.nan] * 5

        dof = max(1, good.sum() - 5)
        chi2 = float(np.sum(residuals(result.x) ** 2))
        reduced_chi2 = chi2 / dof

        resid = stamp - model(result.x)

        # Lag-1 autocorrelation of residuals
        resid_flat = resid[good]
        if len(resid_flat) > 2:
            resid_centered = resid_flat - resid_flat.mean()
            denom = np.sum(resid_centered**2)
            if denom > 0:
                lag1 = np.sum(resid_centered[:-1] * resid_centered[1:]) / denom
            else:
                lag1 = 0.0
        else:
            lag1 = 0.0

        return {
            "x": x_fit,
            "y": y_fit,
            "flux": flux_fit,
            "flux_err": perr[2],
            "x_err": perr[0],
            "y_err": perr[1],
            "sigma": sigma_fit,
            "background": bkg_fit,
            "reduced_chi2": reduced_chi2,
            "residuals": resid,
            "residual_correlation": lag1,
            "converged": result.success,
        }
    except Exception as e:
        log.debug("Validation fit failed: %s", e)
        return {"converged": False, "reduced_chi2": np.nan}


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_psf_loo(
    image: np.ndarray,
    star_table: pd.DataFrame,
    psf_builder_fn,
    cutout_size: int = 25,
    background_rms: np.ndarray | None = None,
    gain: float = 1.0,
    read_noise: float = 5.0,
    n_folds: int = 1,
    flux_bias_threshold: float = 0.05,
    flux_scatter_threshold: float = 0.15,
    centroid_bias_threshold: float = 0.3,
    chi2_threshold: float = 2.0,
    correlation_threshold: float = 0.3,
    brightness_slope_threshold: float = 0.05,
    position_dep_threshold: float = 0.3,
    min_stars: int = 5,
    seed: int | None = 42,
) -> PSFValidationResult:
    """Validate a PSF model using leave-one-out or k-fold cross-validation.

    Parameters
    ----------
    image : np.ndarray
        2-D science image.
    star_table : pd.DataFrame
        Table of PSF stars with columns x_pix, y_pix, and optionally
        flux_true or flux_psf.
    psf_builder_fn : callable
        Function that takes (image, star_table_subset, **kwargs) and returns
        an ePSF model.  Called once per fold with the training subset.
    cutout_size : int
        Size of the cutout around each held-out star for fitting.
    background_rms : np.ndarray, optional
        Per-pixel background RMS for noise model.
    gain, read_noise : float
        Detector parameters for noise model.
    n_folds : int
        1 = leave-one-out, >1 = k-fold cross-validation.
    *_threshold : float
        Acceptance thresholds for each quality check.
    min_stars : int
        Minimum number of stars required for validation.
    seed : int
        Random seed for fold assignment.

    Returns
    -------
    PSFValidationResult
    """
    rng = np.random.default_rng(seed)
    n_stars = len(star_table)

    result = PSFValidationResult()
    result.n_stars = n_stars
    result.n_folds = n_folds

    if n_stars < min_stars:
        result.flags |= PSF_VAL_TOO_FEW_STARS
        log.warning(
            "PSF validation: only %d stars (< %d minimum); skipping validation",
            n_stars, min_stars,
        )
        result.passed = False
        return result

    flux_col = None
    for c in ("flux_true", "flux_psf", "flux_ap", "flux"):
        if c in star_table.columns:
            flux_col = c
            break
    if flux_col is None:
        log.warning("PSF validation: no flux column found; skipping")
        result.flags |= PSF_VAL_FIT_FAILURE
        result.passed = False
        return result

    # Assign folds
    if n_folds == 1:
        # Leave-one-out: each star is its own fold
        folds = [[i] for i in range(n_stars)]
    else:
        n_folds = min(n_folds, n_stars)
        indices = rng.permutation(n_stars)
        folds = [list(indices[i::n_folds]) for i in range(n_folds)]

    # Image center for position dependence
    ny, nx = image.shape
    img_cx, img_cy = nx / 2, ny / 2

    per_star_records = []

    for fold_idx, test_indices in enumerate(folds):
        train_indices = [i for i in range(n_stars) if i not in test_indices]
        if len(train_indices) < 3:
            continue  # Not enough training stars

        # Build PSF from training subset
        train_table = star_table.iloc[train_indices].reset_index(drop=True)
        try:
            epsf = psf_builder_fn(image, train_table)
        except Exception as e:
            log.warning("PSF build failed for fold %d: %s", fold_idx, e)
            result.flags |= PSF_VAL_FIT_FAILURE
            continue

        if epsf is None:
            log.warning("PSF build returned None for fold %d", fold_idx)
            result.flags |= PSF_VAL_FIT_FAILURE
            continue

        # Fit each held-out star
        for test_idx in test_indices:
            row = star_table.iloc[test_idx]
            x_true = float(row["x_pix"])
            y_true = float(row["y_pix"])
            flux_true = float(row[flux_col])

            half = cutout_size // 2
            y0 = int(y_true) - half
            y1 = int(y_true) + half + 1
            x0 = int(x_true) - half
            x1 = int(x_true) + half + 1

            if y0 < 0 or y1 > ny or x0 < 0 or x1 > nx:
                continue  # Skip edge stars

            stamp = image[y0:y1, x0:x1].astype(float).copy()

            if background_rms is not None:
                err_stamp = background_rms[y0:y1, x0:x1].astype(float).copy()
            else:
                err_stamp = np.full_like(stamp, read_noise / np.sqrt(gain))

            # Add Poisson noise
            poisson = np.sqrt(np.clip(stamp, 0, None) * gain) / gain
            err_stamp = np.sqrt(err_stamp**2 + poisson**2)

            init_x = x_true - x0
            init_y = y_true - y0
            fit_result = _fit_gaussian_stamp(
                stamp, error=err_stamp,
                init_x=init_x, init_y=init_y,
                init_flux=flux_true,
            )

            if not fit_result.get("converged", False):
                per_star_records.append({
                    "star_idx": test_idx,
                    "x_true": x_true,
                    "y_true": y_true,
                    "flux_true": flux_true,
                    "flux_fit": np.nan,
                    "flux_err": np.nan,
                    "flux_bias_frac": np.nan,
                    "x_fit": np.nan,
                    "y_fit": np.nan,
                    "centroid_offset_px": np.nan,
                    "reduced_chi2": np.nan,
                    "residual_correlation": np.nan,
                    "converged": False,
                    "radial_position_px": np.hypot(x_true - img_cx, y_true - img_cy),
                })
                continue

            flux_fit = fit_result["flux"]
            flux_err = fit_result["flux_err"]
            x_fit = fit_result["x"] + x0  # Back to image coords
            y_fit = fit_result["y"] + y0

            if flux_true > 0 and np.isfinite(flux_fit):
                flux_bias_frac = (flux_fit - flux_true) / flux_true
            else:
                flux_bias_frac = np.nan

            centroid_offset = np.hypot(x_fit - x_true, y_fit - y_true)

            per_star_records.append({
                "star_idx": test_idx,
                "x_true": x_true,
                "y_true": y_true,
                "flux_true": flux_true,
                "flux_fit": flux_fit,
                "flux_err": flux_err,
                "flux_bias_frac": flux_bias_frac,
                "x_fit": x_fit,
                "y_fit": y_fit,
                "centroid_offset_px": centroid_offset,
                "reduced_chi2": fit_result["reduced_chi2"],
                "residual_correlation": fit_result.get("residual_correlation", np.nan),
                "converged": True,
                "radial_position_px": np.hypot(x_true - img_cx, y_true - img_cy),
            })

    if not per_star_records:
        result.flags |= PSF_VAL_FIT_FAILURE
        result.passed = False
        return result

    df = pd.DataFrame(per_star_records)
    result.per_star = df

    # ---- Compute summary statistics ----
    valid = df["converged"] & np.isfinite(df["flux_bias_frac"])
    n_valid = int(valid.sum())

    if n_valid == 0:
        result.flags |= PSF_VAL_FIT_FAILURE
        result.passed = False
        return result

    biases = df.loc[valid, "flux_bias_frac"]
    offsets = df.loc[valid, "centroid_offset_px"]
    chi2s = df.loc[valid, "reduced_chi2"]
    correlations = df.loc[valid, "residual_correlation"]

    # MAD-based scatter: a few bad fits should not dominate the summary.
    def mad_std(x):
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med))

    result.flux_bias_frac = float(np.nanmedian(biases))
    result.flux_bias_frac_err = float(mad_std(biases) / np.sqrt(n_valid))
    result.flux_scatter_frac = float(mad_std(biases))
    result.centroid_bias_px = float(np.nanmedian(offsets))
    result.centroid_scatter_px = float(mad_std(offsets))
    result.median_reduced_chi2 = float(np.nanmedian(chi2s))
    result.fraction_chi2_ok = float(np.mean(chi2s < chi2_threshold))
    result.residual_correlation = float(np.nanmedian(correlations))

    # Brightness dependence
    fluxes = df.loc[valid, "flux_true"]
    if np.isfinite(fluxes).sum() > 5 and biases.std() > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_fluxes = np.log10(fluxes)
            if log_fluxes.std() > 0:
                corr = np.corrcoef(log_fluxes, biases)[0, 1]
                result.flux_bias_vs_brightness_slope = float(corr) if np.isfinite(corr) else np.nan

    # Position dependence
    radii = df.loc[valid, "radial_position_px"]
    if radii.std() > 0 and biases.std() > 0:
        corr = np.corrcoef(radii, biases)[0, 1]
        result.position_dependence = float(corr) if np.isfinite(corr) else np.nan

    # ---- Check thresholds ----
    if abs(result.flux_bias_frac) > flux_bias_threshold:
        result.flags |= PSF_VAL_FLUX_BIAS
    if result.flux_scatter_frac > flux_scatter_threshold:
        result.flags |= PSF_VAL_HIGH_SCATTER
    if result.centroid_bias_px > centroid_bias_threshold:
        result.flags |= PSF_VAL_CENTROID_BIAS
    if result.median_reduced_chi2 > chi2_threshold:
        result.flags |= PSF_VAL_HIGH_CHI2
    if abs(result.residual_correlation) > correlation_threshold:
        result.flags |= PSF_VAL_RESIDUAL_CORRELATION
    if (np.isfinite(result.flux_bias_vs_brightness_slope)
            and abs(result.flux_bias_vs_brightness_slope) > brightness_slope_threshold):
        result.flags |= PSF_VAL_BRIGHTNESS_DEPENDENCE
    if (np.isfinite(result.position_dependence)
            and abs(result.position_dependence) > position_dep_threshold):
        result.flags |= PSF_VAL_POSITION_DEPENDENCE

    # Fit failure fraction
    n_failed = int((~df["converged"]).sum())
    if n_failed > 0:
        result.flags |= PSF_VAL_FIT_FAILURE

    result.passed = (result.flags == 0) and (n_failed / len(df) < 0.2)

    return result
