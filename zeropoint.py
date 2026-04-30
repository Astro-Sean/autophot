#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zeropoint calibration utilities.

This module cleans reference stars, measures magnitude offsets between
instrumental and catalog photometry, and performs robust fitting (with
optional sigma-clipping and colour-term estimation) to derive the final
photometric zeropoint used by the pipeline.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import time
import warnings
import logging
import traceback

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.stats import (
    sigma_clip,
    sigma_clipped_stats,
    mad_std,
)
from scipy.stats import median_abs_deviation  # single source; avoids duplicate

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import RANSACRegressor

from scipy.optimize import minimize
from scipy.odr import ODR, Model, RealData

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from functions import log_step, snr_err, set_size, calculate_bins, normalize_photometric_filter_name
from plotting_utils import get_color, get_marker_size, get_alpha, get_line_width

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# Module-level sklearn estimator
# ===========================================================================


class PenalisedSlopeRegressor(BaseEstimator, RegressorMixin):
    """
    Linear regressor whose slope is softly constrained to ``slope_constraint``
    via an L-BFGS-B penalty.

    Replaces the near-identical ``ConstantRegressor`` (used in
    ``_robust_RANSAC_fit``) and ``ConstrainedSlopeRegressor`` (used in
    ``fit_color_term``).  Both had the same loss function and fit/predict
    logic; only the default ``slope_tolerance`` differed.

    Parameters
    ----------
    slope_constraint : float   target slope value (default 0 -> constant fit)
    slope_tolerance  : float   deviation from constraint before penalty kicks in
    penalty_weight   : float   penalty multiplier
    """

    def __init__(
        self,
        slope_constraint: float = 0.0,
        slope_tolerance: float = 0.5,
        penalty_weight: float = 100.0,
    ):
        self.slope_constraint = slope_constraint
        self.slope_tolerance = slope_tolerance
        self.penalty_weight = penalty_weight

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        x_flat = X.flatten()

        def _loss(params):
            slope, intercept = params
            residuals = y - (slope * x_flat + intercept)
            mse = np.mean(residuals**2)
            excess = max(0.0, abs(slope - self.slope_constraint) - self.slope_tolerance)
            return mse + self.penalty_weight * excess

        init = [0.1, np.mean(y) - 0.1 * np.mean(x_flat)]
        result = minimize(_loss, init, method="L-BFGS-B")
        self.slope_, self.intercept_ = result.x
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.slope_ * X.flatten() + self.intercept_


# ===========================================================================
# zeropoint class
# ===========================================================================


class Zeropoint:
    """
    Photometric zeropoint calibration against a sequence-star catalog.

    Public methods
    --------------
    clean()               - magnitude / SNR quality filter
    get()                 - simple sigma-clipped offset measurement
    fit_zeropoint()       - RANSAC-weighted fit (ZP vs m_inst)
    estimate_zeropoint()  - sigma-clip histogram method
    fit_color_term()      - RANSAC + ODR colour-term fit
    weighted_average()    - inverse-variance weighted mean
    """

    def _normalize_filter(self, filter_name: str) -> str:
        """Normalize a filter name using the pipeline's canonical mapping."""
        if not filter_name:
            return filter_name
        normalized = normalize_photometric_filter_name(filter_name)
        if normalized and str(normalized).strip():
            return str(normalized)
        return filter_name

    def __init__(self, input_yaml: dict):
        self.input_yaml = input_yaml

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def weighted_average(self, values, errors):
        """
        Deprecated: inverse-variance weighting is disabled by convention.

        Returns a robust median and SE(median) from MAD, ignoring *errors*.
        Kept only for backward compatibility with older call sites.
        """
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan, np.nan
        med = float(np.nanmedian(values))
        mad = float(median_abs_deviation(values, nan_policy="omit"))
        n = int(values.size)
        se_med = (1.858 * mad / np.sqrt(n)) if n >= 2 else mad
        return med, float(se_med)

    def get_color_term_for_filter(self, filter_name: str):
        """
        Return (color_col1, color_col2) for the standard colour index of
        *filter_name*.

        Raises ValueError for unknown filters.
        """
        color_map = {
            "u": ("u", "g"),
            "g": ("g", "r"),
            "r": ("g", "r"),
            "i": ("r", "i"),
            "z": ("i", "z"),
            "J": ("J", "H"),
            "H": ("J", "H"),
            "K": ("H", "K"),
            "U": ("U", "B"),
            "B": ("B", "V"),
            "V": ("B", "V"),
            "R": ("V", "R"),
            "I": ("R", "I"),
        }
        if filter_name not in color_map:
            raise ValueError(f"No colour term defined for filter '{filter_name}'")
        return color_map[filter_name]

    def _fallback_zeropoint(self, catalog: pd.DataFrame, use_filter: str) -> dict:
        """
        Median-based zeropoint fallback used when too few sources survive
        quality cuts.  Shared by fit_zeropoint() and estimate_zeropoint().

        Returns
        -------
        zp_params : dict  keyed by 'AP' and 'PSF'
        """
        zp_params = {}
        if catalog is None or getattr(catalog, "empty", False) or len(catalog) == 0:
            for flux_type in ["AP", "PSF"]:
                zp_params[flux_type] = {
                    "zeropoint": np.nan,
                    "zeropoint_error": np.nan,
                    "has_color_term": False,
                }
            return zp_params
        use_filter = self._normalize_filter(use_filter)
        # Aperture correction is applied at photometry stage (main.py), not here.
        # The zeropoint is calculated from raw instrumental fluxes/magnitudes.

        for flux_type in ["AP", "PSF"]:
            fcol = f"flux_{flux_type}"
            if fcol not in catalog.columns:
                zp_params[flux_type] = {
                    "zeropoint": np.nan,
                    "zeropoint_error": np.nan,
                    "has_color_term": False,
                }
                continue

            if use_filter not in catalog.columns:
                zp_params[flux_type] = {
                    "zeropoint": np.nan,
                    "zeropoint_error": np.nan,
                    "has_color_term": False,
                }
                continue

            flux = np.asarray(catalog[fcol].values, float)
            catmag = np.asarray(catalog[use_filter].values, float)

            ok = np.isfinite(flux) & (flux > 0) & np.isfinite(catmag)
            if ok.sum() == 0:
                median_zp = mad_zp = np.nan
            else:
                delta = catmag[ok] - (-2.5 * np.log10(flux[ok]))
                delta = delta[np.isfinite(delta)]
                n_d = len(delta)
                median_zp = np.nanmedian(delta) if n_d else np.nan
                mad_raw = (
                    float(median_abs_deviation(delta, nan_policy="omit"))
                    if n_d
                    else np.nan
                )
                # SE(median) ~ 1.858 * MAD/sqrt(N); use MAD as fallback if N < 2
                mad_zp = (1.858 * mad_raw / np.sqrt(n_d)) if n_d >= 2 else mad_raw

            zp_params[flux_type] = {
                "zeropoint": median_zp,
                "zeropoint_error": mad_zp,
                "has_color_term": False,
            }
        return zp_params

    @staticmethod
    def _compute_delta_mag(flux, flux_err, catmag, catmag_err):
        """
        Convert flux -> instrumental mag, compute dmag = m_cat - m_inst.

        Parameters
        ----------
        flux, flux_err, catmag, catmag_err : 1-D ndarrays (already finite-filtered)

        Returns
        -------
        inst_mag, inst_mag_err, delta_mag, delta_mag_err : ndarrays
        """
        inst_mag = -2.5 * np.log10(flux)
        inst_mag_err = (2.5 / np.log(10.0)) * (flux_err / flux)
        delta_mag = catmag - inst_mag
        delta_mag_err = np.sqrt(inst_mag_err**2 + catmag_err**2)
        return inst_mag, inst_mag_err, delta_mag, delta_mag_err

    @staticmethod
    def _apply_color_correction(
        delta_mag,
        clean_catalog,
        vmask,
        color1,
        color2,
        fixed_color_coeffs,  # Can be (intercept, slope) for linear, (intercept, slope, quad) for quadratic, or (breakpoints, slopes, intercept) for piecewise
        color_coeff_errors = None,  # Corresponding errors
        fit_mode="polynomial",  # "polynomial" or "piecewise"
        n_segments=1,  # Number of segments for piecewise
    ):
        """
        Subtract the colour term from delta_mag and return the corrected array and
        the propagated correction error. Supports linear, quadratic, and piecewise linear color terms.

        Parameters
        ----------
        fixed_color_coeffs : tuple or None
            For linear: (intercept, slope)
            For quadratic: (intercept, slope, quad_coeff)
            For piecewise (n=2): ((breakpoint,), (slope1, slope2), intercept)
        color_coeff_errors : tuple or None
            Corresponding errors for the coefficients
        fit_mode : str
            "polynomial" or "piecewise"
        n_segments : int
            Number of segments for piecewise fitting

        Returns
        -------
        delta_corr, color_corr_err : ndarrays shaped like delta_mag
        """
        c1_vals = np.asarray(
            clean_catalog[color1].values,
            float,
        )[vmask]
        c2_vals = np.asarray(
            clean_catalog[color2].values,
            float,
        )[vmask]

        c1_err = np.asarray(
            clean_catalog.get(
                f"{color1}_err", pd.Series(0.0, index=clean_catalog.index)
            ).values,
            float,
        )[vmask]
        c2_err = np.asarray(
            clean_catalog.get(
                f"{color2}_err", pd.Series(0.0, index=clean_catalog.index)
            ).values,
            float,
        )[vmask]

        color_diff = c1_vals - c2_vals
        sigma_color = np.sqrt(c1_err**2 + c2_err**2)

        if fit_mode == "piecewise":
            # Piecewise linear color term
            if n_segments == 2:
                breakpoints, slopes, intercept = fixed_color_coeffs
                bp = breakpoints[0]
                slope1, slope2 = slopes

                delta_corr = np.zeros_like(delta_mag)
                color_corr_err = np.zeros_like(delta_mag)

                mask1 = color_diff <= bp
                mask2 = color_diff > bp

                # Extract slope errors if available
                if color_coeff_errors is not None:
                    bp_errs, slope_errs, intercept_err = color_coeff_errors
                    slope1_err, slope2_err = slope_errs
                else:
                    slope1_err, slope2_err = None, None

                # Segment 1 correction with error propagation
                delta_corr[mask1] = delta_mag[mask1] - slope1 * color_diff[mask1]
                term_color_measure1 = np.abs(slope1) * sigma_color[mask1]
                if slope1_err is not None:
                    term_color_slope1 = np.abs(slope1_err) * np.abs(color_diff[mask1])
                    color_corr_err[mask1] = np.sqrt(term_color_measure1**2 + term_color_slope1**2)
                else:
                    color_corr_err[mask1] = term_color_measure1

                # Segment 2 correction (account for continuity) with error propagation
                delta_corr[mask2] = delta_mag[mask2] - (slope2 * color_diff[mask2] + (slope1 - slope2) * bp)
                term_color_measure2 = np.abs(slope2) * sigma_color[mask2]
                if slope2_err is not None:
                    term_color_slope2 = np.abs(slope2_err) * np.abs(color_diff[mask2])
                    color_corr_err[mask2] = np.sqrt(term_color_measure2**2 + term_color_slope2**2)
                else:
                    color_corr_err[mask2] = term_color_measure2
            else:
                raise ValueError(f"Unsupported number of segments for piecewise fitting: {n_segments}")
        elif fit_mode == "polynomial":
            if len(fixed_color_coeffs) == 2:
                # Linear color term: delta_corr = delta_mag - slope * color_diff
                intercept, slope = fixed_color_coeffs
                delta_corr = delta_mag - slope * color_diff

                # Propagate errors
                if color_coeff_errors is not None:
                    intercept_err, slope_err = color_coeff_errors
                    term_color_measure = abs(slope) * sigma_color
                    term_color_slope = abs(slope_err) * np.abs(color_diff)
                    color_corr_err = np.sqrt(term_color_measure**2 + term_color_slope**2)
                else:
                    color_corr_err = abs(slope) * sigma_color
            elif len(fixed_color_coeffs) == 3:
                # Quadratic color term: delta_corr = delta_mag - (quad * color_diff^2 + slope * color_diff)
                intercept, slope, quad = fixed_color_coeffs
                delta_corr = delta_mag - (quad * color_diff**2 + slope * color_diff)

                # Propagate errors
                if color_coeff_errors is not None:
                    intercept_err, slope_err, quad_err = color_coeff_errors
                    # Error propagation for quadratic: d(y)/d(color) = 2*quad*color + slope
                    d_correction = 2 * quad * color_diff + slope
                    term_color_measure = np.abs(d_correction) * sigma_color
                    term_color_slope = abs(slope_err) * np.abs(color_diff)
                    term_color_quad = abs(quad_err) * color_diff**2
                    color_corr_err = np.sqrt(term_color_measure**2 + term_color_slope**2 + term_color_quad**2)
                else:
                    d_correction = 2 * quad * color_diff + slope
                    color_corr_err = np.abs(d_correction) * sigma_color
            else:
                raise ValueError(
                    f"Unsupported number of polynomial coefficients: {len(fixed_color_coeffs)}"
                )
        else:
            raise ValueError(f"Unknown fit_mode: {fit_mode!r}")

        return delta_corr, color_corr_err

    def _prepare_catalog(
        self, catalog: pd.DataFrame, threshold: float, use_filter: str, min_sources: int
    ):
        """
        Apply sky-clipping, SNR, and magnitude-error quality cuts.

        Returns
        -------
        clean_catalog : DataFrame or None (None -> caller should use fallback)
        """
        df = catalog.copy()
        if "sky" in df.columns:
            sky_mask = sigma_clip(np.abs(df["sky"].values), sigma=5, maxiters=10)
            df = df[~sky_mask.mask]
        if "threshold" in df.columns:
            df = df[df["threshold"] >= threshold]

        err_col = f"{use_filter}_err"
        if err_col not in df.columns:
            logger.warning(
                "_prepare_catalog: missing error column '%s'; cannot apply quality cuts.",
                err_col,
            )
            return None
        error_mask = np.asarray(df[err_col].values, float) < 0.5
        clean = df[error_mask].copy()

        if len(clean) < min_sources:
            logger.warning(
                f"Too few sources ({len(clean)} < {min_sources}) after quality cuts."
            )
            return None
        return clean

    def _finite_vmask(
        self, clean_catalog: pd.DataFrame, flux_type: str, use_filter: str
    ):
        """
        Boolean mask for rows with all required finite, positive values.

        Returns
        -------
        (flux, flux_err, catmag, catmag_err, vmask) or None if < 3 rows survive
        """
        fcol = f"flux_{flux_type}"
        ecol = f"flux_{flux_type}_err"
        flux = np.asarray(clean_catalog[fcol].values, float)
        flux_err = np.asarray(clean_catalog[ecol].values, float)
        catmag = np.asarray(clean_catalog[use_filter].values, float)
        catmag_err = np.asarray(clean_catalog[f"{use_filter}_err"].values, float)

        # Aperture correction is applied at photometry stage (main.py), not here.
        # The zeropoint is calculated from raw instrumental fluxes/magnitudes.

        vmask = (
            np.isfinite(flux)
            & (flux > 0)
            & np.isfinite(flux_err)
            & (flux_err > 0)
            & np.isfinite(catmag)
            & np.isfinite(catmag_err)
            & (catmag_err > 0)
        )
        if vmask.sum() < 3:
            logger.warning(f"{flux_type}: only {vmask.sum()} valid sources; skipping.")
            return None

        return (
            flux[vmask],
            flux_err[vmask],
            catmag[vmask],
            catmag_err[vmask],
            vmask,
        )

    # -----------------------------------------------------------------------
    # Public: quality filter
    # -----------------------------------------------------------------------

    def clean(
        self,
        sources: pd.DataFrame,
        upperMaglimit: float = 11.0,
        lowerMaglimit: float = 100.0,
        threshold_limit: float = 3.0,
    ) -> pd.DataFrame:
        """
        Remove sources that are too bright, too faint, or have low SNR.

        Parameters
        ----------
        sources       : input catalog
        upperMaglimit : discard sources brighter than this
        lowerMaglimit : discard sources fainter than this
        threshold_limit: minimum detection threshold (S/N proxy)

        Returns
        -------
        cleaned_sources : DataFrame, or None on error
        """
        logger.info(log_step("Zeropoint: clean sequence stars"))

        try:
            filter_col = self.input_yaml.get("imageFilter")
            filter_col = self._normalize_filter(filter_col)
            
            # Filter catalog to only include sources with current image measurements
            # This ensures we don't use accumulated sequence catalog sources from multiple observations
            # Keep sources that have at least one of AP or PSF flux measurements
            if sources is not None and len(sources) > 0:
                has_ap = sources["flux_AP"].notna() if "flux_AP" in sources.columns else pd.Series(False, index=sources.index)
                has_psf = sources["flux_PSF"].notna() if "flux_PSF" in sources.columns else pd.Series(False, index=sources.index)
                has_any_flux = has_ap | has_psf
                n_before = len(sources)
                sources = sources[has_any_flux].copy()
                n_after = len(sources)
                if n_before - n_after > 0:
                    logger.info(
                        f"Filtered {n_before - n_after} sources with no flux measurements (AP or PSF); {n_after} remaining."
                    )
            
            if not filter_col:
                logger.warning(
                    "No input_yaml.imageFilter provided; skipping sequence-star clean filter."
                )
                return sources
            if sources is None:
                return None

            if filter_col not in sources.columns:
                logger.warning(
                    "Filter column '%s' not found in sources; skipping magnitude/colour clean.",
                    filter_col,
                )
                return sources

            # Newer SExtractor outputs use `snr` instead of `threshold`.
            if "threshold" not in sources.columns:
                if "snr" in sources.columns:
                    sources = sources.copy()
                    sources["threshold"] = sources["snr"]
                    logger.info(
                        "Using sources['snr'] as 'threshold' proxy for zeropoint cleaning."
                    )
                else:
                    # If we have neither threshold nor snr, skip threshold-based SNR cuts.
                    logger.warning(
                        "Missing both 'threshold' and 'snr' columns in sources; skipping threshold/SNR cleaning."
                    )
                    valid_mags = sources[filter_col].notna()
                    cleaned = sources.loc[valid_mags].copy()
                    logger.info("%d sources remaining after magnitude-only clean", len(cleaned))
                    return cleaned

            valid_mags = sources[filter_col].notna()
            n_missing = (~valid_mags).sum()
            if n_missing > 0:
                logger.info(f"Removing {n_missing} sources with missing {filter_col}")

            too_bright = sources[filter_col] < upperMaglimit
            too_faint = sources[filter_col] > lowerMaglimit
            n_brightness = (too_bright | too_faint).sum()
            if n_brightness > 0:
                logger.info(
                    "Removing %d sources outside magnitude range %.2f-%.2f mag",
                    n_brightness,
                    upperMaglimit,
                    lowerMaglimit,
                )

            low_snr = sources["threshold"] < threshold_limit
            n_snr = low_snr.sum()
            if n_snr > 0:
                logger.info(
                    "Removing %d sources with detection threshold < %.1f",
                    n_snr,
                    threshold_limit,
                )

            # Reject likely saturated/non-linear calibrators when peak flux is available.
            zp_cfg = self.input_yaml.get("zeropoint", {}) or {}
            reject_nonlinear = bool(zp_cfg.get("reject_nonlinear_sources", True))
            nonlin_peak_frac = float(zp_cfg.get("nonlinear_peak_frac", 0.85))
            sat_peak_frac = float(zp_cfg.get("saturation_peak_frac", 0.99))
            saturate_level = float(self.input_yaml.get("saturate", np.inf))
            non_linear_mask = np.zeros(len(sources), dtype=bool)
            saturated_mask = np.zeros(len(sources), dtype=bool)
            if (
                reject_nonlinear
                and "peak_flux" in sources.columns
                and np.isfinite(saturate_level)
                and saturate_level > 0
            ):
                peak_flux = np.asarray(sources["peak_flux"], float)
                finite_peak = np.isfinite(peak_flux)
                saturated_mask = finite_peak & (peak_flux >= sat_peak_frac * saturate_level)
                non_linear_mask = finite_peak & (
                    peak_flux >= nonlin_peak_frac * saturate_level
                )
                n_sat = int(np.count_nonzero(saturated_mask))
                n_nonlin = int(np.count_nonzero(non_linear_mask & ~saturated_mask))
                if n_sat > 0:
                    logger.info(
                        "Removing %d saturated zeropoint sources (peak_flux >= %.2f x saturate).",
                        n_sat,
                        sat_peak_frac,
                    )
                if n_nonlin > 0:
                    logger.info(
                        "Removing %d near non-linear zeropoint sources (peak_flux >= %.2f x saturate).",
                        n_nonlin,
                        nonlin_peak_frac,
                    )

            mask = (
                valid_mags
                & (sources[filter_col] >= upperMaglimit)
                & (sources[filter_col] <= lowerMaglimit)
                & (sources["threshold"] >= threshold_limit)
                & (~non_linear_mask)
            )
            cleaned = sources.loc[mask].copy()
            logger.info("%d sources remaining after quality cuts", len(cleaned))
            return cleaned

        except Exception as exc:
            logger.error(f"Error in clean(): {exc}\n{traceback.format_exc()}")
            return None

    # -----------------------------------------------------------------------
    # Public: simple sigma-clipped offset
    # -----------------------------------------------------------------------

    def get(
        self,
        sources: pd.DataFrame,
        useMean: bool = True,
        useMedian: bool = False,
        weighted_average: bool = False,
    ):
        """
        Measure the photometric zeropoint offset via sigma-clipped statistics.

        Returns
        -------
        (sources, output_zp) : (DataFrame, dict)
        """
        logger.info(log_step("Zeropoint: offset vs catalog"))

        methods = ["AP"] + (["PSF"] if "flux_PSF" in sources.columns else [])
        method_labels = {"AP": "Aperture", "PSF": "PSF"}
        output_zp = {}

        image_filter = self.input_yaml["imageFilter"]
        image_filter = self._normalize_filter(image_filter)
        
        mag_col = sources[image_filter]
        mag_err_col = sources[f"{image_filter}_err"]

        for method in methods:
            try:
                src = sources.copy()

                if method == "PSF":
                    if "flags" in src.columns:
                        before = len(src)
                        src = src[src["flags"] <= 0]
                        logger.info(f"Removed {before - len(src)} flagged sources")

                    if "qfit" in src.columns:
                        before = len(src)
                        qfit_clip = sigma_clip(
                            src["qfit"],
                            sigma=3,
                            sigma_lower=np.inf,
                            maxiters=5,
                            masked=True,
                            cenfunc=np.nanmedian,
                            stdfunc=mad_std,
                        )
                        src = src[~qfit_clip.mask]
                        logger.info(f"Removed {before - len(src)} qfit outliers")

                inst_mag = src[f"inst_{image_filter}_{method}"]
                snr = src["SNR"]
                error_snr = snr_err(snr)

                zp = (mag_col.loc[src.index] - inst_mag).values
                zp_err = np.sqrt(error_snr**2 + mag_err_col.loc[src.index].values ** 2)

                zp_col = f"zp_{image_filter}_{method}"
                zp_err_col = f"{zp_col}_err"
                src[zp_col] = zp
                src[zp_err_col] = zp_err

                # Build combined bad-value mask.
                mask = ~np.isfinite(zp)
                clip1 = sigma_clip(
                    mag_col.loc[src.index].values, sigma=5, masked=True, maxiters=10
                )
                mask |= clip1.mask
                clip2 = sigma_clip(
                    zp.values,
                    sigma=5,
                    masked=True,
                    maxiters=10,
                    cenfunc=np.nanmedian,
                    stdfunc=mad_std,
                )
                mask |= clip2.mask

                n_bad = int(mask.sum())
                if n_bad:
                    logger.info(
                        f"[{method_labels[method]}] Removing {n_bad} non-finite/outlier sources"
                    )

                valid = ~mask
                n_valid = int(np.sum(valid))
                if weighted_average:
                    image_zp, image_zp_err = self.weighted_average(
                        zp.values[valid], zp_err.values[valid]
                    )
                else:
                    image_zp, _, mad_val = sigma_clipped_stats(
                        zp.values[valid],
                        sigma=3.0,
                        cenfunc=np.nanmedian,
                        stdfunc=mad_std,
                    )
                    # SE(median) ~ 1.858 * MAD/sqrt(N); use MAD as fallback if N < 2
                    image_zp_err = (
                        (1.858 * mad_val / np.sqrt(n_valid))
                        if n_valid >= 2
                        else mad_val
                    )

                src[f"mask_{method}"] = mask
                sources.loc[src.index, src.columns] = src
                output_zp[method] = [image_zp, image_zp_err]
                logger.info(
                    f"[{method_labels[method]}] ZP = {image_zp:.3f} +/- {image_zp_err:.3f}"
                )

            except Exception as exc:
                logger.error(
                    f"Error in get() for {method}: {exc}\n{traceback.format_exc()}"
                )

        return sources, output_zp

    # -----------------------------------------------------------------------
    # RANSAC robust fit helper
    # -----------------------------------------------------------------------

    def _robust_RANSAC_fit(
        self,
        x_indep,
        delta_mag,
        w_mag,
        x_err=None,
        slope: float = 0.0,
        slope_err: float = 0.0,
        slope_tolerance: float = 0.5,
        penalty_weight: float = 100.0,
        max_trials: int = 1000,
        ransac_min_samples: int = 2,
        random_state: int = 42,
    ):
        """
        Fit delta_mag = ZP (constant) vs *x_indep* with RANSAC outlier rejection
        and an inverse-variance weighted mean on the inlier set.

        Parameters
        ----------
        x_indep   : independent variable (instrumental mag or colour)
        delta_mag : dependent variable  (m_ref - m_inst[+/-colour])
        w_mag     : inverse-variance weights
        x_err     : optional uncertainty on x_indep
        slope     : forced slope value (0 -> constant)
        slope_err : uncertainty on the slope (passed through to covariance)
        max_trials, ransac_min_samples, random_state : RANSAC settings

        Returns
        -------
        (ZP, slope, inlier_mask, cov) : (float, float, ndarray, ndarray)
        """
        logger.info("Starting RANSAC constant fit (robust errors).")

        x = np.asarray(x_indep, float)
        y = np.asarray(delta_mag, float)
        w0 = np.asarray(w_mag, float)
        x_err = np.zeros_like(x) if x_err is None else np.asarray(x_err, float)

        orig_size = len(x)

        # Quality filter: finite values, positive weights, error <= 0.5 mag.
        # NOTE: weights are still used internally for RANSAC conditioning, but we do not
        # propagate weighted errors as output uncertainties.
        mag_err = 1.0 / np.sqrt(np.clip(w0, 1e-12, None))
        finite = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(w0)
            & (w0 > 0)
            & np.isfinite(x_err)
            & (mag_err <= 0.5)
        )
        x, y, w0, x_err = x[finite], y[finite], w0[finite], x_err[finite]
        keep_idx = np.where(finite)[0]
        logger.info(f"Filtered {len(x)}/{orig_size} points.")

        # Trivial fallback for very sparse data (robust median and SE from MAD).
        if len(x) < 3:
            yv = y[np.isfinite(y)]
            ZP = float(np.nanmedian(yv)) if yv.size else np.nan
            mad0 = float(median_abs_deviation(yv, nan_policy="omit")) if yv.size else np.nan
            n0 = int(yv.size)
            zp_se = (1.858 * mad0 / np.sqrt(n0)) if n0 >= 2 else mad0
            full = np.zeros(orig_size, dtype=bool)
            full[keep_idx] = True
            return ZP, slope, full, np.diag([zp_se**2, slope_err**2])

        # RANSAC threshold from MAD of residuals (unweighted centre).
        r0 = y - np.nanmedian(y)
        mad = np.nanmedian(np.abs(r0)) * 1.4826 + 1e-12

        # Use smarter min_samples: at least 5 points or 30% of data, whichever is larger
        # This prevents overfitting with just 2 points
        n_points = len(x)
        smart_min_samples = max(5, min(n_points, int(0.3 * n_points)))
        effective_min_samples = max(ransac_min_samples, smart_min_samples)

        # Relax residual threshold: 0.15 mag cap instead of 0.1
        # This allows slightly more scatter while still rejecting true outliers
        residual_threshold = min(3.0 * mad, 0.15)

        ransac = RANSACRegressor(
            PenalisedSlopeRegressor(
                slope_constraint=0.0,
                slope_tolerance=slope_tolerance,
                penalty_weight=penalty_weight,
            ),
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            min_samples=effective_min_samples,
            random_state=random_state,
        )
        ransac.fit(x[:, None], y)
        inlier_mask = ransac.inlier_mask_
        logger.info(f"RANSAC: {inlier_mask.sum()}/{len(x)} inliers (threshold={residual_threshold:.3f}, min_samples={effective_min_samples})")

        # If RANSAC rejects too many points (>50%), fall back to all points
        # The fit may have larger scatter but uses more data
        if inlier_mask.sum() < 2:
            inlier_mask = np.ones(len(x), dtype=bool)
            logger.warning("RANSAC found too few inliers; using all points.")
        elif inlier_mask.sum() < 0.5 * n_points and n_points > 10:
            logger.warning(f"RANSAC rejected >50% of points ({inlier_mask.sum()}/{n_points}); consider checking data quality")

        xi, yi = x[inlier_mask], y[inlier_mask]
        yi = yi[np.isfinite(yi)]
        intercept = float(np.nanmedian(yi)) if yi.size else np.nan
        mad_i = float(median_abs_deviation(yi, nan_policy="omit")) if yi.size else np.nan
        n_i = int(yi.size)
        intercept_err = (1.858 * mad_i / np.sqrt(n_i)) if n_i >= 2 else mad_i

        cov = np.diag([intercept_err**2, slope_err**2])
        full = np.zeros(orig_size, dtype=bool)
        full[keep_idx[inlier_mask]] = True

        logger.info(f"ZP = {intercept:.4f} +/- {intercept_err:.4f}")
        return intercept, slope, full, cov

    # -----------------------------------------------------------------------
    # Public: RANSAC-based zeropoint fit
    # -----------------------------------------------------------------------

    def fit_zeropoint(
        self,
        catalog: pd.DataFrame,
        threshold: float = 5.0,
        max_trials: int = 4000,
        ransac_min_samples: int = 2,
        n_jobs: int | None = 1,
        random_state: int = 42,
        min_sources: int = 1,
        fixed_color_coeffs = None,
        fixed_color_coeff_errors = None,
        fit_mode="polynomial",
        n_segments=1,
    ):
        """
        Fit ZP = m_cat - m_inst[+/-c*(c1-c2)] vs m_inst via RANSAC.
        Supports linear, quadratic, and piecewise linear color terms.

        Parameters
        ----------
        fixed_color_coeffs : tuple or None
            For linear: (intercept, slope)
            For quadratic: (intercept, slope, quad_coeff)
            For piecewise (n=2): ((breakpoint,), (slope1, slope2), intercept)
        fixed_color_coeff_errors : tuple or None
            Corresponding errors for the coefficients
        fit_mode : str
            "polynomial" or "piecewise"
        n_segments : int
            Number of segments for piecewise fitting

        Returns
        -------
        (clean_catalog, fit_params) : (DataFrame, dict)
        """
        t0 = time.time()
        fit_params = {"AP": {}, "PSF": {}}
        logger.info("Fitting zeropoints vs m_inst.")

        try:
            fpath = self.input_yaml.get("fpath", "")
            base_name = os.path.splitext(os.path.basename(fpath))[0] or "zeropoint"
            write_dir = os.path.dirname(fpath) or "."
            use_filter = self.input_yaml.get("imageFilter")
            use_filter = self._normalize_filter(use_filter)

            if not use_filter:
                raise ValueError("Missing 'imageFilter' in input YAML.")

            required = [
                "flux_AP",
                "flux_AP_err",
                "flux_PSF",
                "flux_PSF_err",
                use_filter,
                f"{use_filter}_err",
            ]
            missing = [c for c in required if c not in catalog.columns]
            if missing:
                raise KeyError(f"Missing columns: {missing}")

            try:
                color1, color2 = self.get_color_term_for_filter(use_filter)
                has_color_term = (color1 in catalog.columns) and (
                    color2 in catalog.columns
                )
            except Exception:
                has_color_term, color1, color2 = False, None, None

            clean_catalog = self._prepare_catalog(
                catalog, threshold, use_filter, min_sources
            )
            if clean_catalog is None:
                fit_params = self._fallback_zeropoint(catalog, use_filter)
                return catalog, fit_params

            from plotting_utils import apply_autophot_mplstyle, ransac_legend_top_outside, set_mag_axes_inverted_xy

            apply_autophot_mplstyle()
            fig, ax = plt.subplots(1, 1, figsize=set_size(540, 1))
            inlier_masks_full = {
                k: np.zeros(len(clean_catalog), dtype=bool) for k in ["AP", "PSF"]
            }
            colors = {"AP": get_color('inliers'), "PSF": get_color('robust')}
            labels = {"AP": "Aperture", "PSF": "PSF"}
            global_xmins, global_xmaxs, global_ymins, global_ymaxs = [], [], [], []

            for flux_type in ["AP", "PSF"]:
                pack = self._finite_vmask(clean_catalog, flux_type, use_filter)
                if pack is None:
                    continue
                flux, flux_err, catmag_v, catmag_err_v, vmask = pack

                inst_mag, inst_mag_err, delta_mag, delta_mag_err = (
                    self._compute_delta_mag(flux, flux_err, catmag_v, catmag_err_v)
                )

                # Colour correction with full error propagation (catalog + colour-term slope).
                color_corr_err = np.zeros_like(delta_mag)
                if has_color_term and fixed_color_coeffs is not None:
                    delta_mag, color_corr_err = self._apply_color_correction(
                        delta_mag,
                        clean_catalog,
                        vmask,
                        color1,
                        color2,
                        fixed_color_coeffs,
                        fixed_color_coeff_errors,
                        fit_mode=fit_mode,
                        n_segments=n_segments,
                    )

                # RANSAC still uses weights for outlier rejection conditioning, but
                # we do not propagate inverse-variance weighting into reported errors.
                yerr = np.sqrt(delta_mag_err**2 + color_corr_err**2)
                weights = 1.0 / (yerr**2 + 1e-12)

                ZP, _, inlier_short, cov = self._robust_RANSAC_fit(
                    inst_mag,
                    delta_mag,
                    weights,
                    x_err=inst_mag_err,
                    max_trials=max_trials,
                    ransac_min_samples=ransac_min_samples,
                    random_state=random_state,
                )
                zp_std = float(np.sqrt(np.clip(cov[0, 0], 0, np.inf)))

                fit_params[flux_type].update(
                    {
                        "zeropoint": ZP,
                        "zeropoint_error": zp_std,
                        "has_color_term": bool(
                            has_color_term and fixed_color_coeffs is not None
                        ),
                    }
                )
                if has_color_term and fixed_color_coeffs is not None:
                    # Extract slope for backwards compatibility
                    slope_for_params = fixed_color_coeffs[1] if len(fixed_color_coeffs) >= 2 else 0.0
                    slope_err_for_params = fixed_color_coeff_errors[1] if fixed_color_coeff_errors is not None and len(fixed_color_coeff_errors) >= 2 else 0.0
                    fit_params[flux_type].update(
                        {
                            "color_term": slope_for_params,
                            "color_term_error": slope_err_for_params,
                            "color1": color1,
                            "color2": color2,
                        }
                    )

                msg = f"[{flux_type}] ZP={ZP:.3f} +/- {zp_std:.3f}"
                logger.info(msg)

                # ---- Plot: m_inst vs m_cal = m_inst + dmag (slope-1 locus y = x + ZP) ----
                m_cal = inst_mag + delta_mag
                m_cal_err = yerr
                in_x = inst_mag[inlier_short]
                m_cal_in = m_cal[inlier_short]
                in_e = m_cal_err[inlier_short]

                ax.errorbar(
                    in_x,
                    m_cal_in,
                    xerr=inst_mag_err[inlier_short],
                    yerr=in_e,
                    fmt="o",
                    ms=get_marker_size("medium"),
                    color=colors[flux_type],
                    ecolor="lightgrey",
                    alpha=0.75,
                    capsize=1.5,
                    label=f"{labels[flux_type]} inliers",
                )
                out_mask = ~inlier_short
                if out_mask.any():
                    ax.errorbar(
                        inst_mag[out_mask],
                        m_cal[out_mask],
                        xerr=inst_mag_err[out_mask],
                        yerr=m_cal_err[out_mask],
                        fmt="x",
                        ms=get_marker_size("medium"),
                        color=colors[flux_type],
                        ecolor=colors[flux_type],
                        alpha=0.5,
                        capsize=0,
                        elinewidth=0.4,
                    )

                xs = np.linspace(in_x.min() - 0.5, in_x.max() + 0.5, 200)
                ys1 = xs + ZP
                ax.plot(
                    xs,
                    ys1,
                    "--",
                    color=colors[flux_type],
                    lw=get_line_width("medium"),
                    label=msg + "  (m_cal = m_inst + ZP, slope=1)",
                )
                ax.fill_between(
                    xs,
                    xs + ZP - zp_std,
                    xs + ZP + zp_std,
                    color=colors[flux_type],
                    alpha=get_alpha("light"),
                )

                global_xmins.append(xs[0])
                global_xmaxs.append(xs[-1])
                _em = float(np.nanmean(in_e)) if np.size(in_e) else 0.0
                global_ymins.append(
                    min(
                        float(m_cal_in.min()) - _em,
                        float(np.min(ys1)) - 2.0 * float(zp_std),
                    )
                )
                global_ymaxs.append(
                    max(
                        float(m_cal_in.max()) + _em,
                        float(np.max(ys1)) + 2.0 * float(zp_std),
                    )
                )

                full_mask = np.zeros(len(clean_catalog), dtype=bool)
                full_mask[np.flatnonzero(vmask)] = inlier_short
                inlier_masks_full[flux_type] = full_mask

            # Axis limits.
            if global_xmins:
                xr = max(global_xmaxs) - min(global_xmins)
                yr = max(global_ymaxs) - min(global_ymins)
                ax.set_xlim(
                    min(global_xmins) - 0.05 * xr, max(global_xmaxs) + 0.05 * xr
                )
                ax.set_ylim(min(global_ymins) - 0.1 * yr, max(global_ymaxs) + 0.1 * yr)

            y_label = rf"Catalog $m_{{\mathrm{{cal,{use_filter}}}}}$ [mag]"
            if has_color_term and fixed_color_coeffs is not None:
                slope_for_display = (
                    fixed_color_coeffs[1] if len(fixed_color_coeffs) >= 2 else 0.0
                )
                sign = r" + " if slope_for_display <= 0 else r" - "
                y_label += (
                    rf" (after colour term{sign}{abs(slope_for_display):.2f} "
                    rf"$(m_{{\mathrm{{cal,{color1}}}}} - m_{{\mathrm{{cal,{color2}}}}})$)"
                )
            ax.set_xlabel(rf"Instrumental $m_{{\mathrm{{inst,{use_filter}}}}}$ [mag]")
            ax.set_ylabel(y_label)
            ax.grid(True, ls="--", alpha=0.5)
            ransac_legend_top_outside(ax, ncol=2)
            set_mag_axes_inverted_xy(ax)

            fig.savefig(
                os.path.join(write_dir, f"Zeropoint_{base_name}.png"),
                bbox_inches="tight",
                dpi=150,
                facecolor="white",
            )
            plt.close(fig)

            # Build joint inlier mask only from flux types that actually
            # contributed inliers, so that missing PSF or AP measurements
            # do not zero-out the catalog.
            valid_masks = [m for m in inlier_masks_full.values() if np.any(m)]
            if valid_masks:
                joint_mask = np.logical_and.reduce(valid_masks)
                clean_catalog = clean_catalog[joint_mask]
            else:
                logger.warning(
                    "fit_zeropoint: no inliers from any flux type; "
                    "returning unfiltered clean_catalog."
                )

            logger.info(f"[fit_zeropoint] Done in {time.time() - t0:.3f}s")
            return clean_catalog, fit_params

        except Exception as exc:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(
                f"fit_zeropoint failed: {exc_type.__name__} in "
                f"{fname}:{exc_tb.tb_lineno}: {exc}"
            )
            return catalog, fit_params

    # -----------------------------------------------------------------------
    # Public: sigma-clip histogram estimate
    # -----------------------------------------------------------------------

    def estimate_zeropoint(
        self,
        catalog: pd.DataFrame,
        threshold: float = 5.0,
        sigma_clip_sigma: float = 3.5,
        sigma_clip_maxiters: int = 10,
        min_sources: int = 1,
        fixed_color_coeffs = None,
        fixed_color_coeff_errors = None,
        fit_mode="polynomial",
        n_segments=1,
    ):
        """
        Estimate ZP via sigma-clipped median of (m_ref - m_inst).
        Produces a combined histogram PDF for AP and PSF.
        Supports linear, quadratic, and piecewise linear color terms.

        Parameters
        ----------
        fixed_color_coeffs : tuple or None
            For linear: (intercept, slope)
            For quadratic: (intercept, slope, quad_coeff)
            For piecewise (n=2): ((breakpoint,), (slope1, slope2), intercept)
        fixed_color_coeff_errors : tuple or None
            Corresponding errors for the coefficients
        fit_mode : str
            "polynomial" or "piecewise"
        n_segments : int
            Number of segments for piecewise fitting

        Returns
        -------
        (clean_catalog, zp_params) : (DataFrame, dict)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            t0 = time.time()
            zp_params = {"AP": {}, "PSF": {}}
            logger.info("Estimating zeropoint via sigma clipping.")

            try:
                if catalog is None or getattr(catalog, "empty", False) or len(catalog) == 0:
                    logger.warning(
                        "estimate_zeropoint: catalog is None/empty; returning NaN zeropoint."
                    )
                    return catalog, {
                        "AP": {"zeropoint": np.nan, "zeropoint_error": np.nan, "has_color_term": False},
                        "PSF": {"zeropoint": np.nan, "zeropoint_error": np.nan, "has_color_term": False},
                    }

                fpath = self.input_yaml.get("fpath", "")
                base_name = os.path.splitext(os.path.basename(fpath))[0] or "zeropoint"
                write_dir = os.path.dirname(fpath) or "."
                use_filter = self.input_yaml.get("imageFilter")
                use_filter = self._normalize_filter(use_filter)
                if not use_filter:
                    raise ValueError("Missing 'imageFilter' in input YAML.")

                required = [
                    "flux_AP",
                    "flux_AP_err",
                    "flux_PSF",
                    "flux_PSF_err",
                    use_filter,
                    f"{use_filter}_err",
                ]
                missing = [c for c in required if c not in catalog.columns]
                if missing:
                    # If filter columns are missing, go directly to fallback
                    zp_params = self._fallback_zeropoint(catalog, use_filter)
                    return catalog, zp_params

                try:
                    color1, color2 = self.get_color_term_for_filter(use_filter)
                    has_color_term = (color1 in catalog.columns) and (
                        color2 in catalog.columns
                    )
                except Exception:
                    has_color_term, color1, color2 = False, None, None

                clean_catalog = self._prepare_catalog(
                    catalog, threshold, use_filter, min_sources
                )
                if clean_catalog is None:
                    zp_params = self._fallback_zeropoint(catalog, use_filter)
                    return catalog, zp_params

                _style = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
                )
                if os.path.exists(_style):
                    plt.style.use(_style)
                fig_hist, ax_hist = plt.subplots(1, 1, figsize=set_size(540, 1))
                colors = {"AP": get_color('inliers'), "PSF": get_color('robust')}
                labels_base = {"AP": "Aperture", "PSF": "PSF"}
                inlier_masks_full = {
                    k: np.zeros(len(clean_catalog), dtype=bool) for k in ["AP", "PSF"]
                }
                # Track global x-range for plot padding
                x_min = np.inf
                x_max = -np.inf
                # Track global y-range so top markers don't overlap histograms
                y_max = 0.0

                # Aperture correction is applied at photometry stage (main.py), not during ZP calculation.
                # The following is for visualization only: show what the ZP would be with aperture correction.
                ap_corr_mag = float(self.input_yaml.get("aperture_correction", 0.0) or 0.0)
                ap_corr_err_mag = float(
                    self.input_yaml.get("aperture_correction_err", 0.0) or 0.0
                )
                ap_zp_raw = np.nan

                for flux_type in ["AP", "PSF"]:
                    pack = self._finite_vmask(clean_catalog, flux_type, use_filter)
                    if pack is None:
                        zp_params[flux_type] = {
                            "zeropoint": np.nan,
                            "zeropoint_error": np.nan,
                            "has_color_term": False,
                        }
                        continue

                    flux, flux_err, catmag_v, catmag_err_v, vmask = pack
                    _, _, delta_mag, delta_mag_err = self._compute_delta_mag(
                        flux, flux_err, catmag_v, catmag_err_v
                    )

                    delta_no_corr = delta_mag.copy()  # before colour correction
                    delta_no_corr_err = delta_mag_err.copy()

                    if has_color_term and fixed_color_coeffs is not None:
                        # Log statistics before color correction
                        zp_no_corr = float(np.nanmedian(delta_no_corr[np.isfinite(delta_no_corr)]))
                        std_no_corr = float(
                            median_abs_deviation(delta_no_corr[np.isfinite(delta_no_corr)], nan_policy="omit")
                        )
                        # Extract slope for logging (second element in tuple)
                        if fit_mode == "piecewise" and n_segments == 2:
                            slope_for_log = fixed_color_coeffs[1][0]  # First slope
                        else:
                            slope_for_log = fixed_color_coeffs[1] if len(fixed_color_coeffs) >= 2 else 0.0
                        logger.info(
                            f"[{flux_type}] Before color correction: ZP={zp_no_corr:.3f}, "
                            f"std={std_no_corr:.3f}, color_term={slope_for_log:.4f}"
                        )

                        delta_mag, color_corr_err = self._apply_color_correction(
                            delta_mag,
                            clean_catalog,
                            vmask,
                            color1,
                            color2,
                            fixed_color_coeffs,
                            fixed_color_coeff_errors,
                            fit_mode=fit_mode,
                            n_segments=n_segments,
                        )
                        # Propagate colour-correction uncertainty into delta_mag_err.
                        delta_mag_err = np.sqrt(delta_mag_err**2 + color_corr_err**2)

                    finite_mask = np.isfinite(delta_mag) & np.isfinite(delta_mag_err)
                    delta_mag = delta_mag[finite_mask]
                    delta_mag_err = delta_mag_err[finite_mask]
                    delta_no_corr = delta_no_corr[finite_mask]
                    delta_no_corr_err = delta_no_corr_err[finite_mask]
                    # Keep the full finite distribution (pre sigma-clip) for plotting.
                    delta_mag_full = delta_mag.copy()

                    # Track which vmask sources survive finite filtering
                    vmask_finite_idx = np.flatnonzero(vmask)[finite_mask]
                    if len(delta_mag) == 0:
                        logger.warning(
                            f"{flux_type}: no finite delta_mag after masking; skipping."
                        )
                        zp_params[flux_type] = {
                            "zeropoint": np.nan,
                            "zeropoint_error": np.nan,
                            "has_color_term": False,
                        }
                        continue

                    clipped = sigma_clip(
                        delta_mag,
                        sigma=sigma_clip_sigma,
                        maxiters=sigma_clip_maxiters,
                        cenfunc=np.nanmedian,
                        stdfunc=mad_std,
                    )
                    inlier_mask = ~clipped.mask
                    inlier_deltas = clipped.data[inlier_mask]
                    inlier_delta_err = delta_mag_err[inlier_mask]

                    # Track which catalog sources survive sigma clipping
                    vmask_sigma_idx = vmask_finite_idx[inlier_mask]

                    finite2 = np.isfinite(inlier_deltas) & np.isfinite(inlier_delta_err)
                    inlier_deltas = inlier_deltas[finite2]
                    inlier_delta_err = inlier_delta_err[finite2]
                    # Update to account for finite2 filtering
                    vmask_sigma_idx = vmask_sigma_idx[finite2]

                    if len(inlier_deltas) == 0:
                        logger.warning(
                            f"{flux_type}: no inliers after sigma clipping; skipping."
                        )
                        zp_params[flux_type] = {
                            "zeropoint": np.nan,
                            "zeropoint_error": np.nan,
                            "has_color_term": False,
                        }
                        continue

                    # Robust central value from the median (stable against
                    # residual outliers and imperfect error modelling).
                    zp_final = float(np.nanmedian(inlier_deltas))
                    if flux_type == "AP":
                        ap_zp_raw = zp_final

                    n_inl = len(inlier_deltas)
                    mad_zp = float(
                        median_abs_deviation(inlier_deltas, nan_policy="omit")
                    )
                    # SE(median) ~ 1.858 * MAD/sqrt(N) (1.253*sigma/sqrt(N), sigma~1.4826*MAD)
                    zp_std = (1.858 * mad_zp / np.sqrt(n_inl)) if n_inl >= 2 else mad_zp

                    # Zeropoint uncertainty: always use empirical scatter (SE of median).
                    # Per your convention, we do not use inverse-variance weighting for errors.
                    zp_err = float(zp_std)

                    zp_params[flux_type].update(
                        {
                            "zeropoint": zp_final,
                            "zeropoint_error": zp_err,
                            "has_color_term": bool(
                                has_color_term and fixed_color_coeffs is not None
                            ),
                        }
                    )
                    if has_color_term and fixed_color_coeffs is not None:
                        # Extract slope for backwards compatibility
                        slope_for_params = fixed_color_coeffs[1] if len(fixed_color_coeffs) >= 2 else 0.0
                        slope_err_for_params = fixed_color_coeff_errors[1] if fixed_color_coeff_errors is not None and len(fixed_color_coeff_errors) >= 2 else 0.0
                        zp_params[flux_type].update(
                            {
                                "color_term": slope_for_params,
                                "color_term_error": slope_err_for_params,
                                "color1": color1,
                                "color2": color2,
                            }
                        )

                    logger.info(
                        "[%s] Zeropoint: %.3f +/- %.3f mag",
                        flux_type,
                        zp_final,
                        zp_err,
                    )

                    # ---- Histogram (colour-corrected) ----------------------
                    bin_edges = np.histogram_bin_edges(inlier_deltas, bins="fd")
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    width = bin_edges[1] - bin_edges[0]
                    counts, _ = np.histogram(
                        inlier_deltas, bins=bin_edges, density=True
                    )
                    try:
                        y_max = max(float(y_max), float(np.nanmax(counts)))
                    except Exception:
                        pass

                    # n_inliers = number of unique catalog sources surviving all cuts
                    # vmask_sigma_idx contains the catalog indices of sources that survive all filtering steps
                    n_sources_used = len(vmask_sigma_idx)

                    ax_hist.bar(
                        bin_centers,
                        counts,
                        width=width,
                        color=colors[flux_type],
                        edgecolor="none",
                        alpha=0.8,
                        zorder=4,
                        label=(
                            f"{labels_base[flux_type]} (color corr., "
                            f"N={n_sources_used}, "
                            f"ZP={zp_final:.3f}+/-{zp_err:.3f})"
                        ),
                    )
                    # Errorbar marker at top of plot (xerr = zeropoint_error).
                    # Use a blended transform: x in data units, y in axes fraction.
                    try:
                        y_ax = 0.95
                        if np.isfinite(zp_final) and np.isfinite(zp_err) and zp_err > 0:
                            ax_hist.errorbar(
                                [zp_final],
                                [y_ax],
                                xerr=[zp_err],
                                fmt="o",
                                markersize=3,
                                color=colors[flux_type],
                                elinewidth=1.0,
                                capsize=2,
                                alpha=0.95,
                                zorder=20,
                                transform=ax_hist.get_xaxis_transform(),
                                clip_on=False,
                            )
                    except Exception:
                        pass

                    # Update global x-range for padding (use the actual inlier deltas)
                    try:
                        d_lo = float(np.nanmin(inlier_deltas))
                        d_hi = float(np.nanmax(inlier_deltas))
                        if np.isfinite(d_lo) and np.isfinite(d_hi):
                            x_min = min(x_min, d_lo)
                            x_max = max(x_max, d_hi)
                    except Exception:
                        pass

                    # Optional: also show the aperture-corrected AP zeropoint distribution as
                    # a separate (step) histogram for visualization when an aperture correction
                    # was measured.
                    if (
                        flux_type == "AP"
                        and np.isfinite(ap_corr_mag)
                        and ap_corr_mag != 0.0
                    ):
                        try:
                            inlier_apcorr = inlier_deltas - float(ap_corr_mag)
                            # Use the same bin width as the inlier histogram, shifted in x.
                            be_apcorr = bin_edges - float(ap_corr_mag)
                            ct_apcorr, _ = np.histogram(
                                inlier_apcorr, bins=be_apcorr, density=True
                            )
                            bc_apcorr = (be_apcorr[:-1] + be_apcorr[1:]) / 2.0
                            try:
                                y_max = max(float(y_max), float(np.nanmax(ct_apcorr)))
                            except Exception:
                                pass
                            # Propagate aperture-correction uncertainty into the error reported for the
                            # corrected zeropoint marker/legend.
                            xerr_corr = float(zp_err)
                            if np.isfinite(ap_corr_err_mag) and ap_corr_err_mag > 0:
                                xerr_corr = float(
                                    np.sqrt(float(zp_err) ** 2 + float(ap_corr_err_mag) ** 2)
                                )
                            ax_hist.hist(
                                inlier_apcorr,
                                bins=be_apcorr,
                                density=True,
                                histtype="step",
                                linestyle="--",
                                linewidth=1.4,
                                color=colors["AP"],
                                alpha=0.85,
                                zorder=5,
                                label=(
                                    f"{labels_base['AP']} (+ap corr., "
                                    f"N={n_sources_used}, "
                                    f"ZP={(zp_final - ap_corr_mag):.3f}+/-{xerr_corr:.3f})"
                                ),
                            )
                            # Errorbar marker at top of plot for the aperture-corrected distribution.
                            try:
                                x_corr = float(zp_final - ap_corr_mag)
                                y_ax = 0.95
                                if np.isfinite(x_corr) and np.isfinite(xerr_corr) and xerr_corr > 0:
                                    ax_hist.errorbar(
                                        [x_corr],
                                        [y_ax],
                                        xerr=[xerr_corr],
                                        fmt="o",
                                        markersize=3,
                                        color=colors["AP"],
                                        elinewidth=1.0,
                                        capsize=2,
                                        alpha=0.95,
                                        zorder=20,
                                        transform=ax_hist.get_xaxis_transform(),
                                        clip_on=False,
                                    )
                            except Exception:
                                pass
                            d_lo = float(np.nanmin(inlier_apcorr))
                            d_hi = float(np.nanmax(inlier_apcorr))
                            if np.isfinite(d_lo) and np.isfinite(d_hi):
                                x_min = min(x_min, d_lo)
                                x_max = max(x_max, d_hi)
                        except Exception:
                            pass

                    # ---- Histogram (without colour correction) -------------
                    if has_color_term and fixed_color_coeffs is not None:
                        dnc = delta_no_corr[np.isfinite(delta_no_corr)]
                        # Track which vmask_finite_idx sources are finite in delta_no_corr
                        vmask_nc_finite_idx = vmask_finite_idx[np.isfinite(delta_no_corr)]

                        clipped_nc = sigma_clip(
                            dnc,
                            sigma=sigma_clip_sigma,
                            maxiters=sigma_clip_maxiters,
                            cenfunc=np.nanmedian,
                            stdfunc=mad_std,
                        )
                        inl_nc = clipped_nc.data[~clipped_nc.mask]
                        inl_nc = inl_nc[np.isfinite(inl_nc)]

                        # Track which catalog sources survive sigma clipping
                        vmask_nc_sigma_idx = vmask_nc_finite_idx[~clipped_nc.mask]
                        # Update to account for final finite filter
                        vmask_nc_sigma_idx = vmask_nc_sigma_idx[np.isfinite(inl_nc)]
                        n_sources_nc = len(vmask_nc_sigma_idx)

                        if len(inl_nc) > 0:
                            be_nc = np.histogram_bin_edges(inl_nc, bins="fd")
                            bc_nc = (be_nc[:-1] + be_nc[1:]) / 2
                            w_nc = be_nc[1] - be_nc[0]
                            ct_nc, _ = np.histogram(inl_nc, bins=be_nc, density=True)
                            zp_nc = float(np.nanmedian(inl_nc))
                            std_nc = float(
                                median_abs_deviation(inl_nc, nan_policy="omit")
                            )
                            try:
                                y_max = max(float(y_max), float(np.nanmax(ct_nc)))
                            except Exception:
                                pass

                            ax_hist.bar(
                                bc_nc,
                                ct_nc,
                                width=w_nc,
                                color="none",
                                edgecolor=colors[flux_type],
                                linewidth=0.5,
                                alpha=0.6,
                                zorder=3,
                                label=(
                                    f"{labels_base[flux_type]} (no corr., "
                                    f"N={n_sources_nc}, ZP={zp_nc:.3f}+/-{std_nc:.3f})"
                                ),
                            )
                            try:
                                d_lo = float(np.nanmin(inl_nc))
                                d_hi = float(np.nanmax(inl_nc))
                                if np.isfinite(d_lo) and np.isfinite(d_hi):
                                    x_min = min(x_min, d_lo)
                                    x_max = max(x_max, d_hi)
                            except Exception:
                                pass

                    # Update inlier mask.
                    full_mask = np.zeros(len(clean_catalog), dtype=bool)
                    full_mask[vmask_sigma_idx] = True
                    inlier_masks_full[flux_type] = full_mask

                # Ensure histogram does not fill the full plot: pad xlim by at least 0.1 mag.
                if np.isfinite(x_min) and np.isfinite(x_max):
                    pad = 0.1
                    ax_hist.set_xlim(x_min - pad, x_max + pad)
                # Add headroom so top markers (y=0.95 axes fraction) sit above distributions.
                if np.isfinite(y_max) and y_max > 0:
                    ax_hist.set_ylim(0.0, float(y_max) / 0.9)

                ax_hist.set_xlabel("Zeropoint [mag]")
                ax_hist.set_ylabel("Density")
                ax_hist.grid(True, ls="-", alpha=0.25, zorder=0)
                ax_hist.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.0),
                    frameon=False,
                    ncol=2,
                )
                for patch in ax_hist.patches:
                    patch.set_zorder(3)

                fig_hist.tight_layout()
                os.makedirs(write_dir, exist_ok=True)
                fig_hist.savefig(
                    os.path.join(
                        write_dir, f"Zeropoint_Hist_Combined_{base_name}.png"
                    ),
                    bbox_inches="tight",
                    dpi=150,
                    facecolor="white",
                )
                plt.close(fig_hist)

                # Combine inliers only over flux types that actually had
                # valid measurements, to avoid emptying the catalog when
                # only AP or PSF is present.
                valid_masks = [m for m in inlier_masks_full.values() if np.any(m)]
                if valid_masks:
                    joint_mask = np.logical_and.reduce(valid_masks)
                    clean_catalog = clean_catalog[joint_mask]
                else:
                    logger.warning(
                        "No inliers from any flux type; returning unfiltered clean_catalog."
                    )

                return clean_catalog, zp_params

            except Exception as exc:
                exc_type, _, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(
                    f"estimate_zeropoint failed: {exc_type.__name__} in "
                    f"{fname}:{exc_tb.tb_lineno}: {exc}"
                )
                zp_params = self._fallback_zeropoint(
                    catalog, self.input_yaml.get("imageFilter")
                )
                return catalog, zp_params

    # -----------------------------------------------------------------------
    # Public: colour-term fit
    # -----------------------------------------------------------------------

    def _plot_piecewise_color_term(self, xi, yi, xe, ye, coefficients, coefficient_errors, n_segments, color1, color2, use_filter, inlier_mask=None, overall_method="RANSAC", output_dir=None):
        """Generate color term plot for piecewise linear fitting."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from plotting_utils import get_color, get_alpha
        import os

        _style = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
        )
        if os.path.exists(_style):
            plt.style.use(_style)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(340, 2.0), sharex=True)
        plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.1, left=0.15, right=0.95)

        # Consistent colors from plotting_utils palette
        inlier_color = get_color('inliers')
        outlier_color = get_color('outliers')

        # Top panel: uncorrected data
        if inlier_mask is not None:
            # Plot inliers
            ax1.errorbar(
                xi[inlier_mask],
                yi[inlier_mask],
                xerr=xe[inlier_mask],
                yerr=ye[inlier_mask],
                fmt="o",
                ms=get_marker_size('medium'),
                color=inlier_color,
                ecolor="lightgrey",
                alpha=0.8,
                capsize=2,
                lw=0.5,
                label="Inliers",
            )
        else:
            ax1.errorbar(
                xi,
                yi,
                xerr=xe,
                yerr=ye,
                fmt="o",
                ms=get_marker_size('medium'),
                color=inlier_color,
                ecolor="lightgrey",
                alpha=0.8,
                capsize=2,
                lw=0.5,
                label="Data",
            )

        # Plot piecewise linear fit
        x_plot = np.linspace(xi.min() * 0.95, xi.max() * 1.05, 200)
        if n_segments == 2:
            breakpoints, slopes, intercept = coefficients
            bp = breakpoints[0]
            slope1, slope2 = slopes

            # Get coefficient errors for shading
            if coefficient_errors is not None and len(coefficient_errors) == 3:
                bp_err = coefficient_errors[0][0] if coefficient_errors[0] else 0.0
                slope1_err = coefficient_errors[1][0] if len(coefficient_errors[1]) > 0 else 0.0
                slope2_err = coefficient_errors[1][1] if len(coefficient_errors[1]) > 1 else 0.0
                intercept_err = coefficient_errors[2] if coefficient_errors[2] else 0.0
            else:
                bp_err = slope1_err = slope2_err = intercept_err = 0.0

            # Plot two line segments
            y_plot = np.zeros_like(x_plot)
            y_plot_upper = np.zeros_like(x_plot)
            y_plot_lower = np.zeros_like(x_plot)
            mask1 = x_plot <= bp
            mask2 = x_plot > bp
            y_plot[mask1] = intercept + slope1 * x_plot[mask1]
            y_plot[mask2] = (intercept + slope1 * bp) + slope2 * (x_plot[mask2] - bp)

            # Calculate error bands (propagate slope and intercept errors)
            y_err_upper = np.zeros_like(x_plot)
            y_err_lower = np.zeros_like(x_plot)
            y_err_upper[mask1] = intercept_err + slope1_err * x_plot[mask1]
            y_err_lower[mask1] = intercept_err + slope1_err * x_plot[mask1]
            y_err_upper[mask2] = (intercept_err + slope1_err * bp) + slope2_err * (x_plot[mask2] - bp)
            y_err_lower[mask2] = (intercept_err + slope1_err * bp) + slope2_err * (x_plot[mask2] - bp)

            y_plot_upper = y_plot + np.abs(y_err_upper)
            y_plot_lower = y_plot - np.abs(y_err_lower)

            label_text = f"Piecewise {overall_method}: bp={bp:.3f}, slope1={slope1:.3f}, slope2={slope2:.3f}"

            ax1.plot(
                x_plot,
                y_plot,
                color=get_color('fit'),
                linestyle="--",
                lw=1.0,
                label=label_text,
            )
            # Add error shading
            ax1.fill_between(
                x_plot,
                y_plot_lower,
                y_plot_upper,
                color=get_color('error_region'),
                alpha=get_alpha('light'),
                label="Error band",
            )
            ax1.axvline(bp, color="gray", linestyle=":", alpha=0.5, label=f"Breakpoint: {bp:.3f}")

        ax1.set_xlim(xi.min() - 0.1 * np.ptp(xi), xi.max() + 0.1 * np.ptp(xi))
        ax1.set_ylim(yi.min() - 0.1 * np.ptp(yi), yi.max() + 0.1 * np.ptp(yi))
        ax1.set_ylabel(
            rf"$m_\mathrm{{cal,{color1}}} - m_\mathrm{{inst,{use_filter}}}$ [mag]"
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8, ncol=2)

        # Bottom panel: color corrected data
        if n_segments == 2:
            breakpoints, slopes, intercept = coefficients
            bp = breakpoints[0]
            slope1, slope2 = slopes

            yi_corrected = np.zeros_like(yi)
            ye_corrected = np.zeros_like(ye)

            mask1 = xi <= bp
            mask2 = xi > bp

            # Segment 1 correction
            yi_corrected[mask1] = yi[mask1] - slope1 * xi[mask1]
            ye_corrected[mask1] = np.sqrt(ye[mask1]**2 + (slope1 * xe[mask1])**2)

            # Segment 2 correction (account for continuity)
            yi_corrected[mask2] = yi[mask2] - (slope2 * xi[mask2] + (slope1 - slope2) * bp)
            ye_corrected[mask2] = np.sqrt(ye[mask2]**2 + (slope2 * xe[mask2])**2)

        std_uncorrected = float(median_abs_deviation(yi, nan_policy="omit"))
        std_corrected = float(median_abs_deviation(yi_corrected, nan_policy="omit"))

        if inlier_mask is not None:
            # Plot corrected inliers
            ax2.errorbar(
                xi[inlier_mask],
                yi_corrected[inlier_mask],
                xerr=xe[inlier_mask],
                yerr=ye_corrected[inlier_mask],
                fmt="o",
                ms=get_marker_size('medium'),
                color=get_color('robust'),
                ecolor="lightgrey",
                alpha=0.8,
                capsize=2,
                label=f"Corrected inliers [{np.sum(inlier_mask)}]",
            )
            # Plot outliers (corrected) if any
            outlier_mask = ~inlier_mask
            if np.any(outlier_mask):
                ax2.errorbar(
                    xi[outlier_mask],
                    yi_corrected[outlier_mask],
                    xerr=xe[outlier_mask],
                    yerr=ye_corrected[outlier_mask],
                    fmt="x",
                    ms=get_marker_size('medium'),
                    color=outlier_color,
                    ecolor="lightgrey",
                    alpha=0.6,
                    capsize=2,
                    lw=0.5,
                    label="Outliers (corrected)",
                )
        else:
            ax2.errorbar(
                xi,
                yi_corrected,
                xerr=xe,
                yerr=ye_corrected,
                fmt="o",
                ms=get_marker_size('medium'),
                color=get_color('robust'),
                ecolor="lightgrey",
                alpha=0.8,
                capsize=2,
                lw=0.5,
                label="Corrected data",
            )

        ax2.axhline(np.median(yi_corrected), color="gray", linestyle=":", alpha=0.5)
        ax2.set_xlabel(rf"$m_\mathrm{{cal,{color1}}} - m_\mathrm{{cal,{color2}}}$ [mag]")
        ax2.set_ylabel("Corrected [mag]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8, ncol=2)

        # Set y-limits to center on median with +/- 5*std of inliers
        if inlier_mask is not None:
            median_val = np.median(yi_corrected[inlier_mask])
            std_val = np.std(yi_corrected[inlier_mask])
            y_min = median_val - 5 * std_val
            y_max = median_val + 5 * std_val
            ax2.set_ylim(y_min, y_max)
        else:
            median_val = np.median(yi_corrected)
            std_val = np.std(yi_corrected)
            y_min = median_val - 5 * std_val
            y_max = median_val + 5 * std_val
            ax2.set_ylim(y_min, y_max)

        # Save plot to same directory as input file (per-image output folder)
        fpath = self.input_yaml.get("fpath", "")
        if output_dir is None:
            write_dir = os.path.dirname(fpath) or "."
        else:
            write_dir = output_dir
        base_name = os.path.splitext(os.path.basename(fpath))[0] or "color_term"
        plot_file = os.path.join(write_dir, f"Color_Term_{base_name}_piecewise.png")
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"fit_color_term: saved piecewise color term plot to {plot_file}")

    def _fit_piecewise_linear(self, x, y, x_err, y_err, n_segments):
        """
        Fit piecewise linear color term with n segments using RANSAC regression for outlier rejection.

        For n segments, there are n-1 breakpoints. Each segment is a linear fit.
        Segments are continuous at breakpoints.

        Returns
        -------
        (coefficients, coefficient_errors, inlier_mask) : (tuple, tuple, ndarray)
            coefficients: (breakpoints, slopes, intercept)
                breakpoints: tuple of n-1 breakpoint x-values
                slopes: tuple of n slopes
                intercept: single intercept (first segment intercept)
            coefficient_errors: corresponding errors
            inlier_mask: boolean array indicating which points are inliers
        """

        # Sort data by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        x_err_sorted = x_err[sort_idx]
        y_err_sorted = y_err[sort_idx]

        # For n=2 segments, find optimal single breakpoint
        if n_segments == 2:
            # Check minimum data requirements
            if len(x) < 8:
                logger.warning(f"fit_color_term: insufficient data ({len(x)} points) for piecewise fitting, falling back to linear")
                slope, intercept = np.polyfit(x, y, 1)
                slope_err = np.std(y - (slope * x + intercept)) / np.sqrt(len(x))
                return ((), (slope,), intercept), ((), (slope_err,), slope_err), np.ones(len(x), dtype=bool), "linear"

            # Search for optimal breakpoint using grid search on full data
            x_min, x_max = x_sorted.min(), x_sorted.max()
            x_range = x_max - x_min
            search_min = x_min + 0.05 * x_range
            search_max = x_max - 0.05 * x_range

            def objective(bp):
                """Objective function: weighted residual sum of squares (no continuity penalty)"""
                # Fit two segments
                mask1 = x_sorted <= bp
                mask2 = x_sorted > bp

                if np.sum(mask1) < 3 or np.sum(mask2) < 3:
                    return 1e10  # Penalty for insufficient data

                # Segment 1
                x1, y1, ye1 = x_sorted[mask1], y_sorted[mask1], y_err_sorted[mask1]
                try:
                    slope1, intercept1 = np.polyfit(x1, y1, 1)
                    resid1 = y1 - (slope1 * x1 + intercept1)
                    chi2_1 = np.sum((resid1 / ye1)**2)
                except Exception:
                    return 1e10

                # Segment 2
                x2, y2, ye2 = x_sorted[mask2], y_sorted[mask2], y_err_sorted[mask2]
                try:
                    slope2, intercept2 = np.polyfit(x2, y2, 1)
                    resid2 = y2 - (slope2 * x2 + intercept2)
                    chi2_2 = np.sum((resid2 / ye2)**2)
                except Exception:
                    return 1e10

                # No continuity penalty - allow segments to be independent
                return chi2_1 + chi2_2

            # Try grid search for robust breakpoint finding
            try:
                # Test 50 candidate breakpoints evenly spaced
                bp_candidates = np.linspace(search_min, search_max, 50)
                best_bp = bp_candidates[0]
                best_score = 1e10

                for bp in bp_candidates:
                    score = objective(bp)
                    if score < best_score:
                        best_score = score
                        best_bp = bp

                optimal_bp = best_bp
                logger.info(f"fit_color_term: grid search found optimal breakpoint at {optimal_bp:.3f}")
            except Exception as exc:
                logger.warning(f"fit_color_term: breakpoint optimization failed ({exc}), falling back to linear")
                slope, intercept = np.polyfit(x, y, 1)
                slope_err = np.std(y - (slope * x + intercept)) / np.sqrt(len(x))
                return ((), (slope,), intercept), ((), (slope_err,), slope_err), np.ones(len(x), dtype=bool), "linear"

            # Apply RANSAC regression to each segment separately for outlier rejection
            mask1 = x_sorted <= optimal_bp
            mask2 = x_sorted > optimal_bp

            # Segment 1 RANSAC regression
            x1, y1 = x_sorted[mask1], y_sorted[mask1]
            ye1 = y_err_sorted[mask1]
            method1 = "none"
            if len(x1) >= 4:
                try:
                    ransac1 = RANSACRegressor(
                        PenalisedSlopeRegressor(
                            slope_constraint=0.0,
                            slope_tolerance=0.5,
                            penalty_weight=100.0,
                        ),
                        residual_threshold=0.1,
                        max_trials=1000,
                        min_samples=max(2, int(0.2 * len(x1))),
                        random_state=42,
                    )
                    X1 = x1.reshape(-1, 1)
                    ransac1.fit(X1, y1)
                    inlier_mask1 = ransac1.inlier_mask_
                    x1_clean = x1[inlier_mask1]
                    y1_clean = y1[inlier_mask1]
                    slope1, intercept1 = ransac1.estimator_.slope_, ransac1.estimator_.intercept_
                    method1 = "RANSAC"
                    logger.info(f"fit_color_term: segment 1 RANSAC: {np.sum(inlier_mask1)}/{len(x1)} inliers")
                except Exception:
                    logger.warning("fit_color_term: segment 1 RANSAC failed, using polyfit")
                    x1_clean, y1_clean = x1, y1
                    slope1, intercept1 = np.polyfit(x1, y1, 1)
                    inlier_mask1 = np.ones(len(x1), dtype=bool)
                    method1 = "polyfit"
            else:
                x1_clean, y1_clean = x1, y1
                slope1, intercept1 = np.polyfit(x1, y1, 1)
                inlier_mask1 = np.ones(len(x1), dtype=bool)
                method1 = "polyfit"

            # Segment 2 RANSAC regression
            x2, y2 = x_sorted[mask2], y_sorted[mask2]
            ye2 = y_err_sorted[mask2]
            method2 = "none"
            if len(x2) >= 4:
                try:
                    ransac2 = RANSACRegressor(
                        PenalisedSlopeRegressor(
                            slope_constraint=0.0,
                            slope_tolerance=0.5,
                            penalty_weight=100.0,
                        ),
                        residual_threshold=0.1,
                        max_trials=1000,
                        min_samples=max(2, int(0.2 * len(x2))),
                        random_state=42,
                    )
                    X2 = x2.reshape(-1, 1)
                    ransac2.fit(X2, y2)
                    inlier_mask2 = ransac2.inlier_mask_
                    x2_clean = x2[inlier_mask2]
                    y2_clean = y2[inlier_mask2]
                    slope2, intercept2 = ransac2.estimator_.slope_, ransac2.estimator_.intercept_
                    method2 = "RANSAC"
                    logger.info(f"fit_color_term: segment 2 RANSAC: {np.sum(inlier_mask2)}/{len(x2)} inliers")
                except Exception:
                    logger.warning("fit_color_term: segment 2 RANSAC failed, using polyfit")
                    x2_clean, y2_clean = x2, y2
                    slope2, intercept2 = np.polyfit(x2, y2, 1)
                    inlier_mask2 = np.ones(len(x2), dtype=bool)
                    method2 = "polyfit"
            else:
                x2_clean, y2_clean = x2, y2
                slope2, intercept2 = np.polyfit(x2, y2, 1)
                inlier_mask2 = np.ones(len(x2), dtype=bool)
                method2 = "polyfit"

            # Check if we have enough inliers
            if len(x1_clean) < 2 or len(x2_clean) < 2:
                logger.warning(f"fit_color_term: insufficient inliers (seg1: {len(x1_clean)}, seg2: {len(x2_clean)}), falling back to linear")
                slope, intercept = np.polyfit(x, y, 1)
                slope_err = np.std(y - (slope * x + intercept)) / np.sqrt(len(x))
                return ((), (slope,), intercept), ((), (slope_err,), slope_err), np.ones(len(x), dtype=bool), "linear"

            # Do NOT enforce continuity - let segments be independent
            # This allows each segment to fit its data without artificial constraints
            coefficients = ((optimal_bp,), (slope1, slope2), intercept1)

            # Error estimates based on RANSAC fit residuals
            if len(x1_clean) > 1:
                residuals1 = y1_clean - (slope1 * x1_clean + intercept1)
                slope1_err = np.std(residuals1) / np.sqrt(len(x1_clean))
            else:
                slope1_err = 0.0

            if len(x2_clean) > 1:
                residuals2 = y2_clean - (slope2 * x2_clean + intercept2)
                slope2_err = np.std(residuals2) / np.sqrt(len(x2_clean))
            else:
                slope2_err = 0.0

            bp_err = x_range * 0.05  # Rough estimate: 5% of range
            coefficient_errors = ((bp_err,), (slope1_err, slope2_err), slope1_err)

            # Combine inlier masks from segment-specific RANSAC
            combined_inlier_mask_sorted = np.zeros(len(x_sorted), dtype=bool)
            combined_inlier_mask_sorted[mask1] = inlier_mask1
            combined_inlier_mask_sorted[mask2] = inlier_mask2
            # Unsort to match original data order
            inlier_mask = np.zeros(len(x), dtype=bool)
            inlier_mask[sort_idx] = combined_inlier_mask_sorted

            # Determine overall method (use segment with more points as primary)
            overall_method = method1 if len(x1) >= len(x2) else method2

            logger.info(
                f"fit_color_term: piecewise linear: breakpoint={optimal_bp:.3f}, "
                f"slope1={slope1:.4f} ({method1}), slope2={slope2:.4f} ({method2}), intercept={intercept1:.4f}"
            )

            return coefficients, coefficient_errors, inlier_mask, overall_method
        else:
            # For n > 2, fall back to linear (not implemented yet)
            logger.warning(f"fit_color_term: n_segments={n_segments} not yet implemented, falling back to linear")
            slope, intercept = np.polyfit(x, y, 1)
            slope_err = np.std(y - (slope * x + intercept)) / np.sqrt(len(x))
            return ((), (slope,), intercept), ((), (slope_err,), slope_err), np.ones(len(x), dtype=bool), "linear"

    def fit_color_term(self, catalog: pd.DataFrame):
        """
        Fit the colour term c in ZP(m_inst) = a + c*(col1 - col2) via
        RANSAC regression (robust outlier rejection) followed by ODR (error-in-variables
        refinement on inliers). Can also fit quadratic: a + b*x + c*x^2,
        or piecewise linear with n segments.

        Returns
        -------
        (coefficients, coefficient_errors) : (tuple, tuple)
            For linear: (intercept, slope), (intercept_err, slope_err)
            For quadratic: (intercept, slope, quad_coeff), (intercept_err, slope_err, quad_coeff_err)
            For piecewise linear (n segments): (breakpoints, slopes, intercept), (breakpoint_errs, slope_errs, intercept_err)
            Returns (None, None) on failure.
        """
        try:
            if catalog is None or getattr(catalog, "empty", False) or len(catalog) == 0:
                logger.warning("fit_color_term: catalog is None/empty; skipping.")
                return None, None

            use_filter = self.input_yaml.get("imageFilter")
            use_filter = self._normalize_filter(use_filter)
            if not use_filter:
                raise ValueError("Missing 'imageFilter' in input YAML.")

            color1, color2 = self.get_color_term_for_filter(use_filter)

            # Read fitting mode from config
            phot_cfg = self.input_yaml.get("photometry", {}) or {}
            n_segments = int(phot_cfg.get("color_term_n_segments", 1))
            poly_order = int(phot_cfg.get("color_term_poly_order", 1))

            # n_segments > 1 overrides poly_order
            if n_segments > 1:
                logger.info(f"fit_color_term: using piecewise linear with {n_segments} segments")
                fit_mode = "piecewise"
            else:
                if poly_order not in [1, 2]:
                    logger.warning(f"fit_color_term: invalid poly_order {poly_order}, using 1 (linear)")
                    poly_order = 1
                logger.info(f"fit_color_term: using polynomial order {poly_order} ({'linear' if poly_order == 1 else 'quadratic'})")
                fit_mode = "polynomial"

            df = catalog.copy()
            if "sky" in df.columns:
                sky_mask = sigma_clip(np.abs(df["sky"].values), sigma=5, maxiters=10)
                df = df[~sky_mask.mask]
            if "threshold" in df.columns:
                df = df[df["threshold"] >= 5]

            clean = df[df[f"{use_filter}_err"] < 0.32].copy()

            required = [
                use_filter,
                f"{use_filter}_err",
                color1,
                color2,
                f"{color1}_err",
                f"{color2}_err",
                "flux_AP",
                "flux_AP_err",
            ]
            missing = [c for c in required if c not in clean.columns]
            if missing:
                raise KeyError(f"Missing columns: {missing}")

            flux_ap = np.asarray(clean["flux_AP"], float)
            flux_err = np.asarray(clean["flux_AP_err"], float)
            x = np.asarray(clean[color1] - clean[color2], float)
            y = np.asarray(clean[use_filter], float) - (-2.5 * np.log10(flux_ap))
            x_err = np.sqrt(
                clean[f"{color1}_err"] ** 2 + clean[f"{color2}_err"] ** 2
            ).values
            y_err = np.sqrt(
                clean[f"{use_filter}_err"] ** 2
                + (2.5 / np.log(10) * flux_err / flux_ap) ** 2
            ).values

            finite = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(x_err)
                & np.isfinite(y_err)
                & (flux_ap > 0)
            )
            x, y, x_err, y_err = x[finite], y[finite], x_err[finite], y_err[finite]
            flux_ap, flux_err = flux_ap[finite], flux_err[finite]

            # Filter by S/N >= 5
            snr = flux_ap / flux_err
            snr_mask = snr >= 5
            n_before_snr = len(x)
            x, y, x_err, y_err = x[snr_mask], y[snr_mask], x_err[snr_mask], y_err[snr_mask]
            flux_ap, flux_err = flux_ap[snr_mask], flux_err[snr_mask]
            n_after_snr = len(x)
            if n_before_snr - n_after_snr > 0:
                logger.info(f"fit_color_term: removed {n_before_snr - n_after_snr} sources with S/N < 5")

            # Log color distribution for diagnostics
            logger.info(
                f"fit_color_term: color distribution before filtering: "
                f"min={x.min():.3f}, max={x.max():.3f}, median={np.median(x):.3f}, "
                f"n_sources={len(x)}"
            )

            # Filter extreme color sources if configured
            zp_cfg = self.input_yaml.get("zeropoint", {}) or {}
            extreme_color_sigma = zp_cfg.get("extreme_color_sigma", 2.5)

            # Branch based on fitting mode
            if fit_mode == "piecewise":
                # Piecewise linear fitting with n segments
                coefficients, coefficient_errors, inlier_mask, overall_method = self._fit_piecewise_linear(
                    x, y, x_err, y_err, n_segments
                )

                # If fallback returned linear, switch to polynomial flow
                if overall_method == "linear":
                    logger.info(
                        "fit_color_term: piecewise fallback returned linear fit; switching to polynomial flow."
                    )
                    fit_mode = "polynomial"
                    poly_order = 1
                else:
                    # For piecewise, use inlier_mask from RANSAC
                    xi, yi, xe, ye = x, y, x_err, y_err
                    n_in = np.sum(inlier_mask)
                    x_range = float(np.ptp(x))

                    # Plot for piecewise linear (save to same directory as input file)
                    self._plot_piecewise_color_term(xi, yi, xe, ye, coefficients, coefficient_errors, n_segments, color1, color2, use_filter, inlier_mask, overall_method)

                    # Return after plotting
                    return coefficients, coefficient_errors
            else:
                # Polynomial fitting (existing logic)
                if extreme_color_sigma is not None:
                    try:
                        extreme_color_sigma = float(extreme_color_sigma)
                        if extreme_color_sigma > 0 and len(x) > 10:
                            # Sigma-clip in color space to remove extreme colors
                            color_clipped = sigma_clip(x, sigma=extreme_color_sigma, maxiters=5)
                            n_extreme = np.sum(color_clipped.mask)
                            if n_extreme > 0:
                                logger.info(
                                    f"fit_color_term: removing {n_extreme} extreme color sources "
                                    f"(beyond {extreme_color_sigma} sigma in color space)"
                                )
                                x, y, x_err, y_err = (
                                    x[~color_clipped.mask],
                                    y[~color_clipped.mask],
                                    x_err[~color_clipped.mask],
                                    y_err[~color_clipped.mask],
                                )
                    except Exception as exc:
                        logger.warning(f"fit_color_term: extreme color filtering failed: {exc}")

            # 1. Pre-RANSAC sigma clipping to remove extreme outliers
            try:
                ols_slope_pre = np.polyfit(x, y, 1)[0]
                ols_intercept_pre = np.median(y - ols_slope_pre * x)
                ols_resid_pre = y - (ols_slope_pre * x + ols_intercept_pre)
                pre_clip = sigma_clip(ols_resid_pre, sigma=5, maxiters=3, masked=True)
                n_pre_outliers = np.sum(pre_clip.mask)
                if n_pre_outliers > 0:
                    logger.info(f"fit_color_term: pre-RANSAC clipping removed {n_pre_outliers} extreme outliers")
                    x, y, x_err, y_err = x[~pre_clip.mask], y[~pre_clip.mask], x_err[~pre_clip.mask], y_err[~pre_clip.mask]
            except Exception as exc:
                logger.debug(f"fit_color_term: pre-RANSAC clipping skipped: {exc}")

            _min_sources_cfg = int(zp_cfg.get("min_source_no", 5))
            min_sources = max(3, _min_sources_cfg)  # Hard floor of 3 for statistical validity
            if _min_sources_cfg < 3:
                logger.debug(
                    f"fit_color_term: min_source_no={_min_sources_cfg} is below minimum (3); using 3."
                )
            min_color_range = 0.05  # mag
            if len(x) < min_sources:
                logger.warning(
                    f"fit_color_term: too few sources ({len(x)} < {min_sources})."
                )
                return 0.0, np.nan
            x_range = float(np.ptp(x))
            if x_range < min_color_range:
                logger.warning(
                    f"fit_color_term: colour range too small ({x_range:.3f} mag)."
                )
                return 0.0, np.nan

            # 7. Stratified color sampling - ensure adequate coverage across color range
            try:
                color_percentiles = np.percentile(x, [0, 25, 50, 75, 100])
                bin_counts = []
                for i in range(len(color_percentiles) - 1):
                    lo, hi = color_percentiles[i], color_percentiles[i+1]
                    if i == len(color_percentiles) - 2:  # Last bin includes upper edge
                        count = np.sum((x >= lo) & (x <= hi))
                    else:
                        count = np.sum((x >= lo) & (x < hi))
                    bin_counts.append(count)
                min_per_bin = min(bin_counts)
                if min_per_bin < 3:
                    logger.warning(
                        f"fit_color_term: poor color coverage (min {min_per_bin} sources per quartile). "
                        "Results may be unreliable."
                    )
            except Exception as exc:
                logger.debug(f"fit_color_term: stratified sampling check skipped: {exc}")

            # 6. RANSAC regression for robust outlier rejection
            # For polynomial fitting, use polynomial features
            if poly_order == 1:
                X_poly = x[:, None]
            else:
                X_poly = np.column_stack([x**2, x])

            # RANSAC loop for polynomial fitting
            inlier_mask = None
            n_inliers = 0

            # Density-based downsampling to avoid cluster bias
            # If data has dense clusters, downsample them to prevent RANSAC from being biased
            try:
                if len(x) > 100:  # Only apply downsampling if we have enough points
                    # Calculate local density using distance to nearest neighbor
                    from scipy.spatial import KDTree
                    kdtree = KDTree(x[:, None])
                    distances, _ = kdtree.query(x[:, None], k=2)  # Get distance to 2nd nearest neighbor (self is 1st)
                    local_density = 1.0 / (distances[:, 1] + 1e-10)
                    
                    # Identify dense regions (density > 75th percentile)
                    density_threshold = np.percentile(local_density, 75)
                    dense_mask = local_density > density_threshold
                    
                    if np.sum(dense_mask) > 10:  # If we have enough dense points
                        # Downsample dense regions to 50% of their original size
                        dense_indices = np.where(dense_mask)[0]
                        # Use seeded RNG for reproducibility
                        rng = np.random.default_rng(42)
                        keep_dense = rng.choice(dense_indices, size=len(dense_indices)//2, replace=False)
                        sparse_indices = np.where(~dense_mask)[0]
                        
                        # Combine downsampled dense points with all sparse points
                        downsample_indices = np.concatenate([keep_dense, sparse_indices])
                        x_orig, y_orig, x_err_orig, y_err_orig = x.copy(), y.copy(), x_err.copy(), y_err.copy()
                        x, y, x_err, y_err = x_orig[downsample_indices], y_orig[downsample_indices], x_err_orig[downsample_indices], y_err_orig[downsample_indices]
                        X_poly = np.column_stack([x**2, x]) if poly_order == 2 else x[:, None]
                        logger.info(f"fit_color_term: downsampled {len(dense_indices)} dense points to {len(keep_dense)} to avoid cluster bias")
            except Exception as exc:
                logger.debug(f"fit_color_term: density downsampling skipped: {exc}")

            # Single seed for speed (was 5 seeds)
            ransac = RANSACRegressor(
                PenalisedSlopeRegressor(
                    slope_constraint=0.0,
                    slope_tolerance=0.5,
                    penalty_weight=100.0,
                ),
                residual_threshold=0.15,  # Increased from 0.1 to prevent overfitting
                max_trials=500,  # Reduced from 2000 for speed
                min_samples=max(10, int(0.5 * len(x))),  # Increased from 20% to 50% to prevent overfitting
                random_state=42,
            )
            ransac.fit(X_poly, y)
            inlier_mask = ransac.inlier_mask_
            n_inliers = np.sum(inlier_mask)

            if inlier_mask is None or n_inliers < 5:
                logger.warning("fit_color_term: RANSAC failed to find sufficient inliers, using all points")
                inlier_mask = np.ones(len(x), dtype=bool)
                n_inliers = len(x)

            logger.info(f"fit_color_term: RANSAC found {n_inliers} inliers")

            min_inlier_frac = 0.25
            min_inliers = max(5, int(len(x) * min_inlier_frac))

            if n_inliers < min_inliers:
                logger.warning(f"fit_color_term: RANSAC inliers {n_inliers} < {min_inliers} ({int(100*min_inlier_frac)}%), using sigma-clip fallback")
                # Sigma-clip fallback
                r0 = y - np.nanmedian(y)
                clipped = sigma_clip(
                    r0,
                    sigma=3.0,
                    maxiters=5,
                    cenfunc=np.nanmedian,
                    stdfunc=mad_std,
                )
                inlier_mask = ~clipped.mask
                if np.sum(inlier_mask) >= 5:
                    n_inliers = np.sum(inlier_mask)
                else:
                    logger.warning("fit_color_term: sigma-clip also found too few inliers, using all points")
                    inlier_mask = np.ones(len(x), dtype=bool)

            xi, yi = x[inlier_mask], y[inlier_mask]
            xe, ye = x_err[inlier_mask], y_err[inlier_mask]
            n_in = int(np.sum(inlier_mask))

            # Initial fit on inliers for ODR initialization
            if poly_order == 1:
                initial_slope, initial_intercept = np.polyfit(xi, yi, 1)
            else:
                quad_coeffs = np.polyfit(xi, yi, 2)
                initial_quad, initial_slope, initial_intercept = quad_coeffs

            # ODR for error-in-variables refinement.
            if poly_order == 1:
                # Linear model: y = slope * x + intercept
                odr_out = ODR(
                    RealData(xi, yi, sx=xe, sy=ye),
                    Model(lambda B, x: B[0] * x + B[1]),
                    beta0=[initial_slope, initial_intercept],
                ).run()
                coefficients = (float(odr_out.beta[1]), float(odr_out.beta[0]))  # (intercept, slope)
                # Estimate errors from covariance matrix
                if odr_out.cov_beta is not None and np.all(np.isfinite(odr_out.cov_beta)):
                    coefficient_errors = (np.sqrt(odr_out.cov_beta[1, 1]), np.sqrt(odr_out.cov_beta[0, 0]))
                else:
                    coefficient_errors = (np.nan, np.nan)
            else:
                # Quadratic model: y = a*x^2 + b*x + c
                # Use polynomial fit as initial guess
                poly_coeffs = np.polyfit(xi, yi, 2)
                odr_out = ODR(
                    RealData(xi, yi, sx=xe, sy=ye),
                    Model(lambda B, x: B[0] * x**2 + B[1] * x + B[2]),
                    beta0=poly_coeffs,
                ).run()
                coefficients = (float(odr_out.beta[2]), float(odr_out.beta[1]), float(odr_out.beta[0]))  # (intercept, slope, quad)
                # Estimate errors from covariance matrix
                if odr_out.cov_beta is not None and np.all(np.isfinite(odr_out.cov_beta)):
                    coefficient_errors = (np.sqrt(odr_out.cov_beta[2, 2]), np.sqrt(odr_out.cov_beta[1, 1]), np.sqrt(odr_out.cov_beta[0, 0]))
                else:
                    coefficient_errors = (np.nan, np.nan, np.nan)

            # Log the fitted color term for debugging
            if fit_mode == "polynomial" and poly_order == 1:
                logger.info(
                    f"fit_color_term: fitted linear color term: slope={coefficients[1]:.4f}, intercept={coefficients[0]:.4f}, "
                    f"n_inliers={n_in}, color_range={x_range:.3f} mag"
                )
            elif fit_mode == "polynomial" and poly_order == 2:
                logger.info(
                    f"fit_color_term: fitted quadratic color term: quad={coefficients[2]:.4f}, slope={coefficients[1]:.4f}, intercept={coefficients[0]:.4f}, "
                    f"n_inliers={n_in}, color_range={x_range:.3f} mag"
                )

            # If color term is very small (< 0.01), treat it as zero to avoid adding noise
            if fit_mode == "polynomial" and poly_order == 1 and abs(coefficients[1]) < 0.01:
                logger.info(
                    f"fit_color_term: linear slope {coefficients[1]:.4f} is negligible (< 0.01), "
                    "returning 0 to avoid adding noise"
                )
                return (0.0, 0.0), (0.0, 0.0)
            elif fit_mode == "polynomial" and poly_order == 2 and abs(coefficients[2]) < 0.01:
                logger.info(
                    f"fit_color_term: quadratic coefficient {coefficients[2]:.4f} is negligible (< 0.01), "
                    "falling back to linear fit"
                )
                # Fall back to linear if quadratic term is negligible
                coefficients = (coefficients[0], coefficients[1])
                coefficient_errors = (coefficient_errors[0], coefficient_errors[1])
                poly_order = 1

            # Extract plotting variables for polynomial fits
            plot_intercept = coefficients[0]
            plot_slope = coefficients[1] if poly_order == 1 else coefficients[1]
            plot_quad = coefficients[2] if poly_order == 2 else 0.0

            # ---- Plot ------------------------------------------------------
            _style = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
            )
            if os.path.exists(_style):
                plt.style.use(_style)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(340, 2.0), sharex=True)
            plt.subplots_adjust(hspace=0.35, top=0.95, bottom=0.1, left=0.15, right=0.95)

            # Consistent colors from plotting_utils palette
            inlier_color = get_color('inliers')
            outlier_color = get_color('outliers')
            all_sources_color = get_color('all_sources')

            # Top panel: uncorrected data - show cleaned distribution (inliers)
            ax1.errorbar(
                xi,
                yi,
                xerr=xe,
                yerr=ye,
                fmt="o",
                ms=get_marker_size('medium'),
                color=inlier_color,
                ecolor="lightgrey",
                alpha=0.8,
                capsize=1,
                lw=get_line_width('thin') * 0.5,
                label=f"Inliers [{np.sum(inlier_mask)}]",
            )

            x_plot = np.linspace(xi.min() * 0.95, xi.max() * 1.05, 200)
            if poly_order == 1:
                y_plot = plot_intercept + plot_slope * x_plot
                label_text = f"Linear: slope={plot_slope:.3f} +/- {coefficient_errors[1]:.3f}"
            else:
                y_plot = plot_intercept + plot_slope * x_plot + plot_quad * x_plot**2
                label_text = f"Quad: quad={plot_quad:.3f}, slope={plot_slope:.3f}"

            ax1.plot(
                x_plot,
                y_plot,
                color=get_color('fit'),
                linestyle="--",
                lw=get_line_width('thin'),
                label=label_text,
            )

            pad = 0.25

            def _padded_lim(arr):
                lo, hi = arr.min(), arr.max()
                delta = hi - lo
                return lo - pad * delta, hi + pad * delta

            ax1.set_xlim(*_padded_lim(xi))
            ax1.set_ylim(*_padded_lim(yi))
            ax1.set_ylabel(
                rf"$m_\mathrm{{cal,{color1}}} - m_\mathrm{{inst,{use_filter}}}$ [mag]"
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)

            # Bottom panel: color corrected data (should be flat)
            if poly_order == 1:
                yi_corrected = yi - plot_slope * xi
                ye_corrected = np.sqrt(ye**2 + (plot_slope * xe)**2)
            else:
                yi_corrected = yi - (plot_quad * xi**2 + plot_slope * xi)
                ye_corrected = np.sqrt(ye**2 + ((2 * plot_quad * xi + plot_slope) * xe)**2)
            std_uncorrected = float(median_abs_deviation(yi, nan_policy="omit"))
            std_corrected = float(median_abs_deviation(yi_corrected, nan_policy="omit"))

            # Bottom panel: color corrected data - show cleaned distribution (inliers)
            ax2.errorbar(
                xi,
                yi_corrected,
                xerr=xe,
                yerr=ye_corrected,
                fmt="o",
                ms=get_marker_size('medium'),
                color=inlier_color,
                ecolor="lightgrey",
                alpha=0.8,
                capsize=1,
                lw=get_line_width('thin') * 0.5,
                label=f"Inliers (corrected) [{np.sum(inlier_mask)}]",
            )

            y_plot_corrected = np.full_like(x_plot, plot_intercept)
            ax2.plot(
                x_plot,
                y_plot_corrected,
                color=get_color('fit'),
                linestyle="-",
                lw=get_line_width('medium'),
                label=f"Flat (intercept={plot_intercept:.3f})",
            )

            ax2.set_ylim(*_padded_lim(yi_corrected))
            ax2.set_xlabel(
                rf"$m_\mathrm{{cal,{color1}}} - m_\mathrm{{cal,{color2}}}$ [mag]"
            )
            ax2.set_ylabel(
                rf"Corrected $m_\mathrm{{cal,{color1}}} - m_\mathrm{{inst,{use_filter}}}$ [mag]"
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)

            fpath = self.input_yaml.get("fpath", "")
            base_name = os.path.splitext(os.path.basename(fpath))[0] or "color_term"
            write_dir = os.path.dirname(fpath) or "."
            fig.savefig(
                os.path.join(write_dir, f"Color_Term_{base_name}.png"),
                bbox_inches="tight",
                dpi=150,
                facecolor="white",
            )
            plt.close(fig)

            return coefficients, coefficient_errors

        except Exception as exc:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(
                f"fit_color_term failed: {exc_type.__name__} in "
                f"{fname}:{exc_tb.tb_lineno}: {exc}"
            )
            return None, None
