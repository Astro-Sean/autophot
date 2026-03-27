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
from functions import border_msg, snr_err, set_size, calculate_bins

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

    def __init__(self, input_yaml: dict):
        self.input_yaml = input_yaml

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def weighted_average(self, values, errors):
        """
        Inverse-variance weighted mean and its propagated uncertainty.

        Parameters
        ----------
        values, errors : array-like

        Returns
        -------
        (weighted_avg, weighted_error) : (float, float)
        """
        values = np.asarray(values, float)
        errors = np.asarray(errors, float)
        if len(values) != len(errors):
            raise ValueError("values and errors must have the same length")

        # Use only finite, strictly positive uncertainties to form an
        # inverse-variance weighted mean.  This avoids overweighting
        # poorly measured points and protects against NaNs/infs.
        mask = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        if mask.sum() == 0:
            return np.nan, np.nan

        v = values[mask]
        e = errors[mask]
        weights = 1.0 / (e**2)
        weighted_avg = np.sum(weights * v) / np.sum(weights)
        weighted_err = np.sqrt(1.0 / np.sum(weights))
        return weighted_avg, weighted_err

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
        # Optional aperture correction (AP -> total flux) in magnitudes.
        # Only applied when apply_aperture_correction is True; otherwise stored for use later.
        ap_corr_mag = float(self.input_yaml.get("aperture_correction", 0.0) or 0.0)
        apply_ap = bool((self.input_yaml.get("photometry") or {}).get("apply_aperture_correction", False))
        ap_scale = (
            10.0 ** (-0.4 * ap_corr_mag)
            if (apply_ap and np.isfinite(ap_corr_mag) and ap_corr_mag != 0.0)
            else 1.0
        )

        for flux_type in ["AP", "PSF"]:
            fcol = f"flux_{flux_type}"
            if fcol not in catalog.columns:
                zp_params[flux_type] = {
                    "zeropoint": np.nan,
                    "zeropoint_error": np.nan,
                    "has_color_term": False,
                }
                continue

            flux = np.asarray(catalog[fcol].values, float)
            if flux_type == "AP" and ap_scale != 1.0:
                flux = flux * ap_scale

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
        fixed_color_slope,
        color_slope_err: float = 0.0,
    ):
        """
        Subtract the colour term from delta_mag and return the corrected array and
        the propagated correction error.

        Returns
        -------
        delta_corr, color_corr_err : ndarrays shaped like delta_mag
        """
        c1_vals = np.asarray(clean_catalog[color1].values, float)[vmask]
        c2_vals = np.asarray(clean_catalog[color2].values, float)[vmask]

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

        # Propagate both catalog colour errors and uncertainty in the
        # colour-term slope itself, if provided.
        sigma_color = np.sqrt(c1_err**2 + c2_err**2)
        term_color_measure = abs(fixed_color_slope) * sigma_color
        term_color_slope = abs(color_slope_err) * np.abs(color_diff)
        color_corr_err = np.sqrt(term_color_measure**2 + term_color_slope**2)

        delta_corr = delta_mag - fixed_color_slope * color_diff
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
        error_mask = np.asarray(df[err_col].values, float) < 0.32
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

        # Optional aperture correction (AP -> total flux) in magnitudes; apply only
        # when apply_aperture_correction is True (otherwise stored for use later).
        if flux_type == "AP":
            ap_corr_mag = float(self.input_yaml.get("aperture_correction", 0.0) or 0.0)
            apply_ap = bool((self.input_yaml.get("photometry") or {}).get("apply_aperture_correction", False))
            if apply_ap and np.isfinite(ap_corr_mag) and ap_corr_mag != 0.0:
                ap_scale = 10.0 ** (-0.4 * ap_corr_mag)
                flux = flux * ap_scale
                flux_err = flux_err * ap_scale

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
        upperMaglimit: float = 13.0,
        lowerMaglimit: float = 100.0,
        threshold_limit: float = 5.0,
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
        logger.info(border_msg("Cleaning Sequence Stars for Zeropoint"))

        try:
            filter_col = self.input_yaml.get("imageFilter")
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
            if n_missing:
                logger.info(f"Removing {n_missing} sources with missing {filter_col}")

            too_bright = sources[filter_col] < upperMaglimit
            too_faint = sources[filter_col] > lowerMaglimit
            n_brightness = (too_bright | too_faint).sum()
            if n_brightness:
                logger.info(
                    "Removing %d sources outside magnitude range %.2f-%.2f mag",
                    n_brightness,
                    upperMaglimit,
                    lowerMaglimit,
                )

            low_snr = sources["threshold"] < threshold_limit
            n_snr = low_snr.sum()
            if n_snr:
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
        logger.info(border_msg("Measuring Zeropoint Offset"))

        methods = ["AP"] + (["PSF"] if "flux_PSF" in sources.columns else [])
        method_labels = {"AP": "Aperture", "PSF": "PSF"}
        output_zp = {}

        image_filter = self.input_yaml["imageFilter"]
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

                zp = mag_col.loc[src.index] - inst_mag
                zp_err = np.sqrt(error_snr**2 + mag_err_col.loc[src.index] ** 2)

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
        logger.info("Starting RANSAC + weighted constant fit.")

        x = np.asarray(x_indep, float)
        y = np.asarray(delta_mag, float)
        w0 = np.asarray(w_mag, float)
        x_err = np.zeros_like(x) if x_err is None else np.asarray(x_err, float)

        orig_size = len(x)

        # Quality filter: finite, positive weight, error <= 0.5 mag.
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

        # Trivial fallback for very sparse data.
        if len(x) < 3:
            ZP = np.average(y, weights=w0) if len(y) else np.nan
            sum_w = np.sum(w0)
            zp_se = 1.0 / np.sqrt(sum_w) if (len(y) and sum_w > 0) else np.nan
            full = np.zeros(orig_size, dtype=bool)
            full[keep_idx] = True
            return ZP, slope, full, np.diag([zp_se**2, slope_err**2])

        # RANSAC threshold from MAD of residuals.
        r0 = y - np.average(y, weights=w0)
        mad = np.nanmedian(np.abs(r0)) * 1.4826 + 1e-12

        ransac = RANSACRegressor(
            PenalisedSlopeRegressor(
                slope_constraint=0.0,
                slope_tolerance=slope_tolerance,
                penalty_weight=penalty_weight,
            ),
            residual_threshold=min(3.0 * mad, 0.1),
            max_trials=max_trials,
            min_samples=ransac_min_samples,
            random_state=random_state,
        )
        ransac.fit(x[:, None], y)
        inlier_mask = ransac.inlier_mask_
        logger.info(f"RANSAC: {inlier_mask.sum()}/{len(x)} inliers")

        if inlier_mask.sum() < 2:
            inlier_mask = np.ones(len(x), dtype=bool)
            logger.warning("RANSAC found too few inliers; using all points.")

        xi, yi, wi = x[inlier_mask], y[inlier_mask], w0[inlier_mask]
        intercept = np.average(yi, weights=wi)
        # SE(weighted mean) = 1/sqrt(sum(w_i)) for w_i = 1/sigma_i^2
        sum_wi = np.sum(wi)
        intercept_err = 1.0 / np.sqrt(sum_wi) if sum_wi > 0 else np.nan

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
        fixed_color_slope: float = None,
        fixed_color_slope_err: float = None,
    ):
        """
        Fit ZP = m_cat - m_inst[+/-c*(c1-c2)] vs m_inst via RANSAC.

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

            _style = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
            )
            if os.path.exists(_style):
                plt.style.use(_style)
            fig, ax = plt.subplots(1, 1, figsize=set_size(540, 1))
            inlier_masks_full = {
                k: np.zeros(len(clean_catalog), dtype=bool) for k in ["AP", "PSF"]
            }
            colors = {"AP": "blue", "PSF": "green"}
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
                if has_color_term and fixed_color_slope is not None:
                    delta_mag, color_corr_err = self._apply_color_correction(
                        delta_mag,
                        clean_catalog,
                        vmask,
                        color1,
                        color2,
                        fixed_color_slope,
                        color_slope_err=fixed_color_slope_err or 0.0,
                    )

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
                            has_color_term and fixed_color_slope is not None
                        ),
                    }
                )
                if has_color_term and fixed_color_slope is not None:
                    fit_params[flux_type].update(
                        {
                            "color_term": fixed_color_slope,
                            "color_term_error": fixed_color_slope_err or 0.0,
                            "color1": color1,
                            "color2": color2,
                        }
                    )

                msg = f"[{flux_type}] ZP={ZP:.3f} +/- {zp_std:.3f}"
                logger.info(msg)

                # ---- Plot --------------------------------------------------
                in_x = inst_mag[inlier_short]
                in_y = delta_mag[inlier_short]
                in_e = yerr[inlier_short]

                ax.errorbar(
                    in_x,
                    in_y,
                    xerr=inst_mag_err[inlier_short],
                    yerr=in_e,
                    fmt="o",
                    ms=3,
                    color=colors[flux_type],
                    alpha=0.75,
                    capsize=1.5,
                    label=f"{labels[flux_type]} inliers",
                )
                out_mask = ~inlier_short
                if out_mask.any():
                    ax.scatter(
                        inst_mag[out_mask],
                        delta_mag[out_mask],
                        s=20,
                        marker="x",
                        alpha=0.5,
                        color=colors[flux_type],
                    )

                xs = np.linspace(in_x.min() - 0.5, in_x.max() + 0.5, 200)
                ys = np.full_like(xs, ZP)
                sig_y = np.full_like(xs, zp_std)
                ax.plot(xs, ys, "--", color=colors[flux_type], label=msg)
                ax.fill_between(
                    xs, ys - sig_y, ys + sig_y, color=colors[flux_type], alpha=0.2
                )

                global_xmins.append(xs[0])
                global_xmaxs.append(xs[-1])
                global_ymins.append(min(in_y.min() - in_e.mean(), ys[0] - 2 * sig_y[0]))
                global_ymaxs.append(max(in_y.max() + in_e.mean(), ys[0] + 2 * sig_y[0]))

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

            y_label = (
                rf"$m_\mathrm{{cal,{use_filter}}} - m_\mathrm{{inst,{use_filter}}}$"
            )
            if has_color_term and fixed_color_slope is not None:
                sign = r" + " if fixed_color_slope <= 0 else r" - "
                y_label += (
                    rf"{sign}{abs(fixed_color_slope):.2f}"
                    rf"($m_\mathrm{{cal,{color1}}} - m_\mathrm{{cal,{color2}}}$)"
                )
            y_label += " [mag]"
            ax.set_xlabel(rf"$m_\mathrm{{inst,{use_filter}}}$ [mag]")
            ax.set_ylabel(y_label)
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(ncol=2, loc="upper left", frameon=False)

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
        sigma_clip_sigma: float = 3.0,
        sigma_clip_maxiters: int = 10,
        min_sources: int = 1,
        fixed_color_slope: float = None,
        fixed_color_slope_err: float = None,
    ):
        """
        Estimate ZP via sigma-clipped median of (m_ref - m_inst).
        Produces a combined histogram PDF for AP and PSF.

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
                    zp_params = self._fallback_zeropoint(catalog, use_filter)
                    return catalog, zp_params

                _style = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
                )
                if os.path.exists(_style):
                    plt.style.use(_style)
                fig_hist, ax_hist = plt.subplots(1, 1, figsize=set_size(540, 1))
                colors = {"AP": "blue", "PSF": "green"}
                labels_base = {"AP": "Aperture", "PSF": "PSF"}
                inlier_masks_full = {
                    k: np.zeros(len(clean_catalog), dtype=bool) for k in ["AP", "PSF"]
                }

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

                    if has_color_term and fixed_color_slope is not None:
                        delta_mag, color_corr_err = self._apply_color_correction(
                            delta_mag,
                            clean_catalog,
                            vmask,
                            color1,
                            color2,
                            fixed_color_slope,
                            color_slope_err=fixed_color_slope_err
                            or 0.0,
                        )
                        # Propagate colour-correction uncertainty into delta_mag_err.
                        delta_mag_err = np.sqrt(delta_mag_err**2 + color_corr_err**2)

                    finite_mask = np.isfinite(delta_mag) & np.isfinite(delta_mag_err)
                    delta_mag = delta_mag[finite_mask]
                    delta_mag_err = delta_mag_err[finite_mask]
                    delta_no_corr = delta_no_corr[finite_mask]
                    delta_no_corr_err = delta_no_corr_err[finite_mask]
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
                        stdfunc=median_abs_deviation,
                    )
                    inlier_mask = ~clipped.mask
                    inlier_deltas = clipped.data[inlier_mask]
                    inlier_delta_err = delta_mag_err[inlier_mask]

                    finite2 = np.isfinite(inlier_deltas) & np.isfinite(inlier_delta_err)
                    inlier_deltas = inlier_deltas[finite2]
                    inlier_delta_err = inlier_delta_err[finite2]

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

                    n_inl = len(inlier_deltas)
                    mad_zp = float(
                        median_abs_deviation(inlier_deltas, nan_policy="omit")
                    )
                    # SE(median) ~ 1.858 * MAD/sqrt(N) (1.253*sigma/sqrt(N), sigma~1.4826*MAD)
                    zp_std = (1.858 * mad_zp / np.sqrt(n_inl)) if n_inl >= 2 else mad_zp

                    # Inverse-variance weighted uncertainty from propagated
                    # per-star delta_mag_err.
                    vm = np.isfinite(inlier_delta_err) & (inlier_delta_err > 0)
                    zp_err_weighted = np.nan
                    if np.sum(vm) >= 2:
                        w = 1.0 / (inlier_delta_err[vm] ** 2)
                        zp_err_weighted = float(np.sqrt(1.0 / np.sum(w)))

                    zp_err = (
                        float(zp_err_weighted)
                        if np.isfinite(zp_err_weighted)
                        else float(zp_std)
                    )

                    zp_params[flux_type].update(
                        {
                            "zeropoint": zp_final,
                            "zeropoint_error": zp_err,
                            "has_color_term": bool(
                                has_color_term and fixed_color_slope is not None
                            ),
                        }
                    )
                    if has_color_term and fixed_color_slope is not None:
                        zp_params[flux_type].update(
                            {
                                "color_term": fixed_color_slope,
                                "color_term_error": fixed_color_slope_err or 0.0,
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
                            f"N={len(inlier_deltas)}, "
                            f"ZP={zp_final:.3f}+/-{zp_err:.3f})"
                        ),
                    )

                    # ---- Histogram (without colour correction) -------------
                    if has_color_term and fixed_color_slope is not None:
                        dnc = delta_no_corr[np.isfinite(delta_no_corr)]
                        clipped_nc = sigma_clip(
                            dnc,
                            sigma=sigma_clip_sigma,
                            maxiters=sigma_clip_maxiters,
                            cenfunc=np.nanmedian,
                            stdfunc=median_abs_deviation,
                        )
                        inl_nc = clipped_nc.data[~clipped_nc.mask]
                        inl_nc = inl_nc[np.isfinite(inl_nc)]

                        if len(inl_nc) > 0:
                            be_nc = np.histogram_bin_edges(inl_nc, bins="fd")
                            bc_nc = (be_nc[:-1] + be_nc[1:]) / 2
                            w_nc = be_nc[1] - be_nc[0]
                            ct_nc, _ = np.histogram(inl_nc, bins=be_nc, density=True)
                            zp_nc = float(np.nanmedian(inl_nc))
                            std_nc = float(
                                median_abs_deviation(inl_nc, nan_policy="omit")
                            )

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
                                    f"N={len(inl_nc)}, ZP={zp_nc:.3f}+/-{std_nc:.3f})"
                                ),
                            )

                    # Update inlier mask.
                    full_mask = np.zeros(len(clean_catalog), dtype=bool)
                    full_mask[np.flatnonzero(vmask)] = ~clipped.mask
                    inlier_masks_full[flux_type] = full_mask

                ax_hist.set_xlabel("Zeropoint [mag]")
                ax_hist.set_ylabel("Density")
                ax_hist.grid(True, ls="--", alpha=0.3, zorder=0)
                ax_hist.legend(loc="best", frameon=False)
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

    def fit_color_term(self, catalog: pd.DataFrame):
        """
        Fit the colour term c in ZP(m_inst) = a + c*(col1 - col2) via
        RANSAC (outlier rejection) followed by ODR (error-in-variables
        refinement on inliers).

        Returns
        -------
        (color_term, color_term_error) : (float, float)
            Returns (None, None) on failure.
        """
        try:
            if catalog is None or getattr(catalog, "empty", False) or len(catalog) == 0:
                logger.warning("fit_color_term: catalog is None/empty; skipping.")
                return None, None

            use_filter = self.input_yaml.get("imageFilter")
            if not use_filter:
                raise ValueError("Missing 'imageFilter' in input YAML.")

            color1, color2 = self.get_color_term_for_filter(use_filter)

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

            zp_cfg = self.input_yaml.get("zeropoint", {}) or {}
            min_sources = int(zp_cfg.get("min_source_no", 1))
            if min_sources < 1:
                min_sources = 1
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

            # Adaptive RANSAC residual threshold from initial OLS residuals.
            try:
                ols_slope = np.polyfit(x, y, 1)[0]
                ols_resid = y - (ols_slope * x + np.median(y - ols_slope * x))
                mad_resid = median_abs_deviation(ols_resid, scale="normal")
                residual_threshold = float(np.clip(2.5 * mad_resid, 0.05, 0.35))
            except Exception:
                residual_threshold = 0.1

            base_estimator = PenalisedSlopeRegressor(
                slope_constraint=0.0,
                slope_tolerance=1.0,
                penalty_weight=0.0,
            )
            min_inlier_frac = 0.25
            min_inliers = max(5, int(len(x) * min_inlier_frac))
            ransac_seeds = [42, 0, 123, 456, 789]
            best_inlier_mask = None
            best_n_inliers = -1
            best_med_abs_resid = np.inf
            best_estimator = None

            for seed in ransac_seeds:
                ransac = RANSACRegressor(
                    base_estimator,
                    residual_threshold=residual_threshold,
                    max_trials=4000,
                    min_samples=min(4, len(x) - 1),
                    random_state=seed,
                )
                ransac.fit(x[:, None], y)
                inlier_mask = ransac.inlier_mask_
                n_in = int(np.sum(inlier_mask))
                if n_in < 4:
                    continue
                resid = y - ransac.predict(x[:, None])
                med_abs = float(np.median(np.abs(resid[inlier_mask])))
                if n_in > best_n_inliers or (
                    n_in == best_n_inliers and med_abs < best_med_abs_resid
                ):
                    best_n_inliers = n_in
                    best_med_abs_resid = med_abs
                    best_inlier_mask = inlier_mask
                    best_estimator = ransac.estimator_

            if best_inlier_mask is None or best_n_inliers < 4:
                logger.warning("fit_color_term: RANSAC found no usable inlier set.")
                return 0.0, np.nan

            inlier_mask = best_inlier_mask
            ransac_slope = best_estimator.slope_
            ransac_intercept = best_estimator.intercept_

            if best_n_inliers < min_inliers:
                # Fallback: sigma-clip residuals and use all remaining points.
                pred = ransac_slope * x + ransac_intercept
                resid = y - pred
                clipped = sigma_clip(resid, sigma=3, maxiters=5)
                inlier_mask = ~clipped.mask
                if np.sum(inlier_mask) >= 5:
                    ransac_slope = float(
                        np.polyfit(x[inlier_mask], y[inlier_mask], 1)[0]
                    )
                    ransac_intercept = float(
                        np.median(y[inlier_mask] - ransac_slope * x[inlier_mask])
                    )
                else:
                    inlier_mask = best_inlier_mask

            xi, yi = x[inlier_mask], y[inlier_mask]
            xe, ye = x_err[inlier_mask], y_err[inlier_mask]
            n_in = int(np.sum(inlier_mask))

            # ODR for error-in-variables refinement.
            odr_out = ODR(
                RealData(xi, yi, sx=xe, sy=ye),
                Model(lambda B, x: B[0] * x + B[1]),
                beta0=[ransac_slope, ransac_intercept],
            ).run()

            color_term = float(odr_out.beta[0])
            if odr_out.cov_beta is not None and np.all(np.isfinite(odr_out.cov_beta)):
                cov00 = odr_out.cov_beta[0, 0]
                if cov00 > 0:
                    color_term_error = float(np.sqrt(cov00))
                else:
                    color_term_error = np.nan
            else:
                color_term_error = np.nan

            # Fallback if ODR did not converge or covariance invalid.
            odr_ok = (
                getattr(odr_out, "stopreason", None) is not None
                and getattr(odr_out, "info", None) is not None
                and odr_out.info <= 4
                and np.isfinite(color_term)
            )
            if not odr_ok or not np.isfinite(color_term_error):
                color_term = ransac_slope
                resid_in = yi - (ransac_slope * xi + ransac_intercept)
                color_term_error = float(
                    mad_std(resid_in) / np.sqrt(max(1, n_in - 2))
                    if n_in > 2
                    else np.nan
                )
                if not np.isfinite(color_term_error) or color_term_error <= 0:
                    color_term_error = 0.2

            # Sanity: reject physically unreasonable colour terms.
            max_slope = 2.0
            if not np.isfinite(color_term) or abs(color_term) > max_slope:
                logger.warning(
                    f"fit_color_term: slope {color_term} out of range [-{max_slope},{max_slope}], "
                    "returning 0 +/- 0.5."
                )
                return 0.0, 0.5

            plot_intercept = (
                float(odr_out.beta[1])
                if odr_ok and np.isfinite(odr_out.beta[1])
                else ransac_intercept
            )

            # ---- Plot ------------------------------------------------------
            _style = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
            )
            if os.path.exists(_style):
                plt.style.use(_style)
            fig, ax = plt.subplots(figsize=set_size(340, 1))

            # Okabe-Ito palette for consistent, colorblind-friendly plots.
            okabe_blue = "#0000FF"
            okabe_orange = "#FF0000"

            ax.errorbar(
                xi,
                yi,
                xerr=xe,
                yerr=ye,
                fmt="o",
                ms=4,
                color=okabe_blue,
                ecolor="lightgrey",
                alpha=0.8,
                capsize=2,
                lw=0.5,
                label="Inliers",
            )

            out_mask = ~inlier_mask
            if out_mask.any():
                ax.scatter(
                    x[out_mask],
                    y[out_mask],
                    s=30,
                    marker="x",
                    alpha=0.4,
                    color=okabe_orange,
                    label="Outliers",
                )

            x_plot = np.linspace(xi.min() * 0.95, xi.max() * 1.05, 200)
            y_plot = plot_intercept + color_term * x_plot
            if (
                odr_ok
                and odr_out.cov_beta is not None
                and np.all(np.isfinite(odr_out.cov_beta))
            ):
                cov = odr_out.cov_beta
                sig_y = np.sqrt(
                    cov[1, 1] + x_plot**2 * cov[0, 0] + 2 * x_plot * cov[0, 1]
                )
                ax.fill_between(
                    x_plot,
                    y_plot - sig_y,
                    y_plot + sig_y,
                    color=okabe_orange,
                    alpha=0.15,
                )
            ax.plot(
                x_plot,
                y_plot,
                color=okabe_orange,
                linestyle="--",
                lw=0.5,
                label=f"Slope={color_term:.3f} +/- {color_term_error:.3f}",
            )

            pad = 0.25

            def _padded_lim(arr):
                lo, hi = arr.min(), arr.max()
                delta = hi - lo
                return lo - pad * delta, hi + pad * delta

            ax.set_xlim(*_padded_lim(xi))
            ax.set_ylim(*_padded_lim(yi))
            ax.set_xlabel(
                rf"$m_\mathrm{{cal,{color1}}} - m_\mathrm{{cal,{color2}}}$ [mag]"
            )
            ax.set_ylabel(
                rf"$m_\mathrm{{cal,{color1}}} - m_\mathrm{{inst,{use_filter}}}$ [mag]"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", frameon=False)

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

            logger.info(f"Color term: {color_term:.3f} +/- {color_term_error:.3f}")
            return color_term, color_term_error

        except Exception as exc:
            logger.error(f"fit_color_term failed: {exc}")
            return None, None
