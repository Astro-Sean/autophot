#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FWHM Measurement and Source Detection for Astronomical Images

This module provides tools for detecting point-like sources in astronomical images,
estimating the Full Width at Half Maximum (FWHM), and filtering sources based on various criteria.
It is designed for robustness and efficiency, with support for background estimation,
source segmentation, and outlier removal.

Author: Sean Brennan
Date: 2022-09-28 (updated 2026-02-19)
"""

# --- Standard Library Imports ---
import os
import sys
import logging
import warnings
import time
from typing import Optional, Tuple, List, Dict, Any, Union

# --- Third-Party Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, median_abs_deviation
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, binary_dilation, label
from scipy.spatial import cKDTree, distance
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std, SigmaClip
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyWarning
from astropy.table import Table
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.detection import StarFinder, IRAFStarFinder, DAOStarFinder, find_peaks
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from photutils.segmentation import (
    detect_threshold,
    detect_sources,
    deblend_sources,
    SourceCatalog,
)
from photutils.centroids import (
    centroid_1dg,
    centroid_2dg,
    centroid_com,
    centroid_quadratic,
)
from skimage import feature, transform, exposure, morphology, draw
from lmfit import Model
from lmfit.models import Gaussian2dModel
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# --- Local Imports ---
from functions import pix_dist, border_msg, pad_ones, set_size, get_normalized_histogram

# --- Logging and Warnings ---
warnings.simplefilter("ignore", category=AstropyWarning)
logger = logging.getLogger(__name__)

# --- Constants ---
SQRT2LOG2 = 2 * np.sqrt(2 * np.log(2))  # 2.354820045


class Find_FWHM:
    """
    A class for detecting point-like sources in astronomical images and estimating the FWHM.
    Supports background estimation, source segmentation, and robust outlier removal.
    """

    def __init__(self, input_yaml: Dict[str, Any]):
        """
        Initialize the FindFWHM class with configuration from input_yaml.

        Args:
            input_yaml (dict): Configuration parameters for the FWHM calculation.
        """
        self.input_yaml = input_yaml
        self.logger = logger

    # =============================================================================
    #  Utility Functions
    # =============================================================================

    def create_circular_mask(
        self,
        h: int,
        w: int,
        center: Optional[Tuple[float, float]] = None,
        radius: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create a circular mask centered within an image.

        Args:
            h (int): Height of the image.
            w (int): Width of the image.
            center (tuple, optional): Pixel location of the mask center. Defaults to image center.
            radius (float, optional): Radius of the mask in pixels. Defaults to smallest distance to image edge.

        Returns:
            np.ndarray: Boolean mask with shape (h, w).
        """
        if center is None:
            center = (int(w / 2), int(h / 2))
        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist_from_center <= radius

    def filter_isolated_sources(
        self,
        fwhm_table: pd.DataFrame,
        x_col: str = "x_pix",
        y_col: str = "y_pix",
        min_distance: float = 5.0,
    ) -> pd.DataFrame:
        """
        Return sources with no neighbors within `min_distance` using KDTree.

        Args:
            fwhm_table (pd.DataFrame): Table of sources with x and y coordinates.
            x_col (str): Column name for x coordinates.
            y_col (str): Column name for y coordinates.
            min_distance (float): Minimum distance (in pixels) to consider a source as isolated.

        Returns:
            pd.DataFrame: Only sources with no neighbors within `min_distance`.
        """
        if len(fwhm_table) < 2:
            return fwhm_table

        coords = fwhm_table[[x_col, y_col]].values
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)
        nearest_distances = distances[:, 1]
        cleaned_sources = fwhm_table[nearest_distances > min_distance].reset_index(
            drop=True
        )
        self.logger.info(
            f"Cleaned {len(fwhm_table)} sources to {len(cleaned_sources)} with isolation distance [{min_distance}]"
        )
        return cleaned_sources

    def isolated_via_segmentation(
        self,
        image: np.ndarray,
        coordinates_df: pd.DataFrame,
        fwhm: float = 3.0,
        npixels: int = 5,
        contrast: float = 0.005,
        min_distance: float = 10.0,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Perform image segmentation, identify well-isolated sources, and return a cleaned DataFrame.
        Sources are removed if they are within `min_distance` pixels of any segment's center of mass (COM),
        excluding the segment the source is likely already in.

        Args:
            image (np.ndarray): 2D numpy array representing the image.
            coordinates_df (pd.DataFrame): DataFrame with columns 'x_pix' and 'y_pix'.
            fwhm (float): Full Width at Half Maximum for source detection.
            npixels (int): Minimum number of connected pixels for a source.
            contrast (float): Minimum contrast ratio for deblending.
            min_distance (float): Minimum distance (in pixels) from any segment's COM.
            plot (bool): If True, saves a plot of the segmented image.

        Returns:
            pd.DataFrame: Cleaned DataFrame containing only well-isolated sources.
        """
        from astropy.stats import sigma_clipped_stats
        from photutils.segmentation import SourceCatalog
        from sklearn.metrics import pairwise_distances_argmin_min
        from astropy.visualization import ZScaleInterval, ImageNormalize

        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        threshold = detect_threshold(image, nsigma=3)
        segment_map = detect_sources(image, threshold, npixels=npixels)
        deblended_map = deblend_sources(
            image,
            segment_map,
            npixels=npixels,
            contrast=contrast,
            mode="exponential",
            progress_bar=False,
        )
        catalog = SourceCatalog(image, deblended_map)
        coms = np.column_stack((catalog.xcentroid, catalog.ycentroid))
        segment_ids = catalog.label

        source_coords = np.column_stack(
            (coordinates_df["x_pix"], coordinates_df["y_pix"])
        )
        all_dists = np.sqrt(
            ((source_coords[:, np.newaxis, :] - coms[np.newaxis, :, :]) ** 2).sum(
                axis=-1
            )
        )
        coordinates_df["is_isolated"] = True

        for i, src_coord in enumerate(source_coords):
            closest_seg_idx = np.argmin(all_dists[i])
            own_seg_id = segment_ids[closest_seg_idx]
            own_idxs = np.where(segment_ids == own_seg_id)[0]
            row_dists = all_dists[i].copy()
            row_dists[own_idxs] = np.inf
            if np.min(row_dists) < min_distance:
                coordinates_df.at[i, "is_isolated"] = False

        cleaned_df = coordinates_df[coordinates_df["is_isolated"]].copy()

        if plot:
            zscale = ZScaleInterval()
            norm = ImageNormalize(image, interval=zscale)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image, cmap="gray", origin="lower", norm=norm)
            ax.contour(
                deblended_map.data,
                levels=np.unique(deblended_map.data[deblended_map.data > 0]),
                colors="red",
                linewidths=0.5,
            )
            ax.scatter(
                cleaned_df["x_pix"],
                cleaned_df["y_pix"],
                color="blue",
                label="Isolated Sources",
                s=30,
            )
            ax.scatter(
                coms[:, 0],
                coms[:, 1],
                color="green",
                marker="x",
                label="Segment COMs",
                s=50,
            )
            ax.legend()
            fpath = self.input_yaml["fpath"]
            write_dir = self.input_yaml["write_dir"]
            base = os.path.basename(fpath).split(".")[0]
            pdf_out = os.path.join(write_dir, f"segmentation_{base}.pdf")
            png_out = os.path.join(write_dir, f"Segmentation_{base}.png")
            fig.savefig(pdf_out, bbox_inches="tight")
            fig.savefig(png_out, bbox_inches="tight")
            plt.close(fig)

        return cleaned_df

    # =============================================================================
    #  Linearity and Fitting
    # =============================================================================

    def check_linearity(
        self,
        catalog: pd.DataFrame,
        threshold: float = 3,
        method: str = "RANSAC",
        residual_threshold: float = 0.25,
        write_dir: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[float]]:
        """
        Fast detector linearity check using RANSAC for robust fitting.
        Identifies sources where the relationship between max pixel value and enclosed flux is linear.

        Args:
            catalog (pd.DataFrame): DataFrame of sources with flux and peak columns.
            threshold (float): Minimum threshold for source inclusion.
            method (str): Fitting method (default: "RANSAC").
            residual_threshold (float): Residual threshold for RANSAC.
            write_dir (bool): If True, saves a diagnostic plot.

        Returns:
            tuple: (linear_sources, fit_params, saturation_range)
        """

        def _mad(x: np.ndarray) -> float:
            """Compute robust median absolute deviation (MAD) estimator of scatter."""
            x = np.asarray(x)
            med = np.nanmedian(x)
            return 1.4826 * np.nanmedian(np.abs(x - med))

        class ConstrainedSlopeRegressor(BaseEstimator, RegressorMixin):
            def __init__(
                self,
                slope_constraint: float = 0.0,
                slope_tolerance: float = 0.0,
                penalty_weight: float = 100.0,
            ):
                self.slope_constraint = slope_constraint
                self.slope_tolerance = slope_tolerance
                self.penalty_weight = penalty_weight

            def fit(self, X: np.ndarray, y: np.ndarray):
                X, y = check_X_y(X, y)

                def loss(params: np.ndarray) -> float:
                    slope, intercept = params
                    pred = slope * X.flatten() + intercept
                    mse = np.mean((y - pred) ** 2)
                    slope_diff = abs(slope - self.slope_constraint)
                    penalty = self.penalty_weight * max(
                        0, slope_diff - self.slope_tolerance
                    )
                    return mse + penalty

                slope_init = 0.0
                intercept_init = np.mean(y) - slope_init * np.mean(X)
                init = [slope_init, intercept_init]
                result = minimize(loss, init, method="L-BFGS-B")
                self.slope_, self.intercept_ = result.x
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                check_is_fitted(self)
                X = check_array(X)
                return self.slope_ * X.flatten() + self.intercept_

        fit_params = {
            "intercept": np.nan,
            "intercept_error": np.nan,
            "sigma_mad": np.nan,
            "n_input": int(len(catalog) if catalog is not None else 0),
            "n_after_quality_cuts": 0,
            "n_linear_inliers": 0,
            "method_used": method,
        }
        saturation_range = [0.0, np.inf]

        try:
            if not isinstance(catalog, pd.DataFrame) or len(catalog) == 0:
                logger.warning("Empty or invalid catalog.")
                return catalog, fit_params, saturation_range

            req = ["flux_AP", "maxPixel"]
            missing = [c for c in req if c not in catalog.columns]
            if missing:
                raise KeyError(f"Missing required columns: {missing}")

            df = catalog.copy()

            # --- Quality cuts ---
            m0 = np.isfinite(df["flux_AP"].values) & (df["flux_AP"].values > 0)
            m0 &= np.isfinite(df["maxPixel"].values) & (df["maxPixel"].values > 0)
            if "threshold" in df.columns:
                m0 &= np.isfinite(df["threshold"].values) & (
                    df["threshold"].values >= float(threshold)
                )
            if "flags" in df.columns:
                m0 &= np.isfinite(df["flags"].values) & (df["flags"].values == 0)
            if "ellipticity" in df.columns:
                ell = df["ellipticity"].values
                m0 &= np.isfinite(ell) & (ell <= 0.25)
            if "fwhm" in df.columns:
                fwhm = df["fwhm"].values
                med_fwhm = (
                    np.nanmedian(fwhm[np.isfinite(fwhm)])
                    if np.any(np.isfinite(fwhm))
                    else np.nan
                )
                if np.isfinite(med_fwhm) and med_fwhm > 0:
                    m0 &= (
                        np.isfinite(fwhm)
                        & (fwhm > 0.6 * med_fwhm)
                        & (fwhm < 1.4 * med_fwhm)
                    )

            excluded_sources = df.loc[~m0]
            logger.info(
                f"Excluded {len(excluded_sources)} sources based on quality cuts "
                f"(kept {len(df) - len(excluded_sources)}/{len(df)})."
            )
            df = df.loc[m0].copy()
            fit_params["n_after_quality_cuts"] = int(len(df))

            # --- Convert fluxes -> magnitudes ---
            flux = df["flux_AP"].values.astype(float)
            peak = df["maxPixel"].values.astype(float)
            df["m_inst"] = -2.5 * np.log10(flux)
            df["m_peak"] = -2.5 * np.log10(peak)

            # --- Magnitude errors ---
            if "flux_AP_err" in df.columns and np.any(np.isfinite(df["flux_AP_err"])):
                fe = df["flux_AP_err"].values.astype(float)
                df["m_inst_err"] = (2.5 / np.log(10)) * (fe / flux)
            else:
                df["m_inst_err"] = np.nan
            if "maxPixel_err" in df.columns and np.any(np.isfinite(df["maxPixel_err"])):
                pe = df["maxPixel_err"].values.astype(float)
                df["m_peak_err"] = (2.5 / np.log(10)) * (pe / peak)
            else:
                df["m_peak_err"] = np.nan

            # --- Per-source weights ---
            var = np.square(df["m_inst_err"].values) + np.square(
                df["m_peak_err"].values
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                w = 1.0 / var
            w[~np.isfinite(w)] = 1.0
            w = np.clip(w, 1e-6, np.inf)

            # --- RANSAC fitting ---
            delta = df["m_peak"] - df["m_inst"]
            X = df["m_inst"].values.reshape(-1, 1)
            y = delta.values
            n_pts = len(df)
            min_samples = min(10, max(2, n_pts))
            if n_pts < 2:
                logger.warning("Too few points for linearity RANSAC (need at least 2).")
                return catalog, fit_params, saturation_range

            ransac = RANSACRegressor(
                estimator=ConstrainedSlopeRegressor(
                    slope_constraint=0.0, slope_tolerance=0.0
                ),
                residual_threshold=residual_threshold,
                max_trials=1000,
                min_samples=min_samples,
                random_state=42,
            )
            ransac.fit(X, y)
            b = ransac.estimator_.intercept_

            inlier_mask = ransac.inlier_mask_.copy()

            # Bin-wise majority filter: in low-S/N magnitude bins, reject the bin
            # unless the majority of sources in that bin are inliers (avoids
            # accepting isolated points in noisy regimes).
            m_inst = df["m_inst"].values
            if inlier_mask.sum() >= 10:
                try:
                    bin_width = 0.5
                    mag_min, mag_max = float(np.nanmin(m_inst)), float(
                        np.nanmax(m_inst)
                    )
                    if (
                        np.isfinite(mag_min)
                        and np.isfinite(mag_max)
                        and mag_max > mag_min
                    ):
                        edges = np.arange(mag_min, mag_max + bin_width, bin_width)
                        min_bin_count = 5
                        for i in range(len(edges) - 1):
                            in_bin = (m_inst >= edges[i]) & (m_inst < edges[i + 1])
                            count = int(in_bin.sum())
                            if count < min_bin_count:
                                continue
                            frac_inlier = float(np.mean(inlier_mask[in_bin]))
                            if frac_inlier < 0.5:
                                inlier_mask[in_bin] = False
                        # No continuity constraint for linearity: allow inliers in non-contiguous
                        # magnitude bins so the detector linearity fit is not over-restricted.
                except Exception:
                    logger.debug(
                        "Linearity bin-wise refinement skipped.", exc_info=True
                    )

            df_lin = df.loc[inlier_mask].copy()
            df_out = df.loc[~inlier_mask].copy()

            if len(df_lin) < 3:
                logger.warning(
                    "Too few linear inliers after RANSAC and bin-wise filter."
                )
                return df_lin, fit_params, saturation_range

            # Recompute intercept and scatter from final selected inliers
            y_sel = df_lin["m_peak"].values - df_lin["m_inst"].values
            b = float(np.nanmean(y_sel))
            res_all = y - b
            res_sel = res_all[inlier_mask]
            sigma = _mad(res_sel)
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = np.nanstd(res_sel) if np.nanstd(res_sel) > 0 else 1e-3

            fit_params["intercept"] = float(b)
            fit_params["sigma_mad"] = float(sigma)
            n_inl = max(1, int(np.isfinite(res_sel).sum()))
            fit_params["intercept_error"] = float(1.253 * sigma / np.sqrt(n_inl))
            fit_params["n_linear_inliers"] = int(len(df_lin))

            min_flux = np.nanmin(df_lin["flux_AP"].values)
            max_flux = np.nanmax(df_lin["flux_AP"].values)
            saturation_range = [float(min_flux), float(max_flux)]

            # --- Diagnostic plot (small markers, minimal overlap) ---
            try:
                plt.ioff()
                fig, ax = plt.subplots(figsize=(8, 6))
                # All quality-cut sources (faint background)
                ax.errorbar(
                    df["m_inst"],
                    df["m_peak"],
                    xerr=df.get("m_inst_err", None),
                    yerr=df.get("m_peak_err", None),
                    fmt="o",
                    ms=1.8,
                    color="lightgray",
                    alpha=0.25,
                    lw=0.3,
                    capsize=0,
                    elinewidth=0.4,
                    label=f"All ({len(df)})",
                )
                # Selected linear inliers - small outlined markers to reduce overlap
                ax.errorbar(
                    df_lin["m_inst"],
                    df_lin["m_peak"],
                    xerr=df_lin.get("m_inst_err", None),
                    yerr=df_lin.get("m_peak_err", None),
                    fmt="o",
                    ms=2.8,
                    mfc="none",
                    mec="green",
                    ecolor="green",
                    elinewidth=0.4,
                    capsize=0,
                    alpha=0.85,
                    label=f"Selected inliers ({len(df_lin)})",
                )
                # Rejected
                ax.errorbar(
                    df_out["m_inst"],
                    df_out["m_peak"],
                    xerr=df_out.get("m_inst_err", None),
                    yerr=df_out.get("m_peak_err", None),
                    fmt="x",
                    ms=2.2,
                    color="red",
                    alpha=0.5,
                    lw=0.5,
                    capsize=0,
                    elinewidth=0.4,
                    label=f"Rejected ({len(df_out)})",
                )
                xx = np.linspace(np.nanmin(df["m_inst"]), np.nanmax(df["m_inst"]), 200)
                ax.plot(xx, xx + b, "k--", lw=1.0, label=f"m_peak = m_inst + {b:.3f}")
                ax.set_xlabel(
                    r"Instrumental magnitude [$-2.5 \log_{10}(\mathrm{Flux})$]"
                )
                ax.set_ylabel(
                    r"Peak magnitude [$-2.5 \log_{10}(\mathrm{Flux}_{\max})$]"
                )
                ax.invert_xaxis()
                ax.invert_yaxis()
                ax.grid(alpha=0.3, ls="--", lw=0.6)
                ax.legend(frameon=False, fontsize=8)
                if write_dir:
                    fpath = self.input_yaml["fpath"]
                    write_dir = self.input_yaml["write_dir"]
                    base = os.path.basename(fpath).split(".")[0]
                    pdf_out = os.path.join(write_dir, f"linear_{base}.pdf")
                    png_out = os.path.join(write_dir, f"Linear_{base}.png")
                    fig.savefig(pdf_out, bbox_inches="tight")
                    fig.savefig(png_out, bbox_inches="tight")
                plt.close(fig)
            except Exception as _pe:
                logger.debug(f"Plotting skipped: {_pe}")

            return df_lin.reset_index(drop=True), fit_params, saturation_range

        except Exception as e:
            logger.error(f"Linearity check failed: {e}")
            return catalog, fit_params, saturation_range

    def fit_gaussian(
        self,
        data: np.ndarray,
        x: Optional[float] = None,
        y: Optional[float] = None,
        dx: float = 3,
        dy: float = 3,
        sigma: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fit a 2D Gaussian to the input data using lmfit.

        Args:
            data (np.ndarray): 2D numpy array of data to fit.
            x (float, optional): Initial x center position.
            y (float, optional): Initial y center position.
            dx (float): Maximum allowed x displacement from initial position.
            dy (float): Maximum allowed y displacement from initial position.
            sigma (float, optional): Initial sigma value for both axes.

        Returns:
            dict: Dictionary containing fit results, or None on failure.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            return None
        height, width = data.shape
        if height < 3 or width < 3:
            return None
        if x is None or (isinstance(x, (int, float)) and not np.isfinite(x)):
            x = width / 2
        if y is None or (isinstance(y, (int, float)) and not np.isfinite(y)):
            y = height / 2
        x, y = float(x), float(y)
        dx = min(abs(dx), 3) if dx is not None else 3
        dy = min(abs(dy), 3) if dy is not None else 3

        n_valid = np.isfinite(data).sum()
        if n_valid < 5:
            logger.debug(
                f"fit_gaussian: too few valid pixels ({n_valid}), skipping fit."
            )
            return {
                "fwhmx": np.nan,
                "fwhmy": np.nan,
                "xfit": np.nan,
                "yfit": np.nan,
                "amplitude": np.nan,
                "success": False,
            }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ygrid, xgrid = np.mgrid[:height, :width]
                data_max = np.nanmax(data)
                data_min = np.nanmin(data)
                model = Gaussian2dModel()
                params = model.make_params()
                params["centerx"].set(
                    value=x, min=max(1, x - dx), max=(min(width - 1, x + dx))
                )
                params["centery"].set(
                    value=y, min=max(1, y - dy), max=(min(height - 1, y + dy))
                )
                params["amplitude"].set(
                    value=data_max * 0.25, min=data_min * 1e-6, max=data_max * 1e6
                )
                if sigma is not None:
                    min_sigma = 0.5 / SQRT2LOG2
                    max_sigma = 30 / SQRT2LOG2
                    params["sigmax"].set(value=sigma, min=min_sigma, max=max_sigma)
                    params["sigmay"].set(value=sigma, min=min_sigma, max=max_sigma)
                    params["sigmay"].set(expr="sigmax")
                result = model.fit(
                    data, x=xgrid, y=ygrid, params=params, nan_policy="omit"
                )
                return {
                    "fwhmx": result.params["fwhmx"].value,
                    "fwhmy": result.params["fwhmy"].value,
                    "xfit": result.params["centerx"].value,
                    "yfit": result.params["centery"].value,
                    "amplitude": result.params["height"].value,
                    "success": True,
                }
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = (
                os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                if exc_tb
                else "unknown"
            )
            lineno = exc_tb.tb_lineno if exc_tb else -1
            logger.error(
                f"Error in fit_gaussian: {exc_type} in {fname} at line {lineno}: {str(e)}"
            )
            return {
                "fwhmx": np.nan,
                "fwhmy": np.nan,
                "xfit": np.nan,
                "yfit": np.nan,
                "amplitude": np.nan,
                "success": False,
            }

    def remove_fwhm_outliers(
        self, source: pd.DataFrame, sigma: float = 5.0, maxiters: int = 25
    ) -> pd.DataFrame:
        """
        Remove outliers from the source DataFrame based on 'fwhmx' and 'fwhmy' using sigma clipping with MAD standard deviation.

        Args:
            source (pd.DataFrame): Input DataFrame with 'fwhmx' and 'fwhmy' columns.
            sigma (float): Clipping threshold in sigma units.
            maxiters (int): Maximum number of clipping iterations.

        Returns:
            pd.DataFrame: Cleaned DataFrame with outliers removed.
        """
        if source.empty:
            logger.warning("Input DataFrame is empty; nothing to clean.")
            return source
        data = np.vstack([source["fwhmx"].values, source["fwhmy"].values])
        clipped = sigma_clip(
            data,
            sigma=sigma,
            maxiters=maxiters,
            cenfunc=np.nanmedian,
            stdfunc=mad_std,
            axis=1,
        )
        combined_mask = np.any(clipped.mask, axis=0)
        num_outliers = np.count_nonzero(combined_mask)
        self.logger.info(
            f"Removed {num_outliers} outliers out of {len(source)} sources"
        )
        self.logger.info(f"Remaining sources: {len(source) - num_outliers}")
        return source.loc[~combined_mask].reset_index(drop=True)

    # =============================================================================
    #  Core Measurement and Detection
    # =============================================================================

    def measure_image(
        self,
        image: np.ndarray,
        scale: int = 31,
        fwhm: Optional[float] = None,
        sigma: Optional[float] = None,
        fwhm_initial: float = 3,
        mask_sources_XY_R: List[Tuple[float, float, float]] = [],
        dontClean: bool = False,
        initial_sources: Optional[pd.DataFrame] = None,
        default_scale: float = 5.5,
        mask: Optional[np.ndarray] = None,
        no_clean: bool = False,
    ) -> Tuple[float, pd.DataFrame, float]:
        """
        Detect point-like sources and estimate the global image FWHM.

        Args:
            image (np.ndarray): Input image array.
            scale (int): Legacy cutout half-size in pixels.
            fwhm (float, optional): If provided with `sigma`, perform a direct IRAFStarFinder run.
            sigma (float, optional): Detection threshold in multiples of background std if `fwhm` also provided.
            fwhm_initial (float): Initial guess for FWHM when auto-estimating.
            mask_sources_XY_R (list): List of circular regions to mask.
            dontClean (bool): Deprecated. Ignored in favor of `no_clean`.
            initial_sources (pd.DataFrame, optional): Optional seed sources with columns `x_pix`, `y_pix`.
            default_scale (float): Minimum allowed detection cutout scale.
            mask (np.ndarray, optional): Boolean mask: True for bad pixels.
            no_clean (bool): If True, skip filtering steps.

        Returns:
            tuple: (fwhm_global, sources_dataframe, scale_pixels)
        """
        t0 = time.time()
        logger = logging.getLogger(__name__)

        try:
            self.logger.info(border_msg("Point-source detection and FWHM measurement"))

            # --- Configuration ---
            cfg = getattr(self, "input_yaml", {}) or {}
            src_cfg = cfg.get("source_detection", {}) or {}
            scale_multiplier = float(
                src_cfg.get("scale_multiplier", src_cfg.get("scale_multipler", 5.0))
            )
            saturate = float(cfg.get("saturate", 65000.0))

            ny, nx = image.shape
            if mask is None:
                mask = ~np.isfinite(image)

            # --- Mask user-specified circles ---
            if mask_sources_XY_R:
                yy, xx = np.indices(image.shape)
                for x0, y0, r in mask_sources_XY_R:
                    circle = (xx - x0) ** 2 + (yy - y0) ** 2 <= r**2
                    mask = np.logical_or(mask, circle)

            # --- Background and noise ---
            use_bkg2d = (nx >= 64) and (ny >= 64)
            if use_bkg2d:
                box = max(32, int(min(nx, ny) // 20))
                bkg2d = Background2D(
                    image,
                    box_size=box,
                    filter_size=3,
                    bkg_estimator=MedianBackground(),
                    mask=mask,
                )
                bkg = bkg2d.background
                bkg_rms = bkg2d.background_rms
                mean, med, std = (
                    np.nanmean(bkg),
                    np.nanmedian(bkg),
                    np.nanmedian(bkg_rms),
                )
            else:
                mean, med, std = sigma_clipped_stats(image[~mask], sigma=3.0)
                bkg = np.full_like(image, med)
                bkg_rms = np.full_like(image, std)

            # --- Pre-smoothing ---
            sigma_smooth = max(0.8, 0.42466 * fwhm_initial)
            kernel = Gaussian2DKernel(
                sigma_smooth, x_size=7, y_size=7, mode="oversample"
            )
            smooth = convolve(image - bkg, kernel, normalize_kernel=True)

            # --- Direct run with provided fwhm and sigma ---
            if (fwhm is not None) and (sigma is not None):
                thr = sigma * std
                finder = IRAFStarFinder(
                    fwhm=fwhm,
                    threshold=thr,
                    minsep_fwhm=1.0,
                    exclude_border=True,
                    peakmax=0.98 * saturate,
                )
                tbl = finder(image - med, mask=mask)
                if tbl is None or len(tbl) == 0:
                    self.logger.info(
                        "No sources found with provided parameters", "warning"
                    )
                    return np.nan, pd.DataFrame(), float(max(scale, default_scale))

                df = tbl.to_pandas()
                df["x_pix"] = df["xcentroid"]
                df["y_pix"] = df["ycentroid"]
                df["s2n"] = df["peak"] / std
                fwhm_list = []
                half = int(max(default_scale, np.ceil(scale_multiplier * fwhm / 2)))
                for _, r in df.iterrows():
                    cut = Cutout2D(
                        image,
                        (float(r.x_pix), float(r.y_pix)),
                        2 * half,
                        mode="partial",
                        fill_value=np.nan,
                    ).data
                    fit = self._fit_gaussian_2d(cut)
                    fwhm_list.append(np.nanmean(fit) if fit else np.nan)
                df["fwhm"] = np.array(fwhm_list, dtype=float)

                fwhm_global = (
                    np.nanmedian(df["fwhm"])
                    if np.isfinite(df["fwhm"]).any()
                    else float(fwhm)
                )
                scale_out = float(
                    max(default_scale, np.ceil(scale_multiplier * fwhm_global))
                )
                self.logger.info(
                    f"Detected {len(df)} sources, FWHM ~ {fwhm_global:.3f} px"
                )
                self.logger.info(f"Cutout scale = {scale_out:.1f} px")
                self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
                return float(fwhm_global), df.reset_index(drop=True), scale_out

            # --- Automatic detection and FWHM estimation ---
            thr_img = detect_threshold(smooth, nsigma=5.0, mask=mask)
            thr = float(np.nanmedian(thr_img))
            fwhm_fp = max(2.0, float(fwhm_initial))
            finder = IRAFStarFinder(
                fwhm=fwhm_fp,
                threshold=thr,
                minsep_fwhm=1.0,
                roundlo=0.0,
                roundhi=1.0,
                sharplo=0.2,
                sharphi=2.0,
                exclude_border=True,
                peakmax=0.98 * saturate,
            )
            tbl = finder(smooth, mask=mask)
            if tbl is None or len(tbl) == 0:
                self.logger.info("No sources in first pass", "warning")
                self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            df = tbl.to_pandas()

            # --- Cleaning: saturation and edge ---
            df = df[df["peak"] < 0.98 * saturate]
            if len(df) == 0:
                self.logger.info("All detections saturated", "warning")
                self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            edge = int(np.ceil(3 * fwhm_fp))
            df = df[
                (df["xcentroid"] > edge)
                & (df["xcentroid"] < nx - edge)
                & (df["ycentroid"] > edge)
                & (df["ycentroid"] < ny - edge)
            ]

            if not no_clean:
                # --- Cleaning: crowding ---
                min_sep_pix = max(5.0, 2.5 * fwhm_fp)
                df = self._crowding_filter(df, min_sep_pix=min_sep_pix)

                # --- Cleaning: clip on roundness and sharpness ---
                for col in [c for c in ["roundness", "sharpness"] if c in df.columns]:
                    df = self._clip_column(df, col, sigma=5.0, maxiters=5)

            if len(df) == 0:
                self.logger.info("All detections rejected by cleaning", "warning")
                self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            # --- Per-source FWHM ---
            half = int(max(default_scale, np.ceil(scale_multiplier * fwhm_fp / 2)))
            fwhm_meas, s2n_list = [], []
            for _, r in df.iterrows():
                x0, y0 = float(r["xcentroid"]), float(r["ycentroid"])
                cut = Cutout2D(
                    image, (x0, y0), 2 * half, mode="partial", fill_value=np.nan
                ).data
                mean_c, med_c, std_c = sigma_clipped_stats(cut, sigma=3.0)
                fit = self._fit_gaussian_2d(cut)
                fwhm_meas.append(np.nanmean(fit) if fit else np.nan)
                s2n_list.append(float(r["peak"]) / max(std, 1e-12))
            df["fwhm"] = np.asarray(fwhm_meas, dtype=float)
            df["s2n"] = np.asarray(s2n_list, dtype=float)
            df["x_pix"] = df["xcentroid"].astype(float)
            df["y_pix"] = df["ycentroid"].astype(float)

            if not no_clean:
                df = df[np.isfinite(df["fwhm"])]
                if len(df) == 0:
                    self.logger.info("No finite FWHM fits", "warning")
                    self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
                    return np.nan, pd.DataFrame(), float(max(scale, default_scale))
                df = self._clip_column(df, "fwhm", sigma=5.0, maxiters=8)
                df = df[df["s2n"] >= 3.0]

            if len(df) == 0:
                self.logger.info("No sources after final quality cuts", "warning")
                self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            # --- Global FWHM and final cutout scale ---
            fwhm_global = float(np.nanmedian(df["fwhm"]))
            scale_out = float(
                max(default_scale, np.ceil(scale_multiplier * fwhm_global))
            )

            self.logger.info(f"Accepted sources: {len(df)}")
            self.logger.info(f"Image FWHM ~ {fwhm_global:.3f} px")
            self.logger.info(f"Cutout scale = {scale_out:.1f} px")
            self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
            return fwhm_global, df.reset_index(drop=True), scale_out

        except Exception as e:
            self.logger.info(f"Error in measure_image: {e!r}", "error")
            self.logger.info(f"Elapsed: {time.time() - t0:.3f} s")
            return float("nan"), pd.DataFrame(), float("nan")

    # =============================================================================
    #  Streak Detection and Masking
    # =============================================================================

    def hough_transform_streak_mask(
        self,
        data: np.ndarray,
        canny_sigma: float = 3.0,
        hough_threshold: float = 0.3,
        dilation_radius: int = 7,
        sigma_clip: float = 5.0,
        bkg: Optional[float] = None,
        sigma: Optional[float] = None,
        min_line_frac: float = 0.1,
        min_line_len_frac: float = 0.3,
        enhance_contrast: bool = True,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Detect streaks in an image using Hough transform and return a binary mask.

        Args:
            data (np.ndarray): Input 2D image array.
            canny_sigma (float): Gaussian smoothing sigma for Canny edge detection.
            hough_threshold (float): Fraction of max Hough accumulator votes required to keep a line.
            dilation_radius (int): Radius for binary dilation to broaden streak mask.
            sigma_clip (float): Minimum sigma above background required for pixels to count as streak.
            bkg (float, optional): Background level; if None, estimated via sigma clipping.
            sigma (float, optional): Background standard deviation; if None, estimated via sigma clipping.
            min_line_frac (float): Minimum fraction of pixels along line above threshold to accept.
            min_line_len_frac (float): Minimum length of line (fraction of min image dimension) to accept.
            enhance_contrast (bool): Apply adaptive histogram equalization before edge detection.

        Returns:
            tuple: (mask_broadened, bkg, sigma)
        """
        self.logger.info(
            border_msg("Searching for streaks using robust Hough Transform")
        )
        try:
            ny, nx = data.shape
            data_finite = data[np.isfinite(data)]
            if data_finite.size == 0:
                logger.warning("Input data contains no finite values.")
                return np.zeros_like(data, dtype=bool), np.nan, np.nan

            # --- Background estimation ---
            if bkg is None or sigma is None:
                bkg, _, sigma = sigma_clipped_stats(data_finite, sigma=3.0, maxiters=5)
            sigma = sigma if np.isfinite(sigma) and sigma > 0 else np.std(data_finite)
            bkg = bkg if np.isfinite(bkg) else np.median(data_finite)
            threshold = bkg + sigma_clip * sigma

            # --- Normalize image ---
            vmin, vmax = np.percentile(data_finite, [5, 99.5])
            if vmax <= vmin:
                vmax = vmin + 1.0
            data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            if enhance_contrast:
                data_norm = exposure.equalize_adapthist(data_norm, clip_limit=0.03)

            # --- Edge detection ---
            data_smooth = gaussian_filter(data_norm, sigma=1.0)
            edges = feature.canny(data_smooth, sigma=canny_sigma)

            # --- Hough transform ---
            hspace, angles, dists = transform.hough_line(edges)
            accum, angles_peaks, dists_peaks = transform.hough_line_peaks(
                hspace,
                angles,
                dists,
                threshold=hough_threshold * hspace.max(),
                min_distance=20,
                min_angle=10,
            )
            mask = np.zeros_like(data, dtype=bool)
            streak_count = 0
            min_len = min_line_len_frac * min(nx, ny)

            # --- Loop over detected lines ---
            for angle, dist in zip(angles_peaks, dists_peaks):
                x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
                dx, dy = nx * -np.sin(angle), ny * np.cos(angle)
                x1, y1 = np.clip(int(x0 + dx), 0, nx - 1), np.clip(
                    int(y0 + dy), 0, ny - 1
                )
                x2, y2 = np.clip(int(x0 - dx), 0, nx - 1), np.clip(
                    int(y0 - dy), 0, ny - 1
                )
                rr, cc = draw.line(y1, x1, y2, x2)
                if rr.size == 0:
                    continue
                line_vals = data[rr, cc]
                if (
                    np.count_nonzero(line_vals > threshold) / len(line_vals)
                    < min_line_frac
                    or np.median(line_vals) < bkg + 3 * sigma
                    or np.hypot(x2 - x1, y2 - y1) < min_len
                ):
                    continue
                mask[rr, cc] = True
                streak_count += 1

            self.logger.info(f"Number of robust streaks detected: {streak_count}")
            selem = morphology.disk(radius=dilation_radius)
            mask_broadened = morphology.binary_dilation(mask, selem)
            return mask_broadened, bkg, sigma

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = (
                os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                if exc_tb
                else "unknown"
            )
            lineno = exc_tb.tb_lineno if exc_tb else -1
            logger.exception(f"Exception {exc_type} in {fname} at line {lineno}: {e}")
            return np.zeros_like(data, dtype=bool), np.nan, np.nan

    def remove_sources_near_spikes(
        self,
        sources: Table,
        spike_mask: np.ndarray,
        x_col: str = "x_pix",
        y_col: str = "y_pix",
        radius: int = 3,
    ) -> Tuple[Table, Table]:
        """
        Remove sources that are located within a given pixel distance of spike regions.

        Args:
            sources (astropy.table.Table): Table of detected sources with x_pix and y_pix columns.
            spike_mask (np.ndarray): Binary mask where True indicates spike regions.
            x_col (str): Name of the x-coordinate column.
            y_col (str): Name of the y-coordinate column.
            radius (int): Pixel distance around spike regions to consider for exclusion.

        Returns:
            tuple: (sources_clean, sources_rejected)
        """
        if sources is None or len(sources) == 0 or spike_mask is None:
            self.logger.info("No sources provided. Returning None.")
            return sources, None

        self.logger.info(
            border_msg(f"Excluding sources within {radius} pixels of spikes")
        )
        self.logger.info(f"Total sources before filtering: {len(sources)}")

        yy, xx = sources[y_col], sources[x_col]
        y_idx = np.round(yy).astype(int)
        x_idx = np.round(xx).astype(int)
        y_idx = np.clip(y_idx, 0, spike_mask.shape[0] - 1)
        x_idx = np.clip(x_idx, 0, spike_mask.shape[1] - 1)
        radius = int(radius)
        dilated_mask = binary_dilation(
            spike_mask, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        )
        keep_mask = ~dilated_mask[y_idx, x_idx]
        sources_clean = sources[keep_mask]
        sources_rejected = sources[~keep_mask]
        self.logger.info(f"Sources kept: {len(sources_clean)}")
        self.logger.info(f"Sources rejected: {len(sources_rejected)}")
        return sources_clean, sources_rejected

    # =============================================================================
    #  Internal Helper Functions
    # =============================================================================

    def _pix_dist(
        self, x0: float, xs: np.ndarray, y0: float, ys: np.ndarray
    ) -> np.ndarray:
        """Vectorized pixel distance from one point to arrays of points."""
        return np.hypot(xs - x0, ys - y0)

    def _fit_gaussian_2d(self, cutout: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Fit a symmetric 2D Gaussian to a small cutout and return (fwhm_x, fwhm_y).
        Returns None if fit fails. Uses robust initial guesses from image moments.
        """
        data = np.array(cutout, dtype=float)
        if not np.isfinite(data).any():
            return None
        mean, med, std = sigma_clipped_stats(data, sigma=3.0)
        data = data - med
        ny, nx = data.shape
        y, x = np.mgrid[0:ny, 0:nx]
        total = np.abs(data).sum()
        if total <= 0:
            return None
        x0 = (x * np.abs(data)).sum() / total
        y0 = (y * np.abs(data)).sum() / total
        amp0 = np.nanmax(data)
        sig0 = max(1.0, 0.5 * min(nx, ny) / 6.0)
        g0 = models.Gaussian2D(
            amplitude=amp0,
            x_mean=x0,
            y_mean=y0,
            x_stddev=sig0,
            y_stddev=sig0,
            theta=0.0,
        )
        fitter = fitting.LevMarLSQFitter()
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                g = fitter(g0, x, y, data)
            fwhm_x = 2.354820045 * float(abs(g.x_stddev.value))
            fwhm_y = 2.354820045 * float(abs(g.y_stddev.value))
            if not np.isfinite(fwhm_x) or not np.isfinite(fwhm_y):
                return None
            return fwhm_x, fwhm_y
        except Exception:
            return None

    def _crowding_filter(self, df: pd.DataFrame, min_sep_pix: float) -> pd.DataFrame:
        """
        Remove sources that have a neighbor closer than min_sep_pix using KDTree.
        Keeps sources that are at least min_sep_pix from any other detection.
        """
        if len(df) < 2:
            return df
        xy = np.vstack([df["xcentroid"].values, df["ycentroid"].values]).T
        tree = cKDTree(xy)
        dists, _ = tree.query(xy, k=2)
        nn = dists[:, 1]
        keep = nn >= min_sep_pix
        return df.loc[keep].copy()

    def _clip_column(
        self, df: pd.DataFrame, col: str, sigma: float = 3.0, maxiters: int = 5
    ) -> pd.DataFrame:
        """
        Sigma-clip a numeric column and return a filtered DataFrame.
        Uses robust MAD-based std for stability.
        """
        arr = df[col].to_numpy(dtype=float)
        sc = SigmaClip(sigma=sigma, maxiters=maxiters, stdfunc=mad_std)
        mask = sc(arr).mask
        if mask is np.ma.nomask:
            return df
        keep = ~mask
        return df.loc[keep].copy()
