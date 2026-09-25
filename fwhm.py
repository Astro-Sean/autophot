#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FWHM measurement and source detection for astronomical images.

Detects point-like sources, estimates the image FWHM, and filters
detections by saturation, edge, crowding, and linearity criteria.

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
from scipy.stats import linregress
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
from photutils.background import Background2D, MedianBackground, BiweightScaleBackgroundRMS
from photutils.utils import circular_footprint
from photutils.profiles import RadialProfile
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
from lmfit.models import Gaussian2dModel, ConstantModel
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# --- Local Imports ---
from functions import (
    get_normalized_histogram,
    log_step,
    pad_ones,
    pix_dist,
    set_size,
    biweight_sky_sigma,
)

# --- Logging and Warnings ---
warnings.simplefilter("ignore", category=AstropyWarning)
logger = logging.getLogger(__name__)

# --- Constants ---
SQRT2LOG2 = 2 * np.sqrt(2 * np.log(2))  # Gaussian sigma -> FWHM


class Find_FWHM:
    """
    Detect point-like sources in an image and estimate the FWHM.
    Includes background estimation, segmentation, and sigma-clip outlier removal.
    """

    def __init__(self, input_yaml: Dict[str, Any]):
        """
        Initialize the FindFWHM class with configuration from input_yaml.

        Args:
            input_yaml (dict): Configuration parameters for the FWHM calculation.
        """
        self.input_yaml = input_yaml
        self.logger = logger
        # FWHM uncertainty (SE of median), set by measure_image().
        self.fwhm_err = np.nan

    # =============================================================================
    #  Utility Functions
    # =============================================================================

    @staticmethod
    def _adaptive_detection_params(fwhm_px: float, finder: str = "iraf") -> dict:
        """Return FWHM-adaptive finder sharpness/roundness bounds.

        Undersampled data (FWHM < 2 px) has broader intrinsic PSF shapes
        because a single pixel can contain most of the flux.  The default
        sharpness/roundness cuts are too restrictive and reject real
        detections.  See Howell (1989) sampling parameter discussion.

        DAOStarFinder sharpness uses a different normalization than
        IRAFStarFinder's: a perfectly matched Gaussian measures ~0.4,
        so the IRAF-calibrated bands would reject real stars outright.
        The DAO bands are centred on that matched value instead.

        Parameters
        ----------
        fwhm_px : float
            Estimated FWHM in pixels.
        finder : str
            'dao' or 'iraf'.

        Returns
        -------
        dict with keys: sharplo, sharphi, roundlo, roundhi
        """
        fwhm_px = float(fwhm_px) if np.isfinite(fwhm_px) else 3.0
        if finder == "dao":
            # DAO bounds stay permissive on purpose: a real PSF's
            # ellipticity and Moffat wings shift the matched-filter
            # statistics systematically (e.g. GROND PSFs measure
            # roundness ~-0.3, sharpness ~0.55), so tight bands reject
            # real stars wholesale.  The post-detection sigma-clips do
            # the calibrated rejection; these only keep out pathological
            # morphology.
            if fwhm_px < 2.0:
                return dict(sharplo=0.1, sharphi=1.6, roundlo=-1.0, roundhi=1.0)
            elif fwhm_px < 3.0:
                return dict(sharplo=0.15, sharphi=1.4, roundlo=-0.9, roundhi=0.9)
            else:
                return dict(sharplo=0.15, sharphi=1.3, roundlo=-0.8, roundhi=0.8)
        if fwhm_px < 2.0:
            # Undersampled: broader PSF tolerance, allow more ellipticity
            return dict(sharplo=0.2, sharphi=1.5, roundlo=-1.0, roundhi=1.0)
        elif fwhm_px < 3.0:
            # Critically sampled: moderate tolerance
            return dict(sharplo=0.4, sharphi=1.2, roundlo=-0.6, roundhi=0.6)
        else:
            # Well/oversampled: standard tight cuts
            return dict(sharplo=0.5, sharphi=1.0, roundlo=-0.3, roundhi=0.3)

    @staticmethod
    def _source_finder_name(src_cfg: dict) -> str:
        """Resolve the configured point-source finder to 'dao' or 'iraf'.

        Config key ``source_detection.finder`` defaults to ``dao``.
        """
        raw = str((src_cfg or {}).get("finder", "dao")).strip().lower()
        aliases = {
            "dao": "dao",
            "daofind": "dao",
            "daostarfinder": "dao",
            "iraf": "iraf",
            "starfind": "iraf",
            "irafstarfinder": "iraf",
        }
        resolved = aliases.get(raw)
        if resolved is None:
            logger.warning(
                "Unknown source_detection.finder=%r; falling back to 'dao'",
                raw,
            )
            resolved = "dao"
        return resolved

    @staticmethod
    def _build_star_finder(
        finder_name: str,
        fwhm_px: float,
        threshold,
        det_params: dict,
        saturate: float,
        ratio: float = 1.0,
        theta: float = 0.0,
    ):
        """Construct the configured point-source finder.

        DAOStarFinder convolves internally with a zero-sum Gaussian
        kernel (matched filter and local-sky subtraction in one step);
        with scale_threshold=True photutils rescales ``threshold`` onto
        the convolved-image noise scale, so it is passed in raw-image
        units and the finder runs on the unsmoothed data.  ``ratio`` and
        ``theta`` describe the kernel's ellipticity (minor/major axis
        ratio and major-axis position angle in degrees CCW from +x);
        they only matter for DAO - IRAFStarFinder has no kernel-shape
        parameters of its own.
        """
        if finder_name == "dao":
            return DAOStarFinder(
                threshold=threshold,
                fwhm=fwhm_px,
                ratio=ratio,
                theta=theta,
                sharpness_range=(det_params["sharplo"], det_params["sharphi"]),
                roundness_range=(det_params["roundlo"], det_params["roundhi"]),
                exclude_border=True,
                peak_max=0.98 * saturate,
                min_separation=fwhm_px,
                scale_threshold=True,
            )
        return IRAFStarFinder(
            fwhm=fwhm_px,
            threshold=threshold,
            min_separation=fwhm_px,
            exclude_border=True,
            peak_max=0.98 * saturate,
            sharpness_range=(det_params["sharplo"], det_params["sharphi"]),
            roundness_range=(det_params["roundlo"], det_params["roundhi"]),
        )

    @staticmethod
    def _dao_significance_map(finder, data: np.ndarray, noise_std: float):
        """Per-pixel detection-significance map for the DAO finder.

        DAOStarFinder detects on the kernel-convolved image, so a
        detection's significance is convolved_peak / (std * ||kernel||_2).
        The catalog ``peak`` column is the raw pixel value, which reads
        ~1/||k||_2 (about 3x for a matched kernel) too low and starves
        S/N cuts of real detections.
        """
        try:
            k = np.asarray(finder.kernel.data, dtype=float)
            l2 = float(np.sqrt(np.sum(k * k)))
            if not np.isfinite(l2) or l2 <= 0 or not noise_std > 0:
                return None
            filled = np.where(np.isfinite(data), data, 0.0)
            return convolve(filled, k, normalize_kernel=False) / (noise_std * l2)
        except Exception:
            return None

    @staticmethod
    def _finder_to_dataframe(tbl) -> pd.DataFrame:
        """Normalize finder output to a common column schema.

        DAOStarFinder reports per-axis roundness1/roundness2 and no
        unified ``roundness``; downstream cleaning clips a single
        ``roundness`` column, so the classical DAOFIND combination
        (the sum) is synthesized here.
        """
        df = tbl.to_pandas()
        if "roundness" not in df.columns:
            if "roundness1" in df.columns and "roundness2" in df.columns:
                df["roundness"] = df["roundness1"] + df["roundness2"]
            elif "roundness1" in df.columns:
                df["roundness"] = df["roundness1"]
        return df

    def _bright_psf_shape(
        self, image: np.ndarray, df: pd.DataFrame, half: int, n_max: int = 25
    ) -> Optional[Tuple[float, float, float]]:
        """Estimate the stellar PSF size and elongation from bright
        detections.

        The matched-filter kernel only discriminates stars from defects
        when its FWHM is near the true PSF; an undersized kernel gives
        compact junk the best response and biases every downstream
        median.  The brightest unsaturated detections are nearly always
        real point sources - defects and noise excursions concentrate at
        the detection threshold - so second moments on their cutouts
        give a kernel estimate that is insensitive to a junk-dominated
        source count.

        Returns (major-axis FWHM px, minor/major ratio, major-axis
        position angle in degrees CCW from +x), or None when too few
        cutouts yield usable moments.
        """
        _xcol = "x_centroid" if "x_centroid" in df.columns else "xcentroid"
        _ycol = "y_centroid" if "y_centroid" in df.columns else "ycentroid"
        peaks = df["peak"].values.astype(float)
        order = np.argsort(peaks)[::-1][:n_max]
        xs = df[_xcol].values.astype(float)
        ys = df[_ycol].values.astype(float)
        fwhm_maj, ratios, thetas = [], [], []
        for i in order:
            if not np.isfinite(xs[i]) or not np.isfinite(ys[i]):
                continue
            cut = Cutout2D(
                image, (xs[i], ys[i]), 2 * half, mode="partial",
                fill_value=np.nan,
            ).data
            if not np.isfinite(cut).any():
                continue
            w = np.clip(cut - np.nanmedian(cut), 0.0, None)
            w[~np.isfinite(w)] = 0.0
            # Isophotal moments: without a floor the box's noise pixels
            # carry as much weight as the star wings and the moments
            # blow up toward the box scale, especially on faint
            # sources.
            w_max = float(w.max())
            if w_max <= 0:
                continue
            w = np.where(w > 0.1 * w_max, w, 0.0)
            w_tot = float(w.sum())
            if w_tot <= 0:
                continue
            yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
            x0 = float((xx * w).sum() / w_tot)
            y0 = float((yy * w).sum() / w_tot)
            dx, dy = xx - x0, yy - y0
            mxx = float((dx * dx * w).sum() / w_tot)
            myy = float((dy * dy * w).sum() / w_tot)
            mxy = float((dx * dy * w).sum() / w_tot)
            lam = 0.5 * (mxx + myy)
            disc = np.sqrt(max(0.0, 0.25 * (mxx - myy) ** 2 + mxy * mxy))
            l1, l2 = lam + disc, lam - disc
            if l1 <= 0 or l2 <= 0:
                continue
            fmaj = 2.354820045 * np.sqrt(l1)
            # Moments from a source that fills the box are truncation-
            # dominated (merged structure or bloom wings spilling past
            # the edge) and read far too large.
            if fmaj > half:
                continue
            fwhm_maj.append(fmaj)
            ratios.append(np.sqrt(l2 / l1))
            thetas.append(np.degrees(0.5 * np.arctan2(2 * mxy, mxx - myy)))
        if len(fwhm_maj) < 3:
            return None
        return (
            float(np.median(fwhm_maj)),
            float(np.clip(np.median(ratios), 0.3, 1.0)),
            float(np.median(thetas)),
        )

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
        all_sources: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Return sources with no neighbors within `min_distance` using KDTree.

        Args:
            fwhm_table (pd.DataFrame): Table of sources with x and y coordinates.
            x_col (str): Column name for x coordinates.
            y_col (str): Column name for y coordinates.
            min_distance (float): Minimum distance (in pixels) to consider a source as isolated.
            all_sources (pd.DataFrame): Optional full raw catalog to check neighbors against.
                If provided, isolation is checked against ALL detections (including blended or
                flagged sources that were removed from fwhm_table), preventing PSF contamination
                from unfiltered neighbors.

        Returns:
            pd.DataFrame: Only sources with no neighbors within `min_distance`.
        """
        if len(fwhm_table) < 2:
            return fwhm_table

        candidate_coords = fwhm_table[[x_col, y_col]].values

        if all_sources is not None and len(all_sources) > 0 and x_col in all_sources.columns and y_col in all_sources.columns:
            neighbor_coords = all_sources[[x_col, y_col]].values
            neighbor_tree = cKDTree(neighbor_coords)
            counts = neighbor_tree.query_ball_point(candidate_coords, r=min_distance, return_length=True)
            is_isolated = counts <= 1
        else:
            tree = cKDTree(candidate_coords)
            distances, _ = tree.query(candidate_coords, k=2)
            is_isolated = distances[:, 1] > min_distance

        cleaned_sources = fwhm_table[is_isolated].reset_index(drop=True)
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
        from astropy.visualization import ZScaleInterval, ImageNormalize

        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        threshold = detect_threshold(image, nsigma=3)
        segment_map = detect_sources(image, threshold, npixels=npixels)
        if segment_map is None or segment_map.nlabels == 0:
            self.logger.info("Segmentation found no sources; returning empty table.")
            return coordinates_df.iloc[[]].copy()
        try:
            deblended_map = deblend_sources(
                image,
                segment_map,
                npixels=npixels,
                contrast=contrast,
                mode="exponential",
                progress_bar=False,
            )
        except Exception:
            # Deblending can fail on pathological segment maps; use the raw map.
            deblended_map = segment_map
        catalog = SourceCatalog(image, deblended_map)
        if len(catalog) == 0:
            return coordinates_df.iloc[[]].copy()
        coms = np.column_stack((
            catalog.x_centroid if hasattr(catalog, 'x_centroid') else catalog.xcentroid,
            catalog.y_centroid if hasattr(catalog, 'y_centroid') else catalog.ycentroid,
        ))
        segment_ids = catalog.label

        source_coords = np.column_stack(
            (coordinates_df["x_pix"], coordinates_df["y_pix"])
        )
        all_dists = np.sqrt(
            ((source_coords[:, np.newaxis, :] - coms[np.newaxis, :, :]) ** 2).sum(
                axis=-1
            )
        )
        coordinates_df = coordinates_df.reset_index(drop=True)
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
            from plotting_utils import (
                apply_autophot_mplstyle,
                get_marker_size,
                get_plot_ext,
                overlay_mask_hatch,
            )
            apply_autophot_mplstyle()
            zscale = ZScaleInterval()
            norm = ImageNormalize(image, interval=zscale)
            fig, ax = plt.subplots(figsize=set_size(540, aspect=1.3))
            cmap = plt.get_cmap("gray").copy()
            cmap.set_bad(color="none")
            ax.imshow(image, cmap=cmap, origin="lower", norm=norm)
            overlay_mask_hatch(ax, ~np.isfinite(np.asarray(image)))
            ax.contour(
                deblended_map.data,
                levels=np.unique(deblended_map.data[deblended_map.data > 0]),
                colors="#D94F4F",
                linewidths=0.5,
            )
            ax.scatter(
                cleaned_df["x_pix"],
                cleaned_df["y_pix"],
                color="blue",
                label="Isolated Sources",
                s=get_marker_size('medium'),
            )
            ax.scatter(
                coms[:, 0],
                coms[:, 1],
                    color="#00AA00",
                marker="x",
                label="Segment COMs",
                s=get_marker_size('medium'),
            )
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      frameon=False)
            fpath = self.input_yaml["fpath"]
            write_dir = self.input_yaml["write_dir"]
            base = os.path.basename(fpath).split(".")[0]
            png_out = os.path.join(write_dir, f"Segmentation_{base}{get_plot_ext(self.input_yaml)}")
            fig.savefig(png_out, bbox_inches="tight", dpi=150, facecolor="white")
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
        Detector linearity check via RANSAC on the m_peak - m_inst offset.
        For linear (unsaturated) sources that offset is a constant; saturated
        stars bend away from it.

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
            """Median absolute deviation scaled to a standard-deviation equivalent."""
            x = np.asarray(x)
            med = np.nanmedian(x)
            return 1.4826 * np.nanmedian(np.abs(x - med))

        class ConstantOffsetRegressor(BaseEstimator, RegressorMixin):
            """
            Zero-slope regressor for RANSAC: in linearity space
            delta_mag = m_peak - m_inst is a constant, so the median is a
            closed-form fit and avoids per-trial numerical optimization.
            """

            def fit(self, X: np.ndarray, y: np.ndarray):
                X, y = check_X_y(X, y)
                self.slope_ = 0.0
                self.intercept_ = float(np.nanmedian(y))
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                check_is_fitted(self)
                X = check_array(X)
                return np.full(X.shape[0], float(self.intercept_), dtype=float)

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

            # --- Fluxes -> magnitudes ---
            flux = df["flux_AP"].values.astype(float)
            peak = df["maxPixel"].values.astype(float)
            # NaN for non-positive fluxes (same convention as functions.mag)
            flux_safe = flux.astype(float).copy()
            flux_safe[flux_safe <= 0] = np.nan
            peak_safe = peak.astype(float).copy()
            peak_safe[peak_safe <= 0] = np.nan
            df["m_inst"] = -2.5 * np.log10(flux_safe)
            df["m_peak"] = -2.5 * np.log10(peak_safe)

            # --- Magnitude errors ---
            if "flux_AP_err" in df.columns and np.any(np.isfinite(df["flux_AP_err"])):
                fe = df["flux_AP_err"].values.astype(float)
                df["m_inst_err"] = (2.5 / np.log(10)) * (fe / flux_safe)
            else:
                df["m_inst_err"] = np.nan
            if "maxPixel_err" in df.columns and np.any(np.isfinite(df["maxPixel_err"])):
                pe = df["maxPixel_err"].values.astype(float)
                df["m_peak_err"] = (2.5 / np.log(10)) * (pe / peak_safe)
            else:
                df["m_peak_err"] = np.nan

            # --- RANSAC fitting ---
            delta = df["m_peak"] - df["m_inst"]
            X = df["m_inst"].values.reshape(-1, 1)
            y = delta.values
            n_pts = len(df)
            # Larger min_samples keeps RANSAC from locking onto clustered points.
            min_samples = min(15, max(5, int(0.5 * n_pts)))
            # RANSAC requires min_samples <= n_samples. For very small catalogs
            # (e.g. 4 sources), clamp to avoid ValueError.
            min_samples = min(min_samples, n_pts)
            if n_pts < 2:
                logger.warning("Too few points for linearity RANSAC (need at least 2).")
                return df.reset_index(drop=True), fit_params, saturation_range

            max_trials = int(min(500, max(100, 15 * n_pts)))

            # Fit on high-S/N sources only: low-S/N points add scatter, not
            # signal. combined_err < 0.3 mag ~ S/N > 3.6 (conservative).
            combined_err = np.sqrt(np.square(df["m_inst_err"].values) + np.square(df["m_peak_err"].values))
            snr_mask = combined_err < 0.3

            # Undersampled data: the m_peak - m_inst offset is dominated by
            # subpixel-phase sampling jitter rather than detector
            # non-linearity, so a fixed small residual threshold rejects a
            # large fraction of genuine stars.  Scale the RANSAC residual
            # threshold to the measured high-S/N offset scatter (capped at
            # 0.75 mag so genuinely saturated/non-linear sources are still
            # excluded) and skip the bin-wise majority filter below.
            phot_cfg = self.input_yaml.get("photometry", {}) or {}
            _us_fwhm_thr = float(phot_cfg.get("undersampled_fwhm_threshold", 2.5))
            try:
                _img_fwhm = float(self.input_yaml.get("fwhm", np.nan))
            except (TypeError, ValueError):
                _img_fwhm = np.nan
            undersampled = np.isfinite(_img_fwhm) and _img_fwhm <= _us_fwhm_thr
            residual_threshold_eff = float(residual_threshold)
            if undersampled:
                _ref = snr_mask if int(snr_mask.sum()) >= 5 else np.ones(n_pts, dtype=bool)
                _offset_scatter = _mad(y[_ref])
                _adaptive_thr = min(
                    0.75, max(residual_threshold_eff, 3.0 * _offset_scatter)
                )
                if _adaptive_thr > residual_threshold_eff + 1e-9:
                    residual_threshold_eff = _adaptive_thr
                    logger.info(
                        "Undersampled image (FWHM=%.2f px): relaxed linearity "
                        "residual threshold to %.2f mag (high-S/N offset "
                        "scatter %.2f mag).",
                        _img_fwhm,
                        residual_threshold_eff,
                        _offset_scatter,
                    )
            if snr_mask.sum() >= 2:
                X_snr = X[snr_mask]
                y_snr = y[snr_mask]
                ransac = RANSACRegressor(
                    estimator=ConstantOffsetRegressor(),
                    residual_threshold=residual_threshold_eff,
                    max_trials=max_trials,
                    min_samples=min(min(15, max(5, int(0.5 * len(y_snr)))), len(y_snr)),
                    random_state=42,
                )
                ransac.fit(X_snr, y_snr)
                b = float(ransac.estimator_.intercept_)
                # inlier_mask_ indexes the high-S/N subset; expand to full frame.
                inlier_mask_full = np.zeros(n_pts, dtype=bool)
                inlier_mask_full[snr_mask] = ransac.inlier_mask_
                inlier_mask = inlier_mask_full
            else:
                # Not enough high-S/N sources: fit everything.
                ransac = RANSACRegressor(
                    estimator=ConstantOffsetRegressor(),
                    residual_threshold=residual_threshold_eff,
                    max_trials=max_trials,
                    min_samples=min_samples,
                    random_state=42,
                )
                ransac.fit(X, y)
                b = float(ransac.estimator_.intercept_)
                inlier_mask = ransac.inlier_mask_.copy()

            # Inliers concentrated in a narrow magnitude range indicate a
            # biased fit; fall back to the plain median offset.
            if inlier_mask.sum() >= 10:
                mag_range = np.nanpercentile(df["m_inst"].values[inlier_mask], [5, 95])
                mag_span = mag_range[1] - mag_range[0]
                total_mag_range = np.nanpercentile(df["m_inst"].values, [5, 95])
                total_mag_span = total_mag_range[1] - total_mag_range[0]
                if total_mag_span > 0 and mag_span / total_mag_span < 0.3:
                    logger.warning(
                        f"RANSAC inliers are clustered in magnitude space (span={mag_span:.2f} vs total={total_mag_span:.2f}); using median offset instead"
                    )
                    b = float(np.nanmedian(y))
                    inlier_mask = np.abs(y - b) < residual_threshold_eff

            # Bin-wise majority filter: reject a magnitude bin unless most of
            # its sources are inliers, so isolated points in noisy bins do not
            # survive.
            m_inst = df["m_inst"].values
            if inlier_mask.sum() >= 10 and not undersampled:
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

            # Recompute the offset so the reported intercept (and plotted
            # line) matches the post-refinement inlier set.
            if int(np.sum(inlier_mask)) >= 3:
                try:
                    b = float(np.nanmedian(y[inlier_mask]))
                except Exception:
                    pass

            df_lin = df.loc[inlier_mask].copy()
            df_out = df.loc[~inlier_mask].copy()

            # If the inlier test rejects most (or nearly all) sources it is
            # not vetting stars reliably -- e.g. undersampled peak-flux
            # scatter, heteroscedastic errors, or a genuinely broad offset
            # distribution.  Keep all quality-cut sources for downstream
            # selection; saturation_range still marks the fitted linear
            # range for the catalog-level saturation check.
            min_inlier_frac = float(
                phot_cfg.get("linearity_min_inlier_frac", 0.35)
            )
            if len(df_lin) < 3 or len(df_lin) < min_inlier_frac * len(df):
                logger.warning(
                    "Linearity inlier set is too small to vet sources "
                    "reliably (%d/%d kept, min fraction %.2f); keeping all "
                    "%d quality-cut sources.",
                    len(df_lin),
                    len(df),
                    min_inlier_frac,
                    len(df),
                )
                fit_params["intercept"] = float(b)
                fit_params["n_linear_inliers"] = int(len(df_lin))
                if len(df_lin) >= 2:
                    min_flux = np.nanmin(df_lin["flux_AP"].values)
                    max_flux = np.nanmax(df_lin["flux_AP"].values)
                    saturation_range = [float(min_flux), float(max_flux)]
                return df.reset_index(drop=True), fit_params, saturation_range

            res_sel = (y - b)[inlier_mask]
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
                from plotting_utils import (
                    get_plot_ext,
                    get_ransac_color, get_marker_size, get_alpha, get_line_width,
                    apply_autophot_mplstyle, ransac_legend_top_outside, ransac_grid, ransac_savefig,
                    format_log_colorbar_ticks,
                )
                plt.ioff()
                apply_autophot_mplstyle()
                fig, ax = plt.subplots(figsize=set_size(540, 1))

                inlier_color = get_ransac_color('linearity')
                outlier_color = get_ransac_color('outliers')
                fit_color = get_ransac_color('fit')
                err_color = get_ransac_color('error_band')

                xerr_vals = df_lin["m_inst_err"].values if "m_inst_err" in df_lin.columns else None
                yerr_vals = df_lin["m_peak_err"].values if "m_peak_err" in df_lin.columns else None
                xerr = xerr_vals if xerr_vals is not None and np.any(np.isfinite(xerr_vals)) else None
                yerr = yerr_vals if yerr_vals is not None and np.any(np.isfinite(yerr_vals)) else None

                # Per-source S/N for marker colouring
                def _frame_snr(frame):
                    if "flux_AP_err" in frame.columns and np.any(np.isfinite(frame["flux_AP_err"])):
                        with np.errstate(divide="ignore", invalid="ignore"):
                            return np.abs(frame["flux_AP"].values.astype(float)) / np.maximum(
                                np.abs(frame["flux_AP_err"].values.astype(float)), 1e-12
                            )
                    if "m_inst_err" in frame.columns:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            return 1.0857 / np.abs(frame["m_inst_err"].values.astype(float))
                    return np.full(len(frame), np.nan)

                snr_lin = _frame_snr(df_lin)
                # Norm range is over inliers only -- outliers are always drawn in
                # the flat outlier colour, never the S/N colormap.
                finite_snr = snr_lin[np.isfinite(snr_lin) & (snr_lin > 0)]
                snr_norm = None
                if finite_snr.size >= 3 and np.nanmax(finite_snr) > np.nanmin(finite_snr):
                    from matplotlib.colors import LogNorm
                    vmin = max(1.0, float(np.nanmin(finite_snr)))
                    vmax = float(np.nanpercentile(finite_snr, 98))
                    if vmax <= vmin:
                        vmax = vmin * 10.0
                    snr_norm = LogNorm(vmin=vmin, vmax=vmax)

                ms_area = get_marker_size('medium') ** 2
                if len(df_out) > 0:
                    ax.errorbar(
                        df_out["m_inst"],
                        df_out["m_peak"],
                        xerr=df_out["m_inst_err"].values if "m_inst_err" in df_out.columns else None,
                        yerr=df_out["m_peak_err"].values if "m_peak_err" in df_out.columns else None,
                        fmt="none",
                        ecolor="lightgrey",
                        alpha=get_alpha('medium'),
                        capsize=get_marker_size('medium') / 4,
                        elinewidth=0.5,
                        zorder=2,
                    )
                    ax.plot(
                        df_out["m_inst"], df_out["m_peak"],
                        "x", ms=get_marker_size('medium'),
                        color=outlier_color, alpha=get_alpha('medium'),
                        label=f"Outliers [{len(df_out)}]",
                        zorder=3,
                    )
                ax.errorbar(
                    df_lin["m_inst"],
                    df_lin["m_peak"],
                    xerr=xerr,
                    yerr=yerr,
                    fmt="none",
                    ecolor="lightgrey",
                    alpha=get_alpha('dark'),
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    zorder=4,
                )
                if snr_norm is not None:
                    sc_in = ax.scatter(
                        df_lin["m_inst"], df_lin["m_peak"],
                        c=snr_lin, cmap="viridis", norm=snr_norm,
                        marker="o", s=ms_area,
                        alpha=get_alpha('dark'),
                        label=f"Inliers [{len(df_lin)}]",
                        zorder=5,
                    )
                    cb = fig.colorbar(sc_in, ax=ax, label="S/N", pad=0.02)
                    format_log_colorbar_ticks(cb, snr_norm.vmin, snr_norm.vmax)
                else:
                    ax.plot(
                        df_lin["m_inst"], df_lin["m_peak"],
                        "o", ms=get_marker_size('medium'),
                        color=inlier_color, alpha=get_alpha('dark'),
                        label=f"Inliers [{len(df_lin)}]",
                    )
                xx = np.linspace(np.nanmin(df["m_inst"]), np.nanmax(df["m_inst"]), 200)
                yy = xx + b
                intercept_error = fit_params.get("intercept_error", 0.0)
                ax.fill_between(
                    xx, yy - intercept_error, yy + intercept_error,
                    color=err_color, alpha=get_alpha('very_light'),
                )
                ax.plot(xx, yy, color=fit_color, linestyle="--", lw=get_line_width('medium'),
                        label=f"Fit: $m_{{\\mathrm{{peak}}}} = m_{{\\mathrm{{inst}}}} + {b:.3f}$")
                
                # Saturation marker: only drawn when data reach ~90% of it.
                # Fallback constant shared with main.py.
                try:
                    from main import SATURATE_INTERNAL_FALLBACK
                except ImportError:
                    SATURATE_INTERNAL_FALLBACK = np.inf

                saturate = self.input_yaml.get("saturate", SATURATE_INTERNAL_FALLBACK)
                if np.isfinite(saturate) and saturate > 0:
                    # SATURATE is in ADU/pixel but maxPixel/m_peak use the
                    # pipeline flux convention (e-/s). Convert so the marker
                    # lands at the correct position on the m_inst axis.
                    try:
                        from aperture import (
                            resolve_exposure_time_seconds,
                            resolve_gain_e_per_adu,
                        )
                        _gain = resolve_gain_e_per_adu(None, self.input_yaml)
                        _expt = resolve_exposure_time_seconds(None, self.input_yaml)
                        saturate_rate = saturate * _gain / _expt
                    except Exception:
                        saturate_rate = saturate
                    saturate_safe = max(saturate_rate, 1e-10)
                    saturate_mag = -2.5 * np.log10(saturate_safe)

                    peak_values = df["maxPixel"].values
                    if np.any(peak_values > 0.9 * saturate_rate):
                        xlim = ax.get_xlim()
                        if xlim[0] <= saturate_mag <= xlim[1]:
                            ax.axvline(saturate_mag, color='gray', linestyle=':', lw=get_line_width('medium'),
                                      alpha=get_alpha('medium'), label=f'Saturation ({saturate:.0f} ADU)')

                            # Axes are inverted below, so xlim[1] is the
                            # bright (saturated) side.
                            ax.axvspan(saturate_mag, xlim[1], color='gray',
                                       alpha=get_alpha('very_light'), label='Saturation regime')
                
                ax.set_xlabel(
                    r"Instrumental magnitude $m_\mathrm{inst}$ "
                    r"[$-2.5\,\log_{10}(\mathrm{Flux}_{e^-/s})$]"
                )
                ax.set_ylabel(
                    r"Peak magnitude $m_\mathrm{peak}$ "
                    r"[$-2.5\,\log_{10}(\mathrm{maxPixel}_{e^-/s})$]"
                )
                ax.invert_xaxis()
                ax.invert_yaxis()
                ransac_grid(ax)
                ransac_legend_top_outside(
                    ax, ncol=max(1, len(ax.get_legend_handles_labels()[0]))
                )
                if write_dir:
                    fpath = self.input_yaml["fpath"]
                    _write_dir = self.input_yaml["write_dir"]
                    base = os.path.basename(fpath).split(".")[0]
                    png_out = os.path.join(_write_dir, f"Linearity_{base}{get_plot_ext(self.input_yaml)}")
                    ransac_savefig(fig, png_out)
                plt.close(fig)
            except Exception as _pe:
                logger.debug("Plotting skipped: %s", _pe)

            return df_lin.reset_index(drop=True), fit_params, saturation_range

        except Exception as e:
            logger.error("Linearity check failed: %s", e)
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
                data_median = float(np.nanmedian(data))
                # Composite model: 2D Gaussian + constant background so the fit
                # does not try to absorb background into the Gaussian amplitude.
                model = Gaussian2dModel() + ConstantModel()
                params = model.make_params()
                params["centerx"].set(
                    value=x, min=max(1, x - dx), max=(min(width - 1, x + dx))
                )
                params["centery"].set(
                    value=y, min=max(1, y - dy), max=(min(height - 1, y + dy))
                )
                # Amplitude init: source peak above the fitted background
                # (data_max - median), not a fraction of the raw max which is
                # wrong when the background dominates.  Bounds must span the
                # data range in EITHER sign: on difference images the source
                # can be entirely negative (data_min < data_max < 0), where
                # min=data_min*1e-6 > max=data_max*1e6 would invert the bounds
                # and break the fit.
                amp0 = float(data_max - data_median)
                if not np.isfinite(amp0) or amp0 == 0.0:
                    amp0 = float(data_max) if np.isfinite(data_max) else 1.0
                amp_span = max(abs(float(data_max)), abs(float(data_min)), abs(amp0), 1.0)
                params["amplitude"].set(
                    value=amp0, min=-10.0 * amp_span, max=10.0 * amp_span
                )
                params["c"].set(
                    value=data_median, min=data_min - abs(data_max), max=data_max + abs(data_max)
                )
                if sigma is not None:
                    min_sigma = 0.5 / SQRT2LOG2
                    _max_sigma_px = float(
                        (self.input_yaml or {}).get("fwhm_max_sigma_px", 30)
                    )
                    max_sigma = _max_sigma_px / SQRT2LOG2
                    params["sigmax"].set(value=sigma, min=min_sigma, max=max_sigma)
                    params["sigmay"].set(value=sigma, min=min_sigma, max=max_sigma)
                    params["sigmay"].set(expr="sigmax")
                result = model.fit(
                    data, x=xgrid, y=ygrid, params=params, nan_policy="omit"
                )
                return {
                    "fwhmx": result.params["sigmax"].value * SQRT2LOG2,
                    "fwhmy": result.params["sigmay"].value * SQRT2LOG2,
                    "xfit": result.params["centerx"].value,
                    "yfit": result.params["centery"].value,
                    "amplitude": result.params["amplitude"].value,
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
        self.logger.info("Remaining sources: %s", len(source) - num_outliers)
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
            self.logger.info(log_step("SExtractor: detect sources & FWHM"))

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
                    bkg_rms_estimator=BiweightScaleBackgroundRMS(),
                    sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
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
                mean, med, _ = sigma_clipped_stats(image[~mask], sigma=3.0)
                std = biweight_sky_sigma(image, mask=mask)
                bkg = np.full_like(image, med)
                bkg_rms = np.full_like(image, std)

            # --- Direct run with provided fwhm and sigma ---
            if (fwhm is not None) and (sigma is not None):
                finder_name = self._source_finder_name(src_cfg)
                _det_params = self._adaptive_detection_params(fwhm, finder_name)
                thr = sigma * std
                if finder_name == "dao":
                    # Convolved noise excursions mimic stellar morphologies
                    # under the matched filter, so DAOFIND needs ~4 sigma on
                    # the convolved scale where IRAF needed 3 on raw pixels.
                    # scale_threshold converts this raw-unit threshold.
                    _dao_nsigma = float(src_cfg.get("dao_nsigma", 4.0))
                    thr = max(sigma, _dao_nsigma) * std
                finder = self._build_star_finder(
                    finder_name, fwhm, thr, _det_params, saturate
                )
                tbl = finder(image - med, mask=mask)
                if tbl is None or len(tbl) == 0:
                    self.logger.info(
                        "No sources found with provided parameters", "warning"
                    )
                    return np.nan, pd.DataFrame(), float(max(scale, default_scale))

                df = self._finder_to_dataframe(tbl)
                _xcol = "x_centroid" if "x_centroid" in df.columns else "xcentroid"
                _ycol = "y_centroid" if "y_centroid" in df.columns else "ycentroid"
                df["x_pix"] = df[_xcol]
                df["y_pix"] = df[_ycol]
                _xs = df["x_pix"].values.astype(float)
                _ys = df["y_pix"].values.astype(float)
                std_safe = np.maximum(std, 1e-12)
                if finder_name == "dao":
                    # Significance on the convolved detection image, not
                    # raw peak/std (see auto path for why).
                    _dao_sig = self._dao_significance_map(
                        finder, image - med, float(std)
                    )
                    if _dao_sig is not None:
                        _xi = np.clip(np.rint(_xs).astype(int), 0, nx - 1)
                        _yi = np.clip(np.rint(_ys).astype(int), 0, ny - 1)
                        df["s2n"] = _dao_sig[_yi, _xi]
                    else:
                        df["s2n"] = df["peak"] / std_safe
                else:
                    df["s2n"] = df["peak"] / std_safe
                fwhm_list = []
                half = int(max(default_scale, np.ceil(scale_multiplier * fwhm / 2)))
                for i in range(len(df)):
                    cut = Cutout2D(
                        image,
                        (_xs[i], _ys[i]),
                        2 * half,
                        mode="partial",
                        fill_value=np.nan,
                    ).data
                    fit = self._fit_gaussian_2d(cut)
                    if fit is not None and all(np.isfinite(v) for v in fit):
                        fwhm_list.append(float(np.mean(fit)))
                    else:
                        fwhm_list.append(np.nan)
                df["fwhm"] = np.array(fwhm_list, dtype=float)

                # Same elite-subset estimate as the auto path: the
                # near-threshold population is dominated by sub-stellar
                # junk that drags a flat median low.
                _est_lo = float(src_cfg.get("fwhm_est_band_lo", 0.5))
                _est_hi = float(src_cfg.get("fwhm_est_band_hi", 2.0))
                _est_s2n = float(src_cfg.get("fwhm_est_s2n_min", 8.0))
                _elite = df[
                    (df["s2n"] >= _est_s2n)
                    & (df["fwhm"] >= _est_lo * fwhm)
                    & (df["fwhm"] <= _est_hi * fwhm)
                ]
                _fwhm_src = _elite if len(_elite) >= 3 else df
                fwhm_global = (
                    np.nanmedian(_fwhm_src["fwhm"])
                    if np.isfinite(_fwhm_src["fwhm"]).any()
                    else float(fwhm)
                )
                scale_out = float(
                    max(default_scale, np.ceil(scale_multiplier * fwhm_global))
                )
                self.logger.info(
                    "Detected %d sources (%s), FWHM ~ %.3f px",
                    len(df), finder_name, fwhm_global,
                )
                self.logger.info("Cutout scale = %.1f px", scale_out)
                self.logger.info("Elapsed: %.3f s", time.time() - t0)
                return float(fwhm_global), df.reset_index(drop=True), scale_out

            # --- Automatic detection and FWHM estimation ---
            finder_name = self._source_finder_name(src_cfg)
            fwhm_fp = max(2.0, float(fwhm_initial))
            kernel_ratio, kernel_theta = 1.0, 0.0
            df = None
            # The detection kernel is only matched to the data when
            # fwhm_fp ~ the true PSF size.  An undersized kernel gives
            # compact junk the best matched response, so the detection
            # population - and every median derived from it - becomes
            # defect-dominated.  After a first pass, re-estimate the
            # kernel size (and, for DAO, ellipticity) from the brightest
            # detections and re-detect when the guess was badly off.
            for _det_pass in range(3):
                # Pre-smoothing (only needed on this path -- the direct run
                # above operates on the unsmoothed image).
                sigma_smooth = max(0.8, 0.42466 * fwhm_fp)
                kernel = Gaussian2DKernel(
                    sigma_smooth, x_size=7, y_size=7, mode="oversample"
                )
                smooth = convolve(image - bkg, kernel, normalize_kernel=True)
                # Smoothing suppresses the per-pixel noise by the kernel's
                # L2 norm: std_smooth = std * sqrt(sum(kernel^2)).
                # Needed when comparing smoothed-image peaks against the
                # detection S/N.
                kernel_l2 = float(
                    np.sqrt(np.sum(np.asarray(kernel.array, dtype=float) ** 2))
                )
                std_smooth = (
                    float(std) * kernel_l2
                    if np.isfinite(kernel_l2) and kernel_l2 > 0
                    else float(std)
                )

                # Use 3.0 sigma to match SExtractor's default detection
                # threshold.  The old 5.0 sigma was too high, producing far
                # fewer sources than SExtractor and making the pythonic
                # fallback unreliable.
                thr_img = detect_threshold(smooth, n_sigma=3.0, mask=mask)
                # photutils >=3.0 supports spatially varying 2D threshold
                # arrays.  Use the full 2D threshold image for better
                # detection near chip gaps/gradients.
                _det_params = self._adaptive_detection_params(
                    fwhm_fp, finder_name
                )
                if finder_name == "dao":
                    # DAO applies its own matched Gaussian kernel
                    # internally, so it runs on the unsmoothed image.  The
                    # threshold is the raw-image noise map;
                    # scale_threshold rescales it onto the kernel-convolved
                    # noise scale.  Convolved noise excursions mimic
                    # stellar morphologies, so the canonical DAOFIND
                    # detection level is ~4 sigma rather than the 3 sigma
                    # used on the pre-smoothed image; a matched star at
                    # raw S/N ~5 still passes because the convolution
                    # boosts its significance by ~1/l2.
                    _dao_nsigma = float(src_cfg.get("dao_nsigma", 4.0))
                    thr_det = detect_threshold(
                        image - bkg, n_sigma=_dao_nsigma, mask=mask
                    )
                    det_input = image - bkg
                else:
                    thr_det = thr_img
                    det_input = smooth
                finder = self._build_star_finder(
                    finder_name, fwhm_fp, thr_det, _det_params, saturate,
                    ratio=kernel_ratio, theta=kernel_theta,
                )
                tbl = finder(det_input, mask=mask)
                if tbl is None or len(tbl) == 0:
                    if _det_pass == 0 and fwhm_fp > 4.0:
                        # An oversized matched kernel detects nothing;
                        # retry at a smaller scale before giving up.
                        fwhm_fp *= 0.5
                        continue
                    self.logger.warning("No sources in first pass")
                    self.logger.info("Elapsed: %.3f s", time.time() - t0)
                    return np.nan, pd.DataFrame(), float(max(scale, default_scale))

                df = self._finder_to_dataframe(tbl)

                if _det_pass < 2:
                    # The moment box must cover a PSF much larger than
                    # the current guess, or every usable bright star is
                    # rejected as box-filling and the estimate is lost;
                    # but an oversized box integrates bloom wings and
                    # neighbors and inflates the moments instead.
                    _shape = self._bright_psf_shape(
                        image, df,
                        half=int(np.clip(3 * fwhm_fp, 20, 25)),
                    )
                    if _shape is not None:
                        _f_new, _r_new, _t_new = _shape
                        if _f_new > 0 and not (
                            0.75 <= _f_new / fwhm_fp <= 1.33
                        ):
                            self.logger.info(
                                "Refining detection kernel: FWHM %.1f -> "
                                "%.1f px, axis ratio %.2f",
                                fwhm_fp, _f_new, _r_new,
                            )
                            fwhm_fp = float(_f_new)
                            if finder_name == "dao":
                                kernel_ratio = _r_new
                                kernel_theta = _t_new
                            continue
                break

            if df is None or len(df) == 0:
                self.logger.warning("No sources in first pass")
                self.logger.info("Elapsed: %.3f s", time.time() - t0)
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            # --- Cleaning: saturation and edge ---
            df = df[df["peak"] < 0.98 * saturate]
            if len(df) == 0:
                self.logger.warning("All detections saturated")
                self.logger.info("Elapsed: %.3f s", time.time() - t0)
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            edge = int(np.ceil(3 * fwhm_fp))
            _xcol = "x_centroid" if "x_centroid" in df.columns else "xcentroid"
            _ycol = "y_centroid" if "y_centroid" in df.columns else "ycentroid"
            df = df[
                (df[_xcol] > edge)
                & (df[_xcol] < nx - edge)
                & (df[_ycol] > edge)
                & (df[_ycol] < ny - edge)
            ]

            if not no_clean:
                # --- Cleaning: mask proximity ---
                # Exclude sources within X FWHM of masked regions (chip gaps, bad pixels, etc.)
                mask_buffer_fwhm = src_cfg.get("mask_buffer_fwhm", 2.0)
                if mask is not None and mask_buffer_fwhm > 0 and len(df) > 0:
                    from scipy import ndimage
                    # binary_dilation iterations = buffer radius in pixels.
                    buffer_px = int(mask_buffer_fwhm * fwhm_fp)
                    dilated_mask = ndimage.binary_dilation(mask, iterations=buffer_px)

                    _xcol = "x_centroid" if "x_centroid" in df.columns else "xcentroid"
                    _ycol = "y_centroid" if "y_centroid" in df.columns else "ycentroid"
                    source_coords = np.round(df[[_xcol, _ycol]].values).astype(int)
                    source_coords[:, 0] = np.clip(source_coords[:, 0], 0, dilated_mask.shape[1] - 1)
                    source_coords[:, 1] = np.clip(source_coords[:, 1], 0, dilated_mask.shape[0] - 1)

                    near_mask = dilated_mask[source_coords[:, 1], source_coords[:, 0]]
                    n_near_mask = near_mask.sum()
                    
                    if n_near_mask > 0:
                        self.logger.info(
                            f"Removing {int(n_near_mask)} sources within {mask_buffer_fwhm} FWHM "
                            f"of masked/chip gap regions"
                        )
                        df = df[~near_mask]

                # --- Cleaning: crowding ---
                min_sep_pix = max(5.0, 2.5 * fwhm_fp)
                df = self._crowding_filter(df, min_sep_pix=min_sep_pix)

                # --- Cleaning: clip on roundness and sharpness ---
                for col in [c for c in ["roundness", "sharpness"] if c in df.columns]:
                    df = self._clip_column(df, col, sigma=5.0, maxiters=5)

            if len(df) == 0:
                self.logger.warning("All detections rejected by cleaning")
                self.logger.info("Elapsed: %.3f s", time.time() - t0)
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            # --- Per-source FWHM ---
            half = int(max(default_scale, np.ceil(scale_multiplier * fwhm_fp / 2)))
            fwhm_meas, s2n_list = [], []
            _xcol = "x_centroid" if "x_centroid" in df.columns else "xcentroid"
            _ycol = "y_centroid" if "y_centroid" in df.columns else "ycentroid"
            _xs = df[_xcol].values.astype(float)
            _ys = df[_ycol].values.astype(float)
            _peaks = df["peak"].values.astype(float)
            # For DAO, significance must be measured on the convolved
            # image the finder detected on; the raw peak/std underestimates
            # it ~3x and would wrongly reject nearly every detection.
            _dao_signif = (
                self._dao_significance_map(finder, det_input, float(std))
                if finder_name == "dao"
                else None
            )
            _s2n_r = max(1, int(round(0.3 * fwhm_fp)))
            for i in range(len(df)):
                cut = Cutout2D(
                    image, (_xs[i], _ys[i]), 2 * half, mode="partial", fill_value=np.nan
                ).data
                fit = self._fit_gaussian_2d(cut)
                if fit is not None and all(np.isfinite(v) for v in fit):
                    fwhm_meas.append(float(np.mean(fit)))
                else:
                    fwhm_meas.append(np.nan)
                if _dao_signif is not None:
                    # Centroid can sit ~1 px off the convolved peak;
                    # sample the local max within a small window.
                    _px = int(np.clip(np.rint(_xs[i]), 0, nx - 1))
                    _py = int(np.clip(np.rint(_ys[i]), 0, ny - 1))
                    s2n_list.append(
                        float(
                            np.nanmax(
                                _dao_signif[
                                    max(0, _py - _s2n_r) : min(
                                        ny, _py + _s2n_r + 1
                                    ),
                                    max(0, _px - _s2n_r) : min(
                                        nx, _px + _s2n_r + 1
                                    ),
                                ]
                            )
                        )
                    )
                else:
                    # IRAF peaks are measured on the smoothed image, so
                    # compare against the smoothed noise scale.
                    s2n_list.append(_peaks[i] / max(std_smooth, 1e-12))
            df["fwhm"] = np.asarray(fwhm_meas, dtype=float)
            df["s2n"] = np.asarray(s2n_list, dtype=float)
            df["x_pix"] = df[_xcol].astype(float)
            df["y_pix"] = df[_ycol].astype(float)

            if not no_clean:
                df = df[np.isfinite(df["fwhm"])]
                if len(df) == 0:
                    self.logger.warning("No finite FWHM fits")
                    self.logger.info("Elapsed: %.3f s", time.time() - t0)
                    return np.nan, pd.DataFrame(), float(max(scale, default_scale))
                df = self._clip_column(df, "fwhm", sigma=5.0, maxiters=8)
                df = df[df["s2n"] >= 3.0]

            if len(df) == 0:
                self.logger.warning("No sources after final quality cuts")
                self.logger.info("Elapsed: %.3f s", time.time() - t0)
                return np.nan, pd.DataFrame(), float(max(scale, default_scale))

            # --- Global FWHM and final cutout scale ---
            # The detection population near threshold is dominated by
            # sub-stellar junk (convolved noise excursions, compact
            # defects), which drags a flat median low.  Estimate the image
            # FWHM from high-significance detections whose fitted size is
            # consistent with the detection kernel, falling back to the
            # full set when that subset is too sparse.
            _est_lo = float(src_cfg.get("fwhm_est_band_lo", 0.5))
            _est_hi = float(src_cfg.get("fwhm_est_band_hi", 2.0))
            _est_s2n = float(src_cfg.get("fwhm_est_s2n_min", 8.0))
            _elite = df[
                (df["s2n"] >= _est_s2n)
                & (df["fwhm"] >= _est_lo * fwhm_fp)
                & (df["fwhm"] <= _est_hi * fwhm_fp)
            ]
            _fwhm_src = _elite if len(_elite) >= 3 else df
            if len(_elite) >= 3 and len(_elite) < len(df):
                self.logger.info(
                    "FWHM estimate uses %d high-S/N kernel-consistent "
                    "sources (s2n>=%.0f, fwhm within [%.1f, %.1f]x kernel) "
                    "out of %d detections",
                    len(_elite), _est_s2n, _est_lo, _est_hi, len(df),
                )
            fwhm_global = float(np.nanmedian(_fwhm_src["fwhm"]))
            # FWHM uncertainty: standard error of the median,
            # SE_median = 1.858 * MAD / sqrt(N) (same convention as the
            # zeropoint and aperture-correction errors). Captures star-to-star
            # scatter (PSF variation, fitting noise). N < 2 -> NaN.
            _fwhm_finite = _fwhm_src["fwhm"].values[
                np.isfinite(_fwhm_src["fwhm"].values)
            ]
            _n_fwhm = len(_fwhm_finite)
            if _n_fwhm >= 2:
                _fwhm_mad = float(np.nanmedian(np.abs(_fwhm_finite - fwhm_global)))
                fwhm_err = float(1.858 * _fwhm_mad / np.sqrt(_n_fwhm))
            else:
                fwhm_err = np.nan
            # Store on the instance for callers that check self.fwhm_err
            self.fwhm_err = fwhm_err

            scale_out = float(
                max(default_scale, np.ceil(scale_multiplier * fwhm_global))
            )

            self.logger.info("Accepted sources: %s (finder=%s)", len(df), finder_name)
            self.logger.info(
                "Image FWHM ~ %.3f +/- %.3f px (N=%d, SE of median)",
                fwhm_global, fwhm_err if np.isfinite(fwhm_err) else float("nan"),
                _n_fwhm,
            )
            self.logger.info("Cutout scale = %.1f px", scale_out)
            self.logger.info("Elapsed: %.3f s", time.time() - t0)
            return fwhm_global, df.reset_index(drop=True), scale_out

        except Exception as e:
            self.logger.error("Error in measure_image: %s", e)
            self.logger.info("Elapsed: %.3f s", time.time() - t0)
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
            log_step("Streak search (Hough)")
        )
        try:
            ny, nx = data.shape
            data_finite = data[np.isfinite(data)]
            if data_finite.size == 0:
                logger.warning("Input data contains no finite values.")
                return np.zeros_like(data, dtype=bool), np.nan, np.nan

            # --- Background estimation ---
            if bkg is None:
                _, bkg, _ = sigma_clipped_stats(data_finite, sigma=3.0, maxiters=5)
            if sigma is None:
                sigma = biweight_sky_sigma(data_finite)
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
                if len(line_vals) == 0:
                    continue
                if (
                    np.count_nonzero(line_vals > threshold) / len(line_vals)
                    < min_line_frac
                    or np.median(line_vals) < bkg + 3 * sigma
                    or np.hypot(x2 - x1, y2 - y1) < min_len
                ):
                    continue
                mask[rr, cc] = True
                streak_count += 1

            self.logger.info("Number of robust streaks detected: %s", streak_count)
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
            logger.exception("Exception %s in %s at line %s: %s", exc_type, fname, lineno, e)
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
            log_step(f"Exclude sources near streaks: {radius} px")
        )
        self.logger.info("Total sources before filtering: %s", len(sources))

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
        self.logger.debug(
            "Spike-mask source filter: kept %s, rejected %s",
            len(sources_clean), len(sources_rejected),
        )
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
        Estimate FWHM from a small cutout.

        Primary method (photutils >=3.0): fit a Moffat profile to the radial
        profile using RadialProfile.moffat_fwhm - more accurate for PSF wings.
        Fallback: fit a 2D Gaussian via LevMarLSQFitter.
        """
        data = np.array(cutout, dtype=float)
        if not np.isfinite(data).any():
            return None
        mean, med, std = sigma_clipped_stats(data, sigma=3.0)
        data = data - med
        ny, nx = data.shape
        total = np.abs(data).sum()
        if total <= 0:
            return None
        y_arr, x_arr = np.mgrid[0:ny, 0:nx]
        # Centroid weights: positive part of the background-subtracted data.
        # |data| also works but lets negative noise pixels pull the centroid
        # toward the cutout centre on faint sources; positive weights are the
        # standard choice.  Fall back to |data| when nothing is positive
        # (e.g. negative-residual cutouts on difference images).
        w_pos = np.clip(data, 0.0, None)
        w_tot = float(w_pos.sum())
        if w_tot > 0 and np.isfinite(w_tot):
            w_cen = w_pos
            w_sum = w_tot
        else:
            w_cen = np.abs(data)
            w_sum = total
        x0 = (x_arr * w_cen).sum() / w_sum
        y0 = (y_arr * w_cen).sum() / w_sum

        # --- Primary: Moffat radial profile fit (photutils 3.0) ---
        try:
            xycen = np.array([x0, y0])
            max_r = 0.5 * min(nx, ny)
            edge_radii = np.arange(0, max_r + 1, 0.5)
            rp = RadialProfile(data, xycen, edge_radii, mask=~np.isfinite(data))
            moffat_fwhm = rp.moffat_fwhm
            if np.isfinite(moffat_fwhm) and moffat_fwhm > 0:
                return float(moffat_fwhm), float(moffat_fwhm)
        except Exception:
            pass

        # --- Fallback: 2D Gaussian fit ---
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
                g = fitter(g0, x_arr, y_arr, data)
            fwhm_x = 2.354820045 * float(abs(g.x_stddev.value))
            fwhm_y = 2.354820045 * float(abs(g.y_stddev.value))
            if not np.isfinite(fwhm_x) or not np.isfinite(fwhm_y):
                return None
            return fwhm_x, fwhm_y
        except Exception:
            return None

    def _crowding_filter(self, df: pd.DataFrame, min_sep_pix: float) -> pd.DataFrame:
        """Drop sources with a neighbor closer than min_sep_pix (KDTree)."""
        if len(df) < 2:
            return df
        _xcol = "x_centroid" if "x_centroid" in df.columns else "xcentroid"
        _ycol = "y_centroid" if "y_centroid" in df.columns else "ycentroid"
        xy = np.vstack([df[_xcol].values, df[_ycol].values]).T
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
        MAD-based std so outliers do not inflate the clip scale.
        """
        arr = df[col].to_numpy(dtype=float)
        sc = SigmaClip(sigma=sigma, maxiters=maxiters, stdfunc=mad_std)
        mask = sc(arr).mask
        if mask is np.ma.nomask:
            return df
        keep = ~mask
        return df.loc[keep].copy()
