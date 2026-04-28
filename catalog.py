#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:47:32 2022
@author: seanbrennan
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import os
import sys
import logging
import pathlib
import warnings
import shutil
import requests
import random
import string
from functools import reduce
from math import ceil
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize
from scipy.ndimage import gaussian_laplace

# Astropy ecosystem imports
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip, mad_std
from astropy.table import Table
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS

from astroquery.vizier import Vizier

# Photutils imports
from photutils.centroids import centroid_sources, centroid_com, centroid_2dg

# Scikit-learn imports
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


import traceback

# Local imports
from functions import (
    AutophotYaml,
    log_step,
    pix_dist,
    mag,
    snr,
    set_size,
    normalize_photometric_filter_name,
    parse_supported_filter_group_key,
    log_warning_from_exception,
)
from aperture import Aperture

# Initialize logger
logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================


def cross_match_sources(given_catalog, variable_catalog, match_radius_pix=5):
    """
    Remove sources from given_catalog that match any source in variable_catalog within match_radius_pix pixels.

    Parameters:
    -----------
    given_catalog : pd.DataFrame
        DataFrame with ['x_pix', 'y_pix'] columns.
    variable_catalog : pd.DataFrame
        DataFrame with ['x_pix', 'y_pix', 'otype'] columns.
    match_radius_pix : float, optional
        Matching radius in pixels (default is 5).

    Returns:
    --------
    filtered_catalog : pd.DataFrame
        DataFrame with matched sources removed.
    """
    if len(variable_catalog) == 0:
        return given_catalog

    x_given = given_catalog["x_pix"].values
    y_given = given_catalog["y_pix"].values
    x_var = variable_catalog["x_pix"].values
    y_var = variable_catalog["y_pix"].values

    keep_mask = np.ones(len(given_catalog), dtype=bool)
    removed_indices = []

    for i, (xg, yg) in enumerate(zip(x_given, y_given)):
        dx = x_var - xg
        dy = y_var - yg
        dist = np.sqrt(dx**2 + dy**2)
        within_radius = dist <= match_radius_pix

        if np.any(within_radius):
            matched_var_idx = np.where(within_radius)[0][
                0
            ]  # First matched variable source index
            keep_mask[i] = False
            removed_indices.append(i)
            otype = variable_catalog.iloc[matched_var_idx]["OTYPE_opt"]
            logger.debug(
                f"Removing source at index {i} (x={xg:.2f}, y={yg:.2f}) due to match with variable source [{otype}]"
            )

    filtered_catalog = given_catalog[keep_mask].reset_index(drop=True)
    logger.info(f"Number of sources removed due to matching: {len(removed_indices)}")
    logger.info(f"Number of sources remaining after filtering: {len(filtered_catalog)}")

    return filtered_catalog


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================
class Catalog:
    def __init__(self, input_yaml):
        """
        Initialize the catalog class with the input YAML configuration.

        Parameters:
        -----------
        input_yaml : dict
            Configuration dictionary loaded from YAML.
        """
        self.input_yaml = input_yaml

    def _require_catalog_selected(self, catalogName: Optional[str]) -> str:
        """
        Ensure a catalog backend is selected.

        The pipeline historically allowed `catalog.use_catalog` to be null for
        workflows that do not require catalog calibration, but when a catalog
        query is requested we must fail fast with a clear message.
        """
        catalogName = self._resolve_catalog_for_filter(catalogName)
        if (
            catalogName is None
            or str(catalogName).strip() == ""
            or str(catalogName).lower() == "none"
        ):
            raise ValueError(
                "No catalog selected. Set `default_input.catalog.use_catalog` in your YAML "
                "(e.g. 'gaia', 'panstarrs'/'pan_starrs', 'sdss', 'apass', '2mass', 'legacy', 'refcat', 'custom', or 'gaia_custom')."
            )
        return str(catalogName).strip()

    def _resolve_catalog_for_filter(self, catalog_choice):
        """
        Resolve catalog selection from a scalar or per-filter mapping.

        Supported mapping examples:
            {'ugriz': 'sdss', 'UBVRI': 'apass', 'default': 'gaia'}
            {'g': 'sdss', 'r': 'sdss', 'default': 'apass'}
        """
        if isinstance(catalog_choice, dict):
            use_filter = str(self.input_yaml.get("imageFilter", "") or "").strip()
            use_filter_norm = normalize_photometric_filter_name(use_filter)
            # Canonical band for group membership (e.g. imageFilter "h" -> "H" in JHK / grizJHK).
            band_for_group = (
                use_filter_norm if use_filter_norm is not None else use_filter
            )
            # Warn when the current filter matches multiple mapping keys.
            membership_matches = []
            if use_filter:
                for key, value in catalog_choice.items():
                    if value is None:
                        continue
                    key_s = str(key).strip()
                    key_l = key_s.lower()
                    if key_l in {"default", "*", "all"}:
                        continue
                    key_bands = parse_supported_filter_group_key(key_s)
                    if key_bands and band_for_group in key_bands:
                        membership_matches.append(str(key))
                if len(membership_matches) > 1:
                    logger.warning(
                        "catalog.use_catalog mapping is ambiguous for filter '%s': matched keys=%s. "
                        "Using precedence: exact key > first membership key > default.",
                        use_filter,
                        membership_matches,
                    )

            # 1) exact key match first (e.g. {"g": "sdss"})
            for key, value in catalog_choice.items():
                key_s = str(key).strip()
                if value is None or key_s.lower() in {"default", "*", "all"}:
                    continue
                key_norm = normalize_photometric_filter_name(key_s)
                if key_s == use_filter and key_norm is not None:
                    return value
            # Backward-compatible fallback: normalized exact match.
            for key, value in catalog_choice.items():
                key_s = str(key).strip()
                if value is None or key_s.lower() in {"default", "*", "all"}:
                    continue
                key_norm = normalize_photometric_filter_name(key_s)
                if (
                    key_norm is not None
                    and use_filter_norm is not None
                    and key_norm == use_filter_norm
                ):
                    return value

            # 2) grouped bands by membership string/list (e.g. {"ugriz": "sdss"})
            for key, value in catalog_choice.items():
                if value is None:
                    continue
                key_str = str(key).strip()
                if key_str.lower() in {"default", "*", "all"}:
                    continue
                if not use_filter:
                    continue
                key_bands = parse_supported_filter_group_key(key_str)
                if key_bands and band_for_group in key_bands:
                    return value

            # 3) explicit default
            for dkey in ("default", "*", "all"):
                if dkey in catalog_choice and catalog_choice[dkey] is not None:
                    return catalog_choice[dkey]

            logger.warning(
                "catalog.use_catalog mapping has no match for filter '%s'; available keys=%s",
                use_filter,
                list(catalog_choice.keys()),
            )
            return None
        # Normalize scalar catalog names for backward compatibility
        if isinstance(catalog_choice, str):
            catalog_choice = self._normalize_catalog_name(catalog_choice)
        return catalog_choice

    @staticmethod
    def _catalog_len(obj) -> int:
        """Best-effort length for DataFrame/Table-like results."""
        if obj is None:
            return 0
        try:
            return int(len(obj))
        except Exception:
            return 0

    @staticmethod
    def _normalize_catalog_name(catalog_name: str) -> str:
        """
        Normalize catalog name aliases to canonical form.
        
        Accepts 'panstarrs' (no underscore) and converts to 'pan_starrs'.
        This ensures backward compatibility with both naming conventions.
        """
        if not catalog_name:
            return catalog_name
        name = str(catalog_name).strip().lower()
        # Map common aliases to canonical names
        aliases = {
            "panstarrs": "pan_starrs",
            "pan-starrs": "pan_starrs",
            "ps1": "pan_starrs",
        }
        return aliases.get(name, catalog_name)

    def _require_nonempty_catalog(
        self, selectedCatalog, catalogName: str, target_coords, radius_arcmin: float
    ) -> None:
        """
        Stop the pipeline if the catalog query returns zero sources.

        This is treated as "catalog does not cover that part of the sky" (or a
        service/query failure) and continuing would produce misleading results.
        """
        n = self._catalog_len(selectedCatalog)
        if n == 0:
            ra = float(target_coords.ra.degree)
            dec = float(target_coords.dec.degree)
            raise RuntimeError(
                f"{catalogName.upper()} catalog query returned 0 sources for "
                f"RA={ra:.6f} deg, Dec={dec:.6f} deg within r={radius_arcmin:.2f} arcmin. "
                "Assuming this catalog does not cover the field (or the query failed); stopping."
            )

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def gaia_synthetic_photometry(
        self,
        ra,
        dec,
        radius=0.1,
        max_sources=5000,
        photometric_systems=None,
    ):
        from gaiaxpy import PhotometricSystem
        import pandas as pd
        import warnings
        import numpy as np
        import logging

        from autophot_gaia_curves.gaia_archive import (
            gaia_xp_source_query,
            gaia_xp_sql_top_n,
            generate_source_ids_batched,
            launch_gaia_adql_to_pandas,
            sort_gaia_table_nearest_to_target,
        )

        logger = logging.getLogger(__name__)
        cat_cfg = self.input_yaml.get("catalog", {}) or {}
        query_pause_b = float(cat_cfg.get("gaia_archive_query_pause_before_sec", 1.0))
        query_pause_a = float(cat_cfg.get("gaia_archive_query_pause_after_sec", 1.0))
        xp_batch_size = int(cat_cfg.get("gaia_xp_batch_size", 200))
        xp_batch_pause = float(cat_cfg.get("gaia_xp_batch_pause_sec", 1.0))
        archive_retries = int(cat_cfg.get("gaia_archive_max_retries", 3))
        retry_base_delay = float(cat_cfg.get("gaia_archive_retry_base_delay_sec", 2.0))
        xp_order = str(cat_cfg.get("gaia_xp_order_by", "brightness")).strip().lower()
        xp_show_progress = bool(cat_cfg.get("gaia_xp_show_progress", False))
        prefetch_factor = int(cat_cfg.get("gaia_nearest_prefetch_factor", 50))
        prefetch_min = int(cat_cfg.get("gaia_nearest_prefetch_min", 500))
        prefetch_max = int(cat_cfg.get("gaia_nearest_prefetch_max", 10000))

        sql_top, sort_by_distance = gaia_xp_sql_top_n(
            max_sources,
            xp_order,
            prefetch_factor=prefetch_factor,
            prefetch_min=prefetch_min,
            prefetch_max=prefetch_max,
        )

        # ADQL radius is in degrees; caller passes degrees (e.g. radius_deg).
        query = gaia_xp_source_query(
            ra,
            dec,
            radius,
            sql_top,
            include_bp_rp=True,
        )

        try:
            logger.info(
                "Querying Gaia DR3 (synthetic photometry, SQL TOP %d -> target %d "
                "sources; paced archive: pause %.2fs before/after ADQL)...",
                sql_top,
                max_sources,
                max(query_pause_b, query_pause_a),
            )
            results = launch_gaia_adql_to_pandas(
                query,
                pause_before_sec=query_pause_b,
                pause_after_sec=query_pause_a,
                max_retries=archive_retries,
                retry_base_delay_sec=retry_base_delay,
                logger=logger,
                op_name="Gaia ADQL (XP sources for synthetic photometry)",
            )
            logger.info("Gaia DR3 query returned %d sources.", len(results))

            if sort_by_distance and not results.empty:
                logger.info(
                    "Sorting %d rows by on-sky distance to target (Gaia ADQL "
                    "does not support ORDER BY distance); keeping nearest %d.",
                    len(results),
                    max_sources,
                )
                results = sort_gaia_table_nearest_to_target(
                    results, ra, dec, max_rows=max_sources
                )
                logger.info("After distance trim: %d sources.", len(results))

            if results.empty:
                return pd.DataFrame()

            source_ids = results["source_id"].astype(str).tolist()
            # Resolve requested GaiaXPy photometric systems.
            #
            # If `photometric_systems` is None: keep prior behaviour (both).
            # If it is an empty list/[]/None: skip GaiaXPy synthetic photometry and
            # fall back to base DR3 photometry only (much faster).
            if photometric_systems is None:
                phot_systems = [
                    PhotometricSystem.SDSS_Std,
                    PhotometricSystem.JKC_Std,
                ]
            else:
                cfg = photometric_systems
                if isinstance(cfg, (str, bytes)):
                    cfg = [cfg]
                cfg_list = list(cfg) if cfg else []

                if not cfg_list:
                    logger.info(
                        "Gaia XP synthetic photometry disabled (empty photometric systems)."
                    )
                    return results

                phot_systems = []
                for sys_name in cfg_list:
                    if isinstance(sys_name, str):
                        sys_name = sys_name.strip()
                        phot_systems.append(getattr(PhotometricSystem, sys_name))
                    else:
                        # Allow passing enum values directly.
                        phot_systems.append(sys_name)

            logger.info(
                "Downloading Gaia XP spectra and synthetic photometry with GaiaXPy "
                "(batched: size=%d, inter-batch pause=%.2fs)...",
                xp_batch_size if xp_batch_size > 0 else len(source_ids),
                xp_batch_pause,
            )
            try:
                logger.info(
                    "GaiaXPy photometric systems: %s",
                    [getattr(p, "name", str(p)) for p in phot_systems],
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    photometry = generate_source_ids_batched(
                        source_ids,
                        phot_systems,
                        batch_size=xp_batch_size,
                        inter_batch_pause_sec=xp_batch_pause,
                        max_retries=archive_retries,
                        retry_base_delay_sec=retry_base_delay,
                        logger=logger,
                        show_progress=xp_show_progress,
                        error_correction=bool(
                            cat_cfg.get("gaia_xp_generate_error_correction", True)
                        ),
                        truncation=bool(
                            cat_cfg.get("gaia_xp_generate_truncation", False)
                        ),
                    )
                results = results.copy()
                results["source_id"] = results["source_id"].astype(str)
                photometry = photometry.copy()
                photometry["source_id"] = photometry["source_id"].astype(str)
                merged = pd.merge(
                    results, photometry, on="source_id", how="inner"
                )
            except Exception as exc:
                logger.warning(
                    "Gaia XP synthetic photometry failed (%s); falling back to base DR3 photometry only.",
                    exc,
                )
                merged = results

            # Compute magnitude errors from flux and flux_error where available:
            # sigma_m = 2.5 / ln(10) * (sigma_F / F).
            factor = 2.5 / np.log(10)

            # Avoid division by zero or missing columns; skip bands that are not
            # present in the merged table (e.g. when XP failed and only base DR3
            # photometry is available).
            for band in ["u", "g", "r", "i", "z"]:
                f_col = f"SdssStd_flux_{band}"
                e_col = f"SdssStd_flux_error_{band}"
                m_err_col = f"SdssStd_mag_error_{band}"
                if f_col not in merged.columns or e_col not in merged.columns:
                    continue
                valid = merged[f_col] > 0
                merged[m_err_col] = np.nan
                merged.loc[valid, m_err_col] = (
                    factor * merged.loc[valid, e_col] / merged.loc[valid, f_col]
                )

            for band in ["U", "B", "V", "R", "I"]:
                f_col = f"JkcStd_flux_{band}"
                e_col = f"JkcStd_flux_error_{band}"
                m_err_col = f"JkcStd_mag_error_{band}"
                if f_col not in merged.columns or e_col not in merged.columns:
                    continue
                valid = merged[f_col] > 0
                merged[m_err_col] = np.nan
                merged.loc[valid, m_err_col] = (
                    factor * merged.loc[valid, e_col] / merged.loc[valid, f_col]
                )

            logger.info(
                "Successfully merged Gaia DR3 catalog and XP photometry for %d sources.",
                len(merged),
            )
            return merged

        except Exception as exc:
            logger.error(
                "Gaia DR3 synthetic photometry query failed: %s", exc, exc_info=True
            )
            return pd.DataFrame()

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def query_legacy_survey(self, ra, dec, radius=0.1):
        """
        Query the Legacy Survey using the NOIRLab Data Lab TAP service via astroquery's TAP+.

        Parameters:
        -----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        radius : float, optional
            Search radius in degrees

        Returns:
        --------
        str
            Path to the downloaded file, or None if an error occurred.
        """
        # TAP service URL for NOIRLab Data Lab
        tap_service_url = "https://datalab.noirlab.edu/tap"
        # radius_deg = radius / 60

        # SQL query to search the Legacy Survey catalog using ADQL
        query = f"""
            SELECT TOP 1000 ra, dec, mag_g, mag_r, mag_i, mag_z,
                sqrt(power(ra - {ra}, 2) + power(dec - {dec}, 2)) AS angular_distance
            FROM ls_dr10.tractor
            WHERE 't'= Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, {radius})
            ORDER BY angular_distance ASC
        """

        try:
            logging.info(
                f"Fetching Legacy Survey Dr10 catalog over {radius:.1f} field-of-view centered at ra = {ra:.1f} dec = {dec:.1f}"
            )
            logging.info(query)

            from astroquery.utils.tap import TapPlus

            # Initialize the TAP service using TapPlus from astroquery
            tap = TapPlus(url=tap_service_url)

            # Perform the query using TAP+ (synchronous execution)
            result = tap.launch_job(query)

            # Convert the result to an Astropy Table
            table = result.get_results().to_pandas()
            filters = ["g", "r", "i", "z"]

            for f in filters:
                table[f"mag_{f}_e"] = [0.01] * len(table)

            return table

        except Exception as e:
            logger.info(f"Error during query: {e}")
            return None

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def fetch_refcat2_field(self, ra, dec, credentials, nsources=1000, sr=0.5):
        """
        Fetch ATLAS-RefCat2 catalog from MAST.

        Parameters:
        -----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        credentials : dict
            Dictionary containing MAST credentials.
        nsources : int, optional
            Maximum number of sources to fetch (default is 1000).
        sr : float, optional
            Search radius in degrees (default is 0.5).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the catalog data.
        """
        try:
            # Generate a unique name for the catalog
            name = f"autophot_{''.join(random.choices(string.ascii_uppercase, k=5))}"
            logger.info(
                f"Fetching ATLAS-RefCat2 catalog from MAST over {sr:.3f} deg field-of-view centered at ra = {ra:.1f} dec = {dec:.1f}"
            )

            # Define the columns to be retrieved from the database
            table = [
                "RA",
                "Dec",
                "g",
                "dg",
                "r",
                "dr",
                "i",
                "di",
                "z",
                "dz",
                "J",
                "dJ",
                "H",
                "dH",
                "K",
                "dK",
            ]

            # Create the SQL query to fetch data from the database
            q = """
            SELECT TOP {max} {columns}
            INTO MyDB.{name}
            FROM fGetNearbyObjEq({ra}, {dec}, {sr}) as n
            INNER JOIN refcat2 AS r ON (n.objid = r.objid)
            WHERE r.dr < 0.1
            ORDER BY n.distance
            """.format(
                max=nsources,
                columns="r." + ",r.".join(table),
                name=name,
                ra=ra,
                dec=dec,
                sr=sr,
            )

            logger.debug(f"SQL Query: {q}")
            from mastcasjobs import MastCasJobs

            # Create a job to execute the SQL query
            job = MastCasJobs(context="HLSP_ATLAS_REFCAT2", **credentials)
            job.drop_table_if_exists(name)
            jobid = job.submit(q, task_name=f"refcat catalog search {ra:.5f} {dec:.5f}")

            # Monitor the status of the job
            status = job.monitor(jobid)

            # Check if the job status indicates an error
            if status[0] in (3, 4):
                raise Exception(f"Job failed with status {status[0]}: {status[1]}")

            # Retrieve the result table and drop the temporary table
            tab = job.get_table(name, format="CSV")
            job.drop_table_if_exists(name)

            # Convert the result table to a pandas DataFrame
            tab = tab.to_pandas()

            # Remove any row where any value is zero
            tab = tab[~(tab == 0).any(axis=1)]
            tab.reset_index(drop=True, inplace=True)

        except Exception as e:
            logger.error("\n> Catalog retrieval failed! \n")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("%s %s %d %s", exc_type, fname, exc_tb.tb_lineno, e)
            return None

        return tab

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def download(
        self,
        target_coords,
        catalogName,
        radius=10,
        target_name=None,
        catalog_custom_fpath=None,
        include_IR_sequence_data=True,
        max_sources=None,
    ):
        """
        Download and process catalog data for a given target.

        Parameters:
        -----------
        target_coords : SkyCoord
            SkyCoord object with the RA and DEC of the target.
        catalogName : str
            Name of the catalog to fetch ('refcat', 'gaia', 'apass', '2mass', 'sdss', 'skymapper', 'panstarrs'/'pan_starrs', 'custom').
        radius : float, optional
            Search radius around the target in degrees (default is 15).
        target_name : str, optional
            Optional name for the target.
        catalog_custom_fpath : str, optional
            File path to a custom catalog (used if catalogName is 'custom').
        include_IR_sequence_data : bool, optional
            Boolean to include IR sequence data from 2MASS (default is True).
        max_sources : int, optional
            Maximum number of sources to return (default None for no limit).
            If set, catalogs will be limited to this many sources after download.

        Returns:
        --------
        DataFrame
            DataFrame containing the catalog data, or None if an error occurs.
        """
        logger.info(log_step("Catalog: sequence sources in field"))

        try:
            catalogName = self._require_catalog_selected(catalogName)

            # If target or its RA/DEC is set, set target name
            target_ra = target_coords.ra.degree
            target_dec = target_coords.dec.degree

            # Set default target name if not provided
            if target_name is None:
                if target_ra is not None and target_dec is not None:
                    target_name = f"target_ra_{target_ra:.6f}_dec_{target_dec:.6f}"
                else:
                    target_name = "target"
            else:
                if "Unknown" not in target_name:
                    target_name = self.input_yaml.get("target_name", "Transient")

            # Set default custom catalog path if not provided
            if not catalog_custom_fpath:
                catalog_custom_fpath = self.input_yaml["catalog"].get(
                    "catalog_custom_fpath", None
                )

            # Working directory
            wdir = self.input_yaml.get("wdir")
            if not wdir:
                raise ValueError("Working directory (wdir) is not set in input YAML.")

            # Create directories for storing catalog data
            dirname = os.path.join(wdir, "catalog_queries")
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            catalog_dir = os.path.join(dirname, catalogName)
            pathlib.Path(catalog_dir).mkdir(parents=True, exist_ok=True)
            target_dir = reduce(
                os.path.join, [dirname, catalogName, target_name.lower()]
            )
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

            # Generate file name for the catalog
            fname = f"{target_name}_r_{radius:.1f}arcmins_{catalogName}_target_ra_{target_ra:.6f}_dec_{target_dec:.6f}".replace(
                "", ""
            )

            # Convert from arcmins to degrees
            radius_deg = radius / 60

            # Handle different catalog sources
            if catalogName == "custom":
                if not catalog_custom_fpath:
                    logger.critical(
                        'Custom catalog selected but "catalog_custom_fpath" is not defined.'
                    )
                    return None
                selectedCatalog = pd.read_csv(catalog_custom_fpath)

            elif os.path.isfile(os.path.join(target_dir, f"{fname}.csv")):
                logger.info(
                    f'Existing {catalogName.upper()} catalog found for {target_name}\n{os.path.join(target_dir, f"{fname}.csv")}'
                )
                selectedCatalog = (
                    Table.read(os.path.join(target_dir, f"{fname}.csv"), format="csv")
                    .to_pandas()
                    .fillna(np.nan)
                )

            else:
                selectedCatalog = []
                from astroquery.mast import Catalogs

                if catalogName == "tic":
                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    coord = SkyCoord(
                        ra=target_coords.ra.degree,
                        dec=target_coords.dec.degree,
                        unit="deg",
                    )
                    result = Catalogs.query_region(
                        coord, radius=5 * u.arcmin, catalog="TIC"
                    )
                    if len(result) == 0:
                        selectedCatalog = pd.DataFrame()
                        self._require_nonempty_catalog(
                            selectedCatalog, catalogName, target_coords, radius
                        )

                    # Columns to extract (RA/DEC, r-band + error, others optional)
                    selected_cols = [
                        "ID",
                        "ra",
                        "dec",
                        "rmag",
                        "e_rmag",
                        "gmag",
                        "e_gmag",
                        "imag",
                        "e_imag",
                        "zmag",
                        "e_zmag",
                        "umag",
                        "e_umag",
                        "Bmag",
                        "e_Bmag",
                        "Vmag",
                        "e_Vmag",
                        "Jmag",
                        "e_Jmag",
                        "Hmag",
                        "e_Hmag",
                        "Kmag",
                        "e_Kmag",
                        "GAIAmag",
                        "e_GAIAmag",
                        "Tmag",
                        "e_Tmag",
                    ]

                    # Ensure all selected columns exist in the result table
                    available_cols = [
                        col for col in selected_cols if col in result.colnames
                    ]

                    # Build final catalog table
                    selectedCatalog = result[available_cols].to_pandas()
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(
                        os.path.join(os.getcwd(), f"{fname}.csv"),
                        os.path.join(target_dir, f"{fname}.csv"),
                    )

                elif catalogName == "gaia":
                    # For Gaia DR3+XP, use a smaller radius than the full catalog
                    # search radius to reduce load on the archive. Hard-limit to
                    # at most 10 arcmin (10/60 deg), and optionally smaller if
                    # catalog.gaia_xp_radius_deg is set.
                    max_gaia_deg = 10.0 / 60.0  # 10 arcmin
                    cfg_radius = float(
                        self.input_yaml.get("catalog", {}).get(
                            "gaia_xp_radius_deg", max_gaia_deg
                        )
                    )
                    xp_radius_deg = min(radius_deg, cfg_radius, max_gaia_deg)
                    gaia_xp_max_sources = int(
                        self.input_yaml.get("catalog", {}).get(
                            "gaia_xp_max_sources", 5000
                        )
                    )
                    gaia_xp_photometric_systems = (
                        self.input_yaml.get("catalog", {}).get(
                            "gaia_xp_photometric_systems", None
                        )
                    )
                    result = self.gaia_synthetic_photometry(
                        ra=target_coords.ra.degree,
                        dec=target_coords.dec.degree,
                        radius=xp_radius_deg,
                        max_sources=gaia_xp_max_sources,
                        photometric_systems=gaia_xp_photometric_systems,
                    )

                    # Build final catalog table
                    selectedCatalog = result
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(
                        os.path.join(os.getcwd(), f"{fname}.csv"),
                        os.path.join(target_dir, f"{fname}.csv"),
                    )

                elif catalogName == "refcat":

                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    logger.warning(
                        "REFCAT requires MAST CasJobs credentials. Set `default_input.catalog.MASTcasjobs_wsid` "
                        "and `default_input.catalog.MASTcasjobs_pwd` (or provide them via environment/local overrides)."
                    )
                    # Normalise types: many auth backends are strict about
                    # receiving strings, so cast to str (and strip) here.
                    userid = self.input_yaml["catalog"].get("MASTcasjobs_wsid")
                    password = self.input_yaml["catalog"].get("MASTcasjobs_pwd")
                    if userid is not None:
                        userid = str(userid).strip()
                    if password is not None:
                        password = str(password).strip()

                    credentials = {"userid": userid, "password": password}
                    userid_ok = (
                        credentials["userid"] is not None
                        and str(credentials["userid"]).strip() != ""
                    )
                    pwd_ok = (
                        credentials["password"] is not None
                        and str(credentials["password"]).strip() != ""
                    )
                    if not userid_ok or not pwd_ok:
                        raise RuntimeError(
                            "Refcat selected but MAST CasJobs credentials are missing/empty. "
                            f"MASTcasjobs_wsid set: {userid_ok}, MASTcasjobs_pwd set: {pwd_ok}. "
                            "Set `default_input.catalog.MASTcasjobs_wsid` and `default_input.catalog.MASTcasjobs_pwd` "
                            "(or export MASTCASJOBS_WSID/MASTCASJOBS_PWD)."
                        )

                    selectedCatalog = self.fetch_refcat2_field(
                        ra=target_coords.ra.degree,
                        dec=target_coords.dec.degree,
                        credentials=credentials,
                        nsources=500,
                        sr=radius_deg,
                    )
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(
                        os.path.join(os.getcwd(), f"{fname}.csv"),
                        os.path.join(target_dir, f"{fname}.csv"),
                    )

                elif catalogName == "legacy":

                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    selectedCatalog = self.query_legacy_survey(
                        ra=target_coords.ra.degree,
                        dec=target_coords.dec.degree,
                        radius=radius_deg,
                    )
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(
                        os.path.join(os.getcwd(), f"{fname}.csv"),
                        os.path.join(target_dir, f"{fname}.csv"),
                    )

                elif catalogName in ["apass", "2mass", "sdss"]:
                    Vizier.ROW_LIMIT = -1
                    logger.info(
                        f"Downloading Sequence Stars from {catalogName.upper()}"
                    )
                    catalog_search = Vizier.query_region(
                        target_coords,
                        radius=Angle(radius_deg, "deg"),
                        catalog=catalogName,
                    )
                    if len(catalog_search) < 1:
                        selectedCatalog = pd.DataFrame()
                    else:
                        selectedCatalog = catalog_search[0].to_pandas()
                        if catalogName == "sdss":
                            selectedCatalog = selectedCatalog[
                                selectedCatalog["mode"] == 1
                            ]
                            selectedCatalog = selectedCatalog[
                                selectedCatalog["cl"] == 6
                            ]
                        selectedCatalog.to_csv(
                            f"{fname}.csv", index=False, na_rep=np.nan
                        )
                        shutil.move(
                            os.path.join(os.getcwd(), f"{fname}.csv"),
                            os.path.join(target_dir, f"{fname}.csv"),
                        )
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )

                elif catalogName == "skymapper":
                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    server = "http://skymapper.anu.edu.au/sm-cone/public/query?"
                    params = {
                        "RA": target_coords.ra.degree,
                        "DEC": target_coords.dec.degree,
                        "SR": radius_deg,
                        "RESPONSEFORMAT": "VOTABLE",
                    }
                    with open("temp.vot", "wb") as f:
                        logger.info("Downloading Sequence Stars from SkyMapper")
                        response = requests.get(server, params=params)
                        f.write(response.content)
                    selectedCatalog = (
                        parse_single_table("temp.vot")
                        .to_table(use_names_over_ids=True)
                        .to_pandas()
                    )
                    selectedCatalog = selectedCatalog[
                        selectedCatalog["class_star"] > 0.8
                    ]
                    selectedCatalog = selectedCatalog[selectedCatalog["flags"] <= 1]
                    os.remove("temp.vot")
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(
                        os.path.join(os.getcwd(), f"{fname}.csv"),
                        os.path.join(target_dir, f"{fname}.csv"),
                    )

                elif catalogName == "pan_starrs":
                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    # Use direct API request to handle string "None" values properly
                    try:
                        import requests
                        
                        ra = float(target_coords.ra.degree)
                        dec = float(target_coords.dec.degree)
                        radius_deg = 0.1
                        
                        # Direct MAST API call for Pan-STARRS
                        url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/ps1/search"
                        params = {
                            "ra": ra,
                            "dec": dec,
                            "radius": radius_deg,
                            "pagesize": 10000,
                            "format": "json"
                        }
                        
                        response = requests.get(url, params=params, timeout=60)
                        response.raise_for_status()
                        data = response.json()
                        
                        if not data:
                            selectedCatalog = pd.DataFrame()
                        else:
                            # Convert to DataFrame, handling None strings
                            selectedCatalog = pd.DataFrame(data)
                            # Replace all forms of None/null with np.nan
                            for col in selectedCatalog.columns:
                                if selectedCatalog[col].dtype == object:
                                    selectedCatalog[col] = selectedCatalog[col].replace(
                                        ['None', 'none', 'NONE', 'null', 'NULL', 'nan', 'NaN'], np.nan
                                    )
                            # Convert numeric columns
                            for col in selectedCatalog.columns:
                                if col not in ['objName', 'objAltName1', 'objAltName2', 'objAltName3']:
                                    try:
                                        selectedCatalog[col] = pd.to_numeric(selectedCatalog[col], errors='coerce')
                                    except Exception:
                                        pass
                            
                        logger.info(f"Retrieved {len(selectedCatalog)} Pan-STARRS sources")
                        
                    except Exception as api_exc:
                        logger.warning(f"Direct Pan-STARRS API failed ({api_exc}), using empty catalog")
                        selectedCatalog = pd.DataFrame()
                    
                    # Replace all common null representations
                    selectedCatalog = selectedCatalog.replace([-999, -999.0, "None", "none", "NONE", "null", "NULL"], np.nan)
                    columns = [
                        "raMean",
                        "decMean",
                        "raMeanErr",
                        "decMeanErr",
                        "gMeanPSFMag",
                        "gMeanPSFMagErr",
                        "rMeanPSFMag",
                        "rMeanPSFMagErr",
                        "iMeanPSFMag",
                        "iMeanPSFMagErr",
                        "zMeanPSFMag",
                        "zMeanPSFMagErr",
                        "yMeanPSFMag",
                        "yMeanPSFMagErr",
                    ]
                    selectedCatalog = selectedCatalog[columns]
                    coords = SkyCoord(
                        ra=selectedCatalog["raMean"].values * u.deg,
                        dec=selectedCatalog["decMean"].values * u.deg,
                    )
                    distances = target_coords.separation(coords)
                    selectedCatalog["distance"] = distances.arcsecond
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(
                        os.path.join(os.getcwd(), f"{fname}.csv"),
                        os.path.join(target_dir, f"{fname}.csv"),
                    )

                else:
                    logger.critical("Catalog %s is not recognized.", catalogName)
                    sys.exit()

                logger.info(
                    "%s catalog contains %d sources",
                    catalogName.upper(),
                    len(selectedCatalog),
                )
                warnings.filterwarnings("default")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.info("%s %s %d %s", exc_type, fname, exc_tb.tb_lineno, e)
            # Propagate catalog failures as hard stops: downstream calibration
            # should not proceed without a valid catalog.
            raise

        # Limit catalog size if max_sources is specified
        if max_sources is not None and selectedCatalog is not None and len(selectedCatalog) > max_sources:
            logger.info(
                f"Limiting catalog from {len(selectedCatalog)} to {max_sources} sources"
            )
            # Sort by distance to target if RA and DEC columns are available
            if "RA" in selectedCatalog.columns and "DEC" in selectedCatalog.columns:
                from astropy.coordinates import SkyCoord
                from astropy import units as u
                
                catalog_coords = SkyCoord(
                    ra=selectedCatalog["RA"].values * u.degree,
                    dec=selectedCatalog["DEC"].values * u.degree
                )
                distances = catalog_coords.separation(target_coords)
                selectedCatalog = selectedCatalog.assign(distance=distances.arcsecond)
                selectedCatalog = selectedCatalog.nsmallest(max_sources, "distance")
                selectedCatalog = selectedCatalog.drop(columns=["distance"])
            else:
                # If RA/DEC not available, just take the first max_sources rows
                selectedCatalog = selectedCatalog.head(max_sources)
            logger.info(f"Catalog limited to {len(selectedCatalog)} sources")

        return selectedCatalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================
    def clean(
        self,
        selectedCatalog,
        catalogName=None,
        usefilter=None,
        magCutoff=30,
        border=0,
        image_wcs=None,
        fwhm=5,
        get_local_sources=False,
        full_clean=True,
        update_names_only=False,
        index=0,
    ):
        """
        Clean the catalog of sources by applying various filters and removing unwanted sources.

        Parameters:
        -----------
        selectedCatalog : pd.DataFrame
            DataFrame containing the source catalog.
        catalogName : str, optional
            Name of the catalog (default is None).
        usefilter : list, optional
            List of filters to use (default is None).
        magCutoff : float or list, optional
            Magnitude cutoff for filtering (default is 30).
        border : int, optional
            Border size in pixels (default is 0).
        image_wcs : WCS, optional
            WCS object for coordinate conversion (default is None).
        fwhm : float, optional
            FWHM in pixels (default is 5).
        get_local_sources : bool, optional
            Flag to get local sources (default is False).
        full_clean : bool, optional
            Flag for full cleaning (default is True).
        update_names_only : bool, optional
            Flag to update names only (default is False).
        index : int, optional
            Index for WCS (default is 0).

        Returns:
        --------
        pd.DataFrame
            Cleaned catalog DataFrame.
        """
        import logging
        import os
        import numpy as np
        import pandas as pd

        logger = logging.getLogger(__name__)

        if full_clean:
            logger.info(f"Cleaning {len(selectedCatalog)} sources")

        try:
            # Early exit if catalog is None or empty (e.g. Gaia service failure)
            if selectedCatalog is None or len(selectedCatalog) == 0:
                logger.warning("Selected catalog is empty; skipping catalog cleaning.")
                return None

            # --- Load catalog configuration ---
            filepath = os.path.dirname(os.path.abspath(__file__))
            catalog_autophot_input_yml = "catalog.yml"
            catalogName = catalogName or self.input_yaml["catalog"]["use_catalog"]

            # Load catalog keyword mappings
            catalog_keywords = AutophotYaml(
                os.path.join(filepath, "databases", catalog_autophot_input_yml),
                catalogName,
            ).load()

            max_distance = self.input_yaml["catalog"].get(
                "max_distance", 10
            )  # arcminutes

            # --- Filter sources by distance ---
            if "distance" in selectedCatalog:
                too_far = selectedCatalog["distance"] > max_distance * 60  # arcseconds
                if too_far.any():
                    logger.info(
                        f"Removing {too_far.sum()} sources that are greater than {max_distance:.1f} arcmins from the target"
                    )
                    selectedCatalog = selectedCatalog[~too_far]

            # --- Prepare output catalog with RA/DEC ---
            ra_key = catalog_keywords.get("RA")
            dec_key = catalog_keywords.get("DEC")
            if ra_key not in selectedCatalog or dec_key not in selectedCatalog:
                raise KeyError(
                    f"Required RA/DEC columns '{ra_key}', '{dec_key}' not found in catalog."
                )

            outputCatalog = pd.DataFrame(
                {
                    "RA": selectedCatalog[ra_key].values,
                    "DEC": selectedCatalog[dec_key].values,
                }
            )

            # --- Convert RA/DEC to pixel coordinates if WCS is provided ---
            if image_wcs:
                try:
                    ra_values = np.asarray(
                        selectedCatalog[catalog_keywords["RA"]].values, dtype=float
                    )
                    dec_values = np.asarray(
                        selectedCatalog[catalog_keywords["DEC"]].values, dtype=float
                    )

                    # Angular pre-filter: must match how the catalog was queried (usually
                    # within max_distance arcmin of the science target). Using CRVAL + a
                    # fixed 1° cap wrongly drops on-chip sources on wide stacks / coadds
                    # where CRVAL sits far from the field geometric center (common after
                    # astrometry.net SIP updates on template images).
                    cfg_cat = self.input_yaml.get("catalog", {}) or {}
                    max_dist_arcmin = float(cfg_cat.get("max_distance", 10.0))
                    t_ra = self.input_yaml.get("target_ra")
                    t_dec = self.input_yaml.get("target_dec")
                    if (
                        t_ra is not None
                        and t_dec is not None
                        and np.isfinite(t_ra)
                        and np.isfinite(t_dec)
                    ):
                        cen_ra = float(t_ra)
                        cen_dec = float(t_dec)
                        # Query radius plus a generous margin (arcsec)
                        max_distance_threshold = (max_dist_arcmin + 5.0) * 60.0
                    else:
                        shape = getattr(image_wcs, "array_shape", None)
                        if shape is not None and len(shape) == 2:
                            ny, nx = int(shape[0]), int(shape[1])
                            cx = 0.5 * float(max(nx - 1, 0))
                            cy = 0.5 * float(max(ny - 1, 0))
                            cen_ra, cen_dec = image_wcs.all_pix2world(
                                np.asarray([cx]),
                                np.asarray([cy]),
                                0,
                            )
                            cen_ra = float(np.asarray(cen_ra).ravel()[0])
                            cen_dec = float(np.asarray(cen_dec).ravel()[0])
                            corners_x = np.array(
                                [0.0, float(nx - 1), 0.0, float(nx - 1)], dtype=float
                            )
                            corners_y = np.array(
                                [0.0, 0.0, float(ny - 1), float(ny - 1)], dtype=float
                            )
                            cra, cde = image_wcs.all_pix2world(corners_x, corners_y, 0)
                            sc_cen = SkyCoord(
                                cen_ra * u.deg, cen_dec * u.deg, frame="icrs"
                            )
                            sc_corner = SkyCoord(
                                np.asarray(cra).ravel() * u.deg,
                                np.asarray(cde).ravel() * u.deg,
                                frame="icrs",
                            )
                            max_sep = float(
                                np.max(sc_cen.separation(sc_corner).to(u.arcsec).value)
                            )
                            max_distance_threshold = max(3600.0, max_sep * 1.25)
                        else:
                            cen_ra, cen_dec = image_wcs.wcs.crval
                            max_distance_threshold = 4 * 3600.0

                    dra = (ra_values - cen_ra) * np.cos(np.radians(cen_dec))
                    ddec = dec_values - cen_dec
                    distance = np.sqrt(dra**2 + ddec**2) * 3600.0  # arcseconds
                    valid_indices = distance < max_distance_threshold
                    if not np.any(valid_indices):
                        logger.warning(
                            "Catalog WCS pre-filter removed all sources (sky vs ref); "
                            "relaxing filter and keeping full list for pixel conversion."
                        )
                        valid_indices = np.ones(len(ra_values), dtype=bool)

                    ra_values = ra_values[valid_indices]
                    dec_values = dec_values[valid_indices]

                    # Full distortion (SIP, etc.): avoid wcs_world2pix, which can disagree
                    # with all_pix2world / solve-field SIP headers used elsewhere.
                    coords = SkyCoord(
                        ra=ra_values * u.deg, dec=dec_values * u.deg, frame="icrs"
                    )
                    x_pix, y_pix = image_wcs.world_to_pixel(coords)
                    x_pix = np.asarray(x_pix, dtype=float).ravel()
                    y_pix = np.asarray(y_pix, dtype=float).ravel()

                    outputCatalog = outputCatalog.iloc[valid_indices].copy()
                    outputCatalog["x_pix"] = x_pix
                    outputCatalog["y_pix"] = y_pix

                except Exception as e:
                    logger.warning(
                        f"Failed to convert RA/DEC to pixel coordinates: {e}. Skipping pixel coordinate conversion."
                    )
                    # Keep schema stable for downstream code that expects pixel columns.
                    outputCatalog = outputCatalog.copy()
                    outputCatalog["x_pix"] = np.nan
                    outputCatalog["y_pix"] = np.nan

            # Populate photometric band columns for the current filter.
            image_filter = self.input_yaml["imageFilter"]
            logger.debug(f"Populating filter columns for {image_filter}; input catalog columns: {list(selectedCatalog.columns)}")

            # For custom catalogs, auto-detect all filter columns (pattern: <filter> and <filter>_err)
            # This handles arbitrary filter names without requiring catalog.yml entries
            if catalogName == "custom":
                import re
                # Find all columns that look like filter magnitudes (have corresponding _err column)
                filter_cols = []
                for col in selectedCatalog.columns:
                    if str(col).endswith('_err'):
                        continue
                    err_col = f"{col}_err"
                    if err_col in selectedCatalog.columns:
                        filter_cols.append(col)
                        logger.debug(f"Auto-detected filter column pair: {col} / {err_col}")

                # Copy all detected filter columns to output
                for col in filter_cols:
                    err_col = f"{col}_err"
                    outputCatalog[col] = selectedCatalog[col].values
                    outputCatalog[err_col] = selectedCatalog[err_col].values
                    logger.debug(f"Auto-copied custom filter {col} from catalog")

            # Always ensure the current image filter is populated (primary logic)
            for col in [image_filter, f"{image_filter}_err"]:
                if col in outputCatalog.columns:
                    logger.debug(f"Column {col} already present in output catalog")
                    continue
                # First try catalog.yml mapping
                if col in catalog_keywords and catalog_keywords[col] in selectedCatalog:
                    outputCatalog[col] = selectedCatalog[catalog_keywords[col]].values
                    logger.debug(f"Mapped {col} from catalog_keywords")
                # Fallback: copy directly if column exists (for custom catalogs)
                elif col in selectedCatalog.columns:
                    outputCatalog[col] = selectedCatalog[col].values
                    logger.debug(f"Copied {col} directly from catalog")
                # Case-insensitive fallback for custom catalogs only (not standard UBVRI/ugriz/JHK)
                # to avoid conflating different photometric systems (e.g., r vs R)
                elif col not in catalog_keywords:
                    col_lower = col.lower()
                    for cat_col in selectedCatalog.columns:
                        if str(cat_col).lower() == col_lower:
                            outputCatalog[col] = selectedCatalog[cat_col].values
                            logger.debug(f"Mapped {col} case-insensitively from {cat_col}")
                            break
                else:
                    logger.warning(f"Could not find column {col} in catalog; available columns: {list(selectedCatalog.columns)}")

            # --- Retrieve all available filters ---
            baseDatabase = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "databases"
            )
            filters_yml = "filters.yml"
            availableFilters = AutophotYaml(
                os.path.join(baseDatabase, filters_yml)
            ).load()

            for filter_x in availableFilters["default_dmag"].keys():
                for col in [filter_x, f"{filter_x}_err"]:
                    # First try catalog.yml mapping
                    if (
                        filter_x in catalog_keywords
                        and catalog_keywords.get(col) in selectedCatalog
                    ):
                        outputCatalog[col] = selectedCatalog[
                            catalog_keywords[col]
                        ].values
                    # Fallback: copy directly if column exists (for custom catalogs)
                    elif col in selectedCatalog.columns:
                        outputCatalog[col] = selectedCatalog[col].values
                    # Case-insensitive fallback for custom catalogs only (not standard UBVRI/ugriz/JHK)
                    elif filter_x not in catalog_keywords:
                        col_lower = col.lower()
                        for cat_col in selectedCatalog.columns:
                            if str(cat_col).lower() == col_lower:
                                outputCatalog[col] = selectedCatalog[cat_col].values
                                break

            # --- Early return if only updating names ---
            if update_names_only:
                logger.info(
                    f"{len(outputCatalog)} sources in output catalog (names only)"
                )
                return outputCatalog

            # --- Determine filters to use ---
            usefilter = usefilter or [self.input_yaml["imageFilter"]]
            magCutoff = (
                [magCutoff] if isinstance(magCutoff, (int, float)) else magCutoff
            )

            # --- Apply magnitude cutoff filter ---
            if not full_clean:
                tooFaint = np.zeros(len(outputCatalog), dtype=bool)
                for i, usefilter_i in enumerate(usefilter):
                    if usefilter_i in outputCatalog.columns:
                        cutoff_i = magCutoff[i] if i < len(magCutoff) else magCutoff[0]
                        tooFaint |= outputCatalog[usefilter_i].values > cutoff_i
                if tooFaint.any():
                    logger.info(
                        f"Removing {tooFaint.sum()} sources that are fainter than the cutoff"
                    )
                    outputCatalog = outputCatalog.loc[~tooFaint]

            # --- Full cleaning procedures ---
            if full_clean:
                # Remove sources with missing filter information
                image_filter = self.input_yaml["imageFilter"]
                if image_filter in outputCatalog.columns:
                    hasFilterinfo = np.isfinite(outputCatalog[image_filter].values)
                    if (~hasFilterinfo).any():
                        logger.info(
                            f"Excluding {(~hasFilterinfo).sum()} sources with no {image_filter} band information"
                        )
                        outputCatalog = outputCatalog[hasFilterinfo]
                else:
                    logger.warning(
                        f"Filter column '{image_filter}' not found in output catalog; "
                        f"available columns: {list(outputCatalog.columns)}"
                    )

            logger.info(f"{len(outputCatalog)} sources in output catalog")
            logger.debug(f"Output catalog columns: {list(outputCatalog.columns)}")
            return outputCatalog

        except Exception as e:
            import traceback

            logger.error(f"Error in catalog cleaning: {e}")
            logger.error(traceback.format_exc())
            return None

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def recenter(self, selectedCatalog, image, boxsize=None, error=None):
        """
        Recenter sources in an image, selecting centroiding method from FWHM.
        Undersampled (FWHM < 3 px): 2D Gaussian fit for subpixel accuracy.
        Well-sampled (FWHM >= 3 px): center-of-mass. Robust against fully masked cutouts.
        Error-weighted centroiding has been removed; this routine always uses
        unweighted centroiding for stability across diverse background/error maps.
        """
        try:
            num_sources = len(selectedCatalog)
            logger.info(
                log_step(
                    f"Recentering {num_sources} source{'s' if num_sources > 1 else ''}"
                )
            )

            required_columns = {"x_pix", "y_pix"}
            if not required_columns.issubset(selectedCatalog.columns):
                logger.error(
                    "Selected catalog missing required columns: 'x_pix', 'y_pix'"
                )
                return selectedCatalog

            fwhm = float(self.input_yaml.get("fwhm", 3))
            undersampled_thr = float(
                (self.input_yaml.get("photometry", {}) or {}).get(
                    "undersampled_fwhm_threshold", 2.5
                )
            )
            undersampled = bool(
                self.input_yaml.get("undersampled_mode", fwhm <= undersampled_thr)
            )

            # Box size: default ~3xFWHM, odd, at least 3; for undersampled use at least 7 for 2DG
            if boxsize is None:
                boxsize = max(3, int(np.ceil(fwhm) * 3))
            boxsize = int(boxsize)
            if undersampled and boxsize < 7:
                boxsize = 7
            boxsize = boxsize if boxsize % 2 != 0 else boxsize + 1
            boxsize = max(boxsize, 3)
            border = boxsize
            logger.info(f"Boxsize: {boxsize} px (undersampled={undersampled})")

            # Filter sources near image borders
            height, width = image.shape
            mask_x = (selectedCatalog["x_pix"] > border) & (
                selectedCatalog["x_pix"] < width - border
            )
            mask_y = (selectedCatalog["y_pix"] > border) & (
                selectedCatalog["y_pix"] < height - border
            )
            mask = mask_x & mask_y

            if num_sources > 1:
                logger.info(f"Recentering {sum(mask)} sources within border")
                selectedCatalog = selectedCatalog.loc[mask].copy()

            # Extract initial coordinates
            old_x = selectedCatalog["x_pix"].values
            old_y = selectedCatalog["y_pix"].values
            nan_mask = ~np.isfinite(image)

            # Pre-check cutouts to avoid fully masked regions
            valid_sources = self._check_valid_cutouts(
                image, old_x, old_y, boxsize, nan_mask
            )
            if valid_sources.sum() == 0:
                logger.warning("No sources have valid cutouts - skipping recentering")
                return selectedCatalog.loc[valid_sources]

            logger.info(f"{valid_sources.sum()} sources have valid cutouts")

            old_x_valid = old_x[valid_sources]
            old_y_valid = old_y[valid_sources]

            # Undersampled (FWHM < 3): 2D Gaussian; well-sampled: COM
            if undersampled:
                logger.info("Using 2D Gaussian centroiding (FWHM < 3 px, undersampled)")
                centroid_func = centroid_2dg
            else:
                logger.info("Using center-of-mass centroiding (FWHM >= 3 px)")
                centroid_func = centroid_com
            try:
                x_valid, y_valid = centroid_sources(
                    image,
                    old_x_valid,
                    old_y_valid,
                    box_size=boxsize,
                    centroid_func=centroid_func,
                    mask=nan_mask,
                )
                x_valid = np.asarray(x_valid)
                y_valid = np.asarray(y_valid)
            except Exception as e:
                log_warning_from_exception(
                    logger, "Centroiding failed even on valid sources", e
                )
                x_valid = old_x_valid.copy()
                y_valid = old_y_valid.copy()
            x_err_valid = np.full(len(x_valid), np.nan)
            y_err_valid = np.full(len(y_valid), np.nan)

            # Create full arrays with NaNs for invalid sources
            x = np.full(len(old_x), np.nan)
            y = np.full(len(old_y), np.nan)
            x[valid_sources] = x_valid
            y[valid_sources] = y_valid

            x_err = np.full(len(old_x), np.nan)
            y_err = np.full(len(old_y), np.nan)
            x_err[valid_sources] = x_err_valid
            y_err[valid_sources] = y_err_valid

            # Update coordinates
            selectedCatalog.loc[:, "x_pix"] = x
            selectedCatalog.loc[:, "y_pix"] = y
            if "x_pix_err" not in selectedCatalog.columns:
                selectedCatalog["x_pix_err"] = np.nan
            if "y_pix_err" not in selectedCatalog.columns:
                selectedCatalog["y_pix_err"] = np.nan
            selectedCatalog.loc[:, "x_pix_err"] = x_err
            selectedCatalog.loc[:, "y_pix_err"] = y_err

            # Filter sources that moved outside the border
            mask_x = (selectedCatalog["x_pix"] >= border) & (
                selectedCatalog["x_pix"] < width - border
            )
            mask_y = (selectedCatalog["y_pix"] >= border) & (
                selectedCatalog["y_pix"] < height - border
            )
            mask = mask_x & mask_y

            if sum(~mask) > 0:
                logger.info(f"Failed to recenter {sum(~mask)} sources - ignoring")
                selectedCatalog = selectedCatalog.loc[mask].copy()

            # Compute median offset
            valid = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(old_x)
                & np.isfinite(old_y)
            )
            if valid.any():
                average_offset = np.nanmedian(
                    pix_dist(x[valid], old_x[valid], y[valid], old_y[valid])
                )
                logger.info(f"Median pixel correction: {average_offset:.1f} px")
            else:
                logger.warning("No valid sources to compute median offset.")

            # Remove NaNs
            selectedCatalog = selectedCatalog.loc[
                selectedCatalog[["x_pix", "y_pix"]].notna().all(axis=1)
            ]
            logger.info(
                f"Recentered {len(selectedCatalog)} source{'s' if len(selectedCatalog) > 1 else ''}"
            )

        except Exception as e:
            logger.error("Error occurred during recentering: %s", e)
            logger.error(traceback.format_exc())

        return selectedCatalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def _check_valid_cutouts(
        self, image, x_coords, y_coords, boxsize, mask, min_unmasked_frac=0.0
    ):
        """
        Check which source cutouts contain at least one unmasked pixel.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        x_coords, y_coords : array
            Source coordinates
        boxsize : int
            Cutout box size
        mask : numpy.ndarray
            Mask array (True where masked)

        Returns:
        --------
        valid_mask : numpy.ndarray
            Boolean mask of valid cutouts
        min_unmasked_frac : float
            Minimum required fraction of unmasked pixels in a cutout.
        """
        height, width = image.shape
        half_box = boxsize // 2
        valid_mask = np.ones(len(x_coords), dtype=bool)

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            # Define cutout bounds
            x_min = int(x - half_box)
            x_max = int(x + half_box + 1)
            y_min = int(y - half_box)
            y_max = int(y + half_box + 1)

            # Check bounds
            if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
                valid_mask[i] = False
                continue

            # Check if cutout has enough unmasked pixels
            cutout_mask = mask[y_min:y_max, x_min:x_max]
            n_pix = cutout_mask.size
            n_unmasked = int(np.sum(~cutout_mask))
            frac_unmasked = n_unmasked / max(1, n_pix)
            if frac_unmasked <= float(min_unmasked_frac):
                valid_mask[i] = False

        return valid_mask

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def find_source(self, ra, dec, catalog, tolerance=3.0):
        """
        Find sources in the catalog that match the given RA and DEC
        within a specified tolerance using vectorized operations.

        Parameters:
        -----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        catalog : pd.DataFrame
            DataFrame containing the source catalog.
        tolerance : float, optional
            Tolerance in arcseconds (default is 3.0).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing matching sources.
        """
        if catalog.empty:
            return catalog

        # Vectorized angular separation using small-angle approximation
        # (valid for separations << 1 degree, which tolerance in arcsec guarantees)
        dec_rad = np.radians(dec)
        ra_vals = catalog["RA"].values
        dec_vals = catalog["DEC"].values

        delta_ra = (ra_vals - ra) * np.cos(dec_rad)   # arcsec-equivalent in degrees
        delta_dec = dec_vals - dec
        separation_deg = np.sqrt(delta_ra**2 + delta_dec**2)
        separation_arcsec = separation_deg * 3600.0

        match_mask = separation_arcsec <= tolerance
        return catalog[match_mask]

    # =========================================================================
    # METHOD: build_complete_catalog
    # =========================================================================
    def build_complete_catalog(
        self,
        target_coords,
        target_name=None,
        catalog_list=["refcat", "sdss", "pan_starrs", "apass", "2mass"],
        radius=2,
        max_separation=3,
        **kwargs,
    ):
        """
        Build a complete catalog by combining multiple catalogs.

        Parameters:
        -----------
        target_coords : SkyCoord
            SkyCoord object with the RA and DEC of the target.
        target_name : str, optional
            Name for the target (default is None).
        catalog_list : list, optional
            List of catalogs to combine (default is ['refcat', 'sdss', 'pan_starrs', 'apass', '2mass']).
        radius : float, optional
            Search radius in arcminutes (default is 2).
        max_separation : float, optional
            Maximum separation in arcseconds for matching sources (default is 3).
        # Backward compatibility: support legacy misspelled keyword.
        if "max_seperation" in kwargs:
            legacy_value = kwargs.pop("max_seperation")
            if legacy_value is not None:
                max_separation = legacy_value
        if kwargs:
            logger.warning(
                "Ignoring unexpected keyword(s) in build_complete_catalog: %s",
                ", ".join(sorted(kwargs.keys())),
            )


        Returns:
        --------
        pd.DataFrame
            Combined catalog DataFrame.
        """
        catalog_list_str = ",".join([i.upper() for i in catalog_list])
        logger.info(log_step(f"Custom catalog: {catalog_list_str}"))

        # Set default target name if not provided
        if not target_name:
            target_name = self.input_yaml.get("target_name", "Transient")

        # Generate file name for the catalog
        fname = f"{target_name}_r_{radius}arcmins_CUSTOM.csv".replace("", "")
        wdir = self.input_yaml.get("wdir")
        if not wdir:
            raise ValueError("Working directory (wdir) is not set in input YAML.")

        dirname = os.path.join(wdir, "catalog_queries")
        dirname = os.path.join(dirname, "custom")

        # Create directories for storing catalog data
        dirname = os.path.join(wdir, "catalog_queries")
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        catalog_dir = os.path.join(dirname, "custom_builds")
        pathlib.Path(catalog_dir).mkdir(parents=True, exist_ok=True)
        fpath = os.path.join(catalog_dir, fname)

        filter_list = [
            "u",
            "g",
            "r",
            "i",
            "z",
            "U",
            "B",
            "V",
            "R",
            "I",
            "Z",
            "J",
            "H",
            "K",
        ]

        # Create a new list by appending '_err' to each element
        updated_filter_list = []
        for filter in filter_list:
            updated_filter_list.append(filter)
            updated_filter_list.append(f"{filter}_err")

        # Matching tolerance in arcseconds (used by `find_source`)
        tolerance_arcsec = max_separation

        cols = ["RA", "DEC"] + updated_filter_list
        
        # Check if output catalog already exists and load it (with deduplication)
        if os.path.isfile(fpath):
            logger.info(f"Loading existing custom catalog from {fpath}")
            existing_catalog = pd.read_csv(fpath)
            if not existing_catalog.empty:
                # Deduplicate existing catalog
                coords_existing = SkyCoord(
                    ra=existing_catalog["RA"].values * u.degree,
                    dec=existing_catalog["DEC"].values * u.degree
                )
                idx_match, sep, _ = coords_existing.match_to_catalog_sky(coords_existing, nthneighbor=2)
                keep_mask = sep > 1.0 * u.arcsec
                existing_catalog = existing_catalog[keep_mask].reset_index(drop=True)
                n_dups = len(coords_existing) - len(existing_catalog)
                if n_dups > 0:
                    logger.warning(f"Removed {n_dups} duplicates from existing cached catalog - RESAVING clean version")
                    # Save the deduplicated catalog back to file
                    existing_catalog.to_csv(fpath, index=False, float_format="%.6f")
                return existing_catalog
        
        output_catalog = pd.DataFrame(columns=cols)

        for catalogName in catalog_list:
            logger.info(f"Getting {catalogName} catalog")
            catalog_i = self.download(
                target_coords=target_coords,
                catalogName=catalogName,
                radius=radius,
                target_name=target_name,
            )
            if catalog_i is None:
                continue

            catalog_i = self.clean(
                catalog_i, catalogName=catalogName, update_names_only=True
            )

            # Collect new rows to avoid repeated concat in loop
            new_rows = []
            
            # Loop over each entry in the current catalog
            for index, entry in catalog_i.iterrows():
                ra = entry["RA"]
                dec = entry["DEC"]

                # Check if the source exists in the output catalog
                existing_source = self.find_source(
                    ra, dec, output_catalog, tolerance=tolerance_arcsec
                )

                if existing_source.empty:
                    # Source does not exist, add to new rows list
                    new_row = {col: entry.get(col, np.nan) for col in cols}
                    new_rows.append(new_row)
                else:
                    # Source exists, update the row with missing filter data
                    index = existing_source.index[0]
                    for filter_name in updated_filter_list:
                        if filter_name in entry and pd.isna(
                            output_catalog.at[index, filter_name]
                        ):
                            output_catalog.at[index, filter_name] = entry[filter_name]
            
            # Concatenate all new rows at once for performance
            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)
                n_before = len(output_catalog)
                output_catalog = pd.concat(
                    [output_catalog, new_rows_df], ignore_index=True
                )
                logger.info(f"Added {len(new_rows_df)} sources from {catalogName}, catalog now has {len(output_catalog)} sources")

        # Final deduplication: remove any duplicate sources that may have been added
        if not output_catalog.empty:
            coords_final = SkyCoord(
                ra=output_catalog["RA"].values * u.degree,
                dec=output_catalog["DEC"].values * u.degree
            )
            idx_match, sep, _ = coords_final.match_to_catalog_sky(coords_final, nthneighbor=2)
            keep_mask = sep > 1.0 * u.arcsec  # Keep sources with no close neighbour within 1 arcsec
            n_before_dedup = len(output_catalog)
            output_catalog = output_catalog[keep_mask].reset_index(drop=True)
            n_removed = n_before_dedup - len(output_catalog)
            logger.info(f"Final catalog: {len(output_catalog)} sources (removed {n_removed} duplicates)")
            if n_removed > 0:
                logger.warning(f"Removed {n_removed} duplicate sources from final combined catalog")
        
        # Final output catalog is ready
        output_catalog.to_csv(fpath, index=False, float_format="%.6f")
        logger.info(f"Saved clean catalog to {fpath}")
        return output_catalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def check_saturation_range(self, catalog, threshold=5):
        """
        Checks the saturation range of a given catalog by plotting catalog magnitude vs. instrumental magnitude
        and robustly fitting a straight line with slope constrained to ~1 using RANSAC.
        Determines the linearity range within the 0.5 to 95 flux range of the inliers.

        Parameters:
        -----------
        catalog : pd.DataFrame
            DataFrame containing astronomical catalog data.
        threshold : float, optional
            Threshold value for filtering sources (default is 5).

        Returns:
        --------
        tuple
            (pd.DataFrame, dict, list) Updated clean catalog containing only inliers,
            fit parameters including errors, and saturation range [min_flux, max_flux].
        """
        # Initialize fit parameters dictionary
        fit_params = {
            "slope": None,
            "intercept": None,
            "intercept_error": None,
            "n_inliers": 0,
        }
        saturation_range = [0, np.inf]

        try:
            # Extract file paths
            fpath = self.input_yaml.get("fpath", "")
            if not fpath:
                raise ValueError("File path is missing from input_yaml.")

            base_name = os.path.splitext(os.path.basename(fpath))[0]
            write_dir = os.path.dirname(fpath)
            logger.info(
                log_step(f"Linearity: saturation check ({len(catalog)} sources)")
            )

            # Extract filter
            use_filter = self.input_yaml.get("imageFilter")
            if not use_filter:
                raise ValueError("Missing 'imageFilter' in input YAML.")

            # Check for required columns
            required_columns = ["flux_AP", use_filter, f"{use_filter}_err"]
            missing_columns = [
                col for col in required_columns if col not in catalog.columns
            ]
            if missing_columns:
                raise KeyError(
                    f"Missing required columns in catalog: {missing_columns}"
                )

            # Sigma-clip sky background
            if "sky" in catalog.columns:
                noise_mask = sigma_clip(
                    np.abs(catalog["sky"].values), sigma=5, maxiters=10
                )
                if np.any(noise_mask.mask):
                    logger.info(
                        f"Masking {np.sum(noise_mask.mask)} sources with irregular backgrounds"
                    )
                    catalog = catalog[~noise_mask.mask]

            # Threshold filtering
            if "threshold" in catalog.columns:
                threshold_cut = catalog["threshold"] < threshold
                logger.info(
                    f"Removing {np.sum(threshold_cut)} sources with threshold < {threshold}"
                )
                catalog = catalog[~threshold_cut]

            # Calculate instrumental magnitude
            flux = catalog["flux_AP"].values
            flux_err = catalog["flux_AP_err"].values
            inst_mag = -2.5 * np.log10(flux)
            inst_mag_err = 2.5 / np.log(10) * (flux_err / flux)
            catalog_mag = catalog[use_filter].values
            catalog_mag_err = catalog[f"{use_filter}_err"].values

            # Filter out high-error points
            error_mask = np.sqrt(catalog_mag_err**2 + inst_mag_err**2) < 0.32
            clean_catalog = catalog[error_mask].copy()

            if len(clean_catalog) < 2:
                logger.warning("Too few points left after error cut.")
                return clean_catalog, fit_params, saturation_range

            # Calculate S/N and select continuous high S/N subset
            # This avoids scattered low S/N inliers that break the linearity fit
            flux = clean_catalog["flux_AP"].values
            flux_err = clean_catalog["flux_AP_err"].values
            snr = flux / flux_err

            # Find a high S/N threshold that gives a continuous bright subset
            # Start with S/N > 50 for very high quality, then relax if needed
            min_snr_thresholds = [100, 75, 50, 30, 20, 10]
            selected_indices = None
            for min_snr in min_snr_thresholds:
                high_snr_mask = snr >= min_snr
                n_high_snr = np.sum(high_snr_mask)
                if n_high_snr >= 15:  # Need at least 15 sources for robust fit
                    selected_indices = high_snr_mask
                    logger.info(
                        f"Selected {n_high_snr} high S/N sources (SNR >= {min_snr}) for linearity fit"
                    )
                    break

            if selected_indices is not None and np.sum(selected_indices) >= 10:
                clean_catalog = clean_catalog[selected_indices].copy()
                flux = clean_catalog["flux_AP"].values
                flux_err = clean_catalog["flux_AP_err"].values
                snr = snr[selected_indices]

            inst_mag_linear = -2.5 * np.log10(flux)
            inst_mag_err_linear = 2.5 / np.log(10) * (flux_err / flux)
            catalog_mag_linear = clean_catalog[use_filter].values
            catalog_mag_err_linear = clean_catalog[f"{use_filter}_err"].values

            class ConstrainedSlopeRegressor(BaseEstimator, RegressorMixin):
                def __init__(
                    self, slope_constraint=1.0, slope_tolerance=1e-3, sample_weight=None
                ):
                    self.slope_constraint = slope_constraint
                    self.slope_tolerance = slope_tolerance
                    self.sample_weight = sample_weight  # weights = 1/sigma^2

                def fit(self, X, y):
                    X, y = check_X_y(X, y)
                    weights = None
                    if self.sample_weight is not None:
                        weights = np.array(self.sample_weight)
                        if weights.shape[0] != len(y):
                            raise ValueError(
                                "sample_weight must match number of samples"
                            )

                    def loss(params):
                        slope, intercept = params
                        pred = slope * X.flatten() + intercept
                        resid = y - pred
                        if weights is not None:
                            mse = np.average(resid**2, weights=weights)
                        else:
                            mse = np.mean(resid**2)
                        slope_penalty = 100 * max(
                            0, abs(slope - self.slope_constraint) - self.slope_tolerance
                        )
                        return mse + slope_penalty

                    init_params = [1.0, np.mean(y) - np.mean(X)]
                    result = minimize(loss, init_params, method="L-BFGS-B")
                    self.slope_, self.intercept_ = result.x
                    return self

                def predict(self, X):
                    check_is_fitted(self, "slope_")
                    X = check_array(X)
                    return self.slope_ * X.flatten() + self.intercept_

            # Fit with RANSAC
            if len(inst_mag_linear) > 1:
                base_estimator = ConstrainedSlopeRegressor(
                    slope_constraint=1.0, slope_tolerance=0  # Keep slope strictly fixed to 1
                )
                # Adaptive residual threshold based on data scatter
                initial_mad = np.median(np.abs(catalog_mag_linear - np.median(catalog_mag_linear)))
                ransac_residual_threshold = max(3.0 * initial_mad, 0.1)  # More lenient threshold for better fit
                ransac = RANSACRegressor(
                    estimator=base_estimator,
                    residual_threshold=ransac_residual_threshold,
                    max_trials=2000,  # More trials for better convergence
                    min_samples=0.25,  # Require slightly more samples for stability
                )
                X = inst_mag_linear.reshape(-1, 1)
                y = catalog_mag_linear
                ransac.fit(X, y)
                slope = ransac.estimator_.slope_
                intercept = ransac.estimator_.intercept_
                inlier_mask = ransac.inlier_mask_

                # Post-RANSAC sigma clipping to remove remaining outliers
                if np.sum(inlier_mask) > 5:
                    inlier_X = X[inlier_mask]
                    inlier_y = y[inlier_mask]
                    residuals = inlier_y - (slope * inlier_X.flatten() + intercept)
                    # Apply sigma clipping on residuals
                    clipped = sigma_clip(residuals, sigma=2.5, maxiters=5)
                    # Keep only points within sigma-clipped range
                    residual_mask = np.asarray(~clipped.mask, dtype=bool).flatten()
                    # Ensure shapes match before assignment
                    if residual_mask.shape[0] == np.sum(inlier_mask):
                        inlier_mask[inlier_mask] = residual_mask
                    else:
                        logger.warning(f"Shape mismatch in residual masking: {residual_mask.shape[0]} vs {np.sum(inlier_mask)}, skipping sigma clip update")
                    n_sigma_outliers = np.sum(~residual_mask)
                    if n_sigma_outliers > 0:
                        logger.info(f"Post-RANSAC sigma clipping removed {n_sigma_outliers} additional outliers")

                # Recompute intercept on the final inlier set so the plotted fit line
                # matches the points that survive post-RANSAC clipping.
                if np.sum(inlier_mask) > 1 and np.isfinite(slope):
                    try:
                        intercept = float(
                            np.nanmedian(
                                y[inlier_mask] - slope * X[inlier_mask].flatten()
                            )
                        )
                    except Exception:
                        pass

                # Calculate intercept error
                inlier_X = X[inlier_mask]
                inlier_y = y[inlier_mask]
                residuals = inlier_y - (slope * inlier_X.flatten() + intercept)
                residual_std = np.std(residuals)
                n_points = len(inlier_X)
                x_mean = np.mean(inlier_X)
                x_var = np.var(inlier_X)

                # Standard error of the intercept
                intercept_error = residual_std * np.sqrt(
                    1 / n_points + x_mean**2 / ((n_points - 1) * x_var)
                )

                # Update fit parameters
                fit_params.update(
                    {
                        "slope": slope,
                        "intercept": intercept,
                        "intercept_error": intercept_error,
                        "n_inliers": n_points,
                    }
                )
                logger.info(
                    f"Constrained fit results:  Zeropoint = {intercept:.3f} +/- {intercept_error:.3f}"
                )

                # Calculate linearity range (0.5 to 95 flux range of inliers)
                inlier_flux = flux[inlier_mask]
                if len(inlier_flux) > 0:
                    min_flux = np.percentile(inlier_flux, 0.5)
                    max_flux = np.percentile(inlier_flux, 95)
                    saturation_range = [min_flux, max_flux]
                    logger.info(
                        f"Linearity range (0.5-95% flux percentiles): {min_flux:.1f} - {max_flux:.1f}"
                    )
                else:
                    logger.warning("No inliers found for linearity range calculation")

                fit_line = lambda x: slope * x + intercept
            else:
                logger.warning("Not enough sources for linear fitting.")
                fit_line = None
                inlier_mask = np.ones_like(catalog_mag_linear, dtype=bool)

            # Plotting
            from plotting_utils import (
                apply_autophot_mplstyle,
                ransac_legend_top_outside,
                set_mag_axes_inverted_xy,
            )

            apply_autophot_mplstyle()
            plt.ioff()
            fig, ax1 = plt.subplots(figsize=set_size(340, 1))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            # Import centralized plotting utilities for consistent formatting
            from plotting_utils import get_color, get_marker_size, get_alpha, get_line_width

            if fit_line:
                predicted = fit_line(inst_mag_linear)
                residuals = catalog_mag_linear - predicted
                # Use RANSAC inlier mask for plotting consistency
                ransac_inliers = inlier_mask if 'inlier_mask' in locals() else np.ones(len(inst_mag_linear), dtype=bool)
                ransac_outliers = ~ransac_inliers
                ax1.errorbar(
                    inst_mag_linear[ransac_outliers],
                    catalog_mag_linear[ransac_outliers],
                    yerr=catalog_mag_err_linear[ransac_outliers],
                    xerr=inst_mag_err_linear[ransac_outliers],
                    fmt="x",
                    color=get_color('outliers'),
                    ecolor="lightgrey",
                    markersize=get_marker_size('medium'),
                    alpha=get_alpha('medium'),
                    lw=get_line_width('thin'),
                    capsize=0,
                    elinewidth=0.4,
                    linestyle="None",
                    label=f"Outliers [{np.sum(ransac_outliers)}]",
                )
                ax1.errorbar(
                    inst_mag_linear[ransac_inliers],
                    catalog_mag_linear[ransac_inliers],
                    yerr=catalog_mag_err_linear[ransac_inliers],
                    xerr=inst_mag_err_linear[ransac_inliers],
                    fmt="o",
                    markersize=get_marker_size('medium'),
                    mfc=get_color('inliers'),
                    mec=get_color('inliers'),
                    ecolor="lightgrey",
                    alpha=get_alpha('dark'),
                    capsize=0,
                    elinewidth=0.4,
                    linestyle="None",
                    label=f"Inliers [{np.sum(ransac_inliers)}]",
                )
                x_range = np.linspace(inst_mag_linear[ransac_inliers].min(), inst_mag_linear[ransac_inliers].max(), 100)
                y_fit = fit_line(x_range)
                ax1.plot(
                    x_range,
                    y_fit,
                    color=get_color('fit'),
                    linestyle="--",
                    lw=get_line_width('thick'),
                    zorder=10,
                    label=(
                        f"m_cal = m_inst + {intercept:.2f} ± {intercept_error:.2f} "
                        f"(slope fixed to 1.0)"
                    ),
                )
                # Add shaded error region
                ax1.fill_between(
                    x_range,
                    y_fit - intercept_error,
                    y_fit + intercept_error,
                    color=get_color('error_region'),
                    alpha=get_alpha('light'),
                    label=f"Fit error (+/- {intercept_error:.2f})",
                )

                # Add vertical lines showing linearity range in instrumental magnitude
                if len(inlier_flux) > 0:
                    min_inst_mag = -2.5 * np.log10(
                        max_flux
                    )  # Note: brighter objects have smaller magnitudes
                    max_inst_mag = -2.5 * np.log10(min_flux)
                    ax1.axvline(
                        x=min_inst_mag,
                        color="#00AA00",
                        linestyle=":",
                        alpha=0.7,
                        label=f"Linearity range: {min_flux:.1f}-{max_flux:.1f} flux",
                    )
                    ax1.axvline(
                        x=max_inst_mag,
                        color="#00AA00",
                        linestyle=":",
                        alpha=0.7,
                    )

            ax1.set_xlabel(r"Instrumental $m_\mathrm{inst}$ [mag]")
            ax1.set_ylabel(rf"Catalog $m_\mathrm{{cal,{use_filter}}}$ [mag]")
            set_mag_axes_inverted_xy(ax1)
            ransac_legend_top_outside(ax1, ncol=2)
            ax1.grid(True, linestyle="--", alpha=0.5, zorder=0, lw=0.5)
            save_path = os.path.join(
                write_dir, f"Saturation_{base_name}.png"
            )
            fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close(fig)

            # Robust selection: find continuous linear region with tight scatter requirement
            if fit_line:
                # Get inlier sources from RANSAC
                inlier_catalog = clean_catalog[inlier_mask].copy()
                inlier_flux = flux[inlier_mask]
                inlier_inst_mag = inst_mag_linear[inlier_mask]
                
                if len(inlier_catalog) > 5:
                    # Calculate residuals for all inliers
                    predicted_mag = fit_line(inlier_inst_mag.reshape(-1, 1))
                    residuals = catalog_mag_linear[inlier_mask] - predicted_mag
                    
                    # Sort by flux (bright to faint - smaller mag = brighter)
                    sort_idx = np.argsort(inlier_flux)[::-1]  # Bright first
                    sorted_flux = inlier_flux[sort_idx]
                    sorted_residuals = np.asarray(residuals[sort_idx]).flatten()  # Ensure 1D
                    sorted_mag = inlier_inst_mag[sort_idx]
                    
                    # TIGHT residual threshold: use 2-sigma instead of 3-sigma for stricter selection
                    # Calculate from central 50% of sources (most linear region)
                    central_start = len(sorted_residuals) // 4
                    central_end = 3 * len(sorted_residuals) // 4
                    central_residuals = sorted_residuals[central_start:central_end]
                    
                    if len(central_residuals) > 3:
                        median_resid = np.median(central_residuals)
                        mad_residual = np.median(np.abs(central_residuals - median_resid))
                        # Relaxed threshold: 3.0 * MAD or 0.1 mag to fit majority of points better
                        residual_threshold = float(np.maximum(3.0 * mad_residual, 0.1))
                    else:
                        residual_threshold = 0.15
                    
                    # Find bright end: cut where residuals exceed threshold (saturation/non-linear)
                    bright_cut_idx = 0
                    for i in range(len(sorted_residuals)):
                        resid_val = float(sorted_residuals[i])  # Ensure scalar
                        if np.abs(resid_val) > residual_threshold:
                            bright_cut_idx = i + 1  # Cut this and brighter
                        else:
                            break
                    
                    # Find faint end: detect where scatter systematically increases
                    # Use smaller window for tighter control, and require multiple consecutive windows
                    window_size = max(3, len(sorted_residuals) // 15)  # Smaller window
                    faint_cut_idx = len(sorted_residuals)
                    
                    # Track consecutive high-scatter windows
                    high_scatter_count = 0
                    required_consecutive = 2  # Require 2 consecutive windows with high scatter
                    
                    for i in range(window_size, len(sorted_residuals) - window_size):
                        window_residuals = sorted_residuals[i-window_size:i+window_size]
                        window_mad = np.median(np.abs(window_residuals - np.median(window_residuals)))

                        # If local scatter exceeds 2.0x central scatter, mark as high scatter (relaxed from 1.5x)
                        if window_mad > 2.0 * mad_residual:
                            high_scatter_count += 1
                            if high_scatter_count >= required_consecutive:
                                faint_cut_idx = i - window_size  # Cut before this region
                                break
                        else:
                            high_scatter_count = 0  # Reset if scatter drops
                    
                    # Additional faint cut: ensure we're not using sources with large individual residuals
                    # Scan from faint end and find where residuals become acceptable
                    for i in range(len(sorted_residuals) - 1, faint_cut_idx - 1, -1):
                        if np.abs(sorted_residuals[i]) > 1.5 * residual_threshold:
                            faint_cut_idx = i  # Cut this and fainter
                        else:
                            break
                    
                    # Apply flux range cuts (bright_cut_idx to faint_cut_idx)
                    if bright_cut_idx > 0 or faint_cut_idx < len(sorted_flux):
                        min_linear_flux = sorted_flux[min(faint_cut_idx, len(sorted_flux)-1)]
                        max_linear_flux = sorted_flux[bright_cut_idx] if bright_cut_idx < len(sorted_flux) else sorted_flux[0]
                        
                        # Ensure we have a valid range
                        if min_linear_flux < max_linear_flux:
                            # Create mask for continuous linear region
                            linear_flux_mask = (flux >= min_linear_flux) & (flux <= max_linear_flux)
                            
                            # Apply tight residual threshold to remove individual outliers
                            # Use inlier arrays only to avoid mismatch with clean_catalog
                            inlier_catalog_mag_linear = catalog_mag_linear[inlier_mask]
                            inlier_inst_mag_linear = inst_mag_linear[inlier_mask]
                            all_residuals = inlier_catalog_mag_linear - fit_line(inlier_inst_mag_linear.reshape(-1, 1))
                            inlier_residual_mask = np.abs(np.asarray(all_residuals).flatten()) < residual_threshold
                            
                            # Expand inlier residual mask back to full array size
                            linear_residual_mask = np.zeros(len(flux), dtype=bool)
                            linear_residual_mask[inlier_mask] = inlier_residual_mask
                            
                            # Combined mask: must be in flux range AND have good residual
                            final_linear_mask = linear_flux_mask & linear_residual_mask
                            
                            n_bright_cut = np.sum(flux > max_linear_flux)
                            n_faint_cut = np.sum(flux < min_linear_flux)
                            n_outlier_cut = np.sum(linear_flux_mask & ~linear_residual_mask)
                            n_selected = np.sum(final_linear_mask)
                            
                            logger.info(
                                f"Robust linear selection: flux range {min_linear_flux:.1f} - {max_linear_flux:.1f}, "
                                f"residual thresh {residual_threshold:.3f} mag, "
                                f"cut {n_bright_cut} bright + {n_faint_cut} faint + {n_outlier_cut} outlier, "
                                f"kept {n_selected} sources"
                            )
                            
                            if n_selected > 0:
                                clean_catalog = clean_catalog[final_linear_mask]
                                # Update saturation range from final selection
                                saturation_range = [
                                    np.percentile(clean_catalog["flux_AP"].values, 0.5),
                                    np.percentile(clean_catalog["flux_AP"].values, 99.5)
                                ]
                            else:
                                # No sources passed tight criteria, fall back to central region only
                                central_flux_mask = (flux >= sorted_flux[central_end]) & (flux <= sorted_flux[central_start])
                                if np.sum(central_flux_mask) > 0:
                                    clean_catalog = clean_catalog[central_flux_mask]
                                    logger.warning(f"No sources passed tight criteria, using central {len(clean_catalog)} sources")
                                else:
                                    clean_catalog = inlier_catalog
                                    logger.warning(f"No sources passed tight criteria, using all {len(clean_catalog)} inliers")
                        else:
                            # Invalid range, use central region
                            central_flux_mask = (flux >= sorted_flux[central_end]) & (flux <= sorted_flux[central_start])
                            if np.sum(central_flux_mask) > 0:
                                clean_catalog = clean_catalog[central_flux_mask]
                    else:
                        # No cuts needed but still apply tight residual filter
                        tight_residual_mask = np.abs(residuals) < residual_threshold
                        n_tight_outliers = (~tight_residual_mask).sum()
                        if n_tight_outliers > 0:
                            logger.info(f"Applying tight residual filter: removed {n_tight_outliers} sources")
                        clean_catalog = inlier_catalog[tight_residual_mask]
                else:
                    # Too few sources for robust selection
                    clean_catalog = inlier_catalog
                    logger.warning(f"Only {len(inlier_catalog)} inliers, skipping robust selection")

            logger.info(f"Returning {len(clean_catalog)} linear sources")
            return clean_catalog, fit_params, saturation_range

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(
                f"Error in build_psf: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {str(e)}"
            )
            return catalog, fit_params, saturation_range

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def downsample_sources_by_position(
        self,
        df: pd.DataFrame,
        x_col: str = "x_pix",
        y_col: str = "y_pix",
        nmax: int = 300,
        snr_col: Optional[str] = "SNR",
    ) -> pd.DataFrame:
        """
        Downsample a dataframe of sources by distributing them evenly across
        spatial bins while prioritizing higher SNR sources.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing source information.
        x_col : str, optional
            Column name for x-coordinates (default: 'x_pix').
        y_col : str, optional
            Column name for y-coordinates (default: 'y_pix').
        nmax : int, optional
            Maximum number of sources to keep (default: 300).
        snr_col : str, optional
            Column name for SNR values to prioritize selection (default: 'SNR').

        Returns:
        --------
        pd.DataFrame
            Downsampled dataframe with at most nmax sources.
        """
        n_src = len(df)

        # If dataframe is already small enough, return quickly without extra logging or work.
        if n_src <= nmax:
            logger.info("Downsampling skipped: %d sources (<= %d target).", n_src, nmax)
            return df.copy()

        # Only emit the full banner when we are actually going to downsample.
        logger.info(
            log_step(f"Downsampling: {n_src} sources -> {nmax} max")
        )

        # Check if required columns exist
        if x_col not in df.columns or y_col not in df.columns:
            error_msg = f"DataFrame must contain '{x_col}' and '{y_col}' columns"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate number of bins (square grid)
        n_bins = int(np.sqrt(nmax))
        sources_per_bin = max(1, nmax // (n_bins**2))
        logger.info(
            f"Using {n_bins}x{n_bins} grid with ~{sources_per_bin} sources per bin"
        )

        # Create spatial bins
        x_bins = np.linspace(df[x_col].min(), df[x_col].max(), n_bins + 1)
        y_bins = np.linspace(df[y_col].min(), df[y_col].max(), n_bins + 1)
        logger.debug(f"X bins range: {x_bins[0]:.2f} to {x_bins[-1]:.2f}")
        logger.debug(f"Y bins range: {y_bins[0]:.2f} to {y_bins[-1]:.2f}")

        selected_indices = []
        bin_stats = []  # Track bin statistics for logging

        # Iterate through each spatial bin
        logger.info("Processing spatial bins...")
        for i in range(n_bins):
            for j in range(n_bins):
                # Find sources in current bin
                in_xbin = (df[x_col] >= x_bins[i]) & (df[x_col] < x_bins[i + 1])
                in_ybin = (df[y_col] >= y_bins[j]) & (df[y_col] < y_bins[j + 1])
                bin_indices = np.where(in_xbin & in_ybin)[0]
                bin_stats.append(len(bin_indices))

                if len(bin_indices) > 0:
                    # Sort by SNR (highest first) if SNR column exists
                    if snr_col and snr_col in df.columns:
                        bin_indices = bin_indices[
                            np.argsort(df[snr_col].iloc[bin_indices])[::-1]
                        ]
                        logger.debug(
                            f"Bin ({i},{j}): {len(bin_indices)} sources, sorted by {snr_col}"
                        )
                    else:
                        logger.debug(f"Bin ({i},{j}): {len(bin_indices)} sources")

                    # Take top sources from this bin
                    selected_indices.extend(bin_indices[:sources_per_bin])

        # Log bin statistics
        logger.info(
            f"Bin statistics: min={min(bin_stats)}, max={max(bin_stats)}, "
            f"avg={np.mean(bin_stats):.1f} sources per bin"
        )
        logger.info(f"Selected {len(selected_indices)} sources from spatial bins")

        # If we didn't get enough sources, fill with highest SNR remaining sources
        if len(selected_indices) < nmax:
            logger.warning(
                f"Only {len(selected_indices)} sources selected from bins, "
                f"filling with top {nmax - len(selected_indices)} remaining sources"
            )
            all_indices = set(range(len(df)))
            remaining_indices = list(all_indices - set(selected_indices))

            if snr_col and snr_col in df.columns:
                # Sort remaining sources by SNR (highest first)
                remaining_indices = sorted(
                    remaining_indices,
                    key=lambda idx: df[snr_col].iloc[idx],
                    reverse=True,
                )
                logger.debug("Sorted remaining sources by SNR")

            # Add top remaining sources
            additional_count = nmax - len(selected_indices)
            selected_indices.extend(remaining_indices[:additional_count])
            logger.info(
                f"Added {additional_count} additional sources from remaining pool"
            )

        # Final validation
        final_count = len(selected_indices)
        if final_count > nmax:
            logger.warning(f"Selected {final_count} sources (exceeds target {nmax})")
        else:
            logger.info(f"Final selection: {final_count} sources")

        # Return the downsampled dataframe
        logger.info("Downsampling complete")
        return df.iloc[selected_indices].copy()

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def check_starshape(
        self,
        image,
        catalog,
        scale=25,
        threshold=5,
        fwhm_threshold=1.5,
        roundness_threshold=0.2,
        sharpness_threshold=0.4,
    ):
        """
        Identify and exclude extended sources using multiple criteria:
        1. Radial profile consistency (original method)
        2. FWHM (Full Width at Half Maximum) comparison
        3. Roundness (1 - minor_axis/major_axis)
        4. Sharpness (central pixel concentration)

        Parameters:
        -----------
        image : numpy.ndarray
            2D array of the image.
        catalog : pd.DataFrame
            DataFrame with source positions (x_pix, y_pix).
        scale : int or tuple, optional
            Size of cutout around each source (default is 25).
        threshold : float, optional
            Sigma threshold for radial profile outliers (default is 5).
        fwhm_threshold : float, optional
            Max allowed FWHM in pixels for point sources (default is 1.5).
        roundness_threshold : float, optional
            Max allowed deviation from roundness (1-perfect circle) (default is 0.2).
        sharpness_threshold : float, optional
            Min required sharpness (higher = more point-like) (default is 0.4).

        Returns:
        --------
        pd.DataFrame
            Cleaned catalog DataFrame.
        """
        if isinstance(scale, int):
            scale = (scale, scale)

        catalog.reset_index(inplace=True, drop=True)
        required_columns = {"x_pix", "y_pix"}
        if not required_columns.issubset(catalog.columns):
            raise ValueError(
                f"Catalog missing required columns: {required_columns - set(catalog.columns)}"
            )

        def radial_profile(data):
            y, x = np.indices(data.shape)
            r = np.sqrt((x - data.shape[1] // 2) ** 2 + (y - data.shape[0] // 2) ** 2)
            r = r.astype(int)
            tbin = np.bincount(r.ravel(), data.ravel())
            nr = np.bincount(r.ravel())
            return tbin / np.maximum(nr, 1)

        def calculate_fwhm(data):
            """Estimate FWHM from radial profile"""
            profile = radial_profile(data)
            half_max = np.max(profile) * 0.5
            above = np.where(profile >= half_max)[0]
            return 2 * (above[-1] - above[0]) if len(above) > 1 else np.nan

        def calculate_roundness(data):
            """Calculate roundness (1 - b/a) from moments"""
            y, x = np.indices(data.shape)
            xc, yc = data.shape[1] / 2, data.shape[0] / 2
            x = x - xc
            y = y - yc

            # Calculate second moments
            mxx = np.sum(x**2 * data) / np.sum(data)
            myy = np.sum(y**2 * data) / np.sum(data)
            mxy = np.sum(x * y * data) / np.sum(data)

            # Calculate eigenvalues (major/minor axes)
            term1 = (mxx + myy) / 2
            term2 = np.sqrt(((mxx - myy) / 2) ** 2 + mxy**2)
            a = term1 + term2
            b = term1 - term2
            return 1 - np.sqrt(b / a)  # 0=perfect circle

        def norm(array):
            """
            Normalize a NumPy array to the range [0, 1].

            Parameters:
            -----------
            array : numpy.ndarray
                Input array to normalize.

            Returns:
            --------
            numpy.ndarray
                Normalized array.
            """
            array = np.asarray(array)
            min_val = np.min(array)
            max_val = np.max(array)
            if max_val == min_val:
                return np.zeros_like(
                    array
                )  # Avoid division by zero; all elements are the same.
            return (array - min_val) / (max_val - min_val)

        def calculate_sharpness(data):
            """Measure central concentration using Laplacian"""
            lap = gaussian_laplace(data, sigma=1)
            center = data.shape[0] // 2, data.shape[1] // 2
            radius = min(center) // 2
            y, x = np.indices(data.shape)
            r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
            central = np.mean(np.abs(lap)[r <= radius])
            outer = np.mean(np.abs(lap)[r > radius * 2])
            return central / max(outer, 1e-6)  # Avoid division by zero

        metrics = {"fwhm": [], "roundness": [], "sharpness": []}
        cutouts = []
        radial_profiles = []
        valid_indices = []

        logger.info(log_step(f"Radial profile: {len(catalog)} sources"))

        for i, (x, y) in enumerate(zip(catalog["x_pix"], catalog["y_pix"])):
            position = (float(x), float(y))
            try:
                cutout = Cutout2D(
                    image, position=position, size=scale, mode="partial", fill_value=np.nan
                )
                cutout_data = cutout.data
                total_flux = np.nanmax(cutout_data)
                if np.isfinite(total_flux) and total_flux > 0:
                    normalized_cutout = norm(cutout_data / total_flux)

                    # Calculate all metrics
                    fwhm = calculate_fwhm(normalized_cutout)
                    roundness = calculate_roundness(normalized_cutout)
                    sharpness = calculate_sharpness(normalized_cutout)

                    metrics["fwhm"].append(fwhm)
                    metrics["roundness"].append(roundness)
                    metrics["sharpness"].append(sharpness)
                    cutouts.append(normalized_cutout)
                    radial_profiles.append(norm(radial_profile(normalized_cutout)))
                    valid_indices.append(i)
                else:
                    logger.info(f"Star {i} rejected: zero or negative flux.")

            except Exception as e:
                logger.warning(
                    f"Star {i} at position {position} skipped due to error: {e}"
                )
                for key in metrics:
                    metrics[key].append(np.nan)

        # Convert metrics to arrays
        for key in metrics:
            metrics[key] = np.array(metrics[key])

        # Create selection masks for each criterion
        fwhm_mask = np.isfinite(metrics["fwhm"])  # & (metrics['fwhm'] < fwhm_threshold)

        # Apply sigma clipping to roundness and sharpness
        roundness_sigma_clip = sigma_clip(
            metrics["roundness"],
            sigma=threshold,
            masked=True,
            cenfunc=np.nanmedian,
            stdfunc=np.nanstd,
        )
        sharpness_sigma_clip = sigma_clip(
            metrics["sharpness"],
            sigma=threshold,
            masked=True,
            cenfunc=np.nanmedian,
            stdfunc=np.nanstd,
        )

        # Mask for roundness and sharpness after sigma clipping
        roundness_mask = np.isfinite(metrics["roundness"]) & ~roundness_sigma_clip.mask
        sharpness_mask = np.isfinite(metrics["sharpness"]) & ~sharpness_sigma_clip.mask

        # Combine with radial profile outlier rejection
        if len(radial_profiles) > 0:
            radial_profiles = np.array(radial_profiles)
            clipped = sigma_clip(
                radial_profiles,
                sigma=threshold,
                axis=0,
                masked=True,
                # cenfunc=np.nanmedian, stdfunc=mad_std
            )
            profile_mask = ~np.any(clipped.mask, axis=1)
        else:
            profile_mask = np.zeros(len(valid_indices), dtype=bool)

        # Final combined mask (profile_mask only; roundness/sharpness still logged above)
        combined_mask = profile_mask

        # Ensure that at least a few sources remain
        if np.sum(combined_mask) < 5:
            logger.warning("All sources have been rejected by the masking process.")
            # Keep all sources or apply a fallback strategy
            combined_mask = np.ones_like(combined_mask, dtype=bool)

        # Identify inliers and outliers
        inliers = [
            valid_indices[i] for i, is_good in enumerate(combined_mask) if is_good
        ]
        outliers = [
            valid_indices[i] for i, is_good in enumerate(combined_mask) if not is_good
        ]

        # Log the results
        logger.info(
            f"Point sources: {len(inliers)} | Extended/rejected sources: {len(outliers)}"
        )
        logger.info(
            f"Rejection breakdown - FWHM: {sum(~fwhm_mask)}, "
            f"Roundness: {sum(~roundness_mask)}, "
            f"Sharpness: {sum(~sharpness_mask)}, "
            f"Profile: {sum(~profile_mask)}"
        )

        # Check the cleaned catalog
        if len(inliers) > 0:
            cleaned_catalog = catalog.iloc[inliers].copy()
        else:
            logger.warning(
                "No sources remained after cleaning. Using original catalog."
            )
            cleaned_catalog = catalog.copy()  # Fallback to original catalog

        # Add metrics to catalog for diagnostics
        for key in metrics:
            cleaned_catalog[f"star_{key}"] = metrics[key][combined_mask]

        if self.input_yaml.get("save_cutout_plot", True):
            index_map = {idx: cutout for idx, cutout in zip(valid_indices, cutouts)}
            stars = [index_map[i] for i in inliers]
            num_stars = len(stars)
            side_length = int(np.ceil(np.sqrt(num_stars)))
            ncols = side_length
            nrows = ceil(num_stars / ncols)
            _style = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
            )
            if os.path.exists(_style):
                plt.style.use(_style)
            plt.ioff()
            fpath = self.input_yaml["fpath"]
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            fig = plt.figure(figsize=set_size(540, 1))
            grid = fig.add_gridspec(
                nrows=nrows,
                ncols=ncols + 3,
                width_ratios=[1] * ncols + [0.1, 1, 1],
                height_ratios=[1] * nrows,
                hspace=0.01,
                wspace=0.01,
            )
            for i in range(num_stars):
                row, col = divmod(i, ncols)
                ax = fig.add_subplot(grid[row, col])
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(np.asarray(stars[i]))
                norm = ImageNormalize(vmin=vmin, vmax=vmax)
                ax.imshow(
                    stars[i],
                    origin="lower",
                    cmap="viridis",
                    norm=norm,
                    interpolation="none",
                )
                ax.text(
                    0.98,
                    0.98,
                    f"{i + 1}/{num_stars}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    color="white",
                )
                ax.set_xticks([])
                ax.set_yticks([])

            ax_right = fig.add_subplot(grid[:, -2:])
            radii = np.arange(radial_profiles.shape[1])
            accepted_profiles = [
                radial_profiles[valid_indices.index(i)] for i in inliers
            ]
            for profile in accepted_profiles:
                ax_right.step(radii, profile, color="black", alpha=0.3, linewidth=0.5)
            median_profile = np.median(accepted_profiles, axis=0)
            ax_right.step(
                radii,
                median_profile,
                color="#FF0000",
                linewidth=0.5,
                label="Median\nRadial\nProfile",
            )
            ax_right.set_xlabel("Radius [pixels]")
            ax_right.set_ylabel("Normalized Flux")
            ax_right.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
            ax_right.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax_right.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
            pos = ax_right.get_position()
            ax_right.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])
            output_path = os.path.join(
                write_dir, f"Zeropoint_Sources_{base}.png"
            )
            fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close()

        logger.info(f"Returning {len(cleaned_catalog)} well-behaved sources")
        return cleaned_catalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def measure(self, selectedCatalog, image):
        """
        Measure the flux and signal-to-noise ratio (SNR) of sources in an image using aperture photometry.

        Parameters:
        -----------
        selectedCatalog : pd.DataFrame
            DataFrame containing the sources with initial coordinates.
        image : numpy.ndarray
            2D array representing the image where sources are located.

        Returns:
        --------
        pd.DataFrame
            Updated DataFrame with measured flux, instrumental magnitude, and SNR for each source.
        """
        # Initialize logger
        logger = logging.getLogger(__name__)

        # Log the start of the measurement process
        logger.info(
            log_step(
                f"Aperture photometry: {len(selectedCatalog)} field sources"
            )
        )

        # Initialize the aperture photometry object with the provided input YAML and image
        initialAperture = Aperture(
            input_yaml=self.input_yaml,
            image=image,
        )

        # Measure the sources using aperture photometry
        selectedCatalog = initialAperture.measure(sources=selectedCatalog)

        # Rename the flux column to reflect the aperture measurement
        selectedCatalog.rename(columns={"flux": "flux_AP"}, inplace=True)

        # Calculate the instrumental magnitude based on the measured flux (mag() does not mutate flux_AP)
        instMag = mag(selectedCatalog["flux_AP"])
        inst_col = "inst_" + self.input_yaml["imageFilter"] + "_AP"
        selectedCatalog[inst_col] = np.round(instMag, 3)

        # Calculate the SNR for each source
        sourceSNR = snr(selectedCatalog["maxPixel"], selectedCatalog["noiseSky"])
        selectedCatalog["snr"] = np.round(sourceSNR, 1)

        # Log the number of sources with valid instrumental magnitude (finite flux > 0)
        logger.info(
            "Instrumental magnitude of %d sources measured"
            % sum(~np.isnan(selectedCatalog[inst_col]))
        )

        return selectedCatalog
