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
import time
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
    if variable_catalog is None or len(variable_catalog) == 0:
        return given_catalog

    x_given = given_catalog["x_pix"].values
    y_given = given_catalog["y_pix"].values
    x_var = variable_catalog["x_pix"].values
    y_var = variable_catalog["y_pix"].values

    # KDTree for O(N log M) matching; replaces the O(N*M) per-source
    # distance loop.
    from scipy.spatial import cKDTree as _cKDTree
    var_tree = _cKDTree(np.column_stack([x_var, y_var]))
    given_xy = np.column_stack([x_given, y_given])
    dist_nearest, idx_nearest = var_tree.query(given_xy, k=1, workers=1)
    keep_mask = dist_nearest > match_radius_pix

    removed_indices = np.where(~keep_mask)[0].tolist()
    for i in removed_indices:
        otype = variable_catalog.iloc[idx_nearest[i]]["OTYPE_opt"]
        logger.debug(
            f"Removing source at index {i} (x={x_given[i]:.2f}, y={y_given[i]:.2f}) due to match with variable source [{otype}]"
        )

    filtered_catalog = given_catalog[keep_mask].reset_index(drop=True)
    logger.info(
        "Variable-source match: removed %d, %d sources remain",
        len(removed_indices), len(filtered_catalog),
    )

    return filtered_catalog


# =============================================================================
# Deduplication helper
# =============================================================================

def _skycoord_dedup_keep_one(catalog_df, sep_threshold_arcsec=0.1):
    """
    Remove duplicate sky positions from *catalog_df*, keeping exactly one
    member of each close pair rather than dropping both.

    Algorithm
    ---------
    For every source i, find its nearest *other* source (nthneighbor=2) and
    its index j.  If sep(i,j) < threshold AND j < i then source i is the
    later-arriving duplicate and is marked for removal.  This guarantees the
    lower-index entry of any close pair survives.

    Parameters
    ----------
    catalog_df : pd.DataFrame  - must contain "RA" and "DEC" columns (degrees)
    sep_threshold_arcsec : float - angular separation below which two sources
                                   are considered duplicates (default 0.1 arcsec)

    Returns
    -------
    pd.DataFrame  - deduplicated catalog with reset integer index
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if catalog_df is None or len(catalog_df) < 2:
        return catalog_df
    if not {"RA", "DEC"}.issubset(catalog_df.columns):
        return catalog_df

    coords = SkyCoord(
        ra=catalog_df["RA"].values * u.degree,
        dec=catalog_df["DEC"].values * u.degree,
    )
    idx_match, sep2, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    thr = sep_threshold_arcsec * u.arcsec
    # Vectorized: drop source i if its nearest neighbour j is closer than the
    # threshold AND has a lower index (meaning j is the "first" copy to keep).
    drop = (sep2 <= thr) & (idx_match < np.arange(len(catalog_df)))
    return catalog_df[~drop].reset_index(drop=True)


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================
class Catalog:
    """Catalog query and cross-matching for photometric calibration.

    Wraps online catalog services (Pan-STARRS, SDSS, Skymapper, Gaia, 2MASS,
    WISE) and provides local catalog operations: source cross-matching,
    saturation/linearity filtering, and zeropoint fitting preparation.
    """

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
            # Warn on ambiguous mappings (multiple keys match this filter).
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
                        "catalog.use_catalog mapping is ambiguous for filter '%s': matched keys=%s. Using precedence: exact key > first membership key > default.",
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
        Normalize catalog name aliases to canonical form (lowercase).
        
        Accepts case-insensitive input (e.g., 'GAIA', 'Gaia', 'gaia') and
        converts to lowercase. Also handles 'panstarrs' (no underscore) and
        converts to 'pan_starrs' for backward compatibility.
        """
        if not catalog_name:
            return catalog_name
        name = str(catalog_name).strip().lower()
        aliases = {
            "panstarrs": "pan_starrs",
            "pan-starrs": "pan_starrs",
            "ps1": "pan_starrs",
        }
        return aliases.get(name, name)

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
        query_pause_b = float(cat_cfg.get("gaia_archive_query_pause_before_sec", 0.25))
        query_pause_a = float(cat_cfg.get("gaia_archive_query_pause_after_sec", 0.25))
        xp_batch_size = int(cat_cfg.get("gaia_xp_batch_size", 200))
        xp_batch_pause = float(cat_cfg.get("gaia_xp_batch_pause_sec", 0.5))
        archive_retries = int(cat_cfg.get("gaia_archive_max_retries", 3))
        retry_base_delay = float(cat_cfg.get("gaia_archive_retry_base_delay_sec", 2.0))
        xp_order = str(cat_cfg.get("gaia_xp_order_by", "brightness")).strip().lower()
        xp_show_progress = bool(cat_cfg.get("gaia_xp_show_progress", False))
        prefetch_factor = int(cat_cfg.get("gaia_nearest_prefetch_factor", 50))
        prefetch_min = int(cat_cfg.get("gaia_nearest_prefetch_min", 200))
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
                "Querying Gaia DR3 (synthetic photometry, SQL TOP %d -> target %d sources; paced archive: pause %.2fs before/after ADQL)...",
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
                    "Sorting %d rows by on-sky distance to target (Gaia ADQL does not support ORDER BY distance); keeping nearest %d.",
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
            # photometric_systems=None keeps prior behaviour (both systems);
            # an empty list skips GaiaXPy and returns base DR3 photometry
            # only (much faster).
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
                    logger.warning(
                        "gaia_xp_photometric_systems is empty; returning base Gaia DR3 "
                        "photometry only. The catalog will have no SdssStd/JkcStd "
                        "columns, so no standard-band magnitudes can be mapped "
                        "downstream."
                    )
                    return results

                phot_systems = []
                for sys_name in cfg_list:
                    if isinstance(sys_name, str):
                        name = sys_name.strip()
                        try:
                            phot_systems.append(getattr(PhotometricSystem, name))
                        except AttributeError:
                            raise ValueError(
                                f"Unknown GaiaXPy photometric system '{name}'. "
                                f"Valid options: {[p.name for p in PhotometricSystem]}"
                            )
                    else:
                        # Enum values are accepted directly.
                        phot_systems.append(sys_name)

            logger.info(
                "Downloading Gaia XP spectra and synthetic photometry with GaiaXPy (batched: size=%d, inter-batch pause=%.2fs)...",
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

            # Magnitude errors from flux: sigma_m = 2.5/ln(10) * (sigma_F/F).
            factor = 2.5 / np.log(10)

            # Skip bands absent from the merged table (e.g. when XP failed and
            # only base DR3 photometry is available); also guards F<=0.
            for prefix, bands in (("SdssStd", "ugriz"), ("JkcStd", "UBVRI")):
                for band in bands:
                    f_col = f"{prefix}_flux_{band}"
                    e_col = f"{prefix}_flux_error_{band}"
                    m_err_col = f"{prefix}_mag_error_{band}"
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
        tap_service_url = "https://datalab.noirlab.edu/tap"

        # `type` is selected so point sources (PSF) can be separated from
        # extended objects (REX, EXP, DEV, SER = galaxies/resolved sources).
        query = f"""
            SELECT TOP 1000 ra, dec, type, mag_g, mag_r, mag_i, mag_z,
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

            tap = TapPlus(url=tap_service_url)
            result = tap.launch_job(query)
            table = result.get_results().to_pandas()

            # Point sources only; non-PSF types are extended/galaxies.
            if "type" in table.columns:
                n_before = len(table)
                table = table[table["type"].astype(str).str.upper() == "PSF"].copy()
                n_gal = n_before - len(table)
                if n_gal > 0:
                    logging.info(
                        "Legacy Survey: removed %d extended sources (type != PSF); %d point sources remain.",
                        n_gal, len(table),
                    )
            table = table.drop(columns=["type"], errors="ignore")

            filters = ["g", "r", "i", "z"]

            for f in filters:
                table[f"mag_{f}_e"] = [0.01] * len(table)

            return table

        except Exception as e:
            logger.warning("Error during query: %s", e)
            return None

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def fetch_refcat2_field(self, ra, dec, credentials, nsources=1000, sr=0.5,
                             max_retries=3, retry_delay=5.0):
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
        max_retries : int, optional
            Maximum number of retry attempts for transient MAST errors (default 3).
        retry_delay : float, optional
            Initial delay between retries in seconds; doubles each retry (default 5.0).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the catalog data, or None if all retries fail.
        """
        from mastcasjobs import MastCasJobs

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Random suffix gives each attempt a unique MAST table name.
                name = f"autophot_{''.join(random.choices(string.ascii_uppercase, k=5))}"
                logger.info(
                    f"Fetching ATLAS-RefCat2 catalog from MAST over {sr:.3f} deg field-of-view centered at ra = {ra:.1f} dec = {dec:.1f}"
                    + (f" (attempt {attempt}/{max_retries})" if attempt > 1 else "")
                )

                table = [
                    "RA", "Dec", "g", "dg", "r", "dr",
                    "i", "di", "z", "dz", "J", "dJ", "H", "dH", "K", "dK",
                ]

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

                logger.debug("SQL Query: %s", q)

                job = MastCasJobs(context="HLSP_ATLAS_REFCAT2", **credentials)

                # drop_table_if_exists can fail with transient MAST SQL errors;
                # since we use random table names, the table almost never exists.
                # Make this non-fatal so the actual query can still proceed.
                try:
                    job.drop_table_if_exists(name)
                except Exception as drop_err:
                    logger.debug(
                        f"drop_table_if_exists failed (non-fatal, table likely new): {drop_err}"
                    )

                jobid = job.submit(q, task_name=f"refcat catalog search {ra:.5f} {dec:.5f}")

                status = job.monitor(jobid)

                # CasJobs status 3/4 means the job failed.
                if status[0] in (3, 4):
                    raise Exception(f"Job failed with status {status[0]}: {status[1]}")

                tab = job.get_table(name, format="CSV")
                try:
                    job.drop_table_if_exists(name)
                except Exception as drop_err:
                    logger.debug("Post-query drop_table failed (non-fatal): %s", drop_err)

                tab = tab.to_pandas()

                # Drop rows containing a 0 (missing photometry sentinel).
                tab = tab[~(tab == 0).any(axis=1)]
                tab.reset_index(drop=True, inplace=True)

                return tab

            except Exception as e:
                last_error = e
                logger.warning(
                    f"RefCat2 fetch attempt {attempt}/{max_retries} failed: {e}"
                )
                if attempt < max_retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    logger.info("Retrying in %.0fs...", delay)
                    time.sleep(delay)

        logger.error("\n> Catalog retrieval failed after %d attempts! \n", max_retries)
        logger.error("Last error: %s: %s", type(last_error).__name__, last_error)
        return None

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
            Search radius around the target in **arcminutes** (default is 10).
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

            target_ra = target_coords.ra.degree
            target_dec = target_coords.dec.degree

            if target_name is None:
                if target_ra is not None and target_dec is not None:
                    target_name = f"target_ra_{target_ra:.6f}_dec_{target_dec:.6f}"
                else:
                    target_name = "target"
            else:
                if "Unknown" not in target_name:
                    target_name = self.input_yaml.get("target_name", "Transient")

            if not catalog_custom_fpath:
                catalog_custom_fpath = self.input_yaml["catalog"].get(
                    "catalog_custom_fpath", None
                )

            wdir = self.input_yaml.get("wdir")
            if not wdir:
                raise ValueError("Working directory (wdir) is not set in input YAML.")

            dirname = os.path.join(wdir, "catalog_queries")
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            catalog_dir = os.path.join(dirname, catalogName)
            pathlib.Path(catalog_dir).mkdir(parents=True, exist_ok=True)
            target_dir = reduce(
                os.path.join, [dirname, catalogName, target_name.lower()]
            )
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

            fname = f"{target_name}_r_{radius:.1f}arcmins_{catalogName}_target_ra_{target_ra:.6f}_dec_{target_dec:.6f}"

            # radius is arcmin; catalog queries take degrees.
            radius_deg = radius / 60

            if catalogName == "custom":
                if not catalog_custom_fpath:
                    logger.critical(
                        'Custom catalog selected but "catalog_custom_fpath" is not defined.'
                    )
                    return None
                selectedCatalog = pd.read_csv(catalog_custom_fpath)

            elif os.path.isfile(os.path.join(target_dir, f"{fname}.csv")):
                logger.info("Existing %s catalog found for %s", catalogName.upper(), target_name)
                selectedCatalog = (
                    Table.read(os.path.join(target_dir, f"{fname}.csv"), format="csv")
                    .to_pandas()
                    .fillna(np.nan)
                )
                # Dedup cached catalog: earlier runs may have written duplicates.
                if not selectedCatalog.empty and {"RA", "DEC"}.issubset(selectedCatalog.columns):
                    n_before = len(selectedCatalog)
                    selectedCatalog = _skycoord_dedup_keep_one(selectedCatalog, sep_threshold_arcsec=0.1)
                    n_dups = n_before - len(selectedCatalog)
                    if n_dups > 0:
                        logger.warning(
                            f"Removed {n_dups} duplicates from cached {catalogName.upper()} catalog"
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
                    else:
                        # objType is kept so point sources can be separated
                        # from galaxies below.
                        selected_cols = [
                            "ID",
                            "ra",
                            "dec",
                            "objType",
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

                        # Keep only columns the result table actually has.
                        available_cols = [
                            col for col in selected_cols if col in result.colnames
                        ]

                        selectedCatalog = result[available_cols].to_pandas()

                        # Point sources only: TIC objType bitmask STAR = 0x300000;
                        # galaxies and extended objects are excluded.
                        if "objType" in selectedCatalog.columns:
                            n_before = len(selectedCatalog)
                            _obj_type = pd.to_numeric(selectedCatalog["objType"], errors="coerce")
                            _is_star = _obj_type == 0x300000
                            if _is_star.sum() > 0:
                                selectedCatalog = selectedCatalog[_is_star].copy()
                                n_gal = n_before - len(selectedCatalog)
                                if n_gal > 0:
                                    logger.info(
                                        "TIC: removed %d non-stellar sources (objType != STAR); %d stars remain.",
                                        n_gal, len(selectedCatalog),
                                    )
                            selectedCatalog = selectedCatalog.drop(columns=["objType"], errors="ignore")

                        # Write to target_dir (not cwd) to avoid misplaced files.
                        csv_path = os.path.join(target_dir, f"{fname}.csv")
                        selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "gaia":
                    # Gaia DR3+XP uses a smaller radius than the full catalog
                    # search radius to reduce archive load. Hard limit of
                    # 10 arcmin (10/60 deg), optionally smaller if
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
                            "gaia_xp_max_sources", 500
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

                    selectedCatalog = result
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "refcat":

                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    logger.warning(
                        "REFCAT requires MAST CasJobs credentials. Set `default_input.catalog.MASTcasjobs_wsid` and `default_input.catalog.MASTcasjobs_pwd` (or provide them via environment/local overrides)."
                    )
                    # Some auth backends reject non-str; cast and strip here.
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
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

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
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

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
                            # Guard column existence before filtering
                            if "mode" in selectedCatalog.columns:
                                selectedCatalog = selectedCatalog[
                                    selectedCatalog["mode"] == 1
                                ]
                            if "cl" in selectedCatalog.columns:
                                selectedCatalog = selectedCatalog[
                                    selectedCatalog["cl"] == 6
                                ]
                    if catalogName == "apass":
                            # APASS `cls` column: 'A' = stellar, 'G' = galaxy.
                            # Stars only for zeropoint calibration.
                            if "cls" in selectedCatalog.columns:
                                n_before = len(selectedCatalog)
                                selectedCatalog = selectedCatalog[
                                    selectedCatalog["cls"].astype(str).str.upper() == "A"
                                ]
                                n_gal = n_before - len(selectedCatalog)
                                if n_gal > 0:
                                    logger.info(
                                        "APASS: removed %d non-stellar sources (cls != 'A'); %d stars remain.",
                                        n_gal, len(selectedCatalog),
                                    )
                    # Validate before writing (covers empty query and post-filter empty).
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

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
                    # Write temp file to target_dir (not cwd) to avoid misplaced files.
                    temp_vot_path = os.path.join(target_dir, "temp.vot")
                    try:
                        logger.info("Downloading Sequence Stars from SkyMapper")
                        response = requests.get(server, params=params, timeout=60)
                        response.raise_for_status()
                        with open(temp_vot_path, "wb") as f:
                            f.write(response.content)
                        selectedCatalog = (
                            parse_single_table(temp_vot_path)
                            .to_table(use_names_over_ids=True)
                            .to_pandas()
                        )
                    finally:
                        if os.path.exists(temp_vot_path):
                            os.remove(temp_vot_path)
                    # Guard column existence before filtering.
                    if "class_star" in selectedCatalog.columns:
                        selectedCatalog = selectedCatalog[
                            selectedCatalog["class_star"] > 0.8
                        ]
                    if "flags" in selectedCatalog.columns:
                        selectedCatalog = selectedCatalog[selectedCatalog["flags"] <= 1]
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "pan_starrs":
                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    # Direct API request: the MAST JSON endpoint returns
                    # string "None" values that need explicit handling below.
                    try:
                        ra = float(target_coords.ra.degree)
                        dec = float(target_coords.dec.degree)

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
                            selectedCatalog = pd.DataFrame(data)
                            # Normalize null-like strings to NaN.
                            selectedCatalog = selectedCatalog.replace(
                                ['None', 'none', 'NONE', 'null', 'NULL', 'nan', 'NaN'], np.nan
                            )
                            # Coerce numeric columns; name columns stay strings.
                            for col in selectedCatalog.columns:
                                if col not in ['objName', 'objAltName1', 'objAltName2', 'objAltName3']:
                                    try:
                                        selectedCatalog[col] = pd.to_numeric(selectedCatalog[col], errors='coerce')
                                    except Exception:
                                        pass
                            
                        logger.info("Retrieved %s Pan-STARRS sources", len(selectedCatalog))
                        
                    except Exception as api_exc:
                        logger.warning("Direct Pan-STARRS API failed (%s), using empty catalog", api_exc)
                        selectedCatalog = pd.DataFrame()
                    
                    # Catch remaining null sentinels (numeric -999 included).
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
                        # Kron magnitudes for star-galaxy separation:
                        # stars have PSF ~ Kron; galaxies have Kron > PSF.
                        "rMeanKronMag",
                        "rMeanKronMagErr",
                    ]
                    # Keep only columns present in the API response.
                    missing_cols = [c for c in columns if c not in selectedCatalog.columns]
                    if missing_cols:
                        logger.warning(
                            "Pan-STARRS response missing expected columns: %s - they will be absent from the catalog",
                            missing_cols,
                        )
                    available_columns = [c for c in columns if c in selectedCatalog.columns]
                    selectedCatalog = selectedCatalog[available_columns]

                    # Star-galaxy separation: compare PSF and Kron magnitudes.
                    # Stars: |PSF - Kron| < 0.1 mag (point sources).
                    # Galaxies: Kron > PSF by > 0.1 mag (extended flux).
                    if {"rMeanPSFMag", "rMeanKronMag"}.issubset(selectedCatalog.columns):
                        _psf = pd.to_numeric(selectedCatalog["rMeanPSFMag"], errors="coerce")
                        _kron = pd.to_numeric(selectedCatalog["rMeanKronMag"], errors="coerce")
                        _both_finite = np.isfinite(_psf) & np.isfinite(_kron)
                        _is_star = _both_finite & (np.abs(_psf - _kron) < 0.1)
                        # Conservative keep: star-like, or Kron mag missing.
                        _keep = _is_star | ~np.isfinite(_kron)
                        n_before = len(selectedCatalog)
                        selectedCatalog = selectedCatalog[_keep].copy()
                        n_gal = n_before - len(selectedCatalog)
                        if n_gal > 0:
                            logger.info(
                                "Pan-STARRS: removed %d extended sources (|PSF-Kron| >= 0.1 mag); %d point sources remain.",
                                n_gal, len(selectedCatalog),
                            )
                        # Kron columns were only needed for the star cut.
                        selectedCatalog = selectedCatalog.drop(
                            columns=[c for c in ["rMeanKronMag", "rMeanKronMagErr"] if c in selectedCatalog.columns],
                            errors="ignore",
                        )

                    if {"raMean", "decMean"}.issubset(selectedCatalog.columns):
                        coords = SkyCoord(
                            ra=selectedCatalog["raMean"].values * u.deg,
                            dec=selectedCatalog["decMean"].values * u.deg,
                        )
                        distances = target_coords.separation(coords)
                        selectedCatalog["distance"] = distances.arcsecond
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, target_coords, radius
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

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
            logger.warning("%s %s %d %s", exc_type, fname, exc_tb.tb_lineno, e)
            # Propagate catalog failures as hard stops: downstream calibration
            # should not proceed without a valid catalog.
            raise

        if max_sources is not None and selectedCatalog is not None and len(selectedCatalog) > max_sources:
            logger.info(
                "Limiting catalog from %s to %s sources",
                len(selectedCatalog), max_sources,
            )
            # Prefer the sources nearest the target when RA/DEC exist.
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
                selectedCatalog = selectedCatalog.head(max_sources)
            logger.info("Catalog limited to %s sources", len(selectedCatalog))

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
            logger.info("Cleaning %s sources", len(selectedCatalog))

        try:
            # Catalog can be empty after a service failure (e.g. Gaia).
            if selectedCatalog is None or len(selectedCatalog) == 0:
                logger.warning("Selected catalog is empty; skipping catalog cleaning.")
                return None

            # --- Load catalog configuration ---
            filepath = os.path.dirname(os.path.abspath(__file__))
            catalog_autophot_input_yml = "catalog.yml"
            catalogName = catalogName or self.input_yaml["catalog"]["use_catalog"]

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
                n_far = too_far.sum()
                if n_far > 0:
                    logger.info(
                        f"Removing {n_far} sources that are greater than {max_distance:.1f} arcmins from the target"
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

                    # Angular pre-filter must match how the catalog was queried
                    # (usually within max_distance arcmin of the target). A
                    # CRVAL + fixed 1 deg cap wrongly drops on-chip sources on
                    # wide stacks/coadds where CRVAL sits far from the field
                    # center (common after astrometry.net SIP updates).
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
                        # Query radius plus a generous margin, in arcsec.
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
                            "Catalog WCS pre-filter removed all sources (sky vs ref); relaxing filter and keeping full list for pixel conversion."
                        )
                        valid_indices = np.ones(len(ra_values), dtype=bool)

                    ra_values = ra_values[valid_indices]
                    dec_values = dec_values[valid_indices]

                    # world_to_pixel applies full distortion (SIP etc.);
                    # wcs_world2pix can disagree with all_pix2world /
                    # solve-field SIP headers used elsewhere.
                    coords = SkyCoord(
                        ra=ra_values * u.deg, dec=dec_values * u.deg, frame="icrs"
                    )
                    x_pix, y_pix = image_wcs.world_to_pixel(coords)
                    x_pix = np.asarray(x_pix, dtype=float).ravel()
                    y_pix = np.asarray(y_pix, dtype=float).ravel()

                    outputCatalog = outputCatalog.iloc[valid_indices].copy()
                    selectedCatalog = selectedCatalog.iloc[valid_indices].copy()
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
            logger.debug("Populating filter columns for %s; input catalog columns: %s", image_filter, list(selectedCatalog.columns))

            # Custom catalogs: auto-detect filter columns as <band>/<band>_err
            # pairs so arbitrary filter names work without catalog.yml entries.
            if catalogName == "custom":
                import re
                filter_cols = []
                for col in selectedCatalog.columns:
                    if str(col).endswith('_err'):
                        continue
                    err_col = f"{col}_err"
                    if err_col in selectedCatalog.columns:
                        filter_cols.append(col)
                        logger.debug("Auto-detected filter column pair: %s / %s", col, err_col)

                for col in filter_cols:
                    err_col = f"{col}_err"
                    outputCatalog[col] = selectedCatalog[col].values
                    outputCatalog[err_col] = selectedCatalog[err_col].values
                    logger.debug("Auto-copied custom filter %s from catalog", col)

            # The current image filter must always be populated.
            for col in [image_filter, f"{image_filter}_err"]:
                if col in outputCatalog.columns:
                    logger.debug("Column %s already present in output catalog", col)
                    continue
                # catalog.yml mapping first, then exact-name fallback.
                if col in catalog_keywords and catalog_keywords[col] in selectedCatalog:
                    outputCatalog[col] = selectedCatalog[catalog_keywords[col]].values
                    logger.debug("Mapped %s from catalog_keywords", col)
                elif col in selectedCatalog.columns:
                    outputCatalog[col] = selectedCatalog[col].values
                    logger.debug("Copied %s directly from catalog", col)
                else:
                    logger.warning("Could not find column %s in catalog; available columns: %s", col, list(selectedCatalog.columns))

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
                    # catalog.yml mapping first, then exact-name fallback.
                    if (
                        filter_x in catalog_keywords
                        and catalog_keywords.get(col) in selectedCatalog
                    ):
                        outputCatalog[col] = selectedCatalog[
                            catalog_keywords[col]
                        ].values
                    elif col in selectedCatalog.columns:
                        outputCatalog[col] = selectedCatalog[col].values
                    # NOTE: no case-insensitive fallback - it would conflate
                    # different photometric systems (e.g. SDSS r vs Cousins R).

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
                n_faint = tooFaint.sum()
                if n_faint > 0:
                    logger.info(
                        f"Removing {n_faint} sources that are fainter than the cutoff"
                    )
                    outputCatalog = outputCatalog.loc[~tooFaint]

            # --- Full cleaning procedures ---
            if full_clean:
                # Drop sources missing the image-filter magnitude.
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

            # Final deduplication before returning.
            if not outputCatalog.empty and {"RA", "DEC"}.issubset(outputCatalog.columns):
                n_before = len(outputCatalog)
                outputCatalog = _skycoord_dedup_keep_one(outputCatalog, sep_threshold_arcsec=0.1)
                n_dups = n_before - len(outputCatalog)
                if n_dups > 0:
                    logger.info("Removed %s duplicates during catalog cleaning", n_dups)

            logger.info("%s sources in output catalog", len(outputCatalog))
            logger.debug("Output catalog columns: %s", list(outputCatalog.columns))
            return outputCatalog

        except Exception as e:
            import traceback

            logger.error("Error in catalog cleaning: %s", e)
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
        Undersampled (FWHM <= undersampled_fwhm_threshold, default 2.5 px): 2D Gaussian fit for subpixel accuracy.
        Well-sampled (FWHM > threshold): center-of-mass. Tolerates fully masked cutouts.
        Error-weighted centroiding was removed; this routine always uses
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

            # Box size ~3xFWHM (odd, >=3); undersampled needs >=7 for the
            # 2D Gaussian fit to constrain the core.
            if boxsize is None:
                boxsize = max(3, int(np.ceil(fwhm) * 3))
            boxsize = int(boxsize)
            if undersampled and boxsize < 7:
                boxsize = 7
            boxsize = boxsize if boxsize % 2 != 0 else boxsize + 1
            boxsize = max(boxsize, 3)
            # Border must clear the aperture annulus (aperture + gap + width);
            # annulus parameters match aperture.py conventions.
            phot_cfg = self.input_yaml.get("photometry", {}) or {}
            _ap_radius = phot_cfg.get("aperture_radius")
            ap_radius = float(_ap_radius if _ap_radius is not None else fwhm * 1.7)
            _gap = phot_cfg.get("annulus_gap_fwhm")
            gap_fwhm = float(_gap if _gap is not None else 0.75)
            _width = phot_cfg.get("annulus_width_fwhm")
            width_fwhm = float(_width if _width is not None else 2.0)
            annulus_outer = ap_radius + (gap_fwhm + width_fwhm) * fwhm
            border = max(boxsize, int(np.ceil(annulus_outer)))
            logger.debug("Boxsize: %s px, Border: %s px (annulus outer=%.1f, undersampled=%s)", boxsize, border, annulus_outer, undersampled)

            # Drop sources whose cutout would cross the image border.
            height, width = image.shape
            mask_x = (selectedCatalog["x_pix"] > border) & (
                selectedCatalog["x_pix"] < width - border
            )
            mask_y = (selectedCatalog["y_pix"] > border) & (
                selectedCatalog["y_pix"] < height - border
            )
            mask = mask_x & mask_y

            if num_sources > 1:
                logger.debug("Recentering %s sources within border", sum(mask))
                selectedCatalog = selectedCatalog.loc[mask].copy()

            old_x = selectedCatalog["x_pix"].values
            old_y = selectedCatalog["y_pix"].values
            nan_mask = ~np.isfinite(image)

            # Skip sources whose cutout is fully masked.
            valid_sources = self._check_valid_cutouts(
                image, old_x, old_y, boxsize, nan_mask
            )
            if valid_sources.sum() == 0:
                logger.warning("No sources have valid cutouts - skipping recentering")
                return selectedCatalog.loc[valid_sources]

            logger.debug("%s sources have valid cutouts", valid_sources.sum())

            old_x_valid = old_x[valid_sources]
            old_y_valid = old_y[valid_sources]

            # Undersampled: 2D Gaussian for subpixel accuracy; else COM.
            if undersampled:
                centroid_method = "2D Gaussian"
                centroid_func = centroid_2dg
            else:
                centroid_method = "center-of-mass"
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

            # NaN-fill invalid sources so indices stay aligned.
            x = np.full(len(old_x), np.nan)
            y = np.full(len(old_y), np.nan)
            x[valid_sources] = x_valid
            y[valid_sources] = y_valid

            x_err = np.full(len(old_x), np.nan)
            y_err = np.full(len(old_y), np.nan)
            x_err[valid_sources] = x_err_valid
            y_err[valid_sources] = y_err_valid

            selectedCatalog.loc[:, "x_pix"] = x
            selectedCatalog.loc[:, "y_pix"] = y
            if "x_pix_err" not in selectedCatalog.columns:
                selectedCatalog["x_pix_err"] = np.nan
            if "y_pix_err" not in selectedCatalog.columns:
                selectedCatalog["y_pix_err"] = np.nan
            selectedCatalog.loc[:, "x_pix_err"] = x_err
            selectedCatalog.loc[:, "y_pix_err"] = y_err

            # Drop sources recentered outside the border.
            mask_x = (selectedCatalog["x_pix"] >= border) & (
                selectedCatalog["x_pix"] < width - border
            )
            mask_y = (selectedCatalog["y_pix"] >= border) & (
                selectedCatalog["y_pix"] < height - border
            )
            mask = mask_x & mask_y

            if sum(~mask) > 0:
                logger.warning("Failed to recenter %s sources - ignoring", sum(~mask))
                selectedCatalog = selectedCatalog.loc[mask].copy()

            valid = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(old_x)
                & np.isfinite(old_y)
            )
            average_offset = np.nan
            if valid.any():
                average_offset = np.nanmedian(
                    pix_dist(x[valid], old_x[valid], y[valid], old_y[valid])
                )
            else:
                logger.warning("No valid sources to compute median offset.")

            selectedCatalog = selectedCatalog.loc[
                selectedCatalog[["x_pix", "y_pix"]].notna().all(axis=1)
            ]
            offset_txt = (
                f", median offset {average_offset:.1f} px"
                if np.isfinite(average_offset)
                else ""
            )
            logger.info(
                "Recentered %d sources (%s centroiding%s)",
                len(selectedCatalog),
                centroid_method,
                offset_txt,
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
            x_min = int(x - half_box)
            x_max = int(x + half_box + 1)
            y_min = int(y - half_box)
            y_max = int(y + half_box + 1)

            if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
                valid_mask[i] = False
                continue

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

        # Small-angle approximation is fine here: tolerance is in arcsec,
        # so separations are always << 1 degree.
        dec_rad = np.radians(dec)
        ra_vals = catalog["RA"].values
        dec_vals = catalog["DEC"].values

        delta_ra = (ra_vals - ra) * np.cos(dec_rad)   # cos(dec) correction
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

        Returns:
        --------
        pd.DataFrame
            Combined catalog DataFrame.
        """
        catalog_list_str = ",".join([i.upper() for i in catalog_list])
        logger.info(log_step(f"Custom catalog: {catalog_list_str}"))

        if not target_name:
            target_name = self.input_yaml.get("target_name", "Transient")

        # Target RA/DEC in the filename makes the cache field-specific;
        # otherwise catalogs from different targets with the same name get
        # reused (N in the ZP legend then exceeds the real source count).
        target_ra = target_coords.ra.degree
        target_dec = target_coords.dec.degree
        fname = f"{target_name}_r_{radius}arcmins_target_ra_{target_ra:.6f}_dec_{target_dec:.6f}_CUSTOM.csv"
        wdir = self.input_yaml.get("wdir")
        if not wdir:
            raise ValueError("Working directory (wdir) is not set in input YAML.")

        dirname = os.path.join(wdir, "catalog_queries")
        dirname = os.path.join(dirname, "custom")

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

        updated_filter_list = []
        for filter in filter_list:
            updated_filter_list.append(filter)
            updated_filter_list.append(f"{filter}_err")

        # Matching tolerance in arcseconds (used by `find_source`).
        tolerance_arcsec = max_separation

        cols = ["RA", "DEC"] + updated_filter_list

        # A cached combined catalog is reused (after deduplication).
        if os.path.isfile(fpath):
            logger.info("Loading existing custom catalog from %s", fpath)
            existing_catalog = pd.read_csv(fpath)
            if not existing_catalog.empty and {"RA", "DEC"}.issubset(existing_catalog.columns):
                # Keep one member of every close pair.
                n_before = len(existing_catalog)
                existing_catalog = _skycoord_dedup_keep_one(existing_catalog, sep_threshold_arcsec=0.1)
                n_dups = n_before - len(existing_catalog)
                if n_dups > 0:
                    logger.warning(
                        f"Removed {n_dups} duplicates from existing cached catalog - resaving clean version"
                    )
                    existing_catalog.to_csv(fpath, index=False, float_format="%.6f")
            elif not existing_catalog.empty:
                logger.warning(
                    "Existing cached catalog at %s is missing RA/DEC columns - skipping deduplication",
                    fpath,
                )
            return existing_catalog
        
        output_catalog = pd.DataFrame(columns=cols)

        for catalogName in catalog_list:
            logger.info("Getting %s catalog", catalogName)
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

            # Cross-match via a single match_to_catalog_sky call (O(N log N)),
            # then split into new / update groups. The empty-catalog branch
            # (first iteration) just takes every row.
            new_rows = []
            if output_catalog.empty:
                # to_dict('records') avoids one Series per row (iterrows is
                # much slower on large catalogs).
                catalog_cols = [c for c in cols if c in catalog_i.columns]
                missing_cols = [c for c in cols if c not in catalog_i.columns]
                sub = catalog_i[catalog_cols].copy()
                for c in missing_cols:
                    sub[c] = np.nan
                new_rows = sub[cols].to_dict("records")
            else:
                coords_existing = SkyCoord(
                    ra=output_catalog["RA"].values * u.degree,
                    dec=output_catalog["DEC"].values * u.degree,
                )
                coords_new = SkyCoord(
                    ra=catalog_i["RA"].values * u.degree,
                    dec=catalog_i["DEC"].values * u.degree,
                )
                idx_match, sep2d, _ = coords_new.match_to_catalog_sky(coords_existing)
                tol_deg = tolerance_arcsec / 3600.0
                matched = sep2d.deg < tol_deg

                # New (unmatched) sources via boolean mask - no iterrows().
                new_mask = ~matched
                if new_mask.any():
                    catalog_cols = [c for c in cols if c in catalog_i.columns]
                    missing_cols = [c for c in cols if c not in catalog_i.columns]
                    sub = catalog_i.loc[new_mask, catalog_cols].copy()
                    for c in missing_cols:
                        sub[c] = np.nan
                    new_rows = sub[cols].to_dict("records")

                # Fill in missing filter values for matched sources.
                if matched.any():
                    matched_new_idx = np.where(matched)[0]
                    out_indices = output_catalog.index[idx_match[matched_new_idx]]
                    for filter_name in updated_filter_list:
                        if filter_name not in catalog_i.columns:
                            continue
                        new_vals = catalog_i[filter_name].to_numpy()
                        existing_vals = output_catalog.loc[out_indices, filter_name].to_numpy()
                        fill_mask = pd.isna(existing_vals) & pd.notna(new_vals[matched_new_idx])
                        if fill_mask.any():
                            output_catalog.loc[
                                out_indices[fill_mask], filter_name
                            ] = new_vals[matched_new_idx[fill_mask]]

            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)
                output_catalog = pd.concat(
                    [output_catalog, new_rows_df], ignore_index=True
                )
                logger.info(
                    "Added %d sources from %s, catalog now has %d sources",
                    len(new_rows_df), catalogName, len(output_catalog),
                )

        # Final deduplication: keep exactly one member of every close pair.
        if not output_catalog.empty:
            n_before_dedup = len(output_catalog)
            output_catalog = _skycoord_dedup_keep_one(output_catalog, sep_threshold_arcsec=0.1)
            n_removed = n_before_dedup - len(output_catalog)
            logger.info(
                f"Final catalog: {len(output_catalog)} sources (removed {n_removed} duplicates)"
            )
            if n_removed > 0:
                logger.warning(
                    f"Removed {n_removed} duplicate sources from final combined catalog"
                )
        
        output_catalog.to_csv(fpath, index=False, float_format="%.6f")
        logger.debug("Saved clean catalog to %s", fpath)
        return output_catalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def check_saturation_range(self, catalog, threshold=5):
        """
        Checks the saturation range of a given catalog by plotting catalog magnitude vs. instrumental magnitude
        and fitting a straight line with slope fixed to ~1 using RANSAC.
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
        fit_params = {
            "slope": None,
            "intercept": None,
            "intercept_error": None,
            "n_inliers": 0,
        }
        saturation_range = [0, np.inf]

        try:
            fpath = self.input_yaml.get("fpath", "")
            if not fpath:
                raise ValueError("File path is missing from input_yaml.")

            base_name = os.path.splitext(os.path.basename(fpath))[0]
            write_dir = os.path.dirname(fpath)
            logger.info(
                log_step(f"Linearity: saturation check ({len(catalog)} sources)")
            )

            use_filter = self.input_yaml.get("imageFilter")
            if not use_filter:
                raise ValueError("Missing 'imageFilter' in input YAML.")

            required_columns = ["flux_AP", use_filter, f"{use_filter}_err"]
            missing_columns = [
                col for col in required_columns if col not in catalog.columns
            ]
            if missing_columns:
                raise KeyError(
                    f"Missing required columns in catalog: {missing_columns}"
                )

            # NOTE: sky sigma-clipping and threshold filtering are already done
            # by Zeropoint.clean() before this method is called.  Repeating them
            # here with different parameters (threshold=5 vs 3.0) causes additional
            # source loss.  Only apply a threshold cut if it wasn't already applied.
            if "threshold" in catalog.columns:
                threshold_cut = catalog["threshold"] < threshold
                n_thresh = int(np.sum(threshold_cut))
                if n_thresh > 0:
                    logger.info(
                        f"Removing {n_thresh} sources with threshold < {threshold}"
                    )
                    catalog = catalog[~threshold_cut]

            flux = catalog["flux_AP"].values
            flux_err = catalog["flux_AP_err"].values
            # NaN for non-positive fluxes (same convention as functions.mag)
            flux_safe = flux.astype(float).copy()
            flux_safe[flux_safe <= 0] = np.nan
            inst_mag = -2.5 * np.log10(flux_safe)
            inst_mag_err = 2.5 / np.log(10) * (flux_err / flux_safe)
            catalog_mag = catalog[use_filter].values
            catalog_mag_err = catalog[f"{use_filter}_err"].values

            # 0.5 mag combined-error cut matches the _prepare_catalog threshold.
            error_mask = np.sqrt(catalog_mag_err**2 + inst_mag_err**2) < 0.5
            clean_catalog = catalog[error_mask].copy()

            if len(clean_catalog) < 2:
                logger.warning("Too few points left after error cut.")
                return clean_catalog, fit_params, saturation_range

            # The high-S/N subset is used ONLY to fit the RANSAC model; the
            # model is then applied to the FULL catalog so faint but valid
            # sources can still be inliers.
            flux = clean_catalog["flux_AP"].values
            flux_err = clean_catalog["flux_AP_err"].values
            flux_err_safe = np.maximum(flux_err, 1e-10)  # guard flux_err=0
            snr_values = np.abs(flux) / flux_err_safe

            # Step down the S/N floor until RANSAC has enough sources.
            min_snr_thresholds = [100, 75, 50, 30, 20, 10]
            selected_indices = None
            for min_snr in min_snr_thresholds:
                high_snr_mask = snr_values >= min_snr
                n_high_snr = np.sum(high_snr_mask)
                if n_high_snr >= 10:  # RANSAC needs at least ~10 sources
                    selected_indices = high_snr_mask
                    logger.debug(
                        f"Selected {n_high_snr} high S/N sources (SNR >= {min_snr}) for linearity fit"
                    )
                    break

            # Instrumental magnitudes for the FULL clean catalog.
            flux_safe = flux.astype(float).copy()
            flux_safe[flux_safe <= 0] = np.nan
            inst_mag_linear = -2.5 * np.log10(flux_safe)
            inst_mag_err_linear = 2.5 / np.log(10) * (flux_err / flux_safe)
            catalog_mag_linear = clean_catalog[use_filter].values
            catalog_mag_err_linear = clean_catalog[f"{use_filter}_err"].values

            # Local import: reuse zeropoint's regressor, avoid duplication.
            from zeropoint import PenalisedSlopeRegressor
            ConstrainedSlopeRegressor = PenalisedSlopeRegressor

            # Fit on the high-S/N subset if available, then apply the model
            # to the FULL catalog to identify all inliers.
            X_full = inst_mag_linear.reshape(-1, 1)
            y_full = catalog_mag_linear

            if selected_indices is not None and np.sum(selected_indices) >= 10:
                X_fit = X_full[selected_indices]
                y_fit = y_full[selected_indices]
            else:
                X_fit = X_full
                y_fit = y_full

            if len(X_fit) > 1:
                base_estimator = ConstrainedSlopeRegressor(
                    slope_constraint=1.0, slope_tolerance=0  # slope fixed to 1
                )
                # Residual threshold adapts to the scatter in this field.
                initial_mad = np.median(np.abs(y_fit - np.median(y_fit)))
                ransac_residual_threshold = max(3.0 * initial_mad, 0.1)
                ransac = RANSACRegressor(
                    estimator=base_estimator,
                    residual_threshold=ransac_residual_threshold,
                    max_trials=500,
                    min_samples=0.25,
                )
                ransac.fit(X_fit, y_fit)
                slope = ransac.estimator_.slope_
                intercept = ransac.estimator_.intercept_

                # Inliers are computed on the FULL catalog, not the fit subset.
                residuals_full = y_full - (slope * X_full.flatten() + intercept)
                inlier_mask = np.abs(residuals_full) < ransac_residual_threshold

                # Post-RANSAC sigma clip on the full-catalog inliers;
                # sigma=3.0 (not 2.5) is more stable for small samples.
                n_sigma_outliers = 0
                if np.sum(inlier_mask) > 5:
                    inlier_residuals = residuals_full[inlier_mask]
                    clip_sigma = 3.0 if np.sum(inlier_mask) < 30 else 2.5
                    clipped = sigma_clip(inlier_residuals, sigma=clip_sigma, maxiters=5)
                    residual_mask = np.asarray(~clipped.mask, dtype=bool).flatten()
                    if residual_mask.shape[0] == np.sum(inlier_mask):
                        inlier_mask[inlier_mask] = residual_mask
                    else:
                        logger.warning("Shape mismatch in residual masking: %s vs %s, skipping sigma clip update", residual_mask.shape[0], np.sum(inlier_mask))
                    n_sigma_outliers = int(np.sum(~residual_mask))

                # Recompute the intercept on the final inlier set.
                if np.sum(inlier_mask) > 1 and np.isfinite(slope):
                    try:
                        intercept = float(
                            np.nanmedian(
                                y_full[inlier_mask] - slope * X_full[inlier_mask].flatten()
                            )
                        )
                    except Exception:
                        pass

                logger.debug(
                    f"RANSAC: {np.sum(inlier_mask)}/{len(X_full)} inliers "
                    f"(fit on {len(X_fit)}, applied to {len(X_full)}), "
                    f"ZP={intercept:.3f}"
                )

                # Intercept error: standard error of the intercept.
                inlier_X = X_full[inlier_mask]
                inlier_y = y_full[inlier_mask]
                residuals = inlier_y - (slope * inlier_X.flatten() + intercept)
                residual_std = np.std(residuals, ddof=1)  # sample std (ddof=1)
                n_points = len(inlier_X)
                x_mean = np.mean(inlier_X)
                x_var = np.var(inlier_X, ddof=1)  # sample variance (ddof=1)

                if x_var <= 0 or not np.isfinite(x_var):
                    logger.warning("Zero or invalid variance in instrumental magnitudes")
                    intercept_error = np.nan
                else:
                    intercept_error = residual_std * np.sqrt(
                        1 / n_points + x_mean**2 / ((n_points - 1) * x_var)
                    )

                fit_params.update(
                    {
                        "slope": slope,
                        "intercept": intercept,
                        "intercept_error": intercept_error,
                        "n_inliers": n_points,
                    }
                )

                # Linearity range = 0.5th to 95th flux percentile of inliers.
                inlier_flux = flux[inlier_mask]
                linear_range_str = ""
                if len(inlier_flux) > 0:
                    min_flux = np.percentile(inlier_flux, 0.5)
                    max_flux = np.percentile(inlier_flux, 95)
                    saturation_range = [min_flux, max_flux]
                    linear_range_str = f" | flux range {min_flux:.0f}-{max_flux:.0f}"
                else:
                    logger.warning("No inliers found for linearity range calculation")

                _zp_err_str = (
                    f" +/- {intercept_error:.3f}"
                    if np.isfinite(intercept_error)
                    else ""
                )
                logger.info(
                    f"Linearity fit: {n_points}/{len(X_full)} inliers "
                    f"(RANSAC on {len(X_fit)} high-S/N, -{n_sigma_outliers} clip) | "
                    f"ZP={intercept:.3f}{_zp_err_str}{linear_range_str}"
                )

                fit_line = lambda x: slope * x + intercept
            else:
                logger.warning("Not enough sources for linear fitting.")
                fit_line = None
                inlier_mask = np.ones_like(catalog_mag_linear, dtype=bool)

            # Plotting
            from plotting_utils import (
                apply_autophot_mplstyle, get_ransac_color, get_marker_size,
                get_alpha, get_line_width, ransac_grid, ransac_savefig,
                set_mag_axes_inverted_xy, format_log_colorbar_ticks,
                get_plot_ext,
            )

            apply_autophot_mplstyle()
            plt.ioff()
            fig, ax1 = plt.subplots(figsize=set_size(540, 1))

            if fit_line:
                predicted = fit_line(inst_mag_linear)
                residuals = catalog_mag_linear - predicted
                # Plot with the same RANSAC inlier mask used for the fit.
                ransac_inliers = inlier_mask if 'inlier_mask' in locals() else np.ones(len(inst_mag_linear), dtype=bool)
                ransac_outliers = ~ransac_inliers

                # S/N-driven marker colours; the norm range covers inliers
                # only -- outliers always get the flat outlier colour.
                _snr_in = snr_values[ransac_inliers] if 'snr_values' in locals() else np.array([])
                finite_snr = _snr_in[np.isfinite(_snr_in) & (_snr_in > 0)]
                snr_norm = None
                if finite_snr.size >= 3 and np.nanmax(finite_snr) > np.nanmin(finite_snr):
                    from matplotlib.colors import LogNorm
                    vmin = max(1.0, float(np.nanmin(finite_snr)))
                    vmax = float(np.nanpercentile(finite_snr, 98))
                    if vmax <= vmin:
                        vmax = vmin * 10.0
                    snr_norm = LogNorm(vmin=vmin, vmax=vmax)
                ms_area = get_marker_size('medium') ** 2

                ax1.errorbar(
                    inst_mag_linear[ransac_outliers],
                    catalog_mag_linear[ransac_outliers],
                    yerr=catalog_mag_err_linear[ransac_outliers],
                    xerr=inst_mag_err_linear[ransac_outliers],
                    fmt="none",
                    ecolor="lightgrey",
                    alpha=get_alpha('medium'),
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    linestyle="None",
                    zorder=2,
                )
                ax1.plot(
                    inst_mag_linear[ransac_outliers],
                    catalog_mag_linear[ransac_outliers],
                    "x", markersize=get_marker_size('medium'),
                    color=get_ransac_color('outliers'),
                    alpha=get_alpha('medium'), linestyle="None",
                    label=f"Outliers [{np.sum(ransac_outliers)}]",
                    zorder=3,
                )
                ax1.errorbar(
                    inst_mag_linear[ransac_inliers],
                    catalog_mag_linear[ransac_inliers],
                    yerr=catalog_mag_err_linear[ransac_inliers],
                    xerr=inst_mag_err_linear[ransac_inliers],
                    fmt="none",
                    ecolor="lightgrey",
                    alpha=get_alpha('dark'),
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    linestyle="None",
                    zorder=4,
                )
                if snr_norm is not None:
                    sc_in = ax1.scatter(
                        inst_mag_linear[ransac_inliers],
                        catalog_mag_linear[ransac_inliers],
                        c=snr_values[ransac_inliers], cmap="viridis", norm=snr_norm,
                        marker="o", s=ms_area, alpha=get_alpha('dark'),
                        label=f"Inliers [{np.sum(ransac_inliers)}]",
                        zorder=5,
                    )
                    cb = fig.colorbar(sc_in, ax=ax1, label="S/N", pad=0.02)
                    format_log_colorbar_ticks(cb, snr_norm.vmin, snr_norm.vmax)
                else:
                    ax1.plot(
                        inst_mag_linear[ransac_inliers],
                        catalog_mag_linear[ransac_inliers],
                        "o", markersize=get_marker_size('medium'),
                        color=get_ransac_color('zeropoint_ap'),
                        alpha=get_alpha('dark'), linestyle="None",
                        label=f"Inliers [{np.sum(ransac_inliers)}]",
                    )
                x_range = np.linspace(inst_mag_linear[ransac_inliers].min(), inst_mag_linear[ransac_inliers].max(), 100)
                y_fit = fit_line(x_range)
                ax1.fill_between(
                    x_range,
                    y_fit - intercept_error,
                    y_fit + intercept_error,
                    color=get_ransac_color('error_band'),
                    alpha=get_alpha('very_light'),
                )
                ax1.plot(
                    x_range,
                    y_fit,
                    color=get_ransac_color('fit'),
                    linestyle="--",
                    lw=get_line_width('medium'),
                    zorder=10,
                    label=(
                        rf"$m_\mathrm{{cal}} = m_\mathrm{{inst}} + {intercept:.2f} \pm {intercept_error:.2f}$"
                    ),
                )

                # Mark the linearity range in instrumental magnitude.
                if len(inlier_flux) > 0:
                    min_inst_mag = -2.5 * np.log10(
                        max_flux
                    )  # Note: brighter objects have smaller magnitudes
                    max_inst_mag = -2.5 * np.log10(min_flux)
                    ax1.axvline(
                        x=min_inst_mag,
                        color=get_ransac_color('error_band'),
                        linestyle=":",
                        lw=get_line_width('thin'),
                        alpha=0.7,
                    )
                    ax1.axvline(
                        x=max_inst_mag,
                        color=get_ransac_color('error_band'),
                        linestyle=":",
                        lw=get_line_width('thin'),
                        alpha=0.7,
                    )

            ax1.set_xlabel(r"Instrumental Magnitude $m_\mathrm{inst}$ [mag]")
            ax1.set_ylabel(rf"Catalog Magnitude $m_\mathrm{{cal,{use_filter}}}$ [mag]")
            set_mag_axes_inverted_xy(ax1)
            ax1.legend(
                loc="upper left", ncol=1, fontsize=8, frameon=False,
            )
            ransac_grid(ax1)
            save_path = os.path.join(write_dir, f"Saturation_{base_name}{get_plot_ext(self.input_yaml)}")
            ransac_savefig(fig, save_path)
            plt.close(fig)

            # Locate the continuous linear region via residual scatter.
            if fit_line:
                # Guard against length mismatches before indexing masks.
                n_clean = len(clean_catalog)
                n_flux = len(flux)
                n_mag = len(catalog_mag_linear)
                n_inst = len(inst_mag_linear)
                n_inlier_mask = len(inlier_mask)
                if not (n_clean == n_flux == n_mag == n_inst == n_inlier_mask):
                    logger.error(
                        f"Array length mismatch: clean_catalog={n_clean}, flux={n_flux}, "
                        f"catalog_mag_linear={n_mag}, inst_mag_linear={n_inst}, inlier_mask={n_inlier_mask}. "
                        f"Skipping robust selection."
                    )
                    clean_catalog = clean_catalog[inlier_mask] if n_clean == n_inlier_mask else clean_catalog
                    return clean_catalog, saturation_range
                
                inlier_catalog = clean_catalog[inlier_mask].copy()
                inlier_flux = flux[inlier_mask]
                inlier_inst_mag = inst_mag_linear[inlier_mask]

                # Keep all RANSAC inliers for zeropoint fitting. The aggressive
                # flux-range selection below removes too many valid faint sources
                # (photon noise is mistaken for non-linearity on modern CCDs).
                # NOTE: Don't reassign clean_catalog yet - we need the full arrays
                # for mask computation. We'll reassign after selection is complete.

                if len(inlier_catalog) > 5:
                    # predict() can return shape (N,1) depending on the sklearn
                    # version; flatten so the subtraction stays (N,)-(N,) and
                    # cannot broadcast to (N,N), which would silently corrupt
                    # central_start/central_end and cause an IndexError when
                    # indexing sorted_flux (size N).
                    predicted_mag = np.asarray(
                        fit_line(inlier_inst_mag.reshape(-1, 1))
                    ).flatten()
                    residuals = (
                        np.asarray(catalog_mag_linear[inlier_mask]).flatten() - predicted_mag
                    )

                    # Sort by flux, bright first (smaller mag = brighter).
                    sort_idx = np.argsort(inlier_flux)[::-1]
                    sorted_flux = inlier_flux[sort_idx]
                    sorted_residuals = residuals[sort_idx]  # already 1D
                    sorted_mag = inlier_inst_mag[sort_idx]

                    # Residual threshold comes from the central 50% of
                    # sources, the most linear region.
                    central_start = len(sorted_residuals) // 4
                    central_end = 3 * len(sorted_residuals) // 4
                    central_residuals = sorted_residuals[central_start:central_end]

                    if len(central_residuals) > 3:
                        median_resid = np.median(central_residuals)
                        mad_residual = np.median(np.abs(central_residuals - median_resid))
                        # Residual floor: 3*MAD, minimum 0.1 mag.
                        residual_threshold = float(np.maximum(3.0 * mad_residual, 0.1))
                    else:
                        residual_threshold = 0.15

                    # Bright end: cut where residuals exceed the threshold
                    # (saturation / non-linearity).
                    bright_cut_idx = 0
                    for i in range(len(sorted_residuals)):
                        resid_val = float(sorted_residuals[i])  # ensure scalar
                        if np.abs(resid_val) > residual_threshold:
                            bright_cut_idx = i + 1  # cut this and brighter
                        else:
                            break

                    # Faint end: cut where the local scatter systematically
                    # exceeds the central scatter. Require 2 consecutive
                    # high-scatter windows so a single noisy window does not
                    # trigger the cut.
                    window_size = max(3, len(sorted_residuals) // 15)
                    faint_cut_idx = len(sorted_residuals)

                    high_scatter_count = 0
                    required_consecutive = 2

                    for i in range(window_size, len(sorted_residuals) - window_size):
                        window_residuals = sorted_residuals[i-window_size:i+window_size]
                        window_mad = np.median(np.abs(window_residuals - np.median(window_residuals)))

                        # High scatter = local MAD > 2x central MAD
                        # (relaxed from 1.5x).
                        if window_mad > 2.0 * mad_residual:
                            high_scatter_count += 1
                            if high_scatter_count >= required_consecutive:
                                faint_cut_idx = i - window_size  # cut before this region
                                break
                        else:
                            high_scatter_count = 0  # reset if scatter drops

                    # Second faint cut: drop sources with large individual
                    # residuals, scanning in from the faint end.
                    for i in range(len(sorted_residuals) - 1, faint_cut_idx - 1, -1):
                        if np.abs(sorted_residuals[i]) > 1.5 * residual_threshold:
                            faint_cut_idx = i  # cut this and fainter
                        else:
                            break

                    # Apply flux range cuts (bright_cut_idx to faint_cut_idx).
                    if bright_cut_idx > 0 or faint_cut_idx < len(sorted_flux):
                        min_linear_flux = sorted_flux[min(faint_cut_idx, len(sorted_flux)-1)]
                        max_linear_flux = sorted_flux[bright_cut_idx] if bright_cut_idx < len(sorted_flux) else sorted_flux[0]

                        if min_linear_flux < max_linear_flux:
                            linear_flux_mask = (flux >= min_linear_flux) & (flux <= max_linear_flux)

                            # Residual filter on the inlier arrays only;
                            # mixing with clean_catalog would mismatch lengths.
                            inlier_catalog_mag_linear = catalog_mag_linear[inlier_mask]
                            inlier_inst_mag_linear = inst_mag_linear[inlier_mask]
                            preds = np.asarray(fit_line(inlier_inst_mag_linear.reshape(-1, 1))).flatten()
                            all_residuals = np.asarray(inlier_catalog_mag_linear).flatten() - preds
                            inlier_residual_mask = np.abs(all_residuals) < residual_threshold
                            
                            # Expand the inlier residual mask back to full
                            # size; lengths must match to avoid indexing errors.
                            n_inliers = np.sum(inlier_mask)
                            if len(inlier_residual_mask) != n_inliers:
                                logger.warning(
                                    f"Array length mismatch: inlier_residual_mask={len(inlier_residual_mask)}, "
                                    f"inlier_mask.sum()={n_inliers}. Skipping residual masking."
                                )
                                linear_residual_mask = inlier_mask.copy()
                            else:
                                linear_residual_mask = np.zeros(len(flux), dtype=bool)
                                linear_residual_mask[inlier_mask] = inlier_residual_mask
                            
                            # Keep sources in the flux range AND with a good residual.
                            final_linear_mask = linear_flux_mask & linear_residual_mask

                            n_bright_cut = np.sum(flux > max_linear_flux)
                            n_faint_cut = np.sum(flux < min_linear_flux)
                            n_outlier_cut = np.sum(linear_flux_mask & ~linear_residual_mask)
                            n_selected = np.sum(final_linear_mask)
                            
                            logger.info(
                                f"Robust linear selection: flux range {min_linear_flux:.1f} - {max_linear_flux:.1f}, "
                                f"residual thresh {residual_threshold:.3f} mag, "
                                f"cut {n_bright_cut} bright + {n_faint_cut} faint + {n_outlier_cut} outlier, "
                                f"would keep {n_selected} sources (keeping all {len(inlier_catalog)} inliers for ZP fit)"
                            )
                            
                            if n_selected > 0:
                                # Saturation range is informational only; all
                                # inliers are kept for zeropoint fitting to
                                # avoid over-aggressive faint-end cuts.
                                inlier_flux = inlier_catalog["flux_AP"].values
                                inlier_linear_mask = (inlier_flux >= min_linear_flux) & (inlier_flux <= max_linear_flux)
                                # linear_residual_mask already matches inlier_catalog size.
                                inlier_linear_mask = inlier_linear_mask & linear_residual_mask[inlier_mask]
                                saturation_range = [
                                    np.percentile(inlier_catalog["flux_AP"].values[inlier_linear_mask], 0.5),
                                    np.percentile(inlier_catalog["flux_AP"].values[inlier_linear_mask], 99.5)
                                ]
                            else:
                                # Nothing passed; fall back to the central
                                # region (inlier_catalog avoids length mismatch).
                                inlier_flux = inlier_catalog["flux_AP"].values
                                inlier_sorted_flux = np.sort(inlier_flux)[::-1]
                                central_start = max(0, len(inlier_sorted_flux) // 3)
                                central_end = min(len(inlier_sorted_flux), 2 * len(inlier_sorted_flux) // 3)
                                central_flux_mask = (inlier_flux >= inlier_sorted_flux[central_end]) & (inlier_flux <= inlier_sorted_flux[central_start])
                                if np.sum(central_flux_mask) > 0:
                                    logger.warning("No sources passed tight criteria, using central %s sources", np.sum(central_flux_mask))
                                else:
                                    logger.warning("No sources passed tight criteria, using all %s inliers", len(inlier_catalog))
                        else:
                            # Invalid range; use the central region
                            # (inlier_catalog avoids length mismatch).
                            inlier_flux = inlier_catalog["flux_AP"].values
                            inlier_sorted_flux = np.sort(inlier_flux)[::-1]
                            central_start = max(0, len(inlier_sorted_flux) // 3)
                            central_end = min(len(inlier_sorted_flux), 2 * len(inlier_sorted_flux) // 3)
                            central_flux_mask = (inlier_flux >= inlier_sorted_flux[central_end]) & (inlier_flux <= inlier_sorted_flux[central_start])
                            if np.sum(central_flux_mask) > 0:
                                pass
                    else:
                        # No cuts needed; still report the tight residual filter.
                        tight_residual_mask = np.abs(residuals) < residual_threshold
                        n_tight_outliers = (~tight_residual_mask).sum()
                        if n_tight_outliers > 0:
                            logger.info("Tight residual filter would remove %s sources (keeping all inliers)", n_tight_outliers)
                else:
                    # Too few inliers for the linear-range selection.
                    logger.warning("Only %s inliers, skipping robust selection", len(inlier_catalog))

                # Reassign clean_catalog only after all mask computations.
                clean_catalog = inlier_catalog

            logger.info("Returning %s sources for zeropoint fitting", len(clean_catalog))
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

        # Small enough: return early without extra logging or work.
        if n_src <= nmax:
            logger.info("Downsampling skipped: %d sources (<= %d target).", n_src, nmax)
            return df.copy()

        # Banner only when downsampling actually runs.
        logger.info(
            log_step(f"Downsampling: {n_src} sources -> {nmax} max")
        )

        if x_col not in df.columns or y_col not in df.columns:
            error_msg = f"DataFrame must contain '{x_col}' and '{y_col}' columns"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Square spatial grid.
        n_bins = int(np.sqrt(nmax))
        sources_per_bin = max(1, nmax // (n_bins**2))
        logger.info(
            f"Using {n_bins}x{n_bins} grid with ~{sources_per_bin} sources per bin"
        )

        x_bins = np.linspace(df[x_col].min(), df[x_col].max(), n_bins + 1)
        y_bins = np.linspace(df[y_col].min(), df[y_col].max(), n_bins + 1)
        logger.debug("X bins range: %.2f to %.2f", x_bins[0], x_bins[-1])
        logger.debug("Y bins range: %.2f to %.2f", y_bins[0], y_bins[-1])

        selected_indices = []
        bin_stats = []  # Track bin statistics for logging

        logger.info("Processing spatial bins...")
        for i in range(n_bins):
            for j in range(n_bins):
                in_xbin = (df[x_col] >= x_bins[i]) & (df[x_col] < x_bins[i + 1])
                in_ybin = (df[y_col] >= y_bins[j]) & (df[y_col] < y_bins[j + 1])
                bin_indices = np.where(in_xbin & in_ybin)[0]
                bin_stats.append(len(bin_indices))

                if len(bin_indices) > 0:
                    # Highest SNR first so bins keep their best sources.
                    if snr_col and snr_col in df.columns:
                        bin_indices = bin_indices[
                            np.argsort(df[snr_col].iloc[bin_indices])[::-1]
                        ]
                        logger.debug(
                            f"Bin ({i},{j}): {len(bin_indices)} sources, sorted by {snr_col}"
                        )
                    else:
                        logger.debug("Bin (%s,%s): %s sources", i, j, len(bin_indices))

                    selected_indices.extend(bin_indices[:sources_per_bin])

        logger.info(
            f"Bin statistics: min={min(bin_stats)}, max={max(bin_stats)}, "
            f"avg={np.mean(bin_stats):.1f} sources per bin"
        )
        logger.info("Selected %s sources from spatial bins", len(selected_indices))

        # Under-filled bins: top up with the highest-SNR remaining sources.
        if len(selected_indices) < nmax:
            logger.warning(
                f"Only {len(selected_indices)} sources selected from bins, "
                f"filling with top {nmax - len(selected_indices)} remaining sources"
            )
            all_indices = set(range(len(df)))
            remaining_indices = list(all_indices - set(selected_indices))

            if snr_col and snr_col in df.columns:
                remaining_indices = sorted(
                    remaining_indices,
                    key=lambda idx: df[snr_col].iloc[idx],
                    reverse=True,
                )
                logger.debug("Sorted remaining sources by SNR")

            additional_count = nmax - len(selected_indices)
            selected_indices.extend(remaining_indices[:additional_count])
            logger.info(
                f"Added {additional_count} additional sources from remaining pool"
            )

        final_count = len(selected_indices)
        if final_count > nmax:
            logger.warning("Selected %s sources (exceeds target %s)", final_count, nmax)
        else:
            logger.info("Final selection: %s sources", final_count)

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
            """Estimate FWHM from the radial profile."""
            profile = radial_profile(data)
            half_max = np.max(profile) * 0.5
            above = np.where(profile >= half_max)[0]
            return 2 * (above[-1] - above[0]) if len(above) > 1 else np.nan

        def calculate_roundness(data):
            """Calculate roundness (1 - b/a) from image moments."""
            y, x = np.indices(data.shape)
            xc, yc = data.shape[1] / 2, data.shape[0] / 2
            x = x - xc
            y = y - yc

            # Second moments -> eigenvalues give major/minor axes.
            mxx = np.sum(x**2 * data) / np.sum(data)
            myy = np.sum(y**2 * data) / np.sum(data)
            mxy = np.sum(x * y * data) / np.sum(data)

            term1 = (mxx + myy) / 2
            term2 = np.sqrt(((mxx - myy) / 2) ** 2 + mxy**2)
            a = term1 + term2
            b = term1 - term2
            return 1 - np.sqrt(b / max(a, 1e-10))  # 0=perfect circle

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
            """Measure central concentration via the Laplacian."""
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
                    logger.debug("Star %s rejected: zero or negative flux.", i)

            except Exception as e:
                logger.warning(
                    f"Star {i} at position {position} skipped due to error: {e}"
                )
                for key in metrics:
                    metrics[key].append(np.nan)

        for key in metrics:
            metrics[key] = np.array(metrics[key])

        fwhm_mask = np.isfinite(metrics["fwhm"])  # & (metrics['fwhm'] < fwhm_threshold)

        # Sigma-clip roundness and sharpness to reject outliers.
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

        roundness_mask = np.isfinite(metrics["roundness"]) & ~roundness_sigma_clip.mask
        sharpness_mask = np.isfinite(metrics["sharpness"]) & ~sharpness_sigma_clip.mask

        # Reject radial-profile outliers.
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

        # Only profile_mask gates rejection; roundness/sharpness are
        # diagnostic-only (still logged above).
        combined_mask = profile_mask

        # Never let the masking empty the catalog.
        if np.sum(combined_mask) < 5:
            logger.warning("All sources have been rejected by the masking process.")
            combined_mask = np.ones_like(combined_mask, dtype=bool)

        inliers = [
            valid_indices[i] for i, is_good in enumerate(combined_mask) if is_good
        ]
        outliers = [
            valid_indices[i] for i, is_good in enumerate(combined_mask) if not is_good
        ]

        logger.info(
            f"Point sources: {len(inliers)} | Extended/rejected sources: {len(outliers)}"
        )
        logger.info(
            f"Rejection breakdown - FWHM: {sum(~fwhm_mask)}, "
            f"Roundness: {sum(~roundness_mask)}, "
            f"Sharpness: {sum(~sharpness_mask)}, "
            f"Profile: {sum(~profile_mask)}"
        )

        if len(inliers) > 0:
            cleaned_catalog = catalog.iloc[inliers].copy()
        else:
            logger.warning(
                "No sources remained after cleaning. Using original catalog."
            )
            cleaned_catalog = catalog.copy()  # fall back to the full catalog

        # Keep per-source metrics in the catalog for diagnostics.
        for key in metrics:
            cleaned_catalog[f"star_{key}"] = metrics[key][combined_mask]

        if self.input_yaml.get("save_cutout_plot", True):
            index_map = {idx: cutout for idx, cutout in zip(valid_indices, cutouts)}
            stars = [index_map[i] for i in inliers]
            num_stars = len(stars)
            side_length = int(np.ceil(np.sqrt(num_stars)))
            ncols = side_length
            nrows = ceil(num_stars / ncols)
            from plotting_utils import apply_autophot_mplstyle, get_plot_ext
            apply_autophot_mplstyle()
            plt.ioff()
            fpath = self.input_yaml["fpath"]
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            # Size each grid cell to the cutout aspect so the equal-aspect
            # panels fill their cells; a fixed figsize shrinks them and
            # leaves gaps no wspace can remove. The 3 trailing columns are
            # a 0.1 spacer plus the radial-profile panel.
            cut_h, cut_w = np.asarray(stars[0]).shape
            ax_h = 1.2
            cell_w = ax_h * (cut_w / max(cut_h, 1))
            left, right, bottom, top = 0.05, 0.985, 0.07, 0.88
            hspace = wspace = 0.01
            fig_w = (ncols + 2.1) * cell_w * (1 + wspace) / (right - left)
            fig_h = nrows * ax_h * (1 + hspace) / (top - bottom)
            fig = plt.figure(figsize=(fig_w, fig_h))
            grid = fig.add_gridspec(
                nrows=nrows,
                ncols=ncols + 3,
                width_ratios=[1] * ncols + [0.1, 1, 1],
                height_ratios=[1] * nrows,
                hspace=hspace,
                wspace=wspace,
                left=left,
                right=right,
                bottom=bottom,
                top=top,
            )
            for i in range(num_stars):
                row, col = divmod(i, ncols)
                ax = fig.add_subplot(grid[row, col])
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(np.asarray(stars[i]))
                norm = ImageNormalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad(color="magenta")
                ax.imshow(
                    stars[i],
                    origin="lower",
                    cmap=cmap,
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
                color="#D94F4F",
                linewidth=0.5,
                label="Median\nRadial\nProfile",
            )
            ax_right.set_xlabel("Radius [pixels]")
            ax_right.set_ylabel("Normalized Flux")
            ax_right.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                            frameon=False, fontsize=8)
            ax_right.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax_right.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
            pos = ax_right.get_position()
            ax_right.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])
            output_path = os.path.join(
                write_dir, f"Zeropoint_Sources_{base}{get_plot_ext(self.input_yaml)}"
            )
            fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close(fig)

        logger.info("Returning %s well-behaved sources", len(cleaned_catalog))
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
        logger = logging.getLogger(__name__)

        logger.info(
            log_step(
                f"Aperture photometry: {len(selectedCatalog)} field sources"
            )
        )

        initialAperture = Aperture(
            input_yaml=self.input_yaml,
            image=image,
        )

        selectedCatalog = initialAperture.measure(sources=selectedCatalog)

        # mag() does not mutate flux_AP.
        instMag = mag(selectedCatalog["flux_AP"])
        inst_col = "inst_" + self.input_yaml["imageFilter"] + "_AP"
        selectedCatalog[inst_col] = np.round(instMag, 3)

        sourceSNR = snr(selectedCatalog["maxPixel"], selectedCatalog["noiseSky"])
        selectedCatalog["snr"] = np.round(sourceSNR, 1)

        logger.info(
            "Instrumental magnitude of %d sources measured"
            % sum(~np.isnan(selectedCatalog[inst_col]))
        )

        return selectedCatalog
