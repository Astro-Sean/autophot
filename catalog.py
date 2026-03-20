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

# Optional: error-weighted centroiding (when error is provided)
try:
    from psf import (
        _centroid_sources_with_error_impl,
        centroid_com_with_error,
        centroid_2dg_with_error,
    )

    _HAS_PSF_CENTROID_ERROR = True
except ImportError:
    _HAS_PSF_CENTROID_ERROR = False

# Scikit-learn imports
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


import traceback

# Local imports
from functions import AutophotYaml, border_msg, pix_dist, mag, snr, set_size
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
        if (
            catalogName is None
            or str(catalogName).strip() == ""
            or str(catalogName).lower() == "none"
        ):
            raise ValueError(
                "No catalog selected. Set `default_input.catalog.use_catalog` in your YAML "
                "(e.g. 'gaia', 'pan_starrs', 'sdss', 'apass', '2mass', 'legacy', 'refcat', or 'custom')."
            )
        return str(catalogName).strip()

    @staticmethod
    def _catalog_len(obj) -> int:
        """Best-effort length for DataFrame/Table/list-like results."""
        if obj is None:
            return 0
        try:
            return int(len(obj))
        except Exception:
            return 0

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

    def gaia_synthetic_photometry(self, ra, dec, radius=0.1, max_sources=5000):
        from gaiaxpy import generate, PhotometricSystem
        from astroquery.gaia import Gaia
        import pandas as pd
        import warnings
        from tqdm import tqdm
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        # ADQL radius is in degrees; caller passes degrees (e.g. radius_deg).
        query = f"""
        SELECT TOP {max_sources} 
               source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
        FROM gaiadr3.gaia_source AS gs
        WHERE has_xp_continuous = 'True'
          AND CONTAINS(POINT('ICRS', gs.ra, gs.dec), 
                       CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
          AND phot_g_mean_mag IS NOT NULL
        ORDER BY phot_g_mean_mag
        """

        try:
            logger.info(
                "Querying Gaia DR3 (synthetic photometry, max %d sources)...",
                max_sources,
            )
            job = Gaia.launch_job(query, dump_to_file=False)
            results = job.get_results().to_pandas()
            logger.info("Gaia DR3 query returned %d sources.", len(results))

            if results.empty:
                return pd.DataFrame()

            source_ids = results["source_id"].astype(str).tolist()
            phot_systems = [PhotometricSystem.SDSS_Std, PhotometricSystem.JKC_Std]

            logger.info(
                "Downloading Gaia XP spectra and synthetic photometry with GaiaXPy..."
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    photometry = generate(source_ids, photometric_system=phot_systems)
                merged = pd.merge(results, photometry, on="source_id", how="inner")
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
    ):
        """
        Download and process catalog data for a given target.

        Parameters:
        -----------
        target_coords : SkyCoord
            SkyCoord object with the RA and DEC of the target.
        catalogName : str
            Name of the catalog to fetch ('refcat', 'gaia', 'apass', '2mass', 'sdss', 'skymapper', 'pan_starrs', 'custom').
        radius : float, optional
            Search radius around the target in degrees (default is 15).
        target_name : str, optional
            Optional name for the target.
        catalog_custom_fpath : str, optional
            File path to a custom catalog (used if catalogName is 'custom').
        include_IR_sequence_data : bool, optional
            Boolean to include IR sequence data from 2MASS (default is True).

        Returns:
        --------
        DataFrame
            DataFrame containing the catalog data, or None if an error occurs.
        """
        logger.info(border_msg("Collecting Sequence sources in field"))

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
                    target_name = self.input_yaml.get("target_name", "target")

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
                    result = self.gaia_synthetic_photometry(
                        ra=target_coords.ra.degree,
                        dec=target_coords.dec.degree,
                        radius=xp_radius_deg,
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
                    selectedCatalog = Catalogs.query_region(
                        target_coords, radius=0.1 * u.deg, catalog="Panstarrs"
                    ).to_pandas()
                    selectedCatalog = selectedCatalog.replace(-999, np.nan)
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
                    # Filter out coordinates that are too far from the image center
                    ra_values = selectedCatalog[catalog_keywords["RA"]].values
                    dec_values = selectedCatalog[catalog_keywords["DEC"]].values

                    # Get the image center
                    center_ra, center_dec = image_wcs.wcs.crval

                    # Calculate angular distance from the center
                    dra = (ra_values - center_ra) * np.cos(np.radians(center_dec))
                    ddec = dec_values - center_dec
                    distance = np.sqrt(dra**2 + ddec**2) * 3600  # arcseconds

                    # Define a reasonable threshold (e.g., 1 degree = 3600 arcseconds)
                    max_distance_threshold = 3600  # arcseconds

                    # Filter out sources too far from the center
                    valid_indices = distance < max_distance_threshold
                    ra_values = ra_values[valid_indices]
                    dec_values = dec_values[valid_indices]

                    # Convert valid coordinates to pixel coordinates
                    x_pix, y_pix = image_wcs.wcs_world2pix(ra_values, dec_values, index)

                    # Update outputCatalog with valid sources only
                    outputCatalog = outputCatalog.iloc[valid_indices].copy()
                    outputCatalog["x_pix"] = x_pix
                    outputCatalog["y_pix"] = y_pix

                except Exception as e:
                    logger.warning(
                        f"Failed to convert RA/DEC to pixel coordinates: {e}. Skipping pixel coordinate conversion."
                    )

            # --- Handle HST mode filtering ---
            if not self.input_yaml.get("HST_mode", False):
                image_filter = self.input_yaml["imageFilter"]
                for col in [image_filter, f"{image_filter}_err"]:
                    if (
                        col in catalog_keywords
                        and catalog_keywords[col] in selectedCatalog
                    ):
                        outputCatalog[col] = selectedCatalog[
                            catalog_keywords[col]
                        ].values

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
                    if (
                        filter_x in catalog_keywords
                        and catalog_keywords.get(col) in selectedCatalog
                    ):
                        outputCatalog[col] = selectedCatalog[
                            catalog_keywords[col]
                        ].values

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

            logger.info(f"{len(outputCatalog)} sources in output catalog")
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
        When `error` is provided (e.g. background RMS), uses uncertainty-weighted centroiding
        and returns centroid errors in x_pix_err, y_pix_err when available.
        """
        try:
            num_sources = len(selectedCatalog)
            logger.info(
                border_msg(
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
            undersampled = fwhm < 3.0

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

            # When error is provided, use uncertainty-weighted centroiding (psf module)
            use_error_weighted = _HAS_PSF_CENTROID_ERROR and error is not None
            if use_error_weighted:
                err_arr = np.asarray(error, dtype=float).copy()
                if err_arr.shape != image.shape:
                    if np.isscalar(err_arr) or err_arr.size == 1:
                        err_arr = np.full(image.shape, float(np.nanmedian(err_arr)))
                    else:
                        err_arr = np.broadcast_to(
                            np.nanmedian(err_arr), image.shape
                        ).copy()
                # Sanitize: zeros/NaNs or very small values make weighted 2DG fit fail
                err_median = np.nanmedian(err_arr)
                if not np.isfinite(err_median) or err_median <= 0:
                    logger.debug(
                        "Error array has no valid positive median; skipping error-weighted centroiding."
                    )
                    use_error_weighted = False
                    err_arr = None
                else:
                    err_arr = np.nan_to_num(
                        err_arr, nan=err_median, posinf=err_median, neginf=err_median
                    )
                    err_floor = max(1e-30, float(err_median) * 1e-6)
                    np.clip(err_arr, err_floor, None, out=err_arr)
            else:
                err_arr = None

            if use_error_weighted:
                # Undersampled: 2DG with error weighting; well-sampled: COM with error propagation
                centroid_func = (
                    centroid_2dg_with_error if undersampled else centroid_com_with_error
                )
                logger.info(
                    "Using %s centroiding with uncertainty weighting",
                    "2D Gaussian" if undersampled else "center-of-mass",
                )
                try:
                    x_valid, y_valid, x_err_valid, y_err_valid = (
                        _centroid_sources_with_error_impl(
                            image,
                            old_x_valid,
                            old_y_valid,
                            box_size=boxsize,
                            mask=nan_mask,
                            error=err_arr,
                            centroid_func=centroid_func,
                        )
                    )
                    good = np.isfinite(x_valid) & np.isfinite(y_valid)
                    good_frac = good.sum() / max(1, len(x_valid))
                    # If error-weighted centroiding produces too few valid centroids
                    # (e.g. pathologically masked/noisy images), fall back to the
                    # more robust unweighted centroiding instead of discarding most
                    # sources.
                    if good_frac < 0.5:
                        logger.debug(
                            "Error array (for centroiding): min=%.2e, max=%.2e, median=%.2e",
                            np.nanmin(err_arr),
                            np.nanmax(err_arr),
                            np.nanmedian(err_arr),
                        )
                        logger.warning(
                            "Error-weighted centroiding produced only %.1f%% valid centroids "
                            "(%d/%d); falling back to unweighted centroiding.",
                            100.0 * good_frac,
                            good.sum(),
                            len(x_valid),
                        )
                        use_error_weighted = False
                except Exception as e:
                    logger.warning(
                        f"Error-weighted centroiding failed: {e}; falling back to unweighted"
                    )
                    use_error_weighted = False
            if not use_error_weighted:
                # Undersampled (FWHM < 3): 2D Gaussian; well-sampled: COM
                if undersampled:
                    logger.info(
                        "Using 2D Gaussian centroiding (FWHM < 3 px, undersampled)"
                    )
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
                    logger.warning(f"Centroiding failed even on valid sources: {e}")
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

    def _check_valid_cutouts(self, image, x_coords, y_coords, boxsize, mask):
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

            # Check if cutout has any unmasked pixels
            cutout_mask = mask[y_min:y_max, x_min:x_max]
            if np.all(cutout_mask):
                valid_mask[i] = False

        return valid_mask

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def find_source(self, ra, dec, catalog, tolerance=3.0):
        """
        Find sources in the catalog that match the given RA and DEC within a specified tolerance.

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
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        import numpy as np

        # Convert the RA, Dec of the input source to a SkyCoord object
        target_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

        # Prepare an empty list to store matching indices
        matching_indices = []

        # Loop over each entry in the catalog
        for i in range(len(catalog)):
            # Get the RA and DEC of the current catalog entry
            catalog_coord = SkyCoord(
                ra=catalog["RA"][i] * u.degree, dec=catalog["DEC"][i] * u.degree
            )

            # Calculate the angular separation between the target and the catalog entry
            separation = target_coord.separation(catalog_coord)

            # If the separation is within the tolerance (in arcseconds), store the index
            if separation.arcsecond <= tolerance:
                matching_indices.append(i)

        # Return the filtered catalog based on the matching indices
        return catalog[np.array(matching_indices)]

    # =========================================================================
    # METHOD: build_complete_catalog
    # =========================================================================
    def build_complete_catalog(
        self,
        target_coords,
        target_name=None,
        catalog_list=["refcat", "sdss", "pan_starrs", "apass", "2mass"],
        radius=2,
        max_seperation=3,
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
        max_seperation : float, optional
            Maximum separation in arcseconds for matching sources (default is 3).

        Returns:
        --------
        pd.DataFrame
            Combined catalog DataFrame.
        """
        catalog_list_str = ",".join([i.upper() for i in catalog_list])
        logger.info(
            border_msg(f"Building Custom catalog using from {catalog_list_str}")
        )

        # Set default target name if not provided
        if not target_name:
            target_name = self.input_yaml.get("target_name", "target")

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
        tolerance_arcsec = max_seperation

        cols = ["RA", "DEC"] + updated_filter_list
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

            # Loop over each entry in the current catalog
            for index, entry in catalog_i.iterrows():
                ra = entry["RA"]
                dec = entry["DEC"]

                # Check if the source exists in the output catalog
                existing_source = self.find_source(
                    ra, dec, output_catalog, tolerance=tolerance_arcsec
                )

                if existing_source.empty:
                    # Source does not exist, add a new row
                    new_row = {col: entry.get(col, np.nan) for col in cols}
                    new_row_df = pd.DataFrame([new_row])
                    output_catalog = pd.concat(
                        [output_catalog, new_row_df], ignore_index=True
                    )
                else:
                    # Source exists, update the row with missing filter data
                    index = existing_source.index[0]
                    for filter_name in updated_filter_list:
                        if filter_name in entry and pd.isna(
                            output_catalog.at[index, filter_name]
                        ):
                            output_catalog.at[index, filter_name] = entry[filter_name]

        # Final output catalog is ready
        output_catalog.to_csv(fpath, index=False, float_format="%.6f")
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
                border_msg(f"Checking saturation level using {len(catalog)} sources")
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

            flux = clean_catalog["flux_AP"].values
            flux_err = clean_catalog["flux_AP_err"].values
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
                    slope_constraint=1.0, slope_tolerance=0
                )
                ransac = RANSACRegressor(
                    estimator=base_estimator,
                    residual_threshold=0.2,
                    max_trials=500,
                    min_samples=0.33,
                )
                X = inst_mag_linear.reshape(-1, 1)
                y = catalog_mag_linear
                ransac.fit(X, y)
                slope = ransac.estimator_.slope_
                intercept = ransac.estimator_.intercept_
                inlier_mask = ransac.inlier_mask_

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
            _style = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "autophot.mplstyle"
            )
            if os.path.exists(_style):
                plt.style.use(_style)
            plt.ioff()
            fig, ax1 = plt.subplots(figsize=set_size(340, 1))
            ax1.errorbar(
                inst_mag,
                catalog_mag,
                yerr=catalog_mag_err,
                fmt="o",
                alpha=0.35,
                color="gray",
                markersize=2.2,
                capsize=0,
                elinewidth=0.4,
                linestyle="None",
                label=f"All Sources [{len(inst_mag)}]",
            )

            if fit_line:
                predicted = fit_line(inst_mag_linear)
                residuals = catalog_mag_linear - predicted
                outlier_mask = np.abs(residuals) > 0.25
                ax1.errorbar(
                    inst_mag_linear[outlier_mask],
                    catalog_mag_linear[outlier_mask],
                    yerr=catalog_mag_err_linear[outlier_mask],
                    xerr=inst_mag_err_linear[outlier_mask],
                    fmt="o",
                    color="#D55E00",
                    markersize=2.2,
                    capsize=0,
                    elinewidth=0.4,
                    linestyle="None",
                    label=f"Outliers [{np.sum(outlier_mask)}]",
                )
                ax1.errorbar(
                    inst_mag_linear[inlier_mask],
                    catalog_mag_linear[inlier_mask],
                    yerr=catalog_mag_err_linear[inlier_mask],
                    xerr=inst_mag_err_linear[inlier_mask],
                    fmt="o",
                    color="blue",
                    markersize=2.2,
                    capsize=0,
                    elinewidth=0.4,
                    linestyle="None",
                    label=f"Inliers [{np.sum(inlier_mask)}]",
                )
                x_range = np.linspace(inst_mag.min(), inst_mag.max(), 100)
                ax1.plot(
                    x_range,
                    fit_line(x_range),
                    color="black",
                    linestyle="--",
                    label=f"Fit: y = {slope:.3f}x + {intercept:.2f} +/- {intercept_error:.2f}",
                )

                # Add vertical lines showing linearity range in instrumental magnitude
                if len(inlier_flux) > 0:
                    min_inst_mag = -2.5 * np.log10(
                        max_flux
                    )  # Note: brighter objects have smaller magnitudes
                    max_inst_mag = -2.5 * np.log10(min_flux)
                    ax1.axvline(
                        x=min_inst_mag,
                        color="#009E73",
                        linestyle=":",
                        alpha=0.7,
                        label=f"Linearity range: {min_flux:.1f}-{max_flux:.1f} flux",
                    )
                    ax1.axvline(
                        x=max_inst_mag,
                        color="#009E73",
                        linestyle=":",
                        alpha=0.7,
                    )

            ax1.set_xlabel("Instrumental Magnitude [mag]")
            ax1.set_ylabel(f"Catalog {use_filter} Band [mag]")
            ax1.invert_yaxis()
            ax1.invert_xaxis()
            ax1.legend(frameon=False)
            ax1.grid(True, linestyle="--", alpha=0.5, zorder=0, lw=0.5)
            save_path = os.path.join(
                write_dir, f"Saturation_{base_name}.png"
            )
            fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close(fig)

            # Filter clean catalog by inliers
            if fit_line:
                clean_catalog = clean_catalog[inlier_mask]

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
            border_msg(
                f"Starting downsampling: {n_src} sources -> target {nmax} sources"
            )
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

        logger.info(border_msg(f"Checking radial profile of {len(catalog)} sources"))

        for i, (x, y) in enumerate(zip(catalog["x_pix"], catalog["y_pix"])):
            position = (float(x), float(y))
            try:
                cutout = Cutout2D(
                    image, position=position, size=scale, mode="partial", fill_value=0
                )
                cutout_data = np.nan_to_num(cutout.data, nan=0.0)
                total_flux = np.max(cutout_data)
                if total_flux > 0:
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
                ax_right.step(radii, profile, color="gray", alpha=0.3, linewidth=0.5)
            median_profile = np.median(accepted_profiles, axis=0)
            ax_right.step(
                radii,
                median_profile,
                color="#D55E00",
                linewidth=0.5,
                label="Median\nRadial\nProfile",
            )
            ax_right.set_xlabel("Radius [pixels]")
            ax_right.set_ylabel("Normalized Flux")
            ax_right.legend(loc="upper right", frameon=False)
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
            border_msg(
                f"Measuring {len(selectedCatalog)} Sources in the Field using Aperture Photometry"
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
        selectedCatalog["SNR"] = np.round(sourceSNR, 1)

        # Log the number of sources with valid instrumental magnitude (finite flux > 0)
        logger.info(
            "Instrumental magnitude of %d sources measured"
            % sum(~np.isnan(selectedCatalog[inst_col]))
        )

        return selectedCatalog
