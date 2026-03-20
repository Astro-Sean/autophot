#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aperture photometry and aperture correction.

This module measures fluxes using circular apertures, estimates an
optimum-radius (curve-of-growth style) when requested, and computes
aperture-correction factors used to calibrate the photometry.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import logging
import warnings

# Safeguard: force BLAS/OpenMP to 1 thread before importing numpy (avoids exhausting
# process/thread limits when using multiprocessing on HPC; OpenBLAS often defaults to 128).
for _env in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env] = "1"

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import mstats

from astropy.stats import (
    biweight_midvariance,
    sigma_clipped_stats,
    mad_std,
    biweight_location,
    sigma_clip,
)
from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus,
)
from photutils.utils import calc_total_error
from photutils.profiles import CurveOfGrowth
from astropy.visualization import ImageNormalize, ZScaleInterval

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from functions import mag, snr_err, set_size, border_msg, log_exception

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NSOURCES = 10  # minimum source count to justify spawning worker processes
MAX_WORKERS_DEFAULT = (
    16  # cap on default n_jobs to avoid exhausting HPC process/thread limits
)


def _resolve_n_jobs(n_jobs, half_cpus=False):
    """Resolve and cap worker count for multiprocessing (safeguards HPC process/thread limits)."""
    from multiprocessing import cpu_count

    n = cpu_count()
    default = (n // 2) if half_cpus else n
    raw = (
        min(MAX_WORKERS_DEFAULT, max(1, default))
        if n_jobs is None
        else max(1, int(n_jobs))
    )
    return min(MAX_WORKERS_DEFAULT, raw)


# ===========================================================================
# Module-level worker functions  (MUST be at module scope for pickle)
# ===========================================================================


def _measure_worker(args):
    """
    Perform aperture photometry for a single source.

    Parameters
    ----------
    args : tuple
        (i, aperture_masks, annulus_masks, image_e, error,
         read_noise_sq, inv_exposure_time, area, phot, gain, verbose)

    Returns
    -------
    dict  - result keyed by 'idx'; contains 'fail_reason' on failure.
    """
    (
        i,
        aperture_masks,
        annulus_masks,
        image_e,
        error,
        read_noise_sq,
        inv_exposure_time,
        area,
        phot,
        gain,
        verbose,
    ) = args

    try:
        ap_mask = aperture_masks[i]
        an_mask = annulus_masks[i]

        ap_pix = ap_mask.get_values(image_e)
        bkg_pix = an_mask.get_values(image_e)

        # Optional per-pixel uncertainty (e.g. from Background2D / calc_total_error).
        ap_err_pix = None
        if error is not None:
            ap_err_pix = ap_mask.get_values(error)

        # Remove NaNs and exact zeros (flagged bad pixels).
        ap_pix = ap_pix[np.isfinite(ap_pix) & (ap_pix != 0.0)]
        bkg_pix = bkg_pix[np.isfinite(bkg_pix) & (bkg_pix != 0.0)]

        if ap_pix.size == 0 or bkg_pix.size == 0:
            return {"idx": i, "fail_reason": "empty_pixels"}

        # Robust background via MAD (handles negatives cleanly).
        bkg_value = np.median(bkg_pix)
        empirical_std = 1.4826 * np.median(np.abs(bkg_pix - bkg_value))

        if empirical_std <= 0 or not np.isfinite(empirical_std):
            return {"idx": i, "fail_reason": "bkg_invalid"}

        row = phot.iloc[i]
        raw_aperture_sum = row.aperture_sum
        if not np.isfinite(raw_aperture_sum):
            return {"idx": i, "fail_reason": "aperture_sum_invalid"}

        # Use effective aperture area from mask when available (subpixel consistency).
        try:
            effective_area = float(np.sum(ap_mask.data))
            if effective_area <= 0 or not np.isfinite(effective_area):
                effective_area = area
        except Exception:
            effective_area = area
        aperture_bkg = bkg_value * effective_area
        aperture_sum = raw_aperture_sum - aperture_bkg

        raw_max = np.max(ap_pix)
        max_val = raw_max - bkg_value

        # Variance model: prefer fully propagated per-pixel uncertainties
        # when available; otherwise fall back to a simple Poisson+sky+read-noise
        # approximation based on the empirical background standard deviation.
        sqrt_var = np.nan
        if ap_err_pix is not None:
            # error array is in electrons; propagate by summing variances.
            var_from_error = np.nansum(ap_err_pix.astype(float) ** 2)
            if var_from_error > 0 and np.isfinite(var_from_error):
                sqrt_var = np.sqrt(var_from_error)

        if not np.isfinite(sqrt_var):
            # Fallback variance: |source| + area * (sigma_sky^2 + RN^2)
            total_var = np.abs(aperture_sum) + effective_area * (
                empirical_std**2 + read_noise_sq
            )
            if total_var > 0 and np.isfinite(total_var):
                sqrt_var = np.sqrt(total_var)

        if np.isfinite(sqrt_var) and sqrt_var > 0:
            snr = aperture_sum / sqrt_var
        else:
            sqrt_var = np.nan
            snr = 0.0

        # Flux in e/s (counts per second) for consistency with PSF; mag = -2.5*log10(flux).
        # This is aperture-only flux (light within the aperture); for total flux use aperture correction.
        flux_ap = aperture_sum * inv_exposure_time

        try:
            mag_val = mag(flux_ap)
        except Exception:
            mag_val = np.nan

        try:
            mag_err_val = snr_err(snr)
        except Exception:
            mag_err_val = np.nan

        rn_term = empirical_std**2 + read_noise_sq
        max_flux_err = np.sqrt(np.abs(raw_max) + rn_term) * inv_exposure_time

        return {
            "idx": i,
            "maxPixel": max_val * inv_exposure_time,
            "maxPixel_err": max_flux_err,
            "area": effective_area,
            "counts_AP": aperture_sum,
            "flux_AP": flux_ap,
            "flux_AP_err": (
                sqrt_var * inv_exposure_time if np.isfinite(sqrt_var) else np.nan
            ),
            "sky_bkg_total": aperture_bkg,
            "sky_bkg_total_flux": aperture_bkg * inv_exposure_time,
            "noiseSky": empirical_std * inv_exposure_time,
            "threshold": max_val / empirical_std,
            "SNR": snr,
            "bkg_std_method": "MAD",
            "mag": mag_val,
            "mag_err": mag_err_val,
        }

    except Exception as exc:
        if verbose >= 2:
            logging.getLogger(__name__).error(
                f"Error processing source {i}: {exc}", exc_info=True
            )
        return {"idx": i, "fail_reason": str(exc)}


def _cog_profile_worker(args):
    """
    Compute a normalised Curve-of-Growth profile for one source.

    Parameters
    ----------
    args : tuple
        (idx, x_pix, y_pix, fwhm, radii, image, error)

    Returns
    -------
    dict or None
    """
    idx, x_pix, y_pix, fwhm, radii, image, error = args
    try:
        xycen = np.array([x_pix, y_pix])
        cog = CurveOfGrowth(
            image, xycen, radii, error=error, mask=None, method="subpixel"
        )
        cog.normalize()
        return {"idx": idx, "radii": cog.radii, "profile": cog.profile}
    except Exception:
        return None


def _optimum_radius_worker(args):
    """
    Compute the per-source optimum aperture radius at the target encircled
    energy fraction (norm_factor), with an SNR guard.

    Parameters
    ----------
    args : tuple
        (idx, x_pix, y_pix, fwhm, radii, image, error,
         norm_factor, stability_threshold)

    Returns
    -------
    dict or None
    """
    idx, x_pix, y_pix, fwhm, radii, image, error, norm_factor, stability_threshold = (
        args
    )

    try:
        xycen = np.array([x_pix, y_pix])
        cog = CurveOfGrowth(
            image, xycen, radii, error=error, mask=None, method="subpixel"
        )
        cog.normalize()

        norm_profile = cog.profile
        norm_profile_err = cog.profile_error

        r_at_norm = cog.calc_radius_at_ee(norm_factor)
        if not np.isfinite(r_at_norm):
            return None

        opt_r_fwhm = float(r_at_norm / fwhm)

        # SNR guard at the chosen radius.
        r_pix = opt_r_fwhm * fwhm
        idx_r = int(np.argmin(np.abs(cog.radii - r_pix)))
        enc_f = norm_profile[idx_r]
        enc_err = norm_profile_err[idx_r] if norm_profile_err is not None else None
        if enc_err is not None and enc_err > 0 and (enc_f / enc_err) < 3:
            return None

        # Mean slope inside the per-source optimum radius (monotonicity proxy).
        within_opt = (cog.radii / fwhm) <= opt_r_fwhm
        if np.count_nonzero(within_opt) > 1:
            mean_slope = float(
                np.nanmean(np.gradient(norm_profile[within_opt], cog.radii[within_opt]))
            )
        else:
            mean_slope = float("nan")

        # Local tail deviation and excess above 1 (neighbour contamination).
        beyond_local = (cog.radii / fwhm) > opt_r_fwhm
        tail_max_dist = (
            float(np.nanmax(np.abs(norm_profile[beyond_local] - 1.0)))
            if np.any(beyond_local)
            else float("nan")
        )
        # Excess above 1 at large radius: CoG increasing there indicates a neighbouring source.
        tail_excess = (
            float(max(0.0, np.nanmax(norm_profile[beyond_local]) - 1.0))
            if np.any(beyond_local)
            else 0.0
        )

        # Surrounding environment: robust scatter in an annulus just outside the star.
        # Prefer low local std (clean background, no bright neighbour or gradient).
        local_env_std = float("nan")
        try:
            inner_r = max(r_at_norm * 1.2, 2.0 * fwhm)
            outer_r = 4.0 * fwhm
            if outer_r > inner_r + fwhm * 0.5:
                annulus = CircularAnnulus(xycen, inner_r, outer_r)
                amask = annulus.to_mask(method="center")
                apix = amask.get_values(image)
                apix = apix[np.isfinite(apix)]
                if len(apix) >= 10:
                    local_env_std = float(mad_std(apix))
        except Exception:
            pass

        return {
            "idx": idx,
            "optimum_radius": opt_r_fwhm,
            "mean_slope": mean_slope,
            "tail_max_dist_local": tail_max_dist,
            "tail_excess": tail_excess,
            "local_env_std": local_env_std,
            "profile": norm_profile,
        }

    except Exception:
        return None


# ===========================================================================
# Aperture class
# ===========================================================================


class Aperture:
    """
    Circular aperture photometry with background annulus subtraction.

    Provides:
    * measure()               - per-source flux / magnitude measurements
    * measure_optimum_radius() - data-driven aperture radius selection
    * compute_aperture_correction() - CoG-based aperture correction
    """

    def __init__(self, input_yaml: dict, image: np.ndarray, verbose: int = 1):
        """
        Parameters
        ----------
        input_yaml : dict   pipeline configuration
        image      : ndarray  2-D science image
        verbose    : int      0 = quiet, 1 = normal, 2 = debug
        """
        self.input_yaml = input_yaml
        self.image = image
        self.verbose = verbose

    # -----------------------------------------------------------------------
    # Background statistics helpers
    # -----------------------------------------------------------------------

    def optimal_background_std_estimation(
        self, bkg_pixels: np.ndarray, verbose: bool = False
    ):
        """
        Estimate the background standard deviation with a cascade of robust
        estimators (best -> worst).

        Parameters
        ----------
        bkg_pixels : 1-D ndarray
        verbose    : bool

        Returns
        -------
        (std_bkg, method_name) : (float, str)
        """
        logger = logging.getLogger(__name__)

        def _percentile_std(data):
            p16, p84 = np.percentile(data, [16, 84])
            return 0.5 * (p84 - p16)

        def _winsorized_std(data):
            return float(np.sqrt(mstats.winsorize(data, limits=(0.05, 0.05)).var()))

        methods = [
            ("MAD", lambda d: mad_std(d, ignore_nan=True)),
            ("Biweight", lambda d: np.sqrt(biweight_midvariance(d, c=6.0))),
            ("Percentile", _percentile_std),
            ("Winsorized", _winsorized_std),
            ("SigmaClip", lambda d: sigma_clipped_stats(d, sigma=3, maxiters=5)[2]),
        ]

        for name, fn in methods:
            try:
                std_value = fn(bkg_pixels)
                if std_value > 0 and np.isfinite(std_value):
                    if verbose:
                        logger.debug(
                            f"Background std [{name}]: sigma = {std_value:.3f}"
                        )
                    return std_value, name
            except Exception as exc:
                if verbose:
                    logger.debug(f"{name} failed: {exc}")

        return np.nan, "None"

    def enhanced_background_estimation(
        self, bkg_pixels: np.ndarray, image_gain: float = 1.0, verbose: bool = False
    ):
        """
        Estimate background level and standard deviation with Poisson sanity check.

        Returns
        -------
        (bkg_level, std, method_name) or (nan, nan, 'InsufficientData')
        """
        logger = logging.getLogger(__name__)
        finite = np.isfinite(bkg_pixels)
        if finite.sum() < 10:
            return np.nan, np.nan, "InsufficientData"

        clean = bkg_pixels[finite]
        bkg_lvl = biweight_location(clean, c=6.0)

        # Biweight midvariance with Poisson sanity check.
        try:
            bw_var = biweight_midvariance(clean, c=6.0, M=bkg_lvl)
            if bw_var > 0 and np.isfinite(bw_var):
                emp_std = np.sqrt(bw_var)
                if bkg_lvl > 0:
                    poisson_std = np.sqrt(bkg_lvl / image_gain)
                    ratio = emp_std / poisson_std
                    if 0.5 < ratio < 5.0:
                        return bkg_lvl, emp_std, "Biweight"
                    if verbose:
                        logger.debug(f"Biweight ratio outside range: {ratio:.2f}")
        except Exception:
            pass

        # MAD fallback.
        try:
            ms = mad_std(clean)
            if ms > 0 and np.isfinite(ms):
                return bkg_lvl, ms, "MAD"
        except Exception:
            pass

        # Percentile fallback.
        try:
            p16, p84 = np.percentile(clean, [16, 84])
            ps = 0.5 * (p84 - p16)
            if ps > 0:
                return bkg_lvl, ps, "Percentile"
        except Exception:
            pass

        return bkg_lvl, np.nan, "Failed"

    # -----------------------------------------------------------------------
    # Aperture photometry
    # -----------------------------------------------------------------------

    def measure(
        self,
        sources: pd.DataFrame,
        ap_size: float = None,
        exposure_time: float = None,
        read_noise: float = None,
        gain: float = None,
        plot: bool = False,
        background_rms: np.ndarray = None,
        saveTarget: bool = False,
        verbose: int = 1,
        n_jobs: int = None,
    ) -> pd.DataFrame:
        """
        Measure aperture photometry for all sources in *sources*.

        Parameters
        ----------
        sources        : DataFrame with 'x_pix', 'y_pix' columns
        ap_size        : aperture radius (pixels); read from config if None
        exposure_time  : seconds; read from config if None
        read_noise     : electrons; read from config if None
        gain           : e-/ADU; read from config if None
        plot           : save per-source diagnostic PDF
        background_rms : 2-D RMS map for error model
        saveTarget     : use filename stem (not index) when naming plot files
        verbose        : 0 quiet, 1 normal, 2 debug
        n_jobs         : worker processes; None -> cpu_count() // 2

        Returns
        -------
        sources : DataFrame (in-place columns added / updated)
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        pd.options.mode.chained_assignment = None

        logger = logging.getLogger(__name__)

        # ---- Configuration -------------------------------------------------
        fwhm = self.input_yaml["fwhm"]
        gain = gain or self.input_yaml.get("gain", 1.0)
        exposure_time = exposure_time or self.input_yaml.get("exposure_time", 30.0)
        read_noise = read_noise or self.input_yaml.get("readnoise", 0.0)
        ap_size = ap_size or self.input_yaml["photometry"]["aperture_radius"]

        crowded = self.input_yaml.get("photometry", {}).get("crowded_field", False)
        if crowded:
            # Tighter annulus for crowded fields: smaller gap and width to avoid neighboring sources.
            annulusIN = float(np.ceil(ap_size + 0.35 * fwhm))
            annulusOUT = float(np.ceil(annulusIN + 0.65 * fwhm))
        else:
            annulusIN = float(np.ceil(ap_size + 0.5 * fwhm))
            annulusOUT = float(np.ceil(annulusIN + 1.0 * fwhm))
        area = np.pi * ap_size**2

        image_e = self.image * gain
        image_e[image_e == 0.0] = np.nan

        inv_exp_time = 1.0 / exposure_time
        read_noise_sq = read_noise**2

        filt = self.input_yaml["imageFilter"]
        mag_col = f"inst_{filt}_AP"
        err_col = f"inst_{filt}_AP_err"

        # ---- Ensure output columns exist -----------------------------------
        float_cols = [
            "maxPixel",
            "maxPixel_err",
            "counts_AP",
            "flux_AP",
            "flux_AP_err",
            "sky_bkg_total",
            "sky_bkg_total_flux",
            "noiseSky",
            "SNR",
            "SNR_err",
            mag_col,
            err_col,
            "threshold",
            "area",
        ]
        str_cols = ["bkg_std_method", "fail_reason"]
        for col in float_cols:
            if col not in sources.columns:
                sources[col] = np.nan
        for col in str_cols:
            if col not in sources.columns:
                sources[col] = ""

        # ---- Error model ---------------------------------------------------
        if background_rms is not None:
            # Poisson term must be non-negative. Slightly negative backgrounds
            # (common after background subtraction / difference imaging) can
            # produce NaNs in calc_total_error and degrade injection/recovery.
            image_e_pois = np.where(
                np.isfinite(image_e), np.maximum(image_e, 0.0), np.nan
            )
            error = calc_total_error(
                image_e_pois, background_rms * gain, effective_gain=1
            )
        else:
            error = None

        # ---- Validate source positions -------------------------------------
        x, y = sources["x_pix"].values, sources["y_pix"].values
        valid_mask = (
            (x >= 0) & (x < self.image.shape[1]) & (y >= 0) & (y < self.image.shape[0])
        )
        sources = sources[valid_mask].reset_index(drop=True)
        if sources.empty:
            if verbose:
                logger.warning("No valid sources within image bounds.")
            return sources

        positions = list(zip(sources["x_pix"], sources["y_pix"]))

        # ---- Aperture objects & photutils photometry -----------------------
        apertures_obj = CircularAperture(positions, r=ap_size)
        annuli_obj = CircularAnnulus(positions, r_in=annulusIN, r_out=annulusOUT)

        phot = aperture_photometry(image_e, apertures_obj, error=error).to_pandas()
        aperture_masks = [
            ap.to_mask(method="exact", subpixels=15) for ap in apertures_obj
        ]
        annulus_masks = [an.to_mask(method="exact", subpixels=15) for an in annuli_obj]

        n_jobs = _resolve_n_jobs(n_jobs, half_cpus=True)

        # ---- Build argument list -------------------------------------------
        args_list = [
            (
                i,
                aperture_masks,
                annulus_masks,
                image_e,
                error,
                read_noise_sq,
                inv_exp_time,
                area,
                phot,
                gain,
                verbose,
            )
            for i in range(len(sources))
        ]

        # ---- Dispatch (parallel for large catalogs) ------------------------
        if len(sources) >= NSOURCES:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_measure_worker, args_list)
        else:
            results = [_measure_worker(a) for a in args_list]

        # ---- Batch update DataFrame (one dict -> update is much faster) -----
        # Collect all successful results into column-keyed lists, then assign.
        updates: dict[str, list] = {c: [np.nan] * len(sources) for c in float_cols}
        for c in str_cols:
            updates[c] = [""] * len(sources)

        fail_count = 0
        for res in results:
            i = res.pop("idx")
            if "fail_reason" in res:
                updates["fail_reason"][i] = res["fail_reason"]
                fail_count += 1
                continue

            updates["maxPixel"][i] = res["maxPixel"]
            updates["maxPixel_err"][i] = res["maxPixel_err"]
            updates["area"][i] = res["area"]
            updates["counts_AP"][i] = res["counts_AP"]
            updates["flux_AP"][i] = res["flux_AP"]
            updates["flux_AP_err"][i] = res["flux_AP_err"]
            updates["sky_bkg_total"][i] = res["sky_bkg_total"]
            updates["sky_bkg_total_flux"][i] = res["sky_bkg_total_flux"]
            updates["noiseSky"][i] = res["noiseSky"]
            updates["threshold"][i] = res["threshold"]
            updates["SNR"][i] = res["SNR"]
            updates["bkg_std_method"][i] = res["bkg_std_method"]
            updates[mag_col][i] = res["mag"]
            updates[err_col][i] = res["mag_err"]

        # Single DataFrame assignment per column - avoids .at[] overhead.
        for col, vals in updates.items():
            sources[col] = vals

        if verbose >= 1 and fail_count > 0:
            logger.warning(
                f"{fail_count}/{len(results)} sources failed. "
                "Check 'fail_reason' column."
            )

        # ---- Per-source diagnostic plots -----------------------------------
        if plot:
            for i in range(len(sources)):
                if sources.at[i, "fail_reason"]:
                    continue
                try:
                    self._generate_plot(
                        image=image_e,
                        cutout_center=(
                            float(sources.at[i, "x_pix"]),
                            float(sources.at[i, "y_pix"]),
                        ),
                        ap_size=ap_size,
                        annulusIN=annulusIN,
                        annulusOUT=annulusOUT,
                        fwhm=fwhm,
                        saveTarget=saveTarget,
                        index=i,
                        error=error,
                    )
                except Exception as exc:
                    logger.exception(f"Plot failed for source {i}: {exc}")

        return sources

    # -----------------------------------------------------------------------
    # Diagnostic plot
    # -----------------------------------------------------------------------

    def _generate_plot(
        self,
        image,
        cutout_center,
        ap_size,
        annulusIN,
        annulusOUT,
        fwhm,
        saveTarget,
        index,
        error=None,
    ):
        """
        Three-panel diagnostic: main image + right / bottom flux profiles.
        """
        plt.ioff()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        style = os.path.join(dir_path, "autophot.mplstyle")
        if os.path.exists(style):
            plt.style.use(style)

        fpath = self.input_yaml["fpath"]
        write_dir = os.path.dirname(fpath)
        base = os.path.splitext(os.path.basename(fpath))[0]

        fig = plt.figure(figsize=set_size(340, 1))
        grid = GridSpec(
            2,
            2,
            width_ratios=[1, 0.25],
            height_ratios=[1, 0.25],
            wspace=0.01,
            hspace=0.01,
        )

        ax_main = fig.add_subplot(grid[0, 0])
        ax_right = fig.add_subplot(grid[0, 1], sharey=ax_main)
        ax_bottom = fig.add_subplot(grid[1, 0], sharex=ax_main)

        ax_right.yaxis.tick_right()
        ax_right.xaxis.tick_top()
        ax_main.xaxis.tick_top()

        cx, cy = cutout_center
        aperture_scale = int(annulusOUT + fwhm)
        zoom_size = 1.25 * aperture_scale
        ax_main.set_xlim(cx - zoom_size, cx + zoom_size)
        ax_main.set_ylim(cy - zoom_size, cy + zoom_size)

        x_min = int(np.floor(ax_main.get_xlim()[0]))
        x_max = int(np.ceil(ax_main.get_xlim()[1]))
        y_min = int(np.floor(ax_main.get_ylim()[0]))
        y_max = int(np.ceil(ax_main.get_ylim()[1]))

        zoom_image = image[y_min:y_max, x_min:x_max]
        zoom_error = (
            error[y_min:y_max, x_min:x_max]
            if error is not None
            else np.zeros_like(zoom_image)
        )

        norm = ImageNormalize(zoom_image, interval=ZScaleInterval())
        ax_main.imshow(image, origin="lower", norm=norm, cmap="viridis", aspect="auto")

        for radius, color, style in [
            (ap_size, "lime", "-"),
            (annulusIN, "red", "--"),
            (annulusOUT, "red", "--"),
        ]:
            ax_main.add_patch(
                Circle((cx, cy), radius, ec=color, fc="none", lw=0.5, ls=style)
            )

        kw = dict(ls=":", color="white", lw=0.5, alpha=0.7)
        ax_main.axvline(cx, **kw)
        ax_main.axhline(cy, **kw)
        ax_bottom.axvline(cx, **kw)
        ax_right.axhline(cy, **kw)

        hx = zoom_image.mean(axis=0)
        hy = zoom_image.mean(axis=1)
        # SE of mean = sqrt(sum(sigma^2))/N (not mean(sigma))
        n_rows, n_cols = zoom_error.shape[0], zoom_error.shape[1]
        hx_err = np.sqrt(np.nansum(zoom_error**2, axis=0)) / max(n_rows, 1)
        hy_err = np.sqrt(np.nansum(zoom_error**2, axis=1)) / max(n_cols, 1)
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)

        kw_step = dict(color="dodgerblue", where="mid", lw=0.5)
        ax_bottom.step(x_range, hx, **kw_step)
        ax_bottom.fill_between(
            x_range, hx - hx_err, hx + hx_err, color="dodgerblue", alpha=0.3, step="mid"
        )

        ax_right.step(hy, y_range, **kw_step)
        ax_right.fill_betweenx(
            y_range, hy - hy_err, hy + hy_err, color="dodgerblue", alpha=0.3, step="mid"
        )

        ax_bottom.set_xlabel("X position (pixels)")
        ax_bottom.set_ylabel(r"Counts [$e^-$]")
        ax_right.set_xlabel(r"Counts [$e^-$]")
        ax_right.yaxis.set_label_position("right")
        ax_right.yaxis.tick_right()
        ax_bottom.set_xlim(ax_main.get_xlim())
        ax_right.set_ylim(ax_main.get_ylim())

        label = base if saveTarget else index
        save_name = os.path.join(write_dir, f"aperture_{label}.pdf")
        fig.savefig(save_name, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def interpolate_nans(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate (and extrapolate) NaN values in *y*.

        Parameters
        ----------
        x, y : array-like

        Returns
        -------
        y_interp : ndarray
        """
        x = np.asarray(x)
        y = np.asarray(y, dtype=float)

        nan_mask = np.isnan(y)
        # BUG FIX: original used `len(nan_indices) == 0` which checks the
        # *length* of the boolean array (always == len(y)), never the count
        # of True entries.  Fixed to `.any()`.
        if not nan_mask.any():
            return y

        interp = interp1d(
            x[~nan_mask], y[~nan_mask], kind="linear", fill_value="extrapolate"
        )
        y_out = y.copy()
        y_out[nan_mask] = interp(x[nan_mask])
        return y_out

    # -----------------------------------------------------------------------
    # Optimum aperture radius
    # -----------------------------------------------------------------------

    def measure_optimum_radius(
        self,
        sources: pd.DataFrame,
        plot: bool = True,
        norm_factor: float = 0.8,
        aperture_norm_factor: float = 0.95,
        sigma: float = 5.0,
        background_rms: np.ndarray = None,
        max_radius: float = 3.5,
        stability_threshold: float = 0.2,
        max_tail_excess: float = 0.8,
        min_tail_flux: float = 0.8,
        fwhm_uncertainty: float = 0.5,
        n_jobs: int = None,
        crowded: bool = False,
    ):
        """
        Data-driven selection of the optimal aperture radius.

        Selection logic
        ---------------
        1. Per-source optimum radii via CoG + SNR guard.
        2. Preliminary global radius = sigma-clipped median of per-source radii.
        3. Global stability: reject sources with |profile - 1| > threshold or
           profile > 1 + max_tail_excess beyond the preliminary global radius.
        4. After final optimum_radius is set: exclude sources with profile
           < min_tail_flux or > 1 + max_tail_excess beyond the final optimum radius.
           CoG is normalized 0->1 at aperture radius = max_radius (default 3.5 FWHM).
        5. Final global radius = sigma-clipped median of stable sources.
        6. Tail check w.r.t. final optimum (min_tail_flux / max_tail_excess).
        7. Optional EE model refinement at *aperture_norm_factor*.
        8. optimum_scale = ceil(optimum_radius * fwhm) + 0.5

        Returns
        -------
        (filtered_sources, optimum_radius, optimum_scale)
        """
        logger = logging.getLogger(__name__)

        # Crowded fields: always use a robust fixed aperture radius of
        # ~1.5 FWHM to avoid failures or unstable behaviour in very dense
        # regions.  This radius is in FWHM units and can be overridden by
        # `photometry.crowded_optimum_radius_fwhm` in the config.
        if crowded:
            phot_cfg = self.input_yaml.get("photometry", {})
            fixed_radius = float(phot_cfg.get("crowded_optimum_radius_fwhm", 1.5))
            fwhm = float(self.input_yaml["fwhm"])
            optimum_radius = fixed_radius
            optimum_scale = max(7, int(np.ceil(fixed_radius * fwhm))) + 0.5
            logger.info(
                "Crowded field: skipping data-driven optimum-radius search; "
                "using fixed aperture radius of %.2f FWHM (%.2f pixels).",
                optimum_radius,
                optimum_radius * fwhm,
            )
            return sources, optimum_radius, optimum_scale

        # Sparse/normal fields: use data-driven optimisation with a slightly
        # larger default fallback radius.
        fallback_radius = 1.7
        optimum_radius = fallback_radius
        optimum_scale = (
            max(7, int(np.ceil(fallback_radius * self.input_yaml["fwhm"]))) + 0.5
        )

        # Relaxed stability/tail criteria for crowded fields (more neighbour contamination)
        if crowded:
            stability_threshold = max(stability_threshold, 0.35)
            max_tail_excess = max(max_tail_excess, 1.0)
            min_tail_flux = min(min_tail_flux, 0.6)

        # ---- SNR pre-filter (slightly relaxed for crowded) ------------------------------------------------
        snr_min = 3.0 if crowded else 5.0
        sources = sources[(sources["SNR"] > snr_min) & (sources["SNR"] < 10000)].copy()
        sources.reset_index(inplace=True)
        n_sources = len(sources)

        if n_sources == 0:
            logger.warning("No sources passed SNR cut. Using default radius/scale.")
            return sources, optimum_radius, optimum_scale

        fwhm = self.input_yaml["fwhm"]
        radii_fwhm = np.arange(0.05, max_radius + 1e-9, 0.1)
        radii = radii_fwhm * fwhm
        logger.info(border_msg(f"Finding optimum aperture using {n_sources} sources"))

        sources["optimum_radius"] = np.nan
        sources["mean_slope"] = np.nan
        sources["tail_max_dist_local"] = np.nan
        sources["tail_max_dist_global"] = np.nan
        sources["tail_excess_global"] = np.nan
        sources["stable_beyond_global"] = False
        sources["local_env_std"] = np.nan

        gain = self.input_yaml.get("gain", 1.0)
        error = (
            None
            if background_rms is None
            else calc_total_error(
                self.image, background_rms, effective_gain=float(gain)
            )
        )
        n_jobs = _resolve_n_jobs(n_jobs, half_cpus=False)

        # ---- Parallel CoG analysis -----------------------------------------
        args_list = [
            (
                idx,
                row["x_pix"],
                row["y_pix"],
                fwhm,
                radii,
                self.image,
                error,
                norm_factor,
                stability_threshold,
            )
            for idx, row in sources.iterrows()
        ]

        if n_sources < NSOURCES:
            results = [_optimum_radius_worker(a) for a in args_list]
        else:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_optimum_radius_worker, args_list)

        # ---- Collect results -----------------------------------------------
        # KEY OPTIMISATION: profiles are already in `results`; no second Pool
        # needed for the plot pass.  Store them here for reuse.
        profiles_map: dict[int, np.ndarray] = {}
        for res in results:
            if res is None:
                continue
            idx = res["idx"]
            sources.at[idx, "optimum_radius"] = res["optimum_radius"]
            sources.at[idx, "mean_slope"] = res["mean_slope"]
            sources.at[idx, "tail_max_dist_local"] = res["tail_max_dist_local"]
            sources.at[idx, "local_env_std"] = res.get("local_env_std", np.nan)
            profiles_map[idx] = res["profile"]

        # ---- Preliminary global radius (no stability cut yet) --------------
        prelim_mask = np.isfinite(sources["optimum_radius"].values) & np.isfinite(
            sources["mean_slope"].values
        )
        if not prelim_mask.any():
            logger.warning("No valid sources for global radius. Using default.")
            return sources.iloc[[]], optimum_radius, optimum_scale

        prelim_r = sources.loc[prelim_mask, "optimum_radius"].to_numpy(float)
        prelim_s = sources.loc[prelim_mask, "mean_slope"].to_numpy(float)

        try:
            cr = sigma_clip(prelim_r, sigma=sigma, stdfunc=mad_std)
            cs = sigma_clip(prelim_s, sigma=sigma, stdfunc=mad_std)
            prelim_keep = ~np.asanyarray(cr.mask) & ~np.asanyarray(cs.mask)
        except Exception:
            prelim_keep = np.ones(len(prelim_r), dtype=bool)

        if not prelim_keep.any():
            logger.warning(
                "Preliminary sigma-clip rejected all sources. Using default."
            )
            return sources.iloc[[]], fallback_radius, optimum_scale

        global_optimum_pre = float(np.nanmedian(prelim_r[prelim_keep]))

        # ---- Global stability filter (tail w.r.t. preliminary radius) -------
        beyond_global = (radii / fwhm) > global_optimum_pre
        for idx in range(len(sources)):
            prof = profiles_map.get(idx)
            if prof is None or not beyond_global.any():
                continue
            tail_region = prof[beyond_global]
            tmd = float(np.nanmax(np.abs(tail_region - 1.0)))
            tail_excess_global = float(max(0.0, np.nanmax(tail_region) - 1.0))
            sources.at[idx, "tail_max_dist_global"] = tmd
            sources.at[idx, "tail_excess_global"] = tail_excess_global
            if (
                np.isfinite(tmd)
                and tmd < stability_threshold
                and np.isfinite(tail_excess_global)
                and tail_excess_global < max_tail_excess
                and np.isfinite(sources.at[idx, "mean_slope"])
            ):
                sources.at[idx, "stable_beyond_global"] = True

        stable_mask = sources["stable_beyond_global"].values & np.isfinite(
            sources["optimum_radius"].values
        )
        if not stable_mask.any():
            # Crowded fallback: use median of all preliminary radii (clipped) instead of failing
            if crowded and prelim_mask.any():
                r_clip = np.clip(prelim_r, 0.5, 2.0)
                fallback_from_data = float(np.nanmedian(r_clip))
                if np.isfinite(fallback_from_data):
                    logger.warning(
                        "No globally stable sources; using median of preliminary radii (crowded): %.2f FWHM",
                        fallback_from_data,
                    )
                    optimum_scale = (
                        max(
                            12,
                            int(np.ceil(fallback_from_data * self.input_yaml["fwhm"])),
                        )
                        + 0.5
                    )
                    return sources.iloc[[]], fallback_from_data, optimum_scale
            logger.warning("No globally stable sources. Using default.")
            return sources.iloc[[]], fallback_radius, optimum_scale

        # Use all globally stable sources (as many as possible) with a mild
        # sanity cut on radius; no additional behaviour-score culling.
        opt_r_arr = sources.loc[stable_mask, "optimum_radius"].to_numpy(float)
        slopes_arr = sources.loc[stable_mask, "mean_slope"].to_numpy(float)
        kept_indices = np.where(stable_mask)[0]
        profiles = np.array(
            [profiles_map[i] for i in kept_indices if i in profiles_map], dtype=float
        )

        # Keep only reasonable radii but otherwise retain the full stable set.
        min_r_ok = 0.5
        max_r_ok = max_radius
        if len(opt_r_arr) > 0:
            radius_ok = (opt_r_arr >= min_r_ok) & (opt_r_arr <= max_r_ok)
            if radius_ok.any():
                opt_r_arr = opt_r_arr[radius_ok]
                slopes_arr = slopes_arr[radius_ok]
                kept_indices = kept_indices[radius_ok]
                if profiles.size > 0:
                    profiles = profiles[radius_ok]

        logger.info(
            f"Using {len(kept_indices)} sources with stable profiles "
            f"for optimum-radius and PSF selection."
        )

        # ---- Final sigma-clipping ------------------------------------------
        # Gentle clip: only drop extreme radius outliers, keep almost all stable profiles.
        sigma_radius = max(sigma, 10.0)
        try:
            cr_f = sigma_clip(opt_r_arr, sigma=sigma_radius, stdfunc=mad_std)
            final_mask = ~np.asanyarray(cr_f.mask)
        except Exception:
            final_mask = np.ones(len(opt_r_arr), dtype=bool)

        if final_mask.any():
            final_indices = kept_indices[final_mask]
            filtered_sources = sources.iloc[final_indices].copy()
            optimum_radius = float(np.nanmedian(opt_r_arr[final_mask]))
        else:
            logger.warning(
                "Final sigma-clip rejected all stable sources. Using default."
            )
            return sources.iloc[[]], fallback_radius, optimum_scale

        # ---- Tail check w.r.t. final optimum radius -----------------------
        # Exclude sources whose normalized flux goes < min_tail_flux or
        # > 1 + max_tail_excess beyond the final optimum radius.
        beyond_final = (radii / fwhm) > optimum_radius
        if beyond_final.any():
            still_ok = []
            for i in final_indices:
                prof = profiles_map.get(i)
                if prof is None:
                    still_ok.append(i)
                    continue
                tail = prof[beyond_final]
                t_min = float(np.nanmin(tail))
                t_excess = float(max(0.0, np.nanmax(tail) - 1.0))
                if (
                    np.isfinite(t_min)
                    and t_min >= min_tail_flux
                    and np.isfinite(t_excess)
                    and t_excess < max_tail_excess
                ):
                    still_ok.append(i)
            if len(still_ok) < len(final_indices):
                final_indices = np.array(still_ok, dtype=int)
                filtered_sources = sources.iloc[final_indices].copy()
                if len(final_indices) > 0:
                    optimum_radius = float(
                        np.nanmedian(
                            sources.loc[final_indices, "optimum_radius"].values
                        )
                    )

        # ---- Optional EE model refinement ----------------------------------
        fine_r, fine_profile = None, None
        good_profiles = np.array(
            [profiles_map[i] for i in final_indices if i in profiles_map], dtype=float
        )
        if len(final_indices) > 0 and good_profiles.size > 0:
            try:
                if good_profiles.ndim == 2 and good_profiles.shape[0] > 0:
                    mean_profile = np.nanmedian(good_profiles, axis=0)

                    def ee_model(r, alpha, beta):
                        return 1.0 - np.exp(-((r / radii[-1] / alpha) ** beta))

                    popt, _ = curve_fit(
                        ee_model,
                        radii,
                        mean_profile,
                        p0=[0.3, 2.0],
                        bounds=(0, np.inf),
                        maxfev=10000,
                    )
                    fine_r = np.linspace(0, radii[-1], 500)
                    fine_profile = ee_model(fine_r, *popt)
                    r_target_pix = np.interp(aperture_norm_factor, fine_profile, fine_r)
                    if np.isfinite(r_target_pix) and r_target_pix > 0:
                        optimum_radius = float(r_target_pix / fwhm)
            except Exception as exc:
                logger.warning(f"EE model fit failed: {exc}")

        # ---- Optimum scale -------------------------------------------------
        optimum_scale = max(12, int(np.ceil(optimum_radius + 2 * fwhm))) + 0.5
        if (2 * optimum_scale) % 2 == 0:
            optimum_scale += 0.5

        # ---- Plotting (reuses profiles already in profiles_map) ------------
        if plot:
            # No second Pool - profiles were computed in the analysis pass.
            dir_path = os.path.dirname(os.path.realpath(__file__))
            try:
                plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))
            except Exception:
                pass

            save_loc = os.path.join(
                self.input_yaml["write_dir"],
                f'optimum_aperture_{self.input_yaml["base"]}.pdf',
            )
            fig = plt.figure(figsize=set_size(340, 1.5))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)

            kept_set = set(final_indices.tolist())
            # Plot all profiles; grey-out rejected ones.
            for idx, prof in profiles_map.items():
                in_kept = idx in kept_set
                ax1.plot(
                    radii / fwhm,
                    prof,
                    color=None if in_kept else "grey",
                    alpha=1.0 if in_kept else 0.3,
                )

            if fine_r is not None:
                ax1.plot(fine_r / fwhm, fine_profile, ls="--", color="black")

            ax1.axvline(
                global_optimum_pre, color="grey", ls=":", label="Initial global"
            )
            ax1.axvline(optimum_radius, color="black", ls="--", label="Final optimum")
            ax1.set_ylabel("Normalized Flux")
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.legend(loc="lower right", fontsize=8)

            per_source = (
                sources.loc[list(kept_set), "optimum_radius"].values
                if kept_set
                else np.array([])
            )
            other = sources.loc[
                [i for i in range(len(sources)) if i not in kept_set], "optimum_radius"
            ].values
            all_radii = np.concatenate(
                [np.atleast_1d(per_source), np.atleast_1d(other)]
            )
            all_radii = all_radii[np.isfinite(all_radii)]

            if len(all_radii) > 0:
                r_min, r_max = float(np.nanmin(all_radii)), float(np.nanmax(all_radii))
                r_range = max(r_max - r_min, 0.1)
                n_bins = min(25, max(8, int(np.ceil(len(all_radii) / 4))))
                bins = np.linspace(
                    max(0, r_min - 0.02 * r_range),
                    min(max_radius, r_max + 0.02 * r_range),
                    n_bins + 1,
                )
                if len(per_source) > 0:
                    ax2.hist(
                        per_source,
                        bins=bins,
                        facecolor="tab:blue",
                        alpha=0.85,
                        label="Selected",
                        zorder=1,
                    )
                if len(other) > 0:
                    ax2.hist(
                        other,
                        bins=bins,
                        facecolor="tab:red",
                        alpha=0.4,
                        label="Rejected",
                        zorder=0,
                    )
                ax2.legend(loc="upper right", fontsize=7)

            ax2.axvline(optimum_radius, color="black", ls="--", label="Final")
            ax2.set_xlabel("Aperture Radius [FWHM]")
            ax2.set_ylabel("Count")
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlim(-0.05, max_radius + 0.05)

            fig.savefig(save_loc, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close(fig)

        logger.info(
            f"Returning {len(filtered_sources)} sources, "
            f"optimum radius: {optimum_radius:.2f} * FWHM, "
            f"scale: {optimum_scale}"
        )
        return filtered_sources, optimum_radius, optimum_scale

    # -----------------------------------------------------------------------
    # Aperture correction
    # -----------------------------------------------------------------------

    def compute_aperture_correction(
        self,
        image: np.ndarray,
        sources: pd.DataFrame,
        n_samples: int = 25,
        fwhm: float = None,
        ap_size: float = None,
        max_radius: float = 5.0,
        background_rms: np.ndarray = None,
        plot: bool = True,
    ):
        """
        Compute the aperture correction (ap_size -> inf) via Curve of Growth.

        Parameters
        ----------
        image          : 2-D ndarray
        sources        : DataFrame with 'x_pix', 'y_pix', 'flux_AP'
        n_samples      : bright stars to include
        fwhm           : PSF FWHM in pixels (required)
        ap_size        : science aperture radius in pixels (required)
        max_radius     : CoG extent in FWHM units
        background_rms : optional 2-D RMS map
        plot           : save histogram PDF

        Returns
        -------
        (correction, correction_err) : (float, float)  [mag]
        """
        logger = logging.getLogger(__name__)
        write_dir = self.input_yaml["write_dir"]
        base_name = self.input_yaml["base"]

        if len(sources) < 5:
            logger.warning(f"Too few sources [{len(sources)}] for aperture correction.")
            return np.nan, np.nan

        if fwhm is None or ap_size is None:
            raise ValueError("fwhm and ap_size are required.")

        radii = np.arange(0.05, max_radius, 0.1) * fwhm
        error = (
            calc_total_error(image, background_rms, effective_gain=1)
            if background_rms is not None
            else None
        )

        selected = sources.sort_values("flux_AP", ascending=False).head(n_samples)

        corrections = []
        for _, row in selected.iterrows():
            try:
                xycen = np.array([row["x_pix"], row["y_pix"]])
                cog = CurveOfGrowth(
                    image, xycen, radii, error=error, mask=None, method="subpixel"
                )
                cog.normalize()
                frac = np.interp(ap_size, cog.radii, cog.profile)
                if 0 < frac <= 1:
                    corrections.append(-2.5 * np.log10(1.0 / frac))
            except Exception as exc:
                logger.warning(f"Skipping star: {exc}")

        if not corrections:
            logger.warning("No valid aperture corrections computed.")
            return np.nan, np.nan

        corrections = np.asarray(corrections, dtype=float)

        # Sigma-clip outliers.
        clipped = sigma_clip(
            corrections,
            sigma=3,
            masked=True,
            cenfunc=np.nanmedian,
            stdfunc=mad_std,
        )
        corrections = corrections[~clipped.mask]
        correction = float(np.nanmedian(corrections))
        correction_err = float(np.nanstd(corrections))
        logger.info(f"Aperture correction: {correction:.3f} +/- {correction_err:.3f}")

        if plot:
            plt.ioff()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(
                corrections, bins=15, alpha=0.7, color="steelblue", edgecolor="black"
            )
            ax.axvline(
                correction, color="r", ls="--", label=f"Median: {correction:.3f}"
            )
            ax.set_xlabel("Aperture Correction [mag]")
            ax.set_ylabel("Frequency")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(
                os.path.join(write_dir, f"aperture_correction_{base_name}.pdf"),
                format="pdf",
                bbox_inches="tight",
                dpi=150,
                facecolor="white",
            )
            plt.close(fig)

        return correction, correction_err
