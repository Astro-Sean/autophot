#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limiting-magnitude estimation utilities.

This module supports both background-based limiting estimates and injection/
recovery experiments by simulating PSF sources into science cutouts and
measuring the detection threshold required for robust photometry.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache


@contextmanager
def _pool_or_serial(n_jobs: int):
    """Yield a ProcessPoolExecutor when n_jobs > 1, else None (serial; avoids fork on HPC)."""
    if n_jobs <= 1:
        yield None
        return
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        yield pool


# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from functions import (
    set_size,
    gauss_1d,  # used directly - NOT redefined inside methods
    flux_upper_limit,
    mag,
    points_in_circum,
    beta_aperture,
    border_msg,
)
from aperture import Aperture

# ===========================================================================
# Module-level worker functions
# ===========================================================================
# Workers must live at module scope so pickle (used by multiprocessing) can
# find them by name.  Instance methods cannot be pickled on all platforms.


def _fake_aperture_worker(args):
    """
    Draw *n_trials* random fake apertures from unmasked background pixels and
    return their summed flux.

    Parameters
    ----------
    args : tuple
        (cutout_e, includ_zip, aperture_area, n_trials, seed)
        cutout_e      - gain-scaled cutout (2-D ndarray)
        includ_zip    - (N, 2) array of valid (row, col) background pixel coords
        aperture_area - number of pixels per synthetic aperture
        n_trials      - number of apertures to draw in this chunk
        seed          - RNG seed for reproducibility
    """
    cutout_e, includ_zip, aperture_area, n_trials, seed = args
    rng = np.random.default_rng(seed)
    # Sample pixel indices with replacement so every trial is independent.
    idx = rng.integers(
        0, len(includ_zip), size=(n_trials, aperture_area), endpoint=False
    )
    return np.nansum(cutout_e[includ_zip[idx, 0], includ_zip[idx, 1]], axis=1)


def _injection_worker(args):
    """
    Inject a scaled PSF at a jittered position and attempt aperture recovery.

    Parameters
    ----------
    args : tuple
        (x_inj, y_inj, F_amp, gridx, gridy, cutout,
         epsf_model, input_yaml, background_rms,
         detection_limit, DETECTION_PROB_THRESH)

    Returns
    -------
    (detection_flag, beta_p) : (bool, float)
    """
    (
        x_inj,
        y_inj,
        F_amp,
        gridx,
        gridy,
        cutout,
        epsf_model,
        input_yaml,
        background_rms,
        detection_limit,
        DETECTION_PROB_THRESH,
    ) = args

    try:
        psf_img = epsf_model.evaluate(
            x=gridx, y=gridy, flux=F_amp, x_0=x_inj, y_0=y_inj
        )
        new_img = cutout + psf_img

        ap = Aperture(input_yaml=input_yaml, image=new_img)
        trial_df = pd.DataFrame({"x_pix": [x_inj], "y_pix": [y_inj]})
        mres = ap.measure(
            sources=trial_df, plot=False, background_rms=background_rms, verbose=0
        )

        beta_p = beta_aperture(
            n=detection_limit,
            flux_aperture=float(mres["flux_AP"].iloc[0]),
            sigma=float(mres["noiseSky"].iloc[0]),
            npix=float(mres["area"].iloc[0]),
        )
        return beta_p >= DETECTION_PROB_THRESH, beta_p

    except Exception:
        return False, 0.0


# ===========================================================================
# limits class
# ===========================================================================


class Limits:
    """
    Compute limiting magnitudes for a single astronomical image frame using:

    * Probabilistic background sampling  (getProbableLimit)
    * PSF injection / recovery           (getInjectedLimit)
    """

    def __init__(self, input_yaml: dict):
        """
        Parameters
        ----------
        input_yaml : dict
            Pipeline configuration loaded from YAML.  Expected keys include
            ``fwhm``, ``scale``, ``gain``, ``exposure_time``, ``target_x_pix``,
            ``target_y_pix``, ``photometry.aperture_radius``,
            ``limiting_magnitude.inject_source_location``, ``fpath``.
        """
        self.input_yaml = input_yaml

        # Optional RNG seed for reproducible limiting-magnitude experiments.
        seed = self.input_yaml.get("rng_seed", None)
        self._rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

    # -----------------------------------------------------------------------
    # Cutout helper
    # -----------------------------------------------------------------------

    def get_cutout(self, image: np.ndarray, position=None) -> np.ndarray | None:
        """
        Extract a square cutout centred on the target (or supplied) position.

        The half-size of the cutout is ``ceil(inject_source_location * fwhm + scale)``.

        Parameters
        ----------
        image : ndarray  - full science frame
        position : (x, y) tuple, optional
            Pixel coordinates for the cutout centre.  Defaults to
            ``target_x_pix`` / ``target_y_pix`` from config.

        Returns
        -------
        ndarray or None
        """
        logger = logging.getLogger(__name__)
        try:
            if position is None:
                tx = self.input_yaml["target_x_pix"]
                ty = self.input_yaml["target_y_pix"]
            else:
                tx, ty = position
            if not (np.isfinite(tx) and np.isfinite(ty)):
                logger.warning(
                    "getCutout: target position (%.2f, %.2f) is not finite; skipping cutout.",
                    tx,
                    ty,
                )
                return None

            fwhm = self.input_yaml["fwhm"]
            scale = self.input_yaml["scale"]
            mag_factor = self.input_yaml["limiting_magnitude"]["inject_source_location"]

            half = int(np.ceil(mag_factor * fwhm + scale))
            cutout = Cutout2D(image, position=(tx, ty), size=(2 * half, 2 * half))
            return cutout.data

        except Exception as exc:
            logger.debug(f"getCutout failed: {exc}")
            return None

    # -----------------------------------------------------------------------
    # PSF helpers
    # -----------------------------------------------------------------------

    def _normalize_psf(self, epsf_model, shape: tuple):
        """
        Ensure the PSF model integrates to unity over *shape* pixels.

        Parameters
        ----------
        epsf_model : photutils ePSF model
        shape      : (height, width)

        Returns
        -------
        epsf_model  (modified in-place and returned for convenience)
        """
        grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        centre = shape[0] // 2
        psf_data = epsf_model.evaluate(
            x=grid[0], y=grid[1], flux=1.0, x_0=centre, y_0=centre
        )
        total = np.nansum(psf_data)
        if not np.isclose(total, 1.0, rtol=0.01):
            epsf_model.flux = 1.0 / total
        return epsf_model

    @staticmethod
    def _downsample_psf(psf_os: np.ndarray, oversampling: int) -> np.ndarray:
        """
        Flux-conserving block-average downsampling of an oversampled PSF.

        Reshapes into (H, os, W, os) blocks and takes the mean over the
        oversampling axes - exact for integer oversampling factors.

        Parameters
        ----------
        psf_os      : 2-D ndarray  (H*os, W*os)
        oversampling: int

        Returns
        -------
        psf_unit : 2-D ndarray  (H, W),  sum ~ 1/(os^2) * sum(psf_os)
        """
        H_os, W_os = psf_os.shape
        H = H_os // oversampling
        W = W_os // oversampling
        # Trim to exact multiple then reshape and average.
        return (
            psf_os[: H * oversampling, : W * oversampling]
            .reshape(H, oversampling, W, oversampling)
            .mean(axis=(1, 3))
        )

    # -----------------------------------------------------------------------
    # Probabilistic limiting magnitude
    # -----------------------------------------------------------------------

    def get_probable_limit(
        self,
        cutout: np.ndarray,
        bkg_level: float = 3.0,
        detection_limit: float = 3.0,
        useBeta: bool = True,
        beta: float = 0.5,
        plot: bool = True,
        unityPSF=None,
        residualTable=None,
        functionParams=None,
        n_jobs: int = None,
    ) -> float:
        """
        Estimate the limiting magnitude from background-noise statistics.

        Fake apertures are drawn from unmasked background pixels; a Gaussian
        is fitted to their flux distribution; the sigma of that Gaussian is used
        to compute an upper-limit count rate.

        Parameters
        ----------
        cutout          : 2-D ndarray  (sky-subtracted, un-scaled)
        bkg_level       : sigma threshold for source detection
        detection_limit : S/N threshold for upper limit (n*sigma or beta formalism)
        useBeta         : use Bayesian beta formalism (True) or n*sigma (False)
        beta            : beta probability threshold
        plot            : unused placeholder (kept for API compatibility)
        n_jobs          : worker processes; defaults to os.cpu_count()

        Returns
        -------
        float  - limiting magnitude (instrumental), or np.nan on failure
        """
        logger = logging.getLogger(__name__)

        try:
            # ---- Gain scaling ------------------------------------------------
            cutout_e = cutout * self.input_yaml["gain"]

            # ---- Background statistics ---------------------------------------
            _, _, bkg_std = sigma_clipped_stats(
                cutout_e,
                sigma=3.0,
                cenfunc=np.nanmedian,
                stdfunc="mad_std",
            )

            # ---- Source detection --------------------------------------------
            daofind = DAOStarFinder(
                fwhm=self.input_yaml["fwhm"],
                threshold=bkg_level * bkg_std,
                sharplo=0.2,
                sharphi=1.0,
                roundlo=-1.0,
                roundhi=1.0,
            )
            sources = daofind(cutout_e)

            positions = (
                list(zip(sources["xcentroid"], sources["ycentroid"]))
                if sources is not None
                else []
            )
            # Always exclude the image centre (target location).
            positions.append((cutout_e.shape[1] / 2.0, cutout_e.shape[0] / 2.0))

            # ---- Exclusion mask ----------------------------------------------
            source_r = self.input_yaml["photometry"]["aperture_radius"]
            aperture_area = int(np.pi * source_r**2)
            # Crowded fields: exclude a full aperture radius around each source
            # so fake apertures do not overlap source wings (crowd-safe).
            exclusion_r = (
                source_r
                if self.input_yaml.get("photometry", {}).get("crowded_field", False)
                else source_r / 2
            )
            masks = CircularAperture(positions, r=exclusion_r).to_mask(method="center")

            # Vectorised: build a (n_masks, H, W) boolean stack then collapse.
            # Much faster than a Python loop over full-image arrays.
            mask_stack = np.stack(
                [m.to_image(cutout_e.shape) > 0 for m in masks], axis=0
            )
            mask_sumed = mask_stack.any(axis=0).astype(np.uint8)

            # Pixels that are unmasked AND finite are eligible background pixels.
            bg_image = cutout_e * (1 - mask_sumed)
            row_idx, col_idx = np.where(bg_image != 0)
            includ_zip = np.column_stack([row_idx, col_idx])  # shape (N, 2)

            # Fallback: if masking removed too many pixels use all finite pixels.
            if len(includ_zip) < aperture_area:
                row_idx, col_idx = np.where(np.isfinite(bg_image))
                includ_zip = np.column_stack([row_idx, col_idx])

            if len(includ_zip) < aperture_area:
                logger.warning("Too few background pixels; result may be unreliable.")
                return np.nan

            # ---- Parallel or serial fake-aperture draws ----------------------
            number_of_points = 150
            n_jobs = min(n_jobs or os.cpu_count(), number_of_points)
            # Serial path when n_jobs==1 to avoid fork/ProcessPool (HPC resource limits).
            if n_jobs == 1:
                fake_sums = _fake_aperture_worker(
                    (
                        cutout_e,
                        includ_zip,
                        aperture_area,
                        number_of_points,
                        int(self._rng.integers(0, 2**32 - 1)),
                    )
                )
            elif number_of_points < 10:
                # Avoid process-spawn overhead for tiny trial counts.
                fake_sums = _fake_aperture_worker(
                    (
                        cutout_e,
                        includ_zip,
                        aperture_area,
                        number_of_points,
                        int(self._rng.integers(0, 2**32 - 1)),
                    )
                )
            else:
                base_trials = number_of_points // n_jobs
                remainder = number_of_points % n_jobs
                seeds = self._rng.integers(0, 2**32 - 1, size=n_jobs, dtype=np.uint64)
                args_list = [
                    (
                        cutout_e,
                        includ_zip,
                        aperture_area,
                        base_trials + (1 if j < remainder else 0),
                        int(seeds[j]),
                    )
                    for j in range(n_jobs)
                ]
                with ProcessPoolExecutor(max_workers=n_jobs) as exe:
                    results = list(exe.map(_fake_aperture_worker, args_list))
                fake_sums = np.concatenate(results)

            # ---- Gaussian fit to flux distribution --------------------------
            hist, bins = np.histogram(fake_sums, bins=number_of_points, density=True)
            centres = (bins[:-1] + bins[1:]) / 2.0

            try:
                popt, _ = curve_fit(
                    gauss_1d,
                    centres,
                    hist,
                    p0=[np.nanmax(hist), np.nanmean(fake_sums), np.nanstd(fake_sums)],
                    absolute_sigma=True,
                    maxfev=5000,
                )
                std = abs(popt[2])
            except Exception as fit_err:
                logger.warning(f"Gaussian fit failed ({fit_err}); using robust std")
                std = np.nanstd(fake_sums)
                if not np.isfinite(std) or std <= 0:
                    std = float(mad_std(fake_sums, ignore_nan=True))
            if not np.isfinite(std) or std <= 0:
                logger.warning(
                    "Background flux dispersion invalid; cannot compute limiting mag"
                )
                return np.nan

            # ---- Upper-limit counts -> flux -> magnitude ----------------------
            counts_upper = (
                flux_upper_limit(n=detection_limit, beta_p=beta, sigma=std)
                if useBeta
                else detection_limit * std
            )
            flux_upper = counts_upper / self.input_yaml["exposure_time"]
            if not np.isfinite(flux_upper) or flux_upper <= 0:
                logger.warning("Upper-limit flux invalid; limiting mag set to NaN")
                magUpperlimit = np.nan
            else:
                magUpperlimit = mag(flux_upper)

        except Exception as exc:
            logger.error(f"getProbableLimit failed: {exc}", exc_info=True)
            magUpperlimit = np.nan

        return magUpperlimit

    # -----------------------------------------------------------------------
    # PSF injection / recovery limiting magnitude
    # -----------------------------------------------------------------------

    def get_injected_limit(
        self,
        cutout: np.ndarray,
        position,
        epsf_model=None,
        initialGuess: float = -5.0,
        detection_limit: float = 3.0,
        detection_cutoff: float = 0.5,
        plot: bool = True,
        background_rms: np.ndarray = None,
        subtraction_ready: bool = False,
        zeropoint: float = None,
        n_jobs: int = None,
    ) -> float:
        """
        Bracket-and-bisect search for the limiting magnitude by injecting
        artificial PSF sources and measuring their recoverability.

        A *single* ProcessPoolExecutor is created for the entire search and
        reused by the nested ``run_trials_at_mag`` helper - avoiding the
        ~40-60 Pool creations that the original code performed.

        Parameters
        ----------
        cutout          : full 2-D science image (pre-subtraction)
        position        : (x, y) target pixel coords used to centre the cutout
        epsf_model      : photutils ePSF model
        initialGuess    : starting instrumental magnitude for the bracket
        detection_limit : S/N or beta threshold for a "detection"
        detection_cutoff: fraction of trials that must be detections to count
        plot            : save diagnostic completeness PDF
        background_rms  : full-frame RMS map (optional)
        subtraction_ready: unused placeholder
        zeropoint       : adds an apparent-magnitude axis to the plot
        n_jobs          : worker processes; defaults to os.cpu_count()

        Returns
        -------
        float - limiting instrumental magnitude, or np.nan on failure
        """
        logger = logging.getLogger(__name__)
        start_time = time.time()

        try:
            # =================================================================
            # Validation
            # =================================================================
            if epsf_model is None:
                logger.info("No PSF model - skipping limiting magnitude")
                return np.nan

            if np.isnan(initialGuess):
                initialGuess = -5.0

            # =================================================================
            # Extract cutouts (ONCE for both science and RMS maps)
            # =================================================================
            cutout = self.get_cutout(image=cutout, position=position)
            if cutout is None:
                logger.warning("getCutout returned None; aborting")
                return np.nan

            if background_rms is not None:
                background_rms = self.get_cutout(
                    image=background_rms, position=position
                )

            # Re-centre position to the middle of the extracted cutout.
            H, W = cutout.shape
            position = [W / 2.0, H / 2.0]

            # Pixel grids for PSF evaluation (integer pixels, no oversampling yet).
            gridx, gridy = np.meshgrid(np.arange(W), np.arange(H))

            # =================================================================
            # Oversampling - compute grids ONCE
            # =================================================================
            raw_os = getattr(epsf_model, "oversampling", 1)
            if isinstance(raw_os, (list, tuple, np.ndarray)):
                oversampling = int(raw_os[0])
            elif np.isscalar(raw_os) and raw_os > 1:
                oversampling = int(raw_os)
            else:
                oversampling = 1

            logger.info(f"PSF oversampling factor: {oversampling}x")

            if oversampling > 1:
                # Fine grid spanning exactly the same pixel extent as gridx/gridy.
                gx_os = np.linspace(0, W - 1, W * oversampling)
                gy_os = np.linspace(0, H - 1, H * oversampling)
                gridx_os, gridy_os = np.meshgrid(gx_os, gy_os)
            else:
                gridx_os, gridy_os = gridx, gridy

            # =================================================================
            # PSF calibration: flux=1 -> what instrumental magnitude?
            # =================================================================
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            psf_os = epsf_model.evaluate(
                x=gridx_os, y=gridy_os, flux=1.0, x_0=cx, y_0=cy
            )
            logger.info(
                f"Oversampled PSF shape={psf_os.shape}, " f"sum={np.nansum(psf_os):.4f}"
            )

            psf_unit = (
                self._downsample_psf(psf_os, oversampling)
                if oversampling > 1
                else psf_os
            )
            psf_unit = psf_unit / np.nansum(psf_unit)  # exact unit normalisation

            psf_ap = Aperture(input_yaml=self.input_yaml, image=psf_unit)
            psf_meas = psf_ap.measure(
                pd.DataFrame({"x_pix": [W / 2.0], "y_pix": [H / 2.0]}),
                plot=False,
                verbose=0,
            )
            F_ref = float(psf_meas["flux_AP"].iloc[0])
            m_ref = mag(F_ref)
            logger.info(f"PSF calibration: flux={F_ref:.4e} -> m_ref={m_ref:.3f}")

            # Memoised so repeated calls at the same magnitude are free.
            @lru_cache(maxsize=512)
            def flux_for_mag(m: float) -> float:
                return 10.0 ** (-0.4 * (m - m_ref))

            # =================================================================
            # Choose quiet injection sites
            # =================================================================
            DETECTION_PROB_THRESH = detection_cutoff
            sourceNum = 10
            distance_factor = 1.0
            injection_df = pd.DataFrame()

            for attempt in range(2):
                inj_dist = (
                    self.input_yaml["limiting_magnitude"].get(
                        "inject_source_location", 3
                    )
                    * self.input_yaml["fwhm"]
                    * distance_factor
                )
                pts = points_in_circum(inj_dist, center=position, n=sourceNum)
                xran = [p[0] for p in pts]
                yran = [p[1] for p in pts]
                df = pd.DataFrame({"x_pix": xran, "y_pix": yran})

                ini_ap = Aperture(input_yaml=self.input_yaml, image=cutout)
                df = ini_ap.measure(
                    sources=df,
                    plot=False,
                    background_rms=background_rms,
                    verbose=0,
                )
                p_det = df.apply(
                    lambda row: beta_aperture(
                        n=detection_limit,
                        flux_aperture=row["flux_AP"],
                        sigma=row["noiseSky"],
                        npix=row["area"],
                    ),
                    axis=1,
                )
                # "Quiet" sites: choose positions with detection probability below
                # the same cutoff used later to define what counts as a detection.
                injection_df = df[p_det < DETECTION_PROB_THRESH].copy()

                if len(injection_df) > 0:
                    break
                logger.info("No quiet positions; retrying at 2x distance ...")
                distance_factor = 2.0

            if len(injection_df) == 0:
                # Fallback: use cutout centre only so we still get a limit in crowded/empty fields
                logger.info("No quiet positions; using cutout centre only")
                injection_df = pd.DataFrame(
                    {"x_pix": [position[0]], "y_pix": [position[1]]}
                )

            if len(injection_df) > sourceNum:
                injection_df = injection_df.sample(sourceNum).reset_index(drop=True)

            x_pix_arr = injection_df["x_pix"].to_numpy()
            y_pix_arr = injection_df["y_pix"].to_numpy()
            n_sites = len(injection_df)

            n_jobs = n_jobs or os.cpu_count()
            # Cap workers to avoid HPC fork/resource limits (serial when 1).
            n_jobs = max(1, min(n_jobs, 8))

            # =================================================================
            # Trial runner - captures the shared pool from the outer scope.
            #
            # KEY OPTIMISATION: the ProcessPoolExecutor is created once below
            # and referenced here.  The original code called Pool() inside
            # this closure on every single magnitude evaluation.
            # =================================================================
            rng = self._rng

            def run_trials_at_mag(m: float, redo: int = 3, pool=None):
                """
                Inject at *m* at all sites with *redo* sub-pixel jitter
                repetitions and return (mean_detection_rate, beta_array).
                """
                dx = rng.random((n_sites, redo)) - 0.5
                dy = rng.random((n_sites, redo)) - 0.5
                F = flux_for_mag(m)

                tasks = [
                    (
                        x_pix_arr[k] + dx[k, j],
                        y_pix_arr[k] + dy[k, j],
                        F,
                        gridx,
                        gridy,
                        cutout,
                        epsf_model,
                        self.input_yaml,
                        background_rms,
                        detection_limit,
                        DETECTION_PROB_THRESH,
                    )
                    for k in range(n_sites)
                    for j in range(redo)
                ]

                if pool is not None:
                    results = list(pool.map(_injection_worker, tasks))
                else:
                    results = [_injection_worker(t) for t in tasks]

                det_flags = np.array([r[0] for r in results], dtype=bool)
                betas = np.array([r[1] for r in results], dtype=float)
                return float(det_flags.mean()), betas

            # =================================================================
            # Single ProcessPoolExecutor for the ENTIRE search (or serial if n_jobs==1)
            # =================================================================
            inject_lmag = np.nan
            bracket_steps: list[tuple] = []
            bisect_steps: list[tuple] = []

            with _pool_or_serial(n_jobs) as pool:

                # ---- Bracket phase ------------------------------------------
                step = 0.5
                max_steps = 30
                m_bright = float(initialGuess)
                c_bright, _ = run_trials_at_mag(m_bright, pool=pool)
                going_faint = c_bright >= detection_cutoff
                m_faint, c_faint = m_bright, c_bright

                bracket_steps.append((m_bright, c_bright))

                for _ in range(max_steps):
                    m_test = m_faint + step if going_faint else m_bright - step
                    c_test, _ = run_trials_at_mag(m_test, pool=pool)
                    bracket_steps.append((m_test, c_test))

                    if going_faint:
                        m_faint, c_faint = m_test, c_test
                        if c_faint < detection_cutoff:
                            break
                    else:
                        m_bright, c_bright = m_test, c_test
                        if c_bright >= detection_cutoff:
                            break

                bracketed = (c_bright >= detection_cutoff) and (
                    c_faint < detection_cutoff
                )
                if not bracketed:
                    # Retry once with initial guess from probabilistic limit (serial to avoid nested pools)
                    probable = self.get_probable_limit(
                        cutout,
                        bkg_level=3.0,
                        detection_limit=detection_limit,
                        useBeta=True,
                        beta=detection_cutoff,
                        plot=False,
                        n_jobs=1,
                    )
                    if np.isfinite(probable):
                        logger.info(
                            f"Bracket failed; retrying with initialGuess={probable - 0.5:.2f} from probable limit"
                        )
                        m_bright = float(probable - 0.5)
                        c_bright, _ = run_trials_at_mag(m_bright, pool=pool)
                        going_faint = c_bright >= detection_cutoff
                        m_faint, c_faint = m_bright, c_bright
                        bracket_steps.append((m_bright, c_bright))
                        for _ in range(35):
                            m_test = m_faint + step if going_faint else m_bright - step
                            c_test, _ = run_trials_at_mag(m_test, pool=pool)
                            bracket_steps.append((m_test, c_test))
                            if going_faint:
                                m_faint, c_faint = m_test, c_test
                                if c_faint < detection_cutoff:
                                    break
                            else:
                                m_bright, c_bright = m_test, c_test
                                if c_bright >= detection_cutoff:
                                    break
                        bracketed = (c_bright >= detection_cutoff) and (
                            c_faint < detection_cutoff
                        )

                if not bracketed:
                    # Conservative estimate: faintest mag tried when stepping faint, else brightest when stepping bright
                    inject_lmag = float(m_faint if going_faint else m_bright)
                    logger.info(
                        f"Could not bracket cutoff; using conservative limit {inject_lmag:.3f}"
                    )

                else:
                    # ---- Bisect phase ----------------------------------------
                    lo_m, lo_c = m_bright, c_bright
                    hi_m, hi_c = m_faint, c_faint
                    bisect_steps = [(lo_m, lo_c), (hi_m, hi_c)]

                    for _ in range(30):
                        mid_m = 0.5 * (lo_m + hi_m)
                        mid_c, _ = run_trials_at_mag(mid_m, pool=pool)
                        bisect_steps.append((mid_m, mid_c))

                        if mid_c >= detection_cutoff:
                            lo_m, lo_c = mid_m, mid_c
                        else:
                            hi_m, hi_c = mid_m, mid_c

                        if abs(hi_m - lo_m) < 0.02:
                            break

                    inject_lmag = 0.5 * (lo_m + hi_m)

                # ---- Plot completeness curve (still inside pool context) -----
                if plot:
                    m_lo = (m_bright if np.isfinite(m_bright) else initialGuess) - 1.0
                    m_hi = (m_faint if np.isfinite(m_faint) else initialGuess) + 1.0
                    sample_mags = np.linspace(m_lo, m_hi, 11)

                    completeness_groups, medians = [], []
                    for m in sample_mags:
                        _, betas = run_trials_at_mag(m, pool=pool)
                        completeness_groups.append(betas)
                        medians.append(np.mean(betas >= DETECTION_PROB_THRESH))

                    self._plot_completeness(
                        sample_mags,
                        completeness_groups,
                        medians,
                        bracket_steps,
                        bisect_steps,
                        inject_lmag,
                        detection_cutoff,
                        DETECTION_PROB_THRESH,
                        zeropoint,
                    )

            # =================================================================
            # Log result
            # =================================================================
            elapsed = time.time() - start_time
            if np.isfinite(inject_lmag):
                app_str = f" ({zeropoint + inject_lmag:.3f} app)" if zeropoint else ""
                logger.info(
                    f"Limiting mag ~ {inject_lmag:.3f}{app_str}  [{elapsed:.1f}s]"
                )
            else:
                logger.info(f"Limiting magnitude search failed [{elapsed:.1f}s]")

            return float(inject_lmag)

        except Exception as exc:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning(
                f"getInjectedLimit failed: {exc} "
                f"[{exc_type.__name__}, {fname}:{exc_tb.tb_lineno}]"
            )
            return np.nan

    # -----------------------------------------------------------------------
    # Diagnostic plot
    # -----------------------------------------------------------------------

    def _plot_completeness(
        self,
        sample_mags,
        completeness_groups,
        medians,
        bracket_steps,
        bisect_steps,
        inject_lmag,
        detection_cutoff,
        DETECTION_PROB_THRESH,
        zeropoint,
    ) -> None:
        """
        Save a completeness curve PNG next to the FITS file.

        Extracted from getInjectedLimit to keep that method focused on the
        search logic and to make the plot code independently testable.
        """
        order = np.argsort(sample_mags)
        mags_sorted = sample_mags[order]
        groups_sorted = [completeness_groups[i] for i in order]
        medians_sorted = np.asarray([medians[i] for i in order], float)

        fpath = self.input_yaml["fpath"]
        base = os.path.splitext(os.path.basename(fpath))[0]
        write_dir = os.path.dirname(fpath)

        # Use the project-wide plotting style for consistency.
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, "autophot.mplstyle"))
        except Exception:
            # Fall back silently if the style cannot be loaded.
            pass

        plt.ioff()
        fig, ax = plt.subplots(figsize=set_size(540, 1))

        # Box plots for per-trial beta distributions at each sampled magnitude.
        for pos, group in zip(mags_sorted, groups_sorted):
            ax.boxplot(
                group,
                positions=[pos],
                widths=0.15,
                showfliers=False,
                patch_artist=False,
                manage_ticks=False,
                boxprops=dict(linewidth=0.6, color="#4D4D4D"),
                whiskerprops=dict(linewidth=0.6, color="#4D4D4D"),
                capprops=dict(linewidth=0.6, color="#4D4D4D"),
                medianprops=dict(linewidth=0.8, color="#FF0000"),
            )

        # Bracket and bisect search trajectories.
        if bracket_steps:
            bm, bc = zip(*bracket_steps)
            ax.plot(
                bm,
                bc,
                "o-",
                ms=3.5,
                lw=0.8,
                color="#0000FF",
                label="Bracket path",
            )
        if bisect_steps:
            bm, bc = zip(*bisect_steps)
            ax.plot(
                bm,
                bc,
                "s--",
                ms=3.5,
                lw=0.8,
                color="#00AA00",
                label="Bisect path",
            )

        # Reference lines.
        ax.axhline(0.5, color="0.7", lw=0.5, ls="--", zorder=0)
        ax.text(
            0.98,
            0.5,
            "50%",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="right",
            color="0.5",
        )

        if detection_cutoff != 0.5:
            ax.axhline(detection_cutoff, color="0.7", lw=0.5, ls="-.", zorder=0)
            ax.text(
                0.98,
                detection_cutoff,
                f"{int(100 * detection_cutoff)}%",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                ha="right",
                color="0.5",
            )

        if np.isfinite(inject_lmag):
            ax.axvline(
                inject_lmag,
                color="k",
                lw=0.5,
                ls="--",
                label="Adopted limit",
            )

        ax.set_xlabel("Injected ePSF instrumental magnitude")
        ax.set_ylabel("Per-trial detection probability (β)")
        ax.set_ylim(-0.05, 1.05)
        ax.invert_xaxis()
        ax.legend(loc="lower left", fontsize=7)

        # Optional apparent-magnitude secondary axis.
        if zeropoint is not None:
            secax = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda m: m + zeropoint,
                    lambda m: m - zeropoint,
                ),
            )
            secax.set_xlabel("Apparent magnitude")
            secax.invert_xaxis()

        fig.tight_layout()
        save_loc_png = os.path.join(write_dir, f"Completeness_{base}.png")
        fig.savefig(save_loc_png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
