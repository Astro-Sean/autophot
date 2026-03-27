# =============================================================================
# background_subtractor.py - Improved BackgroundSubtractor
# =============================================================================
# Key fixes for smooth backgrounds and effective source/galaxy masking:
#
#   1. MUCH LARGER BOX SIZES - mesh_scale raised from 2.5 to 10, with a hard
#      minimum of 64px.  Small boxes track sources instead of sky.
#   2. ITERATIVE SOURCE MASKING - 3-pass detect->mask->re-estimate cycle.
#      Pass 1: bright sources on raw image.
#      Pass 2: subtract preliminary background, detect fainter sources.
#      Pass 3: refine with updated mask.
#      This catches extended wings and faint field sources that single-pass misses.
#   3. MUCH LARGER FILTER SIZE - smoothing filter now scales as box_size//2
#      (not FWHM/2), ensuring the interpolated mesh is genuinely smooth.
#   4. BkgZoomInterpolator - replaces the default piecewise interpolator with
#      a spline-based zoom that produces a smooth continuous surface.
#   5. AGGRESSIVE DILATION - dilate_factor raised to 3.0 and iterations to 3,
#      ensuring source wings and PSF halos are fully excluded.
#   6. SATURATION MASKING - pixels near/above saturation are masked before
#      background estimation (bright stars leave extended ghosts otherwise).
#   7. BiweightLocationBackground - more robust to outlier-contaminated boxes
#      than MMMBackground when sources leak through the mask.
#   8. Lower exclude_percentile (80%) - boxes with >20% masked pixels are
#      interpolated over rather than estimated, preventing partial-source boxes
#      from pulling the background up.
# =============================================================================


# TODO: Mask out regions where the pixel shares the same value as it's nearest nighbout. Also dilute this to avoid aretifacts at the edges


# --- Standard Library ---
import os
import logging
from functools import lru_cache

# --- Third-Party ---
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from astropy.stats import (
    SigmaClip,
    sigma_clipped_stats,
    mad_std,
    gaussian_fwhm_to_sigma,
)
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.wcs as awcs

import pandas as pd

from photutils.background import (
    Background2D,
    MedianBackground,
    BiweightLocationBackground,
    BiweightScaleBackgroundRMS,
    MADStdBackgroundRMS,
    MMMBackground,
)
from photutils.background.interpolators import BkgZoomInterpolator, BkgIDWInterpolator
from photutils.segmentation import detect_threshold, detect_sources

from scipy.ndimage import binary_dilation, uniform_filter, gaussian_filter, label as ndi_label
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Local ---
from functions import border_msg, set_size, log_warning_from_exception
from wcs import get_wcs


# ---------------------------------------------------------------------------
# Helper: build a transparent-red overlay colormap
# ---------------------------------------------------------------------------
def _make_red_overlay_cmap(alpha: float = 0.5) -> mcolors.Colormap:
    colors_rgba = np.zeros((256, 4))
    colors_rgba[:, 0] = np.linspace(0, 1, 256)
    colors_rgba[:, 3] = np.linspace(0, alpha, 256)
    return mcolors.ListedColormap(colors_rgba)


_RED_OVERLAY_CMAP = _make_red_overlay_cmap(alpha=0.5)


# ---------------------------------------------------------------------------
# Helper: filled circular structuring element (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _disk_structuring_element(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


# ---------------------------------------------------------------------------
# Constant-region mask for source filtering (chip gaps, dead areas, flat borders)
# ---------------------------------------------------------------------------
def _constant_region_mask(
    image: np.ndarray,
    box: int = 5,
    variance_threshold: float | None = None,
    dilate_pixels: int = 2,
) -> np.ndarray | None:
    """
    Build a boolean mask of pixels lying in constant-value regions (e.g. chip
    gaps, dead columns, constant borders). Used to reject spurious sources
    detected there.

    Parameters
    ----------
    image : np.ndarray
        2D image (float).
    box : int
        Side length of the window used to compute local variance (default 5).
    variance_threshold : float or None
        Pixels with local variance <= this are considered constant. If None,
        use a small fraction of the image's robust std.
    dilate_pixels : int
        Radius (in pixels) to dilate the constant region so edges are excluded
        (default 2).

    Returns
    -------
    np.ndarray or None
        Boolean mask same shape as image (True = bad/constant region), or None
        if image is not 2D.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        return None
    # Ensure odd box
    box = max(3, int(box)) | 1
    # Local variance: E[X^2] - E[X]^2 over box (NaN propagates in uniform_filter)
    mean = uniform_filter(image, size=box, mode="constant", cval=np.nan)
    mean_sq = uniform_filter(image * image, size=box, mode="constant", cval=np.nan)
    local_var = mean_sq - mean * mean
    finite = np.isfinite(image)
    if variance_threshold is None:
        global_std = np.nanstd(image)
        variance_threshold = (max(global_std, 1e-10) * 1e-6) ** 2
    constant = (local_var <= variance_threshold) & finite
    if dilate_pixels > 0:
        r = int(dilate_pixels)
        constant = binary_dilation(constant, structure=_disk_structuring_element(r))
    return constant


# =============================================================================
# BackgroundSubtractor
# =============================================================================
class BackgroundSubtractor:
    """
    Estimate and subtract the sky background from an astronomical FITS image.

    Uses iterative source masking, large mesh boxes, and spline interpolation
    to produce a smooth background surface free of source/galaxy ghosts.
    """

    def __init__(self, config: dict):
        self.config = config

        dir_path = os.path.dirname(os.path.realpath(__file__))
        style_path = os.path.join(dir_path, "autophot.mplstyle")
        plt.style.use(style_path if os.path.exists(style_path) else "default")

        self.logger = logging.getLogger(__name__)
        self._box_size_cache: dict = {}

    # -------------------------------------------------------------------------
    # WCS helpers
    # -------------------------------------------------------------------------
    def get_rotation_angle(self, header) -> float:
        wcs = get_wcs(header)
        try:
            if hasattr(wcs.wcs, "cd"):
                cd = wcs.wcs.cd
            else:
                cd = wcs.wcs.cdelt[0] * wcs.wcs.pc
            angle_rad = np.arctan2(cd[1][0], cd[0][0])
            return np.degrees(angle_rad)
        except Exception as exc:
            log_warning_from_exception(
                self.logger, "Could not compute rotation angle", exc
            )
            return 0.0

    # -------------------------------------------------------------------------
    # Galaxy masking
    # -------------------------------------------------------------------------
    def masked_simbad_galaxies(
        self,
        galaxies: pd.DataFrame,
        image: np.ndarray,
        header,
        scale_factor: float = 2.0,
    ) -> np.ndarray:
        """
        Build a boolean mask covering extended SIMBAD galaxies.

        Parameters
        ----------
        scale_factor : float
            Multiply galaxy semi-axes by this factor to ensure the full extent
            (including faint outskirts) is masked.  Default 2.0 (was implicitly
            1.0 in the original, which left galaxy wings unmasked).
        """
        wcs = get_wcs(header)
        rotation_angle = self.get_rotation_angle(header)

        xy_pixel_scales = awcs.utils.proj_plane_pixel_scales(wcs)
        pix_scale_arcsec = xy_pixel_scales[0] * 3600.0

        valid = galaxies["galdim_majaxis"].apply(
            lambda x: isinstance(x, (int, float)) and not pd.isna(x) and x > 0
        ) & galaxies["galdim_minaxis"].apply(
            lambda x: isinstance(x, (int, float)) and not pd.isna(x) and x > 0
        )
        filtered_galaxies = galaxies[valid]
        self.logger.info(f"Masking {len(filtered_galaxies)} galaxies from SIMBAD")

        mask = np.zeros(image.shape, dtype=bool)
        y_grid, x_grid = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]

        for _, row in filtered_galaxies.iterrows():
            try:
                sky_pos = SkyCoord(ra=row["RA"] * u.deg, dec=row["DEC"] * u.deg)
                xpix, ypix = wcs.world_to_pixel(sky_pos)

                # FIX: scale_factor ensures faint galaxy outskirts are masked.
                maj_pix = (
                    scale_factor * (row["galdim_majaxis"] * 60.0) / pix_scale_arcsec
                )
                min_pix = (
                    scale_factor * (row["galdim_minaxis"] * 60.0) / pix_scale_arcsec
                )

                delta = 1 if rotation_angle < 0 else -1
                ellipse_angle = (
                    90.0 + delta * (row["galdim_angle"] + rotation_angle)
                ) % 360.0

                theta = np.radians(ellipse_angle)
                cos_t, sin_t = np.cos(-theta), np.sin(-theta)

                x_rel = x_grid - xpix
                y_rel = y_grid - ypix

                x_rot = cos_t * x_rel - sin_t * y_rel
                y_rot = sin_t * x_rel + cos_t * y_rel

                mask |= (x_rot / maj_pix) ** 2 + (y_rot / min_pix) ** 2 <= 1.0

            except Exception as exc:
                self.logger.info(
                    f"Error masking galaxy {row.get('MAIN_ID', 'unknown')}: {exc}"
                )

        self.logger.info(
            f"Galaxy mask excludes {np.sum(mask)} px "
            f"({100.0 * np.sum(mask) / image.size:.2f}%)"
        )
        return mask

    # -------------------------------------------------------------------------
    # Box / filter size estimation
    # -------------------------------------------------------------------------
    def _compute_box_sizes(
        self,
        image: np.ndarray,
        mask,
        fwhm_pixels: float = None,
        mesh_scale: float = 10.0,
        region_fraction_limit: float = 0.20,
        min_box: int = 64,
    ):
        """
        Derive Background2D box and smoothing-filter sizes from the image PSF.

        Design rationale
        ----------------
        * mesh_scale = 6 -> boxes are ~6x FWHM.  Large enough that individual
          sources don't dominate a box, but small enough that the mesh resolves
          real sky gradients (vignetting, scattered light, etc.).
        * min_box = 32 prevents pathologically small meshes.
        * filter_size is 3 or 5 **mesh boxes** (NOT pixels!).  In Background2D
          this is a median filter applied to the *mesh grid*, so filter_size=3
          means a 3x3 box neighbourhood.  Values much larger than 5 flatten
          the surface to a constant.
        * The BkgZoomInterpolator (set in _estimate_background) handles the
          smooth interpolation between mesh nodes - the filter_size just
          suppresses box-to-box noise in the mesh.
        """
        shape = image.shape

        # --- Estimate FWHM from autocorrelation ---
        if fwhm_pixels is None:
            min_dim = min(shape)
            try:
                sample_size = min(128, min_dim // 2)
                cy, cx = shape[0] // 2, shape[1] // 2
                hs = sample_size // 2
                sample = image[cy - hs : cy + hs, cx - hs : cx + hs]
                sample = sample - np.nanmedian(sample)

                ac = np.fft.fftshift(
                    np.abs(np.fft.ifft2(np.abs(np.fft.fft2(sample)) ** 2))
                )
                c0, c1 = np.array(ac.shape) // 2
                prof = ac[c0, c1:]
                half_max = prof[0] / 2.0
                half_idx = np.argmax(prof < half_max)
                fwhm_pixels = float(half_idx) if half_idx > 0 else 3.0
            except Exception:
                fwhm_pixels = 3.0

        # --- Compute mesh (box) size ---
        # Allow config to tighten/relax mesh behaviour.
        cfg_bkg = (
            self.config.get("background", {}) if isinstance(self.config, dict) else {}
        )
        fast_mode = bool(cfg_bkg.get("fast_mode", False))
        mesh_min = float(cfg_bkg.get("mesh_scale_min", 4.0))
        mesh_max = float(cfg_bkg.get("mesh_scale_max", 10.0))
        mesh_scale = float(mesh_scale)
        mesh_scale = max(mesh_min, min(mesh_scale, mesh_max))

        base = max(min_box, int(mesh_scale * fwhm_pixels))
        max_allowed = max(min_box, int(min(shape) * region_fraction_limit))
        base = min(base, max_allowed)
        base = base | 1  # force odd

        box_size = (base, base)

        # --- Compute smoothing filter size (in mesh-box units!) ---
        # Background2D applies a median filter to the *grid of box estimates*.
        # Here we favour a smooth global surface by default, but allow the
        # user to dial this via background.mesh_filter_size_mesh_units. In
        # fast_mode we default to a lighter filter (3x3 mesh boxes).
        default_filter = 3 if fast_mode else 5
        mesh_filter = int(cfg_bkg.get("mesh_filter_size_mesh_units", default_filter))
        if mesh_filter < 3:
            mesh_filter = 3
        if mesh_filter % 2 == 0:
            mesh_filter += 1
        filter_size = mesh_filter

        self.logger.info(
            "Background mesh: fwhm=%.2f px  mesh_scale=%.2f  box_size=%s  filter_size=%d",
            fwhm_pixels,
            mesh_scale,
            box_size,
            filter_size,
        )

        self._box_size_cache[shape] = (box_size, filter_size, fwhm_pixels)
        return self._box_size_cache[shape]

    # -------------------------------------------------------------------------
    # Source masking - ITERATIVE (the key fix for ghost elimination)
    # -------------------------------------------------------------------------
    def _make_source_mask(
        self,
        image: np.ndarray,
        nsigma: float = 2.5,
        npixels: int = 3,
        fwhm_pixels: float = 3.0,
        dilate_factor: float = 1.2,
        n_iterations: int = 3,  # NEW: iterative masking
    ) -> np.ndarray:
        """
        Iteratively detect and mask sources.

        The key improvement: a single-pass mask misses sources whose flux is
        comparable to the (source-contaminated) background estimate.  By
        iterating - detect sources -> estimate background -> subtract -> detect
        again - we progressively reveal fainter sources and their wings.

        Parameters
        ----------
        n_iterations : int
            Number of detect-mask-reestimate cycles.  3 is usually sufficient.
        dilate_factor : float
            Dilation radius = dilate_factor x fwhm_pixels.  Raised to 3.0 to
            cover PSF wings and diffuse halos that otherwise leak flux into
            background boxes.
        """
        # Guard against pathological or unknown FWHM values so the kernel
        # construction never divides by zero or creates a degenerate kernel.
        try:
            fwhm_val = float(fwhm_pixels)
        except Exception:
            fwhm_val = 0.0
        if not np.isfinite(fwhm_val) or fwhm_val <= 0:
            fwhm_val = 3.0  # conservative fallback in pixels

        sigma = fwhm_val * gaussian_fwhm_to_sigma
        size = int(max(3.0, 4.0 * fwhm_val)) | 1
        kern = Gaussian2DKernel(sigma, x_size=size, y_size=size)
        kern.normalize()

        # Dilation structuring element (cached).
        r_dilate = max(2, int(dilate_factor * fwhm_pixels))
        selem = _disk_structuring_element(r_dilate)

        cfg_bkg = (
            (self.config.get("background", {}) or {})
            if isinstance(self.config, dict)
            else {}
        )
        # Inspired by the STScI notebook: convolve before thresholding so
        # structured noise doesn't fragment detections into tiny islands.
        use_convolved_detection = bool(cfg_bkg.get("source_mask_convolve", False))
        # When True, only smooth on the first iteration (avoids N full-image convolves).
        convolve_first_iter_only = bool(
            cfg_bkg.get("source_mask_convolve_first_iter_only", True)
        )
        # Optional: update residual between iterations using a quick Background2D
        # fit (closer to the notebook's "detect -> estimate -> subtract -> detect").
        update_residual_with_background2d = bool(
            cfg_bkg.get("source_mask_iterative_bkg_update", False)
        )

        mask = np.zeros(image.shape, dtype=bool)
        residual = image.copy()

        # Safety: cap mask fraction so Background2D always has enough data to
        # fit a gradient.  If the mask exceeds this, stop iterating.
        max_mask_fraction = 0.45

        for iteration in range(n_iterations):
            try:
                work = residual
                if use_convolved_detection and (
                    not convolve_first_iter_only or iteration == 0
                ):
                    try:
                        # Smooth only for detection; do not change the data used
                        # for Background2D later.
                        work = convolve(
                            residual,
                            kern,
                            boundary="extend",
                            nan_treatment="interpolate",
                            preserve_nan=True,
                            normalize_kernel=True,
                        )
                    except Exception:
                        work = residual

                threshold = detect_threshold(work, nsigma=nsigma, mask=mask)

                # Use a small Gaussian filter kernel to suppress pixel noise
                # before segmentation.  Newer versions of photutils expect
                # this to be passed as "filter_kernel" (the legacy "kernel"
                # keyword is no longer supported).
                segm = detect_sources(
                    work,
                    threshold=threshold,
                    npixels=npixels,
                    connectivity=8,
                )
                if segm is None:
                    break

                new_mask = segm.data.astype(bool)
                # Dilate to cover source wings / PSF halos - 2 iterations is
                # enough with dilate_factor=2.0; 3 was too aggressive and
                # caused the mask to swallow most of the sky.
                new_mask = binary_dilation(new_mask, structure=selem, iterations=2)

                combined = mask | new_mask

                # Stop if we've hit the mask budget.
                frac = combined.mean()
                if frac > max_mask_fraction:
                    self.logger.info(
                        f"Source mask would reach {frac:.1%} (limit {max_mask_fraction:.0%}) "
                        f"- stopping at iteration {iteration + 1}"
                    )
                    # Keep the *previous* mask that was within budget, unless
                    # this is the first iteration (then accept what we have).
                    if iteration > 0:
                        break
                    mask = combined
                    break

                if np.array_equal(combined, mask):
                    self.logger.debug(
                        f"Source mask converged after {iteration + 1} iterations"
                    )
                    break
                mask = combined

                # Subtract a quick background estimate so the next iteration
                # can detect fainter sources hidden under the sky gradient.
                if iteration < n_iterations - 1:
                    if update_residual_with_background2d:
                        # Cheap and robust: coarse mesh, minimal smoothing.
                        # This is intentionally best-effort; if it fails, fall
                        # back to a flat-median subtraction.
                        try:
                            bkg = Background2D(
                                image,
                                mask=mask,
                                box_size=(64, 64),
                                filter_size=3,
                                sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
                                bkg_estimator=MedianBackground(),
                                bkgrms_estimator=MADStdBackgroundRMS(),
                                interpolator=BkgZoomInterpolator(order=1),
                                edge_method="pad",
                                exclude_percentile=90.0,
                            )
                            residual = image - np.asarray(bkg.background, dtype=float)
                        except Exception:
                            _, gmed, _ = sigma_clipped_stats(
                                image, sigma=3.0, mask=mask
                            )
                            residual = image - float(gmed)
                    else:
                        _, gmed, _ = sigma_clipped_stats(residual, sigma=3.0, mask=mask)
                        residual = image - float(gmed)

            except Exception as exc:
                log_warning_from_exception(
                    self.logger,
                    f"Source mask iteration {iteration} failed",
                    exc,
                )
                break

        n_masked = np.sum(mask)
        self.logger.info(
            "Source mask contains %d pixels (%.2f%% of the image) after %d iteration(s)",
            n_masked,
            100.0 * n_masked / image.size,
            min(iteration + 1, n_iterations),
        )
        return mask

    # -------------------------------------------------------------------------
    # Saturation mask (NEW - bright star ghosts)
    # -------------------------------------------------------------------------
    def _make_saturation_mask(
        self, image: np.ndarray, saturate: float, dilate_radius: int = 8
    ) -> np.ndarray:
        """
        Mask saturated pixels and their immediate surroundings.
        """
        sat_mask = image >= 0.90 * saturate
        if np.any(sat_mask):
            selem = _disk_structuring_element(dilate_radius)
            sat_mask = binary_dilation(sat_mask, structure=selem, iterations=1)
            self.logger.info(
                f"Saturation mask: {np.sum(sat_mask)} px "
                f"({100.0 * np.sum(sat_mask) / image.size:.2f}%)"
            )
        return sat_mask

    # -------------------------------------------------------------------------
    # Saturation streak / bleed mask (CCD bleed, diffraction spikes, curved trails)
    # -------------------------------------------------------------------------
    def _line_pixels(
        self, cy: float, cx: float, angle_rad: float, half_length: int, ny: int, nx: int
    ) -> tuple:
        """Return (rr, cc) integer pixel indices along a straight line from center."""
        dx = half_length * np.cos(angle_rad)
        dy = half_length * np.sin(angle_rad)
        n_pts = max(2, 2 * half_length)
        t = np.linspace(-1, 1, n_pts)
        yy = cy + t * dy
        xx = cx + t * dx
        rr = np.clip(np.round(yy).astype(int), 0, ny - 1)
        cc = np.clip(np.round(xx).astype(int), 0, nx - 1)
        return rr, cc

    def _follow_streak_path(
        self,
        image: np.ndarray,
        r0: int,
        c0: int,
        angle_rad: float,
        max_steps: int,
        flux_threshold: float,
        cone_deg: float = 38.0,
    ) -> tuple:
        """
        Follow a possibly curved streak by stepping toward the brightest pixel
        in a narrow cone ahead. Returns (rr, cc) of all visited pixels.
        """
        ny, nx = image.shape
        path_r, path_c = [r0], [c0]
        r, c = r0, c0
        half_cone = np.deg2rad(cone_deg)
        for _ in range(max_steps - 1):
            best_r, best_c, best_pixel_value = None, None, -np.inf
            for dangle in (0.0, half_cone, -half_cone):
                a = angle_rad + dangle
                dc = np.cos(a)
                dr = np.sin(a)
                for step in (1, 2, 3):
                    cn = int(round(c + step * dc))
                    rn = int(round(r + step * dr))
                    if 0 <= rn < ny and 0 <= cn < nx:
                        pixel_value = float(image[rn, cn])
                        if (
                            pixel_value >= flux_threshold
                            and pixel_value > best_pixel_value
                        ):
                            best_pixel_value = pixel_value
                            best_r, best_c = rn, cn
            if best_r is None:
                break
            r, c = best_r, best_c
            path_r.append(r)
            path_c.append(c)
            if len(path_r) >= 2:
                angle_rad = np.arctan2(r - path_r[-2], c - path_c[-2])
        return np.array(path_r), np.array(path_c)

    def _make_saturation_streak_mask(
        self,
        image: np.ndarray,
        saturate: float,
        saturate_frac: float = 0.90,
        bleed_half_length: int = 100,
        use_principal_axis: bool = True,
        follow_curved: bool = True,
        streak_flux_frac: float = 0.08,
    ) -> np.ndarray:
        """
        Mask streaks and bleed trails emanating from saturated pixels.

        If follow_curved is True (default): for each saturated blob, follow the
        streak along a path that can curve (e.g. tracking drift), by stepping
        toward the brightest pixel in a narrow cone ahead. Handles non-straight
        trails. If False, use straight-line extension along principal axes.
        """
        sat_core = np.asarray(image >= (saturate_frac * saturate), dtype=bool)
        if not np.any(sat_core):
            return np.zeros_like(image, dtype=bool)
        ny, nx = image.shape
        streak_mask = np.zeros_like(image, dtype=bool)
        half = max(1, int(bleed_half_length))
        # Flux threshold for "still on streak" - lower to catch fainter tails
        sigma = float(np.nanstd(image)) if np.isfinite(np.nanstd(image)) else 0.0
        flux_thresh = float(np.nanmedian(image)) + 1.5 * sigma
        flux_thresh = max(flux_thresh, streak_flux_frac * saturate)
        flux_thresh = min(flux_thresh, 0.35 * saturate)

        struct = np.ones((3, 3), dtype=int)
        labels, n_comp = ndi_label(sat_core, structure=struct)

        for idx in range(1, n_comp + 1):
            blob = labels == idx
            n_pix = np.sum(blob)
            streak_mask |= blob
            if n_pix < 2:
                continue
            rr, cc = np.where(blob)
            cy, cx = np.mean(rr), np.mean(cc)
            r0, c0 = int(round(cy)), int(round(cx))
            dr, dc = rr - cy, cc - cx
            mrr = np.mean(dr * dr)
            mcc = np.mean(dc * dc)
            mrc = np.mean(dr * dc)
            if np.abs(mrc) > 1e-10:
                angle1 = 0.5 * np.arctan2(2 * mrc, mrr - mcc)
            else:
                angle1 = 0.0 if mrr >= mcc else np.pi / 2
            angle2 = angle1 + np.pi / 2

            if follow_curved:
                # Principal axes + diagonals (8 directions) to catch oblique streaks
                angles = [
                    angle1,
                    angle1 + np.pi,
                    angle2,
                    angle2 + np.pi,
                    angle1 + np.pi / 4,
                    angle1 + np.pi * 5 / 4,
                    angle2 + np.pi / 4,
                    angle2 + np.pi * 5 / 4,
                ]
                for angle in angles:
                    pr, pc = self._follow_streak_path(
                        image, r0, c0, angle, half, flux_thresh, cone_deg=38.0
                    )
                    streak_mask[pr, pc] = True
            elif use_principal_axis:
                for angle in (angle1, angle2):
                    lr, lc = self._line_pixels(cy, cx, angle, half, ny, nx)
                    streak_mask[lr, lc] = True
            else:
                # Fallback: row/column only (handled per full image below)
                pass

        if not use_principal_axis and not follow_curved:
            # Row/column extension only
            rows_with_sat = np.any(sat_core, axis=1)
            for r in np.where(rows_with_sat)[0]:
                cols = np.where(sat_core[r, :])[0]
                if len(cols) == 0:
                    continue
                c1 = max(0, int(cols.min()) - half)
                c2 = min(nx, int(cols.max()) + 1 + half)
                streak_mask[r, c1:c2] = True
            cols_with_sat = np.any(sat_core, axis=0)
            for c in np.where(cols_with_sat)[0]:
                rows = np.where(sat_core[:, c])[0]
                if len(rows) == 0:
                    continue
                r1 = max(0, int(rows.min()) - half)
                r2 = min(ny, int(rows.max()) + 1 + half)
                streak_mask[r1:r2, c] = True
            streak_mask |= sat_core
        else:
            streak_mask |= sat_core

        # Widen slightly so streak edges are fully masked
        streak_mask = binary_dilation(streak_mask, structure=struct, iterations=2)
        n_streak = np.sum(streak_mask)
        if n_streak > np.sum(sat_core):
            self.logger.info(
                "Saturation streak/bleed mask: %d px (extended from %d saturated, curved=%s)",
                n_streak,
                int(np.sum(sat_core)),
                follow_curved,
            )
        return streak_mask

    # -------------------------------------------------------------------------
    # Satellite trail mask (long thin ridges without requiring saturation)
    # -------------------------------------------------------------------------
    def _make_satellite_trail_mask(
        self,
        image: np.ndarray,
        n_sigma: float = 3.0,
        min_length_px: int = 40,
        min_aspect_ratio: float = 5.0,
        min_area_px: int = 100,
        dilate_iterations: int = 2,
    ) -> np.ndarray:
        """
        Mask satellite trails: long, thin linear features that need not be saturated.

        Detects connected components above median + n_sigma * sigma (similar in
        spirit to ASTA-style trail masks), then:

        - removes tiny components (area < min_area_px),
        - measures each remaining blob's principal axis to estimate length/width,
        - keeps only long, thin blobs (length >= min_length_px and
          length/width >= min_aspect_ratio),
        - finally applies a small morphological dilation to connect segments.
        """
        img = np.asarray(image, dtype=float)
        ny, nx = img.shape
        med = float(np.nanmedian(img))
        sig = float(np.nanstd(img)) if np.isfinite(np.nanstd(img)) else 0.0
        if sig <= 0:
            return np.zeros_like(img, dtype=bool)
        thresh = med + n_sigma * sig
        bright = (img >= thresh) & np.isfinite(img)
        if not np.any(bright):
            return np.zeros_like(img, dtype=bool)

        struct = np.ones((3, 3), dtype=int)
        labels, n_comp = ndi_label(bright, structure=struct)
        trail_mask = np.zeros_like(img, dtype=bool)
        n_trails = 0
        for idx in range(1, n_comp + 1):
            blob = labels == idx
            n_pix = int(np.sum(blob))
            if n_pix < max(3, min_area_px):
                continue
            rr, cc = np.where(blob)
            cy, cx = np.mean(rr), np.mean(cc)
            dr, dc = rr - cy, cc - cx
            mrr = np.mean(dr * dr)
            mcc = np.mean(dc * dc)
            mrc = np.mean(dr * dc)
            if np.abs(mrc) > 1e-12:
                angle = 0.5 * np.arctan2(2 * mrc, mrr - mcc)
            else:
                angle = 0.0 if mrr >= mcc else np.pi / 2
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            proj_major = dc * cos_a + dr * sin_a
            proj_minor = -dc * sin_a + dr * cos_a
            length = 2.0 * (float(np.max(np.abs(proj_major))) + 0.5)
            width = 2.0 * (float(np.max(np.abs(proj_minor))) + 0.5)
            width = max(width, 1.0)
            if length >= min_length_px and (length / width) >= min_aspect_ratio:
                trail_mask |= blob
                n_trails += 1
        if np.any(trail_mask):
            trail_mask = binary_dilation(
                trail_mask, structure=struct, iterations=dilate_iterations
            )
            self.logger.info(
                "Satellite/trail mask: %d px (%d elongated component(s))",
                int(np.sum(trail_mask)),
                n_trails,
            )
        return trail_mask

    # -------------------------------------------------------------------------
    # Adaptive box resizing
    # -------------------------------------------------------------------------
    def _adaptive_box_for_mask(
        self,
        mask: np.ndarray,
        init_box: tuple,
        min_frac: float = 0.15,
        min_box: int = 32,
    ) -> tuple:
        valid = (~mask).astype(np.float32)
        bx = init_box[0]

        global_unmasked = valid.mean()
        if global_unmasked >= min_frac:
            return (bx | 1, bx | 1)

        while bx >= min_box:
            cov = uniform_filter(valid, size=bx, mode="nearest")
            if np.nanmedian(cov) >= min_frac:
                return (bx | 1, bx | 1)
            bx = int(bx * 0.75)

        return (min_box | 1, min_box | 1)

    # -------------------------------------------------------------------------
    # Automatic field classification (sparse / crowded / nebulous)
    # -------------------------------------------------------------------------
    def _classify_field(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        fwhm_pixels: float | None = None,
    ) -> dict:
        """
        Classify the field regime and derive background parameters.

        The classification is based on:
        - bright_frac      : fraction of pixels > median + 4 * MAD  (crowding)
        - large_scale_ratio: fraction of variance on large spatial scales
                             (diffuse nebulosity / strong gradients)
        """
        finite = np.isfinite(image) & ~mask
        if not np.any(finite):
            # Fallback to conservative defaults.
            return {
                "regime": "unknown",
                "nsigma": 2.5,
                "n_iterations": 3,
                "dilate_factor": 2.0,
                "mesh_scale": 6.0,
                "exclude_percentile": 90.0,
            }

        data = image[finite]
        med = np.nanmedian(data)
        try:
            scatter = mad_std(data)
            if not np.isfinite(scatter) or scatter <= 0:
                raise ValueError
        except Exception:
            scatter = np.nanstd(data)
            if not np.isfinite(scatter) or scatter <= 0:
                scatter = 1.0

        # Use a relatively high threshold so that only genuinely crowded
        # fields (with many bright pixels) are flagged as such. A 4-sigma cut
        # is much less sensitive to mild wings or a few bright stars than 3-sigma.
        bright_thresh = med + 4.0 * scatter
        bright_frac = float(np.mean(data > bright_thresh))

        # Large-scale structure estimator: compare a heavily smoothed image
        # to small-scale residuals.  High large_scale_ratio indicates strong
        # nebulosity / gradients.
        try:
            if fwhm_pixels is not None:
                sigma_smooth = max(3.0 * fwhm_pixels, 20.0)
            else:
                sigma_smooth = max(min(image.shape) / 50.0, 10.0)

            img_finite = np.where(np.isfinite(image), image, med)
            smooth = gaussian_filter(
                img_finite, sigma=float(sigma_smooth), mode="nearest"
            )
            resid = img_finite - smooth

            std_smooth = np.nanstd(smooth[finite])
            std_resid = np.nanstd(resid[finite])
            large_scale_ratio = float(std_smooth / (std_smooth + std_resid + 1e-6))
        except Exception:
            large_scale_ratio = 0.0

        # Heuristic regime classification.
        regime = "sparse"
        # Crowded only if a substantial fraction of all *unmasked* pixels lie
        # above the 4-sigma threshold. This is intended to track the regime
        # where genuine background pixels are scarce (image effectively filled
        # with stellar profiles), not simply "many stars present".
        cfg_bkg = (
            self.config.get("background", {}) if isinstance(self.config, dict) else {}
        )
        fast_mode = bool(cfg_bkg.get("fast_mode", False))
        crowded_bright_min = float(cfg_bkg.get("crowded_bright_frac_min", 0.40))
        if bright_frac >= crowded_bright_min:
            regime = "crowded"
        if large_scale_ratio > 0.4:
            regime = "nebulous" if regime == "sparse" else "crowded_nebulous"

        self.logger.info(
            "Background regime=%s  bright_frac=%.3f  large_scale_ratio=%.3f",
            regime,
            bright_frac,
            large_scale_ratio,
        )

        # Map regime -> tuned parameters. The dilate_factor values control how
        # far masks expand around detected sources; keep them modest so marked
        # regions remain compact in the diagnostics.
        if regime == "sparse":
            params = dict(
                nsigma=3.0,
                n_iterations=2,
                dilate_factor=1.2,
                mesh_scale=10.0,
                exclude_percentile=90.0,
            )
        elif regime == "crowded":
            params = dict(
                nsigma=2.5,
                n_iterations=3,
                dilate_factor=1.5,
                mesh_scale=8.0,
                exclude_percentile=85.0,
            )
        elif regime == "nebulous":
            params = dict(
                nsigma=2.2,
                n_iterations=3,
                dilate_factor=1.6,
                mesh_scale=8.0,
                exclude_percentile=80.0,
            )
        else:  # crowded_nebulous or unknown complex regime
            params = dict(
                nsigma=2.2,
                n_iterations=4,
                dilate_factor=1.8,
                mesh_scale=7.0,
                exclude_percentile=80.0,
            )

        # Fast mode: lighter masking and somewhat coarser mesh for speed.
        if fast_mode:
            self.logger.info(
                "Background: fast_mode=True - using lighter masking and coarser mesh for speed."
            )
            params["n_iterations"] = max(1, min(params["n_iterations"], 2))
            params["dilate_factor"] = min(params["dilate_factor"], 2.0)
            params["mesh_scale"] = max(5.0, params["mesh_scale"] * 0.8)

        params["regime"] = regime
        return params

    # -------------------------------------------------------------------------
    # Background2D estimation
    # -------------------------------------------------------------------------
    def _estimate_background(
        self,
        image: np.ndarray,
        mask,
        box_size: tuple,
        filter_size: int,
        exclude_percentile: float = 90.0,
    ):
        """
        Fit a smooth sky-background surface with photutils Background2D.

        Uses BiweightLocation (robust to outlier boxes) and BkgZoomInterpolator
        (smooth spline).  On failure, retries with progressively larger boxes
        before falling back to a flat global-median surface.
        """
        # Try the requested box size, then progressively larger ones.
        attempts = [
            (box_size, filter_size, mask),
        ]
        # Fallback 1: double the box size (fewer, larger boxes = more robust).
        big = (box_size[0] * 2) | 1
        attempts.append(((big, big), max(3, filter_size), mask))
        # Fallback 2: no mask + large boxes (last resort before flat).
        attempts.append(((big, big), 3, None))

        cfg_bkg = (
            (self.config.get("background", {}) or {})
            if isinstance(self.config, dict)
            else {}
        )
        fast_mode = bool(cfg_bkg.get("fast_mode", False))
        global_interpolator = str(cfg_bkg.get("global_interpolator", "zoom")).strip().lower()
        if fast_mode and global_interpolator == "idw":
            global_interpolator = "zoom"
            self.logger.info(
                "Background: fast_mode=True - using zoom interpolator (IDW is slow on large images)."
            )
        global_interp_order = int(cfg_bkg.get("global_interp_order", 3))
        clip_maxiters = int(cfg_bkg.get("global_sigma_clip_maxiters", 10))
        if fast_mode:
            clip_maxiters = min(clip_maxiters, 5)
        clip_maxiters = max(1, clip_maxiters)
        for i, (bs, fs, m) in enumerate(attempts):
            try:
                if global_interpolator == "idw":
                    interp = BkgIDWInterpolator()
                else:
                    interp = BkgZoomInterpolator(order=max(0, int(global_interp_order)))
                bkg = Background2D(
                    image,
                    mask=m,
                    box_size=bs,
                    filter_size=fs,
                    sigma_clip=SigmaClip(sigma=3.0, maxiters=clip_maxiters),
                    bkg_estimator=BiweightLocationBackground(),
                    bkgrms_estimator=MADStdBackgroundRMS(),
                    interpolator=interp,
                    edge_method="pad",
                    exclude_percentile=float(exclude_percentile),
                )

                bkg_rms = np.asarray(bkg.background_rms, dtype=float)
                rms_median = np.nanmedian(bkg_rms)
                rms_floor = max(rms_median * 0.5, 1e-30)
                bkg_rms = np.clip(
                    np.nan_to_num(
                        bkg_rms, nan=rms_median * 0.1, posinf=1e10, neginf=0.0
                    ),
                    rms_floor,
                    None,
                )

                # Use the raw Background2D background map without any
                # additional median or Gaussian post-processing so that
                # execution time stays minimal.
                bkg_data = np.asarray(bkg.background, dtype=float)

                self.logger.info(
                    "Background2D final: attempt=%d box_size=%s filter_size=%d "
                    "exclude_percentile=%.1f",
                    i + 1,
                    bs,
                    fs,
                    float(exclude_percentile),
                )

                bkg_median = float(
                    np.nanmedian(bkg.background_median)
                    if np.ndim(bkg.background_median)
                    else bkg.background_median
                )

                if i > 0:
                    self.logger.info(
                        f"Background2D succeeded on attempt {i + 1} "
                        f"(box_size={bs}, mask={'yes' if m is not None else 'no'})"
                    )
                return True, bkg_data, bkg_rms, bkg_median

            except Exception as exc:
                self.logger.warning(
                    f"Background2D attempt {i + 1} failed (box={bs}): {exc}"
                )
                continue

        # All attempts failed - flat fallback.
        self.logger.warning("All Background2D attempts failed - using global stats")
        gmean, gmed, gstd = sigma_clipped_stats(image, sigma=3.0, mask=mask)
        bkg_surface = np.full_like(image, gmed, dtype=float)
        bkg_rms = np.full_like(image, max(gstd, 1.0), dtype=float)
        return True, bkg_surface, bkg_rms, gmed

    # -------------------------------------------------------------------------
    # Saturation guard
    # -------------------------------------------------------------------------
    def _check_saturation(self, bkg_median: float, saturate: float) -> float:
        if bkg_median >= 0.95 * saturate:
            self.logger.warning(
                f"Background median [{bkg_median:.3e}] near saturation "
                f"[{saturate:.3e}].  Disabling saturation clipping."
            )
            return 1e12
        return saturate

    # -------------------------------------------------------------------------
    # Figure saving helper
    # -------------------------------------------------------------------------
    def _save_figure(self, fig: plt.Figure, fpath: str, prefix: str) -> None:
        outdir = os.path.dirname(fpath)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        base = os.path.splitext(os.path.basename(fpath))[0]
        prefix_title = "_".join([p[:1].upper() + p[1:] for p in prefix.split("_")])
        save_path_png = os.path.join(outdir, f"{prefix_title}_{base}.png")

        fig.savefig(
            save_path_png, dpi=150, bbox_inches="tight", facecolor="white"
        )
        plt.close(fig)

    # =========================================================================
    # Public API: global background subtraction
    # =========================================================================
    def remove(
        self,
        image: np.ndarray,
        header=None,
        galaxies=None,
        plot: bool = False,
        mask: np.ndarray = None,
        pixel_scale: float = None,
        fwhm: float = None,
        mask_simbad_galaxies: bool = False,
    ) -> dict:
        """
        Estimate and subtract a global sky-background model.

        The method now uses iterative source masking (3 passes), large mesh
        boxes, spline interpolation, and saturation masking to produce a
        genuinely smooth background surface without source/galaxy ghosts.
        """
        if galaxies is None:
            galaxies = []

        self.logger.info(border_msg("Measuring image background statistics"))

        # Treat near-zero values as invalid.
        image = np.where(np.abs(image) < 1e-29, np.nan, image)

        # ---- Build combined mask ----
        if mask is None:
            mask = np.zeros(image.shape, dtype=bool)

        nan_mask = ~np.isfinite(image)
        zero_mask = image == 0.0

        total = image.size
        self.logger.info(
            f"NaN pixels: {np.sum(nan_mask)} ({100.0 * np.sum(nan_mask) / total:.2f}%)"
        )

        # Derive FWHM in pixels (if provided); otherwise allow helper to estimate.
        if fwhm is not None:
            fwhm_pixels = fwhm / pixel_scale if pixel_scale else fwhm
        else:
            fwhm_pixels = None

        # Classify field regime (sparse / crowded / nebulous) and derive
        # dynamic parameters for masking and Background2D. If config marks
        # the image as crowded, use crowded regime so background is crowd-safe.
        regime_params = self._classify_field(
            image,
            mask | nan_mask | zero_mask,
            fwhm_pixels=fwhm_pixels,
        )
        if self.config.get("photometry", {}).get("crowded_field", False):
            regime_params = {
                "regime": "crowded",
                "nsigma": 2.5,
                "n_iterations": 3,
                "dilate_factor": 2.0,
                "mesh_scale": 5.0,
                "exclude_percentile": 85.0,
            }
            self.logger.info(
                "Config crowded_field=True; using crowded background regime."
            )
        nsigma_src = regime_params["nsigma"]
        n_iter_src = regime_params["n_iterations"]
        dilate_factor = regime_params["dilate_factor"]
        mesh_scale = regime_params["mesh_scale"]
        exclude_percentile = regime_params["exclude_percentile"]

        # Compute box and filter sizes using the regime-dependent mesh_scale.
        box_size, filter_size, fwhm_pixels = self._compute_box_sizes(
            image,
            mask,
            fwhm_pixels=fwhm_pixels,
            mesh_scale=mesh_scale,
        )

        # ---- Saturation mask (NEW) ----
        sat_mask = self._make_saturation_mask(
            np.nan_to_num(image, nan=0.0), self.config["saturate"]
        )
        # ---- Saturation streak / bleed mask (avoid regions from saturated stars) ----
        bleed_half = int(self.config.get("saturate_streak_bleed_half_length", 100))
        use_pa = self.config.get("saturate_streak_principal_axis", True)
        follow_curved = self.config.get("saturate_streak_follow_curved", True)
        streak_flux_frac = float(self.config.get("saturate_streak_flux_frac", 0.08))
        if bleed_half > 0:
            streak_mask = self._make_saturation_streak_mask(
                np.nan_to_num(image, nan=0.0),
                self.config["saturate"],
                saturate_frac=0.90,
                bleed_half_length=bleed_half,
                use_principal_axis=use_pa,
                follow_curved=follow_curved,
                streak_flux_frac=streak_flux_frac,
            )
        else:
            streak_mask = np.zeros_like(image, dtype=bool)

        # ---- Satellite trail mask (long thin ridges, no saturation required) ----
        trail_mask = np.zeros_like(image, dtype=bool)
        if self.config.get("mask_satellite_trails", True):
            trail_n_sigma = float(self.config.get("satellite_trail_n_sigma", 3.0))
            trail_min_length = int(self.config.get("satellite_trail_min_length_px", 40))
            trail_min_aspect = float(
                self.config.get("satellite_trail_min_aspect_ratio", 5.0)
            )
            trail_min_area = int(self.config.get("satellite_trail_min_area_px", 100))
            trail_mask = self._make_satellite_trail_mask(
                np.nan_to_num(image, nan=0.0),
                n_sigma=trail_n_sigma,
                min_length_px=trail_min_length,
                min_aspect_ratio=trail_min_aspect,
                min_area_px=trail_min_area,
                dilate_iterations=2,
            )

        # ---- Iterative source mask ----
        source_mask = self._make_source_mask(
            image,
            nsigma=nsigma_src,
            npixels=5,
            fwhm_pixels=fwhm_pixels if fwhm_pixels is not None else 3.0,
            dilate_factor=dilate_factor,
            n_iterations=n_iter_src,
        )

        # ---- SIMBAD galaxy mask (with expanded ellipses) ----
        galaxy_mask = np.zeros(image.shape, dtype=bool)
        if mask_simbad_galaxies and len(galaxies) > 0:
            galaxy_mask = self.masked_simbad_galaxies(
                galaxies,
                image,
                header,
                scale_factor=2.0,
            )

        # Union of all mask layers.
        mask = (
            mask
            | nan_mask
            | source_mask
            | galaxy_mask
            | zero_mask
            | sat_mask
            | streak_mask
            | trail_mask
        )

        masked_frac = mask.mean()
        self.logger.info(f"Masked fraction (total): {masked_frac:.2%}")

        # If essentially the entire image is saturated or masked, this frame is
        # not usable for reliable background / photometry. Bail out early with
        # a clear error instead of letting downstream steps fail in obscure ways.
        sat_frac = float(np.mean(sat_mask)) if sat_mask is not None else 0.0
        if sat_frac > 0.95:
            self.logger.error(
                "Image appears fully saturated (saturation mask covers %.2f%% of pixels); "
                "background estimation and photometry are not reliable. "
                "Aborting background subtraction for this frame.",
                100.0 * sat_frac,
            )
            raise RuntimeError(
                "BackgroundSubtractor: image is effectively fully saturated; cannot "
                "estimate a meaningful sky background."
            )

        # ---- Adaptive box size for heavily masked images ----
        if masked_frac > 0.30:
            new_box = self._adaptive_box_for_mask(mask, box_size)
            if new_box != box_size:
                self.logger.info(f"Adjusted box_size {box_size} -> {new_box}")
                box_size = new_box

        self.logger.info(f"box_size={box_size}  filter_size={filter_size}")

        # ---- Global fallback statistics ----
        gmean, gmed, gstd = sigma_clipped_stats(
            image, sigma=3.0, mask=mask, cenfunc=np.nanmedian, stdfunc=mad_std
        )
        self.logger.info(
            f"Global stats - mean {gmean:.3e}  median {gmed:.3e}  std {gstd:.3e}"
        )

        # ---- Background estimation (with built-in retry chain) ----
        success, bkg_surface, bkg_rms, bkg_median = self._estimate_background(
            image,
            mask,
            box_size,
            filter_size,
            exclude_percentile=exclude_percentile,
        )

        # Optional: damp Background2D interpolation overshoot near masked regions.
        # This reduces "bowls/rings" around bright sources/galaxy masks caused by
        # spline/zoom interpolation across large masked holes.
        try:
            cfg_bkg = (self.config.get("background", {}) or {}) if isinstance(self.config, dict) else {}
            if bool(cfg_bkg.get("global_mask_edge_flatten", False)) and mask is not None:
                # Radius (px) around masks to replace with a smoothed estimate.
                raw_r_flat = cfg_bkg.get("global_mask_edge_flatten_radius_px", None)
                r_flat = float(raw_r_flat) if raw_r_flat is not None else 0.0
                if not np.isfinite(r_flat) or r_flat <= 0:
                    # Default: ~1 FWHM (>=2 px).
                    r_flat = max(2.0, float(fwhm_pixels) if fwhm_pixels is not None else 3.0)

                m = np.asarray(mask, dtype=bool)
                bkg = np.asarray(bkg_surface, dtype=float)

                # Distance-to-mask band without an expensive distance transform:
                # just grow the mask by r_flat pixels and correct only the ring.
                grow = binary_dilation(m, structure=_disk_structuring_element(int(np.ceil(r_flat))))
                band = grow & (~m)

                # Smooth background using only unmasked pixels.
                bkg_u = bkg.copy()
                bkg_u[m] = np.nan
                w = np.isfinite(bkg_u).astype(float)
                # Gaussian smoothing + normalization by weights.
                raw_sig = cfg_bkg.get("global_mask_edge_flatten_sigma_px", None)
                sig = float(raw_sig) if raw_sig is not None else (float(r_flat) / 2.0)
                sig = max(0.8, float(sig))
                num = gaussian_filter(np.nan_to_num(bkg_u, nan=0.0), sigma=sig, mode="nearest")
                den = gaussian_filter(w, sigma=sig, mode="nearest")
                smooth = num / np.maximum(den, 1e-12)

                if np.any(band) and np.any(np.isfinite(smooth[band])):
                    bkg = bkg.copy()
                    bkg[band] = smooth[band]
                    bkg_surface = bkg
        except Exception:
            pass

        self.logger.info(
            f"Background median {bkg_median:.3e}  RMS mean {np.nanmean(bkg_rms):.3e}"
        )

        # ---- Subtract background ----
        sub = image - bkg_surface
        self.config["saturate"] = self._check_saturation(
            bkg_median, self.config["saturate"]
        )

        if plot:
            self._plot_diagnostics(
                image, bkg_surface, sub, bkg_rms, self.config["fpath"], mask
            )

        # Defects for downstream: NaN, saturation cores, and saturation streaks/bleed.
        defects_mask = nan_mask | sat_mask | streak_mask | trail_mask
        return {
            "image": sub,
            "background": bkg_surface,
            "background_rms": bkg_rms,
            "defects_mask": defects_mask,
            "source_mask": source_mask,
        }

    # =========================================================================
    # Public API: local background subtraction around a target
    # =========================================================================
    def remove_local_surface(
        self,
        image: np.ndarray,
        x0: float,
        y0: float,
        box_half_size: int = 100,
        fwhm_pixels: float = None,
        exclude_inner_radius: float = None,
        dilate_factor: float = 1,  # balanced: masks wings without over-masking
        plot: bool = True,
    ):
        """
        Fit and subtract a local 2-D background surface around a target.

        Uses iterative masking and spline interpolation for a smooth result.
        """
        ny, nx = image.shape
        x0, y0 = int(np.round(x0)), int(np.round(y0))

        if fwhm_pixels is None:
            fwhm_pixels = 5.0

        if not (0 <= x0 < nx and 0 <= y0 < ny):
            raise ValueError(
                f"Target coordinates ({x0}, {y0}) outside image ({nx}x{ny})."
            )
        if box_half_size <= 0:
            raise ValueError("box_half_size must be positive.")

        # ---- Extract cutout ----
        x_min = max(0, x0 - box_half_size)
        x_max = min(nx, x0 + box_half_size)
        y_min = max(0, y0 - box_half_size)
        y_max = min(ny, y0 + box_half_size)
        cutout = image[y_min:y_max, x_min:x_max]

        # ---- Local config overrides (optional) ----
        # These live under the `background:` block in the YAML.
        bcfg = (
            (self.config.get("background", {}) or {})
            if isinstance(self.config, dict)
            else {}
        )
        # Default to a finer mesh for local transient fits so the cutout background
        # is as flat as possible for PSF measurement.
        local_mesh_scale = float(bcfg.get("local_mesh_scale", 0.8))
        local_region_fraction_limit = float(
            bcfg.get("local_region_fraction_limit", 0.15)
        )
        local_min_box = int(bcfg.get("local_min_box", 6))
        local_nsigma = float(bcfg.get("local_source_mask_nsigma", 5.0))
        local_npixels = int(bcfg.get("local_source_mask_npixels", 7))
        local_mask_iterations = int(bcfg.get("local_source_mask_iterations", 2))
        local_exclude_percentile = float(bcfg.get("local_exclude_percentile", 95.0))
        local_sigma = float(bcfg.get("local_sigma_clip", 3.0))
        local_sigma_maxiters = int(bcfg.get("local_sigma_clip_maxiters", 10))
        local_filter_size_max = bcfg.get("local_filter_size_max", 3)
        local_interpolator = str(bcfg.get("local_interpolator", "zoom")).strip().lower()
        local_interp_order = int(bcfg.get("local_interp_order", 1))
        # Allow background mesh override directly (finest control).
        local_box_size_override = bcfg.get("local_box_size", None)
        local_filter_size_override = bcfg.get("local_filter_size", None)

        # Inner exclusion radius: ensure the target core/PSF is never used for
        # background estimation. Default to 2.5*FWHM when not provided.
        if exclude_inner_radius is None:
            exclude_inner_radius = 2.5 * float(fwhm_pixels)
        exclude_inner_radius = (
            float(exclude_inner_radius) if exclude_inner_radius is not None else 0.0
        )

        # ---- Box/filter sizes for the cutout ----
        # Use a smaller mesh_scale for local fits so the background model can
        # follow oversubtracted / galaxy gradients around the target more
        # closely, while still avoiding pixel-scale noise.
        box_size, filter_size, _ = self._compute_box_sizes(
            cutout,
            mask=None,
            fwhm_pixels=fwhm_pixels,
            mesh_scale=local_mesh_scale,
            region_fraction_limit=local_region_fraction_limit,
            min_box=local_min_box,
        )
        fixed_local_box = False
        if local_box_size_override is not None:
            try:
                box_size = int(local_box_size_override)
                fixed_local_box = True
            except Exception:
                pass
        if local_filter_size_override is not None:
            try:
                filter_size = int(local_filter_size_override)
            except Exception:
                pass
        else:
            # Cap smoothing to avoid an overly blurred local surface.
            try:
                if local_filter_size_max is not None:
                    filter_size = min(int(filter_size), int(local_filter_size_max))
            except Exception:
                pass
        # Background2D expects odd filter_size >= 1
        try:
            filter_size = max(1, int(filter_size))
            if filter_size % 2 == 0:
                filter_size += 1
        except Exception:
            filter_size = 1
        self.logger.info(
            "Local background: box_size=%s  filter_size=%s  mesh_scale=%.3g  exclude_inner_radius=%.3g px",
            str(box_size),
            str(filter_size),
            float(local_mesh_scale),
            float(exclude_inner_radius),
        )

        # ---- Source mask in cutout (iterative) ----
        # Use a higher nsigma and fewer iterations here so the local mask is
        # conservative, focusing on obvious compact sources and not broad
        # background gradients.
        source_mask = self._make_source_mask(
            cutout,
            nsigma=local_nsigma,
            npixels=local_npixels,
            fwhm_pixels=fwhm_pixels,
            dilate_factor=dilate_factor,
            n_iterations=local_mask_iterations,
        )

        # Mask the target core region explicitly so the background estimator never
        # tries to model it (prevents "holes"/oversubtraction around transients).
        if exclude_inner_radius and exclude_inner_radius > 0:
            cy = float(y0 - y_min)
            cx = float(x0 - x_min)
            yy, xx = np.mgrid[0 : cutout.shape[0], 0 : cutout.shape[1]]
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            source_mask |= r2 <= float(exclude_inner_radius) ** 2

        # ---- Adaptive box size ----
        if fixed_local_box:
            self.logger.info(
                "Local background: using fixed box_size=%s (no adaptive up-sizing).",
                str(box_size),
            )
        else:
            masked_frac = source_mask.mean()
            if masked_frac > 0.30:
                new_box = self._adaptive_box_for_mask(source_mask, box_size, min_box=16)
                if new_box != box_size:
                    self.logger.info(f"Adjusted local box_size {box_size} -> {new_box}")
                    box_size = new_box

        # ---- Fit background on cutout ----
        try:
            if local_interpolator == "idw":
                interp = BkgIDWInterpolator()
            else:
                # Zoom interpolator with low order keeps the surface less smoothed than the default cubic.
                interp = BkgZoomInterpolator(order=max(0, int(local_interp_order)))
            bkg = Background2D(
                cutout,
                box_size=box_size,
                filter_size=filter_size,
                sigma_clip=SigmaClip(sigma=local_sigma, maxiters=local_sigma_maxiters),
                bkg_estimator=BiweightLocationBackground(),
                bkgrms_estimator=MADStdBackgroundRMS(),
                interpolator=interp,
                mask=source_mask,
                exclude_percentile=local_exclude_percentile,
                edge_method="pad",
            )
            bkg_surface_local = bkg.background
        except Exception as exc:
            self.logger.warning(
                f"Background2D failed on cutout: {exc} - using median fallback"
            )
            _, local_med, _ = sigma_clipped_stats(cutout, sigma=3.0, mask=source_mask)
            bkg_surface_local = np.full_like(cutout, local_med, dtype=float)

        # ---- Guard against negative "bowls" around the target ----
        # Even with the core masked, interpolation can overshoot near the masked
        # region and produce an artificial negative bowl/ring after subtraction.
        # To stabilize targeted photometry, flatten the background model in the
        # central region to match a robust local ring level outside the exclusion.
        if exclude_inner_radius and exclude_inner_radius > 0:
            try:
                cy = float(y0 - y_min)
                cx = float(x0 - x_min)
                yy, xx = np.mgrid[0 : cutout.shape[0], 0 : cutout.shape[1]]
                rr = np.hypot(xx - cx, yy - cy)
                rin = float(exclude_inner_radius)
                ring_width = max(2.0, float(fwhm_pixels))

                # Use a reference ring just outside the exclusion to estimate
                # the local background level, but sample it slightly farther out
                # to avoid the steepest interpolation curvature right at the edge.
                ring_ref = (rr > (rin + ring_width)) & (rr <= (rin + 2.0 * ring_width))
                ring_ref &= ~source_mask
                ring_ref &= np.isfinite(bkg_surface_local) & np.isfinite(cutout)

                # Flatten the model inside this radius. This is where "bowls"
                # are visually obvious and most damaging for PSF/aperture fits.
                r_flat = rin + ring_width
                flat_region = rr <= float(r_flat)

                if np.any(flat_region) and np.any(ring_ref):
                    ring_level = float(np.nanmedian(bkg_surface_local[ring_ref]))
                    bkg_surface_local = np.asarray(bkg_surface_local, dtype=float)
                    bkg_surface_local[flat_region] = ring_level
            except Exception:
                pass

        # ---- Subtract locally and insert back ----
        corrected_cutout = cutout - bkg_surface_local
        image_sub = image.copy()
        image_sub[y_min:y_max, x_min:x_max] = corrected_cutout

        # ---- Full-image RMS for downstream use ----
        full_mask = self._make_source_mask(
            image,
            nsigma=3,
            npixels=5,
            fwhm_pixels=fwhm_pixels,
            dilate_factor=1.5,
            n_iterations=2,
        )
        box_size_full, filter_size_full, _ = self._compute_box_sizes(
            image, full_mask, fwhm_pixels
        )
        _, _, bkg_rms_full, _ = self._estimate_background(
            image, full_mask, box_size_full, filter_size_full
        )

        if plot:
            # For plotting, show only the conservative source mask (target
            # region is not specially excluded).
            self._plot_local_diagnostics(
                cutout,
                bkg_surface_local,
                corrected_cutout,
                self.config["fpath"],
                source_mask,
            )

        return image_sub, bkg_surface_local, bkg_rms_full

    # =========================================================================
    # Diagnostic plots
    # =========================================================================
    def _plot_diagnostics(
        self,
        image,
        background,
        subtracted,
        rms,
        fpath,
        mask=None,
    ) -> None:
        arrays = [image, background, rms, subtracted]
        titles = ["Science", "Background", "Noise RMS", "Subtracted"]
        interval = ZScaleInterval()

        fig, axes = plt.subplots(1, 4, figsize=set_size(540, 1))
        axes = np.atleast_1d(axes)

        for i, (ax, data, title) in enumerate(zip(axes, arrays, titles)):
            vmin, vmax = self._safe_zlimits(data, interval)
            disp = np.where(np.isfinite(data), data, vmin)

            im = ax.imshow(
                disp,
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="none",
            )
            self._attach_colorbar(fig, ax, im, title)
            ax.set_aspect("equal")

            if i > 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            # Leftmost subplot: overlay masked pixels as a second image:
            # an RGBA array where masked pixels are semi-transparent red and
            # unmasked pixels are fully transparent.
            if (
                i == 0
                and mask is not None
                and np.any(mask)
                and mask.shape == data.shape
            ):
                try:
                    ny, nx = mask.shape
                    overlay = np.zeros((ny, nx, 4), dtype=float)
                    overlay[..., 0] = mask.astype(float)  # red channel = 1 where masked
                    overlay[..., 3] = (
                        mask.astype(float) * 0.6
                    )  # alpha = 0.6 where masked

                    ax.imshow(
                        overlay,
                        origin="lower",
                        interpolation="nearest",
                        zorder=10,
                    )
                except Exception:
                    pass

        plt.tight_layout()
        self._save_figure(fig, fpath, "background")

    def _plot_local_diagnostics(
        self,
        cutout,
        background,
        subtracted,
        fpath,
        mask=None,
    ) -> None:
        arrays = [cutout, background, subtracted]
        titles = ["Science", "Background", "Subtracted"]
        interval = ZScaleInterval()

        fig, axes = plt.subplots(1, 3, figsize=set_size(540, 1))
        axes = np.atleast_1d(axes)

        for i, (ax, data, title) in enumerate(zip(axes, arrays, titles)):
            vmin, vmax = self._safe_zlimits(data, interval)
            disp = np.where(np.isfinite(data), data, vmin)

            im = ax.imshow(
                disp,
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="none",
            )
            self._attach_colorbar(fig, ax, im, title)
            ax.set_aspect("equal")

            if i == 0 and mask is not None and np.any(mask):
                try:
                    ny, nx = mask.shape
                    overlay = np.zeros((ny, nx, 4), dtype=float)
                    overlay[..., 0] = mask.astype(float)
                    overlay[..., 3] = mask.astype(float) * 0.6

                    ax.imshow(
                        overlay,
                        origin="lower",
                        interpolation="nearest",
                    )
                except Exception:
                    pass

        plt.tight_layout()
        self._save_figure(fig, fpath, "local_background")

    # -------------------------------------------------------------------------
    # Private plot helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_zlimits(data, interval):
        try:
            return interval.get_limits(data)
        except Exception:
            finite = data[np.isfinite(data)]
            if finite.size:
                return np.nanpercentile(finite, [5, 95])
            return 0.0, 1.0

    @staticmethod
    def _attach_colorbar(fig, ax, im, label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.set_label(str(label))
