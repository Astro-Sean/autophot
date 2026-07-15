#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized cosmic ray removal using astroscrappy or ccdproc.cosmicray_lacosmic with robust background handling,
side-by-side visualization, and improved mask dilation logic with hole filling.
Author: Sean Brennan
"""

# --- Standard Library Imports ---
import logging
import os
import warnings

# --- Third-Party Imports ---
import numpy as np
import astroscrappy
from ccdproc import cosmicray_lacosmic
from typing import Optional, Tuple
from scipy.ndimage import binary_dilation, binary_fill_holes
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from skimage.morphology import disk

from functions import log_warning_from_exception


# --- Main Class ---
class RemoveCosmicRays:
    """
    A class to handle cosmic ray removal from astronomical images using astroscrappy or ccdproc.cosmicray_lacosmic.
    Includes background estimation, masking, visualization, and improved mask dilation with hole filling.
    """

    def __init__(
        self,
        input_yaml: str,
        fpath: str,
        image: np.ndarray,
        header: dict,
        use_lacosmic: bool = False,
    ):
        """
        Initialize the cosmic ray removal process.

        Args:
            input_yaml: Configuration dictionary.
            fpath: Path to the FITS file.
            image: Image data as a numpy array.
            header: FITS header as a dictionary.
            use_lacosmic: Flag to use L.A.Cosmic algorithm (default: False).
        """
        self.input_yaml = input_yaml
        self.fpath = fpath
        self.image = image
        self.header = header
        self.use_lacosmic = use_lacosmic
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # --- Mask Creation ---
    def _create_mask(self, image: np.ndarray, satlevel: float = np.inf) -> np.ndarray:
        """
        Create a mask for saturated and bright pixels in the image.

        Args:
            image: Input image as a numpy array.
            satlevel: Saturation level (default: infinity).

        Returns:
            Boolean mask as a numpy array.
        """
        sat_mask = image > satlevel
        # Use robust MAD-based sigma instead of np.std — cosmic rays and
        # bright stars inflate std, making the mask ineffective at catching
        # the very pixels it should protect.
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return sat_mask
        med = np.median(finite)
        mad = np.median(np.abs(finite - med))
        robust_sigma = 1.4826 * mad if mad > 0 else np.std(finite)
        bright_mask = image > (med + 5 * robust_sigma)
        return sat_mask | bright_mask

    # --- Cosmic Ray Mask Dilation and Hole Filling ---
    def dilate_cosmic_ray_mask(
        self,
        cr_mask: np.ndarray,
        fwhm_pixels: float,
        dilate_factor: float = 1.0,
        iterations: int = 2,
        fill_holes: bool = True,
    ) -> np.ndarray:
        """
        Dilate a cosmic ray mask using a circular kernel and optionally fill enclosed holes.

        Args:
            cr_mask: Boolean mask of cosmic ray pixels.
            fwhm_pixels: FWHM of the PSF in pixels.
            dilate_factor: Scaling factor for the dilation radius (default: 1.0).
            iterations: Number of dilation iterations (default: 2).
            fill_holes: If True, fill enclosed holes after dilation (default: True).

        Returns:
            Processed boolean mask as a numpy array.
        """
        try:
            # Calculate the dilation radius
            r = max(1, int(dilate_factor * fwhm_pixels))

            # Use skimage's optimised disk structuring element
            selem = disk(r)

            # Dilate the mask
            dilated_mask = binary_dilation(
                cr_mask, structure=selem, iterations=iterations
            )

            # Fill enclosed holes if requested
            if fill_holes:
                dilated_mask = binary_fill_holes(dilated_mask)

            return dilated_mask

        except Exception as e:
            log_warning_from_exception(
                self.logger, "Cosmic ray mask dilation failed", e
            )
            return cr_mask  # Return original mask on failure

    # --- Visualization ---
    def plot_comparison(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        cr_mask: np.ndarray,
        title: str = "Cosmic Ray Removal",
        dilation_size: int = 2,
    ):
        """
        Plot a side-by-side comparison of the original and cleaned images.
        The left panel shows the original image with cosmic-ray-contaminated
        pixels highlighted in red; the right panel shows the cleaned image.

        Args:
            original: Original image (before cosmic ray removal).
            cleaned: Cleaned image (after cosmic ray removal).
            cr_mask: Boolean mask of cosmic ray pixels.
            title: Title for the plot (default: "Cosmic Ray Removal").
            dilation_size: Size of dilation for marking cosmic ray regions (default: 2).
        """
        fpath = self.input_yaml.get("fpath", "")
        image_filter = self.input_yaml.get("imageFilter", "")
        if not fpath or not image_filter:
            raise ValueError("Missing 'fpath' or 'imageFilter' in input_yaml.")
        base = os.path.splitext(os.path.basename(fpath))[0]
        write_dir = os.path.dirname(fpath)

        from functions import set_size

        fig, axes = plt.subplots(
            1, 2, figsize=set_size(540, aspect=0.65)
        )

        # --- Apply zscale for optimal contrast ---
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(original)

        # --- Left panel: original with CR pixels in red ---
        # Mask CR pixels so they show as the "bad" colour (red) in the colormap
        original_masked = np.ma.array(original, mask=cr_mask)
        cmap_orig = plt.get_cmap("gray").copy()
        cmap_orig.set_bad(color="red")
        im0 = axes[0].imshow(
            original_masked,
            cmap=cmap_orig,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        n_cr = int(np.count_nonzero(cr_mask))
        axes[0].set_title(
            f"Original ({n_cr:,} CR pixels)", fontsize=9
        )
        fig.colorbar(im0, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04)

        # --- Right panel: cleaned image ---
        cmap_clean = plt.get_cmap("gray").copy()
        cmap_clean.set_bad(color="white")
        im1 = axes[1].imshow(
            cleaned,
            cmap=cmap_clean,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title("Cleaned", fontsize=9)
        fig.colorbar(im1, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)

        # --- Finalize ---
        fig.suptitle(title, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        png_path = os.path.join(write_dir, f"Cosmic_Rays_{base}.png")
        fig.savefig(
            png_path, bbox_inches="tight", dpi=150, facecolor="white"
        )
        plt.close(fig)

    # --- Cosmic Ray Removal ---
    def remove(
        self,
        bkg: Optional[np.ndarray] = None,
        bkg_rms: Optional[np.ndarray] = None,
        invar: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        satlevel: float = np.inf,
        gain: float = 1.0,
        readnoise: Optional[float] = None,
        psf_fwhm: float = 3,
        sigclip: float = 4.5,
        sigfrac: float = 0.3,
        objlim: float = 10.0,
        dilate_factor: float = 1.0,
        dilate_iterations: int = 2,
        plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove cosmic rays using astroscrappy or ccdproc.cosmicray_lacosmic with robust background and masking.

        Args:
            bkg: Background image to subtract (same shape as self.image).
            bkg_rms: Background RMS (same shape as self.image).
            invar: Variance map (noise squared, same shape as self.image); in counts^2.
            mask: Input mask (same shape as self.image).
            satlevel: Saturation level (default: inf).
            gain: Detector gain (default: 1.0).
            readnoise: Read noise (default: from header or 6.5).
            psf_fwhm: PSF FWHM in pixels (default: 3.0).
            sigclip: Detection threshold for cosmic rays (default: 7.0).
            sigfrac: Fraction of pixels used for detection (default: 0.3).
            objlim: Object detection threshold (default: 10.0).
            plot: If True, plot a side-by-side comparison (default: True).

        Returns:
            Tuple of (cleaned image, processed cosmic ray mask) as numpy arrays.
        """
        self.logger.info("Starting cosmic ray removal")

        # Preserve NaNs (chip gaps) before cosmic ray removal
        nan_mask = np.isnan(self.image)

        # --- Convert image to float32 ---
        self.image = self.image.astype(np.float32)

        # --- Convert optional arrays to float32 ---
        if invar is not None:
            invar = invar.astype(np.float32)
        if bkg is not None:
            bkg = bkg.astype(np.float32)
        if bkg_rms is not None:
            bkg_rms = bkg_rms.astype(np.float32)

        # --- Input Validation ---
        if not isinstance(self.image, np.ndarray):
            self.logger.error("Input image must be a numpy array")
            return self.image, np.zeros_like(self.image, dtype=bool)

        # --- Read Noise Handling ---
        if readnoise is None:
            readnoise = float(self.header.get("RDNOISE", 6.5))
        if readnoise < 0.0:
            readnoise = 0.0

        # --- Masking ---
        if mask is None:
            mask = self._create_mask(self.image, satlevel)
            self.logger.info("Created mask for saturated and bright pixels.")
        else:
            if mask.shape != self.image.shape:
                self.logger.warning("Mask shape doesn't match image. Ignoring mask.")
                mask = self._create_mask(self.image, satlevel)

        # --- Variance map (astroscrappy/ccdproc expect variance = sigma^2 in counts^2) ---
        if invar is None:
            from photutils.utils import calc_total_error

            if bkg_rms is None:
                bkg_rms = np.zeros_like(self.image, dtype=np.float32)
            sigma = calc_total_error(self.image, bkg_rms, effective_gain=gain)
            invar = np.asarray(sigma, dtype=np.float32) ** 2
            self.logger.info("Computed variance map from total error.")

        # --- PSF Size Calculation ---
        psf_size = int(np.ceil(3 * psf_fwhm))
        if psf_size % 2 == 0:
            psf_size += 1  # Ensure it's odd
        self.logger.info(f"Using PSF size: {psf_size} (FWHM: {psf_fwhm:.1f} pixels)")

        # --- Run cosmic ray removal ---
        try:
            if self.use_lacosmic:
                self.logger.info(
                    "Using ccdproc.cosmicray_lacosmic for cosmic ray removal"
                )
                clean_image, cr_mask = cosmicray_lacosmic(
                    self.image,
                    sigclip=sigclip,
                    sigfrac=sigfrac,
                    objlim=objlim,
                    gain=gain,
                    readnoise=readnoise,
                    satlevel=satlevel,
                    pssl=0.0,
                    niter=4,
                    sepmed=True,
                    cleantype="meanmask",
                    fsmode="median",
                    psfmodel="gauss",
                    psffwhm=psf_fwhm,
                    psfsize=psf_size,
                    psfk=None,
                    psfbeta=4.765,
                    verbose=False,
                    gain_apply=False,
                    inbkg=bkg,
                    invar=invar,
                )
            else:
                self.logger.info("Using astroscrappy for cosmic ray removal")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cr_mask, clean_image = astroscrappy.detect_cosmics(
                        self.image,
                        gain=gain,
                        inbkg=bkg,
                        invar=invar,
                        readnoise=readnoise,
                        satlevel=satlevel,
                        sepmed=True,
                        cleantype="medmask",
                        fsmode="median",
                        psfmodel="gauss",
                        psffwhm=psf_fwhm,
                        sigclip=sigclip,
                        sigfrac=sigfrac,
                        objlim=objlim,
                        verbose=False,
                    )
                # cleanarr is returned in same units as input (counts/ADU); do not divide by gain

            # --- Post-processing: Dilate and fill holes in the cosmic ray mask ---
            processed_mask = self.dilate_cosmic_ray_mask(
                cr_mask,
                fwhm_pixels=psf_fwhm,
                dilate_factor=dilate_factor,
                iterations=dilate_iterations,
                fill_holes=True,
            )

            # --- Log Results ---
            n_cr = np.count_nonzero(processed_mask)
            total_pixels = self.image.size
            cr_fraction = n_cr / total_pixels

            if cr_fraction > 0.1:
                self.logger.warning(
                    f"High cosmic ray fraction: {cr_fraction:.2%}. Check parameters."
                )
            self.logger.info(
                f"Removed {n_cr:,} contaminated cosmic ray pixels ({cr_fraction:.2%} of image)"
            )

            # --- Plot Comparison ---
            if plot:
                try:
                    self.plot_comparison(self.image, clean_image, processed_mask)
                except Exception as plot_err:
                    self.logger.warning("CR comparison plot failed: %s", plot_err)

            # --- Update Header ---
            method = (
                "ccdproc.cosmicray_lacosmic" if self.use_lacosmic else "astroscrappy"
            )
            self.header["CRAY_RMD"] = (True, "Cosmic rays removed; skip CR step on rerun")
            self.header.add_history(f"Cosmic ray removal: {n_cr} pixels cleaned")
            self.header["CRPIXELS"] = (n_cr, "Cosmic ray pixels removed")
            self.header["CRMETHOD"] = (method, "Cosmic ray removal method")
            self.header["CRSTATUS"] = ("success", "Cosmic ray removal successful")
            self.header["CRPARAMS"] = (
                f"sigclip={sigclip}, sigfrac={sigfrac}, psffwhm={psf_fwhm:.2f}, "
                f"objlim={objlim}, readnoise={readnoise}, gain={gain}",
                "Parameters used for cosmic ray removal",
            )

            # --- Return the cleaned image and processed mask ---
            # Restore NaNs (chip gaps) after cosmic ray removal
            clean_image[nan_mask] = np.nan
            return clean_image, processed_mask

        # --- Error Handling ---
        except Exception as e:
            self.logger.error("Cosmic ray removal failed. Returning original image.")
            self.logger.exception(e)
            self.header["CRSTATUS"] = ("failed", "Cosmic ray removal failed")
            return self.image, np.zeros_like(self.image, dtype=bool)
