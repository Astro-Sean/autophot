"""
Drop-in module: ImageDistortionCorrector
Purpose:
  - Build robust SExtractor catalogs (optional PSFEx)
  - Cross-match science/reference sources into 1<->1 tables, using extended sources as alignment anchors
  - Align via SCAMP+SWarp, or Reproject (WCS), then AstroAlign on failure
  - Refine WCS with SCAMP (.head handling fixed)
  - Optimize convolution kernel if FWHM is available in the header
Key guarantees:
  - After filter_matched_sources(): same length catalogs, row i in science <-> row i in reference, shared MATCH_ID
External tools required on PATH:
  - sex, psfex (optional), scamp, swarp
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List, Optional, Tuple, Union

import astroalign as aa
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.visualization import ZScaleInterval
import astropy.wcs as WCS
from matplotlib.patches import Circle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import minimize
import astropy.units as u

sys.path.append(str(Path(__file__).parent.parent))
from functions import remove_wcs_from_header, log_warning_from_exception
from wcs import get_wcs

try:
    from reproject import reproject_interp

    HAS_REPROJECT = True
except ImportError:
    HAS_REPROJECT = False


class ImageDistortionCorrector:
    """
    Process astronomical images with SExtractor, SCAMP, SWarp, and AstroAlign,
    using extended sources as alignment anchors and optimizing the convolution kernel if FWHM is available.
    """

    # --------------------------- Default tool configs ---------------------------
    DEFAULT_SEX_CONFIG = {
        "CATALOG_TYPE": "FITS_LDAC",
        "DETECT_THRESH": 1.5,
        "ANALYSIS_THRESH": 1.2,
        "DETECT_MINAREA": 3,
        "BACK_SIZE": 32,
        "DEBLEND_NTHRESH": 64,
        "BACK_TYPE": "MANUAL",
        "DEBLEND_MINCONT": 0.001,
        "BACK_FILTERSIZE": 5,
        "FILTER": "Y",
        "CLEAN": "Y",
        "CLEAN_PARAM": 1,
        "PHOT_APERTURES": 10,
        "VERBOSE_TYPE": "QUIET",
    }

    # Maximum FWHM (pixels) for sources used in alignment; sources with FWHM > this are excluded
    ALIGNMENT_MAX_FWHM_PIX = 50.0

    # Crowded-field overrides: maximize source detection (tighter deblending, smaller back mesh, lower thresholds)
    CROWDED_SEX_CONFIG = {
        "DEBLEND_MINCONT": 0.0001,
        "DEBLEND_NTHRESH": 128,
        "BACK_SIZE": 16,
        "BACK_FILTERSIZE": 3,
        "DETECT_MINAREA": 2,
        "DETECT_THRESH": 1.2,
        "ANALYSIS_THRESH": 1.0,
    }

    DEFAULT_SCAMP_CONFIG = {
        "SOLVE_ASTROM": "Y",
        "SOLVE_PHOTOM": "Y",
        "REF_TIMEOUT": 60,
        "REF_SERVER": "vizier.cfa.harvard.edu",
        "DISTORT_DEGREES": None,  # Will be set from config
        "MATCH": "Y",
        "MATCH_RESOL": 0,
        "MATCH_FLIPPED": "Y",
        "WRITE_XML": "Y",
        "VERBOSE_TYPE": "LOG",
        "ASTREF_WEIGHT": 1,
        "ASTREFMAG_KEY": "MAG_AUTO",
        "ASTREFMAGERR_KEY": "MAGERR_AUTO",
        "ASTREFCENT_KEYS": "ALPHA_J2000,DELTA_J2000",
        "ASTREFERR_KEYS": "ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD",
        "DISTORT_KEYS": "XWIN_IMAGE,YWIN_IMAGE",
        "ELLIPTICITY_MAX": 0.5,
        "MOSAIC_TYPE": "UNCHANGED",
        "STABILITY_TYPE": "EXPOSURE",
        "SN_THRESHOLDS": "3.0,100000.0",
    }

    DEFAULT_SWARP_CONFIG = {
        "COMBINE": "N",
        "COMBINE_TYPE": "MEDIAN",
        "INTERPOLATE": "N",
        "OVERSAMPLING": 0,
        "SUBTRACT_BACK": "N",
        "RESAMPLE": "Y",
        "RESAMPLE_DIR": ".",
        "RESAMPLE_SUFFIX": ".resamp.fits",
        "WRITE_XML": "Y",
        "VERBOSE_TYPE": "LOG",
        "BLANK_BADPIXELS": "Y",
        "FILL_VALUE": "NAN",
        "EDGE_THRESH": 0.0,
        "CELESTIAL_TYPE": "NATIVE",
        "PROJECTION_TYPE": "TAN",
        "FSCALASTRO_TYPE": "NONE",
        "COPY_KEYWORDS": "TELESCOP,FILTER,INSTRUME,EXPTIME,GAIN,OBSMJD,RDNOISE,APER,FWHM",
        # Allow full WCS transformations including rotation
        "ROTATE": "Y",
    }

    # ---------------------------- Constructor / logger ----------------------------
    def __init__(
        self, input_yaml, verbose_level: int = 1, delete_originals: bool = True
    ):
        """
        Initialize the ImageDistortionCorrector.

        Args:
            input_yaml: Configuration YAML for the pipeline.
            verbose_level: Logging verbosity (0: quiet, 1: info, 2: verbose).
            delete_originals: Remove original FITS after creating *_align products.
        """
        self.verbose_level = verbose_level
        self.delete_originals = delete_originals
        self.logger = logging.getLogger(__name__)
        self.default_threads = 4
        self._executables: Dict[str, str] = {}
        self._temp_dirs: set[str] = set()
        self.cleanup_intermediate = True
        self.input_yaml = input_yaml

    # -------------------------------- Utilities --------------------------------
    @staticmethod
    def _header_indicates_resampled(header) -> bool:
        """Return True if the FITS header suggests the image was already resampled (e.g. by SWarp)."""
        for key in ("RESAMPLED", "SWARPED", "RESAMPLE"):
            if key not in header:
                continue
            header_value = header[key]
            if isinstance(header_value, bool) and header_value:
                return True
            if isinstance(header_value, str) and header_value.upper() in (
                "T",
                "TRUE",
                "Y",
                "YES",
                "1",
            ):
                return True
            if isinstance(header_value, (int, float)) and header_value:
                return True
        return False

    @staticmethod
    def _header_indicates_scamp(
        header, pattern: str = "Astrometric solution by SCAMP"
    ) -> bool:
        """Return True if the FITS header has HISTORY or COMMENT indicating a SCAMP astrometric solution."""
        pattern_lower = pattern.lower()
        for key in ("HISTORY", "COMMENT"):
            if key not in header:
                continue
            # Header can have multiple cards with the same keyword
            for card in header.cards:
                if card.keyword != key:
                    continue
                history_text = str(card.value).strip() if card.value is not None else ""
                if pattern_lower in history_text.lower():
                    return True
        return False

    @staticmethod
    def _ascii_safe(line: str) -> str:
        """Replace non-ASCII chars in a line with '?' safely (fixes ord() bug)."""
        return "".join((ch if ord(ch) < 128 else "?") for ch in line)

    def _create_conv_file(self, path: str, fwhm_pixels: float = 3.0) -> None:
        """
        Create a convolution kernel for SExtractor, optimized for the given FWHM.

        Args:
            path: Path to save the convolution file.
            fwhm_pixels: FWHM in pixels, used to optimize the kernel size.
        """

        fwhm_pixels = float(max(1.0, min(fwhm_pixels, 15.0)))
        kernel_size = max(3, int(np.ceil(fwhm_pixels * 2)))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
        kernel_size = min(kernel_size, 21)  # SExtractor hard limit
        self.logger.info(
            f"Creating Convolution Kernel with FWHM: {fwhm_pixels:.1f} pixels"
        )
        center = kernel_size // 2
        sigma = fwhm_pixels / 2.355  # FWHM = 2.355 * sigma
        conv_text = f"CONV NORM\n# {kernel_size}x{kernel_size} convolution mask with FWHM = {fwhm_pixels:.1f} pixels\n"
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                weight_value = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
                conv_text += f"{weight_value:.6f} "
            conv_text += "\n"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(conv_text)

    @staticmethod
    def _create_nnw_file(path: str) -> None:
        """Default SExtractor stellarity network."""
        import re

        nnw_text = r"""
NNW
# Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
# inputs: 9 for profile parameters + 1 for seeing.
# outputs: Stellarity index (0.0 to 1.0)
 3 10 10  1
-1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01
 3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00
-3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00
 4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
-2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00
 4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e-01 -1.01592e-02 -7.54052e+00
 1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
-5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01
 2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01
 3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01
-8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
-3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00
-1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00
 0.00000e+00
 1.00000e+00
"""
        # Remove indentation introduced by quoting
        nindent = len(re.split("NNW", nnw_text.split("\n", 1)[1])[0])
        nnw_text = "\n".join([line[nindent:] for line in nnw_text.split("\n")[1:]])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(nnw_text)

    def _validate_output_dir(
        self, output_dir: Optional[str], prefix: Optional[str] = None
    ) -> str:
        """Create and register a temporary output directory if none provided."""
        if output_dir is None:
            output_dir = mkdtemp(prefix=prefix)
            self._temp_dirs.add(output_dir)
            self.logger.info(f"Created temporary directory: {output_dir}")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    def plot_sextractor_sources(
        self,
        image_path: str,
        catalog_path: str,
        output_plot_path: str = "sextractor_sources.png",
        cmap: str = "viridis",
        figsize: Optional[tuple] = None,
        max_sources: int = 1000,
        **imshow_kwargs,
    ) -> None:
        """
        Plot an astronomical image and overlay circles around all sources detected by SExtractor.
        The circle radius is set to the source's FWHM.

        Args:
            image_path: Path to the FITS image.
            catalog_path: Path to the SExtractor catalog (FITS_LDAC).
            output_plot_path: Path to save the output plot.
            cmap: Colormap for the image.
            figsize: Figure size.
            max_sources: Maximum number of sources to plot.
            **imshow_kwargs: Additional arguments for imshow.
        """
        try:
            if figsize is None:
                # Prefer AutoPHoT figure sizing (golden-ratio based).
                from functions import set_size

                golden_ratio = (5**0.5 + 1) / 2
                figsize = set_size(340, aspect=golden_ratio)  # square-ish

            # Load image
            with fits.open(image_path) as hdul:
                data = hdul[0].data.astype(np.float32)

            # Load catalog
            with fits.open(catalog_path) as hdul:
                catalog = Table(hdul[2].data)

            # ZScale for contrast
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(data)

            # Plot image
            fig, ax = plt.subplots(figsize=figsize)
            # Render NaNs as white "no data" regions.
            cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
            try:
                cmap = cmap.copy()
            except Exception:
                pass
            try:
                cmap.set_bad(color="white")
            except Exception:
                pass
            im = ax.imshow(
                data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", **imshow_kwargs
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Plot circles for each source
            for i, row in enumerate(catalog[:max_sources]):
                if (
                    "XWIN_IMAGE" in row.colnames
                    and "YWIN_IMAGE" in row.colnames
                    and "FWHM_IMAGE" in row.colnames
                ):
                    x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                    fwhm = row["FWHM_IMAGE"]
                    # Convert to 0-based if needed
                    x_0based, y_0based = x - 1, y - 1
                    circle = Circle(
                        (x_0based, y_0based),
                        fwhm,
                        facecolor="none",
                        edgecolor="#FF0000",
                        linewidth=0.5,
                        alpha=0.7,
                    )
                    ax.add_patch(circle)
                    ax.text(
                        x_0based,
                        y_0based - fwhm - 2,
                        str(i),
                        color="#FF0000",
                        fontsize=8,
                        ha="center",
                        va="top",
                    )

            plt.tight_layout()
            plt.savefig(output_plot_path, dpi=150)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting sources: {e}")
            raise

    def _check_executable(self, name: str) -> str:
        """Resolve tool path or raise. Cached for speed."""
        if name in self._executables:
            return self._executables[name]
        for cmd in (name, f"{name}.exe"):
            try:
                subprocess.run(
                    [cmd, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
                self._executables[name] = cmd
                return cmd
            except (OSError, subprocess.CalledProcessError):
                continue
        raise FileNotFoundError(f"Executable not found: {name}")

    @staticmethod
    def angular_distance(ra1, dec1, ra2, dec2):
        """Small-angle distance in degrees; uses cos(dec2) projection for RA."""
        dra = (ra1 - ra2) * np.cos(np.radians(dec2))
        ddec = dec1 - dec2
        return np.sqrt(dra**2 + ddec**2)

    def update_fits_catalog(
        self, catalog_path, target_ra, target_dec, sigma_deg=0.01, tbhdu=2
    ):
        """
        Inflate errors far from target to bias SCAMP matching near a position.
        Uses SExtractor-style column names: ALPHA_J2000, DELTA_J2000, ERRA_WORLD, ERRB_WORLD.
        """
        with fits.open(catalog_path, mode="update") as hdul:
            catalog = Table(hdul[tbhdu].data)
            required_cols = ["ALPHA_J2000", "DELTA_J2000", "ERRA_WORLD", "ERRB_WORLD"]
            for col in required_cols:
                if col not in catalog.colnames:
                    raise ValueError(f"Column '{col}' not found in catalog")
            ra = catalog["ALPHA_J2000"]
            dec = catalog["DELTA_J2000"]
            err_ra = catalog["ERRA_WORLD"]
            err_dec = catalog["ERRB_WORLD"]
            dist = self.angular_distance(ra, dec, target_ra, target_dec)
            weights = np.exp(-0.5 * (dist / sigma_deg) ** 2)
            min_err = 1e-4
            catalog["ERRA_WORLD"] = np.clip(err_ra / weights, min_err, None)
            catalog["ERRB_WORLD"] = np.clip(err_dec / weights, min_err, None)
            hdul[tbhdu].data = catalog.as_array()

    def clean_log(self, input_file, output_file=None):
        """Strip ANSI escape codes and blank lines from a log file."""
        import re

        with open(input_file, "r", errors="ignore") as f:
            content = f.read()
        ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean = ansi.sub("", content)
        clean = "\n".join([line for line in clean.split("\n") if line.strip()])
        output_file = input_file if output_file is None else output_file
        with open(output_file, "w") as f:
            f.write(clean)
        return output_file

    def determine_saturation_level(self, fits_path: str) -> float:
        """Heuristic saturation estimate for SExtractor SATUR_LEVEL."""
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            # Store original dtype for saturation check before conversion
            original_dtype = data.dtype
            # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
            if data.dtype.kind != 'f':
                data = data.astype(np.float32)
            if np.issubdtype(original_dtype, np.integer):
                max_possible = np.iinfo(original_dtype).max
                if np.nanmax(data) >= 0.99 * max_possible:
                    return float(max_possible)
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return float(1e6)
            hist, bins = np.histogram(finite, bins=1000)
            peak_idx = int(np.argmax(hist))
            pct = float(np.nanpercentile(finite, 99.99))
            return float(max(bins[min(peak_idx, len(bins) - 2)], pct))

    # ----------------------------- SExtractor + PSFEx -----------------------------
    @staticmethod
    def _guess_map_weight_path(fits_image: str) -> Optional[str]:
        """
        Best-effort MAP_WEIGHT companion for SExtractor alignment.

        The photometry pipeline commonly writes ``<stem>.weight<ext>`` next to the
        science image. If present, using it in SExtractor improves robustness on
        mosaics with NaNs / no-coverage bands encoded as zeros.
        """
        try:
            p = Path(str(fits_image))
            candidates = [
                p.with_name(p.name + ".weight" + p.suffix),  # e.g. foo.fits -> foo.weight.fits
                p.with_suffix(".weight.fits"),  # legacy
            ]
            for c in candidates:
                if c.is_file():
                    return str(c)
        except Exception:
            return None
        return None

    def run_sextractor(
        self,
        fits_image: str,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
        SEEING_FWHM: float = 1.5,
        PIXEL_SCALE: float = 0,
        use_psfex: bool = False,
        aperture_radius=None,
        weight_path: Optional[str] = None,
        crowded: bool = False,
    ) -> Dict:
        """
        Build a clean catalog using SExtractor, prioritizing extended sources.
        Optimizes the convolution kernel if FWHM is available in the header.

        If ``weight_path`` points to an existing FITS weight map, SExtractor is run with
        ``WEIGHT_TYPE MAP_WEIGHT`` so invalid pixels can be down-weighted (important for
        NaN-heavy stacks and SWarp no-coverage bands).
        If crowded is True, uses parameters tuned for crowded fields (tighter deblending,
        smaller background mesh, more deblend levels).
        """
        try:
            output_dir = self._validate_output_dir(output_dir, prefix="sex_")

            stem = Path(fits_image).stem
            conv_path = str(Path(output_dir) / f"{stem}_default.conv")
            nnw_path = str(Path(output_dir) / f"{stem}_default.nnw")
            self._create_nnw_file(nnw_path)

            # Get FWHM from header if available, otherwise use default
            with fits.open(fits_image) as hdul:
                header = hdul[0].header
                fwhm_pixels = header.get("FWHM", 2.0)
                pixel_scale_header = header.get("PIXSCALE", header.get("CDELT2", 0))
                if pixel_scale_header:
                    pixel_scale_header = abs(float(pixel_scale_header)) * 3600.0
                else:
                    pixel_scale_header = PIXEL_SCALE if PIXEL_SCALE > 0 else 1.0
                seeing_fwhm_arcsec = float(fwhm_pixels) * pixel_scale_header

            self._create_conv_file(conv_path, fwhm_pixels=fwhm_pixels)

            final_config = self.DEFAULT_SEX_CONFIG.copy()
            if crowded:
                final_config.update(self.CROWDED_SEX_CONFIG)
                self.logger.info(
                    "Using SExtractor crowded-field config (tighter deblending, smaller back mesh)"
                )
            final_config.update(
                {
                    # 'CHECKIMAGE_NAME': 'check_seg.fits,check_aper.fits',
                    "SATUR_LEVEL": self.determine_saturation_level(fits_image),
                    "FILTER_NAME": conv_path,
                    "STARNNW_NAME": nnw_path,
                    "NTHREADS": self.default_threads,
                    "PIXEL_SCALE": PIXEL_SCALE,
                    "CATALOG_TYPE": "FITS_LDAC",
                    "SEEING_FWHM": round(seeing_fwhm_arcsec, 3),
                }
            )

            # Optional MAP_WEIGHT (0/1 or continuous) for robust detection on masked mosaics.
            wpath = str(weight_path) if weight_path else ""
            if wpath and os.path.isfile(wpath):
                final_config["WEIGHT_TYPE"] = "MAP_WEIGHT"
                final_config["WEIGHT_IMAGE"] = wpath

            if config:
                final_config.update(config)
            if aperture_radius:
                final_config["PHOT_APERTURES"] = aperture_radius

            sex_cmd = self._check_executable("sex")
            param_file = str(Path(output_dir) / f"{stem}_sextractor.param")
            params = [
                "NUMBER",
                "X_IMAGE",
                "Y_IMAGE",
                "FLUX_AUTO",
                "FLUXERR_AUTO",
                "FLUX_APER",
                "FLUXERR_APER",
                "FWHM_IMAGE",
                "ELLIPTICITY",
                "ELONGATION",
                "CLASS_STAR",
                "FLAGS",
                "FLAGS_WEIGHT",
                "XWIN_IMAGE",
                "YWIN_IMAGE",
                "ERRAWIN_IMAGE",
                "ERRBWIN_IMAGE",
                "ERRTHETAWIN_IMAGE",
                "XWIN_WORLD",
                "YWIN_WORLD",
                "X_WORLD",
                "Y_WORLD",
                "ERRA_WORLD",
                "ERRB_WORLD",
                "ERRTHETA_WORLD",
                "MAG_AUTO",
                "MAGERR_AUTO",
                "ALPHA_J2000",
                "DELTA_J2000",
                "FLUX_RADIUS",
                "BACKGROUND",
                "ISOAREA_IMAGE",
                "SNR_WIN",
                "A_IMAGE",
                "B_IMAGE",
                "THETA_IMAGE",
            ]
            Path(param_file).write_text("\n".join(params))
            config_file = str(Path(output_dir) / f"{stem}_default.sex")
            with open(config_file, "w") as f:
                for k, v in final_config.items():
                    f.write(f"{k}\t{v}\n")

            # Use the input filename (without extension) for the catalog
            catalog_name = Path(fits_image).stem + ".cat"
            catalog_path = str(Path(output_dir) / catalog_name)

            # Build SExtractor command
            cmd1 = [
                sex_cmd,
                fits_image,
                "-c",
                config_file,
                "-PARAMETERS_NAME",
                param_file,
                "-CATALOG_NAME",
                catalog_path,
                "-NTHREADS",
                str(final_config["NTHREADS"]),
            ]

            subprocess.run(cmd1, check=True, text=True)

            tbhdu = 2
            with fits.open(catalog_path, mode="update") as hdul:
                catalog = Table(hdul[tbhdu].data)
                if len(catalog) > 0:
                    good = np.ones(len(catalog), dtype=bool)
                    filt = {
                        "SNR_WIN": lambda c: c > 1,
                    }
                    for col, cond in filt.items():
                        if col in catalog.colnames:
                            good &= cond(catalog[col])
                    # Alignment source selection:
                    # Default behavior prefers star-like sources by excluding very large FWHM objects.
                    # For sparse fields, extended sources (galaxies) can be useful alignment anchors.
                    # Opt-in via YAML:
                    #   template_subtraction.alignment_use_extended_sources: True
                    # Optionally override the max-FWHM cutoff:
                    #   template_subtraction.alignment_max_fwhm_pix: <float>
                    iy = getattr(self, "input_yaml", None) or {}
                    ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
                    use_ext = bool(
                        isinstance(ts, dict)
                        and ts.get("alignment_use_extended_sources", False)
                    )
                    max_fwhm = float(
                        ts.get("alignment_max_fwhm_pix", self.ALIGNMENT_MAX_FWHM_PIX)
                    ) if isinstance(ts, dict) else self.ALIGNMENT_MAX_FWHM_PIX
                    if not use_ext:
                        if "FWHM_IMAGE" in catalog.colnames:
                            would_keep = good & (catalog["FWHM_IMAGE"] <= max_fwhm)
                            n_removed = np.sum(
                                good & (catalog["FWHM_IMAGE"] > max_fwhm)
                            )
                            # If excluding would leave 0 sources, keep all (avoid forcing empty catalog)
                            if np.sum(would_keep) > 0:
                                good &= catalog["FWHM_IMAGE"] <= max_fwhm
                                if n_removed > 0:
                                    self.logger.info(
                                        "Alignment SExtractor: excluding %d sources with FWHM > %.0f px",
                                        int(n_removed),
                                        float(max_fwhm),
                                    )
                            elif n_removed > 0:
                                self.logger.info(
                                    "Alignment SExtractor: all %d source(s) have FWHM > %.0f px; keeping them so alignment can proceed",
                                    int(n_removed),
                                    float(max_fwhm),
                                )
                    else:
                        self.logger.info(
                            "Alignment SExtractor: extended sources enabled (keeping large-FWHM objects)."
                        )
                    cleaned = catalog[good]
                    # cleaned = self.filter_well_defined_positions(cleaned)
                    if "SNR_WIN" in cleaned.colnames:
                        cleaned.sort("SNR_WIN", reverse=True)
                    hdul[tbhdu].data = cleaned.as_array()
                    hdul.flush()
                else:
                    cleaned = catalog

            self.logger.info(f"Sextractor found {len(cleaned)} sources")
            if len(cleaned) == 0 or "FWHM_IMAGE" not in cleaned.colnames:
                fwhm = (
                    float(fwhm_pixels)
                    if np.isfinite(fwhm_pixels) and fwhm_pixels > 0
                    else 2.5
                )
            else:
                from astropy.stats import sigma_clipped_stats
                _, fwhm, _ = sigma_clipped_stats(cleaned["FWHM_IMAGE"], sigma=3)
            if not np.isfinite(fwhm) or fwhm <= 0:
                fwhm = 2.5

        except Exception as e:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            self.logger.info(
                f"Type: {exc_type.__name__} File: {fname} Line: {lineno} Message: {e}"
            )
            raise

        return {
            "output_dir": output_dir,
            "catalog": cleaned,
            "catalog_path": catalog_path,
            "fwhm": fwhm,
            "checkimages": None,
            "config": final_config,
            "weight_path": weight_path,
        }

    def place_science_header_as_ahead(
        self,
        science_image: str,
        reference_dir: str,
        ahead_filename: str = "science.ahead",
    ) -> str:
        """
        Extract the header from the science image and save it as an `.ahead` file in the reference directory.
        This ensures the reference image uses the science image's WCS and metadata for alignment.

        Args:
            science_image: Path to the science image (FITS file).
            reference_dir: Path to the directory where the `.ahead` file will be saved.
            ahead_filename: Name of the output `.ahead` file.

        Returns:
            Path to the created `.ahead` file.
        """
        from astropy.io import fits
        import os

        # Ensure the reference directory exists
        os.makedirs(reference_dir, exist_ok=True)

        # Open the science image and extract its header
        with fits.open(science_image) as hdul:
            header = hdul[0].header

        # Save the header as an `.ahead` file in the reference directory
        ahead_path = os.path.join(reference_dir, ahead_filename)
        with open(ahead_path, "w") as f:
            for card in header.cards:
                # Use str(card) to get the full card as a string
                f.write(f"{str(card)}\n")

        self.logger.debug("Science header written as .ahead file: %s", ahead_path)

        return ahead_path

    def _extract_wcs_and_scale(self, fits_path):
        from astropy.wcs.utils import proj_plane_pixel_scales
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            wcs_obj = get_wcs(header)
            pixel_scale = proj_plane_pixel_scales(wcs_obj)[0] * 3600.0
            return wcs_obj, pixel_scale, header

    # ------------------------ Align + resample via SCAMP/SWarp ------------------------
    def align_and_resample_both_images(
        self,
        science_image: str,
        reference_image: str,
        output_dir: Optional[str] = None,
        science_already_resampled: bool = False,
        reference_already_resampled: bool = False,
    ) -> Optional[dict]:
        """
        Align and resample both science and reference images to the science image's grid.
        SCAMP is run on the reference image only (against the science catalog) to compute
        the WCS correction. Both images are then resampled together in a single SWarp call
        onto a common grid defined by the science image pixel centre and shape.
        If an image is already resampled (flag or header RESAMPLED/SWARPED), SWarp is skipped
        for that image to avoid degrading the data.

        Args:
            science_image: Path to science image.
            reference_image: Path to reference image.
            output_dir: Output directory for aligned images.
            science_already_resampled: If True, skip SWarp for science; use image as-is (avoids double resampling).
            reference_already_resampled: If True, skip SWarp for reference; use image as-is.
        Returns:
            Dictionary with paths to aligned images and alignment metadata.
        """
        try:
            output_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(science_image).parent
            )
            # Make aligned directories unique per science image to prevent crosstalk
            # when multiple images are processed in the same output directory
            science_base_name = Path(science_image).stem
            science_aligned_dir = output_dir / f"aligned_sci_{science_base_name}"
            reference_aligned_dir = output_dir / f"aligned_ref_{science_base_name}"

            # Delete existing directories if they exist
            if science_aligned_dir.exists():
                shutil.rmtree(science_aligned_dir)
            if reference_aligned_dir.exists():
                shutil.rmtree(reference_aligned_dir)

            # Create new, empty directories
            science_aligned_dir.mkdir(parents=True, exist_ok=True)
            reference_aligned_dir.mkdir(parents=True, exist_ok=True)

            # Copy and clean images
            sci_image_copy = science_aligned_dir / "science_image.fits"
            ref_image_copy = reference_aligned_dir / "reference_image.fits"

            # Copy science and reference images (weight maps not used for alignment)
            shutil.copy2(science_image, sci_image_copy)
            shutil.copy2(reference_image, ref_image_copy)

            # self.clean_image(sci_image_copy)
            # self.clean_image(ref_image_copy)

            # Extract sources from both images
            with (
                fits.open(sci_image_copy) as hdul,
                fits.open(ref_image_copy) as hdul_ref,
            ):
                sci_data, sci_head = hdul[0].data, hdul[0].header
                ref_data, ref_head = hdul_ref[0].data, hdul_ref[0].header
                from astropy.wcs.utils import proj_plane_pixel_scales
                sci_wcs = get_wcs(sci_head)
                ref_wcs = get_wcs(ref_head)
                sci_shape = sci_data.shape
                ref_shape = ref_data.shape
                sci_pix_scale = proj_plane_pixel_scales(sci_wcs)[0] * 3600.0
                ref_pix_scale = proj_plane_pixel_scales(ref_wcs)[0] * 3600.0
                # Use the science image pixel center as the SWarp grid CENTER, and the
                # science image shape as IMAGE_SIZE.  This is the only combination that
                # guarantees SWarp produces an output of exactly sci_shape for the science
                # image (it completely fills its own pixel-center grid) so no post-hoc
                # cropping is required.  The reference is resampled onto the identical grid
                # and will therefore have the same shape.
                # Use the integer center pixel (0-based).  For odd-sized dimensions
                # this lands exactly on the middle pixel; for even dimensions on the
                # lower-middle pixel.  Passing a half-pixel (shape/2.0) as CENTER
                # causes SWarp to compute CRPIX differently for science vs reference
                # (each image's WCS rounds the sub-pixel position independently),
                # producing the observed 1-px CRPIX offset between outputs.
                _cx_pix = sci_shape[1] // 2  # integer, 0-based
                _cy_pix = sci_shape[0] // 2  # integer, 0-based
                ra_arr, dec_arr = sci_wcs.all_pix2world(
                    [_cx_pix], [_cy_pix], 0
                )
                center_ra, center_dec = float(ra_arr[0]), float(dec_arr[0])
                
                # Compute optimal output size as intersection of valid (non-NaN) regions
                # from both images. This reduces computation when coverage differs.
                # Ensure the target (center_ra, center_dec) is always included in output.
                output_width, output_height = self._compute_optimal_output_shape(
                    sci_data, sci_wcs, ref_data, ref_wcs, sci_pix_scale,
                    target_ra=center_ra, target_dec=center_dec
                )
                self.logger.info(
                    "Optimal output shape: %dx%d (science was %dx%d, reference was %dx%d)",
                    output_width, output_height, sci_shape[1], sci_shape[0], ref_shape[1], ref_shape[0]
                )
                
                # Force SWARP to always resample both images (removed skip check)
                science_skip_resample = False
                reference_skip_resample = False
                reference_already_scamp = self._header_indicates_scamp(ref_head)

            # Extract sources
            sci_aperture_radius = sci_head.get("APER", 7)
            ref_aperture_radius = ref_head.get("APER", 7)
            self.logger.info(
                f"Extracting sources with aperture radii: science = {sci_aperture_radius:.1f} pixels, "
                f"reference = {ref_aperture_radius:.1f} pixels"
            )

            # Use crowded-field SExtractor config when requested (better deblending in dense fields)
            iy = getattr(self, "input_yaml", None) or {}
            ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
            templates_cfg = iy.get("templates", {}) if isinstance(iy, dict) else {}
            phot = iy.get("photometry") if isinstance(iy, dict) else None
            phot_crowded = bool(
                isinstance(phot, dict) and phot.get("crowded_field", False)
            )
            sextractor_crowded = ts.get(
                "sextractor_crowded", templates_cfg.get("crowded_field", phot_crowded)
            )
            sci_w = self._guess_map_weight_path(str(science_image))
            ref_w = self._guess_map_weight_path(str(reference_image))
            if sci_w:
                self.logger.info("Alignment SExtractor: using science MAP_WEIGHT %s", sci_w)
            if ref_w:
                self.logger.info("Alignment SExtractor: using reference MAP_WEIGHT %s", ref_w)

            sci_sex = self.run_sextractor(
                str(sci_image_copy),
                output_dir=str(science_aligned_dir),
                aperture_radius=sci_aperture_radius,
                weight_path=sci_w,
                PIXEL_SCALE=sci_pix_scale,
                crowded=sextractor_crowded,
            )
            ref_sex = self.run_sextractor(
                str(ref_image_copy),
                output_dir=str(reference_aligned_dir),
                aperture_radius=ref_aperture_radius,
                weight_path=ref_w,
                PIXEL_SCALE=ref_pix_scale,
                crowded=sextractor_crowded,
            )

            n_sci = len(sci_sex.get("catalog", []))
            n_ref = len(ref_sex.get("catalog", []))
            self.logger.info(
                "Alignment source counts: science image = %d, reference/template = %d",
                n_sci,
                n_ref,
            )

            fwhm_sci_pix = float(sci_sex["fwhm"]) if "fwhm" in sci_sex else 2.5
            fwhm_ref_pix = float(ref_sex["fwhm"]) if "fwhm" in ref_sex else 2.5

            fwhm_sci_arcsec = fwhm_sci_pix * sci_pix_scale
            fwhm_ref_arcsec = fwhm_ref_pix * ref_pix_scale
            self.logger.info(
                "Measured FWHM: science = %.2f pixels (%.2f arcsec), reference = %.2f pixels (%.2f arcsec)",
                fwhm_sci_pix,
                fwhm_sci_arcsec,
                fwhm_ref_pix,
                fwhm_ref_arcsec,
            )


            sci_is_undersampled = fwhm_sci_pix < 2.0
            ref_is_undersampled = fwhm_ref_pix < 2.0
            self.logger.info(
                "Undersampling flags: science = %s, reference = %s",
                sci_is_undersampled,
                ref_is_undersampled,
            )

            crossid_radius = max(max(fwhm_sci_arcsec, fwhm_ref_arcsec), 15.0)
            self.logger.info("Using cross-match radius: %.2f arcsec", crossid_radius)

            def _do_match(radius_arcsec: float):
                return self.filter_matched_sources(
                    sci_cat_path=sci_sex["catalog_path"],
                    ref_cat_path=ref_sex["catalog_path"],
                    match_radius_arcsec=radius_arcsec,
                    sci_image_path=str(sci_image_copy),
                    ref_image_path=str(ref_image_copy),
                )

            try:
                _num_matched, _ = _do_match(crossid_radius)
            except Exception as e:
                self.logger.warning(
                    "Source matching failed: %s. Falling back to reproject/AstroAlign.",
                    e,
                )
                return self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )


            self.logger.info(
                "Matched sources for SCAMP/SWarp: %d (science %d, reference %d)",
                _num_matched,
                n_sci,
                n_ref,
            )

            try:
                self.plot_matched_sources_side_by_side(
                    sci_image_path=str(sci_image_copy),
                    ref_image_path=str(ref_image_copy),
                    sci_cat_path=sci_sex["catalog_path"],
                    ref_cat_path=ref_sex["catalog_path"],
                    output_plot_path=output_dir / f"matched_sources_{Path(sci_image_copy).stem}.png",
                    label_color="#FF0000",
                    label_fontsize=7,
                    circle_radius_sci=fwhm_sci_pix,
                    circle_radius_ref=fwhm_ref_pix,
                    matched_circle_color="#FF0000",
                    unmatched_circle_color="blue",
                )
            except Exception as e:
                self.logger.debug("Matched-sources plot failed (non-fatal): %s", e)

            self.logger.info(
                "Proceeding with SCAMP + SWarp alignment (%d matched sources).",
                _num_matched,
            )
            pix_scale = sci_pix_scale

            # SCAMP: derive parameters from FWHM and pixel scale
            crossid_arcsec = max(
                1.5 * max(fwhm_sci_arcsec, fwhm_ref_arcsec), 1.5 * pix_scale, 2.0
            )
            # Adaptive POSITION_MAXERR: tighter for good fields (0.3"), more permissive for sparse fields
            # With few sources, we need to allow more positional uncertainty for SCAMP to find a solution
            is_sparse_field = _num_matched < 30
            position_maxerr_arcsec = max(1.5 * pix_scale, 1.0 if is_sparse_field else 0.3)
            # Adaptive SN threshold: lower minimum for sparse fields to include more sources
            sn_thresholds = "3.0,1000.0" if is_sparse_field else "5.0,1000.0"
            scamp_config_base = {
                "CROSSID_RADIUS": crossid_arcsec,
                "POSITION_MAXERR": position_maxerr_arcsec,
                "SN_THRESHOLDS": sn_thresholds,
                "MATCH_RESOL": "0.0",  # Auto-select best matching resolution
                "ASTREF_WEIGHT": "1",  # Weight by magnitude (prioritize bright, well-measured)
                "VERBOSE_TYPE": "FULL" if self.verbose_level >= 2 else "NORMAL",
                "WRITE_XML": "Y",
            }
            scamp_config_sci = {
                **scamp_config_base,
                "FWHM_THRESHOLDS": f"{0.3*fwhm_sci_pix:.2f},{10*fwhm_sci_pix:.2f}",
            }
            scamp_config_ref = {
                **scamp_config_base,
                "FWHM_THRESHOLDS": f"{0.3*fwhm_ref_pix:.2f},{10*fwhm_ref_pix:.2f}",
            }
            # Config for running SCAMP on both catalogs together
            # Use wider FWHM thresholds to accommodate both science and reference sources
            scamp_config_both = {
                **scamp_config_base,
                "FWHM_THRESHOLDS": (
                    f"{min(0.3*fwhm_sci_pix, 0.3*fwhm_ref_pix):.2f},"
                    f"{max(10*fwhm_sci_pix, 10*fwhm_ref_pix):.2f}"
                ),
            }
            sparse_note = " (sparse field mode)" if is_sparse_field else ""
            self.logger.info(
                'SCAMP: CROSSID_RADIUS=%.2f" POSITION_MAXERR=%.2f" SN_THRESHOLDS=%s%s FWHM_THRESHOLDS sci=[%.2f,%.2f] ref=[%.2f,%.2f]',
                crossid_arcsec,
                position_maxerr_arcsec,
                sn_thresholds,
                sparse_note,
                0.3 * fwhm_sci_pix,
                10 * fwhm_sci_pix,
                0.3 * fwhm_ref_pix,
                10 * fwhm_ref_pix,
            )

            # SWarp: resampling type from FWHM/undersampling
            def _swarp_resampling_type(is_undersampled: bool, fwhm_pix: float) -> str:
                if is_undersampled or fwhm_pix < 2.2:
                    return "BILINEAR"
                if fwhm_pix < 3.0:
                    return "LANCZOS2"
                return "LANCZOS3"

            sci_resampling_method = _swarp_resampling_type(
                sci_is_undersampled, fwhm_sci_pix
            )
            ref_resampling_method = _swarp_resampling_type(
                ref_is_undersampled, fwhm_ref_pix
            )

            # Allow YAML to override the automatic SWarp resampling choice for expert tuning.
            # templates.swarp_resampling_science / templates.swarp_resampling_reference can be
            # set to any SWarp-supported RESAMPLING_TYPE (e.g. 'BILINEAR', 'LANCZOS2', 'LANCZOS3').
            override_sci = templates_cfg.get("swarp_resampling_science")
            override_ref = templates_cfg.get("swarp_resampling_reference")
            if isinstance(override_sci, str) and override_sci.strip():
                self.logger.info(
                    "Overriding science SWarp RESAMPLING_TYPE from config: %s -> %s",
                    sci_resampling_method,
                    override_sci.strip().upper(),
                )
                sci_resampling_method = override_sci.strip().upper()
            if isinstance(override_ref, str) and override_ref.strip():
                self.logger.info(
                    "Overriding reference SWarp RESAMPLING_TYPE from config: %s -> %s",
                    ref_resampling_method,
                    override_ref.strip().upper(),
                )
                ref_resampling_method = override_ref.strip().upper()

            swarp_config = {
                "CENTER_TYPE": "MANUAL",
                "CENTER": f"{center_ra:.8f},{center_dec:.8f}",
                "PIXEL_SCALE": pix_scale,
                "PIXELSCALE_TYPE": "MANUAL",
                "IMAGE_SIZE": f"{output_width},{output_height}",
                "RESAMPLING_TYPE": sci_resampling_method,
                # OVERSAMPLING>0 causes SWarp to compute grid extents at N× sub-pixel
                # resolution internally then round back, producing off-by-one shape
                # mismatches when two images are resampled onto the same grid.
                # OVERSAMPLING=0 disables this entirely.
                "OVERSAMPLING": 0,
            }
            swarp_config_sci = {
                **swarp_config,
                "RESAMPLING_TYPE": sci_resampling_method,
            }
            swarp_config_ref = {
                **swarp_config,
                "RESAMPLING_TYPE": ref_resampling_method,
            }

            self.logger.info(
                "SWarp resampling: science=%s (FWHM=%.2f px%s) reference=%s (FWHM=%.2f px%s)",
                sci_resampling_method,
                fwhm_sci_pix,
                ", undersampled" if sci_is_undersampled else "",
                ref_resampling_method,
                fwhm_ref_pix,
                ", undersampled" if ref_is_undersampled else "",
            )

            self.logger.info(
                "SWarp output grid: CENTER=(%.6f,%.6f) IMAGE_SIZE=(%d,%d) PIXEL_SCALE=%.4f arcsec/px",
                center_ra, center_dec, output_width, output_height, pix_scale,
            )

            # Run SCAMP on BOTH catalogs with the science catalog as reference.
            # SCAMP solves both on the same astrometric grid and writes one .head
            # per input catalog.  However, SCAMP 2.14.0 groups catalogs by instrument
            # metadata (e.g., FILTER) and only writes one .head per instrument.
            # To force SCAMP to write separate .head files for both catalogs,
            # create temporary copies with different FILTER values.

            sci_cat_tmp_stem = None  # will be set below if SCAMP runs with temp catalogs
            ref_cat_tmp_stem = None  # will be set below if SCAMP runs with temp catalogs

            if reference_already_scamp:
                self.logger.info(
                    "Reference header has SCAMP HISTORY; skipping SCAMP, using existing WCS."
                )
                scamp_result = {}
            else:
                # Run SCAMP on both catalogs with distinct FILTER values so SCAMP
                # produces a separate .head file for each.  The science .head corrects
                # the science WCS; the reference .head corrects the reference WCS.
                # Both are placed next to their respective images before SWarp so
                # each is resampled with its SCAMP-refined WCS onto the same fixed grid.
                sci_cat_path = Path(sci_sex["catalog_path"])
                ref_cat_path = Path(ref_sex["catalog_path"])
                sci_cat_tmp = reference_aligned_dir / f"{sci_cat_path.stem}_sci.cat"
                ref_cat_tmp = reference_aligned_dir / f"{ref_cat_path.stem}_ref.cat"
                sci_cat_tmp_stem = sci_cat_tmp.stem
                ref_cat_tmp_stem = ref_cat_tmp.stem

                try:
                    for cat_path, cat_tmp, filter_val in [
                        (sci_cat_path, sci_cat_tmp, "w_sci"),
                        (ref_cat_path, ref_cat_tmp, "w_ref"),
                    ]:
                        with fits.open(cat_path, memmap=False) as hdul:
                            if len(hdul) > 1 and "FILTER" in hdul[1].header:
                                hdul[1].header["FILTER"] = filter_val
                            elif len(hdul) > 0 and "FILTER" in hdul[0].header:
                                hdul[0].header["FILTER"] = filter_val
                            hdul.writeto(cat_tmp, overwrite=True)
                        self.logger.debug(
                            "Created temporary catalog with FILTER=%s: %s", filter_val, cat_tmp
                        )

                    self.logger.info(
                        "Running SCAMP on both catalogs (science + reference)..."
                    )
                    scamp_result = self.run_scamp(
                        [str(sci_cat_tmp), str(ref_cat_tmp)],
                        reference_cat=sci_sex["catalog_path"],
                        output_dir=str(reference_aligned_dir),
                        config=scamp_config_both,
                    )

                    for cat_tmp in [sci_cat_tmp, ref_cat_tmp]:
                        try:
                            cat_tmp.unlink(missing_ok=True)
                        except Exception as e:
                            self.logger.debug("Could not remove temp catalog %s: %s", cat_tmp, e)

                except Exception as e:
                    self.logger.error("Failed to create temporary catalogs for SCAMP: %s", e)
                    self.logger.info("Falling back to single-catalog SCAMP on reference...")
                    scamp_result = self.run_scamp(
                        ref_sex["catalog_path"],
                        reference_cat=sci_sex["catalog_path"],
                        output_dir=str(reference_aligned_dir),
                        config=scamp_config_ref,
                    )
                    
                if scamp_result is None:
                    self.logger.info(
                        "SCAMP failed. Falling back to AstroAlign."
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

            # Copy both science and reference .head files next to their respective images
            # so SWarp applies the SCAMP WCS correction to each independently.
            # Both images are resampled onto the same fixed grid (CENTER/IMAGE_SIZE/PIXEL_SCALE),
            # so output shapes are guaranteed to match regardless of which images have .head files.
            head_by_stem = (
                scamp_result.get("head_files_by_stem", {})
                if isinstance(scamp_result, dict)
                else {}
            )
            self.logger.debug(
                "SCAMP produced .head files for stems: %s", list(head_by_stem.keys())
            )

            for label, cat_tmp_stem, orig_cat_path, fits_copy in [
                ("science", sci_cat_tmp_stem, sci_sex["catalog_path"], sci_image_copy),
                ("reference", ref_cat_tmp_stem, ref_sex["catalog_path"], ref_image_copy),
            ]:
                # Try temp-catalog stem first (most common path), then original stem
                head_src = None
                if cat_tmp_stem:
                    head_src = head_by_stem.get(cat_tmp_stem)
                if not head_src:
                    orig_stem = Path(orig_cat_path).stem
                    head_src = head_by_stem.get(orig_stem)
                # Fallback: single .head key (non-temp-catalog SCAMP path)
                if not head_src and label == "reference":
                    head_src = scamp_result.get("head_file") if isinstance(scamp_result, dict) else None

                if head_src and Path(head_src).exists():
                    head_dst = fits_copy.with_suffix(".head")
                    if Path(head_src).resolve() != head_dst.resolve():
                        try:
                            shutil.copy2(head_src, head_dst)
                            self.logger.info(
                                "Copied SCAMP .head (%s) to %s for SWarp.", label, head_dst
                            )
                        except Exception as e:
                            log_warning_from_exception(
                                self.logger, f"Could not copy .head for {label}", e
                            )
                else:
                    self.logger.warning(
                        "No SCAMP .head found for %s image (searched stems: %s, available: %s)",
                        label, [cat_tmp_stem, Path(orig_cat_path).stem], list(head_by_stem.keys()),
                    )

            resample_dir = science_aligned_dir / "resampled_output"
            resample_dir.mkdir(parents=True, exist_ok=True)
            swarp_res = None

            if science_skip_resample and reference_skip_resample:
                # Both already resampled: skip SWarp entirely to avoid double resampling.
                self.logger.info(
                    "Science and reference already resampled (header/flag); skipping SWarp."
                )
                resampled_sub = resample_dir / "resampled"
                resampled_sub.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    sci_image_copy, resampled_sub / "science_image.resamp.fits"
                )
                shutil.copy2(
                    ref_image_copy, resampled_sub / "reference_image.resamp.fits"
                )
                resampled_dir = resampled_sub
            elif science_skip_resample:
                self.logger.info(
                    "Science already resampled; running SWarp on reference only."
                )
                swarp_res = self.run_swarp(
                    [str(ref_image_copy)],
                    scamp_results=scamp_result,
                    output_dir=str(resample_dir),
                    config=swarp_config,
                )
                if swarp_res is None:
                    self.logger.info("SWarp failed. Falling back to AstroAlign.")
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                resampled_dir = Path(swarp_res["resampled_dir"])
                shutil.copy2(
                    sci_image_copy, resampled_dir / "science_image.resamp.fits"
                )
            elif reference_skip_resample:
                self.logger.info(
                    "Reference already resampled; running SWarp on science only."
                )
                swarp_res = self.run_swarp(
                    [str(sci_image_copy)],
                    scamp_results=None,
                    output_dir=str(resample_dir),
                    config=swarp_config,
                )
                if swarp_res is None:
                    self.logger.info("SWarp failed. Falling back to AstroAlign.")
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                resampled_dir = Path(swarp_res["resampled_dir"])
                shutil.copy2(
                    ref_image_copy, resampled_dir / "reference_image.resamp.fits"
                )
            else:
                # Resample both images onto the same grid in separate SWarp calls.
                # Science has no .head file (it defines the output grid — applying a
                # SCAMP self-reference .head would shift pixels off the grid -> all NaN).
                # Reference has its SCAMP .head placed next to it so SWarp applies the
                # WCS correction before resampling onto the fixed CENTER/IMAGE_SIZE grid.
                resample_dir_sci = resample_dir / "sci"
                resample_dir_ref = resample_dir / "ref"
                resample_dir_sci.mkdir(parents=True, exist_ok=True)
                resample_dir_ref.mkdir(parents=True, exist_ok=True)

                self.logger.debug("Running SWarp on science image...")
                swarp_res_sci = self.run_swarp(
                    [str(sci_image_copy)],
                    scamp_results=None,
                    output_dir=str(resample_dir_sci),
                    config=swarp_config_sci,
                    no_weight_maps=True,
                )
                self.logger.debug("Running SWarp on reference image...")
                swarp_res_ref = self.run_swarp(
                    [str(ref_image_copy)],
                    scamp_results=None,
                    output_dir=str(resample_dir_ref),
                    config=swarp_config_ref,
                    no_weight_maps=True,
                )

                if swarp_res_sci is None or swarp_res_ref is None:
                    self.logger.info("SWarp failed. Falling back to AstroAlign.")
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

                sci_resamp = next(Path(swarp_res_sci["resampled_dir"]).glob("*.resamp.fits"), None)
                ref_resamp = next(Path(swarp_res_ref["resampled_dir"]).glob("*.resamp.fits"), None)
                if sci_resamp is None or ref_resamp is None:
                    self.logger.info("SWarp resampled outputs not found. Falling back.")
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )

                canonical_dir = resample_dir / "resampled"
                canonical_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sci_resamp, canonical_dir / "science_image.resamp.fits")
                shutil.copy2(ref_resamp, canonical_dir / "reference_image.resamp.fits")
                resampled_dir = canonical_dir

            self.logger.debug("Looking for resampled images in %s", resampled_dir)
            aligned_sci = next(
                resampled_dir.glob("science_image.resamp.fits"), None
            )
            aligned_ref = next(
                resampled_dir.glob("reference_image.resamp.fits"), None
            )

            if aligned_sci is None or aligned_ref is None:
                self.logger.info(
                    "Could not find resampled images. Falling back to AstroAlign."
                )
                return self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )

            # Log SWarp output WCS for both images so alignment can be verified.
            # Since CENTER = science pixel center and IMAGE_SIZE = science shape,
            # both outputs should have identical shapes.
            for _label, _path in [("science", aligned_sci), ("reference", aligned_ref)]:
                try:
                    with fits.open(_path) as _hdul:
                        _hdr = _hdul[0].header
                        _ny, _nx = _hdul[0].data.shape
                    self.logger.info(
                        "SWarp output WCS [%s]: shape=(%d,%d) CRPIX=(%.2f,%.2f) "
                        "CRVAL=(%.6f,%.6f) CTYPE=(%s,%s)",
                        _label, _ny, _nx,
                        float(_hdr.get("CRPIX1", 0)), float(_hdr.get("CRPIX2", 0)),
                        float(_hdr.get("CRVAL1", 0)), float(_hdr.get("CRVAL2", 0)),
                        _hdr.get("CTYPE1", "?"), _hdr.get("CTYPE2", "?"),
                    )
                except Exception as _e:
                    self.logger.debug("Could not read WCS from SWarp output [%s]: %s", _label, _e)

            # Verify output shapes match. With both SCAMP .head files placed and
            # SWarp using a fixed CENTER/IMAGE_SIZE/PIXEL_SCALE grid, both outputs
            # must be identical. A mismatch means something went structurally wrong.
            try:
                with fits.open(aligned_sci) as _h:
                    _sci_shape = _h[0].data.shape
                with fits.open(aligned_ref) as _h:
                    _ref_shape = _h[0].data.shape

                if _sci_shape != _ref_shape:
                    self.logger.warning(
                        "SWarp outputs have different shapes (sci=%s, ref=%s) — "
                        "SCAMP/SWarp alignment failed, falling back.",
                        _sci_shape, _ref_shape,
                    )
                    return self._align_fallback_reproject_then_astroalign(
                        science_image, reference_image, output_dir
                    )
                else:
                    self.logger.info(
                        "SWarp outputs match: both images shape=%s.", _sci_shape,
                    )
            except Exception as _e:
                self.logger.debug("Could not verify SWarp output shapes: %s", _e)

            # Overwrite the reference_image path in-place with the aligned version.
            # reference_image is already a per-science-image copy so overwriting it
            # does not affect other science images using the same original template.
            aligned_science_fpath = science_image
            aligned_reference_fpath = reference_image

            # SWarp reads the SCAMP .head file next to the input FITS during resampling
            # and writes the correct aligned WCS into the .resamp.fits output.
            # No post-SWarp WCS injection is needed or correct: the SCAMP .head has
            # CRPIX/CRVAL from the original (pre-resample) image space, so injecting
            # it after SWarp would overwrite the valid resampled-grid WCS with
            # coordinates from the wrong pixel space.

            # Validate WCS of aligned images before overwriting originals
            # Test both pix2world and world2pix to ensure WCS is fully invertible
            try:
                with fits.open(aligned_ref) as hdul:
                    ref_wcs = get_wcs(hdul[0].header)
                    cx, cy = hdul[0].data.shape[1]/2, hdul[0].data.shape[0]/2
                    ref_ra, ref_dec = ref_wcs.all_pix2world([cx], [cy], 0)
                    if not (np.isfinite(ref_ra[0]) and np.isfinite(ref_dec[0])):
                        self.logger.error(
                            f"SWarp-aligned reference has invalid WCS pix2world: RA={ref_ra[0]}, Dec={ref_dec[0]}. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
                    # Test world2pix (critical for Cutout2D)
                    test_px, test_py = ref_wcs.all_world2pix(ref_ra[0], ref_dec[0], 0)
                    if not (np.isfinite(test_px) and np.isfinite(test_py)):
                        self.logger.error(
                            f"SWarp-aligned reference has non-invertible WCS: world2pix returned NaN/Inf. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
            except Exception as e:
                self.logger.error(f"Failed to validate WCS of SWarp-aligned reference: {e}")
                return self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )

            try:
                with fits.open(aligned_sci) as hdul:
                    sci_wcs = get_wcs(hdul[0].header)
                    cx, cy = hdul[0].data.shape[1]/2, hdul[0].data.shape[0]/2
                    sci_ra, sci_dec = sci_wcs.all_pix2world([cx], [cy], 0)
                    if not (np.isfinite(sci_ra[0]) and np.isfinite(sci_dec[0])):
                        self.logger.error(
                            f"SWarp-aligned science has invalid WCS pix2world: RA={sci_ra[0]}, Dec={sci_dec[0]}. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
                    # Test world2pix (critical for Cutout2D)
                    test_px, test_py = sci_wcs.all_world2pix(sci_ra[0], sci_dec[0], 0)
                    if not (np.isfinite(test_px) and np.isfinite(test_py)):
                        self.logger.error(
                            f"SWarp-aligned science has non-invertible WCS: world2pix returned NaN/Inf. "
                            f"Falling back to AstroAlign."
                        )
                        return self._align_fallback_reproject_then_astroalign(
                            science_image, reference_image, output_dir
                        )
            except Exception as e:
                self.logger.error(f"Failed to validate WCS of SWarp-aligned science: {e}")
                return self._align_fallback_reproject_then_astroalign(
                    science_image, reference_image, output_dir
                )

            try:
                shutil.copyfile(aligned_ref, aligned_reference_fpath)
                self.logger.debug(
                    "Saved aligned reference image to: %s",
                    aligned_reference_fpath,
                )
            except Exception as e:
                self.logger.warning(
                    "Could not save aligned reference image %s: %s",
                    aligned_reference_fpath,
                    e,
                )

            try:
                shutil.copyfile(aligned_sci, aligned_science_fpath)
                self.logger.debug(
                    "Overwrote science image with aligned version: %s",
                    aligned_science_fpath,
                )
            except Exception as e:
                self.logger.warning(
                    "Could not overwrite science image %s: %s",
                    aligned_science_fpath,
                    e,
                )

            # Remove the aligned working directory after copying aligned images over the originals.
            if science_aligned_dir.exists():
                shutil.rmtree(science_aligned_dir, ignore_errors=True)
                self.logger.debug(
                    "Removed aligned working dir: %s", science_aligned_dir
                )
            if (
                reference_aligned_dir.exists()
                and reference_aligned_dir != science_aligned_dir
            ):
                shutil.rmtree(reference_aligned_dir, ignore_errors=True)
                self.logger.debug(
                    "Removed aligned working dir: %s", reference_aligned_dir
                )

            return {
                "science_aligned": str(aligned_science_fpath),
                "reference_aligned": str(aligned_reference_fpath),
                "science_resampling_method": sci_resampling_method,
                "reference_resampling_method": ref_resampling_method,
                "science_undersampled": sci_is_undersampled,
                "reference_undersampled": ref_is_undersampled,
                "science_fwhm_pixels": fwhm_sci_pix,
                "reference_fwhm_pixels": fwhm_ref_pix,
                "alignment_method": "scamp_swarp",
            }

        except Exception as e:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            self.logger.error(
                f"Alignment failed: Type: {exc_type.__name__} File: {fname} Line: {lineno} Message: {e}"
            )
            return None

    # ------------------------------ Reproject (WCS-based) path ------------------------------

    def align_with_reproject(
        self,
        science_image: str,
        reference_image: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict]:
        """
        Align the reference image onto the science image pixel grid using WCS reprojection.
        Requires valid WCS in both FITS headers. Returns same dict as align_with_astroalign, or None on failure.
        """
        if not HAS_REPROJECT:
            self.logger.debug(
                "reproject package not available; skipping reproject alignment."
            )
            return None
        try:
            output_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(science_image).parent
            )
            with fits.open(science_image, mode="readonly") as h_sci:
                sci_data = h_sci[0].data
                sci_header = h_sci[0].header
                if sci_data is None or sci_data.size == 0:
                    return None
                # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
                if sci_data.dtype.kind != 'f':
                    sci_data = sci_data.astype(np.float32)
            with fits.open(reference_image, mode="readonly") as h_ref:
                ref_data = h_ref[0].data
                ref_header = h_ref[0].header
                if ref_data is None or ref_data.size == 0:
                    return None
                # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
                if ref_data.dtype.kind != 'f':
                    ref_data = ref_data.astype(np.float32)
            sci_wcs = get_wcs(sci_header)
            ref_wcs = get_wcs(ref_header)
            if sci_wcs is None or ref_wcs is None:
                self.logger.debug(
                    "Reproject: missing WCS in science or reference header."
                )
                return None
            # Reproject reference onto science grid
            aligned_ref, footprint = reproject_interp(
                (ref_data, ref_wcs),
                sci_wcs,
                shape_out=sci_data.shape,
                order="bilinear",
                roundtrip_coords=True,
            )
            aligned_ref = np.asarray(aligned_ref, dtype=float)
            out_header = ref_header.copy()
            out_header = remove_wcs_from_header(out_header)
            out_header.update(sci_wcs.to_header(), relax=True)
            base_ref = Path(reference_image).stem
            ext_ref = Path(reference_image).suffix
            aligned_reference_fpath = output_dir / f"{base_ref}{ext_ref}"
            from functions import safe_fits_write
            safe_fits_write(str(aligned_reference_fpath), aligned_ref, out_header)
            self.logger.info("Alignment via WCS reproject succeeded.")
            return {
                "science_aligned": science_image,
                "reference_aligned": str(aligned_reference_fpath),
                "alignment_method": "reproject",
            }
        except Exception as e:
            self.logger.info("Reproject alignment failed: %s", e)
            return None

    def _compute_optimal_output_shape(
        self,
        sci_data: np.ndarray,
        sci_wcs,
        ref_data: np.ndarray,
        ref_wcs,
        pix_scale: float,
        target_ra: Optional[float] = None,
        target_dec: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Compute optimal output shape as intersection of valid (non-NaN) regions.
        
        This reduces output size when images have different coverage or significant
        masked edges, avoiding computation on regions where only one image has data.
        
        Parameters
        ----------
        sci_data, ref_data : ndarray
            Image data arrays (may contain NaN for masked regions)
        sci_wcs, ref_wcs : WCS
            World coordinate systems for each image
        pix_scale : float
            Output pixel scale in arcsec/pixel
        target_ra, target_dec : float, optional
            Target position in degrees. If provided, the output shape is guaranteed
            to include this position (expanded if necessary, up to science image bounds).
            
        Returns
        -------
        (width, height) : tuple of int
            Optimal output dimensions in pixels
        """
        try:
            # Find valid (finite) pixel regions in each image
            sci_valid = np.isfinite(sci_data)
            ref_valid = np.isfinite(ref_data)
            
            # Get bounding boxes of valid regions
            def _valid_bbox(valid_mask):
                rows = np.any(valid_mask, axis=1)
                cols = np.any(valid_mask, axis=0)
                if not np.any(rows) or not np.any(cols):
                    return None
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return (cmin, rmin, cmax, rmax)  # x1,y1,x2,y2 in pixel coords
            
            sci_bbox = _valid_bbox(sci_valid)
            ref_bbox = _valid_bbox(ref_valid)
            
            if sci_bbox is None or ref_bbox is None:
                # Fall back to science shape if no valid data found
                return sci_data.shape[1], sci_data.shape[0]
            
            # Convert bounding box corners to world coordinates
            def _bbox_corners(bbox):
                x1, y1, x2, y2 = bbox
                return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            
            sci_corners = _bbox_corners(sci_bbox)
            ref_corners = _bbox_corners(ref_bbox)
            
            # Convert to RA/Dec
            sci_world = sci_wcs.all_pix2world(sci_corners, 0)
            ref_world = ref_wcs.all_pix2world(ref_corners, 0)
            
            # Find overlapping RA/Dec region
            ra_min = max(np.min(sci_world[:, 0]), np.min(ref_world[:, 0]))
            ra_max = min(np.max(sci_world[:, 0]), np.max(ref_world[:, 0]))
            dec_min = max(np.min(sci_world[:, 1]), np.min(ref_world[:, 1]))
            dec_max = min(np.max(sci_world[:, 1]), np.max(ref_world[:, 1]))
            
            # Check if there is any overlap
            if ra_min >= ra_max or dec_min >= dec_max:
                self.logger.warning(
                    "No overlap in valid regions; using science image shape."
                )
                return sci_data.shape[1], sci_data.shape[0]
            
            # Convert overlap size to pixels at output pixel scale
            # cos(dec) factor for RA separation
            cos_dec = np.cos(np.radians((dec_min + dec_max) / 2))
            ra_sep = (ra_max - ra_min) * cos_dec * 3600  # arcsec
            dec_sep = (dec_max - dec_min) * 3600  # arcsec
            
            width = int(ra_sep / pix_scale)
            height = int(dec_sep / pix_scale)
            
            # Ensure minimum size and add small margin
            width = max(width, 100)
            height = max(height, 100)
            
            # Ensure target is included in output (if target position provided)
            if target_ra is not None and target_dec is not None:
                # Convert target world coords to pixel coords in output grid
                # Output grid center is at (width/2, height/2) in pixel coords
                # with pix_scale arcsec/pixel
                cos_dec = np.cos(np.radians(target_dec))
                
                # Target offset from center in arcsec
                dra_arcsec = (target_ra - (ra_min + ra_max) / 2) * cos_dec * 3600
                ddec_arcsec = (target_dec - (dec_min + dec_max) / 2) * 3600
                
                # Convert to pixels
                dx_pix = dra_arcsec / pix_scale
                dy_pix = ddec_arcsec / pix_scale
                
                # Check if target falls within current output bounds
                half_w = width / 2
                half_h = height / 2
                
                # Expand width if needed
                if abs(dx_pix) >= half_w - 5:  # 5 pixel margin
                    new_half_w = abs(dx_pix) + 10  # Add margin
                    width = int(2 * new_half_w)
                    self.logger.debug(
                        "Expanded output width to include target: %d -> %d", 
                        int(2 * half_w), width
                    )
                
                # Expand height if needed
                if abs(dy_pix) >= half_h - 5:  # 5 pixel margin
                    new_half_h = abs(dy_pix) + 10  # Add margin
                    height = int(2 * new_half_h)
                    self.logger.debug(
                        "Expanded output height to include target: %d -> %d",
                        int(2 * half_h), height
                    )
            
            # Cap at original science size (don't expand beyond input)
            width = min(width, sci_data.shape[1])
            height = min(height, sci_data.shape[0])
            
            return width, height
            
        except Exception as e:
            self.logger.warning(
                "Could not compute optimal output shape: %s. Using science shape.", e
            )
            return sci_data.shape[1], sci_data.shape[0]

    def _align_fallback_reproject_then_astroalign(
        self,
        science_image: str,
        reference_image: str,
        output_dir,
    ) -> Dict:
        """Try reproject first; if it fails or is unavailable, fall back to AstroAlign."""
        result = self.align_with_reproject(
            science_image, reference_image, output_dir=output_dir
        )
        if result is not None and result.get("reference_aligned"):
            return result
        self.logger.info("Reproject failed. Falling back to AstroAlign.")
        return self.align_with_astroalign(
            science_image, reference_image, output_dir
        )

    # ------------------------------ AstroAlign path ------------------------------

    def align_with_astroalign(
        self,
        science_image: str,
        reference_image: str,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Align the reference image onto the science image pixel grid using AstroAlign.
        Uses SExtractor catalogs and a 1<->1 filter to build control points, prioritizing extended sources.
        """
        try:
            # Define output directories
            output_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(science_image).parent
            )
            # Make aligned directories unique per science image to prevent crosstalk
            # when multiple images are processed in the same output directory
            science_base_name = Path(science_image).stem
            science_aligned_dir = output_dir / f"aligned_sci_{science_base_name}"
            reference_aligned_dir = output_dir / f"aligned_ref_{science_base_name}"

            # Delete existing directories if they exist
            if science_aligned_dir.exists():
                shutil.rmtree(science_aligned_dir)
            if reference_aligned_dir.exists():
                shutil.rmtree(reference_aligned_dir)

            # Create new, empty directories
            science_aligned_dir.mkdir(parents=True, exist_ok=True)
            reference_aligned_dir.mkdir(parents=True, exist_ok=True)

            sci_image_copy = science_aligned_dir / "science_image.fits"
            ref_image_copy = reference_aligned_dir / "reference_image.fits"
            shutil.copy2(science_image, sci_image_copy)
            shutil.copy2(reference_image, ref_image_copy)
            sci_wcs, sci_pix_scale, sci_head = self._extract_wcs_and_scale(
                sci_image_copy
            )
            ref_wcs, ref_pix_scale, ref_head = self._extract_wcs_and_scale(
                ref_image_copy
            )

            def _clean_image_data(fits_path):
                with fits.open(fits_path, mode="update") as hdul:
                    data = hdul[0].data
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    # Preserve NaNs (chip gaps) instead of replacing with sentinel
                    data[~np.isfinite(data)] = np.nan
                    hdul[0].data = data
                    hdul.flush()

            iy = getattr(self, "input_yaml", None) or {}
            ts = iy.get("template_subtraction", {}) if isinstance(iy, dict) else {}
            templates_cfg = iy.get("templates", {}) if isinstance(iy, dict) else {}
            sextractor_crowded = ts.get(
                "sextractor_crowded", templates_cfg.get("crowded_field", False)
            )

            def _extract_sources(fits_path, output_dir, aperture_radius, weight_path=None):
                return self.run_sextractor(
                    str(fits_path),
                    output_dir=str(output_dir),
                    aperture_radius=aperture_radius,
                    weight_path=weight_path,
                    crowded=sextractor_crowded,
                )

            sci_aperture_radius = sci_head.get("APER", 7)
            ref_aperture_radius = ref_head.get("APER", 7)
            self.logger.info(
                f"Extracting sources from science image with aperture radius - {sci_aperture_radius:.1f} [px]"
            )
            sci_w = self._guess_map_weight_path(str(science_image))
            ref_w = self._guess_map_weight_path(str(reference_image))
            if sci_w:
                self.logger.info("AstroAlign SExtractor: using science MAP_WEIGHT %s", sci_w)
            if ref_w:
                self.logger.info("AstroAlign SExtractor: using reference MAP_WEIGHT %s", ref_w)

            sci_sex = _extract_sources(
                sci_image_copy, science_dir, sci_aperture_radius, weight_path=sci_w
            )
            self.logger.info(
                f"Extracting sources from reference image with aperture radius - {ref_aperture_radius:.1f} [px]"
            )
            ref_sex = _extract_sources(
                ref_image_copy, reference_dir, ref_aperture_radius, weight_path=ref_w
            )
            fwhm_sci_pix = float(sci_sex.get("fwhm", 2.5))
            fwhm_ref_pix = float(ref_sex.get("fwhm", 2.5))
            fwhm_sci_arcsec = fwhm_sci_pix * sci_pix_scale
            fwhm_ref_arcsec = fwhm_ref_pix * ref_pix_scale
            crossid_radius = max(max(fwhm_sci_arcsec, fwhm_ref_arcsec), 15.0)

            def _load_matched_catalogs(sci_cat_path, ref_cat_path):
                with fits.open(sci_cat_path) as hs, fits.open(ref_cat_path) as hr:
                    sci_tab = Table(hs[2].data)
                    ref_tab = Table(hr[2].data)
                if (
                    len(sci_tab) == 0
                    or len(ref_tab) == 0
                    or len(sci_tab) != len(ref_tab)
                ):
                    raise ValueError(
                        "No 1:1 matched sources after filter_matched_sources"
                    )
                return sci_tab, ref_tab

            _num_matched, _ = self.filter_matched_sources(
                sci_cat_path=sci_sex["catalog_path"],
                ref_cat_path=ref_sex["catalog_path"],
                match_radius_arcsec=crossid_radius,
                sci_image_path=str(sci_image_copy),
                ref_image_path=str(ref_image_copy),
            )
            try:
                self.plot_matched_sources_side_by_side(
                    sci_image_path=str(sci_image_copy),
                    ref_image_path=str(ref_image_copy),
                    sci_cat_path=sci_sex["catalog_path"],
                    ref_cat_path=ref_sex["catalog_path"],
                    output_plot_path=science_dir / f"matched_sources_{Path(sci_image_copy).stem}.png",
                    label_color="#FF0000",
                    label_fontsize=10,
                    circle_radius_sci=fwhm_sci_pix,
                    circle_radius_ref=fwhm_ref_pix,
                )
            except Exception as _plot_exc:
                self.logger.debug("Matched-sources plot failed (non-fatal): %s", _plot_exc)
            sci_tab, ref_tab = _load_matched_catalogs(
                sci_sex["catalog_path"], ref_sex["catalog_path"]
            )

            def _extract_xy_snr(tbl):
                xcol = "XWIN_IMAGE" if "XWIN_IMAGE" in tbl.colnames else "X_IMAGE"
                ycol = "YWIN_IMAGE" if "YWIN_IMAGE" in tbl.colnames else "Y_IMAGE"
                snrcol = "SNR_WIN" if "SNR_WIN" in tbl.colnames else None
                x = np.asarray(tbl[xcol], float) - 1.0
                y = np.asarray(tbl[ycol], float) - 1.0
                if snrcol:
                    snr = np.asarray(tbl[snrcol], float)
                else:
                    snr = np.asarray(
                        (
                            tbl["FLUX_WIN"]
                            if "FLUX_WIN" in tbl.colnames
                            else tbl["FLUX_AUTO"]
                        ),
                        float,
                    )
                return x, y, snr

            sx, sy, ssnr = _extract_xy_snr(sci_tab)
            rx, ry, rsnr = _extract_xy_snr(ref_tab)
            joint = np.minimum(ssnr, rsnr)
            order = np.argsort(joint)[::-1]
            pts_sci = np.vstack([sy[order], sx[order]]).T
            pts_ref = np.vstack([ry[order], rx[order]]).T

            def _load_and_clean_image(fits_path):
                with fits.open(fits_path) as hdul:
                    # Keep float32 to avoid doubling memory on full frames.
                    img = np.asarray(hdul[0].data, dtype=np.float32)
                img = np.nan_to_num(img, nan=np.nan, posinf=np.nan, neginf=np.nan)
                return img

            ref_img = _load_and_clean_image(ref_image_copy)
            sci_img = _load_and_clean_image(sci_image_copy)
            MAX_CONTROL_POINTS = 300
            use_aafitrans = False
            try:
                import aafitrans

                tform, (matched_src, matched_dst) = aafitrans.find_transform(
                    pts_ref,
                    pts_sci,
                    max_control_points=min(MAX_CONTROL_POINTS, len(pts_ref)),
                    ttype="affine",
                    pixel_tolerance=max(fwhm_sci_pix, fwhm_ref_pix),
                    min_matches=3,
                    num_nearest_neighbors=3,
                    kdtree_search_radius=0.05,
                    n_samples=1,
                    get_best_fit=True,
                    seed=None,
                )
                use_aafitrans = True
                self.logger.info("Using aafitrans for alignment")
            except Exception as e:
                self.logger.info(
                    f"aafitrans not available or failed ({e}), using astroalign"
                )
                tform, (matched_src, matched_dst) = aa.find_transform(
                    pts_ref,
                    pts_sci,
                    max_control_points=min(MAX_CONTROL_POINTS, len(pts_ref)),
                )
            if use_aafitrans:
                try:
                    from scipy import ndimage

                    matrix = tform.params if hasattr(tform, "params") else tform._matrix
                    inv_matrix = np.linalg.inv(matrix)
                    aligned_ref_img = ndimage.affine_transform(
                        ref_img,
                        inv_matrix[:2, :2],
                        offset=inv_matrix[:2, 2],
                        output_shape=sci_img.shape,
                        order=3,
                        cval=np.nan,
                    )
                    footprint = np.isfinite(aligned_ref_img)
                except Exception as e:
                    self.logger.info(
                        f"aafitrans apply failed: {e}, falling back to astroalign"
                    )
                    use_aafitrans = False
                    tform, (matched_src, matched_dst) = aa.find_transform(
                        pts_ref,
                        pts_sci,
                        max_control_points=min(MAX_CONTROL_POINTS, len(pts_ref)),
                    )
            if not use_aafitrans:
                aligned_ref_img, footprint = aa.apply_transform(tform, ref_img, sci_img)
            # Preserve NaNs (chip gaps) instead of replacing with sentinel
            aligned_ref_img = np.nan_to_num(
                aligned_ref_img, nan=np.nan, posinf=np.nan, neginf=np.nan
            )
            aligned_ref_img[~footprint.astype(bool)] = np.nan

            def _save_aligned_image(data, header, output_path):
                from functions import safe_fits_write
                safe_fits_write(output_path, data, header)

            base_ref = os.path.splitext(os.path.basename(reference_image))[0]
            ext_ref = os.path.splitext(reference_image)[1]
            aligned_reference_fpath = os.path.join(output_dir, f"{base_ref}{ext_ref}")
            aligned_science_fpath = science_image
            out_hdr = ref_head.copy()
            out_hdr = remove_wcs_from_header(out_hdr)
            out_hdr.update(sci_wcs.to_header(), relax=True)
            _save_aligned_image(aligned_ref_img, out_hdr, aligned_reference_fpath)
            try:
                if use_aafitrans:
                    try:
                        src_w = tform(matched_src)  # skimage Transform.__call__ works
                    except TypeError:
                        # Fallback for transforms without __call__
                        src_w = tform.inverse(matched_src)
                else:
                    src_w = aa.matrix_transform(matched_src, tform.params)
                resid = np.sqrt(np.sum((src_w - matched_dst) ** 2, axis=1))
                rms_pix = float(np.sqrt(np.mean(resid**2))) if resid.size else np.nan
                self.logger.info(
                    f"AstroAlign: {len(matched_dst)} matches, RMS={rms_pix:.3f} px"
                )
            except Exception as e:
                self.logger.info(f"Could not compute alignment metrics: {e}")

            # Remove aligned working directories and their contents after successful alignment.
            if science_dir.exists():
                shutil.rmtree(science_dir, ignore_errors=True)
                self.logger.debug("Removed aligned working dir: %s", science_dir)
            if reference_dir.exists():
                shutil.rmtree(reference_dir, ignore_errors=True)
                self.logger.debug("Removed aligned working dir: %s", reference_dir)

            return {
                "science_aligned": aligned_science_fpath,
                "reference_aligned": aligned_reference_fpath,
                "alignment_method": "astroalign",
            }
        except Exception as e:
            import sys

            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            self.logger.error(
                f"Type: {exc_type.__name__} File: {fname} Line: {lineno} Message: {e}"
            )
            return {
                "science_aligned": None,
                "reference_aligned": None,
                "alignment_offset_arcsec": None,
            }

    # -------------------------------- SCAMP + SWarp --------------------------------
    def _parse_scamp_xml(self, xml_file: str) -> Optional[Dict]:
        """Extract a few WCS keys from SCAMP XML. Extend as needed."""
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            wcs = {}
            tags = {
                "CD1_1",
                "CD1_2",
                "CD2_1",
                "CD2_2",
                "CRVAL1",
                "CRVAL2",
                "CRPIX1",
                "CRPIX2",
            }
            for param in root.findall(".//Astrometry"):
                for child in param:
                    if child.tag in tags:
                        wcs[child.tag] = float(child.text)
            return {"distortion_coeffs": {}, "wcs_params": wcs}
        except Exception as e:
            self.logger.info(f"Could not parse SCAMP XML: {e}")
            return None

    def run_swarp(
        self,
        input_images: Union[str, List[str]],
        scamp_results: Optional[Dict] = None,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
        head_path: Optional[str] = None,
        no_weight_maps: bool = False,
    ) -> Dict:
        """
        Resample images using SWarp onto a common grid.

        For alignment of two images, place a SCAMP-generated .head file next to each
        input FITS with the same base name (e.g. ref.fits -> ref.head) so SWarp uses
        the refined WCS. Callers (e.g. align_and_resample_both_images) are responsible
        for copying the .head to the correct path before invoking this method.

        Config should set CENTER_TYPE/MANUAL, PIXEL_SCALE, IMAGE_SIZE to define the
        output grid; RESAMPLING_TYPE can be set per use (e.g. LANCZOS3, BILINEAR for
        undersampled). Returns resampled product paths and metadata.
        """
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image = str(output_dir / "output.fits")
        output_weights = str(output_dir / "weights.fits")
        log_file = str(output_dir / "swarp.log")

        # --- Config setup ---
        final_config = self.DEFAULT_SWARP_CONFIG.copy()
        final_config.update(
            {
                "WEIGHTOUT_NAME": output_weights,
                "NTHREADS": self.default_threads,
            }
        )
        if config:
            final_config.update(config)

        # --- Check SWarp executable ---
        swarp_cmd = self._check_executable("swarp")

        # --- Write SWarp config file ---
        config_file = str(output_dir / "default.swarp")
        with open(config_file, "w") as f:
            for k, v in final_config.items():
                f.write(f"{k}\t{v}\n")

        # --- Prepare input_images ---
        if isinstance(input_images, str):
            input_images = [input_images]
        input_images = [str(p) for p in input_images]

        # ------------------------------------------------------------------
        # Preserve NaNs through SWarp resampling via weight maps.
        #
        # SWarp does not reliably propagate NaNs in resampling kernels; the
        # supported mechanism for "no data" is a weight map (0 = invalid).
        # We therefore generate a MAP_WEIGHT image for each input:
        #   weight = 1 where finite, 0 where NaN/Inf.
        # With BLANK_BADPIXELS=Y and FILL_VALUE=NAN, SWarp will blank regions
        # with zero weight in the resampled products as NaN instead of 0.
        # ------------------------------------------------------------------
        weight_images: List[str] = []
        if no_weight_maps:
            weight_images = []
        for img_path in ([] if no_weight_maps else input_images):
            try:
                with fits.open(img_path, memmap=False) as hdul:
                    data = np.asarray(hdul[0].data, dtype=float)
                # Base weight: 1 for finite pixels, 0 for NaN/Inf
                w = np.isfinite(data).astype(np.float32)

                w_path = str(output_dir / (Path(img_path).stem + ".weight.fits"))
                fits.writeto(w_path, w.astype(np.float32), overwrite=True)
                weight_images.append(w_path)
            except Exception as e:
                # If weight generation fails, proceed without weights (legacy behaviour)
                log_warning_from_exception(
                    self.logger, f"Could not build SWarp weight map for {img_path}", e
                )
                weight_images = []
                break

        # --- Prepare output paths ---
        xml_file = str(output_dir / "swarp.xml")
        resample_dir = str(output_dir / "resampled")
        Path(resample_dir).mkdir(parents=True, exist_ok=True)

        # # --- Move .head files to head_path if specified ---
        # if head_path:
        #     head_path = Path(head_path)
        #     head_path.mkdir(parents=True, exist_ok=True)
        #     for img in input_images:
        #         head_file = Path(img).with_suffix('.head')
        #         if head_file.exists():
        #             try:
        #                 shutil.move(str(head_file), str(head_path / head_file.name))
        #             except Exception:
        #                 pass

        # --- Build SWarp command ---
        cmd = [
            swarp_cmd,
            *input_images,
            "-c",
            config_file,
            "-IMAGEOUT_NAME",
            output_image,
            "-RESAMPLE_DIR",
            resample_dir,
            "-XML_NAME",
            xml_file,
            "-NTHREADS",
            str(final_config["NTHREADS"]),
        ]

        # Enable weight-map propagation to preserve NaN/no-data regions.
        if weight_images:
            cmd.extend(
                [
                    "-WEIGHT_TYPE",
                    "MAP_WEIGHT",
                    "-WEIGHT_IMAGE",
                    ",".join(weight_images),
                ]
            )

        # # --- Add HEAD_PATH to SWarp command if specified ---
        # if head_path:
        #     cmd.extend(['-HEAD_PATH', str(head_path)])

        # --- Logging ---
        if self.verbose_level >= 2:
            import shlex

            self.logger.debug("Output image: %s", output_image)
            self.logger.debug("Running SWarp...\n%s", shlex.join(cmd))

        def _cleanup_swarp_outputs():
            """Remove SWarp output files on failure so the directory is clean for fallback/retry."""
            to_remove = [
                output_image,
                output_weights,
                config_file,
                xml_file,
            ]
            for p in to_remove:
                try:
                    if p and os.path.isfile(p):
                        os.remove(p)
                        self.logger.debug("Removed SWarp output: %s", p)
                except OSError as e:
                    self.logger.debug("Could not remove %s: %s", p, e)
            try:
                resample_path = Path(resample_dir)
                if resample_path.is_dir():
                    for f in resample_path.iterdir():
                        try:
                            f.unlink() if f.is_file() else shutil.rmtree(f)
                        except OSError as e:
                            self.logger.debug("Could not remove %s: %s", f, e)
            except Exception as e:
                self.logger.debug(
                    "Could not clean resample dir %s: %s", resample_dir, e
                )

        # --- Execute SWarp ---
        result = None
        try:
            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=300,
                )
            self.clean_log(log_file)
        except subprocess.TimeoutExpired:
            with open(log_file, "a") as f:
                f.write("\nERROR: SWarp execution timed out after 300 seconds\n")
            self.logger.warning("SWarp timed out (300 s). See %s", log_file)
            _cleanup_swarp_outputs()
            return None
        except Exception as e:
            log_warning_from_exception(
                self.logger, f"SWarp execution failed (see {log_file})", e
            )
            _cleanup_swarp_outputs()
            return None

        # --- Handle single input case ---
        if len(input_images) == 1:
            resamp_files = list(Path(resample_dir).glob("*.resamp.fits"))
            if resamp_files:
                output_image = str(resamp_files[0])
                if self.verbose_level >= 2:
                    self.logger.info(f"Using resampled file: {output_image}")

        # --- Check for errors ---
        if result.returncode != 0:
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                tail = log_content.strip().splitlines()[-5:] if log_content else []
                self.logger.warning(
                    "SWarp failed (return code %s). Last lines of %s:\n%s",
                    result.returncode,
                    log_file,
                    "\n".join(tail),
                )
            except Exception:
                self.logger.warning(
                    "SWarp failed with return code %s. See %s",
                    result.returncode,
                    log_file,
                )
            _cleanup_swarp_outputs()
            return None

        # ------------------------------------------------------------------
        # Re-impose NaNs on uncovered regions using SWarp weights.
        #
        # Some SWarp builds still emit exact 0.0 in no-data regions even when
        # BLANK_BADPIXELS=Y and FILL_VALUE=NAN. The robust indicator of coverage
        # is the corresponding weight map: weight <= 0 (or non-finite) means the
        # output pixel has no valid contributing input data.
        # ------------------------------------------------------------------
        def _apply_weight_nan_mask(resamp_fits: Path, weight_fits: Path) -> None:
            try:
                if not resamp_fits.is_file() or not weight_fits.is_file():
                    return
                with fits.open(resamp_fits, mode="update", memmap=False) as hdul_img:
                    img = np.asarray(hdul_img[0].data, dtype=float)
                    if img.ndim != 2:
                        return
                    with fits.open(weight_fits, mode="readonly", memmap=False) as hdul_w:
                        w = np.asarray(hdul_w[0].data, dtype=float)
                    if w.shape != img.shape:
                        return
                    bad = (~np.isfinite(w)) | (w <= 0)
                    if not np.any(bad):
                        return
                    n_before = int(np.count_nonzero(~np.isfinite(img)))
                    img[bad] = np.nan
                    hdul_img[0].data = img.astype(np.float32, copy=False)
                    hdul_img.flush()
                    n_after = int(np.count_nonzero(~np.isfinite(img)))
                    if self.verbose_level >= 2:
                        self.logger.debug(
                            "SWarp NaN re-mask: %s using %s (NaN/inf %d -> %d; bad=%d)",
                            str(resamp_fits),
                            str(weight_fits),
                            n_before,
                            n_after,
                            int(np.count_nonzero(bad)),
                        )
            except Exception as e:
                log_warning_from_exception(
                    self.logger,
                    f"Could not re-mask no-data pixels in {resamp_fits.name}",
                    e,
                )

        try:
            resamp_files = list(Path(resample_dir).glob("*.resamp.fits"))
            # Find per-resampled weight files if present; else fall back to SWarp WEIGHTOUT_NAME.
            fallback_weight = Path(output_weights)
            for rf in resamp_files:
                # Common SWarp naming conventions:
                #   image.resamp.fits -> image.resamp.weight.fits
                #   image.resamp.fits -> image.weight.fits
                candidates = [
                    rf.with_name(rf.name.replace(".resamp.fits", ".resamp.weight.fits")),
                    rf.with_name(rf.name.replace(".resamp.fits", ".weight.fits")),
                ]
                # Also allow any weight file sharing the same stem prefix.
                try:
                    candidates += list(rf.parent.glob(rf.stem + "*.weight*.fits"))
                except Exception:
                    pass

                picked = None
                for c in candidates:
                    if c.is_file():
                        picked = c
                        break
                if picked is None and fallback_weight.is_file():
                    picked = fallback_weight
                if picked is not None:
                    _apply_weight_nan_mask(rf, picked)
        except Exception:
            pass

        return {
            "output_dir": str(output_dir),
            "corrected_image": output_image,
            "weight_image": output_weights,
            "resampled_dir": resample_dir,
            "log_file": log_file,
            "config": final_config,
        }

    def run_scamp(
        self,
        catalog_paths,
        reference_cat: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Run SCAMP on one or more LDAC catalogs in a single call.

        Parameters
        ----------
        catalog_paths : str or list of str
            One or more LDAC catalog paths to pass to SCAMP. When multiple
            catalogs are supplied, SCAMP solves them together on the same
            astrometric grid and produces one .head file per catalog.
        reference_cat : str, optional
            Path to the astrometric reference catalog (ASTREF_CATALOG=FILE).
            When None, GAIA-DR3 is used.
        """
        if isinstance(catalog_paths, (str, Path)):
            catalog_paths = [str(catalog_paths)]
        else:
            catalog_paths = [str(p) for p in catalog_paths]

        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        stem = Path(catalog_paths[0]).stem

        # Read scamp_distort_degrees from wcs config for consistency with main pipeline
        iy = getattr(self, "input_yaml", None) or {}
        wcs_cfg = iy.get("wcs", {}) if isinstance(iy, dict) else {}
        distort_degrees = int(wcs_cfg.get("scamp_distort_degrees", 4))

        final_config = {
            **self.DEFAULT_SCAMP_CONFIG,
            "DISTORT_DEGREES": distort_degrees,
            "ASTREF_CATALOG": "FILE" if reference_cat else "GAIA-DR3",
            "ASTREFCAT_NAME": reference_cat,
            "NTHREADS": self.default_threads,
            **(config or {}),
        }
        # COMBINE is a SWarp keyword, not a SCAMP keyword. Do not pass it to SCAMP.
        final_config.pop("COMBINE", None)
        final_config = {k: v for k, v in final_config.items() if v is not None}

        config_file = (
            str(output_path / f"{stem}_default.scamp")
            if output_path
            else f"{stem}_default.scamp"
        )
        with open(config_file, "w") as f:
            for k, v in final_config.items():
                f.write(f"{k}\t{v}\n")

        xml_file = (
            str(output_path / f"{stem}_scamp.xml")
            if output_path
            else f"{stem}_scamp.xml"
        )
        log_file = (
            str(output_path / f"{stem}_scamp.log")
            if output_path
            else f"{stem}_scamp.log"
        )

        cmd = [
            self._check_executable("scamp"),
            *catalog_paths,
            "-c",
            config_file,
            "-XML_NAME",
            xml_file,
            "-NTHREADS",
            str(final_config["NTHREADS"]),
        ]

        if self.verbose_level >= 2:
            import shlex

            self.logger.info(f"Running SCAMP: {shlex.join(cmd)}")

        def _clean_scamp_outputs():
            if output_path:
                for p in [config_file, xml_file, log_file] + list(
                    Path(output_path).glob("*.head")
                ):
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError as e:
                        self.logger.debug("Could not remove SCAMP output %s: %s", p, e)

        try:
            with open(log_file, "w") as log_f:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=90,
                )
            self.clean_log(log_file)
        except subprocess.TimeoutExpired:
            self.logger.error("SCAMP execution timed out after 90 seconds")
            _clean_scamp_outputs()
            raise

        # Check if log file exists and read it
        if not os.path.isfile(log_file):
            self.logger.warning(
                f"SCAMP log file not created at {log_file}; catalog may be invalid"
            )
            _clean_scamp_outputs()
            return None

        with open(log_file) as log_f:
            log_content = log_f.read()

        # Log SCAMP return code and key messages for debugging
        self.logger.info(
            f"SCAMP return code: {result.returncode}, log file: {log_file}"
        )
        if "Not enough matched detections" in log_content:
            self.logger.warning("SCAMP: Not enough matched detections in catalog")
        if "FATAL ERROR" in log_content:
            self.logger.warning(f"SCAMP: FATAL ERROR in log: {log_content[:500]}")

        if result.returncode != 0 or "Not enough matched detections" in log_content:
            self.logger.warning(
                f"SCAMP failed (code {result.returncode}). See {log_file}"
            )
            return None

        distortion = self._parse_scamp_xml(xml_file)

        # Map each input catalog stem to its output .head file.
        # SCAMP names .head files after the catalog stem (same as the FITS stem).
        head_files_found = list(Path(output_path).glob("*.head")) if output_path else []
        head_files_by_stem = {p.stem: str(p) for p in head_files_found}

        # Legacy single-catalog callers expect a "head_file" key.
        head_file = str(head_files_found[0]) if head_files_found else None

        return {
            "output_dir": str(output_path) if output_path else None,
            "xml_file": xml_file,
            "log_file": log_file,
            "head_file": head_file,
            "head_files_by_stem": head_files_by_stem,
            "distortion": distortion,
            "config": final_config,
        }

    def run_scamp_swarp(
        self,
        cat_path,
        ref_cat_path,
        image_path,
        out_dir,
        is_undersampled,
        fwhm_pixels,
        image_type,
        scamp_config,
        swarp_config,
        target_ra=None,
        target_dec=None,
    ):
        """Run SCAMP then SWarp to align an image to a reference.

        Args:
            cat_path: Path to SExtractor catalog for the image to align.
            ref_cat_path: Path to reference catalog.
            image_path: Path to the FITS image.
            out_dir: Output directory for alignment products.
            is_undersampled: Whether the image is undersampled.
            fwhm_pixels: FWHM in pixels.
            image_type: Type of image (e.g., 'science', 'reference').
            scamp_config: SCAMP configuration dict.
            swarp_config: SWarp configuration dict.
            target_ra: Optional target RA (degrees).
            target_dec: Optional target Dec (degrees).
        """
        # Run SCAMP to compute WCS solution for image_path aligned to ref_cat_path.
        # run_scamp signature: (catalog_path, reference_cat, output_dir, config)
        scamp_res = self.run_scamp(
            catalog_path=cat_path,
            reference_cat=ref_cat_path,
            output_dir=str(out_dir),
            config=scamp_config,
        )
        if scamp_res is None:
            self.logger.info("SCAMP failed. Check logs for details.")
            return None
        head_file = Path(scamp_res.get("head_file", ""))

        if not head_file.exists() or head_file.stat().st_size == 0:
            self.logger.info(f"SCAMP did not produce a valid .head file at {head_file}")
            return None

        self.logger.debug("SCAMP produced .head file: %s", head_file)

        # SWarp looks for <stem>.head next to the input FITS to pick up the SCAMP WCS.
        # Copy the .head file to sit next to image_path so SWarp finds it automatically.
        image_head_dst = Path(image_path).with_suffix(".head")
        try:
            if head_file.resolve() != image_head_dst.resolve():
                shutil.copy2(str(head_file), str(image_head_dst))
                self.logger.info(
                    "Copied SCAMP .head to %s for SWarp to use.", image_head_dst
                )
        except Exception as _he:
            log_warning_from_exception(
                self.logger, f"Could not copy SCAMP .head next to {image_path}", _he
            )

        self.logger.debug("Running SWarp on %s", image_path)
        swarp_res = self.run_swarp(
            [str(image_path)],
            scamp_results=scamp_res,
            output_dir=str(out_dir),
            config=swarp_config,
        )
        if swarp_res is None:
            self.logger.info("SWarp failed. Check logs for details.")
            return None
        aligned_image = swarp_res["corrected_image"]
        self.clean_image(Path(aligned_image))
        return aligned_image

    def clean_image(self, path: Path):
        """Replace non-finite values (NaN/inf) in a FITS image with NaN.

        Note: exact zeros are NOT replaced — zero is a valid pixel value on
        background-subtracted images and replacing it with NaN would create
        artificial holes in the data that corrupt downstream subtraction.
        """
        with fits.open(path, mode="update") as hdul:
            data = np.asarray(hdul[0].data, dtype=float)
            data[~np.isfinite(data)] = np.nan
            hdul[0].data = data
            hdul.flush()

    # ------------------------------ Lifecycle -------------------------------
    def cleanup(self) -> None:
        """Remove all temporary directories created by this instance."""
        for temp_dir in list(self._temp_dirs):
            try:
                p = Path(temp_dir)
                if p.exists():
                    import shutil as _shutil

                    _shutil.rmtree(p)
            except Exception as e:
                self.logger.info(
                    f"Failed to remove temporary directory {temp_dir}: {e}"
                )
            finally:
                self._temp_dirs.discard(temp_dir)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------- Matched sources (1<->1) -------------------------
    def filter_well_defined_positions(
        self, catalog: Table, max_position_error_arcsec: float = 3
    ) -> Table:
        """
        Filter SExtractor catalog to keep only sources with well-defined positions.

        Parameters:
        - catalog: SExtractor catalog (astropy Table).
        - max_position_error_arcsec: Maximum allowed positional error in arcseconds.

        Returns:
        - filtered_catalog: Table with only sources that have well-defined positions.
        """
        good = np.ones(len(catalog), dtype=bool)
        if "ERRA_WORLD" in catalog.colnames and "ERRB_WORLD" in catalog.colnames:
            positional_error = (
                np.sqrt(catalog["ERRA_WORLD"] ** 2 + catalog["ERRB_WORLD"] ** 2) * 3600
            )
            good &= positional_error <= max_position_error_arcsec
        return catalog[good]

    def plot_matched_sources_side_by_side(
        self,
        sci_image_path: str,
        ref_image_path: str,
        sci_cat_path: str,
        ref_cat_path: str,
        output_plot_path: str = "matched_sources_side_by_side.png",
        label_color: str = "#FF0000",
        label_fontsize: int = 8,
        figsize: Optional[tuple] = None,
        cmap: str = "viridis",
        draw_lines: bool = True,
        line_color: str = "#FF0000",
        line_alpha: float = 0.5,
        line_width: float = 0.5,
        circle_radius_sci: float = 5.0,
        circle_radius_ref: float = 5.0,
        matched_circle_color: str = "#FF0000",
        unmatched_circle_color: str = "blue",
        circle_alpha: float = 0.7,
        circle_edge_color: str = "none",
        circle_color: str = "none",
        circle_edge_width: float = 0.5,
        max_sources: int = 100,
        **imshow_kwargs,
    ):
        try:
            import gc

            gc.collect()
            if figsize is None:
                from functions import set_size

                # Original default was (12, 6): height/width = 0.5
                # set_size uses height/width = aspect / golden_ratio
                golden_ratio = (5**0.5 + 1) / 2
                figsize = set_size(540, aspect=0.5 * golden_ratio)

            with (
                fits.open(sci_cat_path) as sci_hdul,
                fits.open(ref_cat_path) as ref_hdul,
            ):
                sci_cat = Table(sci_hdul[2].data)
                ref_cat = Table(ref_hdul[2].data)
            with (
                fits.open(sci_image_path) as sci_hdul,
                fits.open(ref_image_path) as ref_hdul,
            ):
                sci_data = sci_hdul[0].data.astype(np.float32)
                ref_data = ref_hdul[0].data.astype(np.float32)
            # Track if we rebin and the scale factor
            rebin_scale = 1.0
            if sci_data.size > 5e6:
                from astropy.nddata import block_reduce

                sci_data = block_reduce(sci_data, block_size=(4, 4), func=np.mean)
                ref_data = block_reduce(ref_data, block_size=(4, 4), func=np.mean)
                rebin_scale = 0.25  # Coordinates must be scaled by 1/4
                logging.info(f"Rebinned images by 4x for display, coordinates will be scaled by {rebin_scale}")
            
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, constrained_layout=True
            )
            zscale_sci = ZScaleInterval()
            zscale_ref = ZScaleInterval()
            vmin_sci, vmax_sci = zscale_sci.get_limits(sci_data)
            vmin_ref, vmax_ref = zscale_ref.get_limits(ref_data)
            cmap_img = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
            try:
                cmap_img = cmap_img.copy()
            except Exception:
                pass
            try:
                cmap_img.set_bad(color="white")
            except Exception:
                pass
            im1 = ax1.imshow(
                sci_data,
                cmap=cmap_img,
                vmin=vmin_sci,
                vmax=vmax_sci,
                **imshow_kwargs,
                origin="lower",
            )
            ax1.set_title("Science Image")
            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Science counts", fontsize=7)
            cbar1.ax.tick_params(labelsize=6)
            ax1.set_xlabel("X [Pixel]")
            ax1.set_ylabel("Y [Pixel]")
            im2 = ax2.imshow(
                ref_data,
                cmap=cmap_img,
                vmin=vmin_ref,
                vmax=vmax_ref,
                **imshow_kwargs,
                origin="lower",
            )
            ax2.set_title("Reference Image")
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Reference counts", fontsize=7)
            cbar2.ax.tick_params(labelsize=6)
            ax2.set_xlabel("X [Pixel]")
            ax2.set_ylabel("Y [Pixel]")

            def int_to_label(i):
                if i < 26:
                    return chr(65 + i)
                else:
                    return f"{chr(65 + (i // 26) - 1)}{chr(65 + (i % 26))}"

            sci_positions = []
            sci_h, sci_w = sci_data.shape
            for i, row in enumerate(sci_cat[:max_sources]):
                if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames:
                    x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                    # Scale coordinates if image was rebinned, then convert to 0-based
                    x_scaled = x * rebin_scale
                    y_scaled = y * rebin_scale
                    x_0based, y_0based = x_scaled - 1, y_scaled - 1
                    
                    # Validate coordinates are within image bounds
                    if not (0 <= x_0based < sci_w and 0 <= y_0based < sci_h):
                        logging.debug(f"Science source {i} out of bounds: ({x_0based:.1f}, {y_0based:.1f}) vs image ({sci_w}, {sci_h})")
                        continue
                    
                    sci_positions.append((x_0based, y_0based))
                    color = (
                        matched_circle_color
                        if "MATCH_ID" in row.colnames
                        else unmatched_circle_color
                    )
                    circle = Circle(
                        (x_0based, y_0based),
                        circle_radius_sci * rebin_scale,  # Scale circle radius too
                        facecolor=circle_color,
                        alpha=circle_alpha,
                        edgecolor=color,
                        linewidth=circle_edge_width,
                    )
                    ax1.add_patch(circle)
                    ax1.text(
                        x_0based,
                        y_0based - circle_radius_sci * rebin_scale - 2,
                        int_to_label(i),
                        color=label_color,
                        fontsize=label_fontsize,
                        ha="center",
                        va="top",
                    )
            
            ref_positions = []
            ref_h, ref_w = ref_data.shape
            for i, row in enumerate(ref_cat[:max_sources]):
                if "XWIN_IMAGE" in row.colnames and "YWIN_IMAGE" in row.colnames:
                    x, y = row["XWIN_IMAGE"], row["YWIN_IMAGE"]
                    # Scale coordinates if image was rebinned, then convert to 0-based
                    x_scaled = x * rebin_scale
                    y_scaled = y * rebin_scale
                    x_0based, y_0based = x_scaled - 1, y_scaled - 1
                    
                    # Validate coordinates are within image bounds
                    if not (0 <= x_0based < ref_w and 0 <= y_0based < ref_h):
                        logging.debug(f"Reference source {i} out of bounds: ({x_0based:.1f}, {y_0based:.1f}) vs image ({ref_w}, {ref_h})")
                        continue
                    
                    ref_positions.append((x_0based, y_0based))
                    color = (
                        matched_circle_color
                        if "MATCH_ID" in row.colnames
                        else unmatched_circle_color
                    )
                    circle = Circle(
                        (x_0based, y_0based),
                        circle_radius_ref * rebin_scale,  # Scale circle radius too
                        facecolor=circle_color,
                        alpha=circle_alpha,
                        edgecolor=color,
                        linewidth=circle_edge_width,
                    )
                    ax2.add_patch(circle)
                    ax2.text(
                        x_0based,
                        y_0based - circle_radius_ref * rebin_scale - 2,
                        int_to_label(i),
                        color=label_color,
                        fontsize=label_fontsize,
                        ha="center",
                        va="top",
                    )
            
            logging.info(f"Plotted {len(sci_positions)} science and {len(ref_positions)} reference sources within image bounds")
            # `constrained_layout=True` keeps colorbars and labels from
            # overlapping, so no additional tight_layout is needed.
            plt.savefig(output_plot_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            gc.collect()
        except Exception as e:
            import traceback

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            logging.info(
                f"\n\n Issue creating side by side plot \n"
                f"\n"
                f"Type     : {exc_type.__name__}\n"
                f"File     : {fname}\n"
                f"Line     : {lineno}\n"
                f"Message  : {str(e)}\n"
                f"\n" + traceback.format_exc()
            )

    def filter_matched_sources(
        self,
        sci_cat_path: str,
        ref_cat_path: str,
        match_radius_arcsec: float = 5.0,
        nmax: int = 1e6,
        sci_image_path: str = None,
        ref_image_path: str = None,
    ) -> tuple[int, float]:
        """
        Build 1-to-1 matched catalogs using SNR_APER >= 3, adding MAG_APER, MAGERR_APER, and SNR_APER columns.
        Uses RANSAC for alignment matching and spatial downsampling if too many sources.
        Prioritizes extended sources for alignment.
        Returns the number of matched sources and the match radius used.
        """
        if not all([sci_cat_path, ref_cat_path, sci_image_path, ref_image_path]):
            raise ValueError(
                "All paths (sci_cat_path, ref_cat_path, sci_image_path, ref_image_path) are required."
            )
        nmax = int(nmax)
        if nmax <= 0:
            raise ValueError("nmax must be positive.")

        def read_ldac(path: str) -> tuple[Table, fits.Header]:
            """Read LDAC catalog and return table and header."""
            with fits.open(path) as hdul:
                return Table(hdul[2].data), hdul[2].header

        def write_ldac(path: str, table: Table, header: fits.Header) -> None:
            """Write LDAC catalog with updated table."""
            with fits.open(path, mode="update") as hdul:
                hdul[2].data = table.as_array()
                hdul.flush()

        class ConstrainedSlopeRegressor(BaseEstimator, RegressorMixin):
            def __init__(
                self, slope_constraint: float = 1.0, slope_tolerance: float = 0.0
            ):
                self.slope_constraint = slope_constraint
                self.slope_tolerance = slope_tolerance

            def fit(self, X, y):
                X, y = check_X_y(X, y)

                def loss(params):
                    slope, intercept = params
                    pred = slope * X.flatten() + intercept
                    mse = np.mean((y - pred) ** 2)
                    penalty = 100 * max(
                        0, abs(slope - self.slope_constraint) - self.slope_tolerance
                    )
                    return mse + penalty

                init = [1.0, float(np.mean(y) - np.mean(X))]
                result = minimize(loss, init, method="L-BFGS-B")
                self.slope_, self.intercept_ = result.x
                return self

            def predict(self, X):
                check_is_fitted(self)
                X = check_array(X)
                return self.slope_ * X.flatten() + self.intercept_

        sci_cat, sci_header = read_ldac(sci_cat_path)
        ref_cat, ref_header = read_ldac(ref_cat_path)

        # Initialize intercept before conditional blocks to prevent NameError
        intercept = None

        def add_mag_snr_aper(tbl: Table, use_magauto: bool = True) -> Table:
            """
            Add magnitude, magnitude error, and SNR columns.
            Optionally use MAG_AUTO, MAGERR_AUTO, and SNR_WIN if use_magauto is True.
            """
            if use_magauto and "MAG_AUTO" in tbl.colnames:
                # Use MAG_AUTO, MAGERR_AUTO, and SNR_WIN if available and requested
                mag = np.array(tbl["MAG_AUTO"], float)
                magerr = (
                    np.array(tbl["MAGERR_AUTO"], float)
                    if "MAGERR_AUTO" in tbl.colnames
                    else 0.02
                )
                snr = (
                    np.array(tbl["SNR_WIN"], float)
                    if "SNR_WIN" in tbl.colnames
                    else 0.0
                )
                tbl["MAG_APER"], tbl["MAGERR_APER"], tbl["SNR_APER"] = mag, magerr, snr
            elif "FLUX_APER" in tbl.colnames:
                # Fall back to FLUX_APER and FLUXERR_APER
                flux = np.array(tbl["FLUX_APER"], float)
                flux_err = (
                    np.array(tbl["FLUXERR_APER"], float)
                    if "FLUXERR_APER" in tbl.colnames
                    else np.sqrt(np.abs(flux))
                )
                mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
                magerr = np.where(
                    (flux > 0) & (flux_err > 0),
                    2.5 / np.log(10) * (flux_err / flux),
                    0.02,
                )
                snr = np.where(flux_err > 0, np.abs(flux) / flux_err, 0.0)
                tbl["MAG_APER"], tbl["MAGERR_APER"], tbl["SNR_APER"] = mag, magerr, snr
            else:
                # If neither is available, fill with NaN and default values
                tbl["MAG_APER"], tbl["MAGERR_APER"], tbl["SNR_APER"] = np.nan, 0.02, 0.0
            return tbl

        sci_cat = add_mag_snr_aper(sci_cat)
        ref_cat = add_mag_snr_aper(ref_cat)
        sci_mask = sci_cat["SNR_APER"] >= 1.0
        ref_mask = ref_cat["SNR_APER"] >= 1.0
        sci_cat_filtered = sci_cat[sci_mask]
        ref_cat_filtered = ref_cat[ref_mask]
        if len(sci_cat_filtered) == 0 or len(ref_cat_filtered) == 0:
            sci_cat_filtered["MATCH_ID"] = np.arange(len(sci_cat_filtered), dtype=int)
            ref_cat_filtered["MATCH_ID"] = np.arange(len(ref_cat_filtered), dtype=int)
            write_ldac(sci_cat_path, sci_cat_filtered, sci_header)
            write_ldac(ref_cat_path, ref_cat_filtered, ref_header)
            return 0, match_radius_arcsec

        def get_xy(
            tbl: Table,
            prefer_win: bool = True,
            input_origin: int = 1,
            output_origin: int = 1,
        ) -> tuple[np.ndarray, np.ndarray, str]:
            """Get coordinates and convert between origin systems."""
            if (
                prefer_win
                and "XWIN_IMAGE" in tbl.colnames
                and "YWIN_IMAGE" in tbl.colnames
            ):
                x, y = tbl["XWIN_IMAGE"], tbl["YWIN_IMAGE"]
                coord_type = "WIN"
            elif "X_IMAGE" in tbl.colnames and "Y_IMAGE" in tbl.colnames:
                x, y = tbl["X_IMAGE"], tbl["Y_IMAGE"]
                coord_type = "ISO"
            else:
                raise ValueError(
                    "Neither XWIN_IMAGE/YWIN_IMAGE nor X_IMAGE/Y_IMAGE found in catalog."
                )
            if input_origin != output_origin:
                offset = input_origin - output_origin
                x = x - offset
                y = y - offset
            return x, y, coord_type

        sci_x, sci_y, sci_coord_type = get_xy(
            sci_cat_filtered, prefer_win=True, input_origin=1, output_origin=1
        )
        ref_x, ref_y, ref_coord_type = get_xy(
            ref_cat_filtered, prefer_win=True, input_origin=1, output_origin=1
        )

        def get_ra_dec(tbl: Table) -> tuple:
            if "XWIN_WORLD" in tbl.colnames and "YWIN_WORLD" in tbl.colnames:
                return tbl["XWIN_WORLD"], tbl["YWIN_WORLD"]
            if "X_WORLD" in tbl.colnames and "Y_WORLD" in tbl.colnames:
                return tbl["X_WORLD"], tbl["Y_WORLD"]
            raise ValueError(
                "Neither XWIN_WORLD/YWIN_WORLD nor X_WORLD/Y_WORLD found in catalog."
            )

        sci_ra, sci_dec = get_ra_dec(sci_cat_filtered)
        ref_ra, ref_dec = get_ra_dec(ref_cat_filtered)
        sci_coords = SkyCoord(sci_ra * u.deg, sci_dec * u.deg)
        ref_coords = SkyCoord(ref_ra * u.deg, ref_dec * u.deg)
        idx_sci, idx_ref, _, _ = search_around_sky(
            sci_coords, ref_coords, match_radius_arcsec * u.arcsec
        )
        best_pairs, used_ref = {}, set()
        for si, ri in zip(idx_sci, idx_ref):
            dist = sci_coords[si].separation(ref_coords[ri]).arcsec
            if si not in best_pairs or dist < best_pairs[si][1]:
                best_pairs[si] = (ri, dist)
        final_pairs = []
        for si, (ri, dist) in best_pairs.items():
            if ri not in used_ref and dist <= match_radius_arcsec:
                final_pairs.append((si, ri))
                used_ref.add(ri)
        if final_pairs:
            sci_idx, ref_idx = map(np.array, zip(*final_pairs))
            sci_cat_matched = sci_cat_filtered[sci_idx]
            ref_cat_matched = ref_cat_filtered[ref_idx]
            match_id = np.arange(len(sci_cat_matched), dtype=int)
            sci_cat_matched["MATCH_ID"] = match_id
            ref_cat_matched["MATCH_ID"] = match_id
        else:
            sci_cat_matched = sci_cat_filtered[:0]
            ref_cat_matched = ref_cat_filtered[:0]
            sci_cat_matched["MATCH_ID"] = np.array([], dtype=int)
            ref_cat_matched["MATCH_ID"] = np.array([], dtype=int)
        # Only apply RANSAC mag filtering if we have many initial matches (>= 50)
        # For fewer matches, use all matched sources to avoid over-filtering
        if len(sci_cat_matched) >= 50:
            sci_mag = np.array(sci_cat_matched["MAG_APER"], dtype=float)
            ref_mag = np.array(ref_cat_matched["MAG_APER"], dtype=float)
            valid = np.isfinite(sci_mag) & np.isfinite(ref_mag)
            if np.sum(valid) >= 10:
                sci_mag_clean = sci_mag[valid]
                ref_mag_clean = ref_mag[valid]
                intercept = None
                try:
                    # Use RANSAC for alignment matching
                    # residual_threshold=2.0 mag is more permissive to handle different zero points
                    ransac = RANSACRegressor(
                        estimator=ConstrainedSlopeRegressor(
                            slope_constraint=1.0, slope_tolerance=0.0
                        ),
                        residual_threshold=2.0,
                        max_trials=500,
                        min_samples=2,
                    )
                    ransac.fit(ref_mag_clean.reshape(-1, 1), sci_mag_clean)
                    intercept = ransac.estimator_.intercept_
                    inlier_mask_clean = ransac.inlier_mask_

                    if inlier_mask_clean.any():
                        inliers = np.zeros(len(sci_cat_matched), dtype=bool)
                        inliers[valid] = inlier_mask_clean
                        sci_cat_matched = sci_cat_matched[inliers]
                        ref_cat_matched = ref_cat_matched[inliers]
                        match_id = np.arange(len(sci_cat_matched), dtype=int)
                        sci_cat_matched["MATCH_ID"] = match_id
                        ref_cat_matched["MATCH_ID"] = match_id
                        self.logger.info(
                            f"RANSAC mag filter kept {len(sci_cat_matched)} sources, zp={intercept:.3f} mag"
                        )
                except Exception as exc:
                    self.logger.warning(f"RANSAC regression failed: {exc}")
                    # Fall back to simple median filtering with more permissive threshold
                    median_diff = np.median(sci_mag_clean - ref_mag_clean)
                    intercept = median_diff
                    inlier_mask_clean = np.abs((sci_mag_clean - ref_mag_clean) - median_diff) < 1.0
                    if inlier_mask_clean.any():
                        inliers = np.zeros(len(sci_cat_matched), dtype=bool)
                        inliers[valid] = inlier_mask_clean
                        sci_cat_matched = sci_cat_matched[inliers]
                        ref_cat_matched = ref_cat_matched[inliers]
                        match_id = np.arange(len(sci_cat_matched), dtype=int)
                        sci_cat_matched["MATCH_ID"] = match_id
                        ref_cat_matched["MATCH_ID"] = match_id
                        self.logger.info(
                            f"Median fallback kept {len(sci_cat_matched)} sources"
                        )
        else:
            # For fewer than 50 initial matches, skip RANSAC to avoid over-filtering
            self.logger.info(
                f"Skipping RANSAC mag filter (only {len(sci_cat_matched)} initial matches < 50 threshold)"
            )
            # Calculate simple intercept for plotting
            sci_mag = np.array(sci_cat_matched["MAG_APER"], dtype=float)
            ref_mag = np.array(ref_cat_matched["MAG_APER"], dtype=float)
            valid = np.isfinite(sci_mag) & np.isfinite(ref_mag)
            if np.sum(valid) >= 2:
                intercept = np.median(sci_mag[valid] - ref_mag[valid])
            else:
                intercept = None

        if len(sci_cat_matched) >= 5 and intercept is not None:
            from functions import set_size
            from plotting_utils import get_color, get_marker_size, get_alpha, get_line_width

            fig, ax = plt.subplots(figsize=set_size(340, 1))
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            ax.errorbar(
                ref_cat_matched["MAG_APER"],
                sci_cat_matched["MAG_APER"],
                xerr=ref_cat_matched["MAGERR_APER"],
                yerr=sci_cat_matched["MAGERR_APER"],
                fmt="o",
                markersize=get_marker_size('medium'),
                mfc="none",
                mec=get_color('inliers'),
                ecolor=get_color('inliers'),
                alpha=get_alpha('dark'),
                label=f"Inliers [{len(sci_cat_matched)}]",
            )
            x_fit = np.linspace(
                np.min(ref_cat_matched["MAG_APER"]),
                np.max(ref_cat_matched["MAG_APER"]),
                100,
            )
            # Slope is forced to 1.0, intercept is the zeropoint
            y_fit = x_fit + intercept
            ax.plot(
                x_fit,
                y_fit,
                color=get_color('fit'),
                linestyle="--",
                lw=get_line_width('medium'),
                label=f"Fit: slope=1.000, intercept={intercept:.3f}",
            )
            # Add shaded error region (calculate from residuals)
            sci_mag = np.array(sci_cat_matched["MAG_APER"], dtype=float)
            ref_mag = np.array(ref_cat_matched["MAG_APER"], dtype=float)
            residuals = sci_mag - (ref_mag + intercept)
            residual_std = np.std(residuals)
            n_points = len(sci_mag)
            intercept_error = residual_std / np.sqrt(n_points)
            ax.fill_between(
                x_fit,
                y_fit - intercept_error,
                y_fit + intercept_error,
                color=get_color('error_region'),
                alpha=get_alpha('light'),
            )
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xlabel("Reference MAG_APER")
            ax.set_ylabel("Science MAG_APER")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)
            ax.grid(True)
            fig.savefig(
                sci_cat_path.replace(".cat", "_Mag_Fit.png"), dpi=150
            )
            plt.close(fig)

        if len(sci_cat_matched) > nmax:
            n_bins = int(np.sqrt(nmax))
            spp = max(1, nmax // (n_bins**2))
            sci_x_bin, sci_y_bin, _ = get_xy(
                sci_cat_matched, prefer_win=True, input_origin=1, output_origin=1
            )
            x_bins = np.linspace(np.min(sci_x_bin), np.max(sci_x_bin), n_bins + 1)
            y_bins = np.linspace(np.min(sci_y_bin), np.max(sci_y_bin), n_bins + 1)
            sel = []
            for i in range(n_bins):
                for j in range(n_bins):
                    m = (
                        (sci_x_bin >= x_bins[i])
                        & (sci_x_bin < x_bins[i + 1])
                        & (sci_y_bin >= y_bins[j])
                        & (sci_y_bin < y_bins[j + 1])
                    )
                    idx = np.where(m)[0]
                    if idx.size:
                        snr_vals = sci_cat_matched["SNR_APER"][idx]
                        idx_sorted = idx[np.argsort(snr_vals)[::-1]]
                        sel.extend(idx_sorted[:spp])
            sel = np.array(sorted(set(sel)), dtype=int)[:nmax]
            sci_cat_matched = sci_cat_matched[sel]
            ref_cat_matched = ref_cat_matched[sel]
            match_id = np.arange(len(sci_cat_matched), dtype=int)
            sci_cat_matched["MATCH_ID"] = match_id
            ref_cat_matched["MATCH_ID"] = match_id
        write_ldac(sci_cat_path, sci_cat_matched, sci_header)
        write_ldac(ref_cat_path, ref_cat_matched, ref_header)
        self.logger.info(
            f"Final matched sources: {len(sci_cat_matched)}, match radius={match_radius_arcsec:.1f} arcsec"
        )
        return len(sci_cat_matched), match_radius_arcsec
