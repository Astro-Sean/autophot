#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized and robust WCS solver and FITS header cleaner.
Handles faint sources, cosmic rays, and edge cases gracefully.
Uses SExtractor with a standard Gaussian convolution filter for source detection by default,
with fallback to astrometry.net-only mode.
"""

# --- Standard Library Imports ---
import os
import re
import glob
import shutil
import logging
import warnings
import subprocess
import numpy as np
import tempfile
from pathlib import Path
from contextlib import contextmanager

# --- Third-Party Imports ---
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning

# --- Local Imports ---
from functions import border_msg, get_header, get_image

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Suppress Astropy Warnings ---
warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

# --- Compiled regex for log cleaning (reuse) ---
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# =============================================================================
# Helper Functions
# =============================================================================


@contextmanager
def silence_astropy_wcs_info():
    """
    Context manager to silence Astropy WCS info-level logs.
    Useful for reducing log clutter during WCS operations.
    """
    wcs_logger = logging.getLogger("astropy.wcs.wcs")
    prev_level = wcs_logger.level
    wcs_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        wcs_logger.setLevel(prev_level)


def update_wcs_center(
    fits_path: str, ra: float, dec: float, overwrite: bool = True
) -> fits.Header:
    """
    Updates the WCS header of a FITS file to center the reference pixel and set the reference sky coordinates.
    Uses FITS 1-indexed convention: reference pixel at image center is (NAXIS+1)/2.

    Args:
        fits_path (str): Path to the FITS file.
        ra (float): Right Ascension for the reference pixel (degrees).
        dec (float): Declination for the reference pixel (degrees).
        overwrite (bool): Overwrite the file if it exists.

    Returns:
        fits.Header: Updated header, or None on failure.
    """
    try:
        with fits.open(fits_path, mode="update") as hdul:
            header = hdul[0].header
            image = hdul[0].data
            naxis1 = header.get("NAXIS1", image.shape[1])
            naxis2 = header.get("NAXIS2", image.shape[0])
            # FITS center pixel (1-indexed) is (naxis+1)/2
            header["CRPIX1"] = (naxis1 + 1) / 2.0
            header["CRPIX2"] = (naxis2 + 1) / 2.0
            header["CRVAL1"] = ra
            header["CRVAL2"] = dec
            hdul.flush()
            logger.info(
                "WCS header updated: CRPIX1=%.2f CRPIX2=%.2f CRVAL1=%.6f CRVAL2=%.6f",
                header["CRPIX1"],
                header["CRPIX2"],
                ra,
                dec,
            )
            return header
    except Exception as e:
        logger.exception(f"Failed to update WCS header: {e}")
        return None


def table_to_ldac(table, header=None, writeto=None) -> fits.HDUList:
    """
    Converts an Astropy table into the LDAC format.

    Args:
        table (astropy.table.Table): Table to convert.
        header (fits.Header): Header to include in the output.
        writeto (str): Path to save the output.

    Returns:
        fits.HDUList: HDUList in LDAC format.
    """
    primary_hdu = fits.PrimaryHDU()
    header_str = header.tostring(endcard=True)
    header_str += fits.Header().tostring(endcard=True)
    header_col = fits.Column(
        name="Field Header Card", format=f"{len(header_str)}A", array=[header_str]
    )
    header_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([header_col]))
    header_hdu.header["EXTNAME"] = "LDAC_IMHEAD"
    data_hdu = fits.table_to_hdu(table)
    data_hdu.header["EXTNAME"] = "LDAC_OBJECTS"
    hdulist = fits.HDUList([primary_hdu, header_hdu, data_hdu])
    if writeto is not None:
        hdulist.writeto(writeto, overwrite=True)
    return hdulist


def get_wcs(header: fits.Header) -> WCS:
    """
    Create a WCS object from a FITS header, handling SIP if present.
    Returns a 2D celestial WCS so it is safe for reproject and other 2D image operations.

    Args:
        header (fits.Header): FITS header.

    Returns:
        WCS: 2D celestial WCS object.
    """
    sip_keywords = ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]
    if any(k in header for k in sip_keywords):
        header = header.copy()
        for ctype in ["CTYPE1", "CTYPE2"]:
            if ctype not in header or not str(header.get(ctype, "")).endswith("-SIP"):
                header[ctype] = f"{header.get(ctype, 'RA---TAN')}-SIP"
    with warnings.catch_warnings():
        with silence_astropy_wcs_info():
            wcs = WCS(header, fix=True, relax=True)
    # Ensure 2D celestial WCS so reproject and callers get consistent pixel grid
    naxis = getattr(wcs.wcs, "naxis", 2)
    if naxis > 2:
        wcs = wcs.celestial
    return wcs


def load_background_std(background_std) -> np.ndarray:
    """
    Load background standard deviation from a FITS file or use a 2D array.
    If None, return None.

    Args:
        background_std (str, np.ndarray, or None): Background standard deviation.

    Returns:
        np.ndarray or None: Background standard deviation.
    """
    if background_std is None:
        return None
    if isinstance(background_std, str):
        with fits.open(background_std) as hdul:
            return hdul[0].data
    if isinstance(background_std, np.ndarray):
        return background_std
    raise TypeError(
        "background_std must be a string (FITS filepath), a 2D numpy array, or None."
    )


def create_conv_file(path: str, fwhm_pixels: float = 3.0, force: bool = False) -> None:
    """
    Create a convolution kernel for SExtractor, optimized for the given FWHM.
    Skips writing if the file already exists unless force=True.
    Args:
        path: Path to save the convolution file.
        fwhm_pixels: FWHM in pixels, used to optimize the kernel size.
        force: If True, overwrite even when the file exists.
    """
    if not force and os.path.isfile(path):
        return
    kernel_size = max(3, int(np.ceil(fwhm_pixels * 3)))
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
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


def create_nnw_file(path: str) -> None:
    """
    Create a default SExtractor neural network weights file for star/galaxy classification.
    """
    nnw_text = """
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
    nnw_text = "\n".join(line.strip() for line in nnw_text.split("\n") if line.strip())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(nnw_text)


# =============================================================================
# Main Class
# =============================================================================


class WCSSolver:
    """
    WCS solver and FITS header cleaner.
    Handles faint sources, cosmic rays, and edge cases gracefully.
    """

    def __init__(
        self, fpath: str, image: np.ndarray, header: fits.Header, default_input: dict
    ):
        """
        Initialize the WCS checker and solver.

        Args:
            fpath (str): Path to the FITS file.
            image (np.ndarray): Image data.
            header (fits.Header): FITS header.
            default_input (dict): Default input parameters.
        """
        self.fpath = fpath
        self.header = header
        self.image = image
        self.default_input = default_input

    # --- Remove WCS Keys ---
    def remove_wcs_keys(self, delete_keys: bool = True) -> fits.Header:
        """
        Removes or renames specific WCS-related keywords from the FITS header.

        Args:
            delete_keys (bool): If True, delete the keys; otherwise, rename them.

        Returns:
            fits.Header: Updated header.
        """
        logger.info("Removing any pre-existing WCS keys")
        keywords = {
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CUNIT1",
            "CUNIT2",
            "CTYPE1",
            "CTYPE2",
            "WCSAXES",
            "EQUINOX",
            "LONPOLE",
            "LATPOLE",
            "CDELT1",
            "CDELT2",
            "A_ORDER",
            "A_0_0",
            "A_0_1",
            "A_0_2",
            "A_1_0",
            "A_1_1",
            "A_2_0",
            "B_ORDER",
            "B_0_0",
            "B_0_1",
            "B_0_2",
            "B_1_0",
            "B_1_1",
            "B_2_0",
            "AP_ORDER",
            "AP_0_0",
            "AP_0_1",
            "AP_0_2",
            "AP_1_0",
            "AP_1_1",
            "AP_2_0",
            "BP_ORDER",
            "BP_0_0",
            "BP_0_1",
            "BP_0_2",
            "BP_1_0",
            "BP_1_1",
            "BP_2_0",
            "PROJP1",
            "PROJP3",
            "RADECSYS",
            "PV1_1",
            "PV1_2",
            "PV2_1",
            "PV2_2",
            "LTV1",
            "LTV2",
            "LTM1_1",
            "LTM2_2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
            "RADESYS",
            "PV1_0",
            "PV1_1",
            "PV1_2",
            "PV1_3",
            "PV1_4",
            "PV1_5",
            "PV1_6",
            "PV1_7",
            "PV1_8",
            "PV1_9",
            "PV1_10",
            "PV1_11",
            "PV1_12",
            "PV1_13",
            "PV1_14",
            "PV1_15",
            "PV1_16",
            "PV1_17",
            "PV1_18",
            "PV1_19",
            "PV1_20",
            "PV1_21",
            "PV1_22",
            "TNX_0_0",
            "TNX_1_0",
            "TNX_0_1",
            "TNX_2_0",
            "TNX_1_1",
            "TNX_0_2",
            "TNX_3_0",
            "TNX_2_1",
            "TNX_1_2",
            "TNX_0_3",
            "TNX_4_0",
            "TNX_3_1",
            "TNX_2_2",
            "TNX_1_3",
            "TNX_0_4",
            "TNX_5_0",
            "TNX_4_1",
            "TNX_3_2",
            "TNX_2_3",
            "TNX_1_4",
            "TNX_0_5",
            "TNX_6_0",
            "TNX_5_1",
            "TNX_4_2",
            "TNX_3_3",
            "TNX_2_4",
            "TNX_1_5",
            "TNX_0_6",
            "PC001001",
            "PC001002",
            "PC002001",
            "PC002002",
            "A_1_1",
            "A_1_2",
            "A_1_3",
            "A_2_0",
            "A_2_1",
            "A_2_2",
            "A_2_3",
            "A_3_0",
            "A_3_1",
            "A_3_2",
            "A_3_3",
            "B_1_1",
            "B_1_2",
            "B_1_3",
            "B_2_0",
            "B_2_1",
            "B_2_2",
            "B_2_3",
            "B_3_0",
            "B_3_1",
            "B_3_2",
            "B_3_3",
            "SIP_A",
            "SIP_B",
            "SIP_C",
            "SIP_D",
            "SIP_AP",
            "SIP_BP",
            "SIP_CP",
            "SIP_DP",
        }
        for key in keywords:
            try:
                if key in self.header:
                    if delete_keys:
                        del self.header[key]
                    else:
                        new_key = f"_{key[1:]}" if len(key) > 1 else f"_{key}"
                        self.header.rename_keyword(key, new_key)
            except Exception as e:
                logger.exception(f"Error handling key '{key}': {e}")
        return self.header

    # --- Clean Log ---
    def clean_log(self, input_file: str, output_file: str = None) -> str:
        """
        Clean ANSI escape sequences and extra lines from log files.

        Args:
            input_file (str): Path to the input log file.
            output_file (str): Path to save the cleaned log file.

        Returns:
            str: Path to the cleaned log file.
        """
        with open(input_file, "r") as f:
            content = f.read()
        clean_content = _ANSI_ESCAPE.sub("", content)
        clean_content = "\n".join(
            line for line in clean_content.split("\n") if line.strip()
        )
        output_file = output_file or input_file
        with open(output_file, "w") as f:
            f.write(clean_content)
        return output_file

    def _run_solve_field(
        self,
        args: list,
        wcs_file: str,
        timeout_sec: float,
        logpath: str,
    ) -> bool:
        """Run solve-field with given args; return True if wcs_file was created."""
        try:
            with open(logpath, "a", encoding="utf-8") as logf:
                logf.write(" ".join(map(str, args)) + "\n")
                kwargs = dict(shell=False, stdout=logf, stderr=subprocess.STDOUT)
                if os.name != "nt":
                    kwargs["preexec_fn"] = os.setsid
                pro = subprocess.Popen(args, **kwargs)
            try:
                pro.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(os.getpgid(pro.pid), 9)
                    except (OSError, AttributeError):
                        pro.kill()
                else:
                    pro.kill()
                pro.wait()
                logger.warning("solve-field exceeded timeout (%s s)", timeout_sec)
            self.clean_log(logpath)
            return os.path.isfile(wcs_file)
        except Exception as e:
            logger.warning("solve-field run failed: %s", e)
            return False

    def _plate_solve_scamp(
        self, wcs_cfg: dict, cpulimit: float | None = None
    ) -> fits.Header | float:
        """
        Attempt WCS solution using SExtractor + SCAMP.

        Returns a FITS header with updated WCS on success, or np.nan on failure.
        """
        logger.info(border_msg("Solving for WCS with SCAMP"))

        scamp_exe = wcs_cfg.get("scamp_exe_loc") or shutil.which("scamp")
        sex_exe = wcs_cfg.get("sextractor_exe_loc") or shutil.which("sex")
        if not scamp_exe and shutil.which("scamp") is None:
            logger.warning("SCAMP executable not found; cannot use SCAMP solver.")
            return np.nan
        if not sex_exe and shutil.which("sex") is None:
            logger.warning("SExtractor executable not found; cannot use SCAMP solver.")
            return np.nan
        scamp_exe = str(
            scamp_exe
            if scamp_exe and os.path.isfile(str(scamp_exe))
            else shutil.which("scamp")
        )
        sex_exe = str(
            sex_exe if sex_exe and os.path.isfile(str(sex_exe)) else shutil.which("sex")
        )

        if not os.path.isfile(self.fpath):
            logger.error("Image file not found: %s", self.fpath)
            return np.nan

        timeout_sec = (
            float(cpulimit)
            if cpulimit is not None
            else float(wcs_cfg.get("cpulimit", 60))
        )

        dirname = os.path.dirname(self.fpath)
        base = os.path.splitext(os.path.basename(self.fpath))[0]
        scamp_log = os.path.join(dirname, f"scamp_{base}.log")

        with tempfile.TemporaryDirectory() as temp_dir:
            param_file = os.path.join(temp_dir, "scamp.param")
            params = [
                "XWIN_IMAGE",
                "YWIN_IMAGE",
                "FLUX_AUTO",
                "FLUXERR_AUTO",
                "MAG_AUTO",
                "MAGERR_AUTO",
                "FLUX_RADIUS",
                "FWHM_IMAGE",
                "CLASS_STAR",
                "ELLIPTICITY",
                "BACKGROUND",
                "THRESHOLD",
                "FLAGS",
                "SNR_WIN",
                "XPEAK_IMAGE",
                "YPEAK_IMAGE",
            ]
            Path(param_file).write_text("\n".join(params))

            nnw_file = os.path.join(temp_dir, "default.nnw")
            create_nnw_file(nnw_file)
            config_file = os.path.join(temp_dir, "scamp.sex")

            pixel_scale = self.default_input.get("pixel_scale") or 0
            if not pixel_scale and self.header.get("CDELT1") is not None:
                try:
                    pixel_scale = abs(float(self.header["CDELT1"])) * 3600.0
                except (TypeError, KeyError):
                    pass
            if not pixel_scale:
                logger.warning(
                    "SCAMP: no pixel_scale in config or header; using 0.1--5 arcsec/pix"
                )

            gain_val = (
                self.header.get("GAIN")
                or self.header.get("gain")
                or self.default_input.get("gain")
            )
            gain_str = str(float(gain_val)) if gain_val is not None else "0"

            detect_thresh = str(float(wcs_cfg.get("sextractor_detect_thresh", 1.5)))
            final_config = {
                "DETECT_TYPE": "CCD",
                "DETECT_MINAREA": str(int(wcs_cfg.get("sextractor_detect_minarea", 5))),
                "DETECT_THRESH": detect_thresh,
                "ANALYSIS_THRESH": str(
                    float(wcs_cfg.get("sextractor_analysis_thresh", 1.2))
                ),
                "DEBLEND_NTHRESH": str(
                    int(wcs_cfg.get("sextractor_deblend_nthresh", 32))
                ),
                "DEBLEND_MINCONT": str(
                    float(wcs_cfg.get("sextractor_deblend_mincont", 0.005))
                ),
                "FILTER": "Y",
                "FILTER_NAME": os.path.join(
                    os.path.dirname(__file__), "gaussian_7x7.conv"
                ),
                "CLEAN": "Y",
                "CLEAN_PARAM": "1",
                "PHOT_AUTOPARAMS": "2.5,3.5",
                "GAIN": gain_str,
                "PIXEL_SCALE": str(pixel_scale),
                "VERBOSE_TYPE": "NORMAL",
                "CATALOG_TYPE": "FITS_LDAC",
                "PARAMETERS_NAME": param_file,
                "STARNNW_NAME": nnw_file,
            }
            with open(config_file, "w") as f:
                for k, v in final_config.items():
                    f.write(f"{k}\t{v}\n")

            cat_path = os.path.join(temp_dir, f"{base}.ldac")
            sex_cmd = [
                sex_exe,
                self.fpath,
                "-c",
                config_file,
                "-CATALOG_NAME",
                cat_path,
            ]
            try:
                logger.info(
                    "Running SExtractor for SCAMP: %s", " ".join(map(str, sex_cmd))
                )
                subprocess.run(
                    sex_cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "SExtractor for SCAMP exceeded timeout (%.1f s)", timeout_sec
                )
                return np.nan
            if not os.path.isfile(cat_path):
                logger.warning(
                    "SCAMP: SExtractor catalog not created; aborting SCAMP solve."
                )
                return np.nan

            head_path = os.path.join(temp_dir, f"{base}.head")
            astref_cat = str(wcs_cfg.get("scamp_astref_catalog", "GAIA-DR3"))
            scamp_cmd = [
                scamp_exe,
                cat_path,
                "-ASTREF_CATALOG",
                astref_cat,
                "-CHECKPLOT_TYPE",
                "NONE",
                "-WRITE_XML",
                "N",
                "-SAVE_TYPE",
                "HEAD",
                "-HEAD_NAME",
                head_path,
            ]
            try:
                logger.info("Running SCAMP: %s", " ".join(map(str, scamp_cmd)))
                with open(scamp_log, "w", encoding="utf-8") as logf:
                    subprocess.run(
                        scamp_cmd,
                        check=False,
                        stdout=logf,
                        stderr=subprocess.STDOUT,
                        timeout=timeout_sec,
                    )
            except subprocess.TimeoutExpired:
                logger.warning("SCAMP exceeded timeout (%.1f s)", timeout_sec)
                return np.nan

            if not os.path.isfile(head_path):
                logger.warning("SCAMP did not produce a .head file; WCS not updated.")
                return np.nan

            try:
                wcs_header = fits.Header.fromtextfile(head_path)
                from functions import remove_wcs_from_header

                self.header = remove_wcs_from_header(self.header)
                _wcs_prefixes = (
                    "CRPIX",
                    "CRVAL",
                    "CTYPE",
                    "CD",
                    "PC",
                    "CDELT",
                    "CROTA",
                    "PV",
                    "LONPOLE",
                    "LATPOLE",
                    "EQUINOX",
                    "WCSNAME",
                    "CUNIT",
                    "WCSAXES",
                    "PROJP",
                    "LTV",
                    "LTM",
                    "RADECSYS",
                    "RADESYS",
                    "RADYSYS",
                    "LONGPOLE",
                    "TNX",
                    "SIP_",
                )
                _wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
                for key in wcs_header:
                    if key in ("NAXIS", "NAXIS1", "NAXIS2"):
                        continue
                    is_wcs = any(key.startswith(p) for p in _wcs_prefixes)
                    if not is_wcs and "_" in key:
                        stem = key.split("_")[0] + "_"
                        is_wcs = stem in _wcs_stems and key.startswith(stem.rstrip("_"))
                    if is_wcs:
                        self.header[key] = wcs_header[key]
                self.header["NAXIS1"] = self.image.shape[1]
                self.header["NAXIS2"] = self.image.shape[0]
                logger.info("SCAMP WCS solution applied to header.")
                return self.header
            except Exception as exc:
                logger.warning("Failed to apply SCAMP WCS header: %s", exc)
                return np.nan

    # --- Plate Solve ---
    def plate_solve(
        self,
        solvefield_exe: str = None,
        downsample: int = None,
        cpulimit: int = None,
        background_std: str = None,
        skip_verify: bool = True,
    ) -> fits.Header:
        """
        Run a WCS solver for this image.

        When wcs.solver is "scamp" and SCAMP/sex are available, use SCAMP
        (with SExtractor) to derive the WCS. Otherwise, fall back to
        Astrometry.net solve-field.
        """
        wcs_cfg = self.default_input.get("wcs") or {}
        solver = str(wcs_cfg.get("solver", "astrometry")).strip().lower()

        # Optional SCAMP path
        if solver == "scamp":
            try:
                header = self._plate_solve_scamp(wcs_cfg=wcs_cfg, cpulimit=cpulimit)
                if isinstance(header, fits.Header):
                    return header
                logger.warning(
                    "SCAMP solve failed or unavailable; falling back to Astrometry.net."
                )
            except Exception as exc:
                logger.warning(
                    "SCAMP solve raised an exception (%s); falling back to Astrometry.net.",
                    exc,
                )

        logger.info(
            border_msg(
                f"Solving for WCS with Astrometry.net ({os.path.basename(self.fpath)})"
            )
        )
        if not solvefield_exe:
            logger.warning(
                "Astrometry.net 'solve-field' executable not found; skipping solve-field WCS step. "
                "To install with conda:\n"
                "  conda install -c conda-forge astrometry\n"
                "Then ensure 'solve-field' is on PATH or set wcs.solve_field_exe_loc in your YAML."
            )
            return np.nan
        if not os.path.isfile(self.fpath):
            logger.error("Image file not found: %s", self.fpath)
            return np.nan

        # Re-fetch in case SCAMP path modified config
        wcs_cfg = self.default_input.get("wcs") or {}
        # Optional profile-based overrides (useful for crowded fields).
        # Set in YAML as:
        #   wcs:
        #     profile: crowded
        #     crowded: { ... overrides ... }
        try:
            profile = str(wcs_cfg.get("profile", "default")).strip().lower()
        except Exception:
            profile = "default"
        if profile == "crowded":
            overrides = wcs_cfg.get("crowded") or {}
            if isinstance(overrides, dict) and overrides:
                merged = dict(wcs_cfg)
                merged.update(overrides)
                wcs_cfg = merged
                logger.info("WCS profile=crowded: applying crowded-field overrides")
        timeout_sec = float(
            cpulimit if cpulimit is not None else wcs_cfg.get("cpulimit", 45)
        )
        downsample = (
            int(downsample)
            if downsample is not None
            else int(wcs_cfg.get("downsample", 2))
        )

        # --- Create Gaussian convolution filter file if it doesn't exist ---
        conv_filter_path = os.path.join(os.path.dirname(__file__), "gaussian_7x7.conv")
        create_conv_file(conv_filter_path)

        # --- Prepare paths (fixed temp names to avoid collisions) ---
        dirname = os.path.dirname(self.fpath)
        base = os.path.splitext(os.path.basename(self.fpath))[0]
        wcs_file = os.path.join(dirname, "astrometry_temp.wcs.fits")
        astrometry_log_fpath = os.path.join(dirname, f"astrometry_{base}.log")
        use_sextractor = shutil.which("sex") is not None
        # use_sextractor = False
        # --- Create a temporary SExtractor config file ---
        with tempfile.TemporaryDirectory() as temp_dir:
            param_file = os.path.join(temp_dir, "default.param")
            params = [
                "XWIN_IMAGE",
                "YWIN_IMAGE",
                "FLUX_AUTO",
                "FLUXERR_AUTO",
                "MAG_AUTO",
                "MAGERR_AUTO",
                "FLUX_RADIUS",
                "FWHM_IMAGE",
                "CLASS_STAR",
                "ELLIPTICITY",
                "BACKGROUND",
                "THRESHOLD",
                "FLAGS",
                "SNR_WIN",
                "XPEAK_IMAGE",
                "YPEAK_IMAGE",
            ]
            Path(param_file).write_text("\n".join(params))
            nnw_file = os.path.join(temp_dir, "default.nnw")
            create_nnw_file(nnw_file)
            config_file = os.path.join(temp_dir, "default.sex")
            pixel_scale = self.default_input.get("pixel_scale") or 0
            if not pixel_scale and self.header.get("CDELT1") is not None:
                try:
                    pixel_scale = (
                        abs(float(self.header["CDELT1"])) * 3600.0
                    )  # deg to arcsec/pix
                except (TypeError, KeyError):
                    pass
            if not pixel_scale:
                logger.warning(
                    "No pixel_scale in config or header; using 0.1--5 arcsec/pix"
                )
            # SExtractor: use header gain when available for better centroid weighting
            gain_val = (
                self.header.get("GAIN")
                or self.header.get("gain")
                or self.default_input.get("gain")
            )
            gain_str = str(float(gain_val)) if gain_val is not None else "0"
            detect_thresh = str(float(wcs_cfg.get("sextractor_detect_thresh", 1.5)))
            final_config = {
                "DETECT_TYPE": "CCD",
                "DETECT_MINAREA": str(int(wcs_cfg.get("sextractor_detect_minarea", 5))),
                "DETECT_THRESH": detect_thresh,
                "ANALYSIS_THRESH": str(
                    float(wcs_cfg.get("sextractor_analysis_thresh", 1.2))
                ),
                # Deblending helps split overlapping sources in crowded fields.
                "DEBLEND_NTHRESH": str(
                    int(wcs_cfg.get("sextractor_deblend_nthresh", 32))
                ),
                "DEBLEND_MINCONT": str(
                    float(wcs_cfg.get("sextractor_deblend_mincont", 0.005))
                ),
                "FILTER": "Y",
                "FILTER_NAME": conv_filter_path,
                "CLEAN": "Y",
                "CLEAN_PARAM": "1",
                "PHOT_AUTOPARAMS": "2.5,3.5",
                "GAIN": gain_str,
                "PIXEL_SCALE": str(pixel_scale),
                "VERBOSE_TYPE": "NORMAL",
                "CATALOG_TYPE": "FITS_LDAC",
                "PARAMETERS_NAME": param_file,
                "STARNNW_NAME": nnw_file,
            }
            with open(config_file, "w") as f:
                for k, v in final_config.items():
                    f.write(f"{k}\t{v}\n")

            # --- Prepare SExtractor command ---
            sextractor_cmd = None
            if use_sextractor:
                logger.info(
                    "Using SExtractor with Gaussian convolution filter for source detection."
                )
                sextractor_cmd = f"sex -c {config_file}"
            else:
                logger.warning("SExtractor not found. Proceeding without.")

            # --- Scale bounds: use range around known pixel scale when available ---
            # A wrong pixel_scale can prevent convergence. We therefore build two sets:
            #   - constrained: around the hint (if present)
            #   - wide: 0.1--5.0 arcsec/pix (no hint)
            ra, dec = self.default_input.get("target_ra"), self.default_input.get(
                "target_dec"
            )
            radius_deg = float(wcs_cfg.get("search_radius", 0.5))

            def _build_scale_args(
                scale_low: float, scale_high: float, include_pos: bool
            ) -> list[str]:
                out = [
                    "--scale-units",
                    "arcsecperpix",
                    "--scale-low",
                    str(scale_low),
                    "--scale-high",
                    str(scale_high),
                ]
                if background_std is not None:
                    out += ["--sigma", str(background_std)]
                if include_pos and ra is not None and dec is not None:
                    out += [
                        "--ra",
                        str(ra),
                        "--dec",
                        str(dec),
                        "--radius",
                        str(radius_deg),
                    ]
                return out

            if pixel_scale and pixel_scale > 0:
                scale_low_hint = pixel_scale * 0.6
                scale_high_hint = pixel_scale * 1.5
            else:
                scale_low_hint, scale_high_hint = 0.1, 5.0

            scale_low_wide, scale_high_wide = 0.1, 5.0

            scale_args_constrained = _build_scale_args(
                scale_low_hint, scale_high_hint, include_pos=True
            )
            scale_only_args_constrained = _build_scale_args(
                scale_low_hint, scale_high_hint, include_pos=False
            )
            scale_args_wide = _build_scale_args(
                scale_low_wide, scale_high_wide, include_pos=True
            )
            scale_only_args_wide = _build_scale_args(
                scale_low_wide, scale_high_wide, include_pos=False
            )

            # Start with constrained args (if pixel_scale provided); we will drop the hint on retry.
            scale_args = scale_args_constrained
            scale_only_args = scale_only_args_constrained

            # # --- solve-field matching and depth ---
            # code_tolerance = float(wcs_cfg.get("code_tolerance", 0.01))
            # scale_args += ["--code-tolerance", str(code_tolerance)]

            # --- Downsample: lighten astrometry.net load for large images ---
            ny, nx = self.image.shape[0], self.image.shape[1]
            if nx * ny < 1500 * 1500:
                downsample = max(1, downsample - 1)
            downsample = max(1, int(downsample))

            if os.path.exists(astrometry_log_fpath):
                os.remove(astrometry_log_fpath)
            new_fits_temp = os.path.join(dirname, "astrometry_newfits_temp.fits")
            scamp_temp = os.path.join(dirname, "astrometry_temp_scamp")
            corr_temp = os.path.join(dirname, "astrometry_temp.corr")
            rdl_temp = os.path.join(dirname, "astrometry_temp.rdls")
            match_temp = os.path.join(dirname, "astrometry_temp.match")
            solved_temp = os.path.join(dirname, "astrometry_temp.solved")
            common_args = [
                str(solvefield_exe),
                "--no-remove-lines",
                "--overwrite",
                "--downsample",
                str(downsample),
                "--new-fits",
                new_fits_temp,
                "--wcs",
                str(wcs_file),
                "--scamp",
                scamp_temp,
                "--corr",
                corr_temp,
                "--rdls",
                rdl_temp,
                "--match",
                match_temp,
                "--solved",
                solved_temp,
                "--no-plots",
                "--cpulimit",
                str(int(timeout_sec)),
                str(self.fpath),
            ]
            # Limit the number of sources used by solve-field (helpful in very crowded images).
            objs = wcs_cfg.get("objs", None)
            if objs is not None:
                try:
                    common_args.insert(-1, "--objs")
                    common_args.insert(-1, str(int(objs)))
                except Exception:
                    pass
            # Reference pixel at image center improves consistency with pipelines
            common_args.insert(-1, "--crpix-center")
            # Tweak order: try moderate SIP orders first (2, 1, 0)
            tweak_orders = [2, 1, 0]

            # --- Step 1: Optional verify existing WCS ---
            if not skip_verify:
                logger.info("Attempting to verify WCS with --no-verify")
                args_verify = common_args + ["--no-verify"] + scale_args
                if use_sextractor and sextractor_cmd:
                    args_verify += [
                        "--use-source-extractor",
                        "--source-extractor-path",
                        sextractor_cmd,
                        "--x-column",
                        "XWIN_IMAGE",
                        "--y-column",
                        "YWIN_IMAGE",
                        "--sort-column",
                        "MAG_AUTO",
                    ]
                if self._run_solve_field(
                    args_verify, wcs_file, timeout_sec, astrometry_log_fpath
                ):
                    logger.info("WCS verified successfully")
                else:
                    logger.warning("Could not verify WCS - proceeding to blind solve")

            def _attempt_solve_with_args(scale_args_use: list[str], label: str) -> None:
                # Step 2: Solve with (optional) SExtractor, tweak orders 2,1,0
                for tweak_order in tweak_orders:
                    if os.path.isfile(wcs_file):
                        break
                    logger.info(
                        "Attempting WCS solve (%s) with tweak order %s",
                        label,
                        tweak_order,
                    )
                    args = (
                        common_args
                        + ["--tweak-order", str(tweak_order), "--no-verify"]
                        + scale_args_use
                    )
                    if use_sextractor and sextractor_cmd:
                        # XWIN_IMAGE/YWIN_IMAGE = weighted centroid, more accurate than X_IMAGE/Y_IMAGE
                        args += [
                            "--use-source-extractor",
                            "--source-extractor-path",
                            sextractor_cmd,
                            "--x-column",
                            "XWIN_IMAGE",
                            "--y-column",
                            "YWIN_IMAGE",
                            "--sort-column",
                            "MAG_AUTO",
                            "--sort-ascending",
                        ]
                    if self._run_solve_field(
                        args, wcs_file, timeout_sec, astrometry_log_fpath
                    ):
                        logger.info(
                            "WCS solved (%s) with tweak order %s", label, tweak_order
                        )
                        break
                    logger.warning(
                        "No solution (%s) with tweak order %s", label, tweak_order
                    )

            # First attempt: use the pixel-scale hint when available
            _attempt_solve_with_args(
                scale_args,
                label=(
                    "with pixel-scale hint"
                    if (pixel_scale and pixel_scale > 0)
                    else "wide scale"
                ),
            )

            # If still not solved and we had a pixel-scale hint, drop it and retry once.
            if not os.path.isfile(wcs_file) and (pixel_scale and pixel_scale > 0):
                logger.warning(
                    "WCS not solved with pixel-scale hint (%.4g arcsec/pix). Dropping pixel scale constraint and retrying with 0.1--5 arcsec/pix.",
                    float(pixel_scale),
                )
                scale_args = scale_args_wide
                scale_only_args = scale_only_args_wide
                _attempt_solve_with_args(
                    scale_args, label="wide scale (dropped pixel-scale hint)"
                )

            # --- Step 3: Retry without SExtractor ---
            if not os.path.isfile(wcs_file):
                logger.warning("Retrying without SExtractor")
                for tweak_order in tweak_orders:
                    if os.path.isfile(wcs_file):
                        break
                    logger.info(
                        "Attempting WCS solve without SExtractor (tweak order %s)",
                        tweak_order,
                    )
                    args_no_sex = (
                        common_args
                        + ["--tweak-order", str(tweak_order), "--no-verify"]
                        + scale_args
                    )
                    if self._run_solve_field(
                        args_no_sex, wcs_file, timeout_sec, astrometry_log_fpath
                    ):
                        logger.info(
                            "WCS solved without SExtractor (tweak order %s)",
                            tweak_order,
                        )
                        break
                    logger.warning(
                        "No solution without SExtractor (tweak order %s)", tweak_order
                    )

            # --- Step 4: If position-constrained solve failed, retry blind (no RA/Dec/radius) ---
            if not os.path.isfile(wcs_file) and ra is not None and dec is not None:
                logger.warning("Retrying with blind solve (no position constraint).")
                for tweak_order in tweak_orders:
                    if os.path.isfile(wcs_file):
                        break
                    logger.info("Blind solve with tweak order %s", tweak_order)
                    args_blind = (
                        common_args
                        + ["--tweak-order", str(tweak_order), "--no-verify"]
                        + scale_only_args
                    )
                    if use_sextractor and sextractor_cmd:
                        args_blind += [
                            "--use-source-extractor",
                            "--source-extractor-path",
                            sextractor_cmd,
                            "--x-column",
                            "XWIN_IMAGE",
                            "--y-column",
                            "YWIN_IMAGE",
                            "--sort-column",
                            "MAG_AUTO",
                            "--sort-ascending",
                        ]
                    if self._run_solve_field(
                        args_blind, wcs_file, timeout_sec, astrometry_log_fpath
                    ):
                        logger.info(
                            "WCS solved with blind solve (tweak order %s)", tweak_order
                        )
                        break
                    logger.warning(
                        "No solution with blind solve (tweak order %s)", tweak_order
                    )

            # --- If WCS still not solved, try SCAMP as fallback (e.g. crowded or IR fields) ---
            if not os.path.isfile(wcs_file):
                logger.warning(
                    "Astrometry.net did not solve WCS; trying SCAMP as fallback."
                )
                try:
                    scamp_header = self._plate_solve_scamp(
                        wcs_cfg=wcs_cfg, cpulimit=cpulimit
                    )
                    if isinstance(scamp_header, fits.Header):
                        logger.info("SCAMP fallback succeeded.")
                        return scamp_header
                except Exception as exc:
                    logger.warning("SCAMP fallback failed: %s", exc)

            # --- If still not solved, return NaN ---
            if not os.path.isfile(wcs_file):
                for f in [new_fits_temp, scamp_temp, os.path.join(dirname, "none")]:
                    if os.path.isfile(f):
                        os.remove(f)
                logger.warning("\tCould not solve WCS - returning NaN")
                return np.nan

            # --- Clean up temporary files ---
            for pattern in [".corr", ".axy", ".match", ".rdls", ".solved", ".xyls"]:
                for f in glob.glob(os.path.join(dirname, f"*{pattern}")):
                    os.remove(f)
            for f in [new_fits_temp, scamp_temp, os.path.join(dirname, "none")]:
                if os.path.isfile(f):
                    os.remove(f)

            # --- Update FITS header with WCS from the .wcs.fits file ---
            try:
                with fits.open(wcs_file) as wcs_hdul:
                    wcs_header = wcs_hdul[0].header
                from functions import remove_wcs_from_header

                # Remove all previous WCS from science header, then add only WCS keywords from solver
                self.header = remove_wcs_from_header(self.header)
                _wcs_prefixes = (
                    "CRPIX",
                    "CRVAL",
                    "CTYPE",
                    "CD",
                    "PC",
                    "CDELT",
                    "CROTA",
                    "PV",
                    "LONPOLE",
                    "LATPOLE",
                    "EQUINOX",
                    "WCSNAME",
                    "CUNIT",
                    "WCSAXES",
                    "PROJP",
                    "LTV",
                    "LTM",
                    "RADECSYS",
                    "RADESYS",
                    "RADYSYS",
                    "LONGPOLE",
                    "TNX",
                    "SIP_",
                )
                _wcs_stems = ("A_", "B_", "AP_", "BP_", "D_", "DP_", "PV_")
                for key in wcs_header:
                    if key in ("NAXIS", "NAXIS1", "NAXIS2"):
                        continue
                    is_wcs = any(key.startswith(p) for p in _wcs_prefixes)
                    if not is_wcs and "_" in key:
                        stem = key.split("_")[0] + "_"
                        is_wcs = stem in _wcs_stems and key.startswith(stem.rstrip("_"))
                    if is_wcs:
                        self.header[key] = wcs_header[key]
                self.header["NAXIS1"] = self.image.shape[1]
                self.header["NAXIS2"] = self.image.shape[0]

                # Astrometry.net solve-field outputs WCS in standard FITS 1-based convention,
                # so no CRPIX offset is needed by default. Optional crpix_offset (e.g. 0.5 or -0.5)
                # only if your solver uses a different pixel convention.
                crpix_offset = wcs_cfg.get("crpix_offset")
                if crpix_offset is None and wcs_cfg.get("crpix_fits_convention", False):
                    crpix_offset = 0.5  # legacy: old configs that applied +0.5
                if crpix_offset is None:
                    crpix_offset = 0.0
                if crpix_offset != 0.0:
                    with silence_astropy_wcs_info():
                        try:
                            tmp_wcs = get_wcs(self.header)
                            crpix1 = float(self.header["CRPIX1"])
                            crpix2 = float(self.header["CRPIX2"])
                            rd = tmp_wcs.all_pix2world([[crpix1]], [[crpix2]], 1)
                            self.header["CRPIX1"] = crpix1 + crpix_offset
                            self.header["CRPIX2"] = crpix2 + crpix_offset
                            self.header["CRVAL1"] = float(rd[0].flat[0])
                            self.header["CRVAL2"] = float(rd[1].flat[0])
                            logger.debug(
                                "Applied CRPIX offset %.2f for solver convention",
                                crpix_offset,
                            )
                        except Exception as e:
                            logger.warning("CRPIX offset correction skipped: %s", e)

                # Validate merged WCS before writing
                with silence_astropy_wcs_info():
                    merged_wcs = get_wcs(self.header)
                    try:
                        from astropy.wcs import utils as wcs_utils

                        scale_val = wcs_utils.proj_plane_pixel_scales(merged_wcs)[0]
                        scale = (
                            float(scale_val * 3600)
                            if hasattr(scale_val, "value")
                            else float(scale_val * 3600)
                        )  # arcsec/pix
                        if not np.isfinite(scale) or scale <= 0 or scale > 3600:
                            logger.warning(
                                "Merged WCS has invalid pixel scale (%.6f); rejecting",
                                scale,
                            )
                            return np.nan
                    except Exception as ev:
                        logger.warning("Could not validate merged WCS: %s", ev)

                fits.writeto(
                    self.fpath,
                    self.image,
                    self.header,
                    overwrite=True,
                    output_verify="ignore",
                )
                if os.path.isfile(wcs_file):
                    os.remove(wcs_file)
                logger.info("WCS information updated in the FITS header")
                return self.header
            except Exception as e:
                for f in [
                    new_fits_temp,
                    scamp_temp,
                    wcs_file,
                    os.path.join(dirname, "none"),
                ]:
                    if f and os.path.isfile(f):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                logger.exception("Failed to update header from WCS file: %s", e)
                return np.nan
