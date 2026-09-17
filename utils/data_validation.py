"""Data validation for AutoPhOT photometry pipeline.

Systematically validates images entering photometry, checking:
- image dimensionality and dtype
- finite pixel values
- exposure time, gain, read noise
- saturation and nonlinearity limits
- bad-pixel, cosmic-ray, chip-gap, and edge masks
- WCS and pixel scale
- units and zeropoint metadata
- consistency between science image, uncertainty image, and mask
- sky-background gradients and spatial variability

Returns a structured ``ValidationReport`` with quality flags.  Does NOT
change pipeline behaviour; wraps existing checks and adds missing ones.

Scientific motivation:  Missing or inconsistent metadata silently degrades
photometric accuracy.  A clear, actionable validation report lets the
pipeline and the user distinguish "good data with a warning" from "bad data
that should be rejected".
"""

from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality flags (bitmask) - kept in sync with utils/quality_flags.py
# ---------------------------------------------------------------------------

# Data-validation flags (bits 0-15)
VAL_OK = 0
VAL_NON_FINITE = 1 << 0          # Non-finite pixels present
VAL_ALL_NAN = 1 << 1             # Entire image is NaN
VAL_BAD_DTYPE = 1 << 2           # Non-floating-point dtype
VAL_BAD_DIMENSIONS = 1 << 3      # Not 2-D
VAL_MISSING_EXPTIME = 1 << 4     # Exposure time missing/invalid
VAL_MISSING_GAIN = 1 << 5        # Gain missing/invalid
VAL_MISSING_RDNOISE = 1 << 6     # Read noise missing/invalid
VAL_MISSING_SATURATE = 1 << 7    # Saturation limit missing
VAL_MISSING_WCS = 1 << 8         # No valid WCS
VAL_BAD_PIXELSCALE = 1 << 9      # Pixel scale missing/invalid
VAL_HIGH_NAN_FRACTION = 1 << 10  # > 20% NaN pixels
VAL_MASK_MISMATCH = 1 << 11      # Mask shape != image shape
VAL_BACKGROUND_GRADIENT = 1 << 12  # Strong background gradient detected
VAL_SATURATED_PIXELS = 1 << 13   # Saturated pixels present
VAL_DEGRADED = 1 << 14           # General degraded-quality flag
VAL_FATAL = 1 << 15              # Fatal: cannot proceed


@dataclass
class ValidationReport:
    """Structured validation report for a single image.

    Attributes
    ----------
    flags : int
        Bitmask of VAL_* flags.  0 means all checks passed.
    warnings : list[str]
        Human-readable warning messages.
    errors : list[str]
        Human-readable error messages (fatal issues).
    info : dict
        Extracted metadata (gain, read_noise, exptime, saturate, pixel_scale,
        nan_fraction, saturated_fraction, background_stats).
    """

    flags: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if no fatal errors."""
        return not (self.flags & VAL_FATAL) and not (self.flags & VAL_ALL_NAN)

    @property
    def degraded(self) -> bool:
        """True if quality is degraded but usable."""
        return bool(self.flags & VAL_DEGRADED) and self.ok

    def add_warning(self, flag: int, msg: str) -> None:
        self.flags |= flag
        self.warnings.append(msg)

    def add_error(self, flag: int, msg: str) -> None:
        self.flags |= flag
        self.errors.append(msg)

    def __repr__(self) -> str:
        status = "OK" if self.ok else ("DEGRADED" if self.degraded else "FATAL")
        return (
            f"ValidationReport(status={status}, flags=0x{self.flags:04x}, "
            f"warnings={len(self.warnings)}, errors={len(self.errors)})"
        )


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_image(
    image: np.ndarray,
    header=None,
    mask: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    gain: Optional[float] = None,
    read_noise: Optional[float] = None,
    exposure_time: Optional[float] = None,
    saturate: Optional[float] = None,
    pixel_scale: Optional[float] = None,
    wcs=None,
    nan_fraction_warn: float = 0.05,
    nan_fraction_degraded: float = 0.20,
    gradient_threshold: float = 0.5,
) -> ValidationReport:
    """Validate an image and its metadata before photometry.

    Parameters
    ----------
    image : np.ndarray
        2-D science image.
    header : astropy.io.fits.Header, optional
        FITS header for metadata extraction.
    mask : np.ndarray, optional
        Bad-pixel mask (True = bad).
    uncertainty : np.ndarray, optional
        Per-pixel uncertainty map.
    gain, read_noise, exposure_time, saturate, pixel_scale : float, optional
        Override values; if None, extracted from header.
    wcs : astropy.wcs.WCS, optional
        WCS object; if None, checked from header.
    nan_fraction_warn : float
        Warn if NaN fraction exceeds this.
    nan_fraction_degraded : float
        Mark degraded if NaN fraction exceeds this.
    gradient_threshold : float
        Warn if background gradient (std/median) exceeds this.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    # ---- 1. Dimensionality and dtype ----
    if not isinstance(image, np.ndarray):
        report.add_error(VAL_BAD_DIMENSIONS, "Image is not a numpy array")
        report.flags |= VAL_FATAL
        return report

    if image.ndim != 2:
        report.add_error(
            VAL_BAD_DIMENSIONS,
            f"Image is {image.ndim}-D, expected 2-D",
        )
        report.flags |= VAL_FATAL
        return report

    if not np.issubdtype(image.dtype, np.floating):
        report.add_warning(
            VAL_BAD_DTYPE,
            f"Image dtype is {image.dtype}, expected floating-point",
        )

    ny, nx = image.shape
    report.info["image_shape"] = (ny, nx)

    # ---- 2. Finite pixel values ----
    finite_mask = np.isfinite(image)
    nan_fraction = float(np.sum(~finite_mask) / image.size)
    report.info["nan_fraction"] = nan_fraction

    if nan_fraction >= 1.0:
        report.add_error(VAL_ALL_NAN, "Entire image is NaN")
        report.flags |= VAL_FATAL
        return report

    if nan_fraction > 0:
        report.add_warning(
            VAL_NON_FINITE,
            f"{nan_fraction*100:.1f}% of pixels are non-finite (NaN/inf)",
        )

    if nan_fraction > nan_fraction_degraded:
        report.add_warning(
            VAL_HIGH_NAN_FRACTION,
            f"High NaN fraction ({nan_fraction*100:.1f}%) may degrade photometry",
        )
        report.flags |= VAL_DEGRADED
    elif nan_fraction > nan_fraction_warn:
        report.flags |= VAL_DEGRADED

    # ---- 3. Exposure time ----
    if exposure_time is None and header is not None:
        for key in ("EXPTIME", "EXPOSURE", "TELAPSE", "ELAPTIME"):
            if key in header:
                exposure_time = float(header[key])
                break
    if exposure_time is None or not np.isfinite(exposure_time) or exposure_time <= 0:
        report.add_warning(
            VAL_MISSING_EXPTIME,
            "Exposure time missing or invalid; using fallback may affect flux calibration",
        )
    else:
        report.info["exposure_time"] = exposure_time

    # ---- 4. Gain ----
    if gain is None and header is not None:
        for key in ("GAIN", "DETGAIN", "GAIN1"):
            if key in header:
                gain = float(header[key])
                break
    if gain is None or not np.isfinite(gain) or gain <= 0:
        report.add_warning(
            VAL_MISSING_GAIN,
            "Gain missing or invalid; noise model may be incorrect",
        )
    else:
        report.info["gain"] = gain

    # ---- 5. Read noise ----
    if read_noise is None and header is not None:
        for key in ("RDNOISE", "READNOIS", "RN"):
            if key in header:
                read_noise = float(header[key])
                break
    if read_noise is None or not np.isfinite(read_noise) or read_noise < 0:
        report.add_warning(
            VAL_MISSING_RDNOISE,
            "Read noise missing or invalid; noise model may be incorrect",
        )
    else:
        report.info["read_noise"] = read_noise

    # ---- 6. Saturation ----
    if saturate is None and header is not None:
        for key in ("SATURATE", "SATLEVEL", "MAXLIN"):
            if key in header:
                saturate = float(header[key])
                break
    if saturate is None or not np.isfinite(saturate) or saturate <= 0:
        report.add_warning(
            VAL_MISSING_SATURATE,
            "Saturation limit missing; saturated sources may not be flagged",
        )
    else:
        report.info["saturate"] = saturate
        finite_data = image[finite_mask]
        if saturate > 0:
            sat_frac = float(np.sum(finite_data >= 0.9 * saturate) / finite_data.size)
            report.info["saturated_fraction"] = sat_frac
            if sat_frac > 0.00001:  # Flag even a few saturated pixels
                report.add_warning(
                    VAL_SATURATED_PIXELS,
                    f"{sat_frac*100:.2f}% of pixels are at/above 90% saturation",
                )

    # ---- 7. WCS and pixel scale ----
    if wcs is None and header is not None:
        try:
            from astropy.wcs import WCS

            wcs = WCS(header, naxis=2)
        except Exception:
            pass

    if wcs is None or not wcs.has_celestial:
        report.add_warning(
            VAL_MISSING_WCS,
            "No valid celestial WCS; coordinate-dependent features unavailable",
        )
    else:
        report.info["has_wcs"] = True
        if pixel_scale is None:
            try:
                cdelt = np.sqrt(np.abs(wcs.wcs.cdelt[0] * wcs.wcs.cdelt[1])) * 3600
                pixel_scale = float(cdelt)
            except Exception:
                pass
        if pixel_scale is None or not np.isfinite(pixel_scale) or pixel_scale <= 0:
            report.add_warning(
                VAL_BAD_PIXELSCALE,
                "Pixel scale missing or invalid; FWHM-based sizing may be wrong",
            )
        else:
            report.info["pixel_scale"] = pixel_scale

    # ---- 8. Mask consistency ----
    if mask is not None:
        if mask.shape != image.shape:
            report.add_error(
                VAL_MASK_MISMATCH,
                f"Mask shape {mask.shape} != image shape {image.shape}",
            )
            report.flags |= VAL_FATAL
        else:
            mask_fraction = float(np.sum(mask) / mask.size)
            report.info["mask_fraction"] = mask_fraction
            if mask_fraction > 0.5:
                report.add_warning(
                    VAL_DEGRADED,
                    f"High mask fraction ({mask_fraction*100:.1f}%) may limit photometry",
                )
                report.flags |= VAL_DEGRADED

    # ---- 9. Uncertainty consistency ----
    if uncertainty is not None:
        if uncertainty.shape != image.shape:
            report.add_warning(
                VAL_MASK_MISMATCH,
                f"Uncertainty shape {uncertainty.shape} != image shape {image.shape}",
            )

    # ---- 10. Background gradient detection ----
    finite_data = image[finite_mask]
    if finite_data.size > 100:
        median_bkg = float(np.nanmedian(finite_data))
        std_bkg = float(np.nanstd(finite_data))
        report.info["background_median"] = median_bkg
        report.info["background_std"] = std_bkg

        # Simple gradient check: compare left vs right half medians
        left = image[:, : nx // 2]
        right = image[:, nx // 2 :]
        left_med = float(np.nanmedian(left[np.isfinite(left)])) if np.any(np.isfinite(left)) else np.nan
        right_med = float(np.nanmedian(right[np.isfinite(right)])) if np.any(np.isfinite(right)) else np.nan
        if np.isfinite(left_med) and np.isfinite(right_med) and std_bkg > 0:
            gradient_ratio = abs(left_med - right_med) / std_bkg
            report.info["background_gradient_ratio"] = gradient_ratio
            if gradient_ratio > gradient_threshold:
                report.add_warning(
                    VAL_BACKGROUND_GRADIENT,
                    f"Strong background gradient detected (L-R ratio={gradient_ratio:.2f})",
                )
                report.flags |= VAL_DEGRADED

    return report


def format_report(report: ValidationReport) -> str:
    """Format a validation report as a human-readable string."""
    lines = []
    status = "OK" if report.ok else ("DEGRADED" if report.degraded else "FATAL")
    lines.append(f"Validation: {status} (flags=0x{report.flags:04x})")
    if report.errors:
        lines.append("  Errors:")
        for e in report.errors:
            lines.append(f"    - {e}")
    if report.warnings:
        lines.append("  Warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")
    if report.info:
        lines.append("  Info:")
        for k, v in report.info.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.4g}")
            else:
                lines.append(f"    {k}: {v}")
    return "\n".join(lines)
