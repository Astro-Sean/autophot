"""Tests for the data validation module (utils/data_validation.py)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from autophot.core.data_validation import (
    validate_image,
    format_report,
    ValidationReport,
    VAL_OK,
    VAL_NON_FINITE,
    VAL_ALL_NAN,
    VAL_BAD_DTYPE,
    VAL_BAD_DIMENSIONS,
    VAL_MISSING_EXPTIME,
    VAL_MISSING_GAIN,
    VAL_MISSING_RDNOISE,
    VAL_MISSING_SATURATE,
    VAL_MISSING_WCS,
    VAL_HIGH_NAN_FRACTION,
    VAL_MASK_MISMATCH,
    VAL_BACKGROUND_GRADIENT,
    VAL_SATURATED_PIXELS,
    VAL_DEGRADED,
    VAL_FATAL,
)


class TestValidateImage:
    """Test the validate_image function."""

    def _make_header(self, **kwargs):
        """Create a FITS header with standard metadata."""
        h = fits.Header()
        h["TELESCOP"] = "Test"
        h["INSTRUME"] = "TestCam"
        h["EXPTIME"] = 30.0
        h["GAIN"] = 1.0
        h["RDNOISE"] = 5.0
        h["SATURATE"] = 50000.0
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [128, 128]
        wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
        wcs.wcs.crval = [150.0, 30.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        h.update(wcs.to_header())
        for k, v in kwargs.items():
            h[k] = v
        return h

    def test_valid_image(self):
        """A well-formed image should pass validation."""
        image = np.random.normal(100, 10, (256, 256)).astype(np.float64)
        header = self._make_header()
        report = validate_image(image, header)
        assert report.ok
        assert report.flags == VAL_OK
        assert len(report.errors) == 0

    def test_non_array(self):
        """Non-array input should be fatal."""
        report = validate_image([1, 2, 3])
        assert report.flags & VAL_FATAL
        assert not report.ok

    def test_wrong_dimensions(self):
        """1-D array should be fatal."""
        report = validate_image(np.ones(100, dtype=float))
        assert report.flags & VAL_FATAL
        assert report.flags & VAL_BAD_DIMENSIONS

    def test_all_nan(self):
        """All-NaN image should be fatal."""
        image = np.full((64, 64), np.nan)
        report = validate_image(image)
        assert report.flags & VAL_FATAL
        assert report.flags & VAL_ALL_NAN

    def test_some_nan(self):
        """Some NaN pixels should warn but not be fatal."""
        image = np.ones((64, 64), dtype=float)
        image[:5, :5] = np.nan  # ~0.6% NaN
        report = validate_image(image)
        assert report.ok
        assert report.flags & VAL_NON_FINITE
        assert "non-finite" in report.warnings[0].lower()

    def test_high_nan_fraction(self):
        """High NaN fraction should be degraded."""
        image = np.ones((64, 64), dtype=float)
        image[:20, :] = np.nan  # ~31% NaN
        report = validate_image(image, nan_fraction_degraded=0.20)
        assert report.degraded
        assert report.flags & VAL_HIGH_NAN_FRACTION

    def test_bad_dtype(self):
        """Integer dtype should warn."""
        image = np.ones((64, 64), dtype=np.int32) * 100
        report = validate_image(image)
        assert report.flags & VAL_BAD_DTYPE

    def test_missing_exptime(self):
        """Missing exposure time should warn."""
        image = np.ones((64, 64), dtype=float) * 100
        header = self._make_header()
        del header["EXPTIME"]
        report = validate_image(image, header)
        assert report.flags & VAL_MISSING_EXPTIME

    def test_missing_gain(self):
        """Missing gain should warn."""
        image = np.ones((64, 64), dtype=float) * 100
        header = self._make_header()
        del header["GAIN"]
        report = validate_image(image, header)
        assert report.flags & VAL_MISSING_GAIN

    def test_missing_rdnoise(self):
        """Missing read noise should warn."""
        image = np.ones((64, 64), dtype=float) * 100
        header = self._make_header()
        del header["RDNOISE"]
        report = validate_image(image, header)
        assert report.flags & VAL_MISSING_RDNOISE

    def test_missing_saturate(self):
        """Missing saturation should warn."""
        image = np.ones((64, 64), dtype=float) * 100
        header = self._make_header()
        del header["SATURATE"]
        report = validate_image(image, header)
        assert report.flags & VAL_MISSING_SATURATE

    def test_saturated_pixels(self):
        """Saturated pixels should be flagged."""
        image = np.ones((64, 64), dtype=float) * 100
        image[:10, :10] = 50000.0  # At saturation
        header = self._make_header()
        report = validate_image(image, header)
        assert report.flags & VAL_SATURATED_PIXELS

    def test_mask_mismatch(self):
        """Mask with wrong shape should be fatal."""
        image = np.ones((64, 64), dtype=float)
        mask = np.ones((32, 32), dtype=bool)
        report = validate_image(image, mask=mask)
        assert report.flags & VAL_FATAL
        assert report.flags & VAL_MASK_MISMATCH

    def test_valid_mask(self):
        """Valid mask should be recorded in info."""
        image = np.ones((64, 64), dtype=float) * 100
        mask = np.zeros((64, 64), dtype=bool)
        mask[:5, :5] = True
        report = validate_image(image, mask=mask)
        assert report.ok
        assert report.info["mask_fraction"] > 0

    def test_background_gradient(self):
        """Strong background gradient should be flagged."""
        image = np.ones((128, 128), dtype=float) * 100
        # Add a strong left-right gradient
        gradient = np.linspace(0, 100, 128)[np.newaxis, :]
        image += gradient
        header = self._make_header()
        report = validate_image(image, header, gradient_threshold=0.5)
        assert report.flags & VAL_BACKGROUND_GRADIENT

    def test_no_gradient(self):
        """Uniform background should not trigger gradient flag."""
        image = np.random.normal(100, 10, (128, 128))
        header = self._make_header()
        report = validate_image(image, header, gradient_threshold=0.5)
        assert not (report.flags & VAL_BACKGROUND_GRADIENT)

    def test_override_values(self):
        """Override values should be used instead of header."""
        image = np.ones((64, 64), dtype=float) * 100
        header = self._make_header()
        report = validate_image(
            image, header,
            gain=2.0, read_noise=10.0, exposure_time=60.0, saturate=60000.0,
        )
        assert report.info["gain"] == 2.0
        assert report.info["read_noise"] == 10.0
        assert report.info["exposure_time"] == 60.0
        assert report.info["saturate"] == 60000.0

    def test_info_populated(self):
        """Info dict should be populated with metadata."""
        image = np.random.normal(100, 10, (64, 64))
        header = self._make_header()
        report = validate_image(image, header)
        assert "image_shape" in report.info
        assert "nan_fraction" in report.info
        assert "exposure_time" in report.info
        assert "gain" in report.info
        assert "read_noise" in report.info
        assert "saturate" in report.info
        assert "background_median" in report.info

    def test_format_report(self):
        """format_report should produce readable output."""
        report = validate_image(np.ones((32, 32), dtype=float) * 100)
        text = format_report(report)
        assert "Validation:" in text
        assert "OK" in text or "DEGRADED" in text or "FATAL" in text


class TestValidationReport:
    """Test the ValidationReport dataclass."""

    def test_ok_by_default(self):
        report = ValidationReport()
        assert report.ok
        assert not report.degraded

    def test_add_warning(self):
        report = ValidationReport()
        report.add_warning(VAL_NON_FINITE, "test warning")
        assert report.flags & VAL_NON_FINITE
        assert len(report.warnings) == 1

    def test_add_error(self):
        report = ValidationReport()
        report.add_error(VAL_FATAL, "test error")
        assert report.flags & VAL_FATAL
        assert not report.ok
        assert len(report.errors) == 1

    def test_degraded(self):
        report = ValidationReport()
        report.add_warning(VAL_DEGRADED, "degraded")
        assert report.degraded
        assert report.ok  # degraded but not fatal
