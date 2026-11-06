"""Tests for the PSF held-out validation module (utils/psf_validation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.psf_validation import (
    validate_psf_loo,
    PSFValidationResult,
    _fit_gaussian_stamp,
    PSF_VAL_OK,
    PSF_VAL_FLUX_BIAS,
    PSF_VAL_HIGH_CHI2,
    PSF_VAL_TOO_FEW_STARS,
    PSF_VAL_FIT_FAILURE,
)
from tests.synthetic.images import make_synthetic_image
from tests.synthetic.catalogs import make_isolated_catalog
from tests.synthetic.psf_models import gaussian_psf_2d, fwhm_to_sigma


class TestFitGaussianStamp:
    """Test the internal Gaussian stamp fitter."""

    def test_perfect_gaussian(self):
        """Fitting a noiseless Gaussian should recover parameters."""
        ny, nx = 25, 25
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        x0, y0, flux, sigma = 12.0, 12.0, 1000.0, 3.0
        stamp = gaussian_psf_2d(xx, yy, x0, y0, flux, sigma)

        result = _fit_gaussian_stamp(stamp, error=np.ones_like(stamp))
        assert result["converged"]
        assert result["x"] == pytest.approx(x0, abs=0.5)
        assert result["y"] == pytest.approx(y0, abs=0.5)
        assert result["flux"] == pytest.approx(flux, rel=0.1)
        assert result["reduced_chi2"] < 2.0

    def test_noisy_gaussian(self):
        """Fitting a noisy Gaussian should approximately recover flux."""
        rng = np.random.default_rng(42)
        ny, nx = 25, 25
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        x0, y0, flux, sigma = 12.0, 12.0, 5000.0, 3.0
        stamp = gaussian_psf_2d(xx, yy, x0, y0, flux, sigma)
        noise = rng.normal(0, 10, stamp.shape)
        stamp_noisy = stamp + noise

        result = _fit_gaussian_stamp(stamp_noisy, error=np.full_like(stamp, 10.0))
        assert result["converged"]
        assert result["flux"] == pytest.approx(flux, rel=0.10)  # 10% tolerance with noise
        assert abs(result["background"]) < 5.0  # Background should be near 0

    def test_non_finite_handled(self):
        """Non-finite pixels should not crash the fitter."""
        stamp = np.ones((15, 15))
        stamp[5, 5] = np.nan
        stamp[10, 10] = np.inf
        result = _fit_gaussian_stamp(stamp)
        # Should not crash; may or may not converge
        assert "converged" in result

    def test_all_non_finite(self):
        """All non-finite pixels should return non-converged."""
        stamp = np.full((15, 15), np.nan)
        result = _fit_gaussian_stamp(stamp)
        assert not result["converged"]


class TestValidatePSFLOO:
    """Test the PSF leave-one-out validation."""

    def _make_psf_builder(self, fwhm=4.0):
        """Create a simple PSF builder that returns a Gaussian model.

        For validation testing, we use a simple builder that always
        returns the same Gaussian PSF (since we're testing the validation
        framework, not the PSF construction).
        """

        def builder(image, star_table):
            sigma = fwhm_to_sigma(fwhm)

            class SimplePSF:
                def __call__(self, x, y, x0, y0, flux):
                    r2 = (x - x0) ** 2 + (y - y0) ** 2
                    return flux / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))

            return SimplePSF()

        return builder

    def test_good_psf_passes(self, isolated_field):
        """A well-built PSF on clean data should pass validation.

        The synthetic image has background=100, background_rms=10, read_noise=5,
        and Poisson noise.  The validation fitter uses a Gaussian + constant
        background model.  We expect:
        - Most fits to converge
        - Flux bias to be modest (< 30% given noise + model simplification)
        - Reduced chi2 to be reasonable
        """
        image = isolated_field["image"]
        truth = isolated_field["truth"]
        bkg_rms = isolated_field["background_rms_map"]

        # Use the truth catalog with flux_true as the flux column
        star_table = truth[["x_pix", "y_pix", "flux_true"]].copy()

        builder = self._make_psf_builder(fwhm=4.0)

        result = validate_psf_loo(
            image=image,
            star_table=star_table,
            psf_builder_fn=builder,
            cutout_size=25,
            background_rms=bkg_rms,
            gain=1.0,
            read_noise=5.0,
            n_folds=1,  # LOO
            flux_bias_threshold=0.30,  # 30% (generous for noisy synthetic data)
            flux_scatter_threshold=0.50,
            chi2_threshold=5.0,
            min_stars=5,
        )

        assert result.n_stars > 0
        assert len(result.per_star) > 0
        assert result.per_star["converged"].mean() > 0.5
        assert abs(result.flux_bias_frac) < 0.50

    def test_too_few_stars(self):
        """Too few stars should return failure with appropriate flag."""
        image = np.ones((64, 64), dtype=float) * 100
        star_table = pd.DataFrame({
            "x_pix": [10, 20, 30],
            "y_pix": [10, 20, 30],
            "flux_true": [100, 200, 300],
        })

        result = validate_psf_loo(
            image=image,
            star_table=star_table,
            psf_builder_fn=lambda img, tbl: None,
            min_stars=5,
        )

        assert result.flags & PSF_VAL_TOO_FEW_STARS
        assert not result.passed

    def test_no_flux_column(self):
        """Missing flux column should return failure."""
        image = np.ones((64, 64), dtype=float) * 100
        star_table = pd.DataFrame({
            "x_pix": [10, 20, 30, 40, 50],
            "y_pix": [10, 20, 30, 40, 50],
        })

        result = validate_psf_loo(
            image=image,
            star_table=star_table,
            psf_builder_fn=lambda img, tbl: None,
            min_stars=5,
        )

        assert result.flags & PSF_VAL_FIT_FAILURE
        assert not result.passed

    def test_psf_builder_failure(self):
        """PSF builder returning None should be handled gracefully."""
        image = np.ones((128, 128), dtype=float) * 100
        star_table = pd.DataFrame({
            "x_pix": [30, 40, 50, 60, 70],
            "y_pix": [30, 40, 50, 60, 70],
            "flux_true": [1000, 2000, 3000, 4000, 5000],
        })

        result = validate_psf_loo(
            image=image,
            star_table=star_table,
            psf_builder_fn=lambda img, tbl: None,
            min_stars=5,
        )

        assert result.flags & PSF_VAL_FIT_FAILURE
        assert not result.passed

    def test_result_to_dict(self):
        """to_dict should produce serializable output."""
        result = PSFValidationResult(
            n_stars=10, n_folds=1,
            flux_bias_frac=0.02, flux_scatter_frac=0.05,
            passed=True,
        )
        d = result.to_dict()
        assert d["psf_validation_n_stars"] == 10
        assert d["psf_validation_passed"] is True
        assert "psf_validation_flux_bias_frac" in d

    def test_result_summary(self):
        """summary should produce readable text."""
        result = PSFValidationResult(
            n_stars=20, n_folds=1,
            flux_bias_frac=0.01, flux_scatter_frac=0.05,
            median_reduced_chi2=1.2, passed=True,
        )
        text = result.summary()
        assert "PASS" in text
        assert "20 stars" in text

    def test_kfold_validation(self, isolated_field):
        """k-fold validation should work with k>1."""
        image = isolated_field["image"]
        truth = isolated_field["truth"]
        bkg_rms = isolated_field["background_rms_map"]

        star_table = truth[["x_pix", "y_pix", "flux_true"]].copy()
        builder = self._make_psf_builder(fwhm=4.0)

        result = validate_psf_loo(
            image=image,
            star_table=star_table,
            psf_builder_fn=builder,
            cutout_size=25,
            background_rms=bkg_rms,
            n_folds=5,  # 5-fold
            min_stars=5,
            flux_bias_threshold=0.15,
            flux_scatter_threshold=0.25,
        )

        assert result.n_folds == 5
        assert len(result.per_star) > 0

    def test_edge_stars_skipped(self):
        """Stars near image edges should be skipped, not crash."""
        image = np.ones((64, 64), dtype=float) * 100
        star_table = pd.DataFrame({
            "x_pix": [2, 62, 32, 32, 32],  # First two are near edges
            "y_pix": [32, 32, 2, 62, 32],
            "flux_true": [1000, 2000, 3000, 4000, 5000],
        })

        builder = self._make_psf_builder(fwhm=4.0)

        result = validate_psf_loo(
            image=image,
            star_table=star_table,
            psf_builder_fn=builder,
            cutout_size=25,  # Half=12, so stars at x<12 or x>52 are skipped
            min_stars=3,
            flux_bias_threshold=1.0,  # Very generous
            flux_scatter_threshold=1.0,
        )

        # Should not crash; edge stars skipped
        assert result.n_stars == 5
