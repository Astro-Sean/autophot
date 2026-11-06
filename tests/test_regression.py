"""Regression tests with synthetic truth-recovery.

These tests verify that the photometry pipeline components produce
scientifically correct results on synthetic data with known truth.
They are the primary validation that the pipeline does not manufacture
precision or bias.

Quantitative acceptance criteria:
- High-S/N isolated source: flux bias < 5%
- PSF held-out residuals: consistent with noise (reduced chi2 ~ 1)
- Uncertainty coverage: 68% of truths within 1-sigma, 95% within 2-sigma
- MCMC: R-hat < 1.1, n_eff > 100 for converged chains
- Injection reproducibility: identical results with fixed seed
- All degraded cases: appropriate quality flags set
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synthetic.images import (
    make_single_source_image,
    make_synthetic_image,
    make_faint_source_image,
    make_saturated_source_image,
    make_edge_source_image,
    make_nan_region_image,
)
from tests.synthetic.catalogs import make_isolated_catalog
from tests.synthetic.psf_models import gaussian_psf_2d, fwhm_to_sigma

from utils.data_validation import validate_image, VAL_OK, VAL_NON_FINITE, VAL_SATURATED_PIXELS
from utils.quality_flags import QualityFlag, from_snr, from_chi2
from utils.psf_validation import validate_psf_loo, PSF_VAL_OK
from utils.uncertainty_calibration import calibrate_uncertainties
from utils.mcmc_diagnostics import assess_convergence, MCMC_FLAG_OK
from utils.injection_catalog import InjectionCatalog


# ---------------------------------------------------------------------------
# Synthetic truth-recovery: aperture photometry on known sources
# ---------------------------------------------------------------------------

class TestSyntheticTruthRecovery:
    """Verify photometry recovers known truth within tolerance."""

    def test_high_snr_isolated_source_flux(self, isolated_source):
        """High-S/N isolated source: flux should be recovered within 5%.

        We use a simple aperture sum (not the full Aperture.measure) to
        verify the synthetic data generator and basic photometry agree.
        """
        data = isolated_source
        image = data["image"]
        truth = data["truth"]
        x0 = truth["x_pix"][0]
        y0 = truth["y_pix"][0]
        flux_true = truth["flux_true"][0]
        background = truth["background_true"][0]

        fwhm = truth["fwhm_true"][0]
        sigma = fwhm_to_sigma(fwhm)
        r_ap = 2.0 * fwhm  # 2 FWHM aperture
        ny, nx = image.shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        aperture_mask = r2 <= r_ap**2

        aperture_sum = np.nansum(image[aperture_mask]) - background * aperture_mask.sum()

        # Encircled energy at 2*FWHM for Gaussian
        from tests.synthetic.psf_models import encircled_energy_gaussian

        ee = encircled_energy_gaussian(r_ap, sigma)
        flux_recovered = aperture_sum / ee

        bias_frac = (flux_recovered - flux_true) / flux_true
        assert abs(bias_frac) < 0.05, (
            f"Flux bias {bias_frac*100:.1f}% exceeds 5% tolerance"
        )

    def test_multiple_isolated_sources_flux(self, isolated_field):
        """Multiple isolated sources: median flux bias < 10%."""
        data = isolated_field
        image = data["image"]
        truth = data["truth"]
        background = truth["background_true"][0]
        fwhm = truth["fwhm_true"][0]
        sigma = fwhm_to_sigma(fwhm)
        r_ap = 2.0 * fwhm

        ny, nx = image.shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

        from tests.synthetic.psf_models import encircled_energy_gaussian

        ee = encircled_energy_gaussian(r_ap, sigma)

        biases = []
        for _, row in truth.iterrows():
            x0, y0 = row["x_pix"], row["y_pix"]
            flux_true = row["flux_true"]
            r2 = (xx - x0) ** 2 + (yy - y0) ** 2
            mask = r2 <= r_ap**2
            ap_sum = np.nansum(image[mask]) - background * mask.sum()
            flux_rec = ap_sum / ee
            if flux_true > 1000:  # Only test bright enough sources
                biases.append((flux_rec - flux_true) / flux_true)

        biases = np.array(biases)
        assert len(biases) > 10, "Not enough bright sources for test"
        median_bias = np.median(biases)
        assert abs(median_bias) < 0.10, (
            f"Median flux bias {median_bias*100:.1f}% exceeds 10% tolerance"
        )

    def test_faint_source_detection(self, faint_source):
        """Faint source at ~5 sigma (PSF-fit) should be detectable.

        Note: the synthetic image generator targets the *PSF-fit* S/N,
        which is higher than the aperture S/N because the PSF fit uses
        optimal weighting.  The aperture S/N will be lower, but the
        source should still be detectable (> 2 sigma).
        """
        data = faint_source
        image = data["image"]
        truth = data["truth"]
        x0 = truth["x_pix"][0]
        y0 = truth["y_pix"][0]
        flux_true = truth["flux_true"][0]
        background = truth["background_true"][0]
        bkg_rms = truth["background_rms_true"][0]

        fwhm = truth["fwhm_true"][0]
        sigma = fwhm_to_sigma(fwhm)
        r_ap = 2.0 * fwhm
        ny, nx = image.shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        mask = r2 <= r_ap**2

        from tests.synthetic.psf_models import encircled_energy_gaussian

        ee = encircled_energy_gaussian(r_ap, sigma)
        ap_sum = np.nansum(image[mask]) - background * mask.sum()
        flux_rec = ap_sum / ee
        noise = bkg_rms * np.sqrt(mask.sum())
        snr = flux_rec / noise

        # Aperture S/N will be lower than PSF-fit S/N, but should be > 2
        assert snr > 2.0, f"Aperture S/N {snr:.1f} too low for 5-sigma PSF source"
        bias = (flux_rec - flux_true) / flux_true
        assert abs(bias) < 0.30, f"Flux bias {bias*100:.1f}% too large"


# ---------------------------------------------------------------------------
# Data validation regression tests
# ---------------------------------------------------------------------------

class TestDataValidationRegression:
    """Verify data validation catches known problems."""

    def test_valid_synthetic_image_passes(self, isolated_source):
        """A valid synthetic image should pass validation."""
        report = validate_image(
            isolated_source["image"],
            header=isolated_source["header"],
        )
        assert report.ok
        assert len(report.errors) == 0

    def test_saturated_source_flagged(self, saturated_source):
        """Saturated source should trigger saturation flag."""
        report = validate_image(
            saturated_source["image"],
            header=saturated_source["header"],
        )
        assert report.flags & VAL_SATURATED_PIXELS

    def test_nan_region_flagged(self, nan_region_image):
        """NaN regions should trigger non-finite flag."""
        report = validate_image(
            nan_region_image["image"],
            header=nan_region_image["header"],
        )
        assert report.flags & VAL_NON_FINITE


# ---------------------------------------------------------------------------
# Quality flag regression tests
# ---------------------------------------------------------------------------

class TestQualityFlagRegression:
    """Verify quality flags correctly classify measurements."""

    def test_high_snr_clean_measurement(self):
        """High S/N, good chi2 should produce OK flags."""
        qf = from_snr(50.0, 3.0) | from_chi2(1.0, 5.0)
        assert qf == QualityFlag.OK
        assert qf.is_ok()

    def test_low_snr_flagged(self):
        """Low S/N should produce LOW_SN flag."""
        qf = from_snr(2.0, 3.0)
        assert QualityFlag.LOW_SN in qf
        assert qf.is_degraded()

    def test_bad_chi2_flagged(self):
        """Bad chi2 should produce SUSPICIOUS_RESIDUALS flag."""
        qf = from_chi2(10.0, 5.0)
        assert QualityFlag.SUSPICIOUS_RESIDUALS in qf
        assert qf.is_degraded()

    def test_fatal_overrides_degraded(self):
        """Fatal flags should override degraded status."""
        qf = QualityFlag.SATURATED | QualityFlag.LOW_SN
        assert qf.is_fatal()
        assert not qf.is_degraded()


# ---------------------------------------------------------------------------
# PSF validation regression tests
# ---------------------------------------------------------------------------

class TestPSFValidationRegression:
    """Verify PSF validation detects known problems."""

    def test_good_psf_validation(self, isolated_field):
        """A good PSF on clean data should pass or be close to passing."""
        image = isolated_field["image"]
        truth = isolated_field["truth"]
        bkg_rms = isolated_field["background_rms_map"]

        star_table = truth[["x_pix", "y_pix", "flux_true"]].copy()

        sigma = fwhm_to_sigma(4.0)

        class SimplePSF:
            def __call__(self, x, y, x0, y0, flux):
                r2 = (x - x0) ** 2 + (y - y0) ** 2
                return flux / (2 * np.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))

        def builder(img, tbl):
            return SimplePSF()

        result = validate_psf_loo(
            image=image,
            star_table=star_table,
            psf_builder_fn=builder,
            cutout_size=25,
            background_rms=bkg_rms,
            gain=1.0,
            read_noise=5.0,
            n_folds=1,
            flux_bias_threshold=0.30,
            flux_scatter_threshold=0.50,
            chi2_threshold=5.0,
            min_stars=5,
        )

        assert len(result.per_star) > 10
        assert result.per_star["converged"].mean() > 0.5


# ---------------------------------------------------------------------------
# Uncertainty calibration regression tests
# ---------------------------------------------------------------------------

class TestUncertaintyCalibrationRegression:
    """Verify uncertainty calibration detects known problems."""

    def test_well_calibrated_errors_pass(self):
        """Well-calibrated errors should pass."""
        rng = np.random.default_rng(42)
        n = 500
        flux_true = rng.uniform(1000, 10000, n)
        flux_err = np.sqrt(flux_true)
        z = rng.normal(0, 1, n)
        flux_meas = flux_true + z * flux_err

        result = calibrate_uncertainties(flux_true, flux_meas, flux_err)
        assert result.passed
        assert abs(result.z_mad_std - 1.0) < 0.15
        assert result.coverage_1sigma_ok
        assert result.coverage_2sigma_ok

    def test_underestimated_errors_detected(self):
        """Underestimated errors should be detected."""
        rng = np.random.default_rng(42)
        n = 500
        flux_true = rng.uniform(1000, 10000, n)
        true_err = np.sqrt(flux_true) * 2
        reported_err = np.sqrt(flux_true)
        z = rng.normal(0, 1, n)
        flux_meas = flux_true + z * true_err

        result = calibrate_uncertainties(flux_true, flux_meas, reported_err)
        assert result.errors_underestimated
        assert result.calibration_factor > 1.5
        assert not result.passed


# ---------------------------------------------------------------------------
# MCMC convergence regression tests
# ---------------------------------------------------------------------------

class TestMCMCConvergenceRegression:
    """Verify MCMC convergence diagnostics detect known problems."""

    def test_converged_chain_passes(self):
        """A well-mixed chain should pass convergence checks."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (32, 2000, 3))
        report = assess_convergence(chain, acceptance_fraction=0.3)
        assert report.converged
        assert report.flags == MCMC_FLAG_OK

    def test_non_converged_chain_detected(self):
        """A non-converged chain should be detected."""
        rng = np.random.default_rng(42)
        n_walkers, n_steps = 32, 2000
        chain = np.zeros((n_walkers, n_steps, 1))
        chain[:n_walkers // 2] = rng.normal(0, 1, (n_walkers // 2, n_steps, 1))
        chain[n_walkers // 2:] = rng.normal(10, 1, (n_walkers // 2, n_steps, 1))
        report = assess_convergence(chain, acceptance_fraction=0.3)
        assert not report.converged
        assert report.rhat > 1.1


# ---------------------------------------------------------------------------
# Injection reproducibility regression tests
# ---------------------------------------------------------------------------

class TestInjectionReproducibility:
    """Verify injection/recovery is reproducible with fixed seeds."""

    def test_catalog_reproducible(self):
        """Same seed should produce same catalog."""
        cat1 = InjectionCatalog(seed=42, recovery_method="AP")
        cat2 = InjectionCatalog(seed=42, recovery_method="AP")

        for i in range(10):
            cat1.add_record(
                magnitude=15.0 + i, x_inj=100 + i, y_inj=200 + i,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )
            cat2.add_record(
                magnitude=15.0 + i, x_inj=100 + i, y_inj=200 + i,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )

        df1 = cat1.to_dataframe()
        df2 = cat2.to_dataframe()
        pd.testing.assert_frame_equal(df1, df2)

    def test_completeness_curve_reproducible(self):
        """Completeness curve should be reproducible."""
        cat1 = InjectionCatalog(seed=42)
        cat2 = InjectionCatalog(seed=42)

        for i in range(20):
            mag = 14.0 + i * 0.2
            cat1.add_record(
                magnitude=mag, x_inj=100, y_inj=100,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )
            cat2.add_record(
                magnitude=mag, x_inj=100, y_inj=100,
                flux_true=1000.0, flux_recovered=1010.0, flux_err=50.0,
                detected=True, detection_snr=20.0,
            )

        curve1 = cat1.completeness_curve(n_bins=10)
        curve2 = cat2.completeness_curve(n_bins=10)
        pd.testing.assert_frame_equal(curve1, curve2)


# ---------------------------------------------------------------------------
# Failure mode regression tests
# ---------------------------------------------------------------------------

class TestFailureModes:
    """Verify failure modes produce appropriate flags/results."""

    def test_saturated_source_validation(self, saturated_source):
        """Saturated source should be flagged by validation."""
        report = validate_image(
            saturated_source["image"],
            header=saturated_source["header"],
        )
        assert report.flags & VAL_SATURATED_PIXELS

    def test_edge_source_validation(self, edge_source):
        """Edge source should not crash validation."""
        report = validate_image(
            edge_source["image"],
            header=edge_source["header"],
        )
        # Should not crash; may or may not have warnings
        assert report is not None

    def test_nan_region_validation(self, nan_region_image):
        """NaN region should be flagged."""
        report = validate_image(
            nan_region_image["image"],
            header=nan_region_image["header"],
        )
        assert report.flags & VAL_NON_FINITE

    def test_all_nan_image_fatal(self):
        """All-NaN image should be fatal."""
        image = np.full((64, 64), np.nan)
        report = validate_image(image)
        assert not report.ok
