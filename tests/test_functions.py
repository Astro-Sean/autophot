"""Tests for core utility functions in functions.py.

These tests verify the scientific correctness of key functions that are
used throughout the photometry pipeline, including uncertainty propagation,
detection confidence, magnitude calculation, and sampling regime classification.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf, erfinv

from autophot.core.functions import (
    quadrature_add,
    beta_psf,
    beta_aperture,
    mag,
    flux_upper_limit,
    classify_sampling_regime,
    adaptive_aperture_radius,
    adaptive_annulus_radii,
    snr,
    snr_err,
    gauss_1d,
    fwhm_gaussian,
    sigma_gaussian,
    fwhm_moffat,
    alpha_moffat,
    pix_dist,
    invalid_use_catalog_keys,
    parse_supported_filter_group_key,
    check_input_config,
    format_config_errors,
    load_default_input_schema,
    strict_config_enabled,
    canonical_target_name,
)


# ---------------------------------------------------------------------------
# quadrature_add
# ---------------------------------------------------------------------------

class TestQuadratureAdd:
    """Test the quadrature error propagation function."""

    def test_simple_quadrature(self):
        """sqrt(3^2 + 4^2) = 5."""
        result = quadrature_add([3.0, 4.0])
        assert result == pytest.approx(5.0, abs=1e-10)

    def test_single_value(self):
        """Single value returns abs(value)."""
        result = quadrature_add([5.0])
        assert result == pytest.approx(5.0, abs=1e-10)

    def test_zero_values(self):
        """All zeros returns 0."""
        result = quadrature_add([0.0, 0.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_nan_skipped(self):
        """NaN terms are skipped, not propagated (BUG-3 fix)."""
        result = quadrature_add([3.0, np.nan, 4.0])
        assert result == pytest.approx(5.0, abs=1e-10)

    def test_all_nan_returns_nan(self):
        """If all terms are NaN, return NaN."""
        result = quadrature_add([np.nan, np.nan])
        assert np.isnan(result)

    def test_inf_skipped(self):
        """Inf terms are skipped."""
        result = quadrature_add([3.0, np.inf, 4.0])
        assert result == pytest.approx(5.0, abs=1e-10)

    def test_mixed_nan_and_finite(self):
        """Mixed NaN and finite values: only finite contribute."""
        result = quadrature_add([np.nan, 1.0, np.nan, 1.0])
        assert result == pytest.approx(np.sqrt(2.0), abs=1e-10)

    def test_empty_list_returns_nan(self):
        """Empty list returns NaN."""
        result = quadrature_add([])
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# beta_psf
# ---------------------------------------------------------------------------

class TestBetaPSF:
    """Test the PSF detection confidence function."""

    def test_high_snr_detection(self):
        """A 5-sigma detection should have beta near 0.5 (50% above 3-sigma)."""
        # flux = 5*err, n=3: z = (3*err - 5*err)/(sqrt(2)*err) = -2/sqrt(2)
        # beta = 0.5*(1 - erf(-2/sqrt(2))) = 0.5*(1 + erf(1.414)) ~ 0.921
        flux = 5.0
        err = 1.0
        result = beta_psf(3, flux, err)
        expected = 0.5 * (1 - erf((3 * err - flux) / (np.sqrt(2) * err)))
        assert result == pytest.approx(expected, abs=1e-6)
        assert result > 0.5

    def test_negative_flux_symmetric(self):
        """Negative flux should give same beta as positive (BUG-5 fix)."""
        beta_pos = beta_psf(3, 5.0, 1.0)
        beta_neg = beta_psf(3, -5.0, 1.0)
        assert beta_pos == pytest.approx(beta_neg, abs=1e-10)

    def test_zero_flux(self):
        """Zero flux should give low beta."""
        result = beta_psf(3, 0.0, 1.0)
        assert result < 0.5

    def test_below_threshold(self):
        """Flux below n-sigma should give low beta."""
        result = beta_psf(5, 1.0, 1.0)  # 1-sigma flux, 5-sigma threshold
        assert result < 0.01

    def test_well_above_threshold(self):
        """Flux well above n-sigma should give high beta."""
        result = beta_psf(3, 10.0, 1.0)  # 10-sigma flux, 3-sigma threshold
        assert result > 0.99

    def test_beta_in_valid_range(self):
        """Beta should always be in [0, 1]."""
        for flux in [-100, -10, -1, 0, 1, 10, 100]:
            result = beta_psf(3, flux, 1.0)
            assert 0.0 <= result <= 1.0

    def test_zero_error_handled(self):
        """Zero error should not crash (uses tiny epsilon)."""
        result = beta_psf(3, 100.0, 0.0)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# beta_aperture
# ---------------------------------------------------------------------------

class TestBetaAperture:
    """Test the aperture detection confidence function."""

    def test_high_snr_detection(self):
        """High S/N aperture detection should have high beta."""
        # flux = 100, npix = 50, sigma = 1, noise = 0
        # source_flux = 100, sigma_ap = 1*sqrt(50) = 7.07
        # z = (3*7.07 - 100)/(sqrt(2)*7.07) = (21.2 - 100)/10 = -7.88
        # beta ~ 0.5*(1 - erf(-7.88)) ~ 1.0
        result = beta_aperture(3, 100.0, 50, 1.0, noise=0)
        assert result > 0.99

    def test_zero_flux(self):
        """Zero flux should give low beta."""
        result = beta_aperture(3, 0.0, 50, 1.0, noise=0)
        assert result < 0.5

    def test_beta_in_valid_range(self):
        """Beta should always be in [0, 1]."""
        for flux in [0, 10, 50, 100, 500]:
            result = beta_aperture(3, flux, 50, 1.0, noise=0)
            assert 0.0 <= result <= 1.0

    def test_noise_subtraction(self):
        """Background noise should reduce effective source flux."""
        beta_no_noise = beta_aperture(3, 100.0, 50, 1.0, noise=0)
        beta_with_noise = beta_aperture(3, 100.0, 50, 1.0, noise=0.5)
        assert beta_with_noise <= beta_no_noise

    def test_negative_flux_symmetric(self):
        """Negative flux should give same beta as positive (difference-image fix).

        Consistent with the beta_psf fix (BUG-5): detection confidence is
        symmetric - a -5 sigma dip is as significant as a +5 sigma peak.
        """
        beta_pos = beta_aperture(3, 100.0, 50, 1.0, noise=0)
        beta_neg = beta_aperture(3, -100.0, 50, 1.0, noise=0)
        assert beta_pos == pytest.approx(beta_neg, abs=1e-10)


# ---------------------------------------------------------------------------
# mag
# ---------------------------------------------------------------------------

class TestMag:
    """Test the magnitude calculation function."""

    def test_positive_flux(self):
        """-2.5 * log10(100) = -5.0."""
        assert mag(100.0) == pytest.approx(-5.0, abs=1e-10)

    def test_unit_flux(self):
        """-2.5 * log10(1) = 0."""
        assert mag(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_zero_flux_returns_nan(self):
        """Zero flux returns NaN."""
        assert np.isnan(mag(0.0))

    def test_negative_flux_returns_nan(self):
        """Negative flux returns NaN."""
        assert np.isnan(mag(-100.0))

    def test_array_input(self):
        """Array input returns array of magnitudes."""
        fluxes = np.array([1.0, 10.0, 100.0])
        result = mag(fluxes)
        expected = np.array([0.0, -2.5, -5.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_array_with_zeros(self):
        """Array with zeros returns NaN for zero entries."""
        fluxes = np.array([100.0, 0.0, 50.0])
        result = mag(fluxes)
        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])

    def test_input_not_mutated(self):
        """Input array should not be mutated."""
        fluxes = np.array([100.0, 0.0, 50.0])
        original = fluxes.copy()
        _ = mag(fluxes)
        np.testing.assert_array_equal(fluxes, original)


# ---------------------------------------------------------------------------
# flux_upper_limit
# ---------------------------------------------------------------------------

class TestFluxUpperLimit:
    """Test the flux upper limit calculation."""

    def test_basic_upper_limit(self):
        """Upper limit should be finite and positive for reasonable inputs."""
        result = flux_upper_limit(3, 10.0, 0.5)
        assert np.isfinite(result)
        assert result > 0

    def test_consistency_with_beta(self):
        """Upper limit should be consistent with beta_aperture inversion."""
        n = 3
        sigma = 10.0
        beta_target = 0.9
        flux_limit = flux_upper_limit(n, sigma, beta_target)
        beta_check = beta_aperture(n, flux_limit, 1, sigma, noise=0)
        assert beta_check == pytest.approx(beta_target, abs=0.01)


# ---------------------------------------------------------------------------
# classify_sampling_regime
# ---------------------------------------------------------------------------

class TestClassifySamplingRegime:
    """Test the FWHM sampling regime classifier."""

    def test_undersampled(self):
        """FWHM < 2.5 px is undersampled."""
        assert classify_sampling_regime(2.0) == "undersampled"

    def test_critical(self):
        """FWHM 2.5-3.0 px is critical."""
        assert classify_sampling_regime(2.7) == "critical"

    def test_well_sampled(self):
        """FWHM 3.0-5.0 px is well_sampled."""
        assert classify_sampling_regime(4.0) == "well_sampled"

    def test_oversampled(self):
        """FWHM > 5.0 px is oversampled."""
        assert classify_sampling_regime(6.0) == "oversampled"

    def test_boundary_undersampled(self):
        """FWHM = 2.5 is exactly at boundary (undersampled)."""
        assert classify_sampling_regime(2.5) == "undersampled"

    def test_boundary_critical(self):
        """FWHM = 3.0 is exactly at boundary (critical)."""
        assert classify_sampling_regime(3.0) == "critical"

    def test_boundary_oversampled(self):
        """FWHM = 5.0 is exactly at boundary (well_sampled)."""
        assert classify_sampling_regime(5.0) == "well_sampled"

    def test_invalid_fwhm(self):
        """Invalid FWHM returns safe default."""
        assert classify_sampling_regime(0.0) == "well_sampled"
        assert classify_sampling_regime(-1.0) == "well_sampled"
        assert classify_sampling_regime(np.nan) == "well_sampled"

    def test_custom_thresholds(self):
        """Custom thresholds from input_yaml."""
        yaml = {"photometry": {"undersampled_fwhm_threshold": 3.0}}
        assert classify_sampling_regime(2.5, yaml) == "undersampled"


# ---------------------------------------------------------------------------
# adaptive_aperture_radius
# ---------------------------------------------------------------------------

class TestAdaptiveApertureRadius:
    """Test the FWHM-adaptive aperture radius function."""

    def test_undersampled(self):
        """Undersampled: max(4, 2.5*FWHM)."""
        result = adaptive_aperture_radius(2.0)
        assert result == pytest.approx(max(4.0, 2.5 * 2.0))

    def test_critical(self):
        """Critical: 2.0*FWHM."""
        result = adaptive_aperture_radius(2.7)
        assert result == pytest.approx(2.0 * 2.7)

    def test_well_sampled(self):
        """Well-sampled: 1.5*FWHM."""
        result = adaptive_aperture_radius(4.0)
        assert result == pytest.approx(1.5 * 4.0)

    def test_oversampled(self):
        """Oversampled: 1.8*FWHM."""
        result = adaptive_aperture_radius(6.0)
        assert result == pytest.approx(1.8 * 6.0)

    def test_monotonic_increase(self):
        """Aperture radius should generally increase with FWHM."""
        fwhms = [1.5, 2.0, 2.7, 4.0, 6.0, 8.0]
        radii = [adaptive_aperture_radius(f) for f in fwhms]
        # Check general trend (allowing for regime transitions)
        assert radii[-1] > radii[0]


# ---------------------------------------------------------------------------
# Gaussian / Moffat helpers
# ---------------------------------------------------------------------------

class TestGaussianHelpers:
    """Test Gaussian/Moffat conversion helpers."""

    def test_fwhm_sigma_roundtrip(self):
        """FWHM <-> sigma roundtrip."""
        sigma = 2.0
        fwhm = fwhm_gaussian(sigma)
        assert sigma_gaussian(fwhm) == pytest.approx(sigma, abs=1e-10)

    def test_fwhm_alpha_roundtrip(self):
        """FWHM <-> alpha roundtrip for Moffat."""
        alpha = 3.0
        fwhm = fwhm_moffat(alpha, beta=4.765)
        result = alpha_moffat(fwhm, beta=4.765)
        assert result == pytest.approx(alpha, abs=1e-8)

    def test_gauss_1d(self):
        """1-D Gaussian evaluation."""
        x = np.array([0.0, 1.0, 2.0])
        result = gauss_1d(x, A=1.0, x0=0.0, sigma=1.0)
        # Peak at x=0
        assert result[0] == pytest.approx(1.0, abs=1e-10)
        # Symmetric
        assert result[1] == pytest.approx(gauss_1d(np.array([-1.0]), 1.0, 0.0, 1.0)[0], abs=1e-10)


# ---------------------------------------------------------------------------
# SNR helpers
# ---------------------------------------------------------------------------

class TestSNRHelpers:
    """Test SNR calculation helpers."""

    def test_snr_basic(self):
        """SNR = maxPixel / noiseBkg."""
        assert snr(100.0, 10.0) == pytest.approx(10.0)

    def test_snr_err(self):
        """Magnitude error from SNR: 1.0857/SNR."""
        result = snr_err(10.0)
        assert result == pytest.approx(1.0857 / 10.0, abs=1e-4)


# ---------------------------------------------------------------------------
# pix_dist
# ---------------------------------------------------------------------------

class TestPixDist:
    """Test pixel distance calculation."""

    def test_basic_distance(self):
        """3-4-5 triangle."""
        assert pix_dist(0, 3, 0, 4) == pytest.approx(5.0)

    def test_zero_distance(self):
        """Same point = 0 distance."""
        assert pix_dist(5, 5, 5, 5) == pytest.approx(0.0)

    def test_negative_coords(self):
        """Negative coordinates work."""
        assert pix_dist(-3, 0, 0, -4) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# invalid_use_catalog_keys
# ---------------------------------------------------------------------------

class TestInvalidUseCatalogKeys:
    """Keys with unknown band letters silently fell back to 'default'
    for those bands (stacked-field ZP failures)."""

    def test_unknown_band_letters_flagged(self):
        mapping = {"uXQ": "gaia", "default": "refcat"}
        assert invalid_use_catalog_keys(mapping) == ["uXQ"]

    def test_mixed_family_key_now_valid(self):
        # Mixed families parse per-char (u from ugriz, R/I from UBVRI).
        mapping = {"uRI": "gaia", "default": "refcat"}
        assert invalid_use_catalog_keys(mapping) == []

    def test_group_and_single_band_keys_ok(self):
        mapping = {"griz": "sdss", "UBVRI": "apass", "g": "sdss"}
        assert invalid_use_catalog_keys(mapping) == []

    def test_default_and_wildcard_ignored(self):
        mapping = {"default": "gaia", "*": "sdss", "all": "apass"}
        assert invalid_use_catalog_keys(mapping) == []

    def test_none_values_skipped(self):
        mapping = {"uXQ": None, "zz": None}
        assert invalid_use_catalog_keys(mapping) == []

    def test_scalar_and_none_passthrough(self):
        assert invalid_use_catalog_keys("gaia") == []
        assert invalid_use_catalog_keys(None) == []


class TestParseFilterGroupKey:
    """Friendly grammar: band letters are unique across families, so
    mixed keys parse unambiguously one char at a time."""

    def test_mixed_family_letters(self):
        assert parse_supported_filter_group_key("uRI") == ("u", "R", "I")

    def test_mixed_family_arbitrary(self):
        assert parse_supported_filter_group_key("gBVw") == ("g", "B", "V", "w")
        assert parse_supported_filter_group_key("rJ") == ("r", "J")

    def test_single_family_subsets_unchanged(self):
        assert parse_supported_filter_group_key("griz") == ("g", "r", "i", "z")
        assert parse_supported_filter_group_key("BV") == ("B", "V")
        assert parse_supported_filter_group_key("HK") == ("H", "K")

    def test_full_groups_unchanged(self):
        assert parse_supported_filter_group_key("UBVRI") == ("U", "B", "V", "R", "I")
        assert parse_supported_filter_group_key("ugriz") == ("u", "g", "r", "i", "z")
        assert parse_supported_filter_group_key("extended") == ("Y", "w")

    def test_composite_keys_unchanged(self):
        assert parse_supported_filter_group_key("grizJHK") == tuple("grizJHK")
        assert parse_supported_filter_group_key("grizjhk") == tuple("grizJHK")
        assert parse_supported_filter_group_key("GRIZJHK") == tuple("grizJHK")

    def test_separators(self):
        assert parse_supported_filter_group_key("u,RI") == ("u", "R", "I")
        assert parse_supported_filter_group_key("u | RI") == ("u", "R", "I")
        assert parse_supported_filter_group_key("UBVRI,JHK") == tuple("UBVRIJHK")

    def test_lowercase_all_unknown_rescued(self):
        # No valid band letters at all -> whole-token uppercase rescue.
        assert parse_supported_filter_group_key("jhk") == ("J", "H", "K")
        assert parse_supported_filter_group_key("bv") == ("B", "V")

    def test_case_never_swaps_bands(self):
        # A token mixing real band letters with unknown ones fails: the
        # rescue must not turn r/i/u into R/I/U.
        assert parse_supported_filter_group_key("bvri") is None
        assert parse_supported_filter_group_key("ubvri") is None
        assert parse_supported_filter_group_key("bvrI") is None
        # w is a real lowercase band, so "yw" fails rather than
        # half-promoting to Y,w.
        assert parse_supported_filter_group_key("yw") is None

    def test_written_case_preserved(self):
        assert parse_supported_filter_group_key("uri") == ("u", "r", "i")
        assert parse_supported_filter_group_key("URI") == ("U", "R", "I")
        assert parse_supported_filter_group_key("gri") == ("g", "r", "i")

    def test_unknown_letters_rejected(self):
        assert parse_supported_filter_group_key("xyz") is None
        assert parse_supported_filter_group_key("uXQ") is None

    def test_bad_token_rejects_whole_key(self):
        # Partially valid key must not silently map fewer bands.
        assert parse_supported_filter_group_key("u,XQ") is None

    def test_dedup_and_none(self):
        assert parse_supported_filter_group_key("gg") == ("g",)
        assert parse_supported_filter_group_key(None) is None
        assert parse_supported_filter_group_key("") is None
        assert parse_supported_filter_group_key("   ") is None


class TestAsciiOutputFormat:
    """Shared ASCII output helpers keep a consistent table style."""

    def test_border_msg_framed(self):
        from autophot.core.functions import border_msg

        out = border_msg("Source detection")
        lines = out.split("\n")
        # Leading blank line separates the banner from preceding output.
        assert lines[0] == ""
        assert lines[1] == "=" * 80
        assert lines[2] == "Source detection"
        assert lines[3] == "=" * 80

    def test_log_step_framed_dashes(self):
        from autophot.core.functions import log_step

        out = log_step("Sub step")
        lines = out.split("\n")
        assert lines[0] == ""
        assert lines[1] == "-" * 70
        assert lines[2] == "Sub step"
        assert lines[3] == "-" * 70

    def test_border_msg_metadata_inside_frame(self):
        from autophot.core.functions import border_msg

        out = border_msg("Title", metadata="a=1 b=2", width=60)
        lines = out.split("\n")
        assert lines[0] == ""
        assert lines[1] == "=" * 60
        assert lines[2] == "Title"
        assert lines[3] == "a=1 b=2"
        assert lines[4] == "=" * 60

    def test_ascii_kv_framed(self):
        from autophot.core.functions import ascii_kv

        out = ascii_kv("IMAGE", [("Exposure time", "30 s"), ("Airmass", "1.9")],
                       framed=True, width=60)
        lines = out.split("\n")
        assert lines[0] == ""
        assert lines[1] == "=" * 60
        assert lines[2] == "IMAGE"
        assert lines[3] == "=" * 60
        assert "Exposure time" in lines[4]
        assert "30 s" in lines[4]
        assert lines[-1] == "=" * 60

    def test_ascii_kv_sub_block(self):
        from autophot.core.functions import ascii_kv

        out = ascii_kv("Attrition", [("generated", "300")], width=50)
        lines = out.split("\n")
        assert lines[0] == ""
        assert lines[1] == "Attrition:"
        assert lines[2].strip().startswith("generated")
        assert "300" in lines[2]

    def test_ascii_kv_empty(self):
        from autophot.core.functions import ascii_kv

        assert ascii_kv("Empty", []) == ""

    def test_ascii_table(self):
        from autophot.core.functions import ascii_table

        out = ascii_table(
            "Zeropoint",
            ["Method", "ZP", "err", "N"],
            [["AP", "24.547", "0.003", "159"], ["PSF", "24.635", "0.005", "157"]],
            width=60,
        )
        lines = out.split("\n")
        assert lines[0] == ""
        assert lines[1] == "-" * 60
        assert lines[2] == "Zeropoint"
        assert lines[3] == "-" * 60
        assert "Method" in lines[4] and "ZP" in lines[4]
        # Column underline row: only dashes and separating spaces.
        assert set(lines[5].strip()) <= {"-", " "} and "-" in lines[5]
        assert "AP" in lines[6] and "24.547" in lines[6]
        assert "PSF" in lines[7]
        assert lines[-1] == "-" * 60
        # No trailing whitespace on any line.
        assert all(ln == ln.rstrip() for ln in lines)

    def test_ascii_table_empty(self):
        from autophot.core.functions import ascii_table

        assert ascii_table("T", ["a"], []) == ""

    def test_ascii_card(self):
        from autophot.core.functions import ascii_card

        out = ascii_card("Error", ["line one", "", "line two"], width=50)
        lines = out.split("\n")
        assert lines[0] == ""
        assert lines[1] == "=" * 50
        assert lines[2] == "Error"
        assert lines[3] == "=" * 50
        assert "line one" in lines[4]
        assert lines[5] == ""  # blank body lines preserved
        assert "line two" in lines[6]
        assert lines[-1] == "=" * 50
        assert ascii_card("Empty", ["", ""]) == ""


# ---------------------------------------------------------------------------
# get_image_and_header: unscaled-integer FITS warning
# ---------------------------------------------------------------------------

class TestUnscaledIntWarning:
    """Warn when integer-stored FITS data arrives with BLANK but no
    BZERO/BSCALE - the signature of a scaled/compressed HDU rewritten
    with do_not_scale_image_data=True (raw stored codes left in the
    pixels, e.g. the ~-44k PS1-warp pedestal that produced a -32.6k ADU
    'background')."""

    def _write_int_fits(self, path, median=-32600.0, with_scaling=False):
        from astropy.io import fits

        data = np.full((32, 32), int(median), dtype=np.int16)
        data[0, 0] = 32767
        hdr = fits.Header()
        hdr["BLANK"] = 32767
        if with_scaling:
            hdr["BZERO"] = 44044.65918338
            hdr["BSCALE"] = 1.350712392867
        fits.writeto(path, data, hdr, output_verify="ignore")
        return path

    def test_warns_on_blank_without_bzero(self, tmp_path, caplog):
        import logging
        from autophot.core.functions import get_image_and_header

        f = self._write_int_fits(tmp_path / "unscaled.fits")
        with caplog.at_level(logging.WARNING, logger="functions"):
            image, header = get_image_and_header(str(f))
        assert "BZERO" in caplog.text
        assert "scaling" in caplog.text.lower()

    def test_no_warning_for_positive_median(self, tmp_path, caplog):
        import logging
        from autophot.core.functions import get_image_and_header

        f = self._write_int_fits(tmp_path / "normal.fits", median=1500)
        with caplog.at_level(logging.WARNING, logger="functions"):
            get_image_and_header(str(f))
        assert "raw stored integers" not in caplog.text


class TestBiweightSkySigma:
    """biweight_sky_sigma is the canonical pixel-noise estimator: it must
    stay continuous on quantized stacks (where MAD staircases) and return
    finite, positive values or NaN - never a silent wrong answer."""

    def test_recovers_gaussian_sigma(self):
        from autophot.core.functions import biweight_sky_sigma

        rng = np.random.default_rng(42)
        data = rng.normal(100.0, 5.0, 400 * 400)
        sig = biweight_sky_sigma(data)
        assert sig == pytest.approx(5.0, rel=0.05)

    def test_continuous_on_quantized_data(self):
        """Quantized sky noise (delta-spaced lattice) must not snap the
        estimate to an integer multiple of delta * 1.4826 the way MAD does."""
        from astropy.stats import mad_std
        from autophot.core.functions import biweight_sky_sigma

        rng = np.random.default_rng(7)
        delta = 1.34  # ADU quantization step, as on PS1 pswarp stacks
        true_sigma = 15.0
        q = np.round(rng.normal(0.0, true_sigma, 200 * 200) / delta) * delta

        mad_val = mad_std(q, ignore_nan=True)
        bw_val = biweight_sky_sigma(q)

        # MAD sits on a staircase rung: an exact multiple of delta*1.4826.
        rung = delta * 1.4826
        assert abs(mad_val / rung - round(mad_val / rung)) < 1e-3
        # Biweight is continuous and closer to the true width.
        assert abs(bw_val / rung - round(bw_val / rung)) > 1e-3
        assert abs(bw_val - true_sigma) <= abs(mad_val - true_sigma)

    def test_ignores_nan_and_mask(self):
        from autophot.core.functions import biweight_sky_sigma

        rng = np.random.default_rng(1)
        data = rng.normal(0.0, 2.0, (100, 100))
        data[10:20, 10:20] = np.nan
        mask = np.zeros_like(data, dtype=bool)
        mask[80:, :] = True  # exclude a chunk of real pixels
        sig = biweight_sky_sigma(data, mask=mask)
        assert np.isfinite(sig)
        assert sig == pytest.approx(2.0, rel=0.1)

    def test_all_nan_returns_nan(self):
        from autophot.core.functions import biweight_sky_sigma

        assert np.isnan(biweight_sky_sigma(np.full((10, 10), np.nan)))

    def test_constant_returns_zero(self):
        """A constant image honestly has zero pixel noise; callers guard
        sig <= 0 themselves."""
        from autophot.core.functions import biweight_sky_sigma

        assert biweight_sky_sigma(np.full((50, 50), 3.0)) == 0.0

    def test_get_image_stats_uses_biweight(self):
        """get_image_stats' std must come from the same continuous estimator
        as the rest of the pipeline (not mad_std)."""
        from astropy.stats import mad_std
        from autophot.core.functions import get_image_stats

        rng = np.random.default_rng(3)
        delta = 1.34
        q = np.round(rng.normal(0.0, 15.0, (150, 150)) / delta) * delta
        _, _, std_val = get_image_stats(q, sigma=3.0)
        rung = delta * 1.4826
        assert abs(std_val / rung - round(std_val / rung)) > 1e-3
        assert std_val != pytest.approx(mad_std(q, ignore_nan=True))


# ---------------------------------------------------------------------------
# Input-config validation (check_input_config)
# ---------------------------------------------------------------------------

class TestInputConfigValidation:
    """Unrecognised keys and out-of-range dispatch values must surface
    immediately (main.py -c file and the driver dict share the same
    schema check) rather than silently running with defaults."""

    def test_shipped_schema_validates_clean(self):
        # The shipped defaults (plus the load()-added empty sections)
        # must produce an empty report -- a generated input.yaml always
        # round-trips.
        report = check_input_config(load_default_input_schema())
        assert report["unknown"] == []
        assert report["deprecated"] == []
        assert report["bad_values"] == []

    def test_typo_key_flagged_with_suggestion(self):
        report = check_input_config(
            {"photometry": {"apertuer_radius": 5.0}}
        )
        assert "photometry.apertuer_radius" in report["unknown"]
        assert "photometry.aperture_radius" in format_config_errors(report)

    def test_nested_unknown_key_flagged(self):
        report = check_input_config(
            {"template_subtraction": {"sfft_min_class_star": 0.5}}
        )
        assert report["unknown"] == [
            "template_subtraction.sfft_min_class_star"
        ]

    def test_runtime_and_data_keys_skipped(self):
        # Generated input.yaml files carry pipeline-injected payloads that
        # are not user options.
        report = check_input_config(
            {
                "_additional_targets_resolved": [{"x": 1}],
                "variable_sources": [{"a": 1}],
                "name_prefix": "SN",
                "objname": "2024pba",
            }
        )
        assert report["unknown"] == []

    def test_deprecated_key_warns_not_fails(self):
        report = check_input_config(
            {"template_subtraction": {"sfft_forceconv": "SCI"}}
        )
        assert report["unknown"] == []
        assert report["deprecated"] == [
            (
                "template_subtraction.sfft_forceconv",
                "template_subtraction.forceconv",
            )
        ]

    def test_removed_key_warns_not_fails(self):
        # zogy_update shipped in old default_input.yml versions; existing
        # configs must keep working after the key was dropped.
        report = check_input_config(
            {"template_subtraction": {"zogy_update": True}}
        )
        assert report["unknown"] == []
        assert report["removed"] == ["template_subtraction.zogy_update"]
        assert format_config_errors(report) == ""

    def test_bad_enum_values_flagged(self):
        report = check_input_config(
            {
                "template_subtraction": {
                    "method": "SFTT",
                    "forceconv": "FOO",
                },
                "wcs": {"solver": "scmap"},
                "catalog": {
                    "use_catalog": {"g": "gaia", "r": "notacatalog"}
                },
            }
        )
        paths = {p for p, _, _ in report["bad_values"]}
        assert "template_subtraction.method" in paths
        assert "template_subtraction.forceconv" in paths
        assert "wcs.solver" in paths
        assert "catalog.use_catalog" in paths

    def test_scalar_where_section_expected_flagged(self):
        report = check_input_config({"photometry": 5})
        assert report["unknown"]

    def test_null_section_is_unset_not_invalid(self):
        report = check_input_config({"wcs": {"crowded": None}})
        assert report["unknown"] == []

    def test_strict_env_toggle(self, monkeypatch):
        monkeypatch.delenv("AUTOPHOT_STRICT_CONFIG", raising=False)
        assert strict_config_enabled()
        monkeypatch.setenv("AUTOPHOT_STRICT_CONFIG", "0")
        assert not strict_config_enabled()


class TestFWHMOutlierClip:
    """fwhm_outlier_clip: MAD sigma-clip with a relative scatter floor.

    The floor matters in two regimes the plain estimators each fail:
    a plain-std clip absorbs the outliers it should reject (std is
    inflated by them), and an unfloored MAD clip collapses onto the
    majority locus on a defect-dominated pool and flags the real stars.
    """

    def test_rejects_extended_outliers(self):
        from autophot.core.functions import fwhm_outlier_clip

        vals = np.array([4.0, 4.1, 3.9, 4.05, 4.0, 8.0, 12.0])
        rej, med, sig = fwhm_outlier_clip(vals, sigma=2.5)
        assert rej.tolist() == [False] * 5 + [True, True]
        assert med == pytest.approx(4.05, abs=0.1)

    def test_defect_dominated_pool_keeps_stellar_side(self):
        # 9 defect rows at fwhm 3.5, 4 genuine stars at 4.0: MAD is
        # pinned to the defect locus (MAD=0); the relative floor keeps
        # the 12%-offset stars in the band.
        from autophot.core.functions import fwhm_outlier_clip

        vals = np.concatenate([np.full(9, 3.5), np.full(4, 4.0)])
        rej, med, sig = fwhm_outlier_clip(vals, sigma=2.5)
        assert not rej.any()
        assert sig == pytest.approx(0.10 * 3.5)

    def test_nonfinite_kept(self):
        from autophot.core.functions import fwhm_outlier_clip

        vals = np.array([4.0, np.nan, -1.0, 4.2, 20.0])
        rej, _, _ = fwhm_outlier_clip(vals, sigma=2.5)
        assert rej.tolist() == [False, False, False, False, True]

    def test_sigma_zero_disables(self):
        from autophot.core.functions import fwhm_outlier_clip

        vals = np.array([4.0, 4.1, 20.0])
        rej, med, sig = fwhm_outlier_clip(vals, sigma=0.0)
        assert not rej.any()
        assert not np.isfinite(med)


class TestCanonicalTargetName:
    """The catalog cache label must agree across the driver pre-fetch,
    per-image main.py, and catalog.download no matter how target
    identification was supplied."""

    def test_real_name_is_normalized(self):
        assert (
            canonical_target_name({"target_name": "SN2024pba"}) == "2024pba"
        )

    def test_none_name_with_coords_is_transient(self):
        cfg = {"target_name": None, "target_ra": 25.3, "target_dec": 49.0}
        assert canonical_target_name(cfg) == "Transient"

    def test_blank_name_with_coords_is_transient(self):
        cfg = {"target_name": "  ", "target_ra": 25.3, "target_dec": 49.0}
        assert canonical_target_name(cfg) == "Transient"

    def test_missing_name_with_coords_is_transient(self):
        cfg = {"target_ra": 25.3, "target_dec": 49.0}
        assert canonical_target_name(cfg) == "Transient"

    def test_placeholder_name_passes_through(self):
        cfg = {
            "target_name": "Transient",
            "target_ra": 25.3,
            "target_dec": 49.0,
        }
        assert canonical_target_name(cfg) == "Transient"

    def test_nothing_given_is_center_of_field(self):
        assert canonical_target_name({}) == "Center of Field"


class TestExternalWeightToRms:
    """external_weight_to_rms folds a delivered companion map into the
    measured RMS map.  SWarp-style weights carry relative depth; the
    measured map keeps the absolute scale.  Maps marked MAPTYPE='RMS'
    fold as sigma ratios.  Unusable companions return None."""

    def _write(self, path, data, header=None):
        from astropy.io import fits

        hdr = fits.Header() if header is None else header
        fits.writeto(path, np.asarray(data, dtype=np.float32), hdr)
        return str(path)

    def test_flat_weight_preserves_ref(self, tmp_path):
        from astropy.io import fits

        from autophot.core.functions import external_weight_to_rms

        p = self._write(tmp_path / "w.fits", np.ones((20, 20)))
        ref = np.full((20, 20), 2.5, np.float32)
        out = external_weight_to_rms(p, ref, fits.getheader(p))
        np.testing.assert_allclose(out, ref)

    def test_inverse_variance_structure(self, tmp_path):
        from astropy.io import fits

        from autophot.core.functions import external_weight_to_rms

        w = np.ones((20, 20), np.float32)
        w[:, 10:] = 4.0  # 4x deeper coverage -> half the sigma
        p = self._write(tmp_path / "w.fits", w)
        ref = np.full((20, 20), 2.5, np.float32)
        out = external_weight_to_rms(p, ref, fits.getheader(p))
        ratio = out[10, 5] / out[10, 15]
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_invalid_weight_pixels_become_nan(self, tmp_path):
        from astropy.io import fits

        from autophot.core.functions import external_weight_to_rms

        w = np.ones((20, 20), np.float32)
        w[:, :3] = 0.0
        w[:, -2:] = np.nan
        p = self._write(tmp_path / "w.fits", w)
        ref = np.full((20, 20), 2.5, np.float32)
        out = external_weight_to_rms(p, ref, fits.getheader(p))
        assert np.isnan(out[:, :3]).all()
        assert np.isnan(out[:, -2:]).all()
        assert np.isfinite(out[:, 3:-2]).all()

    def test_rms_maptype_folds_sigma_ratio(self, tmp_path):
        from astropy.io import fits

        from autophot.core.functions import external_weight_to_rms

        hdr = fits.Header()
        hdr["MAPTYPE"] = "RMS"
        m = np.linspace(1.0, 4.0, 100, dtype=np.float32).reshape(10, 10)
        p = self._write(tmp_path / "rms.fits", m, header=hdr)
        ref = np.full((10, 10), 3.0, np.float32)
        out = external_weight_to_rms(p, ref, fits.getheader(p))
        # Ratio structure is exact (rel = sigma/median-sigma); the absolute
        # scale tracks ref within the even-count median normalization.
        assert out[5, 9] / out[5, 0] == pytest.approx(
            m[5, 9] / m[5, 0], rel=1e-6
        )
        np.testing.assert_allclose(out, 3.0 * m / np.median(m), rtol=1e-3)

    def test_missing_file_returns_none(self, tmp_path):
        from autophot.core.functions import external_weight_to_rms

        ref = np.full((10, 10), 1.0, np.float32)
        assert external_weight_to_rms(None, ref, None) is None
        assert (
            external_weight_to_rms(tmp_path / "nope.fits", ref, None) is None
        )

    def test_shape_mismatch_without_wcs_returns_none(self, tmp_path):
        from astropy.io import fits

        from autophot.core.functions import external_weight_to_rms

        p = self._write(tmp_path / "w.fits", np.ones((8, 8)))
        ref = np.full((10, 10), 1.0, np.float32)
        assert external_weight_to_rms(p, ref, fits.getheader(p)) is None

    def test_wcs_reprojection_keeps_footprint(self, tmp_path):
        pytest.importorskip("reproject")
        from astropy.io import fits
        from astropy.wcs import WCS

        from autophot.core.functions import external_weight_to_rms

        def _wcs(n, cd):
            w = WCS(naxis=2)
            w.wcs.crpix = [n / 2, n / 2]
            w.wcs.crval = [150.0, 2.0]
            w.wcs.cdelt = [cd, cd]
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            return w

        w = np.ones((20, 20), np.float32)
        w[:, 10:] = 4.0
        p = self._write(
            tmp_path / "w_big.fits", w, header=_wcs(20, 0.001).to_header()
        )
        ref = np.full((10, 10), 3.0, np.float32)
        out = external_weight_to_rms(
            p, ref, _wcs(10, 0.002).to_header()
        )
        assert out.shape == (10, 10)
        # 4:1 depth step survives the resample onto the coarser grid
        assert out[5, 0] / out[5, -1] == pytest.approx(2.0, rel=0.05)
