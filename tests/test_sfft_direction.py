"""Regression tests for SFFT convolution-direction selection and the
ConstPhotRatio retry adoption metric.

_select_forceconv resolves forceconv=REF/SCI/AUTO using the measured
post-alignment FWHMs so that SFFT's internal header-FWHM AUTO (which SWarp
resampling can mislead, BUG 122) is never reached.  _diff_resid_at_sources
measures the residual flux at matched-source positions and is used to adopt
or reject the ConstPhotRatio retry (ConstPhotRatio changes the spatial
uniformity of the solved flux scaling; it does not repin the kernel sum to
the SExtractor photometric ratio, so the FSCAL metric cannot judge it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions import (
    _sfft_actual_convd,
    _sfft_difference_gain,
    _sfft_photometric_flux_correction,
)
from templates import (
    _apply_sfft_deconvolution_veto,
    _diff_resid_at_sources,
    _select_forceconv,
)


class TestSelectForceconv:
    def test_explicit_ref(self):
        # 2.46 vs 2.53 = 2.8% gap < 5% tol -> not flagged as deconvolution
        d, deconv, _ = _select_forceconv("REF", 2.46, 2.53)
        assert d == "REF"
        assert deconv is False

    def test_explicit_ref_beyond_tol(self):
        # template clearly broader -> convolving REF sharpens it -> flag
        d, deconv, _ = _select_forceconv("REF", 2.0, 3.0)
        assert d == "REF"
        assert deconv is True

    def test_explicit_ref_no_deconv(self):
        d, deconv, _ = _select_forceconv("REF", 3.0, 2.5)
        assert d == "REF"
        assert deconv is False

    def test_explicit_sci(self):
        d, deconv, _ = _select_forceconv("SCI", 2.46, 2.53)
        assert d == "SCI"
        # convolving science (2.46) up to template (2.53): well-posed blur
        assert deconv is False

    def test_explicit_sci_deconv(self):
        d, deconv, _ = _select_forceconv("SCI", 3.0, 2.5)
        assert d == "SCI"
        assert deconv is True

    def test_auto_template_broader(self):
        # template 3.0 vs science 2.0 -> convolve science (well-posed)
        d, deconv, _ = _select_forceconv("AUTO", 2.0, 3.0)
        assert d == "SCI"
        assert deconv is False

    def test_auto_science_broader(self):
        d, deconv, _ = _select_forceconv("AUTO", 3.0, 2.0)
        assert d == "REF"
        assert deconv is False

    def test_auto_within_tol_keeps_ref(self):
        # 2.46 vs 2.53 is 2.8% apart < 5% tol -> keep REF, not deconvolution
        d, deconv, note = _select_forceconv("AUTO", 2.46, 2.53)
        assert d == "REF"
        assert deconv is False
        assert "AUTO->REF" in note

    def test_auto_tol_configurable(self):
        # with a 1% tolerance the same pair flips to SCI
        d, deconv, _ = _select_forceconv("AUTO", 2.46, 2.53, auto_tol=0.01)
        assert d == "SCI"
        assert deconv is False

    def test_unknown_defaults_ref(self):
        d, _, _ = _select_forceconv("bogus", 2.5, 2.5)
        assert d == "REF"

    def test_nan_fwhm_safe(self):
        d, deconv, _ = _select_forceconv("AUTO", np.nan, 2.5)
        assert d == "REF"
        assert deconv is False
        d2, deconv2, _ = _select_forceconv("AUTO", 2.5, np.inf)
        assert d2 == "REF"
        assert deconv2 is False

    def test_shipped_default_is_ref(self):
        # The packaged default is REF: the difference image keeps the
        # native science PSF so the science ePSF applies directly and no
        # PSF-shape correction can bias the measured flux.  When the
        # template is broader than the science image the direction is
        # physically impossible (deconvolution), so the kernel-probe veto
        # -- not this resolution -- protects that case.
        import yaml

        cfg = yaml.safe_load((ROOT / "databases" / "default_input.yml").read_text())
        ts = cfg["default_input"]["template_subtraction"]
        cfg_dir = str(ts["forceconv"]).strip().upper()
        assert cfg_dir == "REF"
        assert ts["sfft_veto_deconvolution"] is True
        tol = float(ts["sfft_forceconv_auto_tol"])
        # Explicit REF always resolves to REF; the deconvolution flag is
        # raised so the caller can veto when the request is impossible.
        d, deconv, _ = _select_forceconv(cfg_dir, 3.0, 2.0, auto_tol=tol)
        assert d == "REF" and deconv is False
        d, deconv, _ = _select_forceconv(cfg_dir, 2.0, 3.0, auto_tol=tol)
        assert d == "REF" and deconv is True
        d, deconv, _ = _select_forceconv(cfg_dir, 2.46, 2.53, auto_tol=tol)
        assert d == "REF" and deconv is False

    def test_auto_never_deconvolves(self):
        # AUTO is the opt-in escape from deconvolution: across FWHM orderings
        # it must always resolve to the well-posed direction.
        for fs, ft in [(2.0, 3.0), (3.0, 2.0), (2.5, 2.5), (2.46, 2.53)]:
            d, deconv, _ = _select_forceconv("AUTO", fs, ft)
            assert d in ("REF", "SCI")
            assert deconv is False


class TestSfftDeconvolutionVeto:
    def test_ref_sharpening_flips_to_sci(self):
        d, deconv, _ = _select_forceconv("REF", 1.77, 2.39)
        assert (d, deconv) == ("REF", True)
        d, deconv, vetoed = _apply_sfft_deconvolution_veto(d, deconv)
        assert (d, deconv, vetoed) == ("SCI", False, True)

    def test_sci_sharpening_flips_to_ref(self):
        d, deconv, vetoed = _apply_sfft_deconvolution_veto("SCI", True)
        assert (d, deconv, vetoed) == ("REF", False, True)

    def test_veto_can_be_disabled(self):
        d, deconv, vetoed = _apply_sfft_deconvolution_veto(
            "REF", True, enabled=False
        )
        assert (d, deconv, vetoed) == ("REF", True, False)

    def test_well_posed_direction_unchanged(self):
        d, deconv, vetoed = _apply_sfft_deconvolution_veto("REF", False)
        assert (d, deconv, vetoed) == ("REF", False, False)


class TestSfftPhotometricScale:
    def test_convd_overrides_requested_forcecon(self):
        hdr = {
            "FORCECON": "REF",
            "CONVD": "SCI",
            "FSCAL_CONV": 0.95,
            "FSCAL_PHOT": 1.00,
            "FSCAL_DISC": 5.0,
        }
        assert _sfft_actual_convd(hdr) == "SCI"
        assert _sfft_photometric_flux_correction(hdr) == pytest.approx(
            1.00 / 0.95
        )

    def test_calibrated_magnitude_matches_across_directions(self):
        # A science transient of F counts appears as F in CONVD=REF and as
        # F * FSCAL_CONV in CONVD=SCI.  The effective gain must cancel the
        # SFFT pixel scale so both paths share the science-image zeropoint.
        true_adu = 1000.0
        gain_sci = 6.5
        fscal = 0.9566976121
        hdr_ref = {"CONVD": "REF", "FSCAL_CONV": 1.0418340538}
        hdr_sci = {
            "CONVD": "SCI",
            "FSCAL_CONV": fscal,
            "GAIN": gain_sci / fscal,
        }

        flux_ref = true_adu * _sfft_difference_gain(hdr_ref, gain_sci)
        flux_sci = (
            true_adu
            * fscal
            * _sfft_difference_gain(hdr_sci, gain_sci)
        )
        assert flux_sci == pytest.approx(flux_ref, rel=1e-12)

        # Calibrate through a science-image calibrator so the check covers
        # the same flux -> instrumental magnitude -> zeropoint path as the
        # pipeline, not just raw electron invariance.
        exptime = 30.0
        cal_adu = 5000.0
        cal_mag = 18.0
        zeropoint = cal_mag + 2.5 * np.log10(
            cal_adu * gain_sci / exptime
        )
        mag_ref = -2.5 * np.log10(flux_ref / exptime) + zeropoint
        mag_sci = -2.5 * np.log10(flux_sci / exptime) + zeropoint
        assert mag_sci == pytest.approx(mag_ref)

    def test_convolved_science_flux_error_keeps_science_rate_scale(self):
        gain_sci = 6.5
        fscal = 0.72
        exptime = 180.0
        source_adu = 2400.0
        background_rms_adu = 3.5
        diff_gain = _sfft_difference_gain(
            {"CONVD": "SCI", "FSCAL_CONV": fscal}, gain_sci
        )

        science_variance_e = source_adu * gain_sci + (
            background_rms_adu * gain_sci
        ) ** 2
        difference_variance_e = (
            source_adu * fscal * diff_gain
            + (background_rms_adu * fscal * diff_gain) ** 2
        )

        science_error_rate = np.sqrt(science_variance_e) / exptime
        difference_error_rate = np.sqrt(difference_variance_e) / exptime
        assert difference_error_rate == pytest.approx(science_error_rate)

    def test_photometric_correction_keeps_science_scale(self):
        # When the SCI diff is rescaled from FSCAL_CONV to FSCAL_PHOT, the
        # gain must use FSCAL_PHOT.  Pairing the corrected image with either
        # the science gain or FSCAL_CONV gain would shift the calibrated mag.
        true_adu = 1000.0
        gain_sci = 6.5
        hdr = {
            "CONVD": "SCI",
            "FSCAL_CONV": 0.90,
            "FSCAL_PHOT": 1.10,
            "FSCAL_DISC": 18.0,
        }
        correction = _sfft_photometric_flux_correction(hdr)
        assert correction == pytest.approx(1.10 / 0.90)
        gain_diff = _sfft_difference_gain(
            hdr, gain_sci, flux_corrected=True
        )
        corrected_e = true_adu * 0.90 * correction * gain_diff
        assert corrected_e == pytest.approx(true_adu * gain_sci)

    def test_ref_direction_never_gets_whole_image_scale(self):
        # The old REF direction scales only the convolved reference; the
        # transient itself remains in native science ADU.
        hdr = {
            "CONVD": "REF",
            "FSCAL_CONV": 0.8539798,
            "FSCAL_PHOT": 1.0407898,
            "FSCAL_DISC": 17.95,
        }
        assert _sfft_photometric_flux_correction(hdr) is None
        assert _sfft_difference_gain(hdr, 6.5) == pytest.approx(6.5)

    def test_auto_and_header_fallbacks(self):
        assert _sfft_actual_convd(
            {"FORCECON": "AUTO", "FWHM_SCI": 2.0, "FWHM_REF": 3.0}
        ) == "SCI"
        assert _sfft_actual_convd(
            {"FORCECON": "AUTO", "FWHM_SCI": 3.0, "FWHM_REF": 2.0}
        ) == "REF"
        # Very old SFFT products may lack FSCAL_* but still carry the adjusted
        # difference-image gain.
        assert _sfft_difference_gain(
            {"CONVD": "SCI", "GAIN": 6.8}, 6.5
        ) == pytest.approx(6.8)
        assert _sfft_difference_gain(
            {"CONVD": "SCI", "FSCAL": 0.95}, 6.5
        ) == pytest.approx(6.5 / 0.95)
        # A spatially varying kernel carries its own local flux scale; use
        # that target kernel sum instead of the field-mean FSCAL_CONV.
        assert _sfft_difference_gain(
            {"CONVD": "SCI", "FSCAL_CONV": 1.0},
            6.5,
            flux_scale=0.98,
        ) == pytest.approx(6.5 / 0.98)
        # An explicit local scale must not silently fall back to another
        # gain; a bad kernel realization means the target scale is unknown.
        assert np.isnan(
            _sfft_difference_gain(
                {"CONVD": "SCI", "FSCAL_CONV": 1.0},
                6.5,
                flux_scale=0.0,
            )
        )


class TestDiffResidAtSources:
    def _write_diff(self, tmp_path, data):
        from astropy.io import fits
        p = tmp_path / "diff.fits"
        fits.writeto(str(p), data.astype(np.float64), overwrite=True)
        return p

    def test_clean_diff_near_zero(self, tmp_path):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1.0, (60, 60))
        p = self._write_diff(tmp_path, data)
        m = _diff_resid_at_sources(str(p), [(30, 30), (20, 20), (45, 45)], radius=4)
        assert m is not None and m < 10.0

    def test_oversubtracted_sources_detected(self, tmp_path):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1.0, (60, 60))
        yy, xx = np.mgrid[0:60, 0:60]
        for x, y in [(30, 30), (20, 20), (45, 45)]:
            data += -50.0 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 1.0 ** 2)
        p = self._write_diff(tmp_path, data)
        m = _diff_resid_at_sources(str(p), [(30, 30), (20, 20), (45, 45)], radius=4)
        assert m is not None and m > 100.0

    def test_retry_adoption_direction(self, tmp_path):
        """A 'retry' diff with larger residuals scores higher -> reject."""
        rng = np.random.default_rng(2)
        first = rng.normal(0, 1.0, (60, 60))
        retry = first.copy()
        yy, xx = np.mgrid[0:60, 0:60]
        for x, y in [(30, 30), (20, 20), (45, 45)]:
            retry += -80.0 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 1.2 ** 2)
        srcs = [(30, 30), (20, 20), (45, 45)]
        pf = self._write_diff(tmp_path, first)
        pr = tmp_path / "retry.fits"
        from astropy.io import fits
        fits.writeto(str(pr), retry.astype(np.float64), overwrite=True)
        m_first = _diff_resid_at_sources(str(pf), srcs, radius=4)
        m_retry = _diff_resid_at_sources(str(pr), srcs, radius=4)
        assert m_retry > 1.10 * m_first

    def test_missing_inputs(self, tmp_path):
        assert _diff_resid_at_sources(str(tmp_path / "nope.fits"), [(1, 1)]) is None
        data = np.zeros((20, 20))
        p = self._write_diff(tmp_path, data)
        assert _diff_resid_at_sources(str(p), []) is None
        assert _diff_resid_at_sources(str(p), [(1, 1)]) is None  # <2 valid
