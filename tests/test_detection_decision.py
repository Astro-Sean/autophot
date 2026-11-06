"""Regression tests for ``main._decide_detection``.

The per-image detection gate decides whether a forced-photometry row is a
detection or an upper limit.  The SN2024pba run reported genuine ~12-sigma
matched-filter signals as limiting magnitudes because the photometric SNR
was suppressed by an inflated error budget; the matched-filter rescue
(criterion 5b) recovers those rows while the false-alarm and centroid
vetoes still apply.
"""

import numpy as np
import pytest

from main import _decide_detection


def _out(**kw):
    """Minimal output dict for _decide_detection."""
    base = {
        "snr": np.nan,
        "snr_ap": np.nan,
        "flux_psf": np.nan,
        "flux_ap": np.nan,
        "mf_snr": np.nan,
        "prob_detect": np.nan,
        "reduced_chi2": np.nan,
        "empirical_p_false": np.nan,
        "empirical_n_sites": 0,
        "separation": np.nan,
    }
    base.update(kw)
    return base


class TestDecideDetectionPrimary:
    """Conventional gates keep working unchanged."""

    def test_high_snr_positive_flux_detects(self):
        out = _out(snr=6.0, flux_psf=120.0, flux_ap=110.0)
        assert _decide_detection(out, {}, 3.0, False, False)

    def test_low_snr_is_nondetection(self):
        out = _out(snr=1.0, flux_psf=5.0, flux_ap=4.0)
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_negative_flux_rejected(self):
        out = _out(snr=8.0, flux_psf=-80.0, flux_ap=-70.0)
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_bad_chi2_rejected(self):
        out = _out(
            snr=6.0, flux_psf=120.0, flux_ap=110.0, reduced_chi2=50.0
        )
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_ap_fallback(self):
        out = _out(snr=2.0, snr_ap=6.0, flux_psf=4.0, flux_ap=30.0)
        assert _decide_detection(out, {}, 3.0, False, False)


class TestDecideDetectionMfRescue:
    """Criterion 5b: strong matched-filter statistics rescue a real
    signal whose photometric SNR was crushed by systematic error terms."""

    def test_sn2024pba_epoch_rescued(self):
        # 2024-08-09 g-band: photometric SNR 1.4 from an inflated error
        # budget, but mf_snr 12.3 / prob_detect 1.0 / positive flux.
        out = _out(
            snr=1.42,
            snr_ap=2.1,
            flux_psf=88.88,
            flux_ap=95.0,
            mf_snr=12.26,
            prob_detect=1.0,
            reduced_chi2=1.3,
        )
        assert _decide_detection(out, {}, 3.0, False, False)

    def test_rescue_needs_prob_detect(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=90.0,
            mf_snr=12.0,
            prob_detect=0.5,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_rescue_needs_mf_floor(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=90.0,
            mf_snr=3.0,
            prob_detect=1.0,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_negative_dip_never_rescued(self):
        # A deep negative residual can score a large |mf_snr|; it is a
        # noise dip or oversubtraction, not a detection.
        out = _out(
            snr=1.4,
            flux_psf=-90.0,
            flux_ap=-85.0,
            mf_snr=15.0,
            prob_detect=0.99,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_empirical_fap_still_vetoes_rescue(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=90.0,
            mf_snr=12.0,
            prob_detect=1.0,
            empirical_p_false=0.5,
            empirical_n_sites=60,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_centroid_veto_still_applies(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=90.0,
            mf_snr=12.0,
            prob_detect=1.0,
            separation=5.0,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_rescue_configurable(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=90.0,
            mf_snr=6.0,
            prob_detect=1.0,
        )
        assert _decide_detection(out, {}, 3.0, False, False)
        assert not _decide_detection(
            out, {"detection_mf_snr_min": 10.0}, 3.0, False, False
        )

    def test_inverted_fit_not_rescued(self):
        out = _out(
            snr=1.4,
            flux_psf=-90.0,
            flux_ap=-85.0,
            mf_snr=15.0,
            prob_detect=1.0,
        )
        assert not _decide_detection(out, {}, 3.0, False, True)

    def test_aperture_only_skips_rescue(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=90.0,
            mf_snr=15.0,
            prob_detect=1.0,
        )
        assert not _decide_detection(out, {}, 3.0, True, False)

    def test_bad_chi2_not_rescued(self):
        # 2025-02-10 g-band: snr 4.1 passes the SNR gate outright and
        # mf_snr 20.9 is huge, but reduced_chi2=15.5 means the PSF fit
        # failed -- the quality veto stands and the rescue must not
        # re-promote the row.
        out = _out(
            snr=4.08,
            snr_ap=1.6,
            flux_psf=126.2,
            flux_ap=33.3,
            mf_snr=20.94,
            prob_detect=1.0,
            delta_chi2=438.7,
            reduced_chi2=15.52,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)

    def test_inconsistent_flux_not_rescued(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=-40.0,
            mf_snr=12.0,
            prob_detect=1.0,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)
        # With the consistency check disabled the same row rescues.
        assert _decide_detection(
            out, {"detection_flux_consistency": False}, 3.0, False, False
        )

    def test_negative_delta_chi2_not_rescued(self):
        out = _out(
            snr=1.4,
            flux_psf=90.0,
            flux_ap=90.0,
            mf_snr=12.0,
            prob_detect=1.0,
            delta_chi2=-3.0,
        )
        assert not _decide_detection(out, {}, 3.0, False, False)
