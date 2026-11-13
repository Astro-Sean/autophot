"""Tests for the post-processing transient color correction.

The correction solves m = m0 + c*(X - X_ref) and X = m1 - m2 by
fixed-point iteration over paired same-telescope / same-instrument
epochs.  Only rows whose filter is one band of their image's color
index and which have a partner band within dt days are corrected.
"""

import numpy as np
import pandas as pd

from autophot.calibration.color_correction import apply_transient_color_correction


def _row(fpath, mjd, filt, mag, err=0.02, tel="P48", inst="ZTF"):
    return dict(
        filename_path=fpath,
        mjd=mjd,
        filter=filt,
        telescope=tel,
        instrument=inst,
        mag_psf=mag,
        mag_psf_err=err,
        mag_ap=mag,
        mag_ap_err=err,
    )


def _apply(rows, terms, dt=1.0):
    df = pd.DataFrame(rows)
    return apply_transient_color_correction(df, terms, dt=dt)


class TestTransientColorCorrection:
    def test_pair_recovers_true_mags(self):
        # True g=18.0, r=17.0 (X=1.0); c_g=0.10, c_r=0.05, X_ref=0.5.
        # Observed mags are biased by c*(X - X_ref).
        rows = [
            _row("/d1/g.fits", 100.0, "g", 18.0 - 0.10 * 0.5),
            _row("/d2/r.fits", 100.5, "r", 17.0 - 0.05 * 0.5),
        ]
        terms = {
            "/d1": (0.10, 0.005, "g", "r", 0.5),
            "/d2": (0.05, 0.005, "g", "r", 0.5),
        }
        out = _apply(rows, terms)
        assert out.loc[0, "cc_applied_psf"]
        assert out.loc[1, "cc_applied_psf"]
        assert abs(out.loc[0, "mag_psf_cc"] - 18.0) < 1e-6
        assert abs(out.loc[1, "mag_psf_cc"] - 17.0) < 1e-6
        assert abs(out.loc[0, "cc_color_psf"] - 1.0) < 1e-6

    def test_iteration_matches_closed_form(self):
        # Linear two-equation system: X = (X0 - c1*xr1 + c2*xr2)/(1 - c1 + c2).
        m0g, m0r = 17.9, 16.9
        c_g, c_r, xr_g, xr_r = 0.12, 0.04, 0.6, 0.55
        rows = [_row("/d1/g.fits", 100.0, "g", m0g),
                _row("/d2/r.fits", 100.2, "r", m0r)]
        terms = {"/d1": (c_g, 0.005, "g", "r", xr_g),
                 "/d2": (c_r, 0.005, "g", "r", xr_r)}
        out = _apply(rows, terms)
        x0 = m0g - m0r
        x_exp = (x0 - c_g * xr_g + c_r * xr_r) / (1 - c_g + c_r)
        assert abs(out.loc[0, "cc_color_psf"] - x_exp) < 1e-5
        assert abs(out.loc[0, "mag_psf_cc"] - (m0g + c_g * (x_exp - xr_g))) < 1e-5
        assert abs(out.loc[1, "mag_psf_cc"] - (m0r + c_r * (x_exp - xr_r))) < 1e-5

    def test_no_partner_within_dt(self):
        rows = [
            _row("/d1/g.fits", 100.0, "g", 17.95),
            _row("/d2/r.fits", 105.0, "r", 16.975),  # 5 days away
        ]
        terms = {"/d1": (0.10, 0.005, "g", "r", 0.5),
                 "/d2": (0.05, 0.005, "g", "r", 0.5)}
        out = _apply(rows, terms, dt=1.0)
        assert not out["cc_applied_psf"].any()

    def test_different_telescope_not_paired(self):
        rows = [
            _row("/d1/g.fits", 100.0, "g", 17.95, tel="P48", inst="ZTF"),
            _row("/e2/r.fits", 100.1, "r", 16.975, tel="Other", inst="X"),
        ]
        terms = {"/d1": (0.10, 0.005, "g", "r", 0.5),
                 "/e2": (0.05, 0.005, "g", "r", 0.5)}
        out = _apply(rows, terms)
        assert not out["cc_applied_psf"].any()

    def test_missing_term_not_corrected(self):
        rows = [
            _row("/d1/g.fits", 100.0, "g", 17.95),
            _row("/d2/r.fits", 100.2, "r", 16.975),
        ]
        terms = {"/d1": None, "/d2": (0.05, 0.005, "g", "r", 0.5)}
        out = _apply(rows, terms)
        assert not out.loc[0, "cc_applied_psf"]
        # The r row still has a term and a g partner: it is corrected
        # using the g row's uncorrected magnitude.
        assert out.loc[1, "cc_applied_psf"]

    def test_iband_uses_r_minus_i(self):
        # i band index is (r, i): i row sign is -1, partner is r.
        rows = [
            _row("/d1/i.fits", 100.0, "i", 17.0 - 0.20 * 0.5),
            _row("/d2/r.fits", 100.3, "r", 16.5),
        ]
        terms = {"/d1": (0.20, 0.01, "r", "i", 0.5),
                 "/d2": None}
        out = _apply(rows, terms)
        assert out.loc[0, "cc_applied_psf"]
        # X = r - i = 16.5 - i_corr; i_corr = 16.9 + 0.2*(X - 0.5)
        # Solve: X = 16.5 - 16.9 - 0.2*(X - 0.5) -> X = (-0.4+0.1)/0.8
        x_exp = (16.5 - 16.9 - 0.2 * (0 - 0.5)) / (1 + 0.2)
        assert abs(out.loc[0, "cc_color_psf"] - x_exp) < 1e-5

    def test_low_snr_partner_rejected(self):
        # A partner with err >> max_err is a non-detection and must not
        # anchor a color.
        rows = [
            _row("/d1/g.fits", 100.0, "g", 17.95),
            _row("/d2/r.fits", 100.1, "r", 24.0, err=8.3),
        ]
        terms = {"/d1": (0.10, 0.005, "g", "r", 0.5),
                 "/d2": (0.05, 0.005, "g", "r", 0.5)}
        out = _apply(rows, terms)
        assert not out["cc_applied_psf"].any()

    def test_errors_propagated(self):
        rows = [
            _row("/d1/g.fits", 100.0, "g", 17.95, err=0.03),
            _row("/d2/r.fits", 100.2, "r", 16.975, err=0.04),
        ]
        terms = {"/d1": (0.10, 0.01, "g", "r", 0.5),
                 "/d2": (0.05, 0.01, "g", "r", 0.5)}
        out = _apply(rows, terms)
        assert out.loc[0, "mag_psf_cc_err"] > 0.03
        assert np.isfinite(out.loc[0, "cc_color_err_psf"])
