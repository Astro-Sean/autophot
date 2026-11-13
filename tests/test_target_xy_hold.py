"""Regression tests for S/N-adaptive target centroid freedom.

A detected target may re-center inside ``fitting_xy_bounds`` (absorbs
bad WCS / proper motion), while a target that has faded below
``fitting_xy_bounds_detect_snr`` is held near the catalog position
(``fitting_xy_bounds_hold_px``) so the fit becomes forced photometry
instead of chasing a noise peak.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

photutils = pytest.importorskip("photutils", reason="photutils")
from photutils.psf import CircularGaussianPRF

import autophot.photometry.psf as psf_mod

FWHM = 3.0
SIG = FWHM / 2.3548
SHAPE = (61, 61)


def _yaml(**overrides):
    phot = {
        "aperture_radius": 4.5,
        "fitting_xy_bounds": 3.0,
        "fitting_xy_bounds_detect_snr": 4.0,
        "fitting_xy_bounds_hold_px": 0.5,
    }
    phot.update(overrides)
    return {
        "fwhm": FWHM,
        "pixel_scale": 1.0,
        "gain": 1.0,
        "read_noise": 0.0,
        "exposure_time": 30.0,
        "imageFilter": "r",
        "photometry": phot,
    }


def _fit(img, rows, header=None, **fit_kwargs):
    p = psf_mod.PSF(_yaml(), img, header=header or {})
    model = CircularGaussianPRF(fwhm=FWHM)
    src = pd.DataFrame(rows)
    rms = np.ones_like(img)
    return p.fit(
        model, src, is_target_fit=True, background_rms=rms, **fit_kwargs
    )


def _noise(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, SHAPE)


def _add_source(img, x0, y0, peak):
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    img = img.copy()
    img += peak * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * SIG**2))
    return img


class TestTargetXYHold:
    def test_detected_offset_source_recenters(self, caplog):
        # A high-S/N source 1.3 px off the catalog position must be
        # allowed to move -- otherwise the fit is held and cannot
        # reach it.
        img = _add_source(_noise(1), 30.6, 30.4, 4000.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        with caplog.at_level(logging.INFO, logger="psf"):
            out = _fit(img, rows)
        assert out["x_fit"][0] == pytest.approx(30.6, abs=0.15)
        assert out["y_fit"][0] == pytest.approx(30.4, abs=0.15)
        assert out["xy_bounds_psf"][0] == pytest.approx(4.5)
        assert "free within" in caplog.text

    def test_moderate_snr_gets_intermediate_bound(self):
        # A moderate-S/N source inside the ~0.5 FWHM same-source window
        # counts as a detection and gets a bound ramped between the held
        # and free values by detection S/N.
        img = _add_source(_noise(13), 30.7, 30.3, 8.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [100.0]}
        out = _fit(img, rows)
        snr = float(out["detect_snr_psf"][0])
        bound = float(out["xy_bounds_psf"][0])
        expected = 0.5 + np.clip((snr - 4.0) / (20.0 - 4.0), 0.0, 1.0) * 4.0
        assert 4.0 <= snr < 20.0
        assert bound == pytest.approx(expected)
        assert 0.5 < bound < 4.5
        assert abs(out["x_fit"][0] - 29.5) <= bound + 1e-6
        assert abs(out["y_fit"][0] - 29.5) <= bound + 1e-6

    def test_bright_neighbour_does_not_free_target(self, caplog):
        # A bright peak several FWHM from the catalog position is a
        # neighbour.  Its wing inside the seed aperture must not inflate
        # the detection S/N and free the centroid onto it - the failure
        # mode reported for a low-S/N target near unrelated flux.
        img = _add_source(_noise(14), 35.0, 30.5, 4000.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        with caplog.at_level(logging.INFO, logger="psf"):
            out = _fit(img, rows)
        assert float(out["detect_snr_psf"][0]) < 4.0
        assert abs(out["x_fit"][0] - 29.5) <= 0.5 + 1e-6
        assert abs(out["y_fit"][0] - 29.5) <= 0.5 + 1e-6
        assert "held to" in caplog.text

    def test_faint_target_next_to_neighbour_stays_held(self):
        # Faint flux at the seed plus a bright neighbour outside the
        # same-source window: the neighbour's wing must be discounted so
        # the target's own (sub-threshold) flux leaves the fit held.
        img = _add_source(_noise(19), 29.5, 29.5, 3.0)
        img = _add_source(img, 35.0, 30.5, 4000.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [100.0]}
        out = _fit(img, rows)
        assert abs(out["x_fit"][0] - 29.5) <= 0.5 + 1e-6
        assert abs(out["y_fit"][0] - 29.5) <= 0.5 + 1e-6

    def test_faded_target_position_held(self, caplog):
        img = _noise(2)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        with caplog.at_level(logging.INFO, logger="psf"):
            out = _fit(img, rows)
        # Held to <= hold_px from the catalog position.
        assert abs(out["x_fit"][0] - 29.5) <= 0.5 + 1e-6
        assert abs(out["y_fit"][0] - 29.5) <= 0.5 + 1e-6
        assert "held to" in caplog.text

    def test_mixed_targets_split_hold_and_free(self):
        # Primary target is real (offset); companion faded.  The real
        # one must still re-center while the companion is pinned.
        img = _add_source(_noise(3), 30.6, 30.4, 4000.0)
        rows = {
            "x_pix": [29.5, 45.0],
            "y_pix": [29.5, 45.0],
            "counts_AP": [40000.0, 100.0],
        }
        out = _fit(img, rows)
        primary = out.iloc[0]
        companion = out.iloc[1]
        assert primary["x_fit"] == pytest.approx(30.6, abs=0.15)
        assert abs(companion["x_fit"] - 45.0) <= 0.5 + 1e-6
        assert abs(companion["y_fit"] - 45.0) <= 0.5 + 1e-6

    def test_nan_window_holds_position(self):
        img = _add_source(_noise(4), 30.6, 30.4, 4000.0)
        img[24:37, 24:37] = np.nan  # window around the catalog position
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        out = _fit(img, rows)
        # Nothing verifiable -> hold, not a free wander onto NaN-edge data.
        assert abs(out["x_fit"][0] - 29.5) <= 0.5 + 1e-6 or not np.isfinite(
            out["x_fit"][0]
        )

    def test_detection_check_disabled_stays_free(self):
        # detect_snr <= 0 restores the legacy behaviour.
        img = _add_source(_noise(5), 30.6, 30.4, 4000.0)
        p = psf_mod.PSF(
            _yaml(fitting_xy_bounds_detect_snr=0.0), img, header={}
        )
        src = pd.DataFrame(
            {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        )
        out = p.fit(
            CircularGaussianPRF(fwhm=FWHM),
            src,
            is_target_fit=True,
            background_rms=np.ones_like(img),
        )
        assert out["x_fit"][0] == pytest.approx(30.6, abs=0.15)

    def test_difference_image_negative_source_detected(self):
        # A fading source is a *negative* peak on a difference image;
        # with the signed-residual opt-out it must still count as
        # detected and be allowed to re-center.
        img = _add_source(_noise(6), 30.6, 30.4, -4000.0)
        p = psf_mod.PSF(
            _yaml(psf_diff_negative_flux=True),
            img,
            header={"FORCECON": "ZOGY"},
        )
        out = p.fit(
            CircularGaussianPRF(fwhm=FWHM),
            pd.DataFrame(
                {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [1.0]}
            ),
            is_target_fit=True,
            background_rms=np.ones_like(img),
        )
        assert out["x_fit"][0] == pytest.approx(30.6, abs=0.15)
        assert out["y_fit"][0] == pytest.approx(30.4, abs=0.15)

    def test_difference_image_negative_flux_bounded_by_default(self):
        # Default policy: the amplitude bound applies on difference
        # images too - a fading source must not report a negative
        # primary flux.
        img = _add_source(_noise(6), 30.6, 30.4, -4000.0)
        out = _fit(
            img,
            {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [1.0]},
            header={"FORCECON": "ZOGY"},
        )
        fx = float(out["flux_PSF"][0])
        assert not np.isfinite(fx) or fx >= 0.0

    def test_force_photometry_holds_everything(self, caplog):
        # Even a high-S/N source stays pinned under the explicit
        # forced-photometry override.
        img = _add_source(_noise(7), 30.6, 30.4, 4000.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        with caplog.at_level(logging.INFO, logger="psf"):
            out = _fit(img, rows, forcePhotometry=True)
        assert abs(out["x_fit"][0] - 29.5) <= 0.5 + 1e-6
        assert abs(out["y_fit"][0] - 29.5) <= 0.5 + 1e-6
        assert "forcePhotometry" in caplog.text

    def test_astrometric_scatter_widens_hold(self, caplog):
        # The measured dx/dy registration scatter plus the matched-source
        # position error must widen the held bound so forced photometry
        # can still reach a genuinely displaced source.
        yaml = _yaml()
        yaml["align_offset_rms_x_px"] = 1.0
        yaml["align_offset_rms_y_px"] = 1.0
        yaml["align_offset_pos_err_px"] = 0.3
        yaml["align_offset_n"] = 50
        img = _add_source(_noise(15), 30.6, 30.4, 4000.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        p = psf_mod.PSF(yaml, img, header={})
        with caplog.at_level(logging.INFO, logger="psf"):
            out = p.fit(
                CircularGaussianPRF(fwhm=FWHM),
                pd.DataFrame(rows),
                is_target_fit=True,
                forcePhotometry=True,
                background_rms=np.ones_like(img),
            )
        expected = np.sqrt(0.5**2 + 1.0**2 + 1.0**2 + 0.3**2)
        assert out["xy_bounds_psf"][0] == pytest.approx(expected, rel=1e-3)
        assert out["x_fit"][0] == pytest.approx(30.6, abs=0.15)
        assert out["y_fit"][0] == pytest.approx(30.4, abs=0.15)
        assert "Astrometric scatter" in caplog.text

    def test_astrometric_bound_capped_at_free_bound(self):
        # A pathologically large measured scatter must not give a held
        # source more freedom than a detected one.
        yaml = _yaml()
        yaml["wcs_offset_rms_x_px"] = 20.0
        yaml["wcs_offset_rms_y_px"] = 20.0
        yaml["wcs_offset_n"] = 50
        img = _add_source(_noise(16), 30.6, 30.4, 4000.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        p = psf_mod.PSF(yaml, img, header={})
        out = p.fit(
            CircularGaussianPRF(fwhm=FWHM),
            pd.DataFrame(rows),
            is_target_fit=True,
            forcePhotometry=True,
            background_rms=np.ones_like(img),
        )
        # Free bound for this config: 1.5*FWHM floor = 4.5 px.
        assert out["xy_bounds_psf"][0] <= 4.5 + 1e-6

    def test_astrometric_widening_disabled(self):
        # fitting_xy_bounds_astrometric=False restores the fixed hold.
        yaml = _yaml(fitting_xy_bounds_astrometric=False)
        yaml["align_offset_rms_x_px"] = 1.0
        yaml["align_offset_rms_y_px"] = 1.0
        yaml["align_offset_n"] = 50
        img = _add_source(_noise(17), 30.6, 30.4, 4000.0)
        rows = {"x_pix": [29.5], "y_pix": [29.5], "counts_AP": [40000.0]}
        p = psf_mod.PSF(yaml, img, header={})
        out = p.fit(
            CircularGaussianPRF(fwhm=FWHM),
            pd.DataFrame(rows),
            is_target_fit=True,
            forcePhotometry=True,
            background_rms=np.ones_like(img),
        )
        assert out["xy_bounds_psf"][0] == pytest.approx(0.5)
        assert abs(out["x_fit"][0] - 29.5) <= 0.5 + 1e-6

    def test_target_arcsec_cap_tightens_bound(self, caplog):
        # A detected target is normally free within the fit-shape limit
        # (~5.4 px here); fitting_xy_bounds_target_arcsec must cap that
        # to the configured arcsec value (5 px at pixel_scale=1).
        yaml = _yaml(
            fitting_xy_bounds=10.0, fitting_xy_bounds_target_arcsec=5.0
        )
        img = _add_source(_noise(20), 30.5, 30.5, 4000.0)
        rows = {"x_pix": [30.5], "y_pix": [30.5], "counts_AP": [40000.0]}
        p = psf_mod.PSF(yaml, img, header={})
        with caplog.at_level(logging.INFO, logger="psf"):
            out = p.fit(
                CircularGaussianPRF(fwhm=FWHM),
                pd.DataFrame(rows),
                is_target_fit=True,
                background_rms=np.ones_like(img),
            )
        assert out["xy_bounds_psf"][0] == pytest.approx(5.0)
        assert "capping xy_bounds" in caplog.text

    def test_target_arcsec_cap_never_below_floor(self):
        # The FWHM-relative floor (1.5*FWHM = 4.5 px) wins over a cap
        # that would make re-centering impossible on coarse pixels.
        yaml = _yaml(
            fitting_xy_bounds=10.0, fitting_xy_bounds_target_arcsec=2.0
        )
        img = _add_source(_noise(21), 30.5, 30.5, 4000.0)
        rows = {"x_pix": [30.5], "y_pix": [30.5], "counts_AP": [40000.0]}
        p = psf_mod.PSF(yaml, img, header={})
        out = p.fit(
            CircularGaussianPRF(fwhm=FWHM),
            pd.DataFrame(rows),
            is_target_fit=True,
            background_rms=np.ones_like(img),
        )
        assert out["xy_bounds_psf"][0] == pytest.approx(4.5)

    def test_no_target_arcsec_cap_keeps_free_bound(self):
        # Without the cap the free bound reaches the fit-shape limit.
        yaml = _yaml(fitting_xy_bounds=10.0)
        img = _add_source(_noise(22), 30.5, 30.5, 4000.0)
        rows = {"x_pix": [30.5], "y_pix": [30.5], "counts_AP": [40000.0]}
        p = psf_mod.PSF(yaml, img, header={})
        out = p.fit(
            CircularGaussianPRF(fwhm=FWHM),
            pd.DataFrame(rows),
            is_target_fit=True,
            background_rms=np.ones_like(img),
        )
        assert out["xy_bounds_psf"][0] > 5.0


class TestCatalogXYHold:
    def _fit_catalog(self, img, rows):
        p = psf_mod.PSF(_yaml(), img, header={})
        src = pd.DataFrame(rows)
        return p.fit(
            CircularGaussianPRF(fwhm=FWHM),
            src,
            is_target_fit=False,
            background_rms=np.ones_like(img),
        )

    def test_catalog_undetected_source_held(self):
        # Row 0 is real (high aperture S/N -> shortcut, free).
        # Row 1 is a spurious catalog entry on noise (S/N < 4 -> window
        # check fails -> held at its seed position).
        img = _add_source(_noise(8), 30.0, 30.0, 4000.0)
        rows = {
            "x_pix": [30.0, 45.0],
            "y_pix": [30.0, 45.0],
            "counts_AP": [40000.0, 100.0],
            "SNR": [50.0, 1.0],
        }
        out = self._fit_catalog(img, rows)
        assert out["x_fit"][0] == pytest.approx(30.0, abs=0.15)
        assert out["xy_held_psf"][0] == 0.0
        assert out["xy_held_psf"][1] == 1.0
        assert abs(out["x_fit"][1] - 45.0) <= 0.5 + 1e-6

    def test_catalog_displaced_source_rescued_by_window(self):
        # Aperture S/N at the catalog position is low (source sits 2 px
        # away), but the window check must find it and leave it free.
        img = _add_source(_noise(9), 47.0, 45.0, 4000.0)
        rows = {
            "x_pix": [45.0],
            "y_pix": [45.0],
            "counts_AP": [100.0],
            "SNR": [1.0],
        }
        out = self._fit_catalog(img, rows)
        assert out["xy_held_psf"][0] == 0.0
        assert out["x_fit"][0] == pytest.approx(47.0, abs=0.2)


class TestMaskedLocalBackground:
    def _annulus_mask(self, r_in=3.0, r_out=20.0):
        yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
        rr = np.hypot(xx - 30.5, yy - 30.5)
        return (rr >= r_in) & (rr <= r_out)

    def test_fully_masked_annulus_falls_back(self, caplog):
        # Finite-but-masked pixels covering every candidate annulus:
        # counting them as usable leaves photutils an empty annulus and
        # a NaN local_bkg; the mask-aware check must fall back so the
        # fit still returns a finite flux.
        img = _add_source(_noise(10), 30.5, 30.5, 4000.0)
        mask = self._annulus_mask()
        img[mask] = 50000.0  # finite defect values, must not enter the bkg
        rows = {"x_pix": [30.5], "y_pix": [30.5], "counts_AP": [40000.0]}
        model = CircularGaussianPRF(fwhm=FWHM)
        # A free fwhm on a core-only cutout is degenerate; real ePSF
        # fits fix the morphology, so freeze it here.
        model.fwhm.fixed = True
        with caplog.at_level(logging.WARNING, logger="psf"):
            p = psf_mod.PSF(_yaml(), img, header={})
            out = p.fit(
                model,
                pd.DataFrame(rows),
                is_target_fit=True,
                background_rms=np.ones_like(img),
                mask=mask,
            )
        assert "masked/NaN" in caplog.text or "mask-heavy" in caplog.text
        assert np.isfinite(out["flux_PSF"][0])
        # flux_PSF is a rate (flux / exposure_time=30): ~40k e- / 30 s.
        assert out["flux_PSF"][0] == pytest.approx(40775.0 / 30.0, rel=0.2)
        # NON_FINITE_LOCALBKG (2048) must not fire -- the fallback supplied
        # a finite bkg.  N_PIXELS_FIT_PARTIAL (1) is expected.
        assert int(out["flags"][0]) & 2048 == 0

    def test_partially_masked_annulus_warns(self, caplog):
        # >50% masked annulus must trigger the shrink path even though
        # every pixel is finite -- the old isfinite-only check passed it.
        img = _add_source(_noise(11), 30.5, 30.5, 4000.0)
        yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
        rr = np.hypot(xx - 30.5, yy - 30.5)
        mask = (rr >= 5.0) & (rr <= 13.5) & (xx >= 30.5)  # half the annulus ring
        img[mask] = 25000.0
        rows = {"x_pix": [30.5], "y_pix": [30.5], "counts_AP": [40000.0]}
        with caplog.at_level(logging.WARNING, logger="psf"):
            p = psf_mod.PSF(_yaml(), img, header={})
            p.fit(
                CircularGaussianPRF(fwhm=FWHM),
                pd.DataFrame(rows),
                is_target_fit=True,
                background_rms=np.ones_like(img),
                mask=mask,
            )
        assert "masked/NaN" in caplog.text or "mask-heavy" in caplog.text

    def test_all_nan_region_prefiltered(self, caplog):
        # A source seeded on an all-NaN patch (no explicit mask) is
        # unfittable; it must be dropped by the pre-filter, not crash or
        # reach photutils' fully-masked-cutout error.
        img = _add_source(_noise(12), 30.5, 30.5, 4000.0)
        img[22:39, 22:39] = np.nan  # covers the whole fit_shape cutout
        rows = {"x_pix": [30.5], "y_pix": [30.5], "counts_AP": [40000.0]}
        with caplog.at_level(logging.WARNING, logger="psf"):
            out = _fit(img, rows)
        assert "masked regions" in caplog.text


class TestApplyPositionBounds:
    def _compound_model(self):
        # Mimic the suffixed model photutils hands the fitter for a
        # grouped fit.
        m1 = CircularGaussianPRF(fwhm=FWHM)
        m2 = CircularGaussianPRF(fwhm=FWHM)
        m1.x_0 = 25.0
        m1.y_0 = 25.0
        m2.x_0 = 29.0
        m2.y_0 = 27.0
        m1.flux = 100.0
        m2.flux = 80.0
        return m1 + m2

    def test_intersect_with_photutils_bounds(self):
        m = self._compound_model()
        # photutils-style per-source bound tighter than delta on src 0.
        m.x_0_0.bounds = (24.5, 25.5)
        psf_mod._apply_position_bounds(m, m.parameters, delta=4.5)
        # src 0: tighter photutils bound wins
        assert m.x_0_0.bounds == (24.5, 25.5)
        # src 1: no prior bound -> delta window around p0
        assert m.x_0_1.bounds == (29.0 - 4.5, 29.0 + 4.5)

    def test_flux_not_matched(self):
        m = self._compound_model()
        m.flux_0.bounds = (None, None)
        psf_mod._apply_position_bounds(m, m.parameters, delta=4.5)
        # "x_0" inside "flux_0" must not clamp the flux amplitude.
        assert m.flux_0.bounds == (None, None)

    def test_single_source_param_names(self):
        m = CircularGaussianPRF(fwhm=FWHM)
        m.x_0 = 25.0
        m.y_0 = 25.0
        m.x_0.bounds = (24.9, 25.1)
        psf_mod._apply_position_bounds(m, m.parameters, delta=4.5)
        assert m.x_0.bounds == (24.9, 25.1)
        assert m.y_0.bounds == (20.5, 29.5)
