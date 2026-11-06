"""Tests for the source-detection cleaning layer in Find_FWHM.measure_image.

The detection stage is the first place that decides what counts as a
point source: saturated stars must not bias the detection-kernel shape
estimate, galaxies/trails must be rejected by shape (the scalar Moffat
FWHM is azimuthally averaged and blind to elongation), and compact
defects must be caught by absolute-size floors and a minimum connected
area rather than by population-relative clips alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

photutils = pytest.importorskip("photutils")

from fwhm import Find_FWHM


def _stellar_field(n_stars=8, fwhm=4.0, noise=3.0, seed=0, shape=(220, 220)):
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, noise, shape)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    sig = fwhm / 2.355
    gx, gy = np.meshgrid(
        np.arange(40, shape[1], 60), np.arange(40, shape[0], 60)
    )
    xs, ys = gx.ravel()[:n_stars], gy.ravel()[:n_stars]
    for cx, cy in zip(xs, ys):
        img += 2500.0 * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig**2)
        )
    return img


def _fwhm_obj(**src_overrides):
    src = {"finder": "dao"}
    src.update(src_overrides)
    return Find_FWHM(
        {
            "saturate": 60000.0,
            "fwhm": 4.0,
            "source_detection": src,
            "fpath": "/tmp/fwhm_cleaning.fits",
            "write_dir": "/tmp",
        }
    )


class TestShapeDiagnostics:
    """_fit_gaussian_2d returns per-axis shape and fit quality, not just
    a scalar FWHM."""

    def test_round_star_shape(self):
        yy, xx = np.mgrid[0:21, 0:21]
        sig = 4.0 / 2.355
        cut = 5000.0 * np.exp(
            -((xx - 10.0) ** 2 + (yy - 10.0) ** 2) / (2 * sig**2)
        ) + 100.0
        fit = _fwhm_obj()._fit_gaussian_2d(cut)
        assert fit is not None
        assert fit["fwhm"] == pytest.approx(4.0, abs=0.4)
        assert fit["elongation"] == pytest.approx(1.0, abs=0.15)
        assert np.isfinite(fit["chi2_red"])

    def test_blend_cutout_rejected(self):
        # Two equal-brightness peaks 8 px apart inside one cutout: the
        # secondary-peak veto must veto the cutout entirely.
        yy, xx = np.mgrid[0:31, 0:31]
        sig = 4.0 / 2.355
        cut = (
            4000.0 * np.exp(-((xx - 11.0) ** 2 + (yy - 15.0) ** 2) / (2 * sig**2))
            + 4000.0 * np.exp(-((xx - 19.0) ** 2 + (yy - 15.0) ** 2) / (2 * sig**2))
            + 100.0
        )
        assert _fwhm_obj()._fit_gaussian_2d(cut) is None

    def test_nan_fill_cutout(self):
        # Partial cutout (frame edge) carries NaN fill; the fit must not
        # propagate NaN into the background level or centroid.
        yy, xx = np.mgrid[0:21, 0:21]
        sig = 4.0 / 2.355
        cut = 5000.0 * np.exp(
            -((xx - 10.0) ** 2 + (yy - 10.0) ** 2) / (2 * sig**2)
        ) + 100.0
        cut[:4, :] = np.nan
        fit = _fwhm_obj()._fit_gaussian_2d(cut)
        assert fit is not None
        assert np.isfinite(fit["fwhm"])
        assert fit["fwhm"] == pytest.approx(4.0, abs=0.6)


class TestKernelShapeSaturation:
    """_bright_psf_shape ranks by peak; saturated sources must be
    excluded before ranking or their bloomed moments bias the kernel."""

    def test_saturated_excluded(self):
        img = np.full((260, 260), 100.0)
        yy, xx = np.mgrid[0:260, 0:260]
        sig = 5.0 / 2.355
        xs, ys, peaks = [], [], []
        # Four clean round stars (FWHM ~5 px), >=60 px from every blob
        # so the 40-px moment box sees no contamination.
        for cx, cy in [(60, 60), (140, 60), (60, 140), (140, 140)]:
            img += 3000.0 * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig**2)
            )
            xs.append(cx)
            ys.append(cy)
            peaks.append(3000.0)
        # Four saturated bloomed "stars": brighter than the clean ones
        # and elongated (sigma_x ~3x sigma_y).  Without the saturation
        # exclusion they are 4 of the 8 brightest rows and drag the
        # median FWHM to ~9 px.
        sig_x, sig_y = 4.0, 1.7
        for cx, cy in [(220, 60), (220, 140), (60, 220), (140, 220)]:
            img += 90000.0 * np.exp(
                -(
                    (xx - cx) ** 2 / (2 * sig_x**2)
                    + (yy - cy) ** 2 / (2 * sig_y**2)
                )
            )
            xs.append(cx)
            ys.append(cy)
            peaks.append(90000.0)
        df = pd.DataFrame(
            {"xcentroid": xs, "ycentroid": ys, "peak": peaks}
        )
        f = _fwhm_obj()
        shape = f._bright_psf_shape(img, df, half=20, saturate=60000.0)
        assert shape is not None
        fwhm_maj, ratio, _theta = shape
        assert fwhm_maj == pytest.approx(5.0, rel=0.25)
        assert ratio > 0.8

    def test_saturated_excluded_via_measure(self):
        # End-to-end: the brightest source on the field is saturated;
        # kernel refinement must still converge on the true PSF.
        img = _stellar_field(n_stars=8, fwhm=4.0)
        yy, xx = np.mgrid[0:220, 0:220]
        img += 90000.0 * np.exp(
            -((xx - 30.0) ** 2 / (2 * 4.0**2) + (yy - 30.0) ** 2 / (2 * 1.5**2))
        )
        fw, df, _ = _fwhm_obj().measure_image(img, fwhm_initial=7.0)
        assert np.isfinite(fw)
        assert fw == pytest.approx(4.0, rel=0.25)


class TestDetectionCleaning:
    def test_shape_columns_present(self):
        img = _stellar_field()
        _, df, _ = _fwhm_obj().measure_image(img, fwhm_initial=4.0)
        for col in ("fwhm", "elongation", "ellipticity", "fit_chi2_red"):
            assert col in df.columns
        assert (df["elongation"] < 1.5).all()

    def test_elongated_galaxy_rejected(self):
        img = _stellar_field()
        yy, xx = np.mgrid[0:220, 0:220]
        # Galaxy at (245, 20)-ish is too near the edge; use a free corner
        # of the 60-px grid instead.
        img += 2000.0 * np.exp(
            -(
                (xx - 215.0) ** 2 / (2 * 5.0**2)
                + (yy - 15.0) ** 2 / (2 * 1.2**2)
            )
        )
        _, df, _ = _fwhm_obj().measure_image(img, fwhm_initial=4.0)
        assert len(df) > 0
        near = df[
            (np.hypot(df["x_pix"] - 215.0, df["y_pix"] - 15.0) < 8)
        ]
        assert len(near) == 0

    def test_hot_pixel_rejected(self):
        img = _stellar_field()
        img[150, 210] += 8000.0  # isolated single-pixel spike
        _, df, _ = _fwhm_obj().measure_image(img, fwhm_initial=4.0)
        near = df[
            (np.hypot(df["x_pix"] - 210.0, df["y_pix"] - 150.0) < 3)
        ]
        assert len(near) == 0

    def test_defect_dominance_warning(self, caplog):
        # A floor raised artificially high puts the whole surviving pool
        # within 20% of it, exercising the defect-dominance diagnostic.
        img = _stellar_field()
        f = _fwhm_obj(fwhm_abs_floor_px=3.4, defect_dominance_frac=0.2)
        with caplog.at_level("WARNING"):
            fw, df, _ = f.measure_image(img, fwhm_initial=4.0)
        assert np.isfinite(fw)
        assert any(
            "defect-dominated" in r.getMessage().lower() for r in caplog.records
        )

    def test_undersampled_roundness_is_a_real_cut(self):
        # +-1.0 is the statistic's natural range (i.e. no cut); the
        # undersampled band must stay a genuine screen against CR tracks
        # and bad columns.
        p = Find_FWHM._adaptive_detection_params(1.5, "dao")
        assert p["roundlo"] > -1.0 and p["roundhi"] < 1.0
        p = Find_FWHM._adaptive_detection_params(1.5, "iraf")
        assert p["roundlo"] > -1.0 and p["roundhi"] < 1.0

    def test_segment_isolation_can_be_disabled(self):
        img = _stellar_field()
        f = _fwhm_obj(segment_isolation=False)
        fw, df, _ = f.measure_image(img, fwhm_initial=4.0)
        assert np.isfinite(fw)
        assert len(df) >= 6


class TestCheckLinearityEllipticity:
    """The ellipticity quality cut in check_linearity must engage even
    when the input catalog carries SExtractor A_IMAGE/B_IMAGE instead of
    a literal 'ellipticity' column."""

    def _catalog(self):
        # Six linear sources; row 5 is highly elongated (a/b = 4).
        n = 6
        flux = 1e4 * 10.0 ** (0.4 * np.arange(n))
        df = pd.DataFrame(
            {
                "flux_AP": flux,
                "maxPixel": 0.3 * flux,
                "a": np.full(n, 2.0),
                "b": np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.5]),
                "x_pix": np.arange(n) * 30.0 + 20.0,
                "y_pix": np.arange(n) * 30.0 + 20.0,
            }
        )
        return df

    def test_ellipticity_derived_from_axes(self):
        f = Find_FWHM({"write_dir": "/tmp", "fpath": "/tmp/x.fits"})
        df = self._catalog()
        out, fit_params, _ = f.check_linearity(df, write_dir=False)
        # The a/b = 4 source (ellipticity 0.75 > 0.25) is removed by the
        # quality cuts before RANSAC ever sees it.
        assert fit_params["n_after_quality_cuts"] == 5
        assert len(out) <= 5


class TestFitGaussianMinPixels:
    """fit_gaussian fits a 7-parameter model (Gaussian2d + constant); the
    valid-pixel floor must keep the fit over-determined."""

    def test_too_few_valid_pixels_rejected(self):
        cut = np.full((5, 5), np.nan)
        cut[1:3, 1:3] = 100.0  # 4 valid pixels for a 7-param model
        res = _fwhm_obj().fit_gaussian(cut)
        assert res is not None
        assert res["success"] is False
        assert np.isnan(res["fwhmx"])

    def test_normal_cutout_still_fits(self):
        yy, xx = np.mgrid[0:15, 0:15]
        sig = 4.0 / 2.355
        cut = 3000.0 * np.exp(
            -((xx - 7.0) ** 2 + (yy - 7.0) ** 2) / (2 * sig**2)
        ) + 50.0
        res = _fwhm_obj().fit_gaussian(cut)
        assert res["success"] is True
        assert np.isfinite(res["fwhmx"])
