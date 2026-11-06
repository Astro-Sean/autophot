"""Regression tests for the spatial ZP surface.

zeropoint.spatial_zp_surface models the zeropoint as
ZP(x, y) = ZP_0 + s(x, y) with s a low-order polynomial in detector
position.  A smooth residual pattern (vignetting, flat-field residuals,
mis-registration) should correct the calibration at the target position
instead of only flagging a spatial gradient / inflated error.
"""

import numpy as np
import pandas as pd
import pytest

from zeropoint import Zeropoint


def _yaml(tmp_path, enabled=True, order=1, **overrides):
    cfg = {
        "enabled": enabled,
        "order": order,
        "min_inliers": 12,
        "sigma_clip": 4.0,
        "min_improvement_frac": 0.10,
        "min_f_stat": 4.0,
        "max_correction_mag": 0.3,
    }
    cfg.update(overrides)
    return {
        "fpath": str(tmp_path / "img.fits"),
        "imageFilter": "r",
        "zeropoint": {"fit_method": "odr", "spatial_zp_surface": cfg},
    }


def _tilted_catalog(n=40, zp_true=24.5, tilt_x=0.4, seed=7):
    """delta = ZP + s(x): a linear tilt of +/-tilt_x/2 across x."""
    rng = np.random.default_rng(seed)
    catmag = rng.uniform(16.0, 21.0, n)
    x_pix = rng.uniform(0.0, 1000.0, n)
    y_pix = rng.uniform(0.0, 1000.0, n)
    s = tilt_x * (x_pix - 500.0) / 1000.0
    inst = catmag - zp_true - s + rng.normal(0.0, 0.01, n)
    flux = 10.0 ** (-0.4 * inst)
    return pd.DataFrame(
        {
            "r": catmag,
            "r_err": np.full(n, 0.02),
            "flux_AP": flux,
            "flux_AP_err": 0.02 * flux,
            "flux_PSF": flux,
            "flux_PSF_err": 0.02 * flux,
            "x_pix": x_pix,
            "y_pix": y_pix,
        }
    )


def _flat_catalog(n=40, zp_true=24.5, seed=9):
    rng = np.random.default_rng(seed)
    catmag = rng.uniform(16.0, 21.0, n)
    inst = catmag - zp_true + rng.normal(0.0, 0.02, n)
    flux = 10.0 ** (-0.4 * inst)
    return pd.DataFrame(
        {
            "r": catmag,
            "r_err": np.full(n, 0.02),
            "flux_AP": flux,
            "flux_AP_err": 0.02 * flux,
            "flux_PSF": flux,
            "flux_PSF_err": 0.02 * flux,
            "x_pix": rng.uniform(0.0, 1000.0, n),
            "y_pix": rng.uniform(0.0, 1000.0, n),
        }
    )


class TestSurfaceFitHelpers:
    def test_recovers_linear_plane(self):
        rng = np.random.default_rng(3)
        n = 60
        x = rng.uniform(0.0, 1000.0, n)
        y = rng.uniform(0.0, 800.0, n)
        s_true = (
            0.05
            + 0.15 * (x - 500.0) / 500.0
            + 0.08 * (y - 400.0) / 400.0
        )
        r = s_true + rng.normal(0.0, 0.01, n)
        surf = Zeropoint._fit_spatial_zp_surface(
            r, x, y, err=np.full(n, 0.02), order=1
        )
        assert surf is not None
        val, err, inside = Zeropoint.eval_spatial_zp_surface(
            surf, 750.0, 600.0
        )
        expect = 0.05 + 0.15 * 250.0 / 500.0 + 0.08 * 200.0 / 400.0
        assert val == pytest.approx(expect, abs=0.02)
        assert np.isfinite(err) and err > 0
        assert inside

    def test_eval_flags_extrapolation(self):
        rng = np.random.default_rng(3)
        n = 40
        x = rng.uniform(100.0, 900.0, n)
        y = rng.uniform(100.0, 700.0, n)
        r = 0.1 * (x - 500.0) / 500.0 + rng.normal(0.0, 0.01, n)
        surf = Zeropoint._fit_spatial_zp_surface(
            r, x, y, err=np.full(n, 0.02), order=1
        )
        _, _, inside_out = Zeropoint.eval_spatial_zp_surface(
            surf, 5000.0, 400.0
        )
        assert not inside_out
        _, _, inside_in = Zeropoint.eval_spatial_zp_surface(
            surf, 500.0, 400.0
        )
        assert inside_in

    def test_eval_empty_surface_returns_zero(self):
        assert Zeropoint.eval_spatial_zp_surface(None, 1.0, 2.0) == (
            0.0,
            0.0,
            False,
        )
        assert Zeropoint.eval_spatial_zp_surface(
            {"applied": False}, 1.0, 2.0
        ) == (0.0, 0.0, False)

    def test_too_few_points_returns_none(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 100.0, 4)
        y = rng.uniform(0.0, 100.0, 4)
        r = rng.normal(0.0, 0.01, 4)
        assert (
            Zeropoint._fit_spatial_zp_surface(
                r, x, y, err=np.full(4, 0.02), order=1
            )
            is None
        )

    def test_degenerate_geometry_returns_none(self):
        # All calibrators on one column: dx > 0 but dy = 0.
        x = np.linspace(0.0, 1000.0, 20)
        y = np.full(20, 400.0)
        r = 0.1 * (x - 500.0) / 500.0
        assert (
            Zeropoint._fit_spatial_zp_surface(
                r, x, y, err=np.full(20, 0.02), order=1
            )
            is None
        )


class TestFitZeropointIntegration:
    def test_tilted_field_surface_applied_and_gradient_cleared(
        self, tmp_path
    ):
        # Without the surface this same catalog trips the
        # spatial-gradient reliability gate (0.4 mag tilt, >>3 sigma).
        zp = Zeropoint(_yaml(tmp_path, enabled=True))
        _, params = zp.fit_zeropoint(_tilted_catalog())
        ap = params["AP"]
        assert ap["zp_surface_applied"] is True
        assert not any(
            "gradient" in f for f in ap.get("reliability_flags", [])
        )
        # Surface at the field edges should be ~ +/-0.2 mag.
        v_lo, _, _ = Zeropoint.eval_spatial_zp_surface(
            ap["zp_surface"], 0.0, 500.0
        )
        v_hi, _, _ = Zeropoint.eval_spatial_zp_surface(
            ap["zp_surface"], 1000.0, 500.0
        )
        assert v_lo == pytest.approx(-0.2, abs=0.08)
        assert v_hi == pytest.approx(0.2, abs=0.08)
        # Corrected inlier scatter should be much smaller than the
        # raw tilt amplitude.
        assert ap["zeropoint_scatter"] < 0.1

    def test_same_tilt_flags_gradient_when_disabled(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path, enabled=False))
        _, params = zp.fit_zeropoint(_tilted_catalog())
        ap = params["AP"]
        assert ap["zp_surface_applied"] is False
        assert any(
            "gradient" in f for f in ap.get("reliability_flags", [])
        )

    def test_flat_field_not_applied(self, tmp_path):
        # iid residuals: the surface must not chase noise-level RMS
        # improvements through the gates.
        zp = Zeropoint(_yaml(tmp_path, enabled=True))
        _, params = zp.fit_zeropoint(_flat_catalog())
        assert params["AP"]["zp_surface_applied"] is False

    def test_disabled_when_key_absent(self, tmp_path):
        y = _yaml(tmp_path, enabled=True)
        del y["zeropoint"]["spatial_zp_surface"]
        zp = Zeropoint(y)
        _, params = zp.fit_zeropoint(_tilted_catalog())
        ap = params["AP"]
        assert ap["zp_surface_applied"] is False
        assert ap["zp_surface"] is None

    def test_sparse_field_rejected_by_min_inliers(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path, enabled=True, min_inliers=20))
        _, params = zp.fit_zeropoint(_tilted_catalog(n=15, seed=11))
        assert params["AP"]["zp_surface_applied"] is False

    def test_huge_correction_rejected(self, tmp_path):
        # A 1.0 mag tilt exceeds max_correction_mag: pathological, the
        # reliability gates must still see it, not have it smoothed away.
        zp = Zeropoint(_yaml(tmp_path, enabled=True))
        _, params = zp.fit_zeropoint(_tilted_catalog(tilt_x=1.0, seed=13))
        ap = params["AP"]
        assert ap["zp_surface_applied"] is False
        assert any(
            "gradient" in f for f in ap.get("reliability_flags", [])
        )
