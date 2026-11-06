"""Tests for the spatially-varying (gridded) ePSF path.

Covers the safety gating of ``PSF._build_spatial_grid_epsf`` -- the grid is
only used when every (or enough, per the fill policy) detector cell has
sufficient vetted sources -- plus the helpers ``epsf_at_position`` and
``convolve_gridded_epsf_with_kernel``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel, ImagePSF

import psf as psf_mod
from psf import (
    PSF,
    convolve_gridded_epsf_with_kernel,
    epsf_at_position,
)


def _gauss_stamp(size=21, fwhm=3.0, peak=1.0):
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[:size, :size]
    s = fwhm / 2.3548
    return peak * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s * s))


def _make_gridded(n=2, size=15, nx=200.0, ny=200.0, peaks=None):
    """GriddedPSFModel on an n x n grid; per-cell stamps carry a unique
    peak so cell provenance is identifiable in tests."""
    xypos = []
    stamps = []
    edges = np.linspace(0.0, nx, n + 1)
    ey = np.linspace(0.0, ny, n + 1)
    fid_x = 0.5 * (edges[:-1] + edges[1:])
    fid_y = 0.5 * (ey[:-1] + ey[1:])
    k = 0
    for iy in range(n):
        for ix in range(n):
            peak = 1.0 + k if peaks is None else peaks[k]
            stamps.append(_gauss_stamp(size=size, peak=peak))
            xypos.append((float(fid_x[ix]), float(fid_y[iy])))
            k += 1
    nd = NDData(
        np.stack(stamps),
        meta={"grid_xypos": xypos, "oversampling": 1, "fill_value": 0.0},
    )
    return GriddedPSFModel(nd, fill_value=0.0)


class TestEpsfAtPosition:
    def test_non_gridded_passthrough(self):
        m = ImagePSF(data=_gauss_stamp(), flux=1.0)
        assert epsf_at_position(m, 10.0, 10.0) is m

    def test_at_fiducial_returns_cell_stamp(self):
        g = _make_gridded(n=2)
        # fiducial (50, 50) is cell index 0 -> peak 1.0
        loc = epsf_at_position(g, 50.0, 50.0)
        assert isinstance(loc, ImagePSF)
        assert np.isclose(np.nanmax(loc.data), 1.0, atol=0.05)

    def test_at_fiducial_other_cell(self):
        g = _make_gridded(n=2)
        # fiducial (150, 150) is cell index 3 -> peak 4.0
        loc = epsf_at_position(g, 150.0, 150.0)
        assert np.isclose(np.nanmax(loc.data), 4.0, atol=0.2)

    def test_midpoint_interpolates(self):
        g = _make_gridded(n=2)
        loc = epsf_at_position(g, 100.0, 100.0)
        # midpoint of all four cells: average peak ~ 2.5
        peak = np.nanmax(loc.data)
        assert 1.5 < peak < 3.5


class TestConvolveGridded:
    def test_delta_kernel_identity(self):
        g = _make_gridded(n=2)
        k = np.zeros((5, 5))
        k[2, 2] = 1.0
        out, nat_in, nat_out = convolve_gridded_epsf_with_kernel(
            g, [k] * 4
        )
        assert isinstance(out, GriddedPSFModel)
        assert out.data.shape == g.data.shape
        # delta kernel: convolved ~= original per cell
        for i in range(4):
            assert np.allclose(
                np.asarray(out.data[i]) / np.nanmax(out.data[i]),
                np.asarray(g.data[i]) / np.nanmax(g.data[i]),
                atol=0.02,
            )

    def test_per_cell_kernels_applied(self):
        g = _make_gridded(n=2)
        delta = np.zeros((5, 5))
        delta[2, 2] = 1.0
        wide = np.ones((5, 5)) / 25.0  # strong smoothing
        kernels = [delta, wide, delta, wide]
        out, _, _ = convolve_gridded_epsf_with_kernel(g, kernels)
        # cells 1 and 3 were box-smoothed -> lower peak than delta cells
        p0 = np.nanmax(out.data[0])
        p1 = np.nanmax(out.data[1])
        assert p1 < 0.8 * p0

    def test_kernel_count_mismatch_raises(self):
        g = _make_gridded(n=2)
        with pytest.raises(ValueError):
            convolve_gridded_epsf_with_kernel(g, [np.eye(3)] * 3)


# ---------------------------------------------------------------------------
# _build_spatial_grid_epsf gating
# ---------------------------------------------------------------------------

_IMG = np.zeros((200, 200), dtype=float)


def _star_df(nx, ny, n_grid, min_per_cell, skip_cells=()):
    """Stars spread uniformly over the cells; ``skip_cells`` entries are
    (ix, iy) tuples left empty."""
    edges = np.linspace(0, nx, n_grid + 1)
    rows = []
    for iy in range(n_grid):
        for ix in range(n_grid):
            if (ix, iy) in skip_cells:
                continue
            x0, x1 = edges[ix], edges[ix + 1]
            y0, y1 = edges[iy], edges[iy + 1]
            rng = np.random.default_rng(ix * 10 + iy)
            xs = rng.uniform(x0 + 5, x1 - 5, min_per_cell)
            ys = rng.uniform(y0 + 5, y1 - 5, min_per_cell)
            for x, y in zip(xs, ys):
                rows.append({"x_pix": x, "y_pix": y, "SNR": 100.0})
    return pd.DataFrame(rows)


def _fake_cell_build(n_stars_report=None, fwhm_by_cell=None, ring_cells=None):
    """Replacement for PSF.build used by per-cell builds: returns a usable
    ImagePSF whose peak encodes the cell indices parsed from the per-cell
    ``base`` name (``<base>_cellx<ix>y<iy>``). ``fwhm_by_cell`` maps
    (ix, iy) to an alternate stamp FWHM (still inside the sanity band);
    ``ring_cells`` is a set of (ix, iy) -- or the string ``"all"`` --
    whose stamps get a strong negative ring (degenerate-stack
    signature)."""

    def _build(self, psfSources=None, **kwargs):
        base = str(
            self.input_yaml.get("_psf_base_override")
            or self.input_yaml.get("base", "")
        )
        m = re.search(r"cellx(\d+)y(\d+)", base)
        ix, iy = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        fwhm = (
            fwhm_by_cell.get((ix, iy), 3.0)
            if fwhm_by_cell
            else 3.0
        )
        stamp = _gauss_stamp(size=21, fwhm=fwhm, peak=1.0 + 10 * ix + iy)
        if ring_cells == "all" or (
            ring_cells is not None and (ix, iy) in ring_cells
        ):
            # collapse-sharp core plus a wide negative moat: the stamp's
            # gaussian-fit FWHM stays in band but >10% of pixels go
            # negative, like a starved undersampled stack.
            yy, xx = np.mgrid[:21, :21]
            r = np.hypot(yy - 10.0, xx - 10.0)
            stamp = stamp + np.where(
                (r > 4) & (r < 9), -0.05 * np.nanmax(stamp), 0.0
            )
            stamp += 0.02  # keep the sum positive
        model = ImagePSF(data=stamp, flux=1.0, oversampling=1)
        self.n_epsf_stars = (
            len(psfSources)
            if n_stars_report is None
            else n_stars_report
        )
        return model, psfSources

    return _build


def _grid_builder(tmp_path, phot_cfg):
    input_yaml = {
        "write_dir": str(tmp_path),
        "base": "testimg",
        "photometry": dict(phot_cfg),
    }
    return PSF(input_yaml=input_yaml, image=_IMG)


def _call_grid(obj, df, grid_mode, phot_cfg):
    return obj._build_spatial_grid_epsf(
        df,
        xcol="x_pix",
        ycol="y_pix",
        grid_mode=grid_mode,
        phot_cfg=phot_cfg,
        fwhm=3.0,
        undersampled=False,
        oversample_psf=False,
        oversample=1,
    )


@pytest.fixture
def cell_build_patched(monkeypatch):
    monkeypatch.setattr(psf_mod.PSF, "build", _fake_cell_build())
    return monkeypatch


class TestGridSufficiencyGating:
    def test_full_2x2_grid_built(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        res = _call_grid(obj, df, 2, cfg)
        assert res is not None
        model, df_out = res
        assert isinstance(model, GriddedPSFModel)
        assert model.data.shape[0] == 4
        assert obj.psf_model_kind == "gridded-epsf-2x2"
        # standard-name FITS stays a 2-D stamp; cube is separate.  Both
        # live under PSF_MODELS/ and the cube name carries the grid shape.
        stamp_file = tmp_path / "PSF_MODELS" / "PSF_model_image_testimg.fits"
        grid_file = (
            tmp_path / "PSF_MODELS" / "PSF_model_grid2x2_testimg.fits"
        )
        assert stamp_file.exists() and grid_file.exists()
        from astropy.io import fits

        assert fits.getdata(stamp_file).ndim == 2
        assert fits.getdata(grid_file).ndim == 3
        # per-cell diagnostics are suppressed -- the grid plot covers them
        assert not (tmp_path / "PSF_Image_testimg.png").exists()

    def test_too_few_stars_returns_none(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        # 3 stars total in one corner cell -> <50% cells populated -> None
        df = pd.DataFrame(
            {"x_pix": [10, 20, 30], "y_pix": [10, 20, 30], "SNR": 100.0}
        )
        assert _call_grid(obj, df, 2, cfg) is None

    def test_one_empty_cell_nearest_fill(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8, skip_cells={(1, 1)})
        res = _call_grid(obj, df, 2, cfg)
        assert res is not None
        model, _ = res
        # grid_xypos is iy-major: (1,1) is index 3; its stamp must be a
        # copy of a built cell (peak encodes donor provenance)
        assert isinstance(model, GriddedPSFModel)
        assert model.data.shape[0] == 4
        filled_peak = np.nanmax(model.data[3])
        donor_peaks = {np.nanmax(model.data[i]) for i in range(3)}
        assert any(np.isclose(filled_peak, p) for p in donor_peaks)

    def test_one_empty_cell_fail_policy(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "fail",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8, skip_cells={(1, 1)})
        assert _call_grid(obj, df, 2, cfg) is None

    def test_failed_cell_falls_to_nearest(self, tmp_path, monkeypatch):
        def _build(self, psfSources=None, **kwargs):
            base = str(
                self.input_yaml.get("_psf_base_override")
                or self.input_yaml.get("base", "")
            )
            if "cellx1y1" in base:
                raise RuntimeError("cell build exploded")
            return _fake_cell_build()(self, psfSources=psfSources, **kwargs)

        monkeypatch.setattr(psf_mod.PSF, "build", _build)
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        res = _call_grid(obj, df, 2, cfg)
        assert res is not None
        model, _ = res
        assert isinstance(model, GriddedPSFModel)

    def test_failed_cell_fail_policy(self, tmp_path, monkeypatch):
        def _build(self, psfSources=None, **kwargs):
            base = str(
                self.input_yaml.get("_psf_base_override")
                or self.input_yaml.get("base", "")
            )
            if "cellx1y1" in base:
                return None, psfSources  # unusable model
            return _fake_cell_build()(self, psfSources=psfSources, **kwargs)

        monkeypatch.setattr(psf_mod.PSF, "build", _build)
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "fail",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        assert _call_grid(obj, df, 2, cfg) is None

    def test_filled_target_cell_rejects_grid(self, tmp_path, monkeypatch):
        """When the target's own cell cannot be built, the grid offers no
        local PSF where it matters most - reject it so the caller falls
        back to the field-wide ePSF."""

        def _build(self, psfSources=None, **kwargs):
            base = str(
                self.input_yaml.get("_psf_base_override")
                or self.input_yaml.get("base", "")
            )
            if "cellx1y1" in base:
                raise RuntimeError("cell build exploded")
            return _fake_cell_build()(self, psfSources=psfSources, **kwargs)

        monkeypatch.setattr(psf_mod.PSF, "build", _build)
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            # target-aware off: the gate must still know the target cell
            "psf_grid_target_aware": False,
        }
        obj = _grid_builder(tmp_path, cfg)
        obj.input_yaml["target_x_pix"] = 150.0
        obj.input_yaml["target_y_pix"] = 150.0  # inside cell (1, 1)
        df = _star_df(200, 200, 2, 8)
        assert _call_grid(obj, df, 2, cfg) is None

    def test_filled_target_cell_kept_when_not_required(
        self, tmp_path, monkeypatch
    ):
        """psf_grid_require_target_cell=False keeps the grid: the target
        cell is filled from a neighbour and a warning is logged."""

        def _build(self, psfSources=None, **kwargs):
            base = str(
                self.input_yaml.get("_psf_base_override")
                or self.input_yaml.get("base", "")
            )
            if "cellx1y1" in base:
                raise RuntimeError("cell build exploded")
            return _fake_cell_build()(self, psfSources=psfSources, **kwargs)

        monkeypatch.setattr(psf_mod.PSF, "build", _build)
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            "psf_grid_target_aware": False,
            "psf_grid_require_target_cell": False,
        }
        obj = _grid_builder(tmp_path, cfg)
        obj.input_yaml["target_x_pix"] = 150.0
        obj.input_yaml["target_y_pix"] = 150.0
        df = _star_df(200, 200, 2, 8)
        res = _call_grid(obj, df, 2, cfg)
        assert res is not None
        model, _ = res
        assert isinstance(model, GriddedPSFModel)
        assert model.data.shape[0] == 4

    def test_survival_floor_rejects_sparse_cell(
        self, tmp_path, monkeypatch
    ):
        """A cell whose vetting left almost no stars is unpopulated even
        though the candidate-count gate passed."""
        monkeypatch.setattr(
            psf_mod.PSF, "build", _fake_cell_build(n_stars_report=1)
        )
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        # every cell survives vetting with only 1 star -> all unpopulated
        # -> <50% populated -> None
        assert _call_grid(obj, df, 2, cfg) is None

    def test_auto_prefers_3x3_when_possible(
        self, tmp_path, cell_build_patched
    ):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            "psf_grid_max_cells": 9,
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 3, 8)
        res = _call_grid(obj, df, "auto", cfg)
        assert res is not None
        assert obj.psf_model_kind == "gridded-epsf-3x3"
        assert res[0].data.shape[0] == 9

    def test_max_cells_merges_down_to_budget(
        self, tmp_path, cell_build_patched
    ):
        """A fully-populated 3x3 exceeds the default 6-cell cap and is
        merged down to a 2x3/3x2 grid."""
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            "psf_grid_max_cells": 6,
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 3, 8)
        res = _call_grid(obj, df, 3, cfg)
        assert res is not None
        model, _ = res
        assert model.data.shape[0] <= 6
        assert obj.psf_model_kind in ("gridded-epsf-2x3", "gridded-epsf-3x2")

    def test_max_cells_floored_at_2x2(self, tmp_path, cell_build_patched):
        """A cap below 4 cannot shrink a 2x2 grid further."""
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            "psf_grid_max_cells": 2,
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        res = _call_grid(obj, df, 2, cfg)
        assert res is not None
        assert res[0].data.shape[0] == 4

    def test_auto_falls_to_2x2(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        # enough for 2x2 (8/cell) but not 3x3 (9 cells x 8)
        df = _star_df(200, 200, 2, 8)
        res = _call_grid(obj, df, "auto", cfg)
        assert res is not None
        assert obj.psf_model_kind == "gridded-epsf-2x2"

    def test_auto_none_when_too_few(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = pd.DataFrame(
            {
                "x_pix": np.linspace(5, 195, 12),
                "y_pix": np.linspace(5, 195, 12),
                "SNR": 100.0,
            }
        )
        assert _call_grid(obj, df, "auto", cfg) is None

    def test_ringing_cell_demoted_by_negfrac(self, tmp_path, monkeypatch):
        """A cell whose stamp rings (>10% negative pixels) is demoted and
        filled from a healthy neighbour even though its FWHM is in band."""
        monkeypatch.setattr(
            psf_mod.PSF, "build", _fake_cell_build(ring_cells={(1, 1)})
        )
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        res = _call_grid(obj, df, 2, cfg)
        assert res is not None
        model, _ = res
        # (1,1) is index 3 in iy-major order; its stamp must be a donor
        # copy, so it cannot carry the negative ring.
        assert np.mean(np.asarray(model.data[3]) < 0) <= 0.10

    def test_outlier_cell_demoted_by_consistency(
        self, tmp_path, monkeypatch
    ):
        """A cell whose stamp is much broader than the ensemble (in-band
        FWHM, but inconsistent sharpness) is demoted and filled."""
        monkeypatch.setattr(
            psf_mod.PSF,
            "build",
            _fake_cell_build(fwhm_by_cell={(1, 1): 4.9}),
        )
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            # A 2x2 grid is below the shipped minimum (5 cells) for the
            # consistency gate - the ensemble median is too fragile there
            # - so opt back in to exercise the demotion path.
            "psf_grid_cell_consistency_min_cells": 4,
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        res = _call_grid(obj, df, 2, cfg)
        assert res is not None
        model, _ = res
        # donor stamps are all fwhm=3.0; the filled (1,1) stamp must
        # match a donor's width, not its own 4.4-px build.
        from psf import measure_epsf_fwhm_native

        assert np.isclose(
            measure_epsf_fwhm_native(
                np.asarray(model.data[3], float), 1
            ),
            3.0,
            atol=0.3,
        )

    def test_all_degenerate_cells_reject_grid(
        self, tmp_path, monkeypatch
    ):
        """When every cell's stamp is degenerate the grid is rejected
        outright in favour of the single ePSF."""
        monkeypatch.setattr(
            psf_mod.PSF, "build", _fake_cell_build(ring_cells="all")
        )
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        assert _call_grid(obj, df, 2, cfg) is None

    def test_grid_descends_oversampling_when_cells_fail(
        self, tmp_path, monkeypatch
    ):
        """When too few cells build at the picked factor the whole grid
        is retried at the next lower oversampling, preserving shape
        uniformity, rather than abandoning the grid."""

        def _build(self, psfSources=None, **kwargs):
            osamp = int(
                self.input_yaml["photometry"].get("psf_oversample", 1)
            )
            if osamp > 1:
                raise RuntimeError("cell build exploded at x%d" % osamp)
            return _fake_cell_build()(self, psfSources=psfSources, **kwargs)

        monkeypatch.setattr(psf_mod.PSF, "build", _build)
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        df = _star_df(200, 200, 2, 8)
        res = obj._build_spatial_grid_epsf(
            df,
            xcol="x_pix",
            ycol="y_pix",
            grid_mode=2,
            phot_cfg=cfg,
            fwhm=3.0,
            undersampled=False,
            oversample_psf=False,
            oversample=2,
        )
        assert res is not None
        model, _ = res
        assert isinstance(model, GriddedPSFModel)
        assert model.data.shape[0] == 4


def _dense_df(nx=200, ny=200, step=9):
    """Near-uniform lattice of stars - any reasonable cell is populated."""
    xs, ys = np.meshgrid(np.arange(5, nx, step), np.arange(5, ny, step))
    return pd.DataFrame(
        {"x_pix": xs.ravel(), "y_pix": ys.ravel(), "SNR": 100.0}
    )


def _grid_fiducials(model):
    return np.asarray(model.meta["grid_xypos"], dtype=float)


class TestTargetAwareEdges:
    def test_edges_recenter_target_on_boundary(
        self, tmp_path, cell_build_patched
    ):
        """Target sitting on the uniform-grid corner should get a cell
        whose fiducial is much closer than the uniform one."""
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            "psf_grid_target_aware": True,
        }
        obj = _grid_builder(tmp_path, cfg)
        obj.input_yaml["target_x_pix"] = 99.5
        obj.input_yaml["target_y_pix"] = 99.5  # the uniform 2x2 corner
        res = _call_grid(obj, _dense_df(), 2, cfg)
        assert res is not None
        fids = _grid_fiducials(res[0])
        nearest = np.hypot(
            fids[:, 0] - 99.5, fids[:, 1] - 99.5
        ).min()
        # uniform fiducials sit at (50,50)/(150,150) -> ~70 px away
        assert nearest < 50.0

    def test_disabled_keeps_uniform(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
            "psf_grid_target_aware": False,
        }
        obj = _grid_builder(tmp_path, cfg)
        obj.input_yaml["target_x_pix"] = 99.5
        obj.input_yaml["target_y_pix"] = 99.5
        res = _call_grid(obj, _dense_df(), 2, cfg)
        assert res is not None
        fids = _grid_fiducials(res[0])
        assert np.allclose(np.unique(fids[:, 0]), [50.0, 150.0])

    def test_no_target_keeps_uniform(self, tmp_path, cell_build_patched):
        cfg = {
            "psf_grid_min_stars_per_cell": 8,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        res = _call_grid(obj, _dense_df(), 2, cfg)
        assert res is not None
        fids = _grid_fiducials(res[0])
        assert np.allclose(np.unique(fids[:, 0]), [50.0, 150.0])


class TestChooseTargetAwareEdges:
    def test_target_cell_always_populated(self):
        # sources only in the lower-left quadrant; target upper-right
        xs, ys = np.meshgrid(np.linspace(5, 90, 20), np.linspace(5, 90, 20))
        got = psf_mod._choose_target_aware_edges(
            xs.ravel(), ys.ravel(), 2, 200.0, 200.0, 180.0, 180.0,
            min_per_cell=8, require_all=False,
        )
        assert got is not None
        xe, ye, cnt = got
        ix = int(np.clip(np.digitize([180.0], xe[1:-1]), 0, 1)[0])
        iy = int(np.clip(np.digitize([180.0], ye[1:-1]), 0, 1)[0])
        assert cnt[iy, ix] >= 8

    def test_require_all_rejects_sparse_field(self):
        # 10 sources total, all in one corner: no all-populated 2x2 layout
        xs = np.linspace(5, 40, 10)
        ys = np.linspace(5, 40, 10)
        assert (
            psf_mod._choose_target_aware_edges(
                xs, ys, 2, 200.0, 200.0, 180.0, 180.0,
                min_per_cell=8, require_all=True,
            )
            is None
        )

    def test_uniform_layout_is_a_candidate(self):
        xs, ys = np.meshgrid(np.linspace(5, 195, 25), np.linspace(5, 195, 25))
        got = psf_mod._choose_target_aware_edges(
            xs.ravel(), ys.ravel(), 2, 200.0, 200.0, 100.0, 100.0,
            min_per_cell=8, require_all=True,
        )
        assert got is not None
        xe, ye, cnt = got
        assert int(cnt.min()) >= 8


class TestRepairStarvedCells:
    def _counts(self, xs, ys, xe, ye):
        return np.histogram2d(ys, xs, bins=[ye, xe])[0].astype(int)

    def test_grow_annexes_neighbour_sources(self):
        # corner cell (0,0) has 3 stars; right neighbour has 20 -> the
        # shared x edge should move right to annex sources, not merge.
        rng = np.random.default_rng(0)
        xs = np.concatenate(
            [
                rng.uniform(5, 95, 3),
                rng.uniform(105, 195, 20),
                rng.uniform(5, 95, 12),
                rng.uniform(105, 195, 12),
            ]
        )
        ys = np.concatenate(
            [
                rng.uniform(5, 95, 3),
                rng.uniform(5, 95, 20),
                rng.uniform(105, 195, 12),
                rng.uniform(105, 195, 12),
            ]
        )
        xe = np.array([0.0, 100.0, 200.0])
        ye = np.array([0.0, 100.0, 200.0])
        xe2, ye2, cnt = psf_mod._repair_starved_cells(
            xs, ys, xe, ye, min_per_cell=10, min_w=20.0
        )
        # still 2x2 -- growth, not merge
        assert len(xe2) == 3 and len(ye2) == 3
        assert xe2[1] > 100.0
        assert cnt.min() >= 10

    def test_merge_drops_edge_when_neighbour_too_poor(self):
        # every cell holds exactly min; corner cell starved -> cannot
        # grow (annexing would starve the neighbour) -> merge a column.
        rows_x, rows_y = [], []
        edges = np.linspace(0.0, 300.0, 4)
        rng = np.random.default_rng(1)
        for iy in range(3):
            for ix in range(3):
                n = 3 if (ix, iy) == (0, 0) else 10
                rows_x.append(
                    rng.uniform(edges[ix] + 3, edges[ix + 1] - 3, n)
                )
                rows_y.append(
                    rng.uniform(edges[iy] + 3, edges[iy + 1] - 3, n)
                )
        xs = np.concatenate(rows_x)
        ys = np.concatenate(rows_y)
        xe2, ye2, cnt = psf_mod._repair_starved_cells(
            xs, ys, edges.copy(), edges.copy(), min_per_cell=10,
            min_w=20.0,
        )
        # one axis merged: 2x3 or 3x2, every remaining cell populated
        dims = sorted((len(xe2) - 1, len(ye2) - 1))
        assert dims == [2, 3]
        assert cnt.min() >= 10

    def test_axis_never_merges_below_two_cells(self):
        xs = np.linspace(5, 95, 5)
        ys = np.linspace(5, 95, 5)
        xe = np.array([0.0, 100.0, 200.0])
        ye = np.array([0.0, 100.0, 200.0])
        xe2, ye2, cnt = psf_mod._repair_starved_cells(
            xs, ys, xe, ye, min_per_cell=10, min_w=20.0
        )
        # cannot reach the floor: axes stay at 2 cells, starved cells
        # are left for the fill/fail policy
        assert len(xe2) == 3 and len(ye2) == 3

    def test_target_cell_repaired_first(self):
        # two starved cells; the target's must be the one grown first -
        # here both can be fixed, so just assert the final layout is
        # legal and the target cell is populated.
        rng = np.random.default_rng(2)
        xs = np.concatenate(
            [
                rng.uniform(5, 95, 3),      # cell (0,0) starved
                rng.uniform(105, 195, 30),  # cell (1,0) rich
                rng.uniform(5, 95, 30),     # cell (0,1) rich
                rng.uniform(105, 195, 30),  # cell (1,1) rich
            ]
        )
        ys = np.concatenate(
            [
                rng.uniform(5, 95, 3),
                rng.uniform(5, 95, 30),
                rng.uniform(105, 195, 30),
                rng.uniform(105, 195, 30),
            ]
        )
        xe = np.array([0.0, 100.0, 200.0])
        ye = np.array([0.0, 100.0, 200.0])
        xe2, ye2, cnt = psf_mod._repair_starved_cells(
            xs, ys, xe, ye, min_per_cell=10, min_w=20.0,
            tx=50.0, ty=50.0,
        )
        tix = int(np.clip(np.digitize([50.0], xe2[1:-1]), 0, len(xe2) - 2)[0])
        tiy = int(np.clip(np.digitize([50.0], ye2[1:-1]), 0, len(ye2) - 2)[0])
        assert cnt[tiy, tix] >= 10


class TestCellMergeIntegration:
    def test_sparse_3x3_merges_to_rectangular_grid(
        self, tmp_path, cell_build_patched
    ):
        """A starved cell whose neighbours are all exactly at the floor
        cannot grow -> the edge is removed and the grid becomes 2x3."""
        cfg = {
            "psf_grid_min_stars_per_cell": 10,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        edges = np.linspace(0.0, 300.0, 4)
        rng = np.random.default_rng(3)
        rows = []
        for iy in range(3):
            for ix in range(3):
                n = 3 if (ix, iy) == (0, 0) else 10
                for x, y in zip(
                    rng.uniform(edges[ix] + 3, edges[ix + 1] - 3, n),
                    rng.uniform(edges[iy] + 3, edges[iy + 1] - 3, n),
                ):
                    rows.append({"x_pix": x, "y_pix": y, "SNR": 100.0})
        obj.image = np.zeros((300, 300))
        res = _call_grid(obj, pd.DataFrame(rows), 3, cfg)
        assert res is not None
        model, _ = res
        assert obj.psf_model_kind in ("gridded-epsf-2x3", "gridded-epsf-3x2")
        assert model.data.shape[0] == 6

    def test_starved_cell_grows_instead_of_filling(
        self, tmp_path, cell_build_patched
    ):
        """A cell just under the floor next to a rich neighbour annexes
        sources and is BUILT (peak encodes cell index), not filled."""
        cfg = {
            "psf_grid_min_stars_per_cell": 10,
            "psf_spatial_fill_missing": "nearest",
        }
        obj = _grid_builder(tmp_path, cfg)
        obj.image = np.zeros((200, 200))
        rng = np.random.default_rng(4)
        rows = []
        # cell (0,0) starved; (1,0) and (1,1) rich enough that moving the
        # shared x edge right annexes a strip without starving either.
        for (ix, iy), n in {
            (0, 0): 3, (1, 0): 30, (0, 1): 12, (1, 1): 30
        }.items():
            for x, y in zip(
                rng.uniform(ix * 100 + 3, (ix + 1) * 100 - 3, n),
                rng.uniform(iy * 100 + 3, (iy + 1) * 100 - 3, n),
            ):
                rows.append({"x_pix": x, "y_pix": y, "SNR": 100.0})
        res = _call_grid(obj, pd.DataFrame(rows), 2, cfg)
        assert res is not None
        model, _ = res
        assert obj.psf_model_kind == "gridded-epsf-2x2"
        # iy-major order: cell (0,0) is index 0; peak 1.0 means it was
        # built from its (grown) star set, not copied from a neighbour.
        assert np.isclose(np.nanmax(model.data[0]), 1.0, atol=0.01)
