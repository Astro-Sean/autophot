"""Tests for template-image alignment quality measurement and verification.

Covers ``compute_alignment_rms`` (the shared post-alignment diagnostic used by
the spalipy/SWarp/reproject/AstroAlign gates), the ``_wcs_footprints_overlap``
helper, the SCAMP catalog pre-filter
``ImageDistortionCorrector._filter_sci_to_ref_counterparts``, and YAML wiring
of ``run_scamp``.  Focus: sparse fields, coherent misregistration detection,
and matched-source spatial coverage.

These tests do not require the external ``sex``/``scamp``/``swarp`` binaries:
source detection inside ``compute_alignment_rms`` is monkeypatched.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import templates
from templates import (
    compute_alignment_rms,
    _wcs_footprints_overlap,
    _pad_to_shape,
    _merge_detection_lists,
    _close_group_survivor_indices,
    _finite_coverage_floor,
)

try:
    from utils.run_IDC import ImageDistortionCorrector
except ImportError:  # astroalign / optional deps not installed
    ImageDistortionCorrector = None

try:
    import spalipy as _spalipy_mod
except ImportError:
    _spalipy_mod = None

_needs_spalipy = pytest.mark.skipif(
    _spalipy_mod is None, reason="spalipy not installed"
)

_needs_run_idc = pytest.mark.skipif(
    ImageDistortionCorrector is None, reason="utils.run_IDC not importable"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spread_xy(n: int, nx: int = 512, ny: int = 512, margin: float = 60.0):
    """n well-separated source positions spanning the image."""
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    xs = np.linspace(margin, nx - margin, ncol)
    ys = np.linspace(margin, ny - margin, nrow)
    return np.array([(x, y) for y in ys for x in xs], dtype=float)[:n]


def _fake_det(xy):
    """Return value matching _detect_sextractor_sources' (xy, flux, fwhm)."""
    xy = np.asarray(xy, float)
    n = len(xy)
    return xy, np.full(n, 1.0e4), np.full(n, 4.0)


def _patch_detection(monkeypatch, ref_xy):
    """Monkeypatch SExtractor detection to always return *ref_xy*."""
    monkeypatch.setattr(
        templates,
        "_detect_sextractor_sources",
        lambda *a, **k: _fake_det(ref_xy),
    )


def _write_ldac(path, ra_deg, dec_deg):
    """Write a minimal FITS-LDAC catalog (primary + LDAC_IMHEAD + objects)."""
    ra = np.asarray(ra_deg, float)
    dec = np.asarray(dec_deg, float)
    imhead = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="Field Header Card",
                    array=np.array(["END" + " " * 77], dtype="S80"),
                    format="80A",
                )
            ]
        )
    )
    imhead.name = "LDAC_IMHEAD"
    objs = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(name="XWIN_WORLD", array=ra, format="D"),
                fits.Column(name="YWIN_WORLD", array=dec, format="D"),
                fits.Column(
                    name="XWIN_IMAGE", array=np.arange(len(ra)) + 10.0, format="D"
                ),
                fits.Column(
                    name="YWIN_IMAGE", array=np.arange(len(dec)) + 10.0, format="D"
                ),
            ]
        )
    )
    objs.name = "LDAC_OBJECTS"
    fits.HDUList([fits.PrimaryHDU(), imhead, objs]).writeto(path, overwrite=True)


def _tan_header(ra_deg, dec_deg):
    """Minimal TAN WCS header centred on (ra_deg, dec_deg)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [128.0, 128.0]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.cdelt = [-0.0003, 0.0003]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w.to_header()


# ---------------------------------------------------------------------------
# compute_alignment_rms
# ---------------------------------------------------------------------------

class TestComputeAlignmentRMS:
    """Behaviour of the shared post-alignment RMS diagnostic."""

    def test_well_aligned_field(self, monkeypatch):
        """A well-aligned field returns small metrics and good coverage."""
        sci = _spread_xy(12)
        ref = sci + np.array([0.15, -0.10])
        img = np.zeros((512, 512))
        _patch_detection(monkeypatch, ref)

        res = compute_alignment_rms(
            img, img.copy(), 4.0,
            sci_xy_override=sci, return_per_quadrant=True,
        )
        assert res is not None and len(res) == 4
        med, rms, p90, quad = res
        assert med == pytest.approx(np.hypot(0.15, 0.10), abs=0.05)
        assert rms < 0.5
        assert quad["coverage_ok"] is True

    def test_coherent_misregistration_detected(self, monkeypatch):
        """A coherent offset larger than max_sep must still be measured.

        Regression: previously every pair exceeded max_sep, fewer than 10
        matches survived, and the function returned None -- so a gross
        misregistration was silently accepted as "unverifiable".
        """
        sci = _spread_xy(12)
        ref = sci + np.array([15.0, 8.0])  # 17 px offset, > max_sep (10 px)
        img = np.zeros((512, 512))
        _patch_detection(monkeypatch, ref)

        res = compute_alignment_rms(
            img, img.copy(), 4.0,
            sci_xy_override=sci, return_per_quadrant=True,
        )
        assert res is not None, "coherent misregistration was not measured"
        med, rms, p90, quad = res
        assert med == pytest.approx(np.hypot(15.0, 8.0), rel=0.05)

    def test_sparse_aligned_field_returns_metrics(self, monkeypatch):
        """Five well-aligned sources should still yield a measurement."""
        sci = _spread_xy(5)
        ref = sci + np.array([0.2, 0.1])
        img = np.zeros((512, 512))
        _patch_detection(monkeypatch, ref)

        res = compute_alignment_rms(
            img, img.copy(), 4.0,
            sci_xy_override=sci, return_per_quadrant=True,
        )
        assert res is not None
        med, rms, p90, quad = res
        assert med == pytest.approx(np.hypot(0.2, 0.1), abs=0.05)

    def test_scattered_false_matches_rejected(self, monkeypatch):
        """Uncorrelated catalogs (rotation about centre) must not fake a
        coherent offset -- the measurement should be reported unverifiable."""
        sci = _spread_xy(12)
        cx, cy = 256.0, 256.0
        th = np.deg2rad(45.0)
        rot = np.array(
            [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        )
        ref = (sci - [cx, cy]) @ rot.T + [cx, cy]
        img = np.zeros((512, 512))
        _patch_detection(monkeypatch, ref)

        res = compute_alignment_rms(
            img, img.copy(), 4.0,
            sci_xy_override=sci, return_per_quadrant=True,
        )
        assert res is None

    def test_clustered_sources_flag_poor_coverage(self, monkeypatch):
        """Matched sources confined to one quadrant must flag coverage."""
        sci = np.array(
            [
                [50.0, 50.0], [80.0, 60.0], [110.0, 55.0],
                [70.0, 90.0], [100.0, 100.0], [60.0, 80.0],
            ]
        )
        ref = sci + np.array([0.2, 0.15])
        img = np.zeros((512, 512))
        _patch_detection(monkeypatch, ref)

        res = compute_alignment_rms(
            img, img.copy(), 4.0,
            sci_xy_override=sci, return_per_quadrant=True,
        )
        assert res is not None
        quad = res[3]
        assert quad["coverage_ok"] is False
        assert quad["n_quadrants"] == 1

    def test_too_few_detections_returns_none(self, monkeypatch):
        """Fewer than 5 detections per image cannot be verified."""
        sci = _spread_xy(4)
        ref = sci + 0.2
        img = np.zeros((512, 512))
        _patch_detection(monkeypatch, ref)
        assert (
            compute_alignment_rms(img, img.copy(), 4.0, sci_xy_override=sci)
            is None
        )


# ---------------------------------------------------------------------------
# _wcs_footprints_overlap
# ---------------------------------------------------------------------------

class TestWcsFootprintsOverlap:
    def test_same_pointing_overlaps(self):
        h = _tan_header(150.0, 20.0)
        assert _wcs_footprints_overlap(h, (256, 256), h, (256, 256))

    def test_distant_pointing_no_overlap(self):
        h1 = _tan_header(150.0, 20.0)
        h2 = _tan_header(150.0, 30.0)
        assert not _wcs_footprints_overlap(h1, (256, 256), h2, (256, 256))

    def test_partial_overlap_detected(self):
        h1 = _tan_header(150.0, 20.0)
        # ~0.038 deg field; shift by half a field in Dec
        h2 = _tan_header(150.0, 20.0 + 0.02)
        assert _wcs_footprints_overlap(h1, (256, 256), h2, (256, 256))


# ---------------------------------------------------------------------------
# _filter_sci_to_ref_counterparts
# ---------------------------------------------------------------------------

@_needs_run_idc
class TestFilterSciToRefCounterparts:
    """The SCAMP pre-filter must never write an empty science catalog."""

    def _idc(self):
        return ImageDistortionCorrector({"fwhm": 4.0})

    def test_all_sources_no_counterpart_leaves_catalog(self, tmp_path):
        """Regression: when no science source has a reference counterpart
        (e.g. a large WCS offset), the catalog must be left intact instead of
        being silently emptied."""
        sci_cat = tmp_path / "sci.cat"
        ref_cat = tmp_path / "ref.cat"
        _write_ldac(sci_cat, ra_deg=[150.0, 150.001, 150.002, 149.999, 150.003],
                    dec_deg=[20.0, 20.001, 20.002, 19.999, 20.003])
        # Reference field 1 deg away -> no counterparts within 5"
        _write_ldac(ref_cat, ra_deg=[151.0, 151.001], dec_deg=[20.0, 20.001])

        n = self._idc()._filter_sci_to_ref_counterparts(
            sci_cat_path=str(sci_cat),
            ref_cat_path=str(ref_cat),
            match_radius_arcsec=5.0,
        )
        assert n == 5
        assert len(fits.getdata(sci_cat, ext=2)) == 5

    def test_partial_filtering_still_works(self, tmp_path):
        sci_cat = tmp_path / "sci.cat"
        ref_cat = tmp_path / "ref.cat"
        _write_ldac(sci_cat, ra_deg=[150.0, 150.001, 160.0],
                    dec_deg=[20.0, 20.001, 25.0])
        _write_ldac(ref_cat, ra_deg=[150.0, 150.001], dec_deg=[20.0, 20.001])

        n = self._idc()._filter_sci_to_ref_counterparts(
            sci_cat_path=str(sci_cat),
            ref_cat_path=str(ref_cat),
            match_radius_arcsec=5.0,
        )
        assert n == 2
        assert len(fits.getdata(sci_cat, ext=2)) == 2


# ---------------------------------------------------------------------------
# run_scamp YAML wiring
# ---------------------------------------------------------------------------

@_needs_run_idc
class TestRunScampConfig:
    """run_scamp must honour the documented scamp_* keys in wcs config."""

    def test_yaml_scamp_keys_reach_config_file(self, tmp_path, monkeypatch):
        idc = ImageDistortionCorrector(
            {
                "wcs": {
                    "scamp_ref_timeout": 120,
                    "scamp_ref_server": "vizier.unistra.fr",
                    "scamp_distort_degrees": 3,
                    "scamp_crossid_radius": 2.5,
                    "scamp_position_maxerr": 1.0,
                    "scamp_fwhm_thresholds": "1.0,15.0",
                }
            }
        )
        fake_cat = tmp_path / "fake.cat"
        fake_cat.write_bytes(b"")
        ref_cat = tmp_path / "ref.cat"
        ref_cat.write_bytes(b"")

        # The config file is written before the executable check raises;
        # simulate scamp being absent so the test does not depend on
        # whether the host has it installed.
        def _missing(name):
            raise FileNotFoundError(f"Executable not found: {name}")

        monkeypatch.setattr(idc, "_check_executable", _missing)
        with pytest.raises(FileNotFoundError):
            idc.run_scamp(
                catalog_paths=str(fake_cat),
                reference_cat=str(ref_cat),
                output_dir=str(tmp_path),
            )

        cfg_text = (tmp_path / "fake_default.scamp").read_text()
        assert "REF_TIMEOUT\t120" in cfg_text
        assert "REF_SERVER\tvizier.unistra.fr" in cfg_text
        assert "DISTORT_DEGREES\t3" in cfg_text
        assert "CROSSID_RADIUS\t2.5" in cfg_text
        assert "POSITION_MAXERR\t1.0" in cfg_text
        assert "FWHM_THRESHOLDS\t1.0,15.0" in cfg_text
        assert "ASTREF_CATALOG\tFILE" in cfg_text

    def test_explicit_config_overrides_yaml(self, tmp_path, monkeypatch):
        idc = ImageDistortionCorrector(
            {"wcs": {"scamp_ref_timeout": 120}}
        )
        fake_cat = tmp_path / "fake.cat"
        fake_cat.write_bytes(b"")

        def _missing(name):
            raise FileNotFoundError(f"Executable not found: {name}")

        monkeypatch.setattr(idc, "_check_executable", _missing)
        with pytest.raises(FileNotFoundError):
            idc.run_scamp(
                catalog_paths=str(fake_cat),
                reference_cat=str(tmp_path / "ref.cat"),
                output_dir=str(tmp_path),
                config={"REF_TIMEOUT": 45},
            )
        cfg_text = (tmp_path / "fake_default.scamp").read_text()
        assert "REF_TIMEOUT\t45" in cfg_text


# ---------------------------------------------------------------------------
# _pad_to_shape + spalipy cutout-template regression
# ---------------------------------------------------------------------------

class TestPadToShape:
    """Unit tests for the padding helper used by the spalipy path."""

    def test_pads_short_axes_only(self):
        img = np.ones((100, 300))
        out, m = _pad_to_shape(img, (200, 300))
        assert out.shape == (200, 300)
        assert (out[:100] == 1).all()
        assert np.isnan(out[100:]).all()
        assert m is None

    def test_oversized_axis_not_cropped(self):
        img = np.ones((500, 100))
        out, _ = _pad_to_shape(img, (200, 400))
        assert out.shape == (500, 400)
        assert np.isnan(out[:, 100:]).all()

    def test_mask_padded_true(self):
        img = np.zeros((10, 10))
        mask = np.zeros((10, 10), dtype=bool)
        _, mo = _pad_to_shape(img, (14, 12), mask=mask)
        assert mo.shape == (14, 12)
        assert mo[10:].all() and mo[:, 10:].all()
        assert not mo[:10, :10].any()


@_needs_spalipy
class TestSpalipyCutoutTemplate:
    """Regression for the SN2024pba i-band failure mode.

    spalipy sub-tiles and edge-clips the *template*-role (science)
    detections using the *source* image's shape.  A cutout template
    smaller than the science frame therefore drops every science anchor
    outside the template's own pixel bounds -- on the real field only
    ~10 of ~100 matched sources survived and quad matching starved.
    Padding the source data/mask to the science frame restores the full
    anchor set.
    """

    @staticmethod
    def _make_pair(seed=7, n=80):
        """Science frame + smaller template cutout sharing the same stars.

        Returns (sci, tpl, sci_det, tpl_det); the template footprint maps
        to sci x in [250, 550], y in [150, 450] (a 300x300 cutout).
        """
        from astropy.table import Table

        rng = np.random.default_rng(seed)
        sci_shape = (600, 800)
        tpl_shape = (300, 300)
        tpl_origin = np.array([250.0, 150.0])  # (x0, y0) in sci frame
        margin = 20.0

        sci_xy = np.column_stack(
            [
                rng.uniform(
                    tpl_origin[0] + margin,
                    tpl_origin[0] + tpl_shape[1] - margin,
                    n,
                ),
                rng.uniform(
                    tpl_origin[1] + margin,
                    tpl_origin[1] + tpl_shape[0] - margin,
                    n,
                ),
            ]
        )
        tpl_xy = sci_xy - tpl_origin
        flux = 1.0e4 * (1.0 + rng.random(n))

        sci = np.zeros(sci_shape)
        tpl = np.zeros(tpl_shape)
        yy_s, xx_s = np.mgrid[0 : sci_shape[0], 0 : sci_shape[1]]
        yy_t, xx_t = np.mgrid[0 : tpl_shape[0], 0 : tpl_shape[1]]
        for (xs, ys), (xt, yt), f in zip(sci_xy, tpl_xy, flux):
            sci += f * np.exp(
                -((xx_s - xs) ** 2 + (yy_s - ys) ** 2) / (2 * 1.5**2)
            )
            tpl += f * np.exp(
                -((xx_t - xt) ** 2 + (yy_t - yt) ** 2) / (2 * 1.5**2)
            )

        def _det(xy, fl):
            return Table(
                {
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "flux": fl,
                    "fwhm": np.full(len(xy), 4.0),
                    "flag": np.zeros(len(xy), dtype=int),
                }
            )

        return sci, tpl, _det(sci_xy, flux), _det(tpl_xy, flux)

    @staticmethod
    def _call(tpl_data, tpl_mask, sci, sci_det, tpl_det):
        """Mirror the Spalipy call in templates._spalipy."""
        from spalipy import Spalipy

        return Spalipy(
            tpl_data,
            source_mask=tpl_mask if (tpl_mask is not None and tpl_mask.any()) else None,
            template_data=sci,
            source_det=tpl_det,
            template_det=sci_det,
            output_shape=sci.shape,
            min_n_match=15,
            n_quad_det=25,
            max_quad_hash_dist=0.16,
            max_match_dist=2.0,
            min_quad_sep=50.0,
            quad_edge_buffer=4,
            max_quad_cand=10,
            min_sep=8.0,
            interp_order=3,
            sub_tile=1,
            spline_order=2,
            cval=np.nan,
        )

    def test_unpadded_cutout_starves_quad_matching(self):
        sci, tpl, sci_det, tpl_det = self._make_pair()
        sp = self._call(
            tpl, ~np.isfinite(tpl), sci, sci_det, tpl_det
        )
        # Sci anchors live at x,y >= 150 but the template quadlist clips
        # them to the source's 300x300 bounds -> too few quads to match.
        with pytest.raises(Exception):
            sp.align()

    def test_padded_cutout_aligns(self):
        sci, tpl, sci_det, tpl_det = self._make_pair()
        # Median fill, not NaN: the order>1 spline prefilter would spread
        # a NaN pad across the whole resampled image.  The pad is masked
        # so downstream re-masking turns it back into NaN.
        padded, pad_mask = _pad_to_shape(
            tpl, sci.shape, fill=float(np.median(tpl)), mask=~np.isfinite(tpl)
        )
        sp = self._call(padded, pad_mask, sci, sci_det, tpl_det)
        sp.align()
        assert sp.aligned_data is not None
        n_matched = len(sp._source_det_matched[0])
        assert n_matched >= 15
        # The 300x300 template covers ~19% of the 600x800 science frame;
        # the resample must actually deliver data there (an all-NaN
        # aligned_data would mean the pad poisoned the spline filter).
        assert np.isfinite(sp.aligned_data).mean() > 0.1


# ---------------------------------------------------------------------------
# spalipy source-pool helpers
# ---------------------------------------------------------------------------

class TestMergeDetectionLists:
    """Deep-pass merge used by _detect_for_spalipy."""

    def test_adds_new_sources(self):
        xy_a = np.array([[10.0, 10.0], [50.0, 50.0]])
        xy_b = np.array([[10.2, 10.1], [90.0, 90.0]])
        fa = np.array([1.0, 1.0])
        wa = np.array([3.0, 3.0])
        xy, _, _ = _merge_detection_lists(
            xy_a, fa, wa, xy_b, fa, wa
        )
        assert len(xy) == 3
        # First list's centroid wins on the overlapping source.
        np.testing.assert_allclose(xy[0], [10.0, 10.0])
        np.testing.assert_allclose(xy[2], [90.0, 90.0])

    def test_empty_inputs(self):
        xy = np.array([[1.0, 2.0]])
        f = np.array([1.0])
        out = _merge_detection_lists(None, None, None, xy, f, f)
        assert len(out[0]) == 1
        out = _merge_detection_lists(xy, f, f, None, None, None)
        assert len(out[0]) == 1


class TestCloseGroupSurvivorIndices:
    """Brightest-per-close-group crowding filter for spalipy dets."""

    def test_pair_keeps_brighter(self):
        xy = np.array([[10.0, 10.0], [12.0, 10.0], [100.0, 100.0]])
        flux = np.array([5.0, 9.0, 1.0])
        keep = _close_group_survivor_indices(xy, flux, min_sep=5.0)
        assert sorted(keep) == [1, 2]

    def test_chain_collapses_to_one_group(self):
        # A-B and B-C are each within min_sep; all three form one group.
        xy = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [50.0, 50.0]])
        flux = np.array([1.0, 7.0, 3.0, 1.0])
        keep = _close_group_survivor_indices(xy, flux, min_sep=4.0)
        assert sorted(keep) == [1, 3]

    def test_sparse_list_unchanged(self):
        xy = np.array([[0.0, 0.0], [30.0, 30.0], [60.0, 0.0]])
        keep = _close_group_survivor_indices(
            xy, np.ones(3), min_sep=5.0
        )
        assert sorted(keep) == [0, 1, 2]

    def test_zero_sep_no_op(self):
        xy = np.array([[0.0, 0.0], [0.1, 0.0]])
        keep = _close_group_survivor_indices(xy, np.ones(2), min_sep=0.0)
        assert len(keep) == 2


class TestFiniteCoverageFloor:
    """Coverage gate relative to the template's attainable footprint."""

    def test_full_frame_uses_config_floor(self):
        floor, att = _finite_coverage_floor(0.25, 400 * 400, 400 * 400)
        assert att == pytest.approx(1.0)
        assert floor == pytest.approx(0.25)

    def test_stamp_scales_floor_down(self):
        # 300x300 stamp on 600x800 science: attainable ~0.19 -> floor ~0.15,
        # which the old absolute 0.25 gate would have failed at ~0.19 retained.
        floor, att = _finite_coverage_floor(0.25, 300 * 300, 600 * 800)
        assert att == pytest.approx(300 * 300 / (600 * 800))
        assert floor == pytest.approx(0.8 * att)
        assert floor < att < 0.25

    def test_zero_size_guard(self):
        floor, _ = _finite_coverage_floor(0.25, 0, 0)
        assert floor == pytest.approx(0.0)


class TestSpalipyDenseCrowding:
    """Keep-brightest dedup preserves anchors spalipy would drop.

    spalipy's own min_sep filter removes BOTH members of every pair
    closer than min_sep; our pre-filter keeps the brightest of each
    close group.  A field built from close pairs therefore feeds
    usable anchors instead of collapsing to zero detections.
    """

    def test_dedup_keeps_one_anchor_per_pair(self):
        from astropy.table import Table

        rng = np.random.default_rng(3)
        # 60 sources: 30 well-separated + a faint companion 2 px away.
        base = np.column_stack(
            [rng.uniform(30, 570, 30), rng.uniform(30, 570, 30)]
        )
        comp = base + np.array([2.0, 0.5])
        xy = np.vstack([base, comp])
        flux = np.concatenate([np.full(30, 1e4), np.full(30, 1e2)])
        det = Table(
            {
                "x": xy[:, 0],
                "y": xy[:, 1],
                "flux": flux,
                "fwhm": np.full(len(xy), 4.0),
                "flag": np.zeros(len(xy), dtype=int),
            }
        )
        keep = _close_group_survivor_indices(
            np.column_stack([det["x"], det["y"]]), det["flux"], min_sep=8.0
        )
        assert len(keep) == 30
        # Bright member of each pair survives (indices 0..29).
        assert np.all(np.asarray(keep) < 30)
        # The surviving table is a fixed point of spalipy's own min_sep
        # filter - running the same rule again removes nothing more.
        keep2 = _close_group_survivor_indices(
            np.column_stack(
                [np.asarray(det["x"])[keep], np.asarray(det["y"])[keep]]
            ),
            np.asarray(det["flux"])[keep],
            min_sep=8.0,
        )
        assert len(keep2) == 30
