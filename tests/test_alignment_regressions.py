"""Regression tests for the alignment audit fixes.

Covers the distortion-aware WCS cache key and mutation protection
(``wcs.py``), the no-unaligned-fallback contract of ``Templates.align``
(structured ``AlignmentResult``), chi2_shift validity-mask propagation and
the common-grid guard, the ``alignment_max_p90_px`` rename, RA-wraparound
footprint overlap, NaN-safe ``find_non_uniform_center``, the shared
pass/warn/fail classifier, atomic output writes, and the per-run
``Alignment_<base>.json`` provenance report.

No external binaries (sex/scamp/swarp) are required: alignment backends are
disabled or stubbed via monkeypatching.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from autophot.subtraction import templates
import autophot.pipeline.wcs as wcs_mod
from autophot.subtraction.templates import (
    AlignmentResult,
    Templates,
    _WARN_DEMOTED_FROM_FAIL,
    _alignment_gate_accept,
    _alignment_gate_thresholds,
    _atomic_fits_write,
    _classify_alignment_quality,
    _wcs_footprints_overlap,
    _write_alignment_report,
    verify_common_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tan_header(ra_deg, dec_deg, nx=256):
    """Minimal TAN WCS header centred on (ra_deg, dec_deg)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2.0, nx / 2.0]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.cdelt = [-0.0003, 0.0003]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w.to_header()


def _sip_header(ra_deg, dec_deg, a10):
    """TAN-SIP header; ``a10`` sets the A_1_0 SIP coefficient."""
    hdr = _tan_header(ra_deg, dec_deg)
    hdr["CTYPE1"] = "RA---TAN-SIP"
    hdr["CTYPE2"] = "DEC--TAN-SIP"
    hdr["A_ORDER"] = 2
    hdr["B_ORDER"] = 2
    hdr["A_1_0"] = a10
    hdr["A_0_1"] = 0.0
    hdr["A_2_0"] = 0.0
    hdr["B_1_0"] = 0.0
    hdr["B_0_1"] = 0.0
    hdr["B_2_0"] = 0.0
    return hdr


def _write_fits(path, data, header):
    fits.writeto(
        str(path), np.asarray(data, np.float32), header, overwrite=True
    )


def _disable_all_backends(monkeypatch):
    """Force every alignment backend to fail fast."""
    monkeypatch.setattr(templates, "run_IDC", None)
    monkeypatch.setattr(templates, "_HAS_SPALIPY", False)
    monkeypatch.setattr(templates, "_HAS_TWEAKWCS", False)
    monkeypatch.setattr(templates, "_HAS_IMGREG", False)
    monkeypatch.setattr(templates, "_REPROJECT_AVAILABLE", False)
    monkeypatch.setattr(templates, "reproject_interp", None)
    monkeypatch.setattr(templates, "reproject_adaptive", None)
    monkeypatch.setattr(templates, "reproject_exact", None)


def _fake_imgreg(monkeypatch, xoff=2.0, yoff=1.0):
    """Stub chi2_shift + shiftnd so the chi2 path runs without the package."""
    from scipy.ndimage import shift as _scipy_shift

    monkeypatch.setattr(templates, "_HAS_IMGREG", True)
    monkeypatch.setattr(
        templates,
        "chi2_shift",
        lambda *a, **k: (xoff, yoff, 0.05, 0.05),
    )
    monkeypatch.setattr(
        templates,
        "_imgreg_shift",
        types.SimpleNamespace(
            shiftnd=lambda img, off: _scipy_shift(
                img, off, order=3, mode="constant", cval=0.0
            )
        ),
    )


@pytest.fixture
def fits_pair(tmp_path):
    """Write a science/template FITS pair sharing a TAN WCS."""
    rng = np.random.default_rng(1)
    sci = rng.normal(100.0, 5.0, (64, 64))
    tpl = rng.normal(100.0, 5.0, (64, 64))
    hdr = _tan_header(150.0, 20.0)
    sci_p = tmp_path / "sci.fits"
    tpl_p = tmp_path / "tpl.fits"
    _write_fits(sci_p, sci, hdr)
    _write_fits(tpl_p, tpl, hdr)
    return sci_p, tpl_p, sci, tpl, hdr


def _templates_obj():
    return Templates({"fwhm": 3.0, "template_subtraction": {}})


# ---------------------------------------------------------------------------
# WCS cache: distortion-aware key + mutation protection
# ---------------------------------------------------------------------------

class TestWcsCache:
    def setup_method(self):
        wcs_mod.invalidate_wcs_cache()

    def test_sip_coefficients_change_cache_key(self):
        """Two headers differing only in SIP terms must not share a WCS."""
        h1 = _sip_header(150.0, 20.0, a10=1e-4)
        h2 = _sip_header(150.0, 20.0, a10=3e-4)
        assert wcs_mod._wcs_cache_key(h1) != wcs_mod._wcs_cache_key(h2)

        w1 = wcs_mod.get_wcs(h1)
        w2 = wcs_mod.get_wcs(h2)
        assert w1 is not w2
        # SIP parsed -> distinct sky transforms off-axis.
        if w1.sip is not None and w2.sip is not None:
            p1 = w1.all_pix2world(200.0, 200.0, 0)
            p2 = w2.all_pix2world(200.0, 200.0, 0)
            assert not np.allclose(p1, p2, rtol=0.0, atol=1e-9)

    def test_pv_coefficients_change_cache_key(self):
        """TPV/PV distortion terms must also enter the key."""
        h1 = _tan_header(150.0, 20.0)
        h2 = _tan_header(150.0, 20.0)
        h2["PV2_1"] = 1.0
        h2["PV2_3"] = 5e-5
        assert wcs_mod._wcs_cache_key(h1) != wcs_mod._wcs_cache_key(h2)

    def test_caller_mutation_does_not_poison_cache(self):
        """Mutating a returned WCS must not corrupt later cache hits."""
        h = _tan_header(150.0, 20.0)
        w1 = wcs_mod.get_wcs(h)
        assert w1 is not None
        w1.wcs.crpix[:] = [1.0, 1.0]  # caller mutation
        w2 = wcs_mod.get_wcs(h)
        assert np.allclose(w2.wcs.crpix, [128.0, 128.0])

    def test_cache_key_none_header_safe(self):
        assert wcs_mod._wcs_cache_key(None) in (None, ())


# ---------------------------------------------------------------------------
# align() failure contract: structured result, never unaligned originals
# ---------------------------------------------------------------------------

class TestAlignFailureContract:
    def test_all_backends_failed_returns_fail_result(
        self, tmp_path, monkeypatch, fits_pair
    ):
        _disable_all_backends(monkeypatch)
        sci_p, tpl_p, sci, tpl, _ = fits_pair
        res = _templates_obj().align(
            scienceFpath=str(sci_p),
            templateFpath=str(tpl_p),
            method="spalipy",
        )
        assert isinstance(res, AlignmentResult)
        assert res.success is False
        assert res.science_path is None
        assert res.template_path is None
        assert res.quality == "fail"

        # Provenance: every attempted method is recorded.
        report = tmp_path / "Alignment_sci.json"
        assert report.exists()
        rec = json.loads(report.read_text())
        methods = {a["method"] for a in rec["attempts"]}
        assert methods == {
            "spalipy", "swarp", "reproject", "astroalign", "chi2_shift"
        }
        assert all(a["success"] is False for a in rec["attempts"])

    def test_alignment_result_success_property(self):
        ok = AlignmentResult("/s", "/t", "reproject/exact", 0.2)
        bad = AlignmentResult(None, None, "spalipy", None)
        assert ok.success is True
        assert bad.success is False
        assert bad.quality == "fail"


# ---------------------------------------------------------------------------
# chi2_shift: mask propagation + common-grid guard
# ---------------------------------------------------------------------------

class TestChi2Shift:
    def test_invalid_mask_propagates_through_shift(
        self, tmp_path, monkeypatch, fits_pair
    ):
        """NaN regions must stay NaN after the sub-pixel template shift."""
        monkeypatch.setattr(templates, "run_IDC", None)
        monkeypatch.setattr(templates, "_HAS_SPALIPY", False)
        monkeypatch.setattr(templates, "_HAS_TWEAKWCS", False)
        monkeypatch.setattr(templates, "_REPROJECT_AVAILABLE", False)
        monkeypatch.setattr(templates, "reproject_interp", None)
        monkeypatch.setattr(templates, "reproject_adaptive", None)
        monkeypatch.setattr(templates, "reproject_exact", None)
        _fake_imgreg(monkeypatch, xoff=2.0, yoff=1.0)
        # Keep the quality probe deterministic regardless of whether the
        # host has SExtractor installed.
        monkeypatch.setattr(
            templates, "compute_alignment_rms", lambda *a, **k: None
        )

        sci_p, tpl_p, sci, tpl, _ = fits_pair
        tpl[:10, :10] = np.nan  # chip-gap-style invalid corner
        _write_fits(tpl_p, tpl, _tan_header(150.0, 20.0))

        res = _templates_obj().align(
            scienceFpath=str(sci_p),
            templateFpath=str(tpl_p),
            method="chi2_shift",
        )
        assert res.success, res.message
        out = fits.getdata(res.template_path)
        assert out.shape == sci.shape
        # Applied shift is (-yoff, -xoff) = (-1, -2): the invalid 10x10
        # block moves to rows<9, cols<8.  Deep interior stays NaN.
        assert np.isnan(out[:7, :6]).all()
        # The interior must carry shifted template data, not zeros/NaN.
        assert np.isfinite(out[20:60, 20:60]).mean() > 0.95

        hdr = fits.getheader(res.template_path)
        assert hdr["ALIGMETH"] == "chi2_shift"
        assert "ALIGQUAL" in hdr
        assert hdr["ALIGMED"] == pytest.approx(np.hypot(2.0, 1.0), abs=0.01)

        rec = json.loads((tmp_path / "Alignment_sci.json").read_text())
        assert rec["attempts"][0]["method"] == "chi2_shift"
        assert rec["attempts"][0]["success"] is True

    def test_shape_mismatch_rejected_not_cropped(
        self, tmp_path, monkeypatch, fits_pair
    ):
        """Different-shaped inputs must fail, not be top-left cropped."""
        monkeypatch.setattr(templates, "run_IDC", None)
        monkeypatch.setattr(templates, "_HAS_SPALIPY", False)
        monkeypatch.setattr(templates, "_HAS_TWEAKWCS", False)
        monkeypatch.setattr(templates, "_REPROJECT_AVAILABLE", False)
        monkeypatch.setattr(templates, "reproject_interp", None)
        monkeypatch.setattr(templates, "reproject_adaptive", None)
        monkeypatch.setattr(templates, "reproject_exact", None)
        _fake_imgreg(monkeypatch)

        sci_p, tpl_p, sci, tpl, _ = fits_pair
        rng = np.random.default_rng(2)
        _write_fits(
            tpl_p, rng.normal(100.0, 5.0, (48, 48)), _tan_header(150.0, 20.0)
        )

        res = _templates_obj().align(
            scienceFpath=str(sci_p),
            templateFpath=str(tpl_p),
            method="chi2_shift",
        )
        assert res.success is False
        rec = json.loads((tmp_path / "Alignment_sci.json").read_text())
        chi2_attempt = rec["attempts"][0]
        assert chi2_attempt["method"] == "chi2_shift"
        assert "shape" in chi2_attempt["message"].lower()


# ---------------------------------------------------------------------------
# P90/P95 gate naming
# ---------------------------------------------------------------------------

class TestP90GateNaming:
    def test_p90_key_preferred(self):
        _, _, p90 = _alignment_gate_thresholds(
            {"template_subtraction": {"alignment_max_p90_px": 2.5}}, 3.0
        )
        assert p90 == pytest.approx(2.5)

    def test_deprecated_p95_key_falls_back(self, monkeypatch):
        monkeypatch.setattr(templates, "_P95_KEY_DEPRECATION_WARNED", False)
        _, _, p90 = _alignment_gate_thresholds(
            {"template_subtraction": {"alignment_max_p95_px": 3.0}}, 3.0
        )
        assert p90 == pytest.approx(3.0)

    def test_p90_key_wins_over_p95(self):
        _, _, p90 = _alignment_gate_thresholds(
            {
                "template_subtraction": {
                    "alignment_max_p90_px": 2.0,
                    "alignment_max_p95_px": 9.0,
                }
            },
            3.0,
        )
        assert p90 == pytest.approx(2.0)

    def test_fwhm_scaling_applied(self):
        _, _, p90 = _alignment_gate_thresholds(
            {"template_subtraction": {"alignment_max_p90_px": 1.5}}, 6.0
        )
        assert p90 == pytest.approx(3.0)  # scale = 6/3 = 2


# ---------------------------------------------------------------------------
# RA-wraparound footprint overlap
# ---------------------------------------------------------------------------

class TestFootprintRAWrap:
    def test_fields_across_ra_zero_overlap(self):
        """Two adjacent fields straddling RA=0 must register as overlapping."""
        h1 = _tan_header(359.97, 20.0)
        h2 = _tan_header(0.03, 20.0)
        assert _wcs_footprints_overlap(h1, (256, 256), h2, (256, 256))

    def test_wrapped_field_is_not_whole_sky(self):
        """A footprint crossing RA=0 must not span 0-360 deg in the test."""
        h_wrap = _tan_header(359.95, 20.0)
        h_far = _tan_header(180.0, 20.0)
        assert not _wcs_footprints_overlap(
            h_wrap, (256, 256), h_far, (256, 256)
        )


# ---------------------------------------------------------------------------
# find_non_uniform_center: NaN / all-uniform safety
# ---------------------------------------------------------------------------

class TestFindNonUniformCenter:
    def test_all_nan_returns_full_frame(self):
        img = np.full((40, 50), np.nan)
        cy, cx, t, b, l, r = Templates.find_non_uniform_center(img)
        assert (t, b, l, r) == (0, 39, 0, 49)
        assert cy == pytest.approx(20.0)
        assert cx == pytest.approx(25.0)

    def test_all_uniform_returns_full_frame(self):
        img = np.zeros((30, 30))
        cy, cx, t, b, l, r = Templates.find_non_uniform_center(img)
        assert (t, b, l, r) == (0, 29, 0, 29)

    def test_nan_padded_block_bbox(self):
        """NaN padding must be treated as invalid, not as data."""
        img = np.full((64, 64), np.nan)
        rng = np.random.default_rng(0)
        img[10:20, 30:40] = rng.random((10, 10)) + 1.0
        cy, cx, t, b, l, r = Templates.find_non_uniform_center(img)
        assert (t, b, l, r) == (10, 19, 30, 39)
        assert cy == pytest.approx(14.5)
        assert cx == pytest.approx(34.5)

    def test_mixed_nan_and_constant_rows(self):
        """Rows with a single valid pixel are data-bearing."""
        img = np.full((20, 20), np.nan)
        img[5, :] = 7.0  # uniform but finite row
        img[6, 10] = 3.0
        cy, cx, t, b, l, r = Templates.find_non_uniform_center(img)
        assert t <= 6 and b >= 6


# ---------------------------------------------------------------------------
# verify_common_grid
# ---------------------------------------------------------------------------

class TestVerifyCommonGrid:
    def test_identical_wcs_passes(self):
        h = _tan_header(150.0, 20.0)
        assert verify_common_grid(h, h, (256, 256))

    def test_shifted_crval_fails(self):
        h1 = _tan_header(150.0, 20.0)
        h2 = _tan_header(150.0 + 0.03, 20.0)  # ~100 px offset
        assert not verify_common_grid(h1, h2, (256, 256))

    def test_wcs_free_header_is_unverifiable_not_mismatched(self):
        assert verify_common_grid(fits.Header(), _tan_header(150.0, 20.0),
                                  (256, 256))


# ---------------------------------------------------------------------------
# Shared quality classifier
# ---------------------------------------------------------------------------

class TestQualityClassifier:
    GATES = (0.5, 0.75, 1.5)

    def test_no_metrics_is_unverified(self):
        q, reasons = _classify_alignment_quality(
            None, None, None, None, *self.GATES
        )
        assert q == "unverified"
        assert reasons == []

    def test_good_metrics_pass(self):
        q, _ = _classify_alignment_quality(0.2, 0.3, 0.6, True, *self.GATES)
        assert q == "pass"

    def test_over_threshold_fails(self):
        q, reasons = _classify_alignment_quality(
            1.2, 0.3, 0.6, True, *self.GATES
        )
        assert q == "fail"
        assert any("offset" in r for r in reasons)

    def test_poor_coverage_warns(self):
        q, _ = _classify_alignment_quality(0.2, 0.3, 0.6, False, *self.GATES)
        assert q == "warn"

    def test_gate_accept_policy(self):
        assert not _alignment_gate_accept("fail", {})
        assert _alignment_gate_accept("pass", {})
        assert _alignment_gate_accept("warn", {})
        assert _alignment_gate_accept("unverified", {})
        # A warn demoted from fail is measured-bad and always rejects.
        demoted = [_WARN_DEMOTED_FROM_FAIL]
        assert not _alignment_gate_accept("warn", {}, demoted)


# ---------------------------------------------------------------------------
# Atomic writes + alignment report
# ---------------------------------------------------------------------------

class TestOutputArtifacts:
    def test_atomic_fits_write_reopenable(self, tmp_path):
        hdr = _tan_header(150.0, 20.0)
        hdr["TESTKEY"] = 42
        data = np.arange(256, dtype=np.float32).reshape(16, 16)
        out = tmp_path / "prod.fits"
        _atomic_fits_write(str(out), data, hdr)
        assert out.exists()
        assert not (tmp_path / "prod.fits.tmp").exists()
        d2 = fits.getdata(out)
        h2 = fits.getheader(out)
        assert np.allclose(d2, data)
        assert h2["TESTKEY"] == 42

    def test_alignment_report_records_attempts(self, tmp_path):
        attempts = [
            AlignmentResult(
                None, None, "spalipy", None,
                quality="fail", message="no matched quads",
            ),
            AlignmentResult(
                "/sci.fits", "/tpl_al.fits", "reproject/exact", 0.21,
                rms_px=0.3, p90_px=0.6, coverage_ok=True, n_matches=24,
                finite_fraction=0.9, quality="pass",
            ),
        ]
        sci = tmp_path / "sci_field.fits"
        tpl = tmp_path / "tpl.fits"
        _write_alignment_report(
            attempts, str(tmp_path), str(sci), str(tpl)
        )
        report = tmp_path / "Alignment_sci_field.json"
        rec = json.loads(report.read_text())
        assert rec["science"] == "sci_field.fits"
        assert len(rec["attempts"]) == 2
        assert rec["attempts"][0]["method"] == "spalipy"
        assert rec["attempts"][0]["success"] is False
        assert rec["attempts"][1]["quality"] == "pass"
        assert rec["attempts"][1]["n_matches"] == 24
        assert rec["attempts"][1]["median_offset_px"] == pytest.approx(0.21)


class TestSpalipySubTileFeasibility:
    """_check_spalipy_sub_tile_feasibility must mirror the frame spalipy
    actually tiles: the source image AFTER padding up to the science
    frame, with the edge mask and min_quad_sep quad requirement.  A tile
    with >=4 detections can still yield zero valid quads (every 4-source
    combination has a pair closer than min_quad_sep), which left
    spalipy's source hash a 1-D empty array and crashed scipy cdist
    (ValueError: XB must be a 2-dimensional array) - the
    SINGLE_SN2024pba_2025-01-05_ZTF_r failure.
    """

    def test_corner_confined_detections_infeasible_on_padded_frame(self):
        rng = np.random.default_rng(0)
        # 249 detections inside a 476x501 corner of a padded 989x644
        # frame: the old unpadded-shape check passed sub_tile=2.
        det = {
            "x": rng.uniform(0, 476, 249),
            "y": rng.uniform(0, 501, 249),
        }
        assert (
            templates._check_spalipy_sub_tile_feasibility(
                det,
                (989, 644),
                2,
                edge_buffer=2,
                min_quad_sep=50.0,
                n_quad_det=25,
            )
            == 1
        )

    def test_same_detections_feasible_on_unpadded_frame(self):
        rng = np.random.default_rng(0)
        det = {
            "x": rng.uniform(0, 476, 249),
            "y": rng.uniform(0, 501, 249),
        }
        assert (
            templates._check_spalipy_sub_tile_feasibility(
                det,
                (501, 476),
                2,
                edge_buffer=2,
                min_quad_sep=50.0,
                n_quad_det=25,
            )
            == 2
        )

    def test_four_detections_zero_quads_is_infeasible(self):
        rng = np.random.default_rng(1)
        # 300 detections spread over the upper half; the lower-right tile
        # gets exactly 4 clustered detections whose only candidate quad
        # fails min_quad_sep - count check passes, quad check fails.
        det = {
            "x": np.concatenate(
                [rng.uniform(0, 644, 300), np.array([10.0, 12.0, 14.0, 16.0])]
            ),
            "y": np.concatenate(
                [rng.uniform(0, 494, 300), np.array([600.0, 601.0, 602.0, 603.0])]
            ),
        }
        assert (
            templates._check_spalipy_sub_tile_feasibility(
                det,
                (989, 644),
                2,
                edge_buffer=0,
                min_quad_sep=50.0,
                n_quad_det=25,
            )
            == 1
        )

    def test_uniform_coverage_stays_feasible(self):
        rng = np.random.default_rng(2)
        det = {
            "x": rng.uniform(0, 644, 300),
            "y": rng.uniform(0, 989, 300),
        }
        assert (
            templates._check_spalipy_sub_tile_feasibility(
                det,
                (989, 644),
                2,
                edge_buffer=2,
                min_quad_sep=50.0,
                n_quad_det=25,
            )
            == 2
        )


# ---------------------------------------------------------------------------
# Trusted-pair spalipy transform and target-aware quad ranking
# ---------------------------------------------------------------------------

class TestSpalipyPairCorrespondence:
    def test_pair_id_subsets_restore_one_to_one_order(self):
        from astropy.table import Table

        sci = Table(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0],
                "pair_id": [0, 1, 2, 3, 4],
            }
        )
        tpl = Table(
            {
                "x": [14.0, 13.0, 12.0, 11.0, 10.0, 99.0],
                "y": [14.0, 13.0, 12.0, 11.0, 10.0, 99.0],
                "pair_id": [4, 3, 2, 1, 0, 9],
            }
        )
        sci_pair, tpl_pair = templates._paired_detection_subsets(sci, tpl)
        assert len(sci_pair) == 5
        assert np.array_equal(
            np.asarray(sci_pair["pair_id"]), np.asarray(tpl_pair["pair_id"])
        )
        np.testing.assert_allclose(tpl_pair["x"], [10.0, 11.0, 12.0, 13.0, 14.0])

    @pytest.mark.skipif(
        not templates._HAS_SPALIPY, reason="spalipy is not installed"
    )
    def test_paired_affine_clips_false_correspondences(self):
        rng = np.random.default_rng(12)
        src = np.column_stack(
            [rng.uniform(0, 500, 60), rng.uniform(0, 500, 60)]
        )
        theta = np.deg2rad(0.1)
        mat = np.array(
            [[1.01 * np.cos(theta), -1.01 * np.sin(theta)],
             [1.01 * np.sin(theta), 1.01 * np.cos(theta)]]
        )
        tpl = src @ mat.T + np.array([3.0, -2.0])
        tpl[-8:] += np.array([8.0, -6.0])
        transform, keep, residual = templates._fit_spalipy_paired_affine(
            src, tpl, clip_px=1.5
        )
        assert transform is not None
        assert np.count_nonzero(keep) == 52
        assert not keep[-8:].any()
        assert transform.scale == pytest.approx(1.01, abs=0.01)
        assert np.nanmedian(residual[keep]) < 1.0


class TestSpalipyTargetQuadFlux:
    def test_bounded_local_anchor_ranking(self):
        from astropy.table import Table

        xy = np.array(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [5.0, 0.0],
                [7.0, 0.0],
                [9.0, 0.0],
                [100.0, 0.0],
                [200.0, 0.0],
                [300.0, 0.0],
                [400.0, 0.0],
                [500.0, 0.0],
            ]
        )
        det = Table(
            {
                "x": xy[:, 0],
                "y": xy[:, 1],
                "flux": [1e6, 1.0, 2.0, 3.0, 4.0, 90.0, 80.0, 70.0, 60.0, 50.0],
                "fwhm": np.full(10, 3.0),
                "flag": np.zeros(10, dtype=int),
            }
        )
        ranked = templates._spalipy_target_quad_flux(
            det, (0.0, 0.0), 8, 0.5, exclude_px=2.0
        )
        top = set(np.argsort(-np.asarray(ranked["flux"]))[:8])
        assert 0 not in top
        assert {1, 2, 3, 4}.issubset(top)
        assert {5, 6, 7, 8}.issubset(top)

    def test_target_scoring_disabled_without_local_pool(self):
        from astropy.table import Table

        det = Table(
            {
                "x": [100.0, 200.0, 300.0, 400.0],
                "y": [100.0, 200.0, 300.0, 400.0],
                "flux": [4.0, 3.0, 2.0, 1.0],
            }
        )
        out = templates._spalipy_target_quad_flux(
            det, (0.0, 0.0), 4, 0.5, exclude_px=1000.0
        )
        np.testing.assert_allclose(out["flux"], det["flux"])

    def test_local_quota_is_split_across_multiple_targets(self):
        from astropy.table import Table

        det = Table(
            {
                "x": [
                    0.0, 3.0, 5.0, 50.0, 47.0, 45.0,
                    100.0, 200.0, 300.0, 400.0,
                ],
                "y": np.zeros(10),
                "flux": [1e6, 1.0, 2.0, 1e6, 3.0, 4.0, 90.0, 80.0, 70.0, 60.0],
            }
        )
        ranked = templates._spalipy_target_quad_flux(
            det,
            [(0.0, 0.0), (50.0, 0.0)],
            8,
            0.5,
            exclude_px=2.0,
        )
        top = set(np.argsort(-np.asarray(ranked["flux"]))[:8])
        assert 0 not in top
        assert 3 not in top
        assert {1, 2, 4, 5}.issubset(top)
        assert {6, 7, 8, 9}.issubset(top)

    def test_alignment_target_positions_include_additional_targets(self):
        header = fits.Header()
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRPIX1"] = 51.0
        header["CRPIX2"] = 51.0
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CD1_1"] = -1.0 / 3600.0
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1.0 / 3600.0
        yaml = {
            "target_x_pix": 20.0,
            "target_y_pix": 30.0,
            "target_name": "SN2026sqf",
            "_additional_targets_resolved": [
                {
                    "name": "SN2021gmj",
                    "ra": 150.0,
                    "dec": 2.0,
                }
            ],
        }
        targets = templates._resolve_alignment_target_positions(
            yaml, (100, 100), header
        )
        assert [row["name"] for row in targets] == [
            "SN2026sqf",
            "SN2021gmj",
        ]
        np.testing.assert_allclose(
            [targets[1]["x"], targets[1]["y"]], [50.0, 50.0], atol=0.2
        )

    def test_target_region_stats_selects_worst_target(self):
        anchors = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [50.0, 0.0],
                [51.0, 0.0],
                [49.0, 0.0],
                [50.0, 1.0],
            ]
        )
        dxy = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        stats = templates._target_region_alignment_stats(
            dxy, anchors, [(0.0, 0.0), (50.0, 0.0)], radius_px=5.0
        )
        assert stats["worst"]["target_index"] == 0
        assert stats["worst"]["n"] == 4
        assert stats["worst"]["median"] == pytest.approx(1.0)
        assert len(stats["per_target"]) == 2
