"""Tests for PSF candidate-pool vetting inside PSF.build().

Covers:
* Pairwise minimum-separation cut: two candidates closer than
  psf_min_pair_separation_fwhm x FWHM are dropped as an unresolved
  blend, while near-identical entries are deduplicated (better-ranked
  kept).  Catches the same star arriving twice through different
  supplement lists, which per-list isolation checks cannot see.
* Defect-mask proximity: candidates whose core overlaps a masked
  pixel (chip gap, NaN, saturation, streak, cosmic ray) are always
  rejected; supplement-pool members bypass the upstream
  defect-proximity rejection, so the build mask is re-checked here.
* Multi-fail defect veto: candidates failing several independent
  quality cuts are removed even when every per-cut keep-floor would
  skip them - the sparse-pool path that let defect blobs reach the
  ePSF on masked/sparse fields.
"""

from __future__ import annotations

import functools
import logging
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from psf import (
    PSF,
    _psf_mask_distances,
    _psf_pair_separation_drop,
    _psf_spike_flag_counts,
)


class TestPairSeparationDrop:
    def test_blend_pair_drops_both(self):
        xy = np.array([[10.0, 10.0], [13.0, 10.0], [60.0, 60.0]])
        drop = _psf_pair_separation_drop(xy, 10.0, 2.0, np.ones(3))
        assert drop.tolist() == [True, True, False]

    def test_duplicate_keeps_better_ranked(self):
        xy = np.array([[10.0, 10.0], [11.0, 10.0], [60.0, 60.0]])
        drop = _psf_pair_separation_drop(xy, 10.0, 2.0, [5.0, 100.0, 1.0])
        # Entry 1 is the same star at a 1-px offset with higher SNR.
        assert drop.tolist() == [True, False, False]

    def test_duplicate_keeps_first_on_tie(self):
        xy = np.array([[10.0, 10.0], [10.5, 10.0]])
        drop = _psf_pair_separation_drop(xy, 10.0, 2.0, [1.0, 1.0])
        assert drop.tolist() == [False, True]

    def test_wide_pairs_kept(self):
        xy = np.array([[10.0, 10.0], [25.0, 10.0], [60.0, 60.0]])
        drop = _psf_pair_separation_drop(xy, 10.0, 2.0, np.ones(3))
        assert not drop.any()

    def test_chain_of_blends_drops_all(self):
        xy = np.array([[10.0, 10.0], [15.0, 10.0], [20.0, 10.0]])
        drop = _psf_pair_separation_drop(xy, 10.0, 2.0, np.ones(3))
        assert drop.all()

    def test_nonfinite_coords_never_dropped(self):
        xy = np.array([[np.nan, 10.0], [1.0, 10.0], [1.4, 10.0]])
        # Entries 1 and 2 are the same star 0.4 px apart (dedupe);
        # the NaN-coord entry must never enter the pair check.
        drop = _psf_pair_separation_drop(xy, 10.0, 0.5, np.ones(3))
        assert drop.tolist() == [False, False, True]

    def test_disabled_at_zero_separation(self):
        xy = np.array([[10.0, 10.0], [10.1, 10.0]])
        drop = _psf_pair_separation_drop(xy, 0.0, 2.0, np.ones(2))
        assert not drop.any()


class TestMaskDistances:
    def test_distance_to_masked_pixel(self):
        img = np.ones((20, 20))
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        dist = _psf_mask_distances(img, mask, [[10.0, 10.0], [15.0, 10.0], [0.0, 0.0]])
        assert dist[0] == 0.0
        assert dist[1] == pytest.approx(5.0)
        assert dist[2] > 10.0

    def test_no_bad_pixels_gives_inf(self):
        img = np.ones((20, 20))
        dist = _psf_mask_distances(img, None, [[5.0, 5.0]])
        assert np.isinf(dist[0])

    def test_nonfinite_image_counts_as_bad(self):
        img = np.ones((20, 20))
        img[7, 8] = np.nan
        dist = _psf_mask_distances(img, None, [[8.0, 7.0]])
        assert dist[0] == 0.0

    def test_nonfinite_coord_gives_inf(self):
        img = np.ones((20, 20))
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        dist = _psf_mask_distances(img, mask, [[np.nan, 10.0]])
        assert np.isinf(dist[0])

    def test_mask_shape_mismatch_ignored(self):
        img = np.ones((20, 20))
        mask = np.ones((5, 5), dtype=bool)
        dist = _psf_mask_distances(img, mask, [[10.0, 10.0]])
        assert np.isinf(dist[0])


class TestSpikeFlagCounts:
    def test_arm_pixels_counted_in_stamp(self):
        # A spike arm crossing a stamp away from the core: the distance
        # check misses it but the stamp-level count catches it.
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 20] = True  # 20-px vertical arm at x=20
        cnt = _psf_spike_flag_counts(mask, [[20.0, 25.0]], 4)
        assert cnt[0] == 9  # rows 21..29 of the arm, col 20 only

    def test_far_arm_not_counted(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:20, 20] = True
        cnt = _psf_spike_flag_counts(mask, [[20.0, 33.0]], 4)
        assert cnt[0] == 0

    def test_none_and_empty_mask_give_zero(self):
        xy = [[10.0, 10.0]]
        assert _psf_spike_flag_counts(None, xy, 5)[0] == 0
        assert _psf_spike_flag_counts(np.zeros((20, 20), bool), xy, 5)[0] == 0

    def test_box_clips_at_image_edge(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:3, :] = True  # flagged band at top edge
        cnt = _psf_spike_flag_counts(mask, [[5.0, 1.0]], 4)
        assert cnt[0] == 3 * 9  # rows 0..2 x cols 1..9 (box clipped)

    def test_nonfinite_xy_gives_zero(self):
        mask = np.ones((20, 20), dtype=bool)
        cnt = _psf_spike_flag_counts(mask, [[np.nan, 10.0]], 4)
        assert cnt[0] == 0


def _build_yaml(tmp_path, phot_extra=None):
    phot = {
        "aperture_radius": 5.0,
        "psf_cutout_contamination_check": False,
        "psf_fft_rejection": False,
    }
    phot.update(phot_extra or {})
    return {
        "fpath": "test.fits",
        "write_dir": str(tmp_path),
        "fwhm": 4.0,
        "scale": 12.0,
        "gain": 1.0,
        "photometry": phot,
    }


def _defect_candidates(n=8, fwhm_img=4.0):
    """Cosmic-ray/defect-like rows: small FWHM, tiny half-light radius
    and semi-axes, high peak-to-aperture ratio - each row fails
    several independent quality cuts."""
    rng = np.random.default_rng(0)
    x = rng.uniform(15, 60, n)
    y = rng.uniform(15, 60, n)
    return pd.DataFrame(
        {
            "x_pix": x,
            "y_pix": y,
            "fwhm": np.full(n, 0.3 * fwhm_img),
            "flux_radius": np.full(n, 0.1 * fwhm_img),
            "peak_flux": np.full(n, 6e3),
            "flux_AP": np.full(n, 1e4),
            "a": np.full(n, 0.1 * fwhm_img),
            "b": np.full(n, 0.1 * fwhm_img),
            "ELONGATION": np.full(n, 1.1),
            "flags": np.zeros(n),
            "SNR": np.full(n, 20.0),
        }
    )


def _minority_defect_pool(n_clean=8, n_defect=7, fwhm_img=4.0):
    """Clean stellar grid plus a minority defect subset.

    The defect rows fail flux_radius, a/b and peak/aperture while the
    FWHM matches the image value (so they are not removed by the FWHM
    consistency/sigma-clip cuts before the veto).  At 7 fails in 15
    rows each of those cuts is skipped by the keep-floor yet fails a
    minority (<50%) of the pool - the regime the multi-fail veto is
    for.  Positions keep >=14 px (3.5 FWHM) between every pair so the
    isolation cut never fires.
    """
    gx, gy = np.meshgrid(
        np.array([30.0, 65.0, 100.0]), np.array([30.0, 65.0, 100.0])
    )
    x = np.concatenate([gx.ravel()[:n_clean], np.full(n_defect, 130.0)])
    y = np.concatenate(
        [gy.ravel()[:n_clean], np.linspace(25.0, 125.0, n_defect)]
    )
    n = n_clean + n_defect
    df = pd.DataFrame(
        {
            "x_pix": x,
            "y_pix": y,
            "fwhm": np.full(n, fwhm_img),
            "flux_radius": np.concatenate(
                [
                    np.full(n_clean, 0.6 * fwhm_img),
                    np.full(n_defect, 0.1 * fwhm_img),
                ]
            ),
            "peak_flux": np.concatenate(
                [np.full(n_clean, 2e3), np.full(n_defect, 6e3)]
            ),
            "flux_AP": np.concatenate(
                [np.full(n_clean, 4e4), np.full(n_defect, 1e4)]
            ),
            "a": np.concatenate(
                [
                    np.full(n_clean, 0.42 * fwhm_img),
                    np.full(n_defect, 0.1 * fwhm_img),
                ]
            ),
            "b": np.concatenate(
                [
                    np.full(n_clean, 0.42 * fwhm_img),
                    np.full(n_defect, 0.1 * fwhm_img),
                ]
            ),
            "ELONGATION": np.full(n, 1.1),
            "flags": np.zeros(n),
            "SNR": np.full(n, 30.0),
        }
    )
    return df


def _majority_defect_pool(n_clean=4, n_defect=9, fwhm_img=4.0):
    """Defect-dominated pool: n_defect of n rows fail every
    compactness cut (flux_radius, peak/aperture, a/b) while sitting at
    star-like FWHM, so the FWHM-consistency cut sees nothing.  This is
    the GROND-g resampled hot-pixel regime: with 9 of 13 candidates
    failing, each compactness cut is a majority-fail cut and used to
    be exempt from the multi-fail veto entirely.  Positions keep
    >=14 px between every pair so the isolation cut never fires."""
    x = np.concatenate(
        [
            np.array([25.0, 25.0, 125.0, 125.0])[:n_clean],
            np.array(
                [65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 95.0, 95.0, 95.0]
            )[:n_defect],
        ]
    )
    y = np.concatenate(
        [
            np.array([25.0, 125.0, 25.0, 125.0])[:n_clean],
            np.array(
                [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 30.0, 70.0, 110.0]
            )[:n_defect],
        ]
    )
    n = n_clean + n_defect
    is_defect = np.concatenate(
        [np.zeros(n_clean, bool), np.ones(n_defect, bool)]
    )
    return pd.DataFrame(
        {
            "x_pix": x,
            "y_pix": y,
            "fwhm": np.where(is_defect, 0.875 * fwhm_img, fwhm_img),
            "flux_radius": np.where(
                is_defect, 0.25 * fwhm_img, 0.6 * fwhm_img
            ),
            "peak_flux": np.where(is_defect, 1e3, 2e3),
            "flux_AP": np.where(is_defect, 3e3, 4e4),
            "a": np.where(is_defect, 0.2 * fwhm_img, 1.7),
            "b": np.where(is_defect, 0.2 * fwhm_img, 1.7),
            "ELONGATION": np.full(n, 1.1),
            "flags": np.zeros(n),
            "SNR": np.full(n, 30.0),
        }
    )


def _fwhm_outlier_pool(n_clean=5, n_out=3, fwhm_img=4.0):
    """Stellar-morphology pool where n_out rows sit at 2x the locus
    FWHM - inside the absolute [0.5, 2.5] x FWHM consistency band, so
    only the ensemble sigma-clip can remove them (the GROND_g TDP6
    regime: extended companions surviving every compactness cut into
    the ePSF pool).  5 clean + 3 outliers: clipping leaves 5 - below
    the old psf_min_candidates floor of 8, above the new min-keep."""
    gx, gy = np.meshgrid(
        np.array([30.0, 65.0, 100.0]), np.array([30.0, 65.0, 100.0])
    )
    x = np.concatenate(
        [gx.ravel()[:n_clean], np.full(n_out, 130.0)]
    )
    y = np.concatenate(
        [gy.ravel()[:n_clean], np.linspace(25.0, 115.0, n_out)]
    )
    n = n_clean + n_out
    is_out = np.concatenate([np.zeros(n_clean, bool), np.ones(n_out, bool)])
    return pd.DataFrame(
        {
            "x_pix": x,
            "y_pix": y,
            "fwhm": np.where(is_out, 2.0 * fwhm_img, fwhm_img),
            "flux_radius": np.full(n, 0.6 * fwhm_img),
            "peak_flux": np.full(n, 2e3),
            "flux_AP": np.full(n, 4e4),
            "a": np.full(n, 1.7),
            "b": np.full(n, 1.7),
            "ELONGATION": np.full(n, 1.1),
            "flags": np.zeros(n),
            "SNR": np.full(n, 30.0),
        }
    )


class TestFWHMClipKeepFloor:
    """The FWHM sigma-clip is an evidence-based outlier cut: its floor is
    a small absolute count, not psf_min_candidates.  Tying it to the
    target pool size made the clip dead whenever applying it left fewer
    than ~8 candidates - precisely the sparse contaminated pools it
    exists for."""

    def test_clip_fires_below_min_candidates(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        df = _fwhm_outlier_pool()  # 8 rows; clip leaves 5 < old floor 8
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 5
        assert (out["fwhm"] < 5.0).all()

    def test_min_keep_still_protects_tiny_pools(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        df = _fwhm_outlier_pool()
        # Clip would leave 5; raising the floor to 8 must restore it.
        p = PSF(
            _build_yaml(tmp_path, {"psf_fwhm_clip_min_keep": 8}), img
        )
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 8


class TestBuildCandidateVeto:
    """End-to-end through PSF.build(): a minority defect subset must
    not reach the ePSF even when each quality cut it fails is skipped
    by the psf_min_candidates keep-floor.  An all-failing pool is a
    different case: cuts that fail the majority of the candidates are
    field-level misfires (undersampled FLUX_RADIUS, crowded-field
    FLAGS), so their strikes are suppressed and the pool survives."""

    def test_minority_defects_vetoed(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        df = _minority_defect_pool()
        p = PSF(_build_yaml(tmp_path), img)
        model, out = p.build(psfSources=df, plot=False)
        assert out is not None
        # The 7 defect rows are vetoed; the 8 clean rows reach
        # extraction (a noise image yields no usable empirical model).
        assert len(out) == 8
        assert (out["flux_radius"] > 1.0).all()

    def test_all_defect_pool_not_vetoed(self, tmp_path):
        """A pool where every candidate fails the same cuts is a
        field-level misfire, not a defect set: the veto must not empty
        it.  The build still reports no model because a noise image
        yields no usable ePSF - the pipeline falls back to
        aperture-only photometry rather than substituting a Moffat."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        p = PSF(_build_yaml(tmp_path), img)
        model, out = p.build(psfSources=_defect_candidates(), plot=False)
        assert out is not None
        assert len(out) > 0
        assert model is None
        assert p.psf_model_kind == "none"

    def test_failed_build_returns_none_writes_no_model_file(self, tmp_path):
        # A build with no usable empirical model must return None and
        # write no PSF_model_image product - downstream consumers only
        # ever see models genuinely built from field sources.
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        p = PSF(_build_yaml(tmp_path), img)
        model, _ = p.build(psfSources=_defect_candidates(), plot=False)
        assert model is None
        assert p.psf_model_kind == "none"
        assert not (tmp_path / "PSF_model_image_test.fits").is_file()

    def test_veto_disabled_restores_keep_floor(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        p = PSF(_build_yaml(tmp_path, {"psf_defect_veto_min_fails": 0}), img)
        _, out = p.build(psfSources=_defect_candidates(), plot=False)
        # With the veto off, every skipped cut keeps the pool alive, so
        # the defect rows still reach the extraction stage's table.
        assert out is None or len(out) > 0

    def test_majority_compact_failures_vetoed_clean_field(self, tmp_path):
        """On a sparse, well-sampled field a compactness cut failing
        most of the pool is not a field-level misfire -- it means the
        pool itself is defect-dominated.  Reproduces the GROND-g
        resampled-hot-pixel failure: 9 of 13 candidates were defect
        clusters at star-like FWHM, so fr_min/pk_ap/ab_min each failed
        a majority, were exempted, and defect stamps built a needle
        ePSF.  Those strikes must count."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(5)
        img = rng.normal(100.0, 5.0, (150, 150))
        df = _majority_defect_pool()
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 4
        assert (out["flux_radius"] > 1.5).all()

    def test_crowded_majority_compact_failures_not_vetoed(self, tmp_path):
        """The same pool on a crowded field keeps the majority-fail
        exemption: crowding makes FLUX_RADIUS/A-B per-source
        measurements systematically unreliable, so a majority failure
        there is still a field-level misfire."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(6)
        img = rng.normal(100.0, 5.0, (150, 150))
        df = _majority_defect_pool()
        p = PSF(_build_yaml(tmp_path, {"crowded_field": True}), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 13

    def test_blend_pair_removed_from_pool(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        # A 3-px pair at FWHM=4 (0.75 FWHM): an unresolved blend.
        df = pd.DataFrame(
            {
                "x_pix": [30.0, 33.0],
                "y_pix": [40.0, 40.0],
                "fwhm": [4.0, 4.0],
                "flux_radius": [1.6, 1.6],
                "peak_flux": [2e3, 2e3],
                "flux_AP": [4e4, 4e4],
                "a": [1.7, 1.7],
                "b": [1.7, 1.7],
                "ELONGATION": [1.1, 1.1],
                "flags": [0, 0],
                "SNR": [30.0, 30.0],
            }
        )
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 0

    def test_masked_core_candidates_removed(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        mask = np.zeros((80, 80), dtype=bool)
        # Chip-gap band under every candidate: each sits within the
        # 0.75-FWHM core radius of a masked pixel.
        mask[38:43, 28:33] = True
        mask[58:63, 58:63] = True
        df = pd.DataFrame(
            {
                "x_pix": [30.0, 60.0],
                "y_pix": [40.0, 60.0],
                "fwhm": [4.0, 4.0],
                "flux_radius": [1.6, 1.6],
                "peak_flux": [2e3, 2e3],
                "flux_AP": [4e4, 4e4],
                "a": [1.7, 1.7],
                "b": [1.7, 1.7],
                "ELONGATION": [1.1, 1.1],
                "flags": [0, 0],
                "SNR": [30.0, 30.0],
            }
        )
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, mask=mask, plot=False)
        assert out is not None
        assert len(out) == 0


def _clean_candidates(n=8, fwhm_img=4.0, fr_frac=0.6):
    """Stellar-looking rows: per-source FWHM at the image value and a
    half-light radius inside the point-source band (0.5-0.7x FWHM).
    All other quality metrics are clean, so only the profile gate can
    produce a veto strike.  Positions sit on a 20-px grid so the
    isolation cut (3xFWHM) never sees a pair."""
    gx, gy = np.meshgrid(np.arange(15, 80, 20), np.arange(15, 80, 20))
    x = gx.ravel()[:n].astype(float)
    y = gy.ravel()[:n].astype(float)
    return pd.DataFrame(
        {
            "x_pix": x,
            "y_pix": y,
            "fwhm": np.full(n, fwhm_img),
            "flux_radius": np.full(n, fr_frac * fwhm_img),
            "peak_flux": np.full(n, 2e3),
            "flux_AP": np.full(n, 4e4),
            "a": np.full(n, 1.7),
            "b": np.full(n, 1.7),
            "ELONGATION": np.full(n, 1.1),
            "flags": np.zeros(n),
            "SNR": np.full(n, 30.0),
        }
    )


class TestProfileConcentrationCut:
    """The CLASS_STAR replacement: FLUX_RADIUS/FWHM discriminates
    point sources from extended profiles.  Stellar ratios are
    ~0.5-0.7; the psf_flux_radius_max_frac gate (default 0.9) rejects
    galaxies and bloated defect cores."""

    def test_extended_profiles_add_veto_strikes(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        # Minority defect subset: extended profiles + high elongation +
        # defect flags on 7 of 15 rows.  Each cut is keep-floor-skipped
        # (8 kept < 9 required) yet fails a minority, so the strikes
        # count and the veto drops the defect rows.
        df = _minority_defect_pool()
        mask = df["x_pix"] > 120.0
        df.loc[mask, "flux_radius"] = 1.6 * 4.0
        df.loc[mask, "peak_flux"] = 2e3
        df.loc[mask, "flux_AP"] = 4e4
        df.loc[mask, "a"] = 1.7
        df.loc[mask, "b"] = 1.7
        df.loc[mask, "ELONGATION"] = 3.0
        df.loc[mask, "flags"] = 3
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 8
        assert (out["flags"] == 0).all()

    def test_single_extended_fail_alone_not_vetoed(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        # One fail (profile) only: the cut may run, but the veto needs
        # two strikes, so the pool must not be wiped by it.
        df = _clean_candidates(fr_frac=1.6)
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is None or len(out) > 0

    def test_stellar_profiles_keep_pool(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        df = _clean_candidates(fr_frac=0.6)
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        # No strikes anywhere: the pool reaches extraction untouched.
        assert out is None or len(out) > 0

    def test_nan_flux_radius_never_fails(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        df = _clean_candidates()
        df["flux_radius"] = np.nan
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        # Unmeasurable is not failing: no profile strike is possible.
        assert out is None or len(out) > 0


class TestMixedColumnPool:
    """The supplemented pool concatenates frames with disjoint
    columns: catalogue rows carry ELONGATION while detection rows
    only have a/b.  Concatenation fills the gaps with NaN, and a
    NaN elongation must not count as a failed cut - otherwise every
    detection supplement collects a phantom veto strike."""

    def _mixed_pool(self):
        # Detection-style rows: a/b only, NaN ELONGATION.
        det = _clean_candidates(n=6)
        det["ELONGATION"] = np.nan
        # Catalogue-style rows: real ELONGATION, no a/b.
        cat = _clean_candidates(n=2)
        cat[["a", "b"]] = np.nan
        cat["x_pix"] = [70.0, 75.0]
        cat["y_pix"] = [70.0, 75.0]
        return pd.concat([det, cat], ignore_index=True)

    def test_ab_derived_elongation_survives(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=self._mixed_pool(), plot=False)
        # a/b-derived elongation (1.0) keeps detection rows strike-free;
        # NaN catalogue axes stay unmeasurable, not failing.
        assert out is None or len(out) > 0

    def test_ab_elongated_rows_still_vetoed(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (80, 80))
        # Detection-style pool only: elongation must be derived from
        # a/b (no ELONGATION column at all) for the strike to land.
        # One row elongated well past the round field earns the strike;
        # a pool-wide value would be the image PSF and must not count.
        df = _clean_candidates(n=8).drop(columns=["ELONGATION"])
        df.loc[df.index[-1], "a"] = 6.0  # a/b = 3.5: elongated outlier
        df.loc[df.index[-1], "flux_radius"] = 1.6 * 4.0  # second fail
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == len(df) - 1

    def test_peak_aperture_uses_same_units_per_row(self, tmp_path):
        """flux_AP is an e-/s rate on measured (catalogue) rows but
        FLUX_AUTO in image units on detection rows, while
        peak_flux/FLUX_MAX is always image units -- the naive
        peak_flux/flux_AP ratio is inflated by ~exposure_time/gain on
        measured rows and flags every real star.  maxPixel is peak
        e-/s and exists only on measured rows, so the cut must pair
        maxPixel with flux_AP there and peak_flux with flux_AP on
        detection rows."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        det = _clean_candidates(n=7)
        meas = _clean_candidates(n=6)
        meas["x_pix"] = np.linspace(15.0, 135.0, len(meas))
        meas["y_pix"] = 130.0
        # e-/s pair: true ratio 0.02 - stellar.  peak_flux holds the
        # ADU-equivalent value (230x the e-/s rate): a naive
        # peak_flux/flux_AP ratio is ~4.6 and would fail every row.
        meas["flux_AP"] = 1e4
        meas["maxPixel"] = 200.0
        meas["peak_flux"] = 4.6e4
        # One compactness strike each (flux_radius) so a phantom pk_ap
        # strike would veto them: veto needs psf_defect_veto_min_fails=2.
        meas["flux_radius"] = 1.0
        # An e-/s defect row that genuinely fails peak/aperture
        # (0.5 > ~0.28 cap) plus flux_radius: two real strikes, vetoed.
        defect = _clean_candidates(n=1)
        defect["x_pix"] = 135.0
        defect["y_pix"] = 70.0
        defect["flux_AP"] = 1e4
        defect["maxPixel"] = 5e3
        defect["peak_flux"] = 8e3
        defect["flux_radius"] = 1.0
        df = pd.concat([det, meas, defect], ignore_index=True)
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 13
        assert not ((out["x_pix"] == 135.0) & (out["y_pix"] == 70.0)).any()


class TestCutoutValidation:
    """Cutout validation must not reject stars whose stamp carries
    NaN/masked pixels only outside the PSF core.  Aligned frames are
    NaN-rich (resampling, chip gaps); a fully-finite-stamp requirement
    dropped 5 of 7 stars on a real sparse-field build and starved the
    ePSF to two stars."""

    def _star_image(self, rng, n_bad_frac=0.0):
        img = rng.normal(100.0, 5.0, (170, 170))
        yy, xx = np.mgrid[0:170, 0:170]
        sig = 4.0 / 2.354820045
        stars = [
            (30.0, 30.0),
            (30.0, 80.0),
            (30.0, 130.0),
            (80.0, 30.0),
            (80.0, 80.0),
            (80.0, 130.0),
            (130.0, 30.0),
            (130.0, 80.0),
            (130.0, 130.0),
        ]
        for cx, cy in stars:
            img += 3e3 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig))
        # 5x5 NaN patch ~17 px east of every star: inside the cutout
        # stamp (~41 px) but outside both the central-core region and
        # the fit box, and >8 px from every candidate position.
        for cx, cy in stars:
            iy = int(round(cy)) - 3
            ix = int(round(cx)) + 15
            img[iy : iy + 5, ix : ix + 5] = np.nan
        df = pd.DataFrame(
            {
                "x_pix": [c[0] for c in stars],
                "y_pix": [c[1] for c in stars],
                "fwhm": 4.0,
                "flux_radius": 2.4,
                "peak_flux": 3e3,
                "flux_AP": 4e4,
                "a": 1.7,
                "b": 1.7,
                "ELONGATION": 1.05,
                "flags": 0,
                "SNR": 40.0,
            }
        )
        return img, df

    def test_outer_stamp_nans_keep_epsf_stars(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img, df = self._star_image(rng)
        p = PSF(_build_yaml(tmp_path), img)
        model, out = p.build(psfSources=df, plot=False)
        assert model is not None
        # NaN pixels carry weight 0 via the cutout mask; the empirical
        # build must reach and keep most of the injected stars.
        assert p.n_epsf_stars is not None and p.n_epsf_stars >= 6
        assert p.psf_model_kind == "epsf"


class TestStarvedEmpiricalKeep:
    """A starved, FWHM-mismatched empirical model must only be swapped
    for the analytic composite when it actually fits the PSF stars
    worse.  Non-Gaussian PSFs (defocused flat-top: bright plateau core
    over a broad disk) under-measure half-max FWHM relative to the
    catalog image FWHM, so the scalar gate alone condemns a
    structurally-correct model - the Lulin SLT r-band regression where
    a valid defocused ePSF was replaced by a Moffat."""

    def _flattop_image(self, rng):
        img = rng.normal(100.0, 5.0, (190, 190))
        yy, xx = np.mgrid[0:190, 0:190]
        stars = [
            (55.0, 55.0),
            (55.0, 95.0),
            (55.0, 135.0),
            (95.0, 55.0),
            (135.0, 55.0),
            (135.0, 95.0),
        ]
        # Super-Gaussian plateau: flat core to ~R, sharp edge - a
        # defocused-PSF proxy whose half-max FWHM (~5 px) is smaller
        # than the effective image FWHM (8 px) carried in the catalog.
        R = 2.6
        for cx, cy in stars:
            r = np.hypot(xx - cx, yy - cy)
            img += 4e3 * np.exp(-((r / R) ** 8))
        df = pd.DataFrame(
            {
                "x_pix": [c[0] for c in stars],
                "y_pix": [c[1] for c in stars],
                "fwhm": 8.0,
                "flux_radius": 3.0,
                "peak_flux": 4e3,
                "flux_AP": 6e4,
                "a": 2.6,
                "b": 2.6,
                "ELONGATION": 1.05,
                "flags": 0,
                "SNR": 60.0,
            }
        )
        return img, df

    def test_starved_flattop_keeps_empirical_model(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(2)
        img, df = self._flattop_image(rng)
        # fwhm=8.0 matches the catalog values so no build-FWHM clamping
        # rescues the band: the flat-top model measures ~5 px < 0.85*8.
        yaml_ = _build_yaml(tmp_path)
        yaml_["fwhm"] = 8.0
        p = PSF(yaml_, img)
        model, out = p.build(psfSources=df, plot=False)
        assert model is not None
        # 6 stars < min_psf_candidates (8) -> the starved path used to
        # force an analytic swap whenever the measured ePSF FWHM missed
        # the band.  The empirical flat-top model fits its own stars
        # better than any Moffat, so it must be kept.
        assert p.psf_model_kind == "epsf"

    def test_fitter_maxiters_forwarded_to_builder(self, tmp_path, monkeypatch):
        pytest.importorskip("photutils")
        from photutils.psf import EPSFBuilder

        captured = []
        orig_init = EPSFBuilder.__init__

        # functools.wraps is required: build() introspects the signature to
        # pick the photutils >=3.0 kwargs (fit_shape, fitter_maxiters).
        @functools.wraps(orig_init)
        def spy(self, *args, **kwargs):
            captured.append(kwargs)
            orig_init(self, *args, **kwargs)

        monkeypatch.setattr(EPSFBuilder, "__init__", spy)
        img, df = self._flattop_image(np.random.default_rng(2))
        yaml_ = _build_yaml(tmp_path, {"psf_epsf_fitter_maxiters": 37})
        yaml_["fwhm"] = 8.0
        PSF(yaml_, img).build(psfSources=df, plot=False)
        assert captured
        assert all(k.get("fitter_maxiters") == 37 for k in captured)


class TestNonConvergedOutOfBandReject:
    """An ePSF that both failed to converge AND measures outside the
    image-FWHM band is two independent failure signals agreeing: a
    defect stack, not a degraded star ensemble.  On the GROND g field
    a resampled-defect pool produced exactly this (1.6 px model vs
    3.3 px stars, converged=no) and the needle still drove PSF
    photometry.  The build must return no model so the pipeline drops
    to aperture-only photometry."""

    @staticmethod
    def _spike_image(rng):
        img = rng.normal(100.0, 5.0, (160, 160))
        yy, xx = np.mgrid[0:160, 0:160]
        stars = [
            (30.0, 30.0),
            (30.0, 80.0),
            (30.0, 130.0),
            (80.0, 30.0),
            (80.0, 80.0),
            (80.0, 130.0),
            (130.0, 30.0),
            (130.0, 80.0),
            (130.0, 130.0),
        ]
        # Compact defect-like cores (sigma 0.7 px vs 4 px image FWHM):
        # the stacked ePSF measures ~1.7 px, outside the [0.85, 1.5]
        # image-FWHM band.  Row values stay stellar enough to pass the
        # candidate cascade - this test exercises the post-build gate.
        for cx, cy in stars:
            img += 3e3 * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 0.7**2)
            )
        df = pd.DataFrame(
            {
                "x_pix": [c[0] for c in stars],
                "y_pix": [c[1] for c in stars],
                "fwhm": 3.5,
                "flux_radius": 2.0,
                "peak_flux": 3e3,
                "flux_AP": 4e4,
                "a": 1.7,
                "b": 1.6,
                "ELONGATION": 1.05,
                "flags": 0,
                "SNR": 40.0,
            }
        )
        return img, df

    def test_nonconverged_out_of_band_returns_none(self, tmp_path, monkeypatch):
        pytest.importorskip("photutils")
        import psf as psf_module

        # Force the non-converged verdict at the builder boundary: the
        # spike stack converges numerically, so the real signal is
        # patched in.  psf_center_accuracy_relaxed_px=0 also disables
        # the relaxed-acceptance path so the verdict stays negative.
        orig = psf_module._call_build_epsf

        def _forced_nonconverged(builder, epsfstars, init_epsf):
            res = orig(builder, epsfstars, init_epsf)
            if res is not None and hasattr(res, "epsf"):
                res = SimpleNamespace(
                    epsf=res.epsf,
                    fitted_stars=res.fitted_stars,
                    iterations=res.iterations,
                    converged=False,
                    final_center_accuracy=res.final_center_accuracy,
                    n_excluded_stars=getattr(res, "n_excluded_stars", 0),
                    excluded_star_indices=getattr(
                        res, "excluded_star_indices", []
                    ),
                )
            return res

        monkeypatch.setattr(
            psf_module, "_call_build_epsf", _forced_nonconverged
        )
        img, df = self._spike_image(np.random.default_rng(2))
        yaml_ = _build_yaml(
            tmp_path, {"psf_center_accuracy_relaxed_px": 0.0}
        )
        p = PSF(yaml_, img)
        model, _ = p.build(psfSources=df, plot=False)
        assert model is None
        assert p.psf_model_kind == "none"
        assert not (tmp_path / "PSF_model_image_test.fits").is_file()

    def test_converged_out_of_band_kept(self, tmp_path):
        pytest.importorskip("photutils")
        img, df = self._spike_image(np.random.default_rng(2))
        p = PSF(_build_yaml(tmp_path), img)
        model, _ = p.build(psfSources=df, plot=False)
        # The same needle build, converged: kept under the
        # degraded-empirical-model policy (a non-Gaussian PSF can
        # legitimately miss the half-max band).
        assert model is not None
        assert p.psf_model_kind == "epsf"


class TestElongatedField:
    """Field-wide elongation (tracking drift / defocus leaves every
    point source sharing one axis ratio and PA) is the image PSF, not
    a defect: the shape machinery must measure it, the analytic model
    must render it, and the vetting must not strike every real star
    for sharing the field's shape - the Lulin 300 s trailed frames
    (FWHM ~19 px, axis ratio ~0.3, common PA)."""

    @staticmethod
    def _trail(n, q, theta, fwhm, flux, dx=0.0, dy=0.0):
        yy, xx = np.indices((n, n))
        cx, cy = (n - 1) / 2.0 + dx, (n - 1) / 2.0 + dy
        ct, st = np.cos(theta), np.sin(theta)
        xp = (xx - cx) * ct + (yy - cy) * st
        yp = -(xx - cx) * st + (yy - cy) * ct
        sa = fwhm / 2.355
        sb = sa * q
        return flux * np.exp(-0.5 * ((xp / sa) ** 2 + (yp / sb) ** 2))

    @staticmethod
    def _star(data):
        class S:
            pass

        s = S()
        s.data = np.asarray(data, float)
        ny, nx = s.data.shape
        s.cutout_center = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0])
        s.weights = np.ones_like(s.data)
        s.flux = float(np.nansum(s.data))
        return s

    @staticmethod
    def _moments_q_pa(data):
        d = np.asarray(data, float)
        yy, xx = np.indices(d.shape)
        c = (d.shape[0] - 1) / 2.0
        w = np.clip(d - np.median(d), 0.0, None)
        w = w / w.sum()
        mxx = float((w * (xx - c) ** 2).sum())
        myy = float((w * (yy - c) ** 2).sum())
        mxy = float((w * (xx - c) * (yy - c)).sum())
        tr, det = mxx + myy, mxx * myy - mxy * mxy
        sq = np.sqrt(max(tr * tr / 4.0 - det, 0.0))
        q = np.sqrt(max(0.5 * tr - sq, 1e-12) / (0.5 * tr + sq))
        pa = 0.5 * np.degrees(np.arctan2(2.0 * mxy, mxx - myy))
        return q, pa

    def test_elliptical_radius_deforms_isophotes(self):
        from psf import _elliptical_radius

        # Point on the major axis -> r = |x|; on the minor axis the
        # distance is stretched by 1/q so the same isophote reads as
        # one radius.
        ell = (0.5, 0.0)
        assert _elliptical_radius(4.0, 0.0, ell) == pytest.approx(4.0)
        assert _elliptical_radius(0.0, 2.0, ell) == pytest.approx(4.0)
        assert _elliptical_radius(3.0, 4.0, None) == pytest.approx(5.0)

    def test_ensemble_recovers_shared_shape(self):
        from psf import _ensemble_ellipticity

        rng = np.random.default_rng(11)
        theta = np.radians(30.0)
        stars = [
            self._star(
                self._trail(61, 0.30, theta, 19.0, 1000.0)
                + rng.normal(0.0, 0.5, (61, 61))
            )
            for _ in range(8)
        ]
        ell = _ensemble_ellipticity(stars, 19.0)
        assert ell is not None
        assert ell[0] == pytest.approx(0.30, abs=0.06)
        assert np.degrees(ell[1]) == pytest.approx(30.0, abs=3.0)

    def test_ensemble_none_when_round_or_incoherent(self):
        from psf import _ensemble_ellipticity

        rng = np.random.default_rng(3)
        round_stars = [
            self._star(
                self._trail(45, 1.0, 0.0, 8.0, 800.0) + rng.normal(0.0, 0.5, (45, 45))
            )
            for _ in range(6)
        ]
        assert _ensemble_ellipticity(round_stars, 8.0) is None
        # Same elongation per star but PAs 45 deg apart: doubled-angle
        # resultant cancels, so there is no shared axis to measure.
        mix = [
            self._star(
                self._trail(61, 0.3, np.radians(30.0 + i * 45.0), 19.0, 1000.0)
                + rng.normal(0.0, 0.5, (61, 61))
            )
            for i in range(8)
        ]
        assert _ensemble_ellipticity(mix, 19.0) is None

    def test_composite_stamp_matches_ellipticity(self):
        from psf import _composite_moffat_psf, _default_moffat_composite

        comps = _default_moffat_composite(19.0, 3.0)
        ell = (0.30, np.radians(60.0))
        stamp = _composite_moffat_psf(comps, 1, 81, ellipticity=ell)
        q, pa = self._moments_q_pa(stamp.data)
        # PA convention flips sign through second moments vs the
        # (+x toward +y) stamp definition; test the axis ratio and the
        # angle up to the 90-degree frame ambiguity.
        assert q == pytest.approx(0.30, abs=0.06)
        assert abs(pa) == pytest.approx(60.0, abs=4.0)
        stamp_r = _composite_moffat_psf(comps, 1, 81)
        q_r, _ = self._moments_q_pa(stamp_r.data)
        assert q_r == pytest.approx(1.0, abs=0.08)

    @staticmethod
    def _trailed_field(rng):
        """Eight trailed stars (axis ratio ~0.3, PA 35 deg, major-axis
        FWHM ~19 px) on a 260-px image, with SExtractor-consistent
        catalogue values: the per-source fwhm column is the geometric
        mean diameter, flux_radius the circular half-light radius, and
        flux_AP the aperture-integrated flux."""
        img = rng.normal(100.0, 3.0, (340, 340))
        theta = np.radians(35.0)
        sa, sb = 19.0 / 2.355, 19.0 / 2.355 * 0.30
        pos = [
            (80.0, 80.0),
            (80.0, 142.0),
            (80.0, 204.0),
            (142.0, 80.0),
            (142.0, 142.0),
            (142.0, 204.0),
            (204.0, 80.0),
            (204.0, 142.0),
        ]
        yy, xx = np.indices(img.shape)
        for cx, cy in pos:
            ct, st = np.cos(theta), np.sin(theta)
            xp = (xx - cx) * ct + (yy - cy) * st
            yp = -(xx - cx) * st + (yy - cy) * ct
            img += 5e3 * np.exp(-0.5 * ((xp / sa) ** 2 + (yp / sb) ** 2))
        df = pd.DataFrame(
            {
                "x_pix": [p[0] for p in pos],
                "y_pix": [p[1] for p in pos],
                "fwhm": 11.0,
                "flux_radius": 5.2,
                "peak_flux": 5e3,
                "flux_AP": 5e5,
                "a": 8.0,
                "b": 2.4,
                "ELONGATION": 3.33,
                "flags": 0,
                "SNR": 80.0,
            }
        )
        return img, df

    def test_elongated_field_not_vetoed_and_model_elliptical(self, tmp_path):
        pytest.importorskip("photutils")
        img, df = self._trailed_field(np.random.default_rng(5))
        yaml_ = _build_yaml(tmp_path)
        yaml_["fwhm"] = 19.0
        p = PSF(yaml_, img)
        model, out = p.build(psfSources=df, plot=False)
        assert model is not None
        # Every candidate shares the field elongation (a/b = 3.33 >
        # the absolute 1.5 cut): none may be struck out for it, nor for
        # the coupled small-b/flux-radius floors.
        assert len(out) == len(df)
        q, _ = self._moments_q_pa(np.asarray(model.data, float))
        assert q < 0.75

    def test_isolated_extra_elongated_outlier_dropped(self, tmp_path):
        pytest.importorskip("photutils")
        img, df = self._trailed_field(np.random.default_rng(6))
        # One unresolved double stays much more elongated than the
        # field and carries a defect flag: the relaxed cut must still
        # remove it.
        df.loc[df.index[-1], "ELONGATION"] = 7.0
        df.loc[df.index[-1], "a"] = 16.0
        df.loc[df.index[-1], "flags"] = 3
        yaml_ = _build_yaml(tmp_path)
        yaml_["fwhm"] = 19.0
        p = PSF(yaml_, img)
        model, out = p.build(psfSources=df, plot=False)
        assert model is not None
        assert len(out) == len(df) - 1


class TestShapeOutlierDrop:
    """One or two poor cutouts (miscentered, contaminated, deviant
    morphology) keep EPSFBuilder recentering chasing disagreement so
    the build never converges.  The shape check scores each
    flux-normalised cutout against the robust median stack and drops
    only statistical outliers."""

    @staticmethod
    def _gauss(n, fwhm, flux, dx=0.0, dy=0.0):
        yy, xx = np.indices((n, n))
        cx, cy = (n - 1) / 2.0 + dx, (n - 1) / 2.0 + dy
        s = fwhm / 2.355
        return flux * np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / s**2))

    @staticmethod
    def _star(data):
        class S:
            pass

        s = S()
        s.data = np.asarray(data, float)
        s.weights = np.ones_like(s.data)
        s.flux = float(np.nansum(np.clip(s.data, 0.0, None)))
        return s

    def test_single_bad_cutout_dropped(self):
        from psf import _psf_shape_outlier_indices

        rng = np.random.default_rng(9)
        stars = [
            self._star(self._gauss(45, 6.0, 2000.0) + rng.normal(0, 2.0, (45, 45)))
            for _ in range(9)
        ]
        # Bad source: a second nearby star pulled into the cutout.
        bad = self._gauss(45, 6.0, 2000.0) + self._gauss(
            45, 5.0, 1500.0, dx=6.0, dy=-4.0
        )
        stars[4] = self._star(bad + rng.normal(0, 2.0, (45, 45)))
        idx = _psf_shape_outlier_indices(stars, 6.0)
        assert idx == [4]

    def test_miscentred_cutout_dropped(self):
        from psf import _psf_shape_outlier_indices

        rng = np.random.default_rng(10)
        stars = [
            self._star(self._gauss(45, 6.0, 2000.0) + rng.normal(0, 2.0, (45, 45)))
            for _ in range(9)
        ]
        # Same star shape but extracted 4 px off-centre.
        stars[6] = self._star(
            self._gauss(45, 6.0, 2000.0, dx=4.0, dy=3.0) + rng.normal(0, 2.0, (45, 45))
        )
        idx = _psf_shape_outlier_indices(stars, 6.0)
        assert idx == [6]

    def test_uniform_pool_untouched(self):
        from psf import _psf_shape_outlier_indices

        rng = np.random.default_rng(11)
        # Uniformly noisy-but-similar stars: the check must not invent
        # outliers - it only removes stars worse than the ensemble.
        stars = [
            self._star(self._gauss(45, 6.0, 800.0) + rng.normal(0, 30.0, (45, 45)))
            for _ in range(8)
        ]
        assert _psf_shape_outlier_indices(stars, 6.0) == []

    def test_tiny_pool_skipped(self):
        from psf import _psf_shape_outlier_indices

        stars = [self._star(self._gauss(45, 6.0, 2000.0)) for _ in range(4)]
        assert _psf_shape_outlier_indices(stars, 6.0) == []

    def test_removal_bounds(self):
        from psf import _psf_shape_outlier_indices

        rng = np.random.default_rng(12)
        stars = [
            self._star(self._gauss(45, 6.0, 2000.0) + rng.normal(0, 2.0, (45, 45)))
            for _ in range(10)
        ]
        for i in (1, 4, 7):
            bad = self._gauss(45, 6.0, 2000.0) + self._gauss(
                45, 5.0, 1500.0, dx=6.0, dy=-4.0
            )
            stars[i] = self._star(bad + rng.normal(0, 2.0, (45, 45)))
        idx = _psf_shape_outlier_indices(stars, 6.0, max_frac=0.2)
        assert len(idx) <= 2
        idx = _psf_shape_outlier_indices(stars, 6.0, max_frac=0.9, min_keep=9)
        assert len(idx) <= 1
        idx = _psf_shape_outlier_indices(stars, 6.0, max_frac=0.9, min_keep=4)
        assert set(idx) == {1, 4, 7}

    @staticmethod
    def _spike(n, flux=5e3):
        d = np.zeros((n, n))
        c = (n - 1) / 2.0
        d[int(c), int(c)] = flux
        d[int(c), int(c) + 1] = 0.4 * flux
        return d

    def test_sub_floor_cutout_dropped(self):
        """A hot-pixel spike far sharper than the image FWHM is dropped
        by the absolute floor even if it is not an ensemble outlier."""
        from psf import _psf_shape_outlier_indices

        rng = np.random.default_rng(13)
        stars = [
            self._star(self._gauss(45, 6.0, 2000.0) + rng.normal(0, 2.0, (45, 45)))
            for _ in range(8)
        ]
        stars[5] = self._star(self._spike(45) + rng.normal(0, 2.0, (45, 45)))
        idx = _psf_shape_outlier_indices(stars, 6.0, abs_fwhm_floor=0.3)
        assert 5 in idx

    def test_floor_disabled_by_zero(self):
        from psf import _psf_shape_outlier_indices

        # Homogeneous defect pool: with the floor disabled nothing is
        # ensemble-anomalous, so nothing is dropped.
        rng = np.random.default_rng(14)
        stars = [
            self._star(self._spike(45) + rng.normal(0, 2.0, (45, 45)))
            for _ in range(8)
        ]
        assert (
            _psf_shape_outlier_indices(stars, 6.0, abs_fwhm_floor=0.0)
            == []
        )

    def test_defect_dominated_pool_drops_defects_not_stars(self):
        """When the ensemble itself is mostly hot-pixel clusters the
        relative test would flag the real stars; the absolute floor
        removes the defects instead (bounded by min_keep)."""
        from psf import _psf_shape_outlier_indices

        rng = np.random.default_rng(15)
        # 6 spikes (indices 0-5) + 3 real stars (indices 6-8).
        stars = [
            self._star(self._spike(45) + rng.normal(0, 2.0, (45, 45)))
            for _ in range(6)
        ]
        stars += [
            self._star(self._gauss(45, 6.0, 2000.0) + rng.normal(0, 2.0, (45, 45)))
            for _ in range(3)
        ]
        idx = _psf_shape_outlier_indices(
            stars, 6.0, abs_fwhm_floor=0.3, min_keep=4
        )
        # min_keep=4 caps the drop at 9-4=5: the spikes go first, the
        # three real stars always survive.
        assert set(idx).issubset({0, 1, 2, 3, 4, 5})
        assert len(idx) == 5
        assert not set(idx) & {6, 7, 8}


class TestPruneUnsettledRetry:
    """Converge-first flow: build the ePSF on the full pool; only when
    it fails to converge are the sources still drifting (or excluded by
    photutils for repeated fit failures) removed and the build retried
    once on the cleaned pool."""

    def _field_with_double(self, rng):
        img = rng.normal(100.0, 5.0, (300, 300))
        yy, xx = np.indices(img.shape)

        def g(cx, cy, f=4e3, s=2.6):
            return f * np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / s**2))

        pos = [
            (60, 60),
            (60, 140),
            (60, 220),
            (140, 60),
            (140, 140),
            (140, 220),
            (220, 60),
            (220, 140),
        ]
        for cx, cy in pos:
            img += g(cx, cy)
        # Index 8: two equal peaks 7 px apart - the per-iteration
        # centre fit ping-pongs between them and never settles.
        img += g(220, 216.5) + g(220, 223.5)
        pos.append((220, 220))
        df = pd.DataFrame(
            {
                "x_pix": [p[0] for p in pos],
                "y_pix": [p[1] for p in pos],
                "fwhm": 6.0,
                "flux_radius": 2.6,
                "peak_flux": 4e3,
                "flux_AP": 6e4,
                "a": 2.6,
                "b": 2.6,
                "ELONGATION": 1.05,
                "flags": 0,
                "SNR": 60.0,
            }
        )
        return img, df

    def test_unsettled_star_pruned_and_rebuilt(self, tmp_path, caplog):
        pytest.importorskip("photutils")
        img, df = self._field_with_double(np.random.default_rng(0))
        # maxiters_initial=1 makes the first pass deterministically
        # non-converged (first refit always moves centres > 1e-3 px);
        # the pre-build shape filter is off so the double star reaches
        # the builder.
        yaml_ = _build_yaml(
            tmp_path,
            {
                "psf_shape_outlier_rejection": False,
                "psf_maxiters_initial": 1,
                "psf_maxiters_max": 5,
            },
        )
        yaml_["fwhm"] = 6.0
        p = PSF(yaml_, img)
        with caplog.at_level(logging.INFO, logger="psf"):
            model, _ = p.build(psfSources=df, plot=False)
        assert model is not None
        assert p.psf_model_kind == "epsf"
        assert p.n_epsf_stars < 9
        prune_msgs = [
            r.getMessage()
            for r in caplog.records
            if r.name == "psf" and "never settled" in r.getMessage()
        ]
        assert prune_msgs, "expected the non-converged prune-retry to run"
        m = re.search(r"indices \[([0-9, ]+)\]", prune_msgs[0])
        dropped = {int(t) for t in m.group(1).split(",")}
        # The double star (pool index 8, the worst mover) must be dropped.
        assert 8 in dropped
        # The pruned retry converges, so the final verdict must say so.
        assert p.psf_converged is True
        assert any(
            "ePSF converged: yes" in r.getMessage()
            for r in caplog.records
            if r.name == "psf"
        )

    def test_nonconverged_model_reports_status(self, tmp_path, caplog):
        pytest.importorskip("photutils")
        fits = pytest.importorskip("astropy.io.fits")
        img, df = self._field_with_double(np.random.default_rng(0))
        # A single-iteration budget can never converge, so the kept
        # empirical model must report converged=False rather than a
        # silent or misleading verdict.  The relaxed centre-movement
        # acceptance is disabled so the verdict stays strict.
        yaml_ = _build_yaml(
            tmp_path,
            {
                "psf_shape_outlier_rejection": False,
                "psf_maxiters_initial": 1,
                "psf_maxiters_max": 1,
                "psf_center_accuracy_relaxed_px": 0,
            },
        )
        yaml_["fwhm"] = 6.0
        p = PSF(yaml_, img)
        with caplog.at_level(logging.INFO, logger="psf"):
            model, _ = p.build(psfSources=df, plot=False)
        assert model is not None
        assert p.psf_model_kind == "epsf"
        assert p.psf_converged is False
        assert any(
            "ePSF converged: no" in r.getMessage()
            for r in caplog.records
            if r.name == "psf"
        )
        hdr = fits.getheader(str(tmp_path / "PSF_model_image_test.fits"))
        assert hdr["PSFCONV"] is False or hdr["PSFCONV"] == 0

    def test_relaxed_settle_acceptance(self, tmp_path, caplog):
        pytest.importorskip("photutils")
        img, df = self._field_with_double(np.random.default_rng(0))
        # Same single-iteration build, but with the default relaxed
        # bound: once the pruned pool's last-iteration movement falls
        # below psf_center_accuracy_relaxed_px the model is accepted as
        # effectively settled and the verdict says so.
        yaml_ = _build_yaml(
            tmp_path,
            {
                "psf_shape_outlier_rejection": False,
                "psf_maxiters_initial": 1,
                "psf_maxiters_max": 1,
            },
        )
        yaml_["fwhm"] = 6.0
        p = PSF(yaml_, img)
        with caplog.at_level(logging.INFO, logger="psf"):
            model, _ = p.build(psfSources=df, plot=False)
        assert model is not None
        assert p.psf_model_kind == "epsf"
        assert p.psf_converged is True
        assert p.psf_converged_relaxed is True
        msgs = [r.getMessage() for r in caplog.records if r.name == "psf"]
        assert any("effectively settled" in m for m in msgs)
        assert any("ePSF converged: yes (relaxed)" in m for m in msgs)

    def test_bounded_builder_records_moves(self):
        pytest.importorskip("photutils")
        from astropy.nddata import NDData
        from astropy.table import Table
        from photutils.psf import extract_stars

        from psf import (
            _BoundedShiftEPSFBuilder,
            _composite_moffat_psf,
            _default_moffat_composite,
        )

        rng = np.random.default_rng(1)
        img = rng.normal(0.0, 2.0, (300, 300))
        yy, xx = np.indices(img.shape)
        pos = [(60, 60), (60, 140), (140, 60), (140, 140), (220, 60), (220, 140)]
        for cx, cy in pos:
            img += 3000.0 * np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / 2.5**2))
        tbl = Table({"x": [p[0] for p in pos], "y": [p[1] for p in pos]})
        stars = extract_stars(NDData(img), tbl, size=45)
        seed = _composite_moffat_psf(_default_moffat_composite(6.0, 4.765), 1, 45)
        b = _BoundedShiftEPSFBuilder(
            oversampling=1,
            maxiters=4,
            progress_bar=False,
        )
        res = b.build_epsf(stars, epsf=seed)
        assert res is not None
        mv = np.asarray(b._last_move, float)
        # One finite entry per successfully fitted flat star.
        assert mv.shape == (len(stars),)
        assert np.isfinite(mv).any()
        if res.converged:
            assert np.nanmax(mv) < b.center_accuracy_sq


class TestEPSFCrashResilience:
    """A single pathological star must not abort the ePSF build:

    * non-finite cutout pixels are dropped from each per-star fit -
      astropy computes weights*(model - data), so a masked pixel's
      weight 0 does not protect against 0*NaN = NaN;
    * a per-star fitter exception marks the source as failed so the
      photutils exclusion machinery drops it after a few iterations
      instead of the exception propagating out of build_epsf;
    * a build that still crashes salvages the newest completed
      iteration's model as a non-converged empirical ePSF rather than
      losing it to the analytic backdoor.
    """

    def _field(self, rng, nan_pixels=None):
        img = rng.normal(0.0, 2.0, (300, 300))
        yy, xx = np.indices(img.shape)
        pos = [
            (60, 60), (60, 140), (60, 220),
            (140, 60), (140, 140), (140, 220),
            (220, 60), (220, 140), (220, 220),
        ]
        for cx, cy in pos:
            img += 3000.0 * np.exp(
                -0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / 2.5**2)
            )
        for yy_, xx_ in nan_pixels or []:
            img[yy_, xx_] = np.nan
        df = pd.DataFrame(
            {
                "x_pix": [p[0] for p in pos],
                "y_pix": [p[1] for p in pos],
                "fwhm": 6.0,
                "flux_radius": 2.5,
                "peak_flux": 3e3,
                "flux_AP": 5e4,
                "a": 2.5,
                "b": 2.5,
                "ELONGATION": 1.05,
                "flags": 0,
                "SNR": 80.0,
            }
        )
        return img, df

    def _stars(self, img, df):
        from astropy.nddata import NDData
        from astropy.table import Table
        from photutils.psf import extract_stars

        tbl = Table({"x": df["x_pix"].to_numpy(), "y": df["y_pix"].to_numpy()})
        return extract_stars(NDData(img), tbl, size=45)

    def _builder(self, **kw):
        from psf import _BoundedShiftEPSFBuilder

        return _BoundedShiftEPSFBuilder(
            oversampling=1,
            maxiters=6,
            progress_bar=False,
            fit_shape=(21, 21),
            **kw,
        )

    def _seed(self):
        from psf import _composite_moffat_psf, _default_moffat_composite

        return _composite_moffat_psf(
            _default_moffat_composite(6.0, 4.765), 1, 45
        )

    def test_nan_pixel_in_fit_box_does_not_abort_build(self):
        # Two NaN pixels inside the 21x21 fit box of star index 4
        # (resampled/masked frames carry exactly these); the default
        # filter_non_finite drops them from that star's fit and the
        # build completes instead of raising NonFiniteValueError.
        pytest.importorskip("photutils")
        img, df = self._field(
            np.random.default_rng(1), nan_pixels=[(142, 141), (138, 143)]
        )
        stars = self._stars(img, df)
        res = self._builder().build_epsf(stars, epsf=self._seed())
        assert res is not None
        assert np.isfinite(res.epsf.data).all()

    def test_star_fit_exception_excluded_not_fatal(self, monkeypatch):
        # A per-star fitter crash must downgrade to a fit failure so
        # photutils excludes the source after repeated iterations and
        # still returns an empirical model.  photutils >=3.1 masks
        # non-finite pixels before the fitter, so the crash is injected
        # into the base _fit_star directly; the subclass wrapper catches
        # it.  The unreachable center_accuracy keeps the build iterating
        # past the iter>3 exclusion gate.
        pytest.importorskip("photutils")
        from photutils.psf import EPSFBuilder

        img, df = self._field(np.random.default_rng(1))
        stars = self._stars(img, df)
        bad_center = tuple(np.asarray(stars.all_stars[4].center, float))
        orig = EPSFBuilder._fit_star

        def _boom(self, epsf, star, *a, **kw):
            c = getattr(star, "center", None)
            if c is not None and np.allclose(
                np.asarray(c, float), bad_center, atol=0.5
            ):
                raise RuntimeError("injected star-fit crash")
            return orig(self, epsf, star, *a, **kw)

        monkeypatch.setattr(EPSFBuilder, "_fit_star", _boom)
        b = self._builder(center_accuracy=1e-9)
        res = b.build_epsf(stars, epsf=self._seed())
        assert res is not None
        assert np.isfinite(res.epsf.data).all()
        assert res.fitted_stars.all_stars[4]._fit_error_status > 0
        assert 4 in res.excluded_star_indices

    def test_crash_salvages_partial_epsf(self, tmp_path, caplog, monkeypatch):
        pytest.importorskip("photutils")
        from photutils.psf import EPSFBuilder

        img, df = self._field(np.random.default_rng(0))
        orig = EPSFBuilder._process_iteration

        def _boom(self, stars, epsf, iter_num):
            if iter_num >= 3:
                raise RuntimeError("injected mid-build crash")
            return orig(self, stars, epsf, iter_num)

        monkeypatch.setattr(EPSFBuilder, "_process_iteration", _boom)
        # Strict verdict: relaxed acceptance off so a salvaged model
        # can never be reported as settled.
        yaml_ = _build_yaml(
            tmp_path,
            {
                "psf_shape_outlier_rejection": False,
                "psf_center_accuracy_relaxed_px": 0,
            },
        )
        yaml_["fwhm"] = 6.0
        p = PSF(yaml_, img)
        with caplog.at_level(logging.INFO, logger="psf"):
            model, _ = p.build(psfSources=df, plot=False)
        assert model is not None
        assert p.psf_model_kind == "epsf"
        assert p.psf_converged is False
        assert any(
            "salvag" in r.getMessage()
            for r in caplog.records
            if r.name == "psf"
        )


def _clean_catalog_row_pool(positions, fwhm_img=4.0):
    """Catalog-clean rows at the given positions: every value passes the
    quality cuts, so the rows reach extraction regardless of what the
    image actually contains at those positions."""
    n = len(positions)
    x = np.array([p[0] for p in positions], dtype=float)
    y = np.array([p[1] for p in positions], dtype=float)
    return pd.DataFrame(
        {
            "x_pix": x,
            "y_pix": y,
            "fwhm": np.full(n, fwhm_img),
            "flux_radius": np.full(n, 0.6 * fwhm_img),
            "peak_flux": np.full(n, 2e3),
            "flux_AP": np.full(n, 4e4),
            "a": np.full(n, 0.42 * fwhm_img),
            "b": np.full(n, 0.42 * fwhm_img),
            "ELONGATION": np.full(n, 1.1),
            "flags": np.zeros(n),
            "SNR": np.full(n, 30.0),
        }
    )


def _paint_gauss(img, cx, cy, fwhm, flux):
    yy, xx = np.indices(img.shape)
    s = fwhm / 2.355
    img += flux * np.exp(
        -0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / s**2)
    )


class TestPeakApertureHardCap:
    """psf_peak_to_aperture_ratio_max is ceilinged at 0.80 even when the
    configured value is huge: a ratio above ~0.8 means nearly all flux
    sits in 1-2 px, i.e. a hot-pixel/CR cluster, never a star."""

    def _pool(self):
        # 6 clean rows + 4 rows that fail only fr_min and (with the cap)
        # pk_ap.  Without the 0.80 ceiling the configured 5.0 disables the
        # pk/ap strike, leaving one strike per defect and no veto.
        gx, gy = np.meshgrid(
            np.array([30.0, 65.0, 100.0]), np.array([30.0, 65.0, 100.0])
        )
        x = np.concatenate([gx.ravel()[:6], np.full(4, 130.0)])
        y = np.concatenate([gy.ravel()[:6], np.linspace(25.0, 85.0, 4)])
        n = 10
        fwhm_img = 4.0
        return pd.DataFrame(
            {
                "x_pix": x,
                "y_pix": y,
                "fwhm": np.full(n, fwhm_img),
                "flux_radius": np.concatenate(
                    [np.full(6, 0.6 * fwhm_img), np.full(4, 0.1 * fwhm_img)]
                ),
                "peak_flux": np.concatenate(
                    [np.full(6, 2e3), np.full(4, 8.5e3)]
                ),
                "flux_AP": np.concatenate(
                    [np.full(6, 4e4), np.full(4, 1e4)]
                ),
                "a": np.full(n, 0.42 * fwhm_img),
                "b": np.full(n, 0.42 * fwhm_img),
                "ELONGATION": np.full(n, 1.1),
                "flags": np.zeros(n),
                "SNR": np.full(n, 30.0),
            }
        )

    def test_configured_high_value_still_caps_at_080(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(2)
        img = rng.normal(100.0, 5.0, (150, 150))
        yaml_ = _build_yaml(
            tmp_path, {"psf_peak_to_aperture_ratio_max": 5.0}
        )
        p = PSF(yaml_, img)
        _, out = p.build(psfSources=self._pool(), plot=False)
        assert out is not None
        # pk/ap=0.85 > 0.80 + fr_min -> 2 strikes -> vetoed.
        assert len(out) == 6


class TestDefectDominanceDiagnostics:
    """When the candidate pool is mostly hot-pixel/CR clusters the
    ensemble-relative vetoes cannot fix it - the code now warns instead
    of silently building an ePSF from defects."""

    def test_low_fwhm_pool_warns(self, tmp_path, caplog):
        """Catalog stage: most surviving candidates hug the lower FWHM
        bound -> defect-dominance warning."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(3)
        img = rng.normal(100.0, 5.0, (150, 150))
        fwhm_img = 4.0
        # fwhm_lo = 0.5 * 4 = 2.0.  10 rows sit at 2.1 (within 20% of the
        # bound), 3 at the image FWHM.
        xs = np.linspace(20.0, 130.0, 13)
        df = pd.DataFrame(
            {
                "x_pix": xs,
                "y_pix": np.full(13, 75.0),
                "fwhm": np.concatenate(
                    [np.full(10, 2.1), np.full(3, fwhm_img)]
                ),
                "flux_radius": np.concatenate(
                    [np.full(10, 1.5), np.full(3, 0.6 * fwhm_img)]
                ),
                "peak_flux": np.full(13, 2e3),
                "flux_AP": np.full(13, 4e4),
                "a": np.full(13, 1.4),
                "b": np.full(13, 1.4),
                "ELONGATION": np.full(13, 1.1),
                "flags": np.zeros(13),
                "SNR": np.full(13, 30.0),
            }
        )
        p = PSF(_build_yaml(tmp_path), img)
        with caplog.at_level(logging.WARNING, logger="psf"):
            p.build(psfSources=df, plot=False)
        assert any(
            "defect-dominated" in r.getMessage() for r in caplog.records
        )

    def test_sharp_cutout_pool_warns(self, tmp_path, caplog):
        """Post-extraction stage: most cutouts measure far sharper than
        the image FWHM -> defect-dominance warning."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(4)
        img = rng.normal(100.0, 5.0, (170, 170))
        # 5 real stars (FWHM=4) + 11 sharp 2-px-wide blobs, all on an
        # isolated grid.  Catalog rows are uniformly clean: the scenario
        # is defects whose catalog metrics masqueraded as stars.
        xs = np.tile(np.linspace(25.0, 145.0, 4), 4)
        ys = np.repeat(np.linspace(25.0, 145.0, 4), 4)
        positions = list(zip(xs, ys))
        for cx, cy in positions[:5]:
            _paint_gauss(img, cx, cy, 4.0, 4e3)
        for cx, cy in positions[5:]:
            _paint_gauss(img, cx, cy, 1.6, 4e3)
        df = _clean_catalog_row_pool(positions, fwhm_img=4.0)
        yaml_ = _build_yaml(
            tmp_path,
            {
                # Keep the ensemble checks from pruning the pool before
                # the diagnostic runs.
                "psf_shape_outlier_rejection": False,
            },
        )
        p = PSF(yaml_, img)
        with caplog.at_level(logging.WARNING, logger="psf"):
            p.build(psfSources=df, plot=False)
        assert any(
            "dominated by defects" in r.getMessage()
            for r in caplog.records
        )

    def test_fft_relaxed_not_skipped_undersampled(self, tmp_path, caplog):
        """On undersampled data the FFT outlier pass is relaxed to a
        higher sigma rather than skipped outright."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(5)
        img = rng.normal(100.0, 5.0, (170, 170))
        xs = np.tile(np.linspace(25.0, 145.0, 4), 4)
        ys = np.repeat(np.linspace(25.0, 145.0, 4), 4)
        positions = list(zip(xs, ys))
        for cx, cy in positions:
            _paint_gauss(img, cx, cy, 2.0, 4e3)
        fwhm_img = 2.0
        n = len(positions)
        df = pd.DataFrame(
            {
                "x_pix": xs,
                "y_pix": ys,
                "fwhm": np.full(n, fwhm_img),
                "flux_radius": np.full(n, 1.2),
                "peak_flux": np.full(n, 2e3),
                "flux_AP": np.full(n, 2e4),
                "a": np.full(n, 0.84),
                "b": np.full(n, 0.84),
                "ELONGATION": np.full(n, 1.1),
                "flags": np.zeros(n),
                "SNR": np.full(n, 30.0),
            }
        )
        yaml_ = _build_yaml(
            tmp_path,
            {
                "psf_fft_rejection": True,
                "psf_shape_outlier_rejection": False,
            },
        )
        yaml_["fwhm"] = fwhm_img
        p = PSF(yaml_, img)
        with caplog.at_level(logging.INFO, logger="psf"):
            p.build(psfSources=df, plot=False)
        msgs = [r.getMessage() for r in caplog.records if r.name == "psf"]
        assert any("FFT threshold relaxed" in m for m in msgs)
        assert not any("skipping FFT outlier rejection" in m for m in msgs)


def _isolated_clean_pool(positions, snr, fwhm_img=4.0):
    """Stellar-morphology rows at given positions with a shared S/N."""
    n = len(positions)
    return pd.DataFrame(
        {
            "x_pix": [c[0] for c in positions],
            "y_pix": [c[1] for c in positions],
            "fwhm": np.full(n, fwhm_img),
            "flux_radius": np.full(n, 0.6 * fwhm_img),
            "peak_flux": np.full(n, 2e3),
            "flux_AP": np.full(n, 4e4),
            "a": np.full(n, 0.42 * fwhm_img),
            "b": np.full(n, 0.42 * fwhm_img),
            "ELONGATION": np.full(n, 1.1),
            "flags": np.zeros(n),
            "SNR": np.asarray(snr, dtype=float),
        }
    )


class TestIsolationNeighbourSNR:
    """The isolation cut counts only catalog rows with enough S/N to
    contaminate a cutout.  Sub-threshold rows are detection noise whose
    pixels the cutout-level contamination checks already mask; counting
    them vetoes good stars next to noise blobs.  Self-duplicates of the
    candidate inside ~2 px are discounted, not neighbours."""

    _GRID = [
        (30.0, 30.0),
        (65.0, 30.0),
        (100.0, 30.0),
        (30.0, 65.0),
        (65.0, 65.0),
        (100.0, 65.0),
        (30.0, 100.0),
        (65.0, 100.0),
        (100.0, 100.0),
    ]

    def _catalog(self, extra_rows):
        # The merged pool's own entries sit at identical coordinates;
        # they must be self-discounted, not counted as neighbours.
        rows = [(x, y, 50.0) for x, y in self._GRID]
        rows += extra_rows
        return pd.DataFrame(
            {
                "x_pix": [r[0] for r in rows],
                "y_pix": [r[1] for r in rows],
                "SNR": [r[2] for r in rows],
            }
        )

    def _run(self, tmp_path, extra_rows):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        df = _isolated_clean_pool(self._GRID, 30.0)
        yaml_ = _build_yaml(
            tmp_path,
            {"last_raw_sex_catalog": self._catalog(extra_rows)},
        )
        p = PSF(yaml_, img)
        _, out = p.build(psfSources=df, plot=False)
        return out

    def test_low_snr_neighbour_does_not_veto(self, tmp_path):
        # SNR 2 blob 8 px from candidate 0 (iso_r = 3x4 = 12 px).
        out = self._run(tmp_path, [(38.0, 30.0, 2.0)])
        assert out is not None
        assert len(out) == len(self._GRID)

    def test_real_neighbour_still_vetoes(self, tmp_path):
        out = self._run(tmp_path, [(38.0, 30.0, 50.0)])
        assert out is not None
        kept = set(zip(out["x_pix"], out["y_pix"]))
        assert (30.0, 30.0) not in kept
        assert len(out) == len(self._GRID) - 1

    def test_neighbour_snr_floor_configurable(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        df = _isolated_clean_pool(self._GRID, 30.0)
        yaml_ = _build_yaml(
            tmp_path,
            {
                "last_raw_sex_catalog": self._catalog([(38.0, 30.0, 5.0)]),
                "psf_isolation_neighbour_snr_min": 4.0,
            },
        )
        p = PSF(yaml_, img)
        _, out = p.build(psfSources=df, plot=False)
        kept = set(zip(out["x_pix"], out["y_pix"]))
        assert (30.0, 30.0) not in kept


class TestSNRHardFloorRescue:
    """On a starved field the hard S/N floor rescues the best
    sub-floor candidates down to psf_snr_rescue_min so the empirical
    build keeps enough samples; rescued stars still face cutout-level
    vetting downstream."""

    _GOOD = [
        (30.0, 30.0),
        (65.0, 30.0),
        (100.0, 30.0),
        (30.0, 65.0),
        (65.0, 65.0),
    ]
    _SUB = [
        (100.0, 65.0),
        (30.0, 100.0),
        (65.0, 100.0),
        (100.0, 100.0),
    ]

    def _pool(self):
        pos = self._GOOD + self._SUB
        snr = [30.0] * len(self._GOOD) + [7.0] * len(self._SUB)
        return _isolated_clean_pool(pos, snr)

    def test_rescue_keeps_best_subfloor(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=self._pool(), plot=False)
        assert out is not None
        # Hard floor leaves 5 < min_candidates (8): rescue restores 3
        # of the 4 sub-floor rows.
        assert len(out) == 8
        assert int((out["SNR"] < 10.0).sum()) == 3

    def test_rescue_disabled_drops_to_floor_survivors(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        p = PSF(
            _build_yaml(tmp_path, {"psf_snr_rescue_min": 0.0}), img
        )
        _, out = p.build(psfSources=self._pool(), plot=False)
        assert out is not None
        assert len(out) == 5

    def test_rescue_floor_excludes_faint_rejects(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 5.0, (150, 150))
        pos = self._GOOD + self._SUB
        snr = [30.0] * len(self._GOOD) + [7.0, 7.0, 7.0, 3.0]
        df = _isolated_clean_pool(pos, snr)
        p = PSF(_build_yaml(tmp_path), img)
        _, out = p.build(psfSources=df, plot=False)
        assert out is not None
        assert len(out) == 8
        # The SNR 3 row is below psf_snr_rescue_min and never rescued.
        assert float(out["SNR"].min()) >= 5.0


class TestThinEnsembleKeepsEmpirical:
    """A thin but WORKING ePSF stays the delivered model: star count
    below psf_empirical_min_stars alone cannot condemn a build that
    passed every quality gate.  The analytic composite is only
    delivered when the thin model failed a gate or demonstrably loses
    to the analytic fit on its own stars."""

    def _star_image(self, rng):
        img = rng.normal(100.0, 5.0, (190, 190))
        yy, xx = np.mgrid[0:190, 0:190]
        stars = [
            (55.0, 55.0),
            (55.0, 135.0),
            (135.0, 55.0),
            (135.0, 135.0),
            (95.0, 95.0),
        ]
        sig = 4.0 / 2.3548
        for cx, cy in stars:
            img += 4e3 * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig)
            )
        df = _isolated_clean_pool(stars, 60.0)
        return img, df

    def test_thin_pool_delivers_epsf_not_analytic(self, tmp_path):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(3)
        img, df = self._star_image(rng)
        p = PSF(_build_yaml(tmp_path), img)
        model, _ = p.build(psfSources=df, plot=False)
        assert model is not None
        # 5 stars < psf_empirical_min_stars (6): the working thin model
        # is kept (flagged understaffed), not swapped for the analytic
        # composite.
        assert p.psf_model_kind == "epsf"
        assert p.psf_ensemble_understaffed
        assert p.n_epsf_stars is not None and p.n_epsf_stars >= 4
