"""Tests for sparse-field evidence-aware classification.

Covers the transform-aware alignment verification tier (ALIGVER).
"""

from __future__ import annotations

import numpy as np

from autophot.subtraction.templates import _alignment_verification_tier, _classify_alignment_quality


def _quad(n=10, coverage_ok=True, quadrants=4, coherent=False):
    return {
        "n_matched": n,
        "coverage_ok": coverage_ok,
        "n_quadrants": quadrants,
        "coherent_cluster": coherent,
    }


class TestVerificationTier:
    def test_no_matches_unverified(self):
        assert _alignment_verification_tier(None) == "unverified"
        assert _alignment_verification_tier({"n_matched": 0}) == "unverified"

    def test_no_matches_wcs_supported(self):
        assert (
            _alignment_verification_tier(None, wcs_based=True)
            == "wcs_supported"
        )

    def test_too_few_matches_unverified(self):
        # Below the translation threshold nothing is established.
        assert (
            _alignment_verification_tier(_quad(n=2, coverage_ok=False))
            == "unverified"
        )

    def test_coherent_cluster_translation_only(self):
        tier = _alignment_verification_tier(
            _quad(n=12, coherent=True, coverage_ok=True)
        )
        assert tier == "translation_verified"

    def test_clustered_matches_translation_only(self):
        # Many matches confined to one region: coverage_ok=False means the
        # residuals cannot verify rotation/scale field-wide.
        tier = _alignment_verification_tier(
            _quad(n=30, coverage_ok=False, quadrants=1)
        )
        assert tier == "translation_verified"

    def test_distributed_affine(self):
        tier = _alignment_verification_tier(
            _quad(n=10, coverage_ok=True, quadrants=2)
        )
        assert tier == "affine_verified"

    def test_field_wide_spatial(self):
        tier = _alignment_verification_tier(
            _quad(n=25, coverage_ok=True, quadrants=4)
        )
        assert tier == "spatially_verified"

    def test_spatial_needs_quadrant_spread(self):
        # 20+ matches but only 2 occupied quadrants: affine, not spatial.
        tier = _alignment_verification_tier(
            _quad(n=25, coverage_ok=True, quadrants=2)
        )
        assert tier == "affine_verified"


class TestTranslationOnlyQualityCap:
    def test_coherent_cluster_caps_at_warn(self):
        cls, reasons = _classify_alignment_quality(
            0.1, 0.2, 0.3, True, 1.0, 1.0, 1.0,
            n_matched=4, translation_only=True,
        )
        assert cls == "warn"
        assert any("translation-only" in r for r in reasons)

    def test_no_metrics_unverified(self):
        cls, _ = _classify_alignment_quality(
            None, None, None, None, 1.0, 1.0, 1.0
        )
        assert cls == "unverified"
