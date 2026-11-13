"""Tests for the SFFT B-Spline kernel-regularization resolver.

Regularization (Hu et al. 2024) penalizes the matching kernel's second
derivatives at sampled positions; it is only reachable through the Cupy
B-Spline path in sfft 1.7.x, so the resolver is exercised directly rather
than end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autophot.subtraction.run_sfft import _resolve_kernel_regularization


class TestResolveKernelRegularization:
    xy = np.array([[100.5, 200.0], [300.0, 400.0], [50.0, 60.0]])

    def test_auto_enables_below_threshold(self):
        out = _resolve_kernel_regularization("auto", 10, 30, 1e-6, self.xy)
        assert out is not None
        assert out["REGULARIZE_KERNEL"] is True
        assert out["IGNORE_LAPLACIAN_KERCENT"] is True
        assert out["LAMBDA_REGULARIZE"] == pytest.approx(1e-6)
        assert out["WEIGHT_REGULARIZE"] is None
        np.testing.assert_array_equal(out["XY_REGULARIZE"], self.xy)

    def test_auto_disables_above_threshold(self):
        out = _resolve_kernel_regularization("auto", 50, 30, 1e-6, self.xy)
        assert out is None

    def test_true_forces_on(self):
        out = _resolve_kernel_regularization("true", 500, 30, 2e-6, self.xy)
        assert out is not None
        assert out["LAMBDA_REGULARIZE"] == pytest.approx(2e-6)

    def test_false_forces_off(self):
        assert (
            _resolve_kernel_regularization("false", 1, 30, 1e-6, self.xy)
            is None
        )

    def test_no_coordinates_returns_none(self):
        assert (
            _resolve_kernel_regularization("true", 10, 30, 1e-6, None) is None
        )
        assert (
            _resolve_kernel_regularization(
                "true", 10, 30, 1e-6, np.empty((0, 2))
            )
            is None
        )

    def test_nonfinite_rows_dropped(self):
        xy = np.array(
            [[100.0, 200.0], [np.nan, 50.0], [np.inf, 1.0], [10.0, 20.0]]
        )
        out = _resolve_kernel_regularization("true", 10, 30, 1e-6, xy)
        assert out is not None
        assert len(out["XY_REGULARIZE"]) == 2

    def test_all_nonfinite_returns_none(self):
        xy = np.array([[np.nan, 1.0], [np.inf, np.nan]])
        assert _resolve_kernel_regularization("true", 10, 30, 1e-6, xy) is None

    def test_bad_shape_returns_none(self):
        xy = np.array([[1.0, 2.0, 3.0]])
        assert _resolve_kernel_regularization("true", 10, 30, 1e-6, xy) is None

    def test_unknown_mode_disables(self):
        # Anything that is not "auto"/"true" must not silently enable.
        assert (
            _resolve_kernel_regularization("bogus", 1, 30, 1e-6, self.xy)
            is None
        )
