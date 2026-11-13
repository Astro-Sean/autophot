"""Regression tests for error/uncertainty propagation fixes.

Covers:
* catalog._legacy_ivar_to_mag_err: real per-source Legacy Survey magnitude
  errors from the Tractor inverse variance instead of a fabricated 0.01 mag.
* limits._combine_limit_errors: quadrature combination of the site-bootstrap
  and solver limiting-magnitude errors without letting a NaN one destroy a
  valid term.
"""

from __future__ import annotations

import numpy as np
import pytest

from autophot.calibration.catalog import _legacy_ivar_to_mag_err
from autophot.analysis.limits import _combine_limit_errors


class TestLegacyIvarToMagErr:
    def test_known_value(self):
        # mag = 22.5 -> flux = 1 nanomaggy; ivar = 100 -> sigma_f = 0.1.
        out = _legacy_ivar_to_mag_err(np.array([22.5]), np.array([100.0]))
        expected = (2.5 / np.log(10.0)) * 0.1 / 1.0
        np.testing.assert_allclose(out, [expected])

    def test_fainter_source_larger_error(self):
        # Same inverse variance but fainter flux => larger magnitude error.
        bright = _legacy_ivar_to_mag_err(np.array([20.0]), np.array([100.0]))[0]
        faint = _legacy_ivar_to_mag_err(np.array([22.0]), np.array([100.0]))[0]
        assert faint > bright

    def test_missing_coverage_is_nan(self):
        # The combined DR10 table encodes missing (northern i-band) sky as -9999.
        out = _legacy_ivar_to_mag_err(np.array([-9999.0]), np.array([-9999.0]))
        assert np.isnan(out[0])

    def test_nonpositive_ivar_is_nan(self):
        out = _legacy_ivar_to_mag_err(
            np.array([20.0, 20.0]), np.array([0.0, -5.0])
        )
        assert np.all(np.isnan(out))

    def test_nonfinite_mag_is_nan(self):
        out = _legacy_ivar_to_mag_err(np.array([np.nan]), np.array([100.0]))
        assert np.isnan(out[0])

    def test_no_fabricated_finite_floor(self):
        # A source with no usable variance must not be given a finite error.
        out = _legacy_ivar_to_mag_err(np.array([21.0]), np.array([np.nan]))
        assert np.isnan(out[0])


class TestCombineLimitErrors:
    def test_quadrature_when_both_finite(self):
        assert _combine_limit_errors(0.1, 0.2) == pytest.approx(
            np.hypot(0.1, 0.2)
        )

    def test_nan_other_keeps_bootstrap(self):
        assert _combine_limit_errors(0.1, np.nan) == pytest.approx(0.1)

    def test_nan_bootstrap_keeps_other(self):
        assert _combine_limit_errors(np.nan, 0.2) == pytest.approx(0.2)

    def test_both_nan_returns_nan(self):
        assert np.isnan(_combine_limit_errors(np.nan, np.nan))
