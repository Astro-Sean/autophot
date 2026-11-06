"""Tests for per-pixel error sanitization in psf.py.

Regression: np.maximum floors zeros but passes NaN/inf through, so any
non-finite error (NaN difference-image pixel, inf bkg rms) made photutils
abort the whole image's PSF fit.
"""

from __future__ import annotations

import numpy as np

from psf import _sanitize_error_array


class TestSanitizeErrorArray:
    def test_floors_zeros(self):
        err = np.array([0.0, -2.0, 3.0])
        out = _sanitize_error_array(err, fallback=5.0)
        assert np.isfinite(out).all()
        assert out[0] == 1e-12
        assert out[2] == 3.0

    def test_nan_replaced_by_median(self):
        err = np.array([1.0, np.nan, 3.0, 5.0])
        out = _sanitize_error_array(err, fallback=9.0)
        assert np.isfinite(out).all()
        assert out[1] == np.median([1.0, 3.0, 5.0])

    def test_inf_replaced_by_median(self):
        err = np.array([2.0, np.inf, 4.0])
        out = _sanitize_error_array(err, fallback=9.0)
        assert out[1] == 3.0  # median of 2, 4

    def test_neg_inf_floored_then_finite(self):
        # np.maximum(-inf, floor) = floor, so -inf ends finite and small.
        err = np.array([2.0, -np.inf, 4.0])
        out = _sanitize_error_array(err, fallback=9.0)
        assert np.isfinite(out).all()
        assert out[1] == 1e-12

    def test_all_nonfinite_uses_fallback(self):
        err = np.array([np.nan, np.inf, np.nan])
        out = _sanitize_error_array(err, fallback=7.5)
        assert np.isfinite(out).all()
        assert (out == 7.5).all()

    def test_clean_array_unchanged(self):
        err = np.array([1.0, 2.0, 3.0])
        out = _sanitize_error_array(err, fallback=9.0)
        assert np.array_equal(out, err)
