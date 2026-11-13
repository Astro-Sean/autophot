"""Regression tests for the difference-image output mask.

A finite pixel inside the universal mask can carry raw input flux into
the difference image - e.g. a saturated star whose template footprint
was blanked leaves DIFF ~= SCI - Conv(~0) ~= SCI - and reads as a
bright transient downstream.  The full invalid mask (no-data inputs +
universal source mask) must therefore be re-imposed on the output
products, not just the NaN/zero input pixels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autophot.subtraction.run_sfft import _combine_output_masks, _remask_diff_fits


class TestCombineOutputMasks:
    def test_unions_source_mask_with_invalid_inputs(self):
        combined = np.zeros((10, 10), dtype=bool)
        combined[0, 0] = True
        prior = np.zeros((10, 10), dtype=bool)
        prior[5, 5] = True
        prior[0, 0] = True  # overlap counted once
        out = _combine_output_masks(combined, prior)
        assert out[0, 0] and out[5, 5]
        assert int(out.sum()) == 2

    def test_none_source_mask_returns_inputs(self):
        combined = np.zeros((4, 4), dtype=bool)
        combined[1, 1] = True
        out = _combine_output_masks(combined, None)
        assert out.sum() == 1 and out[1, 1]

    def test_does_not_mutate_inputs(self):
        combined = np.zeros((4, 4), dtype=bool)
        prior = np.ones((4, 4), dtype=bool)
        _combine_output_masks(combined, prior)
        assert not combined.any()

    def test_shape_mismatch_source_mask_ignored(self):
        combined = np.zeros((4, 4), dtype=bool)
        prior = np.ones((6, 6), dtype=bool)
        out = _combine_output_masks(combined, prior)
        assert out.shape == (4, 4) and not out.any()


class TestRemaskDiffFits:
    def _write_diff(self, path, data):
        fits.writeto(path, np.asarray(data, dtype=np.float32), overwrite=True)

    def test_masked_bright_pixels_become_nan(self, tmp_path):
        # Saturated star footprint: template blanked -> diff kept raw
        # science flux (~55k) at masked pixels.
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 10.0, (50, 50)).astype(np.float32)
        data[20:25, 20:25] = 55000.0
        mask = np.zeros((50, 50), dtype=bool)
        mask[19:26, 19:26] = True
        p = tmp_path / "diff.fits"
        self._write_diff(p, data)
        n_new, _ = _remask_diff_fits(str(p), mask)
        out = fits.getdata(p)
        assert np.all(~np.isfinite(out[mask]))
        assert n_new == int(np.count_nonzero(np.isfinite(data) & mask))

    def test_unmasked_pixels_unchanged(self, tmp_path):
        rng = np.random.default_rng(1)
        data = rng.normal(5.0, 10.0, (40, 40)).astype(np.float32)
        mask = np.zeros((40, 40), dtype=bool)
        mask[0:5, 0:5] = True
        p = tmp_path / "diff.fits"
        self._write_diff(p, data)
        _remask_diff_fits(str(p), mask)
        out = fits.getdata(p)
        np.testing.assert_array_equal(out[~mask], data[~mask])

    def test_existing_nans_preserved(self, tmp_path):
        data = np.full((30, 30), 100.0, dtype=np.float32)
        data[10:12, 10:12] = np.nan
        mask = np.zeros((30, 30), dtype=bool)
        mask[15:18, 15:18] = True
        p = tmp_path / "diff.fits"
        self._write_diff(p, data)
        _, n_tot = _remask_diff_fits(str(p), mask)
        out = fits.getdata(p)
        assert np.all(~np.isfinite(out[10:12, 10:12]))
        assert np.all(~np.isfinite(out[mask]))
        assert n_tot == int(np.count_nonzero(~np.isfinite(out)))

    def test_shape_mismatch_raises(self, tmp_path):
        data = np.zeros((20, 20), dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        p = tmp_path / "diff.fits"
        self._write_diff(p, data)
        with pytest.raises(ValueError):
            _remask_diff_fits(str(p), mask)

    def test_end_to_end_saturated_star_scenario(self, tmp_path):
        # Reproduce the reported failure: sci ~55k at a saturated
        # footprint, ref ~0 (blanked) -> diff ~ sci.  The combined output
        # mask must invalidate it while leaving the clean unmasked
        # subtraction untouched.
        rng = np.random.default_rng(2)
        data = rng.normal(0.0, 38.0, (80, 80)).astype(np.float32)
        data[30:36, 30:36] = rng.normal(54000.0, 200.0, (6, 6)).astype(np.float32)
        universal = np.zeros((80, 80), dtype=bool)
        universal[29:37, 29:37] = True  # saturated footprint
        combined = np.zeros((80, 80), dtype=bool)
        combined[60:62, 60:62] = True  # chip-gap pixels
        out_mask = _combine_output_masks(combined, universal)
        p = tmp_path / "diff.fits"
        self._write_diff(p, data)
        _remask_diff_fits(str(p), out_mask)
        out = fits.getdata(p)
        assert np.all(~np.isfinite(out[universal]))
        assert np.all(~np.isfinite(out[combined]))
        np.testing.assert_array_equal(out[~out_mask], data[~out_mask])
        assert np.isfinite(out[50, 50])
