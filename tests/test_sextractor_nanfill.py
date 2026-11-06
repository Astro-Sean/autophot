"""Tests for _nan_filled_copy: SExtractor < 2.25.0 detects nothing on
images containing non-finite pixels, so detection runs on a median-filled
copy that keeps the same pixel grid and filename stem."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from utils.run_sex import _nan_filled_copy


def _write_fits(path, data, header=None):
    fits.writeto(path, data, header, overwrite=True)


class TestNanFilledCopy:

    def test_fills_nonfinite_with_median(self, tmp_path):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        data[0, 0] = np.nan
        data[-1, -1] = np.inf
        src = tmp_path / "science_image.fits"
        _write_fits(src, data)
        work = tmp_path / "work"
        work.mkdir()

        out = _nan_filled_copy(src, work)

        assert out != src
        assert out.stem == "science_image"
        out_data = fits.getdata(out)
        assert np.isfinite(out_data).all()
        expected_fill = np.median(data[np.isfinite(data)])
        assert out_data[0, 0] == expected_fill
        assert out_data[-1, -1] == expected_fill
        finite = np.isfinite(data)
        np.testing.assert_array_equal(out_data[finite], data[finite])

    def test_finite_input_returns_original(self, tmp_path):
        data = np.ones((8, 8), dtype=np.float32)
        src = tmp_path / "clean.fits"
        _write_fits(src, data)

        assert _nan_filled_copy(src, tmp_path / "work") == src
        # No stray copy written
        assert not (tmp_path / "work").exists() or not list(
            (tmp_path / "work").iterdir()
        )

    def test_preserves_header(self, tmp_path):
        data = np.full((4, 4), np.nan, dtype=np.float32)
        src = tmp_path / "hdr.fits"
        hdr = fits.Header()
        hdr["SATURATE"] = 54000.0
        hdr["GAIN"] = 1.03
        _write_fits(src, data, hdr)
        work = tmp_path / "work"
        work.mkdir()

        out = _nan_filled_copy(src, work)
        out_hdr = fits.getheader(out)
        assert out_hdr["SATURATE"] == 54000.0
        assert out_hdr["GAIN"] == 1.03
        # All-NaN image falls back to a 0.0 fill rather than NaN
        assert np.isfinite(fits.getdata(out)).all()

    def test_same_name_collision_uses_suffix(self, tmp_path):
        # If the caller's file already lives in temp_dir, the copy must not
        # overwrite the input.
        data = np.full((4, 4), 1.0, dtype=np.float32)
        data[0, 0] = np.nan
        src = tmp_path / "same.fits"
        _write_fits(src, data)

        out = _nan_filled_copy(src, tmp_path)
        assert out != src
        assert out.stem == "same_nanfilled"
        # Original untouched
        assert np.isnan(fits.getdata(src)[0, 0])
