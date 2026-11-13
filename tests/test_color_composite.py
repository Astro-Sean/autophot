"""Tests for the RGB color-composite image product.

Synthetic same-grid frames exercise the whole path (resample passthrough,
flux scaling, robust normalisation, stretch, channel mixing) without
needing real survey data.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from autophot.analysis.color_composite import (
    adjust_rgb,
    assign_channels,
    black_white_levels,
    combine_frames,
    compose_rgb,
    flux_scale,
    generate_color_image,
    normalize_band,
    resample_to_grid,
    stretch_channel,
)


NY, NX = 64, 80


def _tan_header():
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = NX / 2.0
    hdr["CRPIX2"] = NY / 2.0
    hdr["CRVAL1"] = 150.0
    hdr["CRVAL2"] = -20.0
    hdr["CDELT1"] = -0.001
    hdr["CDELT2"] = 0.001
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = NX
    hdr["NAXIS2"] = NY
    return hdr


def _tan_wcs():
    return WCS(_tan_header())


def _frame(seed=0, level=100.0):
    rng = np.random.default_rng(seed)
    return level + rng.normal(0.0, 1.0, (NY, NX))


def test_flux_scale_math():
    assert flux_scale(100.0, 25.0) == pytest.approx(1e8)
    assert flux_scale(None, None) == 1.0
    assert flux_scale(-1.0, np.nan) == 1.0


def test_black_white_levels_and_normalize():
    data = np.concatenate([np.full(900, 10.0), np.full(100, 200.0)])
    black, white = black_white_levels(data, 1.0, 99.0)
    assert black < white
    norm = normalize_band(data, black, white)
    assert norm.min() == 0.0 and norm.max() <= 1.0
    assert norm[data == 10.0].mean() < 0.05


def test_stretch_channel_shapes():
    v = np.linspace(0.0, 1.0, 101)
    for name in ("asinh", "log", "sqrt", "linear"):
        out = stretch_channel(v, name, 10.0)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)
        assert np.all(np.diff(out) >= 0)
        assert np.all((out >= 0) & (out <= 1))
    with pytest.raises(ValueError):
        stretch_channel(v, "nonsense")


def test_assign_channels_orders_by_wavelength():
    w_eff = {"g": 4830, "r": 6260, "i": 7670}
    ch = assign_channels(["g", "r", "i"], w_eff=w_eff)
    assert ch == {"R": ["i"], "G": ["r"], "B": ["g"]}


def test_assign_channels_two_bands_synthesizes_green():
    ch = assign_channels(["r", "i"], w_eff={"r": 6260, "i": 7670})
    assert ch["R"] == ["i"] and ch["B"] == ["r"]
    assert ch["G"] == ["__synthetic__"]


def test_assign_channels_map_and_mixing():
    ch = assign_channels(
        ["g", "r", "V"], channel_map={"R": "r", "G": ["g", "V"]}
    )
    assert ch["R"] == ["r"]
    assert ch["G"] == ["g", "V"]
    assert ch["B"] == []


def test_compose_rgb_two_band_synthetic_green():
    bands = {
        "i": np.full((4, 4), 0.8),
        "r": np.full((4, 4), 0.4),
    }
    ch = {"R": ["i"], "G": ["__synthetic__"], "B": ["r"]}
    rgb = compose_rgb(bands, ch)
    assert rgb.shape == (4, 4, 3)
    assert rgb[0, 0, 0] == pytest.approx(0.8)
    assert rgb[0, 0, 2] == pytest.approx(0.4)
    assert rgb[0, 0, 1] == pytest.approx(0.6)


def test_adjust_rgb_gamma_and_saturation():
    rgb = np.full((4, 4, 3), 0.25)
    bright = adjust_rgb(rgb, gamma=2.0)
    assert bright[0, 0, 0] == pytest.approx(np.sqrt(0.25))
    gray = adjust_rgb(np.tile([0.8, 0.4, 0.2], (4, 4, 1)), saturation=0.0)
    # Zero saturation collapses channels onto the luminance.
    assert np.ptp(gray[0, 0]) < 1e-6


def test_resample_same_grid_passthrough():
    data = _frame()
    w = _tan_wcs()
    out = resample_to_grid(data, w, w, data.shape)
    np.testing.assert_array_equal(out, data)


def test_resample_shifted_grid_interpolates():
    data = _frame()
    w = _tan_wcs()
    shifted = _tan_wcs()
    shifted.wcs.crpix = [NX / 2.0 + 5.0, NY / 2.0]
    out = resample_to_grid(data, w, shifted, data.shape)
    # A 5 px shift moves source content 5 px in the output grid.
    np.testing.assert_allclose(out[:, 5:], data[:, :-5], atol=1e-6)


def test_resample_no_wcs_same_shape_warns_but_passes():
    data = _frame()
    out = resample_to_grid(data, None, None, data.shape)
    np.testing.assert_array_equal(out, data)


def test_resample_no_wcs_wrong_shape_fails():
    with pytest.raises(ValueError):
        resample_to_grid(_frame(), None, _tan_wcs(), (10, 10))


def test_combine_frames_median():
    a = np.full((5, 5), 10.0)
    b = np.full((5, 5), 12.0)
    c = np.full((5, 5), 20.0)
    assert combine_frames([a, b, c])[0, 0] == 12.0
    assert combine_frames([a])[0, 0] == 10.0


def test_generate_color_image_end_to_end(tmp_path):
    hdr = _tan_header()
    rng = np.random.default_rng(1)
    bands = {}
    for band, level in (("i", 120.0), ("r", 100.0), ("g", 90.0)):
        frames = []
        for k in range(2):
            data = level + rng.normal(0, 1.0, (NY, NX))
            frames.append((data, hdr.copy(), 100.0, 25.0))
        bands[band] = frames
    out_png = tmp_path / "Color_Image.png"
    written = generate_color_image(
        bands,
        str(out_png),
        w_eff={"g": 4830, "r": 6260, "i": 7670},
        save_fits=True,
        title="test",
    )
    assert written is not None
    assert out_png.exists() and out_png.stat().st_size > 0
    fits_path = tmp_path / "Color_Image.fits"
    assert fits_path.exists()
    with fits.open(fits_path) as hdul:
        assert hdul[0].data.shape == (3, NY, NX)


def test_generate_color_image_empty():
    assert generate_color_image({}, "unused.png") is None
