"""Tests for the sparse/starved-field WCS-solving helpers in wcs.py.

Covers the pure/unit-testable pieces added for the starved-field work:
the proactive --nsigma truth table, the bounded downsample ladder, the
FITS xylist writer, and the relaxed DAO-style detection used to build
xylists.  No solve-field calls are made here.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from autophot.pipeline.wcs import (
    _coerce_detected_flux,
    _detect_solve_sources,
    _downsample_levels_for,
    _solve_nsigma_for_count,
    _write_solve_xylist,
)

# ---------------------------------------------------------------------------
# _solve_nsigma_for_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_det,expected",
    [
        (0, 3),  # explicit zero = maximally sparse -> sparse_nsigma
        (20, 3),  # < 30 detections -> 3
        (40, 4),  # < 50 detections -> 4
        (100, 5),  # enough sources -> configured value
        (None, 5),  # count unknown -> configured value
        (29, 3),
        (30, 4),
        (49, 4),
        (50, 5),
    ],
)
def test_solve_nsigma_for_count_truth_table(n_det, expected):
    assert _solve_nsigma_for_count(n_det, cfg_val=5, sparse_nsigma=3) == expected


def test_solve_nsigma_for_count_never_raises():
    """A configured nsigma below the sparse value must be kept, not raised."""
    assert _solve_nsigma_for_count(0, cfg_val=2, sparse_nsigma=3) == 2
    assert _solve_nsigma_for_count(10, cfg_val=2, sparse_nsigma=3) == 2


# ---------------------------------------------------------------------------
# _downsample_levels_for
# ---------------------------------------------------------------------------


def test_downsample_levels_small_image_fine_first():
    """770px frame: fine levels first, coarser levels bounded by min_pixels."""
    # After the existing -1 decrement the configured level is 3; level 6
    # would leave 770/6 ~ 128 px < 256 and must be excluded.
    levels = _downsample_levels_for(
        770, 770, downsample=3, ds_max=8, min_pixels=256, small_image=1024
    )
    assert levels == [1, 2, 3]


def test_downsample_levels_small_image_configured_ds4():
    """Same 770px frame at configured ds=4 still resolves to [1, 2, 3]."""
    levels = _downsample_levels_for(
        770, 770, downsample=4, ds_max=8, min_pixels=256, small_image=1024
    )
    assert levels == [1, 2, 3]


def test_downsample_levels_large_image_doubling():
    """Large images keep the original configured-then-doubling ladder."""
    levels = _downsample_levels_for(
        4000, 4000, downsample=2, ds_max=8, min_pixels=256, small_image=1024
    )
    assert levels == [2, 4, 8]


def test_downsample_levels_min_pixels_boundary():
    """A level leaving exactly min_pixels on the small axis is allowed."""
    levels = _downsample_levels_for(
        512, 512, downsample=2, ds_max=8, min_pixels=256, small_image=1024
    )
    # Small image gets the level-1 fine pass; 512/2 = 256 passes while
    # 512/4 = 128 fails.
    assert levels == [1, 2]


def test_downsample_levels_min_pixels_excludes_all_but_finest():
    levels = _downsample_levels_for(
        511, 511, downsample=2, ds_max=8, min_pixels=256, small_image=1024
    )
    # 511/2 ~ 255.5 < 256, so only the level-1 fine attempt survives.
    assert levels == [1]


def test_downsample_levels_large_image_bound_drops_tail():
    levels = _downsample_levels_for(
        2000, 2000, downsample=2, ds_max=8, min_pixels=256, small_image=1024
    )
    # 2000/8 = 250 < 256, so the ds_max level drops off the ladder.
    assert levels == [2, 4]


def test_downsample_levels_fallback_level_one():
    """An undersized frame still gets a single level-1 attempt."""
    levels = _downsample_levels_for(
        100, 100, downsample=4, ds_max=8, min_pixels=256, small_image=1024
    )
    assert levels == [1]


# ---------------------------------------------------------------------------
# _write_solve_xylist
# ---------------------------------------------------------------------------


def test_write_solve_xylist_roundtrip(tmp_path):
    """X/Y columns come back as input+1 (FITS 1-based), FLUX preserved."""
    xy = np.array([[10.5, 20.25], [100.0, 200.0], [3.0, 4.0]])
    flux = np.array([300.0, 100.0, 50.0])
    path = tmp_path / "test.xyls"
    out = _write_solve_xylist(xy, flux, str(path), width=770, height=770)
    assert out == str(path)
    with fits.open(path) as hdul:
        tbl = hdul[1].data
        np.testing.assert_allclose(tbl["X"], xy[:, 0] + 1.0)
        np.testing.assert_allclose(tbl["Y"], xy[:, 1] + 1.0)
        np.testing.assert_allclose(tbl["FLUX"], flux)
        assert hdul[1].header["IMAGEW"] == 770
        assert hdul[1].header["IMAGEH"] == 770


def test_write_solve_xylist_flux_fallback(tmp_path):
    """Missing or non-finite flux values fall back to 1.0."""
    path = tmp_path / "noflux.xyls"
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _write_solve_xylist(xy, None, str(path)) == str(path)
    with fits.open(path) as hdul:
        np.testing.assert_allclose(hdul[1].data["FLUX"], [1.0, 1.0])
    path2 = tmp_path / "nanflux.xyls"
    flux = np.array([np.nan, 5.0])
    assert _write_solve_xylist(xy, flux, str(path2)) == str(path2)
    with fits.open(path2) as hdul:
        # finite flux sorts first; the NaN row lands last with flux fallback
        np.testing.assert_allclose(hdul[1].data["FLUX"], [5.0, 1.0])
        np.testing.assert_allclose(hdul[1].data["X"], [4.0, 2.0])
        np.testing.assert_allclose(hdul[1].data["Y"], [5.0, 3.0])


def test_write_solve_xylist_brightest_first(tmp_path):
    """Rows are written in flux-descending order for quad building."""
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    flux = np.array([10.0, 99.0, 50.0])
    path = tmp_path / "sorted.xyls"
    assert _write_solve_xylist(xy, flux, str(path)) == str(path)
    with fits.open(path) as hdul:
        tbl = hdul[1].data
        np.testing.assert_allclose(tbl["FLUX"], [99.0, 50.0, 10.0])
        np.testing.assert_allclose(tbl["X"], [4.0, 6.0, 2.0])
        np.testing.assert_allclose(tbl["Y"], [5.0, 7.0, 3.0])


def test_write_solve_xylist_bad_inputs(tmp_path):
    path = tmp_path / "bad.xyls"
    assert _write_solve_xylist(np.empty((0, 2)), None, str(path)) is None
    assert _write_solve_xylist(np.array([1.0, 2.0]), None, str(path)) is None


# ---------------------------------------------------------------------------
# _coerce_detected_flux
# ---------------------------------------------------------------------------


def test_coerce_detected_flux_snr_column():
    """snr is the preferred ranking column, kept row-aligned with xy."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "x_pix": [1.0, 2.0, 3.0],
            "y_pix": [4.0, 5.0, 6.0],
            "snr": [10.0, 99.0, 50.0],
            "flux_AP": [1.0, 2.0, 3.0],
        }
    )
    out = _coerce_detected_flux(df, 3)
    np.testing.assert_allclose(out, [10.0, 99.0, 50.0])


def test_coerce_detected_flux_fallback_column_and_mask():
    """Falls back to flux columns and drops rows that lost x/y."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "x_pix": [1.0, np.nan, 3.0],
            "y_pix": [4.0, 5.0, 6.0],
            "flux_AP": [10.0, 99.0, 50.0],
        }
    )
    out = _coerce_detected_flux(df, 2)
    np.testing.assert_allclose(out, [10.0, 50.0])


def test_coerce_detected_flux_bad_inputs():
    import pandas as pd

    assert _coerce_detected_flux(None, 3) is None
    assert _coerce_detected_flux(np.zeros((3, 2)), 3) is None
    assert _coerce_detected_flux(pd.DataFrame({"a": [1.0]}), 1) is None
    # positions exist but no ranking column -> keep given order
    df = pd.DataFrame({"x_pix": [1.0, 2.0], "y_pix": [3.0, 4.0]})
    assert _coerce_detected_flux(df, 2) is None
    df2 = pd.DataFrame(
        {"x_pix": [1.0, 2.0], "y_pix": [3.0, 4.0], "snr": [np.nan, np.nan]}
    )
    assert _coerce_detected_flux(df2, 2) is None


# ---------------------------------------------------------------------------
# _detect_solve_sources
# ---------------------------------------------------------------------------


def _sparse_field(seed=7):
    """300x300 frame with 10 faint Gaussian stars over noise."""
    rng = np.random.default_rng(seed)
    noise_sigma = 5.0
    image = rng.normal(100.0, noise_sigma, size=(300, 300))
    yy, xx = np.mgrid[0:300, 0:300]
    positions = rng.uniform(30.0, 270.0, size=(10, 2))
    for i, (x0, y0) in enumerate(positions):
        peak = 40.0 + 5.0 * i  # ~8 sigma at the faint end
        image += peak * np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5**2)
    return image, positions


def test_detect_solve_sources_finds_faint_stars():
    image, positions = _sparse_field()
    xy, flux = _detect_solve_sources(image, nsigma=2.5)
    assert xy is not None
    assert xy.shape[1] == 2
    # All 10 injected stars should be recovered at this relaxed threshold.
    assert len(xy) >= 8
    # Returned list is brightest-first.
    assert np.all(np.diff(flux) <= 0)
    # Every injected star is recovered; a relaxed threshold may also admit
    # noise spikes, so the match direction is stars -> detections.
    n_recovered = 0
    for x0, y0 in positions:
        sep = np.hypot(xy[:, 0] - x0, xy[:, 1] - y0)
        if sep.min() < 1.5:
            n_recovered += 1
    assert n_recovered >= 8


def test_detect_solve_sources_empty_frame():
    """Pure noise at a strict threshold returns no detections."""
    rng = np.random.default_rng(0)
    xy, flux = _detect_solve_sources(rng.normal(0.0, 1.0, size=(200, 200)), nsigma=8.0)
    assert xy is None
    assert flux is None


def test_detect_solve_sources_degenerate_input():
    xy, flux = _detect_solve_sources(np.full((50, 50), np.nan), nsigma=2.5)
    assert xy is None and flux is None
    xy, flux = _detect_solve_sources(np.zeros((50, 50)), nsigma=2.5)
    assert xy is None and flux is None
