"""Tests for the adaptive WCS-solving helpers in wcs.py.

Covers the pure/unit-testable pieces added for the plate_solve
adaptivity work: input-WCS verification residuals, position-hint
ordering, wide scale-bounds plumbing, and the verify gate truth table.
No solve-field or astroquery calls are made here; _solved_pos_hint
tracking lives inside the solve loop and is exercised only through its
helper inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from autophot.pipeline.wcs import (
    _build_scale_args,
    _choose_position_hints,
    _coerce_detected_xy,
    _should_verify_input_wcs,
    _solve_field_parity,
    _wcs_cfg_float,
    verify_wcs_residuals,
)


def _tan_wcs(scale_arcsec=1.0, crval=(150.0, 30.0), npix=500):
    """Simple TAN WCS at a fixed pixel scale centered on ``crval``."""
    w = WCS(naxis=2)
    scale_deg = scale_arcsec / 3600.0
    w.wcs.crpix = [(npix + 1) / 2.0, (npix + 1) / 2.0]
    w.wcs.crval = [float(crval[0]), float(crval[1])]
    w.wcs.cd = np.array([[-scale_deg, 0.0], [0.0, scale_deg]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _tan_header(**kwargs):
    hdr = _tan_wcs(**kwargs).to_header()
    npix = 500
    hdr["NAXIS1"] = npix
    hdr["NAXIS2"] = npix
    return hdr


def _detected_grid(n=12, lo=60.0, hi=440.0):
    """Well-separated pixel positions covering a 500x500 frame."""
    rng = np.random.default_rng(42)
    return rng.uniform(lo, hi, size=(n, 2))


# ---------------------------------------------------------------------------
# verify_wcs_residuals
# ---------------------------------------------------------------------------


def test_verify_wcs_residuals_self_consistent_catalog():
    """Catalog built from the same WCS must give ~0 residuals."""
    w = _tan_wcs()
    xy = _detected_grid()
    ra, dec = w.all_pix2world(xy[:, 0], xy[:, 1], 0)
    # Extra catalog rows with no detection must not enter the match count.
    ra = np.append(ra, [10.0, 11.0, 12.0])
    dec = np.append(dec, [-40.0, -41.0, -42.0])
    med, p95, n = verify_wcs_residuals(
        w, xy[:, 0], xy[:, 1], ra, dec, match_radius_arcsec=2.0
    )
    assert n == len(xy)
    assert med < 0.01
    assert p95 < 0.01


def test_verify_wcs_residuals_offset_catalog():
    """A 5 arcsec catalog offset reports a ~5 arcsec median residual."""
    w = _tan_wcs()
    xy = _detected_grid()
    ra, dec = w.all_pix2world(xy[:, 0], xy[:, 1], 0)
    dec = dec + 5.0 / 3600.0
    med, p95, n = verify_wcs_residuals(
        w, xy[:, 0], xy[:, 1], ra, dec, match_radius_arcsec=10.0
    )
    assert n == len(xy)
    assert med == pytest.approx(5.0, abs=0.1)
    assert p95 == pytest.approx(5.0, abs=0.1)


def test_verify_wcs_residuals_no_match_returns_nan():
    """Radius tighter than the true offset must yield zero matches."""
    w = _tan_wcs()
    xy = _detected_grid()
    ra, dec = w.all_pix2world(xy[:, 0], xy[:, 1], 0)
    dec = dec + 60.0 / 3600.0
    med, p95, n = verify_wcs_residuals(
        w, xy[:, 0], xy[:, 1], ra, dec, match_radius_arcsec=2.0
    )
    assert n == 0
    assert np.isnan(med)
    assert np.isnan(p95)


def test_verify_wcs_residuals_degenerate_inputs():
    w = _tan_wcs()
    assert verify_wcs_residuals(None, [1], [2], [3], [4], 2.0) == (
        pytest.approx(np.nan, nan_ok=True),
        pytest.approx(np.nan, nan_ok=True),
        0,
    )
    med, p95, n = verify_wcs_residuals(w, [], [], [], [], 2.0)
    assert n == 0 and np.isnan(med) and np.isnan(p95)


# ---------------------------------------------------------------------------
# _choose_position_hints
# ---------------------------------------------------------------------------


def test_position_hints_header_first_when_preferred():
    hdr = _tan_header(crval=(150.0, 30.0))
    hints = _choose_position_hints(hdr, 200.0, -10.0, prefer_header=True)
    assert hints[0] == pytest.approx((150.0, 30.0))
    assert hints[1] == pytest.approx((200.0, -10.0))


def test_position_hints_target_first_when_not_preferred():
    hdr = _tan_header(crval=(150.0, 30.0))
    hints = _choose_position_hints(hdr, 200.0, -10.0, prefer_header=False)
    assert hints[0] == pytest.approx((200.0, -10.0))
    assert hints[1] == pytest.approx((150.0, 30.0))


def test_position_hints_invalid_header_falls_back_to_target():
    hdr = fits.Header()  # no WCS keywords
    hints = _choose_position_hints(hdr, 200.0, -10.0, prefer_header=True)
    assert hints == [pytest.approx((200.0, -10.0))]


def test_position_hints_dedupe():
    """Centers inside the dedupe tolerance collapse to a single hint."""
    hdr = _tan_header(crval=(150.0, 30.0))
    hints = _choose_position_hints(hdr, 150.0 + 0.2 / 3600.0, 30.0, prefer_header=True)
    assert len(hints) == 1


def test_position_hints_no_inputs():
    assert _choose_position_hints(fits.Header(), None, None) == []


def test_position_hint_uses_cropped_image_center():
    hdr = _tan_header()
    hdr["CRPIX1"] = 1180.5
    hdr["CRPIX2"] = -936.5
    hdr["NAXIS1"] = 751
    hdr["NAXIS2"] = 618
    expected = WCS(hdr).all_pix2world([[376.0, 309.5]], 1)[0]
    hints = _choose_position_hints(hdr, None, None)
    assert hints[0] == pytest.approx(expected)


@pytest.mark.parametrize("representation", ["cd", "pc"])
@pytest.mark.parametrize("sign, expected", [(-1, "pos"), (1, "neg")])
def test_solver_parity_uses_astrometry_convention(representation, sign, expected):
    hdr = fits.Header()
    scale = 1.0 / 3600.0
    if representation == "cd":
        hdr["CD1_1"] = sign * scale
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = scale
    else:
        hdr["CDELT1"] = sign * scale
        hdr["CDELT2"] = scale
    assert _solve_field_parity({}, hdr) == expected


@pytest.mark.parametrize("setting", ["pos", "neg"])
def test_solver_parity_preserves_explicit_override(setting):
    assert (
        _solve_field_parity({"solve_field_parity": setting}, fits.Header()) == setting
    )


def test_solver_parity_omits_unusable_header():
    assert _solve_field_parity({}, fits.Header()) is None


# ---------------------------------------------------------------------------
# _build_scale_args / wide bounds config
# ---------------------------------------------------------------------------


def test_build_scale_args_bounds_and_position():
    args = _build_scale_args(
        0.5, 8.0, include_pos=True, ra=10.0, dec=-20.0, radius_deg=0.5
    )
    assert args[args.index("--scale-low") + 1] == "0.5"
    assert args[args.index("--scale-high") + 1] == "8.0"
    assert args[args.index("--ra") + 1] == "10.0"
    assert args[args.index("--dec") + 1] == "-20.0"
    assert args[args.index("--radius") + 1] == "0.5"


def test_build_scale_args_scale_only():
    args = _build_scale_args(0.1, 5.0, include_pos=False)
    assert "--ra" not in args and "--dec" not in args
    assert "--radius" not in args


def test_build_scale_args_sigma():
    args = _build_scale_args(0.1, 5.0, include_pos=False, background_std=12.5)
    assert args[args.index("--sigma") + 1] == "12.5"


def test_wide_scale_cfg_values_feed_bounds():
    """Config values for the wide bounds must reach the args verbatim."""
    cfg = {"wide_scale_low_arcsec": 0.2, "wide_scale_high_arcsec": 9.0}
    low = _wcs_cfg_float(cfg, "wide_scale_low_arcsec", 0.1)
    high = _wcs_cfg_float(cfg, "wide_scale_high_arcsec", 5.0)
    args = _build_scale_args(low, high, include_pos=False)
    assert args[args.index("--scale-low") + 1] == "0.2"
    assert args[args.index("--scale-high") + 1] == "9.0"


def test_wide_scale_cfg_defaults_and_null():
    assert _wcs_cfg_float({}, "wide_scale_low_arcsec", 0.1) == 0.1
    cfg = {"wide_scale_high_arcsec": None}
    assert _wcs_cfg_float(cfg, "wide_scale_high_arcsec", 5.0) == 5.0


# ---------------------------------------------------------------------------
# _should_verify_input_wcs gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "redo,flag,has_wcs,n_det,min_m,expected",
    [
        (True, True, True, 20, 8, False),  # redo_wcs always solves
        (False, True, True, 20, 8, True),
        (False, False, True, 20, 8, False),  # opt-in flag off
        (False, True, False, 20, 8, False),  # no header WCS
        (False, True, True, 5, 8, False),  # too few detections
        (False, True, True, None, 8, False),  # no detections at all
        (False, True, True, 8, 8, True),  # boundary count passes
    ],
)
def test_should_verify_input_wcs_truth_table(
    redo, flag, has_wcs, n_det, min_m, expected
):
    assert _should_verify_input_wcs(redo, flag, has_wcs, n_det, min_m) == expected


# ---------------------------------------------------------------------------
# _coerce_detected_xy
# ---------------------------------------------------------------------------


def test_coerce_detected_xy_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 5.0]])
    out = _coerce_detected_xy(arr)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 4.0]])


def test_coerce_detected_xy_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x_pix": [1.0, 2.0], "y_pix": [3.0, 4.0], "flux": [10.0, 20.0]})
    out = _coerce_detected_xy(df)
    np.testing.assert_allclose(out, [[1.0, 3.0], [2.0, 4.0]])


def test_coerce_detected_xy_bad_inputs():
    assert _coerce_detected_xy(None) is None
    assert _coerce_detected_xy(np.array([1.0, 2.0, 3.0])) is None
    pd = pytest.importorskip("pandas")
    assert _coerce_detected_xy(pd.DataFrame({"a": [1.0]})) is None
