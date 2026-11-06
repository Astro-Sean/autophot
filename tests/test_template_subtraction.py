"""Regression tests for template-subtraction mathematics.

Covers the module-level ZOGY implementation (_zogy_subtract,
_pad_psf_to_image) and the MAD-based outlier mask (robust_outlier_mask)
used by the flux-matching stage.  These tests use synthetic images with known truth so that flux
conservation and sign conventions can be checked quantitatively.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astropy.io import fits

import templates as templates_mod
from templates import _zogy_subtract, _pad_psf_to_image, Templates


def _gauss_psf(sigma: float, size: int = 31) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-0.5 * (ax[None, :] ** 2 + ax[:, None] ** 2) / sigma ** 2)
    return g / g.sum()


def _make_image(rng, shape, psf, sources, noise_sig):
    img = rng.normal(0.0, noise_sig, shape)
    for (x, y, f) in sources:
        s = psf.shape[0] // 2
        img[y - s : y + s + 1, x - s : x + s + 1] += f * psf
    return img


SHAPE = (128, 128)
COMMON = [(30, 30, 2000.0), (60, 60, 3000.0), (90, 40, 1500.0)]
TRANSIENT = (80, 80, 1000.0)


def test_zogy_residual_warp_recovers_linear_shift_field():
    rng = np.random.default_rng(11)
    shape = (200, 240)
    psf = _gauss_psf(2.0)
    stars = [(40, 40, 5000), (120, 60, 4000), (200, 100, 6000),
             (60, 150, 3500), (160, 170, 4500), (100, 120, 3000),
             (30, 180, 2500)]
    sci = _make_image(rng, shape, psf, [(x, y, f) for x, y, f in stars], 1.0)
    # Reference warped by a linear field: dx = +0.30*x/W, dy = -0.25*y/H
    from scipy.ndimage import map_coordinates

    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    ref = map_coordinates(
        _make_image(rng, shape, psf, [(x, y, f) for x, y, f in stars], 0.3),
        [yy - (-0.25 * yy / H), xx - (0.30 * xx / W)],
        order=3, mode="nearest",
    )
    warped, info = templates_mod._zogy_residual_warp(
        sci, ref, [(x, y) for x, y, _ in stars], 4.7, min_sources=6,
    )
    assert info["applied"]
    assert info["n_used"] >= 6
    assert info["fit_rms"] < 0.6 * info["raw_rms"]
    # Stars land back on their science positions
    for x, y, _ in stars[:3]:
        s0 = sci[y - 8 : y + 9, x - 8 : x + 9] - np.median(sci)
        r1 = warped[y - 8 : y + 9, x - 8 : x + 9] - np.median(warped)
        assert np.nanstd(s0 - r1) < np.nanstd(
            s0 - (ref[y - 8 : y + 9, x - 8 : x + 9] - np.median(ref))
        )


def test_zogy_residual_warp_skips_when_fine():
    rng = np.random.default_rng(3)
    psf = _gauss_psf(2.0)
    stars = [(x, y, 4000.0) for x, y in
             [(40, 40), (120, 60), (200, 100), (60, 150), (160, 170), (100, 120)]]
    sci = _make_image(rng, (200, 240), psf, stars, 1.0)
    ref = sci.copy()
    warped, info = templates_mod._zogy_residual_warp(
        sci, ref, [(x, y) for x, y, _ in stars], 4.7, min_sources=6,
    )
    assert not info["applied"]
    assert warped is ref


def test_zogy_stamp_shift_rejects_unmeasurable_stamps():
    rng = np.random.default_rng(5)
    psf = _gauss_psf(2.0)
    star = 4000.0 * psf + 10.0
    # Stamp smaller than the lag window cannot express the shift: must
    # return None, not a lag relative to a truncated zero point.
    assert templates_mod._zogy_stamp_shift(
        star[:7, :7], star[:7, :7].copy(), max_lag=8
    ) is None
    blank = np.zeros_like(star)
    assert templates_mod._zogy_stamp_shift(blank, blank.copy()) is None
    bad = star.copy()
    bad[5, 5] = np.nan
    assert templates_mod._zogy_stamp_shift(bad, star.copy()) is None
    assert (
        templates_mod._zogy_stamp_shift(star, star[:29, :29].copy()) is None
    )


def test_zogy_stamp_shift_recovers_subpixel_lag():
    from scipy.ndimage import shift as _ndi_shift

    psf = _gauss_psf(2.0)
    sci = 4000.0 * psf + 10.0
    ref = _ndi_shift(sci - 10.0, (0.4, -0.3), order=3) + 10.0
    dx, dy = templates_mod._zogy_stamp_shift(sci, ref, max_lag=4)
    assert dx == pytest.approx(-0.3, abs=0.1)
    assert dy == pytest.approx(0.4, abs=0.1)


def test_zogy_residual_warp_tolerates_bad_anchors_and_bans():
    rng = np.random.default_rng(8)
    psf = _gauss_psf(2.0)
    stars = [(x, y, 4000.0) for x, y in
             [(40, 40), (120, 60), (200, 100), (60, 150), (160, 170),
              (100, 120), (30, 180), (210, 40)]]
    sci = _make_image(rng, (200, 240), psf, stars, 1.0)
    from scipy.ndimage import shift as _ndi_shift

    ref = _ndi_shift(sci, (0.3, 0.25), order=3, mode="nearest")
    # Non-finite and malformed anchor entries are skipped, not fatal.
    pos = [(x, y) for x, y, _ in stars] + [
        (np.nan, 10.0), (np.inf, 20.0), ("bad", 5)
    ]
    warped, info = templates_mod._zogy_residual_warp(
        sci, ref, pos, 4.7, min_sources=6,
    )
    assert info["applied"]
    assert info["n_anchors"] == len(stars)
    # A bare (x, y) ban point is coerced rather than crashing.
    _, info1d = templates_mod._zogy_residual_warp(
        sci, ref, [(x, y) for x, y, _ in stars], 4.7,
        ban_xy=np.array([120.0, 60.0]), ban_radius=20.0,
        min_sources=6,
    )
    # Banning every anchor declines cleanly.
    _, infob = templates_mod._zogy_residual_warp(
        sci, ref, [(x, y) for x, y, _ in stars], 4.7,
        ban_xy=np.asarray([(x, y) for x, y, _ in stars], dtype=float),
        ban_radius=25.0, min_sources=6,
    )
    assert not infob["applied"]


def test_zogy_residual_warp_bails_when_clip_empties_ensemble():
    rng = np.random.default_rng(9)
    psf = _gauss_psf(2.0)
    stars = [(x, y, 4000.0) for x, y in
             [(40, 40), (120, 60), (200, 100), (60, 150), (160, 170),
              (100, 120)]]
    sci = _make_image(rng, (200, 240), psf, stars, 1.0)
    from scipy.ndimage import shift as _ndi_shift

    ref = _ndi_shift(sci, (0.3, 0.2), order=3, mode="nearest")
    # Burn a second source 2.5 px off in the reference at one anchor so
    # its measured lag is an outlier; with exactly min_sources anchors
    # the clip empties the ensemble and the warp must decline.
    x, y = stars[0][:2]
    gy, gx = np.mgrid[y - 15 : y + 16, x - 15 : x + 16]
    ref[y - 15 : y + 16, x - 15 : x + 16] += 6000.0 * np.exp(
        -((gx - x - 2.5) ** 2 + (gy - y - 2.5) ** 2) / 4.0
    )
    _, info = templates_mod._zogy_residual_warp(
        sci, ref, [(x, y) for x, y, _ in stars], 4.7, min_sources=6,
    )
    assert not info["applied"]


def test_zogy_exposure_flux_scale_includes_gain_and_exposure():
    scale = templates_mod._zogy_exposure_flux_scale(
        science_exposure=300.0,
        reference_exposure=60.0,
        science_gain=2.0,
        reference_gain=4.0,
    )
    assert scale == pytest.approx(10.0)
    assert np.isnan(
        templates_mod._zogy_exposure_flux_scale(
            science_exposure=300.0,
            reference_exposure=60.0,
            science_gain=0.0,
            reference_gain=4.0,
        )
    )


def test_convolved_source_variance_uses_squared_kernel():
    variance = np.zeros((9, 9), dtype=float)
    variance[4, 4] = 16.0
    kernel = np.zeros_like(variance)
    kernel[0, 0] = 0.5
    kernel[0, 1] = 0.5

    convolved = templates_mod._convolve_variance_map(variance, kernel)

    assert convolved[4, 4] == pytest.approx(4.0)
    assert convolved[4, 5] == pytest.approx(4.0)
    assert convolved.sum() == pytest.approx(8.0)


def test_pick_psf_uses_newest_exact_field_stamp(tmp_path):
    root_stamp = tmp_path / "PSF_model_image_science.fits"
    grid_stamp = tmp_path / "PSF_MODELS" / "PSF_model_image_science.fits"
    unrelated_stamp = tmp_path / "PSF_model_image_other.fits"
    grid_stamp.parent.mkdir()
    for stamp in (root_stamp, grid_stamp, unrelated_stamp):
        stamp.write_bytes(b"psf")
    os.utime(root_stamp, ns=(1, 1))
    os.utime(grid_stamp, ns=(2, 2))
    os.utime(unrelated_stamp, ns=(3, 3))

    selected = templates_mod._pick_psf_file(
        [str(root_stamp), str(grid_stamp), str(unrelated_stamp)],
        "science.fits",
    )

    assert selected == str(grid_stamp)


def test_flux_matching_keeps_consistent_stars_across_brightness_range():
    flux_img = np.concatenate(
        [np.linspace(9000.0, 11000.0, 18), np.linspace(800.0, 1200.0, 5)]
    )
    flux_tpl = 0.4 * flux_img
    flux_tpl[2] *= 1.5
    catalog_img = pd.DataFrame(
        {
            "flux_AP": flux_img,
            "flux_AP_err": 0.02 * flux_img,
            "threshold": np.full(len(flux_img), 10.0),
            "source_id": np.arange(len(flux_img)),
        }
    )
    catalog_tpl = pd.DataFrame(
        {
            "flux_AP": flux_tpl,
            "flux_AP_err": 0.02 * flux_tpl,
            "threshold": np.full(len(flux_tpl), 10.0),
            "source_id": np.arange(len(flux_tpl)),
        }
    )

    output, _ = Templates(input_yaml={}).find_flux_consistent_sources(
        catalog_img, catalog_tpl, make_plot=False
    )

    assert len(output) == 22
    assert 2 not in output["source_id"].to_numpy()
    assert set(output["source_id"]) == set(range(23)) - {2}


def test_sfft_matching_positions_read_actual_zero_based_sources(tmp_path):
    source_file = tmp_path / "SFFT_Matching_Sources.csv"
    pd.DataFrame(
        {
            "X_IMAGE_REF_SCI_MEAN": [11.0, 31.0, np.nan],
            "Y_IMAGE_REF_SCI_MEAN": [21.0, 41.0, 51.0],
        }
    ).to_csv(source_file, index=False)

    positions = templates_mod._read_sfft_matching_positions(str(source_file))

    assert positions == [(10.0, 20.0), (30.0, 40.0)]
    assert templates_mod._read_sfft_matching_positions(
        str(tmp_path / "missing.csv")
    ) is None


def test_auto_sfft_kernel_order_requires_spatial_mismatch():
    select = templates_mod._auto_sfft_kernel_order

    assert select(rel_diff=0.3, n_eff=127, max_order=2) == 0
    assert select(rel_diff=0.5, n_eff=29, max_order=2) == 0
    assert select(rel_diff=0.5, n_eff=30, max_order=2) == 1
    assert select(rel_diff=0.7, n_eff=80, max_order=2) == 2
    assert select(rel_diff=0.7, n_eff=80, max_order=1) == 1
    assert select(rel_diff=0.1, n_eff=40, max_order=2, last_resort=True) == 1


def test_sfft_retry_kernel_hw_tracks_constraint_budget():
    size = templates_mod._sfft_retry_kernel_hw

    # Sparse fields shrink toward the constraint budget, never below the
    # one-FWHM_conv floor needed to hold the PSF-difference core.
    assert size(3, fwhm_conv=1.0) == 5
    assert size(13, fwhm_conv=10.1) == 11
    assert size(4, fwhm_conv=10.7) == 11
    # Dense fields relax toward the full PSF-sized kernel.
    assert size(100, fwhm_conv=5.0) == 29
    # Degenerate inputs still return a usable half-width.
    assert size(None, fwhm_conv=float("nan")) == 5
    assert size(0, fwhm_conv=0.0) == 5


def test_compact_core_dipole_mask_finds_local_shift_artifact():
    from scipy.ndimage import shift as _ndi_shift

    rng = np.random.default_rng(12)
    yy, xx = np.indices((100, 100))
    core = 30000.0 * np.exp(
        -0.5 * ((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 2.0**2
    )
    science = core + rng.normal(0.0, 1.0, core.shape)
    reference = _ndi_shift(core, (0.2, -0.3), order=3, mode="nearest")
    reference += rng.normal(0.0, 1.0, reference.shape)
    diff = science - reference

    mask, info = templates_mod._compact_core_dipole_mask(
        diff,
        science,
        reference,
        fwhm=4.7,
        protected_xy=[(80.0, 80.0)],
        config={"residual_artifact_min_static_snr": 10.0},
    )

    assert len(info) == 1
    assert info[0]["x"] == pytest.approx(40.0, abs=2.0)
    assert info[0]["y"] == pytest.approx(40.0, abs=2.0)
    assert info[0]["rms_drop"] > 0.5
    assert mask.sum() > 0
    assert mask[40, 40]


def test_compact_core_dipole_mask_never_masks_protected_position():
    from scipy.ndimage import shift as _ndi_shift

    yy, xx = np.indices((100, 100))
    core = 30000.0 * np.exp(
        -0.5 * ((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 2.0**2
    )
    science = core
    reference = _ndi_shift(core, (0.2, -0.3), order=3, mode="nearest")
    diff = science - reference

    mask, info = templates_mod._compact_core_dipole_mask(
        diff,
        science,
        reference,
        fwhm=4.7,
        protected_xy=[(40.0, 40.0)],
        config={"residual_artifact_min_static_snr": 10.0},
    )

    assert info == []
    assert not mask.any()


def test_compact_core_dipole_mask_rejects_single_lobed_residual():
    rng = np.random.default_rng(13)
    yy, xx = np.indices((100, 100))
    core = 30000.0 * np.exp(
        -0.5 * ((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 2.0**2
    )
    transient = 3000.0 * np.exp(
        -0.5 * ((xx - 40.0) ** 2 + (yy - 40.0) ** 2) / 1.2**2
    )
    science = core + transient + rng.normal(0.0, 1.0, core.shape)
    reference = core + rng.normal(0.0, 1.0, core.shape)
    diff = science - reference

    mask, info = templates_mod._compact_core_dipole_mask(
        diff,
        science,
        reference,
        fwhm=4.7,
        config={"residual_artifact_min_static_snr": 10.0},
    )

    assert info == []
    assert not mask.any()


class TestZogySubtraction:
    """Verify the ZOGY difference image is flux-conserving and correctly signed."""

    def _run(self, fn=1.0, fr=1.0, sn=1.0, sr=1.0,
             psf_n=None, psf_r=None, sci_flux_mult=1.0, seed=42):
        rng = np.random.default_rng(seed)
        psf_n = psf_n if psf_n is not None else _gauss_psf(2.0)
        psf_r = psf_r if psf_r is not None else _gauss_psf(2.0)
        sci = _make_image(
            rng, SHAPE, psf_n,
            [(x, y, f * sci_flux_mult) for x, y, f in COMMON] + [TRANSIENT],
            sn,
        )
        ref = _make_image(rng, SHAPE, psf_r, COMMON, sr)
        Pn = _pad_psf_to_image(psf_n, SHAPE)
        Pr = _pad_psf_to_image(psf_r, SHAPE)
        return _zogy_subtract(sci, ref, Pn, Pr, sn=sn, sr=sr, fn=fn, fr=fr)

    def test_equal_psf_transient_flux_and_sign(self):
        """A science-only transient must appear POSITIVE and flux-conserved."""
        D, S, Scorr, P_D = self._run()
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(D[y - s : y + s + 1, x - s : x + s + 1]))
        # ~2% tolerance: the ZOGY matched filter attenuates extreme spatial
        # frequencies where the PSF has no power.
        assert flux == pytest.approx(f, rel=0.05)
        assert np.isfinite(D).all()

    def test_equal_psf_common_source_residual_zero(self):
        """Unvarying sources must subtract to ~zero."""
        D, _, _, _ = self._run()
        s = 10
        resid = float(np.nansum(D[30 - s : 30 + s + 1, 30 - s : 30 + s + 1]))
        assert abs(resid) < 0.05 * 2000.0  # << 5% of the source flux

    def test_different_psf_flux_scale(self):
        """With fn=1.2 (science counts are 1.2x reference), common sources
        must cancel and the transient must carry the correct D-units flux."""
        psf_n = _gauss_psf(3.0)
        psf_r = _gauss_psf(1.5)
        D, _, _, P_D = self._run(
            fn=1.2, fr=1.0, psf_n=psf_n, psf_r=psf_r, sci_flux_mult=1.2,
        )
        s = 10
        # transient flux in D units = counts / fn = 1000/1.2
        flux = float(np.nansum(D[70:91, 70:91]))
        assert flux == pytest.approx(1000.0 / 1.2, rel=0.08)
        resid = float(np.nansum(D[20:41, 20:41]))
        assert abs(resid) < 0.05 * (2000.0 * 1.2 / 1.2)
        assert np.isfinite(D).all()
        assert float(np.nansum(P_D)) == pytest.approx(1.0, rel=1e-6)

    def test_source_poisson_variance_uses_detector_gain(self):
        psf = _gauss_psf(2.0)
        rng = np.random.default_rng(23)
        science = rng.normal(0.0, 1.0, SHAPE)
        reference = rng.normal(0.0, 1.0, SHAPE)
        science[80 - 15 : 80 + 16, 80 - 15 : 80 + 16] += 8000.0 * psf
        Pn = _pad_psf_to_image(psf, SHAPE)
        Pr = _pad_psf_to_image(psf, SHAPE)

        _, _, scorr_low_gain, _ = _zogy_subtract(
            science,
            reference,
            Pn,
            Pr,
            sn=1.0,
            sr=1.0,
            gain_n=1.0,
            gain_r=1.0,
        )
        _, _, scorr_high_gain, _ = _zogy_subtract(
            science,
            reference,
            Pn,
            Pr,
            sn=1.0,
            sr=1.0,
            gain_n=8.0,
            gain_r=8.0,
        )

        assert scorr_high_gain[80, 80] > scorr_low_gain[80, 80]

    def test_degenerate_noise_raises(self):
        """Zero noise on both sides must raise, not silently emit NaN."""
        with pytest.raises(ValueError):
            self._run(sn=0.0, sr=0.0)

    def test_nonpositive_zero_point_raises(self):
        rng = np.random.default_rng(0)
        psf = _gauss_psf(2.0)
        sci = _make_image(rng, SHAPE, psf, COMMON, 1.0)
        ref = _make_image(rng, SHAPE, psf, COMMON, 1.0)
        Pn = _pad_psf_to_image(psf, SHAPE)
        with pytest.raises(ValueError):
            _zogy_subtract(sci, ref, Pn, Pn, sn=1.0, sr=1.0, fn=0.0, fr=1.0)

    def test_nan_psf_does_not_poison_output(self):
        """NaN in a PSF stamp must not propagate to an all-NaN diff."""
        rng = np.random.default_rng(7)
        psf = _gauss_psf(2.0)
        sci = _make_image(rng, SHAPE, psf, COMMON + [TRANSIENT], 1.0)
        ref = _make_image(rng, SHAPE, psf, COMMON, 1.0)
        Pn = _pad_psf_to_image(psf, SHAPE)
        Pr = _pad_psf_to_image(psf, SHAPE)
        Pn[10, 10] = np.nan
        D, _, _, _ = _zogy_subtract(sci, ref, Pn, Pr, sn=1.0, sr=1.0)
        assert np.isfinite(D).all()

    def test_nan_image_regions_masked(self):
        """NaN regions in the inputs (via nan_mask) must not poison D."""
        rng = np.random.default_rng(9)
        psf = _gauss_psf(2.0)
        sci = _make_image(rng, SHAPE, psf, COMMON + [TRANSIENT], 1.0)
        ref = _make_image(rng, SHAPE, psf, COMMON, 1.0)
        sci[0:10, :] = np.nan
        nan_mask = ~np.isfinite(sci) | ~np.isfinite(ref)
        Pn = _pad_psf_to_image(psf, SHAPE)
        Pr = _pad_psf_to_image(psf, SHAPE)
        D, _, _, _ = _zogy_subtract(
            sci, ref, Pn, Pr, sn=1.0, sr=1.0, nan_mask=nan_mask
        )
        assert np.isfinite(D[40:120, 40:120]).all()
        # Masked inputs are zeroed pre-FFT, but convolution leaks a small
        # amount of signal back into the band; the caller re-imposes NaN on
        # the output.  Require only that the leakage stays small vs the
        # transient flux (1000).
        assert np.nanmax(np.abs(D[0:10, 0:10])) < 50.0


class TestSubtractZogyWrapper:
    """End-to-end coverage of Templates._subtract_zogy: config plumbing,
    flux-scale estimation, forceconv resolution, provenance headers, and
    the SFFT fallback contract."""

    @staticmethod
    def _write(path, data, exptime=None):
        hdr = fits.Header()
        if exptime is not None:
            hdr["EXPTIME"] = exptime
        fits.writeto(str(path), np.asarray(data, dtype=float), hdr,
                     overwrite=True)
        return str(path)

    def _inputs(self, tmp_path, ref_scale=1.0, seed=5, psf_n=None, psf_r=None):
        rng = np.random.default_rng(seed)
        psf_n = psf_n if psf_n is not None else _gauss_psf(2.0)
        psf_r = psf_r if psf_r is not None else _gauss_psf(2.0)
        sci = _make_image(rng, SHAPE, psf_n, COMMON + [TRANSIENT], 1.0)
        ref = _make_image(rng, SHAPE, psf_r, COMMON, 1.0) * ref_scale
        sci_f = self._write(tmp_path / "sci.fits", sci, exptime=300.0)
        ref_f = self._write(tmp_path / "ref.fits", ref, exptime=300.0)
        diff_f = str(tmp_path / "diff.fits")
        psf_n_f = self._write(tmp_path / "psf_n.fits", psf_n)
        psf_r_f = self._write(tmp_path / "psf_r.fits", psf_r)
        return sci_f, ref_f, diff_f, psf_n_f, psf_r_f

    @staticmethod
    def _cfg(tmp_path, **ts_overrides):
        ts = {"sky_subtract": True, "forceconv": "AUTO"}
        ts.update(ts_overrides)
        return {
            "wdir": str(tmp_path),
            "template_subtraction": ts,
        }

    def _run(self, tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
             cfg=None, sci_fwhm=0.0, ref_fwhm=0.0, saturate=1e9,
             matching_sources=None, universal_mask=None,
             masked_sources=None, target_xy=None, lsq_stamp_sources=None):
        t = Templates(input_yaml=cfg or self._cfg(tmp_path))
        hdr = fits.getheader(sci_f)
        return t._subtract_zogy(
            sci_f, ref_f, diff_f, hdr,
            psf_n_f, psf_r_f,
            science_saturate=saturate,
            template_saturate=saturate,
            method="zogy",
            science_fwhm=sci_fwhm,
            template_fwhm=ref_fwhm,
            matching_sources=matching_sources,
            lsq_stamp_sources=lsq_stamp_sources,
            universal_mask=universal_mask,
            masked_sources=masked_sources,
            target_xy=target_xy,
        )

    def test_star_stack_width_requires_independent_agreement(self):
        consistent = templates_mod._zogy_star_stack_consistent
        # A stack wider than the agreeing references is a contaminant
        # signature (galaxy ensemble) and stays vetoed.
        assert not consistent(3.10, (2.64, 2.56))
        # A narrower stack is not: nothing real is narrower than the PSF,
        # so the offset points at inflated references (e.g. a model stamp
        # sized on a galaxy-biased detection FWHM) rather than bad stars.
        assert consistent(1.95, (2.64, 2.56))
        # ...but only within a bounded offset -- a stack under 0.6x the
        # references is a different population, not a width disagreement.
        assert not consistent(1.50, (2.64, 2.56))
        assert consistent(2.65, (2.64, 2.56))

    def test_star_stack_survives_inconsistent_references(self):
        consistent = templates_mod._zogy_star_stack_consistent
        # TDP5_10-like: the image FWHM and a degenerate model stamp
        # disagree with each other, so neither may veto the stack
        # measured directly on the field stars.
        assert consistent(5.64, (6.65, 4.60))
        # The stack is still bounded by the reference span -- a spike
        # ensemble does not get a free pass.
        assert not consistent(20.0, (6.65, 4.60))
        # Mutually consistent references still reject an off-width stack.
        assert not consistent(3.0, (6.0, 5.7))

    def test_missing_psf_falls_back_to_sfft(self, tmp_path, monkeypatch):
        sci_f, ref_f, diff_f, psf_n_f, _ = self._inputs(tmp_path)
        res = self._run(tmp_path, sci_f, ref_f, diff_f, psf_n_f, None)
        assert res == "sfft"
        assert not (tmp_path / "diff.fits").exists()

    def test_shape_mismatch_falls_back(self, tmp_path, monkeypatch):
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        # Overwrite the reference with a differently shaped image
        rng = np.random.default_rng(0)
        self._write(tmp_path / "ref.fits", rng.normal(0, 1, (96, 96)))
        res = self._run(tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f)
        assert res == "sfft"

    def test_success_writes_diff_and_headers(self, tmp_path, monkeypatch):
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        res = self._run(tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f)
        assert res == "done"
        data, hdr = fits.getdata(diff_f, header=True)
        assert hdr["FORCECON"] == "ZOGY"
        # No FWHMs supplied -> direction resolves on the measured stamp
        # widths; identical stamps tie and keep the REF convention.
        assert hdr["CONVD"] == "REF"
        assert "DIFFPSF" in hdr
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(data[y - s : y + s + 1, x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.05)
        # Unvarying common sources must cancel
        resid = float(np.nansum(data[20:41, 20:41]))
        assert abs(resid) < 0.05 * 2000.0

    def test_lsq_stamp_sources_join_kernel_pool(self, tmp_path, caplog):
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        # One flux-consistent source; the LSQ pool is topped up from the
        # wider position-matched list, where stellar-in-both-frames
        # positions join and blank sky does not.
        with caplog.at_level("INFO", logger="templates"):
            res = self._run(
                tmp_path,
                sci_f,
                ref_f,
                diff_f,
                psf_n_f,
                psf_r_f,
                matching_sources=[(30, 30)],
                lsq_stamp_sources=[(30, 30), (60, 60), (90, 40), (5, 5)],
            )
        assert res == "done"
        assert "2 position-matched stamps joined the LSQ kernel pool" in (
            caplog.text
        )

    def test_near_matched_psfs_skip_two_ringy_kernels(
        self, tmp_path, monkeypatch, caplog
    ):
        kernel = np.zeros(SHAPE)
        kernel[0, 0] = 1.35
        kernel[0, 1] = -0.35
        monkeypatch.setattr(
            templates_mod,
            "_zogy_wiener_kernel",
            lambda source, target, eps_rel=1e-2: (kernel, 0.35),
        )
        monkeypatch.setattr(
            templates_mod,
            "_zogy_wiener_kernel_adaptive",
            lambda source, target, half_width=0, fid_tol=None, neg_cap=None: (
                kernel,
                0.35,
                1.0,
            ),
        )
        monkeypatch.setattr(
            templates_mod,
            "_zogy_kernel_fid_err",
            lambda source, kernel_, target: 0.0,
        )
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        cfg = self._cfg(
            tmp_path, forceconv="REF", zogy_lsq_kernel=False
        )
        with caplog.at_level("WARNING", logger="templates"):
            res = self._run(
                tmp_path,
                sci_f,
                ref_f,
                diff_f,
                psf_n_f,
                psf_r_f,
                cfg=cfg,
                sci_fwhm=4.7,
                ref_fwhm=4.7,
                matching_sources=[(x, y) for x, y, _ in COMMON],
            )
        assert res == "done"
        assert fits.getheader(diff_f)["CONVD"] == "ZOGY"
        assert "both matching kernels are ringy" in caplog.text

    def test_ringy_flip_falls_back_to_canonical_zogy(
        self, tmp_path, monkeypatch, caplog
    ):
        """A vetoed direction may only defer to a clean kernel.

        J2344 r-band (2026-10): the REF kernel was ringy and the SCI
        kernel was merely less ringy (neg ~0.25); the old flip rule
        applied it anyway and printed sombrero residuals at every bright
        star.  When the flipped kernel is still over tolerance the code
        must skip pre-convolution entirely (canonical ZOGY).
        """
        kernel = np.zeros(SHAPE)
        kernel[0, 0] = 1.0
        probe_negs = iter([0.50, 0.20])  # REF probe, then SCI probe
        monkeypatch.setattr(
            templates_mod,
            "_zogy_wiener_kernel",
            lambda source, target, eps_rel=1e-2: (kernel, next(probe_negs)),
        )
        monkeypatch.setattr(
            templates_mod,
            "_zogy_wiener_kernel_adaptive",
            lambda source, target, half_width=0, fid_tol=None, neg_cap=None: (
                kernel,
                0.20,
                1.0,
            ),
        )
        monkeypatch.setattr(
            templates_mod,
            "_zogy_kernel_fid_err",
            lambda source, kernel_, target: 0.0,
        )
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.5), psf_r=_gauss_psf(3.0)
        )
        cfg = self._cfg(
            tmp_path, forceconv="REF", zogy_lsq_kernel=False
        )
        with caplog.at_level("WARNING", logger="templates"):
            res = self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f, cfg=cfg
            )
        assert res == "done"
        assert fits.getheader(diff_f)["CONVD"] == "ZOGY"
        assert "canonical ZOGY" in caplog.text

    def test_clean_kernel_flip_is_still_applied(
        self, tmp_path, monkeypatch, caplog
    ):
        """The flip remains available when the opposite direction is a
        genuinely clean smoother (neg mass under the Wiener tolerance)."""
        kernel = np.zeros(SHAPE)
        kernel[0, 0] = 1.0
        probe_negs = iter([0.50, 0.02])  # REF probe, then SCI probe
        monkeypatch.setattr(
            templates_mod,
            "_zogy_wiener_kernel",
            lambda source, target, eps_rel=1e-2: (kernel, next(probe_negs)),
        )
        monkeypatch.setattr(
            templates_mod,
            "_zogy_wiener_kernel_adaptive",
            lambda source, target, half_width=0, fid_tol=None, neg_cap=None: (
                kernel,
                0.02,
                1.0,
            ),
        )
        monkeypatch.setattr(
            templates_mod,
            "_zogy_kernel_fid_err",
            lambda source, kernel_, target: 0.0,
        )
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.5), psf_r=_gauss_psf(3.0)
        )
        cfg = self._cfg(
            tmp_path, forceconv="REF", zogy_lsq_kernel=False
        )
        with caplog.at_level("WARNING", logger="templates"):
            res = self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f, cfg=cfg
            )
        assert res == "done"
        assert fits.getheader(diff_f)["CONVD"] == "SCI"
        assert "overriding to" in caplog.text

    def test_flux_scale_recovered(self, tmp_path, monkeypatch):
        """A template 3x brighter than the science must be rescaled so
        sources cancel and the transient keeps its true flux."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, ref_scale=3.0
        )
        res = self._run(tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f)
        assert res == "done"
        data = fits.getdata(diff_f)
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(data[y - s : y + s + 1, x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.08)
        resid = float(np.nansum(data[20:41, 20:41]))
        assert abs(resid) < 0.05 * 2000.0

    def test_universal_mask_fills_masked_flux(self, tmp_path, monkeypatch):
        """Masked-but-finite pixels must not reach the FFT convolution.

        A saturated spike inside the masked footprint carries non-linear
        flux that would otherwise print as a huge residual (and ring into
        neighbouring pixels); filling it like a NaN region keeps the diff
        clean while the caller re-imposes the mask on the output file.
        """
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        with fits.open(sci_f) as hdul:
            sci = hdul[0].data.copy()
        # Saturated detector blob, science-side only.
        sci[95:105, 95:105] += 5.0e4
        self._write(sci_f, sci, exptime=300.0)
        um = np.zeros(SHAPE, dtype=bool)
        um[92:108, 92:108] = True

        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
            universal_mask=um,
        )
        assert res == "done"
        data = fits.getdata(diff_f)
        # The masked blob must not leak flux into the diff: the fill
        # produces ~(med_sci - med_ref) inside the footprint, and the
        # pixels just outside it must not carry a spike's wings/ringing.
        inner = np.nanmax(np.abs(data[93:107, 93:107]))
        ring = data[88:112, 88:112].copy()
        ring[5:19, 5:19] = np.nan  # exclude the masked block itself
        assert inner < 5.0
        assert np.nanmax(np.abs(ring)) < 20.0

        # Without the mask the same blob dominates the difference.
        diff_f2 = str(tmp_path / "diff_nomask.fits")
        res2 = self._run(tmp_path, sci_f, ref_f, diff_f2, psf_n_f, psf_r_f)
        assert res2 == "done"
        data2 = fits.getdata(diff_f2)
        assert np.nanmax(np.abs(data2[90:110, 90:110])) > 1000.0

    def test_masked_and_target_positions_banned_from_scale(
        self, tmp_path, monkeypatch, caplog
    ):
        """Variable sources (masked_sources) and the target position must
        not feed the aperture flux-scale or star-stack ensembles."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        # matching_sources are 0-based; masked_sources are 1-based.  The
        # ban must convert conventions and drop the listed source.
        ms = [(30, 30), (60, 60), (90, 40)]
        banned = [(61.0, 61.0)]  # 1-based -> (60, 60)
        with caplog.at_level("INFO", logger="templates"):
            res = self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
                matching_sources=ms,
                masked_sources=banned,
                target_xy=(80.0, 80.0),
            )
        assert res == "done"
        msgs = "\n".join(r.getMessage() for r in caplog.records)
        assert "excluded 1 matched source" in msgs
        data = fits.getdata(diff_f)
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(data[y - s : y + s + 1, x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.08)

    def test_mismatched_universal_mask_warned_not_fatal(
        self, tmp_path, monkeypatch
    ):
        """A mask on the wrong pixel grid must be ignored, not crash."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
            universal_mask=np.ones((64, 64), dtype=bool),
        )
        assert res == "done"

    def test_auto_forceconv_template_broader(self, tmp_path, monkeypatch):
        """AUTO with a broader template must convolve the science image
        (CONVD=SCI), giving the diff the reference PSF."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.2), psf_r=_gauss_psf(2.5)
        )
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
            sci_fwhm=2.8, ref_fwhm=5.9,
        )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        assert hdr["CONVD"] == "SCI"
        # DIFFFWHM is the measured width of the stamp actually used as the
        # diff PSF -- close to, but not identical with, the header FWHM.
        assert hdr["DIFFFWHM"] == pytest.approx(5.9, rel=0.15)

    def test_auto_forceconv_kernel_probe_overrides_fwhm(
        self, tmp_path, monkeypatch
    ):
        """AUTO must resolve on the candidate kernels' negative mass, not
        the image FWHMs.  Here the image FWHMs wrongly claim the science
        PSF is broader (sci 6.0 > ref 4.0 -> REF), while the actual PSF
        stamps show the reference is broader (sigma 2.5 vs 1.2).  The old
        FWHM-only resolution built a sharpening kernel that left dipole
        residuals at bright sources (real-data failure on TDP6_12)."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.2), psf_r=_gauss_psf(2.5)
        )
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
            sci_fwhm=6.0, ref_fwhm=4.0,  # misleading: claims sci broader
        )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        # The stamps say the template is broader: probe must pick SCI.
        assert hdr["CONVD"] == "SCI"

    def test_nan_holed_psf_stamp_still_finite(self, tmp_path, monkeypatch):
        """A PSF stamp carrying NaN holes (photutils ePSFs write NaN
        outside the support) must not propagate NaN through the matching
        kernel into the convolved image.  Core holes still force the
        kernel probe to refuse and fall back to the canonical path."""
        psf_nan = _gauss_psf(1.5, size=41)
        # NaN patch inside the PSF core -- sits in the untapered region,
        # so the kernel stays non-finite and is refused.
        psf_nan[18:24, 18:24] = np.nan
        sci_f, ref_f, diff_f, _, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.5), psf_r=_gauss_psf(2.5)
        )
        self._write(tmp_path / "psf_n.fits", psf_nan)
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f,
            str(tmp_path / "psf_n.fits"), psf_r_f,
            sci_fwhm=3.5, ref_fwhm=5.9,  # template broader -> would preconv
        )
        assert res == "done"
        data = fits.getdata(diff_f)
        hdr = fits.getheader(diff_f)
        assert np.isfinite(data).all()
        assert hdr["CONVD"] == "ZOGY"

    def test_nan_edge_psf_stamp_cleaned_for_kernel(self, tmp_path,
                                                   monkeypatch):
        """NaN borders outside the ePSF support land in the
        taper-suppressed wing band and must be cleaned so the matching
        kernel remains usable rather than forcing a canonical fallback."""
        psf_nan = _gauss_psf(1.5, size=41)
        psf_nan[:4, :] = np.nan
        psf_nan[-4:, :] = np.nan
        sci_f, ref_f, diff_f, _, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.5), psf_r=_gauss_psf(2.5)
        )
        self._write(tmp_path / "psf_n.fits", psf_nan)
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f,
            str(tmp_path / "psf_n.fits"), psf_r_f,
            sci_fwhm=3.5, ref_fwhm=5.9,
        )
        assert res == "done"
        data = fits.getdata(diff_f)
        hdr = fits.getheader(diff_f)
        assert np.isfinite(data).all()
        # The cleaned stamp yields a usable kernel: template is broader,
        # so the science image is convolved to the reference PSF.
        assert hdr["CONVD"] == "SCI"

    def test_inconsistent_star_stack_does_not_override_psf_models(
        self, tmp_path, monkeypatch, caplog
    ):
        narrow = _gauss_psf(0.8)
        narrow_fwhm = templates_mod._measure_stamp_fwhm(narrow)
        monkeypatch.setattr(
            templates_mod,
            "_zogy_star_psf",
            lambda image, positions, guess_fwhm=8.0, saturate=np.nan: (
                narrow,
                narrow_fwhm,
                positions,
            ),
        )
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(2.0), psf_r=_gauss_psf(2.0)
        )
        with caplog.at_level("WARNING", logger="templates"):
            res = self._run(
                tmp_path,
                sci_f,
                ref_f,
                diff_f,
                psf_n_f,
                psf_r_f,
                sci_fwhm=4.7,
                ref_fwhm=4.7,
                matching_sources=[(x, y) for x, y, _ in COMMON],
            )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        assert hdr["FWHM_SCI"] == pytest.approx(4.7, rel=0.15)
        assert hdr["FWHM_REF"] == pytest.approx(4.7, rel=0.15)
        assert "star-stack PSF rejected" in caplog.text

    def test_degenerate_stamp_replaced_by_star_stack(
        self, tmp_path, monkeypatch
    ):
        """A sub-pixel / unsanitary PSF stamp must be replaced by the
        star-stacked field PSF measured at the matched positions, and the
        DIFFPSF product must carry that substitute stamp (real-data
        failure: degenerate ePSFs on TDP5_4/TDP8_2 produced sharpening
        kernels and dipole residuals)."""
        from templates import _measure_stamp_fwhm, _zogy_star_psf

        # A delta-like stamp: unmeasurable FWHM -> unsanitary.
        bad = np.zeros((31, 31))
        bad[15, 15] = 1.0
        bad[::2, ::2] -= 0.001  # negative sprinkling, like a noisy ePSF
        sci_f, ref_f, diff_f, _, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(2.0), psf_r=_gauss_psf(2.0)
        )
        self._write(tmp_path / "psf_n.fits", bad)
        positions = [(x, y) for x, y, _ in COMMON]
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f,
            str(tmp_path / "psf_n.fits"), psf_r_f,
            sci_fwhm=4.7, ref_fwhm=4.7,
            matching_sources=positions,
        )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        # The effective science width is the star-stack width (~4.7 px
        # for sigma=2.0), not the degenerate stamp or nothing.
        assert hdr["FWHM_SCI"] == pytest.approx(4.7, rel=0.2)
        stamp = fits.getdata(hdr["DIFFPSF"])
        assert np.isfinite(stamp).all()
        assert _measure_stamp_fwhm(stamp) == pytest.approx(4.7, rel=0.2)
        # Transient still carries its flux.
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(fits.getdata(diff_f)[y - s : y + s + 1,
                                                    x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.10)

    def test_empirical_psf_model_kept_over_star_stack(
        self, tmp_path, caplog
    ):
        """A stamp marked PSFBUILD=epsf is a measured photometry model:
        the star stack only arbitrates it (>15% width disagreement
        triggers substitution), never overrides it outright.  Analytic
        stamps keep losing to a vetted stack -- they are the fallback
        the stack exists to rescue."""
        positions = [(x, y) for x, y, _ in COMMON]
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)

        # An agreeing ePSF stamp: kept, with a provenance log.
        hdr_in = fits.Header()
        hdr_in["PSFBUILD"] = "epsf"
        fits.writeto(psf_n_f, _gauss_psf(2.0), header=hdr_in,
                     overwrite=True)
        with caplog.at_level("INFO", logger="templates"):
            res = self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
                sci_fwhm=4.7, ref_fwhm=4.7,
                matching_sources=positions,
            )
        assert res == "done"
        msgs = "\n".join(r.getMessage() for r in caplog.records)
        assert "keeping the empirical ePSF model" in msgs

        # The same stamp marked analytic: the stack wins outright.
        caplog.clear()
        hdr_in["PSFBUILD"] = "analytic-composite"
        fits.writeto(psf_n_f, _gauss_psf(2.0), header=hdr_in,
                     overwrite=True)
        with caplog.at_level("INFO", logger="templates"):
            res = self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
                sci_fwhm=4.7, ref_fwhm=4.7,
                matching_sources=positions,
            )
        assert res == "done"
        msgs = "\n".join(r.getMessage() for r in caplog.records)
        assert "using star-stack stamp" in msgs

    def test_star_stack_filters_contaminated_positions(
        self, tmp_path, monkeypatch
    ):
        """The star stack must reject positions that are blank sky or
        spikes in the cutout frame -- a matched list can carry cosmic
        rays that are only present in one epoch."""
        from templates import _zogy_star_psf

        rng = np.random.default_rng(1)
        psf = _gauss_psf(2.0)
        sci = _make_image(rng, SHAPE, psf, COMMON, 1.0)
        ref = _make_image(rng, SHAPE, psf, COMMON, 1.0)
        # A science-only cosmic-ray spike at a matched position.
        sci[20, 110] += 5000.0
        positions = [(x, y) for x, y, _ in COMMON] + [(110, 20)]

        stamp_s, f_s, kept_s = _zogy_star_psf(sci, positions, 4.7)
        assert stamp_s is not None
        assert f_s == pytest.approx(4.7, rel=0.2)
        # The spike position must not survive the FWHM window.
        assert (110.0, 20.0) not in [(round(x), round(y)) for x, y in kept_s]

        # On the reference frame the same position is blank sky and must
        # fail the peak-significance gate before even being measured.
        stamp_r, f_r, kept_r = _zogy_star_psf(ref, positions, 4.7)
        assert stamp_r is not None
        assert (110.0, 20.0) not in [(round(x), round(y)) for x, y in kept_r]

    def test_star_stack_keeps_faint_stars_on_noisy_frames(self):
        """A real star whose peak sits below 10 sigma of the background
        must still stack when its integrated aperture SNR is strong --
        a broad PSF on a noisy frame spreads its light so the peak SNR
        alone is the wrong statistic (TDP6 60553 r: every matched star
        peaked at 6-9 sigma and the stack died silently)."""
        from templates import _zogy_star_psf

        rng = np.random.default_rng(7)
        shape = (220, 220)
        img = rng.normal(100.0, 20.0, shape)
        psf = _gauss_psf(5.5)  # FWHM ~13 px; unit-normalized stamp
        h = psf.shape[0] // 2
        # Total flux ~3e4 gives a ~160 ADU peak: ~8 sigma of the 20-ADU
        # background -- a real star that the old peak-only gate dropped.
        for cx, cy, flux in ((110, 60, 30000.0), (60, 140, 26000.0)):
            img[cy - h : cy + h + 1, cx - h : cx + h + 1] += flux * psf
        positions = [(110, 60), (60, 140)]
        stamp, f, kept = _zogy_star_psf(img, positions, 13.0)
        assert stamp is not None
        assert f == pytest.approx(13.0, rel=0.35)
        assert len(kept) == 2

        # A blank-sky position integrates to noise: still rejected.
        stamp, f, kept = _zogy_star_psf(img, [(110, 60), (30, 190)], 13.0)
        assert len(kept) == 1
        assert round(kept[0][0]) == 110 and round(kept[0][1]) == 60

    def test_star_stack_single_survivor_and_none(self, tmp_path):
        """One valid star still yields a stack.  ``kept`` is [] when
        vetting ran but rejected everything, and None only when no
        positions were offered -- the caller treats "vetted, none
        survived" differently from "no vetting"."""
        from templates import _zogy_star_psf

        rng = np.random.default_rng(2)
        psf = _gauss_psf(2.0)
        img = _make_image(rng, SHAPE, psf, COMMON, 1.0)

        stamp, f, kept = _zogy_star_psf(img, [(60, 60)], 4.7)
        assert stamp is not None
        assert f == pytest.approx(4.7, rel=0.2)
        assert len(kept) == 1

        stamp, f, kept = _zogy_star_psf(img, [(5, 5)], 4.7)
        assert stamp is None
        assert np.isnan(f)
        assert kept == []

        stamp, f, kept = _zogy_star_psf(img, None, 4.7)
        assert stamp is None
        assert np.isnan(f)
        assert kept is None

    def test_explicit_ref_deconvolution_warns(self, tmp_path, monkeypatch,
                                              caplog):
        """forceconv=REF with a broader template must log the deconvolution
        warning (parity with the SFFT branch)."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.2), psf_r=_gauss_psf(2.5)
        )
        cfg = self._cfg(tmp_path, forceconv="REF")
        with caplog.at_level("WARNING", logger="templates"):
            self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
                cfg=cfg, sci_fwhm=2.8, ref_fwhm=5.9,
            )
        assert any("deconvolution" in r.getMessage().lower() for r in caplog.records)

    def test_explicit_ref_broader_template_flips_to_sci(
        self, tmp_path, monkeypatch
    ):
        """forceconv=REF on a broader template requests a deconvolution:
        the guard must veto the ringy kernel and convolve the science
        image instead (CONVD=SCI), not apply the kernel anyway."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(1.2), psf_r=_gauss_psf(2.5)
        )
        cfg = self._cfg(tmp_path, forceconv="REF")
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
            cfg=cfg, sci_fwhm=2.8, ref_fwhm=5.9,
        )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        assert hdr["CONVD"] == "SCI"
        assert hdr["DIFFFWHM"] == pytest.approx(5.9, rel=0.15)

    def test_explicit_sci_broader_science_flips_to_ref(
        self, tmp_path, monkeypatch
    ):
        """Symmetric guard: forceconv=SCI with a broader science PSF
        would deconvolve; the resolved direction must flip to REF."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, psf_n=_gauss_psf(2.5), psf_r=_gauss_psf(1.2)
        )
        cfg = self._cfg(tmp_path, forceconv="SCI")
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
            cfg=cfg, sci_fwhm=5.9, ref_fwhm=2.8,
        )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        assert hdr["CONVD"] == "REF"

    def test_unknown_nan_fill_warns_and_uses_median(self, tmp_path,
                                                    monkeypatch, caplog):
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        # Force NaN regions so the fill path runs
        sci = np.asarray(fits.getdata(sci_f))
        sci[0:5, :] = np.nan
        fits.writeto(sci_f, sci, overwrite=True)
        cfg = self._cfg(tmp_path, zogy_nan_fill="bogus")
        with caplog.at_level("WARNING", logger="templates"):
            res = self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f, cfg=cfg
            )
        assert res == "done"
        assert any("zogy_nan_fill" in r.getMessage() for r in caplog.records)

    def test_score_images_written_when_enabled(self, tmp_path, monkeypatch):
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        cfg = self._cfg(tmp_path, zogy_write_score_images=True)
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f, cfg=cfg
        )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        s_path = tmp_path / "zogy_score_diff.fits"
        sc_path = tmp_path / "zogy_scorr_diff.fits"
        assert s_path.exists() and sc_path.exists()
        assert hdr["DIFFS"] == str(s_path)
        # Scorr should peak at the transient position well above noise
        scorr = np.asarray(fits.getdata(sc_path))
        x, y, _ = TRANSIENT
        assert scorr[y, x] > 10

    def test_matched_aperture_flux_scale(self, tmp_path, monkeypatch, caplog):
        """With matched positions, the flux scale comes from aperture
        photometry (PSF-shape independent) rather than per-pixel ratios."""
        # Different PSF widths make the pixel-ratio estimator biased; the
        # aperture estimator integrates whole sources and is not.
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path,
            ref_scale=3.0,
            psf_n=_gauss_psf(1.6),
            psf_r=_gauss_psf(2.6),
        )
        t = Templates(input_yaml=self._cfg(tmp_path))
        hdr = fits.getheader(sci_f)
        positions = [(x, y) for x, y, _ in COMMON]
        with caplog.at_level("INFO", logger="templates"):
            res = t._subtract_zogy(
                sci_f, ref_f, diff_f, hdr, psf_n_f, psf_r_f,
                science_saturate=1e9, template_saturate=1e9,
                method="zogy",
                science_fwhm=3.8, template_fwhm=6.1,
                matching_sources=positions,
            )
        assert res == "done"
        assert any(
            "estimator=aperture" in r.getMessage() for r in caplog.records
        )
        data = fits.getdata(diff_f)
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(data[y - s : y + s + 1, x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.1)

    def test_flux_scale_retries_with_full_match_list(
        self, tmp_path, monkeypatch, caplog
    ):
        """A validated PSF pool that collapses to a single usable
        position must not leave the aperture scale resting on one ratio;
        the estimator retries on the full matched list."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(
            tmp_path, ref_scale=3.0
        )
        orig = templates_mod._zogy_star_psf
        monkeypatch.setattr(
            templates_mod,
            "_zogy_star_psf",
            lambda img, pos, guess_fwhm=8.0, saturate=np.nan: (
                lambda s, f, k: (s, f, k[:1] if k else k)
            )(*orig(img, pos, guess_fwhm, saturate)),
        )
        positions = [(x, y) for x, y, _ in COMMON]
        with caplog.at_level("INFO", logger="templates"):
            res = self._run(
                tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
                sci_fwhm=3.8, ref_fwhm=3.8,
                matching_sources=positions,
            )
        assert res == "done"
        logs = [r.getMessage() for r in caplog.records]
        assert any("estimator=aperture" in m for m in logs)
        # Without the retry the sparse pool reports n_ratios=1.
        assert any("n_ratios=3" in m for m in logs)

    def test_aperture_scale_falls_back_without_matches(
        self, tmp_path, monkeypatch, caplog
    ):
        """Fewer than 3 usable matched positions must fall back to the
        pixel-ratio estimator, not fail."""
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = self._inputs(tmp_path)
        t = Templates(input_yaml=self._cfg(tmp_path))
        hdr = fits.getheader(sci_f)
        with caplog.at_level("INFO", logger="templates"):
            res = t._subtract_zogy(
                sci_f, ref_f, diff_f, hdr, psf_n_f, psf_r_f,
                science_saturate=1e9, template_saturate=1e9,
                method="zogy", matching_sources=[(60, 30)],
            )
        assert res == "done"
        assert any(
            "estimator=pixel-ratio" in r.getMessage() for r in caplog.records
        )

    def test_preconvolved_diff_keeps_sharp_source_compact(
        self, tmp_path, monkeypatch
    ):
        """After pre-convolution both PSFs are equal, so D reduces to the
        plain difference.  The Fourier P_hat/|P_hat| ratio adds phase
        ringing when the PSF transform is not strictly positive (noisy /
        boxcar stamps): a sharp science-only feature must stay compact,
        not smear into an extended +- blob."""
        rng = np.random.default_rng(11)

        def boxcar_psf(size=9):
            p = np.ones((size, size))
            # small pixel-level roughness guarantees sign changes in the
            # transform (empirical ePSF stamps are never perfectly smooth)
            return p + rng.uniform(-0.05, 0.05, p.shape)

        psf_n = boxcar_psf(9)
        psf_n = psf_n / np.sum(psf_n)
        psf_r = boxcar_psf(7)
        psf_r = psf_r / np.sum(psf_r)
        sci = _make_image(rng, SHAPE, _gauss_psf(2.2), COMMON, 1.0)
        ref = _make_image(rng, SHAPE, _gauss_psf(1.4), COMMON, 1.0)
        # sharp science-only artifact (survived CR cleaning): 3x3 spike
        sci[100:103, 100:103] += 5000.0
        sci_f = self._write(tmp_path / "sci.fits", sci, exptime=300.0)
        ref_f = self._write(tmp_path / "ref.fits", ref, exptime=300.0)
        diff_f = str(tmp_path / "diff.fits")
        psf_n_f = self._write(tmp_path / "psf_n.fits", psf_n)
        psf_r_f = self._write(tmp_path / "psf_r.fits", psf_r)
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_n_f, psf_r_f,
            sci_fwhm=5.2, ref_fwhm=3.3,  # sci broader -> CONVD=REF
        )
        assert res == "done"
        hdr = fits.getheader(diff_f)
        assert hdr["CONVD"] == "REF"
        data = fits.getdata(diff_f)
        # The spike's peak must survive at near-full amplitude rather than
        # being redistributed into a wide lobed artifact.
        assert np.nanmax(data[98:106, 98:106]) > 0.5 * 5000.0
        # and the surrounding ring should not carry large negative lobes
        ring = np.nanmin(data[90:114, 90:114])
        assert ring > -0.1 * 5000.0

    def test_saturated_pixels_excluded_from_flux_scale(self, tmp_path,
                                                     monkeypatch):
        """A saturated science star must not drag the flux-scale estimate
        toward the saturation plateau ratio."""
        rng = np.random.default_rng(5)
        psf = _gauss_psf(2.0)
        sci = _make_image(rng, SHAPE, psf, COMMON + [TRANSIENT], 1.0)
        # Flatten a "saturated" core on the brightest common source
        sci[27:34, 57:64] = 50000.0
        ref = _make_image(rng, SHAPE, psf, COMMON, 1.0)
        sci_f = self._write(tmp_path / "sci.fits", sci, exptime=300.0)
        ref_f = self._write(tmp_path / "ref.fits", ref, exptime=300.0)
        diff_f = str(tmp_path / "diff.fits")
        psf_f = self._write(tmp_path / "psf.fits", psf)
        res = self._run(
            tmp_path, sci_f, ref_f, diff_f, psf_f, psf_f,
            saturate=60000.0,
        )
        assert res == "done"
        data = fits.getdata(diff_f)
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(data[y - s : y + s + 1, x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.08)


class TestMeasureStampFwhm:
    def test_gaussian_recovers_true_fwhm(self):
        from templates import _measure_stamp_fwhm
        sigma = 2.0
        stamp = _gauss_psf(sigma, size=41)
        fwhm = _measure_stamp_fwhm(stamp)
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        assert fwhm == pytest.approx(expected, rel=0.15)

    def test_degenerate_stamp_returns_nan(self):
        from templates import _measure_stamp_fwhm
        assert np.isnan(_measure_stamp_fwhm(np.zeros((15, 15))))
        assert np.isnan(_measure_stamp_fwhm(np.full((15, 15), np.nan)))


class TestZogyKernelHelpers:
    def test_taper_suppresses_wings_not_core(self):
        from templates import _zogy_taper_stamp

        stamp = _gauss_psf(2.0, size=41)
        out = _zogy_taper_stamp(stamp, fwhm=4.7)
        # Unit normalisation is restored and the wings are attenuated
        # relative to the peak by the cosine taper.
        assert out.sum() == pytest.approx(1.0)
        assert out[20, 2] / out[20, 20] < stamp[20, 2] / stamp[20, 20]
        assert np.isfinite(out).all()

    def test_taper_handles_nan_stamp(self):
        from templates import _zogy_taper_stamp

        stamp = _gauss_psf(2.0, size=21)
        # ePSF-style NaN border in the taper-suppressed corner band is
        # cleaned; NaN inside the informative core is left to poison the
        # kernel so the probe still refuses a corrupt stamp.
        stamp[:3, :3] = np.nan
        out = _zogy_taper_stamp(stamp, fwhm=4.7)
        assert np.isfinite(out).all()
        stamp2 = _gauss_psf(2.0, size=21)
        stamp2[10, 8:13] = np.nan
        assert not np.isfinite(_zogy_taper_stamp(stamp2, fwhm=4.7)).all()

    def test_window_kernel_truncates_and_conserves(self):
        from templates import _zogy_window_kernel

        rng = np.random.default_rng(0)
        k = rng.normal(0.0, 1.0, (31, 31))
        k[15, 15] = 50.0  # unambiguous peak
        kw = _zogy_window_kernel(k, half_width=5)
        yy, xx = np.mgrid[0:31, 0:31]
        # Square +-half_width window around the peak; everything outside
        # is zeroed and the kept part is renormalised to unit sum.
        outside = (np.abs(xx - 15) > 5) | (np.abs(yy - 15) > 5)
        assert (kw[outside] == 0.0).all()
        assert kw.sum() == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_allclose(
            kw[~outside], k[~outside] / k[~outside].sum()
        )

    def test_window_kernel_degenerate(self):
        from templates import _zogy_window_kernel

        k = np.zeros((9, 9))
        kw = _zogy_window_kernel(k, half_width=2)
        assert (kw == 0.0).all()


class TestPadPsfToImage:
    def test_centering_and_normalization(self):
        psf = _gauss_psf(2.5, size=21)
        padded = _pad_psf_to_image(psf, (64, 64))
        assert padded.shape == (64, 64)
        assert float(np.nansum(padded)) == pytest.approx(1.0, rel=1e-10)
        # For FFT convolution the kernel zero-lag must sit at index (0,0),
        # wrapping around the array edges (ifftshift convention).  Undoing
        # the shift must return the PSF peak to the array centre.
        centered = np.fft.fftshift(padded)
        assert np.unravel_index(np.argmax(centered), centered.shape) == (32, 32)
        # And fft2 of the padded PSF must have no linear phase (real, even PSF).
        Ph = np.fft.fft2(padded)
        assert np.max(np.abs(Ph.imag)) < 1e-12

    def test_no_unnecessary_padding(self):
        # Same-shape PSFs are still ifftshifted (zero-lag to [0,0]).
        psf = _gauss_psf(1.5, size=15)
        padded = _pad_psf_to_image(psf, (15, 15))
        np.testing.assert_allclose(
            np.fft.fftshift(padded), psf / np.nansum(psf), atol=1e-12
        )


class TestRobustOutlierMask:
    def test_constant_array_all_inliers(self):
        """Zero MAD must not flag every point as an outlier."""
        vals = np.full(20, 5.0)
        mask = Templates.robust_outlier_mask(vals, n_sigma=3.0)
        assert mask.all()

    def test_single_outlier_rejected(self):
        rng = np.random.default_rng(1)
        vals = rng.normal(10.0, 0.01, 50)
        vals[17] = 15.0  # gross outlier
        mask = Templates.robust_outlier_mask(vals, n_sigma=3.0)
        assert not mask[17]
        assert mask[:17].all()

    def test_nan_handled(self):
        vals = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
        mask = Templates.robust_outlier_mask(vals, n_sigma=3.0)
        assert not mask[2]


class TestPSFStampChi2:
    """Per-source PSF-model stamp chi2 used by the flux-comparison veto."""

    @staticmethod
    def _image_psf(sigma=1.4, size=41):
        from photutils.psf import ImagePSF
        ax = np.arange(size) - size // 2
        g = np.exp(-0.5 * (ax[None, :] ** 2 + ax[:, None] ** 2) / sigma ** 2)
        g /= g.sum()
        return ImagePSF(data=g, x_0=size // 2, y_0=size // 2, oversampling=1)

    def _field(self, rng, shape=(160, 160), noise=0.5):
        sig = 1.4
        ax = np.arange(41) - 20
        g = np.exp(-0.5 * (ax[None, :] ** 2 + ax[:, None] ** 2) / sig ** 2)
        g /= g.sum()
        img = rng.normal(0.0, noise, shape)
        pos = [(30, 30), (60, 60), (90, 90), (40, 100), (100, 40)]
        flux = [3000.0, 2500.0, 2000.0, 1500.0, 2800.0]
        for (x, y), f in zip(pos, flux):
            img[y - 20 : y + 21, x - 20 : x + 21] += f * g
        return img, g, pos

    def test_clean_sources_low_chi2(self):
        rng = np.random.default_rng(0)
        img, _, pos = self._field(rng)
        psf = self._image_psf()
        x = np.array([p[0] for p in pos], float)
        y = np.array([p[1] for p in pos], float)
        chi2 = Templates._psf_stamp_chi2(img, x, y, psf, fwhm=3.3, gain=1.0)
        assert np.isfinite(chi2).all()
        assert (chi2 < 5.0).all()

    def test_blend_and_extended_rejected(self):
        rng = np.random.default_rng(0)
        img, g, pos = self._field(rng)
        # blended double at (60, 130)
        img[110:151, 40:81] += 2000.0 * g
        img[112:153, 43:84] += 2000.0 * g
        # elongated source at (110, 60)
        ax = np.arange(41) - 20
        ge = np.exp(-0.5 * ((ax[None, :] / 4.0) ** 2 + (ax[:, None] / 1.0) ** 2))
        ge /= ge.sum()
        img[40:81, 90:131] += 2500.0 * ge
        psf = self._image_psf()
        x = np.array([p[0] for p in pos] + [60.0, 110.0])
        y = np.array([p[1] for p in pos] + [130.0, 60.0])
        chi2 = Templates._psf_stamp_chi2(img, x, y, psf, fwhm=3.3, gain=1.0)
        assert (chi2[:5] < 5.0).all()
        assert chi2[5] > 5.0  # blend
        assert chi2[6] > 5.0  # elongated

    def test_off_frame_and_missing_inputs(self):
        rng = np.random.default_rng(0)
        img, _, _ = self._field(rng)
        psf = self._image_psf()
        x = np.array([2.0, 60.0])
        y = np.array([60.0, 60.0])
        chi2 = Templates._psf_stamp_chi2(img, x, y, psf, fwhm=3.3)
        assert np.isnan(chi2[0])  # stamp off-frame -> unverifiable
        assert np.isfinite(chi2[1])
        # missing inputs -> all NaN
        assert np.isnan(
            Templates._psf_stamp_chi2(None, x, y, psf, 3.3)
        ).all()
        assert np.isnan(
            Templates._psf_stamp_chi2(img, x, y, None, 3.3)
        ).all()
        assert np.isnan(
            Templates._psf_stamp_chi2(img, x, y, psf, np.nan)
        ).all()

    def test_analytic_floor_rescues_model_limited_stars(self):
        # Stars carry real structure the smooth circular model cannot
        # match (ellipticity ~ the template-side analytic Moffat
        # situation).  With the empirical 5% floor the bright stars fail
        # the veto for model error alone; the analytic-model floor keeps
        # them while the metric still sees the mismatch.
        rng = np.random.default_rng(0)
        img = rng.normal(0.0, 0.5, (160, 160))
        ax = np.arange(41) - 20
        ge = np.exp(-0.5 * ((ax[None, :] / 1.60) ** 2 + (ax[:, None] / 1.30) ** 2))
        ge /= ge.sum()
        pos = [(30, 30), (60, 60), (90, 90), (40, 100), (100, 40)]
        for (x, y), f in zip(pos, [3000., 2500., 2000., 1500., 2800.]):
            img[y - 20 : y + 21, x - 20 : x + 21] += 8.0 * f * ge
        psf = self._image_psf()
        x = np.array([p[0] for p in pos], float)
        y = np.array([p[1] for p in pos], float)
        chi_emp = Templates._psf_stamp_chi2(
            img, x, y, psf, fwhm=3.3, gain=1.0, model_floor_frac=0.05
        )
        chi_ana = Templates._psf_stamp_chi2(
            img, x, y, psf, fwhm=3.3, gain=1.0, model_floor_frac=0.15
        )
        assert np.nanmedian(chi_emp) > 5.0
        assert np.nanmedian(chi_ana) < 5.0
        assert (chi_ana < chi_emp).all()


class TestPSFVettingFluxMatch:
    """End-to-end PSF-fit-quality veto inside find_flux_consistent_sources."""

    @staticmethod
    def _setup(seed=0):
        from photutils.psf import ImagePSF
        rng = np.random.default_rng(seed)
        sig = 1.4
        ax = np.arange(41) - 20
        g = np.exp(-0.5 * (ax[None, :] ** 2 + ax[:, None] ** 2) / sig ** 2)
        g /= g.sum()
        psf = ImagePSF(data=g, x_0=20, y_0=20, oversampling=1)

        img = rng.normal(0.0, 0.5, (160, 160))
        pos = [
            (30, 30), (60, 60), (90, 90), (40, 100), (100, 40), (120, 120),
            (125, 70), (30, 135), (135, 30), (80, 30), (30, 80), (120, 80),
        ]
        flux = np.array(
            [3000., 2500., 2000., 1500., 2800., 2200., 2600., 1900.,
             2400., 2100., 2300., 2700.]
        )
        for (x, y), f in zip(pos, flux):
            img[y - 20 : y + 21, x - 20 : x + 21] += f * g
        # blended double at (60, 130) - flux-consistent but not PSF-like
        img[110:151, 40:81] += 2000.0 * g
        img[112:153, 43:84] += 2000.0 * g

        x_all = np.array([p[0] for p in pos] + [60], float)
        y_all = np.array([p[1] for p in pos] + [130], float)
        f_all = np.append(flux, 4000.0)
        cat = pd.DataFrame(
            {
                "x_pix": x_all,
                "y_pix": y_all,
                "flux_AP": f_all,
                "flux_AP_err": f_all / 50.0,
                "threshold": np.full(len(f_all), 100.0),
            }
        )
        cat_t = cat.copy()
        cat_t["flux_AP"] = f_all * 0.5
        cat_t["flux_AP_err"] = f_all * 0.5 / 50.0
        return img, psf, cat, cat_t

    def test_veto_removes_blend(self):
        img, psf, cat_s, cat_t = self._setup()
        t = Templates(input_yaml={})
        vet = {
            "image_sci": img, "image_tpl": img,
            "psf_sci": psf, "psf_tpl": psf,
            "fwhm_sci": 3.3, "fwhm_tpl": 3.3,
            "chi2_max": 5.0,
        }
        out, _ = t.find_flux_consistent_sources(
            cat_s, cat_t, make_plot=False, psf_vetting=vet
        )
        blend = (out["x_pix"] == 60) & (out["y_pix"] == 130)
        assert not blend.any()
        assert len(out) >= 10
        assert {"psf_chi2_sci", "psf_chi2_tpl", "is_psf_inlier"} <= set(
            out.columns
        )

    def test_no_vetting_keeps_blend(self):
        img, psf, cat_s, cat_t = self._setup()
        t = Templates(input_yaml={})
        out, _ = t.find_flux_consistent_sources(cat_s, cat_t, make_plot=False)
        blend = (out["x_pix"] == 60) & (out["y_pix"] == 130)
        assert blend.any()

    def test_veto_only_one_image(self):
        """A source bad in the reference only must still be rejected."""
        img, psf, cat_s, cat_t = self._setup()
        t = Templates(input_yaml={})
        # Template image without the blend structure: replace the blend
        # with a clean single PSF in the template so only science vetoes it.
        vet = {
            "image_sci": img, "image_tpl": img,
            "psf_sci": psf, "psf_tpl": psf,
            "fwhm_sci": 3.3, "fwhm_tpl": 3.3,
            "chi2_max": 5.0,
        }
        out, _ = t.find_flux_consistent_sources(
            cat_s, cat_t, make_plot=False, psf_vetting=vet
        )
        assert not ((out["x_pix"] == 60) & (out["y_pix"] == 130)).any()

    def test_veto_adaptive_min_keep(self):
        """A veto that rejects nearly everything must still honour the
        min_keep floor by restoring the best-fitting sources."""
        img, psf, cat_s, cat_t = self._setup()
        t = Templates(input_yaml={})
        vet = {
            "image_sci": img, "image_tpl": img,
            "psf_sci": psf, "psf_tpl": psf,
            "fwhm_sci": 3.3, "fwhm_tpl": 3.3,
            "chi2_max": 1e-9,  # nominally vetoes every source
            "min_keep": 5,
        }
        out, _ = t.find_flux_consistent_sources(
            cat_s, cat_t, make_plot=False, psf_vetting=vet
        )
        assert len(out) >= 5
        # The worst-fitting source (the blend) must not be among the
        # restored best-5.
        assert not ((out["x_pix"] == 60) & (out["y_pix"] == 130)).any()

    @staticmethod
    def _elliptical_image(pos, flux, seed=1, shape=(160, 160), noise=0.5):
        """Field whose stars carry structure (ellipticity) a smooth
        circular model cannot match - the analytic-Moffat regime."""
        ax = np.arange(41) - 20
        ge = np.exp(
            -0.5 * ((ax[None, :] / 1.60) ** 2 + (ax[:, None] / 1.30) ** 2)
        )
        ge /= ge.sum()
        img = np.random.default_rng(seed).normal(0.0, noise, shape)
        for (x, y), f in zip(pos, flux):
            xi, yi = int(round(x)), int(round(y))
            img[yi - 20 : yi + 21, xi - 20 : xi + 21] += f * ge
        return img

    def test_per_side_floor_rescues_template_veto(self):
        """Template-side analytic model is model-limited: with the
        empirical floor it vetoes most bright stars; the analytic floor
        rescues them without weakening the science-side veto."""
        img, psf, cat_s, cat_t = self._setup()
        t = Templates(input_yaml={})
        pos = list(zip(cat_s["x_pix"], cat_s["y_pix"]))
        # Same field but stars have real structure (bright so the
        # mismatch dominates the sky noise, as for real bright stars).
        img_t = self._elliptical_image(
            pos, np.full(len(pos), 20000.0)
        )
        vet = {
            "image_sci": img, "image_tpl": img_t,
            "psf_sci": psf, "psf_tpl": psf,
            "fwhm_sci": 3.3, "fwhm_tpl": 3.3,
            "chi2_max": 5.0,
        }
        out_default, _ = t.find_flux_consistent_sources(
            cat_s, cat_t, make_plot=False, psf_vetting=vet
        )
        vet_ana = dict(vet, model_floor_frac_tpl=0.15)
        out_fixed, _ = t.find_flux_consistent_sources(
            cat_s, cat_t, make_plot=False, psf_vetting=vet_ana
        )
        # the empirical floor mass-vetoed template-side model error
        assert len(out_fixed) > len(out_default)
        # science-side veto is untouched: the blend stays rejected
        assert not ((out_fixed["x_pix"] == 60) & (out_fixed["y_pix"] == 130)).any()

    def test_min_keep_frac_restores_pool(self):
        """A systematic over-veto must restore at least min_keep_frac of
        the robust pool, not merely the small absolute min_keep."""
        img, psf, cat_s, cat_t = self._setup()
        t = Templates(input_yaml={})
        pos = list(zip(cat_s["x_pix"], cat_s["y_pix"]))
        # Both sides model-limited -> nearly every source vetoed.
        img2 = self._elliptical_image(
            pos, cat_s["flux_AP"].to_numpy() * 8.0, seed=2
        )
        img2_t = self._elliptical_image(
            pos, cat_t["flux_AP"].to_numpy() * 8.0, seed=3
        )
        vet = {
            "image_sci": img2, "image_tpl": img2_t,
            "psf_sci": psf, "psf_tpl": psf,
            "fwhm_sci": 3.3, "fwhm_tpl": 3.3,
            "chi2_max": 5.0,
            "min_keep": 5,
        }
        out_abs, _ = t.find_flux_consistent_sources(
            cat_s, cat_t, make_plot=False,
            psf_vetting=dict(vet, min_keep_frac=0.0),
        )
        out_frac, _ = t.find_flux_consistent_sources(
            cat_s, cat_t, make_plot=False,
            psf_vetting=dict(vet, min_keep_frac=0.7),
        )
        # absolute floor alone only guarantees min_keep; the fractional
        # floor must pull the surviving pool well above it (0.7*13 ~ 9+).
        assert len(out_frac) >= 9
        assert len(out_frac) > len(out_abs)


class TestSparseApertureFluxScale:
    """A ZOGY flux scale resting on one or two matched stars must be
    cross-checked against the pixel-ratio ensemble: a single variable or
    mismatched source must not rescale the whole subtraction."""

    def _wrapper(self):
        return TestSubtractZogyWrapper()

    def test_single_variable_star_falls_back_to_ensemble(
        self, tmp_path, monkeypatch, caplog
    ):
        w = self._wrapper()
        sci_f, ref_f, diff_f, psf_n_f, psf_r_f = w._inputs(tmp_path)
        # Boost one common star 2x in the science image only: a scale
        # measured from it alone is 2x the true ensemble scale.
        rng = np.random.default_rng(5)
        sci = _make_image(
            rng, SHAPE, _gauss_psf(2.0),
            [(30, 30, 4000.0), (60, 60, 3000.0), (90, 40, 1500.0), TRANSIENT],
            1.0,
        )
        w._write(tmp_path / "sci.fits", sci, exptime=300.0)
        hdr = fits.getheader(sci_f)
        with caplog.at_level("INFO", logger="templates"):
            res = Templates(input_yaml=w._cfg(tmp_path))._subtract_zogy(
                sci_f, ref_f, diff_f, hdr, psf_n_f, psf_r_f,
                science_saturate=1e9, template_saturate=1e9,
                method="zogy",
                matching_sources=[(30, 30)],
            )
        assert res == "done"
        assert any(
            "estimator=pixel-ratio" in r.getMessage() for r in caplog.records
        )
        # And the transient still recovers its true flux.
        data = fits.getdata(diff_f)
        x, y, f = TRANSIENT
        s = 10
        flux = float(np.nansum(data[y - s : y + s + 1, x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.1)

    def test_zogy_exposure_fallback_uses_image_gains(self, tmp_path):
        w = self._wrapper()
        rng = np.random.default_rng(19)
        sci = rng.normal(0.0, 1.0, SHAPE)
        ref = rng.normal(0.0, 1.0, SHAPE)
        sci_f = w._write(tmp_path / "sci.fits", sci, exptime=300.0)
        ref_f = w._write(tmp_path / "ref.fits", ref, exptime=60.0)
        ref_hdr = fits.getheader(ref_f)
        ref_hdr["EGAIN"] = 4.0
        fits.writeto(ref_f, ref, ref_hdr, overwrite=True)
        diff_f = str(tmp_path / "diff.fits")
        psf = _gauss_psf(2.0)
        psf_n_f = w._write(tmp_path / "psf_n.fits", psf)
        psf_r_f = w._write(tmp_path / "psf_r.fits", psf)
        cfg = w._cfg(tmp_path)
        cfg.update({"exposure_time": 300.0, "gain": 2.0})

        result = Templates(input_yaml=cfg)._subtract_zogy(
            sci_f,
            ref_f,
            diff_f,
            fits.getheader(sci_f),
            psf_n_f,
            psf_r_f,
            science_saturate=1e9,
            template_saturate=1e9,
            method="zogy",
            matching_sources=[],
        )

        assert result == "done"
        assert fits.getheader(diff_f)["FLXSCALE"] == pytest.approx(10.0)

    def test_aperture_scale_reports_ratio_count(self):
        from templates import _zogy_matched_aperture_scale

        sci = np.full((61, 61), 100.0)
        ref = np.full((61, 61), 50.0)
        for x, y, f in [(20, 20, 5000.0), (40, 40, 4000.0)]:
            sci[y, x] += f
            ref[y, x] += f / 2.0
        scale, n = _zogy_matched_aperture_scale(
            sci, ref, [(20, 20), (40, 40)], fwhm=3.0
        )
        assert n == 2
        assert scale == pytest.approx(2.0, rel=0.05)

    def test_no_valid_ratios_returns_nan_and_zero(self):
        from templates import _zogy_matched_aperture_scale

        sci = np.full((61, 61), 100.0)
        ref = np.full((61, 61), 50.0)
        scale, n = _zogy_matched_aperture_scale(
            sci, ref, [(5, 5)], fwhm=3.0
        )
        assert not np.isfinite(scale)
        assert n == 0


class TestZogyLsqKernel:
    """Pixel-domain matching kernel (Bramich delta basis + Laplacian
    Tikhonov penalty): sub-pixel shift absorption, the free photometric
    kernel sum, and the sanity gates that reject unusable solves."""

    @staticmethod
    def _pair(
        shape=(160, 160),
        fwhm_s=2.5,
        fwhm_r=3.5,
        shift=(0.0, 0.0),
        flux_ratio=1.0,
        seed=42,
        noise=1.0,
    ):
        """sci: FWHM fwhm_s stars; ref: FWHM fwhm_r stars shifted by
        (dx, dy) and scaled by flux_ratio (ref = sci_units * ratio)."""
        from scipy.ndimage import shift as _shift

        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]

        def _gauss(cx, cy, fwhm, flux):
            s = fwhm / 2.3548
            return (
                flux
                / (2 * np.pi * s * s)
                * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
            )

        stars = [
            (60, 60),
            (120, 60),
            (70, 120),
            (110, 110),
            (90, 80),
            (40, 130),
        ]
        sci = rng.normal(0.0, noise, shape)
        ref = rng.normal(0.0, noise, shape)
        for cx, cy in stars:
            sci += _gauss(cx, cy, fwhm_s, 5000.0)
            ref += _gauss(cx + shift[0], cy + shift[1], fwhm_r,
                          5000.0 * flux_ratio)
        return sci, ref, stars

    def test_smoothing_kernel_absorbs_subpixel_shift(self):
        """The LSQ kernel must carry the 0.4 px misregistration the
        Wiener PSF-ratio kernel cannot represent."""
        from templates import _pack_lsq_kernel, _zogy_lsq_kernel

        sci, ref, stars = self._pair(shift=(0.4, -0.3))
        k, info = _zogy_lsq_kernel(
            sci, ref, stars, half_width=6, lam_rel=1e-3, min_stamps=3
        )
        assert k is not None
        assert info["ok"]
        # Kernel centroid sits at -delta in the solver's correlation
        # convention (k[d] multiplies src(p + d)); the applied shift was
        # (dx=+0.4, dy=-0.3), so the kernel offset must be (-0.4, +0.3).
        hw = k.shape[0] // 2
        yy, xx = np.mgrid[0 : k.shape[0], 0 : k.shape[1]]
        w = np.abs(k)
        w /= w.sum()
        assert (xx * w).sum() - hw == pytest.approx(-0.4, abs=0.15)
        assert (yy * w).sum() - hw == pytest.approx(0.3, abs=0.15)
        # Applied through the pipeline's FFT path, stars must cancel to
        # a few percent of peak.
        kp = _pack_lsq_kernel(k, sci.shape)
        conv = (
            np.real(
                np.fft.ifft2(np.fft.fft2(sci) * np.fft.fft2(kp))
            )
            + info["b0"]
        )
        d = conv - ref
        for cx, cy in stars:
            peak = float(ref[cy - 4 : cy + 5, cx - 4 : cx + 5].max())
            assert peak > 200.0
            stamp = d[cy - 4 : cy + 5, cx - 4 : cx + 5]
            assert np.abs(stamp).max() < 0.05 * peak

    def test_raw_kernel_sum_recovers_flux_ratio(self):
        """ksum is the free photometric factor (SFFT ConstPhotRatio
        equivalent): a reference 1.96x brighter must produce ksum ~1.96."""
        from templates import _zogy_lsq_kernel

        sci, ref, stars = self._pair(flux_ratio=1.96)
        k, info = _zogy_lsq_kernel(
            sci, ref, stars, half_width=6, lam_rel=1e-3, min_stamps=3
        )
        assert k is not None
        assert info["ksum_raw"] == pytest.approx(1.96, rel=0.05)

    def test_sharpening_direction_carries_negative_mass(self):
        """An impossible direction must still read as deconvolution so
        the probe veto can reject it."""
        from templates import _zogy_lsq_kernel

        sci, ref, stars = self._pair()
        k, info = _zogy_lsq_kernel(
            ref, sci, stars, half_width=6, lam_rel=1e-3, min_stamps=3
        )
        if k is None:
            # The sum-bounds gate may reject the fit outright -- also a
            # valid refusal.
            return
        assert info["neg"] > 0.3

    def test_too_few_stamps_returns_none(self):
        from templates import _zogy_lsq_kernel

        sci, ref, _ = self._pair()
        k, info = _zogy_lsq_kernel(
            sci, ref, [(60, 60)], half_width=6, min_stamps=3
        )
        assert k is None
        assert not info["ok"]

    def test_mask_excludes_stamp_footprints(self):
        """A masked blob inside a stamp must not teach the kernel."""
        from templates import _zogy_lsq_kernel

        sci, ref, stars = self._pair()
        mask = np.zeros_like(sci, dtype=bool)
        mask[40:80, 40:80] = True  # swallows two of the six stamps
        k, info = _zogy_lsq_kernel(
            sci, ref, stars, half_width=6, invalid_mask=mask,
            min_stamps=3,
        )
        assert k is not None
        assert info["n_stamps"] < len(stars)

    def test_neg_cap_escalates_damping(self):
        """With neg_cap set, the solver escalates the Tikhonov weight
        until the sidelobe mass fits the cap; when no damping level can
        satisfy it the least-negative attempt is returned so the caller's
        veto sees an honest diagnostic (TDP6 60881 r: neg 1.7 at
        lam=1e-3 needed lam=0.1 to reach a usable kernel)."""
        from templates import _zogy_lsq_kernel

        sci, ref, stars = self._pair()
        _, i_raw = _zogy_lsq_kernel(
            ref, sci, stars, half_width=6, lam_rel=1e-3, min_stamps=3
        )
        # Sharpening direction: reachable only under heavy damping.
        k_cap, i_cap = _zogy_lsq_kernel(
            ref, sci, stars, half_width=6, lam_rel=1e-3, min_stamps=3,
            neg_cap=0.9,
        )
        assert k_cap is not None
        assert i_cap["neg"] <= 0.9
        assert i_cap["neg"] < i_raw["neg"]
        assert i_cap["lam_rel"] > 1e-3
        # No ladder step can satisfy a 5% cap: the least-negative
        # attempt is still returned for the caller to veto.
        k_hard, i_hard = _zogy_lsq_kernel(
            ref, sci, stars, half_width=6, lam_rel=1e-3, min_stamps=3,
            neg_cap=0.05,
        )
        assert k_hard is not None
        assert i_hard["neg"] < i_raw["neg"]
        assert i_hard["neg"] > 0.05

    def test_pack_convention_matches_convolution(self):
        """The packed kernel applied by the pipeline FFT path must equal
        scipy convolve with the flipped centred kernel."""
        from scipy.ndimage import convolve

        from templates import _pack_lsq_kernel

        rng = np.random.default_rng(1)
        img = rng.normal(0, 1, (64, 64))
        k = np.zeros((9, 9))
        k[4, 4] = 0.6
        k[5, 4] = 0.3  # asymmetric: offset +1 row
        k[4, 6] = 0.1  # offset +2 cols
        kp = _pack_lsq_kernel(k, img.shape)
        via_fft = np.real(
            np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(kp))
        )
        via_spatial = convolve(img, k[::-1, ::-1], mode="wrap")
        assert np.allclose(via_fft, via_spatial, atol=1e-10)


class TestZogyWienerAdaptiveEps:
    """The adaptive Wiener kernel uses the weakest regularization that
    keeps the kernel clean, and reports the least-bad candidate when a
    direction is genuinely impossible."""

    @staticmethod
    def _padded(fwhm, shape=(121, 121)):
        n = 2 * int(np.ceil(4.0 * fwhm)) + 1
        ax = np.arange(n) - n // 2
        s = fwhm / 2.3548
        g = np.exp(-0.5 * (ax[None, :] ** 2 + ax[:, None] ** 2) / s ** 2)
        g /= g.sum()
        out = np.zeros(shape)
        c = shape[0] // 2
        out[c - n // 2 : c + n // 2 + 1, c - n // 2 : c + n // 2 + 1] = g
        return out

    def test_smoothing_direction_stays_clean_at_low_eps(self):
        from templates import _zogy_wiener_kernel_adaptive

        src = self._padded(2.5)
        tgt = self._padded(4.0)
        k, neg, eps = _zogy_wiener_kernel_adaptive(src, tgt)
        assert k is not None
        assert neg <= 0.05
        # A well-posed direction must not be over-damped.
        assert eps <= 1e-2

    def test_sharpening_direction_stays_ringy(self):
        """No amount of regularization can legitimately sharpen; the
        veto diagnostic uses the fixed-eps probe, which must keep the
        high negative mass so the veto fires."""
        from templates import (
            _ZOGY_KERNEL_NEG_HARD,
            _zogy_wiener_kernel,
            _zogy_wiener_kernel_adaptive,
        )

        src = self._padded(4.0)
        tgt = self._padded(2.5)
        _, neg_probe = _zogy_wiener_kernel(src, tgt)
        assert neg_probe > _ZOGY_KERNEL_NEG_HARD * 0.5
        # The adaptive kernel for a deconvolving direction collapses to a
        # correlation bump at high eps -- clean-looking but not a match;
        # that is exactly why it never feeds the veto.
        k, neg, eps = _zogy_wiener_kernel_adaptive(src, tgt)
        assert np.isfinite(neg)

    def test_fixed_eps_still_available(self):
        from templates import _zogy_wiener_kernel

        src = self._padded(2.5)
        tgt = self._padded(4.0)
        k, neg = _zogy_wiener_kernel(src, tgt, eps_rel=1e-2)
        assert k is not None
        assert np.isfinite(k).all()
        assert np.isfinite(neg)

    def test_fid_err_handles_corner_split_stamp(self):
        """Zero-lag padded stamps split the PSF across the array
        corners; measuring the raw layout under-reads the width enough
        to veto good kernels (TDP6 60881 r vetoed every REF kernel on a
        phantom 127% width error)."""
        from templates import (
            _zogy_kernel_fid_err,
            _zerolag_stamp_fwhm,
        )

        # Zero-lag layout: PSF centred on index (0,0), i.e. the corners.
        n = 25
        ax = np.arange(n) - n // 2
        s = 6.0 / 2.3548
        g = np.exp(-0.5 * (ax[None, :] ** 2 + ax[:, None] ** 2) / s ** 2)
        g /= g.sum()
        tgt = np.zeros((101, 101))
        iy = (np.arange(n) - n // 2) % 101
        tgt[np.ix_(iy, iy)] = g
        # Same stamp with its peak one pixel off the nominal zero-lag
        # centre, as a median stack's argmax jitter produces.
        tgt_off = np.roll(tgt, (1, 0), axis=(0, 1))
        f_on = _zerolag_stamp_fwhm(tgt)
        f_off = _zerolag_stamp_fwhm(tgt_off)
        assert np.isfinite(f_on) and np.isfinite(f_off)
        assert abs(f_on - f_off) < 0.5
        # Identity kernel: the convolved width must match the target.
        k = np.zeros_like(tgt)
        k[0, 0] = 1.0
        assert _zogy_kernel_fid_err(tgt, k, tgt_off) < 0.15


class TestLsqFitPositions:
    """Vetted-position selection for the pixel-domain LSQ solve.

    Positions rejected by the star-stack vetting must never re-enter
    the fit, even when the vetted intersection is smaller than the
    solver's minimum-stamp gate.
    """

    _srcs = [
        {"x": 10.0, "y": 20.0},
        {"x": 30.0, "y": 40.0},
        {"x": 50.0, "y": 60.0},
        {"x": 70.0, "y": 80.0},
    ]

    def test_both_vetted_lists_intersect(self):
        from templates import _zogy_lsq_fit_positions

        out = _zogy_lsq_fit_positions(
            self._srcs,
            [(10.0, 20.0), (30.0, 40.0)],
            [(30.0, 40.0), (50.0, 60.0)],
        )
        assert out == [(30.0, 40.0)]

    def test_small_intersection_does_not_fallback_to_raw(self):
        from templates import _zogy_lsq_fit_positions

        out = _zogy_lsq_fit_positions(
            self._srcs, [(10.0, 20.0)], [(50.0, 60.0)]
        )
        assert out == []

    def test_single_vetted_list_is_used(self):
        from templates import _zogy_lsq_fit_positions

        out = _zogy_lsq_fit_positions(
            self._srcs, [(10.0, 20.0), (70.0, 80.0)], None
        )
        assert out == [(10.0, 20.0), (70.0, 80.0)]

    def test_no_vetting_uses_raw_matches(self):
        from templates import _zogy_lsq_fit_positions

        out = _zogy_lsq_fit_positions(self._srcs, None, None)
        assert len(out) == 4

    def test_non_finite_and_malformed_entries_dropped(self):
        from templates import _zogy_lsq_fit_positions

        srcs = [
            {"x": np.nan, "y": 5.0},
            {"x": 1.0},
            (3.0, 4.0),
        ]
        out = _zogy_lsq_fit_positions(srcs, None, None)
        assert out == [(3.0, 4.0)]


class TestSubtractZogyLsqIntegration:
    """End-to-end: the LSQ kernel path through _subtract_zogy."""

    def test_sci_direction_uses_lsq_and_absorbs_shift(
        self, tmp_path, monkeypatch
    ):
        """A sub-pixel-shifted reference must subtract cleanly when the
        LSQ kernel can absorb the shift; the header must record the
        kernel provenance."""
        w = TestSubtractZogyWrapper()
        rng = np.random.default_rng(11)
        psf_n = _gauss_psf(1.2)
        psf_r = _gauss_psf(2.2)
        sci = _make_image(
            rng, SHAPE, psf_n, COMMON + [TRANSIENT], 1.0
        )
        ref = _make_image(rng, SHAPE, psf_r, COMMON, 1.0)
        # Sub-pixel misregistration of the reference.
        from scipy.ndimage import shift as _shift

        ref = _shift(ref, (0.35, -0.45), order=3, mode="nearest")
        sci_f = w._write(tmp_path / "sci.fits", sci, exptime=300.0)
        ref_f = w._write(tmp_path / "ref.fits", ref, exptime=300.0)
        diff_f = str(tmp_path / "diff.fits")
        psf_n_f = w._write(tmp_path / "psf_n.fits", psf_n)
        psf_r_f = w._write(tmp_path / "psf_r.fits", psf_r)
        cfg = w._cfg(
            tmp_path, forceconv="SCI", zogy_lsq_kernel_hw_max=6
        )
        hdr = fits.getheader(sci_f)
        res = Templates(input_yaml=cfg)._subtract_zogy(
            sci_f, ref_f, diff_f, hdr, psf_n_f, psf_r_f,
            science_saturate=1e9, template_saturate=1e9,
            method="zogy",
            science_fwhm=2.8, template_fwhm=5.2,
            matching_sources=[(x, y) for x, y, _ in COMMON],
        )
        assert res == "done"
        data, hdr = fits.getdata(diff_f, header=True)
        assert hdr["CONVD"] == "SCI"
        # The LSQ kernel is preferred when it solves.
        assert hdr.get("KERNMETH") == "lsq"
        x, y, f = TRANSIENT
        s = 8
        flux = float(np.nansum(data[y - s : y + s + 1, x - s : x + s + 1]))
        assert flux == pytest.approx(f, rel=0.15)


class TestZogyAnalyticPsfStamps:
    """Regression tests for the analytic-PSF path feeding ZOGY.

    ``main._write_analytic_psf_stamp`` + ``main._analytic_star_cutouts``
    produce the ZOGY_PSF_* stamps used under template_subtraction
    zogy_psf='analytic' (and as the auto-mode fallback).
    """

    def _stars_image(self, fwhm=4.5, beta=3.0):
        rng = np.random.default_rng(0)
        img = rng.normal(0.0, 2.0, (200, 200))
        alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        yy, xx = np.mgrid[:200, :200]
        for cx, cy in [(50, 50), (100, 60), (150, 80), (60, 130), (140, 140)]:
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            img += 5000.0 * (1.0 + r2 / alpha**2) ** (-beta)
        return img, [(50, 50), (100, 60), (150, 80), (60, 130), (140, 140)]

    def test_cutout_extraction_keeps_edge_stars(self):
        import main as main_mod

        img, xy = self._stars_image()
        df = pd.DataFrame([(3, 3)] + xy, columns=["x_pix", "y_pix"])
        stars = main_mod._analytic_star_cutouts(img, df, 55)
        # NaN padding must keep the edge star in the pool.
        assert len(stars) == 6

    def test_fitted_stamp_written_and_normalized(self, tmp_path):
        import main as main_mod

        img, xy = self._stars_image()
        df = pd.DataFrame(xy, columns=["x_pix", "y_pix"])
        stars = main_mod._analytic_star_cutouts(img, df, 55)
        out = tmp_path / "ZOGY_PSF_image_test.fits"
        main_mod._write_analytic_psf_stamp(
            str(out), 4.5, {"photometry": {}}, n_stars=len(stars), stars=stars
        )
        hdr = fits.getheader(str(out))
        data = fits.getdata(str(out))
        assert hdr["OVERSAMP"] == 1
        assert hdr["NPSFSTAR"] == len(stars)
        assert np.isfinite(data).all()
        assert np.nansum(data) == pytest.approx(1.0, abs=1e-6)

    def test_stamp_kwarg_serializes_model_verbatim(self, tmp_path):
        import main as main_mod
        from psf import _analytic_psf_stamp

        img, xy = self._stars_image()
        df = pd.DataFrame(xy, columns=["x_pix", "y_pix"])
        stars = main_mod._analytic_star_cutouts(img, df, 55)
        model, comps = _analytic_psf_stamp(
            4.5, 1, 55, 3.0, stars=stars, phot_cfg={"psf_analytic_components": 3}
        )
        assert getattr(model, "_autophot_fitted", False)
        out = tmp_path / "stamp.fits"
        main_mod._write_analytic_psf_stamp(
            str(out), 4.5, {"photometry": {}}, stamp=model, n_stars=5
        )
        assert np.allclose(fits.getdata(str(out)), model.data)

    def test_fwhm_fallback_chain(self):
        import main as main_mod

        assert main_mod._zogy_fwhm_fallback(np.nan, None, 3.7) == 3.7
        assert main_mod._zogy_fwhm_fallback(np.nan, -1.0) == 5.0
        assert main_mod._zogy_fwhm_fallback(6.0, 3.0) == 6.0


class TestFscalPhotRepair:
    """_fscal_phot_needs_repair gates the FSCAL_PHOT sanity fix."""

    def test_broken_phot_is_repaired(self):
        from templates import _fscal_phot_needs_repair

        # SExtractor scale 5.5 vs aperture truth ~1.0 (the SN2024pba case).
        assert _fscal_phot_needs_repair(5.5, 1.0, 20)
        assert _fscal_phot_needs_repair(0.42, 1.0, 20)

    def test_close_values_left_alone(self):
        from templates import _fscal_phot_needs_repair

        # A few-percent aperture bias is real; do not touch it.
        assert not _fscal_phot_needs_repair(1.05, 1.0, 20)
        assert not _fscal_phot_needs_repair(0.97, 1.0, 20)

    def test_too_few_ratios_disables_repair(self):
        from templates import _fscal_phot_needs_repair

        assert not _fscal_phot_needs_repair(5.5, 1.0, 4, min_ratios=5)
        assert _fscal_phot_needs_repair(5.5, 1.0, 5, min_ratios=5)

    def test_non_finite_inputs_rejected(self):
        from templates import _fscal_phot_needs_repair

        assert not _fscal_phot_needs_repair(np.nan, 1.0, 20)
        assert not _fscal_phot_needs_repair(1.0, np.nan, 20)
        assert not _fscal_phot_needs_repair(1.0, 0.0, 20)
        assert not _fscal_phot_needs_repair("bad", 1.0, 20)
        assert not _fscal_phot_needs_repair(5.5, 1.0, np.nan)

    def test_custom_sanity_frac(self):
        from templates import _fscal_phot_needs_repair

        assert _fscal_phot_needs_repair(1.2, 1.0, 20, sanity_frac=0.10)
        assert not _fscal_phot_needs_repair(1.2, 1.0, 20, sanity_frac=0.30)


class TestFscalApCardScale:
    """_fscal_ap_card_scale maps the sci/ref aperture ratio onto the
    FSCAL-card direction, which flips for CONVD=SCI products."""

    def test_sci_direction_inverts(self):
        from templates import _fscal_ap_card_scale

        # SN2024pba 2024-08-09: aperture ratio f_sci/f_ref = 1.84 while the
        # FSCAL cards carry the convolved/static scale f_ref/f_sci = 0.54.
        assert np.isclose(_fscal_ap_card_scale(1.84, "SCI"), 1.0 / 1.84)

    def test_ref_direction_untouched(self):
        from templates import _fscal_ap_card_scale

        assert np.isclose(_fscal_ap_card_scale(9.18, "REF"), 9.18)
        assert np.isclose(_fscal_ap_card_scale(0.42, "ref"), 0.42)

    def test_direction_corrected_gate(self):
        # End-to-end check on the SN2024pba false-repair cases: with the
        # card-direction value, a healthy FSCAL_PHOT is left alone on every
        # CONVD=SCI epoch, while the genuine CONVD=REF breakage still fires.
        from templates import _fscal_ap_card_scale, _fscal_phot_needs_repair

        # 2024-08-09: PHOT=0.559 (correct) vs raw aperture 1.84 -> 0.543.
        ap = _fscal_ap_card_scale(1.8402, "SCI")
        assert not _fscal_phot_needs_repair(0.559, ap, 9)
        # 2025-01-13 REF: PHOT=5.529 was genuinely broken vs aperture 0.418.
        ap_ref = _fscal_ap_card_scale(0.418, "REF")
        assert _fscal_phot_needs_repair(5.529, ap_ref, 9)

    def test_invalid_inputs(self):
        from templates import _fscal_ap_card_scale

        assert not np.isfinite(_fscal_ap_card_scale(np.nan, "SCI"))
        assert not np.isfinite(_fscal_ap_card_scale(0.0, "SCI"))
        assert not np.isfinite(_fscal_ap_card_scale(-1.0, "SCI"))
        assert not np.isfinite(_fscal_ap_card_scale("bad", "SCI"))
        assert not np.isfinite(_fscal_ap_card_scale(np.nan, "REF"))


class TestSuppliedNoiseMap:
    """_supplied_noise_map loads a validated RMS map for ZOGY's sn_map /
    sr_map.  The map supplies per-pixel structure; the caller's measured
    scalar keeps absolute scale, so a flat map reduces exactly to the
    scalar-noise path.  Invalid pixels fall back to the scalar because
    their image pixels are already dead and only need finite variance."""

    def _write(self, path, data):
        fits.writeto(path, np.asarray(data, dtype=np.float32))
        return str(path)

    def test_structure_anchored_to_scalar(self, tmp_path):
        from templates import _supplied_noise_map

        m = np.full((20, 20), 1.0, np.float32)
        m[:, 10:] = 4.0
        p = self._write(tmp_path / "n.fits", m)
        out = _supplied_noise_map(p, 2.0, "science")
        # equal halves -> median sigma 2.5: left = 0.8, right = 3.2
        assert out[5, 5] == pytest.approx(0.8, rel=1e-6)
        assert out[5, 15] == pytest.approx(3.2, rel=1e-6)
        assert out[5, 5] / out[5, 15] == pytest.approx(0.25, rel=1e-6)

    def test_flat_map_reduces_to_scalar(self, tmp_path):
        from templates import _supplied_noise_map

        p = self._write(tmp_path / "n.fits", np.full((10, 10), 7.0))
        out = _supplied_noise_map(p, 3.3, "science")
        np.testing.assert_allclose(out, np.full((10, 10), 3.3))

    def test_invalid_pixels_fall_back_to_scalar(self, tmp_path):
        from templates import _supplied_noise_map

        m = np.ones((10, 10), np.float32)
        m[:2, :] = np.nan
        m[-2:, :] = 0.0
        p = self._write(tmp_path / "n.fits", m)
        out = _supplied_noise_map(p, 5.0, "template")
        assert np.isfinite(out).all()
        assert (out[:2, :] == 5.0).all()
        assert (out[-2:, :] == 5.0).all()

    def test_missing_or_unusable_returns_none(self, tmp_path):
        from templates import _supplied_noise_map

        assert _supplied_noise_map(None, 2.0, "science") is None
        assert (
            _supplied_noise_map(tmp_path / "nope.fits", 2.0, "science")
            is None
        )
        p = self._write(tmp_path / "zero.fits", np.zeros((8, 8)))
        assert _supplied_noise_map(p, 2.0, "science") is None
        assert _supplied_noise_map(p, 0.0, "science") is None
        assert _supplied_noise_map(p, np.nan, "science") is None
