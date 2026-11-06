"""CR mask split: source-exclusion halo vs pixel-defect mask.

The dilated mask is a source-exclusion halo (keeps stars whose stamps
overlap repaired pixels out of fit pools).  The defect mask marks the
pixels that are not real data - flagged pixels plus a small margin -
and is what the subtraction/aperture masks should null.  Regressed on
GROND_g_TDP7_cor_ana_5_MJD61233_28873, where 110 flagged pixels were
inflated into 20968 masked pixels in the difference image.
"""

import numpy as np
import pytest
from astropy.io import fits
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.morphology import disk

import cosmic
from functions import drop_mask_components_at_positions


def _run(monkeypatch, raw_mask, **kwargs):
    rng = np.random.default_rng(42)
    image = (100.0 + rng.normal(0, 5, raw_mask.shape)).astype(np.float32)
    header = fits.Header()
    header["RDNOISE"] = 5.0

    def fake_detect(*args, **kw):
        return raw_mask.copy(), image.copy()

    monkeypatch.setattr(cosmic.astroscrappy, "detect_cosmics", fake_detect)
    cleaner = cosmic.RemoveCosmicRays(
        input_yaml={}, fpath="dummy.fits", image=image, header=header
    )
    return cleaner.remove(plot=False, **kwargs)


@pytest.fixture
def raw_mask():
    m = np.zeros((200, 200), dtype=bool)
    m[50, 50] = True          # isolated single-pixel hit
    m[100:103, 120] = True    # short 3-px track
    m[150, 160] = True
    return m


def test_returns_three_masks(monkeypatch, raw_mask):
    clean, exclusion, defect = _run(monkeypatch, raw_mask, psf_fwhm=8.0)
    assert clean.shape == raw_mask.shape
    assert exclusion.dtype == bool and defect.dtype == bool


def test_defect_mask_is_small_margin_only(monkeypatch, raw_mask):
    _, exclusion, defect = _run(
        monkeypatch, raw_mask, psf_fwhm=8.0, defect_margin_px=2.0
    )
    expected = binary_fill_holes(
        binary_dilation(raw_mask, structure=disk(2))
    )
    np.testing.assert_array_equal(defect, expected)
    assert np.count_nonzero(defect) <= 60


def test_exclusion_halo_is_larger(monkeypatch, raw_mask):
    _, exclusion, defect = _run(
        monkeypatch, raw_mask, psf_fwhm=8.0, defect_margin_px=2.0
    )
    assert np.count_nonzero(exclusion) > 5 * np.count_nonzero(defect)
    # defect mask is a subset of the exclusion halo
    assert np.count_nonzero(defect & ~exclusion) == 0


def test_zero_margin_returns_raw(monkeypatch, raw_mask):
    _, _, defect = _run(
        monkeypatch, raw_mask, psf_fwhm=8.0, defect_margin_px=0.0
    )
    np.testing.assert_array_equal(defect, raw_mask)


def test_empty_detection(monkeypatch):
    empty = np.zeros((100, 100), dtype=bool)
    clean, exclusion, defect = _run(monkeypatch, empty, psf_fwhm=5.0)
    assert not defect.any() and not exclusion.any()


def test_protect_positions_reach_detector_inmask(monkeypatch):
    rng = np.random.default_rng(0)
    image = (100.0 + rng.normal(0, 5, (200, 200))).astype(np.float32)
    captured = {}

    def fake_detect(*args, **kw):
        captured["inmask"] = np.asarray(kw["inmask"])
        return np.zeros(image.shape, bool), image.copy()

    monkeypatch.setattr(cosmic.astroscrappy, "detect_cosmics", fake_detect)
    cleaner = cosmic.RemoveCosmicRays(
        input_yaml={}, fpath="d.fits", image=image, header=fits.Header()
    )
    cleaner.remove(
        plot=False,
        protect_positions=[(100.0, 100.0)],
        protect_radius_px=4.0,
    )
    inmask = captured["inmask"]
    assert inmask[100, 100] and inmask[103, 100]
    assert not inmask[100, 108]
    assert not inmask[50, 50]


def _undersampled_source_image(seed=7):
    # Faint, sharp point source on a noisy high background: only ~2 px
    # rise above the auto-protect threshold, so without explicit
    # protection astroscrappy flags it as a CR cluster.  This is the
    # SN2025abkv failure mode - the isolated target was interpolated and
    # its defect flag nulled the difference image.
    rng = np.random.default_rng(seed)
    image = (700.0 + rng.normal(0, 15, (120, 120))).astype(np.float32)
    yy, xx = np.mgrid[:120, :120]
    image += (
        300.0
        * np.exp(-((xx - 60.3) ** 2 + (yy - 59.7) ** 2) / (2 * 0.55**2))
    ).astype(np.float32)
    image[20, 20] += 5000.0  # genuine CR spike, must stay flagged
    return image


def _run_real_detect(image, **kwargs):
    header = fits.Header()
    header["RDNOISE"] = 5.0
    cleaner = cosmic.RemoveCosmicRays(
        input_yaml={}, fpath="d.fits", image=image.copy(), header=header
    )
    return cleaner.remove(
        bkg=np.full_like(image, 700.0),
        invar=np.full_like(image, 15.0**2 + 700.0),
        gain=1.0,
        readnoise=5.0,
        psf_fwhm=3.5,
        defect_margin_px=2.0,
        plot=False,
        **kwargs,
    )


def test_protected_source_keeps_flux():
    image = _undersampled_source_image()
    clean_u, _, defect_u = _run_real_detect(image)
    # Baseline: without protection the detector flags the source.
    assert defect_u[58:63, 58:63].any()
    assert clean_u[60, 60] < image[60, 60] - 100.0

    clean_p, _, defect_p = _run_real_detect(
        image,
        protect_positions=[(60.3, 59.7)],
        protect_radius_px=4.0,
    )
    assert not defect_p[58:63, 58:63].any()
    np.testing.assert_allclose(
        clean_p[58:63, 58:63], image[58:63, 58:63], rtol=0, atol=1e-5
    )
    # The genuine CR spike is still detected and masked.
    assert defect_p[18:23, 18:23].any()
    assert clean_p[20, 20] < image[20, 20] - 1000.0


class TestDropMaskComponentsAtPositions:
    """Component-level rescue for suspect flags covering known sources."""

    def test_drops_small_component_at_position(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:23, 20:23] = True  # 9-px flag blob on the target
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150
        )
        assert not out.any()
        assert dropped == [(21.0, 21.0)]

    def test_keeps_large_component(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[15:35, 15:35] = True  # 400-px region covering the target
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150
        )
        np.testing.assert_array_equal(out, mask)
        assert dropped == []

    def test_keeps_component_over_nonfinite_pixel(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:23, 20:23] = True
        image = np.full((50, 50), 100.0)
        image[21, 21] = np.nan  # genuinely missing data, not a flag
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150, reference_image=image
        )
        np.testing.assert_array_equal(out, mask)
        assert dropped == []

    def test_leaves_unrelated_components(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:23, 20:23] = True   # on target -> dropped
        mask[40, 40] = True         # unrelated single pixel -> kept
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150
        )
        assert not out[20:23, 20:23].any()
        assert out[40, 40]

    def test_skips_invalid_positions(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:23, 20:23] = True
        out, dropped = drop_mask_components_at_positions(
            mask,
            [(np.nan, 21.0), (200.0, 200.0), (21.0, 21.0)],
            max_area_px=150,
        )
        assert not out.any()
        assert dropped == [(21.0, 21.0)]

    def test_position_off_component_is_ignored(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[40, 40] = True
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150
        )
        np.testing.assert_array_equal(out, mask)
        assert dropped == []

    def test_extent_check_keeps_elongated_component(self):
        # A 35-px line through the target fits the area cap but is not
        # compact - a real defect fragment must survive the rescue.
        mask = np.zeros((50, 50), dtype=bool)
        mask[21, 10:45] = True
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150, max_extent_px=6.0
        )
        np.testing.assert_array_equal(out, mask)
        assert dropped == []

    def test_extent_check_drops_compact_component(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[19:24, 19:24] = True  # 25-px blob, furthest pixel ~2.8 px
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150, max_extent_px=4.0
        )
        assert not out.any()
        assert dropped == [(21.0, 21.0)]

    def test_invalid_extent_falls_back_to_area_only(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:23, 20:23] = True
        out, dropped = drop_mask_components_at_positions(
            mask, [(21.0, 21.0)], max_area_px=150, max_extent_px=np.nan
        )
        assert not out.any()
        assert dropped == [(21.0, 21.0)]


def test_protect_radius_capped_for_runaway_fwhm(monkeypatch):
    # A degenerate (huge) radius must not shield a large patch: the
    # cap keeps the disk at 25 px.
    rng = np.random.default_rng(0)
    image = (100.0 + rng.normal(0, 5, (200, 200))).astype(np.float32)
    captured = {}

    def fake_detect(*args, **kw):
        captured["inmask"] = np.asarray(kw["inmask"])
        return np.zeros(image.shape, bool), image.copy()

    monkeypatch.setattr(cosmic.astroscrappy, "detect_cosmics", fake_detect)
    cleaner = cosmic.RemoveCosmicRays(
        input_yaml={}, fpath="d.fits", image=image, header=fits.Header()
    )
    cleaner.remove(
        plot=False,
        protect_positions=[(100.0, 100.0)],
        protect_radius_px=200.0,
    )
    inmask = captured["inmask"]
    assert inmask[100, 100]
    assert np.count_nonzero(inmask) <= int(np.ceil(np.pi * 25.0**2)) + 8
