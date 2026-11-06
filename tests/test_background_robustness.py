"""Regression tests for BackgroundSubtractor input validation and edge cases.

Covers failure modes that previously crashed deep in the pipeline or
silently produced wrong masks:

* ``_make_source_mask`` with ``n_iterations <= 0`` (unbound ``iteration``).
* Missing required config keys (``saturate`` unconditionally, ``fpath``
  when plotting).
* Mismatched ``mask`` / ``precomputed_rms`` shapes and non-2-D images.
* ``masked_simbad_galaxies`` with ``header=None`` or malformed catalogs.
* Invalid saturation levels (zero, negative, NaN) masking the whole image.
* Fully non-finite local cutouts in ``remove_local_surface``.
* ``initial_mask`` seeding of the iterative source-mask loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from background import BackgroundSubtractor


@pytest.fixture
def flat_image():
    rng = np.random.default_rng(0)
    return 100.0 + rng.normal(0.0, 1.0, size=(128, 128))


@pytest.fixture
def base_config(tmp_path):
    return {
        "saturate": 50000.0,
        "fpath": str(tmp_path / "sci.fits"),
        "background": {},
        "photometry": {},
    }


class TestMaskingLoopEdgeCases:
    def test_zero_iterations_does_not_crash(self, flat_image):
        sub = BackgroundSubtractor({})
        mask = sub._make_source_mask(flat_image, n_iterations=0)
        assert mask.shape == flat_image.shape
        assert not mask.any()

    def test_negative_iterations_does_not_crash(self, flat_image):
        sub = BackgroundSubtractor({})
        mask = sub._make_source_mask(flat_image, n_iterations=-3)
        assert not mask.any()

    def test_initial_mask_preserved(self, flat_image):
        sub = BackgroundSubtractor({})
        init = np.zeros(flat_image.shape, dtype=bool)
        init[:10, :] = True
        mask = sub._make_source_mask(flat_image, n_iterations=1, initial_mask=init)
        assert mask[:10, :].all()

    def test_initial_mask_returned_with_zero_iterations(self, flat_image):
        sub = BackgroundSubtractor({})
        init = np.zeros(flat_image.shape, dtype=bool)
        init[:10, :] = True
        mask = sub._make_source_mask(flat_image, n_iterations=0, initial_mask=init)
        assert np.array_equal(mask, init)
        # Caller array must not be aliased - the loop mutates the mask.
        assert mask is not init

    def test_initial_mask_wrong_shape_raises(self, flat_image):
        sub = BackgroundSubtractor({})
        with pytest.raises(ValueError, match="initial_mask"):
            sub._make_source_mask(
                flat_image,
                n_iterations=1,
                initial_mask=np.zeros((4, 4), dtype=bool),
            )

    def test_zero_iterations_via_local_config(self, flat_image, base_config):
        cfg = dict(base_config)
        cfg["background"] = {"local_source_mask_iterations": 0}
        sub = BackgroundSubtractor(cfg)
        image_sub, bkg_local, rms, nn_meta = sub.remove_local_surface(
            flat_image, 64, 64, box_half_size=30, plot=False
        )
        assert image_sub.shape == flat_image.shape
        assert bkg_local.shape == (60, 60)


class TestConfigValidation:
    def test_missing_saturate_raises_keyerror(self, flat_image):
        sub = BackgroundSubtractor({})
        with pytest.raises(KeyError, match="saturate"):
            sub.remove(flat_image, plot=False)

    def test_missing_fpath_raises_only_when_plotting(self, flat_image):
        sub = BackgroundSubtractor({"saturate": 50000.0})
        with pytest.raises(KeyError, match="fpath"):
            sub.remove(flat_image, plot=True)

    def test_non_dict_config_raises_typeerror(self, flat_image):
        sub = BackgroundSubtractor(["not", "a", "dict"])
        with pytest.raises(TypeError):
            sub.remove(flat_image, plot=False)

    def test_local_missing_fpath_with_plot(self, flat_image):
        # remove_local_surface defaults plot=True, so a missing fpath used
        # to surface as a bare KeyError at the end of the fit.
        sub = BackgroundSubtractor({})
        with pytest.raises(KeyError, match="fpath"):
            sub.remove_local_surface(flat_image, 64, 64, plot=True)


class TestShapeValidation:
    def test_bad_mask_shape(self, flat_image, base_config):
        sub = BackgroundSubtractor(base_config)
        with pytest.raises(ValueError, match="mask"):
            sub.remove(
                flat_image,
                mask=np.zeros((10, 10), dtype=bool),
                plot=False,
            )

    def test_bad_precomputed_rms_shape(self, flat_image, base_config):
        sub = BackgroundSubtractor(base_config)
        with pytest.raises(ValueError, match="precomputed_rms"):
            sub.remove_local_surface(
                flat_image,
                64,
                64,
                plot=False,
                precomputed_rms=np.zeros((5, 5)),
            )

    def test_non_2d_image(self, base_config):
        sub = BackgroundSubtractor(base_config)
        with pytest.raises(ValueError):
            sub.remove(np.zeros((4, 4, 4)), plot=False)

    def test_empty_image(self, base_config):
        sub = BackgroundSubtractor(base_config)
        with pytest.raises(ValueError):
            sub.remove(np.zeros((0, 0)), plot=False)

    def test_non_2d_image_local(self, base_config):
        sub = BackgroundSubtractor(base_config)
        with pytest.raises(ValueError):
            sub.remove_local_surface(np.zeros((4, 4, 4)), 1, 1, plot=False)


class TestGalaxyMaskGuards:
    def test_missing_columns_returns_empty_mask(self, flat_image):
        sub = BackgroundSubtractor({})
        galaxies = pd.DataFrame({"RA": [150.0], "DEC": [30.0]})
        mask = sub.masked_simbad_galaxies(galaxies, flat_image, None)
        assert mask.shape == flat_image.shape
        assert not mask.any()

    def test_header_none_returns_empty_mask(self, flat_image):
        sub = BackgroundSubtractor({})
        galaxies = pd.DataFrame(
            {
                "RA": [150.0],
                "DEC": [30.0],
                "galdim_majaxis": [1.0],
                "galdim_minaxis": [0.5],
                "galdim_angle": [45.0],
            }
        )
        mask = sub.masked_simbad_galaxies(galaxies, flat_image, None)
        assert mask.shape == flat_image.shape
        assert not mask.any()

    def test_non_dataframe_returns_empty_mask(self, flat_image):
        sub = BackgroundSubtractor({})
        mask = sub.masked_simbad_galaxies([{"RA": 1.0}], flat_image, None)
        assert not mask.any()

    def test_get_rotation_angle_none_header(self):
        sub = BackgroundSubtractor({})
        assert sub.get_rotation_angle(None) == 0.0


class TestSaturationGuards:
    @pytest.mark.parametrize("saturate", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_saturate_masks_nothing(self, flat_image, saturate):
        sub = BackgroundSubtractor({})
        mask = sub._make_saturation_mask(flat_image, saturate)
        assert not mask.any()

    def test_streak_mask_invalid_saturate(self, flat_image):
        sub = BackgroundSubtractor({})
        assert not sub._make_saturation_streak_mask(flat_image, 0.0).any()
        assert not sub._make_saturation_streak_mask(flat_image, np.nan).any()

    def test_streak_mask_all_nan_image(self):
        sub = BackgroundSubtractor({})
        nan_img = np.full((32, 32), np.nan)
        assert not sub._make_saturation_streak_mask(nan_img, 100.0).any()

    def test_check_saturation_invalid_returns_inf(self):
        sub = BackgroundSubtractor({})
        assert np.isinf(sub._check_saturation(100.0, 0.0))
        assert np.isinf(sub._check_saturation(100.0, -5.0))
        assert np.isinf(sub._check_saturation(100.0, np.nan))


class TestAdaptiveBox:
    def test_nonpositive_init_box_clamped(self):
        sub = BackgroundSubtractor({})
        mask = np.zeros((64, 64), dtype=bool)
        # With an unmasked image the function returns early; the clamp just
        # guarantees the result is a valid odd box.
        bx = sub._adaptive_box_for_mask(mask, (-5, -5))
        assert bx[0] >= 1 and bx[0] % 2 == 1


class TestRemoveEndToEnd:
    def test_saturate_zero_does_not_abort(self, flat_image, base_config):
        """saturate=0 used to mask the whole image -> RuntimeError."""
        cfg = dict(base_config)
        cfg["saturate"] = 0.0
        sub = BackgroundSubtractor(cfg)
        result = sub.remove(flat_image, plot=False)
        assert result["image"].shape == flat_image.shape
        assert not result["defects_mask"].all()
        assert np.isinf(cfg["saturate"])

    def test_remove_basic_keys(self, flat_image, base_config):
        sub = BackgroundSubtractor(base_config)
        result = sub.remove(flat_image, plot=False)
        for key in (
            "image",
            "background",
            "background_rms",
            "defects_mask",
            "hardware_defects_mask",
            "source_mask",
        ):
            assert key in result
            assert result[key].shape == flat_image.shape

    def test_remove_seeds_source_mask_with_defects(
        self, nan_region_image, base_config, monkeypatch
    ):
        seen = {}
        orig = BackgroundSubtractor._make_source_mask

        def spy(self, image, *args, initial_mask=None, **kwargs):
            seen["initial_mask"] = initial_mask
            return orig(self, image, *args, initial_mask=initial_mask, **kwargs)

        monkeypatch.setattr(BackgroundSubtractor, "_make_source_mask", spy)
        sub = BackgroundSubtractor(base_config)
        img = nan_region_image["image"]
        sub.remove(img, plot=False)

        initial = seen.get("initial_mask")
        assert initial is not None
        nan_px = ~np.isfinite(img)
        assert nan_px.any()
        assert initial[nan_px].all()


class TestLocalCutoutGuards:
    def test_all_nan_cutout_returns_nan_surface(self, flat_image, base_config):
        sub = BackgroundSubtractor(base_config)
        img = flat_image.copy()
        img[50:80, 50:80] = np.nan
        rms_in = np.full(img.shape, 5.0)
        image_sub, bkg_local, rms, nn_meta = sub.remove_local_surface(
            img,
            64,
            64,
            box_half_size=12,
            plot=False,
            precomputed_rms=rms_in,
        )
        # Cutout (52:76, 52:76) sits fully inside the NaN region.
        assert bkg_local.shape == (24, 24)
        assert np.isnan(bkg_local).all()
        assert np.isnan(image_sub[50:80, 50:80]).all()
        assert np.array_equal(rms, rms_in)
        assert nn_meta["lift"] == 0.0
        assert nn_meta["box"] == (52, 76, 52, 76)

    def test_masked_core_surface_interpolated_not_flattened(
        self, base_config
    ):
        # The central masked region used to be flattened to a constant ring
        # level, leaving a visible plateau bump in the map.  The corrected
        # behaviour keeps the mesh-interpolated surface under the mask and
        # only clamps interpolation overshoot.
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:128, 0:128]
        img = (
            100.0
            + 0.05 * (xx - 64.0)
            + 0.04 * (yy - 64.0)
            + rng.normal(0.0, 1.0, size=(128, 128))
        )
        img += 800.0 * np.exp(-((xx - 64.0) ** 2 + (yy - 64.0) ** 2) / (2 * 1.5**2))
        sub = BackgroundSubtractor(base_config)
        _, surf, _, _ = sub.remove_local_surface(
            img,
            64,
            64,
            box_half_size=50,
            fwhm_pixels=3.0,
            exclude_inner_radius=6.0,
            plot=False,
        )
        rr = np.hypot(xx[14:114, 14:114] - 50.0, yy[14:114, 14:114] - 50.0)
        inner = surf[rr < 6.0]
        # Interpolated, not constant: the tilted sky continues under the mask.
        assert np.nanstd(inner) > 0.01
        assert np.isfinite(inner).all()
        # But still bounded: no runaway bowl relative to the surrounding ring.
        ring = surf[(rr >= 9.5) & (rr < 12.5)]
        ring_level = np.nanmedian(ring)
        ring_mad = 1.4826 * np.nanmedian(np.abs(ring - ring_level))
        assert np.nanmax(np.abs(inner - ring_level)) <= 3.0 * ring_mad + 1e-6
        # Continuous across the clamp boundary: no step at the ring edge.
        in_edge = np.nanmedian(surf[(rr >= 8.5) & (rr < 9.5)])
        out_edge = np.nanmedian(surf[(rr >= 9.5) & (rr < 10.5)])
        assert abs(in_edge - out_edge) < 0.5

    def test_local_mask_seeded_with_nonfinite(
        self, flat_image, base_config, monkeypatch
    ):
        seen = []
        orig = BackgroundSubtractor._make_source_mask

        def spy(self, image, *args, initial_mask=None, **kwargs):
            seen.append((image.shape, initial_mask))
            return orig(self, image, *args, initial_mask=initial_mask, **kwargs)

        monkeypatch.setattr(BackgroundSubtractor, "_make_source_mask", spy)
        sub = BackgroundSubtractor(base_config)
        img = flat_image.copy()
        img[60:70, 60:70] = np.nan
        sub.remove_local_surface(img, 64, 64, box_half_size=30, plot=False)

        # First call is on the 60x60 cutout (origin at 34,34); the NaN block
        # at global 60:70 lands at cutout coords 26:36.
        cutout_shape, cutout_seed = seen[0]
        assert cutout_shape == (60, 60)
        assert cutout_seed is not None
        assert cutout_seed[26:36, 26:36].all()
        assert int(cutout_seed.sum()) == 100


class TestDefectColumnMask:
    """Hot-pixel columns/rows are short axis-aligned ridges that the
    satellite trail mask misses (its minimum length is 40 px).  The
    defect-column mask catches them but must leave compact sources and
    diagonal structures alone."""

    def test_vertical_column_masked(self, flat_image):
        img = flat_image.copy()
        img[40:70, 64] = 500.0  # 30-px hot column
        sub = BackgroundSubtractor({})
        m = sub._make_defect_column_mask(img)
        assert m[40:70, 64].all()

    def test_horizontal_row_masked(self, flat_image):
        img = flat_image.copy()
        img[64, 40:70] = 500.0  # 30-px hot row
        sub = BackgroundSubtractor({})
        m = sub._make_defect_column_mask(img)
        assert m[64, 40:70].all()

    def test_compact_source_not_masked(self, flat_image):
        img = flat_image.copy()
        img[60:65, 60:65] = 500.0  # 5x5 blob: a real source
        sub = BackgroundSubtractor({})
        m = sub._make_defect_column_mask(img)
        assert not m[60:65, 60:65].any()

    def test_short_column_not_masked(self, flat_image):
        img = flat_image.copy()
        img[50:58, 30] = 500.0  # 8 px, below the 12-px minimum
        sub = BackgroundSubtractor({})
        m = sub._make_defect_column_mask(img)
        assert not m[50:58, 30].any()

    def test_diagonal_ridge_not_masked(self, flat_image):
        img = flat_image.copy()
        for i in range(20):
            img[40 + i, 40 + i] = 500.0  # 45-deg ridge: trail mask's job
        sub = BackgroundSubtractor({})
        m = sub._make_defect_column_mask(img)
        assert not m.any()

    def test_mask_via_remove(self, flat_image, base_config):
        img = flat_image.copy()
        img[30:60, 90] = 5000.0  # bright column stands out over stars
        sub = BackgroundSubtractor(base_config)
        res = sub.remove(img, plot=False)
        # hardware_defects_mask excludes the source mask, so this asserts
        # the column mask (not source detection) caught the ridge.
        assert res["defects_mask"][30:60, 90].all()
        assert res["hardware_defects_mask"][30:60, 90].all()


class TestTrailMaskAxisAligned:
    """Regression: the principal-axis convention in
    _make_satellite_trail_mask had the variance difference reversed,
    rotating the measured axis by 90 deg.  Exactly axis-aligned ridges
    (vertical/horizontal trails, bad columns) measured aspect < 1 and
    were never masked."""

    def test_vertical_trail_masked(self, flat_image):
        img = flat_image.copy()
        img[20:80, 64] = 500.0  # 60-px vertical trail
        sub = BackgroundSubtractor({})
        m = sub._make_satellite_trail_mask(
            img,
            n_sigma=3.0,
            min_length_px=40,
            min_aspect_ratio=5.0,
            min_area_px=30,
        )
        assert m[20:80, 64].all()

    def test_horizontal_trail_masked(self, flat_image):
        img = flat_image.copy()
        img[64, 20:80] = 500.0
        sub = BackgroundSubtractor({})
        m = sub._make_satellite_trail_mask(
            img,
            n_sigma=3.0,
            min_length_px=40,
            min_aspect_ratio=5.0,
            min_area_px=30,
        )
        assert m[64, 20:80].all()

    def test_compact_blob_still_unmasked(self, flat_image):
        img = flat_image.copy()
        img[55:70, 55:70] = 500.0  # 15x15 bright region
        sub = BackgroundSubtractor({})
        m = sub._make_satellite_trail_mask(
            img,
            n_sigma=3.0,
            min_length_px=40,
            min_aspect_ratio=5.0,
            min_area_px=30,
        )
        assert not m[55:70, 55:70].any()


class TestMaxiMaskIntegration:
    """Optional MaxiMask defect mask (background.use_maximask).

    These tests stub ``_maximask_probabilities`` so the plumbing (class
    selection, thresholding, dilation, union into the defect masks, seeding
    into the source mask) is checked without TensorFlow.  Real end-to-end
    inference is covered by tests/MaxiMask.
    """

    MM_ABBRS = (
        "CR",
        "HCL",
        "DCL",
        "HP",
        "DP",
        "P",
        "TRL",
        "FR",
        "NEB",
        "SAT",
        "SP",
        "OV",
        "BBG",
        "BG",
    )

    @staticmethod
    def _fake_probs(shape, hot_regions):
        """(probs, abbrs, thresh) with the given classes hot on regions."""
        probs = np.zeros(
            (len(TestMaxiMaskIntegration.MM_ABBRS),) + shape, dtype=np.float32
        )
        for name, region in hot_regions.items():
            idx = TestMaxiMaskIntegration.MM_ABBRS.index(name)
            probs[idx][region] = 1.0
        thresh = np.full(len(TestMaxiMaskIntegration.MM_ABBRS), 0.5, dtype=np.float32)
        return probs, TestMaxiMaskIntegration.MM_ABBRS, thresh

    @staticmethod
    def _patch_probs(monkeypatch, fake):
        monkeypatch.setattr(
            BackgroundSubtractor,
            "_maximask_probabilities",
            lambda self, image, batch_size, allow_gpu: fake,
        )

    def _cfg(self, base_config, **bkg):
        cfg = dict(base_config)
        cfg["background"] = bkg
        return cfg

    def test_disabled_by_default(self, flat_image, base_config, monkeypatch):
        monkeypatch.setattr(
            BackgroundSubtractor,
            "_maximask_probabilities",
            lambda *a, **k: pytest.fail("inference ran while disabled"),
        )
        res = BackgroundSubtractor(base_config).remove(flat_image, plot=False)
        assert res["image"].shape == flat_image.shape

    def test_union_into_hardware_and_defects(
        self, flat_image, base_config, monkeypatch
    ):
        fake = self._fake_probs(flat_image.shape, {"SAT": np.s_[40:60, 40:60]})
        self._patch_probs(monkeypatch, fake)
        cfg = self._cfg(base_config, use_maximask=True, maximask_dilate_iterations=0)
        res = BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert res["hardware_defects_mask"][40:60, 40:60].all()
        assert res["defects_mask"][40:60, 40:60].all()

    def test_class_subset_respected(self, flat_image, base_config, monkeypatch):
        fake = self._fake_probs(flat_image.shape, {"SAT": np.s_[40:60, 40:60]})
        self._patch_probs(monkeypatch, fake)
        # SAT is hot but not requested: nothing may be flagged.
        cfg = self._cfg(base_config, use_maximask=True, maximask_classes=["CR"])
        res = BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert not res["hardware_defects_mask"][40:60, 40:60].any()
        # Requesting SAT picks the region up.
        cfg = self._cfg(base_config, use_maximask=True, maximask_classes=["SAT"])
        res = BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert res["hardware_defects_mask"][45:55, 45:55].all()

    def test_seeded_into_source_mask(self, flat_image, base_config, monkeypatch):
        fake = self._fake_probs(flat_image.shape, {"TRL": np.s_[10:20, :]})
        self._patch_probs(monkeypatch, fake)
        seen = {}
        orig = BackgroundSubtractor._make_source_mask

        def spy(self, image, *args, initial_mask=None, **kwargs):
            seen["initial_mask"] = initial_mask
            return orig(self, image, *args, initial_mask=initial_mask, **kwargs)

        monkeypatch.setattr(BackgroundSubtractor, "_make_source_mask", spy)
        cfg = self._cfg(base_config, use_maximask=True, maximask_classes=["TRL"])
        BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert seen["initial_mask"][10:20, :].all()

    def test_inference_failure_is_graceful(self, flat_image, base_config, monkeypatch):
        self._patch_probs(monkeypatch, None)
        cfg = self._cfg(base_config, use_maximask=True)
        res = BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert res["image"].shape == flat_image.shape

    def test_maximask_active_flag(self, flat_image, base_config, monkeypatch):
        fake = self._fake_probs(flat_image.shape, {"CR": np.s_[0:5, 0:5]})
        self._patch_probs(monkeypatch, fake)
        cfg = self._cfg(base_config, use_maximask=True)
        sub = BackgroundSubtractor(cfg)
        assert sub.maximask_active is False
        sub.remove(flat_image, plot=False)
        assert sub.maximask_active is True

    def test_maximask_active_false_on_failure(
        self, flat_image, base_config, monkeypatch
    ):
        self._patch_probs(monkeypatch, None)
        cfg = self._cfg(base_config, use_maximask=True)
        sub = BackgroundSubtractor(cfg)
        sub.remove(flat_image, plot=False)
        # Inference failure must leave the flag unset so fallback defect
        # stages (e.g. astroscrappy) are not skipped on the caller's behalf.
        assert sub.maximask_active is False

    def test_maximask_active_false_when_disabled(
        self, flat_image, base_config, monkeypatch
    ):
        monkeypatch.setattr(
            BackgroundSubtractor,
            "_maximask_probabilities",
            lambda *a, **k: pytest.fail("inference ran while disabled"),
        )
        sub = BackgroundSubtractor(base_config)
        sub.remove(flat_image, plot=False)
        assert sub.maximask_active is False

    def test_string_classes_parsed(self, flat_image, base_config, monkeypatch):
        fake = self._fake_probs(flat_image.shape, {"CR": np.s_[0:5, 0:5]})
        self._patch_probs(monkeypatch, fake)
        cfg = self._cfg(base_config, use_maximask=True, maximask_classes="CR, SAT")
        res = BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert res["hardware_defects_mask"][1:4, 1:4].all()

    def test_defect_columns_read_from_background_section(
        self, flat_image, base_config, monkeypatch
    ):
        """mask_defect_columns/defect_column_* live under the background:
        YAML section; remove() used to read them from the top level so user
        settings never applied."""
        calls = {}
        orig = BackgroundSubtractor._make_defect_column_mask

        def spy(self, image, **kwargs):
            calls.update(kwargs)
            return orig(self, image, **kwargs)

        monkeypatch.setattr(BackgroundSubtractor, "_make_defect_column_mask", spy)
        cfg = self._cfg(
            base_config,
            mask_defect_columns=True,
            defect_column_n_sigma=7.0,
        )
        BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert calls["n_sigma"] == 7.0

        calls.clear()
        cfg = self._cfg(base_config, mask_defect_columns=False)
        BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert calls == {}

    def test_maximask_plot_written_when_enabled(
        self, flat_image, base_config, monkeypatch, tmp_path
    ):
        """With plot=True a Maximask_<base> figure lands next to fpath."""
        fake = self._fake_probs(flat_image.shape, {"TRL": np.s_[40:44, 30:60]})
        self._patch_probs(monkeypatch, fake)
        cfg = self._cfg(base_config, use_maximask=True, maximask_classes=["TRL"])
        sub = BackgroundSubtractor(cfg)
        sub.remove(flat_image, plot=True)
        plots = list(tmp_path.glob("Maximask_*"))
        assert len(plots) == 1
        assert "TRL" in sub._maximask_class_masks

    def test_no_maximask_plot_when_disabled(self, flat_image, base_config, tmp_path):
        res = BackgroundSubtractor(base_config).remove(flat_image, plot=True)
        assert res["image"].shape == flat_image.shape
        assert not list(tmp_path.glob("Maximask_*"))

    def test_maximask_plot_written_without_plot_flag(
        self, flat_image, base_config, monkeypatch, tmp_path
    ):
        """use_maximask must emit Maximask_<base> even on plot=False passes -
        the second/third remove() calls per epoch run without diagnostics."""
        fake = self._fake_probs(flat_image.shape, {"SAT": np.s_[40:44, 30:60]})
        self._patch_probs(monkeypatch, fake)
        cfg = self._cfg(base_config, use_maximask=True, maximask_classes=["SAT"])
        BackgroundSubtractor(cfg).remove(flat_image, plot=False)
        assert len(list(tmp_path.glob("Maximask_*"))) == 1

    def test_maximask_plot_written_on_inference_failure(
        self, flat_image, base_config, monkeypatch, tmp_path
    ):
        """A failed/unavailable MaxiMask stage still leaves a figure behind,
        annotated with the reason, so 'requested but broken' is visible."""
        self._patch_probs(monkeypatch, None)
        cfg = self._cfg(base_config, use_maximask=True)
        sub = BackgroundSubtractor(cfg)
        sub.remove(flat_image, plot=False)
        assert len(list(tmp_path.glob("Maximask_*"))) == 1
        assert sub._maximask_error is not None

    def test_inference_cached_per_image(self, flat_image, base_config, monkeypatch):
        """remove() runs twice per epoch on the same pixels; the second
        call must reuse the cached mask instead of re-running inference."""
        calls = []

        def counting(self, image, batch_size, allow_gpu):
            calls.append(1)
            return TestMaxiMaskIntegration._fake_probs(
                image.shape, {"CR": np.s_[0:5, 0:5]}
            )

        monkeypatch.setattr(BackgroundSubtractor, "_maximask_probabilities", counting)
        cfg = self._cfg(base_config, use_maximask=True)
        sub = BackgroundSubtractor(cfg)
        sub.remove(flat_image, plot=False)
        sub.remove(flat_image, plot=False)
        assert len(calls) == 1

    def test_inference_cached_across_affine_change(
        self, flat_image, base_config, monkeypatch
    ):
        """The mask cache must still hit after the image is background
        subtracted in place: defects do not move under affine flux changes."""
        calls = []

        def counting(self, image, batch_size, allow_gpu):
            calls.append(1)
            return TestMaxiMaskIntegration._fake_probs(
                image.shape, {"CR": np.s_[0:5, 0:5]}
            )

        monkeypatch.setattr(BackgroundSubtractor, "_maximask_probabilities", counting)
        cfg = self._cfg(base_config, use_maximask=True)
        sub = BackgroundSubtractor(cfg)
        sub.remove(flat_image, plot=False)
        sub.remove((flat_image - 99.0) * 1.3, plot=False)
        assert len(calls) == 1
