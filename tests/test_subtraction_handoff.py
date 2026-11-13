"""Regression tests for the main.py -> Templates.subtract() backend handoff.

Covers the defects found in the subtraction-handoff audit:

- noise-map validation (shape/grid agreement with the aligned images);
- pre-subtraction common-grid rejection;
- difference-image WCS validation/injection;
- real saturation masking in ``create_image_mask``;
- HOTPANTS target exclusion in kernel-fitting stamp lists;
- HOTPANTS per-image (-imi/-tmi) and noise (-ini/-tni) arguments.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astropy.io import fits
from astropy.wcs import WCS

from autophot.subtraction.templates import (
    Templates,
    _ensure_diff_wcs,
    _validate_noise_map,
    _wcs_on_common_grid,
)


def _wcs_header(shift_x: float = 0.0) -> fits.Header:
    """Minimal valid 2-D celestial WCS header on a fixed grid."""
    w = WCS(naxis=2)
    w.wcs.crpix = [32.0 + shift_x, 32.0]
    w.wcs.cdelt = [-0.0003, 0.0003]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w.to_header()


def _write_fits(path: Path, data: np.ndarray, header=None) -> str:
    fits.writeto(str(path), data, header, overwrite=True)
    return str(path)


class TestNoiseMapValidation:
    """_validate_noise_map must reject anything not on the image grid."""

    def test_missing_path_returns_none(self, tmp_path):
        assert _validate_noise_map(
            str(tmp_path / "nope.fits"), (8, 8), "science"
        ) is None
        assert _validate_noise_map(None, (8, 8), "science") is None

    def test_wrong_shape_rejected(self, tmp_path):
        p = _write_fits(tmp_path / "w.fits", np.ones((4, 4)))
        assert _validate_noise_map(p, (8, 8), "template") is None

    def test_all_nan_rejected(self, tmp_path):
        p = _write_fits(tmp_path / "w.fits", np.full((8, 8), np.nan))
        assert _validate_noise_map(p, (8, 8), "template") is None

    def test_matching_finite_map_accepted(self, tmp_path):
        p = _write_fits(tmp_path / "w.fits", np.ones((8, 8)))
        assert _validate_noise_map(p, (8, 8), "science") == p

    def test_wcs_mismatch_rejected(self, tmp_path):
        from autophot.pipeline.wcs import get_wcs

        ref_wcs = get_wcs(_wcs_header())
        # Same shape, but the map carries a shifted (stale) grid.
        p = _write_fits(
            tmp_path / "w.fits", np.ones((8, 8)), _wcs_header(shift_x=5.0)
        )
        assert (
            _validate_noise_map(p, (8, 8), "template", ref_wcs=ref_wcs)
            is None
        )

    def test_matching_wcs_map_accepted(self, tmp_path):
        from autophot.pipeline.wcs import get_wcs

        hdr = _wcs_header()
        p = _write_fits(tmp_path / "w.fits", np.ones((8, 8)), hdr)
        assert (
            _validate_noise_map(
                p, (8, 8), "template", ref_wcs=get_wcs(hdr)
            )
            == p
        )


class TestCommonGrid:
    def test_identical_wcs_is_common_grid(self):
        hdr = _wcs_header()
        w1 = w2 = None
        from autophot.pipeline.wcs import get_wcs

        w1 = get_wcs(hdr)
        w2 = get_wcs(hdr.copy())
        assert w1 is not None and w2 is not None
        assert _wcs_on_common_grid(w1, w2, (64, 64), tol_px=0.5)

    def test_shifted_wcs_is_not_common_grid(self):
        from autophot.pipeline.wcs import get_wcs

        w1 = get_wcs(_wcs_header())
        # 1-pixel CRPIX shift moves the grid by ~1 px everywhere.
        w2 = get_wcs(_wcs_header(shift_x=1.0))
        assert not _wcs_on_common_grid(w1, w2, (64, 64), tol_px=0.5)


class TestEnsureDiffWCS:
    def test_missing_diff_wcs_gets_science_wcs(self):
        sci_hdr = _wcs_header()
        diff_hdr = fits.Header()  # no WCS cards
        out = _ensure_diff_wcs(diff_hdr, sci_hdr, (64, 64))
        from autophot.pipeline.wcs import get_wcs

        assert get_wcs(out) is not None

    def test_consistent_diff_wcs_kept(self):
        sci_hdr = _wcs_header()
        diff_hdr = sci_hdr.copy()
        out = _ensure_diff_wcs(diff_hdr, sci_hdr, (64, 64))
        # Round-trip through get_wcs to prove the grid still agrees.
        from autophot.pipeline.wcs import get_wcs

        assert _wcs_on_common_grid(
            get_wcs(out), get_wcs(sci_hdr), (64, 64), tol_px=0.01
        )

    def test_inconsistent_diff_wcs_replaced(self):
        sci_hdr = _wcs_header()
        bad_hdr = _wcs_header(shift_x=3.0)
        out = _ensure_diff_wcs(bad_hdr, sci_hdr, (64, 64))
        from autophot.pipeline.wcs import get_wcs

        # After injection the diff WCS must agree with the science grid.
        assert _wcs_on_common_grid(
            get_wcs(out), get_wcs(sci_hdr), (64, 64), tol_px=0.01
        )


class TestCreateImageMask:
    """create_image_mask must actually mask saturated pixels."""

    @staticmethod
    def _image() -> np.ndarray:
        rng = np.random.default_rng(7)
        img = rng.normal(100.0, 5.0, (96, 96))
        # Saturated star: bright core well above 0.9 * sat_lvl.
        yy, xx = np.ogrid[:96, :96]
        core = (xx - 30) ** 2 + (yy - 30) ** 2 < 9
        img[core] = 65000.0
        # Unsaturated point source (must stay usable when
        # create_source_mask=False).
        src = (xx - 70) ** 2 + (yy - 70) ** 2 < 4
        img[src] += 5000.0
        return img

    def test_saturated_core_masked_unsaturated_kept(self):
        t = Templates(input_yaml={})
        img = self._image()
        mask, centres = t.create_image_mask(
            img,
            sat_lvl=65535,
            saturate_frac=0.90,
            fwhm=5,
            create_source_mask=False,
        )
        assert mask[30, 30] == 1
        # Dilation must cover the bleeding wings beyond the 3-px core.
        assert mask[30, 30 + 4] == 1
        # The unsaturated calibration source must remain unmasked.
        assert mask[70, 70] == 0
        assert len(centres) >= 1

    def test_ignore_position_keeps_target_unmasked(self):
        t = Templates(input_yaml={})
        img = self._image()
        # Move the saturated star onto the target position.
        mask, _ = t.create_image_mask(
            img,
            sat_lvl=65535,
            saturate_frac=0.90,
            fwhm=5,
            create_source_mask=False,
            ignore_position=[(30.0, 30.0)],
        )
        assert mask[30, 30] == 0


class TestSubtractPreValidation:
    """subtract() must refuse geometrically inconsistent inputs."""

    def test_shape_mismatch_rejected(self, tmp_path):
        rng = np.random.default_rng(0)
        sci = _write_fits(
            tmp_path / "sci.fits", rng.normal(0, 1, (64, 64)), _wcs_header()
        )
        ref = _write_fits(
            tmp_path / "ref.fits", rng.normal(0, 1, (32, 32)), _wcs_header()
        )
        t = Templates(input_yaml={})
        out = t.subtract(sci, ref, method="hotpants")
        assert out == (None, None, None, None)

    def test_wcs_grid_mismatch_rejected(self, tmp_path):
        rng = np.random.default_rng(0)
        sci = _write_fits(
            tmp_path / "sci.fits", rng.normal(0, 1, (64, 64)), _wcs_header()
        )
        ref = _write_fits(
            tmp_path / "ref.fits",
            rng.normal(0, 1, (64, 64)),
            _wcs_header(shift_x=5.0),
        )
        t = Templates(input_yaml={})
        out = t.subtract(sci, ref, method="hotpants")
        assert out == (None, None, None, None)

    def test_missing_wcs_rejected(self, tmp_path):
        rng = np.random.default_rng(0)
        sci = _write_fits(
            tmp_path / "sci.fits", rng.normal(0, 1, (64, 64)), _wcs_header()
        )
        ref = _write_fits(tmp_path / "ref.fits", rng.normal(0, 1, (64, 64)))
        t = Templates(input_yaml={})
        out = t.subtract(sci, ref, method="hotpants")
        assert out == (None, None, None, None)


class TestHotpantsAdapter:
    """The HOTPANTS CLI adapter must pass per-image masks, both noise
    maps, and a target-excluded stamp list."""

    @pytest.fixture()
    def fake_hotpants(self, tmp_path, monkeypatch):
        """Install a fake `hotpants` executable that records its argv."""
        arglog = tmp_path / "hp_args.json"
        monkeypatch.setenv("HP_ARGLOG", str(arglog))
        script = tmp_path / "fake_hotpants.py"
        script.write_text(
            f"#!{sys.executable}\n"
            "import sys, json, os\n"
            "from astropy.io import fits\n"
            "import numpy as np\n"
            "out = sys.argv[sys.argv.index('-outim') + 1]\n"
            "sci = sys.argv[sys.argv.index('-inim') + 1]\n"
            "h = fits.open(sci); hdr = h[0].header.copy()\n"
            "data = np.zeros_like(h[0].data); h.close()\n"
            "fits.writeto(out, data, hdr, overwrite=True)\n"
            "json.dump(sys.argv[1:], open(os.environ['HP_ARGLOG'], 'w'))\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        return script, arglog

    def _run_adapter(self, tmp_path, fake, monkeypatch, stamp_loc=None):
        script, arglog = fake
        rng = np.random.default_rng(1)
        shape = (64, 64)
        hdr = _wcs_header()
        sci = _write_fits(tmp_path / "sci.fits", rng.normal(0, 1, shape), hdr)
        ref = _write_fits(tmp_path / "ref.fits", rng.normal(0, 1, shape), hdr)
        mask = _write_fits(tmp_path / "mask.fits", np.zeros(shape, dtype=int))
        sci_mask = _write_fits(
            tmp_path / "sci_mask.fits", np.zeros(shape, dtype=int)
        )
        tpl_mask = _write_fits(
            tmp_path / "tpl_mask.fits", np.zeros(shape, dtype=int)
        )
        sci_noise = _write_fits(tmp_path / "sci_n.fits", np.ones(shape))
        tpl_noise = _write_fits(tmp_path / "tpl_n.fits", np.ones(shape))
        diff = str(tmp_path / "diff.fits")

        # 15 matching sources; one sits on top of the target at (32, 32).
        sources = [(10.0 + 4.0 * i, 50.0) for i in range(15)]
        sources.append((33.0, 32.0))  # inside the target exclusion radius

        t = Templates(
            input_yaml={
                "template_subtraction": {
                    "hotpants_exe_loc": str(script),
                },
                "target_x_pix": 32.0,
                "target_y_pix": 32.0,
            }
        )
        ok = t._subtract_hotpants(
            sci,
            ref,
            diff,
            mask,
            tmp_path,
            "sci.fits",
            0.0,
            1.0,
            0.0,
            1.0,
            60000.0,
            60000.0,
            5.0,
            5.0,
            3.0,
            3.0,
            0,
            stamp_loc,  # None: adapter must build it from matching_sources
            sci_noise,
            15,
            templateNoise=tpl_noise,
            sci_mask_loc=sci_mask,
            tpl_mask_loc=tpl_mask,
            matching_sources=sources,
            target_xy=(32.0, 32.0),
        )
        assert ok
        return json.loads(arglog.read_text()), tmp_path

    def test_noise_maps_and_per_image_masks_passed(
        self, tmp_path, fake_hotpants, monkeypatch
    ):
        args, tmp_path = self._run_adapter(
            tmp_path, fake_hotpants, monkeypatch
        )
        argmap = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
        assert argmap["-ini"].endswith("sci_n.fits")
        assert argmap["-tni"].endswith("tpl_n.fits")
        assert argmap["-imi"].endswith("sci_mask.fits")
        assert argmap["-tmi"].endswith("tpl_mask.fits")

    def test_stamp_file_excludes_target(
        self, tmp_path, fake_hotpants, monkeypatch
    ):
        args, tmp_path = self._run_adapter(
            tmp_path, fake_hotpants, monkeypatch
        )
        argmap = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
        assert "-ssf" in argmap
        stamp_path = Path(argmap["-ssf"])
        assert stamp_path.is_file()
        excl_r = max(3.0 * 3.0, 15.0)
        for line in stamp_path.read_text().splitlines():
            x, y = (float(v) for v in line.split())
            # Stamp positions are 1-based; target is at 0-based (32, 32).
            assert np.hypot(x - 1.0 - 32.0, y - 1.0 - 32.0) > excl_r

    def test_caller_stamp_file_excludes_target(
        self, tmp_path, fake_hotpants, monkeypatch
    ):
        """A caller-provided stamp file (method=hotpants in main.py) must
        get the same target exclusion as the adapter-built list."""
        caller = tmp_path / "caller_stamps.txt"
        # 15 stamps well clear of the target plus one on it (1-based).
        lines = [f"{10.0 + 4.0 * i + 1.0} 51.0" for i in range(15)]
        lines.append("34.0 33.0")  # 0-based (33, 32): inside the exclusion radius
        caller.write_text("\n".join(lines) + "\n")

        args, tmp_path = self._run_adapter(
            tmp_path, fake_hotpants, monkeypatch, stamp_loc=str(caller)
        )
        argmap = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
        stamp_path = Path(argmap["-ssf"])
        pts = [
            tuple(float(v) for v in line.split())
            for line in stamp_path.read_text().splitlines()
        ]
        assert len(pts) == 15
        excl_r = max(3.0 * 3.0, 15.0)
        for x, y in pts:
            assert np.hypot(x - 1.0 - 32.0, y - 1.0 - 32.0) > excl_r
        # The caller's file must not be rewritten in place.
        assert len(caller.read_text().splitlines()) == 16
