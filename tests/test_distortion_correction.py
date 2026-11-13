"""Tests for the opt-in SCAMP+SWarp distortion correction stage.

``ImageDistortionCorrector.undistort_image`` removes optical distortion
(e.g. telescope flexure) by fitting a TPV solution against GAIA-DR3 with
SCAMP and resampling with SWarp.  Every failure path must return None so
the caller keeps the original image.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.wcs import WCS

from autophot.subtraction import run_sex

try:
    from autophot.subtraction.run_IDC import ImageDistortionCorrector
except ImportError:  # astroalign / optional deps not installed
    ImageDistortionCorrector = None

_needs_idc = pytest.mark.skipif(
    ImageDistortionCorrector is None, reason="run_IDC deps not installed"
)


def _tan_wcs():
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 150.0
    hdr["CRVAL2"] = 2.0
    hdr["CRPIX1"] = 50.0
    hdr["CRPIX2"] = 40.0
    hdr["CD1_1"] = -2.7e-4
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 2.7e-4
    return WCS(hdr)


def _make_idc(tmp_path, distort_degrees=3):
    idc = ImageDistortionCorrector(
        input_yaml={
            "wcs": {"scamp_distort_degrees": distort_degrees},
            "global_verbose_level": 0,
        }
    )
    idc._is_executable_available = lambda name: True
    idc._extract_wcs_and_scale = lambda p: (
        _tan_wcs(),
        0.97,
        fits.Header({"NAXIS1": 100, "NAXIS2": 80}),
    )
    return idc


def _fake_sex(tmp_path, n_sources):
    cat = tmp_path / "science_image.cat"
    cat.write_text("LDAC")
    catalog = [{"XWIN_IMAGE": float(i)} for i in range(n_sources)]
    return {"catalog_path": str(cat), "catalog": catalog}


def _fake_scamp(tmp_path):
    head = tmp_path / "science_image.head"
    head.write_text("CTYPE1  = 'RA---TPV'\nEND\n")
    return {
        "head_file": str(head),
        "head_files_by_stem": {"science_image": str(head)},
        "distortion": {
            "astrometric_rms_arcsec": 0.05,
            "n_matched_stars": 42,
        },
        "config": {},
    }


def _fake_swarp(tmp_path, shape=(80, 100)):
    out_dir = tmp_path / "swarp_out"
    out_dir.mkdir(exist_ok=True)
    corr = out_dir / "science_image.resamp.fits"
    fits.writeto(corr, np.zeros(shape, dtype=np.float32), overwrite=True)
    return {
        "corrected_image": str(corr),
        "weight_image": str(out_dir / "weights.fits"),
        "resampled_dir": str(out_dir / "resampled"),
        "log_file": str(out_dir / "swarp.log"),
        "config": {},
    }


@_needs_idc
class TestUndistortImage:

    def test_missing_executables_returns_none(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc._is_executable_available = lambda name: False
        called = mock.Mock()
        idc.run_sextractor = called

        assert idc.undistort_image("img.fits", str(tmp_path)) is None
        called.assert_not_called()

    def test_no_wcs_returns_none(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc._extract_wcs_and_scale = lambda p: (None, 1.0, fits.Header())

        assert idc.undistort_image("img.fits", str(tmp_path)) is None

    def test_too_few_sources_returns_none(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 3)
        scamp = mock.Mock()
        idc.run_scamp = scamp

        assert idc.undistort_image("img.fits", str(tmp_path)) is None
        scamp.assert_not_called()

    def test_scamp_failure_returns_none(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 50)
        idc.run_scamp = lambda *a, **k: None

        assert idc.undistort_image("img.fits", str(tmp_path)) is None

    def test_swarp_failure_returns_none(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 50)
        idc.run_scamp = lambda *a, **k: _fake_scamp(tmp_path)
        idc.run_swarp = lambda *a, **k: None

        assert idc.undistort_image("img.fits", str(tmp_path)) is None

    def test_happy_path(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 400)
        idc.run_scamp = lambda *a, **k: _fake_scamp(tmp_path)
        swarp_calls = []

        def _swarp(images, **kwargs):
            swarp_calls.append(kwargs)
            return _fake_swarp(tmp_path)

        idc.run_swarp = _swarp
        img = tmp_path / "img.fits"
        img.write_text("x")  # placeholder; .head lands next to it

        out = idc.undistort_image(str(img), str(tmp_path))

        assert out is not None and Path(out).exists()
        # Grid preserved: same size/scale/center type in the SWarp config
        cfg = swarp_calls[0]["config"]
        assert cfg["IMAGE_SIZE"] == "100,80"
        assert cfg["PIXELSCALE_TYPE"] == "MANUAL"
        assert cfg["RESAMPLING_TYPE"] == "LANCZOS3"
        # The SCAMP .head staged next to the image is cleaned up
        assert not img.with_suffix(".head").exists()

    def test_scamp_retry_descends_degree(self, tmp_path):
        # A degenerate solve at degree 3 must retry at degree 2, 1.
        idc = _make_idc(tmp_path, distort_degrees=3)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 400)
        seen_degrees = []

        def _scamp(catalog_paths, **kwargs):
            seen_degrees.append((kwargs.get("config") or {})["DISTORT_DEGREES"])
            if seen_degrees[-1] > 2:
                return None  # degenerate solve
            return _fake_scamp(tmp_path)

        idc.run_scamp = _scamp
        idc.run_swarp = lambda *a, **k: _fake_swarp(tmp_path)
        img = tmp_path / "img.fits"
        img.write_text("x")

        out = idc.undistort_image(str(img), str(tmp_path))
        assert out is not None
        assert seen_degrees == [3, 2]

    def test_scamp_all_degrees_fail_returns_none(self, tmp_path):
        idc = _make_idc(tmp_path, distort_degrees=3)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 400)
        seen_degrees = []

        def _scamp(catalog_paths, **kwargs):
            seen_degrees.append((kwargs.get("config") or {})["DISTORT_DEGREES"])
            return None

        idc.run_scamp = _scamp
        swarp = mock.Mock()
        idc.run_swarp = swarp

        assert idc.undistort_image("img.fits", str(tmp_path)) is None
        assert seen_degrees == [3, 2, 1]
        swarp.assert_not_called()

    def test_degree_reduced_on_sparse_field(self, tmp_path):
        idc = _make_idc(tmp_path, distort_degrees=3)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 150)
        captured = {}

        def _scamp(catalog_paths, **kwargs):
            captured.update(kwargs.get("config") or {})
            return _fake_scamp(tmp_path)

        idc.run_scamp = _scamp
        idc.run_swarp = lambda *a, **k: None

        idc.undistort_image("img.fits", str(tmp_path))
        assert captured["DISTORT_DEGREES"] == 2

    def test_resampling_type_override(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc.run_sextractor = lambda *a, **k: _fake_sex(tmp_path, 400)
        idc.run_scamp = lambda *a, **k: _fake_scamp(tmp_path)
        swarp_calls = []

        def _swarp(images, **kwargs):
            swarp_calls.append(kwargs)
            return _fake_swarp(tmp_path)

        idc.run_swarp = _swarp
        img = tmp_path / "img.fits"
        img.write_text("x")

        idc.undistort_image(
            str(img), str(tmp_path), resampling_type="LANCZOS2"
        )
        assert swarp_calls[0]["config"]["RESAMPLING_TYPE"] == "LANCZOS2"


@_needs_idc
class TestDetectionBackground:
    """Regression: BACK_TYPE MANUAL wrote no BACK_VALUE (implicit 0),
    so a detection pass on an image that still carried positive sky
    (~24 ADU on a GROND frame) put the whole frame above DETECT_THRESH
    and merged it into a single giant object.  AUTO estimates the sky
    mesh instead."""

    def test_default_config_uses_auto_background(self):
        assert (
            ImageDistortionCorrector.DEFAULT_SEX_CONFIG["BACK_TYPE"] == "AUTO"
        )

    def test_alignment_overrides_keep_auto_background(self):
        merged = dict(ImageDistortionCorrector.DEFAULT_SEX_CONFIG)
        merged.update(ImageDistortionCorrector.ALIGNMENT_SEX_CONFIG)
        assert merged["BACK_TYPE"] == "AUTO"


@_needs_idc
class TestHeaderFwhmPixels:
    """GROND-style headers carry FWHM in arcseconds and FWHMPIX in
    pixels; reading FWHM as pixels made the detection kernel ~2.4x too
    narrow (1.4 px vs the real ~8.7 px PSF)."""

    def test_fwhmpix_preferred_over_arcsec_fwhm(self):
        hdr = fits.Header({"FWHM": 1.38, "FWHMPIX": 8.73})
        assert ImageDistortionCorrector._header_fwhm_pixels(
            hdr
        ) == pytest.approx(8.73)

    def test_fwhm_only_assumed_pixels(self):
        hdr = fits.Header({"FWHM": 3.2})
        assert ImageDistortionCorrector._header_fwhm_pixels(
            hdr
        ) == pytest.approx(3.2)

    def test_missing_or_invalid_defaults_to_2(self):
        assert ImageDistortionCorrector._header_fwhm_pixels(fits.Header()) == 2.0
        assert ImageDistortionCorrector._header_fwhm_pixels(
            fits.Header({"FWHMPIX": "bad"})
        ) == 2.0
        assert ImageDistortionCorrector._header_fwhm_pixels(
            fits.Header({"FWHM": -1.0})
        ) == 2.0


@_needs_idc
class TestSexBinaryResolution:
    """IDC must run the same SExtractor binary whose version the
    run_sex NaN-fill shim probed (the highest on PATH), not whichever
    'sex' resolves first - a stale low-version binary breaks the shim
    assumption and can lack catalog parameters."""

    def test_prefers_highest_version_binary(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc._executables.clear()
        with mock.patch(
            "utils.run_sex.get_sextractor_executable",
            return_value="/opt/astro/bin/sex",
        ):
            assert idc._check_executable("sex") == "/opt/astro/bin/sex"
        assert idc._executables["sex"] == "/opt/astro/bin/sex"

    def test_falls_back_to_path_probe(self, tmp_path):
        idc = _make_idc(tmp_path)
        idc._executables.clear()
        with mock.patch.object(
            ImageDistortionCorrector, "_resolve_sextractor", return_value=None
        ), mock.patch("subprocess.run", side_effect=OSError), pytest.raises(
            FileNotFoundError
        ):
            idc._check_executable("sex")


class TestNanFilledCopyForDetection:

    def test_new_sextractor_returns_original(self, tmp_path):
        src = tmp_path / "img.fits"
        fits.writeto(src, np.full((4, 4), np.nan, dtype=np.float32))
        with mock.patch.object(
            run_sex, "get_sextractor_executable", return_value="/bin/sex"
        ), mock.patch.object(run_sex, "_SEXTRACTOR_VERSION", (2, 28, 0)):
            out = run_sex.nan_filled_copy_for_detection(src, tmp_path)
        assert out == src

    def test_old_sextractor_fills(self, tmp_path):
        src = tmp_path / "img.fits"
        data = np.ones((4, 4), dtype=np.float32)
        data[0, 0] = np.nan
        fits.writeto(src, data)
        work = tmp_path / "work"
        work.mkdir()
        with mock.patch.object(
            run_sex, "get_sextractor_executable", return_value="/bin/sex"
        ), mock.patch.object(run_sex, "_SEXTRACTOR_VERSION", (2, 5, 0)):
            out = run_sex.nan_filled_copy_for_detection(src, work)
        assert out != src
        assert np.isfinite(fits.getdata(out)).all()


class TestConfigDefaults:

    def test_disabled_by_default(self):
        cfg_path = (
            Path(__file__).resolve().parents[1]
            / "autophot" / "data"
            / "default_input.yml"
        )
        cfg = yaml.safe_load(cfg_path.read_text())
        wcs_cfg = cfg["default_input"]["wcs"]
        assert wcs_cfg["correct_distortion"] is False
        assert wcs_cfg["correct_distortion_resampling_type"] == "LANCZOS3"
