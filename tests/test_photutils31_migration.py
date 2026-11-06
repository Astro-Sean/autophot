"""Regression tests for the photutils 3.1-dev migration (2026-10-10l).

Covers the moved API surface: AperturePhotometry replacing the deprecated
aperture_photometry() function, n_threads wiring through the shared
resolver, and the feature-gated MoffatPRF stamp path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestPhotutilsThreadsResolver:
    def test_defaults_serial(self):
        from functions import photutils_n_threads

        assert photutils_n_threads() == 1
        assert photutils_n_threads(None) == 1
        assert photutils_n_threads({}) == 1

    def test_reads_input_yaml_shape(self):
        from functions import photutils_n_threads

        assert photutils_n_threads({"photometry": {"photutils_n_threads": 4}}) == 4

    def test_reads_phot_cfg_shape(self):
        from functions import photutils_n_threads

        assert photutils_n_threads({"photutils_n_threads": 3}) == 3

    def test_invalid_and_nonpositive(self):
        from functions import photutils_n_threads

        assert photutils_n_threads({"photometry": {"photutils_n_threads": "x"}}) == 1
        assert photutils_n_threads({"photometry": {"photutils_n_threads": 0}}) == 1
        assert photutils_n_threads({"photometry": {"photutils_n_threads": -5}}) == 1

    def test_capped_by_cpu_and_constant(self):
        import os

        from functions import _PHOTUTILS_THREADS_CAP, photutils_n_threads

        try:
            n_cpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_cpu = os.cpu_count() or 1
        expected = min(999, n_cpu, _PHOTUTILS_THREADS_CAP)
        assert photutils_n_threads({"photutils_n_threads": 999}) == expected

    def test_psf_threads_gated_on_gil(self):
        # On stock CPython the fit loop gains nothing from threads; the
        # resolver must return 1 rather than warning through photutils.
        import sys

        from functions import psf_photometry_threads

        is_gil = getattr(sys, "_is_gil_enabled", None)
        free_threaded = callable(is_gil) and not is_gil()
        got = psf_photometry_threads({"photutils_n_threads": 8})
        assert got == (8 if free_threaded else 1)


class TestAperturePhotometryMigration:
    def test_class_api_used_and_flags_surfaced(self):
        # measure() must go through AperturePhotometry (not the deprecated
        # aperture_photometry function) and carry the flags column into
        # the per-source output as ap_flags.
        import aperture as ap_mod
        from photutils.aperture import AperturePhotometry  # noqa: F401

        assert not hasattr(ap_mod, "aperture_photometry") or (
            ap_mod.AperturePhotometry is AperturePhotometry
        )
        img = np.random.RandomState(0).normal(100, 5, (80, 80))
        img[40, 40] += 2000
        sources = pd.DataFrame({"x_pix": [40.0], "y_pix": [40.0]})
        yaml_in = {
            "fwhm": 3.0,
            "imageFilter": "r",
            "exposure_time": 60.0,
            "read_noise": 5.0,
            "gain": 1.0,
            "photometry": {"aperture_n_jobs": 1},
        }
        out = ap_mod.Aperture(yaml_in, img, verbose=0).measure(
            sources,
            ap_size=5.0,
            exposure_time=60.0,
            read_noise=5.0,
            gain=1.0,
            n_jobs=1,
            verbose=0,
        )
        assert "ap_flags" in out.columns
        assert float(out["ap_flags"].iloc[0]) == 0.0
        assert np.isfinite(out["flux_AP"].iloc[0])
        assert float(out["SNR"].iloc[0]) > 5

    def test_partial_overlap_flag_on_edge_source(self):
        # Direct API check: an aperture clipped by the image edge must set
        # the partial_overlap flag bit (0x2) - this is what ap_flags carries.
        from photutils.aperture import AperturePhotometry, CircularAperture

        img = np.random.RandomState(0).normal(100, 5, (40, 40))
        t = (
            AperturePhotometry(img, CircularAperture([(39.0, 39.0)], r=5.0))
            .to_table()
            .to_pandas()
        )
        assert int(t["flags"].iloc[0]) & 0x2 != 0
        t2 = (
            AperturePhotometry(img, CircularAperture([(20.0, 20.0)], r=5.0))
            .to_table()
            .to_pandas()
        )
        assert int(t2["flags"].iloc[0]) == 0


class TestMoffatPrfStamp:
    def test_stamp_matches_epsf_conventions(self):
        from psf import _analytic_moffat_psf, measure_epsf_fwhm_native

        for osamp in (1, 2, 4):
            m = _analytic_moffat_psf(3.5, osamp, 25, 4.765)
            data = np.asarray(m.data)
            assert np.isfinite(data).all()
            # photutils ePSF convention: stamp sums to prod(oversampling)
            assert np.nansum(data) == pytest.approx(float(osamp) ** 2, rel=1e-6)
            fwhm_native = measure_epsf_fwhm_native(data, osamp)
            # Requested 3.5 px; pixel integration legitimately widens the
            # effective PSF, so allow headroom but reject a broken shape.
            assert 3.0 <= fwhm_native <= 4.5

    def test_stamp_is_point_sampled_not_reintegrated(self):
        # The analytic components describe the effective (already
        # pixel-integrated) PSF, so the stamp must point-sample the
        # continuous Moffat: a second pixel integration used to widen
        # the stamp ~10% past the requested FWHM.
        import psf as psf_mod

        fwhm, osamp, n, beta = 3.5, 4, 25, 4.765
        m = psf_mod._analytic_moffat_psf(fwhm, osamp, n, beta)
        data = np.asarray(m.data)
        n_stamp = data.shape[0]
        ctr = (n_stamp - 1) / 2.0
        gamma = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        yy, xx = np.indices(data.shape)
        rr = np.hypot((xx - ctr) / osamp, (yy - ctr) / osamp)
        expected = (1.0 + (rr / gamma) ** 2) ** (-beta)
        expected *= float(osamp) ** 2 / np.nansum(expected)
        np.testing.assert_allclose(data, expected, rtol=1e-12)

    def test_injection_stamp_is_point_sampled(self):
        # Same effective-PSF convention as the psf.py stamps: the
        # injection helper evaluates the continuous Moffat at the grid
        # positions rather than convolving with the pixel response.
        limits = pytest.importorskip("limits")

        fwhm, osamp, beta = 3.5, 4, 4.765
        m = limits._analytic_psf_for_injection(fwhm, oversampling=osamp)
        data = np.asarray(m.data)
        assert np.isfinite(data).all()
        assert np.nansum(data) == pytest.approx(float(osamp) ** 2, rel=1e-6)
        ctr = (data.shape[0] - 1) / 2.0
        gamma = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        yy, xx = np.indices(data.shape)
        rr = np.hypot((xx - ctr) / osamp, (yy - ctr) / osamp)
        expected = (1.0 + (rr / gamma) ** 2) ** (-beta)
        expected *= float(osamp) ** 2 / np.nansum(expected)
        np.testing.assert_allclose(data, expected, rtol=1e-12)

    def test_no_deprecated_kwargs_in_deblend_calls(self):
        # photutils 3.1 deprecates progress_bar/n_processes on
        # deblend_sources and renamed nlevels to n_levels; the call
        # sites must use the current keywords.
        import inspect
        import re

        import functions, fwhm

        for mod in (functions, fwhm):
            src = inspect.getsource(mod)
            for call in re.finditer(
                r"deblend_sources\(([^)]*(?:\([^)]*\)[^)]*)*)\)", src, re.DOTALL
            ):
                body = call.group(1)
                assert "progress_bar" not in body, (
                    f"deprecated kwarg in {mod.__name__}: {call.group(0)[:120]}"
                )
                assert "n_processes" not in body
                assert not re.search(r"\bnlevels\b", body), (
                    f"renamed kwarg in {mod.__name__}: {call.group(0)[:120]}"
                )


class TestSegmentationFlags:
    def test_crowding_neighbors_carry_seg_flags(self):
        # SourceCatalog.flags is new in 3.1; compute_target_crowding must
        # forward it onto each neighbour dict so downstream QC can tell
        # a deblended/edge sibling from a clean one.
        from functions import compute_target_crowding

        rng = np.random.RandomState(1)
        img = rng.normal(50.0, 3.0, (61, 61))
        yy, xx = np.mgrid[0:61, 0:61]
        img += 800 * np.exp(-((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 8.0)
        img += 600 * np.exp(-((xx - 40.0) ** 2 + (yy - 34.0) ** 2) / 8.0)

        out = compute_target_crowding(
            img, (30.0, 30.0), box_half_size=25, neighbor_radius_pix=20.0
        )
        assert out.get("ok", True)
        assert out["neighbors"], "expected the planted neighbour to be found"
        for n in out["neighbors"]:
            assert "seg_flags" in n
            assert n["seg_flags"] >= 0
