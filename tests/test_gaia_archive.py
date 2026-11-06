"""Tests for autophot_gaia_curves helpers.

Covers the pure (non-network) parts of the Gaia archive access layer:
retry policy, prefetch sizing, nearest-source sorting, and the curve-map
builder's band integration and source-id handling. No archive access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autophot_gaia_curves import gaia_archive
from autophot_gaia_curves.gaia_archive import (
    _chunk_list,
    _is_permanent_http_error,
    gaia_xp_sql_top_n,
    gaia_xp_uses_python_distance_sort,
    query_gaia_xp_cone_growing,
    retry_with_backoff,
    sort_gaia_table_nearest_to_target,
)
from autophot_gaia_curves.builder import (
    GaiaCurveCatalogBuilder,
    _format_gaia_source_id,
)


class TestChunkList:
    def test_even_split(self):
        assert _chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_remainder(self):
        assert _chunk_list([1, 2, 3], 2) == [[1, 2], [3]]

    def test_nonpositive_batch_returns_single_chunk(self):
        assert _chunk_list([1, 2, 3], 0) == [[1, 2, 3]]


class TestPermanentHttpError:
    @staticmethod
    def _exc_with_status(code):
        class FakeResponse:
            status_code = code

        err = Exception(f"HTTP {code}")
        err.response = FakeResponse()
        return err

    def test_400_is_permanent(self):
        assert _is_permanent_http_error(self._exc_with_status(400))

    def test_404_is_permanent(self):
        assert _is_permanent_http_error(self._exc_with_status(404))

    def test_429_is_retryable(self):
        assert not _is_permanent_http_error(self._exc_with_status(429))

    def test_408_is_retryable(self):
        assert not _is_permanent_http_error(self._exc_with_status(408))

    def test_500_is_retryable(self):
        assert not _is_permanent_http_error(self._exc_with_status(500))

    def test_no_response_is_retryable(self):
        assert not _is_permanent_http_error(Exception("boom"))


class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        out = retry_with_backoff(
            fn,
            max_retries=3,
            base_delay_sec=0.0,
            logger=None,
            op_name="test",
        )
        assert out == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("transient")
            return "ok"

        out = retry_with_backoff(
            fn,
            max_retries=3,
            base_delay_sec=0.0,
            logger=None,
            op_name="test",
        )
        assert out == "ok"
        assert len(calls) == 3

    def test_exhaustion_raises_last_error(self):
        def fn():
            raise RuntimeError("always")

        with pytest.raises(RuntimeError, match="always"):
            retry_with_backoff(
                fn,
                max_retries=2,
                base_delay_sec=0.0,
                logger=None,
                op_name="test",
            )

    def test_permanent_error_not_retried(self):
        calls = []

        def fn():
            calls.append(1)
            raise TestPermanentHttpError._exc_with_status(400)

        with pytest.raises(Exception, match="HTTP 400"):
            retry_with_backoff(
                fn,
                max_retries=5,
                base_delay_sec=0.0,
                logger=None,
                op_name="test",
            )
        assert len(calls) == 1


class TestSqlTopN:
    def test_brightness_returns_exact_n(self):
        top, sort_by_distance = gaia_xp_sql_top_n(300, "brightness")
        assert top == 300
        assert sort_by_distance is False

    def test_distance_prefetches_and_sorts(self):
        top, sort_by_distance = gaia_xp_sql_top_n(100, "distance")
        assert sort_by_distance is True
        assert top > 100

    def test_distance_prefetch_monotonic_in_n(self):
        tops = [
            gaia_xp_sql_top_n(ms, "distance")[0] for ms in (10, 50, 100, 200, 500)
        ]
        assert all(a <= b for a, b in zip(tops, tops[1:]))

    def test_prefetch_capped_by_max(self):
        top, _ = gaia_xp_sql_top_n(5000, "nearest", prefetch_max=10000)
        assert top <= 10000

    def test_prefetch_floor(self):
        top, _ = gaia_xp_sql_top_n(5, "distance", prefetch_min=200)
        assert top >= 200


class TestDistanceModes:
    @pytest.mark.parametrize("mode", ["distance", "angular", "nearest", "sky"])
    def test_distance_modes(self, mode):
        assert gaia_xp_uses_python_distance_sort(mode)

    def test_brightness_not_distance(self):
        assert not gaia_xp_uses_python_distance_sort("brightness")

    def test_none_defaults_to_brightness(self):
        assert not gaia_xp_uses_python_distance_sort(None)


class TestSortNearest:
    def _df(self):
        # 5 sources on a line of constant dec; ra increases eastward.
        return pd.DataFrame(
            {
                "ra": [10.0, 10.001, 10.01, 10.1, 9.9],
                "dec": [20.0] * 5,
                "phot_g_mean_mag": [15.0, 14.0, 13.0, 12.0, 11.0],
            }
        )

    def test_keeps_nearest_rows(self):
        df = self._df()
        out = sort_gaia_table_nearest_to_target(df, 10.0, 20.0, max_rows=2)
        assert len(out) == 2
        assert set(out["ra"]) == {10.0, 10.001}

    def test_max_rows_larger_than_frame(self):
        df = self._df()
        out = sort_gaia_table_nearest_to_target(df, 10.0, 20.0, max_rows=99)
        assert len(out) == len(df)

    def test_empty_frame(self):
        out = sort_gaia_table_nearest_to_target(
            self._df().iloc[:0], 10.0, 20.0, max_rows=5
        )
        assert out.empty


class TestXpConeGrowth:
    """Adaptive cone growth: re-query at a larger radius when the cone is sparse."""

    @staticmethod
    def _fake_launch(monkeypatch, sizes):
        """Mock the TAP call; returns len-sizes rows per call, records radius.

        COUNT(*) probes (used for cone growth after the first fetch) get a
        one-row ``{"n": ...}`` frame, matching ``gaia_cone_count`` output.
        """
        queries = []

        def fake(query, **kwargs):
            queries.append(query)
            n = sizes[min(len(queries) - 1, len(sizes) - 1)]
            if "COUNT(*)" in query:
                return pd.DataFrame({"n": [n]})
            return pd.DataFrame(
                {"source_id": range(n), "ra": [1.0] * n, "dec": [2.0] * n}
            )

        monkeypatch.setattr(gaia_archive, "launch_gaia_adql_to_pandas", fake)
        return queries

    @staticmethod
    def _data_queries(queries):
        return [q for q in queries if "COUNT(*)" not in q]

    def test_disabled_growth_single_query(self, monkeypatch):
        queries = self._fake_launch(monkeypatch, [3])
        df, used = query_gaia_xp_cone_growing(
            10.0, 20.0, 0.1, 100, min_sources=0, logger=None
        )
        assert len(queries) == 1
        assert used == pytest.approx(0.1)
        assert len(df) == 3

    def test_grows_until_min_sources(self, monkeypatch):
        queries = self._fake_launch(monkeypatch, [4, 60])
        df, used = query_gaia_xp_cone_growing(
            10.0,
            20.0,
            0.1,
            100,
            min_sources=25,
            max_radius_deg=1.0,
            grow_factor=2.0,
            logger=None,
        )
        # fetch@0.1 (4 < 25) -> count@0.2 (60 >= 25) -> final fetch@0.2.
        assert len(queries) == 3
        assert len(self._data_queries(queries)) == 2
        assert used == pytest.approx(0.2)
        assert len(df) == 60
        # Second ADQL carries the grown radius in the CONTAINS circle.
        assert " 0.1" in queries[0] and " 0.2" in queries[1]

    def test_dense_field_single_query(self, monkeypatch):
        queries = self._fake_launch(monkeypatch, [80])
        df, used = query_gaia_xp_cone_growing(
            10.0,
            20.0,
            0.1,
            100,
            min_sources=25,
            max_radius_deg=1.0,
            grow_factor=2.0,
            logger=None,
        )
        assert len(queries) == 1
        assert "COUNT(*)" not in queries[0]
        assert used == pytest.approx(0.1)
        assert len(df) == 80

    def test_growth_capped(self, monkeypatch):
        queries = self._fake_launch(monkeypatch, [2])
        df, used = query_gaia_xp_cone_growing(
            10.0,
            20.0,
            0.1,
            100,
            min_sources=25,
            max_radius_deg=0.25,
            grow_factor=2.0,
            logger=None,
        )
        # 0.1 -> 0.2 -> 0.25 (capped); rows fetched only at start and end.
        assert used == pytest.approx(0.25)
        assert len(df) == 2
        assert len(self._data_queries(queries)) == 2

    def test_count_failure_falls_back_to_fetch(self, monkeypatch):
        queries = []

        def fake(query, **kwargs):
            queries.append(query)
            if "COUNT(*)" in query:
                raise RuntimeError("count backend down")
            n = 4 if " 0.1" in query else 60
            return pd.DataFrame(
                {"source_id": range(n), "ra": [1.0] * n, "dec": [2.0] * n}
            )

        monkeypatch.setattr(gaia_archive, "launch_gaia_adql_to_pandas", fake)
        df, used = query_gaia_xp_cone_growing(
            10.0,
            20.0,
            0.1,
            100,
            min_sources=25,
            max_radius_deg=1.0,
            grow_factor=2.0,
            logger=None,
        )
        assert used == pytest.approx(0.2)
        assert len(df) == 60
        # fetch@0.1, failed count@0.2, fetch@0.2 - same result as fetch-only.
        assert len(self._data_queries(queries)) == 2

    def test_no_growth_when_factor_le_1(self, monkeypatch):
        queries = self._fake_launch(monkeypatch, [1])
        df, used = query_gaia_xp_cone_growing(
            10.0,
            20.0,
            0.1,
            100,
            min_sources=25,
            max_radius_deg=1.0,
            grow_factor=1.0,
            logger=None,
        )
        assert len(queries) == 1
        assert used == pytest.approx(0.1)


class TestCatalogCacheReuse:
    """Cached CSV reuse decisions honour the query footprint provenance."""

    def _df(self, n):
        return pd.DataFrame(
            {"name": range(n), "ra": [10.0] * n, "dec": [20.0] * n}
        )

    def _builder(self):
        return GaiaCurveCatalogBuilder()

    def test_legacy_cache_reused_when_dense(self, tmp_path):
        csv = tmp_path / "cat.csv"
        self._df(50).to_csv(csv, index=False)
        b = self._builder()
        assert b._catalog_cache_usable(
            self._df(50),
            b._meta_path(csv),
            10.0,
            20.0,
            0.1,
            min_sources=25,
            max_radius_deg=1.0,
        )

    def test_legacy_cache_rebuilt_when_sparse(self, tmp_path):
        csv = tmp_path / "cat.csv"
        self._df(4).to_csv(csv, index=False)
        b = self._builder()
        assert not b._catalog_cache_usable(
            self._df(4),
            b._meta_path(csv),
            10.0,
            20.0,
            0.1,
            min_sources=25,
            max_radius_deg=1.0,
        )

    def test_meta_sparse_but_exhausted_reused(self, tmp_path):
        csv = tmp_path / "cat.csv"
        b = self._builder()
        b._write_outputs(csv, self._df(5), 10.0, 20.0, 0.1, 1.0, 1.0)
        assert b._catalog_cache_usable(
            self._df(5),
            b._meta_path(csv),
            10.0,
            20.0,
            0.1,
            min_sources=25,
            max_radius_deg=1.0,
        )

    def test_meta_exhausted_at_smaller_cap_rebuilt(self, tmp_path):
        csv = tmp_path / "cat.csv"
        b = self._builder()
        b._write_outputs(csv, self._df(5), 10.0, 20.0, 0.1, 0.5, 0.5)
        # Raising the cap invalidates the "growth exhausted" provenance.
        assert not b._catalog_cache_usable(
            self._df(5),
            b._meta_path(csv),
            10.0,
            20.0,
            0.1,
            min_sources=25,
            max_radius_deg=1.0,
        )

    def test_meta_different_target_rebuilt(self, tmp_path):
        csv = tmp_path / "cat.csv"
        b = self._builder()
        b._write_outputs(csv, self._df(50), 10.0, 20.0, 0.1, 0.1, 1.0)
        assert not b._catalog_cache_usable(
            self._df(50),
            b._meta_path(csv),
            30.0,
            20.0,
            0.1,
            min_sources=25,
            max_radius_deg=1.0,
        )

    def test_meta_narrower_cone_rebuilt(self, tmp_path):
        csv = tmp_path / "cat.csv"
        b = self._builder()
        b._write_outputs(csv, self._df(50), 10.0, 20.0, 0.1, 0.1, 1.0)
        assert not b._catalog_cache_usable(
            self._df(50),
            b._meta_path(csv),
            10.0,
            20.0,
            0.2,
            min_sources=25,
            max_radius_deg=1.0,
        )


class TestFormatSourceId:
    def test_int_passthrough(self):
        assert _format_gaia_source_id(4295806720) == "4295806720"

    def test_string_digit_passthrough(self):
        assert _format_gaia_source_id("  4295806720 ") == "4295806720"

    def test_float_exact(self):
        assert _format_gaia_source_id(4295806720.0) == "4295806720"

    def test_large_float_no_rounding(self):
        # 64-bit source ids exceed float53 precision; use fixed formatting.
        assert _format_gaia_source_id(1.2345678901234568e18) == "1234567890123456768"

    def test_none_and_nan_empty(self):
        assert _format_gaia_source_id(None) == ""
        assert _format_gaia_source_id(np.nan) == ""

    def test_bool(self):
        assert _format_gaia_source_id(True) == "1"


class TestBandIntegration:
    """AB magnitude from a flat f_nu spectrum must be ~0 for any curve."""

    def _flat_fnu_spectrum(self, wl_nm, f_nu_jy=3631.0):
        f_nu = f_nu_jy * 1e-26  # W m^-2 Hz^-1
        wl_m = wl_nm * 1e-9
        c = 299792458.0
        fl_w_m2_m = f_nu * c / wl_m**2
        fl_w_m2_nm = fl_w_m2_m * 1e-9
        return fl_w_m2_nm

    def test_flat_fnu_gives_ab_zero(self):
        wl = np.linspace(400.0, 900.0, 200)
        flux = self._flat_fnu_spectrum(wl)
        curve_wl = np.array([400.0, 900.0])
        curve_tr = np.array([1.0, 1.0])
        mag, mag_err = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
            wl, flux, None, curve_wl, curve_tr
        )
        assert mag == pytest.approx(0.0, abs=1e-6)
        assert np.isnan(mag_err)

    def test_no_overlap_returns_nan(self):
        wl = np.linspace(400.0, 500.0, 50)
        flux = self._flat_fnu_spectrum(wl)
        curve_wl = np.array([2000.0, 3000.0])
        curve_tr = np.array([1.0, 1.0])
        mag, mag_err = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
            wl, flux, None, curve_wl, curve_tr
        )
        assert np.isnan(mag)
        assert np.isnan(mag_err)

    def test_error_propagates(self):
        wl = np.linspace(400.0, 900.0, 200)
        flux = self._flat_fnu_spectrum(wl)
        err = flux * 0.01  # 1% flux error -> ~0.011 mag
        curve_wl = np.array([400.0, 900.0])
        curve_tr = np.array([1.0, 1.0])
        mag, mag_err = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
            wl, flux, err, curve_wl, curve_tr
        )
        assert mag == pytest.approx(0.0, abs=1e-6)
        # Independent-bin error shrinks with sqrt(N) samples.
        assert 0.0 < mag_err < 0.01

    def test_trapezoid_coeffs_match_trapz(self):
        x = np.array([0.0, 1.0, 2.5, 4.0])
        y = np.array([1.0, 3.0, 2.0, 5.0])
        coeffs = GaiaCurveCatalogBuilder._trapezoid_integration_coeffs(x)
        assert np.dot(coeffs, y) == pytest.approx(np.trapezoid(y, x))

    def test_angstrom_curve_axis_detected(self):
        wl_nm, note = GaiaCurveCatalogBuilder.curve_wavelength_axis_to_nm(
            np.array([4000.0, 9000.0])
        )
        assert wl_nm[0] == pytest.approx(400.0)
        assert "Angstrom" in note

    def test_nm_curve_axis_detected(self):
        wl_nm, note = GaiaCurveCatalogBuilder.curve_wavelength_axis_to_nm(
            np.array([400.0, 900.0])
        )
        assert wl_nm[1] == pytest.approx(900.0)
        assert "nanometres" in note


class TestSpectraMatrix:
    """Stacking GaiaXPy per-row spectra onto a shared wavelength grid."""

    def test_stack_shared_grid(self):
        wave = np.linspace(400.0, 900.0, 50)
        fluxes = [np.full(50, v) for v in (1.0, 2.0, 3.0)]
        df = pd.DataFrame(
            {
                "source_id": [1, 2, 3],
                "flux": fluxes,
                "flux_error": [np.full(50, 0.1)] * 3,
            }
        )
        out = GaiaCurveCatalogBuilder()._spectra_matrix(df, wave)
        assert out is not None
        sids, wv, F, E, ok = out
        assert sids == ["1", "2", "3"]
        np.testing.assert_array_equal(wv, wave)
        assert F.shape == (3, 50)
        np.testing.assert_array_equal(F[1], fluxes[1])
        np.testing.assert_array_equal(E[0], np.full(50, 0.1))
        assert ok.all()

    def test_mismatched_row_marked_not_ok(self):
        wave = np.linspace(400.0, 900.0, 50)
        df = pd.DataFrame(
            {
                "source_id": [1, 2],
                "flux": [np.ones(50), np.ones(30)],
            }
        )
        out = GaiaCurveCatalogBuilder()._spectra_matrix(df, wave)
        _, _, _, E, ok = out
        assert ok.tolist() == [True, False]
        assert E is None

    def test_row_wavelength_must_agree_with_grid(self):
        wave = np.linspace(400.0, 900.0, 50)
        df = pd.DataFrame(
            {
                "source_id": [1],
                "wavelength": [wave + 5.0],
                "flux": [np.ones(50)],
            }
        )
        assert GaiaCurveCatalogBuilder()._spectra_matrix(df, wave) is None

    def test_no_flux_column_returns_none(self):
        df = pd.DataFrame({"source_id": [1], "zz": [np.ones(10)]})
        out = GaiaCurveCatalogBuilder()._spectra_matrix(
            df, np.linspace(400.0, 500.0, 5)
        )
        assert out is None

    def test_heuristic_flux_column_detected(self):
        wave = np.linspace(400.0, 900.0, 50)
        df = pd.DataFrame(
            {
                "source_id": [1],
                "mystery_array": [np.ones(50) * 2.0],
                "mystery_error": [np.ones(50) * 0.1],
            }
        )
        out = GaiaCurveCatalogBuilder()._spectra_matrix(df, wave)
        assert out is not None
        _, _, F, E, ok = out
        assert ok.all()
        np.testing.assert_array_equal(F[0], np.ones(50) * 2.0)
        np.testing.assert_array_equal(E[0], np.ones(50) * 0.1)


class TestBatchBandIntegration:
    """_ab_mags_from_matrix must reproduce the reference per-row path."""

    @staticmethod
    def _flat_fnu(wl_nm, f_nu_jy=3631.0):
        f_nu = f_nu_jy * 1e-26
        wl_m = wl_nm * 1e-9
        return f_nu * 299792458.0 / wl_m**2 * 1e-9

    def test_matches_reference_per_row(self):
        wl = np.linspace(400.0, 900.0, 80)
        F = np.stack(
            [self._flat_fnu(wl, 3631.0 * (1.0 + 0.1 * i)) for i in range(5)]
        )
        E = np.abs(F * 0.01)
        curve_wl = np.array([450.0, 800.0])
        curve_tr = np.array([1.0, 1.0])
        ok = np.ones(5, dtype=bool)
        mags, errs = GaiaCurveCatalogBuilder._ab_mags_from_matrix(
            wl, F, E, ok, curve_wl, curve_tr
        )
        for i in range(5):
            m, e = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
                wl, F[i], E[i], curve_wl, curve_tr
            )
            assert mags[i] == pytest.approx(m, abs=1e-10)
            assert errs[i] == pytest.approx(e, abs=1e-10)

    def test_matches_reference_nonflat(self):
        rng = np.random.default_rng(3)
        wl = np.linspace(350.0, 1000.0, 130)
        F = np.stack(
            [
                self._flat_fnu(wl)
                * (1.0 + 0.2 * np.sin(wl / 50.0) + rng.normal(0.0, 0.02, wl.size))
                for _ in range(4)
            ]
        )
        E = np.abs(F * 0.02)
        curve_wl = np.array([400.0, 500.0, 600.0, 700.0, 800.0, 900.0])
        curve_tr = np.array([0.0, 0.4, 0.9, 0.9, 0.4, 0.0])
        ok = np.ones(4, dtype=bool)
        mags, errs = GaiaCurveCatalogBuilder._ab_mags_from_matrix(
            wl, F, E, ok, curve_wl, curve_tr
        )
        for i in range(4):
            m, e = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
                wl, F[i], E[i], curve_wl, curve_tr
            )
            assert mags[i] == pytest.approx(m, abs=1e-10)
            assert errs[i] == pytest.approx(e, abs=1e-10)

    def test_dirty_row_uses_reference_path(self):
        wl = np.linspace(400.0, 900.0, 80)
        F = np.stack([self._flat_fnu(wl), self._flat_fnu(wl)])
        F[1, 40] = np.nan  # inside the band - not a clean row
        curve_wl = np.array([450.0, 800.0])
        curve_tr = np.array([1.0, 1.0])
        ok = np.ones(2, dtype=bool)
        mags, _ = GaiaCurveCatalogBuilder._ab_mags_from_matrix(
            wl, F, None, ok, curve_wl, curve_tr
        )
        for i in range(2):
            m, _ = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
                wl, F[i], None, curve_wl, curve_tr
            )
            assert mags[i] == pytest.approx(m, abs=1e-10)

    def test_no_overlap_all_nan(self):
        wl = np.linspace(400.0, 500.0, 40)
        F = np.stack([self._flat_fnu(wl)] * 2)
        curve_wl = np.array([2000.0, 3000.0])
        curve_tr = np.array([1.0, 1.0])
        mags, errs = GaiaCurveCatalogBuilder._ab_mags_from_matrix(
            wl, F, None, np.ones(2, dtype=bool), curve_wl, curve_tr
        )
        assert np.isnan(mags).all() and np.isnan(errs).all()


class TestGaiaXPyKwargs:
    """Batched wrappers must not let GaiaXPy write files to cwd."""

    def test_calibrate_passes_save_file_false(self, monkeypatch):
        gaiaxpy = pytest.importorskip("gaiaxpy")

        calls = []

        def fake_calibrate(chunk, **kwargs):
            calls.append(kwargs)
            return (
                pd.DataFrame({"source_id": [1], "flux": [np.ones(3)]}),
                np.array([1.0, 2.0, 3.0]),
            )

        monkeypatch.setattr(gaiaxpy, "calibrate", fake_calibrate)
        gaia_archive.calibrate_source_ids_batched(
            ["1"],
            batch_size=200,
            inter_batch_pause_sec=0.0,
            show_progress=False,
        )
        assert calls and calls[0].get("save_file") is False

    def test_generate_passes_save_file_false(self, monkeypatch):
        gaiaxpy = pytest.importorskip("gaiaxpy")

        calls = []

        def fake_generate(chunk, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame({"source_id": [1]})

        monkeypatch.setattr(gaiaxpy, "generate", fake_generate)
        gaia_archive.generate_source_ids_batched(
            ["1"],
            ["sys"],
            batch_size=200,
            inter_batch_pause_sec=0.0,
            show_progress=False,
        )
        assert calls and calls[0].get("save_file") is False


class TestGaiaXPyRequestCap:
    """Anonymous GaiaXPy archive requests allow at most 50 source IDs
    per call; larger batches are split internally."""

    def test_chunk_size_clamps_to_archive_limit(self):
        from autophot_gaia_curves.gaia_archive import _gaiaxpy_chunk_size

        assert _gaiaxpy_chunk_size(200, 66, None) == 50
        assert _gaiaxpy_chunk_size(0, 66, None) == 50
        assert _gaiaxpy_chunk_size(30, 66, None) == 30
        assert _gaiaxpy_chunk_size(200, 10, None) == 50

    def test_calibrate_splits_overfifty_chunks(self, monkeypatch):
        gaiaxpy = pytest.importorskip("gaiaxpy")

        chunks = []

        def fake_calibrate(chunk, **kwargs):
            chunks.append(list(chunk))
            return (
                pd.DataFrame({"source_id": chunk}),
                np.array([1.0, 2.0, 3.0]),
            )

        monkeypatch.setattr(gaiaxpy, "calibrate", fake_calibrate)
        ids = [str(i) for i in range(66)]
        gaia_archive.calibrate_source_ids_batched(
            ids,
            batch_size=200,
            inter_batch_pause_sec=0.0,
            show_progress=False,
        )
        assert [len(c) for c in chunks] == [50, 16]

    def test_generate_splits_overfifty_chunks(self, monkeypatch):
        gaiaxpy = pytest.importorskip("gaiaxpy")

        chunks = []

        def fake_generate(chunk, **kwargs):
            chunks.append(list(chunk))
            return pd.DataFrame({"source_id": chunk})

        monkeypatch.setattr(gaiaxpy, "generate", fake_generate)
        ids = [str(i) for i in range(66)]
        gaia_archive.generate_source_ids_batched(
            ids,
            ["sys"],
            batch_size=200,
            inter_batch_pause_sec=0.0,
            show_progress=False,
        )
        assert [len(c) for c in chunks] == [50, 16]


class TestGaiaPrintFilter:
    """The quiet-TAP context drops astroquery/GaiaXPy print chatter and
    collapses UWS error dumps to one line."""

    def _filtered(self, text):
        import io

        buf = io.StringIO()
        filt = gaia_archive._FilterGaiaPrintSpam(buf)
        filt.write(text)
        return buf.getvalue()

    def test_status_lines_dropped(self):
        for line in (
            "Running query...",
            "Reading...",
            "Launched uws job: 'anonymous'",
            "Query finished.",
            "Retrieving tables...",
        ):
            assert self._filtered(line) == ""

    def test_release_banner_dropped(self):
        banner = (
            "The current archive release 3.15 contains two breaking "
            "changes. Please see https://www.cosmos.esa.int/web/gaia/"
            "news#GACSrelease315 for details."
        )
        assert self._filtered(banner) == ""

    def test_uws_dump_collapsed_to_digest(self):
        dump = (
            "500 Error 500:\n"
            "Maximum number of Source Ids reached (max: 50, found: 66)\n"
            "\n"
            "Cannot process request: 'https://gea...' "
            "(req: Reqid: anonymous-7fe..., retrieval type: XP_CONTINUOUS)\n"
            " UwsJobOwner{id='anonymous', name='null'}"
        )
        out = self._filtered(dump)
        assert "UwsJobOwner" not in out
        assert "Cannot process request" not in out
        assert "Maximum number of Source Ids" in out

    def test_normal_output_passthrough(self):
        assert self._filtered("Loaded curve g: 392.0-563.0 nm\n") == (
            "Loaded curve g: 392.0-563.0 nm\n"
        )

    def test_quiet_context_suppresses_print_noise(self, capsys):
        with gaia_archive.quiet_gaia_tap_info_logs(True):
            print("Running query...")
            print("real line")
        out = capsys.readouterr().out
        assert "Running query" not in out
        assert "real line" in out


class TestExcDigest:
    def test_multiline_body_collapses(self):
        body = (
            "Error 500:\nMaximum number of Source Ids reached "
            "(max: 50, found: 66)\n\nCannot process request: 'x'\n"
            " UwsJobOwner{id='anonymous'}"
        )
        d = gaia_archive._exc_digest(body)
        assert "\n" not in d
        assert "Maximum number of Source Ids" in d

    def test_single_line_passthrough(self):
        assert gaia_archive._exc_digest(ValueError("boom")) == "boom"

    def test_retry_log_uses_digest(self, caplog):
        import logging

        calls = []

        def fail():
            calls.append(1)
            raise ValueError("line one\nline two\nline three")

        logger = logging.getLogger("test_gaia_retry")
        with pytest.raises(ValueError):
            with caplog.at_level(logging.WARNING, logger="test_gaia_retry"):
                gaia_archive.retry_with_backoff(
                    fail,
                    max_retries=2,
                    base_delay_sec=0.0,
                    logger=logger,
                    op_name="test op",
                )
        assert len(calls) == 2
        msg = caplog.records[0].getMessage()
        assert "line three" not in msg
        assert "line one" in msg
