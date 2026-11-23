"""Tests for uniform catalog coverage across the image FOV.

A cone query that caps rows by distance order, or a release whose
footprint ends inside the field, leaves part of the detector without
calibrators.  These tests cover the spatial-subsample, the coverage
metric, the SDSS release picker, and the FOV-aware query radius.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import autophot.calibration.catalog as catalog_mod
from autophot.calibration.catalog import (
    Catalog,
    _catalog_cone_radius_arcmin,
    _catalog_radec,
    _field_coverage_fraction,
    _footprint_radius_arcmin,
    _pixscale_arcsec,
    _select_sdss_vizier_table,
    _sky_uniform_subsample,
)

CENTER = SkyCoord(150.0, 30.0, unit="deg", frame="icrs")
RADIUS_DEG = 0.2


@pytest.fixture(autouse=True)
def _fresh_coverage_warnings():
    """The download() coverage warning dedups on a module-level set."""
    catalog_mod._COVERAGE_WARNED_KEYS.clear()
    yield
    catalog_mod._COVERAGE_WARNED_KEYS.clear()


def _disk_sources(n, radius_deg=RADIUS_DEG, seed=0):
    """Uniform positions inside a disk around CENTER."""
    rng = np.random.default_rng(seed)
    r = radius_deg * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    ra = CENTER.ra.deg + x / np.cos(np.radians(CENTER.dec.deg))
    dec = CENTER.dec.deg + y
    return ra, dec


def _tan_wcs(naxis=200, scale_as=1.0, crval=None):
    w = WCS(naxis=2)
    w.wcs.crpix = [naxis / 2.0, naxis / 2.0]
    w.wcs.cdelt = [-scale_as / 3600.0, scale_as / 3600.0]
    ra0, dec0 = (crval or (CENTER.ra.deg, CENTER.dec.deg))
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.array_shape = (naxis, naxis)
    return w


class TestCatalogRadec:
    def test_common_column_names(self):
        ra = np.array([1.0, 2.0])
        dec = np.array([3.0, 4.0])
        for ra_col, dec_col in (
            ("RA", "DEC"),
            ("ra", "dec"),
            ("RA_ICRS", "DE_ICRS"),
            ("RAJ2000", "DEJ2000"),
            ("raMean", "decMean"),
        ):
            df = pd.DataFrame({ra_col: ra, dec_col: dec})
            got_ra, got_dec = _catalog_radec(df)
            np.testing.assert_allclose(got_ra, ra)
            np.testing.assert_allclose(got_dec, dec)

    def test_missing_columns_returns_none(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        assert _catalog_radec(df) == (None, None)

    def test_empty_returns_none(self):
        assert _catalog_radec(pd.DataFrame()) == (None, None)


class TestSkyUniformSubsample:
    def test_under_cap_returns_all(self):
        ra, dec = _disk_sources(50)
        df = pd.DataFrame({"ra": ra, "dec": dec})
        out = _sky_uniform_subsample(df, 100, center_ra=CENTER.ra.deg)
        assert len(out) == 50

    def test_spatial_coverage_preserved(self):
        """A nearest-N cap would keep only the centre; the subsample must
        keep the edge ring populated."""
        rng = np.random.default_rng(1)
        n_inner, n_ring = 3000, 600
        ri = 0.06 * np.sqrt(rng.uniform(0, 1, n_inner))
        rr = rng.uniform(0.16, 0.20, n_ring)
        ti = rng.uniform(0, 2 * np.pi, n_inner)
        tr = rng.uniform(0, 2 * np.pi, n_ring)
        r = np.concatenate([ri, rr])
        th = np.concatenate([ti, tr])
        x = r * np.cos(th)
        y = r * np.sin(th)
        df = pd.DataFrame(
            {
                "RA": CENTER.ra.deg
                + x / np.cos(np.radians(CENTER.dec.deg)),
                "DEC": CENTER.dec.deg + y,
            }
        )

        out = _sky_uniform_subsample(df, 250, center_ra=CENTER.ra.deg)
        assert len(out) == 250

        # The outer ring (r > 0.16 deg) must still be represented - a
        # nearest-to-centre cap would keep none of the 600 ring sources.
        out_x = (out["RA"].to_numpy() - CENTER.ra.deg) * np.cos(
            np.radians(CENTER.dec.deg)
        )
        out_r = np.hypot(out_x, out["DEC"].to_numpy() - CENTER.dec.deg)
        assert (out_r > 0.16).sum() > 20

    def test_native_column_names(self):
        """Works on raw catalog column names, not just RA/DEC."""
        ra, dec = _disk_sources(500)
        df = pd.DataFrame({"raMean": ra, "decMean": dec, "g": 15.0})
        out = _sky_uniform_subsample(df, 100, center_ra=CENTER.ra.deg)
        assert len(out) == 100

    def test_no_radec_falls_back_to_head(self):
        df = pd.DataFrame({"a": np.arange(50)})
        out = _sky_uniform_subsample(df, 10)
        assert len(out) == 10
        assert (out["a"].to_numpy() == np.arange(10)).all()


class TestFieldCoverageFraction:
    def test_uniform_disk_full_coverage(self):
        ra, dec = _disk_sources(2000)
        cov = _field_coverage_fraction(
            ra, dec, CENTER.ra.deg, CENTER.dec.deg, RADIUS_DEG
        )
        assert cov > 0.95

    def test_half_plane_flagged(self):
        """SDSS-style survey boundary: sources only on one side."""
        ra, dec = _disk_sources(2000)
        keep = ra <= CENTER.ra.deg  # eastern half only (RA < centre)
        cov = _field_coverage_fraction(
            ra[keep],
            dec[keep],
            CENTER.ra.deg,
            CENTER.dec.deg,
            RADIUS_DEG,
        )
        assert cov < 0.8

    def test_inner_clump_flagged(self):
        """Distance-ordered TOP-N truncation leaves an inner disk."""
        ra, dec = _disk_sources(2000)
        x = (ra - CENTER.ra.deg) * np.cos(np.radians(CENTER.dec.deg))
        r = np.hypot(x, dec - CENTER.dec.deg)
        inner = r < 0.5 * RADIUS_DEG
        cov = _field_coverage_fraction(
            ra[inner],
            dec[inner],
            CENTER.ra.deg,
            CENTER.dec.deg,
            RADIUS_DEG,
        )
        assert cov < 0.8

    def test_empty_catalog(self):
        assert np.isnan(
            _field_coverage_fraction(
                [], [], CENTER.ra.deg, CENTER.dec.deg, RADIUS_DEG
            )
        )
        assert np.isnan(
            _field_coverage_fraction(
                [150.0], [30.0], CENTER.ra.deg, CENTER.dec.deg, 0.0
            )
        )


class TestFootprintRadiusArcmin:
    def test_covers_detector_corners(self):
        w = _tan_wcs(naxis=2000, scale_as=1.0)
        # corner sep ~ sqrt(2)*1000" = 1414" = 23.6'; x1.1 margin ~ 25.9'
        r = _footprint_radius_arcmin(
            [{"wcs": w, "shape": (2000, 2000)}], CENTER
        )
        assert r == pytest.approx(25.9, rel=0.05)

    def test_no_wcs_returns_none(self):
        assert (
            _footprint_radius_arcmin(
                [{"wcs": None, "shape": None}], CENTER
            )
            is None
        )
        assert _footprint_radius_arcmin([], CENTER) is None

    def test_no_wcs_uses_pixscale(self):
        """No-WCS images still size the cone from the detector
        half-diagonal at the image pixel scale."""
        # 2000x2000 px at 1"/px -> half-diagonal ~23.6'; x1.1 ~25.9'.
        r = _footprint_radius_arcmin(
            [{"wcs": None, "shape": (2000, 2000), "pixscale": 1.0}],
            CENTER,
        )
        assert r == pytest.approx(25.9, rel=0.05)

    def test_no_wcs_bad_pixscale_skipped(self):
        for bad in (None, np.nan, 0.0, -1.0, "junk"):
            assert (
                _footprint_radius_arcmin(
                    [
                        {
                            "wcs": None,
                            "shape": (2000, 2000),
                            "pixscale": bad,
                        }
                    ],
                    CENTER,
                )
                is None
            )

    def test_broken_wcs_falls_back_to_pixscale(self):
        """A WCS whose transform raises must not lose the footprint."""
        w = _tan_wcs()
        w.all_pix2world = lambda *a, **k: (_ for _ in ()).throw(
            ValueError("boom")
        )
        r = _footprint_radius_arcmin(
            [{"wcs": w, "shape": (1200, 1200), "pixscale": 0.5}],
            CENTER,
        )
        # half-diagonal = 0.5*hypot(1200,1200)*0.5"/60 ~ 7.07'; x1.1 ~7.8'
        assert r == pytest.approx(7.8, rel=0.05)

    def test_union_over_offsets(self):
        """An offset pointing raises the radius above the centred case."""
        w_near = _tan_wcs(naxis=200, scale_as=1.0)
        w_far = _tan_wcs(
            naxis=200, scale_as=1.0, crval=(CENTER.ra.deg + 0.5, CENTER.dec.deg)
        )
        r_single = _footprint_radius_arcmin(
            [{"wcs": w_near, "shape": (200, 200)}], CENTER
        )
        r_union = _footprint_radius_arcmin(
            [
                {"wcs": w_near, "shape": (200, 200)},
                {"wcs": w_far, "shape": (200, 200)},
            ],
            CENTER,
        )
        assert r_union > r_single + 20.0  # 0.5 deg offset dominates


class TestCatalogConeRadiusArcmin:
    def test_footprint_limits_default(self):
        """A narrow field must shrink the cone below the 10' default -
        the footprint extent limits, not just enlarges."""
        w = _tan_wcs(naxis=200, scale_as=1.0)  # ~2.6' corner sep
        r = _catalog_cone_radius_arcmin(
            [{"wcs": w, "shape": (200, 200)}], CENTER, default_arcmin=10.0
        )
        assert r < 10.0
        assert r == pytest.approx(2.6, rel=0.05)

    def test_footprint_enlarges_default(self):
        w = _tan_wcs(naxis=2000, scale_as=1.0)  # ~25.9' corner sep
        r = _catalog_cone_radius_arcmin(
            [{"wcs": w, "shape": (2000, 2000)}], CENTER,
            default_arcmin=10.0,
        )
        assert r == pytest.approx(25.9, rel=0.05)

    def test_extremes_clamped(self):
        w_huge = _tan_wcs(naxis=30000, scale_as=1.0)  # >5 deg field
        r = _catalog_cone_radius_arcmin(
            [{"wcs": w_huge, "shape": (30000, 30000)}], CENTER,
            default_arcmin=10.0,
        )
        assert r == 60.0
        w_tiny = _tan_wcs(naxis=10, scale_as=0.5)  # ~0.1' footprint
        r = _catalog_cone_radius_arcmin(
            [{"wcs": w_tiny, "shape": (10, 10)}], CENTER,
            default_arcmin=10.0,
        )
        assert r == 2.0

    def test_no_footprint_info_returns_default(self):
        """A dataset with no WCS and no pixel scale keeps the default."""
        infos = [
            {"wcs": None, "shape": None},
            {"wcs": None, "shape": (1000, 1000), "pixscale": None},
            {"wcs": None, "shape": (1000, 1000), "pixscale": np.nan},
        ]
        assert (
            _catalog_cone_radius_arcmin(infos, CENTER, default_arcmin=10.0)
            == 10.0
        )
        assert (
            _catalog_cone_radius_arcmin([], CENTER, default_arcmin=10.0)
            == 10.0
        )

    def test_no_wcs_pixscale_sizes_cone(self):
        r = _catalog_cone_radius_arcmin(
            [{"wcs": None, "shape": (2000, 2000), "pixscale": 1.0}],
            CENTER,
            default_arcmin=10.0,
        )
        assert r == pytest.approx(25.9, rel=0.05)


class TestPixscaleArcsec:
    def test_keyword_priority(self):
        from astropy.io.fits import Header

        hdr = Header()
        hdr["PIXSCALE"] = 0.42
        assert _pixscale_arcsec(hdr) == pytest.approx(0.42)
        hdr["CDELT1"] = -1e-4  # deg/pix = 0.36"/pix; PIXSCALE wins
        assert _pixscale_arcsec(hdr) == pytest.approx(0.42)

    def test_cd_and_cdelt_fallback(self):
        from astropy.io.fits import Header

        hdr = Header()
        hdr["CDELT1"] = -2e-4  # 0.72 "/pix
        assert _pixscale_arcsec(hdr) == pytest.approx(0.72)
        hdr = Header()
        hdr["CD1_1"], hdr["CD1_2"] = -2e-4, 0.0
        hdr["CD2_1"], hdr["CD2_2"] = 0.0, 2e-4
        assert _pixscale_arcsec(hdr) == pytest.approx(0.72)

    def test_missing_or_bad_returns_none(self):
        from astropy.io.fits import Header

        assert _pixscale_arcsec(None) is None
        assert _pixscale_arcsec(Header()) is None
        hdr = Header()
        hdr["PIXSCALE"] = "not-a-number"
        assert _pixscale_arcsec(hdr) is None
        hdr["PIXSCALE"] = 500.0  # beyond the sanity bound
        assert _pixscale_arcsec(hdr) is None


def _sdss_table(name, ra, dec, flags):
    """Fake VizieR table with explicit positions and flag columns."""
    data = {"RA_ICRS": ra, "DE_ICRS": dec}
    data.update(flags)
    t = Table(data)
    t.meta["name"] = name
    return t


class TestSelectSdssVizierTable:
    _geo = {
        "center_ra": CENTER.ra.deg,
        "center_dec": CENTER.dec.deg,
        "radius_deg": RADIUS_DEG,
    }

    def test_better_coverage_beats_newer_release(self):
        """DR9 covering the whole disk beats DR16 covering half of it."""
        ra_full, dec_full = _disk_sources(40)
        ra_h, dec_h = _disk_sources(80)
        half = ra_h <= CENTER.ra.deg
        t9 = _sdss_table(
            "V/139/sdss9",
            ra_full,
            dec_full,
            {"cl": np.full(len(ra_full), 6)},
        )
        t16 = _sdss_table(
            "V/154/sdss16",
            ra_h[half],
            dec_h[half],
            {"class": np.full(int(half.sum()), 6)},
        )
        out = _select_sdss_vizier_table([t9, t16], **self._geo)
        assert "cl" in out.columns  # the DR9 frame was chosen

    def test_equal_coverage_prefers_newer_release(self):
        ra, dec = _disk_sources(30)
        t9 = _sdss_table(
            "V/139/sdss9", ra, dec, {"cl": np.full(len(ra), 6)}
        )
        t16 = _sdss_table(
            "V/154/sdss16", ra, dec, {"class": np.full(len(ra), 6)}
        )
        out = _select_sdss_vizier_table([t9, t16], **self._geo)
        assert len(out) == 30
        assert "class" in out.columns  # DR16 frame chosen

    def test_star_cuts_applied(self):
        """mode!=1, class!=6 and clean==0 rows do not count."""
        ra, dec = _disk_sources(20)
        t16 = _sdss_table(
            "V/154/sdss16",
            ra,
            dec,
            {
                "mode": np.concatenate([np.ones(15), np.full(5, 3)]),
                "class": np.concatenate([np.full(10, 6), np.full(10, 3)]),
                "clean": np.concatenate(
                    [np.ones(8), np.zeros(2), np.zeros(10)]
                ),
            },
        )
        # mode==1: first 15; class==6: first 10; clean==1: first 8 -> 8.
        out = _select_sdss_vizier_table([t16], **self._geo)
        assert len(out) == 8

    def test_all_empty_returns_empty(self):
        t = _sdss_table(
            "V/154/sdss16",
            np.array([]),
            np.array([]),
            {"class": np.array([], dtype=int)},
        )
        out = _select_sdss_vizier_table([t], **self._geo)
        assert len(out) == 0


class TestDownloadCoverageWarning:
    def _cat(self, tmp_path):
        return Catalog(
            input_yaml={
                "wdir": str(tmp_path),
                "target_name": "T1",
                "target_ra": CENTER.ra.deg,
                "target_dec": CENTER.dec.deg,
                "catalog": {"use_catalog": "sdss", "max_distance": 30},
            }
        )

    def _fake_vizier(self, ra, dec):
        t = Table({"RA_ICRS": ra, "DE_ICRS": dec, "cl": np.full(len(ra), 6)})
        t.meta["name"] = "V/139/sdss9"
        return [t]

    def test_partial_coverage_warns(self, tmp_path, monkeypatch, caplog):
        ra, dec = _disk_sources(2000)
        keep = ra <= CENTER.ra.deg  # half-plane, like an SDSS stripe edge

        monkeypatch.setattr(
            catalog_mod.Vizier,
            "query_region",
            lambda *a, **k: self._fake_vizier(ra[keep], dec[keep]),
        )
        cat = self._cat(tmp_path)
        with caplog.at_level("WARNING", logger="catalog"):
            out = cat.download(CENTER, catalogName="sdss", radius=12.0)
        assert len(out) == int(keep.sum())
        assert any("covers only" in r.getMessage() for r in caplog.records)

    def test_full_coverage_no_warning(self, tmp_path, monkeypatch, caplog):
        ra, dec = _disk_sources(2000)
        monkeypatch.setattr(
            catalog_mod.Vizier,
            "query_region",
            lambda *a, **k: self._fake_vizier(ra, dec),
        )
        cat = self._cat(tmp_path)
        with caplog.at_level("WARNING", logger="catalog"):
            cat.download(CENTER, catalogName="sdss", radius=12.0)
        assert not any("covers only" in r.getMessage() for r in caplog.records)

    def test_max_sources_subsample_is_spatial(
        self, tmp_path, monkeypatch
    ):
        """download(max_sources=N) keeps edge coverage, not nearest-N."""
        rng = np.random.default_rng(3)
        n_inner, n_ring = 3000, 800
        ri = 0.05 * np.sqrt(rng.uniform(0, 1, n_inner))
        rr = rng.uniform(0.15, 0.20, n_ring)
        th = rng.uniform(0, 2 * np.pi, n_inner + n_ring)
        r = np.concatenate([ri, rr])
        x = r * np.cos(th)
        y = r * np.sin(th)
        ra = CENTER.ra.deg + x / np.cos(np.radians(CENTER.dec.deg))
        dec = CENTER.dec.deg + y

        monkeypatch.setattr(
            catalog_mod.Vizier,
            "query_region",
            lambda *a, **k: self._fake_vizier(ra, dec),
        )
        cat = self._cat(tmp_path)
        out = cat.download(
            CENTER, catalogName="sdss", radius=12.0, max_sources=200
        )
        assert len(out) == 200
        out_r = np.hypot(
            (out["RA_ICRS"].to_numpy() - CENTER.ra.deg)
            * np.cos(np.radians(CENTER.dec.deg)),
            out["DE_ICRS"].to_numpy() - CENTER.dec.deg,
        )
        assert (out_r > 0.15).sum() > 0


class TestDownloadRowCap:
    """max_download_rows bounds crowded-field catalogs for every caller."""

    def _cat(self, tmp_path, cap):
        return Catalog(
            input_yaml={
                "wdir": str(tmp_path),
                "target_name": "T1",
                "target_ra": CENTER.ra.deg,
                "target_dec": CENTER.dec.deg,
                "catalog": {
                    "use_catalog": "sdss",
                    "max_distance": 30,
                    "max_download_rows": cap,
                },
            }
        )

    def _fake_vizier(self, ra, dec):
        t = Table({"RA_ICRS": ra, "DE_ICRS": dec, "cl": np.full(len(ra), 6)})
        t.meta["name"] = "V/139/sdss9"
        return [t]

    def _patch(self, monkeypatch, n=2000):
        ra, dec = _disk_sources(n)
        monkeypatch.setattr(
            catalog_mod.Vizier,
            "query_region",
            lambda *a, **k: self._fake_vizier(ra, dec),
        )

    def test_config_cap_applies_without_kwarg(self, tmp_path, monkeypatch):
        """A Galactic-field-sized download is trimmed to the config cap."""
        self._patch(monkeypatch, n=2000)
        out = self._cat(tmp_path, cap=200).download(
            CENTER, catalogName="sdss", radius=12.0
        )
        assert len(out) == 200

    def test_explicit_max_sources_beats_config(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, n=2000)
        out = self._cat(tmp_path, cap=200).download(
            CENTER, catalogName="sdss", radius=12.0, max_sources=50
        )
        assert len(out) == 50

    def test_zero_cap_disables(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, n=2000)
        out = self._cat(tmp_path, cap=0).download(
            CENTER, catalogName="sdss", radius=12.0
        )
        assert len(out) == 2000

    def test_cap_keeps_edge_coverage(self, tmp_path, monkeypatch):
        """The cap is a spatial subsample, not a nearest-N truncation."""
        self._patch(monkeypatch, n=2000)
        out = self._cat(tmp_path, cap=200).download(
            CENTER, catalogName="sdss", radius=12.0
        )
        out_r = np.hypot(
            (out["RA_ICRS"].to_numpy() - CENTER.ra.deg)
            * np.cos(np.radians(CENTER.dec.deg)),
            out["DE_ICRS"].to_numpy() - CENTER.dec.deg,
        )
        # A centre-truncated catalog would drop the outer annulus entirely.
        assert (out_r > 0.15).sum() > 0


class TestBuildCatalogRegionCacheKey:
    """The combined-catalog CSV must key on the per-catalog regions."""

    def test_region_change_invalidates_cache(self, tmp_path, monkeypatch):
        calls = []

        def fake_download(self, target_coords, catalogName=None, **kw):
            calls.append(catalogName)
            ra, dec = _disk_sources(20)
            return pd.DataFrame({"raMean": ra, "decMean": dec})

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = Catalog(
            input_yaml={
                "wdir": str(tmp_path),
                "target_name": "T1",
                "target_ra": CENTER.ra.deg,
                "target_dec": CENTER.dec.deg,
                "imageFilter": "r",
                "catalog": {"use_catalog": "pan_starrs"},
            }
        )
        region_a = {"ra": 150.0, "dec": 30.0, "radius_arcmin": 10.0}
        region_b = {"ra": 150.2, "dec": 30.0, "radius_arcmin": 15.0}

        cat.build_complete_catalog(
            CENTER,
            target_name="T1",
            catalog_list=["pan_starrs"],
            regions={"pan_starrs": region_a},
        )
        assert len(calls) == 1
        # Same regions -> the cached combined CSV is reused.
        cat.build_complete_catalog(
            CENTER,
            target_name="T1",
            catalog_list=["pan_starrs"],
            regions={"pan_starrs": region_a},
        )
        assert len(calls) == 1
        # A different footprint region must not reuse the stale build.
        cat.build_complete_catalog(
            CENTER,
            target_name="T1",
            catalog_list=["pan_starrs"],
            regions={"pan_starrs": region_b},
        )
        assert len(calls) == 2


class TestOptimizerCoverageRadius:
    def test_radius_enlarged_and_report_has_coverage(
        self, tmp_path, monkeypatch
    ):
        """A wide footprint must grow the cone beyond the 10 arcmin default."""
        w = _tan_wcs(naxis=2000, scale_as=1.0)  # ~23.6' corner sep
        images = [{"path": "/tmp/i.fits", "band": "r", "wcs": w,
                   "shape": (2000, 2000)}]
        captured = {}

        def fake_download(self, target_coords, catalogName=None, **kw):
            captured["radius"] = kw.get("radius")
            ra, dec = _disk_sources(30)
            return pd.DataFrame(
                {
                    "RA_ICRS": ra,
                    "DE_ICRS": dec,
                    "rmag": np.full(len(ra), 15.0),
                    "e_rmag": np.full(len(ra), 0.01),
                }
            )

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = Catalog(
            input_yaml={
                "wdir": str(tmp_path),
                "target_name": "T1",
                "target_ra": CENTER.ra.deg,
                "target_dec": CENTER.dec.deg,
                "imageFilter": "r",
                "catalog": {"use_catalog": "auto", "max_distance": 30},
                "zeropoint": {"bright_mag_limit": 11.0,
                              "faint_mag_limit": 22.0},
            }
        )
        out = cat.find_optimized_catalog(
            CENTER,
            images=images,
            bands=["r"],
            catalog_names=["sdss"],
            write_report=False,
            write_plot=False,
        )
        assert captured["radius"] > 10.0
        assert "coverage" in out["report"].columns


def _refcat_frame(n=200, seed=3, complete=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "RA": CENTER.ra.deg + (rng.random(n) - 0.5) * 0.3,
            "Dec": CENTER.dec.deg + (rng.random(n) - 0.5) * 0.3,
            "g": rng.uniform(15, 21, n),
            "dg": rng.uniform(0.001, 0.09, n),
            "r": rng.uniform(15, 21, n),
            "dr": rng.uniform(0.001, 0.09, n),
            "i": rng.uniform(15, 21, n),
            "di": rng.uniform(0.01, 0.09, n),
            "z": rng.uniform(15, 21, n),
            "dz": rng.uniform(0.01, 0.09, n),
            "J": rng.uniform(14, 19, n),
            "dJ": rng.uniform(0.01, 0.09, n),
            "H": rng.uniform(14, 19, n),
            "dH": rng.uniform(0.01, 0.09, n),
            "K": rng.uniform(14, 19, n),
            "dK": rng.uniform(0.01, 0.09, n),
        }
    )
    if not complete:
        df.loc[0, "g"] = 0.0
        df.loc[1, "dr"] = 0.5
    return df


class TestRefcatBackends:
    def test_schema_maps_vizier_names(self):
        df = _refcat_frame(20).rename(
            columns={
                "RA": "RA_ICRS", "Dec": "DE_ICRS",
                "g": "gmag", "dg": "e_gmag", "r": "rmag",
                "dr": "e_rmag", "i": "imag", "di": "e_imag",
                "z": "zmag", "dz": "e_zmag", "J": "Jmag",
                "dJ": "e_Jmag", "H": "Hmag", "dH": "e_Hmag",
                "K": "Kmag", "dK": "e_Kmag",
            }
        )
        sch = Catalog({})._refcat2_schema(df)
        assert list(sch.columns)[:2] == ["RA", "Dec"]
        assert len(sch) == 20
        assert np.isfinite(sch["g"]).all()

    def test_schema_drops_incomplete_and_high_derr(self):
        df = _refcat_frame(20, complete=False)
        sch = Catalog({})._refcat2_schema(df)
        # row 0 carries a 0 sentinel, row 1 fails dr<0.1
        assert len(sch) == 18

    def test_local_backend_cone_filter(self, tmp_path):
        zone_dir = tmp_path / "zones"
        zone_dir.mkdir()
        df = _refcat_frame(50)
        df.loc[0, "RA"] = CENTER.ra.deg
        df.loc[0, "Dec"] = CENTER.dec.deg
        df.loc[1:, "Dec"] = 41.5  # far outside the cone
        df.to_csv(zone_dir / "refcat2_dec+40.csv", index=False)
        cat = Catalog({})
        out = cat.fetch_refcat2_local(
            str(zone_dir), CENTER.ra.deg, CENTER.dec.deg, 0.05
        )
        assert out is not None and len(out) == 1
        assert abs(out["RA"].iloc[0] - CENTER.ra.deg) < 1e-4

    def test_local_backend_missing_dir(self, tmp_path):
        cat = Catalog({})
        assert (
            cat.fetch_refcat2_local(str(tmp_path / "none"), 0.0, 0.0, 0.1)
            is None
        )

    def test_vizier_mirror_fallback(self, monkeypatch):
        cat = Catalog({})
        calls = []

        class _FakeVizier:
            def __init__(self, **kw):
                self.server = kw["vizier_server"]

            def query_region(self, coords, radius, catalog):
                calls.append(self.server)
                if "harvard" not in self.server:
                    raise RuntimeError("mirror down")
                tbl = Table.from_pandas(_refcat_frame(30))
                return [tbl]

        monkeypatch.setattr(catalog_mod, "Vizier", _FakeVizier)
        out = cat.fetch_refcat2_vizier(
            CENTER.ra.deg, CENTER.dec.deg, 0.1,
            servers=["vizier.dead.example", "vizier.cfa.harvard.edu"],
        )
        assert calls == [
            "vizier.dead.example",
            "vizier.cfa.harvard.edu",
        ]
        assert out is not None and len(out) == 30

    def test_download_ladder_vizier_then_csv_cache(
        self, tmp_path, monkeypatch
    ):
        cfg = {
            "wdir": str(tmp_path),
            "target_name": "T1",
            "imageFilter": "g",
            "catalog": {
                "use_catalog": "refcat",
                "refcat_backend": "vizier",
                "refcat_nsources": 25,
            },
        }
        cat = Catalog(input_yaml=cfg)

        class _FakeVizier:
            def __init__(self, **kw):
                pass

            def query_region(self, coords, radius, catalog):
                return [Table.from_pandas(_refcat_frame(300))]

        monkeypatch.setattr(catalog_mod, "Vizier", _FakeVizier)
        out = cat.download(CENTER, "refcat", radius=10)
        # 300 rows fetched -> capped + spatially subsampled to 25
        assert len(out) == 25
        cached = list(tmp_path.rglob("*refcat_n25*.csv"))
        assert len(cached) == 1

    def test_download_ladder_local_first(self, tmp_path, monkeypatch):
        zone_dir = tmp_path / "zones"
        zone_dir.mkdir()
        _refcat_frame(50).to_csv(zone_dir / "z.csv", index=False)
        cfg = {
            "wdir": str(tmp_path),
            "target_name": "T1",
            "imageFilter": "g",
            "catalog": {
                "use_catalog": "refcat",
                "refcat_backend": "auto",
                "refcat_local_dir": str(zone_dir),
                "refcat_nsources": 100,
            },
        }
        cat = Catalog(input_yaml=cfg)

        def _boom(*a, **k):
            raise AssertionError("network path must not run")

        monkeypatch.setattr(cat, "fetch_refcat2_vizier", _boom)
        monkeypatch.setattr(cat, "fetch_refcat2_field", _boom)
        out = cat.download(CENTER, "refcat", radius=10)
        assert out is not None and len(out) <= 100
