"""Tests for Catalog.find_optimized_catalog and `use_catalog: "auto"`.

All network access is mocked: ``Catalog.download`` is replaced with a stub
returning synthetic raw catalogs whose column names match
``databases/catalog.yml`` so the real ``clean()`` column mapping runs.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog import (
    AUTO_OPTIMIZE_CATALOGS,
    Catalog,
    _coverage_optimal_center,
    _in_required_region,
    _region_from_corners,
)
from functions import SUPPORTED_PHOTOMETRIC_FILTERS

try:
    from prepare import Prepare
except Exception:  # pragma: no cover - optional heavy deps (tqdm etc.)
    Prepare = None


CENTER = SkyCoord(150.0, 30.0, unit="deg", frame="icrs")


def _tan_wcs():
    """Simple 1 arcsec/pixel TAN WCS centred on the field."""
    w = WCS(naxis=2)
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    w.wcs.crval = [150.0, 30.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.array_shape = (200, 200)
    return w


def _offset_wcs(dra_arcmin, ddec_arcmin=0.0):
    """TAN WCS for the same detector, centred off-target."""
    w = _tan_wcs()
    cosd = np.cos(np.radians(CENTER.dec.deg))
    w.wcs.crval = [
        CENTER.ra.deg + dra_arcmin / 60.0 / cosd,
        CENTER.dec.deg + ddec_arcmin / 60.0,
    ]
    return w


def _field_ras_decs(n):
    """n positions inside a ~100x100 arcsec box around CENTER."""
    idx = np.arange(n)
    # ~10 arcsec steps in each axis, wrapped into the box.
    ra = CENTER.ra.deg + (((idx % 10) - 4.5) * 10.0) / 3600.0 / np.cos(
        np.radians(CENTER.dec.deg)
    )
    dec = CENTER.dec.deg + (((idx // 10) - 2.0) * 10.0) / 3600.0
    return ra, dec


def _raw_catalog(name, n, mag_val):
    """Raw-format rows for a catalog, using its native column names."""
    ra, dec = _field_ras_decs(n)
    base = {}
    if name == "pan_starrs":
        base = {
            "raMean": ra,
            "decMean": dec,
            "gMeanPSFMag": np.full(n, mag_val),
            "gMeanPSFMagErr": np.full(n, 0.01),
            "rMeanPSFMag": np.full(n, mag_val),
            "rMeanPSFMagErr": np.full(n, 0.01),
            "iMeanPSFMag": np.full(n, mag_val),
            "iMeanPSFMagErr": np.full(n, 0.01),
        }
    elif name == "sdss":
        base = {
            "RA_ICRS": ra,
            "DE_ICRS": dec,
            "gmag": np.full(n, mag_val),
            "e_gmag": np.full(n, 0.01),
            "rmag": np.full(n, mag_val),
            "e_rmag": np.full(n, 0.01),
            "imag": np.full(n, mag_val),
            "e_imag": np.full(n, 0.01),
        }
    elif name == "2mass":
        base = {
            "RAJ2000": ra,
            "DEJ2000": dec,
            "Jmag": np.full(n, mag_val),
            "e_Jmag": np.full(n, 0.01),
            "Hmag": np.full(n, mag_val),
            "e_Hmag": np.full(n, 0.01),
            "Kmag": np.full(n, mag_val),
            "e_Kmag": np.full(n, 0.01),
        }
    else:
        base = {"ra": ra, "dec": dec}
    return pd.DataFrame(base)


def _make_catalog(tmp_path, **overrides):
    input_yaml = {
        "wdir": str(tmp_path),
        "target_name": "T1",
        "target_ra": CENTER.ra.deg,
        "target_dec": CENTER.dec.deg,
        "imageFilter": "r",
        "catalog": {"use_catalog": "auto", "max_distance": 10},
        "zeropoint": {"bright_mag_limit": 11.0, "faint_mag_limit": 22.0},
    }
    input_yaml.update(overrides)
    return Catalog(input_yaml=input_yaml)


@pytest.fixture
def image_pair():
    w = _tan_wcs()
    return [
        {"path": "/tmp/img_r.fits", "band": "r", "wcs": w, "shape": (200, 200)},
        {"path": "/tmp/img_j.fits", "band": "J", "wcs": w, "shape": (200, 200)},
    ]


class TestFindOptimizedCatalog:
    def test_winner_per_band(self, tmp_path, monkeypatch, image_pair):
        """pan_starrs wins r (more sources), 2mass is the only J provider."""
        spec = {"pan_starrs": 40, "sdss": 5, "2mass": 25}

        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, spec[catalogName], 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER,
            images=image_pair,
            bands=["r", "J"],
            catalog_names=["pan_starrs", "sdss", "2mass"],
        )

        assert out["winners"]["r"] == "pan_starrs"
        assert out["winners"]["J"] == "2mass"
        assert out["use_catalog"]["r"] == "pan_starrs"
        assert out["use_catalog"]["J"] == "2mass"
        assert out["use_catalog"]["default"] in {"pan_starrs", "2mass"}
        rep = out["report"]
        assert set(rep["mode"]) == {"onchip"}
        # sdss scored but lost r.
        row = rep[(rep.catalog == "sdss") & (rep.band == "r")]
        assert row["n_usable"].iloc[0] == 5

    def test_magnitude_window_respected(self, tmp_path, monkeypatch, image_pair):
        """Sources outside the ZP bright/faint window do not count."""
        # sdss: 40 stars but all too faint (24); pan_starrs: 8 usable.
        def fake_download(self, target_coords, catalogName=None, **kw):
            if catalogName == "sdss":
                return _raw_catalog("sdss", 40, 24.0)
            return _raw_catalog(catalogName, 8, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER,
            images=image_pair,
            bands=["r"],
            catalog_names=["pan_starrs", "sdss"],
        )
        assert out["winners"]["r"] == "pan_starrs"
        row = out["report"][
            (out["report"].catalog == "sdss") & (out["report"].band == "r")
        ]
        assert row["n_usable"].iloc[0] == 0

    def test_off_chip_sources_not_counted(self, tmp_path, monkeypatch):
        """Sources outside the detector footprint do not count."""
        w = _tan_wcs()
        images = [
            {"path": "/tmp/i.fits", "band": "r", "wcs": w, "shape": (200, 200)}
        ]

        def fake_download(self, target_coords, catalogName=None, **kw):
            df = _raw_catalog(catalogName, 6, 15.0)
            # Move every source 10 deg away: on the sky but off the chip.
            for c in df.columns:
                if c.lower().startswith(("ra", "raj")) or c == "RA_ICRS":
                    df[c] = df[c] + 10.0
            return df

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER, images=images, bands=["r"], catalog_names=["sdss"]
        )
        assert out["report"]["n_usable"].max() == 0

    def test_field_level_fallback_without_wcs(self, tmp_path, monkeypatch):
        """No WCS -> field-level count (mode='field'), still usable."""
        images = [{"path": "/tmp/i.fits", "band": "r", "wcs": None, "shape": None}]

        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, 12, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER, images=images, bands=["r"], catalog_names=["sdss"]
        )
        assert out["winners"]["r"] == "sdss"
        assert set(out["report"]["mode"]) == {"field"}
        assert out["report"]["n_usable"].iloc[0] == 12

    def test_download_failure_recorded_not_fatal(
        self, tmp_path, monkeypatch, image_pair
    ):
        """A failing backend lands in 'skipped'; the rest still compete."""

        def fake_download(self, target_coords, catalogName=None, **kw):
            if catalogName == "sdss":
                raise RuntimeError("sdss exploded")
            return _raw_catalog(catalogName, 10, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER,
            images=image_pair,
            bands=["r"],
            catalog_names=["pan_starrs", "sdss"],
        )
        assert "sdss" in out["skipped"]
        assert out["winners"]["r"] == "pan_starrs"

    def test_all_failed_returns_empty_map(self, tmp_path, monkeypatch):
        def fake_download(self, target_coords, catalogName=None, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER,
            images=[],
            bands=["r"],
            catalog_names=["sdss"],
            write_report=False,
        )
        assert out["use_catalog"] == {}
        assert out["winners"] == {}

    def test_report_written(self, tmp_path, monkeypatch, image_pair):
        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, 9, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        cat.find_optimized_catalog(
            CENTER,
            images=image_pair,
            bands=["r"],
            catalog_names=["sdss"],
        )
        report_path = (
            tmp_path / "catalog_queries" / "T1_optimized_catalog_coverage.csv"
        )
        assert report_path.exists()
        df = pd.read_csv(report_path)
        assert {"catalog", "band", "image", "n_usable", "mode"} <= set(
            df.columns
        )

    def test_coverage_plot_written(self, tmp_path, monkeypatch, image_pair):
        """One PNG with per-band panels; plot_data carries footprints+sources."""
        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, 9, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER,
            images=image_pair,
            bands=["r", "J"],
            catalog_names=["pan_starrs", "sdss", "2mass"],
        )
        png = (
            tmp_path / "catalog_queries" / "T1_optimized_catalog_coverage.png"
        )
        assert png.exists() and png.stat().st_size > 1000
        src = out["plot_data"]["sources"]
        assert ("sdss", "r") in src and ("2mass", "J") in src
        assert len(out["plot_data"]["footprints"]) == 2
        assert out["plot_path"] == str(png)

    def test_plot_sources_rebuilt_from_cache(
        self, tmp_path, monkeypatch, image_pair
    ):
        """_optimizer_plot_sources reproduces the usable positions the
        coverage map needs on cached-selection runs, offline."""
        calls = []

        def fake_download(self, target_coords, catalogName=None, **kw):
            calls.append(catalogName)
            return _raw_catalog(catalogName, 9, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat._optimizer_plot_sources(
            ["pan_starrs", "sdss", "2mass"],
            band_set=["r"],
            image_infos=image_pair,
            target_coords=CENTER,
            radius=10,
        )
        assert ("pan_starrs", "r") in out
        assert ("sdss", "r") in out
        assert len(out[("pan_starrs", "r")]) > 0
        assert {"RA", "DEC"} <= set(out[("pan_starrs", "r")].columns)
        # 2mass has no r-band -> absent, not an error.
        assert ("2mass", "r") not in out

    def test_input_imagefilter_restored(self, tmp_path, monkeypatch, image_pair):
        """clean() needs imageFilter swapped per band; original is restored."""
        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, 7, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        cat.find_optimized_catalog(
            CENTER,
            images=image_pair,
            bands=["r", "J"],
            catalog_names=["sdss", "2mass"],
            write_report=False,
        )
        assert cat.input_yaml["imageFilter"] == "r"


class TestCoveragePlotScoreRow:
    """The coverage figure embeds a scoreboard row when a report is
    given, so one PNG carries footprints, source positions, counts, and
    coverage for every catalog."""

    def _report(self):
        rows = []
        for cat, n, cov in [
            ("pan_starrs", 40, 1.0),
            ("sdss", 12, 0.6),
            ("apass", 4, 1.0),
        ]:
            for img in ("a.fits", "b.fits"):
                rows.append(
                    {
                        "catalog": cat,
                        "band": "g",
                        "image": img,
                        "n_usable": n,
                        "coverage": cov,
                        "mode": "onchip",
                    }
                )
        return pd.DataFrame(rows)

    def _plot_data(self):
        w = _tan_wcs()
        ny, nx = 200, 200
        fp_ra, fp_dec = w.all_pix2world(
            [0.0, nx - 1.0, nx - 1.0, 0.0],
            [0.0, 0.0, ny - 1.0, ny - 1.0],
            0,
        )
        ra, dec = _field_ras_decs(9)
        return {
            "footprints": [
                {"band": "g", "ra": fp_ra, "dec": fp_dec, "name": "a.fits"}
            ],
            "sources": {
                ("pan_starrs", "g"): pd.DataFrame({"RA": ra, "DEC": dec})
            },
            "winners": {"g": "pan_starrs"},
            "band_set": ["g"],
            "evaluated": ["pan_starrs", "sdss", "apass"],
        }

    def test_combined_figure_written(self, tmp_path):
        from catalog import plot_optimized_catalog_coverage

        out = plot_optimized_catalog_coverage(
            self._plot_data(),
            target_coords=CENTER,
            report=self._report(),
            skipped={"gaia": "excluded from bulk scan"},
            target_name="T1",
            wdir=str(tmp_path),
        )
        assert out is not None and os.path.exists(out)
        assert out.endswith("T1_optimized_catalog_coverage.png")
        # Score row roughly doubles the figure height vs the map-only PNG.
        assert os.path.getsize(out) > 1000

    def test_missing_coverage_column_tolerated(self, tmp_path):
        from catalog import plot_optimized_catalog_coverage

        rep = self._report().drop(columns=["coverage"])
        out = plot_optimized_catalog_coverage(
            self._plot_data(),
            target_coords=CENTER,
            report=rep,
            wdir=str(tmp_path),
        )
        assert out is not None and os.path.exists(out)

    def test_map_only_without_report(self, tmp_path):
        from catalog import plot_optimized_catalog_coverage

        out = plot_optimized_catalog_coverage(
            self._plot_data(), target_coords=CENTER, wdir=str(tmp_path)
        )
        assert out is not None and os.path.exists(out)

    def test_all_sources_underlay_and_unknown_bands(self, tmp_path):
        """catalog_sources renders as a faint underlay; bands missing
        from the palette still draw (unique hashed fallback colour)."""
        from catalog import plot_optimized_catalog_coverage

        data = self._plot_data()
        data["catalog_sources"] = {
            "pan_starrs": pd.DataFrame(
                {
                    "RA": _field_ras_decs(20)[0],
                    "DEC": _field_ras_decs(20)[1],
                }
            )
        }
        # An unlisted filter name - palette lookup fails, hashed fallback
        # colours the panel elements instead of dropping to shared gray.
        data2 = dict(data)
        data2["band_set"] = ["j", "XX9"]
        data2["footprints"] = [
            dict(fp, band="j") for fp in data["footprints"]
        ] + [dict(fp, band="XX9") for fp in data["footprints"]]
        out = plot_optimized_catalog_coverage(
            data2, target_coords=CENTER, wdir=str(tmp_path)
        )
        assert out is not None and os.path.exists(out)

    def test_raw_only_catalog_gets_legend_entry(self, tmp_path, monkeypatch):
        """A catalog that returned detections but no usable sources on a
        band still appears in the legend so its faint underlay is
        attributable."""
        import matplotlib.figure as mfig

        from catalog import plot_optimized_catalog_coverage

        data = self._plot_data()
        data["catalog_sources"] = {
            "pan_starrs": pd.DataFrame(
                {
                    "RA": _field_ras_decs(20)[0],
                    "DEC": _field_ras_decs(20)[1],
                }
            ),
            "apass": pd.DataFrame(
                {
                    "RA": _field_ras_decs(12)[0],
                    "DEC": _field_ras_decs(12)[1],
                }
            ),
        }
        captured = {}
        orig_savefig = mfig.Figure.savefig

        def spy(self, *a, **kw):
            captured["fig"] = self
            return orig_savefig(self, *a, **kw)

        monkeypatch.setattr(mfig.Figure, "savefig", spy)
        out = plot_optimized_catalog_coverage(
            data, target_coords=CENTER, wdir=str(tmp_path)
        )
        assert out is not None and os.path.exists(out)
        labels = [
            t.get_text()
            for ax in captured["fig"].axes
            if ax.get_legend() is not None
            for t in ax.get_legend().get_texts()
        ]
        assert any(t.startswith("apass raw") for t in labels)
        assert any(t.startswith("pan_starrs") for t in labels)

    def test_plot_sources_return_all(self, tmp_path, monkeypatch, image_pair):
        """return_all=True also yields the full per-catalog source list
        for the faint underlay on cached-run rebuilds."""
        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, 9, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out, all_src = cat._optimizer_plot_sources(
            ["pan_starrs", "sdss"],
            band_set=["r"],
            image_infos=image_pair,
            target_coords=CENTER,
            radius=10,
            return_all=True,
        )
        assert ("pan_starrs", "r") in out
        assert len(all_src["pan_starrs"]) == 9
        assert {"RA", "DEC"} <= set(all_src["pan_starrs"].columns)


class TestAutoGuardAndConfig:
    def test_require_catalog_selected_rejects_auto(self, tmp_path):
        cat = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="auto"):
            cat._require_catalog_selected("auto")

    @pytest.mark.skipif(Prepare is None, reason="prepare.py deps unavailable")
    def test_check_catalog_auto_returns_all_bands(self, tmp_path):
        prep = Prepare(
            {
                "catalog": {"use_catalog": "auto"},
                "imageFilter": "r",
                "wdir": str(tmp_path),
            }
        )
        bands = prep.check_catalog()
        assert set(bands) == set(SUPPORTED_PHOTOMETRIC_FILTERS)

    @pytest.mark.skipif(Prepare is None, reason="prepare.py deps unavailable")
    def test_file_filter_map_initialized(self, tmp_path):
        prep = Prepare({"catalog": {"use_catalog": "gaia"}, "wdir": str(tmp_path)})
        assert prep.file_filter_map == {}


class TestQueryRegions:
    """Per-band/per-catalog query regions: bounding-box-centred, bounded,
    and routed to download() as box (VizieR) or circumscribed cone."""

    def test_region_centres_on_footprint_bbox_not_target(self):
        from catalog import _footprint_region

        # r images on-target; J images offset +20' in RA.
        infos = [
            {"path": "r1.fits", "band": "r", "wcs": _tan_wcs(),
             "shape": (200, 200)},
            {"path": "r2.fits", "band": "r", "wcs": _tan_wcs(),
             "shape": (200, 200)},
            {"path": "j1.fits", "band": "J", "wcs": _offset_wcs(20.0),
             "shape": (200, 200)},
        ]
        reg_r = _footprint_region(infos, ["r"], CENTER)
        reg_j = _footprint_region(infos, ["J"], CENTER)
        # The r region stays centred near the target; the J region
        # follows its own (offset) footprints.
        assert abs(reg_r["ra"] - CENTER.ra.deg) < 0.02
        assert reg_j["offset_arcmin"] > 15.0
        # Both cones must cover their band's corner separations.
        assert reg_r["radius_arcmin"] >= 2.0
        assert reg_j["radius_arcmin"] >= 2.0
        # The union of both bands is wider than either alone.
        reg_all = _footprint_region(infos, None, CENTER)
        assert reg_all["radius_arcmin"] > reg_r["radius_arcmin"]

    def test_region_bounds(self):
        from catalog import _region_from_corners

        # Tiny field -> min bound.
        reg_small = _region_from_corners(
            np.array([150.0, 150.001]), np.array([30.0, 30.001]), CENTER
        )
        assert reg_small["radius_arcmin"] == pytest.approx(2.0)
        # Huge field -> max bound, box rescaled to stay inscribed.
        reg_big = _region_from_corners(
            np.array([140.0, 160.0]), np.array([0.0, 30.0]), None
        )
        assert reg_big["radius_arcmin"] == pytest.approx(60.0)
        # width_deg is angular (cos-dec projected); the box half-diagonal
        # must stay inside the clamped cone.
        hw = 0.5 * np.hypot(reg_big["width_deg"], reg_big["height_deg"]) * 60
        assert hw <= 60.0 * 1.001
        # No corners -> default cone, no box.
        reg_none = _region_from_corners([], [], CENTER)
        assert reg_none["radius_arcmin"] == pytest.approx(10.0)
        assert reg_none["box_deg"] is None

    def test_region_ra_wrap(self):
        from catalog import _footprint_region

        target = SkyCoord(0.02, -10.0, unit="deg", frame="icrs")
        w = _tan_wcs()
        w.wcs.crval = [0.02, -10.0]
        infos = [{"path": "w.fits", "band": "r", "wcs": w, "shape": (200, 200)}]
        reg = _footprint_region(infos, ["r"], target)
        ra_min, ra_max, dec_min, dec_max = reg["box_deg"]
        # Wrap across 0/360: ra_min > ra_max marks the wrapped box; the
        # span must not balloon to ~360 deg.
        span = (ra_max - ra_min) % 360.0
        assert span < 1.0
        assert abs(reg["ra"] - 0.02) < 0.05

    def test_scan_uses_per_catalog_regions(self, tmp_path, monkeypatch):
        """sdss serves r (on-target), 2mass serves J (offset pointing) -
        each must be queried over its own bands' footprints."""
        captured = {}

        def fake_download(self, target_coords, catalogName=None,
                          region=None, **kw):
            captured[catalogName] = region
            return _raw_catalog(catalogName, 9, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        images = [
            {"path": "r.fits", "band": "r", "wcs": _tan_wcs(),
             "shape": (200, 200)},
            {"path": "j.fits", "band": "J", "wcs": _offset_wcs(20.0),
             "shape": (200, 200)},
        ]
        out = cat.find_optimized_catalog(
            CENTER,
            images=images,
            bands=["r", "J"],
            catalog_names=["sdss", "2mass"],
        )
        reg_sdss = captured["sdss"]
        reg_2m = captured["2mass"]
        assert reg_sdss["offset_arcmin"] < 2.0
        assert reg_2m["offset_arcmin"] > 15.0
        assert out["regions"]["sdss"] is reg_sdss
        assert out["band_regions"]["J"]["offset_arcmin"] > 15.0
        # The report carries the queried radius per row.
        assert "r_query_arcmin" in out["report"].columns


class TestDownloadRegionRouting:
    """download(region=...) routes centre/extent per backend shape."""

    def _vizier_table(self, n=6):
        from astropy.table import Table

        ra, dec = _field_ras_decs(n)
        t = Table()
        t["RAJ2000"] = ra
        t["DEJ2000"] = dec
        t["Jmag"] = np.full(n, 15.0)
        t["e_Jmag"] = np.full(n, 0.01)
        t["Hmag"] = np.full(n, 15.0)
        t["e_Hmag"] = np.full(n, 0.01)
        t["Kmag"] = np.full(n, 15.0)
        t["e_Kmag"] = np.full(n, 0.01)
        t["Qflg"] = ["AAA"] * n
        return t

    def test_vizier_box_query_used_when_region_given(
        self, tmp_path, monkeypatch
    ):
        from catalog import _region_from_corners

        calls = {}

        def fake_query(coords, **kwargs):
            calls["coords"] = coords
            calls["kwargs"] = kwargs
            return [self._vizier_table()]

        monkeypatch.setattr("catalog.Vizier.query_region", fake_query)
        cat = _make_catalog(
            tmp_path, catalog={"use_catalog": "2mass", "max_distance": 10}
        )
        region = _region_from_corners(
            np.array([150.0, 150.3]), np.array([30.0, 30.2]), CENTER
        )
        out = cat.download(
            CENTER, catalogName="2mass", region=region, target_name="T1"
        )
        kw = calls["kwargs"]
        assert "width" in kw and "height" in kw
        assert "radius" not in kw
        # Query was centred on the region centre, not the target.
        assert abs(calls["coords"].ra.deg - region["ra"]) < 1e-6
        assert len(out) == 6

    def test_cone_backend_uses_circumscribed_region(
        self, tmp_path, monkeypatch
    ):
        """pan_starrs (cone-only) gets the region centre + circumscribed
        radius, not the box."""
        from catalog import _region_from_corners

        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured.update(params)
            return _FakeResponse(_ps_payload(8))

        monkeypatch.setattr("catalog.requests.get", fake_get)
        cat = _ps_catalog(tmp_path)
        region = _region_from_corners(
            np.array([151.0, 151.4]), np.array([31.0, 31.2]), CENTER
        )
        cat.download(
            CENTER, catalogName="pan_starrs", region=region,
            target_name="T1",
        )
        assert captured["ra"] == pytest.approx(region["ra"])
        assert captured["dec"] == pytest.approx(region["dec"])
        assert captured["radius"] == pytest.approx(
            region["radius_arcmin"] / 60.0
        )

    def test_cone_cache_superset_serves_box_request(
        self, tmp_path, monkeypatch
    ):
        """A box query reuses the wider cone cache at the same centre -
        the cone result is a superset, so no re-download is needed."""
        from catalog import _region_from_corners

        calls = {"n": 0}

        def fake_query(coords, **kwargs):
            calls["n"] += 1
            return [self._vizier_table()]

        monkeypatch.setattr("catalog.Vizier.query_region", fake_query)
        cat = _make_catalog(
            tmp_path, catalog={"use_catalog": "2mass", "max_distance": 10}
        )
        region = _region_from_corners(
            np.array([150.0, 150.3]), np.array([30.0, 30.2]), CENTER
        )
        # First: a cone request writes the cone-named cache file.
        cat.download(
            CENTER, catalogName="2mass", region=None, target_name="T1",
            radius=region["radius_arcmin"],
        )
        assert calls["n"] == 1
        # Cone was centred on the target, so rename it to the region's
        # key: the box request must find it as the cone-superset reuse.
        import glob

        cone_csv = glob.glob(
            str(tmp_path / "catalog_queries" / "2mass" / "**" / "*.csv"),
            recursive=True,
        )
        assert cone_csv, "expected a cached cone CSV"
        new_csv = cone_csv[0].replace(
            f"target_ra_{CENTER.ra.deg:.6f}_dec_{CENTER.dec.deg:.6f}",
            f"target_ra_{region['ra']:.6f}_dec_{region['dec']:.6f}",
        )
        os.rename(cone_csv[0], new_csv)
        out = cat.download(
            CENTER, catalogName="2mass", region=region, target_name="T1"
        )
        assert calls["n"] == 1  # served from the cone cache
        assert len(out) == 6


class _FakeResponse:
    """Minimal requests.Response stand-in for the MAST Pan-STARRS API."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _ps_payload(n):
    cols = [
        "raMean",
        "decMean",
        "gMeanPSFMag",
        "gMeanPSFMagErr",
        "rMeanPSFMag",
        "rMeanPSFMagErr",
        "nStackDetections",
        "nDetections",
        "ng",
        "nr",
    ]
    ra, dec = _field_ras_decs(n)
    rows = [
        [
            ra[i],
            dec[i],
            15.0,
            0.01,
            15.0,
            0.01,
            5,
            5,
            5,
            5,
        ]
        for i in range(n)
    ]
    return {"info": [{"name": c} for c in cols], "data": rows}


def _ps_catalog(tmp_path, retries=3):
    return _make_catalog(
        tmp_path,
        catalog={
            "use_catalog": "pan_starrs",
            "max_distance": 10,
            "pan_starrs_max_retries": retries,
            "pan_starrs_retry_base_delay_sec": 0,
        },
    )


class TestPanStarrsRetry:
    def test_retry_recovers_transient_timeout(self, tmp_path, monkeypatch):
        """Two timeouts then a success: download still returns the catalog."""
        import requests as _req

        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _req.ReadTimeout("simulated MAST timeout")
            return _FakeResponse(_ps_payload(10))

        monkeypatch.setattr("catalog.requests.get", fake_get)
        cat = _ps_catalog(tmp_path)
        out = cat.download(
            CENTER, catalogName="pan_starrs", radius=10, target_name="T1"
        )
        assert calls["n"] == 3
        assert len(out) == 10

    def test_exhausted_retries_raise_query_failed(self, tmp_path, monkeypatch):
        """Exhausted retries raise a distinct error - not '0 sources'."""
        import requests as _req

        def fake_get(url, params=None, timeout=None):
            raise _req.ReadTimeout("simulated MAST timeout")

        monkeypatch.setattr("catalog.requests.get", fake_get)
        cat = _ps_catalog(tmp_path, retries=2)
        with pytest.raises(RuntimeError) as excinfo:
            cat.download(
                CENTER, catalogName="pan_starrs", radius=10, target_name="T1"
            )
        msg = str(excinfo.value)
        assert "query failed" in msg
        assert "0 sources" not in msg

    def test_genuine_empty_still_reports_zero_sources(
        self, tmp_path, monkeypatch
    ):
        """A successful but empty reply keeps the '0 sources' wording so the
        driver treats it as real no-coverage, not a transient failure."""

        def fake_get(url, params=None, timeout=None):
            return _FakeResponse({"info": [{"name": "raMean"}], "data": []})

        monkeypatch.setattr("catalog.requests.get", fake_get)
        cat = _ps_catalog(tmp_path)
        with pytest.raises(RuntimeError, match="0 sources"):
            cat.download(
                CENTER, catalogName="pan_starrs", radius=10, target_name="T1"
            )


class TestUsableMagErrGate:
    """"Usable" = well-measured: a finite in-window magnitude AND a
    finite band error under catalog.optimizer_mag_err_max."""

    def _df(self):
        return pd.DataFrame(
            {
                "g": [15.0, 15.0, 15.0, 15.0, 30.0, np.nan],
                "g_err": [0.02, 0.30, np.nan, np.inf, 0.02, 0.02],
            }
        )

    def test_mask_requires_finite_capped_error(self):
        from catalog import _usable_mag_mask

        mask = _usable_mag_mask(
            self._df(), "g", 11.0, 22.0, mag_err_max=0.2
        )
        assert mask.tolist() == [True, False, False, False, False, False]

    def test_no_cap_still_requires_finite_error(self):
        from catalog import _usable_mag_mask

        mask = _usable_mag_mask(
            self._df(), "g", 11.0, 22.0, mag_err_max=None
        )
        # err <= cap lifted but NaN/inf errors still fail.
        assert mask.tolist() == [True, True, False, False, False, False]

    def test_missing_err_column_tolerated(self):
        from catalog import _usable_mag_mask

        df = pd.DataFrame({"g": [15.0, 30.0]})
        mask = _usable_mag_mask(df, "g", 11.0, 22.0, mag_err_max=0.2)
        assert mask.tolist() == [True, False]

    def test_missing_band_returns_none(self):
        from catalog import _usable_mag_mask

        assert (
            _usable_mag_mask(self._df(), "r", 11.0, 22.0, 0.2) is None
        )
        assert _usable_mag_mask(None, "g", 11.0, 22.0, 0.2) is None

    def test_optimizer_scoring_uses_err_gate(
        self, tmp_path, monkeypatch, image_pair
    ):
        """A catalog full of poorly-measured mags must not win the
        usable count - the plot and the scoreboard share the mask."""
        n = 40

        def fake_download(self, target_coords, catalogName=None, **kw):
            df = _raw_catalog(catalogName, n, 15.0)
            if catalogName == "sdss":
                df["e_rmag"] = np.full(n, 0.5)  # all badly measured
            return df

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        out = cat.find_optimized_catalog(
            CENTER,
            images=image_pair,
            bands=["r"],
            catalog_names=["pan_starrs", "sdss"],
        )
        row = out["report"][
            (out["report"].catalog == "sdss") & (out["report"].band == "r")
        ]
        assert row["n_usable"].iloc[0] == 0
        assert out["winners"]["r"] == "pan_starrs"

    def test_err_gate_configurable(self, tmp_path, monkeypatch):
        """catalog.optimizer_mag_err_max <= 0 lifts the cap."""
        def fake_download(self, target_coords, catalogName=None, **kw):
            df = _raw_catalog(catalogName, 10, 15.0)
            df["e_rmag"] = np.full(10, 0.5)
            return df

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(
            tmp_path,
            catalog={
                "use_catalog": "auto",
                "max_distance": 10,
                "optimizer_mag_err_max": 0,
            },
        )
        w = _tan_wcs()
        images = [
            {"path": "/tmp/i.fits", "band": "r", "wcs": w,
             "shape": (200, 200)}
        ]
        out = cat.find_optimized_catalog(
            CENTER,
            images=images,
            bands=["r"],
            catalog_names=["sdss"],
            write_report=False,
            write_plot=False,
        )
        assert out["report"]["n_usable"].iloc[0] == 10


class TestOptimizerPlotSourcesReport:
    """return_report rebuilds the optimizer report columns so cached
    runs can draw the scoreboard without the coverage CSV."""

    def test_report_columns_and_counts(
        self, tmp_path, monkeypatch, image_pair
    ):
        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, 9, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        regions = {
            "pan_starrs": {
                "ra": CENTER.ra.deg,
                "dec": CENTER.dec.deg,
                "radius_arcmin": 10.0,
            }
        }
        out, all_src, rep = cat._optimizer_plot_sources(
            ["pan_starrs", "sdss"],
            band_set=["r"],
            image_infos=image_pair,
            target_coords=CENTER,
            radius=10,
            regions=regions,
            return_report=True,
        )
        assert ("pan_starrs", "r") in out
        assert {
            "catalog",
            "band",
            "image",
            "n_usable",
            "coverage",
            "r_query_arcmin",
            "mode",
        } <= set(rep.columns)
        ps_rows = rep[rep.catalog == "pan_starrs"]
        # image_pair is one r image + one J image; pan_starrs serves
        # only r, so exactly one row with all 9 sources usable.
        assert len(ps_rows) == 1
        assert (ps_rows["n_usable"] == 9).all()
        assert set(ps_rows["mode"]) == {"onchip"}
        # Coverage was computed against the region entry.
        assert np.isfinite(ps_rows["coverage"]).all()
        # sdss has no region entry -> coverage stays NaN but rows exist.
        sd_rows = rep[rep.catalog == "sdss"]
        assert len(sd_rows) == 1
        assert np.isnan(sd_rows["coverage"]).all()

    def test_scoreboard_draws_on_rebuilt_report(
        self, tmp_path, monkeypatch, image_pair
    ):
        """The rebuilt report feeds the same scoreboard path as the
        persisted CSV - the figure height includes the score row."""
        import matplotlib.figure as mfig

        from catalog import plot_optimized_catalog_coverage

        def fake_download(self, target_coords, catalogName=None, **kw):
            return _raw_catalog(catalogName, 9, 15.0)

        monkeypatch.setattr(Catalog, "download", fake_download)
        cat = _make_catalog(tmp_path)
        sources, all_src, rep = cat._optimizer_plot_sources(
            ["pan_starrs", "sdss"],
            band_set=["r"],
            image_infos=image_pair,
            target_coords=CENTER,
            radius=10,
            return_report=True,
        )
        captured = {}
        orig_savefig = mfig.Figure.savefig

        def spy(self, *a, **kw):
            captured["fig"] = self
            return orig_savefig(self, *a, **kw)

        monkeypatch.setattr(mfig.Figure, "savefig", spy)
        out = plot_optimized_catalog_coverage(
            {
                "footprints": [],
                "sources": sources,
                "winners": {"r": "pan_starrs"},
                "band_set": ["r"],
                "catalog_sources": all_src,
            },
            target_coords=CENTER,
            report=rep,
            wdir=str(tmp_path),
        )
        assert out is not None
        # 2 axes: map panel + scoreboard panel.
        assert len(captured["fig"].axes) == 2


class TestUsableMarkersOpen:
    """Usable sources draw as open markers so coincident calibrators
    from different catalogs stay visible instead of hiding under
    whichever catalog was drawn last."""

    def test_usable_scatter_has_no_facecolors(self, tmp_path):
        import matplotlib.figure as mfig

        from catalog import plot_optimized_catalog_coverage

        ra, dec = _field_ras_decs(9)
        data = {
            "footprints": [],
            "sources": {
                ("pan_starrs", "r"): pd.DataFrame({"RA": ra, "DEC": dec}),
                ("sdss", "r"): pd.DataFrame({"RA": ra, "DEC": dec}),
            },
            "winners": {"r": "pan_starrs"},
            "band_set": ["r"],
        }
        captured = {}
        orig_savefig = mfig.Figure.savefig

        def spy(self, *a, **kw):
            captured["fig"] = self
            return orig_savefig(self, *a, **kw)

        monkeypatch_fixture = pytest.MonkeyPatch()
        monkeypatch_fixture.setattr(mfig.Figure, "savefig", spy)
        try:
            out = plot_optimized_catalog_coverage(
                data, target_coords=CENTER, wdir=str(tmp_path)
            )
        finally:
            monkeypatch_fixture.undo()
        assert out is not None
        colls = [
            c
            for ax in captured["fig"].axes
            for c in ax.collections
            if c.get_label() in ("pan_starrs (9)", "sdss (9)")
        ]
        assert len(colls) == 2
        for c in colls:
            # facecolors='none' -> empty facecolor array.
            assert c.get_facecolor().shape[0] == 0
            assert c.get_edgecolor().shape[0] == 1


class TestRefcatNsourcesScaling:
    """The RefCat2 row cap scales with the queried area and rides in the
    cache filename so an old 500-row file cannot answer a larger ask."""

    def _cat(self, tmp_path, nsources=None):
        cat_cfg = {
            "use_catalog": "refcat",
            "max_distance": 30,
            "MASTcasjobs_wsid": "dummy",
            "MASTcasjobs_pwd": "dummy",
        }
        if nsources is not None:
            cat_cfg["refcat_nsources"] = nsources
        return _make_catalog(tmp_path, catalog=cat_cfg)

    def _fake_field(self, captured):
        def fake(ra, dec, credentials, nsources=500, sr=0.17):
            captured["nsources"] = nsources
            ra_v, dec_v = _field_ras_decs(6)
            return pd.DataFrame({"RA": ra_v, "Dec": dec_v, "g": 15.0})

        return fake

    def test_area_scaled_cap(self, tmp_path, monkeypatch):
        captured = {}
        cat = self._cat(tmp_path)
        monkeypatch.setattr(
            cat, "fetch_refcat2_field", self._fake_field(captured)
        )
        cat.download(
            CENTER, catalogName="refcat", radius=13.5, target_name="T1"
        )
        # ~3/arcmin^2 over pi*13.5^2 ~ 1720, above the old flat 500.
        assert captured["nsources"] > 1500
        # Cache filename carries the cap so a stale 500-row file from
        # older runs can never satisfy this query.
        import glob

        csvs = glob.glob(
            str(tmp_path / "catalog_queries" / "refcat" / "**" / "*.csv"),
            recursive=True,
        )
        assert len(csvs) == 1
        assert f"_n{captured['nsources']}_" in csvs[0]

    def test_default_cone_keeps_floor(self, tmp_path, monkeypatch):
        captured = {}
        cat = self._cat(tmp_path)
        monkeypatch.setattr(
            cat, "fetch_refcat2_field", self._fake_field(captured)
        )
        cat.download(
            CENTER, catalogName="refcat", radius=10.0, target_name="T1"
        )
        assert captured["nsources"] >= 500

    def test_explicit_override(self, tmp_path, monkeypatch):
        captured = {}
        cat = self._cat(tmp_path, nsources=2500)
        monkeypatch.setattr(
            cat, "fetch_refcat2_field", self._fake_field(captured)
        )
        cat.download(
            CENTER, catalogName="refcat", radius=10.0, target_name="T1"
        )
        assert captured["nsources"] == 2500


class TestCoverageRecentre:
    """Coverage-optimised query centres for partial-coverage cone backends."""

    @staticmethod
    def _ps_frame(ra, dec, mag=17.0):
        n = len(ra)
        return pd.DataFrame(
            {
                "raMean": np.asarray(ra, dtype=float),
                "decMean": np.asarray(dec, dtype=float),
                "gMeanPSFMag": np.full(n, mag),
                "gMeanPSFMagErr": np.full(n, 0.01),
                "rMeanPSFMag": np.full(n, mag),
                "rMeanPSFMagErr": np.full(n, 0.01),
                "iMeanPSFMag": np.full(n, mag),
                "iMeanPSFMagErr": np.full(n, 0.01),
            }
        )

    @staticmethod
    def _west_half_sources(c_ra, c_dec, hw_x_deg, hw_y_deg, n, seed=3):
        """Uniform positions in the low-RA (western) half of a box."""
        rng = np.random.default_rng(seed)
        x = rng.uniform(-hw_x_deg, 0.0, n)
        y = rng.uniform(-hw_y_deg, hw_y_deg, n)
        ra = c_ra + x / np.cos(np.radians(c_dec))
        return ra, c_dec + y

    def test_optimal_center_shifts_to_covered_side(self):
        reg = {
            "ra": CENTER.ra.deg,
            "dec": CENTER.dec.deg,
            "radius_arcmin": 30.0,
            "width_deg": 0.6,
            "height_deg": 0.6,
        }
        ra, dec = self._west_half_sources(
            CENTER.ra.deg, CENTER.dec.deg, 0.3, 0.3, 400
        )
        new_c = _coverage_optimal_center(ra, dec, reg)
        assert new_c is not None
        cosd = np.cos(np.radians(CENTER.dec.deg))
        got_x = (new_c[0] - CENTER.ra.deg) * cosd
        # Centroid of the western half of the box.
        assert got_x == pytest.approx(-0.15, abs=0.03)
        assert new_c[1] == pytest.approx(CENTER.dec.deg, abs=0.03)

    def test_optimal_center_none_when_balanced(self):
        reg = {
            "ra": CENTER.ra.deg,
            "dec": CENTER.dec.deg,
            "radius_arcmin": 30.0,
            "width_deg": 0.6,
            "height_deg": 0.6,
        }
        rng = np.random.default_rng(4)
        x = rng.uniform(-0.3, 0.3, 2000)
        y = rng.uniform(-0.3, 0.3, 2000)
        ra = CENTER.ra.deg + x / np.cos(np.radians(CENTER.dec.deg))
        dec = CENTER.dec.deg + y
        assert _coverage_optimal_center(ra, dec, reg) is None

    def test_optimal_center_none_when_sparse(self):
        reg = {
            "ra": CENTER.ra.deg,
            "dec": CENTER.dec.deg,
            "radius_arcmin": 30.0,
            "width_deg": 0.6,
            "height_deg": 0.6,
        }
        ra, dec = self._west_half_sources(
            CENTER.ra.deg, CENTER.dec.deg, 0.3, 0.3, 5
        )
        assert _coverage_optimal_center(ra, dec, reg) is None

    def test_in_required_region_cone_fallback(self):
        reg = {
            "ra": CENTER.ra.deg,
            "dec": CENTER.dec.deg,
            "radius_arcmin": 6.0,
        }
        ra = np.array([CENTER.ra.deg, CENTER.ra.deg + 1.0])
        dec = np.array([CENTER.dec.deg, CENTER.dec.deg])
        mask = _in_required_region(ra, dec, reg)
        assert mask.tolist() == [True, False]

    def test_region_clamped_flag(self):
        reg = _region_from_corners(
            np.array([140.0, 160.0]),
            np.array([25.0, 35.0]),
            CENTER,
            max_arcmin=60.0,
        )
        assert reg["clamped"] is True
        ra, dec = _field_ras_decs(20)
        small = _region_from_corners(ra, dec, CENTER, max_arcmin=60.0)
        assert small["clamped"] is False

    def _optimizer_run(self, cat, monkeypatch, fake_download):
        monkeypatch.setattr(Catalog, "download", fake_download)
        w = _tan_wcs()
        return cat.find_optimized_catalog(
            CENTER,
            images=[
                {
                    "path": "/tmp/i.fits",
                    "band": "r",
                    "wcs": w,
                    "shape": (200, 200),
                }
            ],
            bands=["r"],
            catalog_names=["pan_starrs"],
            target_name="T1",
            write_report=False,
            write_plot=False,
        )

    def _tiny_region_catalog(self, tmp_path, extra=None):
        cfg = {
            "use_catalog": "auto",
            "max_distance": 10,
            # Tiny region cap: the footprint cone clamps, which is the
            # recenter gate for cone backends that are not row-capped.
            "region_max_arcmin": 1.0,
        }
        cfg.update(extra or {})
        return _make_catalog(tmp_path, catalog=cfg)

    def test_optimizer_recentres_clamped_cone(self, tmp_path, monkeypatch):
        calls = []

        def fake_download(self, target_coords, catalogName=None, **kw):
            reg = kw.get("region") or {}
            calls.append(dict(reg))
            c_ra = float(reg.get("ra", CENTER.ra.deg))
            c_dec = float(reg.get("dec", CENTER.dec.deg))
            hw_x = 0.5 * float(reg.get("width_deg") or 0.02)
            hw_y = 0.5 * float(reg.get("height_deg") or 0.02)
            rng = np.random.default_rng(11)
            if reg.get("recentered"):
                x = rng.uniform(-hw_x, hw_x, 200)
            else:
                x = rng.uniform(-hw_x, 0.0, 200)
            y = rng.uniform(-hw_y, hw_y, len(x))
            ra = c_ra + x / np.cos(np.radians(c_dec))
            return TestCoverageRecentre._ps_frame(ra, c_dec + y)

        cat = self._tiny_region_catalog(tmp_path)
        out = self._optimizer_run(cat, monkeypatch, fake_download)
        reg = out["regions"]["pan_starrs"]
        assert reg.get("recentered") is True
        assert len(calls) == 2
        # The adopted centre moved toward the covered (low-RA) side.
        cosd = np.cos(np.radians(CENTER.dec.deg))
        assert (reg["ra"] - calls[0]["ra"]) * cosd < 0

    def test_recenter_rejected_when_no_gain(self, tmp_path, monkeypatch):
        def fake_download(self, target_coords, catalogName=None, **kw):
            reg = kw.get("region") or {}
            c_ra = float(reg.get("ra", CENTER.ra.deg))
            c_dec = float(reg.get("dec", CENTER.dec.deg))
            hw_x = 0.5 * float(reg.get("width_deg") or 0.02)
            hw_y = 0.5 * float(reg.get("height_deg") or 0.02)
            ra, dec = TestCoverageRecentre._west_half_sources(
                c_ra, c_dec, hw_x, hw_y, 200, seed=5
            )
            return TestCoverageRecentre._ps_frame(ra, dec)

        cat = self._tiny_region_catalog(tmp_path)
        out = self._optimizer_run(cat, monkeypatch, fake_download)
        assert "recentered" not in out["regions"]["pan_starrs"]

    def test_recenter_disabled_by_config(self, tmp_path, monkeypatch):
        calls = []

        def fake_download(self, target_coords, catalogName=None, **kw):
            reg = kw.get("region") or {}
            calls.append(dict(reg))
            c_ra = float(reg.get("ra", CENTER.ra.deg))
            c_dec = float(reg.get("dec", CENTER.dec.deg))
            hw_x = 0.5 * float(reg.get("width_deg") or 0.02)
            hw_y = 0.5 * float(reg.get("height_deg") or 0.02)
            ra, dec = TestCoverageRecentre._west_half_sources(
                c_ra, c_dec, hw_x, hw_y, 200
            )
            return TestCoverageRecentre._ps_frame(ra, dec)

        cat = self._tiny_region_catalog(
            tmp_path, {"coverage_recenter_min": 0}
        )
        self._optimizer_run(cat, monkeypatch, fake_download)
        assert len(calls) == 1
