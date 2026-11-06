"""Regression tests for catalog calibrator vetting.

Two failure modes seen on real Pan-STARRS data (SN2025abkv, ZTF i-band
2025-08-07): spurious faint catalog entries sitting on top of bright
stars survive the download-time quality cuts, inherit the bright star's
measured flux, and form a second zeropoint locus ~5 mag above the true
one; and the linearity RANSAC seeds from the highest-S/N subset, which
those artifacts dominate, so it anchored on the false locus and rejected
the majority of real calibrators as outliers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog import Catalog, _drop_fainter_blended_neighbors


def _ps1_frame(rows):
    """Minimal Pan-STARRS-shaped frame for Catalog.clean()."""
    return pd.DataFrame(
        {
            "raMean": [r[0] for r in rows],
            "decMean": [r[1] for r in rows],
            "iMeanPSFMag": [r[2] for r in rows],
            "iMeanPSFMagErr": [r[3] for r in rows],
        }
    )


class TestDropFainterBlendedNeighbors:
    def _df(self, ras, decs, mags):
        return pd.DataFrame({"RA": ras, "DEC": decs, "i": mags})

    def test_phantom_entry_dropped(self):
        # Faint entry 2 arcsec from a 12.7-mag star - the failure mode.
        df = self._df(
            [150.0, 150.0 + 2.0 / 3600.0 / np.cos(np.radians(30.0))],
            [30.0, 30.0],
            [12.7, 17.1],
        )
        out = _drop_fainter_blended_neighbors(df, "i", 6.0, 3.0)
        assert len(out) == 1
        assert np.isclose(out["i"].iloc[0], 12.7)

    def test_comparable_pair_both_kept(self):
        # Equal-brightness close doubles are not phantom artefacts.
        df = self._df(
            [150.0, 150.0 + 2.0 / 3600.0 / np.cos(np.radians(30.0))],
            [30.0, 30.0],
            [16.0, 17.5],
        )
        out = _drop_fainter_blended_neighbors(df, "i", 6.0, 3.0)
        assert len(out) == 2

    def test_distant_faint_kept(self):
        df = self._df(
            [150.0, 150.0 + 20.0 / 3600.0 / np.cos(np.radians(30.0))],
            [30.0, 30.0],
            [12.7, 17.1],
        )
        out = _drop_fainter_blended_neighbors(df, "i", 6.0, 3.0)
        assert len(out) == 2

    def test_dmag_boundary(self):
        ras = [
            150.0,
            150.0 + 2.0 / 3600.0 / np.cos(np.radians(30.0)),
            150.0 + 4.0 / 3600.0 / np.cos(np.radians(30.0)),
        ]
        df = self._df(ras, [30.0] * 3, [13.0, 16.0, 15.9])
        out = _drop_fainter_blended_neighbors(df, "i", 6.0, 3.0)
        mags = sorted(out["i"].tolist())
        # 16.0 is exactly 3 mag fainter -> dropped; 15.9 survives.
        assert mags == [13.0, 15.9]

    def test_nan_mag_neighbour_kept(self):
        df = self._df(
            [150.0, 150.0 + 2.0 / 3600.0 / np.cos(np.radians(30.0))],
            [30.0, 30.0],
            [12.7, np.nan],
        )
        out = _drop_fainter_blended_neighbors(df, "i", 6.0, 3.0)
        assert len(out) == 2

    def test_chain_only_drops_faint_side(self):
        # Two phantoms on one bright host: both drop, host stays.
        cosd = np.cos(np.radians(30.0))
        df = self._df(
            [150.0, 150.0 + 1.0 / 3600.0 / cosd, 150.0 - 1.0 / 3600.0 / cosd],
            [30.0, 30.0, 30.0],
            [12.7, 17.1, 19.1],
        )
        out = _drop_fainter_blended_neighbors(df, "i", 6.0, 3.0)
        assert len(out) == 1
        assert np.isclose(out["i"].iloc[0], 12.7)


class TestCleanBlendCut:
    def _cat(self, tmp_path, **cat_cfg):
        cfg = {"use_catalog": "pan_starrs"}
        cfg.update(cat_cfg)
        return Catalog(
            input_yaml={
                "imageFilter": "i",
                "catalog": cfg,
            }
        )

    def test_clean_removes_phantom(self, tmp_path):
        cosd = np.cos(np.radians(30.0))
        rows = [
            (150.0, 30.0, 12.7, 0.01),  # real bright star
            (150.0 + 2.0 / 3600.0 / cosd, 30.0, 17.1, 0.02),  # phantom
            (150.1, 30.1, 18.5, 0.02),  # isolated faint star
        ]
        cat = self._cat(tmp_path)
        out = cat.clean(_ps1_frame(rows), catalogName="pan_starrs")
        assert len(out) == 2
        assert 17.1 not in out["i"].tolist()

    def test_clean_disabled_keeps_phantom(self, tmp_path):
        cosd = np.cos(np.radians(30.0))
        rows = [
            (150.0, 30.0, 12.7, 0.01),
            (150.0 + 2.0 / 3600.0 / cosd, 30.0, 17.1, 0.02),
        ]
        cat = self._cat(tmp_path, blend_neighbor_radius_arcsec=0.0)
        out = cat.clean(_ps1_frame(rows), catalogName="pan_starrs")
        assert len(out) == 2


def _locus_catalog(seed=7):
    """Catalog mimicking the 2025-08-07 failure: a majority of normal
    calibrators (ZP=24.0, moderate S/N) plus a minority of bright-flux
    artifacts (ZP~28.6, very high S/N) that dominate the >=100 S/N tier."""
    rng = np.random.default_rng(seed)
    n_good = 70
    i_good = rng.uniform(14.0, 20.0, n_good)
    flux_good = 10.0 ** (-0.4 * (i_good - 24.0)) * (
        1.0 + rng.normal(0.0, 0.02, n_good)
    )
    # S/N ~50: below the 100 seed gate on a noisy night.
    ferr_good = 0.02 * flux_good
    n_bad = 12
    i_bad = rng.uniform(17.0, 21.0, n_bad)
    flux_bad = 10.0 ** (-0.4 * (i_bad - 28.6)) * (
        1.0 + rng.normal(0.0, 0.01, n_bad)
    )
    ferr_bad = 0.005 * flux_bad  # S/N ~200: they top the S/N ladder
    return pd.DataFrame(
        {
            "i": np.concatenate([i_good, i_bad]),
            "i_err": np.full(n_good + n_bad, 0.02),
            "flux_AP": np.concatenate([flux_good, flux_bad]),
            "flux_AP_err": np.concatenate([ferr_good, ferr_bad]),
        }
    )


class TestSaturationMajorityLocus:
    def _cat(self, tmp_path):
        return Catalog(
            input_yaml={
                "fpath": str(tmp_path / "img.fits"),
                "imageFilter": "i",
            }
        )

    def test_contaminated_seed_reanchors_on_majority(self, tmp_path):
        cat = self._cat(tmp_path)
        df = _locus_catalog()
        out, params, sat_range = cat.check_saturation_range(df)
        # Without the guard the RANSAC lands at ~28.6 with ~12 inliers.
        assert abs(params["intercept"] - 24.0) < 0.4
        assert params["n_inliers"] >= 50
        # The artifact entries must not survive into the returned catalog.
        good = 10.0 ** (-0.4 * (out["i"] - 24.0))
        assert np.allclose(out["flux_AP"].values, good, rtol=0.5)

    def test_clean_catalog_unchanged(self, tmp_path):
        # With no artifact population the guard must not fire: the
        # RANSAC result and a clean slope-1 fit should agree closely.
        cat = self._cat(tmp_path)
        rng = np.random.default_rng(3)
        i = rng.uniform(14.0, 21.0, 120)
        flux = 10.0 ** (-0.4 * (i - 24.0)) * (1.0 + rng.normal(0, 0.02, 120))
        df = pd.DataFrame(
            {
                "i": i,
                "i_err": np.full(120, 0.02),
                "flux_AP": flux,
                "flux_AP_err": 0.005 * flux,
            }
        )
        out, params, sat_range = cat.check_saturation_range(df)
        assert abs(params["intercept"] - 24.0) < 0.2
        assert params["n_inliers"] >= 100


def _legacy_frame(rows):
    """Minimal Legacy Survey-shaped frame (griz only, no Johnson bands)."""
    return pd.DataFrame(
        {
            "ra": [r[0] for r in rows],
            "dec": [r[1] for r in rows],
            "mag_g": [r[2] for r in rows],
            "mag_g_e": [r[3] for r in rows],
            "mag_r": [r[4] for r in rows],
            "mag_r_e": [r[5] for r in rows],
            "mag_i": [r[6] for r in rows],
            "mag_i_e": [r[7] for r in rows],
            "mag_z": [r[8] for r in rows],
            "mag_z_e": [r[9] for r in rows],
        }
    )


class TestMissingBandRescue:
    """An image band absent from the catalog but synthesizable via
    published transformations must still be populated.  Regression: a
    B-band image calibrated against a legacy catalog previously reached
    fit_zeropoint with no B column and died on KeyError."""

    _ROWS = [
        (142.00, 14.10, 18.0, 0.02, 17.8, 0.02, 17.7, 0.02, 17.6, 0.03),
        (142.01, 14.12, 19.5, 0.03, 19.2, 0.03, 19.0, 0.03, 18.9, 0.04),
        (142.02, 14.14, 17.2, 0.02, 17.0, 0.02, 16.9, 0.02, 16.8, 0.03),
    ]

    def _cat(self, image_filter, **cat_cfg):
        cfg = {"use_catalog": "legacy"}
        cfg.update(cat_cfg)
        return Catalog(
            input_yaml={
                "imageFilter": image_filter,
                "catalog": cfg,
            }
        )

    def test_missing_image_band_synthesized(self):
        # allow_filter_transformations deliberately left unset (off).
        cat = self._cat("B")
        out = cat.clean(_legacy_frame(self._ROWS), catalogName="legacy")
        assert "B" in out.columns
        assert np.isfinite(out["B"]).all()
        assert "B_err" in out.columns
        assert np.isfinite(out["B_err"]).all()
        # Synthesized errors must exceed the input band errors - the
        # published r.m.s. floor is propagated.
        assert (out["B_err"] > out["g_err"]).all()

    def test_nontransformable_band_stays_absent(self):
        # Legacy cannot synthesize u; the column stays missing so the
        # downstream zeropoint error can name the cause.
        cat = self._cat("u")
        out = cat.clean(_legacy_frame(self._ROWS), catalogName="legacy")
        assert "u" not in out.columns

    def test_native_band_untouched(self):
        # A band the survey measures must not be overwritten by a
        # synthesized value.
        cat = self._cat("r")
        out = cat.clean(_legacy_frame(self._ROWS), catalogName="legacy")
        assert out["r"].tolist() == [17.8, 19.2, 17.0]
