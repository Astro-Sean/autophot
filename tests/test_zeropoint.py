"""Regression tests for Zeropoint.fit_zeropoint.

Covers the full fit + diagnostic-plot path, which produced a NameError
(zeropoint.py legend label) after a logging refactor: the exception was
swallowed by the outer try/except, returning fit_params with only the AP
entry so PSF magnitudes stayed NaN downstream.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from autophot.calibration.zeropoint import Zeropoint


def _catalog(n=30, zp_true=24.5, seed=1):
    rng = np.random.default_rng(seed)
    catmag = rng.uniform(16.0, 21.0, n)
    catmag_err = np.full(n, 0.02)
    # inst = catmag - zp_true (slope-1 relation)
    inst = catmag - zp_true + rng.normal(0.0, 0.01, n)
    flux = 10.0 ** (-0.4 * inst)
    flux_err = 0.02 * flux
    return pd.DataFrame(
        {
            "r": catmag,
            "r_err": catmag_err,
            "flux_AP": flux,
            "flux_AP_err": flux_err,
            "flux_PSF": flux * 1.02,
            "flux_PSF_err": flux_err,
        }
    )


def _yaml(tmp_path, fit_method="ransac"):
    return {
        "fpath": str(tmp_path / "img.fits"),
        "imageFilter": "r",
        "zeropoint": {"fit_method": fit_method},
    }


def _offset_catalog(n=40, zp_true=24.5, seed=31):
    """PSF fluxes 2x aperture fluxes -> ZP_PSF sits 0.7526 mag high."""
    rng = np.random.default_rng(seed)
    catmag = rng.uniform(16.0, 21.0, n)
    inst = catmag - zp_true + rng.normal(0.0, 0.01, n)
    flux = 10.0 ** (-0.4 * inst)
    return pd.DataFrame(
        {
            "r": catmag,
            "r_err": np.full(n, 0.01),
            "flux_AP": flux,
            "flux_AP_err": 0.01 * flux,
            "flux_PSF": 2.0 * flux,
            "flux_PSF_err": 0.01 * flux,
        }
    )


class TestFitZeropoint:
    def test_psf_fit_positions_are_used_for_psf_calibrators(self):
        catalog = pd.DataFrame(
            {
                "x_pix": [10.0, 20.0],
                "y_pix": [11.0, 21.0],
                "x_fit": [10.4, np.nan],
                "y_fit": [11.3, 20.8],
            }
        )

        x_psf, y_psf = Zeropoint._photometry_positions(catalog, "PSF")
        x_ap, y_ap = Zeropoint._photometry_positions(catalog, "AP")

        np.testing.assert_allclose(x_psf, [10.4, 20.0])
        np.testing.assert_allclose(y_psf, [11.3, 20.8])
        np.testing.assert_allclose(x_ap, [10.0, 20.0])
        np.testing.assert_allclose(y_ap, [11.0, 21.0])

    def test_psf_zeropoint_uses_fitted_positions_for_spatial_errors(
        self, tmp_path, monkeypatch
    ):
        catalog = _catalog(n=30)
        x_pix = np.linspace(10.0, 100.0, len(catalog))
        y_pix = np.linspace(20.0, 120.0, len(catalog))
        catalog["x_pix"] = x_pix
        catalog["y_pix"] = y_pix
        catalog["x_fit"] = x_pix + 3.0
        catalog["y_fit"] = y_pix - 2.0
        captured = []

        def capture_positions(residuals, x, y):
            captured.append((np.asarray(x).copy(), np.asarray(y).copy()))
            return np.nan, np.nan

        monkeypatch.setattr(
            Zeropoint,
            "_block_bootstrap_se",
            staticmethod(capture_positions),
        )
        Zeropoint(_yaml(tmp_path)).fit_zeropoint(catalog)

        assert len(captured) == 2
        np.testing.assert_allclose(captured[0][0], x_pix)
        np.testing.assert_allclose(captured[0][1], y_pix)
        np.testing.assert_allclose(captured[1][0], x_pix + 3.0)
        np.testing.assert_allclose(captured[1][1], y_pix - 2.0)

    def test_both_methods_return_finite_zp(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path))
        cat, params = zp.fit_zeropoint(_catalog())
        for method in ("AP", "PSF"):
            entry = params.get(method) or {}
            assert np.isfinite(
                entry.get("zeropoint", np.nan)
            ), f"{method} zeropoint missing/non-finite: {entry}"
            assert abs(entry["zeropoint"] - 24.5) < 0.2

    def test_single_calibrator_still_yields_zeropoint(self, tmp_path):
        # A sparse field can leave exactly one valid calibrator: too few
        # for the model fitters but still enough to anchor a zeropoint
        # via the direct median estimate (flagged unreliable).
        zp = Zeropoint(_yaml(tmp_path))
        cat_out, params = zp.fit_zeropoint(_catalog(n=1))
        for method in ("AP", "PSF"):
            entry = params.get(method) or {}
            assert np.isfinite(
                entry.get("zeropoint", np.nan)
            ), f"{method} zeropoint missing/non-finite: {entry}"
            assert abs(entry["zeropoint"] - 24.5) < 0.2
            assert entry.get("n_sources") == 1
            assert entry.get("reliable") is False
        # The contributing calibrator is retained, not filtered out.
        assert len(cat_out) == 1

    def test_psf_survives_when_ap_ran_first(self, tmp_path):
        # The NameError fired during AP plotting, so PSF never ran.
        zp = Zeropoint(_yaml(tmp_path))
        _, params = zp.fit_zeropoint(_catalog())
        assert params["PSF"].get("n_sources", 0) >= 2

    def test_all_outlier_mask_does_not_abort(self, tmp_path, monkeypatch):
        # The MCMC mixture can legally return an all-False inlier mask
        # (f_out pinned at its 0.5 prior cap, or an unconverged chain).
        # Before the guard, the plotting code crashed on min() of an empty
        # array and the whole fit aborted: the PSF pass never ran, the
        # diagnostic figure was never written, and the RAW catalog was
        # returned instead of clean_catalog.
        zp = Zeropoint(_yaml(tmp_path, fit_method="mcmc"))

        def _no_inliers(self, x, y, x_err, y_err, **kwargs):
            return 24.5, 0.6, np.zeros(len(x), dtype=bool)

        monkeypatch.setattr(Zeropoint, "_mcmc_fit", _no_inliers)
        cat_in = _catalog()
        cat_out, params = zp.fit_zeropoint(cat_in)
        for method in ("AP", "PSF"):
            entry = params.get(method) or {}
            assert np.isfinite(
                entry.get("zeropoint", np.nan)
            ), f"{method} zeropoint missing: {entry}"
        # Diagnostic figure must be produced even with zero flagged inliers.
        assert list(tmp_path.glob("Zeropoint_*"))
        # The returned catalog must be the cleaned catalog, not the raw input.
        assert cat_out is not cat_in
        assert len(cat_out) == len(cat_in)

    def test_nan_zp_mask_does_not_abort(self, tmp_path, monkeypatch):
        # The fitters return (nan, nan, all-False) when too few finite
        # points exist.  The guard must keep plotting alive so the frame
        # still gets its diagnostic figure.
        zp = Zeropoint(_yaml(tmp_path, fit_method="mcmc"))

        def _no_fit(self, x, y, x_err, y_err, **kwargs):
            return np.nan, np.nan, np.zeros(len(x), dtype=bool)

        monkeypatch.setattr(Zeropoint, "_mcmc_fit", _no_fit)
        cat_out, params = zp.fit_zeropoint(_catalog())
        assert list(tmp_path.glob("Zeropoint_*"))
        assert len(cat_out) == 30


def _color_catalog(n=80, zp_true=25.0, slope=0.15, color_lo=0.2, color_hi=1.5, seed=2):
    """r-band catalog where y = cat_r - inst_r = zp_true + slope*(g - r)."""
    rng = np.random.default_rng(seed)
    r_cat = rng.uniform(16.0, 21.0, n)
    color = rng.uniform(color_lo, color_hi, n)
    g_cat = r_cat + color
    inst = r_cat - zp_true - slope * color + rng.normal(0.0, 0.01, n)
    flux = 10.0 ** (-0.4 * inst)
    return pd.DataFrame(
        {
            "r": r_cat,
            "r_err": np.full(n, 0.02),
            "g": g_cat,
            "g_err": np.full(n, 0.02),
            "flux_AP": flux,
            "flux_AP_err": 0.02 * flux,
            "threshold": np.full(n, 10.0),
        }
    )


def _color_yaml(tmp_path):
    return {
        "fpath": str(tmp_path / "img.fits"),
        "imageFilter": "r",
        "zeropoint": {"min_source_no": 5},
        "photometry": {"color_term_poly_order": 1},
    }


class TestFitColorTerm:
    def test_strong_color_term_accepted(self, tmp_path):
        zp = Zeropoint(_color_yaml(tmp_path))
        coeffs, errs = zp.fit_color_term(_color_catalog(slope=0.15))
        assert coeffs is not None
        assert coeffs[1] != 0.0
        # Fitted slope should be in the vicinity of the injected 0.15.
        assert 0.05 < coeffs[1] < 0.30

    def test_flat_color_term_rejected(self, tmp_path):
        # No true color dependence: a fitted slope here is noise and must
        # not be applied to the zeropoint.
        zp = Zeropoint(_color_yaml(tmp_path))
        coeffs, _ = zp.fit_color_term(_color_catalog(slope=0.0))
        assert coeffs == (0.0, 0.0)

    def test_too_few_sources_rejected(self, tmp_path):
        zp = Zeropoint(_color_yaml(tmp_path))
        coeffs, _ = zp.fit_color_term(_color_catalog(n=4, slope=0.15))
        assert coeffs == (0.0, 0.0)

    def test_narrow_color_range_rejected(self, tmp_path):
        # A real slope cannot be measured off a 0.02 mag color baseline.
        zp = Zeropoint(_color_yaml(tmp_path))
        coeffs, _ = zp.fit_color_term(
            _color_catalog(slope=0.15, color_lo=0.49, color_hi=0.51)
        )
        assert coeffs == (0.0, 0.0)


def _clean_catalog(n=12, with_snr=True):
    """Catalog that passes every Zeropoint.clean() cut."""
    cat = _catalog(n)
    cat["x_pix"] = np.arange(n) * 10.0 + 5.0
    cat["y_pix"] = 50.0
    if with_snr:
        cat["snr"] = 50.0
    return cat


def _heteroscedastic_catalog(n=24, zp_true=24.5, seed=3):
    """Slope-1 catalog with strongly heteroscedastic errors and a mild
    coherent bright-end tilt: the brightest three calibrators carry
    +0.08/+0.04/+0.02 mag offsets (a small nonlinearity signature) with
    very small errors.  Under squared-weight fitting the three points
    dominate the free-slope diagnostic and drag it off the bulk."""
    rng = np.random.default_rng(seed)
    inst = np.sort(rng.uniform(17.0, 22.0, n))
    catmag = inst + zp_true + rng.normal(0.0, 0.015, n)
    inst_err = np.linspace(0.004, 0.04, n)
    cat_err = np.linspace(0.005, 0.05, n)
    catmag[:3] += np.array([0.08, 0.04, 0.02])
    flux = 10.0 ** (-0.4 * inst)
    # inst_mag_err = (2.5/ln10) * flux_err/flux inside _compute_delta_mag.
    flux_err = flux * (inst_err / (2.5 / np.log(10.0)))
    return pd.DataFrame(
        {
            "r": catmag,
            "r_err": cat_err,
            "flux_AP": flux,
            "flux_AP_err": flux_err,
            "flux_PSF": flux,
            "flux_PSF_err": flux_err,
        }
    )


class TestFreeSlopeDiagnostic:
    """The free-slope fit solves a WLS problem; lstsq must see the
    sqrt-weight whitened system.  The previous form, lstsq(W@A, W@y),
    applies the inverse-variance weights twice (effective 1/sigma^4)
    so a handful of bright low-error calibrators -- exactly the
    population a magnitude-dependent bias shows up in -- controls the
    line and the plotted diagnostic looks wrong.
    """

    def test_free_slope_not_dragged_by_bright_deviants(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path, fit_method="odr"))
        _, params = zp.fit_zeropoint(_heteroscedastic_catalog())
        for method in ("AP", "PSF"):
            entry = params.get(method) or {}
            fs = entry.get("free_slope", np.nan)
            assert np.isfinite(fs), f"{method} free_slope missing"
            # Squared-weight fitting returns ~0.94 on this catalog; a
            # correctly weighted fit stays above 0.95 (OLS ~0.99).
            assert fs > 0.95, f"{method} free_slope={fs:.4f}"

    def test_free_slope_recovers_unity_on_clean_catalog(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path, fit_method="odr"))
        _, params = zp.fit_zeropoint(_catalog())
        for method in ("AP", "PSF"):
            fs = params[method].get("free_slope", np.nan)
            assert np.isfinite(fs)
            assert abs(fs - 1.0) < 0.1


class TestCleanFWHMReject:
    """Zeropoint.clean() FWHM outlier rejection must also engage on
    small calibrator sets - sparse fields are where extended sources do
    the most damage to the fit."""

    def test_extended_source_dropped_small_n(self, tmp_path):
        cat = _clean_catalog(n=6)
        cat["fwhm"] = [7.3, 7.5, 7.4, 7.6, 7.2, 14.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 5
        assert out["fwhm"].max() < 10

    def test_narrow_outlier_dropped_small_n(self, tmp_path):
        cat = _clean_catalog(n=6)
        cat["fwhm"] = [7.3, 7.5, 7.4, 7.6, 7.2, 1.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 5

    def test_keep_floor_protects_starved_clean(self, tmp_path):
        cat = _clean_catalog(n=4)
        cat["fwhm"] = [7.3, 7.5, 14.0, 15.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 4

    def test_below_min_sources_skips(self, tmp_path):
        cat = _clean_catalog(n=3)
        cat["fwhm"] = [7.3, 7.5, 14.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 3

    def test_disable_with_zero_sigma(self, tmp_path):
        cat = _clean_catalog(n=6)
        cat["fwhm"] = [7.3, 7.5, 7.4, 7.6, 7.2, 14.0]
        y = _yaml(tmp_path)
        y["zeropoint"]["fwhm_reject_sigma"] = 0.0
        out = Zeropoint(y).clean(cat)
        assert len(out) == 6


class TestCleanDefectFlag:
    """Zeropoint.clean() defect_flag rejection (main.py flags calibrators
    whose footprint overlaps hardware_defects_mask)."""

    def test_flagged_sources_dropped(self, tmp_path):
        cat = _clean_catalog()
        cat["defect_flag"] = False
        cat.loc[[2, 5, 9], "defect_flag"] = True
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == len(cat) - 3
        assert not out["defect_flag"].astype(bool).any()

    def test_reject_disabled_keeps_flagged(self, tmp_path):
        cat = _clean_catalog()
        cat["defect_flag"] = True
        y = _yaml(tmp_path)
        y["zeropoint"]["reject_defect_sources"] = False
        out = Zeropoint(y).clean(cat)
        assert len(out) == len(cat)

    def test_no_flag_column_unchanged(self, tmp_path):
        cat = _clean_catalog()
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == len(cat)

    def test_flag_applies_without_snr_column(self, tmp_path):
        # The defect cut must run on the magnitude-only path too (sources
        # with no threshold/snr columns return early).
        cat = _clean_catalog(with_snr=False)
        cat["defect_flag"] = False
        cat.loc[[0, 11], "defect_flag"] = True
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == len(cat) - 2

    def test_flagged_kept_when_pool_below_floor(self, tmp_path):
        # Sparse field: dropping every flagged calibrator would empty
        # the ZP pool; the best flagged sources are rescued up to
        # defect_reject_min_keep instead.
        cat = _clean_catalog(n=4)
        cat["defect_flag"] = False
        cat.loc[[1, 2, 3], "defect_flag"] = True
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 3
        assert out["defect_flag"].astype(bool).sum() == 2

    def test_flagged_all_rescued_when_pool_below_floor(self, tmp_path):
        # The reported failure mode: every calibrator is defect-flagged
        # on a sparse field.  The floor keeps them all rather than
        # leaving zero sources for the ZP fit.
        cat = _clean_catalog(n=2)
        cat["defect_flag"] = True
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 2


class TestCleanMinKeep:
    """Keep-floor on the combined quality mask in Zeropoint.clean():
    the union of soft cuts must not starve a sparse calibrator pool."""

    def test_soft_reject_rescued_below_floor(self, tmp_path):
        cat = _clean_catalog(n=4)
        cat.loc[[2, 3], "snr"] = 0.5  # below the S/N cut
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 3

    def test_no_mag_row_never_rescued(self, tmp_path):
        # A row without a finite catalog magnitude cannot calibrate and
        # must not be rescued.
        cat = _clean_catalog(n=4)
        cat.loc[3, "r"] = np.nan
        cat.loc[[2, 3], "snr"] = 0.5
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        # The salvageable row is rescued; the NaN-mag row is not.
        assert len(out) == 3
        assert out["r"].notna().all()


class TestCleanPSFFitQuality:
    """Zeropoint.clean() rejects sources with pathological PSF-fit
    quality (qfit, |cfit|, reduced_chi2) - blended or non-converged fits
    that still carry plausible fluxes and can anchor a wrong locus."""

    def test_qfit_outlier_dropped(self, tmp_path):
        cat = _clean_catalog(n=8)
        cat["qfit"] = [3.0] * 7 + [80.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 7
        assert out["qfit"].max() < 10

    def test_cfit_abs_outlier_dropped(self, tmp_path):
        # |cfit| is clipped: a large negative centroid correction is as
        # suspect as a large positive one.
        cat = _clean_catalog(n=8)
        cat["cfit"] = [0.01] * 7 + [-2.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 7

    def test_reduced_chi2_outlier_dropped(self, tmp_path):
        cat = _clean_catalog(n=8)
        cat["reduced_chi2"] = [1.0] * 7 + [50.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 7

    def test_disabled_with_zero_sigma(self, tmp_path):
        cat = _clean_catalog(n=8)
        cat["qfit"] = [3.0] * 7 + [80.0]
        y = _yaml(tmp_path)
        y["zeropoint"]["psf_fit_reject_sigma"] = 0.0
        out = Zeropoint(y).clean(cat)
        assert len(out) == 8

    def test_few_finite_skips_clip(self, tmp_path):
        # Fewer than fwhm_reject_min_sources finite values: the clip is
        # too unstable to trust.
        cat = _clean_catalog(n=6)
        cat["qfit"] = [np.nan] * 4 + [3.0, 80.0]
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == 6

    def test_missing_columns_unchanged(self, tmp_path):
        cat = _clean_catalog()
        out = Zeropoint(_yaml(tmp_path)).clean(cat)
        assert len(out) == len(cat)


class TestODRLocusAnchor:
    """The slope-1 fit must anchor on the densest delta locus, not the
    median or an all-points mean: with ~50% contamination the median
    lands between loci and an unbounded clip width then swallows both
    (the 2025-08-07 ZTF-i failure mode, where spurious faint catalog
    entries on bright stars form a second ZP locus ~5 mag high)."""

    def _bimodal(self, n_good=45, n_bad=60, seed=7):
        rng = np.random.default_rng(seed)
        x = np.concatenate(
            [rng.uniform(16, 21, n_good), rng.uniform(16, 21, n_bad)]
        )
        # Bogus population is the majority but spread over a wide range;
        # the good population is the densest single locus.
        delta = np.concatenate(
            [
                24.0 + rng.normal(0.0, 0.02, n_good),
                rng.uniform(26.5, 29.5, n_bad),
            ]
        )
        errs = np.full(n_good + n_bad, 0.02)
        return x, x + delta, errs, n_good

    def test_odr_recovers_dominant_locus(self):
        x, y, errs, n_good = self._bimodal()
        zp, zp_err, mask = Zeropoint({"zeropoint": {}})._odr_slope1_fit(
            x, y, errs, errs
        )
        assert abs(zp - 24.0) < 0.1
        # the dense locus must dominate the inlier set
        assert int(mask[:n_good].sum()) >= 40
        assert int(mask[n_good:].sum()) < 20

    def test_odr_clean_data_unchanged(self):
        rng = np.random.default_rng(9)
        x = rng.uniform(16, 21, 40)
        delta = 24.5 + rng.normal(0.0, 0.05, 40)
        errs = np.full(40, 0.03)
        zp, zp_err, mask = Zeropoint({"zeropoint": {}})._odr_slope1_fit(
            x, x + delta, errs, errs
        )
        assert abs(zp - 24.5) < 0.05
        assert mask.sum() >= 35


class TestMCMCFastPath:
    """2025-08-07 ZTF-i: a calibrator set entirely on a bogus locus had
    MAD ~1 mag but the fast-path "clean data" test measured cleanliness
    against the systematic floor (which is MAD^2 by construction), so
    the weighted mean ran on garbage and reported a tiny formal error."""

    def _scattered(self, n=15, seed=11):
        rng = np.random.default_rng(seed)
        x = rng.uniform(16, 21, n)
        delta = 28.5 + rng.normal(0.0, 0.5, n)
        errs = np.full(n, 0.02)
        return x, x + delta, errs

    def test_scattered_data_skips_fast_path(self, caplog):
        x, y, errs = self._scattered()
        zp_obj = Zeropoint({"zeropoint": {}})
        with caplog.at_level(logging.INFO, logger="zeropoint"):
            _, zp_err, _ = zp_obj._mcmc_fit(
                x, y, errs, errs, max_steps=3000
            )
        assert "fast-path" not in caplog.text.lower()
        # Whatever the mixture returns, the error must reflect the
        # observed mag-scale scatter, not the tiny formal errors.
        assert zp_err > 0.05

    def test_clean_data_still_fast_paths(self, caplog):
        rng = np.random.default_rng(13)
        x = rng.uniform(16, 21, 20)
        delta = 24.0 + rng.normal(0.0, 0.01, 20)
        errs = np.full(20, 0.02)
        zp_obj = Zeropoint({"zeropoint": {}})
        with caplog.at_level(logging.INFO, logger="zeropoint"):
            zp, zp_err, _ = zp_obj._mcmc_fit(x, x + delta, errs, errs)
        assert "fast-path" in caplog.text.lower()
        assert abs(zp - 24.0) < 0.05
        assert zp_err < 0.05

    def test_single_dominant_source_skips_fast_path(self, caplog):
        # One tiny-error source carrying >50% of the total weight must
        # not let a weighted mean masquerade as a clean fit.
        rng = np.random.default_rng(17)
        n = 12
        x = rng.uniform(16, 21, n)
        delta = 24.0 + rng.normal(0.0, 0.03, n)
        delta[-1] = 24.05
        errs = np.full(n, 0.05)
        errs[-1] = 0.0005  # dominates sum(1/var) by ~10^4x
        zp_obj = Zeropoint({"zeropoint": {}})
        with caplog.at_level(logging.INFO, logger="zeropoint"):
            zp_obj._mcmc_fit(x, x + delta, errs, errs, max_steps=3000)
        assert "fast-path" not in caplog.text.lower()


class TestZPErrorScatterFloor:
    """The reported zeropoint error is floored at the inlier
    dispersion/sqrt(N): a formal error tighter than the observed scatter
    is dishonest (08-07 reported +/-0.021 mag with inlier MAD = 1.4)."""

    def test_scatter_floor_raises_tiny_formal_error(
        self, tmp_path, monkeypatch
    ):
        rng = np.random.default_rng(5)
        n = 30
        catmag = rng.uniform(16.0, 21.0, n)
        # 0.2 mag true scatter but tiny claimed errors
        inst = catmag - 24.5 + rng.normal(0.0, 0.2, n)
        flux = 10.0 ** (-0.4 * inst)
        df = pd.DataFrame(
            {
                "r": catmag,
                "r_err": np.full(n, 0.01),
                "flux_AP": flux,
                "flux_AP_err": 0.01 * flux,
                "flux_PSF": flux,
                "flux_PSF_err": 0.01 * flux,
            }
        )

        monkeypatch.setattr(
            Zeropoint,
            "_mcmc_fit",
            lambda self, x, y, x_err, y_err, **k: (
                24.5,
                0.001,
                np.ones(len(x), dtype=bool),
            ),
        )
        zp = Zeropoint(_yaml(tmp_path, fit_method="mcmc"))
        _, params = zp.fit_zeropoint(df)
        # floor = SE(median) = 1.858 * 0.2 / sqrt(30) ~ 0.068
        assert params["AP"]["zeropoint_error"] >= 0.04
        assert params["AP"]["n_inliers"] == n


class TestReliabilityFlags:
    """A mis-stacked or mis-registered image has no dominant 1:1 locus:
    the fit must still return a number but flag it as unreliable rather
    than letting a scatter-floored ZP pass quietly."""

    def test_clean_catalog_is_reliable(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path))
        _, params = zp.fit_zeropoint(_catalog())
        for method in ("AP", "PSF"):
            assert params[method]["reliable"] is True
            assert "reliability_flags" not in params[method]

    def test_huge_scatter_flagged_unreliable(self, tmp_path):
        # delta scatter ~0.8 mag: no coherent 1:1 locus exists.
        rng = np.random.default_rng(21)
        n = 30
        catmag = rng.uniform(16.0, 21.0, n)
        inst = catmag - 24.5 + rng.normal(0.0, 0.8, n)
        flux = 10.0 ** (-0.4 * inst)
        df = pd.DataFrame(
            {
                "r": catmag,
                "r_err": np.full(n, 0.02),
                "flux_AP": flux,
                "flux_AP_err": 0.02 * flux,
                "flux_PSF": flux,
                "flux_PSF_err": 0.02 * flux,
            }
        )
        zp = Zeropoint(_yaml(tmp_path))
        _, params = zp.fit_zeropoint(df)
        assert params["AP"]["reliable"] is False
        assert any(
            "scatter" in f for f in params["AP"]["reliability_flags"]
        )
        # Still returns a number - degraded, not dead.
        assert np.isfinite(params["AP"]["zeropoint"])

    def test_spatial_gradient_flagged_unreliable(self, tmp_path):
        # Mis-stacking signature: flat in magnitude space but a >3-sigma
        # tilt of the residuals across the detector.
        rng = np.random.default_rng(23)
        n = 40
        catmag = rng.uniform(16.0, 21.0, n)
        x_pix = rng.uniform(0.0, 1000.0, n)
        inst = catmag - 24.5 - 0.35 * x_pix / 1000.0 + rng.normal(
            0.0, 0.01, n
        )
        flux = 10.0 ** (-0.4 * inst)
        df = pd.DataFrame(
            {
                "r": catmag,
                "r_err": np.full(n, 0.02),
                "flux_AP": flux,
                "flux_AP_err": 0.02 * flux,
                "flux_PSF": flux,
                "flux_PSF_err": 0.02 * flux,
                "x_pix": x_pix,
                "y_pix": rng.uniform(0.0, 1000.0, n),
            }
        )
        zp = Zeropoint(_yaml(tmp_path))
        _, params = zp.fit_zeropoint(df)
        assert params["AP"]["reliable"] is False
        assert any(
            "gradient" in f for f in params["AP"]["reliability_flags"]
        )

    def test_minority_locus_flagged_unreliable(
        self, tmp_path, monkeypatch
    ):
        # A fitter that converged on a minority locus (e.g. the
        # contaminated-seed failure) must not read as reliable even
        # though the surviving subset is internally tight.
        n = 30
        mask = np.zeros(n, dtype=bool)
        mask[:3] = True

        def _minority_fit(self, x, y, x_err, y_err, **k):
            return 28.5, 0.01, mask

        monkeypatch.setattr(Zeropoint, "_mcmc_fit", _minority_fit)
        zp = Zeropoint(_yaml(tmp_path, fit_method="mcmc"))
        _, params = zp.fit_zeropoint(_catalog(n=n))
        assert params["AP"]["reliable"] is False
        assert any(
            "minority" in f for f in params["AP"]["reliability_flags"]
        )

    def test_nonfinite_zp_flagged_unreliable(self, tmp_path, monkeypatch):
        # A failed fit returns NaN; reliable=True next to
        # ZP_<m>="unknown" in the header is a contradiction.
        def _nan_fit(self, x, y, x_err, y_err, **k):
            return np.nan, np.nan, np.zeros(len(x), dtype=bool)

        monkeypatch.setattr(Zeropoint, "_mcmc_fit", _nan_fit)
        zp = Zeropoint(_yaml(tmp_path, fit_method="mcmc"))
        _, params = zp.fit_zeropoint(_catalog())
        for method in ("AP", "PSF"):
            assert params[method]["reliable"] is False
            assert any(
                "finite" in f for f in params[method]["reliability_flags"]
            )


class TestApPsfAgreement:
    """AP fluxes are measured at the finite science aperture while PSF
    fluxes are total-flux.  The consistency check bridges the two scales
    with the shared-inlier offset d = inst_PSF - inst_AP, and compares
    that measured offset against the external CoG aperture correction -
    a uniform offset the correction cannot explain is flagged."""

    def test_uncorrected_offset_flags_both(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path, fit_method="odr"))
        _, params = zp.fit_zeropoint(_offset_catalog())
        for method in ("AP", "PSF"):
            assert params[method]["reliable"] is False
            assert any(
                "AP" in f and "PSF" in f
                for f in params[method]["reliability_flags"]
            )

    def test_aperture_correction_explains_offset(self, tmp_path):
        # ZP_AP = ZP_true + ap_corr, so with ap_corr = -0.7526 the
        # corrected comparison is consistent and no flag fires.
        y = _yaml(tmp_path, fit_method="odr")
        y["aperture_correction"] = -2.5 * np.log10(2.0)
        y["aperture_correction_err"] = 0.01
        zp = Zeropoint(y)
        _, params = zp.fit_zeropoint(_offset_catalog())
        for method in ("AP", "PSF"):
            assert params[method]["reliable"] is True
            assert "reliability_flags" not in params[method]


class TestMCMCReproducibility:
    """The zeropoint must be a deterministic function of the image: an
    unseeded sampler reports a slightly different ZP and error on every
    run of the same data, which no downstream consumer can justify."""

    def _contaminated(self, n=25, seed=41):
        rng = np.random.default_rng(seed)
        x = rng.uniform(16, 21, n)
        delta = 24.0 + rng.normal(0.0, 0.05, n)
        delta[:4] += rng.uniform(0.6, 1.2, n - (n - 4))
        errs = np.full(n, 0.03)
        return x, x + delta, errs

    def test_seeded_runs_identical(self):
        x, y, errs = self._contaminated()
        zp_obj = Zeropoint({"zeropoint": {}})
        kw = dict(n_burn=100, n_steps=200, max_steps=2000, random_state=42)
        a = zp_obj._mcmc_fit(x, y, errs, errs, **kw)
        b = zp_obj._mcmc_fit(x, y, errs, errs, **kw)
        assert a[0] == b[0] and a[1] == b[1]
        assert (a[2] == b[2]).all()

    def test_standard_mode_no_crash(self):
        # robust=False used to die on a NameError: zp_init_spread was
        # only defined in the robust branch.
        x, y, errs = self._contaminated()
        zp_obj = Zeropoint({"zeropoint": {}})
        zp, zp_err, mask = zp_obj._mcmc_fit(
            x, y, errs, errs, robust=False,
            n_burn=100, n_steps=200, max_steps=2000, random_state=42,
        )
        assert np.isfinite(zp)
        assert np.isfinite(zp_err)
        assert mask.sum() > 0

    def test_fit_zeropoint_deterministic(self, tmp_path):
        # End-to-end: the mcmc path must give the same ZP/error twice.
        y = _yaml(tmp_path, fit_method="mcmc")
        rng = np.random.default_rng(43)
        n = 20
        catmag = rng.uniform(16.0, 21.0, n)
        inst = catmag - 24.5 + rng.normal(0.0, 0.4, n)
        flux = 10.0 ** (-0.4 * inst)
        df = pd.DataFrame(
            {
                "r": catmag,
                "r_err": np.full(n, 0.01),
                "flux_AP": flux,
                "flux_AP_err": 0.01 * flux,
                "flux_PSF": flux,
                "flux_PSF_err": 0.01 * flux,
            }
        )
        zp = Zeropoint(y)
        _, p1 = zp.fit_zeropoint(df.copy())
        _, p2 = zp.fit_zeropoint(df.copy())
        assert (
            p1["AP"]["zeropoint"] == p2["AP"]["zeropoint"]
            and p1["AP"]["zeropoint_error"] == p2["AP"]["zeropoint_error"]
        )
        assert p1["AP"]["fit_method"] == "mcmc"


class TestFallbackSchema:
    """The fallback and sigma-clip paths must carry the same
    reliable/n_inliers/fit_method keys as fit_zeropoint - downstream
    writes ZP_<m>_OK header cards off them."""

    def test_fallback_marks_sparse_sample_unreliable(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path))
        cat = _catalog(n=1)
        params = zp._fallback_zeropoint(cat, "r")
        assert params["AP"]["reliable"] is False
        assert params["AP"]["fit_method"] == "median_fallback"
        assert np.isfinite(params["AP"]["zeropoint"])
        # error must not be 0 for a single calibrator
        assert params["AP"]["zeropoint_error"] > 0.001

    def test_fallback_healthy_sample_reliable(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path))
        params = zp._fallback_zeropoint(_catalog(n=30), "r")
        assert params["AP"]["reliable"] is True
        assert params["AP"]["n_inliers"] == 30

    def test_estimate_zeropoint_without_psf_columns(self, tmp_path):
        # A catalog with AP photometry only must still get an AP
        # zeropoint from the sigma-clip path rather than falling back.
        cat = _catalog(n=30).drop(columns=["flux_PSF", "flux_PSF_err"])
        zp = Zeropoint(_yaml(tmp_path))
        _, params = zp.estimate_zeropoint(cat)
        assert np.isfinite(params["AP"]["zeropoint"])
        assert params["AP"]["reliable"] is True
        assert params["PSF"]["reliable"] is False
        assert params["PSF"]["fit_method"] == "sigma_clip"

    def test_estimate_zeropoint_flags_scatter(self, tmp_path):
        rng = np.random.default_rng(51)
        n = 30
        catmag = rng.uniform(16.0, 21.0, n)
        inst = catmag - 24.5 + rng.normal(0.0, 0.8, n)
        flux = 10.0 ** (-0.4 * inst)
        df = pd.DataFrame(
            {
                "r": catmag,
                "r_err": np.full(n, 0.02),
                "flux_AP": flux,
                "flux_AP_err": 0.02 * flux,
                "flux_PSF": flux,
                "flux_PSF_err": 0.02 * flux,
            }
        )
        zp = Zeropoint(_yaml(tmp_path))
        _, params = zp.estimate_zeropoint(df)
        assert params["AP"]["reliable"] is False
        assert any(
            "scatter" in f for f in params["AP"]["reliability_flags"]
        )


class TestSystematicTerm:
    """zeropoint.systematic_err is folded in quadrature into the
    reported zeropoint_error.  The fit statistical error
    (scatter/sqrt(N)) only sees per-source dispersion; field-level
    calibration systematics (catalog ties, flat-field residuals) do
    not average down with N.  zeropoint_error_stat and
    zeropoint_scatter keep the two components inspectable."""

    def test_systematic_raises_reported_error(self, tmp_path):
        y = _yaml(tmp_path, fit_method="odr")
        y["zeropoint"]["systematic_err"] = 0.02
        zp = Zeropoint(y)
        _, params = zp.fit_zeropoint(_catalog())
        ap = params["AP"]
        stat = ap["zeropoint_error_stat"]
        assert np.isfinite(stat)
        assert ap["zeropoint_error"] == pytest.approx(
            np.hypot(stat, 0.02), rel=1e-9
        )

    def test_scatter_reported_separately(self, tmp_path):
        # Injected inst noise sigma=0.01 -> inlier sigma ~0.01, far
        # below the mean error N would imply.
        zp = Zeropoint(_yaml(tmp_path, fit_method="odr"))
        _, params = zp.fit_zeropoint(_catalog())
        scat = params["AP"]["zeropoint_scatter"]
        assert np.isfinite(scat)
        assert scat == pytest.approx(0.01, abs=0.005)

    def test_systematic_disabled_with_zero(self, tmp_path):
        y = _yaml(tmp_path, fit_method="odr")
        y["zeropoint"]["systematic_err"] = 0.0
        zp = Zeropoint(y)
        _, params = zp.fit_zeropoint(_catalog())
        ap = params["AP"]
        assert ap["zeropoint_error"] == ap["zeropoint_error_stat"]

    def test_estimate_path_folds_systematic(self, tmp_path):
        y = _yaml(tmp_path)
        y["zeropoint"]["systematic_err"] = 0.05
        zp = Zeropoint(y)
        _, params = zp.estimate_zeropoint(_catalog())
        ap = params["AP"]
        assert ap["zeropoint_error"] > ap["zeropoint_error_stat"]
        assert np.isfinite(ap["zeropoint_scatter"])

    def test_cross_check_uses_stat_error(self, tmp_path):
        # A large configured systematic must not mask a real AP/PSF
        # split: it is common-mode between the two measurements and
        # cancels in the difference.
        y = _yaml(tmp_path, fit_method="odr")
        y["zeropoint"]["systematic_err"] = 0.2
        zp = Zeropoint(y)
        _, params = zp.fit_zeropoint(_offset_catalog())
        for method in ("AP", "PSF"):
            assert params[method]["reliable"] is False


class TestSpatialCorrelation:
    """Calibrator residuals varying smoothly across the detector are
    spatially correlated: neighbouring calibrators do not give
    independent measurements of the zeropoint.  The block-bootstrap
    standard error (Kunsch 1989; Carlstein 1986) folds that
    correlation into the reported error and deflates the effective
    calibrator count."""

    def _patchy_catalog(self, n=60, zp_true=24.5, seed=77):
        rng = np.random.default_rng(seed)
        catmag = rng.uniform(16.0, 21.0, n)
        x_pix = rng.uniform(0.0, 1000.0, n)
        y_pix = rng.uniform(0.0, 1000.0, n)
        # A two-region field offset: residuals share a sign within
        # each half -> whole blocks correlate with each other.
        patch = np.where(x_pix > 500, 0.15, -0.15)
        inst = catmag - zp_true - patch + rng.normal(0.0, 0.01, n)
        flux = 10.0 ** (-0.4 * inst)
        return pd.DataFrame(
            {
                "r": catmag,
                "r_err": np.full(n, 0.02),
                "flux_AP": flux,
                "flux_AP_err": 0.02 * flux,
                "flux_PSF": flux,
                "flux_PSF_err": 0.02 * flux,
                "x_pix": x_pix,
                "y_pix": y_pix,
            }
        )

    def test_correlated_field_inflates_error_and_flags(self, tmp_path):
        zp = Zeropoint(_yaml(tmp_path, fit_method="odr"))
        _, params = zp.fit_zeropoint(self._patchy_catalog())
        ap = params["AP"]
        # The field offset is +/-0.15 mag, so a naive sigma/sqrt(N)
        # error (~0.02) understates the true uncertainty.
        assert np.isfinite(ap["n_eff"])
        assert ap["n_eff"] < 0.5 * ap["n_inliers"]
        assert ap["zeropoint_error_stat"] > 0.05
        assert ap["reliable"] is False
        assert any(
            "spatially correlated" in f
            for f in ap["reliability_flags"]
        )

    def test_iid_residuals_not_flagged(self, tmp_path):
        rng = np.random.default_rng(88)
        n = 60
        catmag = rng.uniform(16.0, 21.0, n)
        inst = catmag - 24.5 + rng.normal(0.0, 0.05, n)
        flux = 10.0 ** (-0.4 * inst)
        df = pd.DataFrame(
            {
                "r": catmag,
                "r_err": np.full(n, 0.02),
                "flux_AP": flux,
                "flux_AP_err": 0.02 * flux,
                "flux_PSF": flux,
                "flux_PSF_err": 0.02 * flux,
                "x_pix": rng.uniform(0.0, 1000.0, n),
                "y_pix": rng.uniform(0.0, 1000.0, n),
            }
        )
        zp = Zeropoint(_yaml(tmp_path, fit_method="odr"))
        _, params = zp.fit_zeropoint(df)
        ap = params["AP"]
        assert not any(
            "spatially correlated" in f
            for f in ap.get("reliability_flags", [])
        )


class TestBootstrapSE:
    """Seeded bootstrap resampling of the inlier locus gives a
    nonparametric SE of the median - a check on the MAD-based error
    that catches skewed or multimodal loci."""

    def test_recovers_median_se_on_iid(self):
        rng = np.random.default_rng(1)
        vals = rng.normal(0.0, 0.1, 50)
        se = Zeropoint({"zeropoint": {}})._bootstrap_se(
            vals, 512, np.random.default_rng(42)
        )
        # SE(median) = 1.2533*sigma/sqrt(50) ~ 0.018
        assert se == pytest.approx(0.018, abs=0.008)

    def test_deterministic_given_seed(self):
        vals = np.arange(30.0)
        obj = Zeropoint({"zeropoint": {}})
        a = obj._bootstrap_se(vals, 256, np.random.default_rng(7))
        b = obj._bootstrap_se(vals, 256, np.random.default_rng(7))
        assert a == b

    def test_disabled_by_zero_config(self, tmp_path):
        y = _yaml(tmp_path, fit_method="odr")
        y["zeropoint"]["bootstrap_resamples"] = 0
        zp = Zeropoint(y)
        _, params = zp.fit_zeropoint(_catalog())
        assert np.isnan(params["AP"]["bootstrap_se"])
