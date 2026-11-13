"""Tests for the Lomb-Scargle period analysis of the light curve."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from autophot.analysis.periodogram import (
    _truthy,
    load_target_series,
    lomb_scargle_power,
    phase_fold,
    run_periodogram,
    select_top_peaks,
)


TRUE_PERIOD = 0.317  # days - safely below any 2x-cadence limit of the grid


def _sine_series(n=220, period=TRUE_PERIOD, amp=0.08, noise=0.01, seed=7):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 60.0, n))
    y = 1.0 + amp * np.sin(2 * np.pi * (t - 4.0) / period)
    y = y + rng.normal(0.0, noise, n)
    e = np.full(n, noise)
    return t, y, e


def test_lomb_scargle_recovers_period():
    t, y, e = _sine_series()
    periods, power, fap_levels, fap_of = lomb_scargle_power(
        t, y, e, min_period=0.1
    )
    best = periods[np.argmax(power)]
    assert best == pytest.approx(TRUE_PERIOD, rel=0.01)
    assert fap_of(power.max()) < 0.01


def test_select_top_peaks_spacing():
    periods = np.geomspace(0.05, 30.0, 4000)
    power = np.exp(-((np.log10(periods) - np.log10(0.3)) ** 2) / 0.001)
    power += 0.8 * np.exp(-((np.log10(periods) - np.log10(2.0)) ** 2) / 0.001)
    idx = select_top_peaks(periods, power, n_peaks=2, baseline=60.0)
    assert len(idx) == 2
    assert periods[idx[0]] == pytest.approx(0.3, rel=0.02)
    assert periods[idx[1]] == pytest.approx(2.0, rel=0.02)


def test_phase_fold_range():
    ph = phase_fold(np.array([0.0, 1.0, 2.0, 2.5]), 1.0)
    np.testing.assert_allclose(ph, [0.0, 0.0, 0.0, 0.5])


def test_truthy_parses_strings():
    s = pd.Series(["True", "False", True, False, "1", "0", None])
    assert _truthy(s).tolist() == [True, False, True, False, True, False, False]


def _write_differential_csv(path, t, y, e, band="r"):
    df = pd.DataFrame(
        {
            "mjd": t - 0.5 / 86400.0,
            "mjd_mid": t,
            "bjd_tdb": t,
            "filter": band,
            "inst_mag": -2.5 * np.log10(y),
            "inst_mag_err": e / y,
            "ensemble_mag": 0.0,
            "diff_mag": -2.5 * np.log10(y),
            "diff_mag_err": e / y,
            "rel_flux": y,
            "rel_flux_err": e,
            "transparency": 1.0,
            "n_ensemble": 5,
            "airmass": 1.2,
            "fwhm": 3.0,
            "systematics": 0.0,
            "jumped": False,
            "outlier": False,
            "used": True,
        }
    )
    df.to_csv(path, index=False)


def test_load_target_series_differential(tmp_path):
    t, y, e = _sine_series()
    csv = tmp_path / "Differential_Photometry_PSF.csv"
    _write_differential_csv(csv, t, y, e)
    series = load_target_series(str(csv), method="PSF")
    assert "r" in series
    tt, yy, ee = series["r"]
    assert len(tt) == len(t)
    np.testing.assert_allclose(tt, t, rtol=1e-9)


def test_differential_used_flag_filters(tmp_path):
    t, y, e = _sine_series(n=60)
    csv = tmp_path / "Differential_Photometry_PSF.csv"
    _write_differential_csv(csv, t, y, e)
    df = pd.read_csv(csv)
    df.loc[df.index[:50], "used"] = False  # leave 10 usable epochs
    df.to_csv(csv, index=False)
    series = load_target_series(str(csv), method="PSF")
    assert len(series["r"][0]) == 10


def test_load_target_series_lightcurve(tmp_path):
    t, y, e = _sine_series(n=120, noise=0.005)
    mag = -2.5 * np.log10(y)
    df = pd.DataFrame(
        {
            "mjd": t,
            "filter": "Sloan_g",
            "mag_psf": mag - 25.0,  # instrumental: apparent - zp
            "mag_psf_err": e / y,
            "zp_g_PSF": 25.0,
        }
    )
    csv = tmp_path / "LightCurve_Output.csv"
    df.to_csv(csv, index=False)
    series = load_target_series(str(csv), method="PSF")
    assert "g" in series
    tt, yy, ee = series["g"]
    np.testing.assert_allclose(yy, y / np.median(y), rtol=1e-6)


def test_run_periodogram_writes_products(tmp_path):
    t, y, e = _sine_series()
    _write_differential_csv(tmp_path / "Differential_Photometry_PSF.csv", t, y, e)
    lc = tmp_path / "LightCurve_Output.csv"
    lc.write_text("mjd\n1.0\n")  # placeholder; differential CSV is preferred
    results = run_periodogram(
        str(lc),
        outdir=str(tmp_path),
        method="PSF",
        min_period=0.1,
        input_yaml={"plot_format": "png"},
    )
    assert len(results) == 1
    res = results[0]
    assert os.path.exists(res["figure"])
    assert os.path.exists(res["csv"])
    top = res["peaks"].iloc[0]
    assert top["period_days"] == pytest.approx(TRUE_PERIOD, rel=0.01)
    assert top["fap"] < 0.05
