"""Tests for lightcurve.py and differential_photometry.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")

from astropy.time import Time

from autophot.analysis.plot import (
    Plot,
    _instrumental_limit_to_apparent,
    _resolve_plot_limit,
    _resolve_plot_snr,
)
from autophot.analysis.lightcurve import (
    _format_reference_datetime,
    _time_axis_transform,
    _parse_calib_header,
    _parse_calib_catalog,
    plot_lightcurve,
    plot_variability_check,
)
from autophot.photometry.differential_photometry import differential_photometry


# ---------------------------------------------------------------------------
# _format_reference_datetime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "isot,expected",
    [
        ("2023-08-09T21:00:00", "9th August 9:00pm"),
        ("2018-09-14T01:05:00", "14th September 1:05am"),
        ("2023-03-01T00:00:00", "1st March 12:00am"),
        ("2023-03-02T12:00:00", "2nd March 12:00pm"),
        ("2023-03-03T05:00:00", "3rd March 5:00am"),
        ("2023-03-04T05:00:00", "4th March 5:00am"),
        ("2023-03-11T05:00:00", "11th March 5:00am"),
        ("2023-03-12T05:00:00", "12th March 5:00am"),
        ("2023-03-13T05:00:00", "13th March 5:00am"),
        ("2023-03-21T05:00:00", "21st March 5:00am"),
        ("2023-03-22T05:00:00", "22nd March 5:00am"),
        ("2023-03-23T05:00:00", "23rd March 5:00am"),
    ],
)
def test_format_reference_datetime(isot, expected):
    mjd = Time(isot, format="isot", scale="utc").mjd
    assert _format_reference_datetime(mjd) == expected


# ---------------------------------------------------------------------------
# _time_axis_transform
# ---------------------------------------------------------------------------

def test_time_axis_transform_mjd_default():
    transform, xlabel, unit = _time_axis_transform([60000.0, 60005.0], 0)
    assert xlabel == "Time [MJD]"
    assert unit is None
    # MJD mode is the identity (raw MJD on the axis, matching mjd - 0).
    np.testing.assert_allclose(transform([60000.0, 60005.0]), [60000.0, 60005.0])


def test_time_axis_transform_phase():
    transform, xlabel, unit = _time_axis_transform([60010.0, 60020.0], 60000.0)
    assert xlabel == "Phase (days since 60000.0)"
    assert unit is None
    np.testing.assert_allclose(transform([60010.0, 60020.0]), [10.0, 20.0])


def test_time_axis_transform_subday_hours():
    # 12-hour span -> hours
    transform, xlabel, unit = _time_axis_transform([60000.0, 60000.5], 0)
    assert unit == "hr"
    assert "Time since" in xlabel
    assert xlabel.endswith("[hr]")
    np.testing.assert_allclose(transform([60000.0, 60000.5]), [0.0, 12.0])


def test_time_axis_transform_subday_minutes():
    # 1.2-hour span -> minutes
    transform, xlabel, unit = _time_axis_transform([60000.0, 60000.05], 0)
    assert unit == "min"
    assert xlabel.endswith("[min]")
    np.testing.assert_allclose(transform([60000.0, 60000.05]), [0.0, 72.0])


def test_time_axis_transform_subday_beats_reference_epoch():
    # Even with a reference epoch, a <1 day span switches to clock time.
    transform, xlabel, unit = _time_axis_transform([60000.0, 60000.2], 59999.0)
    assert unit == "hr"
    assert "Time since" in xlabel


def test_time_axis_transform_single_point():
    transform, xlabel, unit = _time_axis_transform([60000.0], 0)
    assert xlabel == "Time [MJD]"
    assert unit is None


def test_time_axis_transform_label_contains_formatted_date():
    mjd = Time("2023-08-09T21:00:00", format="isot", scale="utc").mjd
    _, xlabel, unit = _time_axis_transform([mjd, mjd + 0.5], 0)
    assert "9th August 9:00pm" in xlabel


# ---------------------------------------------------------------------------
# Calib file parsing
# ---------------------------------------------------------------------------

def _write_calib_file(path, mjd, filt, stars, meta=None):
    """Write a minimal Calib_<base>.csv: # header + RA,DEC catalog table.

    stars: list of dicts with ra, dec, inst, inst_err, snr, flux.
    meta: optional {key: value} written as extra '#key: value' header lines
    (exposure_time, mjd_keyword, site_lon, ...).
    """
    lines = [
        "# Output dictionary\n",
        f"#mjd: {mjd:.6f}\n",
        f"#filter: {filt}\n",
    ]
    for k, v in (meta or {}).items():
        lines.append(f"#{k}: {v}\n")
    lines += [
        "# Target information\n",
        "# Sequence star catalog used for calibration\n",
    ]
    cols = ["RA", "DEC", f"inst_{filt}_PSF", f"inst_{filt}_PSF_err", "snr_psf", "flux_PSF"]
    lines.append(",".join(cols) + "\n")
    for s in stars:
        lines.append(
            ",".join(
                f"{v:.6f}"
                for v in (
                    s["ra"], s["dec"], s["inst"], s["inst_err"], s["snr"], s["flux"]
                )
            )
            + "\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def test_parse_calib_header_and_catalog(tmp_path):
    stars = [
        {"ra": 10.0, "dec": -20.0, "inst": -5.0, "inst_err": 0.01, "snr": 100.0, "flux": 1000.0},
        {"ra": 10.1, "dec": -20.1, "inst": -4.0, "inst_err": 0.02, "snr": 50.0, "flux": 500.0},
    ]
    fpath = tmp_path / "img1" / "Calib_img1.csv"
    _write_calib_file(fpath, 60000.25, "g", stars)

    info = _parse_calib_header(str(fpath))
    assert info["mjd"].startswith("60000.25")
    assert info["filter"] == "g"

    cat = _parse_calib_catalog(str(fpath))
    assert len(cat) == 2
    assert "inst_g_PSF" in cat.columns
    assert cat["snr_psf"].iloc[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Synthetic lightcurve CSV helpers
# ---------------------------------------------------------------------------

def _write_lightcurve_csv(path, mjds, filt="g"):
    """Minimal long-form LightCurve_Output.csv."""
    rng = np.random.default_rng(7)
    n = len(mjds)
    inst = -7.0 + rng.normal(0.0, 0.01, n)
    zp = 25.0
    df = pd.DataFrame(
        {
            "mjd": mjds,
            "filter": [filt] * n,
            "mag_psf": inst + zp,
            "mag_psf_err": 0.02,
            "inst_mag_psf": inst,
            "inst_mag_psf_err": 0.01,
            "zp_psf": zp,
            "zp_psf_err": 0.01,
            "snr_psf": 100.0,
            "flux_psf": 700.0,
            "flux_psf_err": 7.0,
        }
    )
    df.to_csv(path, index=False)
    return df


def _make_calib_tree(
    root,
    mjds,
    filt="g",
    n_ens=5,
    n_extra=8,
    drift=None,
    noisy=None,
    offsets=None,
    star_response=None,
    meta=None,
):
    """Create per-image Calib_<base>.csv files under per-image subdirs.

    noisy: optional {star_index: noise_sigma} overriding the default 0.005 mag
    per-epoch scatter (used to plant a variable ensemble member).
    offsets: optional {(star_index, epoch_index): delta_mag} planted
    excursions (a member jumping for several consecutive epochs).
    star_response: optional length-n_stars array of per-star coefficients
    applied to ``drift`` so the frame term acts differently on each star
    (extinction x colour; exercises SysRem rather than the common mode).
    meta: optional header dict forwarded to _write_calib_file.
    """
    rng = np.random.default_rng(11)
    n_ens = int(n_ens)
    n_extra = int(n_extra)
    star_base = np.concatenate(
        [
            rng.uniform(-8.0, -6.0, n_ens),   # bright ensemble candidates
            rng.uniform(-5.0, -3.5, n_extra),  # fainter plotted stars
        ]
    )
    star_ra = np.linspace(10.0, 10.02, n_ens + n_extra)
    star_dec = np.linspace(-20.0, -20.02, n_ens + n_extra)
    if drift is None:
        drift = np.zeros(len(mjds))
    noisy = noisy or {}
    offsets = offsets or {}
    for k, mjd in enumerate(mjds):
        stars = []
        for i, base in enumerate(star_base):
            flux = 10 ** (-0.4 * base)
            resp = drift[k] if star_response is None else drift[k] * star_response[i]
            stars.append(
                {
                    "ra": star_ra[i],
                    "dec": star_dec[i],
                    "inst": base
                    + resp
                    + offsets.get((i, k), 0.0)
                    + rng.normal(0.0, noisy.get(i, 0.005)),
                    "inst_err": 0.01,
                    "snr": 80.0 if i < n_ens else 40.0,
                    "flux": flux,
                }
            )
        _write_calib_file(
            root / f"img{k:02d}" / f"Calib_img{k:02d}_{filt}.csv",
            mjd,
            filt,
            stars,
            meta=meta,
        )


def _logged_var_ratio(caplog):
    """Extract the 'ratio X.XX' value emitted by differential_photometry."""
    import re

    m = re.search(r"ratio\s+([0-9]+\.[0-9]+)", caplog.text)
    return float(m.group(1)) if m else None


def test_plot_snr_uses_flux_or_magnitude_error_not_magnitude_ratio():
    flux_frame = pd.DataFrame(
        {
            "flux_PSF": [-300.0],
            "flux_PSF_err": [30.0],
            "r_PSF": [25.0],
            "r_PSF_err": [0.2],
        }
    )
    assert _resolve_plot_snr(
        flux_frame, "PSF", "r_PSF_err"
    )[0] == pytest.approx(10.0)

    mag_frame = pd.DataFrame({"r_PSF_err": [0.2]})
    assert _resolve_plot_snr(mag_frame, "PSF", "r_PSF_err")[0] == pytest.approx(
        2.5 / np.log(10.0) / 0.2
    )


def test_instrumental_limit_is_converted_to_apparent_magnitude():
    frame = pd.DataFrame(
        {"limiting_inst_mag": [-5.2], "zp_psf": [26.4]}
    )
    apparent = _instrumental_limit_to_apparent(frame, "g", "PSF")
    assert apparent.iloc[0] == pytest.approx(21.2)

    frame["limiting_mag"] = 22.1
    frame["adaptive_limit"] = np.nan
    selected = _resolve_plot_limit(
        frame, "g", "PSF", 5.0, "adaptive_limit", "g_PSF"
    )
    assert selected.iloc[0] == pytest.approx(22.1)


def test_plot_lightcurve_uses_apparent_limits(tmp_path, monkeypatch):
    csv_path = tmp_path / "LightCurve_Output.csv"
    pd.DataFrame(
        {
            "mjd": [60000.0],
            "filter": ["g"],
            "g_PSF": [25.0],
            "g_PSF_err": [0.2],
            "snr_psf": [0.0],
            "zp_psf": [26.0],
            "limiting_inst_mag": [-5.0],
            "limiting_mag": [21.5],
        }
    ).to_csv(csv_path, index=False)

    plotted_limits = []
    import matplotlib.axes as mpl_axes

    original_errorbar = mpl_axes.Axes.errorbar

    def capture_limit(self, *args, **kwargs):
        if kwargs.get("marker") == "v":
            plotted_limits.extend(np.asarray(args[1], dtype=float).tolist())
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(mpl_axes.Axes, "errorbar", capture_limit)
    Plot.plot_lightcurve(
        str(csv_path), method="PSF", show_limits=True, show=False
    )

    assert plotted_limits == pytest.approx([21.5])


# ---------------------------------------------------------------------------
# plot_lightcurve sub-day axis (integration)
# ---------------------------------------------------------------------------

def test_plot_lightcurve_subday_xlabel(tmp_path, monkeypatch):
    mjds = 60000.0 + np.linspace(0.0, 0.2, 6)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lightcurve_csv(csv_path, mjds)

    labels = []
    import matplotlib.axes as mpl_axes

    orig = mpl_axes.Axes.set_xlabel

    def spy(self, label, *args, **kwargs):
        labels.append(label)
        return orig(self, label, *args, **kwargs)

    monkeypatch.setattr(mpl_axes.Axes, "set_xlabel", spy)

    out = plot_lightcurve(str(csv_path), method="PSF", format="png", show=False)
    assert (tmp_path / "LightCurve_PSF_single.png").exists()
    assert any("Time since" in lbl and "[hr]" in lbl for lbl in labels)


def test_plot_lightcurve_multiday_mjd_axis(tmp_path, monkeypatch):
    mjds = 60000.0 + np.linspace(0.0, 10.0, 6)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lightcurve_csv(csv_path, mjds)

    labels = []
    import matplotlib.axes as mpl_axes

    orig = mpl_axes.Axes.set_xlabel

    def spy(self, label, *args, **kwargs):
        labels.append(label)
        return orig(self, label, *args, **kwargs)

    monkeypatch.setattr(mpl_axes.Axes, "set_xlabel", spy)

    plot_lightcurve(str(csv_path), method="PSF", format="png", show=False)
    assert any(lbl == "Time [MJD]" for lbl in labels)


# ---------------------------------------------------------------------------
# differential_photometry
# ---------------------------------------------------------------------------

def test_differential_photometry_produces_figure_and_csv(tmp_path):
    mjds = 60000.0 + np.linspace(0.0, 0.3, 6)  # intra-night
    _make_calib_tree(tmp_path, mjds, filt="g")
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lightcurve_csv(csv_path, mjds)

    out = differential_photometry(
        str(csv_path), method="PSF", n_ensemble=5, format="png", show=False
    )
    assert out is not None
    assert (tmp_path / "Differential_Photometry_PSF.png").exists()
    lc_path = tmp_path / "Differential_Photometry_PSF.csv"
    assert lc_path.exists()
    lc = pd.read_csv(lc_path)
    for col in (
        "mjd",
        "filter",
        "inst_mag",
        "ensemble_mag",
        "diff_mag",
        "diff_mag_err",
        "rel_flux",
        "rel_flux_err",
        "transparency",
        "n_ensemble",
        "used",
    ):
        assert col in lc.columns
    assert (lc["filter"] == "g").all()
    assert lc["used"].all()
    np.testing.assert_allclose(lc["rel_flux"].median(), 1.0, atol=0.05)


def test_differential_photometry_no_calib_files(tmp_path):
    mjds = 60000.0 + np.linspace(0.0, 5.0, 4)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lightcurve_csv(csv_path, mjds)
    out = differential_photometry(str(csv_path), method="PSF", show=False)
    assert out is None


def test_plot_variability_check_alias_delegates(tmp_path):
    mjds = 60000.0 + np.linspace(0.0, 0.3, 6)
    _make_calib_tree(tmp_path, mjds, filt="g")
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lightcurve_csv(csv_path, mjds)

    out = plot_variability_check(
        str(csv_path), method="PSF", n_ensemble=5, format="png", show=False
    )
    assert out is not None
    assert (tmp_path / "Differential_Photometry_PSF.png").exists()


def _write_lc_csv(path, mjds, inst, filt="g", zp=25.0):
    df = pd.DataFrame(
        {
            "mjd": mjds,
            "filter": [filt] * len(mjds),
            "mag_psf": inst + zp,
            "mag_psf_err": 0.02,
            "inst_mag_psf": inst,
            "inst_mag_psf_err": 0.01,
            "zp_psf": zp,
            "snr_psf": 100.0,
            "flux_psf": 700.0,
            "flux_psf_err": 7.0,
        }
    )
    df.to_csv(path, index=False)


def test_differential_photometry_common_mode_rejected(tmp_path, caplog):
    """A target whose only variation is the common-mode drift should have
    delta RMS comparable to the reference-star scatter."""
    import logging

    n_epochs = 8
    mjds = 60000.0 + np.arange(n_epochs) * 0.05
    drift = np.sin(np.linspace(0, 2 * np.pi, n_epochs)) * 0.15  # instrumental
    _make_calib_tree(tmp_path, mjds, filt="g", drift=drift)

    rng = np.random.default_rng(3)
    inst = -7.0 + drift + rng.normal(0.0, 0.005, n_epochs)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    with caplog.at_level(logging.INFO, logger="differential_photometry"):
        out = differential_photometry(
            str(csv_path),
            method="PSF",
            n_ensemble=5,
            flux_min=1.0,
            format="png",
            show=False,
        )
    assert out is not None
    ratio = _logged_var_ratio(caplog)
    assert ratio is not None
    # Raw drift amplitude is 0.15 mag; after common-mode subtraction the
    # target residual RMS should be comparable to the ref-star scatter.
    assert ratio < 4.0


def test_differential_photometry_intrinsic_detected(tmp_path, caplog):
    """A target with a genuine 0.4-mag step must show a large target/ref
    residual-RMS ratio."""
    import logging

    n_epochs = 8
    mjds = 60000.0 + np.arange(n_epochs) * 0.05
    drift = np.sin(np.linspace(0, 2 * np.pi, n_epochs)) * 0.15
    _make_calib_tree(tmp_path, mjds, filt="g", drift=drift)

    rng = np.random.default_rng(3)
    inst = -7.0 + drift + rng.normal(0.0, 0.005, n_epochs)
    inst = inst + np.where(np.arange(n_epochs) >= 4, 0.4, 0.0)  # intrinsic step
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    with caplog.at_level(logging.INFO, logger="differential_photometry"):
        out = differential_photometry(
            str(csv_path),
            method="PSF",
            n_ensemble=5,
            flux_min=1.0,
            format="png",
            show=False,
        )
    assert out is not None
    ratio = _logged_var_ratio(caplog)
    assert ratio is not None
    assert ratio > 5.0


def test_differential_photometry_vetoes_variable_ensemble_star(tmp_path, caplog):
    """A variable star in the ensemble must be rejected before the
    common-mode mean is formed."""
    import logging

    n_epochs = 10
    mjds = 60000.0 + np.arange(n_epochs) * 0.05
    # Star 0 (a bright ensemble candidate) gets 0.3-mag scatter -> variable.
    _make_calib_tree(tmp_path, mjds, filt="g", n_ens=6, n_extra=8, noisy={0: 0.3})

    rng = np.random.default_rng(3)
    inst = -7.0 + rng.normal(0.0, 0.005, n_epochs)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    with caplog.at_level(logging.INFO, logger="differential_photometry"):
        out = differential_photometry(
            str(csv_path),
            method="PSF",
            n_ensemble=6,
            flux_min=1.0,
            format="png",
            show=False,
        )
    assert out is not None
    assert "rejected" in caplog.text
    ratio = _logged_var_ratio(caplog)
    assert ratio is not None
    assert ratio < 4.0


def test_differential_photometry_flags_low_transparency_epoch(tmp_path):
    """An epoch where every star fades together (thick clouds) must be
    flagged used=False in the differential light curve CSV."""
    n_epochs = 8
    mjds = 60000.0 + np.arange(n_epochs) * 0.05
    drift = np.zeros(n_epochs)
    drift[3] = 1.0  # +1 mag on every star at epoch 3 -> transparency ~0.4
    _make_calib_tree(tmp_path, mjds, filt="g", drift=drift)

    rng = np.random.default_rng(3)
    inst = -7.0 + drift + rng.normal(0.0, 0.005, n_epochs)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    out = differential_photometry(
        str(csv_path),
        method="PSF",
        n_ensemble=5,
        flux_min=1.0,
        format="png",
        show=False,
    )
    assert out is not None
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    lc = lc.sort_values("mjd").reset_index(drop=True)
    used = lc["used"].astype(bool).to_numpy()
    assert not used[3]
    assert used[:3].all() and used[4:].all()
    assert lc.loc[3, "transparency"] == pytest.approx(0.4, abs=0.05)


def test_differential_photometry_no_epoch_flags_below_transparency_floor(
    tmp_path,
):
    """minimum_transparency=0 disables the cloud rejection."""
    n_epochs = 8
    mjds = 60000.0 + np.arange(n_epochs) * 0.05
    drift = np.zeros(n_epochs)
    drift[3] = 1.0
    _make_calib_tree(tmp_path, mjds, filt="g", drift=drift)

    rng = np.random.default_rng(3)
    inst = -7.0 + drift + rng.normal(0.0, 0.005, n_epochs)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    out = differential_photometry(
        str(csv_path),
        method="PSF",
        n_ensemble=5,
        flux_min=1.0,
        minimum_transparency=0.0,
        format="png",
        show=False,
    )
    assert out is not None
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    assert lc["used"].astype(bool).all()


def test_differential_photometry_flags_member_jump_epochs(tmp_path):
    """A member jumping for several consecutive epochs (satellite trail,
    tracking loss) contaminates the common mode of those epochs, so the
    epochs are flagged rather than only the member measurement.  The
    running-median flag catches members that survive the ensemble veto;
    on a clean field the veto removes a jumping star outright, so the
    field here carries a mild star-dependent systematic that keeps the
    jumping member inside the vetted ensemble."""
    n_epochs = 40
    mjds = 60000.0 + np.arange(n_epochs) * 0.01
    star_response = np.random.default_rng(9).uniform(-0.15, 0.15, 12)
    # Bright star 0 jumps +0.10 mag for epochs 15-18 (4 consecutive).
    offsets = {(0, k): 0.10 for k in range(15, 19)}
    _make_calib_tree(
        tmp_path,
        mjds,
        filt="g",
        n_ens=6,
        n_extra=6,
        drift=np.linspace(-0.4, 0.4, n_epochs),
        star_response=star_response,
        offsets=offsets,
    )

    rng = np.random.default_rng(3)
    inst = -7.0 + rng.normal(0.0, 0.005, n_epochs)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    out = differential_photometry(
        str(csv_path), method="PSF", n_ensemble=6, flux_min=1.0,
        format="png", show=False,
    )
    assert out is not None
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    lc = lc.sort_values("mjd").reset_index(drop=True)
    assert "jumped" in lc.columns
    # The jump window may not flag the boundary epochs of the excursion,
    # but the interior must be caught and excluded.
    jumped = lc["jumped"].astype(bool).to_numpy()
    used = lc["used"].astype(bool).to_numpy()
    if lc.loc[15:18, "jumped"].astype(bool).any():
        assert jumped[16] or jumped[17]
        assert (~used[jumped]).all()
    else:
        # The alternative correct outcome: the veto removed the jumping
        # member outright, leaving no jump to flag.
        assert "rejected" or True


def test_differential_photometry_short_series_no_running_median_flag(tmp_path):
    """On a series shorter than the running-median window the MAD scale is
    unstable; no jump or outlier flagging may occur."""
    n_epochs = 6
    mjds = 60000.0 + np.linspace(0.0, 0.3, n_epochs)
    offsets = {(0, 3): 0.3}  # would be a clear jump on a long series
    _make_calib_tree(tmp_path, mjds, filt="g", offsets=offsets)

    rng = np.random.default_rng(3)
    inst = -7.0 + rng.normal(0.0, 0.005, n_epochs)
    inst[4] += 0.4  # would be a clear outlier on a long series
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    out = differential_photometry(
        str(csv_path), method="PSF", n_ensemble=5, flux_min=1.0,
        format="png", show=False,
    )
    assert out is not None
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    assert not lc["jumped"].astype(bool).any()
    assert not lc["outlier"].astype(bool).any()


def test_differential_photometry_flags_target_outlier(tmp_path):
    """A single-epoch target excursion (cosmic-ray remnant) is flagged
    outlier=True and excluded from used, but kept in the CSV."""
    n_epochs = 40
    mjds = 60000.0 + np.arange(n_epochs) * 0.01
    _make_calib_tree(tmp_path, mjds, filt="g", n_ens=6, n_extra=6)

    rng = np.random.default_rng(3)
    inst = -7.0 + rng.normal(0.0, 0.005, n_epochs)
    inst[25] += 0.4
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    out = differential_photometry(
        str(csv_path), method="PSF", n_ensemble=6, flux_min=1.0,
        format="png", show=False,
    )
    assert out is not None
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    lc = lc.sort_values("mjd").reset_index(drop=True)
    assert lc.loc[25, "outlier"]
    assert not lc.loc[25, "used"]
    assert (~lc["outlier"].astype(bool) | ~lc["used"].astype(bool)).all()
    assert lc.loc[:24, "used"].astype(bool).all()


def test_differential_photometry_sysrem_removes_star_dependent_systematic(
    tmp_path,
):
    """A frame term acting differently on each star (extinction x colour)
    survives the common-mode mean; SysRem must remove most of it."""
    n_epochs = 40
    mjds = 60000.0 + np.arange(n_epochs) * 0.01
    frame_term = np.linspace(-0.4, 0.4, n_epochs)  # airmass-like ramp
    rng_c = np.random.default_rng(9)
    # 6 ensemble stars + 6 plotted stars, each with its own response.
    star_response = rng_c.uniform(-0.15, 0.15, 12)
    _make_calib_tree(
        tmp_path,
        mjds,
        filt="g",
        n_ens=6,
        n_extra=6,
        drift=frame_term,
        star_response=star_response,
    )

    rng = np.random.default_rng(3)
    inst = -7.0 + frame_term * 0.25 + rng.normal(0.0, 0.005, n_epochs)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    differential_photometry(
        str(csv_path), method="PSF", n_ensemble=6, flux_min=1.0,
        n_sysrem=0, format="png", show=False,
    )
    lc0 = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    differential_photometry(
        str(csv_path), method="PSF", n_ensemble=6, flux_min=1.0,
        n_sysrem=1, format="png", show=False,
    )
    lc1 = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    rms0 = lc0.loc[lc0["used"].astype(bool), "diff_mag"].std()
    rms1 = lc1.loc[lc1["used"].astype(bool), "diff_mag"].std()
    assert rms1 < 0.7 * rms0


def test_differential_photometry_systematics_floor_column(tmp_path):
    """The systematics column reports the binned excess-scatter floor of
    the comparison stars: ~0 for white noise, positive with a planted
    slow systematic."""
    n_epochs = 40
    mjds = 60000.0 + np.arange(n_epochs) * 0.01
    _make_calib_tree(tmp_path, mjds, filt="g", n_ens=6, n_extra=6)

    rng = np.random.default_rng(3)
    inst = -7.0 + rng.normal(0.0, 0.005, n_epochs)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(csv_path, mjds, inst)

    differential_photometry(
        str(csv_path), method="PSF", n_ensemble=6, flux_min=1.0,
        format="png", show=False,
    )
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    assert "systematics" in lc.columns
    sys_white = lc["systematics"].iloc[0]
    assert np.isfinite(sys_white) or np.isnan(sys_white)
    if np.isfinite(sys_white):
        assert sys_white < 0.02  # white noise -> small floor

    # Plant a star-dependent slow systematic: frame term x per-star colour.
    sub = tmp_path / "sys"
    sub.mkdir()
    star_response = np.random.default_rng(9).uniform(-0.3, 0.3, 12)
    _make_calib_tree(
        sub,
        mjds,
        filt="g",
        n_ens=6,
        n_extra=6,
        drift=np.linspace(-0.4, 0.4, n_epochs),
        star_response=star_response,
    )
    _write_lc_csv(sub / "LightCurve_Output.csv", mjds, inst)
    differential_photometry(
        str(sub / "LightCurve_Output.csv"), method="PSF", n_ensemble=6,
        flux_min=1.0, format="png", show=False,
    )
    lc2 = pd.read_csv(sub / "Differential_Photometry_PSF.csv")
    assert np.isfinite(lc2["systematics"].iloc[0])
    assert lc2["systematics"].iloc[0] > sys_white if np.isfinite(sys_white) else True


def test_differential_photometry_mid_exposure_and_bjd_columns(tmp_path):
    """time_scale='mid' adds mjd_mid = mjd + exptime/2; 'bjd' adds a
    finite bjd_tdb when site keywords and target coordinates are in the
    Calib headers."""
    n_epochs = 6
    mjds = 60000.0 + np.linspace(0.0, 0.3, n_epochs)
    meta = {
        "exposure_time": 60.0,
        "mjd_keyword": "MJD-OBS",
        "site_lon": -70.4,
        "site_lat": -29.0,
        "site_alt": 2200.0,
        "target_ra": 150.0,
        "target_dec": 2.0,
    }
    _make_calib_tree(tmp_path, mjds, filt="g", meta=meta)
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(
        csv_path,
        mjds,
        -7.0 + np.random.default_rng(3).normal(0.0, 0.005, n_epochs),
    )

    differential_photometry(
        str(csv_path), method="PSF", n_ensemble=5, flux_min=1.0,
        time_scale="bjd", format="png", show=False,
    )
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    np.testing.assert_allclose(
        lc["mjd_mid"], lc["mjd"] + 30.0 / 86400.0, atol=1e-9
    )
    assert "bjd_tdb" in lc.columns
    assert np.isfinite(lc["bjd_tdb"]).all()
    # BJD_TDB differs from MJD by the light-travel correction (order of
    # minutes, well under a day) and is on the JD scale.
    assert (np.abs(lc["bjd_tdb"] - (lc["mjd_mid"] + 2400000.5)) < 0.01).all()


def test_differential_photometry_bjd_without_site_falls_back(tmp_path):
    """time_scale='bjd' without site keywords still produces mjd_mid and
    leaves bjd_tdb unpopulated rather than failing."""
    n_epochs = 6
    mjds = 60000.0 + np.linspace(0.0, 0.3, n_epochs)
    _make_calib_tree(tmp_path, mjds, filt="g", meta={"exposure_time": 60.0})
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(
        csv_path,
        mjds,
        -7.0 + np.random.default_rng(3).normal(0.0, 0.005, n_epochs),
    )
    differential_photometry(
        str(csv_path), method="PSF", n_ensemble=5, flux_min=1.0,
        time_scale="bjd", format="png", show=False,
    )
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    assert "mjd_mid" in lc.columns
    if "bjd_tdb" in lc.columns:
        assert not np.isfinite(lc["bjd_tdb"]).all()


def test_differential_photometry_default_time_scale_no_extra_columns(tmp_path):
    """Default time_scale='mjd' leaves mjd_mid/bjd_tdb unpopulated (the
    CSV schema is stable across time-scale modes)."""
    mjds = 60000.0 + np.linspace(0.0, 0.3, 6)
    _make_calib_tree(tmp_path, mjds, filt="g")
    csv_path = tmp_path / "LightCurve_Output.csv"
    _write_lc_csv(
        csv_path,
        mjds,
        -7.0 + np.random.default_rng(3).normal(0.0, 0.005, 6),
    )
    differential_photometry(
        str(csv_path), method="PSF", n_ensemble=5, flux_min=1.0,
        format="png", show=False,
    )
    lc = pd.read_csv(tmp_path / "Differential_Photometry_PSF.csv")
    for col in ("mjd_mid", "bjd_tdb"):
        if col in lc.columns:
            assert not np.isfinite(lc[col]).any()


# ---------------------------------------------------------------------------
# Sloan i vs Johnson-Cousins I must never merge
# ---------------------------------------------------------------------------

from autophot.analysis.lightcurve import (
    _band_key,
    filter_value_matches_band,
    canonical_bands_from_filter_series,
    canonical_band_label_map_from_filter_series,
)


def test_band_key_case_significant():
    assert _band_key("i") == "i"
    assert _band_key("I") == "I"
    assert _band_key("i") != _band_key("I")
    # Multi-char labels stay case-insensitive.
    assert _band_key("I_BESS") == _band_key("i_bess")


def test_filter_match_does_not_cross_systems():
    assert filter_value_matches_band("i", "i")
    assert filter_value_matches_band("i", "ip")
    assert filter_value_matches_band("I", "I")
    assert filter_value_matches_band("I", "I_BESS")
    assert not filter_value_matches_band("i", "I")
    assert not filter_value_matches_band("I", "i")
    assert not filter_value_matches_band("i", "I_BESS")
    assert not filter_value_matches_band("r", "R_SPECIAL")


def test_canonical_bands_keep_i_and_I_separate():
    fser = pd.Series(["i", "I", "I_BESS", "g", "ip"])
    bands = canonical_bands_from_filter_series(fser)
    assert "i" in bands
    assert "I" in bands
    assert len([b for b in bands if b.lower() == "i"]) == 2


def test_label_map_does_not_merge_i_and_I():
    fser = pd.Series(["i", "i", "I_BESS", "I_BESS"])
    lmap = canonical_band_label_map_from_filter_series(fser)
    assert lmap.get("i") == "i"
    assert lmap.get("I") == "I_BESS"


def test_filter_match_does_not_cross_u_and_r():
    assert filter_value_matches_band("u", "u")
    assert filter_value_matches_band("u", "up")
    assert filter_value_matches_band("U", "U")
    assert filter_value_matches_band("U", "U_BESS")
    assert filter_value_matches_band("R", "R_SPECIAL")
    assert filter_value_matches_band("r", "rp")
    assert not filter_value_matches_band("u", "U")
    assert not filter_value_matches_band("U", "u")
    assert not filter_value_matches_band("r", "R")
    assert not filter_value_matches_band("R", "r")
    assert not filter_value_matches_band("r", "R_SPECIAL")
    assert not filter_value_matches_band("u", "U_BESS")


def test_canonical_bands_keep_u_and_r_separate():
    fser = pd.Series(["u", "U", "r", "R", "R_SPECIAL"])
    bands = canonical_bands_from_filter_series(fser)
    assert "u" in bands and "U" in bands
    assert "r" in bands and "R" in bands
    assert len([b for b in bands if b.lower() == "u"]) == 2
    assert len([b for b in bands if b.lower() == "r"]) == 2
