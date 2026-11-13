"""Lomb-Scargle period analysis of the target light curve.

The preferred input is ``Differential_Photometry_<METHOD>.csv`` (its
relative fluxes are already transparency- and ensemble-corrected); when
absent, calibrated magnitudes are read from ``LightCurve_Output.csv``
and converted to flux.  Per band, a standard-normalized Lomb-Scargle
periodogram is computed over an automatic frequency grid, top peaks and
Baluev false-alarm levels are reported, and a two-panel figure
(periodogram + phased light curve at the best period) plus a peak table
CSV are written.
"""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

from ..core.plotting_utils import (
    apply_autophot_mplstyle,
    get_plot_ext,
    ransac_savefig,
)

log = logging.getLogger(__name__.rsplit(".", 1)[-1])


def _time_column(df):
    """Best available time column: BJD_TDB > mid-exposure MJD > MJD."""
    for col in ("bjd_tdb", "mjd_mid", "mjd"):
        if col in df.columns:
            t = pd.to_numeric(df[col], errors="coerce")
            if t.notna().any():
                return col, t
    return None, None


def _truthy(series):
    """Boolean mask from a column that may hold bools or 'True'/'False' strings."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "t", "yes"))


def _series_from_differential(df, band):
    """(t, rel_flux, err) for one band, restricted to vetted epochs."""
    need = {"filter", "rel_flux", "rel_flux_err", "used"}
    if not need.issubset(df.columns):
        return None
    sub = df[df["filter"].astype(str) == str(band)]
    if "used" in sub.columns:
        sub = sub[_truthy(sub["used"])]
    if "outlier" in sub.columns:
        sub = sub[~_truthy(sub["outlier"])]
    _, t = _time_column(sub)
    if t is None:
        return None
    y = pd.to_numeric(sub["rel_flux"], errors="coerce")
    e = pd.to_numeric(sub["rel_flux_err"], errors="coerce")
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    if ok.sum() < 5:
        return None
    return t[ok].to_numpy(float), y[ok].to_numpy(float), e[ok].to_numpy(float)


def _series_from_lightcurve(df, band, method):
    """(t, flux, err) per band from calibrated magnitudes.

    Flux (normalised to its median) rather than magnitude is used so a
    sinusoidal modulation maps to a sinusoidal signal.
    """
    from .lightcurve import (
        _filter_series_matches_band,
        _normalize_photometry_columns,
        _resolve_band_triplet,
        photometry_filter_series,
    )

    df = _normalize_photometry_columns(df)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    if "mjd" not in df.columns:
        return None
    trip = _resolve_band_triplet(set(df.columns), band, method)
    if trip is None:
        return None
    mag_c, err_c, zp_c = trip
    fser = photometry_filter_series(df)
    if fser is not None and fser.notna().any():
        sub = df[_filter_series_matches_band(band, fser)]
    else:
        sub = df
    t = pd.to_numeric(sub["mjd"], errors="coerce")
    inst = pd.to_numeric(sub[mag_c], errors="coerce")
    inst_e = pd.to_numeric(sub[err_c], errors="coerce")
    zp = pd.to_numeric(sub[zp_c], errors="coerce")
    mag = inst + zp
    ok = np.isfinite(t) & np.isfinite(mag) & np.isfinite(inst_e) & (inst_e > 0)
    if ok.sum() < 5:
        return None
    with np.errstate(invalid="ignore", over="ignore"):
        flux = 10.0 ** (-0.4 * mag[ok])
        ferr = flux * (np.log(10.0) / 2.5) * inst_e[ok]
    good = np.isfinite(flux) & np.isfinite(ferr) & (flux > 0) & (ferr > 0)
    if good.sum() < 5:
        return None
    med = np.median(flux[good])
    return (
        t[ok][good].to_numpy(float),
        (flux[good] / med).to_numpy(float),
        (ferr[good] / med).to_numpy(float),
    )


def load_target_series(csv_path, bands=None, method="PSF", prefer_differential=True):
    """Load per-band (t, y, yerr) series for the target.

    Prefers ``Differential_Photometry_<method>*.csv`` sitting next to
    ``csv_path`` (relative flux, already cleaned); otherwise interprets
    ``csv_path`` itself as the light curve table.  A differential CSV may
    also be passed directly as ``csv_path``.
    """
    method = str(method or "PSF").upper()
    frames = []
    if prefer_differential:
        candidates = []
        base = os.path.basename(str(csv_path))
        if base.startswith("Differential_Photometry"):
            candidates.append(str(csv_path))
        else:
            folder = os.path.dirname(os.path.abspath(str(csv_path))) or "."
            candidates = sorted(
                glob.glob(
                    os.path.join(folder, f"Differential_Photometry_{method}*.csv")
                )
            )
        for cand in candidates:
            try:
                frames.append(pd.read_csv(cand))
            except Exception as exc:
                log.warning("periodogram: could not read %s (%s)", cand, exc)
    series = {}
    if frames:
        df = pd.concat(frames, ignore_index=True)
        # A target renamed between runs can leave several matching CSVs
        # behind; identical epochs must not be measured twice.
        if {"filter", "mjd"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["filter", "mjd"], keep="first")
        if "filter" in df.columns:
            present = [str(b) for b in pd.unique(df["filter"])]
            for band in bands or present:
                s = _series_from_differential(df, band)
                if s is not None:
                    series[str(band)] = s
    if series or not os.path.exists(str(csv_path)):
        return series

    df = pd.read_csv(csv_path)
    from .lightcurve import (
        _normalize_photometry_columns,
        canonical_bands_from_filter_series,
        photometry_filter_series,
    )

    df = _normalize_photometry_columns(df)
    fser = photometry_filter_series(df)
    if fser is not None and fser.notna().any():
        present = canonical_bands_from_filter_series(fser)
    else:
        # No per-row band column: the bands present are encoded in the
        # zp_<band>_<method> column names.  Without this, every generic
        # band guess would resolve to the same shared triplet and the
        # same series would be plotted under many names.
        import re

        present = sorted(
            {
                m.group(1)
                for c in df.columns
                for m in [re.fullmatch(rf"zp_(.+)_{method}", str(c))]
                if m
            }
        ) or ["unk"]
    for band in bands or present:
        try:
            s = _series_from_lightcurve(df, band, method)
        except Exception as exc:
            log.debug("periodogram: band %s skipped (%s)", band, exc)
            continue
        if s is not None:
            series[str(band)] = s
    return series


def lomb_scargle_power(t, y, yerr, min_period=None, max_period=None, samples_per_peak=5):
    """(periods, power, fap_levels, fap_of) for an automatic frequency grid.

    ``power`` uses the standard normalization so the Baluev (2009)
    false-alarm levels in ``fap_levels`` and the ``fap_of(power)``
    evaluator are meaningful.
    """
    from astropy.timeseries import LombScargle

    t = np.asarray(t, dtype=np.float64)
    order = np.argsort(t)
    t, y, yerr = t[order], np.asarray(y)[order], np.asarray(yerr)[order]
    baseline = max(t[-1] - t[0], 1e-6)
    cadence = np.median(np.diff(t)) if len(t) > 1 else baseline
    if min_period is None:
        min_period = max(2.0 * cadence, 1e-4)
    if max_period is None:
        max_period = max(baseline / 2.0, 2.0 * min_period)
    if max_period <= min_period:
        max_period = 2.0 * min_period
    ls = LombScargle(t, y, dy=yerr, normalization="standard")
    freq = ls.autofrequency(
        samples_per_peak=int(samples_per_peak),
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
    )
    power = ls.power(freq)
    try:
        # The Baluev FAP needs the searched frequency range to estimate
        # the effective number of independent frequencies.
        levels = ls.false_alarm_level(
            [0.1, 0.01, 0.001],
            method="baluev",
            samples_per_peak=int(samples_per_peak),
            minimum_frequency=float(freq.min()),
            maximum_frequency=float(freq.max()),
        )
        fap_levels = {"10%": levels[0], "1%": levels[1], "0.1%": levels[2]}

        def fap_of(p):
            return float(
                ls.false_alarm_probability(
                    p,
                    method="baluev",
                    samples_per_peak=int(samples_per_peak),
                    minimum_frequency=float(freq.min()),
                    maximum_frequency=float(freq.max()),
                )
            )

    except Exception:
        fap_levels = {}

        def fap_of(p):
            return np.nan

    return 1.0 / freq, np.asarray(power), fap_levels, fap_of


def select_top_peaks(periods, power, n_peaks=5, baseline=None):
    """Top local maxima, separated by at least ~1/baseline in frequency."""
    p = np.asarray(power)
    if len(p) < 3:
        return []
    loc = np.flatnonzero((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:])) + 1
    if baseline:
        df_min = 1.0 / float(baseline)
    else:
        df_min = 0.0
    picked = []
    freq = 1.0 / np.asarray(periods)
    for idx in loc[np.argsort(-p[loc])]:
        if all(abs(freq[idx] - freq[j]) > df_min for j in picked):
            picked.append(idx)
        if len(picked) >= int(n_peaks):
            break
    return picked


def phase_fold(t, period):
    """Phase in [0, 1) relative to the earliest epoch."""
    return ((np.asarray(t) - np.min(t)) / float(period)) % 1.0


def plot_periodogram(
    band,
    periods,
    power,
    fap_levels,
    peak_idx,
    t,
    y,
    yerr,
    out_path,
    title=None,
):
    """Two-panel product: periodogram and light curve phased at the top peak."""
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
    apply_autophot_mplstyle()
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 6), sharex=False, gridspec_kw={"hspace": 0.35}
    )
    ax = axes[0]
    ax.plot(periods, power, lw=1.0, color="k")
    ax.set_xscale("log")
    for label, level in fap_levels.items():
        ax.axhline(level, ls="--", lw=0.8, color="0.5")
        ax.text(
            0.98,
            level,
            f"FAP {label}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.4",
        )
    if peak_idx:
        ax.plot(
            periods[peak_idx],
            power[peak_idx],
            "o",
            ms=5,
            mfc="none",
            color="tab:red",
        )
        best_p = periods[peak_idx[0]]
        ax.axvline(best_p, ls=":", lw=0.8, color="tab:red")
        ax.set_title(
            (title + " - " if title else "")
            + f"{band}: best period {best_p:.5f} d"
        )
    else:
        ax.set_title((title + " - " if title else "") + f"{band}")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Lomb-Scargle power")

    ax = axes[1]
    if peak_idx:
        phase = phase_fold(t, periods[peak_idx[0]])
        order = np.argsort(phase)
        for k in range(2):
            ax.errorbar(
                phase[order] + k,
                y[order],
                yerr=yerr[order],
                fmt="o",
                ms=3,
                lw=0.8,
                alpha=0.7,
            )
        ax.set_xlim(0, 2)
        ax.set_xlabel(f"Phase (P = {periods[peak_idx[0]]:.5f} d)")
        ax.set_ylabel("Relative flux")
    else:
        ax.text(0.5, 0.5, "no significant peak", ha="center", va="center")
        ax.set_axis_off()
    ransac_savefig(fig, out_path)
    plt.close(fig)
    return out_path


def run_periodogram(
    csv_path,
    outdir=None,
    bands=None,
    method="PSF",
    min_period=None,
    max_period=None,
    samples_per_peak=5,
    top_peaks=5,
    input_yaml=None,
    target_name=None,
):
    """Compute periodograms for every band with a usable series.

    Returns a list of per-band result dicts; writes
    ``Periodogram_<band>.<ext>`` and ``periodogram_<band>.csv`` into
    ``outdir`` (defaults to the light curve's directory).
    """
    outdir = outdir or os.path.dirname(os.path.abspath(str(csv_path))) or "."
    os.makedirs(outdir, exist_ok=True)
    series = load_target_series(csv_path, bands=bands, method=method)
    if not series:
        log.warning("periodogram: no usable light-curve series in %s", csv_path)
        return []
    ext = get_plot_ext(input_yaml)
    results = []
    for band, (t, y, e) in series.items():
        if len(t) < 10:
            log.warning(
                "periodogram: band %s has only %d epochs; skipped.", band, len(t)
            )
            continue
        try:
            periods, power, fap_levels, fap_of = lomb_scargle_power(
                t, y, e,
                min_period=min_period,
                max_period=max_period,
                samples_per_peak=samples_per_peak,
            )
        except Exception as exc:
            log.warning("periodogram: band %s failed (%s)", band, exc)
            continue
        baseline = float(t.max() - t.min())
        peaks = select_top_peaks(
            periods, power, n_peaks=top_peaks, baseline=baseline
        )
        rows = [
            {
                "period_days": float(periods[i]),
                "frequency_per_day": float(1.0 / periods[i]),
                "power": float(power[i]),
                "fap": fap_of(power[i]),
            }
            for i in peaks
        ]
        table = pd.DataFrame(
            rows, columns=["period_days", "frequency_per_day", "power", "fap"]
        )
        csv_out = os.path.join(outdir, f"periodogram_{band}.csv")
        table.to_csv(csv_out, index=False)
        fig_out = os.path.join(outdir, f"Periodogram_{band}{ext}")
        plot_periodogram(
            band,
            periods,
            power,
            fap_levels,
            peaks,
            t,
            y,
            e,
            fig_out,
            title=target_name,
        )
        results.append(
            {
                "band": band,
                "n_epochs": int(len(t)),
                "baseline_days": baseline,
                "peaks": table,
                "figure": fig_out,
                "csv": csv_out,
            }
        )
        if peaks:
            log.info(
                "periodogram: band %s best period %.5f d (FAP %.3g).",
                band,
                periods[peaks[0]],
                fap_of(power[peaks[0]]),
            )
    return results


def main(argv=None):
    """``autophot periodogram`` entry point."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="autophot periodogram",
        description=(
            "Lomb-Scargle period search of the target light curve. "
            "A Differential_Photometry_<method>.csv next to the input is "
            "preferred when present."
        ),
    )
    ap.add_argument("csv", help="LightCurve_Output.csv (or a Differential_Photometry_*.csv)")
    ap.add_argument("--band", action="append", default=None, help="Restrict to band(s)")
    ap.add_argument("--method", default="PSF", choices=["PSF", "AP"])
    ap.add_argument("--min-period", type=float, default=None, help="days")
    ap.add_argument("--max-period", type=float, default=None, help="days")
    ap.add_argument("--samples-per-peak", type=int, default=5)
    ap.add_argument("--top-peaks", type=int, default=5)
    ap.add_argument("--out", default=None, help="Output directory")
    args = ap.parse_args(argv)

    results = run_periodogram(
        args.csv,
        outdir=args.out,
        bands=args.band,
        method=args.method,
        min_period=args.min_period,
        max_period=args.max_period,
        samples_per_peak=args.samples_per_peak,
        top_peaks=args.top_peaks,
    )
    if not results:
        return 1
    for res in results:
        print(
            f"{res['band']}: {res['n_epochs']} epochs over "
            f"{res['baseline_days']:.1f} d -> {res['figure']}"
        )
        if not res["peaks"].empty:
            top = res["peaks"].iloc[0]
            print(
                f"   best period {top['period_days']:.5f} d "
                f"(power {top['power']:.3f}, FAP {top['fap']:.3g})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
