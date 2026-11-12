"""Differential photometry against an ensemble of field reference stars.

For each band present in the lightcurve output, this module builds a
per-epoch common-mode signal from a vetted ensemble of bright sequence
stars (read from the per-image ``Calib_*.csv`` catalogs), removes it from
the target and from the remaining reference stars, writes the normalized
differential light curve to CSV, and produces a diagnostic figure.

Ensemble construction follows standard practice for high-precision
relative photometry:

* members are inverse-variance weighted in the per-epoch mean, so bright
  low-noise stars dominate the common-mode estimate;
* candidate members are vetted by a leave-one-out noise test: each star's
  residual against the ensemble formed without it is compared to the
  scatter expected from its own measurement errors, and the most
  discrepant member is dropped while it remains a >2x outlier relative
  to the group -- this removes variable or blended stars that a static
  brightness cut cannot;
* within each epoch, member measurements that deviate strongly from the
  epoch mean are clipped (cosmic rays, satellite crossings);
* epochs where the ensemble flux falls below a fraction of its median
  (thick clouds) or too few members survive clipping are flagged
  ``used=False`` in the output rather than silently dropped.

The figure shows, per band, the instrumental-drift comparison, a
diagnostics panel (ensemble transparency, airmass and FWHM when the
catalog headers carry them), and the differential residuals of the
reference stars and the target.
"""

import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from functions import set_size, border_msg
from plotting_utils import (
    apply_autophot_mplstyle,
    get_marker_size,
    get_plot_color,
    interactive_display_allowed,
)

from lightcurve import (
    BAND_COLORS,
    _filter_series_matches_band,
    _nearest_epoch,
    _normalize_photometry_columns,
    _parse_calib_catalog,
    _parse_calib_header,
    _resolve_band_triplet,
    _time_axis_transform,
    canonical_bands_from_filter_series,
    filter_value_matches_band,
    photometry_filter_series,
)


# -----------------------------------------------------------------------------
# Weighted ensemble helpers
# -----------------------------------------------------------------------------


class _ignore_nan_warnings(warnings.catch_warnings):
    """Silence 'All-NaN slice' RuntimeWarnings where NaN results are handled."""

    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        return self


def _star_matrix(cat_all, star_ids, column, epochs):
    """Pivot per-epoch per-star ``column`` values into an (E x N) matrix."""
    sub = cat_all[cat_all["star_id"].isin(star_ids)]
    pt = sub.pivot_table(
        index="mjd", columns="star_id", values=column, aggfunc="first"
    )
    pt = pt.reindex(index=list(epochs), columns=list(star_ids))
    return pt.to_numpy(dtype=float)


def _weighted_mean_nan(vals, wts):
    """Row-wise weighted mean ignoring NaN; returns (mean, sum_of_weights)."""
    valid = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    w = np.where(valid, wts, 0.0)
    v = np.where(valid, vals, 0.0)
    wtot = w.sum(axis=1)
    mean = np.divide(
        (w * v).sum(axis=1),
        wtot,
        out=np.full(vals.shape[0], np.nan),
        where=wtot > 0,
    )
    return mean, wtot


def _leave_one_out_veto(dm, wts, err_med, factor=2.0, min_keep=3):
    """Drop candidate ensemble stars by a leave-one-out noise test.

    Each active member is compared against the weighted ensemble mean
    formed without it.  The ratio of measured residual RMS to expected
    noise (measurement error plus the formal error of the leave-one-out
    mean) is scored per member; the worst-scoring member is removed while
    its score exceeds ``factor`` times the median score.  When measurement
    errors are unavailable the expected noise falls back to the median
    residual RMS, so the test degrades to a pure relative-scatter cut.

    Returns (keep_mask, vetoed_count).
    """
    n = dm.shape[1]
    active = np.ones(n, dtype=bool)
    vetoed = 0
    while active.sum() > max(min_keep, 3):
        idx = np.where(active)[0]
        D = dm[:, idx]
        W = wts[:, idx]
        valid = np.isfinite(D)
        W = np.where(valid, W, 0.0)
        D0 = np.where(valid, D, 0.0)
        s = (W * D0).sum(axis=1)
        wtot = W.sum(axis=1)
        numer = s[:, None] - W * D0
        denom = wtot[:, None] - W
        loo_mean = np.divide(
            numer,
            denom,
            out=np.full_like(numer, np.nan),
            where=(denom > 0) & valid,
        )
        R = D - loo_mean
        with np.errstate(invalid="ignore"), _ignore_nan_warnings():
            R = R - np.nanmean(R, axis=0)
            rms = np.nanstd(R, axis=0)
            loo_err = np.sqrt(np.where(denom > 0, 1.0 / denom, np.nan))
            med_loo_err = np.nanmedian(loo_err, axis=0)
        exp = np.sqrt(err_med[idx] ** 2 + med_loo_err**2)
        if not np.isfinite(exp).all():
            with _ignore_nan_warnings():
                rms_med = np.nanmedian(rms[np.isfinite(rms)])
            exp = np.where(np.isfinite(exp), exp, rms_med)
        score = np.divide(
            rms, exp, out=np.full_like(rms, np.nan), where=exp > 0
        )
        finite = np.isfinite(score)
        if not finite.any():
            break
        with _ignore_nan_warnings():
            med = np.nanmedian(score[finite])
        if not np.isfinite(med) or med <= 0:
            break
        worst_local = int(np.nanargmax(score))
        if score[worst_local] <= factor * med:
            break
        active[idx[worst_local]] = False
        vetoed += 1
    return active, vetoed


def _ensemble_epoch_stats(dm, wts, err, sigma_clip=5.0, min_members=3):
    """Weighted common-mode per epoch with outlier clipping.

    ``dm`` is the (E x N) matrix of member magnitudes already centred on
    each star's own mean.  Returns a dict with the common-mode ``c``, its
    error ``c_err`` (formal error floored at the across-member scatter /
    sqrt(n), so member disagreement cannot produce a silently small
    error), the across-member ``scatter``, per-epoch member counts, the
    clipped-measurement keep-mask, and ensemble ``transparency`` per
    epoch (weighted mean relative flux, normalized to its median).
    """
    valid = np.isfinite(dm)
    W = np.where(valid & np.isfinite(wts) & (wts > 0), wts, 0.0)
    D0 = np.where(valid, dm, 0.0)

    for _ in range(3):
        c, wtot = _weighted_mean_nan(D0, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            se_formal = np.sqrt(np.where(wtot > 0, 1.0 / wtot, np.nan))
            # W is zeroed wherever dm is invalid, so masking with W > 0
            # keeps NaN measurements out of the epoch scatter.
            resid = np.where(W > 0, dm - c[:, None], 0.0)
            var = (W * resid**2).sum(axis=1) / np.where(
                wtot > 0, wtot, np.nan
            )
            scatter = np.sqrt(var)
            n_mem = (W > 0).sum(axis=1)
            c_err = np.maximum(
                se_formal, scatter / np.sqrt(np.clip(n_mem, 1, None))
            )
        if sigma_clip is None or sigma_clip <= 0:
            break
        sig = np.sqrt(
            np.where(np.isfinite(err), err, np.nan) ** 2
            + c_err[:, None] ** 2
        )
        # A member with no error estimate is clipped against the pooled
        # epoch scatter so it cannot silently dominate the mean.
        sig = np.where(
            np.isfinite(sig),
            sig,
            np.where(np.isfinite(scatter), scatter, np.inf)[:, None],
        )
        resid_abs = np.where(W > 0, dm - c[:, None], np.nan)
        outlier = (W > 0) & (np.abs(resid_abs) > sigma_clip * sig)
        if not outlier.any():
            break
        W = np.where(outlier, 0.0, W)

    n_mem = (W > 0).sum(axis=1)
    with np.errstate(invalid="ignore"):
        rel_flux = np.where(W > 0, 10.0 ** (-0.4 * dm), np.nan)
        transp, _ = _weighted_mean_nan(rel_flux, W)
    with _ignore_nan_warnings():
        med_t = np.nanmedian(transp[np.isfinite(transp)])
    transp_norm = (
        transp / med_t if np.isfinite(med_t) and med_t > 0 else transp
    )

    return {
        "c": c,
        "c_err": c_err,
        "scatter": scatter,
        "n_mem": n_mem,
        "keep": W > 0,
        "transparency": transp_norm,
        "min_members_ok": n_mem >= int(min_members),
    }


def _red_noise_beta(resid, n_per_bin):
    """Ratio of binned to white-noise-expected RMS (>1 means correlated)."""
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if n_per_bin < 2 or r.size < 2 * n_per_bin:
        return np.nan
    nb = r.size // n_per_bin
    binned = r[: nb * n_per_bin].reshape(nb, n_per_bin).mean(axis=1)
    white = r.std(ddof=1) / np.sqrt(n_per_bin)
    if white <= 0 or not np.isfinite(white):
        return np.nan
    return float(binned.std(ddof=1) / white)


def _p2p_scatter(resid):
    """Point-to-point scatter: RMS of successive differences / sqrt(2).

    Insensitive to slow trends, so it tracks the high-frequency noise of
    the differential curve rather than any real variability signal.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return np.nan
    return float(np.diff(r).std(ddof=1) / np.sqrt(2.0))


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def differential_photometry(
    output_file,
    method="PSF",
    snr_min=25.0,
    flux_min=100.0,
    flux_max=None,
    min_epoch_frac=0.8,
    min_target_flux_frac=0.1,
    n_ensemble=20,
    n_ref_plot=25,
    max_plot_err=0.5,
    ensemble_reject_factor=2.0,
    epoch_sigma_clip=5.0,
    min_ensemble_per_epoch=3,
    minimum_transparency=0.7,
    n_bin=0,
    show_drift_panel=True,
    show_diagnostics=True,
    write_csv=True,
    calib_dir=None,
    format="png",
    dpi=150,
    show=False,
    target_name=None,
):
    """Differential photometry of the target against a reference ensemble.

    For each band present in the lightcurve output, reads the per-image
    ``Calib_*.csv`` sequence-star catalogs (searched recursively under the
    directory containing ``output_file``), builds the per-epoch weighted
    common-mode signal of a vetted reference ensemble, and subtracts it
    from every reference star and from the target.

    Compared to a plain variability check this performs full relative
    photometry: inverse-variance weighted ensemble means, iterative
    leave-one-out rejection of variable ensemble members, per-epoch
    outlier clipping, and transparency/coverage epoch flags.  The
    normalized differential light curve (relative flux, differential
    magnitude, diagnostics and per-epoch ``used`` flags) is written to
    ``Differential_Photometry_<METHOD>.csv`` next to the figure.

    Parameters
    ----------
    output_file : str
        Path to the concatenated photometry CSV (e.g. ``LightCurve_Output.csv``).
    method : str
        Photometry method whose instrumental columns are used ('PSF' or 'AP').
    snr_min, flux_min : float
        Minimum mean S/N and mean flux for a catalog star to join the
        reference ensemble (relaxed automatically if too few stars qualify).
    flux_max : float or None
        Optional maximum flux cut (saturation guard); None disables it.
    min_epoch_frac : float
        Fraction of epochs a star must be present in to qualify (default 0.8;
        progressively relaxed if fewer than 3 stars qualify).
    min_target_flux_frac : float
        Ensemble candidates must have mean flux >= this fraction of the
        target's median flux (default 0.1; inactive when the target flux
        column is missing).
    n_ensemble : int
        Maximum number of brightest vetted stars in the ensemble mean.
    n_ref_plot : int
        Number of additional (non-ensemble) stars drawn as the grey cloud.
    max_plot_err : float
        Maximum target residual error plotted (mag).
    ensemble_reject_factor : float
        Leave-one-out veto: members scoring above this factor times the
        median score are dropped (default 2.0).
    epoch_sigma_clip : float
        Per-epoch clip: member measurements this many sigma from the
        epoch weighted mean are excluded (cosmic rays, satellites).
    min_ensemble_per_epoch : int
        Epochs with fewer surviving ensemble members are flagged unused.
    minimum_transparency : float
        Epochs whose ensemble relative flux falls below this fraction of
        the median are flagged unused (cloud rejection; 0 disables).
    n_bin : int
        If >= 2, overlay binned target points of this many consecutive
        used epochs on the residual panel.
    show_drift_panel : bool
        Show the per-band instrumental-drift comparison panel.
    show_diagnostics : bool
        Show the per-band transparency/airmass/FWHM diagnostics panel.
    write_csv : bool
        Write the differential light curve CSV next to the figure.
    calib_dir : str or None
        Directory searched recursively for ``Calib_*.csv`` catalogs.
        Defaults to the directory containing ``output_file``.
    format, dpi, show : plotting controls
        Same conventions as :func:`lightcurve.plot_lightcurve`.
    target_name : str or None
        Optional label; appended to the output filenames.

    Returns
    -------
    str or None
        Path to the saved figure, or None when insufficient data.
    """
    log = logging.getLogger(__name__)
    log.info(border_msg("Differential photometry"))
    apply_autophot_mplstyle()
    if show and interactive_display_allowed():
        import matplotlib

        current_backend = str(plt.get_backend()).lower()
        if "agg" in current_backend:
            for backend in ("QtAgg", "TkAgg"):
                try:
                    matplotlib.use(backend, force=True)
                    break
                except Exception:
                    continue

    try:
        data = pd.read_csv(output_file)
    except Exception as exc:
        log.error(
            "differential_photometry: cannot read '%s': %s", output_file, exc
        )
        return None
    data = _normalize_photometry_columns(data)
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated()].copy()
    if "mjd" not in data.columns:
        log.warning(
            "differential_photometry: no 'mjd' column in '%s'", output_file
        )
        return None

    save_path = os.path.dirname(os.path.abspath(output_file))
    m_low = str(method).strip().lower()
    method_u = str(method).strip().upper()

    # Discover bands, mirroring plot_lightcurve (long-form via `filter`
    # column, else wide-format per-band triplets).
    long_form = (
        f"mag_{m_low}" in data.columns and f"mag_{m_low}_err" in data.columns
    )
    fser = photometry_filter_series(data)
    bands = []
    if long_form and fser is not None and fser.notna().any():
        bands = canonical_bands_from_filter_series(fser)
    if not bands:
        used_triplets = set()
        for b in "FSDNAuUBgcVwrRoEiIzyYJHKWQ":
            trip = _resolve_band_triplet(set(data.columns), b, method)
            if trip is not None and trip not in used_triplets:
                bands.append(b)
                used_triplets.add(trip)
    if not bands:
        log.info(
            "differential_photometry: no photometric bands found in '%s'.",
            output_file,
        )
        return None

    # Locate per-image Calib catalogs under the reduced output tree.
    search_dir = os.path.abspath(calib_dir) if calib_dir else save_path
    calib_files = sorted(
        glob.glob(os.path.join(search_dir, "**", "Calib_*.csv"), recursive=True)
    )
    if not calib_files:
        log.info(
            "differential_photometry: no Calib_*.csv files found under '%s'.",
            search_dir,
        )
        return None

    def _meta_float(info, *keys):
        for k in keys:
            try:
                v = float(info.get(k, "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                return v
        return np.nan

    calib_meta = []
    for fpath in calib_files:
        info = _parse_calib_header(fpath)
        mjd = _meta_float(info, "mjd")
        if not np.isfinite(mjd):
            continue
        calib_meta.append(
            {
                "path": fpath,
                "mjd": mjd,
                "filter": str(info.get("filter", "")).strip(),
                "airmass": _meta_float(info, "airmass"),
                "fwhm": _meta_float(info, "image_fwhm", "fwhm"),
            }
        )
    if not calib_meta:
        log.info("differential_photometry: no usable Calib headers found.")
        return None

    band_payloads = []
    all_mjds = []
    csv_rows = []

    for band in bands:
        band_files = [
            m for m in calib_meta if filter_value_matches_band(band, m["filter"])
        ]
        if len(band_files) < 2:
            continue

        frames = []
        for meta in band_files:
            cat = _parse_calib_catalog(meta["path"])
            if cat.empty or "RA" not in cat.columns or "DEC" not in cat.columns:
                continue
            raw_filt = meta["filter"]
            cmap = {str(c).lower(): c for c in cat.columns}
            inst_col = cmap.get(f"inst_{raw_filt}_{method_u}".lower())
            if inst_col is None:
                continue
            inst_err_col = cmap.get(f"{inst_col}_err".lower())
            snr_col = cmap.get(f"snr_{m_low}", cmap.get("snr"))
            flux_col = cmap.get(f"flux_{m_low}")
            sub = pd.DataFrame(
                {
                    "ra": pd.to_numeric(cat["RA"], errors="coerce"),
                    "dec": pd.to_numeric(cat["DEC"], errors="coerce"),
                    "mjd": float(meta["mjd"]),
                    "inst": pd.to_numeric(cat[inst_col], errors="coerce"),
                    "inst_err": (
                        pd.to_numeric(cat[inst_err_col], errors="coerce")
                        if inst_err_col
                        else np.nan
                    ),
                    "snr": (
                        pd.to_numeric(cat[snr_col], errors="coerce")
                        if snr_col
                        else np.nan
                    ),
                    "flux": (
                        pd.to_numeric(cat[flux_col], errors="coerce")
                        if flux_col
                        else np.nan
                    ),
                }
            )
            sub = sub[
                np.isfinite(sub["inst"])
                & np.isfinite(sub["ra"])
                & np.isfinite(sub["dec"])
            ]
            if not sub.empty:
                frames.append(sub)
        if not frames:
            continue

        cat_all = pd.concat(frames, ignore_index=True)
        # Star identity key: sequence stars share catalog coordinates, so a
        # rounded (ra, dec) pair is a stable identifier across epochs.  A
        # string key keeps pandas pivot/reindex unambiguous (tuple labels
        # can be read as multi-level keys).
        cat_all["star_id"] = (
            cat_all["ra"].round(4).astype(str)
            + ","
            + cat_all["dec"].round(4).astype(str)
        )
        epochs = np.sort(cat_all["mjd"].unique())
        n_epochs = len(epochs)
        if n_epochs < 2:
            continue

        stats = (
            cat_all.groupby("star_id")
            .agg(
                n_mjd=("mjd", "nunique"),
                mean_snr=("snr", "mean"),
                mean_flux=("flux", "mean"),
                max_flux=("flux", "max"),
            )
            .reset_index()
        )

        # Target instrumental magnitudes for this band (needed early for the
        # relative-brightness candidate cut).
        if long_form and fser is not None and fser.notna().any():
            tmask = _filter_series_matches_band(band, fser)
            tgt = data[tmask.fillna(False)].copy()
        else:
            tgt = data.copy()
        cmap_t = {str(c).lower(): c for c in tgt.columns}
        inst_t_col = None
        for cand in (
            f"inst_mag_{m_low}",
            f"inst_{band}_{method_u}",
            f"inst_{band}_{m_low}",
        ):
            if cand in tgt.columns:
                inst_t_col = cand
                break
            if cand.lower() in cmap_t:
                inst_t_col = cmap_t[cand.lower()]
                break
        if inst_t_col is not None:
            tgt["inst"] = pd.to_numeric(tgt[inst_t_col], errors="coerce")
            err_cand = f"{inst_t_col}_err"
            tgt["inst_err"] = (
                pd.to_numeric(tgt[err_cand], errors="coerce")
                if err_cand in tgt.columns
                else np.nan
            )
        else:
            # Derive instrumental mag = calibrated mag - zeropoint.
            trip = _resolve_band_triplet(set(tgt.columns), band, method)
            if trip is None:
                continue
            mcol, ecol, zcol = trip
            tgt["inst"] = pd.to_numeric(tgt[mcol], errors="coerce") - pd.to_numeric(
                tgt[zcol], errors="coerce"
            )
            tgt["inst_err"] = pd.to_numeric(tgt[ecol], errors="coerce")
        tgt = tgt[
            np.isfinite(tgt["inst"])
            & np.isfinite(pd.to_numeric(tgt["mjd"], errors="coerce"))
        ]
        if tgt.empty:
            continue
        tgt["mjd"] = pd.to_numeric(tgt["mjd"], errors="coerce")

        # Relative-brightness cut mirrors the comparison-star selection used
        # for differential light curves: members much fainter than the target
        # only add noise to the common-mode mean.
        tgt_flux_col = next(
            (
                c
                for c in (
                    f"flux_{m_low}",
                    f"flux_{method_u}",
                    "flux",
                )
                if c in tgt.columns
            ),
            None,
        )
        tgt_med_flux = np.nan
        if tgt_flux_col is not None:
            tgt_med_flux = float(
                pd.to_numeric(tgt[tgt_flux_col], errors="coerce").median()
            )
        flux_floor_rel = np.nan
        if np.isfinite(tgt_med_flux) and min_target_flux_frac:
            flux_floor_rel = min_target_flux_frac * tgt_med_flux

        # Reference-ensemble candidate selection with progressive relaxation
        # so that sparse/noisy fields still produce a diagnostic.
        sel = None
        frac_ladder = [min_epoch_frac] + [
            f for f in (0.6, 0.4, 0.0) if f < min_epoch_frac
        ]
        for frac in frac_ladder:
            for snr_cut in (snr_min, snr_min / 2.0, 0.0):
                need = max(2, int(np.ceil(frac * n_epochs)))
                good = stats[
                    (stats["n_mjd"] >= need)
                    & (stats["mean_snr"].fillna(np.inf) > snr_cut)
                    & (stats["mean_flux"].fillna(np.inf) > flux_min)
                ]
                if np.isfinite(flux_floor_rel):
                    good = good[good["mean_flux"] > flux_floor_rel]
                if flux_max is not None:
                    good = good[good["max_flux"] < flux_max]
                if len(good) >= 3:
                    sel = good.sort_values("mean_flux", ascending=False)
                    break
            if sel is not None:
                break
        if sel is None or sel.empty:
            log.info(
                "differential_photometry: band %s - fewer than 3 usable "
                "reference stars; skipping.",
                band,
            )
            continue

        cand_ids = sel["star_id"].tolist()

        # Demeaned magnitude matrix (E x N_candidates): each star is centred
        # on its own weighted mean so intrinsic brightness differences do not
        # contaminate the common-mode estimate.
        dm_raw = _star_matrix(cat_all, cand_ids, "inst", epochs)
        err_raw = _star_matrix(cat_all, cand_ids, "inst_err", epochs)
        err_floor = np.where(
            np.isfinite(err_raw) & (err_raw > 1e-3), err_raw, np.nan
        )
        with _ignore_nan_warnings():
            col_med_err = np.nanmedian(err_floor, axis=0)
            global_med_err = np.nanmedian(err_floor)
        err_filled = np.where(np.isfinite(err_floor), err_floor, col_med_err)
        err_filled = np.where(
            np.isfinite(err_filled), err_filled, global_med_err
        )
        if not np.isfinite(err_filled).any():
            # No measurement errors at all: uniform weights, and the veto
            # falls back to a relative-scatter comparison.
            wts = np.where(np.isfinite(dm_raw), 1.0, 0.0)
            err_use = np.full_like(dm_raw, np.nan)
            err_med = np.full(len(cand_ids), np.nan)
        else:
            err_use = np.where(np.isfinite(err_filled), err_filled, np.nan)
            wts = np.where(np.isfinite(err_use), 1.0 / err_use**2, 0.0)
            err_med = np.where(
                np.isfinite(col_med_err), col_med_err, global_med_err
            )

        valid0 = np.isfinite(dm_raw)
        w0 = np.where(valid0, wts, 0.0)
        d0 = np.where(valid0, dm_raw, 0.0)
        den0 = w0.sum(axis=0)
        offset = np.divide(
            (w0 * d0).sum(axis=0),
            den0,
            out=np.full(dm_raw.shape[1], np.nan),
            where=den0 > 0,
        )
        with _ignore_nan_warnings():
            col_mean = np.nanmean(dm_raw, axis=0)
        offset = np.where(np.isfinite(offset), offset, col_mean)
        dm = dm_raw - offset

        # Leave-one-out veto of variable/discrepant ensemble candidates.
        keep_mask, n_vetoed = _leave_one_out_veto(
            dm,
            wts,
            err_med,
            factor=ensemble_reject_factor,
            min_keep=3,
        )
        if n_vetoed:
            log.info(
                "differential_photometry: band %s - rejected %d discrepant "
                "ensemble member(s) by leave-one-out noise test.",
                band,
                n_vetoed,
            )
        # Cap at the brightest n_ensemble survivors (cand_ids is already
        # sorted by mean flux descending).
        ensemble_ids = [
            sid for sid, k in zip(cand_ids, keep_mask) if k
        ][:n_ensemble]
        if len(ensemble_ids) < 3:
            log.info(
                "differential_photometry: band %s - fewer than 3 ensemble "
                "stars after vetting; skipping.",
                band,
            )
            continue

        ens_idx = [cand_ids.index(sid) for sid in ensemble_ids]
        ens = _ensemble_epoch_stats(
            dm[:, ens_idx],
            wts[:, ens_idx],
            err_use[:, ens_idx],
            sigma_clip=epoch_sigma_clip,
            min_members=min_ensemble_per_epoch,
        )
        with _ignore_nan_warnings():
            mean_offset = float(np.nanmean(offset[ens_idx]))
        ens_c = ens["c"]
        ens_mean_inst = ens_c + mean_offset  # back on the instrumental scale
        used_epoch = (
            np.isfinite(ens_c)
            & ens["min_members_ok"]
            & (
                ens["transparency"] >= minimum_transparency
                if minimum_transparency
                else np.ones(n_epochs, dtype=bool)
            )
        )
        n_rej = int((~used_epoch).sum())
        if n_rej:
            log.info(
                "differential_photometry: band %s - %d epoch(s) flagged "
                "unused (transparency < %.2f or < %d ensemble members).",
                band,
                n_rej,
                minimum_transparency,
                min_ensemble_per_epoch,
            )

        ens_epoch = pd.DataFrame(
            {
                "mjd": epochs,
                "ens_mean": ens_mean_inst,
                "ens_std": ens["c_err"],
                "ens_n": ens["n_mem"],
                "transparency": ens["transparency"],
                "used": used_epoch,
            }
        )
        # Per-epoch observing diagnostics from the Calib headers.
        meta_by_mjd = {m["mjd"]: m for m in band_files}
        ens_epoch["airmass"] = [
            meta_by_mjd.get(m, {}).get("airmass", np.nan) for m in epochs
        ]
        ens_epoch["fwhm"] = [
            meta_by_mjd.get(m, {}).get("fwhm", np.nan) for m in epochs
        ]

        # Brightest qualifying stars not in the ensemble form the plotted
        # reference cloud (independent of the common-mode correction).
        plot_ids = set(
            sel[~sel["star_id"].isin(ensemble_ids)]
            .head(n_ref_plot)["star_id"]
        )
        if not plot_ids:
            # Fall back to plotting ensemble members themselves: their
            # residuals are then computed against a mean containing
            # themselves, so the plotted scatter understates the true
            # reference scatter (variance biased low by ~1-1/N).
            plot_ids = set(ensemble_ids)
            log.info(
                "differential_photometry: band %s - reference cloud contains "
                "ensemble members; its scatter is not independent of the "
                "correction.",
                band,
            )
        ref_dm = _star_matrix(cat_all, sorted(plot_ids), "inst", epochs)
        with _ignore_nan_warnings():
            ref_dm = ref_dm - np.nanmean(ref_dm, axis=0)
            ref_delta = ref_dm - ens_c[:, None]
            ref_delta = ref_delta - np.nanmean(ref_delta, axis=0)
        ref = pd.DataFrame(
            {
                "mjd": np.repeat(epochs, ref_delta.shape[1]),
                "delta": ref_delta.ravel(),
            }
        )
        ref = ref[np.isfinite(ref["delta"])]
        ref_epoch = (
            ref.groupby("mjd")["delta"].agg(["mean", "std"]).reset_index()
            if not ref.empty
            else pd.DataFrame(columns=["mjd", "mean", "std"])
        )

        # Map target rows to catalog epochs by nearest MJD (headers and the
        # lightcurve CSV may differ at the float-rounding level). Tolerance is
        # half the minimum epoch gap, capped at ~86 s.
        if n_epochs > 1:
            min_gap = float(np.min(np.diff(epochs)))
            epoch_tol = min(1e-3, min_gap / 2.0)
        else:
            epoch_tol = 1e-3
        tgt["epoch"] = _nearest_epoch(tgt["mjd"], epochs, epoch_tol)
        tgt = tgt[np.isfinite(tgt["epoch"])]
        ep_index = {m: i for i, m in enumerate(epochs)}
        tgt["ep_i"] = tgt["epoch"].map(ep_index)
        tgt["ens_mean"] = ens_mean_inst[tgt["ep_i"].to_numpy(dtype=int)]
        tgt["ens_std"] = ens["c_err"][tgt["ep_i"].to_numpy(dtype=int)]
        tgt["ens_n"] = ens["n_mem"][tgt["ep_i"].to_numpy(dtype=int)]
        tgt["transparency"] = ens["transparency"][
            tgt["ep_i"].to_numpy(dtype=int)
        ]
        tgt["used"] = used_epoch[tgt["ep_i"].to_numpy(dtype=int)]
        tgt["diff"] = tgt["inst"] - tgt["ens_mean"]
        # A NaN instrumental error means the uncertainty is unknown, not
        # zero - propagate it so the max_plot_err gate drops the point
        # instead of treating it as perfectly measured.
        tgt["delta_err"] = np.sqrt(
            tgt["inst_err"] ** 2 + tgt["ens_std"].fillna(0.0) ** 2
        )
        if max_plot_err is not None and max_plot_err > 0:
            n_before = len(tgt)
            tgt = tgt[tgt["delta_err"] <= max_plot_err]
            if len(tgt) < n_before:
                log.info(
                    "differential_photometry: band %s - %d target point(s) "
                    "excluded (delta_err > %.2f mag).",
                    band,
                    n_before - len(tgt),
                    max_plot_err,
                )
        if tgt.empty:
            continue
        # Centre on the used subset so rejected epochs do not drag the
        # effective zero point of the plotted/product residuals.
        used_t = tgt["used"].to_numpy(dtype=bool)
        diff_ref = (
            float(tgt.loc[used_t, "diff"].mean())
            if used_t.any()
            else float(tgt["diff"].mean())
        )
        tgt["delta"] = tgt["diff"] - diff_ref
        # Normalized relative flux, unity at the median of the used epochs.
        tgt["rel_flux"] = 10.0 ** (-0.4 * tgt["delta"])
        tgt["rel_flux_err"] = (
            np.log(10.0) / 2.5 * tgt["rel_flux"] * tgt["delta_err"]
        )
        tgt = tgt.sort_values("mjd")

        ref_rms = (
            float(np.nanstd(ref_delta[used_epoch, :]))
            if used_epoch.any() and ref_delta.size
            else np.nan
        )
        tgt_used = tgt[tgt["used"]]
        tgt_rms = (
            float(np.nanstd(tgt_used["delta"])) if len(tgt_used) else np.nan
        )
        var_ratio = (
            tgt_rms / ref_rms
            if np.isfinite(ref_rms) and ref_rms > 0
            else np.nan
        )
        p2p = _p2p_scatter(tgt_used["delta"].to_numpy())
        n_per_bin = max(2, int(np.sqrt(max(len(tgt_used), 1))))
        beta = _red_noise_beta(tgt_used["delta"].to_numpy(), n_per_bin)
        n_ref_plotted = int(ref_delta.shape[1])
        log.info(
            "differential_photometry: band %s - %d ensemble stars, %d "
            "plotted refs, %d epochs (%d rejected); target RMS %.4f mag, "
            "ref RMS %.4f mag, ratio %.2f, p2p %.1f mmag, beta %.2f.",
            band,
            len(ensemble_ids),
            n_ref_plotted,
            n_epochs,
            n_rej,
            tgt_rms,
            ref_rms,
            var_ratio,
            (p2p * 1e3) if np.isfinite(p2p) else np.nan,
            beta,
        )

        # Long-form differential light curve rows for the CSV product.
        for row in tgt.itertuples():
            ei = int(row.ep_i)
            meta = meta_by_mjd.get(epochs[ei], {})
            csv_rows.append(
                {
                    "mjd": row.mjd,
                    "filter": band,
                    "inst_mag": row.inst,
                    "inst_mag_err": row.inst_err,
                    "ensemble_mag": row.ens_mean,
                    "diff_mag": row.diff,
                    "diff_mag_err": row.delta_err,
                    "rel_flux": row.rel_flux,
                    "rel_flux_err": row.rel_flux_err,
                    "transparency": row.transparency,
                    "n_ensemble": int(row.ens_n),
                    "airmass": meta.get("airmass", np.nan),
                    "fwhm": meta.get("fwhm", np.nan),
                    "used": bool(row.used),
                }
            )

        band_payloads.append(
            {
                "band": band,
                "ens_epoch": ens_epoch,
                "ref": ref,
                "ref_epoch": ref_epoch,
                "tgt": tgt,
                "n_ensemble": len(ensemble_ids),
                "n_ref": n_ref_plotted,
                "var_ratio": var_ratio,
                "p2p": p2p,
                "beta": beta,
            }
        )
        all_mjds.extend(tgt["mjd"].tolist())
        all_mjds.extend(ens_epoch["mjd"].tolist())

    if not band_payloads:
        log.info(
            "differential_photometry: insufficient catalog/target overlap; "
            "no figure produced."
        )
        return None

    _tag = f"Differential_Photometry_{method_u}"
    if target_name:
        _safe_tn = (
            str(target_name)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        _tag = f"{_tag}_{_safe_tn}"

    if write_csv and csv_rows:
        csv_path = os.path.join(save_path, f"{_tag}.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        log.info("differential_photometry: saved '%s'", csv_path)

    x_transform, xlabel, subday_unit = _time_axis_transform(all_mjds, 0)

    def _panel_kinds(payload):
        kinds = []
        if show_drift_panel:
            kinds.append("drift")
        if show_diagnostics:
            kinds.append("diag")
        kinds.append("resid")
        return kinds

    panel_plan = [_panel_kinds(p) for p in band_payloads]
    n_rows = sum(len(k) for k in panel_plan)
    height_ratios = [
        {"drift": 1.0, "diag": 0.55, "resid": 1.0}[k]
        for kinds in panel_plan
        for k in kinds
    ]
    width_in = set_size(540, aspect=1)[0]
    height_in = set_size(505, aspect=1)[1] * 0.62 * sum(
        1.0 if k != "diag" else 0.55 for kinds in panel_plan for k in kinds
    )

    fig = plt.figure(figsize=(width_in, height_in))
    gs = GridSpec(
        n_rows, 1, figure=fig, height_ratios=height_ratios, hspace=0.12
    )
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]
    for a in axes[1:]:
        a.sharex(axes[0])

    row = 0
    for i, payload in enumerate(band_payloads):
        band = payload["band"]
        c = BAND_COLORS.get(band, BAND_COLORS.get(str(band).lower(), "k"))
        tgt = payload["tgt"]
        ens_epoch = payload["ens_epoch"]
        ref = payload["ref"]
        ref_epoch = payload["ref_epoch"]

        for kind in panel_plan[i]:
            if kind == "drift":
                ax_drift = axes[row]
                row += 1
                # Centre the ensemble curve over the epochs the target
                # actually covers so the two curves are directly comparable.
                _tgt_ep = ens_epoch["mjd"].isin(set(tgt["epoch"]))
                _ens_ref = ens_epoch.loc[_tgt_ep, "ens_mean"]
                _ens_c0 = (
                    float(_ens_ref.mean())
                    if len(_ens_ref)
                    else float(ens_epoch["ens_mean"].mean())
                )
                ens_c_dr = ens_epoch["ens_mean"] - _ens_c0
                ax_drift.plot(
                    x_transform(ens_epoch["mjd"]),
                    ens_c_dr,
                    "D-",
                    color="dimgrey",
                    markersize=3,
                    lw=0.8,
                    label=f"Ensemble mean ({payload['n_ensemble']} stars)",
                    zorder=2,
                )
                tgt_c = tgt["inst"] - tgt["inst"].mean()
                ax_drift.errorbar(
                    x_transform(tgt["mjd"]),
                    tgt_c,
                    yerr=tgt["inst_err"],
                    fmt="o",
                    color=c,
                    ecolor=c,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=get_marker_size("medium"),
                    capsize=get_marker_size("medium") / 4,
                    lw=0.5,
                    label=f"{band} target",
                    zorder=3,
                )
                ax_drift.set_ylabel("Delta Instrumental Magnitude [mag]")
                ax_drift.invert_yaxis()
                ax_drift.grid(
                    True, which="major", alpha=0.35, linestyle="-", linewidth=0.5
                )
                ax_drift.minorticks_on()
                ax_drift.legend(loc="best", frameon=False, fontsize=7)

            elif kind == "diag":
                ax_diag = axes[row]
                row += 1
                xe = x_transform(ens_epoch["mjd"])
                tr = ens_epoch["transparency"]
                ax_diag.plot(
                    xe,
                    tr,
                    "s-",
                    color="darkslategray",
                    markersize=3,
                    lw=0.8,
                    label="Ensemble transparency",
                    zorder=3,
                )
                if minimum_transparency:
                    ax_diag.axhline(
                        minimum_transparency,
                        color="firebrick",
                        lw=0.8,
                        ls=":",
                        label=f"Rejection limit ({minimum_transparency:.2f})",
                        zorder=2,
                    )
                bad = ~ens_epoch["used"].to_numpy(dtype=bool)
                if bad.any():
                    ax_diag.plot(
                        xe[bad],
                        tr[bad],
                        "x",
                        color="red",
                        markersize=5,
                        label="Rejected epoch",
                        zorder=4,
                    )
                ax_diag.set_ylabel("Transparency")
                ax_diag.set_ylim(
                    bottom=min(
                        0.0,
                        float(np.nanmin(tr[np.isfinite(tr)])) - 0.05
                        if np.isfinite(tr).any()
                        else 0.0,
                    )
                )
                ax_diag.grid(
                    True, which="major", alpha=0.35, linestyle="-", linewidth=0.5
                )
                ax_diag.minorticks_on()
                # Airmass/FWHM on a twin axis only when the Calib headers
                # carried them.
                twin_axes = []
                am = pd.to_numeric(ens_epoch["airmass"], errors="coerce")
                if np.isfinite(am).any():
                    ax_am = ax_diag.twinx()
                    twin_axes.append(ax_am)
                    ax_am.plot(
                        xe,
                        am,
                        "^--",
                        color="steelblue",
                        markersize=2.5,
                        lw=0.6,
                        alpha=0.8,
                        label="Airmass",
                        zorder=1,
                    )
                    ax_am.set_ylabel("Airmass", color="steelblue", fontsize=8)
                    ax_am.tick_params(axis="y", colors="steelblue", labelsize=7)
                fw = pd.to_numeric(ens_epoch["fwhm"], errors="coerce")
                if np.isfinite(fw).any():
                    ax_fw = ax_diag.twinx()
                    if twin_axes:
                        ax_fw.spines["right"].set_position(("outward", 40))
                    twin_axes.append(ax_fw)
                    ax_fw.plot(
                        xe,
                        fw,
                        "v-.",
                        color="darkorchid",
                        markersize=2.5,
                        lw=0.6,
                        alpha=0.8,
                        label="FWHM [pix]",
                        zorder=1,
                    )
                    ax_fw.set_ylabel(
                        "FWHM [pix]", color="darkorchid", fontsize=8
                    )
                    ax_fw.tick_params(axis="y", colors="darkorchid", labelsize=7)
                h1, l1 = ax_diag.get_legend_handles_labels()
                for _ax_t in twin_axes:
                    _h, _l = _ax_t.get_legend_handles_labels()
                    h1 += _h
                    l1 += _l
                ax_diag.legend(h1, l1, loc="best", frameon=False, fontsize=7)

            else:  # resid
                ax = axes[row]
                row += 1
                ax.axhline(0, color="black", lw=0.5, ls="--", zorder=1)
                if not ref.empty:
                    ax.scatter(
                        x_transform(ref["mjd"]),
                        ref["delta"],
                        s=6,
                        c="lightgrey",
                        alpha=0.4,
                        edgecolors="none",
                        label=f"Reference stars ({payload['n_ref']})",
                        zorder=0,
                    )
                if not ref_epoch.empty:
                    ax.errorbar(
                        x_transform(ref_epoch["mjd"]),
                        ref_epoch["mean"],
                        yerr=ref_epoch["std"],
                        fmt="D",
                        color="whitesmoke",
                        ecolor="dimgrey",
                        markersize=2,
                        capsize=2 / 4,
                        lw=0.5,
                        markeredgecolor="dimgrey",
                        markeredgewidth=0.5,
                        label="Per-epoch mean +/- std",
                        zorder=2,
                    )
                used_t = tgt["used"].to_numpy(dtype=bool)
                tu = tgt[used_t]
                ax.errorbar(
                    x_transform(tu["mjd"]),
                    tu["delta"],
                    yerr=tu["delta_err"],
                    fmt="o",
                    color=c,
                    ecolor=c,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=get_marker_size("medium"),
                    capsize=get_marker_size("medium") / 4,
                    lw=0.5,
                    label=f"{band} target",
                    zorder=5,
                )
                trej = tgt[~used_t]
                if not trej.empty:
                    ax.plot(
                        x_transform(trej["mjd"]),
                        trej["delta"],
                        "x",
                        color="red",
                        markersize=6,
                        label="Rejected epoch",
                        zorder=6,
                    )
                if n_bin and n_bin >= 2 and len(tu) >= 2 * n_bin:
                    tb = tu.sort_values("mjd")
                    nbb = len(tb) // n_bin
                    bx, by, be = [], [], []
                    for bidx in range(nbb):
                        chunk = tb.iloc[bidx * n_bin : (bidx + 1) * n_bin]
                        bx.append(chunk["mjd"].mean())
                        by.append(chunk["delta"].mean())
                        be.append(
                            chunk["delta"].std(ddof=1) / np.sqrt(len(chunk))
                            if len(chunk) > 1
                            else 0.0
                        )
                    ax.errorbar(
                        x_transform(np.asarray(bx)),
                        np.asarray(by),
                        yerr=np.asarray(be),
                        fmt="s",
                        color="black",
                        markersize=3.5,
                        capsize=2,
                        lw=0.8,
                        label=f"Binned x{n_bin}",
                        zorder=4,
                    )
                ax.set_ylabel("Residual Instrumental Magnitude [mag]")
                # Fit the y-limits to the target residuals (+/- errors) with
                # a small margin, extended to zero so the residual reference
                # line stays in view; reference-star outliers cannot stretch
                # the axis.
                _tgt_d = np.asarray(tgt["delta"], dtype=float)
                _tgt_e = np.asarray(tgt["delta_err"], dtype=float)
                _y_lo = np.nanmin(_tgt_d - _tgt_e)
                _y_hi = np.nanmax(_tgt_d + _tgt_e)
                if not (np.isfinite(_y_lo) and np.isfinite(_y_hi)):
                    _y_lo, _y_hi = -0.05, 0.05
                _y_lo = min(_y_lo, 0.0)
                _y_hi = max(_y_hi, 0.0)
                _pad = max(0.05 * (_y_hi - _y_lo), 0.01)
                ax.set_ylim(_y_hi + _pad, _y_lo - _pad)
                ax.grid(
                    True, which="major", alpha=0.35, linestyle="-", linewidth=0.5
                )
                ax.minorticks_on()
                note_lines = []
                if np.isfinite(payload["var_ratio"]):
                    note_lines.append(
                        f"target/ref RMS = {payload['var_ratio']:.2f}"
                    )
                if np.isfinite(payload["p2p"]):
                    note_lines.append(f"p2p = {payload['p2p'] * 1e3:.0f} mmag")
                if np.isfinite(payload["beta"]):
                    note_lines.append(f"beta = {payload['beta']:.2f}")
                if note_lines:
                    ax.text(
                        0.02,
                        0.03,
                        "\n".join(note_lines),
                        transform=ax.transAxes,
                        va="bottom",
                        ha="left",
                        fontsize=7,
                        bbox=dict(
                            facecolor="white",
                            alpha=0.9,
                            edgecolor="black",
                            linewidth=0.5,
                        ),
                    )
                ax.legend(loc="best", frameon=False, fontsize=7)

    axes[-1].set_xlabel(xlabel)
    for a in axes[:-1]:
        a.tick_params(axis="x", labelbottom=False)
    if subday_unit == "min":
        for a in axes:
            a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    if target_name:
        fig.suptitle(str(target_name), fontsize=11, y=0.995)

    outpath = os.path.join(save_path, f"{_tag}.{format}")
    save_kw = dict(dpi=dpi) if format.lower() != "pdf" else {}
    fig.savefig(
        outpath,
        **save_kw,
        bbox_inches="tight",
        facecolor=get_plot_color("figure_facecolor"),
    )
    log.info("differential_photometry: saved '%s'", outpath)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outpath
