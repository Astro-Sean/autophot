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
* epochs where a member jumps several sigma from its running median
  (events lasting several frames that per-epoch clipping cannot see) are
  flagged, as are epochs where the ensemble flux falls below a fraction
  of its median (thick clouds) or too few members survive clipping --
  all reported as ``used=False`` in the output rather than dropped;
* target points deviating strongly from their own running median are
  flagged ``outlier=True`` and excluded from ``used`` (single-epoch
  excursions such as cosmic-ray remnants);
* optionally, residual systematics common to many stars (extinction x
  color terms, position-dependent flat errors as the field drifts) are
  removed with SysRem (Tamuz, Mazeh & Zucker 2005); frame coefficients
  are fit only from well-behaved stars so a genuinely variable target
  does not bias the systematic model.

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
from scipy.ndimage import median_filter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..core.functions import set_size, border_msg
from ..core.plotting_utils import (
    apply_autophot_mplstyle,
    get_marker_size,
    get_plot_color,
    interactive_display_allowed,
)

from ..analysis.lightcurve import (
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


def _p2p_scatter_robust(resid):
    """MAD-based point-to-point scatter (insensitive to slow trends)."""
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 6:
        return np.nan
    d = np.diff(r)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def _leave_one_out_residuals(dm, wts):
    """(E x N) residual of each member against the weighted mean of the others."""
    valid = np.isfinite(dm) & np.isfinite(wts) & (wts > 0)
    w = np.where(valid, wts, 0.0)
    d0 = np.where(valid, dm, 0.0)
    s = (w * d0).sum(axis=1)
    wt = w.sum(axis=1)
    denom = wt[:, None] - w
    loo = np.divide(
        s[:, None] - w * d0,
        denom,
        out=np.full_like(dm, np.nan),
        where=denom > 0,
    )
    return dm - loo


def _running_median_mask(resid, usable, window, sigma):
    """Points deviating from a per-series running median by > sigma x MAD scale.

    A running median follows slow stellar and instrumental drift, so a
    sharp excursion stands out even when it persists for several epochs
    (a regime per-epoch clipping cannot see).  ``usable`` restricts both
    the flagged points and the scale estimate; filled values stand in
    for missing points so gaps do not distort the running median.
    """
    resid = np.asarray(resid, dtype=float)
    ok = np.asarray(usable, dtype=bool) & np.isfinite(resid)
    out = np.zeros(resid.shape, dtype=bool)
    win = int(window) if window else 0
    if win < 3 or resid.shape[0] < 3 or sigma <= 0:
        return out
    win = min(win, resid.shape[0])
    if resid.ndim == 1:
        resid = resid[:, None]
        ok = ok[:, None]
        out = out[:, None]
    for j in range(resid.shape[1]):
        # Require a full window of usable points: on short series the MAD
        # scale of |r - running median| is unreliable (the running median
        # nearly interpolates), which would flag plain noise points.
        if np.count_nonzero(ok[:, j]) < window:
            continue
        r = resid[:, j]
        filled = np.where(np.isfinite(r), r, np.nanmedian(r[ok[:, j]]))
        running = median_filter(filled, size=win, mode="nearest")
        dev = np.abs(filled - running)
        scale = 1.4826 * np.median(dev[ok[:, j]])
        if np.isfinite(scale) and scale > 0:
            out[:, j] = ok[:, j] & (dev > sigma * scale)
    return out.squeeze()


def _comparison_systematics(resid, usable, bin_size=10):
    """Median excess scatter of binned comparison residuals beyond white noise.

    For each member, consecutive residuals are averaged in blocks of
    ``bin_size`` epochs; the scatter of the bin means is compared to the
    level expected from the point-to-point (white) noise.  The median
    excess over members is the level of slow systematics (flat-field
    drift, patchy extinction, residual colour terms) that a differential
    light curve of this field carries.  Returns scalar mag, NaN when
    underdetermined.
    """
    resid = np.asarray(resid, dtype=float)
    b = int(bin_size)
    if b < 2 or resid.shape[0] < 3 * b:
        return np.nan
    excess = []
    for j in range(resid.shape[1]):
        ok = np.asarray(usable)[:, j] if np.ndim(usable) == 2 else usable
        r = resid[:, j][ok & np.isfinite(resid[:, j])]
        n_bins = r.size // b
        p2p = _p2p_scatter_robust(r)
        if n_bins < 3 or not np.isfinite(p2p):
            continue
        binned = r[: n_bins * b].reshape(n_bins, b).mean(axis=1)
        excess.append(
            np.sqrt(max(float(np.std(binned)) ** 2 - p2p**2 / b, 0.0))
        )
    return float(np.median(excess)) if excess else np.nan


def _sysrem(resid, errs, n_components, star_weights, epoch_weights=None,
            iterations=15, clip=4.0):
    """SysRem (Tamuz, Mazeh & Zucker 2005) on an (E x S) residual matrix.

    Removes effects common to many stars, each modelled as c_star x
    a_epoch (e.g. extinction x colour, flat-field drift).  Frame
    coefficients ``a`` are fit only from stars with ``star_weights`` > 0
    and from inlier points, so a deeply variable star cannot imprint its
    signal on the systematic model shared by all stars.  ``epoch_weights``
    masks rejected epochs out of the fit.  Returns (cleaned residuals,
    list of (c_star, a_epoch) components).
    """
    finite = np.isfinite(resid) & np.isfinite(errs) & (errs > 0)
    weights = np.where(finite, 1.0 / np.where(finite, errs, 1.0) ** 2, 0.0)
    if epoch_weights is not None:
        weights = weights * np.asarray(epoch_weights, dtype=float)[:, None]
    cleaned = np.where(finite, resid, 0.0)
    with _ignore_nan_warnings():
        col_med = np.nanmedian(
            np.where(finite, resid, np.nan), axis=0, keepdims=True
        )
        star_scatter = 1.4826 * np.nanmedian(
            np.abs(np.where(finite, resid, np.nan) - col_med), axis=0
        )
    inliers = np.abs(cleaned) < clip * np.where(
        np.isfinite(star_scatter), star_scatter, np.inf
    )[None, :]
    frame_weights = weights * np.asarray(star_weights, dtype=float)[None, :] * inliers
    components = []
    for _ in range(int(n_components)):
        a = np.ones(resid.shape[0])
        for _ in range(iterations):
            c = ((cleaned * weights) * a[:, None]).sum(axis=0) / np.maximum(
                (weights * a[:, None] ** 2).sum(axis=0), 1e-30
            )
            a = ((cleaned * frame_weights) * c[None, :]).sum(axis=1) / np.maximum(
                (frame_weights * c[None, :] ** 2).sum(axis=1), 1e-30
            )
        # The residuals are column-centred, so a frame term's mean only
        # re-expresses the centring; removing it keeps the component a
        # pure epoch pattern and stops the mean leaking into the c fit.
        a = a - np.mean(a)
        c = ((cleaned * weights * inliers) * a[:, None]).sum(axis=0) / np.maximum(
            ((weights * inliers) * a[:, None] ** 2).sum(axis=0), 1e-30
        )
        cleaned = cleaned - np.outer(a, c)
        components.append((c, a))
    return np.where(finite, cleaned, np.nan), components


def _sysrem_subtract(delta, delta_err, A_rows, used):
    """Subtract the fitted SysRem frame terms from a light curve.

    ``A_rows`` is the per-point frame-term design matrix (one row per
    light-curve point, one column per SysRem component).  The star's
    response ``c`` is solved by weighted least squares over its used
    points: the frame terms are known, so the model is linear.
    """
    A = np.asarray(A_rows, dtype=float)
    if A.ndim != 2 or A.shape[1] == 0:
        return delta
    ok = np.isfinite(delta) & np.isfinite(delta_err) & (delta_err > 0) & used
    if np.count_nonzero(ok) <= A.shape[1]:
        return delta
    n_comp = A.shape[1]
    # A constant column absorbs the curve's mean level: without it the
    # response fit conflates the instrumental offset with the frame
    # terms whenever a term has a nonzero mean over the used points.
    A = np.column_stack([A, np.ones(A.shape[0])])
    w = 1.0 / delta_err[ok] ** 2
    Aw = A[ok] * np.sqrt(w)[:, None]
    c_t, *_ = np.linalg.lstsq(Aw, delta[ok] * np.sqrt(w), rcond=None)
    return delta - A[:, :n_comp] @ c_t[:n_comp]


def _observatory_location(observatory, meta_list):
    """EarthLocation from an explicit argument or Calib-header site keys.

    ``observatory`` may be an ``EarthLocation`` or a (lon_deg, lat_deg,
    alt_m) tuple; when None the first Calib header carrying
    ``site_lon``/``site_lat`` is used.  Returns None when no location is
    available.
    """
    if observatory is not None:
        from astropy.coordinates import EarthLocation

        if isinstance(observatory, EarthLocation):
            return observatory
        try:
            lon, lat, *rest = np.atleast_1d(observatory)
            alt = rest[0] if rest else 0.0
            return EarthLocation.from_geodetic(
                lon=float(lon), lat=float(lat), height=float(alt)
            )
        except Exception:
            return None
    lon = lat = alt = np.nan
    for m in meta_list:
        try:
            lon = float(m.get("site_lon", "nan"))
            lat = float(m.get("site_lat", "nan"))
            alt = float(m.get("site_alt", "0") or 0.0)
        except (TypeError, ValueError):
            continue
        if np.isfinite(lon) and np.isfinite(lat):
            break
    if not (np.isfinite(lon) and np.isfinite(lat)):
        return None
    try:
        from astropy.coordinates import EarthLocation

        return EarthLocation.from_geodetic(
            lon=lon, lat=lat, height=alt if np.isfinite(alt) else 0.0
        )
    except Exception:
        return None


def _mid_bjd_times(mjds, exposures, location, target_radec):
    """(mjd_mid, bjd_tdb) arrays; bjd_tdb is None without location+coords.

    ``mjd_mid`` adds half the exposure to the stored start-of-exposure
    MJD (epochs lacking an exposure time pass through unchanged).  The
    barycentric correction needs both the observatory location and the
    target coordinates.
    """
    mid = np.asarray(mjds, dtype=float)
    # Header values arrive as strings; coerce junk to NaN rather than fail.
    exp = pd.to_numeric(np.atleast_1d(exposures), errors="coerce").astype(float)
    mid = mid + np.where(np.isfinite(exp) & (exp > 0), 0.5 * exp / 86400.0, 0.0)
    bjd = None
    if location is not None and target_radec is not None:
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u

        coord = SkyCoord(
            ra=float(target_radec[0]) * u.deg,
            dec=float(target_radec[1]) * u.deg,
        )
        t = Time(mid, format="mjd", scale="utc", location=location)
        bjd = (t.tdb + t.light_travel_time(coord)).jd
    return mid, bjd


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
    jump_sigma=5.0,
    jump_window=9,
    outlier_sigma=5.0,
    outlier_window=11,
    n_sysrem=0,
    systematics_bin=10,
    time_scale="mjd",
    observatory=None,
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
    jump_sigma, jump_window : float, int
        Epochs where an ensemble member deviates more than ``jump_sigma``
        times its MAD scale from a running median of ``jump_window``
        epochs are flagged unused (multi-frame events such as satellites
        or tracking loss; <=0 or window <3 disables).
    outlier_sigma, outlier_window : float, int
        Target points deviating more than ``outlier_sigma`` times the MAD
        scale from a running median of ``outlier_window`` epochs are
        flagged ``outlier=True`` and excluded from ``used`` (<=0 or
        window <3 disables).
    n_sysrem : int
        Number of SysRem systematic components removed from the light
        curves (0 disables; typical 1-3 for multi-night data).
    systematics_bin : int
        Epochs per bin for the comparison-systematics floor diagnostic.
    time_scale : str
        'mjd' (default, stored MJD unchanged), 'mid' (add half the
        exposure time; assumes stored MJD is start-of-exposure), or
        'bjd' (mid-exposure barycentric-corrected BJD_TDB; needs the
        observatory location and target coordinates).
    observatory : EarthLocation, tuple or None
        Observatory location for time_scale='bjd': an EarthLocation or a
        (lon_deg, lat_deg, alt_m) tuple.  When None, the ``site_lon`` /
        ``site_lat`` / ``site_alt`` keys written into newer Calib_*.csv
        headers are used.
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
    log = logging.getLogger(__name__.rsplit(".", 1)[-1])
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
                "exposure_time": _meta_float(info, "exposure_time"),
                "site_lon": _meta_float(info, "site_lon"),
                "site_lat": _meta_float(info, "site_lat"),
                "site_alt": _meta_float(info, "site_alt"),
                "mjd_keyword": str(info.get("mjd_keyword", "")).strip(),
                "target_ra": _meta_float(info, "target_ra"),
                "target_dec": _meta_float(info, "target_dec"),
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

        # Running-median jump rejection: a member deviating from its own
        # running median for several consecutive epochs (satellite trails,
        # tracking loss, uncaught variability) contaminates the common
        # mode of every epoch it touches, so the epoch is flagged rather
        # than only the member measurement.  Members already clipped by
        # the per-epoch sigma test are not re-flagged.
        ens_w = np.where(ens["keep"], wts[:, ens_idx], 0.0)
        loo_resid = _leave_one_out_residuals(dm[:, ens_idx], ens_w)
        jumped = (
            _running_median_mask(
                loo_resid, ens["keep"], window=jump_window, sigma=jump_sigma
            ).any(axis=1)
            if loo_resid.size
            else np.zeros(n_epochs, dtype=bool)
        )
        n_jump = int(jumped.sum())
        if n_jump:
            used_epoch = used_epoch & ~jumped
            log.info(
                "differential_photometry: band %s - %d epoch(s) flagged "
                "unused (ensemble member jumped > %.1f sigma from its "
                "running median).",
                band,
                n_jump,
                jump_sigma,
            )
        n_rej = int((~used_epoch).sum())
        if n_rej:
            log.info(
                "differential_photometry: band %s - %d epoch(s) flagged "
                "unused (transparency < %.2f, < %d ensemble members, or "
                "member jump).",
                band,
                n_rej,
                minimum_transparency,
                min_ensemble_per_epoch,
            )

        # Slow-systematics floor: the median excess scatter of binned
        # member residuals beyond what the point-to-point noise predicts.
        sys_floor = _comparison_systematics(
            loo_resid,
            ens["keep"] & used_epoch[:, None],
            bin_size=systematics_bin,
        )

        # SysRem: remove residual systematics shared by many stars that a
        # single common-mode mean cannot (extinction x colour, flat-field
        # errors as the field drifts).  Frame coefficients are fit only
        # from well-behaved stars so a variable star or the target does
        # not leak into the systematic model.
        components = []
        sysrem_model = None
        n_sysrem_use = int(n_sysrem or 0)
        if n_sysrem_use > 0:
            resid_all = dm - ens_c[:, None]
            with _ignore_nan_warnings():
                cover = np.mean(np.isfinite(resid_all[used_epoch]), axis=0)
                col_med = np.nanmedian(resid_all, axis=0, keepdims=True)
                init_scat = 1.4826 * np.nanmedian(
                    np.abs(resid_all - col_med), axis=0
                )
                exp_err = np.nanmedian(err_use, axis=0)
            star_ok = (
                keep_mask
                & (cover > 0.5)
                & np.isfinite(init_scat)
                & (
                    ~np.isfinite(exp_err)
                    | (init_scat < 3.0 * np.maximum(exp_err, 1e-4))
                )
            )
            if np.count_nonzero(star_ok) < 5:
                log.info(
                    "differential_photometry: band %s - fewer than 5 "
                    "well-measured reference stars; SysRem skipped.",
                    band,
                )
                n_sysrem_use = 0
            else:
                resid_clean, components = _sysrem(
                    resid_all,
                    err_use,
                    n_sysrem_use,
                    star_ok.astype(float),
                    epoch_weights=used_epoch.astype(float),
                )
                sysrem_model = resid_all - resid_clean
                n_sysrem_use = len(components)
                log.info(
                    "differential_photometry: band %s - SysRem removed %d "
                    "component(s) fit from %d stars.",
                    band,
                    n_sysrem_use,
                    int(np.count_nonzero(star_ok)),
                )

        ens_epoch = pd.DataFrame(
            {
                "mjd": epochs,
                "ens_mean": ens_mean_inst,
                "ens_std": ens["c_err"],
                "ens_n": ens["n_mem"],
                "transparency": ens["transparency"],
                "used": used_epoch,
                "jumped": jumped,
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

        # Optional mid-exposure / barycentric timestamps for the CSV.  The
        # stored MJD is taken as start-of-exposure unless the Calib header
        # records a mid-exposure keyword (e.g. MJD-MID).
        ts_mode = str(time_scale).strip().lower()
        mjd_mid = bjd_tdb = None
        if ts_mode in ("mid", "bjd", "bjd_tdb"):
            exp_arr = np.array(
                [
                    meta_by_mjd.get(m, {}).get("exposure_time", np.nan)
                    for m in epochs
                ]
            )
            mjd_kw = next(
                (m["mjd_keyword"] for m in band_files if m.get("mjd_keyword")),
                "",
            )
            if "MID" in mjd_kw.upper():
                exp_arr = np.zeros(len(epochs))
            mjd_mid, _ = _mid_bjd_times(epochs, exp_arr, None, None)
            if ts_mode in ("bjd", "bjd_tdb"):
                location = _observatory_location(observatory, calib_meta)
                t_radec = next(
                    (
                        (m["target_ra"], m["target_dec"])
                        for m in band_files
                        if np.isfinite(m.get("target_ra", np.nan))
                        and np.isfinite(m.get("target_dec", np.nan))
                    ),
                    None,
                )
                if location is None or t_radec is None:
                    log.info(
                        "differential_photometry: band %s - time_scale='bjd' "
                        "needs the observatory location (site_lon/site_lat in "
                        "Calib headers or the observatory argument) and target "
                        "coordinates; mjd_mid written instead.",
                        band,
                    )
                else:
                    _, bjd_tdb = _mid_bjd_times(
                        mjd_mid, np.zeros(len(epochs)), location, t_radec
                    )

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
            if sysrem_model is not None:
                # Plotted stars are candidates, so their per-star SysRem
                # response is already fitted; subtract the model columns.
                plot_idx = [
                    cand_ids.index(sid) if sid in cand_ids else None
                    for sid in sorted(plot_ids)
                ]
                for col, idx in enumerate(plot_idx):
                    if idx is not None:
                        ref_delta[:, col] = (
                            ref_delta[:, col] - sysrem_model[:, idx]
                        )
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
        # Time-order the rows first: the running-median filters below
        # assume consecutive rows are consecutive in time.
        tgt = tgt.sort_values("mjd").reset_index(drop=True)
        # SysRem: subtract the target's response to the shared frame
        # terms, solved by weighted least squares over its used points.
        if components:
            A_rows = np.column_stack([a for _, a in components])[
                tgt["ep_i"].to_numpy(dtype=int)
            ]
            tgt["diff"] = _sysrem_subtract(
                tgt["diff"].to_numpy(dtype=float),
                tgt["delta_err"].to_numpy(dtype=float),
                A_rows,
                tgt["used"].to_numpy(dtype=bool),
            )
        # Running-median outlier flag: single- or few-epoch excursions
        # (cosmic-ray remnants, unmasked defects) that the ensemble-level
        # checks cannot attribute.  Outlier points stay in the CSV but
        # are excluded from `used` and from the centring below.
        used_t = tgt["used"].to_numpy(dtype=bool)
        outlier_t = _running_median_mask(
            tgt["diff"].to_numpy(dtype=float),
            used_t,
            window=outlier_window,
            sigma=outlier_sigma,
        )
        tgt["outlier"] = outlier_t
        n_outlier = int(np.count_nonzero(outlier_t))
        if n_outlier:
            used_t = used_t & ~outlier_t
            tgt["used"] = used_t
            log.info(
                "differential_photometry: band %s - %d target point(s) "
                "flagged as outliers (%.1f sigma from running median).",
                band,
                n_outlier,
                outlier_sigma,
            )
        # Centre on the used subset so rejected epochs do not drag the
        # effective zero point of the plotted/product residuals.
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
            "ref RMS %.4f mag, ratio %.2f, p2p %.1f mmag, beta %.2f, "
            "sys floor %.1f mmag.",
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
            (sys_floor * 1e3) if np.isfinite(sys_floor) else np.nan,
        )

        # Long-form differential light curve rows for the CSV product.
        for row in tgt.itertuples():
            ei = int(row.ep_i)
            meta = meta_by_mjd.get(epochs[ei], {})
            csv_rows.append(
                {
                    "mjd": row.mjd,
                    "mjd_mid": (
                        mjd_mid[ei] if mjd_mid is not None else np.nan
                    ),
                    "bjd_tdb": (
                        bjd_tdb[ei] if bjd_tdb is not None else np.nan
                    ),
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
                    "systematics": sys_floor,
                    "jumped": bool(jumped[ei]),
                    "outlier": bool(row.outlier),
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
                "sys_floor": sys_floor,
                "n_sysrem": n_sysrem_use,
                "n_jump": n_jump,
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

    # Layout: panels group by band with a clear gap between groups and a
    # tight stack within one.  The residual panel (the publishable
    # product) gets the most room; legends and the title live above the
    # panel stack, never over data.
    _H = {"drift": 1.0, "diag": 0.55, "resid": 1.3}
    total_rel = sum(_H[k] for kinds in panel_plan for k in kinds)
    n_bands = len(band_payloads)

    width_in = set_size(540, aspect=1)[0]
    height_in = 0.95 * total_rel + 1.5 + 0.5 * (n_bands - 1)

    fig = plt.figure(figsize=(width_in, height_in))
    # Reserve the top of the figure for the title plus one legend row per
    # content class; product entries (~6 + one per band) may wrap.
    _leg_row = 0.17 / height_in
    _prod_rows = max(1, int(np.ceil((n_bands + 6) / 8.0)))
    n_leg = _prod_rows + int(show_diagnostics)
    _head = _leg_row * (1.7 if target_name else 0.4) + n_leg * _leg_row + 0.03
    outer = GridSpec(
        n_bands,
        1,
        figure=fig,
        hspace=0.18,
        left=0.105,
        right=0.885,
        top=1.0 - _head,
        bottom=max(0.55 / height_in, 0.04),
    )
    band_axes = []
    for i, kinds in enumerate(panel_plan):
        sub = outer[i].subgridspec(
            len(kinds), 1, height_ratios=[_H[k] for k in kinds], hspace=0.1
        )
        band_axes.append([fig.add_subplot(sub[j]) for j in range(len(kinds))])
    axes = [a for axs in band_axes for a in axs]
    for a in axes[1:]:
        a.sharex(axes[0])

    saw_airmass = saw_fwhm = saw_binned = saw_rejected = False
    for i, payload in enumerate(band_payloads):
        band = payload["band"]
        c = BAND_COLORS.get(band, BAND_COLORS.get(str(band).lower(), "k"))
        tgt = payload["tgt"]
        ens_epoch = payload["ens_epoch"]
        ref = payload["ref"]
        ref_epoch = payload["ref_epoch"]

        # Band tag on the group's top panel replaces the old per-panel
        # legends as the group identifier.
        band_axes[i][0].text(
            0.99,
            0.96,
            f"{band} band",
            transform=band_axes[i][0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=c,
        )

        for ax, kind in zip(band_axes[i], panel_plan[i]):
            if kind == "drift":
                ax_drift = ax
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
                ax_drift.set_ylabel("Delta Mag [mag]", fontsize=8)
                ax_drift.invert_yaxis()
                ax_drift.grid(
                    True, which="major", alpha=0.35, linestyle="-", linewidth=0.5
                )
                ax_drift.minorticks_on()

            elif kind == "diag":
                ax_diag = ax
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
                    saw_rejected = True
                    ax_diag.plot(
                        xe[bad],
                        tr[bad],
                        "x",
                        color="red",
                        markersize=5,
                        label="Rejected epoch",
                        zorder=4,
                    )
                ax_diag.set_ylabel("Transp.", fontsize=8)
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
                    saw_airmass = True
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
                    saw_fwhm = True
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

            else:  # resid
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
                    saw_rejected = True
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
                    saw_binned = True
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
                ax.set_ylabel("Resid. Mag [mag]", fontsize=8)
                # Fit the y-limits to the used target residuals (+/- errors)
                # with a small margin, extended to zero so the residual
                # reference line stays in view.  Rejected epochs cannot
                # stretch the axis (they are flagged in the CSV and on the
                # diagnostics panel).
                _sel = tgt[used_t] if used_t.any() else tgt
                _tgt_d = np.asarray(_sel["delta"], dtype=float)
                _tgt_e = np.asarray(_sel["delta_err"], dtype=float)
                _y_lo = np.nanmin(_tgt_d - _tgt_e)
                _y_hi = np.nanmax(_tgt_d + _tgt_e)
                if not (np.isfinite(_y_lo) and np.isfinite(_y_hi)):
                    _y_lo, _y_hi = -0.05, 0.05
                _y_lo = min(_y_lo, 0.0)
                _y_hi = max(_y_hi, 0.0)
                _pad = max(0.08 * (_y_hi - _y_lo), 0.02)
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
                if np.isfinite(payload.get("sys_floor", np.nan)):
                    note_lines.append(
                        f"sys floor = {payload['sys_floor'] * 1e3:.0f} mmag"
                    )
                if payload.get("n_sysrem"):
                    note_lines.append(f"SysRem x{payload['n_sysrem']}")
                n_out = int((~used_t).sum())
                if n_out:
                    note_lines.append(f"{n_out} rejected")
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

    # Figure-level legends above the panel stack keep symbols out of the
    # data axes: products on the top row(s), diagnostics below them.
    _ms = get_marker_size("medium")
    _r1 = [
        Line2D(
            [0],
            [0],
            marker="D",
            ls="-",
            color="dimgrey",
            markersize=3,
            lw=0.8,
            label="Ensemble mean",
        )
    ]
    for p in band_payloads:
        _b = p["band"]
        _bc = BAND_COLORS.get(_b, BAND_COLORS.get(str(_b).lower(), "k"))
        _r1.append(
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                color=_bc,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=_ms,
                label=f"{_b} target",
            )
        )
    _r2 = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="lightgrey",
            markeredgecolor="none",
            markersize=3.5,
            alpha=0.9,
            label="Ref. stars",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            ls="",
            color="whitesmoke",
            markeredgecolor="dimgrey",
            markeredgewidth=0.5,
            markersize=3,
            label="Epoch mean",
        ),
    ]
    if saw_rejected:
        _r2.append(
            Line2D(
                [0],
                [0],
                marker="x",
                ls="",
                color="red",
                markersize=5,
                label="Rejected",
            )
        )
    if saw_binned:
        _r2.append(
            Line2D(
                [0],
                [0],
                marker="s",
                ls="",
                color="black",
                markersize=3.5,
                label=f"Binned x{n_bin}",
            )
        )
    # matplotlib fills legends column-major: interleave the two logical
    # rows so targets sit together on the top row.
    prod_h = []
    if _prod_rows == 1:
        prod_h = _r1 + _r2
    else:
        for _i in range(max(len(_r1), len(_r2))):
            if _i < len(_r1):
                prod_h.append(_r1[_i])
            if _i < len(_r2):
                prod_h.append(_r2[_i])

    _top_y = 1.0 - _head + 0.012
    _y_prod = _top_y + (n_leg - _prod_rows) * _leg_row
    fig.legend(
        handles=prod_h,
        loc="lower center",
        bbox_to_anchor=(0.5, _y_prod),
        ncol=max(1, int(np.ceil(len(prod_h) / _prod_rows))),
        fontsize=7,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
    )
    if show_diagnostics:
        diag_h = [
            Line2D(
                [0],
                [0],
                marker="s",
                ls="-",
                color="darkslategray",
                markersize=3,
                lw=0.8,
                label="Transparency",
            )
        ]
        if minimum_transparency:
            diag_h.append(
                Line2D(
                    [0],
                    [0],
                    ls=":",
                    color="firebrick",
                    lw=0.8,
                    label="Rej. limit",
                )
            )
        if saw_airmass:
            diag_h.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    ls="--",
                    color="steelblue",
                    markersize=2.5,
                    lw=0.6,
                    label="Airmass",
                )
            )
        if saw_fwhm:
            diag_h.append(
                Line2D(
                    [0],
                    [0],
                    marker="v",
                    ls="-.",
                    color="darkorchid",
                    markersize=2.5,
                    lw=0.6,
                    label="FWHM",
                )
            )
        fig.legend(
            handles=diag_h,
            loc="lower center",
            bbox_to_anchor=(0.5, _top_y),
            ncol=len(diag_h),
            fontsize=7,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.9,
        )

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
