"""Post-process transient photometry with per-image zeropoint color terms.

Each image's zeropoint fit may carry a color term c_f for a color index
(f1 - f2), defined by

    mag_catalog = inst_mag + ZP + c_f * (f1 - f2).

The zeropoints stored in the Output CSV are anchored to the median
calibrator color X_ref, so a target whose true color X differs from
X_ref needs

    mag_cc = mag + c_f * (X - X_ref).

For a transient the true color is not in the catalog; it is recovered
from the transient's own paired observations.  With bands f1 and f2
observed on the same telescope and instrument within dt days, the
corrected mags and the color form two equations in two unknowns:

    m1 = m1_0 + c_1 * (X - X_ref1)
    m2 = m2_0 + c_2 * (X - X_ref2)
    X  = m1 - m2

which are solved by fixed-point iteration over every matched pair at
once (the linear case also has the closed form
X = (X0 - c1*X_ref1 + c2*X_ref2) / (1 - c1 + c2); iteration is used so
the scheme generalises to non-shared anchors and future piecewise
terms).

Only rows sharing the same telescope and instrument are paired, so a
ZTF g measurement is only ever paired with a ZTF r measurement (and
likewise r-i for i band), never across telescopes or photometric
systems.

Usage (standalone):

    python color_correction.py <reduced_dir> [--dt 1.0]

reads <reduced_dir>/LightCurve_Output.csv (falling back to a glob of
the per-image Output_*.csv) and writes
<reduced_dir>/LightCurve_Output_colorcorrected.csv.
"""

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METHODS = ("psf", "ap")


def _color_index_for_filter(filter_name):
    """Return (f1, f2) for the color index of *filter_name*, or None."""
    from zeropoint import Zeropoint

    try:
        return Zeropoint({"imageFilter": filter_name}).get_color_term_for_filter(
            filter_name
        )
    except Exception:
        return None


def _reference_color(calib, use_filter, color1, color2):
    """Median calibrator color over the same sample fit_color_term uses.

    The stored zeropoint is anchored to this color, so it is the zero
    point of the post-hoc correction.
    """
    try:
        df = calib
        if "threshold" in df.columns:
            df = df[df["threshold"] >= 5]
        if f"{use_filter}_err" in df.columns:
            df = df[df[f"{use_filter}_err"] < 0.32]
        flux = np.asarray(df["flux_AP"], float)
        ferr = np.asarray(df["flux_AP_err"], float)
        x = np.asarray(df[color1] - df[color2], float)
        ok = (
            np.isfinite(x)
            & np.isfinite(flux)
            & (flux > 0)
            & (np.abs(flux) / np.maximum(ferr, 1e-10) >= 5)
        )
        x = x[ok]
        if x.size == 0:
            x_all = np.asarray(calib[color1] - calib[color2], float)
            x = x_all[np.isfinite(x_all)]
        return float(np.median(x)) if x.size else np.nan
    except Exception:
        return np.nan


def _measure_term_for_dir(image_dir, use_filter, calib_path=None):
    """Return (slope, slope_err, color1, color2, x_ref) or None.

    Re-runs the gated fit_color_term on the image's Calib catalog, so
    the same acceptance rules apply as in the pipeline.
    """
    from zeropoint import Zeropoint

    if calib_path is None:
        cands = glob.glob(os.path.join(image_dir, "Calib_*.csv"))
        if not cands:
            return None
        calib_path = cands[0]
    try:
        calib = pd.read_csv(calib_path, comment="#")
        zp = Zeropoint(
            {
                "imageFilter": use_filter,
                "fpath": os.path.join(image_dir, "color_term_ref.fits"),
                "zeropoint": {"min_source_no": 5},
                "photometry": {"color_term_poly_order": 1},
            }
        )
        coeffs, errs = zp.fit_color_term(calib)
        if (
            coeffs is None
            or errs is None
            or len(coeffs) < 2
            or not np.isfinite(coeffs[1])
            or abs(coeffs[1]) < 1e-9
        ):
            return None
        color1, color2 = zp.get_color_term_for_filter(use_filter)
        x_ref = _reference_color(calib, use_filter, color1, color2)
        return float(coeffs[1]), float(errs[1]), color1, color2, x_ref
    except Exception as exc:
        logger.debug("color term measurement failed for %s: %s", image_dir, exc)
        return None


def collect_color_terms(df, term_by_dir=None):
    """Build a per-row term table keyed on each row's image directory.

    Prefers the color_term_* columns written by the pipeline; falls
    back to re-measuring from the sibling Calib_*.csv for outputs made
    before those columns existed.
    """
    df = df.copy()
    df["_image_dir"] = df["filename_path"].astype(str).map(os.path.dirname)
    if term_by_dir is None:
        term_by_dir = {}
    terms = {}
    for image_dir, grp in df.groupby("_image_dir"):
        if image_dir in term_by_dir:
            terms[image_dir] = term_by_dir[image_dir]
            continue
        row = grp.iloc[0]
        use_filter = str(row.get("filter", ""))
        c_col = row.get("color_term_psf", np.nan)
        if np.isfinite(c_col) and abs(c_col) > 1e-9:
            idx = str(row.get("color_index", ""))
            if "-" in idx:
                c1, c2 = idx.split("-", 1)
            else:
                pair = _color_index_for_filter(use_filter)
                c1, c2 = pair if pair else (None, None)
            # A ZP fit that already included the term is anchored at
            # color 0; an uncorrected ZP is anchored at the median
            # calibrator color.
            x_ref = (
                0.0
                if bool(row.get("has_color_term", False))
                else float(row.get("color_ref", np.nan))
            )
            terms[image_dir] = (
                float(c_col),
                float(row.get("color_term_psf_err", 0.0)),
                c1,
                c2,
                x_ref,
            )
        else:
            terms[image_dir] = _measure_term_for_dir(image_dir, use_filter)
    return df, terms


def apply_transient_color_correction(
    df, terms, dt=1.0, max_err=0.5, max_iter=200, tol=1e-6
):
    """Apply the iterative transient color correction in-place on *df*.

    Adds, per photometry method m in {"psf", "ap"}:
      mag_{m}_cc, mag_{m}_cc_err, cc_color_{m}, cc_color_err_{m},
      cc_dt_{m}, cc_index_{m}, cc_applied_{m}
    """
    for m in METHODS:
        df[f"mag_{m}_cc"] = np.nan
        df[f"mag_{m}_cc_err"] = np.nan
        df[f"cc_color_{m}"] = np.nan
        df[f"cc_color_err_{m}"] = np.nan
        df[f"cc_dt_{m}"] = np.nan
        df[f"cc_index_{m}"] = ""
        df[f"cc_applied_{m}"] = False

    if "_image_dir" not in df.columns:
        df["_image_dir"] = df["filename_path"].astype(str).map(os.path.dirname)
    df["_row_id"] = np.arange(len(df))
    group_cols = [c for c in ("telescope", "instrument") if c in df.columns]

    for _, grp in df.groupby(group_cols or ["_image_dir"]):
        grp = grp[pd.notna(grp["mjd"])].copy()
        if len(grp) < 2:
            continue
        for m in METHODS:
            mag_col, err_col = f"mag_{m}", f"mag_{m}_err"
            if mag_col not in grp.columns:
                continue
            _correct_group(
                grp, terms, m, mag_col, err_col, dt, max_err, max_iter, tol
            )
            for col in (
                f"mag_{m}_cc",
                f"mag_{m}_cc_err",
                f"cc_color_{m}",
                f"cc_color_err_{m}",
                f"cc_dt_{m}",
                f"cc_index_{m}",
                f"cc_applied_{m}",
            ):
                df.loc[grp.index, col] = grp[col]
    df.drop(columns=["_row_id", "_image_dir"], inplace=True)
    return df


def _correct_group(
    grp, terms, method, mag_col, err_col, dt, max_err, max_iter, tol
):
    """Fixed-point solve of corrected mags within one instrument group."""
    idx = grp.index.to_numpy()
    mjd = grp["mjd"].to_numpy(float)
    m0 = grp[mag_col].to_numpy(float)
    e0 = (
        grp[err_col].to_numpy(float)
        if err_col in grp.columns
        else np.full(len(grp), np.nan)
    )

    slope = np.full(len(grp), np.nan)
    slope_err = np.zeros(len(grp))
    xref = np.full(len(grp), np.nan)
    partner = np.full(len(grp), -1, dtype=int)
    sign = np.zeros(len(grp))  # +1 if row filter is f1 of the index, -1 if f2

    for j, i in enumerate(idx):
        info = terms.get(grp.at[i, "_image_dir"])
        if not info:
            continue
        c, c_err, c1, c2, xr = info
        filt = str(grp.at[i, "filter"])
        if filt == c1:
            sign[j] = 1.0
        elif filt == c2:
            sign[j] = -1.0
        else:
            continue
        slope[j], slope_err[j], xref[j] = c, c_err, xr

    if not np.any(np.isfinite(slope)):
        return

    # Pair each correctable row with the nearest same-group row in the
    # partner band inside the dt window.
    filters = grp["filter"].astype(str).to_numpy()
    for j, i in enumerate(idx):
        if not np.isfinite(slope[j]):
            continue
        info = terms[grp.at[i, "_image_dir"]]
        # sign>0: row filter is f1, partner band is f2 (and vice versa).
        other = info[3] if sign[j] > 0 else info[2]
        cand = np.where(filters == other)[0]
        # Non-detections and low-S/N points cannot anchor a color.
        cand = cand[np.isfinite(m0[cand]) & (e0[cand] <= max_err)]
        if cand.size == 0:
            continue
        dt_abs = np.abs(mjd[cand] - mjd[j])
        k = cand[np.argmin(dt_abs)]
        if dt_abs.min() <= dt and k != j:
            partner[j] = k
            grp.at[i, f"cc_dt_{method}"] = float(dt_abs.min())
            grp.at[i, f"cc_index_{method}"] = f"{info[2]}-{info[3]}"

    ok = (
        np.isfinite(slope)
        & np.isfinite(xref)
        & (partner >= 0)
        & np.isfinite(m0)
        & (e0 <= max_err)
    )
    if not np.any(ok):
        return

    # Fixed-point iteration: corrected mags feed back into the colors.
    corr = m0.copy()
    for _ in range(max_iter):
        delta = 0.0
        new = corr.copy()
        for j in np.where(ok)[0]:
            k = partner[j]
            color = sign[j] * (corr[j] - corr[k])
            v = m0[j] + slope[j] * (color - xref[j])
            delta = max(delta, abs(v - corr[j]))
            new[j] = v
        corr = new
        if delta < tol:
            break

    # Errors: sigma_corr^2 = (c_err*|X-Xref|)^2 + (|c|*sigma_X)^2 with
    # sigma_X^2 = e1^2 + e2^2 (each side's best current estimate).
    err_cc = e0.copy()
    for _ in range(2):  # one refinement pass picks up partner corr errors
        for j in np.where(ok)[0]:
            k = partner[j]
            color = sign[j] * (corr[j] - corr[k])
            resid = color - xref[j]
            sig_x = np.hypot(
                err_cc[j] if np.isfinite(err_cc[j]) else 0.0,
                err_cc[k] if np.isfinite(err_cc[k]) else 0.0,
            )
            corr_err = np.hypot(abs(slope_err[j]) * abs(resid),
                                abs(slope[j]) * sig_x)
            e = e0[j] if np.isfinite(e0[j]) else 0.0
            err_cc[j] = np.hypot(e, corr_err)

    for j in np.where(ok)[0]:
        i = idx[j]
        k = partner[j]
        grp.at[i, f"mag_{method}_cc"] = corr[j]
        grp.at[i, f"mag_{method}_cc_err"] = err_cc[j]
        grp.at[i, f"cc_color_{method}"] = sign[j] * (corr[j] - corr[k])
        grp.at[i, f"cc_color_err_{method}"] = np.hypot(
            err_cc[j], err_cc[k]
        )
        grp.at[i, f"cc_applied_{method}"] = True


def correct_lightcurve(
    lc_path, reduced_dir=None, dt=1.0, max_err=0.5, out_path=None
):
    """Correct a light-curve CSV; returns the output path or None."""
    if not os.path.exists(lc_path):
        logger.warning("correct_lightcurve: %s not found", lc_path)
        return None
    if reduced_dir is None:
        reduced_dir = os.path.dirname(os.path.abspath(lc_path))
    df = pd.read_csv(lc_path, comment="#")
    needed = {"mjd", "filter", "filename_path"}
    if not needed.issubset(df.columns):
        logger.warning(
            "correct_lightcurve: %s lacks columns %s", lc_path, needed - set(df.columns)
        )
        return None
    df, terms = collect_color_terms(df)
    n_terms = sum(1 for v in terms.values() if v)
    logger.info("color terms available for %d/%d images", n_terms, len(terms))
    df = apply_transient_color_correction(df, terms, dt=dt, max_err=max_err)
    if out_path is None:
        root, ext = os.path.splitext(lc_path)
        out_path = f"{root}_colorcorrected{ext or '.csv'}"
    df.to_csv(out_path, index=False)
    logger.info("color-corrected light curve written to %s", out_path)
    return out_path


def _print_summary(df):
    from functions import ascii_table

    rows = []
    for m in METHODS:
        flag = f"cc_applied_{m}"
        if flag not in df.columns:
            continue
        for filt, sub in df[df[flag] == True].groupby("filter"):
            corr = sub[f"mag_{m}_cc"] - sub[f"mag_{m}"]
            rows.append(
                [
                    m.upper(),
                    filt,
                    str(len(sub)),
                    f"{sub[f'cc_index_{m}'].iloc[0]}",
                    f"{np.nanmedian(corr):+.4f}",
                    f"{np.nanmedian(sub[f'cc_dt_{m}']):.3f}",
                    f"{np.nanmedian(sub[f'cc_color_{m}']):+.3f}",
                ]
            )
    if not rows:
        print("No color corrections applied (no paired epochs or no terms).")
        return
    print(
        ascii_table(
            "Transient color correction",
            ["method", "filter", "n", "index", "med corr", "med dt(d)", "med color"],
            rows,
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Post-process transient photometry with zeropoint color terms."
    )
    ap.add_argument("path", help="Reduced dir or LightCurve CSV")
    ap.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Max |t1 - t2| in days for pairing color-index filters (default 1.0)",
    )
    ap.add_argument(
        "--max-err",
        type=float,
        default=0.5,
        help="Max magnitude error for both members of a pair (default 0.5)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from functions import cap_console_lines
    cap_console_lines(logging.getLogger().handlers, prefix_fmt="{levelname} ")

    if os.path.isdir(args.path):
        reduced_dir = args.path
        lc_path = os.path.join(reduced_dir, "LightCurve_Output.csv")
        if os.path.exists(lc_path):
            df = pd.read_csv(lc_path, comment="#")
        else:
            outs = sorted(
                glob.glob(os.path.join(reduced_dir, "*", "Output_*.csv"))
            )
            if not outs:
                logger.error("No Output_*.csv under %s", reduced_dir)
                return 1
            df = pd.concat(
                [pd.read_csv(f, comment="#") for f in outs], ignore_index=True
            )
        out_path = os.path.join(
            reduced_dir, "LightCurve_Output_colorcorrected.csv"
        )
    else:
        reduced_dir = os.path.dirname(os.path.abspath(args.path))
        df = pd.read_csv(args.path, comment="#")
        root, ext = os.path.splitext(args.path)
        out_path = f"{root}_colorcorrected{ext or '.csv'}"

    df, terms = collect_color_terms(df)
    n_terms = sum(1 for v in terms.values() if v)
    print(f"Color terms measured/loaded for {n_terms}/{len(terms)} images")
    df = apply_transient_color_correction(
        df, terms, dt=args.dt, max_err=args.max_err
    )
    df.to_csv(out_path, index=False)
    _print_summary(df)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
