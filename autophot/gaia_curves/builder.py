from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .gaia_archive import (
    _gaiaxpy_chunk_size,
    calibrate_source_ids_batched,
    gaia_xp_sql_top_n,
    query_gaia_xp_cone_growing,
    retry_with_backoff,
    sort_gaia_table_nearest_to_target,
)

# Column-name conventions seen in GaiaXPy calibrate() output across versions.
_SPECTRUM_WAVE_KEYS = ("wavelength", "wavelength_nm", "wl", "lambda")
_SPECTRUM_FLUX_KEYS = (
    "flux",
    "flux_w_m2_nm",
    "flux_density",
    "flambda",
    "calibrated_flux",
    "xp_flux",
    "sampled_flux",
    "xp_sampled_flux",
    "mean_spectrum",
    "xp_mean_spectrum",
)
_SPECTRUM_ERR_KEYS = (
    "flux_error",
    "flux_uncertainty",
    "sigma_flux",
    "calibrated_flux_error",
    "flux_err",
)
_SPECTRUM_NONFLUX_NAMES = {
    "source_id",
    "ra",
    "dec",
    "designation",
    "flux_error",
    "flux_uncertainty",
    "correlation",
}

_C_LIGHT_M_S = 299792458.0
_F_NU_AB_ZERO = 3631e-26


def _flux_col_score(name: str) -> int:
    n = name.lower()
    if "error" in n or "uncert" in n or "corr" in n:
        return 200
    if n == "flux":
        return 0
    if "flux" in n:
        return 2
    return 10


def _as_1d_float_array(v: Any) -> Optional[np.ndarray]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return a if a.ndim == 1 else None


def _format_gaia_source_id(val: Any) -> str:
    """
    Stable string ID for cross-matching TAP rows to GaiaXPy output.

    Gaia ``source_id`` is a 64-bit integer; if it becomes ``float64`` in pandas,
    string conversion can round and break joins - prefer integer paths.
    """
    if val is None:
        return ""
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return ""
        if s.isdigit():
            return s
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            return s
    if isinstance(val, (bool, np.bool_)):
        return str(int(val))
    if isinstance(val, (np.integer, int)):
        return str(int(val))
    if isinstance(val, (np.floating, float)):
        if not np.isfinite(val):
            return ""
        x = float(val)
        if abs(x) >= 1e15:
            return f"{x:.0f}"
        return str(int(round(x)))
    return str(val).strip()


@dataclass
class GaiaCurveCatalogBuilder:
    """
    Build an AutoPHoT-compatible custom catalog from Gaia DR3 + XP spectra
    using user passband transmission curves.
    """

    logger: logging.Logger = logging.getLogger("autophot_gaia_curves")

    @staticmethod
    def parse_mapping(items: Iterable[str], flag_name: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for raw in items:
            if "=" not in raw:
                raise ValueError(f"{flag_name} expects BAND=VALUE format, got: {raw}")
            band, value = raw.split("=", 1)
            band = band.strip()
            value = value.strip()
            if not band or not value:
                raise ValueError(f"Invalid {flag_name} entry: {raw}")
            mapping[band] = value
        return mapping

    @staticmethod
    def download_svo_curve(
        svo_id: str,
        dest: Path,
        timeout_sec: int = 60,
    ) -> Path:
        url = f"https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={svo_id}&FORMAT=ascii"
        resp = requests.get(url, timeout=timeout_sec)
        resp.raise_for_status()
        dest.write_text(resp.text)
        return dest

    @staticmethod
    def curve_wavelength_axis_to_nm(wl_raw: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Map filter curve wavelengths to nanometres for overlap with Gaia XP (~330-1050 nm).

        Files may list **Angstrom** (typical SVO / instrument: ~3000-11000 A) or **nm**
        (~300-1100). Heuristic: if max wavelength > 2500, treat values as A and divide
        by 10; otherwise assume they are already nm. Mis-classifying A as nm was the
        main cause of all-NaN synthetic mags (no spectral overlap after scaling).
        """
        wl_raw = np.asarray(wl_raw, dtype=float)
        w_max = float(np.nanmax(wl_raw))
        w_min = float(np.nanmin(wl_raw))
        if w_max > 2500.0:
            curve_nm = wl_raw / 10.0
            note = f"wavelengths interpreted as Angstrom (raw {w_min:.1f}-{w_max:.1f} A)"
        else:
            curve_nm = wl_raw.copy()
            note = f"wavelengths interpreted as nanometres ({w_min:.1f}-{w_max:.1f} nm)"
        return curve_nm, note

    @staticmethod
    def read_curve(curve_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a curve file with two numeric columns:
            wavelength (Angstrom **or** nm - see :meth:`curve_wavelength_axis_to_nm`),
            throughput (arbitrary positive)
        """
        wl: List[float] = []
        tr: List[float] = []
        for line in curve_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                t = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(w) and math.isfinite(t):
                wl.append(w)
                tr.append(max(0.0, t))
        if len(wl) < 2:
            raise ValueError(f"No usable transmission data in {curve_path}")
        wl_arr = np.asarray(wl, dtype=float)
        tr_arr = np.asarray(tr, dtype=float)
        order = np.argsort(wl_arr)
        return wl_arr[order], tr_arr[order]

    @staticmethod
    def extract_spectrum_columns(
        row: pd.Series,
        wavelength_nm: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get wavelength (nm), flux (W m^-2 nm^-1), and optional flux uncertainty arrays.

        GaiaXPy often provides a global wavelength grid as the second return value
        of ``calibrate()``; per-row data then only contains flux (and errors).
        """
        wave = None
        flux = None
        for k in _SPECTRUM_WAVE_KEYS:
            if k in row and row[k] is not None:
                wave = np.asarray(row[k], dtype=float)
                break
        if wave is None and wavelength_nm is not None:
            wave = np.asarray(wavelength_nm, dtype=float)
        for k in _SPECTRUM_FLUX_KEYS:
            if k in row and row[k] is not None:
                flux = np.asarray(row[k], dtype=float)
                break

        # GaiaXPy column names vary by version: pick any 1-D numeric column that
        # matches the calibrated wavelength grid length (skip obvious non-flux).
        if flux is None and wavelength_nm is not None:
            nexp = int(np.asarray(wavelength_nm).shape[0])
            matched: List[Tuple[str, np.ndarray]] = []
            for k in row.index:
                kl = str(k).lower()
                if (
                    k in _SPECTRUM_NONFLUX_NAMES
                    or kl.endswith("_error")
                    or kl.endswith("_uncertainty")
                ):
                    continue
                v = row[k]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                try:
                    a = np.asarray(v, dtype=float)
                except (TypeError, ValueError):
                    continue
                if a.ndim == 1 and a.shape[0] == nexp and np.any(np.isfinite(a)):
                    matched.append((str(k), a))

            if matched:
                matched.sort(key=lambda kv: (_flux_col_score(kv[0]), kv[0]))
                flux = matched[0][1]

        if wave is None or flux is None:
            raise KeyError(
                "Could not find wavelength/flux arrays in GaiaXPy output row. "
                "Inspect calibrate(...) output columns for your GaiaXPy version."
            )

        if wave.shape != flux.shape:
            raise ValueError(
                f"Wavelength grid length {wave.shape} != flux length {flux.shape}."
            )

        err: Optional[np.ndarray] = None
        for ek in _SPECTRUM_ERR_KEYS:
            if ek in row and row[ek] is not None:
                try:
                    e = np.asarray(row[ek], dtype=float)
                except (TypeError, ValueError):
                    continue
                if e.shape == flux.shape:
                    err = e
                    break
        if err is None and wavelength_nm is not None:
            nexp = int(flux.shape[0])
            for k in row.index:
                kl = str(k).lower()
                if (
                    "error" not in kl
                    and "uncert" not in kl
                    and "sigma" not in kl
                ):
                    continue
                if "corr" in kl:
                    continue
                v = row[k]
                if v is None:
                    continue
                try:
                    e = np.asarray(v, dtype=float)
                except (TypeError, ValueError):
                    continue
                if e.ndim == 1 and e.shape[0] == nexp:
                    err = e
                    break

        valid = np.isfinite(wave) & np.isfinite(flux)
        if err is not None:
            valid &= np.isfinite(err) & (err >= 0.0)
        if np.count_nonzero(valid) < 2:
            raise ValueError("Not enough valid spectral samples.")
        if err is not None:
            return wave[valid], flux[valid], err[valid]
        return wave[valid], flux[valid], None

    @staticmethod
    def _trapezoid_integration_coeffs(x: np.ndarray) -> np.ndarray:
        """
        Coefficients ``c`` with ``np.trapz(y, x) == np.dot(c, y)`` for any ``y``.

        Propagates uncorrelated per-bin uncertainties through the trapezoid rule.
        Gaia XP flux errors are correlated between bins; this is an approximation.
        """
        x = np.asarray(x, dtype=float)
        n = int(x.size)
        if n < 2:
            return np.zeros(n, dtype=float)
        c = np.zeros(n, dtype=float)
        c[0] = 0.5 * (x[1] - x[0])
        c[-1] = 0.5 * (x[-1] - x[-2])
        c[1:-1] = 0.5 * (x[2:] - x[:-2])
        return c

    @staticmethod
    def ab_mag_and_err_from_curve(
        wavelength_nm: np.ndarray,
        flux_w_m2_nm: np.ndarray,
        flux_err_w_m2_nm: Optional[np.ndarray],
        curve_wavelength_nm: np.ndarray,
        curve_throughput: np.ndarray,
    ) -> Tuple[float, float]:
        """
        AB magnitude and approximate 1-sigma magnitude error from band integration.

        Uses GaiaXPy ``flux_error`` (W m^-2 nm^-1), propagated through the same f_nu
        mapping and trapezoidal band average as the magnitude. Bin errors are treated
        as independent (ignores covariance). Returns ``(nan, nan)`` if the band
        integral is undefined; ``mag_err`` is ``nan`` if ``flux_err_w_m2_nm`` is
        ``None``.
        """
        curve_nm = np.asarray(curve_wavelength_nm, dtype=float)
        t = np.interp(
            wavelength_nm,
            curve_nm,
            curve_throughput,
            left=0.0,
            right=0.0,
        )
        valid = (
            np.isfinite(wavelength_nm)
            & np.isfinite(flux_w_m2_nm)
            & np.isfinite(t)
            & (t > 1e-15)
        )
        if flux_err_w_m2_nm is not None:
            valid &= np.isfinite(flux_err_w_m2_nm) & (flux_err_w_m2_nm >= 0.0)
        if np.count_nonzero(valid) < 2:
            return np.nan, np.nan

        wl_nm = wavelength_nm[valid]
        fl_nm = flux_w_m2_nm[valid]
        tv = t[valid]

        fl_m = fl_nm * 1e9
        wl_m = wl_nm * 1e-9
        f_nu = fl_m * (wl_m**2) / _C_LIGHT_M_S

        if hasattr(np, "trapezoid"):
            _trapz = np.trapezoid  # NumPy >= 2.0
        else:
            _trapz = np.trapz  # pragma: no cover
        y = f_nu * tv
        num = _trapz(y, wl_nm)
        den = _trapz(tv, wl_nm)
        if den <= 0 or not np.isfinite(den):
            return np.nan, np.nan
        f_nu_band = num / den
        if f_nu_band <= 0 or not np.isfinite(f_nu_band):
            return np.nan, np.nan

        mag = -2.5 * np.log10(f_nu_band / _F_NU_AB_ZERO)

        mag_err = np.nan
        if flux_err_w_m2_nm is not None:
            sig_fl = flux_err_w_m2_nm[valid]
            sig_fn = sig_fl * 1e9 * (wl_m**2) / _C_LIGHT_M_S
            coeffs = GaiaCurveCatalogBuilder._trapezoid_integration_coeffs(wl_nm)
            var_num = float(np.sum((coeffs * tv * sig_fn) ** 2))
            sig_num = math.sqrt(var_num) if var_num > 0.0 else 0.0
            sig_fband = sig_num / den
            if f_nu_band > 0 and np.isfinite(sig_fband) and sig_fband >= 0.0:
                mag_err = (2.5 / math.log(10.0)) * (sig_fband / f_nu_band)

        return float(mag), float(mag_err)

    @staticmethod
    def ab_mag_from_curve(
        wavelength_nm: np.ndarray,
        flux_w_m2_nm: np.ndarray,
        curve_wavelength_nm: np.ndarray,
        curve_throughput: np.ndarray,
    ) -> float:
        """
        Approximate AB magnitude by integrating f_nu weighted by throughput.

        Assumes GaiaXPy calibrated spectra in W m^-2 nm^-1 and filter wavelengths
        already in **nm** (use :meth:`curve_wavelength_axis_to_nm` when loading files).
        """
        m, _ = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
            wavelength_nm,
            flux_w_m2_nm,
            None,
            curve_wavelength_nm,
            curve_throughput,
        )
        return m

    @staticmethod
    def _spectrum_column_names(
        df: pd.DataFrame, n_exp: int
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Pick (wavelength, flux, flux_error) columns once for a whole table.

        Mirrors :meth:`extract_spectrum_columns` using the first row as the
        format probe: GaiaXPy tables are column-uniform, so rescanning per
        row only repeats the same decision ``n_rows`` times. ``n_exp`` is the
        shared wavelength-grid length used by the length-match heuristics.
        """
        if df.empty:
            return None, None, None
        cols = list(df.columns)
        first = df.iloc[0]

        def _present(k: str) -> bool:
            return k in cols and first[k] is not None

        wave_col = next((k for k in _SPECTRUM_WAVE_KEYS if _present(k)), None)
        flux_col = next((k for k in _SPECTRUM_FLUX_KEYS if _present(k)), None)
        if flux_col is None and n_exp:
            matched: List[str] = []
            for c in cols:
                kl = str(c).lower()
                if (
                    c in _SPECTRUM_NONFLUX_NAMES
                    or kl.endswith("_error")
                    or kl.endswith("_uncertainty")
                ):
                    continue
                a = _as_1d_float_array(first[c])
                if (
                    a is not None
                    and a.shape[0] == n_exp
                    and np.any(np.isfinite(a))
                ):
                    matched.append(str(c))
            if matched:
                matched.sort(key=lambda nm: (_flux_col_score(nm), nm))
                flux_col = matched[0]

        err_col = None
        for k in _SPECTRUM_ERR_KEYS:
            if not _present(k):
                continue
            e = _as_1d_float_array(first[k])
            if e is not None and e.shape[0] == n_exp:
                err_col = k
                break
        if err_col is None and n_exp:
            for c in cols:
                kl = str(c).lower()
                if "corr" in kl:
                    continue
                if "error" not in kl and "uncert" not in kl and "sigma" not in kl:
                    continue
                e = _as_1d_float_array(first[c])
                if e is not None and e.shape[0] == n_exp:
                    err_col = c
                    break
        return wave_col, flux_col, err_col

    def _spectra_matrix(
        self,
        spectra_df: pd.DataFrame,
        xp_sampling_nm: Optional[np.ndarray],
    ) -> Optional[
        Tuple[List[str], np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]
    ]:
        """
        Stack all per-row spectra onto one (n_rows, n_wl) grid.

        Returns ``(source_ids, wave_nm, flux, flux_err, row_ok)`` where
        ``flux_err`` is None when the table has no error column and
        ``row_ok`` flags rows that filled the matrix. Returns None when no
        shared grid exists at all (missing flux column, per-row wavelength
        grids that disagree with each other or with ``xp_sampling_nm``);
        rows with malformed cells keep the original per-row path through
        ``row_ok``.
        """
        if (
            spectra_df is None
            or spectra_df.empty
            or "source_id" not in spectra_df.columns
        ):
            return None

        wave_col_guess = next(
            (
                k
                for k in _SPECTRUM_WAVE_KEYS
                if k in spectra_df.columns and spectra_df[k].iloc[0] is not None
            ),
            None,
        )
        w0: Optional[np.ndarray] = None
        if wave_col_guess is not None:
            w0 = _as_1d_float_array(spectra_df[wave_col_guess].iloc[0])
            if w0 is None:
                return None
            for v in spectra_df[wave_col_guess].tolist():
                wv = _as_1d_float_array(v)
                if (
                    wv is None
                    or wv.shape != w0.shape
                    or not np.allclose(
                        wv, w0, rtol=0.0, atol=1e-6, equal_nan=True
                    )
                ):
                    return None
        grid: Optional[np.ndarray] = None
        if xp_sampling_nm is not None:
            g = np.asarray(xp_sampling_nm, dtype=float)
            if g.ndim == 1 and g.size >= 2:
                grid = g
        if w0 is not None and grid is not None:
            if w0.shape != grid.shape or not np.allclose(
                w0, grid, rtol=0.0, atol=1e-6, equal_nan=True
            ):
                return None
        wave_nm = w0 if w0 is not None else grid
        if wave_nm is None or wave_nm.ndim != 1 or wave_nm.size < 2:
            return None
        m = int(wave_nm.shape[0])

        _, flux_col, err_col = self._spectrum_column_names(spectra_df, m)
        if flux_col is None:
            return None

        n = len(spectra_df)
        row_ok = np.ones(n, dtype=bool)
        flux_mat = np.full((n, m), np.nan, dtype=float)
        for i, v in enumerate(spectra_df[flux_col].tolist()):
            a = _as_1d_float_array(v)
            if a is not None and a.shape[0] == m:
                flux_mat[i] = a
            else:
                row_ok[i] = False
        err_mat: Optional[np.ndarray] = None
        if err_col is not None:
            err_mat = np.full((n, m), np.nan, dtype=float)
            for i, v in enumerate(spectra_df[err_col].tolist()):
                if not row_ok[i]:
                    continue
                e = _as_1d_float_array(v)
                if e is not None and e.shape[0] == m:
                    err_mat[i] = e
                else:
                    row_ok[i] = False

        # Rows with <2 valid samples fail extract_spectrum_columns anyway;
        # route them through the per-row path so they are skipped, not kept
        # with all-NaN magnitudes.
        min_valid = np.isfinite(wave_nm)[None, :] & np.isfinite(flux_mat)
        if err_mat is not None:
            min_valid &= np.isfinite(err_mat) & (err_mat >= 0.0)
        row_ok &= min_valid.sum(axis=1) >= 2

        sids = [
            _format_gaia_source_id(v)
            for v in spectra_df["source_id"].tolist()
        ]
        return sids, wave_nm, flux_mat, err_mat, row_ok

    @staticmethod
    def _band_integration_weights(
        wavelength_nm: np.ndarray,
        curve_wavelength_nm: np.ndarray,
        curve_throughput: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Precompute the shared-grid part of :meth:`ab_mag_and_err_from_curve`.

        Returns ``(idx, weight, denom)`` where ``dot(weight, f_lambda[idx])``
        is the throughput-weighted f_nu numerator, ``dot(weight**2, sig**2)``
        its uncorrelated-error variance, and ``denom`` the throughput
        denominator. ``weight`` folds in the trapezoid coefficients, the
        f_lambda -> f_nu factor and the throughput, so a whole batch of
        spectra needs one matrix product per band instead of a fresh
        ``np.interp`` + ``np.trapezoid`` per source. Returns None when fewer
        than 2 grid points carry throughput (all mags are then NaN anyway).
        """
        wave = np.asarray(wavelength_nm, dtype=float)
        t = np.interp(
            wave,
            np.asarray(curve_wavelength_nm, dtype=float),
            np.asarray(curve_throughput, dtype=float),
            left=0.0,
            right=0.0,
        )
        sel = np.isfinite(wave) & np.isfinite(t) & (t > 1e-15)
        idx = np.flatnonzero(sel)
        if idx.size < 2:
            return None
        wl = wave[idx]
        tv = t[idx]
        coeffs = GaiaCurveCatalogBuilder._trapezoid_integration_coeffs(wl)
        k = 1e9 * ((wl * 1e-9) ** 2) * tv / _C_LIGHT_M_S
        w = coeffs * k
        den = float(np.dot(coeffs, tv))
        return idx, w, den

    @staticmethod
    def _ab_mags_from_matrix(
        wavelength_nm: np.ndarray,
        flux_mat: np.ndarray,
        err_mat: Optional[np.ndarray],
        row_ok: np.ndarray,
        curve_wavelength_nm: np.ndarray,
        curve_throughput: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized equivalent of per-row :meth:`ab_mag_and_err_from_curve`
        for spectra stacked on a shared wavelength grid.

        Rows whose samples inside the band are all valid integrate with one
        matrix product; rows carrying NaN fluxes or bad error samples inside
        the band fall back to the reference per-row path so results stay
        identical. Rows with ``row_ok`` False are left as NaN (the caller
        re-extracts them per row).
        """
        n = int(flux_mat.shape[0])
        mags = np.full(n, np.nan, dtype=float)
        errs = np.full(n, np.nan, dtype=float)
        pre = GaiaCurveCatalogBuilder._band_integration_weights(
            wavelength_nm, curve_wavelength_nm, curve_throughput
        )
        if pre is None:
            return mags, errs
        idx, w, den = pre
        if not np.isfinite(den) or den <= 0:
            return mags, errs
        usable = np.asarray(row_ok, dtype=bool)
        flux_sel = flux_mat[:, idx]
        clean = np.isfinite(flux_sel).all(axis=1) & usable
        err_sel: Optional[np.ndarray] = None
        if err_mat is not None:
            err_sel = err_mat[:, idx]
            clean &= np.isfinite(err_sel).all(axis=1) & (err_sel >= 0).all(axis=1)
        if np.any(clean):
            f_band = (flux_sel[clean] @ w) / den
            pos = np.isfinite(f_band) & (f_band > 0)
            mag_c = np.full(f_band.shape[0], np.nan, dtype=float)
            mag_c[pos] = -2.5 * np.log10(f_band[pos] / _F_NU_AB_ZERO)
            err_c = np.full(f_band.shape[0], np.nan, dtype=float)
            if err_sel is not None:
                sig = np.sqrt((err_sel[clean] ** 2) @ (w * w)) / den
                good = pos & np.isfinite(sig) & (sig >= 0)
                err_c[good] = (2.5 / math.log(10.0)) * (
                    sig[good] / f_band[good]
                )
            mags[clean] = mag_c
            errs[clean] = err_c
        for i in np.flatnonzero(usable & ~clean):
            err_i = err_mat[i] if err_mat is not None else None
            m_i, e_i = GaiaCurveCatalogBuilder.ab_mag_and_err_from_curve(
                wavelength_nm,
                flux_mat[i],
                err_i,
                curve_wavelength_nm,
                curve_throughput,
            )
            mags[i] = m_i
            errs[i] = e_i
        return mags, errs

    def _fill_band_mags(
        self,
        rec: dict,
        wavelength_nm: np.ndarray,
        flux_w_m2_nm: np.ndarray,
        flux_err_w_m2_nm: Optional[np.ndarray],
        loaded_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Per-row band integration: reference path for one spectrum."""
        for band, (cw, ct) in loaded_curves.items():
            mag, mag_err = self.ab_mag_and_err_from_curve(
                wavelength_nm,
                flux_w_m2_nm,
                flux_err_w_m2_nm,
                cw,
                ct,
            )
            rec[band] = mag
            # Default 0.01 mag when GaiaXPy errors missing or non-finite.
            rec[f"{band}_err"] = 0.01 if not math.isfinite(mag_err) else mag_err

    @staticmethod
    def _meta_path(out_path: Path) -> Path:
        """Provenance sidecar recording the query footprint of a built catalog."""
        return out_path.with_suffix(out_path.suffix + ".meta.json")

    def _catalog_cache_usable(
        self,
        cached_df: pd.DataFrame,
        meta_path: Path,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        *,
        min_sources: int,
        max_radius_deg: float,
    ) -> bool:
        """
        Decide whether an existing catalog CSV can be reused.

        Without a provenance sidecar (catalogs written before this existed)
        only the row count is known, so the file is trusted only when it
        already meets ``min_sources``. With provenance, reuse requires the
        same target position and a query cone at least as wide as requested;
        sparse catalogs are reused only when the earlier build already grew
        the cone to its cap.
        """
        needed = max(1, int(min_sources))
        if not meta_path.exists():
            return len(cached_df) >= needed
        try:
            meta = json.loads(meta_path.read_text())
            used_radius = float(meta["used_radius_deg"])
            if (
                abs(float(meta["ra_deg"]) - float(ra_deg)) > 1e-5
                or abs(float(meta["dec_deg"]) - float(dec_deg)) > 1e-5
            ):
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if used_radius < float(radius_deg) - 1e-6:
            return False
        if len(cached_df) >= needed:
            return True
        return (
            bool(meta.get("growth_exhausted"))
            and float(meta.get("max_radius_deg", 0.0))
            >= float(max_radius_deg) - 1e-6
        )

    def _write_outputs(
        self,
        out_path: Path,
        out_df: pd.DataFrame,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        used_radius_deg: float,
        max_radius_deg: float,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        cap = max(float(max_radius_deg), float(radius_deg))
        meta = {
            "ra_deg": float(ra_deg),
            "dec_deg": float(dec_deg),
            "requested_radius_deg": float(radius_deg),
            "used_radius_deg": float(used_radius_deg),
            "max_radius_deg": float(cap),
            "n_rows": int(len(out_df)),
            "growth_exhausted": bool(used_radius_deg >= cap - 1e-9),
        }
        try:
            self._meta_path(out_path).write_text(json.dumps(meta, indent=2))
        except OSError:
            pass

    def build(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        max_sources: int,
        curves: Dict[str, str | Path],
        out_csv: str | Path,
        svo_filters: Optional[Dict[str, str]] = None,
        curve_cache_dir: str | Path = "svo_curves",
        timeout_sec: int = 60,
        gaia_query_pause_before_sec: float = 0.25,
        gaia_query_pause_after_sec: float = 0.25,
        gaia_xp_batch_size: int = 200,
        gaia_xp_batch_pause_sec: float = 0.5,
        gaia_archive_max_retries: int = 3,
        gaia_archive_retry_base_delay_sec: float = 2.0,
        gaia_xp_order_by: str = "distance",
        gaia_xp_show_progress: bool = False,
        gaia_nearest_prefetch_factor: int = 50,
        gaia_nearest_prefetch_min: int = 200,
        gaia_nearest_prefetch_max: int = 10000,
        gaia_xp_min_sources: int = 25,
        gaia_xp_max_radius_deg: float = 1.0,
        gaia_xp_grow_factor: float = 1.5,
    ) -> pd.DataFrame:
        out_path = Path(out_csv)
        if out_path.exists():
            cached_df = pd.read_csv(out_path)
            if self._catalog_cache_usable(
                cached_df,
                self._meta_path(out_path),
                ra_deg,
                dec_deg,
                radius_deg,
                min_sources=gaia_xp_min_sources,
                max_radius_deg=gaia_xp_max_radius_deg,
            ):
                self.logger.info(
                    "Existing Gaia custom catalog found at %s; using cached file instead of re-downloading.",
                    out_path,
                )
                return cached_df
            self.logger.info(
                "Cached catalog at %s has %d row(s) and does not cover the current "
                "query settings; rebuilding with an expanded cone.",
                out_path,
                len(cached_df),
            )

        band_to_curve_path: Dict[str, Path] = {
            b: Path(p).expanduser().resolve() for b, p in curves.items()
        }
        if svo_filters:
            cache_dir = Path(curve_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            for band, svo_id in svo_filters.items():
                out = cache_dir / f"{band}_{svo_id.replace('/', '_')}.dat"
                self.logger.info(
                    "Downloading SVO filter %s for band %s ...", svo_id, band
                )
                retry_with_backoff(
                    lambda: self.download_svo_curve(
                        svo_id, out, timeout_sec=timeout_sec
                    ),
                    max_retries=gaia_archive_max_retries,
                    base_delay_sec=gaia_archive_retry_base_delay_sec,
                    logger=self.logger,
                    op_name=f"SVO filter download {svo_id}",
                )
                band_to_curve_path[band] = out

        if not band_to_curve_path:
            raise ValueError("No passbands provided (curves and/or svo_filters).")

        loaded_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for band, path in band_to_curve_path.items():
            wl_raw, tr = self.read_curve(path)
            curve_nm, wl_note = self.curve_wavelength_axis_to_nm(wl_raw)
            loaded_curves[band] = (curve_nm, tr)
            self.logger.info(
                "Loaded curve %s: %.1f-%.1f nm on XP grid (%s), %s",
                band,
                float(np.nanmin(curve_nm)),
                float(np.nanmax(curve_nm)),
                wl_note,
                path,
            )

        sql_top, sort_by_distance = gaia_xp_sql_top_n(
            max_sources,
            gaia_xp_order_by,
            prefetch_factor=gaia_nearest_prefetch_factor,
            prefetch_min=gaia_nearest_prefetch_min,
            prefetch_max=gaia_nearest_prefetch_max,
        )
        gaia, used_radius_deg = query_gaia_xp_cone_growing(
            ra_deg,
            dec_deg,
            radius_deg,
            sql_top,
            include_bp_rp=False,
            min_sources=gaia_xp_min_sources,
            max_radius_deg=gaia_xp_max_radius_deg,
            grow_factor=gaia_xp_grow_factor,
            pause_before_sec=gaia_query_pause_before_sec,
            pause_after_sec=gaia_query_pause_after_sec,
            max_retries=gaia_archive_max_retries,
            retry_base_delay_sec=gaia_archive_retry_base_delay_sec,
            logger=self.logger,
            op_name="Gaia ADQL (XP sources)",
        )
        if sort_by_distance and not gaia.empty:
            self.logger.info(
                "Gaia TAP does not support ORDER BY sky distance; fetched TOP %d rows, sorting in Python to nearest %d ...",
                sql_top,
                max_sources,
            )
            gaia = sort_gaia_table_nearest_to_target(
                gaia, ra_deg, dec_deg, max_rows=max_sources
            )
        if not gaia.empty and "source_id" in gaia.columns:
            gaia = gaia.copy()
            gaia["_sid_key"] = gaia["source_id"].map(_format_gaia_source_id)
        if gaia.empty:
            self.logger.warning("No Gaia sources found.")
            out_df = pd.DataFrame(columns=["name", "ra", "dec"])
            self._write_outputs(
                out_path,
                out_df,
                ra_deg,
                dec_deg,
                radius_deg,
                used_radius_deg,
                gaia_xp_max_radius_deg,
            )
            return out_df

        source_ids = [s for s in gaia["_sid_key"].tolist() if s]
        if len(source_ids) != len(gaia):
            self.logger.warning(
                "After formatting, %d / %d Gaia rows have empty source_id (skipped for GaiaXPy).",
                len(gaia) - len(source_ids),
                len(gaia),
            )
        # Report the archive-capped request size, not the configured
        # batch size: the batched call clamps to the same limit.
        eff_bs = _gaiaxpy_chunk_size(
            gaia_xp_batch_size, len(source_ids), None
        )
        n_batches = max(1, (len(source_ids) + eff_bs - 1) // eff_bs) if source_ids else 0
        self.logger.info(
            "Calibrating Gaia XP spectra for %d sources (%d batch(es), batch_size=%d, inter_batch_pause=%.2fs) ...",
            len(source_ids),
            n_batches,
            eff_bs,
            gaia_xp_batch_pause_sec,
        )
        spectra_df, xp_sampling_nm = calibrate_source_ids_batched(
            source_ids,
            batch_size=gaia_xp_batch_size,
            inter_batch_pause_sec=gaia_xp_batch_pause_sec,
            max_retries=gaia_archive_max_retries,
            retry_base_delay_sec=gaia_archive_retry_base_delay_sec,
            logger=self.logger,
            show_progress=gaia_xp_show_progress,
        )
        if xp_sampling_nm is not None:
            self.logger.info(
                "GaiaXPy wavelength sampling grid: %d points (nm).",
                int(np.asarray(xp_sampling_nm).shape[0]),
            )
        elif not spectra_df.empty:
            self.logger.warning(
                "GaiaXPy did not return a separate wavelength sampling\n"
                "    array; expecting wavelength columns inside each row.\n"
                "    If the catalog is empty, update autophot_gaia_curves\n"
                "    or inspect calibrate() output."
            )

        if "source_id" not in spectra_df.columns:
            raise KeyError("GaiaXPy calibrate output missing source_id column.")

        rows: List[dict] = []
        # Index the ADQL rows once so the per-source join is O(1) instead of
        # a full-frame scan for every calibrated spectrum.
        gaia_by_sid = gaia.drop_duplicates("_sid_key", keep="first").set_index(
            "_sid_key", drop=False
        )
        join_miss_logged = 0
        extract_fail_logged = 0

        # Stack all spectra onto the shared grid once; each band then needs a
        # single matrix product instead of an interp + trapz per source.
        mat = self._spectra_matrix(spectra_df, xp_sampling_nm)
        band_results: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        row_ok = np.zeros(len(spectra_df), dtype=bool)
        if mat is not None:
            sids, grid_nm, flux_mat, err_mat, row_ok = mat
            band_results = {
                band: self._ab_mags_from_matrix(
                    grid_nm, flux_mat, err_mat, row_ok, cw, ct
                )
                for band, (cw, ct) in loaded_curves.items()
            }
        else:
            sids = [
                _format_gaia_source_id(v)
                for v in spectra_df["source_id"].tolist()
            ]

        n_rows_tbl = len(sids)
        idx_iter = range(n_rows_tbl)
        if gaia_xp_show_progress and n_rows_tbl > 0:
            try:
                from tqdm.auto import tqdm

                idx_iter = tqdm(
                    idx_iter,
                    total=n_rows_tbl,
                    desc="Band-integrate spectra",
                    unit="src",
                )
            except ImportError:
                pass
        for i in idx_iter:
            sid = sids[i]
            if not sid:
                continue
            try:
                base = gaia_by_sid.loc[sid]
            except KeyError:
                if join_miss_logged < 3:
                    self.logger.warning(
                        "No ADQL row for GaiaXPy source_id=%s (join mismatch).",
                        sid,
                    )
                    join_miss_logged += 1
                continue

            rec = {"name": sid, "ra": float(base["ra"]), "dec": float(base["dec"])}
            if band_results is not None and row_ok[i]:
                for band in loaded_curves:
                    mag_arr, err_arr = band_results[band]
                    e = float(err_arr[i])
                    rec[band] = float(mag_arr[i])
                    rec[f"{band}_err"] = 0.01 if not math.isfinite(e) else e
            else:
                # Rows that could not stack keep the per-row column scan.
                try:
                    wavelength_nm, flux_w_m2_nm, flux_err_w_m2_nm = (
                        self.extract_spectrum_columns(
                            spectra_df.iloc[i], wavelength_nm=xp_sampling_nm
                        )
                    )
                except Exception as exc:
                    if extract_fail_logged < 5:
                        self.logger.warning(
                            "Skipping source %s during spectrum extract: %s",
                            sid,
                            exc,
                        )
                        extract_fail_logged += 1
                    else:
                        self.logger.debug("Skipping source %s: %s", sid, exc)
                    continue
                self._fill_band_mags(
                    rec,
                    wavelength_nm,
                    flux_w_m2_nm,
                    flux_err_w_m2_nm,
                    loaded_curves,
                )
            rows.append(rec)

        if not rows:
            cols = ["name", "ra", "dec"]
            for b in loaded_curves:
                cols.append(b)
                cols.append(f"{b}_err")
            out_df = pd.DataFrame(columns=cols)
            if not spectra_df.empty:
                self.logger.warning(
                    "Gaia curve-map catalog has 0 usable rows after band\n"
                    "    integration (%d calibrated sources skipped).\n"
                    "    spectra_df columns=%s; xp_sampling_nm=%s.\n"
                    "    Check XP flux column names, wavelength grid, and\n"
                    "    filter overlap.",
                    len(spectra_df),
                    list(spectra_df.columns),
                    "None"
                    if xp_sampling_nm is None
                    else f"{int(np.asarray(xp_sampling_nm).shape[0])} nm samples",
                )
        else:
            out_df = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Validate (warning-only): non-empty but all-NaN magnitudes is almost
        # always a wavelength-unit/overlap problem (or a GaiaXPy output-format
        # change). We warn loudly but do not raise, since some workflows prefer
        # to proceed and handle missing mags downstream.
        # ------------------------------------------------------------------
        if not out_df.empty and loaded_curves:
            n = int(len(out_df))
            bad_bands = []
            for b in loaded_curves.keys():
                if b not in out_df.columns:
                    bad_bands.append(f"{b} (missing column)")
                    continue
                finite = np.isfinite(pd.to_numeric(out_df[b], errors="coerce").values)
                n_finite = int(np.count_nonzero(finite))
                if n_finite == 0:
                    bad_bands.append(f"{b} (0/{n} finite)")
            if len(bad_bands) == len(list(loaded_curves.keys())):
                self.logger.warning(
                    "Gaia curve-map catalog build produced %d row(s) but no\n"
                    "    finite synthetic magnitudes for any requested\n"
                    "    band(s): %s.\n"
                    "    Likely causes: (1) curve wavelength units\n"
                    "    misinterpreted (A vs nm), (2) no overlap with Gaia\n"
                    "    XP spectra (~330-1050 nm), (3) GaiaXPy output\n"
                    "    format changed (flux column names).",
                    n,
                    ", ".join(bad_bands),
                )
        if not out_df.empty and len(out_df) < max(1, int(gaia_xp_min_sources)):
            self.logger.warning(
                "Gaia curve-map catalog holds only %d source(s) after XP\n"
                "    calibration within a %.4f-deg cone (target >= %d).\n"
                "    Zeropoints on so few stars are fragile; for survey-covered\n"
                "    bands prefer 'refcat'/'pan_starrs', or raise\n"
                "    catalog.gaia_xp_max_radius_deg.",
                len(out_df),
                used_radius_deg,
                int(gaia_xp_min_sources),
            )
        self._write_outputs(
            out_path,
            out_df,
            ra_deg,
            dec_deg,
            radius_deg,
            used_radius_deg,
            gaia_xp_max_radius_deg,
        )
        self.logger.debug("Curve-map catalog written: %d rows -> %s", len(out_df), out_path)
        return out_df


def build_custom_catalog(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    max_sources: int,
    curves: Dict[str, str | Path],
    out_csv: str | Path,
    svo_filters: Optional[Dict[str, str]] = None,
    curve_cache_dir: str | Path = "svo_curves",
    timeout_sec: int = 60,
    gaia_query_pause_before_sec: float = 0.25,
    gaia_query_pause_after_sec: float = 0.25,
    gaia_xp_batch_size: int = 200,
    gaia_xp_batch_pause_sec: float = 0.5,
    gaia_archive_max_retries: int = 3,
    gaia_archive_retry_base_delay_sec: float = 2.0,
    gaia_xp_order_by: str = "distance",
    gaia_xp_show_progress: bool = False,
    gaia_nearest_prefetch_factor: int = 50,
    gaia_nearest_prefetch_min: int = 200,
    gaia_nearest_prefetch_max: int = 10000,
    gaia_xp_min_sources: int = 25,
    gaia_xp_max_radius_deg: float = 1.0,
    gaia_xp_grow_factor: float = 1.5,
    log_level: int = logging.INFO,
) -> pd.DataFrame:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        from ..core.functions import cap_console_lines
    except ImportError:
        pass
    else:
        cap_console_lines(logging.getLogger().handlers, extra_reserve=26)
    builder = GaiaCurveCatalogBuilder()
    return builder.build(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_deg=radius_deg,
        max_sources=max_sources,
        curves=curves,
        out_csv=out_csv,
        svo_filters=svo_filters,
        curve_cache_dir=curve_cache_dir,
        timeout_sec=timeout_sec,
        gaia_query_pause_before_sec=gaia_query_pause_before_sec,
        gaia_query_pause_after_sec=gaia_query_pause_after_sec,
        gaia_xp_batch_size=gaia_xp_batch_size,
        gaia_xp_batch_pause_sec=gaia_xp_batch_pause_sec,
        gaia_archive_max_retries=gaia_archive_max_retries,
        gaia_archive_retry_base_delay_sec=gaia_archive_retry_base_delay_sec,
        gaia_xp_order_by=gaia_xp_order_by,
        gaia_xp_show_progress=gaia_xp_show_progress,
        gaia_nearest_prefetch_factor=gaia_nearest_prefetch_factor,
        gaia_nearest_prefetch_min=gaia_nearest_prefetch_min,
        gaia_nearest_prefetch_max=gaia_nearest_prefetch_max,
        gaia_xp_min_sources=gaia_xp_min_sources,
        gaia_xp_max_radius_deg=gaia_xp_max_radius_deg,
        gaia_xp_grow_factor=gaia_xp_grow_factor,
    )

