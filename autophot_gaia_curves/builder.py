from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from autophot_gaia_curves.gaia_archive import (
    calibrate_source_ids_batched,
    gaia_xp_source_query,
    gaia_xp_sql_top_n,
    launch_gaia_adql_to_pandas,
    sort_gaia_table_nearest_to_target,
)


def _format_gaia_source_id(val: Any) -> str:
    """
    Stable string ID for cross-matching TAP rows to GaiaXPy output.

    Gaia ``source_id`` is a 64-bit integer; if it becomes ``float64`` in pandas,
    string conversion can round and break joins — prefer integer paths.
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
        Map filter curve wavelengths to nanometres for overlap with Gaia XP (≈330–1050 nm).

        Files may list **Angstrom** (typical SVO / instrument: ~3000–11000 Å) or **nm**
        (~300–1100). Heuristic: if max wavelength > 2500, treat values as Å and divide
        by 10; otherwise assume they are already nm. Mis-classifying Å as nm was the
        main cause of all-NaN synthetic mags (no spectral overlap after scaling).
        """
        wl_raw = np.asarray(wl_raw, dtype=float)
        w_max = float(np.nanmax(wl_raw))
        w_min = float(np.nanmin(wl_raw))
        if w_max > 2500.0:
            curve_nm = wl_raw / 10.0
            note = f"wavelengths interpreted as Angstrom (raw {w_min:.1f}–{w_max:.1f} Å)"
        else:
            curve_nm = wl_raw.copy()
            note = f"wavelengths interpreted as nanometres ({w_min:.1f}–{w_max:.1f} nm)"
        return curve_nm, note

    @staticmethod
    def read_curve(curve_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a curve file with two numeric columns:
            wavelength (Angstrom **or** nm — see :meth:`curve_wavelength_axis_to_nm`),
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
        wave_keys = ("wavelength", "wavelength_nm", "wl", "lambda")
        flux_keys = (
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

        wave = None
        flux = None
        for k in wave_keys:
            if k in row and row[k] is not None:
                wave = np.asarray(row[k], dtype=float)
                break
        if wave is None and wavelength_nm is not None:
            wave = np.asarray(wavelength_nm, dtype=float)
        for k in flux_keys:
            if k in row and row[k] is not None:
                flux = np.asarray(row[k], dtype=float)
                break

        # GaiaXPy column names vary by version: pick any 1-D numeric column that
        # matches the calibrated wavelength grid length (skip obvious non-flux).
        if flux is None and wavelength_nm is not None:
            nexp = int(np.asarray(wavelength_nm).shape[0])
            skip = {
                "source_id",
                "ra",
                "dec",
                "designation",
                "flux_error",
                "flux_uncertainty",
                "correlation",
            }
            matched: List[Tuple[str, np.ndarray]] = []
            for k in row.index:
                kl = str(k).lower()
                if k in skip or kl.endswith("_error") or kl.endswith("_uncertainty"):
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

            def _flux_col_score(name: str) -> int:
                n = name.lower()
                if "error" in n or "uncert" in n or "corr" in n:
                    return 200
                if n == "flux":
                    return 0
                if "flux" in n:
                    return 2
                return 10

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
        err_keys = (
            "flux_error",
            "flux_uncertainty",
            "sigma_flux",
            "calibrated_flux_error",
            "flux_err",
        )
        for ek in err_keys:
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
        for j in range(1, n - 1):
            c[j] = 0.5 * (x[j + 1] - x[j - 1])
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
        AB magnitude and approximate 1σ magnitude error from band integration.

        Uses GaiaXPy ``flux_error`` (W m^-2 nm^-1), propagated through the same f_ν
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
        c_light = 299792458.0
        f_nu = fl_m * (wl_m**2) / c_light

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

        f_nu_ab0 = 3631e-26
        mag = -2.5 * np.log10(f_nu_band / f_nu_ab0)

        mag_err = np.nan
        if flux_err_w_m2_nm is not None:
            sig_fl = flux_err_w_m2_nm[valid]
            sig_fn = sig_fl * 1e9 * (wl_m**2) / c_light
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
        gaia_query_pause_before_sec: float = 1.0,
        gaia_query_pause_after_sec: float = 1.0,
        gaia_xp_batch_size: int = 200,
        gaia_xp_batch_pause_sec: float = 1.0,
        gaia_archive_max_retries: int = 3,
        gaia_archive_retry_base_delay_sec: float = 2.0,
        gaia_xp_order_by: str = "distance",
        gaia_xp_show_progress: bool = True,
        gaia_nearest_prefetch_factor: int = 50,
        gaia_nearest_prefetch_min: int = 500,
        gaia_nearest_prefetch_max: int = 10000,
    ) -> pd.DataFrame:
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
                self.download_svo_curve(svo_id, out, timeout_sec=timeout_sec)
                band_to_curve_path[band] = out

        if not band_to_curve_path:
            raise ValueError("No passbands provided (curves and/or svo_filters).")

        loaded_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for band, path in band_to_curve_path.items():
            wl_raw, tr = self.read_curve(path)
            curve_nm, wl_note = self.curve_wavelength_axis_to_nm(wl_raw)
            loaded_curves[band] = (curve_nm, tr)
            self.logger.info(
                "Loaded curve for %s from %s — %s; on XP grid → %.1f–%.1f nm.",
                band,
                path,
                wl_note,
                float(np.nanmin(curve_nm)),
                float(np.nanmax(curve_nm)),
            )

        sql_top, sort_by_distance = gaia_xp_sql_top_n(
            max_sources,
            gaia_xp_order_by,
            prefetch_factor=gaia_nearest_prefetch_factor,
            prefetch_min=gaia_nearest_prefetch_min,
            prefetch_max=gaia_nearest_prefetch_max,
        )
        adql = gaia_xp_source_query(
            ra_deg,
            dec_deg,
            radius_deg,
            sql_top,
            include_bp_rp=False,
        )
        gaia = launch_gaia_adql_to_pandas(
            adql,
            pause_before_sec=gaia_query_pause_before_sec,
            pause_after_sec=gaia_query_pause_after_sec,
            max_retries=gaia_archive_max_retries,
            retry_base_delay_sec=gaia_archive_retry_base_delay_sec,
            logger=self.logger,
            op_name="Gaia ADQL (XP sources)",
        )
        if sort_by_distance and not gaia.empty:
            self.logger.info(
                "Gaia TAP does not support ORDER BY sky distance; "
                "fetched TOP %d rows, sorting in Python to nearest %d ...",
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
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_csv, index=False)
            return out_df

        source_ids = [s for s in gaia["_sid_key"].tolist() if s]
        if len(source_ids) != len(gaia):
            self.logger.warning(
                "After formatting, %d / %d Gaia rows have empty source_id (skipped "
                "for GaiaXPy).",
                len(gaia) - len(source_ids),
                len(gaia),
            )
        eff_bs = (
            gaia_xp_batch_size
            if gaia_xp_batch_size > 0
            else max(len(source_ids), 1)
        )
        n_batches = max(1, (len(source_ids) + eff_bs - 1) // eff_bs) if source_ids else 0
        self.logger.info(
            "Calibrating Gaia XP spectra for %d sources (%d batch(es), "
            "batch_size=%s, inter_batch_pause=%.2fs) ...",
            len(source_ids),
            n_batches,
            gaia_xp_batch_size if gaia_xp_batch_size > 0 else "all",
            gaia_xp_batch_pause_sec,
        )
        spectra_df, xp_sampling_nm = calibrate_source_ids_batched(
            source_ids,
            batch_size=gaia_xp_batch_size,
            inter_batch_pause_sec=gaia_xp_batch_pause_sec,
            max_retries=gaia_archive_max_retries,
            retry_base_delay_sec=gaia_archive_retry_base_delay_sec,
            logger=self.logger,
        )
        if xp_sampling_nm is not None:
            self.logger.info(
                "GaiaXPy wavelength sampling grid: %d points (nm).",
                int(np.asarray(xp_sampling_nm).shape[0]),
            )
        elif not spectra_df.empty:
            self.logger.warning(
                "GaiaXPy did not return a separate wavelength sampling array; "
                "expecting wavelength columns inside each row. If the catalog is "
                "empty, update autophot_gaia_curves or inspect calibrate() output."
            )

        if "source_id" not in spectra_df.columns:
            raise KeyError("GaiaXPy calibrate output missing source_id column.")

        rows: List[dict] = []
        row_iter = spectra_df.iterrows()
        if gaia_xp_show_progress and len(spectra_df) > 0:
            try:
                from tqdm.auto import tqdm

                row_iter = tqdm(
                    row_iter,
                    total=len(spectra_df),
                    desc="Band-integrate spectra",
                    unit="src",
                )
            except ImportError:
                pass
        join_miss_logged = 0
        extract_fail_logged = 0
        for _, row in row_iter:
            sid = _format_gaia_source_id(row["source_id"])
            if not sid:
                continue
            match = gaia.loc[gaia["_sid_key"] == sid]
            if match.empty:
                if join_miss_logged < 3:
                    self.logger.warning(
                        "No ADQL row for GaiaXPy source_id=%s (join mismatch).",
                        sid,
                    )
                    join_miss_logged += 1
                continue
            base = match.iloc[0]
            try:
                wavelength_nm, flux_w_m2_nm, flux_err_w_m2_nm = (
                    self.extract_spectrum_columns(
                        row, wavelength_nm=xp_sampling_nm
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

            rec = {"name": sid, "ra": float(base["ra"]), "dec": float(base["dec"])}
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
                rec[f"{band}_err"] = (
                    0.01 if not math.isfinite(mag_err) else mag_err
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
                    "Gaia curve-map catalog has 0 usable rows after band "
                    "integration (%d calibrated sources skipped). "
                    "spectra_df columns=%s; xp_sampling_nm=%s. "
                    "Check XP flux column names, wavelength grid, and filter overlap.",
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
                    "Gaia curve-map catalog build produced %d row(s) but no finite "
                    "synthetic magnitudes for any requested band(s): %s. "
                    "Likely causes: (1) curve wavelength units misinterpreted (Å vs nm), "
                    "(2) no overlap with Gaia XP spectra (~330–1050 nm), "
                    "(3) GaiaXPy output format changed (flux column names).",
                    n,
                    ", ".join(bad_bands),
                )
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        self.logger.info("Wrote %d rows to %s", len(out_df), out_path)
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
    gaia_query_pause_before_sec: float = 1.0,
    gaia_query_pause_after_sec: float = 1.0,
    gaia_xp_batch_size: int = 200,
    gaia_xp_batch_pause_sec: float = 1.0,
    gaia_archive_max_retries: int = 3,
    gaia_archive_retry_base_delay_sec: float = 2.0,
    gaia_xp_order_by: str = "distance",
    gaia_xp_show_progress: bool = True,
    gaia_nearest_prefetch_factor: int = 50,
    gaia_nearest_prefetch_min: int = 500,
    gaia_nearest_prefetch_max: int = 10000,
    log_level: int = logging.INFO,
) -> pd.DataFrame:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
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
    )

