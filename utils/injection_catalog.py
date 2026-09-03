"""Injection catalog for AutoPhOT artificial-source injection/recovery.

Provides a structured record of every injection trial, including:
- injection position (x, y)
- true injected flux
- recovered flux and uncertainty
- detection flag and detection criteria
- quality flags
- random seed / provenance
- background and crowding metrics at injection site

This catalog can be saved to disk (CSV/FITS) for:
- external validation of completeness curves
- re-analysis with different detection thresholds
- uncertainty calibration (comparing reported errors to empirical scatter)
- audit trails for published results

Scientific motivation:  Without a persistent injection catalog, the
injection/recovery results cannot be independently verified.  The catalog
makes the validation reproducible and auditable.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class InjectionRecord:
    """Record of a single injection/recovery trial.

    Attributes
    ----------
    trial_id : int
        Unique trial identifier.
    magnitude : float
        Injected instrumental magnitude.
    x_inj, y_inj : float
        Injection position (pixels).
    flux_true : float
        True injected flux (e-/s).
    flux_recovered : float
        Recovered flux (e-/s).
    flux_err : float
        Reported flux uncertainty (e-/s).
    detected : bool
        Whether the source was detected.
    detection_snr : float
        Detection S/N.
    detection_method : str
        Recovery method used ("AP", "PSF", "EMCEE").
    background_rms : float
        Local background RMS at injection site.
    crowding_metric : float
        Local source density or crowding score.
    seed : int
        Random seed used for this trial.
    quality_flags : int
        Quality flags bitmask.
    """

    trial_id: int = 0
    magnitude: float = np.nan
    x_inj: float = np.nan
    y_inj: float = np.nan
    flux_true: float = np.nan
    flux_recovered: float = np.nan
    flux_err: float = np.nan
    detected: bool = False
    detection_snr: float = np.nan
    detection_method: str = ""
    background_rms: float = np.nan
    crowding_metric: float = np.nan
    seed: int = 0
    quality_flags: int = 0


class InjectionCatalog:
    """Catalog of injection/recovery trials.

    Stores per-trial records and provides methods for:
    - adding records
    - saving to CSV/FITS
    - computing summary statistics
    - uncertainty calibration analysis
    """

    def __init__(self, seed: int | None = None, recovery_method: str = ""):
        self.records: list[InjectionRecord] = []
        self.seed = seed
        self.recovery_method = recovery_method
        self._next_id = 0

    def add_record(
        self,
        magnitude: float,
        x_inj: float,
        y_inj: float,
        flux_true: float,
        flux_recovered: float,
        flux_err: float,
        detected: bool,
        detection_snr: float = np.nan,
        background_rms: float = np.nan,
        crowding_metric: float = np.nan,
        quality_flags: int = 0,
    ) -> InjectionRecord:
        """Add a single injection/recovery record."""
        record = InjectionRecord(
            trial_id=self._next_id,
            magnitude=magnitude,
            x_inj=x_inj,
            y_inj=y_inj,
            flux_true=flux_true,
            flux_recovered=flux_recovered,
            flux_err=flux_err,
            detected=detected,
            detection_snr=detection_snr,
            detection_method=self.recovery_method,
            background_rms=background_rms,
            crowding_metric=crowding_metric,
            seed=self.seed if self.seed is not None else 0,
            quality_flags=quality_flags,
        )
        self.records.append(record)
        self._next_id += 1
        return record

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=[
                "trial_id", "magnitude", "x_inj", "y_inj",
                "flux_true", "flux_recovered", "flux_err",
                "detected", "detection_snr", "detection_method",
                "background_rms", "crowding_metric", "seed", "quality_flags",
            ])
        return pd.DataFrame([dataclasses.asdict(r) for r in self.records])

    def save_csv(self, path: str | Path) -> None:
        """Save catalog to CSV file."""
        df = self.to_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.6f")
        log.info("Saved injection catalog (%d trials) to %s", len(df), path)

    def summary_statistics(self) -> dict:
        """Compute summary statistics for the catalog.

        Returns
        -------
        dict with keys:
            n_total : int
            n_detected : int
            detection_rate : float
            flux_bias_frac : float (median (F_rec - F_true) / F_true for detected)
            flux_scatter_frac : float (MAD-based scatter)
            magnitude_range : (min, max)
            snr_median : float
        """
        df = self.to_dataframe()
        if df.empty:
            return {
                "n_total": 0,
                "n_detected": 0,
                "detection_rate": np.nan,
                "flux_bias_frac": np.nan,
                "flux_scatter_frac": np.nan,
                "magnitude_range": (np.nan, np.nan),
                "snr_median": np.nan,
            }

        n_total = len(df)
        n_detected = int(df["detected"].sum())
        detection_rate = n_detected / n_total if n_total > 0 else np.nan

        detected = df[df["detected"] & np.isfinite(df["flux_true"]) & (df["flux_true"] != 0)]
        if len(detected) > 0:
            bias = (detected["flux_recovered"] - detected["flux_true"]) / detected["flux_true"]
            flux_bias = float(np.nanmedian(bias))
            flux_scatter = float(1.4826 * np.nanmedian(np.abs(bias - np.nanmedian(bias))))
        else:
            flux_bias = np.nan
            flux_scatter = np.nan

        mags = df["magnitude"]
        mag_range = (float(np.nanmin(mags)), float(np.nanmax(mags))) if mags.notna().any() else (np.nan, np.nan)

        snr_med = float(np.nanmedian(df["detection_snr"])) if df["detection_snr"].notna().any() else np.nan

        return {
            "n_total": n_total,
            "n_detected": n_detected,
            "detection_rate": detection_rate,
            "flux_bias_frac": flux_bias,
            "flux_scatter_frac": flux_scatter,
            "magnitude_range": mag_range,
            "snr_median": snr_med,
        }

    def completeness_curve(
        self,
        mag_bins: np.ndarray | None = None,
        n_bins: int = 20,
    ) -> pd.DataFrame:
        """Compute completeness curve (detection rate vs. magnitude).

        Parameters
        ----------
        mag_bins : array, optional
            Magnitude bin edges.  If None, auto-computed.
        n_bins : int
            Number of bins if mag_bins is None.

        Returns
        -------
        pd.DataFrame with columns: mag_center, n_total, n_detected, completeness, completeness_err
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["mag_center", "n_total", "n_detected", "completeness", "completeness_err"])

        mags = df["magnitude"].dropna()
        if mag_bins is None:
            mag_min, mag_max = mags.min(), mags.max()
            if mag_min == mag_max:
                mag_bins = np.array([mag_min - 0.5, mag_max + 0.5])
            else:
                mag_bins = np.linspace(mag_min, mag_max, n_bins + 1)

        centers = []
        n_total_list = []
        n_detected_list = []
        completeness_list = []
        completeness_err_list = []

        for i in range(len(mag_bins) - 1):
            lo, hi = mag_bins[i], mag_bins[i + 1]
            mask = (df["magnitude"] >= lo) & (df["magnitude"] < hi)
            n_tot = int(mask.sum())
            n_det = int(df.loc[mask, "detected"].sum())
            comp = n_det / n_tot if n_tot > 0 else np.nan
            # Binomial error
            err = np.sqrt(comp * (1 - comp) / n_tot) if n_tot > 0 else np.nan
            centers.append((lo + hi) / 2)
            n_total_list.append(n_tot)
            n_detected_list.append(n_det)
            completeness_list.append(comp)
            completeness_err_list.append(err)

        return pd.DataFrame({
            "mag_center": centers,
            "n_total": n_total_list,
            "n_detected": n_detected_list,
            "completeness": completeness_list,
            "completeness_err": completeness_err_list,
        })

    def uncertainty_calibration_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract data for uncertainty calibration.

        Returns
        -------
        (flux_true, flux_measured, flux_error) arrays for detected sources
        with finite values.
        """
        df = self.to_dataframe()
        valid = (
            df["detected"]
            & np.isfinite(df["flux_true"])
            & np.isfinite(df["flux_recovered"])
            & np.isfinite(df["flux_err"])
            & (df["flux_true"] != 0)
            & (df["flux_err"] > 0)
        )
        sub = df[valid]
        return (
            sub["flux_true"].to_numpy(),
            sub["flux_recovered"].to_numpy(),
            sub["flux_err"].to_numpy(),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"InjectionCatalog(n_trials={len(self.records)}, method={self.recovery_method!r})"
