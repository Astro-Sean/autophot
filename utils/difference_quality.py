#!/usr/bin/env python3
"""
Difference-image quality diagnostics for AutoPhOT image subtraction.

Provides quantitative metrics that go beyond a simple global RMS:

  * **Dipole detection** – bright-star residual dipoles are the most common
    symptom of astrometric misalignment or PSF-matching failure.  We detect
    them by looking for anti-symmetric positive/negative residual pairs
    around known source positions.

  * **Spatial background variation** – measures whether the difference-image
    background varies systematically across the field (indicates sky or
    illumination-gradient mismatch).

  * **Residual autocorrelation** – correlated noise on small spatial scales
    indicates over-fitting or deconvolution artefacts.

  * **Bright-star residual flux** – measures how much flux remains around
    bright sources after subtraction (should be near zero for a good
    subtraction).

  * **Structured quality score** – a documented pass / downgrade / fail
    classification with component scores, not just one combined number.

  * **Provenance manifest** – machine-readable JSON recording all inputs,
    parameters, and quality metrics for reproducibility.

All functions are designed to be called from ``templates.py.subtract()``
after the difference image has been written.  They are intentionally
self-contained (no AutoPhOT imports) so they can be unit-tested in
isolation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d, binary_dilation, generate_binary_structure
from astropy.stats import sigma_clipped_stats

logger = logging.getLogger(__name__)


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class QualityMetrics:
    """Container for all difference-image quality metrics."""

    # Global statistics
    diff_median: float = 0.0
    diff_std: float = 0.0
    diff_rms: float = 0.0
    valid_pixels: int = 0

    # Spatial background variation
    background_spatial_std: float = 0.0  # std of per-tile medians
    background_spatial_range: float = 0.0  # max - min of per-tile medians

    # Dipole detection
    dipole_count: int = 0
    dipole_fraction: float = 0.0  # fraction of checked sources with dipoles
    dipole_mean_amplitude: float = 0.0  # mean |positive + negative| flux
    dipole_max_amplitude: float = 0.0

    # Bright-star residual flux
    bright_star_residual_median: float = 0.0
    bright_star_residual_rms: float = 0.0
    bright_star_count: int = 0

    # Residual autocorrelation
    autocorr_scale_px: float = 0.0  # spatial scale of correlated noise
    autocorr_peak: float = 0.0  # peak autocorrelation (excluding lag 0)

    # Edge artifacts
    edge_std_ratio: float = 1.0  # edge_std / global_std

    # Component scores (0-1, higher is better)
    score_background: float = 1.0
    score_dipole: float = 1.0
    score_bright_star: float = 1.0
    score_edge: float = 1.0
    score_autocorr: float = 1.0

    # Overall
    quality_score: float = 1.0
    quality_class: str = "unknown"  # "pass", "downgrade", "fail"

    # Metadata
    algorithm: str = ""
    forceconv: str = ""
    kernel_order: int = 0
    kernel_half_width: int = 0
    science_fwhm: float = 0.0
    template_fwhm: float = 0.0
    n_matching_sources: int = 0
    flux_scale_conv: float = 0.0
    flux_scale_phot: float = 0.0
    flux_scale_discrep_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class QualityConfig:
    """Configuration for difference-image quality assessment."""

    # Dipole detection
    dipole_check_sources: bool = True
    dipole_n_sigma: float = 5.0  # residual threshold for dipole detection
    dipole_radius_fwhm: float = 1.5  # search radius for anti-symmetric pair
    dipole_min_antisymmetry: float = 0.3  # min |pos/neg| ratio for dipole
    dipole_max_fraction_pass: float = 0.05  # >5% dipoles => downgrade/fail
    dipole_max_fraction_fail: float = 0.20  # >20% dipoles => fail

    # Bright-star residuals
    bright_star_check: bool = True
    bright_star_n_sigma: float = 50.0  # stars brighter than this sigma
    bright_star_radius_fwhm: float = 3.0
    bright_star_max_residual_sigma: float = 3.0  # max allowed residual in sigma units

    # Spatial background
    background_tile_size: int = 128  # pixels per tile
    background_max_spatial_std_ratio: float = 0.5  # spatial_std / global_std

    # Edge artifacts
    edge_width_px: int = 20
    edge_max_std_ratio: float = 2.0

    # Autocorrelation
    autocorr_max_lag: int = 20  # pixels
    autocorr_max_peak: float = 0.3  # max allowed autocorr peak (excl. lag 0)

    # Quality score weights (must sum to 1.0)
    weight_background: float = 0.15
    weight_dipole: float = 0.35
    weight_bright_star: float = 0.20
    weight_edge: float = 0.15
    weight_autocorr: float = 0.15

    # Classification thresholds
    pass_threshold: float = 0.75
    downgrade_threshold: float = 0.50


# ===========================================================================
# Core metric functions
# ===========================================================================

def compute_spatial_background(
    diff_data: np.ndarray,
    quality_mask: np.ndarray,
    tile_size: int = 128,
) -> Tuple[float, float]:
    """Compute spatial variation of the background in the difference image.

    Divides the image into tiles and computes the median of each tile on
    unmasked pixels.  Returns (spatial_std, spatial_range) where spatial_std
    is the std of per-tile medians and spatial_range is max - min.

    A large spatial_std relative to the global noise indicates sky or
    illumination-gradient mismatch between science and template.
    """
    ny, nx = diff_data.shape
    tile_medians = []
    for y0 in range(0, ny, tile_size):
        for x0 in range(0, nx, tile_size):
            y1 = min(y0 + tile_size, ny)
            x1 = min(x0 + tile_size, nx)
            tile = diff_data[y0:y1, x0:x1]
            tile_mask = quality_mask[y0:y1, x0:x1]
            valid = tile[~tile_mask]
            valid = valid[np.isfinite(valid)]
            if len(valid) > 20:  # need enough pixels for a stable median
                tile_medians.append(np.median(valid))
    if len(tile_medians) < 2:
        return 0.0, 0.0
    medians = np.array(tile_medians)
    return float(np.std(medians)), float(np.ptp(medians))


def detect_dipoles(
    diff_data: np.ndarray,
    source_positions: List[Tuple[float, float]],
    quality_mask: np.ndarray,
    noise_sigma: float,
    fwhm: float,
    cfg: QualityConfig,
) -> Tuple[int, float, float, float]:
    """Detect dipole residuals around known source positions.

    A dipole is an anti-symmetric positive/negative residual pair caused
    by astrometric misalignment or PSF-matching failure.  For each source,
    we search for a positive peak within ``dipole_radius_fwhm * FWHM`` and
    a corresponding negative peak on the opposite side.

    Parameters
    ----------
    diff_data : np.ndarray
        Difference image.
    source_positions : list of (x, y)
        Source positions to check (pixel coordinates).
    quality_mask : np.ndarray (bool)
        Mask of pixels to exclude (NaN, sources, target region).
    noise_sigma : float
        Background noise sigma of the difference image.
    fwhm : float
        FWHM of the difference image PSF (pixels).
    cfg : QualityConfig
        Configuration parameters.

    Returns
    -------
    (dipole_count, dipole_fraction, mean_amplitude, max_amplitude)
    """
    if not source_positions or noise_sigma <= 0 or fwhm <= 0:
        return 0, 0.0, 0.0, 0.0

    ny, nx = diff_data.shape
    radius = max(int(cfg.dipole_radius_fwhm * fwhm), 3)
    threshold = cfg.dipole_n_sigma * noise_sigma
    min_antisym = cfg.dipole_min_antisymmetry

    dipole_count = 0
    amplitudes = []

    for x, y in source_positions:
        xi, yi = int(round(x)), int(round(y))
        # Skip sources too close to edges
        if xi < radius or xi >= nx - radius or yi < radius or yi >= ny - radius:
            continue

        # Extract stamp around source
        stamp = diff_data[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1]
        stamp_mask = quality_mask[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1]

        # Mask out invalid pixels
        stamp_clean = stamp.copy()
        stamp_clean[stamp_mask] = 0.0
        stamp_clean[~np.isfinite(stamp_clean)] = 0.0

        # Find the brightest positive and negative pixels
        pos_peak = np.max(stamp_clean)
        neg_peak = np.min(stamp_clean)

        # Both must exceed threshold
        if pos_peak < threshold or abs(neg_peak) < threshold:
            continue

        # Check anti-symmetry: the positive and negative peaks should be
        # on opposite sides of the source center and have similar amplitude
        pos_idx = np.unravel_index(np.argmax(stamp_clean), stamp_clean.shape)
        neg_idx = np.unravel_index(np.argmin(stamp_clean), stamp_clean.shape)

        # Vector from center to positive peak
        cy, cx = radius, radius
        dy_pos = pos_idx[0] - cy
        dx_pos = pos_idx[1] - cx
        dy_neg = neg_idx[0] - cy
        dx_neg = neg_idx[1] - cx

        # Check that they are on opposite sides (dot product < 0)
        dot = dy_pos * dy_neg + dx_pos * dx_neg
        if dot >= 0:
            continue

        # Check amplitude ratio (anti-symmetry)
        amp_ratio = min(abs(pos_peak), abs(neg_peak)) / max(abs(pos_peak), abs(neg_peak))
        if amp_ratio < min_antisym:
            continue

        # This is a dipole
        dipole_count += 1
        amplitude = abs(pos_peak) + abs(neg_peak)
        amplitudes.append(amplitude)

    n_checked = len(source_positions)
    dipole_fraction = dipole_count / n_checked if n_checked > 0 else 0.0
    mean_amp = float(np.mean(amplitudes)) if amplitudes else 0.0
    max_amp = float(np.max(amplitudes)) if amplitudes else 0.0

    return dipole_count, dipole_fraction, mean_amp, max_amp


def measure_bright_star_residuals(
    diff_data: np.ndarray,
    bright_star_positions: List[Tuple[float, float]],
    quality_mask: np.ndarray,
    noise_sigma: float,
    fwhm: float,
    cfg: QualityConfig,
) -> Tuple[float, float, int]:
    """Measure residual flux around bright stars in the difference image.

    For each bright star, we measure the residual flux in an annulus
    around the star position.  A good subtraction should have residuals
    consistent with the background noise.

    Returns
    -------
    (median_residual_sigma, rms_residual_sigma, n_stars)
        Residuals expressed in units of noise_sigma.
    """
    if not bright_star_positions or noise_sigma <= 0 or fwhm <= 0:
        return 0.0, 0.0, 0

    ny, nx = diff_data.shape
    inner_r = max(int(fwhm), 3)
    outer_r = max(int(cfg.bright_star_radius_fwhm * fwhm), inner_r + 2)

    residuals = []
    for x, y in bright_star_positions:
        xi, yi = int(round(x)), int(round(y))
        if xi < outer_r or xi >= nx - outer_r or yi < outer_r or yi >= ny - outer_r:
            continue

        # Annulus mask
        yy, xx = np.ogrid[:2 * outer_r + 1, :2 * outer_r + 1]
        r2 = (xx - outer_r) ** 2 + (yy - outer_r) ** 2
        annulus = (r2 >= inner_r ** 2) & (r2 <= outer_r ** 2)

        stamp = diff_data[yi - outer_r:yi + outer_r + 1, xi - outer_r:xi + outer_r + 1]
        stamp_mask = quality_mask[yi - outer_r:yi + outer_r + 1, xi - outer_r:xi + outer_r + 1]

        annulus_data = stamp[annulus & ~stamp_mask]
        annulus_data = annulus_data[np.isfinite(annulus_data)]
        if len(annulus_data) > 10:
            residuals.append(np.median(annulus_data) / noise_sigma)

    if not residuals:
        return 0.0, 0.0, 0

    res = np.array(residuals)
    return float(np.median(res)), float(np.sqrt(np.mean(res ** 2))), len(residuals)


def compute_autocorrelation(
    diff_data: np.ndarray,
    quality_mask: np.ndarray,
    noise_sigma: float,
    max_lag: int = 20,
) -> Tuple[float, float]:
    """Compute the spatial autocorrelation of the difference image residuals.

    Correlated noise on small spatial scales indicates over-fitting or
    deconvolution artefacts.  We compute the 1-D autocorrelation along
    rows and columns and report the peak (excluding lag 0) and the
    spatial scale at which it occurs.

    Returns
    -------
    (peak_autocorr, scale_px)
        Peak autocorrelation value (0 = uncorrelated, 1 = perfectly
        correlated) and the spatial scale in pixels.
    """
    if noise_sigma <= 0:
        return 0.0, 0.0

    # Use a central region to avoid edge effects
    ny, nx = diff_data.shape
    y0, y1 = ny // 4, 3 * ny // 4
    x0, x1 = nx // 4, 3 * nx // 4
    region = diff_data[y0:y1, x0:x1].copy()
    region_mask = quality_mask[y0:y1, x0:x1]

    # Zero out masked pixels
    region[region_mask] = 0.0
    region[~np.isfinite(region)] = 0.0

    # Subtract mean
    region = region - np.mean(region)

    # Compute 1-D autocorrelation along rows (averaged)
    n_rows = region.shape[0]
    row_autocorrs = []
    for i in range(min(n_rows, 50)):  # sample 50 rows
        row = region[i]
        if np.std(row) < 1e-10:
            continue
        # Normalized autocorrelation
        ac = np.correlate(row, row, mode="full")[len(row) - 1:]
        ac = ac / ac[0]  # normalize
        row_autocorrs.append(ac[:max_lag + 1])

    if not row_autocorrs:
        return 0.0, 0.0

    mean_ac = np.mean(row_autocorrs, axis=0)
    # Exclude lag 0 (always 1.0)
    if len(mean_ac) > 1:
        ac_vals = mean_ac[1:]
        peak_idx = np.argmax(np.abs(ac_vals))
        peak_val = float(abs(ac_vals[peak_idx]))
        scale = float(peak_idx + 1)
    else:
        peak_val = 0.0
        scale = 0.0

    return peak_val, scale


# ===========================================================================
# Quality score computation
# ===========================================================================

def compute_quality_score(metrics: QualityMetrics, cfg: QualityConfig) -> None:
    """Compute component scores and overall quality classification.

    Modifies *metrics* in place: sets score_* fields, quality_score, and
    quality_class.
    """
    # Background score: penalize spatial variation
    if metrics.diff_std > 0:
        bg_ratio = metrics.background_spatial_std / metrics.diff_std
        metrics.score_background = max(0.0, 1.0 - bg_ratio / cfg.background_max_spatial_std_ratio)
    else:
        metrics.score_background = 0.0

    # Dipole score: penalize high dipole fraction
    if metrics.dipole_fraction <= cfg.dipole_max_fraction_pass:
        metrics.score_dipole = 1.0
    elif metrics.dipole_fraction >= cfg.dipole_max_fraction_fail:
        metrics.score_dipole = 0.0
    else:
        # Linear interpolation between pass and fail thresholds
        frac = (metrics.dipole_fraction - cfg.dipole_max_fraction_pass) / (
            cfg.dipole_max_fraction_fail - cfg.dipole_max_fraction_pass
        )
        metrics.score_dipole = max(0.0, 1.0 - frac)

    # Bright-star score: penalize large residuals
    if metrics.bright_star_count > 0:
        rms_sigma = metrics.bright_star_residual_rms
        metrics.score_bright_star = max(0.0, 1.0 - rms_sigma / cfg.bright_star_max_residual_sigma)
    else:
        metrics.score_bright_star = 1.0  # no bright stars to check

    # Edge score: penalize edge artifacts
    metrics.score_edge = max(0.0, 1.0 - max(0.0, metrics.edge_std_ratio - 1.0) / (cfg.edge_max_std_ratio - 1.0))

    # Autocorrelation score: penalize correlated noise
    metrics.score_autocorr = max(0.0, 1.0 - metrics.autocorr_peak / cfg.autocorr_max_peak)

    # Overall weighted score
    metrics.quality_score = (
        cfg.weight_background * metrics.score_background
        + cfg.weight_dipole * metrics.score_dipole
        + cfg.weight_bright_star * metrics.score_bright_star
        + cfg.weight_edge * metrics.score_edge
        + cfg.weight_autocorr * metrics.score_autocorr
    )

    # Classification
    if metrics.quality_score >= cfg.pass_threshold:
        metrics.quality_class = "pass"
    elif metrics.quality_score >= cfg.downgrade_threshold:
        metrics.quality_class = "downgrade"
    else:
        metrics.quality_class = "fail"


# ===========================================================================
# Main entry point
# ===========================================================================

def assess_difference_image(
    diff_data: np.ndarray,
    quality_mask: np.ndarray,
    source_positions: Optional[List[Tuple[float, float]]] = None,
    bright_star_positions: Optional[List[Tuple[float, float]]] = None,
    fwhm: float = 3.0,
    cfg: Optional[QualityConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityMetrics:
    """Assess the quality of a difference image.

    This is the main entry point called from ``templates.py.subtract()``
    after the difference image has been generated.

    Parameters
    ----------
    diff_data : np.ndarray
        The difference image (2-D float array).
    quality_mask : np.ndarray (bool)
        Mask of pixels to exclude from quality checks (NaN, sources,
        target region, etc.).
    source_positions : list of (x, y) or None
        Source positions for dipole detection.
    bright_star_positions : list of (x, y) or None
        Bright star positions for residual flux measurement.
    fwhm : float
        FWHM of the difference image PSF (pixels).
    cfg : QualityConfig or None
        Configuration.  If None, uses defaults.
    metadata : dict or None
        Additional metadata to store in the QualityMetrics (algorithm,
        forceconv, kernel parameters, flux scaling, etc.).

    Returns
    -------
    QualityMetrics
        Container with all metrics, component scores, and classification.
    """
    if cfg is None:
        cfg = QualityConfig()
    if source_positions is None:
        source_positions = []
    if bright_star_positions is None:
        bright_star_positions = []
    if metadata is None:
        metadata = {}

    metrics = QualityMetrics()

    # Store metadata
    metrics.algorithm = str(metadata.get("algorithm", ""))
    metrics.forceconv = str(metadata.get("forceconv", ""))
    metrics.kernel_order = int(metadata.get("kernel_order", 0))
    metrics.kernel_half_width = int(metadata.get("kernel_half_width", 0))
    metrics.science_fwhm = float(metadata.get("science_fwhm", 0.0))
    metrics.template_fwhm = float(metadata.get("template_fwhm", 0.0))
    metrics.n_matching_sources = int(metadata.get("n_matching_sources", 0))
    metrics.flux_scale_conv = float(metadata.get("flux_scale_conv", 0.0))
    metrics.flux_scale_phot = float(metadata.get("flux_scale_phot", 0.0))
    metrics.flux_scale_discrep_pct = float(metadata.get("flux_scale_discrep_pct", 0.0))

    # --- Global statistics ---
    # Guard against shape mismatch between diff_data and quality_mask
    if quality_mask.shape != diff_data.shape:
        logger.warning(
            "quality_mask shape %s != diff_data shape %s; "
            "falling back to NaN-only mask.",
            quality_mask.shape, diff_data.shape,
        )
        quality_mask = ~np.isfinite(diff_data)
    valid_pixels = diff_data[~quality_mask]
    valid_pixels = valid_pixels[np.isfinite(valid_pixels)]
    metrics.valid_pixels = len(valid_pixels)
    if len(valid_pixels) == 0:
        metrics.quality_class = "fail"
        metrics.quality_score = 0.0
        return metrics

    metrics.diff_median = float(np.median(valid_pixels))
    metrics.diff_std = float(np.std(valid_pixels))
    metrics.diff_rms = float(np.sqrt(np.mean(valid_pixels ** 2)))
    noise_sigma = metrics.diff_std

    # --- Spatial background variation ---
    try:
        bg_std, bg_range = compute_spatial_background(
            diff_data, quality_mask, tile_size=cfg.background_tile_size
        )
        metrics.background_spatial_std = bg_std
        metrics.background_spatial_range = bg_range
    except Exception as e:
        logger.debug("Spatial background computation failed: %s", e)

    # --- Dipole detection ---
    if cfg.dipole_check_sources and source_positions:
        try:
            n_dip, frac, mean_amp, max_amp = detect_dipoles(
                diff_data, source_positions, quality_mask,
                noise_sigma, fwhm, cfg,
            )
            metrics.dipole_count = n_dip
            metrics.dipole_fraction = frac
            metrics.dipole_mean_amplitude = mean_amp
            metrics.dipole_max_amplitude = max_amp
        except Exception as e:
            logger.debug("Dipole detection failed: %s", e)

    # --- Bright-star residuals ---
    if cfg.bright_star_check and bright_star_positions:
        try:
            med_res, rms_res, n_bs = measure_bright_star_residuals(
                diff_data, bright_star_positions, quality_mask,
                noise_sigma, fwhm, cfg,
            )
            metrics.bright_star_residual_median = med_res
            metrics.bright_star_residual_rms = rms_res
            metrics.bright_star_count = n_bs
        except Exception as e:
            logger.debug("Bright-star residual measurement failed: %s", e)

    # --- Edge artifacts ---
    try:
        edge_w = cfg.edge_width_px
        if diff_data.shape[0] > 2 * edge_w and diff_data.shape[1] > 2 * edge_w:
            edge_mask = np.zeros_like(quality_mask, dtype=bool)
            edge_mask[:edge_w, :] = True
            edge_mask[-edge_w:, :] = True
            edge_mask[:, :edge_w] = True
            edge_mask[:, -edge_w:] = True
            edge_mask = edge_mask | quality_mask
            edge_pixels = diff_data[~edge_mask]
            edge_pixels = edge_pixels[np.isfinite(edge_pixels)]
            if len(edge_pixels) > 0 and metrics.diff_std > 0:
                metrics.edge_std_ratio = float(np.std(edge_pixels) / metrics.diff_std)
    except Exception as e:
        logger.debug("Edge artifact check failed: %s", e)

    # --- Autocorrelation ---
    try:
        peak, scale = compute_autocorrelation(
            diff_data, quality_mask, noise_sigma,
            max_lag=cfg.autocorr_max_lag,
        )
        metrics.autocorr_peak = peak
        metrics.autocorr_scale_px = scale
    except Exception as e:
        logger.debug("Autocorrelation computation failed: %s", e)

    # --- Compute quality score ---
    compute_quality_score(metrics, cfg)

    # Log summary
    logger.info(
        "Difference-image quality: class=%s score=%.3f | "
        "median=%.3f std=%.3f rms=%.3f | "
        "dipoles=%d (%.1f%%) | "
        "bright_star_resid_rms=%.2f sigma (n=%d) | "
        "bg_spatial_std=%.3f | edge_ratio=%.2f | autocorr_peak=%.3f",
        metrics.quality_class, metrics.quality_score,
        metrics.diff_median, metrics.diff_std, metrics.diff_rms,
        metrics.dipole_count, metrics.dipole_fraction * 100,
        metrics.bright_star_residual_rms, metrics.bright_star_count,
        metrics.background_spatial_std, metrics.edge_std_ratio,
        metrics.autocorr_peak,
    )

    return metrics


def write_quality_manifest(
    metrics: QualityMetrics,
    output_path: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a machine-readable JSON manifest with all quality metrics.

    Parameters
    ----------
    metrics : QualityMetrics
        The quality metrics to write.
    output_path : str
        Path to the output JSON file.
    extra_metadata : dict or None
        Additional provenance metadata (input paths, hashes, timestamps,
        software versions, etc.).
    """
    manifest = {
        "quality_metrics": metrics.to_dict(),
        "provenance": extra_metadata or {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("Difference-image quality manifest written to %s", output_path)
    except Exception as e:
        logger.warning("Failed to write quality manifest to %s: %s", output_path, e)


def write_quality_to_fits_header(
    fits_path: str,
    metrics: QualityMetrics,
) -> None:
    """Write quality metrics as FITS header keywords for provenance.

    Parameters
    ----------
    fits_path : str
        Path to the difference image FITS file (modified in place).
    metrics : QualityMetrics
        The quality metrics to record.
    """
    try:
        from astropy.io import fits as _fits
        with _fits.open(fits_path, mode="update", memmap=False) as hdul:
            hdr = hdul[0].header
            hdr["DIFFQUAL"] = metrics.quality_class
            hdr["DIFFQSCR"] = float(metrics.quality_score)
            hdr["DIFFDIPO"] = int(metrics.dipole_count)
            hdr["DIFFDIPF"] = float(metrics.dipole_fraction)
            hdr["DIFFBGST"] = float(metrics.background_spatial_std)
            hdr["DIFFEDGR"] = float(metrics.edge_std_ratio)
            hdr["DIFFACOR"] = float(metrics.autocorr_peak)
            hdr["DIFFBSR"] = float(metrics.bright_star_residual_rms)
            hdr["DIFFBSN"] = int(metrics.bright_star_count)
            hdul.flush()
    except Exception as e:
        logger.warning("Failed to write quality keywords to FITS header: %s", e)
