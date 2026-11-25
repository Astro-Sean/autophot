#!/usr/bin/env python3
"""
Difference-image quality diagnostics for AutoPhOT image subtraction.

Provides quantitative metrics that go beyond a simple global RMS:

  * Dipole detection - bright-star residual dipoles are the most common
    symptom of astrometric misalignment or PSF-matching failure.  Detected
    as anti-symmetric positive/negative residual pairs around known source
    positions.

  * Spatial background variation - systematic background differences across
    the field indicate sky or illumination-gradient mismatch.

  * Residual autocorrelation - correlated noise on small spatial scales
    indicates over-fitting or deconvolution artefacts.

  * Bright-star residual flux - flux remaining around bright sources after
    subtraction (should be near zero for a good subtraction).

  * Structured quality score - a documented pass / downgrade / fail
    classification with component scores, not just one combined number.

  * Provenance manifest - machine-readable JSON recording all inputs,
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

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


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
    dipole_checked: int = 0  # number of sources actually examined
    dipole_fraction_scored: float = (
        0.0  # fraction used for scoring (Wilson lower bound for small samples)
    )
    dipole_mean_amplitude: float = 0.0  # mean |positive + negative| flux
    dipole_max_amplitude: float = 0.0

    # Bright-star residual flux
    bright_star_residual_median: float = 0.0
    bright_star_residual_rms: float = 0.0
    bright_star_count: int = 0
    # RMS of per-star excess normalised by its allowed level (noise or
    # flux-fraction, whichever is larger); this is what the score uses.
    bright_star_residual_rms_norm: float = 0.0

    # Residual autocorrelation
    autocorr_scale_px: float = 0.0  # spatial scale of correlated noise
    autocorr_peak: float = 0.0  # peak autocorrelation (excluding lag 0)
    autocorr_peak_signed: float = 0.0  # signed value at the peak lag
    # Kernel-implied levels reported by the backend (0 = unknown): positive
    # correlation from the broadening convolution is expected physics and
    # only the excess above it is scored.
    autocorr_expected_pos: float = 0.0
    autocorr_expected_neg: float = 0.0

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
    # False when no dipole/bright-star residual check could run - the
    # classification then caps at "downgrade" because the subtraction's
    # fidelity is unverified rather than verified clean.
    residual_checks_verified: bool = True

    # Names of enabled metric blocks that raised/returned no usable sample;
    # any such failure caps the class at "downgrade" (a broken metric must
    # not improve the reported quality).
    metric_failures: List[str] = field(default_factory=list)

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
        # JSON does not allow bare NaN/Inf; emit null so the manifest stays
        # standards-compliant and parseable by other tools.
        _d = json.loads(json.dumps(self.to_dict(), default=str))
        for _k, _v in _d.items():
            if isinstance(_v, float) and not np.isfinite(_v):
                _d[_k] = None
        return json.dumps(_d, indent=indent, default=str)


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
    # Below this many checked sources the raw dipole fraction is statistically
    # unreliable; the score uses the one-sided Wilson lower bound instead.
    dipole_min_checked: int = 10
    dipole_wilson_z: float = 1.645  # one-sided ~95% lower confidence bound
    # Kernel-mismatch residuals scale with source flux, so a fixed sigma
    # threshold flags percent-level matching error on bright stars as
    # dipoles.  The flag threshold is max(n_sigma*noise, this fraction *
    # source flux) - a real subtraction failure leaves far more than ~10%.
    dipole_min_flux_frac: float = 0.10

    # Bright-star residuals
    bright_star_check: bool = True
    bright_star_n_sigma: float = 50.0  # stars brighter than this sigma
    bright_star_radius_fwhm: float = 3.0
    bright_star_max_residual_sigma: float = 3.0  # max allowed residual in sigma units
    # Same flux-scaling issue as dipoles: annulus excess at a bright star is
    # allowed up to this fraction of its flux before counting against the
    # subtraction.  An unsubtracted or badly-mismatched star exceeds it.
    bright_star_min_flux_frac: float = 0.10

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

    def __post_init__(self) -> None:
        _wsum = (
            self.weight_background
            + self.weight_dipole
            + self.weight_bright_star
            + self.weight_edge
            + self.weight_autocorr
        )
        if _wsum <= 0:
            raise ValueError(
                f"QualityConfig weights must sum to a positive value (got {_wsum})."
            )


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
    source_fluxes: Optional[List[float]] = None,
) -> Tuple[int, float, float, float]:
    """Detect dipole residuals around known source positions.

    Two kernel-mismatch morphologies are caught: an anti-symmetric
    positive/negative lobe pair from astrometric misalignment, and a
    "sombrero" (core of one sign ringed by the opposite sign) from a
    sharpening match kernel.  For each source, we search for a positive
    peak within ``dipole_radius_fwhm * FWHM`` and a corresponding negative
    excursion, then require either opposite-lobe geometry or a bipolar
    cluster of >N-sigma pixels in each sign.

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
    source_fluxes : list of float or None
        Per-source flux in difference-image units, parallel to
        ``source_positions``.  Kernel-mismatch residuals scale with flux,
        so the flag threshold per source is
        ``max(n_sigma * noise, dipole_min_flux_frac * flux)`` - a few-percent
        matching error on a bright star then reads as matching error, not
        a subtraction failure.  None keeps the pure noise threshold.

    Returns
    -------
    (dipole_count, dipole_fraction, mean_amplitude, max_amplitude, n_checked)
        ``n_checked`` is the number of sources actually examined (edge-skipped
        sources are excluded), so the fraction reflects the evaluated sample.
    """
    if (
        not source_positions
        or not np.isfinite(noise_sigma)
        or noise_sigma <= 0
        or not np.isfinite(fwhm)
        or fwhm <= 0
    ):
        return 0, 0.0, 0.0, 0.0, 0

    ny, nx = diff_data.shape
    radius = max(int(cfg.dipole_radius_fwhm * fwhm), 3)
    threshold = cfg.dipole_n_sigma * noise_sigma
    min_antisym = cfg.dipole_min_antisymmetry

    dipole_count = 0
    amplitudes = []
    n_checked = 0

    for _si, (x, y) in enumerate(source_positions):
        xi, yi = int(round(x)), int(round(y))
        # Stamp would fall off the edge.
        if xi < radius or xi >= nx - radius or yi < radius or yi >= ny - radius:
            continue

        stamp = diff_data[yi - radius : yi + radius + 1, xi - radius : xi + radius + 1]
        stamp_mask = quality_mask[
            yi - radius : yi + radius + 1, xi - radius : xi + radius + 1
        ]

        # A nearly fully masked stamp can never flag (its cleaned pixels
        # are all zero) but would still dilute dipole_fraction as
        # "checked" -- skip it entirely.  Eight unmasked pixels is the
        # minimum for the >=4-per-sign bipolar test to have any support.
        if np.count_nonzero(~stamp_mask & np.isfinite(stamp)) < 8:
            continue
        n_checked += 1

        # Per-source threshold: kernel-mismatch amplitude scales with the
        # source flux, so bright stars need the bipolar residual to clear a
        # fraction of their flux, not just the noise floor.
        _thr = threshold
        if source_fluxes is not None and _si < len(source_fluxes):
            _fx = source_fluxes[_si]
            if np.isfinite(_fx) and _fx > 0:
                _thr = max(
                    threshold, cfg.dipole_min_flux_frac * float(_fx)
                )

        stamp_clean = stamp.copy()
        stamp_clean[stamp_mask] = 0.0
        stamp_clean[~np.isfinite(stamp_clean)] = 0.0

        pos_peak = np.max(stamp_clean)
        neg_peak = np.min(stamp_clean)

        if pos_peak < _thr or abs(neg_peak) < _thr:
            continue

        # A dipole's positive and negative peaks sit on opposite sides of
        # the source center with comparable amplitude.
        pos_idx = np.unravel_index(np.argmax(stamp_clean), stamp_clean.shape)
        neg_idx = np.unravel_index(np.argmin(stamp_clean), stamp_clean.shape)

        cy, cx = radius, radius
        dy_pos = pos_idx[0] - cy
        dx_pos = pos_idx[1] - cx
        dy_neg = neg_idx[0] - cy
        dx_neg = neg_idx[1] - cx

        # Two bipolar morphologies both indicate kernel mismatch:
        #
        # 1) Classic dipole: lobes on opposite sides of the source center
        #    with comparable peak amplitudes (misalignment).
        #
        # 2) Sombrero: a sharpening (deconvolving) kernel leaves a core of
        #    one sign ringed by the opposite sign -- the negative power is
        #    spread over an annulus, so neither a lobe-geometry test nor a
        #    peak-amplitude ratio sees it, and masked cores shift the lobes
        #    off the antipodal axis (SN2024pba 2025-02-22 r: +11 kADU core
        #    inside a -1 kADU ring).  Require a handful of >N-sigma pixels
        #    in EACH sign: real variability or flux-scale mismatch leaves
        #    a unipolar residual, and noise rarely produces >=4 such
        #    pixels per sign in one stamp.
        dot = dy_pos * dy_neg + dx_pos * dx_neg
        amp_ratio = min(abs(pos_peak), abs(neg_peak)) / max(
            abs(pos_peak), abs(neg_peak)
        )
        _is_dipole = dot < 0 and amp_ratio >= min_antisym
        _is_bipolar = (
            min(
                int(np.sum(stamp_clean > _thr)),
                int(np.sum(stamp_clean < -_thr)),
            )
            >= 4
        )
        if not (_is_dipole or _is_bipolar):
            continue

        dipole_count += 1
        amplitude = abs(pos_peak) + abs(neg_peak)
        amplitudes.append(amplitude)

    dipole_fraction = dipole_count / n_checked if n_checked > 0 else 0.0
    mean_amp = float(np.mean(amplitudes)) if amplitudes else 0.0
    max_amp = float(np.max(amplitudes)) if amplitudes else 0.0

    return dipole_count, dipole_fraction, mean_amp, max_amp, n_checked


def measure_bright_star_residuals(
    diff_data: np.ndarray,
    bright_star_positions: List[Tuple[float, float]],
    quality_mask: np.ndarray,
    noise_sigma: float,
    fwhm: float,
    cfg: QualityConfig,
    star_fluxes: Optional[List[float]] = None,
) -> Tuple[float, float, int, float]:
    """Measure residual flux around bright stars in the difference image.

    For each bright star, we measure the residual flux in an annulus
    around the star position.  A good subtraction should have residuals
    consistent with the background noise.

    Returns
    -------
    (median_residual_sigma, rms_residual_sigma, n_stars, rms_residual_norm)
        Residuals expressed in units of noise_sigma, plus the RMS of each
        star's excess divided by its allowed level (``max(residual_sigma,
        bright_star_min_flux_frac * flux / noise)``).  The normalised RMS
        is what the score consumes: percent-level kernel-mismatch power at
        a bright star no longer reads as subtraction failure.
    """
    if (
        not bright_star_positions
        or not np.isfinite(noise_sigma)
        or noise_sigma <= 0
        or not np.isfinite(fwhm)
        or fwhm <= 0
    ):
        return 0.0, 0.0, 0, 0.0

    ny, nx = diff_data.shape
    inner_r = max(int(fwhm), 3)
    outer_r = max(int(cfg.bright_star_radius_fwhm * fwhm), inner_r + 2)

    residuals = []
    residuals_norm = []
    for _si, (x, y) in enumerate(bright_star_positions):
        xi, yi = int(round(x)), int(round(y))
        if xi < outer_r or xi >= nx - outer_r or yi < outer_r or yi >= ny - outer_r:
            continue

        # Annulus mask
        yy, xx = np.ogrid[: 2 * outer_r + 1, : 2 * outer_r + 1]
        r2 = (xx - outer_r) ** 2 + (yy - outer_r) ** 2
        annulus = (r2 >= inner_r**2) & (r2 <= outer_r**2)

        stamp = diff_data[
            yi - outer_r : yi + outer_r + 1, xi - outer_r : xi + outer_r + 1
        ]
        stamp_mask = quality_mask[
            yi - outer_r : yi + outer_r + 1, xi - outer_r : xi + outer_r + 1
        ]

        annulus_data = stamp[annulus & ~stamp_mask]
        annulus_data = annulus_data[np.isfinite(annulus_data)]
        if len(annulus_data) > 10:
            # Excess-power RMS, not the median: a dipole residual's
            # positive and negative lobes cancel in a median, so it
            # reports ~0 for the exact failure mode this check exists to
            # catch.  Subtracting the expected noise variance keeps a
            # clean subtraction at ~0 sigma.
            _rms2 = float(np.mean(annulus_data**2))
            _excess = np.sqrt(max(_rms2 - noise_sigma**2, 0.0)) / noise_sigma
            # Kernel-mismatch wings scale with star flux, so the allowed
            # excess is the larger of the flat sigma limit and a fraction
            # of the star's own flux.  An unsubtracted star (~all its flux
            # left over) still exceeds it by an order of magnitude.
            _allowed = cfg.bright_star_max_residual_sigma
            if star_fluxes is not None and _si < len(star_fluxes):
                _fx = star_fluxes[_si]
                if np.isfinite(_fx) and _fx > 0:
                    _allowed = max(
                        _allowed,
                        cfg.bright_star_min_flux_frac * float(_fx) / noise_sigma,
                    )
            residuals.append(_excess)
            residuals_norm.append(_excess / _allowed)

    if not residuals:
        return 0.0, 0.0, 0, 0.0

    res = np.array(residuals)
    res_norm = np.array(residuals_norm)
    return (
        float(np.median(res)),
        float(np.sqrt(np.mean(res**2))),
        len(residuals),
        float(np.sqrt(np.mean(res_norm**2))),
    )


def compute_autocorrelation(
    diff_data: np.ndarray,
    quality_mask: np.ndarray,
    noise_sigma: float,
    max_lag: int = 20,
) -> Tuple[float, float, float]:
    """Compute the spatial autocorrelation of the difference image residuals.

    Correlated noise on small spatial scales indicates over-fitting or
    deconvolution artefacts.  We compute the 1-D autocorrelation along
    rows and columns and report the peak (excluding lag 0) and the
    spatial scale at which it occurs.

    Returns
    -------
    (peak_autocorr, scale_px, peak_signed)
        Peak autocorrelation magnitude (0 = uncorrelated, 1 = perfectly
        correlated), the spatial scale in pixels, and the signed value at
        the peak lag.  The sign separates physically distinct modes:
        positive correlation is expected whenever one input was convolved
        by a broadening kernel, while negative correlation (ringing /
        over-sharpening) has no benign explanation.
    """
    if noise_sigma <= 0:
        return 0.0, 0.0, 0.0

    # Central region only, to keep edge artifacts out of the estimate.
    ny, nx = diff_data.shape
    y0, y1 = ny // 4, 3 * ny // 4
    x0, x1 = nx // 4, 3 * nx // 4
    region = diff_data[y0:y1, x0:x1].copy()
    region_mask = quality_mask[y0:y1, x0:x1]

    valid = ~(region_mask | ~np.isfinite(region))

    # Masked autocorrelation: masked pixels must be EXCLUDED from both the
    # mean and each lag pair.  Zeroing them first injects artificial
    # correlation (runs of zeros correlate) and biases the peak estimate.
    n_rows = region.shape[0]
    n_cols = region.shape[1]

    def _slice_ac(arr: np.ndarray, vmask: np.ndarray):
        """Autocorrelation (lag 0..max_lag) of one 1-D row or column."""
        if vmask.sum() < max_lag + 8:
            return None
        vector = np.where(vmask, arr - np.mean(arr[vmask]), 0.0)
        num = np.correlate(vector, vector, mode="full")[len(vector) - 1 :]
        den = np.correlate(vmask.astype(float), vmask.astype(float), mode="full")[
            len(vmask) - 1 :
        ]
        with np.errstate(invalid="ignore", divide="ignore"):
            ac = np.where(den > 0, num / den, 0.0)
        if ac[0] <= 0:
            return None
        return (ac / ac[0])[: max_lag + 1]

    row_autocorrs = []
    for i in range(min(n_rows, 50)):
        _ac = _slice_ac(region[i], valid[i])
        if _ac is not None:
            row_autocorrs.append(_ac)
    # Columns as well: artifacts aligned with columns (e.g. bad columns,
    # windowing strips) would otherwise be missed by a rows-only scan.
    col_autocorrs = []
    for j in range(min(n_cols, 50)):
        _ac = _slice_ac(region[:, j], valid[:, j])
        if _ac is not None:
            col_autocorrs.append(_ac)

    all_autocorrs = row_autocorrs + col_autocorrs
    if not all_autocorrs:
        return 0.0, 0.0, 0.0

    mean_ac = np.mean(all_autocorrs, axis=0)
    # Report the strongest positive or negative off-zero peak (signed) so
    # ringing/anti-correlation is not scored identically to positive
    # correlation.
    if len(mean_ac) > 1:
        ac_vals = mean_ac[1:]
        peak_idx = int(np.argmax(np.abs(ac_vals)))
        peak_signed = float(ac_vals[peak_idx])
        peak_val = float(abs(peak_signed))
        scale = float(peak_idx + 1)
    else:
        peak_val = 0.0
        peak_signed = 0.0
        scale = 0.0

    return peak_val, scale, peak_signed


# ===========================================================================
# Quality score computation
# ===========================================================================


def _wilson_lower_bound(k: int, n: int, z: float) -> float:
    """One-sided Wilson lower confidence bound for a binomial fraction.

    With few checked sources the observed dipole fraction is dominated by
    counting noise (1 dipole in 5 sources reads as 20%).  The lower bound is
    the smallest true fraction consistent with the data at the ``z``
    confidence level, so scoring on it penalises only dipole rates the data
    actually establish - while still catching egregious small-sample cases
    (e.g. 3/5 has a lower bound above the fail threshold).
    """
    if n <= 0:
        return 0.0
    z2 = float(z) ** 2
    denom = n + z2
    centre = (k + z2 / 2.0) / denom
    half = float(z) * np.sqrt(k * (n - k) / n + z2 / 4.0) / denom
    return max(0.0, centre - half)


def compute_quality_score(metrics: QualityMetrics, cfg: QualityConfig) -> None:
    """Compute component scores and overall quality classification.

    Modifies *metrics* in place: sets score_* fields, quality_score, and
    quality_class.
    """
    # Background score: penalize spatial variation and any global DC offset
    # (a sky/flux mismatch moves every tile median together, so the spatial
    # std is blind to it - the median is the only probe).
    _bg_max = float(cfg.background_max_spatial_std_ratio)
    if metrics.diff_std > 0:
        _bg_ratio = (
            metrics.background_spatial_std / metrics.diff_std if _bg_max > 0 else 0.0
        )
        metrics.score_background = (
            max(0.0, 1.0 - _bg_ratio / _bg_max) if _bg_max > 0 else 1.0
        )
    else:
        # A perfectly uniform difference has zero spatial variation; score it
        # clean rather than as the worst case.
        metrics.score_background = 1.0
    # DC-pedestal penalty: |median| larger than a few sigma of the noise is
    # the classic sky/flux-scale mismatch.  noise_sigma here is the clipped
    # background noise (fall back to std).
    _noise_q = max(metrics.diff_std, 1e-9)
    _offset_sigma = abs(metrics.diff_median) / _noise_q
    if _offset_sigma > 2.0:
        _off_penalty = min(1.0, (_offset_sigma - 2.0) / 6.0)
        metrics.score_background *= 1.0 - 0.5 * _off_penalty
        logger.info(
            "Difference-image median is %.1f sigma from zero - likely sky/"
            "flux-scale mismatch; background score penalized.",
            _offset_sigma,
        )

    # Dipole score: penalize high dipole fraction.  For small checked samples
    # score on the Wilson lower confidence bound so a single dipole in a
    # handful of sources cannot zero out the component on weak evidence.
    if (
        cfg.dipole_min_checked > 0
        and 0 < metrics.dipole_checked < cfg.dipole_min_checked
    ):
        frac_for_score = _wilson_lower_bound(
            metrics.dipole_count, metrics.dipole_checked, cfg.dipole_wilson_z
        )
        if frac_for_score != metrics.dipole_fraction:
            logger.info(
                "Dipole fraction %.1f%% from only %d checked sources; scoring "
                "on Wilson lower bound %.1f%% (z=%.3g).",
                metrics.dipole_fraction * 100,
                metrics.dipole_checked,
                frac_for_score * 100,
                cfg.dipole_wilson_z,
            )
    else:
        frac_for_score = metrics.dipole_fraction
    metrics.dipole_fraction_scored = frac_for_score

    if frac_for_score <= cfg.dipole_max_fraction_pass:
        metrics.score_dipole = 1.0
    elif frac_for_score >= cfg.dipole_max_fraction_fail:
        metrics.score_dipole = 0.0
    else:
        # Linear interpolation between pass and fail thresholds
        frac = (frac_for_score - cfg.dipole_max_fraction_pass) / (
            cfg.dipole_max_fraction_fail - cfg.dipole_max_fraction_pass
        )
        metrics.score_dipole = max(0.0, 1.0 - frac)
    # An enabled dipole check that examined nothing is unverified, not clean.
    if cfg.dipole_check_sources and metrics.dipole_checked == 0:
        metrics.score_dipole = min(metrics.score_dipole, 0.5)

    # Bright-star score: penalize large residuals.  Untested is NOT a clean
    # score: scoring 1.0 for "no bright stars" would let a totally failed
    # subtraction (every source a unipolar positive residual, invisible to
    # the dipole test) pass on exactly the check that would catch it.
    # Residuals arrive already normalised by each star's allowed level
    # (noise limit or flux fraction), so a normalised RMS of 1 means the
    # typical bright star sits at its allowed limit.
    if metrics.bright_star_count > 0:
        metrics.score_bright_star = max(
            0.0, 1.0 - metrics.bright_star_residual_rms_norm
        )
    elif cfg.bright_star_check:
        metrics.score_bright_star = 0.5  # neutral, unverified
    else:
        metrics.score_bright_star = 1.0  # intentionally disabled

    # Edge score: penalize edge artifacts
    _edge_denom = cfg.edge_max_std_ratio - 1.0
    if _edge_denom > 0:
        metrics.score_edge = max(
            0.0, 1.0 - max(0.0, metrics.edge_std_ratio - 1.0) / _edge_denom
        )
    else:
        metrics.score_edge = 0.0 if metrics.edge_std_ratio > 1.0 else 1.0

    # Autocorrelation score: the sign separates two physically different
    # regimes.  Positive lag correlation is expected whenever the backend
    # broadened one input with the matching kernel - score only the excess
    # above the kernel-implied level the backend reported.  Negative
    # correlation (ringing / over-sharpening) has no benign explanation and
    # is scored at full weight.  With no backend estimate both expectations
    # are zero and this reduces to the raw |peak| penalty.
    _ac_signed = (
        metrics.autocorr_peak_signed
        if metrics.autocorr_peak_signed != 0.0
        else metrics.autocorr_peak
    )
    if cfg.autocorr_max_peak > 0:
        if _ac_signed >= 0:
            _ac_excess = max(0.0, _ac_signed - metrics.autocorr_expected_pos)
        else:
            _ac_excess = max(
                0.0, abs(_ac_signed) - abs(metrics.autocorr_expected_neg)
            )
        metrics.score_autocorr = max(
            0.0, 1.0 - _ac_excess / cfg.autocorr_max_peak
        )
    else:
        metrics.score_autocorr = 1.0

    # Clamp every component into [0, 1] so misconfigured weights/thresholds
    # cannot produce scores above 1 or below 0.
    for _name in (
        "score_background",
        "score_dipole",
        "score_bright_star",
        "score_edge",
        "score_autocorr",
    ):
        setattr(metrics, _name, float(min(1.0, max(0.0, getattr(metrics, _name)))))

    # Overall weighted score
    metrics.quality_score = (
        cfg.weight_background * metrics.score_background
        + cfg.weight_dipole * metrics.score_dipole
        + cfg.weight_bright_star * metrics.score_bright_star
        + cfg.weight_edge * metrics.score_edge
        + cfg.weight_autocorr * metrics.score_autocorr
    )
    metrics.quality_score = float(min(1.0, max(0.0, metrics.quality_score)))

    # Classification
    if metrics.quality_score >= cfg.pass_threshold:
        metrics.quality_class = "pass"
    elif metrics.quality_score >= cfg.downgrade_threshold:
        metrics.quality_class = "downgrade"
    else:
        metrics.quality_class = "fail"

    # A "pass" on zero residual checks is unverifiable, not verified:
    # the dipole and bright-star measurements are the only evidence the
    # subtraction actually removed flux, and scoring them 1.0 for
    # "nothing to check" would certify a diff no source ever tested.
    # Cap at "downgrade" so photometry picks up the configured
    # systematic instead of a clean bill.  Skipped only when the user
    # deliberately disabled both residual checks.
    metrics.residual_checks_verified = (
        metrics.dipole_checked > 0 or metrics.bright_star_count > 0
    )
    if (
        metrics.quality_class == "pass"
        and not metrics.residual_checks_verified
        and (cfg.dipole_check_sources or cfg.bright_star_check)
    ):
        metrics.quality_class = "downgrade"
        logger.info(
            "Difference-image quality downgraded: no residual checks "
            "were possible (dipole_checked=0, bright_star_count=0) - "
            "subtraction fidelity is unverified, not verified clean."
        )

    # A metric that raised cannot certify the subtraction.  Downgrade (never
    # upgrade) so a broken check cannot turn a bad subtraction into a pass.
    if metrics.metric_failures:
        logger.warning(
            "Difference-image quality capped at 'downgrade': metric block(s) "
            "raised during assessment: %s",
            ", ".join(sorted(set(metrics.metric_failures))),
        )
        if metrics.quality_class == "pass":
            metrics.quality_class = "downgrade"


# ===========================================================================
# Main entry point
# ===========================================================================


def assess_difference_image(
    diff_data: np.ndarray,
    quality_mask: np.ndarray,
    source_positions: Optional[List[Tuple[float, float]]] = None,
    bright_star_positions: Optional[List[Tuple[float, float]]] = None,
    source_fluxes: Optional[List[float]] = None,
    bright_star_fluxes: Optional[List[float]] = None,
    expected_autocorr_pos: float = 0.0,
    expected_autocorr_neg: float = 0.0,
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
    source_fluxes : list of float or None
        Per-source flux in difference-image units, parallel to
        ``source_positions``.  Enables the flux-scaled dipole threshold so
        percent-level kernel mismatch on bright stars is not flagged.
    bright_star_fluxes : list of float or None
        Same, parallel to ``bright_star_positions``.
    expected_autocorr_pos : float
        Backend-reported kernel-implied positive autocorrelation level
        (0 = unknown).  Positive correlation below this level is expected
        convolution physics and is not penalised.
    expected_autocorr_neg : float
        Kernel-implied negative autocorrelation level (0 = unknown).
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
    # A non-boolean mask (e.g. int 0/1) would make the `~mask` operations a
    # bitwise negation on integers (all True) and zero out valid pixels.
    quality_mask = np.asarray(quality_mask, dtype=bool)

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
    metrics.autocorr_expected_pos = float(expected_autocorr_pos or 0.0)
    metrics.autocorr_expected_neg = float(expected_autocorr_neg or 0.0)

    # --- Global statistics ---
    # Guard against shape mismatch between diff_data and quality_mask
    if quality_mask.shape != diff_data.shape:
        logger.warning(
            "quality_mask shape %s != diff_data shape %s; "
            "falling back to NaN-only mask.",
            quality_mask.shape,
            diff_data.shape,
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
    metrics.diff_rms = float(np.sqrt(np.mean(valid_pixels**2)))
    # The dipole/bright-star thresholds need the *noise* level, not the
    # power of the residuals themselves: a plain std over a diff littered
    # with kernel-sidelobe residuals inflates noise_sigma (51 vs 11 ADU on
    # a ZTF epoch with a deconvolving kernel) and the checks pass vacuously.
    # Sigma-clipping recovers the background noise floor; keep the raw std
    # in the metrics for reporting.
    try:
        _, _, _sigclip = sigma_clipped_stats(valid_pixels, sigma=3.0, maxiters=10)
    except Exception:
        _sigclip = np.nan
    if np.isfinite(_sigclip) and _sigclip > 0:
        noise_sigma = float(_sigclip)
    else:
        noise_sigma = metrics.diff_std

    # --- Spatial background variation ---
    try:
        bg_std, bg_range = compute_spatial_background(
            diff_data, quality_mask, tile_size=cfg.background_tile_size
        )
        metrics.background_spatial_std = bg_std
        metrics.background_spatial_range = bg_range
    except Exception as e:
        metrics.metric_failures.append("background")
        logger.warning("Spatial background computation failed: %s", e)

    # --- Dipole detection ---
    if cfg.dipole_check_sources and source_positions:
        try:
            n_dip, frac, mean_amp, max_amp, n_chk = detect_dipoles(
                diff_data,
                source_positions,
                quality_mask,
                noise_sigma,
                fwhm,
                cfg,
                source_fluxes=source_fluxes,
            )
            metrics.dipole_count = n_dip
            metrics.dipole_fraction = frac
            metrics.dipole_checked = n_chk
            metrics.dipole_mean_amplitude = mean_amp
            metrics.dipole_max_amplitude = max_amp
        except Exception as e:
            metrics.metric_failures.append("dipole")
            logger.warning("Dipole detection failed: %s", e)

    # --- Bright-star residuals ---
    if cfg.bright_star_check and bright_star_positions:
        try:
            med_res, rms_res, n_bs, rms_norm = measure_bright_star_residuals(
                diff_data,
                bright_star_positions,
                quality_mask,
                noise_sigma,
                fwhm,
                cfg,
                star_fluxes=bright_star_fluxes,
            )
            metrics.bright_star_residual_median = med_res
            metrics.bright_star_residual_rms = rms_res
            metrics.bright_star_count = n_bs
            metrics.bright_star_residual_rms_norm = rms_norm
        except Exception as e:
            metrics.metric_failures.append("bright_star")
            logger.warning("Bright-star residual measurement failed: %s", e)

    # --- Edge artifacts ---
    try:
        edge_w = cfg.edge_width_px
        if diff_data.shape[0] > 2 * edge_w and diff_data.shape[1] > 2 * edge_w:
            edge_mask = np.zeros_like(quality_mask, dtype=bool)
            edge_mask[:edge_w, :] = True
            edge_mask[-edge_w:, :] = True
            edge_mask[:, :edge_w] = True
            edge_mask[:, -edge_w:] = True
            # The metric needs the std of pixels IN the edge band (bad
            # pixels still excluded); ~edge_mask would select the interior
            # and report ratio ~1.0 regardless of edge artefacts.  The
            # denominator is the interior-only sigma so the edge band cannot
            # dilute its own artifact.
            edge_pixels = diff_data[edge_mask & ~quality_mask]
            edge_pixels = edge_pixels[np.isfinite(edge_pixels)]
            interior_pixels = diff_data[
                ~edge_mask & ~quality_mask & np.isfinite(diff_data)
            ]
            if len(edge_pixels) > 0 and len(interior_pixels) > 0:
                _int_std = float(np.std(interior_pixels))
                if _int_std > 0:
                    metrics.edge_std_ratio = float(np.std(edge_pixels) / _int_std)
    except Exception as e:
        metrics.metric_failures.append("edge")
        logger.warning("Edge artifact check failed: %s", e)

    # --- Autocorrelation ---
    try:
        peak, scale, peak_signed = compute_autocorrelation(
            diff_data,
            quality_mask,
            noise_sigma,
            max_lag=cfg.autocorr_max_lag,
        )
        metrics.autocorr_peak = peak
        metrics.autocorr_scale_px = scale
        metrics.autocorr_peak_signed = peak_signed
    except Exception as e:
        metrics.metric_failures.append("autocorrelation")
        logger.warning("Autocorrelation computation failed: %s", e)

    # --- Compute quality score ---
    compute_quality_score(metrics, cfg)

    _exp_ac_str = (
        f", expected={metrics.autocorr_expected_pos:+.3f}"
        if metrics.autocorr_expected_pos > 0 or metrics.autocorr_expected_neg < 0
        else ""
    )
    logger.info(
        "Difference-image quality: class=%s score=%.3f\n"
        "    median=%.3f std=%.3f rms=%.3f\n"
        "    dipoles=%d/%d checked (%.1f%%)\n"
        "    bright_star_resid_rms=%.2f sigma, %.2f allowed-norm (n=%d)\n"
        "    bg_spatial_std=%.3f, edge_ratio=%.2f, autocorr_peak=%+.3f%s",
        metrics.quality_class,
        metrics.quality_score,
        metrics.diff_median,
        metrics.diff_std,
        metrics.diff_rms,
        metrics.dipole_count,
        metrics.dipole_checked,
        metrics.dipole_fraction * 100,
        metrics.bright_star_residual_rms,
        metrics.bright_star_residual_rms_norm,
        metrics.bright_star_count,
        metrics.background_spatial_std,
        metrics.edge_std_ratio,
        metrics.autocorr_peak_signed,
        _exp_ac_str,
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
        logger.debug("Difference-image quality manifest written to %s", output_path)
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
            hdr["DIPDCHK"] = int(metrics.dipole_checked)
            hdr["DIFFDIPS"] = float(metrics.dipole_fraction_scored)
            hdr["DIFFBGST"] = float(metrics.background_spatial_std)
            hdr["DIFFEDGR"] = float(metrics.edge_std_ratio)
            hdr["DIFFACOR"] = float(metrics.autocorr_peak)
            hdr["DIFFBSR"] = float(metrics.bright_star_residual_rms)
            hdr["DIFFBSN"] = int(metrics.bright_star_count)
            hdul.flush()
    except Exception as e:
        logger.warning("Failed to write quality keywords to FITS header: %s", e)
