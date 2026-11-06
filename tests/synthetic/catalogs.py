"""Synthetic source catalogs for AutoPhOT tests.

Generates catalogs of point sources with known positions and fluxes
for injection/recovery and photometry validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_isolated_catalog(
    n_sources: int = 50,
    image_shape: tuple[int, int] = (512, 512),
    margin: int = 50,
    min_separation: float = 30.0,
    flux_range: tuple[float, float] = (100.0, 50000.0),
    seed: int | None = 42,
) -> pd.DataFrame:
    """Create a catalog of well-separated point sources.

    Parameters
    ----------
    n_sources : int
        Number of sources to place.
    image_shape : (ny, nx)
        Image dimensions.
    margin : int
        Minimum distance from image edge (pixels).
    min_separation : float
        Minimum separation between sources (pixels).
    flux_range : (flux_min, flux_max)
        Range of total fluxes (counts).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: x_pix, y_pix, flux_true, mag_true
    """
    rng = np.random.default_rng(seed)
    ny, nx = image_shape

    positions = []
    attempts = 0
    max_attempts = n_sources * 1000
    while len(positions) < n_sources and attempts < max_attempts:
        x = rng.uniform(margin, nx - margin)
        y = rng.uniform(margin, ny - margin)
        ok = all(np.hypot(x - px, y - py) >= min_separation for px, py in positions)
        if ok:
            positions.append((x, y))
        attempts += 1

    if len(positions) < n_sources:
        raise RuntimeError(
            f"Could not place {n_sources} sources with min_separation={min_separation} "
            f"in {image_shape} (placed {len(positions)})"
        )

    fluxes = rng.uniform(flux_range[0], flux_range[1], size=n_sources)
    # Instrumental magnitude: -2.5 * log10(flux)
    mags = -2.5 * np.log10(fluxes)

    return pd.DataFrame(
        {
            "x_pix": [p[0] for p in positions],
            "y_pix": [p[1] for p in positions],
            "flux_true": fluxes,
            "mag_true": mags,
        }
    )


def make_blended_catalog(
    n_pairs: int = 10,
    image_shape: tuple[int, int] = (512, 512),
    margin: int = 50,
    separation_range: tuple[float, float] = (2.0, 5.0),
    flux_ratio_range: tuple[float, float] = (0.3, 1.0),
    primary_flux_range: tuple[float, float] = (5000.0, 30000.0),
    seed: int | None = 42,
) -> pd.DataFrame:
    """Create a catalog of blended source pairs.

    Each pair consists of a primary source and a secondary source at
    a controlled separation and flux ratio.
    """
    rng = np.random.default_rng(seed)
    ny, nx = image_shape

    rows = []
    for _ in range(n_pairs):
        x0 = rng.uniform(margin + 20, nx - margin - 20)
        y0 = rng.uniform(margin + 20, ny - margin - 20)
        sep = rng.uniform(*separation_range)
        angle = rng.uniform(0, 2 * np.pi)
        x1 = x0 + sep * np.cos(angle)
        y1 = y0 + sep * np.sin(angle)

        f0 = rng.uniform(*primary_flux_range)
        ratio = rng.uniform(*flux_ratio_range)
        f1 = f0 * ratio

        rows.append({"x_pix": x0, "y_pix": y0, "flux_true": f0, "mag_true": -2.5 * np.log10(f0), "pair_id": len(rows) // 2})
        rows.append({"x_pix": x1, "y_pix": y1, "flux_true": f1, "mag_true": -2.5 * np.log10(f1), "pair_id": len(rows) // 2})

    return pd.DataFrame(rows)


def make_crowded_catalog(
    n_sources: int = 200,
    image_shape: tuple[int, int] = (512, 512),
    margin: int = 20,
    min_separation: float = 5.0,
    flux_range: tuple[float, float] = (200.0, 20000.0),
    seed: int | None = 42,
) -> pd.DataFrame:
    """Create a catalog of crowded sources (small separations)."""
    return make_isolated_catalog(
        n_sources=n_sources,
        image_shape=image_shape,
        margin=margin,
        min_separation=min_separation,
        flux_range=flux_range,
        seed=seed,
    )
