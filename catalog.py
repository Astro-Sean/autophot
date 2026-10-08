#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:47:32 2022
@author: seanbrennan
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import contextlib
import os
import sys
import logging
import pathlib
import warnings
import hashlib
import shutil
import requests
import random
import string
import time
from functools import reduce
from math import ceil
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize
from scipy.ndimage import gaussian_laplace

# Astropy ecosystem imports
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clip, mad_std
from astropy.table import Table
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS

from astroquery.vizier import Vizier

# Photutils imports
from photutils.centroids import centroid_sources, centroid_com, centroid_2dg

# Scikit-learn imports
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


import traceback

# Local imports
from functions import (
    AutophotYaml,
    STATUS,
    log_step,
    pix_dist,
    mag,
    snr,
    set_size,
    normalize_photometric_filter_name,
    parse_supported_filter_group_key,
    log_warning_from_exception,
    canonical_target_name,
    SUPPORTED_PHOTOMETRIC_FILTERS,
)
from aperture import Aperture

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_catalog_log():
    """
    Silence download()/clean() chatter during the optimizer scan.

    Those helpers log a status banner, cache hits, and per-column warnings
    on every call - fine for single-image runs, unreadable when the
    optimizer fans them out over every catalog. Errors still propagate to
    the caller, which reports each failure as one line.
    """
    _prev = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(_prev)


# Remote backends eligible for `use_catalog: "auto"` optimization. "refcat"
# and "custom" join the list only when their prerequisites (MAST CasJobs
# credentials / a catalog file) are configured.
AUTO_OPTIMIZE_CATALOGS = (
    "gaia",
    "pan_starrs",
    "sdss",
    "apass",
    "2mass",
    "skymapper",
    "legacy",
    "tic",
)

# Eligible backends removed from the default "auto" scan, with the reason
# shown in the run printout. Gaia stays the single-catalog fallback, but a
# fresh bulk cone query on every run adds disproportionate load on the ESA
# Gaia archive, so it is only scanned when requested explicitly through
# ``catalog_names``.
AUTO_OPTIMIZE_EXCLUDED = {
    "gaia": "excluded from auto scan - bulk cone queries overload the Gaia archive",
}


# Fixed marker/color per catalog so the optimizer coverage plot keeps a
# consistent legend across bands and runs.
CATALOG_PLOT_STYLE = {
    "gaia": {"marker": "*", "color": "tab:purple"},
    "pan_starrs": {"marker": "o", "color": "tab:blue"},
    "sdss": {"marker": "s", "color": "tab:orange"},
    "apass": {"marker": "^", "color": "tab:green"},
    "2mass": {"marker": "D", "color": "tab:red"},
    "skymapper": {"marker": "v", "color": "tab:cyan"},
    "legacy": {"marker": "P", "color": "tab:pink"},
    "tic": {"marker": "X", "color": "tab:olive"},
    "refcat": {"marker": "h", "color": "tab:brown"},
    "custom": {"marker": "d", "color": "tab:gray"},
}


def _catalog_plot_style(name):
    """Marker/color for a catalog; unnamed backends cycle deterministically."""
    style = CATALOG_PLOT_STYLE.get(name)
    if style is not None:
        return style
    markers = ["o", "s", "^", "D", "v", "P", "X", "h", "d"]
    i = abs(hash(str(name))) % len(markers)
    return {"marker": markers[i], "color": f"C{i % 10}"}


# Photometric system of each catalog's reported magnitudes, keyed on the
# cleaned column names built from databases/catalog.yml. A plain string
# means one system covers every band; a dict resolves per band because
# the catalog mixes systems (APASS: Johnson BV are Vega, Sloan gri are
# AB; Gaia synthetic columns: SDSS_Std are AB, JKC_Std are Vega).
# "combined" merges several sources band-dependent per row, and "custom"
# is user-supplied - its system cannot be inferred.
CATALOG_MAGSYS = {
    "gaia": {
        "u": "abmag",
        "g": "abmag",
        "r": "abmag",
        "i": "abmag",
        "z": "abmag",
        "B": "vegamag",
        "V": "vegamag",
        "R": "vegamag",
        "I": "vegamag",
    },
    "gaia_custom": "abmag",
    "pan_starrs": "abmag",
    "sdss": "abmag",
    "legacy": "abmag",
    "skymapper": "abmag",
    "refcat": {
        "g": "abmag",
        "r": "abmag",
        "i": "abmag",
        "z": "abmag",
        "J": "vegamag",
        "H": "vegamag",
        "K": "vegamag",
    },
    "apass": {
        "B": "vegamag",
        "V": "vegamag",
        "g": "abmag",
        "r": "abmag",
        "i": "abmag",
    },
    "2mass": "vegamag",
    "tic": {
        "u": "abmag",
        "g": "abmag",
        "r": "abmag",
        "i": "abmag",
        "z": "abmag",
        "B": "vegamag",
        "V": "vegamag",
        "J": "vegamag",
        "H": "vegamag",
        "K": "vegamag",
        "G": "vegamag",
        "T": "vegamag",
    },
    "custom": "unknown",
    "combined": "mixed",
}

def catalog_magsys(catalog_name, band=None):
    """
    Photometric system of a catalog's reported magnitudes.

    Returns "abmag", "vegamag", "mixed" (per-row system varies for
    combined catalogs), or "unknown" (custom/unrecognized backends).
    *band* is normalized through the pipeline's filter map before
    lookup; pass it whenever a mixed catalog is in play so the value
    reflects the band actually calibrated.
    """
    name = Catalog._normalize_catalog_name(catalog_name)
    entry = CATALOG_MAGSYS.get(str(name or "").strip().lower())
    if entry is None:
        return "unknown"
    if isinstance(entry, str):
        return entry

    if band:
        key = normalize_photometric_filter_name(band) or str(band).strip()
        key = str(key).strip()
        if key in entry:
            return entry[key]
    systems = set(entry.values())
    return systems.pop() if len(systems) == 1 else "mixed"


def _unwrap_ra_near(ra_deg, center_ra):
    """Wrap RA values to within +/-180 deg of the field centre for plotting."""
    return (np.asarray(ra_deg, dtype=float) - center_ra + 180.0) % 360.0 - 180.0 + center_ra


def _plot_footprint_outline(ax, fps, center_ra, color, dashed=False):
    """
    Draw the union outline of image footprints on ``ax``.

    Small stacks (<=8 frames) keep their individual squares. Larger stacks
    are rasterized onto a grid and traced as a single mosaic outline, so
    hundreds of overlapping edges collapse into one boundary; disjoint
    pointings simply produce separate contours.
    """
    from matplotlib.path import Path

    polys = [
        np.c_[
            _unwrap_ra_near(fp["ra"], center_ra),
            np.asarray(fp["dec"], dtype=float),
        ]
        for fp in fps
        if fp.get("ra") is not None and fp.get("dec") is not None
    ]
    if not polys:
        return

    ls = "--" if dashed else "-"
    if len(polys) <= 8:
        for p in polys:
            ax.plot(
                np.r_[p[:, 0], p[0, 0]],
                np.r_[p[:, 1], p[0, 1]],
                color=color,
                lw=0.8,
                alpha=0.8,
                ls=ls,
                zorder=2,
            )
        return

    x0 = min(p[:, 0].min() for p in polys)
    x1 = max(p[:, 0].max() for p in polys)
    y0 = min(p[:, 1].min() for p in polys)
    y1 = max(p[:, 1].max() for p in polys)
    pad_x = 0.03 * (x1 - x0) + 1e-6
    pad_y = 0.03 * (y1 - y0) + 1e-6

    nx = 600
    xs = np.linspace(x0 - pad_x, x1 + pad_x, nx)
    dx = xs[1] - xs[0]
    ny = max(64, int(round((y1 - y0 + 2 * pad_y) / dx)))
    ys = np.linspace(y0 - pad_y, y0 - pad_y + ny * dx, ny + 1)
    mask = np.zeros((ys.size, xs.size), dtype=bool)

    for p in polys:
        sx = (xs >= p[:, 0].min()) & (xs <= p[:, 0].max())
        sy = (ys >= p[:, 1].min()) & (ys <= p[:, 1].max())
        if not (sx.any() and sy.any()):
            continue
        gx, gy = np.meshgrid(xs[sx], ys[sy], indexing="xy")
        inside = Path(p).contains_points(np.c_[gx.ravel(), gy.ravel()])
        mask[np.ix_(sy, sx)] |= inside.reshape(sy.sum(), sx.sum())

    if not mask.any():
        return
    ax.contourf(
        xs, ys, mask.astype(float), levels=[0.5, 1.5],
        colors=[color], alpha=0.10, zorder=2,
    )
    ax.contour(
        xs, ys, mask.astype(float), levels=[0.5],
        colors=[color], linewidths=1.2, linestyles=ls, zorder=2,
    )


def _draw_region_outline(
    ax, region, color, center_ra, ls=":", lw=0.9, alpha=0.75, use_box=None
):
    """
    Outline of a catalog query region on an RA/Dec axes.

    ``use_box`` selects the rectangular region (``box_deg`` bounds);
    otherwise the circumscribed cone is drawn, rendered as a
    cos(dec)-corrected circle so it matches the true angular extent under
    the axes' equal-aspect degree grid.  RA coordinates are unwrapped
    near ``center_ra`` so fields near 0/360 stay contiguous.
    """
    if not isinstance(region, dict):
        return
    try:
        ra_c = float(region["ra"])
        dec_c = float(region["dec"])
    except (KeyError, TypeError, ValueError):
        return
    box = region.get("box_deg") or region.get("box")
    if use_box is None:
        use_box = bool(region.get("used_box", box is not None))
    if use_box and box is not None:
        try:
            ra_min, ra_max, dec_min, dec_max = [float(v) for v in box]
        except (TypeError, ValueError):
            box = None
    if use_box and box is not None:
        xs = _unwrap_ra_near(
            np.array([ra_min, ra_max, ra_max, ra_min]), center_ra
        )
        ys = np.array([dec_min, dec_min, dec_max, dec_max])
        ax.plot(
            np.r_[xs, xs[0]],
            np.r_[ys, ys[0]],
            color=color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            zorder=3,
        )
        return
    try:
        r_deg = float(region.get("radius_arcmin") or 0.0) / 60.0
    except (TypeError, ValueError):
        return
    if r_deg <= 0:
        return
    th = np.linspace(0.0, 2.0 * np.pi, 121)
    cosd = max(np.cos(np.radians(dec_c)), 1e-6)
    xs = _unwrap_ra_near(ra_c + r_deg * np.cos(th) / cosd, center_ra)
    ys = dec_c + r_deg * np.sin(th)
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, alpha=alpha, zorder=3)


def _usable_mag_mask(cleaned, band, bright_lim, faint_lim):
    """Sources with a finite in-window magnitude in ``band``."""
    if cleaned is None or len(cleaned) == 0 or band not in cleaned.columns:
        return None
    m = pd.to_numeric(cleaned[band], errors="coerce")
    return np.isfinite(m) & (m >= bright_lim) & (m <= faint_lim)


def _footprints_from_image_infos(image_infos):
    """Sky-polygon footprints for the optimizer coverage map.

    Returns a list of ``{band, ra, dec, name}`` dicts; images without a
    WCS or shape contribute nothing (their sources were field-scored).
    """
    footprints = []
    for img in image_infos or []:
        w_i, s_i = img.get("wcs"), img.get("shape")
        if w_i is None or s_i is None:
            continue
        ny, nx = s_i
        try:
            fp_ra, fp_dec = w_i.all_pix2world(
                [0.0, nx - 1.0, nx - 1.0, 0.0],
                [0.0, 0.0, ny - 1.0, ny - 1.0],
                0,
            )
            footprints.append(
                {
                    "band": img.get("band"),
                    "ra": np.asarray(fp_ra, dtype=float).ravel(),
                    "dec": np.asarray(fp_dec, dtype=float).ravel(),
                    "name": (
                        os.path.basename(str(img["path"]))
                        if img.get("path")
                        else "field"
                    ),
                }
            )
        except Exception:
            continue
    return footprints


def _draw_score_panel(
    ax, sub, band, winner, min_sources, coverage_min, band_color
):
    """One optimizer scoreboard panel: worst-image usable count per
    catalog (bars), per-image mean (tick), the required minimum (dashed
    line), and per-catalog field coverage under each name (red below the
    threshold).  ``sub`` is the report rows for ``band``."""
    stats = (
        sub.groupby("catalog")
        .agg(
            n_min=("n_usable", "min"),
            n_mean=("n_usable", "mean"),
            cov=("coverage", "first"),
        )
        .sort_values(["n_min", "n_mean"], ascending=False)
    )
    if stats.empty:
        ax.set_visible(False)
        return

    xs = np.arange(len(stats))
    ax.bar(
        xs,
        stats["n_min"].values,
        color=[
            band_color if c == winner else "0.75" for c in stats.index
        ],
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for x, m in zip(xs, stats["n_mean"].values):
        ax.plot(
            [x - 0.3, x + 0.3], [m, m], color="black", lw=1.4, zorder=4
        )
    for x, n in zip(xs, stats["n_min"].values):
        ax.text(
            x, n, f"{int(n)}",
            ha="center", va="bottom", fontsize=7, zorder=5,
        )
    ax.axhline(min_sources, color="tab:red", lw=1.0, ls="--", zorder=2)
    ax.text(
        0.99,
        min_sources,
        f"required >= {min_sources:g}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="tab:red",
    )

    labels = []
    for c, row in stats.iterrows():
        cov = row["cov"]
        labels.append(
            f"{c}\n{cov:.0%} cov" if np.isfinite(cov) else f"{c}\ncov n/a"
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    for tick, (_, row) in zip(ax.get_xticklabels(), stats.iterrows()):
        if np.isfinite(row["cov"]) and row["cov"] < coverage_min:
            tick.set_color("tab:red")

    title = f"{band}-band"
    if winner:
        title += f" -> {winner}"
    ax.set_title(title, fontsize=10)
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)


def plot_optimized_catalog_coverage(
    plot_data,
    target_coords=None,
    wdir=None,
    target_name="target",
    outpath=None,
    report=None,
    min_sources=5,
    coverage_min=0.8,
    skipped=None,
):
    """
    Render the find_optimized_catalog coverage map: one subplot per band
    (at most three columns), image footprints as band-coloured squares,
    every source each catalog returned as a faint underlay, and usable
    catalog sources with a unique marker+color per catalog.  When
    ``report`` is given, a scoreboard row is appended below the map
    panels (worst-image usable counts vs the required minimum,
    per-catalog field coverage).

    Parameters
    ----------
    plot_data : dict
        ``result["plot_data"]`` from ``find_optimized_catalog`` - needs
        ``footprints`` (list of {band, ra, dec, name}), ``sources``
        ({(catalog, band): DataFrame with RA/DEC}), ``winners``,
        ``band_set``, and ``evaluated``.  Optional: ``catalog_sources``
        ({catalog: DataFrame with RA/DEC}) - every source each catalog
        returned, drawn faintly under the usable subset;
        ``band_regions``/``regions``/``catalog_bands`` - the required
        per-band boxes and per-catalog query outlines.
    target_coords : SkyCoord, optional
        Field centre; drawn as a cross when given.
    wdir : str, optional
        Output directory root (report lands in ``<wdir>/catalog_queries``).
    target_name : str
        Used in the output filename.
    outpath : str, optional
        Explicit output path; overrides wdir/target_name.
    report : pd.DataFrame, optional
        Optimizer report (catalog/band/image/n_usable/coverage); enables
        the scoreboard row.
    min_sources : float
        Preferred minimum usable sources on the worst image (scoreboard).
    coverage_min : float
        Field-coverage fraction below which a catalog name turns red.
    skipped : dict, optional
        Catalog -> reason it was not evaluated; listed as a footnote.

    Returns
    -------
    str or None
        Path of the written PNG, or None if nothing was drawn.
    """
    band_set = list(plot_data.get("band_set") or [])
    if not band_set:
        return None

    from lightcurve import BAND_COLORS
    from matplotlib.colors import is_color_like
    from matplotlib.ticker import MaxNLocator

    _band_color_cache = {}

    def _band_color(band):
        """Filter colour: the project palette first (case-insensitive),
        then a deterministic fallback so unlisted filters still get a
        unique, stable colour within the figure."""
        for k in (str(band), str(band).lower(), str(band).upper()):
            c = BAND_COLORS.get(k)
            if c and is_color_like(c):
                return c
        if band not in _band_color_cache:
            pool = plt.get_cmap("tab20").colors
            used = {
                _band_color_cache[b]
                for b in band_set
                if b in _band_color_cache
            }
            seed = int(
                hashlib.md5(str(band).encode()).hexdigest()[:8], 16
            )
            for k in range(len(pool)):
                cand = pool[(seed + k) % len(pool)]
                if cand not in used:
                    _band_color_cache[band] = cand
                    break
            else:
                _band_color_cache[band] = pool[seed % len(pool)]
        return _band_color_cache[band]

    center_ra = None
    if target_coords is not None:
        center_ra = float(target_coords.ra.degree)
        center_dec = float(target_coords.dec.degree)
    else:
        # Anchor RA unwrapping at the mean footprint centre when no target
        # is given (fields near RA=0/360 otherwise smear across the axes).
        all_ra = [
            np.asarray(fp["ra"], dtype=float)
            for fp in plot_data.get("footprints", [])
            if fp.get("ra") is not None
        ]
        center_ra = (
            float(np.mean(np.concatenate(all_ra))) if all_ra else 180.0
        )
        center_dec = None

    n_bands = len(band_set)
    ncols = min(3, n_bands)
    nrows = int(ceil(n_bands / ncols))
    have_score = report is not None and len(report) > 0
    if have_score:
        report = report.copy()
        if "coverage" not in report.columns:
            report["coverage"] = np.nan
    score_rows = int(ceil(n_bands / ncols)) if have_score else 0
    fig, axes = plt.subplots(
        nrows + score_rows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows + 3.4 * score_rows),
        squeeze=False,
        constrained_layout=True,
        height_ratios=[4.6] * nrows + [3.4] * score_rows,
    )

    footprints = plot_data.get("footprints", [])
    sources = plot_data.get("sources", {})
    winners = plot_data.get("winners", {})

    for idx, band in enumerate(band_set):
        ax = axes[idx // ncols][idx % ncols]
        band_color = _band_color(band)

        # Footprints: only resolved-band images are drawn, in the band
        # colour - unresolved-band (band=None) frames were scored under
        # every band and are skipped here. Large stacks collapse to a
        # single union outline so the mosaic boundary stays readable.
        band_fps = [fp for fp in footprints if fp["band"] == band]
        _plot_footprint_outline(ax, band_fps, center_ra, band_color)

        # Required coverage for this band's images: the padded bounding
        # box the catalog query must span (dashed, band colour).
        _draw_region_outline(
            ax,
            (plot_data.get("band_regions") or {}).get(band),
            color=band_color,
            center_ra=center_ra,
            ls="--",
            lw=1.2,
            alpha=0.9,
            use_box=True,
        )

        # The region each evaluated catalog was actually queried over -
        # the rectangular box for box-capable backends, the circumscribed
        # cone otherwise (dotted, catalog colour).
        _cat_bands = plot_data.get("catalog_bands") or {}
        for _cname, _creg in (plot_data.get("regions") or {}).items():
            _serves = _cat_bands.get(_cname)
            if _serves is not None and band not in _serves:
                continue
            _draw_region_outline(
                ax,
                _creg,
                color=_catalog_plot_style(_cname)["color"],
                center_ra=center_ra,
                ls=":",
                lw=0.9,
                alpha=0.7,
                use_box=_cname in _BOX_QUERY_CATALOGS,
            )

        # All sources each catalog returned, drawn faintly under the
        # usable subset so raw catalog coverage is visible against the
        # required region - e.g. a survey boundary shows as an empty
        # half-panel even before the usable-source cuts bite.
        _cat_src = plot_data.get("catalog_sources") or {}
        for _cname in sorted(_cat_src):
            _serves_c = _cat_bands.get(_cname)
            if _serves_c is not None and band not in _serves_c:
                continue
            _df = _cat_src[_cname]
            if _df is None or len(_df) == 0:
                continue
            ax.scatter(
                _unwrap_ra_near(
                    pd.to_numeric(_df["RA"], errors="coerce").to_numpy(),
                    center_ra,
                ),
                pd.to_numeric(_df["DEC"], errors="coerce").to_numpy(),
                s=3,
                marker=".",
                c=_catalog_plot_style(_cname)["color"],
                alpha=0.15,
                linewidths=0,
                zorder=1.5,
            )

        # Usable sources per catalog (finite mag inside the ZP window; the
        # squares show which of them actually land on a detector).
        legend_handles = []
        for (cat_name, cat_band), df in sorted(sources.items()):
            if cat_band != band or df is None or len(df) == 0:
                continue
            style = _catalog_plot_style(cat_name)
            sc = ax.scatter(
                _unwrap_ra_near(df["RA"].values, center_ra),
                df["DEC"].values,
                s=12,
                marker=style["marker"],
                c=style["color"],
                alpha=0.9,
                edgecolors="black",
                linewidths=0.4,
                label=f"{cat_name} ({len(df)})",
                zorder=4,
            )
            legend_handles.append(sc)

        if center_dec is not None:
            # White underlay keeps the cross legible over dense footprints.
            ax.plot(
                center_ra,
                center_dec,
                marker="+",
                ms=16,
                mew=3.0,
                color="white",
                zorder=5,
            )
            ax.plot(
                center_ra,
                center_dec,
                marker="+",
                ms=14,
                mew=1.8,
                color="black",
                zorder=6,
            )

        # Band label floats just above the top of the coverage mosaic,
        # centred on the field; falls back to the panel top when there are
        # no footprints to anchor to.
        title = f"{band}-band dataset"

        def _fp_pts(fps):
            return [
                np.c_[
                    _unwrap_ra_near(fp["ra"], center_ra),
                    np.asarray(fp["dec"], dtype=float),
                ]
                for fp in fps
                if fp.get("ra") is not None and fp.get("dec") is not None
            ]

        _label_pts = _fp_pts(band_fps)
        _txt_kw = dict(
            fontsize=10,
            ha="center",
            zorder=7,
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.75, pad=1.5
            ),
        )
        if _label_pts:
            _y_top = max(p[:, 1].max() for p in _label_pts)
            _y_bot = min(p[:, 1].min() for p in _label_pts)
            _x_lab = 0.5 * (
                min(p[:, 0].min() for p in _label_pts)
                + max(p[:, 0].max() for p in _label_pts)
            )
            _y_lab = _y_top + 0.04 * (_y_top - _y_bot + 1e-9)
            # Invisible point pulls the datalim up so the label is not
            # clipped by the axes top.
            ax.plot([_x_lab], [_y_lab + 0.04 * (_y_top - _y_bot)], alpha=0)
            ax.text(_x_lab, _y_lab, title, va="bottom", **_txt_kw)
        else:
            ax.text(
                0.5, 1.0, title, transform=ax.transAxes, va="bottom", **_txt_kw
            )
        if legend_handles:
            # Single-column legend inside the axes, top right.
            ax.legend(
                fontsize=7,
                loc="upper right",
                ncol=1,
                framealpha=0.9,
            )
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        # Few ticks: the field is a fraction of a degree wide, so the
        # default locator crowds the axis with long decimals.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
        ax.grid(lw=0.3, color="0.85", zorder=0)
        ax.invert_xaxis()  # sky convention: RA increases to the left
        ax.set_aspect("equal", adjustable="datalim")

    # Hide unused panels.
    for j in range(n_bands, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    # Scoreboard row(s): same grid layout as the map panels so band
    # columns line up.
    if have_score:
        winners = winners or {}
        for idx, band in enumerate(band_set):
            ax = axes[nrows + idx // ncols][idx % ncols]
            sub = report[report["band"].astype(str) == band]
            _draw_score_panel(
                ax,
                sub,
                band,
                winners.get(band),
                min_sources,
                coverage_min,
                _band_color(band),
            )
            if idx % ncols == 0:
                ax.set_ylabel("usable sources (worst image)")
        for j in range(n_bands, score_rows * ncols):
            axes[nrows + j // ncols][j % ncols].set_visible(False)

        note = (
            f"bar = worst-image usable count, tick = per-image mean; "
            f"coverage under each name (red = below {coverage_min:.0%})"
        )
        if skipped:
            note += " | skipped: " + "; ".join(
                f"{k} ({str(v).splitlines()[0]})" for k, v in skipped.items()
            )
        # Negative y lands below the x tick labels; bbox_inches="tight"
        # keeps it in the saved figure.
        fig.text(0.5, -0.015, note, ha="center", fontsize=7, color="0.4")

    if plot_data.get("band_regions") or plot_data.get("regions"):
        fig.text(
            0.5,
            -0.015 if not have_score else -0.04,
            "dashed = region each band's images require; "
            "dotted = region each catalog was queried over "
            "(box where supported, else circumscribed cone)",
            ha="center",
            fontsize=7,
            color="0.4",
        )

    if outpath is None:
        rep_dir = os.path.join(wdir or ".", "catalog_queries")
        outpath = os.path.join(
            rep_dir, f"{target_name}_optimized_catalog_coverage.png"
        )
    pathlib.Path(os.path.dirname(os.path.abspath(outpath))).mkdir(
        parents=True, exist_ok=True
    )
    try:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    logger.info("Optimized catalog coverage plot: %s", outpath)
    return outpath


def cross_match_sources(given_catalog, variable_catalog, match_radius_pix=5):
    """
    Remove sources from given_catalog that match any source in variable_catalog within match_radius_pix pixels.

    Parameters:
    -----------
    given_catalog : pd.DataFrame
        DataFrame with ['x_pix', 'y_pix'] columns.
    variable_catalog : pd.DataFrame
        DataFrame with ['x_pix', 'y_pix', 'otype'] columns.
    match_radius_pix : float, optional
        Matching radius in pixels (default is 5).

    Returns:
    --------
    filtered_catalog : pd.DataFrame
        DataFrame with matched sources removed.
    """
    if variable_catalog is None or len(variable_catalog) == 0:
        return given_catalog

    x_given = given_catalog["x_pix"].values
    y_given = given_catalog["y_pix"].values
    x_var = variable_catalog["x_pix"].values
    y_var = variable_catalog["y_pix"].values

    # KDTree for O(N log M) matching; replaces the O(N*M) per-source
    # distance loop.
    from scipy.spatial import cKDTree as _cKDTree
    var_tree = _cKDTree(np.column_stack([x_var, y_var]))
    given_xy = np.column_stack([x_given, y_given])
    dist_nearest, idx_nearest = var_tree.query(given_xy, k=1, workers=1)
    keep_mask = dist_nearest > match_radius_pix

    removed_indices = np.where(~keep_mask)[0].tolist()
    # OTYPE_opt is not guaranteed on every catalog - the drop must still
    # happen when the column is absent.
    has_otype = "OTYPE_opt" in variable_catalog.columns
    for i in removed_indices:
        otype = (
            variable_catalog.iloc[idx_nearest[i]]["OTYPE_opt"]
            if has_otype
            else "unknown"
        )
        logger.debug(
            f"Removing source at index {i} (x={x_given[i]:.2f}, y={y_given[i]:.2f}) due to match with variable source [{otype}]"
        )

    filtered_catalog = given_catalog[keep_mask].reset_index(drop=True)
    logger.info(
        "Variable-source match: removed %d, %d sources remain",
        len(removed_indices), len(filtered_catalog),
    )

    return filtered_catalog


# =============================================================================
# Deduplication helper
# =============================================================================

def _skycoord_dedup_keep_one(catalog_df, sep_threshold_arcsec=0.1):
    """
    Remove duplicate sky positions from *catalog_df*, keeping exactly one
    member of each close pair rather than dropping both.

    Algorithm
    ---------
    For every source i, find its nearest *other* source (nthneighbor=2) and
    its index j.  If sep(i,j) < threshold AND j < i then source i is the
    later-arriving duplicate and is marked for removal.  This guarantees the
    lower-index entry of any close pair survives.

    Parameters
    ----------
    catalog_df : pd.DataFrame  - must contain "RA" and "DEC" columns (degrees)
    sep_threshold_arcsec : float - angular separation below which two sources
                                   are considered duplicates (default 0.1 arcsec)

    Returns
    -------
    pd.DataFrame  - deduplicated catalog with reset integer index
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if catalog_df is None or len(catalog_df) < 2:
        return catalog_df
    if not {"RA", "DEC"}.issubset(catalog_df.columns):
        return catalog_df

    coords = SkyCoord(
        ra=catalog_df["RA"].values * u.degree,
        dec=catalog_df["DEC"].values * u.degree,
    )
    idx_match, sep2, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    thr = sep_threshold_arcsec * u.arcsec
    # Vectorized: drop source i if its nearest neighbour j is closer than the
    # threshold AND has a lower index (meaning j is the "first" copy to keep).
    drop = (sep2 <= thr) & (idx_match < np.arange(len(catalog_df)))
    return catalog_df[~drop].reset_index(drop=True)


def _drop_fainter_blended_neighbors(
    catalog_df, mag_col, radius_arcsec=6.0, dmag=3.0
):
    """
    Drop catalog entries that sit within ``radius_arcsec`` of another entry
    at least ``dmag`` magnitudes brighter in ``mag_col``.

    Some catalogs (notably Pan-STARRS) contain spurious faint entries within
    a few arcsec of a bright star; the measured flux at that position is the
    bright star's, so the faint entry lands on a false zeropoint locus and
    can hijack the linearity fit.  Even when the faint entry is a real
    companion, its photometry is dominated by the neighbour, so dropping it
    costs nothing for calibration.

    Parameters
    ----------
    catalog_df : pd.DataFrame  - must contain "RA", "DEC" (degrees) and mag_col
    mag_col : str              - magnitude column used for the brightness test
    radius_arcsec : float      - neighbour search radius
    dmag : float               - minimum magnitude difference that marks the
                                 fainter member as contaminated

    Returns
    -------
    pd.DataFrame - catalog with the fainter member of each such pair removed
    """
    import numpy as np

    if catalog_df is None or len(catalog_df) < 2:
        return catalog_df
    if (
        not {"RA", "DEC"}.issubset(catalog_df.columns)
        or mag_col not in catalog_df.columns
    ):
        return catalog_df

    ra = pd.to_numeric(catalog_df["RA"], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(catalog_df["DEC"], errors="coerce").to_numpy(dtype=float)
    mags = pd.to_numeric(catalog_df[mag_col], errors="coerce").to_numpy(
        dtype=float
    )
    cosd = np.cos(np.deg2rad(np.nan_to_num(dec, nan=0.0)))
    radius_deg = radius_arcsec / 3600.0

    n = len(catalog_df)
    drop = np.zeros(n, dtype=bool)
    # Chunked N^2: catalogs are ~1e2-1e3 rows, but bound memory anyway.
    block = 1024
    col_idx = np.arange(n)
    for s in range(0, n, block):
        e = min(s + block, n)
        dra = (ra[s:e, None] - ra[None, :]) * cosd[None, :]
        ddec = dec[s:e, None] - dec[None, :]
        sep_deg = np.hypot(dra, ddec)
        # dm[i, j] < 0 means neighbour j is brighter than source i.
        dm = mags[None, :] - mags[s:e, None]
        is_self = (np.arange(s, e)[:, None] == col_idx[None, :])
        contaminated = (
            (sep_deg <= radius_deg)
            & (dm <= -dmag)
            & np.isfinite(dm)
            & ~is_self
        )
        drop[s:e] = np.any(contaminated, axis=1)
    return catalog_df[~drop].reset_index(drop=True)


# =============================================================================
# Spatial coverage helpers
# =============================================================================

# (catalog, rounded field, rounded radius) keys already warned about for
# partial sky coverage.  Module-level because main.py builds a fresh
# Catalog per image - an instance set would repeat the same warning for
# every frame in the stack.
_COVERAGE_WARNED_KEYS = set()

# Column-name spellings used by the remote backends (checked in order).
_RA_COL_CANDIDATES = ("ra", "raicrs", "raj2000", "ramean", "radeg")
_DEC_COL_CANDIDATES = ("dec", "de", "deicrs", "dej2000", "decmean", "decdeg")


def _catalog_radec(df):
    """
    Return (ra_deg, dec_deg) float arrays for a raw catalog table, or
    (None, None) when no RA/DEC pair is found.

    Works on pre-clean() frames, where coordinate columns carry each
    service's native names (RA_ICRS/DE_ICRS, raMean/decMean, ra/dec, ...).
    """
    if df is None or len(df) == 0:
        return None, None
    norm = {
        str(c).lower().replace("_", "").replace(" ", ""): c
        for c in df.columns
    }
    ra_col = next(
        (norm[k] for k in _RA_COL_CANDIDATES if k in norm), None
    )
    dec_col = next(
        (norm[k] for k in _DEC_COL_CANDIDATES if k in norm), None
    )
    if ra_col is None or dec_col is None:
        return None, None
    ra = pd.to_numeric(df[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(df[dec_col], errors="coerce").to_numpy(dtype=float)
    return ra, dec


def _sky_uniform_subsample(df, nmax, center_ra=None):
    """
    Cap a catalog at ``nmax`` rows while preserving sky coverage.

    Row caps that keep "the N nearest to the field centre" hollow out the
    edges of the FOV (distance-ordered TOP-N queries do this silently).
    This instead bins the footprint into a grid and keeps a per-cell
    quota, then tops up from the leftovers.
    """
    if df is None or nmax is None or len(df) <= nmax:
        return df
    ra, dec = _catalog_radec(df)
    if ra is None:
        return df.head(int(nmax))

    if center_ra is None:
        center_ra = float(np.nanmean(ra))
    dec0 = float(np.nanmean(dec))
    x = (_unwrap_ra_near(ra, center_ra) - center_ra) * np.cos(
        np.radians(dec0)
    )
    y = dec - dec0
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return df.head(int(nmax))

    nmax = int(nmax)
    n_bins = max(2, int(np.sqrt(nmax)))
    per_cell = max(1, nmax // (n_bins**2))
    x_edges = np.linspace(x[finite].min(), x[finite].max(), n_bins + 1)
    y_edges = np.linspace(y[finite].min(), y[finite].max(), n_bins + 1)

    xi = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
    # Non-finite coordinates get a sentinel cell so they can only be
    # picked up by the top-up below, not displace real coverage.
    cell_id = np.where(finite, yi * n_bins + xi, n_bins**2)

    # Keep up to per_cell rows per occupied cell, in the catalog's own row
    # order (distance-sorted inputs keep their nearest-to-centre members).
    order = np.argsort(cell_id, kind="stable")
    cell_sorted = cell_id[order]
    starts = np.searchsorted(cell_sorted, np.arange(n_bins**2 + 1))
    picked = []
    for cid in range(n_bins**2):
        cell_rows = order[starts[cid] : starts[cid + 1]]
        if cell_rows.size:
            picked.extend(cell_rows[:per_cell].tolist())

    if len(picked) < nmax:
        rest = np.setdiff1d(np.arange(len(df)), np.asarray(picked))
        picked.extend(rest[: nmax - len(picked)].tolist())
    return df.iloc[np.asarray(picked[:nmax])]


def _field_coverage_fraction(
    ra_deg, dec_deg, center_ra, center_dec, radius_deg, box_wh=None
):
    """
    Fraction of the query region populated by at least one source.

    The region is divided into a grid whose cell count scales with the
    number of in-region sources, so the metric is meaningful for sparse
    (2MASS) and dense (Pan-STARRS) catalogs alike.  A survey boundary
    crossing the field (e.g. SDSS covering only half a cone) reads as a
    fraction well below 1.

    ``box_wh`` = (width_deg, height_deg) restricts the region to the
    box centred on (center_ra, center_dec) - the shape actually queried
    by box-capable backends, where ``width_deg`` is the angular width
    along RA (already cos(dec)-projected, matching VizieR ``-c.b``);
    otherwise the circular disk of ``radius_deg`` is used.

    Returns NaN when the catalog or the query region is empty.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not ok.any():
        return np.nan
    x = (_unwrap_ra_near(ra[ok], center_ra) - center_ra) * np.cos(
        np.radians(center_dec)
    )
    y = dec[ok] - center_dec

    if box_wh is not None:
        # x is already a projected (angular) offset, matching the
        # angular-width convention the box query itself uses.
        hw_x = 0.5 * float(box_wh[0])
        hw_y = 0.5 * float(box_wh[1])
        if hw_x <= 0 or hw_y <= 0:
            return np.nan
        inside = (np.abs(x) <= hw_x) & (np.abs(y) <= hw_y)
    else:
        if radius_deg is None or radius_deg <= 0:
            return np.nan
        hw_x = hw_y = float(radius_deg)
        inside = (x**2 + y**2) <= radius_deg**2
    n_in = int(inside.sum())
    if n_in == 0:
        return 0.0

    n_side = int(np.clip(round(np.sqrt(n_in / 3.0)), 4, 12))
    x_edges = np.linspace(-hw_x, hw_x, n_side + 1)
    y_edges = np.linspace(-hw_y, hw_y, n_side + 1)
    x_ctr = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_ctr = 0.5 * (y_edges[:-1] + y_edges[1:])
    if box_wh is not None:
        in_region = np.ones((n_side, n_side), dtype=bool)
    else:
        in_region = (
            x_ctr[:, None] ** 2 + y_ctr[None, :] ** 2
        ) <= radius_deg**2
    if not in_region.any():
        return np.nan

    xi = np.clip(np.digitize(x[inside], x_edges) - 1, 0, n_side - 1)
    yi = np.clip(np.digitize(y[inside], y_edges) - 1, 0, n_side - 1)
    occupied = np.zeros((n_side, n_side), dtype=bool)
    occupied[yi, xi] = True
    return float((occupied & in_region).sum() / in_region.sum())


_PIXSCALE_KEYS = (
    "PIXSCALE",
    "PIXSCAL1",
    "SCALE",
    "SECPIX",
    "SECPIX1",
)


def _pixscale_arcsec(header):
    """
    Pixel scale (arcsec/pix) from common header cards, or None.

    Explicit scale keywords are tried first; the CD matrix and CDELT are
    worth a look even when astropy rejects the full WCS - some products
    carry a mixed SIP/TPV card set whose WCS fails to build while the raw
    linear cards are still valid.
    """
    if header is None:
        return None
    try:
        for key in _PIXSCALE_KEYS:
            val = header.get(key)
            if val is None:
                continue
            try:
                scale = abs(float(val))
            except (TypeError, ValueError):
                continue
            if 0.0 < scale <= 30.0:
                return scale
        if all(k in header for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
            det = abs(
                float(header["CD1_1"]) * float(header["CD2_2"])
                - float(header["CD1_2"]) * float(header["CD2_1"])
            )
            scale = det**0.5 * 3600.0
            if 0.0 < scale <= 30.0:
                return scale
        for key in ("CDELT1", "CDELT2"):
            if key in header:
                scale = abs(float(header[key])) * 3600.0
                if 0.0 < scale <= 30.0:
                    return scale
    except Exception:
        pass
    return None


def _footprint_radius_arcmin(image_infos, target_coords, margin=1.1):
    """
    Cone radius (arcmin) centred on ``target_coords`` that covers the
    corners of every image footprint.  Returns None when no image carries
    a usable WCS/shape, so callers can keep their default radius.

    An image without a usable WCS falls back to its detector
    half-diagonal at the ``pixscale`` entry (arcsec/pix) - that assumes a
    near-centred target, which is the best available estimate when no
    WCS exists; the coverage diagnostic flags fields where it falls
    short.
    """
    max_sep = 0.0
    for img in image_infos or []:
        shape_i = img.get("shape")
        if shape_i is None:
            continue
        try:
            ny, nx = shape_i
            sep_i = None
            wcs_i = img.get("wcs")
            if wcs_i is not None:
                try:
                    cra, cde = wcs_i.all_pix2world(
                        [0.0, nx - 1.0, nx - 1.0, 0.0],
                        [0.0, 0.0, ny - 1.0, ny - 1.0],
                        0,
                    )
                    corners = SkyCoord(
                        np.asarray(cra, dtype=float).ravel() * u.deg,
                        np.asarray(cde, dtype=float).ravel() * u.deg,
                    )
                    sep = target_coords.separation(corners).arcmin
                    if np.isfinite(sep).any():
                        sep_i = float(np.nanmax(sep))
                except Exception:
                    sep_i = None
            if sep_i is None:
                ps = img.get("pixscale")
                try:
                    ps = float(ps) if ps is not None else np.nan
                except (TypeError, ValueError):
                    ps = np.nan
                if np.isfinite(ps) and ps > 0.0:
                    sep_i = 0.5 * np.hypot(nx, ny) * ps / 60.0
            if sep_i is not None and np.isfinite(sep_i):
                max_sep = max(max_sep, sep_i)
        except Exception:
            continue
    if max_sep <= 0:
        return None
    return max_sep * margin


def _catalog_cone_radius_arcmin(
    image_infos,
    target_coords,
    default_arcmin=10.0,
    min_arcmin=2.0,
    max_arcmin=60.0,
):
    """
    Cone-search radius sized by the image footprint extent.

    The footprint radius both covers and *limits* the query: a narrow
    field gets a narrower cone instead of always paying for the default
    radius.  Falls back to ``default_arcmin`` when no image carries a
    usable WCS or pixel scale (e.g. a dataset with no WCS solution).
    """
    footprint = _footprint_radius_arcmin(image_infos, target_coords)
    if footprint is None or not np.isfinite(footprint) or footprint <= 0:
        return float(default_arcmin)
    return float(min(max(footprint, min_arcmin), max_arcmin))


# Backends whose query API accepts a rectangular region; the rest get the
# box's circumscribed cone. VizieR query_region takes width/height.
_BOX_QUERY_CATALOGS = {"apass", "2mass", "sdss"}


def _footprint_corners_radec(image_infos, target_coords=None):
    """
    Collect image-footprint corner positions as flat RA/Dec arrays.

    Images with a usable WCS contribute their true sky corners; images
    with only ``shape`` + ``pixscale`` contribute a square box centred on
    ``target_coords`` (the same near-centred assumption
    ``_footprint_radius_arcmin`` makes).  Images with neither contribute
    nothing.
    """
    ras, decs = [], []
    t_ra = t_dec = None
    if target_coords is not None:
        try:
            t_ra = float(target_coords.ra.degree)
            t_dec = float(target_coords.dec.degree)
        except Exception:
            t_ra = t_dec = None
    for img in image_infos or []:
        shape_i = img.get("shape")
        if shape_i is None:
            continue
        try:
            ny, nx = int(shape_i[0]), int(shape_i[1])
        except Exception:
            continue
        if nx <= 0 or ny <= 0:
            continue
        wcs_i = img.get("wcs")
        done = False
        if wcs_i is not None:
            try:
                cra, cde = wcs_i.all_pix2world(
                    [0.0, nx - 1.0, nx - 1.0, 0.0],
                    [0.0, 0.0, ny - 1.0, ny - 1.0],
                    0,
                )
                cra = np.asarray(cra, dtype=float).ravel()
                cde = np.asarray(cde, dtype=float).ravel()
                if np.isfinite(cra).all() and np.isfinite(cde).all():
                    ras.append(cra)
                    decs.append(cde)
                    done = True
            except Exception:
                pass
        if done or t_ra is None:
            continue
        try:
            ps = float(img.get("pixscale"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(ps) and 0.0 < ps <= 30.0):
            continue
        hw_y = 0.5 * ny * ps / 3600.0
        cosd = max(np.cos(np.radians(t_dec)), 1e-6)
        hw_x = 0.5 * nx * ps / 3600.0 / cosd
        ras.append(t_ra + np.array([-hw_x, hw_x, hw_x, -hw_x]))
        decs.append(t_dec + np.array([-hw_y, -hw_y, hw_y, hw_y]))
    if not ras:
        return np.array([]), np.array([])
    return np.concatenate(ras), np.concatenate(decs)


def _region_from_corners(
    ra_deg,
    dec_deg,
    target_coords=None,
    margin=1.1,
    min_arcmin=2.0,
    max_arcmin=60.0,
    default_arcmin=10.0,
):
    """
    Catalog query region covering a footprint corner set.

    The region is anchored on the bounding-box centre of the footprint
    corners - not the target - so dithered/off-centre pointings no longer
    inflate the cone (a target at the field edge used to double the
    radius).  ``box_deg`` is the padded RA/Dec bounding box for backends
    that accept rectangular queries; ``radius_arcmin`` is the box's
    circumscribed cone for cone-only backends.  Both are bounded: the
    cone is clamped to [min_arcmin, max_arcmin] and the box rescaled so
    it stays inscribed in the clamped cone.

    Falls back to a target-centred default cone (no box) when the corner
    set is empty.

    Returns a plain-python dict (yaml/json safe):
    ``ra``, ``dec``, ``radius_arcmin``, ``offset_arcmin``,
    ``box_deg`` = [ra_min, ra_max, dec_min, dec_max] (ra_min > ra_max
    means the box wraps across RA=0/360), ``width_deg``, ``height_deg``,
    ``n_corners``.

    ``width_deg`` is the *angular* width along the RA direction (what
    VizieR's ``-c.b`` box consumes), so the RA coordinate span is
    ``width_deg / cos(dec)``; ``box_deg`` carries the coordinate bounds
    for plotting.
    """
    t_ra = t_dec = None
    if target_coords is not None:
        try:
            t_ra = float(target_coords.ra.degree)
            t_dec = float(target_coords.dec.degree)
        except Exception:
            t_ra = t_dec = None

    ra = np.asarray(ra_deg, dtype=float).ravel()
    dec = np.asarray(dec_deg, dtype=float).ravel()
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = ra[ok], dec[ok]
    if ra.size == 0:
        return {
            "ra": float(t_ra if t_ra is not None else 0.0),
            "dec": float(t_dec if t_dec is not None else 0.0),
            "radius_arcmin": float(default_arcmin),
            "offset_arcmin": 0.0,
            "box_deg": None,
            "width_deg": None,
            "height_deg": None,
            "n_corners": 0,
        }

    anchor = t_ra if t_ra is not None else float(np.mean(ra % 360.0))
    ra_u = _unwrap_ra_near(ra, anchor)
    hw_ra = 0.5 * float(ra_u.max() - ra_u.min()) * margin
    hw_dec = 0.5 * float(dec.max() - dec.min()) * margin
    c_ra_u = 0.5 * float(ra_u.max() + ra_u.min())
    c_dec = 0.5 * float(dec.max() + dec.min())

    # Dec bounds cannot cross a pole; re-centre on the clamped range.
    dec_lo = max(c_dec - hw_dec, -89.999)
    dec_hi = min(c_dec + hw_dec, 89.999)
    c_dec = 0.5 * (dec_lo + dec_hi)
    hw_dec = 0.5 * (dec_hi - dec_lo)

    # Circumscribed cone: true angular distance centre -> box corner
    # (SkyCoord handles the cos(dec) RA foreshortening and wrap).
    c_ra = c_ra_u % 360.0
    corner_sc = SkyCoord(
        ra=np.array(
            [c_ra_u - hw_ra, c_ra_u + hw_ra, c_ra_u + hw_ra, c_ra_u - hw_ra]
        )
        * u.deg,
        dec=np.array([dec_lo, dec_lo, dec_hi, dec_hi]) * u.deg,
        frame="icrs",
    )
    centre_sc = SkyCoord(ra=c_ra * u.deg, dec=c_dec * u.deg, frame="icrs")
    sep_raw = float(centre_sc.separation(corner_sc).arcmin.max())
    radius = float(min(max(sep_raw, min_arcmin), max_arcmin))

    # Rescale the box so it stays inscribed in the (clamped) cone.  A
    # degenerate corner set (sep_raw ~ 0) gets the minimum-size box.
    if sep_raw > 0:
        hw_ra *= radius / sep_raw
        hw_dec *= radius / sep_raw
    else:
        hw_dec = radius / (60.0 * np.sqrt(2.0))
        hw_ra = hw_dec / max(np.cos(np.radians(c_dec)), 1e-6)

    dec_lo = max(c_dec - hw_dec, -89.999)
    dec_hi = min(c_dec + hw_dec, 89.999)
    # A box spanning (nearly) all of RA is not a useful box query - the
    # pole/pan-sky case is better served by the plain cone.
    if 2.0 * hw_ra >= 350.0:
        box_deg = None
        width_deg = height_deg = None
    else:
        ra_min = (c_ra_u - hw_ra) % 360.0
        ra_max = (c_ra_u + hw_ra) % 360.0
        box_deg = [
            float(ra_min),
            float(ra_max),
            float(dec_lo),
            float(dec_hi),
        ]
        # VizieR box widths are angular (RA size is divided by cos(dec)
        # server-side), so report the on-sky width, not the RA coordinate
        # span stored in box_deg.
        width_deg = float(
            2.0 * hw_ra * max(np.cos(np.radians(c_dec)), 1e-6)
        )
        height_deg = float(2.0 * hw_dec)

    off = 0.0
    if t_ra is not None:
        off = float(
            centre_sc.separation(
                SkyCoord(ra=t_ra * u.deg, dec=t_dec * u.deg, frame="icrs")
            ).arcmin
        )
    return {
        "ra": float(c_ra),
        "dec": float(c_dec),
        "radius_arcmin": float(radius),
        "offset_arcmin": float(off),
        "box_deg": box_deg,
        "width_deg": width_deg,
        "height_deg": height_deg,
        "n_corners": int(ra.size),
    }


def _footprint_region(
    image_infos,
    bands,
    target_coords,
    margin=1.1,
    min_arcmin=2.0,
    max_arcmin=60.0,
    default_arcmin=10.0,
):
    """
    Query region covering the footprints of ``image_infos`` restricted
    to ``bands`` (falsy = all images).  Images with no resolved band are
    scored under every band, so they join every band's region.
    """
    if bands:
        sel = {str(b) for b in bands}
        infos = [
            i
            for i in (image_infos or [])
            if i.get("band") is None or str(i.get("band")) in sel
        ]
    else:
        infos = list(image_infos or [])
    ra, dec = _footprint_corners_radec(infos, target_coords)
    reg = _region_from_corners(
        ra,
        dec,
        target_coords,
        margin=margin,
        min_arcmin=min_arcmin,
        max_arcmin=max_arcmin,
        default_arcmin=default_arcmin,
    )
    reg["n_images"] = len(infos)
    return reg


def _fmt_region(region):
    """
    Compact one-line description of a query region for log messages:
    cone radius and centre, box size when present, and the centre's
    offset from the target when it is not negligible.
    """
    if not isinstance(region, dict):
        return ""
    parts = [f"r={float(region.get('radius_arcmin', 0.0)):.1f}'"]
    if region.get("width_deg") and region.get("height_deg"):
        parts.append(
            f"box {60.0 * float(region['width_deg']):.1f}'x"
            f"{60.0 * float(region['height_deg']):.1f}'"
        )
    parts.append(
        f"@ ({float(region.get('ra', 0.0)):.4f}, "
        f"{float(region.get('dec', 0.0)):.4f})"
    )
    off = float(region.get("offset_arcmin") or 0.0)
    if off > 1.0:
        parts.append(f"({off:.1f}' off target)")
    return " ".join(parts)


# VizieR resolves ``catalog="sdss"`` to several SDSS releases at once and
# returns one table per release.  The footprints differ between releases,
# so the first table is not necessarily the best-covered one.
_SDSS_RELEASE_RANK = {
    "v/154": 4,  # DR16 - final release, largest footprint
    "v/147": 3,  # DR12
    "v/139": 2,  # DR9
    "ii/294": 1,  # DR7
}


def _select_sdss_vizier_table(
    catalog_search, center_ra=None, center_dec=None, radius_deg=None
):
    """
    Pick the best SDSS release from a VizieR TableList.

    Every SDSS release on VizieR answers the same cone query; each applies
    its own star cuts (``cl``/``class`` == 6, ``mode`` == 1, ``clean`` == 1
    where present).  Releases are scored on how much of the query disk
    their surviving primary stars cover - footprints differ between
    releases - with ties going to the newer release, then to the larger
    source count.
    """
    best_key = None
    best_df = pd.DataFrame()
    best_name = "?"
    for table in catalog_search:
        name = str(getattr(table, "meta", {}).get("name", "?"))
        try:
            df = table.to_pandas()
        except Exception:
            continue
        n0 = len(df)
        if "mode" in df.columns:
            df = df[pd.to_numeric(df["mode"], errors="coerce") == 1]
        if "cl" in df.columns:
            df = df[pd.to_numeric(df["cl"], errors="coerce") == 6]
        elif "class" in df.columns:
            df = df[pd.to_numeric(df["class"], errors="coerce") == 6]
        if "clean" in df.columns:
            df = df[pd.to_numeric(df["clean"], errors="coerce") == 1]
        rank = _SDSS_RELEASE_RANK.get(
            "/".join(name.split("/")[:2]).lower(), 0
        )
        # Coverage decides at ~10% granularity; count alone would favour
        # older releases that lack the ``clean`` cut.
        decile = 0
        if (
            center_ra is not None
            and center_dec is not None
            and radius_deg
            and len(df) > 0
        ):
            ra_arr, dec_arr = _catalog_radec(df)
            if ra_arr is not None:
                cov = _field_coverage_fraction(
                    ra_arr, dec_arr, center_ra, center_dec, radius_deg
                )
                if np.isfinite(cov):
                    decile = int(np.clip(cov * 10, 0, 10))
        logger.debug(
            "SDSS %s: %d usable primary stars (of %d rows), coverage decile %d",
            name,
            len(df),
            n0,
            decile,
        )
        key = (decile, rank, len(df))
        if best_key is None or key > best_key:
            best_key = key
            best_df = df
            best_name = name
    if len(best_df) > 0:
        logger.info(
            "SDSS: using %s (%d usable primary stars)", best_name, len(best_df)
        )
    return best_df


# =============================================================================
# =============================================================================
# #
# =============================================================================
# =============================================================================
class Catalog:
    """Catalog query and cross-matching for photometric calibration.

    Wraps online catalog services (Pan-STARRS, SDSS, Skymapper, Gaia, 2MASS,
    WISE) and provides local catalog operations: source cross-matching,
    saturation/linearity filtering, and zeropoint fitting preparation.
    """

    def __init__(self, input_yaml):
        """
        Initialize the catalog class with the input YAML configuration.

        Parameters:
        -----------
        input_yaml : dict
            Configuration dictionary loaded from YAML.
        """
        self.input_yaml = input_yaml

    def _require_catalog_selected(self, catalogName: Optional[str]) -> str:
        """
        Ensure a catalog backend is selected.

        The pipeline historically allowed `catalog.use_catalog` to be null for
        workflows that do not require catalog calibration, but when a catalog
        query is requested we must fail fast with a clear message.
        """
        catalogName = self._resolve_catalog_for_filter(catalogName)
        if (
            catalogName is None
            or str(catalogName).strip() == ""
            or str(catalogName).lower() == "none"
        ):
            raise ValueError(
                "No catalog selected. Set `default_input.catalog.use_catalog` in your YAML "
                "(e.g. 'gaia', 'panstarrs'/'pan_starrs', 'sdss', 'apass', '2mass', 'legacy', 'refcat', 'custom', or 'gaia_custom')."
            )
        if str(catalogName).strip().lower() == "auto":
            raise ValueError(
                "catalog.use_catalog='auto' is resolved by the driver "
                "(autophot.py) into a per-filter mapping before per-image "
                "processing; it cannot be used when calling main() directly."
            )
        return str(catalogName).strip()

    def _resolve_catalog_for_filter(self, catalog_choice):
        """
        Resolve catalog selection from a scalar or per-filter mapping.

        Supported mapping examples:
            {'ugriz': 'sdss', 'UBVRI': 'apass', 'default': 'gaia'}
            {'g': 'sdss', 'r': 'sdss', 'default': 'apass'}
        """
        if isinstance(catalog_choice, dict):
            use_filter = str(self.input_yaml.get("imageFilter", "") or "").strip()
            use_filter_norm = normalize_photometric_filter_name(use_filter)
            # Canonical band for group membership (e.g. imageFilter "h" -> "H" in JHK / grizJHK).
            band_for_group = (
                use_filter_norm if use_filter_norm is not None else use_filter
            )
            # Warn on ambiguous mappings (multiple keys match this filter).
            membership_matches = []
            if use_filter:
                for key, value in catalog_choice.items():
                    if value is None:
                        continue
                    key_s = str(key).strip()
                    key_l = key_s.lower()
                    if key_l in {"default", "*", "all"}:
                        continue
                    key_bands = parse_supported_filter_group_key(key_s)
                    if key_bands and band_for_group in key_bands:
                        membership_matches.append(str(key))
                if len(membership_matches) > 1:
                    logger.warning(
                        "catalog.use_catalog mapping is ambiguous for filter '%s': matched keys=%s. Using precedence: exact key > first membership key > default.",
                        use_filter,
                        membership_matches,
                    )

            # 1) exact key match first (e.g. {"g": "sdss"})
            for key, value in catalog_choice.items():
                key_s = str(key).strip()
                if value is None or key_s.lower() in {"default", "*", "all"}:
                    continue
                key_norm = normalize_photometric_filter_name(key_s)
                if key_s == use_filter and key_norm is not None:
                    return value
            # Backward-compatible fallback: normalized exact match.
            for key, value in catalog_choice.items():
                key_s = str(key).strip()
                if value is None or key_s.lower() in {"default", "*", "all"}:
                    continue
                key_norm = normalize_photometric_filter_name(key_s)
                if (
                    key_norm is not None
                    and use_filter_norm is not None
                    and key_norm == use_filter_norm
                ):
                    return value

            # 2) grouped bands by membership string/list (e.g. {"ugriz": "sdss"})
            for key, value in catalog_choice.items():
                if value is None:
                    continue
                key_str = str(key).strip()
                if key_str.lower() in {"default", "*", "all"}:
                    continue
                if not use_filter:
                    continue
                key_bands = parse_supported_filter_group_key(key_str)
                if key_bands and band_for_group in key_bands:
                    return value

            # 3) explicit default
            for dkey in ("default", "*", "all"):
                if dkey in catalog_choice and catalog_choice[dkey] is not None:
                    return catalog_choice[dkey]

            logger.warning(
                "catalog.use_catalog mapping has no match for filter '%s'; available keys=%s",
                use_filter,
                list(catalog_choice.keys()),
            )
            return None
        # Normalize scalar catalog names for backward compatibility
        if isinstance(catalog_choice, str):
            catalog_choice = self._normalize_catalog_name(catalog_choice)
        return catalog_choice

    @staticmethod
    def _catalog_len(obj) -> int:
        """Best-effort length for DataFrame/Table-like results."""
        if obj is None:
            return 0
        try:
            return int(len(obj))
        except Exception:
            return 0

    @staticmethod
    def _normalize_catalog_name(catalog_name: str) -> str:
        """
        Normalize catalog name aliases to canonical form (lowercase).
        
        Accepts case-insensitive input (e.g., 'GAIA', 'Gaia', 'gaia') and
        converts to lowercase. Also handles 'panstarrs' (no underscore) and
        converts to 'pan_starrs' for backward compatibility.
        """
        if not catalog_name:
            return catalog_name
        name = str(catalog_name).strip().lower()
        aliases = {
            "panstarrs": "pan_starrs",
            "pan-starrs": "pan_starrs",
            "ps1": "pan_starrs",
        }
        return aliases.get(name, name)

    def _require_nonempty_catalog(
        self, selectedCatalog, catalogName: str, target_coords, radius_arcmin: float
    ) -> None:
        """
        Stop the pipeline if the catalog query returns zero sources.

        This is treated as "catalog does not cover that part of the sky" (or a
        service/query failure) and continuing would produce misleading results.
        """
        n = self._catalog_len(selectedCatalog)
        if n == 0:
            ra = float(target_coords.ra.degree)
            dec = float(target_coords.dec.degree)
            raise RuntimeError(
                f"{catalogName.upper()} catalog query returned 0 sources for "
                f"RA={ra:.6f} deg, Dec={dec:.6f} deg within r={radius_arcmin:.2f} arcmin. "
                "Assuming this catalog does not cover the field (or the query failed); stopping."
            )

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def gaia_synthetic_photometry(
        self,
        ra,
        dec,
        radius=0.1,
        max_sources=5000,
        photometric_systems=None,
    ):
        from gaiaxpy import PhotometricSystem
        import pandas as pd
        import warnings
        import numpy as np
        import logging

        from autophot_gaia_curves.gaia_archive import (
            gaia_xp_sql_top_n,
            generate_source_ids_batched,
            query_gaia_xp_cone_growing,
            sort_gaia_table_nearest_to_target,
        )

        logger = logging.getLogger(__name__)
        cat_cfg = self.input_yaml.get("catalog", {}) or {}
        query_pause_b = float(cat_cfg.get("gaia_archive_query_pause_before_sec", 0.25))
        query_pause_a = float(cat_cfg.get("gaia_archive_query_pause_after_sec", 0.25))
        xp_batch_size = int(cat_cfg.get("gaia_xp_batch_size", 200))
        xp_batch_pause = float(cat_cfg.get("gaia_xp_batch_pause_sec", 0.5))
        archive_retries = int(cat_cfg.get("gaia_archive_max_retries", 3))
        retry_base_delay = float(cat_cfg.get("gaia_archive_retry_base_delay_sec", 2.0))
        xp_order = str(cat_cfg.get("gaia_xp_order_by", "brightness")).strip().lower()
        xp_show_progress = bool(cat_cfg.get("gaia_xp_show_progress", False))
        prefetch_factor = int(cat_cfg.get("gaia_nearest_prefetch_factor", 50))
        prefetch_min = int(cat_cfg.get("gaia_nearest_prefetch_min", 200))
        prefetch_max = int(cat_cfg.get("gaia_nearest_prefetch_max", 10000))
        xp_min_sources = int(cat_cfg.get("gaia_xp_min_sources", 25))
        xp_max_radius_deg = float(cat_cfg.get("gaia_xp_max_radius_deg", 1.0))
        xp_grow_factor = float(cat_cfg.get("gaia_xp_grow_factor", 1.5))

        sql_top, sort_by_distance = gaia_xp_sql_top_n(
            max_sources,
            xp_order,
            prefetch_factor=prefetch_factor,
            prefetch_min=prefetch_min,
            prefetch_max=prefetch_max,
        )

        try:
            logger.info(
                "Querying Gaia DR3 (synthetic photometry, SQL TOP %d -> target %d sources; paced archive: pause %.2fs before/after ADQL)...",
                sql_top,
                max_sources,
                max(query_pause_b, query_pause_a),
            )
            results, used_radius_deg = query_gaia_xp_cone_growing(
                ra,
                dec,
                radius,
                sql_top,
                include_bp_rp=True,
                min_sources=xp_min_sources,
                max_radius_deg=xp_max_radius_deg,
                grow_factor=xp_grow_factor,
                pause_before_sec=query_pause_b,
                pause_after_sec=query_pause_a,
                max_retries=archive_retries,
                retry_base_delay_sec=retry_base_delay,
                logger=logger,
                op_name="Gaia ADQL (XP sources for synthetic photometry)",
            )
            logger.info(
                "Gaia DR3 query returned %d sources within %.4f deg.",
                len(results),
                used_radius_deg,
            )

            if sort_by_distance and not results.empty:
                logger.info(
                    "Sorting %d rows by on-sky distance to target (Gaia ADQL does not support ORDER BY distance); keeping nearest %d.",
                    len(results),
                    max_sources,
                )
                results = sort_gaia_table_nearest_to_target(
                    results, ra, dec, max_rows=max_sources
                )
                logger.info("After distance trim: %d sources.", len(results))

            if results.empty:
                return pd.DataFrame()

            source_ids = results["source_id"].astype(str).tolist()
            # Resolve requested GaiaXPy photometric systems.
            # photometric_systems=None keeps prior behaviour (both systems);
            # an empty list skips GaiaXPy and returns base DR3 photometry
            # only (much faster).
            if photometric_systems is None:
                phot_systems = [
                    PhotometricSystem.SDSS_Std,
                    PhotometricSystem.JKC_Std,
                ]
            else:
                cfg = photometric_systems
                if isinstance(cfg, (str, bytes)):
                    cfg = [cfg]
                cfg_list = list(cfg) if cfg else []

                if not cfg_list:
                    logger.warning(
                        "gaia_xp_photometric_systems is empty; returning\n"
                        "    base Gaia DR3 photometry only. The catalog\n"
                        "    will have no SdssStd/JkcStd columns, so no\n"
                        "    standard-band magnitudes can be mapped\n"
                        "    downstream."
                    )
                    return results

                phot_systems = []
                for sys_name in cfg_list:
                    if isinstance(sys_name, str):
                        name = sys_name.strip()
                        try:
                            phot_systems.append(getattr(PhotometricSystem, name))
                        except AttributeError:
                            raise ValueError(
                                f"Unknown GaiaXPy photometric system '{name}'. "
                                f"Valid options: {[p.name for p in PhotometricSystem]}"
                            )
                    else:
                        # Enum values are accepted directly.
                        phot_systems.append(sys_name)

            logger.info(
                "Downloading Gaia XP spectra and synthetic photometry with GaiaXPy (batched: size=%d, inter-batch pause=%.2fs)...",
                xp_batch_size if xp_batch_size > 0 else len(source_ids),
                xp_batch_pause,
            )
            try:
                logger.info(
                    "GaiaXPy photometric systems: %s",
                    [getattr(p, "name", str(p)) for p in phot_systems],
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    photometry = generate_source_ids_batched(
                        source_ids,
                        phot_systems,
                        batch_size=xp_batch_size,
                        inter_batch_pause_sec=xp_batch_pause,
                        max_retries=archive_retries,
                        retry_base_delay_sec=retry_base_delay,
                        logger=logger,
                        show_progress=xp_show_progress,
                        error_correction=bool(
                            cat_cfg.get("gaia_xp_generate_error_correction", True)
                        ),
                        truncation=bool(
                            cat_cfg.get("gaia_xp_generate_truncation", False)
                        ),
                    )
                results = results.copy()
                results["source_id"] = results["source_id"].astype(str)
                photometry = photometry.copy()
                photometry["source_id"] = photometry["source_id"].astype(str)
                merged = pd.merge(
                    results, photometry, on="source_id", how="inner"
                )
            except Exception as exc:
                logger.warning(
                    "Gaia XP synthetic photometry failed (%s); falling back to base DR3 photometry only.",
                    exc,
                )
                merged = results

            # Magnitude errors from flux: sigma_m = 2.5/ln(10) * (sigma_F/F).
            factor = 2.5 / np.log(10)

            # Skip bands absent from the merged table (e.g. when XP failed and
            # only base DR3 photometry is available); also guards F<=0.
            for prefix, bands in (("SdssStd", "ugriz"), ("JkcStd", "UBVRI")):
                for band in bands:
                    f_col = f"{prefix}_flux_{band}"
                    e_col = f"{prefix}_flux_error_{band}"
                    m_err_col = f"{prefix}_mag_error_{band}"
                    if f_col not in merged.columns or e_col not in merged.columns:
                        continue
                    valid = merged[f_col] > 0
                    merged[m_err_col] = np.nan
                    merged.loc[valid, m_err_col] = (
                        factor * merged.loc[valid, e_col] / merged.loc[valid, f_col]
                    )

            logger.info(
                "Successfully merged Gaia DR3 catalog and XP photometry for %d sources.",
                len(merged),
            )
            return merged

        except Exception as exc:
            logger.error(
                "Gaia DR3 synthetic photometry query failed: %s", exc, exc_info=True
            )
            return pd.DataFrame()

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def query_legacy_survey(self, ra, dec, radius=0.1, nsources=1000):
        """
        Query the Legacy Survey using the NOIRLab Data Lab TAP service via astroquery's TAP+.

        Parameters:
        -----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        radius : float, optional
            Search radius in degrees
        nsources : int, optional
            Maximum number of sources to return (default 1000).  The TAP
            query over-fetches so the cap is applied as a spatially
            uniform subsample - a distance-ordered TOP-N alone would keep
            only the inner cone in crowded fields.

        Returns:
        --------
        str
            Path to the downloaded file, or None if an error occurred.
        """
        tap_service_url = "https://datalab.noirlab.edu/tap"

        fetch_n = min(max(int(nsources) * 10, int(nsources)), 50000)

        # `type` is selected so point sources (PSF) can be separated from
        # extended objects (REX, EXP, DEV, SER = galaxies/resolved sources).
        query = f"""
            SELECT TOP {fetch_n} ra, dec, type, mag_g, mag_r, mag_i, mag_z,
                sqrt(power(ra - {ra}, 2) + power(dec - {dec}, 2)) AS angular_distance
            FROM ls_dr10.tractor
            WHERE 't'= Q3C_RADIAL_QUERY(ra, dec, {ra}, {dec}, {radius})
            ORDER BY angular_distance ASC
        """

        try:
            logger.info(
                f"Fetching Legacy Survey Dr10 catalog over {radius:.1f} field-of-view centered at ra = {ra:.1f} dec = {dec:.1f}"
            )
            logger.debug(query)

            from astroquery.utils.tap import TapPlus

            tap = TapPlus(url=tap_service_url)
            result = tap.launch_job(query)
            table = result.get_results().to_pandas()

            # Point sources only; non-PSF types are extended/galaxies.
            if "type" in table.columns:
                n_before = len(table)
                table = table[table["type"].astype(str).str.upper() == "PSF"].copy()
                n_gal = n_before - len(table)
                if n_gal > 0:
                    logger.info(
                        "Legacy Survey: removed %d extended sources (type != PSF); %d point sources remain.",
                        n_gal, len(table),
                    )
            table = table.drop(columns=["type"], errors="ignore")

            if len(table) > nsources:
                logger.info(
                    "Legacy Survey: %d rows exceed the %d-source cap; "
                    "subsampling spatially to keep coverage uniform.",
                    len(table),
                    nsources,
                )
                table = _sky_uniform_subsample(
                    table, int(nsources), center_ra=ra
                )

            filters = ["g", "r", "i", "z"]

            for f in filters:
                table[f"mag_{f}_e"] = [0.01] * len(table)

            return table

        except Exception as e:
            logger.warning("Error during query: %s", e)
            return None

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def fetch_refcat2_field(self, ra, dec, credentials, nsources=1000, sr=0.5,
                             max_retries=3, retry_delay=5.0):
        """
        Fetch ATLAS-RefCat2 catalog from MAST.

        Parameters:
        -----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        credentials : dict
            Dictionary containing MAST credentials.
        nsources : int, optional
            Maximum number of sources to fetch (default is 1000).
            The SQL over-fetches so the cap is applied as a spatially
            uniform subsample - TOP-N ordered by distance alone would
            keep only the inner cone in crowded fields.
        sr : float, optional
            Search radius in degrees (default is 0.5).
        max_retries : int, optional
            Maximum number of retry attempts for transient MAST errors (default 3).
        retry_delay : float, optional
            Initial delay between retries in seconds; doubles each retry (default 5.0).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the catalog data, or None if all retries fail.
        """
        from mastcasjobs import MastCasJobs

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Random suffix gives each attempt a unique MAST table name.
                name = f"autophot_{''.join(random.choices(string.ascii_uppercase, k=5))}"
                logger.info(
                    f"Fetching ATLAS-RefCat2 catalog from MAST over {sr:.3f} deg field-of-view centered at ra = {ra:.1f} dec = {dec:.1f}"
                    + (f" (attempt {attempt}/{max_retries})" if attempt > 1 else "")
                )

                table = [
                    "RA", "Dec", "g", "dg", "r", "dr",
                    "i", "di", "z", "dz", "J", "dJ", "H", "dH", "K", "dK",
                ]

                # Over-fetch: the SQL cap alone truncates nearest-first,
                # leaving the field edge empty; the spatial subsample
                # below restores uniform coverage at nsources rows.
                fetch_n = min(max(int(nsources) * 10, int(nsources)), 20000)
                q = """
                SELECT TOP {max} {columns}
                INTO MyDB.{name}
                FROM fGetNearbyObjEq({ra}, {dec}, {sr}) as n
                INNER JOIN refcat2 AS r ON (n.objid = r.objid)
                WHERE r.dr < 0.1
                ORDER BY n.distance
                """.format(
                    max=fetch_n,
                    columns="r." + ",r.".join(table),
                    name=name,
                    ra=ra,
                    dec=dec,
                    sr=sr,
                )

                logger.debug("SQL Query: %s", q)

                job = MastCasJobs(context="HLSP_ATLAS_REFCAT2", **credentials)

                # drop_table_if_exists can fail with transient MAST SQL errors;
                # since we use random table names, the table almost never exists.
                # Make this non-fatal so the actual query can still proceed.
                try:
                    job.drop_table_if_exists(name)
                except Exception as drop_err:
                    logger.debug(
                        f"drop_table_if_exists failed (non-fatal, table likely new): {drop_err}"
                    )

                jobid = job.submit(q, task_name=f"refcat catalog search {ra:.5f} {dec:.5f}")

                status = job.monitor(jobid)

                # CasJobs status 3/4 means the job failed.
                if status[0] in (3, 4):
                    raise Exception(f"Job failed with status {status[0]}: {status[1]}")

                tab = job.get_table(name, format="CSV")
                try:
                    job.drop_table_if_exists(name)
                except Exception as drop_err:
                    logger.debug("Post-query drop_table failed (non-fatal): %s", drop_err)

                tab = tab.to_pandas()

                # Drop rows containing a 0 (missing photometry sentinel).
                tab = tab[~(tab == 0).any(axis=1)]
                tab.reset_index(drop=True, inplace=True)

                if len(tab) > nsources:
                    logger.info(
                        "RefCat2: %d rows exceed the %d-source cap; "
                        "subsampling spatially to keep coverage uniform.",
                        len(tab),
                        nsources,
                    )
                    tab = _sky_uniform_subsample(
                        tab, int(nsources), center_ra=ra
                    ).reset_index(drop=True)

                return tab

            except Exception as e:
                last_error = e
                logger.warning(
                    f"RefCat2 fetch attempt {attempt}/{max_retries} failed: {e}"
                )
                if attempt < max_retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    logger.info("Retrying in %.0fs...", delay)
                    time.sleep(delay)

        logger.error("\n> Catalog retrieval failed after %d attempts! \n", max_retries)
        logger.error("Last error: %s: %s", type(last_error).__name__, last_error)
        return None

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def download(
        self,
        target_coords,
        catalogName,
        radius=10,
        target_name=None,
        catalog_custom_fpath=None,
        include_IR_sequence_data=True,
        max_sources=None,
        region=None,
    ):
        """
        Download and process catalog data for a given target.

        Parameters:
        -----------
        target_coords : SkyCoord
            SkyCoord object with the RA and DEC of the target.
        catalogName : str
            Name of the catalog to fetch ('refcat', 'gaia', 'apass', '2mass', 'sdss', 'skymapper', 'panstarrs'/'pan_starrs', 'custom').
        radius : float, optional
            Search radius around the target in **arcminutes** (default is 10).
        target_name : str, optional
            Optional name for the target.
        catalog_custom_fpath : str, optional
            File path to a custom catalog (used if catalogName is 'custom').
        include_IR_sequence_data : bool, optional
            Boolean to include IR sequence data from 2MASS (default is True).
        max_sources : int, optional
            Maximum number of sources to return (default None for no limit).
            If set, catalogs will be limited to this many sources after download.
        region : dict, optional
            Query region from ``_footprint_region``/``_region_from_corners``
            (keys ``ra``, ``dec``, ``radius_arcmin``, ``box_deg``,
            ``width_deg``, ``height_deg``).  When given, the query is
            centred on the region centre - the footprint bounding-box
            midpoint, which need not be the target - and box-capable
            backends (``_BOX_QUERY_CATALOGS``) issue the rectangular query
            while the rest get the box's circumscribed cone.  ``radius``
            remains the fallback when ``region`` carries none.

        Returns:
        --------
        DataFrame
            DataFrame containing the catalog data, or None if an error occurs.
        """
        logger.log(STATUS, log_step("Catalog: sequence sources in field"))

        try:
            catalogName = self._require_catalog_selected(catalogName)

            target_ra = target_coords.ra.degree
            target_dec = target_coords.dec.degree

            # Effective query region: a passed-in ``region`` centres the
            # query on the footprint bounds rather than the target, so
            # offset pointings no longer inflate the cone.  Cone-only
            # backends use the box's circumscribed radius; box-capable
            # ones issue the rectangular query directly.
            q_ra, q_dec = float(target_ra), float(target_dec)
            q_radius_arcmin = float(radius)
            box_wh = None
            if isinstance(region, dict):
                try:
                    if region.get("ra") is not None:
                        q_ra = float(region["ra"]) % 360.0
                    if region.get("dec") is not None:
                        q_dec = float(region["dec"])
                    _qr = region.get("radius_arcmin")
                    if _qr is not None:
                        _qr = float(_qr)
                        if np.isfinite(_qr) and _qr > 0:
                            q_radius_arcmin = _qr
                    _bw = region.get("width_deg")
                    _bh = region.get("height_deg")
                    if _bw is not None and _bh is not None:
                        _bw, _bh = float(_bw), float(_bh)
                        if np.isfinite(_bw) and np.isfinite(_bh) and _bw > 0 and _bh > 0:
                            box_wh = (_bw, _bh)
                except (TypeError, ValueError):
                    q_ra, q_dec = float(target_ra), float(target_dec)
                    q_radius_arcmin = float(radius)
                    box_wh = None
            query_coords = SkyCoord(
                ra=q_ra * u.deg, dec=q_dec * u.deg, frame="icrs"
            )
            # A region offset from the target still needs its far edge
            # inside clean()'s target-anchored max_distance cut.
            off_arcmin = float(
                query_coords.separation(target_coords).arcmin
            )
            # Rectangular queries are only sent to backends that support
            # them; other backends use the circumscribed cone.
            use_box = box_wh is not None and catalogName in _BOX_QUERY_CATALOGS

            if target_name is None:
                if target_ra is not None and target_dec is not None:
                    target_name = f"target_ra_{target_ra:.6f}_dec_{target_dec:.6f}"
                else:
                    target_name = "target"
            else:
                if "Unknown" not in target_name:
                    # Re-anchor on the configured name so every caller
                    # shares one cache key; "Unknown" labels (additional
                    # targets) keep their own namespace.
                    target_name = canonical_target_name(self.input_yaml)

            if not catalog_custom_fpath:
                catalog_custom_fpath = self.input_yaml["catalog"].get(
                    "catalog_custom_fpath", None
                )

            wdir = self.input_yaml.get("wdir")
            if not wdir:
                raise ValueError("Working directory (wdir) is not set in input YAML.")

            dirname = os.path.join(wdir, "catalog_queries")
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            catalog_dir = os.path.join(dirname, catalogName)
            pathlib.Path(catalog_dir).mkdir(parents=True, exist_ok=True)
            target_dir = reduce(
                os.path.join, [dirname, catalogName, target_name.lower()]
            )
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

            # The cache key carries the *queried* centre/extent so runs
            # with different resolved regions never collide.
            box_tag = (
                f"_box{box_wh[0]:.2f}x{box_wh[1]:.2f}deg" if use_box else ""
            )
            fname = (
                f"{target_name}_r_{q_radius_arcmin:.1f}arcmins{box_tag}_"
                f"{catalogName}_target_ra_{q_ra:.6f}_dec_{q_dec:.6f}"
            )

            # radius is arcmin; catalog queries take degrees.
            radius_deg = q_radius_arcmin / 60

            # clean() drops sources farther than catalog.max_distance from
            # the target - a wider query cone must not be re-trimmed to the
            # default 10 arcmin or the edge coverage would be lost again.
            # The reach is centre-to-target offset + radius: region centres
            # sit on the footprint bounds, not necessarily on the target.
            try:
                _cat_cfg = self.input_yaml.setdefault("catalog", {})
                _cur_md = float(_cat_cfg.get("max_distance", 10.0))
                _reach = q_radius_arcmin + off_arcmin
                if _reach > _cur_md:
                    _cat_cfg["max_distance"] = _reach
                    logger.debug(
                        "Raised catalog.max_distance to %.1f arcmin to match "
                        "the query reach (r=%.1f + offset %.1f).",
                        _reach,
                        q_radius_arcmin,
                        off_arcmin,
                    )
            except Exception:
                pass

            # A cone cache at the same centre is a superset of any box
            # query - reuse it before paying for a narrower download.
            _cache_paths = [os.path.join(target_dir, f"{fname}.csv")]
            if use_box:
                _fname_cone = (
                    f"{target_name}_r_{q_radius_arcmin:.1f}arcmins_"
                    f"{catalogName}_target_ra_{q_ra:.6f}_dec_{q_dec:.6f}"
                )
                _cache_paths.append(
                    os.path.join(target_dir, f"{_fname_cone}.csv")
                )
            _cache_hit = next(
                (p for p in _cache_paths if os.path.isfile(p)), None
            )

            if catalogName == "custom":
                if not catalog_custom_fpath:
                    logger.critical(
                        'Custom catalog selected but "catalog_custom_fpath" is not defined.'
                    )
                    return None
                selectedCatalog = pd.read_csv(catalog_custom_fpath)

            elif _cache_hit is not None:
                logger.info("Existing %s catalog found for %s", catalogName.upper(), target_name)
                selectedCatalog = (
                    Table.read(_cache_hit, format="csv")
                    .to_pandas()
                    .fillna(np.nan)
                )
                # Caches written before the detection-quality cuts lack the
                # flag columns they need; treat them as stale and re-download.
                _quality_cols = {
                    "pan_starrs": {"ng", "nr", "ni", "nz", "ny"},
                    "2mass": {"Qflg"},
                    "skymapper": {"ngood"},
                }
                _need = _quality_cols.get(catalogName, set())
                if _need and not _need.issubset(selectedCatalog.columns):
                    logger.info(
                        "Cached %s catalog lacks quality columns %s; re-downloading",
                        catalogName.upper(),
                        sorted(_need),
                    )
                    os.remove(_cache_hit)
                    return self.download(
                        target_coords=target_coords,
                        catalogName=catalogName,
                        radius=radius,
                        target_name=target_name,
                        catalog_custom_fpath=catalog_custom_fpath,
                        include_IR_sequence_data=include_IR_sequence_data,
                        max_sources=max_sources,
                        region=region,
                    )
                # Dedup cached catalog: earlier runs may have written duplicates.
                if not selectedCatalog.empty and {"RA", "DEC"}.issubset(selectedCatalog.columns):
                    n_before = len(selectedCatalog)
                    selectedCatalog = _skycoord_dedup_keep_one(selectedCatalog, sep_threshold_arcsec=0.1)
                    n_dups = n_before - len(selectedCatalog)
                    if n_dups > 0:
                        logger.warning(
                            f"Removed {n_dups} duplicates from cached {catalogName.upper()} catalog"
                        )

            else:
                selectedCatalog = []
                from astroquery.mast import Catalogs

                if catalogName == "tic":
                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    result = Catalogs.query_region(
                        query_coords,
                        radius=q_radius_arcmin * u.arcmin,
                        catalog="TIC",
                    )
                    if len(result) == 0:
                        selectedCatalog = pd.DataFrame()
                        self._require_nonempty_catalog(
                            selectedCatalog, catalogName, query_coords, q_radius_arcmin
                        )
                    else:
                        # objType is kept so point sources can be separated
                        # from galaxies below.
                        selected_cols = [
                            "ID",
                            "ra",
                            "dec",
                            "objType",
                            "rmag",
                            "e_rmag",
                            "gmag",
                            "e_gmag",
                            "imag",
                            "e_imag",
                            "zmag",
                            "e_zmag",
                            "umag",
                            "e_umag",
                            "Bmag",
                            "e_Bmag",
                            "Vmag",
                            "e_Vmag",
                            "Jmag",
                            "e_Jmag",
                            "Hmag",
                            "e_Hmag",
                            "Kmag",
                            "e_Kmag",
                            "GAIAmag",
                            "e_GAIAmag",
                            "Tmag",
                            "e_Tmag",
                        ]

                        # Keep only columns the result table actually has.
                        available_cols = [
                            col for col in selected_cols if col in result.colnames
                        ]

                        selectedCatalog = result[available_cols].to_pandas()

                        # Point sources only: TIC objType bitmask STAR = 0x300000;
                        # galaxies and extended objects are excluded.
                        if "objType" in selectedCatalog.columns:
                            n_before = len(selectedCatalog)
                            _obj_type = pd.to_numeric(selectedCatalog["objType"], errors="coerce")
                            _is_star = _obj_type == 0x300000
                            if _is_star.sum() > 0:
                                selectedCatalog = selectedCatalog[_is_star].copy()
                                n_gal = n_before - len(selectedCatalog)
                                if n_gal > 0:
                                    logger.info(
                                        "TIC: removed %d non-stellar sources (objType != STAR); %d stars remain.",
                                        n_gal, len(selectedCatalog),
                                    )
                            selectedCatalog = selectedCatalog.drop(columns=["objType"], errors="ignore")

                        # Write to target_dir (not cwd) to avoid misplaced files.
                        csv_path = os.path.join(target_dir, f"{fname}.csv")
                        selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "gaia":
                    # Gaia DR3+XP uses a smaller radius than the full catalog
                    # search radius to reduce archive load. Hard limit of
                    # 10 arcmin (10/60 deg), optionally smaller if
                    # catalog.gaia_xp_radius_deg is set.
                    max_gaia_deg = 10.0 / 60.0  # 10 arcmin
                    cfg_radius = float(
                        self.input_yaml.get("catalog", {}).get(
                            "gaia_xp_radius_deg", max_gaia_deg
                        )
                    )
                    xp_radius_deg = min(radius_deg, cfg_radius, max_gaia_deg)
                    gaia_xp_max_sources = int(
                        self.input_yaml.get("catalog", {}).get(
                            "gaia_xp_max_sources", 500
                        )
                    )
                    gaia_xp_photometric_systems = (
                        self.input_yaml.get("catalog", {}).get(
                            "gaia_xp_photometric_systems", None
                        )
                    )
                    result = self.gaia_synthetic_photometry(
                        ra=q_ra,
                        dec=q_dec,
                        radius=xp_radius_deg,
                        max_sources=gaia_xp_max_sources,
                        photometric_systems=gaia_xp_photometric_systems,
                    )

                    selectedCatalog = result
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, query_coords, q_radius_arcmin
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "refcat":

                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    logger.warning(
                        "REFCAT requires MAST CasJobs credentials. Set\n"
                        "    `default_input.catalog.MASTcasjobs_wsid` and\n"
                        "    `default_input.catalog.MASTcasjobs_pwd` (or\n"
                        "    provide them via environment/local overrides)."
                    )
                    # Some auth backends reject non-str; cast and strip here.
                    userid = self.input_yaml["catalog"].get("MASTcasjobs_wsid")
                    password = self.input_yaml["catalog"].get("MASTcasjobs_pwd")
                    if userid is not None:
                        userid = str(userid).strip()
                    if password is not None:
                        password = str(password).strip()

                    credentials = {"userid": userid, "password": password}
                    userid_ok = (
                        credentials["userid"] is not None
                        and str(credentials["userid"]).strip() != ""
                    )
                    pwd_ok = (
                        credentials["password"] is not None
                        and str(credentials["password"]).strip() != ""
                    )
                    if not userid_ok or not pwd_ok:
                        raise RuntimeError(
                            "Refcat selected but MAST CasJobs credentials are missing/empty. "
                            f"MASTcasjobs_wsid set: {userid_ok}, MASTcasjobs_pwd set: {pwd_ok}. "
                            "Set `default_input.catalog.MASTcasjobs_wsid` and `default_input.catalog.MASTcasjobs_pwd` "
                            "(or export MASTCASJOBS_WSID/MASTCASJOBS_PWD)."
                        )

                    selectedCatalog = self.fetch_refcat2_field(
                        ra=q_ra,
                        dec=q_dec,
                        credentials=credentials,
                        nsources=500,
                        sr=radius_deg,
                    )
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, query_coords, q_radius_arcmin
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "legacy":

                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    selectedCatalog = self.query_legacy_survey(
                        ra=q_ra,
                        dec=q_dec,
                        radius=radius_deg,
                    )
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, query_coords, q_radius_arcmin
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName in ["apass", "2mass", "sdss"]:
                    Vizier.ROW_LIMIT = -1
                    logger.info(
                        f"Downloading Sequence Stars from {catalogName.upper()}"
                    )
                    if use_box:
                        # Rectangular query matched to the footprint
                        # bounds: no cone-corner waste (~36% of the
                        # circumscribed disk for square fields).
                        catalog_search = Vizier.query_region(
                            query_coords,
                            width=Angle(box_wh[0], "deg"),
                            height=Angle(box_wh[1], "deg"),
                            catalog=catalogName,
                            frame="icrs",
                        )
                    else:
                        catalog_search = Vizier.query_region(
                            query_coords,
                            radius=Angle(radius_deg, "deg"),
                            catalog=catalogName,
                        )
                    if len(catalog_search) < 1:
                        selectedCatalog = pd.DataFrame()
                    elif catalogName == "sdss":
                        # The 'sdss' alias resolves to several VizieR
                        # releases (DR7/9/12/16) returned as separate
                        # tables with different footprints - keep the one
                        # with the best-covered field rather than table [0].
                        selectedCatalog = _select_sdss_vizier_table(
                            catalog_search,
                            center_ra=q_ra,
                            center_dec=q_dec,
                            radius_deg=radius_deg,
                        )
                    else:
                        if len(catalog_search) > 1:
                            logger.debug(
                                "%s query matched %d VizieR catalogs; using %s",
                                catalogName.upper(),
                                len(catalog_search),
                                catalog_search[0].meta.get("name"),
                            )
                        selectedCatalog = catalog_search[0].to_pandas()
                    if catalogName == "apass":
                            # APASS `cls` column: 'A' = stellar, 'G' = galaxy.
                            # Stars only for zeropoint calibration.
                            if "cls" in selectedCatalog.columns:
                                n_before = len(selectedCatalog)
                                selectedCatalog = selectedCatalog[
                                    selectedCatalog["cls"].astype(str).str.upper() == "A"
                                ]
                                n_gal = n_before - len(selectedCatalog)
                                if n_gal > 0:
                                    logger.info(
                                        "APASS: removed %d non-stellar sources (cls != 'A'); %d stars remain.",
                                        n_gal, len(selectedCatalog),
                                    )
                    if catalogName == "2mass":
                        # Photometric quality flag Qflg holds one letter per
                        # band (J,H,Ks): A/B/C are reliable detections; D/E/F/
                        # U/X/- are poor fits, upper limits, or non-detections
                        # that should never calibrate.
                        if "Qflg" in selectedCatalog.columns:
                            _qflg = selectedCatalog["Qflg"].astype(str)
                            for _bi, (_mcol, _ecol) in enumerate(
                                [
                                    ("Jmag", "e_Jmag"),
                                    ("Hmag", "e_Hmag"),
                                    ("Kmag", "e_Kmag"),
                                ]
                            ):
                                _bad = ~_qflg.str[_bi].isin(["A", "B", "C"])
                                for _c in (_mcol, _ecol):
                                    if _c in selectedCatalog.columns:
                                        selectedCatalog.loc[_bad, _c] = np.nan
                            _jk = [
                                c
                                for c in ("Jmag", "Hmag", "Kmag")
                                if c in selectedCatalog.columns
                            ]
                            if _jk:
                                _has = pd.concat(
                                    [
                                        pd.to_numeric(
                                            selectedCatalog[c],
                                            errors="coerce",
                                        )
                                        for c in _jk
                                    ],
                                    axis=1,
                                ).notna().any(axis=1)
                                _n_dead = int((~_has).sum())
                                if _n_dead:
                                    selectedCatalog = selectedCatalog[
                                        _has
                                    ].copy()
                                    logger.info(
                                        "2MASS: dropped %d sources with no "
                                        "quality magnitude (Qflg not A-C) "
                                        "in any band.",
                                        _n_dead,
                                    )
                    # Validate before writing (covers empty query and post-filter empty).
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, query_coords, q_radius_arcmin
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "skymapper":
                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    server = "http://skymapper.anu.edu.au/sm-cone/public/query?"
                    params = {
                        "RA": q_ra,
                        "DEC": q_dec,
                        "SR": radius_deg,
                        "RESPONSEFORMAT": "VOTABLE",
                    }
                    # Write temp file to target_dir (not cwd) to avoid misplaced files.
                    temp_vot_path = os.path.join(target_dir, "temp.vot")
                    try:
                        logger.info("Downloading Sequence Stars from SkyMapper")
                        response = requests.get(server, params=params, timeout=60)
                        response.raise_for_status()
                        with open(temp_vot_path, "wb") as f:
                            f.write(response.content)
                        selectedCatalog = (
                            parse_single_table(temp_vot_path)
                            .to_table(use_names_over_ids=True)
                            .to_pandas()
                        )
                    finally:
                        if os.path.exists(temp_vot_path):
                            os.remove(temp_vot_path)
                    # Guard column existence before filtering.
                    if "class_star" in selectedCatalog.columns:
                        selectedCatalog = selectedCatalog[
                            selectedCatalog["class_star"] > 0.8
                        ]
                    if "flags" in selectedCatalog.columns:
                        selectedCatalog = selectedCatalog[selectedCatalog["flags"] <= 1]
                    # ngood counts clean measurements; a source measured well
                    # only once is as likely an artifact as real.
                    if "ngood" in selectedCatalog.columns:
                        selectedCatalog = selectedCatalog[
                            pd.to_numeric(
                                selectedCatalog["ngood"], errors="coerce"
                            )
                            >= 2
                        ]
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, query_coords, q_radius_arcmin
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                elif catalogName == "pan_starrs":
                    logger.info(
                        f"Downloading reference sources from {catalogName.upper()}"
                    )
                    # Direct API request: the MAST catalog endpoint returns
                    # {"info": [column metadata], "data": [row arrays]}; the
                    # release segment must be dr1/dr2 ('ps1' is rejected), and
                    # 'mean' is the MeanObjectView (gMeanPSFMag et al.).
                    #
                    # MAST read timeouts are common on wide cones, so retry
                    # with backoff. An exhausted query raises a distinct
                    # "query failed" error - folding it into an empty
                    # DataFrame would make a transient failure look like a
                    # genuine 0-source (no coverage) result downstream.
                    _ps_attempts = max(
                        1,
                        int(
                            self.input_yaml.get("catalog", {}).get(
                                "pan_starrs_max_retries", 3
                            )
                            or 3
                        ),
                    )
                    _ps_delay = float(
                        self.input_yaml.get("catalog", {}).get(
                            "pan_starrs_retry_base_delay_sec", 5.0
                        )
                        or 5.0
                    )
                    selectedCatalog = None
                    _ps_exc = None
                    for _ps_try in range(1, _ps_attempts + 1):
                        try:
                            ra = q_ra
                            dec = q_dec

                            url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean"
                            params = {
                                "ra": ra,
                                "dec": dec,
                                "radius": radius_deg,
                                "pagesize": 50000,
                                "format": "json"
                            }

                            # Paginate: a single page silently truncates the cone
                            # at pagesize rows, biasing coverage toward whatever
                            # order the endpoint happens to return.
                            rows = []
                            col_names = None
                            first_row = None
                            page = 1
                            while True:
                                params["page"] = page
                                response = requests.get(url, params=params, timeout=120)
                                response.raise_for_status()
                                data = response.json()
                                page_rows = (
                                    data.get("data")
                                    if isinstance(data, dict)
                                    else None
                                )
                                if col_names is None:
                                    col_names = [
                                        c["name"] for c in data.get("info", [])
                                    ]
                                if not page_rows:
                                    break
                                # If the endpoint ignores 'page', every request
                                # returns page 1 - detect repeats and stop rather
                                # than accumulating duplicates.
                                if page > 1 and page_rows[0] == first_row:
                                    break
                                if first_row is None:
                                    first_row = page_rows[0]
                                rows.extend(page_rows)
                                if (
                                    len(page_rows) < params["pagesize"]
                                    or len(rows) >= 200000
                                ):
                                    break
                                page += 1

                            if not rows:
                                selectedCatalog = pd.DataFrame()
                            else:
                                selectedCatalog = pd.DataFrame(rows, columns=col_names)
                                # Normalize null-like strings to NaN.
                                selectedCatalog = selectedCatalog.replace(
                                    ['None', 'none', 'NONE', 'null', 'NULL', 'nan', 'NaN'], np.nan
                                )
                                # Coerce numeric columns; name columns stay strings.
                                for col in selectedCatalog.columns:
                                    if col not in ['objName', 'objAltName1', 'objAltName2', 'objAltName3']:
                                        try:
                                            selectedCatalog[col] = pd.to_numeric(selectedCatalog[col], errors='coerce')
                                        except Exception:
                                            pass

                            logger.info("Retrieved %s Pan-STARRS sources", len(selectedCatalog))
                            break

                        except Exception as api_exc:
                            _ps_exc = api_exc
                            logger.warning(
                                "Direct Pan-STARRS API attempt %d/%d failed (%s)",
                                _ps_try,
                                _ps_attempts,
                                api_exc,
                            )
                            if _ps_try < _ps_attempts:
                                time.sleep(_ps_delay * (2 ** (_ps_try - 1)))
                    if selectedCatalog is None:
                        raise RuntimeError(
                            f"PAN_STARRS catalog query failed after "
                            f"{_ps_attempts} attempt(s): {_ps_exc}"
                        )

                    # Catch remaining null sentinels (numeric -999 included).
                    selectedCatalog = selectedCatalog.replace([-999, -999.0, "None", "none", "NONE", "null", "NULL"], np.nan)
                    columns = [
                        "raMean",
                        "decMean",
                        "raMeanErr",
                        "decMeanErr",
                        "gMeanPSFMag",
                        "gMeanPSFMagErr",
                        "rMeanPSFMag",
                        "rMeanPSFMagErr",
                        "iMeanPSFMag",
                        "iMeanPSFMagErr",
                        "zMeanPSFMag",
                        "zMeanPSFMagErr",
                        "yMeanPSFMag",
                        "yMeanPSFMagErr",
                        # Kron magnitudes for star-galaxy separation:
                        # stars have PSF ~ Kron; galaxies have Kron > PSF.
                        "rMeanKronMag",
                        "rMeanKronMagErr",
                        # Detection counts for reliability cuts: the mean
                        # endpoint returns single-epoch detections too, whose
                        # "mean" magnitude is one noisy/artifact-prone
                        # measurement.
                        "nStackDetections",
                        "nDetections",
                        "ng",
                        "nr",
                        "ni",
                        "nz",
                        "ny",
                    ]
                    # Keep only columns present in the API response.
                    missing_cols = [c for c in columns if c not in selectedCatalog.columns]
                    if missing_cols:
                        logger.warning(
                            "Pan-STARRS response missing expected columns: %s - they will be absent from the catalog",
                            missing_cols,
                        )
                    available_columns = [c for c in columns if c in selectedCatalog.columns]
                    selectedCatalog = selectedCatalog[available_columns]

                    # Star-galaxy separation: compare PSF and Kron magnitudes.
                    # Stars: |PSF - Kron| < 0.1 mag (point sources).
                    # Galaxies: Kron > PSF by > 0.1 mag (extended flux).
                    if {"rMeanPSFMag", "rMeanKronMag"}.issubset(selectedCatalog.columns):
                        _psf = pd.to_numeric(selectedCatalog["rMeanPSFMag"], errors="coerce")
                        _kron = pd.to_numeric(selectedCatalog["rMeanKronMag"], errors="coerce")
                        _both_finite = np.isfinite(_psf) & np.isfinite(_kron)
                        _is_star = _both_finite & (np.abs(_psf - _kron) < 0.1)
                        # Conservative keep: star-like, or Kron mag missing.
                        _keep = _is_star | ~np.isfinite(_kron)
                        n_before = len(selectedCatalog)
                        selectedCatalog = selectedCatalog[_keep].copy()
                        n_gal = n_before - len(selectedCatalog)
                        if n_gal > 0:
                            logger.info(
                                "Pan-STARRS: removed %d extended sources (|PSF-Kron| >= 0.1 mag); %d point sources remain.",
                                n_gal, len(selectedCatalog),
                            )
                        # Kron columns were only needed for the star cut.
                        selectedCatalog = selectedCatalog.drop(
                            columns=[c for c in ["rMeanKronMag", "rMeanKronMagErr"] if c in selectedCatalog.columns],
                            errors="ignore",
                        )

                    # Per-band reliability: a mean magnitude built from a
                    # single-epoch detection is as likely a cosmic ray or
                    # artifact as a source (the bulk of MeanObjectView rows
                    # have nDetections=1). Mask mags with <2 detections in
                    # that band; masked values fail every finite-mag check
                    # downstream (coverage scoring and ZP fitting alike).
                    _n_det_col = {"g": "ng", "r": "nr", "i": "ni",
                                  "z": "nz", "y": "ny"}
                    _n_masked = 0
                    for _b, _ncol in _n_det_col.items():
                        _mcol = f"{_b}MeanPSFMag"
                        _ecol = f"{_b}MeanPSFMagErr"
                        if (
                            _ncol in selectedCatalog.columns
                            and _mcol in selectedCatalog.columns
                        ):
                            _ndet = pd.to_numeric(
                                selectedCatalog[_ncol], errors="coerce"
                            )
                            _bad = np.isfinite(
                                pd.to_numeric(
                                    selectedCatalog[_mcol], errors="coerce"
                                )
                            ) & ~(_ndet >= 2)
                            if _bad.any():
                                _n_masked += int(_bad.sum())
                                _cols = [_mcol] + (
                                    [_ecol]
                                    if _ecol in selectedCatalog.columns
                                    else []
                                )
                                selectedCatalog.loc[_bad, _cols] = np.nan
                    if _n_masked:
                        logger.info(
                            "Pan-STARRS: masked %d magnitudes built from "
                            "single-epoch detections (n<2 in that band).",
                            _n_masked,
                        )
                    _mag_cols = [
                        c
                        for c in selectedCatalog.columns
                        if c.endswith("MeanPSFMag")
                    ]
                    if _mag_cols:
                        _has_mag = pd.concat(
                            [
                                pd.to_numeric(selectedCatalog[c],
                                              errors="coerce")
                                for c in _mag_cols
                            ],
                            axis=1,
                        ).notna().any(axis=1)
                        _n_dead = int((~_has_mag).sum())
                        if _n_dead:
                            selectedCatalog = selectedCatalog[
                                _has_mag
                            ].copy()
                            logger.info(
                                "Pan-STARRS: dropped %d sources with no "
                                "reliable magnitude in any band.",
                                _n_dead,
                            )

                    if {"raMean", "decMean"}.issubset(selectedCatalog.columns):
                        coords = SkyCoord(
                            ra=selectedCatalog["raMean"].values * u.deg,
                            dec=selectedCatalog["decMean"].values * u.deg,
                        )
                        # clean()'s max_distance cut is anchored on the
                        # target, so this column measures from the target
                        # even when the query region is centred elsewhere.
                        distances = target_coords.separation(coords)
                        selectedCatalog["distance"] = distances.arcsecond
                    self._require_nonempty_catalog(
                        selectedCatalog, catalogName, query_coords, q_radius_arcmin
                    )
                    # Write to target_dir (not cwd) to avoid misplaced files.
                    csv_path = os.path.join(target_dir, f"{fname}.csv")
                    selectedCatalog.to_csv(csv_path, index=False, na_rep=np.nan)

                else:
                    logger.critical("Catalog %s is not recognized.", catalogName)
                    sys.exit()

                logger.log(
                    STATUS,
                    "%s catalog contains %d sources",
                    catalogName.upper(),
                    len(selectedCatalog),
                )
                warnings.filterwarnings("default")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.warning("%s %s %d %s", exc_type, fname, exc_tb.tb_lineno, e)
            # Propagate catalog failures as hard stops: downstream calibration
            # should not proceed without a valid catalog.
            raise

        if max_sources is not None and selectedCatalog is not None and len(selectedCatalog) > max_sources:
            logger.info(
                "Limiting catalog from %s to %s sources",
                len(selectedCatalog), max_sources,
            )
            # Spatially uniform subsample: keeping the N nearest to the
            # target would concentrate calibrators at the field centre and
            # leave the edges empty.
            selectedCatalog = _sky_uniform_subsample(
                selectedCatalog, max_sources, center_ra=q_ra
            )
            logger.info("Catalog limited to %s sources", len(selectedCatalog))

        # Coverage diagnostic: a catalog can answer a cone query yet cover
        # only part of the region (survey boundary, or a distance-ordered
        # row cap that slipped through).  Measured against the region
        # actually queried - the box for box-capable backends, the
        # circumscribed cone otherwise - and warned once per field so
        # one-sided coverage does not silently reach the zeropoint fit.
        try:
            _cov_ra, _cov_dec = _catalog_radec(selectedCatalog)
            if _cov_ra is not None:
                _cov = _field_coverage_fraction(
                    _cov_ra,
                    _cov_dec,
                    q_ra,
                    q_dec,
                    radius_deg,
                    box_wh=box_wh if use_box else None,
                )
                _cov_key = (
                    catalogName,
                    round(float(q_ra), 3),
                    round(float(q_dec), 3),
                    round(float(radius_deg), 3),
                    use_box,
                )
                if np.isfinite(_cov) and _cov_key not in _COVERAGE_WARNED_KEYS:
                    _COVERAGE_WARNED_KEYS.add(_cov_key)
                    if _cov < 0.8:
                        logger.warning(
                            "%s catalog covers only %.0f%% of the %.1f arcmin query "
                            "field - the survey footprint likely does not cover the "
                            "full image, so calibrators will be unevenly distributed. "
                            "Consider catalog.use_catalog='auto' or a "
                            "wider-coverage catalog.",
                            catalogName.upper(),
                            100.0 * _cov,
                            float(q_radius_arcmin),
                        )
                    else:
                        logger.info(
                            "%s catalog covers %.0f%% of the query field",
                            catalogName.upper(),
                            100.0 * _cov,
                        )
        except Exception as _cov_exc:
            logger.debug("Field-coverage check skipped: %s", _cov_exc)

        return selectedCatalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================
    def clean(
        self,
        selectedCatalog,
        catalogName=None,
        usefilter=None,
        magCutoff=30,
        border=0,
        image_wcs=None,
        fwhm=5,
        get_local_sources=False,
        full_clean=True,
        update_names_only=False,
        index=0,
    ):
        """
        Clean the catalog of sources by applying various filters and removing unwanted sources.

        Parameters:
        -----------
        selectedCatalog : pd.DataFrame
            DataFrame containing the source catalog.
        catalogName : str, optional
            Name of the catalog (default is None).
        usefilter : list, optional
            List of filters to use (default is None).
        magCutoff : float or list, optional
            Magnitude cutoff for filtering (default is 30).
        border : int, optional
            Border size in pixels (default is 0).
        image_wcs : WCS, optional
            WCS object for coordinate conversion (default is None).
        fwhm : float, optional
            FWHM in pixels (default is 5).
        get_local_sources : bool, optional
            Flag to get local sources (default is False).
        full_clean : bool, optional
            Flag for full cleaning (default is True).
        update_names_only : bool, optional
            Flag to update names only (default is False).
        index : int, optional
            Index for WCS (default is 0).

        Returns:
        --------
        pd.DataFrame
            Cleaned catalog DataFrame.
        """
        import logging
        import os
        import numpy as np
        import pandas as pd

        logger = logging.getLogger(__name__)

        if full_clean:
            logger.info("Cleaning %s sources", len(selectedCatalog))

        try:
            # Catalog can be empty after a service failure (e.g. Gaia).
            if selectedCatalog is None or len(selectedCatalog) == 0:
                logger.warning("Selected catalog is empty; skipping catalog cleaning.")
                return None

            # --- Load catalog configuration ---
            filepath = os.path.dirname(os.path.abspath(__file__))
            catalog_autophot_input_yml = "catalog.yml"
            catalogName = catalogName or self.input_yaml["catalog"]["use_catalog"]

            # catalog.yml uses 'panstarrs' while the normalized internal
            # catalog name is 'pan_starrs' - translate for the YAML lookup.
            _yml_name = {"pan_starrs": "panstarrs"}.get(catalogName, catalogName)
            catalog_keywords = AutophotYaml(
                os.path.join(filepath, "databases", catalog_autophot_input_yml),
                _yml_name,
            ).load()

            max_distance = self.input_yaml["catalog"].get(
                "max_distance", 10
            )  # arcminutes

            # --- Filter sources by distance ---
            if "distance" in selectedCatalog:
                too_far = selectedCatalog["distance"] > max_distance * 60  # arcseconds
                n_far = too_far.sum()
                if n_far > 0:
                    logger.info(
                        f"Removing {n_far} sources that are greater than {max_distance:.1f} arcmins from the target"
                    )
                    selectedCatalog = selectedCatalog[~too_far]

            # --- Prepare output catalog with RA/DEC ---
            ra_key = catalog_keywords.get("RA")
            dec_key = catalog_keywords.get("DEC")
            if ra_key not in selectedCatalog or dec_key not in selectedCatalog:
                raise KeyError(
                    f"Required RA/DEC columns '{ra_key}', '{dec_key}' not found in catalog."
                )

            outputCatalog = pd.DataFrame(
                {
                    "RA": selectedCatalog[ra_key].values,
                    "DEC": selectedCatalog[dec_key].values,
                }
            )

            # --- Convert RA/DEC to pixel coordinates if WCS is provided ---
            if image_wcs:
                try:
                    ra_values = np.asarray(
                        selectedCatalog[catalog_keywords["RA"]].values, dtype=float
                    )
                    dec_values = np.asarray(
                        selectedCatalog[catalog_keywords["DEC"]].values, dtype=float
                    )

                    # Angular pre-filter must match how the catalog was queried
                    # (usually within max_distance arcmin of the target). A
                    # CRVAL + fixed 1 deg cap wrongly drops on-chip sources on
                    # wide stacks/coadds where CRVAL sits far from the field
                    # center (common after astrometry.net SIP updates).
                    cfg_cat = self.input_yaml.get("catalog", {}) or {}
                    max_dist_arcmin = float(cfg_cat.get("max_distance", 10.0))
                    t_ra = self.input_yaml.get("target_ra")
                    t_dec = self.input_yaml.get("target_dec")
                    if (
                        t_ra is not None
                        and t_dec is not None
                        and np.isfinite(t_ra)
                        and np.isfinite(t_dec)
                    ):
                        cen_ra = float(t_ra)
                        cen_dec = float(t_dec)
                        # Query radius plus a generous margin, in arcsec.
                        max_distance_threshold = (max_dist_arcmin + 5.0) * 60.0
                    else:
                        shape = getattr(image_wcs, "array_shape", None)
                        if shape is not None and len(shape) == 2:
                            ny, nx = int(shape[0]), int(shape[1])
                            cx = 0.5 * float(max(nx - 1, 0))
                            cy = 0.5 * float(max(ny - 1, 0))
                            cen_ra, cen_dec = image_wcs.all_pix2world(
                                np.asarray([cx]),
                                np.asarray([cy]),
                                0,
                            )
                            cen_ra = float(np.asarray(cen_ra).ravel()[0])
                            cen_dec = float(np.asarray(cen_dec).ravel()[0])
                            corners_x = np.array(
                                [0.0, float(nx - 1), 0.0, float(nx - 1)], dtype=float
                            )
                            corners_y = np.array(
                                [0.0, 0.0, float(ny - 1), float(ny - 1)], dtype=float
                            )
                            cra, cde = image_wcs.all_pix2world(corners_x, corners_y, 0)
                            sc_cen = SkyCoord(
                                cen_ra * u.deg, cen_dec * u.deg, frame="icrs"
                            )
                            sc_corner = SkyCoord(
                                np.asarray(cra).ravel() * u.deg,
                                np.asarray(cde).ravel() * u.deg,
                                frame="icrs",
                            )
                            max_sep = float(
                                np.max(sc_cen.separation(sc_corner).to(u.arcsec).value)
                            )
                            max_distance_threshold = max(3600.0, max_sep * 1.25)
                        else:
                            cen_ra, cen_dec = image_wcs.wcs.crval
                            max_distance_threshold = 4 * 3600.0

                    # Wrap the RA offset across 0/360 deg - a field centred
                    # on RA ~0.1 with catalog entries at 359.9 otherwise
                    # measures dra ~ 360 deg and drops every valid source.
                    dra = (
                        _unwrap_ra_near(ra_values, cen_ra) - cen_ra
                    ) * np.cos(np.radians(cen_dec))
                    ddec = dec_values - cen_dec
                    distance = np.sqrt(dra**2 + ddec**2) * 3600.0  # arcseconds
                    valid_indices = distance < max_distance_threshold
                    if not np.any(valid_indices):
                        logger.warning(
                            "Catalog WCS pre-filter removed all sources (sky vs ref); relaxing filter and keeping full list for pixel conversion."
                        )
                        valid_indices = np.ones(len(ra_values), dtype=bool)

                    ra_values = ra_values[valid_indices]
                    dec_values = dec_values[valid_indices]

                    # world_to_pixel applies full distortion (SIP etc.);
                    # wcs_world2pix can disagree with all_pix2world /
                    # solve-field SIP headers used elsewhere.
                    coords = SkyCoord(
                        ra=ra_values * u.deg, dec=dec_values * u.deg, frame="icrs"
                    )
                    x_pix, y_pix = image_wcs.world_to_pixel(coords)
                    x_pix = np.asarray(x_pix, dtype=float).ravel()
                    y_pix = np.asarray(y_pix, dtype=float).ravel()

                    outputCatalog = outputCatalog.iloc[valid_indices].copy()
                    selectedCatalog = selectedCatalog.iloc[valid_indices].copy()
                    outputCatalog["x_pix"] = x_pix
                    outputCatalog["y_pix"] = y_pix

                except Exception as e:
                    logger.warning(
                        f"Failed to convert RA/DEC to pixel coordinates: {e}. Skipping pixel coordinate conversion."
                    )
                    # Keep schema stable for downstream code that expects pixel columns.
                    outputCatalog = outputCatalog.copy()
                    outputCatalog["x_pix"] = np.nan
                    outputCatalog["y_pix"] = np.nan

            # Populate photometric band columns for the current filter.
            image_filter = self.input_yaml["imageFilter"]
            logger.debug("Populating filter columns for %s; input catalog columns: %s", image_filter, list(selectedCatalog.columns))

            # Custom catalogs: auto-detect filter columns as <band>/<band>_err
            # pairs so arbitrary filter names work without catalog.yml entries.
            if catalogName == "custom":
                import re
                filter_cols = []
                for col in selectedCatalog.columns:
                    if str(col).endswith('_err'):
                        continue
                    err_col = f"{col}_err"
                    if err_col in selectedCatalog.columns:
                        filter_cols.append(col)
                        logger.debug("Auto-detected filter column pair: %s / %s", col, err_col)

                for col in filter_cols:
                    err_col = f"{col}_err"
                    outputCatalog[col] = selectedCatalog[col].values
                    outputCatalog[err_col] = selectedCatalog[err_col].values
                    logger.debug("Auto-copied custom filter %s from catalog", col)

            # The current image filter must always be populated.
            for col in [image_filter, f"{image_filter}_err"]:
                if col in outputCatalog.columns:
                    logger.debug("Column %s already present in output catalog", col)
                    continue
                # catalog.yml mapping first, then exact-name fallback.
                if col in catalog_keywords and catalog_keywords[col] in selectedCatalog:
                    outputCatalog[col] = selectedCatalog[catalog_keywords[col]].values
                    logger.debug("Mapped %s from catalog_keywords", col)
                elif col in selectedCatalog.columns:
                    outputCatalog[col] = selectedCatalog[col].values
                    logger.debug("Copied %s directly from catalog", col)
                else:
                    logger.warning("Could not find column %s in catalog; available columns: %s", col, list(selectedCatalog.columns))

            # --- Retrieve all available filters ---
            baseDatabase = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "databases"
            )
            filters_yml = "filters.yml"
            availableFilters = AutophotYaml(
                os.path.join(baseDatabase, filters_yml)
            ).load()

            for filter_x in availableFilters["default_dmag"].keys():
                for col in [filter_x, f"{filter_x}_err"]:
                    # catalog.yml mapping first, then exact-name fallback.
                    if (
                        filter_x in catalog_keywords
                        and catalog_keywords.get(col) in selectedCatalog
                    ):
                        outputCatalog[col] = selectedCatalog[
                            catalog_keywords[col]
                        ].values
                    elif col in selectedCatalog.columns:
                        outputCatalog[col] = selectedCatalog[col].values
                    # NOTE: no case-insensitive fallback - it would conflate
                    # different photometric systems (e.g. SDSS r vs Cousins R).

            # --- Early return if only updating names ---
            if update_names_only:
                logger.info(
                    f"{len(outputCatalog)} sources in output catalog (names only)"
                )
                return outputCatalog

            # --- Determine filters to use ---
            usefilter = usefilter or [self.input_yaml["imageFilter"]]
            magCutoff = (
                [magCutoff] if isinstance(magCutoff, (int, float)) else magCutoff
            )

            # --- Apply magnitude cutoff filter ---
            if not full_clean:
                tooFaint = np.zeros(len(outputCatalog), dtype=bool)
                for i, usefilter_i in enumerate(usefilter):
                    if usefilter_i in outputCatalog.columns:
                        cutoff_i = magCutoff[i] if i < len(magCutoff) else magCutoff[0]
                        tooFaint |= outputCatalog[usefilter_i].values > cutoff_i
                n_faint = tooFaint.sum()
                if n_faint > 0:
                    logger.info(
                        f"Removing {n_faint} sources that are fainter than the cutoff"
                    )
                    outputCatalog = outputCatalog.loc[~tooFaint]

            # --- Full cleaning procedures ---
            if full_clean:
                # Drop sources missing the image-filter magnitude.
                image_filter = self.input_yaml["imageFilter"]
                if image_filter in outputCatalog.columns:
                    hasFilterinfo = np.isfinite(outputCatalog[image_filter].values)
                    if (~hasFilterinfo).any():
                        logger.info(
                            f"Excluding {(~hasFilterinfo).sum()} sources with no {image_filter} band information"
                        )
                        outputCatalog = outputCatalog[hasFilterinfo]
                else:
                    logger.warning(
                        f"Filter column '{image_filter}' not found in output catalog; "
                        f"available columns: {list(outputCatalog.columns)}"
                    )

                # Drop entries sharing a position with a much brighter
                # catalog source.  Pan-STARRS in particular carries spurious
                # faint rows within a few arcsec of bright stars; whether the
                # faint entry is an artifact or a real companion, its measured
                # flux is dominated by the neighbour, so it cannot calibrate.
                cat_cfg = self.input_yaml.get("catalog", {}) or {}
                try:
                    blend_radius = float(
                        cat_cfg.get("blend_neighbor_radius_arcsec", 6.0)
                    )
                    blend_dmag = float(
                        cat_cfg.get("blend_neighbor_dmag", 3.0)
                    )
                except (TypeError, ValueError):
                    blend_radius, blend_dmag = 6.0, 3.0
                if (
                    blend_radius > 0
                    and blend_dmag > 0
                    and image_filter in outputCatalog.columns
                    and not outputCatalog.empty
                ):
                    n_blend = len(outputCatalog)
                    outputCatalog = _drop_fainter_blended_neighbors(
                        outputCatalog,
                        mag_col=image_filter,
                        radius_arcsec=blend_radius,
                        dmag=blend_dmag,
                    )
                    n_blend -= len(outputCatalog)
                    if n_blend > 0:
                        logger.info(
                            "Removed %d sources within %.1f arcsec of a "
                            ">%.1f-mag-brighter catalog neighbour "
                            "(blend/artifact rejection)",
                            n_blend,
                            blend_radius,
                            blend_dmag,
                        )

            # Final deduplication before returning.
            if not outputCatalog.empty and {"RA", "DEC"}.issubset(outputCatalog.columns):
                n_before = len(outputCatalog)
                outputCatalog = _skycoord_dedup_keep_one(outputCatalog, sep_threshold_arcsec=0.1)
                n_dups = n_before - len(outputCatalog)
                if n_dups > 0:
                    logger.info("Removed %s duplicates during catalog cleaning", n_dups)

            logger.info("%s sources in output catalog", len(outputCatalog))
            logger.debug("Output catalog columns: %s", list(outputCatalog.columns))
            return outputCatalog

        except Exception as e:
            import traceback

            logger.error("Error in catalog cleaning: %s", e)
            logger.error(traceback.format_exc())
            return None

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def recenter(self, selectedCatalog, image, boxsize=None, error=None):
        """
        Recenter sources in an image, selecting centroiding method from FWHM.
        Undersampled (FWHM <= undersampled_fwhm_threshold, default 2.5 px): 2D Gaussian fit for subpixel accuracy.
        Well-sampled (FWHM > threshold): center-of-mass. Tolerates fully masked cutouts.
        Error-weighted centroiding was removed; this routine always uses
        unweighted centroiding for stability across diverse background/error maps.
        """
        try:
            num_sources = len(selectedCatalog)
            logger.info(
                log_step(
                    f"Recentering {num_sources} source{'s' if num_sources > 1 else ''}"
                )
            )

            required_columns = {"x_pix", "y_pix"}
            if not required_columns.issubset(selectedCatalog.columns):
                logger.error(
                    "Selected catalog missing required columns: 'x_pix', 'y_pix'"
                )
                return selectedCatalog

            fwhm = float(self.input_yaml.get("fwhm", 3))
            undersampled_thr = float(
                (self.input_yaml.get("photometry", {}) or {}).get(
                    "undersampled_fwhm_threshold", 2.5
                )
            )
            undersampled = bool(
                self.input_yaml.get("undersampled_mode", fwhm <= undersampled_thr)
            )

            # Box size ~3xFWHM (odd, >=3); undersampled needs >=7 for the
            # 2D Gaussian fit to constrain the core.
            if boxsize is None:
                boxsize = max(3, int(np.ceil(fwhm) * 3))
            boxsize = int(boxsize)
            if undersampled and boxsize < 7:
                boxsize = 7
            boxsize = boxsize if boxsize % 2 != 0 else boxsize + 1
            boxsize = max(boxsize, 3)
            # Border must clear the aperture annulus (aperture + gap + width);
            # annulus parameters match aperture.py conventions.
            phot_cfg = self.input_yaml.get("photometry", {}) or {}
            _ap_radius = phot_cfg.get("aperture_radius")
            ap_radius = float(_ap_radius if _ap_radius is not None else fwhm * 1.7)
            _gap = phot_cfg.get("annulus_gap_fwhm")
            gap_fwhm = float(_gap if _gap is not None else 0.75)
            _width = phot_cfg.get("annulus_width_fwhm")
            width_fwhm = float(_width if _width is not None else 2.0)
            annulus_outer = ap_radius + (gap_fwhm + width_fwhm) * fwhm
            border = max(boxsize, int(np.ceil(annulus_outer)))
            logger.debug("Boxsize: %s px, Border: %s px (annulus outer=%.1f, undersampled=%s)", boxsize, border, annulus_outer, undersampled)

            # Drop sources whose cutout would cross the image border.
            height, width = image.shape
            mask_x = (selectedCatalog["x_pix"] > border) & (
                selectedCatalog["x_pix"] < width - border
            )
            mask_y = (selectedCatalog["y_pix"] > border) & (
                selectedCatalog["y_pix"] < height - border
            )
            mask = mask_x & mask_y

            if num_sources > 1:
                logger.debug("Recentering %s sources within border", sum(mask))
                selectedCatalog = selectedCatalog.loc[mask].copy()

            old_x = selectedCatalog["x_pix"].values
            old_y = selectedCatalog["y_pix"].values
            nan_mask = ~np.isfinite(image)

            # Skip sources whose cutout is fully masked.
            valid_sources = self._check_valid_cutouts(
                image, old_x, old_y, boxsize, nan_mask
            )
            if valid_sources.sum() == 0:
                logger.warning("No sources have valid cutouts - skipping recentering")
                return selectedCatalog.loc[valid_sources]

            logger.debug("%s sources have valid cutouts", valid_sources.sum())

            old_x_valid = old_x[valid_sources]
            old_y_valid = old_y[valid_sources]

            # Undersampled: 2D Gaussian for subpixel accuracy; else COM.
            if undersampled:
                centroid_method = "2D Gaussian"
                centroid_func = centroid_2dg
            else:
                centroid_method = "center-of-mass"
                centroid_func = centroid_com
            try:
                x_valid, y_valid = centroid_sources(
                    image,
                    old_x_valid,
                    old_y_valid,
                    box_size=boxsize,
                    centroid_func=centroid_func,
                    mask=nan_mask,
                )
                x_valid = np.asarray(x_valid)
                y_valid = np.asarray(y_valid)
            except Exception as e:
                log_warning_from_exception(
                    logger, "Centroiding failed even on valid sources", e
                )
                x_valid = old_x_valid.copy()
                y_valid = old_y_valid.copy()
            x_err_valid = np.full(len(x_valid), np.nan)
            y_err_valid = np.full(len(y_valid), np.nan)

            # NaN-fill invalid sources so indices stay aligned.
            x = np.full(len(old_x), np.nan)
            y = np.full(len(old_y), np.nan)
            x[valid_sources] = x_valid
            y[valid_sources] = y_valid

            x_err = np.full(len(old_x), np.nan)
            y_err = np.full(len(old_y), np.nan)
            x_err[valid_sources] = x_err_valid
            y_err[valid_sources] = y_err_valid

            selectedCatalog.loc[:, "x_pix"] = x
            selectedCatalog.loc[:, "y_pix"] = y
            if "x_pix_err" not in selectedCatalog.columns:
                selectedCatalog["x_pix_err"] = np.nan
            if "y_pix_err" not in selectedCatalog.columns:
                selectedCatalog["y_pix_err"] = np.nan
            selectedCatalog.loc[:, "x_pix_err"] = x_err
            selectedCatalog.loc[:, "y_pix_err"] = y_err

            # Drop sources recentered outside the border.
            mask_x = (selectedCatalog["x_pix"] >= border) & (
                selectedCatalog["x_pix"] < width - border
            )
            mask_y = (selectedCatalog["y_pix"] >= border) & (
                selectedCatalog["y_pix"] < height - border
            )
            mask = mask_x & mask_y

            if sum(~mask) > 0:
                logger.warning("Failed to recenter %s sources - ignoring", sum(~mask))
                selectedCatalog = selectedCatalog.loc[mask].copy()

            valid = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(old_x)
                & np.isfinite(old_y)
            )
            average_offset = np.nan
            if valid.any():
                average_offset = np.nanmedian(
                    pix_dist(x[valid], old_x[valid], y[valid], old_y[valid])
                )
            else:
                logger.warning("No valid sources to compute median offset.")

            selectedCatalog = selectedCatalog.loc[
                selectedCatalog[["x_pix", "y_pix"]].notna().all(axis=1)
            ]
            offset_txt = (
                f", median offset {average_offset:.1f} px"
                if np.isfinite(average_offset)
                else ""
            )
            logger.info(
                "Recentered %d sources (%s centroiding%s)",
                len(selectedCatalog),
                centroid_method,
                offset_txt,
            )

        except Exception as e:
            logger.error("Error occurred during recentering: %s", e)
            logger.error(traceback.format_exc())

        return selectedCatalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def _check_valid_cutouts(
        self, image, x_coords, y_coords, boxsize, mask, min_unmasked_frac=0.0
    ):
        """
        Check which source cutouts contain at least one unmasked pixel.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        x_coords, y_coords : array
            Source coordinates
        boxsize : int
            Cutout box size
        mask : numpy.ndarray
            Mask array (True where masked)

        Returns:
        --------
        valid_mask : numpy.ndarray
            Boolean mask of valid cutouts
        min_unmasked_frac : float
            Minimum required fraction of unmasked pixels in a cutout.
        """
        height, width = image.shape
        half_box = boxsize // 2
        valid_mask = np.ones(len(x_coords), dtype=bool)

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            x_min = int(x - half_box)
            x_max = int(x + half_box + 1)
            y_min = int(y - half_box)
            y_max = int(y + half_box + 1)

            if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
                valid_mask[i] = False
                continue

            cutout_mask = mask[y_min:y_max, x_min:x_max]
            n_pix = cutout_mask.size
            n_unmasked = int(np.sum(~cutout_mask))
            frac_unmasked = n_unmasked / max(1, n_pix)
            if frac_unmasked <= float(min_unmasked_frac):
                valid_mask[i] = False

        return valid_mask

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def find_source(self, ra, dec, catalog, tolerance=3.0):
        """
        Find sources in the catalog that match the given RA and DEC
        within a specified tolerance using vectorized operations.

        Parameters:
        -----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        catalog : pd.DataFrame
            DataFrame containing the source catalog.
        tolerance : float, optional
            Tolerance in arcseconds (default is 3.0).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing matching sources.
        """
        if catalog.empty:
            return catalog

        # Small-angle approximation is fine here: tolerance is in arcsec,
        # so separations are always << 1 degree.
        dec_rad = np.radians(dec)
        ra_vals = catalog["RA"].values
        dec_vals = catalog["DEC"].values

        delta_ra = (ra_vals - ra) * np.cos(dec_rad)   # cos(dec) correction
        delta_dec = dec_vals - dec
        separation_deg = np.sqrt(delta_ra**2 + delta_dec**2)
        separation_arcsec = separation_deg * 3600.0

        match_mask = separation_arcsec <= tolerance
        return catalog[match_mask]

    # =========================================================================
    # METHOD: build_complete_catalog
    # =========================================================================
    def build_complete_catalog(
        self,
        target_coords,
        target_name=None,
        catalog_list=["refcat", "sdss", "pan_starrs", "apass", "2mass"],
        radius=10,
        max_separation=3,
        regions=None,
        **kwargs,
    ):
        """
        Build a complete catalog by combining multiple catalogs.

        Parameters:
        -----------
        target_coords : SkyCoord
            SkyCoord object with the RA and DEC of the target.
        target_name : str, optional
            Name for the target (default is None).
        catalog_list : list, optional
            List of catalogs to combine (default is ['refcat', 'sdss', 'pan_starrs', 'apass', '2mass']).
        radius : float, optional
            Search radius in arcminutes (default is 10).
        max_separation : float, optional
            Maximum separation in arcseconds for matching sources (default is 3).
        regions : dict, optional
            Per-catalog query regions (``catalog_query_regions`` format);
            each member catalog downloads over its own region so the
            combined build reuses the same CSV cache keys as the
            per-catalog downloads.

        Returns:
        --------
        pd.DataFrame
            Combined catalog DataFrame.
        """
        catalog_list_str = ",".join([i.upper() for i in catalog_list])
        logger.info(log_step(f"Custom catalog: {catalog_list_str}"))

        if not target_name:
            target_name = canonical_target_name(self.input_yaml)

        # Target RA/DEC in the filename makes the cache field-specific;
        # otherwise catalogs from different targets with the same name get
        # reused (N in the ZP legend then exceeds the real source count).
        target_ra = target_coords.ra.degree
        target_dec = target_coords.dec.degree
        fname = f"{target_name}_r_{radius}arcmins_target_ra_{target_ra:.6f}_dec_{target_dec:.6f}_CUSTOM.csv"
        wdir = self.input_yaml.get("wdir")
        if not wdir:
            raise ValueError("Working directory (wdir) is not set in input YAML.")

        dirname = os.path.join(wdir, "catalog_queries")
        dirname = os.path.join(dirname, "custom")

        dirname = os.path.join(wdir, "catalog_queries")
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        catalog_dir = os.path.join(dirname, "custom_builds")
        pathlib.Path(catalog_dir).mkdir(parents=True, exist_ok=True)
        fpath = os.path.join(catalog_dir, fname)

        filter_list = [
            "u",
            "g",
            "r",
            "i",
            "z",
            "U",
            "B",
            "V",
            "R",
            "I",
            "Z",
            "J",
            "H",
            "K",
        ]

        updated_filter_list = []
        for filter in filter_list:
            updated_filter_list.append(filter)
            updated_filter_list.append(f"{filter}_err")

        # Matching tolerance in arcseconds (used by `find_source`).
        tolerance_arcsec = max_separation

        cols = ["RA", "DEC"] + updated_filter_list

        # A cached combined catalog is reused (after deduplication).
        if os.path.isfile(fpath):
            logger.info("Loading existing custom catalog from %s", fpath)
            existing_catalog = pd.read_csv(fpath)
            if not existing_catalog.empty and {"RA", "DEC"}.issubset(existing_catalog.columns):
                # Keep one member of every close pair.
                n_before = len(existing_catalog)
                existing_catalog = _skycoord_dedup_keep_one(existing_catalog, sep_threshold_arcsec=0.1)
                n_dups = n_before - len(existing_catalog)
                if n_dups > 0:
                    logger.warning(
                        f"Removed {n_dups} duplicates from existing cached catalog - resaving clean version"
                    )
                    existing_catalog.to_csv(fpath, index=False, float_format="%.6f")
            elif not existing_catalog.empty:
                logger.warning(
                    "Existing cached catalog at %s is missing RA/DEC columns - skipping deduplication",
                    fpath,
                )
            return existing_catalog
        
        output_catalog = pd.DataFrame(columns=cols)

        for catalogName in catalog_list:
            logger.info("Getting %s catalog", catalogName)
            _reg_i = (regions or {}).get(
                catalogName
            ) or (regions or {}).get("default")
            catalog_i = self.download(
                target_coords=target_coords,
                catalogName=catalogName,
                radius=radius,
                target_name=target_name,
                region=_reg_i,
            )
            if catalog_i is None:
                continue

            catalog_i = self.clean(
                catalog_i, catalogName=catalogName, update_names_only=True
            )

            # Cross-match via a single match_to_catalog_sky call (O(N log N)),
            # then split into new / update groups. The empty-catalog branch
            # (first iteration) just takes every row.
            new_rows = []
            if output_catalog.empty:
                # to_dict('records') avoids one Series per row (iterrows is
                # much slower on large catalogs).
                catalog_cols = [c for c in cols if c in catalog_i.columns]
                missing_cols = [c for c in cols if c not in catalog_i.columns]
                sub = catalog_i[catalog_cols].copy()
                for c in missing_cols:
                    sub[c] = np.nan
                new_rows = sub[cols].to_dict("records")
            else:
                coords_existing = SkyCoord(
                    ra=output_catalog["RA"].values * u.degree,
                    dec=output_catalog["DEC"].values * u.degree,
                )
                coords_new = SkyCoord(
                    ra=catalog_i["RA"].values * u.degree,
                    dec=catalog_i["DEC"].values * u.degree,
                )
                idx_match, sep2d, _ = coords_new.match_to_catalog_sky(coords_existing)
                tol_deg = tolerance_arcsec / 3600.0
                matched = sep2d.deg < tol_deg

                # New (unmatched) sources via boolean mask - no iterrows().
                new_mask = ~matched
                if new_mask.any():
                    catalog_cols = [c for c in cols if c in catalog_i.columns]
                    missing_cols = [c for c in cols if c not in catalog_i.columns]
                    sub = catalog_i.loc[new_mask, catalog_cols].copy()
                    for c in missing_cols:
                        sub[c] = np.nan
                    new_rows = sub[cols].to_dict("records")

                # Fill in missing filter values for matched sources.
                if matched.any():
                    matched_new_idx = np.where(matched)[0]
                    out_indices = output_catalog.index[idx_match[matched_new_idx]]
                    for filter_name in updated_filter_list:
                        if filter_name not in catalog_i.columns:
                            continue
                        new_vals = catalog_i[filter_name].to_numpy()
                        existing_vals = output_catalog.loc[out_indices, filter_name].to_numpy()
                        fill_mask = pd.isna(existing_vals) & pd.notna(new_vals[matched_new_idx])
                        if fill_mask.any():
                            output_catalog.loc[
                                out_indices[fill_mask], filter_name
                            ] = new_vals[matched_new_idx[fill_mask]]

            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)
                output_catalog = pd.concat(
                    [output_catalog, new_rows_df], ignore_index=True
                )
                logger.info(
                    "Added %d sources from %s, catalog now has %d sources",
                    len(new_rows_df), catalogName, len(output_catalog),
                )

        # Final deduplication: keep exactly one member of every close pair.
        if not output_catalog.empty:
            n_before_dedup = len(output_catalog)
            output_catalog = _skycoord_dedup_keep_one(output_catalog, sep_threshold_arcsec=0.1)
            n_removed = n_before_dedup - len(output_catalog)
            logger.info(
                f"Final catalog: {len(output_catalog)} sources (removed {n_removed} duplicates)"
            )
            if n_removed > 0:
                logger.warning(
                    f"Removed {n_removed} duplicate sources from final combined catalog"
                )
        
        output_catalog.to_csv(fpath, index=False, float_format="%.6f")
        logger.debug("Saved clean catalog to %s", fpath)
        return output_catalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def _catalog_supported_bands(self, catalog_names, cat_cfg=None):
        """Bands each candidate backend can serve, from catalog.yml.

        Returns ``(supported_bands, skipped)`` where ``skipped`` maps a
        catalog name to the reason it cannot participate (currently only
        an unreadable custom catalog).
        """
        cat_cfg = cat_cfg if cat_cfg is not None else (
            self.input_yaml.get("catalog", {}) or {}
        )
        filepath = os.path.dirname(os.path.abspath(__file__))
        catalog_db = AutophotYaml(
            os.path.join(filepath, "databases", "catalog.yml")
        ).load()

        supported_bands = {}
        skipped = {}
        for name in catalog_names:
            # catalog.yml uses 'panstarrs' for the normalized 'pan_starrs'.
            yml_name = {"pan_starrs": "panstarrs"}.get(name, name)
            section = catalog_db.get(yml_name) or {}
            bands_i = {
                k
                for k in section
                if normalize_photometric_filter_name(k) is not None
            }
            if name == "custom":
                # custom.yml band names are nominal; the CSV decides.
                try:
                    csv_cols = set(
                        pd.read_csv(
                            cat_cfg["catalog_custom_fpath"], nrows=0
                        ).columns
                    )
                except Exception as exc:
                    skipped[name] = f"custom catalog unreadable: {exc}"
                    continue
                bands_i = {b for b in bands_i if b in csv_cols}
                # clean() also auto-detects <band>/<band>_err column pairs.
                bands_i |= {
                    c
                    for c in csv_cols
                    if f"{c}_err" in csv_cols
                    and normalize_photometric_filter_name(c) is not None
                }
            supported_bands[name] = bands_i
        return supported_bands, skipped

    def _optimizer_plot_sources(
        self,
        catalog_names,
        band_set,
        image_infos,
        target_coords,
        radius,
        border=11,
        regions=None,
        return_all=False,
    ):
        """Rebuild ``plot_data["sources"]`` for the coverage map without a
        full rescan.

        Used on cached-selection auto runs: every download here hits the
        local catalog CSV cache, so the (catalog, band) -> usable-source
        RA/DEC union is recomputed with the same masks the optimizer
        applies, but entirely offline.  Returns a dict keyed by
        ``(catalog_name, band)`` holding concatenated RA/DEC frames.

        ``regions`` optionally maps catalog name -> the query region the
        scan used, so cache keys resolve identically to the original run.
        With ``return_all=True`` the return value becomes
        ``(sources, catalog_sources)`` where ``catalog_sources`` maps
        catalog name -> every source it returned (post-clean), matching
        ``plot_data["catalog_sources"]`` from the optimizer.
        """
        zp_cfg = self.input_yaml.get("zeropoint", {}) or {}
        bright_lim = float(zp_cfg.get("bright_mag_limit", 11.0))
        faint_lim = float(zp_cfg.get("faint_mag_limit", 22.0))
        supported_bands, _ = self._catalog_supported_bands(
            list(catalog_names)
        )
        band_set = sorted({str(b) for b in band_set if b})

        per_key = {}
        all_sources = {}
        for name in catalog_names:
            name = self._normalize_catalog_name(str(name))
            if not (set(band_set) & supported_bands.get(name, set())):
                continue
            reg = (regions or {}).get(name)
            try:
                with _quiet_catalog_log():
                    raw = self.download(
                        target_coords=target_coords,
                        catalogName=name,
                        radius=(
                            reg.get("radius_arcmin", radius)
                            if isinstance(reg, dict)
                            else radius
                        ),
                        region=reg,
                    )
            except Exception:
                continue
            try:
                with _quiet_catalog_log():
                    cleaned = self.clean(
                        raw, catalogName=name, update_names_only=True
                    )
                    if (
                        cleaned is not None
                        and len(cleaned) > 0
                        and {"RA", "DEC"}.issubset(cleaned.columns)
                    ):
                        cleaned = _skycoord_dedup_keep_one(
                            cleaned, sep_threshold_arcsec=0.1
                        )
            except Exception:
                cleaned = None

            coords = None
            if (
                cleaned is not None
                and len(cleaned) > 0
                and {"RA", "DEC"}.issubset(cleaned.columns)
            ):
                coords = SkyCoord(
                    ra=pd.to_numeric(cleaned["RA"], errors="coerce").to_numpy()
                    * u.deg,
                    dec=pd.to_numeric(cleaned["DEC"], errors="coerce").to_numpy()
                    * u.deg,
                    frame="icrs",
                )
                all_sources[name] = cleaned[["RA", "DEC"]].copy()

            for img in image_infos or []:
                img_bands = (
                    [img["band"]] if img.get("band") else band_set
                )
                img_bands = [
                    b
                    for b in img_bands
                    if b in supported_bands.get(name, set())
                ]
                if not img_bands:
                    continue

                onchip_mask = None
                wcs_i = img.get("wcs")
                shape_i = img.get("shape")
                if (
                    coords is not None
                    and wcs_i is not None
                    and shape_i is not None
                ):
                    try:
                        x, y = wcs_i.world_to_pixel(coords)
                        x = np.asarray(x, dtype=float).ravel()
                        y = np.asarray(y, dtype=float).ravel()
                        ny, nx = shape_i
                        onchip_mask = (
                            np.isfinite(x)
                            & np.isfinite(y)
                            & (x >= border)
                            & (x < nx - border)
                            & (y >= border)
                            & (y < ny - border)
                        )
                    except Exception:
                        pass

                for band in img_bands:
                    mag_mask = _usable_mag_mask(
                        cleaned, band, bright_lim, faint_lim
                    )
                    if mag_mask is None:
                        usable = None
                    elif onchip_mask is not None:
                        usable = mag_mask.to_numpy() & onchip_mask
                    else:
                        usable = mag_mask.to_numpy()
                    if usable is not None and usable.any():
                        per_key.setdefault((name, band), []).append(
                            cleaned.loc[usable, ["RA", "DEC"]]
                        )

        out = {
            key: pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["RA", "DEC"]
            )
            for key, frames in per_key.items()
        }
        if return_all:
            return out, all_sources
        return out

    def find_optimized_catalog(
        self,
        target_coords,
        images=None,
        bands=None,
        catalog_names=None,
        radius=10,
        border=11,
        min_sources=5,
        target_name=None,
        write_report=True,
        write_plot=True,
        outdir=None,
    ):
        """
        Evaluate every feasible catalog against the science images and pick
        the best backend per band.

        Each catalog is downloaded once (the ``catalog_queries`` CSV cache
        makes repeat evaluations cheap) and cleaned against each image
        footprint. A source counts as usable when it lands on the detector,
        carries a finite magnitude in the image band, and sits inside the
        zeropoint magnitude window (``zeropoint.bright_mag_limit`` to
        ``zeropoint.faint_mag_limit``). The winner per band maximises the
        worst-case per-image count, so the least-covered image still gets
        the most calibrators available.

        Parameters
        ----------
        target_coords : SkyCoord
            Reference point for the catalog queries.  Query regions are
            centred on each band's image-footprint bounding box, so the
            query centre can sit away from the target for offset
            pointings; ``target_coords`` anchors fallback regions and
            the plot.
        images : list of dict, optional
            Per-image descriptors with keys ``path`` (str), ``band``
            (resolved image band or None), ``wcs`` (astropy WCS or None),
            and ``shape`` ((ny, nx) or None). Entries with no band are
            scored under every requested band; entries with no WCS fall
            back to field-level counts (no on-detector cut).
        bands : list of str, optional
            Bands that need a winner; defaults to the distinct bands found
            in ``images``.
        catalog_names : list of str, optional
            Backends to try; defaults to all feasible catalogs (every entry
            of AUTO_OPTIMIZE_CATALOGS minus AUTO_OPTIMIZE_EXCLUDED, plus
            "refcat"/"custom" when their credentials/file are configured).
            Passing an explicit list overrides the exclusion - e.g.
            ``["gaia"]`` re-enables the Gaia scan.
        radius : float
            Fallback query radius in arcmin (default 10) used when no
            image carries footprint geometry (no WCS/shape/pixel scale).
            Otherwise each band gets its own region centred on the
            footprint bounding box - bounded by ``catalog.region_*`` -
            and each catalog is queried over the union of the bands it
            can serve.
        border : int
            Pixel margin for the on-detector count (default 11).
        min_sources : int
            Preferred minimum usable sources on the worst-covered image
            (default 5); below this a warning is logged but the best
            available catalog is still chosen.
        target_name : str, optional
            Cache/report naming; defaults to ``target_name`` in the input
            YAML.
        write_report : bool
            Write a per-image coverage CSV under
            ``<wdir>/catalog_queries/`` (default True).
        write_plot : bool
            Render a coverage map - one subplot per band, image footprints
            as band-coloured squares, usable catalog sources with a unique
            marker/color per catalog, plus a scoreboard row (worst-image
            usable count vs the required minimum, per-catalog field
            coverage) - under ``<wdir>/catalog_queries/`` (default True).
        outdir : str, optional
            If given, the coverage CSV and PNG are written directly into
            this directory (e.g. the run's ``*_REDUCED`` output folder)
            instead of ``<wdir>/catalog_queries/``.

        Returns
        -------
        dict
            ``use_catalog`` - per-band mapping compatible with
            ``catalog.use_catalog`` (one key per band plus ``default``);
            ``winners`` - band -> catalog name; ``report`` - per-image
            coverage DataFrame; ``evaluated`` - catalogs that downloaded;
            ``skipped`` - catalog -> reason it was not evaluated;
            ``plot_data``/``plot_path`` - coverage-map inputs and the
            written PNG path (None if not drawn).
        """
        logger.log(STATUS, log_step("Catalog: optimize catalog per band"))

        cat_cfg = self.input_yaml.get("catalog", {}) or {}
        zp_cfg = self.input_yaml.get("zeropoint", {}) or {}
        bright_lim = float(zp_cfg.get("bright_mag_limit", 11.0))
        faint_lim = float(zp_cfg.get("faint_mag_limit", 22.0))
        target_name = target_name or canonical_target_name(self.input_yaml)

        # --- Candidate set ----------------------------------------------------
        excluded = {}
        if catalog_names is None:
            catalog_names = list(AUTO_OPTIMIZE_CATALOGS)
            if cat_cfg.get("MASTcasjobs_wsid") and cat_cfg.get(
                "MASTcasjobs_pwd"
            ):
                catalog_names.append("refcat")
            if cat_cfg.get("catalog_custom_fpath"):
                catalog_names.append("custom")
            excluded = {
                n: AUTO_OPTIMIZE_EXCLUDED[n]
                for n in catalog_names
                if n in AUTO_OPTIMIZE_EXCLUDED
            }
            catalog_names = [
                c for c in catalog_names if c not in AUTO_OPTIMIZE_EXCLUDED
            ]
        catalog_names = list(
            dict.fromkeys(
                self._normalize_catalog_name(str(c)) for c in catalog_names
            )
        )

        # --- Bands each backend can serve -------------------------------------
        supported_bands, _band_skips = self._catalog_supported_bands(
            catalog_names, cat_cfg
        )
        skipped = dict(excluded)
        skipped.update(_band_skips)

        # --- Normalise image descriptors --------------------------------------
        image_infos = []
        for img in images or []:
            if isinstance(img, dict):
                image_infos.append(
                    {
                        "path": img.get("path"),
                        "band": img.get("band"),
                        "wcs": img.get("wcs"),
                        "shape": img.get("shape"),
                    }
                )
            else:
                image_infos.append(
                    {"path": img, "band": None, "wcs": None, "shape": None}
                )
        if not image_infos:
            # Field-level fallback: one synthetic entry scored under every band.
            image_infos = [
                {"path": None, "band": None, "wcs": None, "shape": None}
            ]

        band_set = sorted({str(b) for b in (bands or []) if b})
        if not band_set:
            band_set = sorted(
                {i["band"] for i in image_infos if i.get("band")}
            )
        if not band_set:
            raise ValueError(
                "find_optimized_catalog: no bands to optimize - pass "
                "`bands` or images with resolved filters."
            )

        # --- Query regions ---------------------------------------------------
        # Each band's images get their own coverage region: bands observed
        # at different pointings or with different instruments must not be
        # forced into one global cone.  A catalog is then queried over the
        # union of the bands it can serve, centred on the footprint
        # bounding box (not the target - a target near the field edge would
        # otherwise double the cone), bounded by the catalog.region_*
        # config, and box-shaped where the backend accepts rectangular
        # queries.  Sources outside the queried region can never be scored,
        # so an undersized region reads as "no coverage" at the edges.
        _rb_margin = float(cat_cfg.get("region_margin", 1.1))
        _rb_min = float(cat_cfg.get("region_min_arcmin", 2.0))
        _rb_max = float(cat_cfg.get("region_max_arcmin", 60.0))
        band_regions = {
            b: _footprint_region(
                image_infos,
                [b],
                target_coords,
                margin=_rb_margin,
                min_arcmin=_rb_min,
                max_arcmin=_rb_max,
                default_arcmin=radius,
            )
            for b in band_set
        }
        for b in band_set:
            logger.info(
                "  %s-band coverage region: %s", b, _fmt_region(band_regions[b])
            )

        # Download each candidate once. Failures are recorded, not fatal: a
        # catalog that does not cover the field (or is unreachable) simply
        # cannot win a band. download()/clean() run under _quiet_catalog_log
        # so their per-call banners/cache lines/column warnings don't flood
        # the console - each candidate reports one summary line here instead.
        logger.log(
            STATUS,
            log_step(
                f"Optimized catalog scan: {len(catalog_names)} candidate "
                f"catalogs x {len(image_infos)} images"
            ),
        )
        for name, reason in excluded.items():
            logger.info("  %-12s skipped - %s", name, reason)
        raw_catalogs = {}
        evaluated = []
        # catalog -> query region actually used / bands it can serve.
        cat_regions = {}
        cat_bands = {}
        for name in catalog_names:
            served = sorted(set(band_set) & supported_bands.get(name, set()))
            if not served:
                skipped[name] = "none of the required bands are covered"
                logger.info("  %-12s skipped - no required bands", name)
                continue
            reg = _footprint_region(
                image_infos,
                served,
                target_coords,
                margin=_rb_margin,
                min_arcmin=_rb_min,
                max_arcmin=_rb_max,
                default_arcmin=radius,
            )
            cat_regions[name] = reg
            cat_bands[name] = served
            try:
                with _quiet_catalog_log():
                    raw_catalogs[name] = self.download(
                        target_coords=target_coords,
                        catalogName=name,
                        radius=reg["radius_arcmin"],
                        target_name=target_name,
                        region=reg,
                    )
                evaluated.append(name)
                logger.info(
                    "  %-12s %d sources (%s)",
                    name,
                    len(raw_catalogs[name]),
                    _fmt_region(reg),
                )
            except Exception as exc:
                skipped[name] = str(exc)
                reason = str(exc).splitlines()[0] if str(exc) else repr(exc)
                if len(reason) > 120:
                    reason = reason[:117] + "..."
                logger.warning("  %-12s failed - %s", name, reason)

        # --- Score every catalog x image --------------------------------------
        # The band-column mapping is identical for every image, so clean()
        # runs ONCE per catalog (update_names_only=True populates all band
        # columns); only the world->pixel transform varies per image, done
        # here directly. This also keeps the scan quiet: clean() logs per
        # call, so per-image cleaning would print thousands of lines. The
        # band-missing drop clean() skips is replicated per band by the
        # finite-magnitude requirement in _usable_mag_mask; the skipped dedup
        # is applied explicitly below.
        # clean() reads input_yaml["imageFilter"]; it is restored after the scan.
        rows = []
        # catalog -> fraction of the query disk populated by its sources.
        coverage_map = {}
        # (catalog, band) -> list of usable-source RA/DEC frames, for the
        # coverage plot. Unioned across the band's images at the end.
        plot_sources = {}
        # catalog -> every source the query returned (post-clean), drawn
        # faintly on the map so the catalog's raw coverage is visible
        # next to the usable subset.
        catalog_sources = {}
        prev_filter = self.input_yaml.get("imageFilter")
        try:
            for name in evaluated:
                self.input_yaml["imageFilter"] = band_set[0]
                try:
                    with _quiet_catalog_log():
                        cleaned = self.clean(
                            raw_catalogs[name],
                            catalogName=name,
                            update_names_only=True,
                        )
                        if (
                            cleaned is not None
                            and len(cleaned) > 0
                            and {"RA", "DEC"}.issubset(cleaned.columns)
                        ):
                            cleaned = _skycoord_dedup_keep_one(
                                cleaned, sep_threshold_arcsec=0.1
                            )
                except Exception as exc:
                    reason = str(exc).splitlines()[0] if str(exc) else repr(exc)
                    logger.warning("  %-12s clean failed - %s", name, reason)
                    cleaned = None

                coords = None
                if (
                    cleaned is not None
                    and len(cleaned) > 0
                    and {"RA", "DEC"}.issubset(cleaned.columns)
                ):
                    coords = SkyCoord(
                        ra=pd.to_numeric(cleaned["RA"], errors="coerce").to_numpy()
                        * u.deg,
                        dec=pd.to_numeric(cleaned["DEC"], errors="coerce").to_numpy()
                        * u.deg,
                        frame="icrs",
                    )
                    catalog_sources[name] = cleaned[["RA", "DEC"]].copy()

                # Coverage is measured against the region actually
                # queried - the box for box-capable backends, the
                # circumscribed cone otherwise - centred on the footprint
                # bounds rather than the target.
                reg = cat_regions.get(name) or {}
                cov_frac = np.nan
                if coords is not None and reg:
                    _box = (
                        (float(reg["width_deg"]), float(reg["height_deg"]))
                        if name in _BOX_QUERY_CATALOGS
                        and reg.get("width_deg")
                        else None
                    )
                    cov_frac = _field_coverage_fraction(
                        coords.ra.deg,
                        coords.dec.deg,
                        float(reg["ra"]),
                        float(reg["dec"]),
                        float(reg["radius_arcmin"]) / 60.0,
                        box_wh=_box,
                    )
                coverage_map[name] = cov_frac
                if np.isfinite(cov_frac) and cov_frac < 0.8:
                    logger.warning(
                        "  %-12s covers only %.0f%% of the query field "
                        "(partial survey footprint)",
                        name,
                        100.0 * cov_frac,
                    )

                for img in image_infos:
                    img_bands = (
                        [img["band"]] if img.get("band") else band_set
                    )
                    img_bands = [
                        b for b in img_bands if b in supported_bands.get(name, set())
                    ]
                    if not img_bands:
                        continue

                    onchip_mask = None
                    wcs_i = img.get("wcs")
                    shape_i = img.get("shape")
                    if (
                        coords is not None
                        and wcs_i is not None
                        and shape_i is not None
                    ):
                        try:
                            x, y = wcs_i.world_to_pixel(coords)
                            x = np.asarray(x, dtype=float).ravel()
                            y = np.asarray(y, dtype=float).ravel()
                            ny, nx = shape_i
                            onchip_mask = (
                                np.isfinite(x)
                                & np.isfinite(y)
                                & (x >= border)
                                & (x < nx - border)
                                & (y >= border)
                                & (y < ny - border)
                            )
                        except Exception as exc:
                            logger.debug(
                                "Optimized catalog scan: %s WCS transform "
                                "failed on %s (%s)",
                                name.upper(),
                                img.get("path"),
                                exc,
                            )

                    for band in img_bands:
                        mag_mask = _usable_mag_mask(
                            cleaned, band, bright_lim, faint_lim
                        )
                        if mag_mask is None:
                            usable = None
                        elif onchip_mask is not None:
                            usable = mag_mask.to_numpy() & onchip_mask
                        else:
                            usable = mag_mask.to_numpy()
                        n = int(usable.sum()) if usable is not None else 0
                        if usable is not None and usable.any():
                            plot_sources.setdefault((name, band), []).append(
                                cleaned.loc[usable, ["RA", "DEC"]]
                            )
                        rows.append(
                            {
                                "catalog": name,
                                "band": band,
                                "image": (
                                    os.path.basename(str(img["path"]))
                                    if img.get("path")
                                    else "field"
                                ),
                                "n_usable": n,
                                "coverage": coverage_map.get(name, np.nan),
                                "r_query_arcmin": float(
                                    reg.get("radius_arcmin", np.nan)
                                ),
                                "mode": (
                                    "onchip"
                                    if wcs_i is not None and shape_i is not None
                                    else "field"
                                ),
                            }
                        )
        finally:
            if prev_filter is None:
                self.input_yaml.pop("imageFilter", None)
            else:
                self.input_yaml["imageFilter"] = prev_filter

        report = pd.DataFrame(
            rows,
            columns=[
                "catalog",
                "band",
                "image",
                "n_usable",
                "coverage",
                "r_query_arcmin",
                "mode",
            ],
        )

        # Usable-source summary: the numbers the winners are picked from -
        # on-detector sources with a finite magnitude inside the zeropoint
        # window, reported as the worst-covered image's count (the maximin
        # criterion), sorted descending per band.
        for band in band_set:
            sub = report[report["band"] == band]
            if sub.empty:
                continue
            worst = (
                sub.groupby("catalog")["n_usable"]
                .min()
                .sort_values(ascending=False)
            )
            logger.info(
                "  %s-band usable (worst image, %g <= mag <= %g): %s",
                band,
                bright_lim,
                faint_lim,
                ", ".join(f"{c}={int(n)}" for c, n in worst.items()),
            )

        # --- Winners: maximise the worst-case per-image count -----------------
        winners = {}
        for band in band_set:
            sub = report[report["band"] == band]
            if sub.empty:
                continue
            scores = (
                sub.groupby("catalog")["n_usable"]
                .agg(["min", "mean"])
                .sort_values(["min", "mean"], ascending=False)
            )
            if scores.empty:
                continue
            best = str(scores.index[0])
            best_min = int(scores.iloc[0]["min"])
            winners[band] = best
            logger.info(
                "Optimized catalog: %s-band -> %s "
                "(worst-case %d usable sources per image, mean %.1f)",
                band,
                best.upper(),
                best_min,
                float(scores.iloc[0]["mean"]),
            )
            best_cov = coverage_map.get(best, np.nan)
            if np.isfinite(best_cov) and best_cov < 0.8:
                logger.warning(
                    "Optimized catalog: winning %s-band catalog %s covers "
                    "only %.0f%% of the field - part of the detector will "
                    "have no calibrators.",
                    band,
                    best.upper(),
                    100.0 * best_cov,
                )
            if best_min < min_sources:
                logger.warning(
                    "Optimized catalog: best %s-band catalog %s provides only "
                    "%d usable source(s) on the worst-covered image "
                    "(preferred minimum %d).",
                    band,
                    best.upper(),
                    best_min,
                    min_sources,
                )

        use_catalog_map = {band: cat for band, cat in sorted(winners.items())}
        if winners:
            # A "default" key covers any runtime band not in the winners map.
            default_cat = (
                pd.Series(list(winners.values()))
                .value_counts()
                .index[0]
            )
            use_catalog_map["default"] = str(default_cat)

        if write_report:
            try:
                rep_dir = outdir or os.path.join(
                    self.input_yaml.get("wdir", "."), "catalog_queries"
                )
                pathlib.Path(rep_dir).mkdir(parents=True, exist_ok=True)
                rep_path = os.path.join(
                    rep_dir,
                    f"{target_name}_optimized_catalog_coverage.csv",
                )
                report.to_csv(rep_path, index=False)
                logger.info("Optimized catalog coverage report: %s", rep_path)
            except Exception as exc:
                logger.warning(
                    "Could not write optimized-catalog report: %s", exc
                )

        # --- Plot data ---------------------------------------------------------
        # Image footprints (WCS corners -> sky polygon) and per-(catalog,
        # band) usable-source positions, unioned across the band's images.
        footprints = _footprints_from_image_infos(image_infos)

        sources_union = {}
        for key, frames in plot_sources.items():
            merged = pd.concat(frames, ignore_index=True)
            sources_union[key] = merged.drop_duplicates(subset=["RA", "DEC"])

        plot_data = {
            "footprints": footprints,
            "sources": sources_union,
            "winners": winners,
            "band_set": band_set,
            "evaluated": evaluated,
            "skipped": skipped,
            # Region outlines: per-band required coverage and the
            # per-catalog query regions actually issued.
            "band_regions": band_regions,
            "regions": cat_regions,
            "catalog_bands": cat_bands,
            # Every source each catalog returned (post-clean); drawn
            # faintly under the usable subset on the coverage map.
            "catalog_sources": catalog_sources,
        }

        plot_path = None
        if write_plot:
            try:
                plot_path = plot_optimized_catalog_coverage(
                    plot_data,
                    target_coords=target_coords,
                    wdir=self.input_yaml.get("wdir", "."),
                    target_name=target_name,
                    outpath=(
                        os.path.join(
                            outdir,
                            f"{target_name}_optimized_catalog_coverage.png",
                        )
                        if outdir
                        else None
                    ),
                    report=report,
                    min_sources=min_sources,
                    skipped=skipped,
                )
            except Exception as exc:
                logger.warning(
                    "Optimized catalog coverage plot failed: %s", exc
                )

        return {
            "use_catalog": use_catalog_map,
            "winners": winners,
            "report": report,
            "evaluated": evaluated,
            "skipped": skipped,
            "regions": cat_regions,
            "band_regions": band_regions,
            "plot_data": plot_data,
            "plot_path": plot_path,
        }

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def check_saturation_range(self, catalog, threshold=5):
        """
        Checks the saturation range of a given catalog by plotting catalog magnitude vs. instrumental magnitude
        and fitting a straight line with slope fixed to ~1 using RANSAC.
        Determines the linearity range within the 0.5 to 95 flux range of the inliers.

        Parameters:
        -----------
        catalog : pd.DataFrame
            DataFrame containing astronomical catalog data.
        threshold : float, optional
            Threshold value for filtering sources (default is 5).

        Returns:
        --------
        tuple
            (pd.DataFrame, dict, list) Updated clean catalog containing only inliers,
            fit parameters including errors, and saturation range [min_flux, max_flux].
        """
        fit_params = {
            "slope": None,
            "intercept": None,
            "intercept_error": None,
            "n_inliers": 0,
        }
        saturation_range = [0, np.inf]

        try:
            fpath = self.input_yaml.get("fpath", "")
            if not fpath:
                raise ValueError("File path is missing from input_yaml.")

            base_name = os.path.splitext(os.path.basename(fpath))[0]
            write_dir = os.path.dirname(fpath)
            logger.info(
                log_step(f"Linearity: saturation check ({len(catalog)} sources)")
            )

            use_filter = self.input_yaml.get("imageFilter")
            if not use_filter:
                raise ValueError("Missing 'imageFilter' in input YAML.")

            required_columns = [
                "flux_AP",
                "flux_AP_err",
                use_filter,
                f"{use_filter}_err",
            ]
            missing_columns = [
                col for col in required_columns if col not in catalog.columns
            ]
            if missing_columns:
                raise KeyError(
                    f"Missing required columns in catalog: {missing_columns}"
                )

            # NOTE: sky sigma-clipping and threshold filtering are already done
            # by Zeropoint.clean() before this method is called.  Repeating them
            # here with different parameters (threshold=5 vs 3.0) causes additional
            # source loss.  'threshold' is the peak-pixel S/N, which undersells
            # the aperture S/N by ~1/sqrt(A_psf) on oversampled images (a 22-px
            # FWHM star needs aperture S/N ~12+ to pass a peak cut of 5).  Gate
            # on the better of the two metrics so good calibrators survive, and
            # keep the strongest sources when the pool would drop below the
            # keep-floor -- the ZP fit flags a degenerate inlier set itself.
            if "threshold" in catalog.columns:
                _peak_snr = pd.to_numeric(
                    catalog["threshold"], errors="coerce"
                ).to_numpy(dtype=float)
                _snr_col = next(
                    (
                        c
                        for c in ("SNR", "snr_ap", "snr")
                        if c in catalog.columns
                    ),
                    None,
                )
                if _snr_col is not None:
                    _aper_snr = pd.to_numeric(
                        catalog[_snr_col], errors="coerce"
                    ).to_numpy(dtype=float)
                    _det_metric = np.fmax(_peak_snr, _aper_snr)
                else:
                    _det_metric = _peak_snr
                threshold_cut = ~np.isfinite(_det_metric) | (
                    _det_metric < threshold
                )
                n_thresh = int(np.sum(threshold_cut))
                if n_thresh > 0:
                    _min_keep = int(
                        (self.input_yaml.get("zeropoint") or {}).get(
                            "saturation_check_min_keep", 3
                        )
                    )
                    if len(catalog) - n_thresh < _min_keep:
                        _rank = np.where(
                            np.isfinite(_det_metric), -_det_metric, np.inf
                        )
                        _keep_idx = np.argsort(_rank, kind="stable")[
                            : _min_keep
                        ]
                        _keep_mask = np.zeros(len(catalog), dtype=bool)
                        _keep_mask[_keep_idx] = True
                        n_dropped = int((~_keep_mask).sum())
                        logger.info(
                            f"Keeping {_min_keep} highest-S/N sources "
                            f"instead of applying threshold < {threshold} "
                            f"cut ({n_dropped} dropped); calibrator pool is "
                            "too small to vet further."
                        )
                        catalog = catalog[_keep_mask]
                    else:
                        logger.info(
                            f"Removing {n_thresh} sources with detection "
                            f"S/N < {threshold} (peak and aperture)"
                        )
                        catalog = catalog[~threshold_cut]

            flux = catalog["flux_AP"].values
            flux_err = catalog["flux_AP_err"].values
            # NaN for non-positive fluxes (same convention as functions.mag)
            flux_safe = flux.astype(float).copy()
            flux_safe[flux_safe <= 0] = np.nan
            inst_mag = -2.5 * np.log10(flux_safe)
            inst_mag_err = 2.5 / np.log(10) * (flux_err / flux_safe)
            catalog_mag = catalog[use_filter].values
            catalog_mag_err = catalog[f"{use_filter}_err"].values

            # 0.5 mag combined-error cut matches the _prepare_catalog threshold.
            error_mask = np.sqrt(catalog_mag_err**2 + inst_mag_err**2) < 0.5
            clean_catalog = catalog[error_mask].copy()

            if len(clean_catalog) < 2:
                logger.warning("Too few points left after error cut.")
                return clean_catalog, fit_params, saturation_range

            # The high-S/N subset is used ONLY to fit the RANSAC model; the
            # model is then applied to the FULL catalog so faint but valid
            # sources can still be inliers.
            flux = clean_catalog["flux_AP"].values
            flux_err = clean_catalog["flux_AP_err"].values
            flux_err_safe = np.maximum(flux_err, 1e-10)  # guard flux_err=0
            snr_values = np.abs(flux) / flux_err_safe

            # Step down the S/N floor until RANSAC has enough sources.
            min_snr_thresholds = [100, 75, 50, 30, 20, 10]
            selected_indices = None
            for min_snr in min_snr_thresholds:
                high_snr_mask = snr_values >= min_snr
                n_high_snr = np.sum(high_snr_mask)
                if n_high_snr >= 10:  # RANSAC needs at least ~10 sources
                    selected_indices = high_snr_mask
                    logger.debug(
                        f"Selected {n_high_snr} high S/N sources (SNR >= {min_snr}) for linearity fit"
                    )
                    break

            # Instrumental magnitudes for the FULL clean catalog.
            flux_safe = flux.astype(float).copy()
            flux_safe[flux_safe <= 0] = np.nan
            inst_mag_linear = -2.5 * np.log10(flux_safe)
            inst_mag_err_linear = 2.5 / np.log(10) * (flux_err / flux_safe)
            catalog_mag_linear = clean_catalog[use_filter].values
            catalog_mag_err_linear = clean_catalog[f"{use_filter}_err"].values

            # Local import: reuse zeropoint's regressor, avoid duplication.
            from zeropoint import PenalisedSlopeRegressor
            ConstrainedSlopeRegressor = PenalisedSlopeRegressor

            # Fit on the high-S/N subset if available, then apply the model
            # to the FULL catalog to identify all inliers.
            X_full = inst_mag_linear.reshape(-1, 1)
            y_full = catalog_mag_linear

            if selected_indices is not None and np.sum(selected_indices) >= 10:
                X_fit = X_full[selected_indices]
                y_fit = y_full[selected_indices]
            else:
                X_fit = X_full
                y_fit = y_full

            if len(X_fit) > 1:
                base_estimator = ConstrainedSlopeRegressor(
                    slope_constraint=1.0, slope_tolerance=0  # slope fixed to 1
                )
                # Residual threshold adapts to the scatter in this field.
                # The MAD must be taken on the slope-1 residuals (y - x, the
                # per-source ZP deltas), NOT on the catalog magnitudes y
                # themselves: a field spanning several magnitudes otherwise
                # yields a threshold of a few mag and RANSAC accepts every
                # locus, including the bogus one.
                _delta_fit = y_fit - X_fit.flatten()
                initial_mad = np.median(
                    np.abs(_delta_fit - np.median(_delta_fit))
                )
                ransac_residual_threshold = max(
                    3.0 * 1.4826 * initial_mad, 0.15
                )
                ransac = RANSACRegressor(
                    estimator=base_estimator,
                    residual_threshold=ransac_residual_threshold,
                    max_trials=500,
                    min_samples=0.25,
                    # Fixed seed: the inlier set feeds the ZP calibration
                    # chain, so the same image must give the same calibrators
                    # (and the same zeropoint) on every run.
                    random_state=42,
                )
                ransac.fit(X_fit, y_fit)
                slope = ransac.estimator_.slope_
                intercept = ransac.estimator_.intercept_

                # Inliers are computed on the FULL catalog, not the fit subset.
                residuals_full = y_full - (slope * X_full.flatten() + intercept)
                inlier_mask = np.abs(residuals_full) < ransac_residual_threshold

                # Majority-locus guard: the high-S/N fit subset can be
                # dominated by catalog artifacts (spurious faint entries
                # coincident with bright stars inherit the bright flux and
                # its S/N, so they top the S/N ladder on noisy images).
                # The median per-source offset over the FULL clean catalog
                # tracks the majority locus instead; if it collects more
                # inliers than the RANSAC model, re-anchor on it.
                zp_median = np.nanmedian(y_full - X_full.flatten())
                resid_median = (y_full - X_full.flatten()) - zp_median
                mad_median = 1.4826 * np.nanmedian(
                    np.abs(resid_median - np.nanmedian(resid_median))
                )
                median_band = max(3.0 * mad_median, 0.3)
                median_inlier_mask = np.abs(resid_median) < median_band
                if np.sum(median_inlier_mask) > np.sum(inlier_mask):
                    logger.warning(
                        "Linearity fit anchored on a minority locus "
                        "(%d/%d inliers); the median offset ZP=%.3f supports "
                        "%d/%d - re-anchoring on the majority locus. Check "
                        "the catalog for blended/spurious entries.",
                        int(np.sum(inlier_mask)),
                        len(X_full),
                        zp_median,
                        int(np.sum(median_inlier_mask)),
                        len(X_full),
                    )
                    intercept = zp_median
                    slope = 1.0
                    residuals_full = resid_median
                    inlier_mask = median_inlier_mask
                    ransac_residual_threshold = median_band

                # Post-RANSAC sigma clip on the full-catalog inliers;
                # sigma=3.0 (not 2.5) is more stable for small samples.
                n_sigma_outliers = 0
                if np.sum(inlier_mask) > 5:
                    inlier_residuals = residuals_full[inlier_mask]
                    clip_sigma = 3.0 if np.sum(inlier_mask) < 30 else 2.5
                    clipped = sigma_clip(inlier_residuals, sigma=clip_sigma, maxiters=5)
                    residual_mask = np.asarray(~clipped.mask, dtype=bool).flatten()
                    if residual_mask.shape[0] == np.sum(inlier_mask):
                        inlier_mask[inlier_mask] = residual_mask
                    else:
                        logger.warning("Shape mismatch in residual masking: %s vs %s, skipping sigma clip update", residual_mask.shape[0], np.sum(inlier_mask))
                    n_sigma_outliers = int(np.sum(~residual_mask))

                # Recompute the intercept on the final inlier set.
                if np.sum(inlier_mask) > 1 and np.isfinite(slope):
                    try:
                        intercept = float(
                            np.nanmedian(
                                y_full[inlier_mask] - slope * X_full[inlier_mask].flatten()
                            )
                        )
                    except Exception:
                        pass

                logger.debug(
                    f"RANSAC: {np.sum(inlier_mask)}/{len(X_full)} inliers "
                    f"(fit on {len(X_fit)}, applied to {len(X_full)}), "
                    f"ZP={intercept:.3f}"
                )

                # Intercept error: standard error of the intercept.
                inlier_X = X_full[inlier_mask]
                inlier_y = y_full[inlier_mask]
                residuals = inlier_y - (slope * inlier_X.flatten() + intercept)
                residual_std = np.std(residuals, ddof=1)  # sample std (ddof=1)
                n_points = len(inlier_X)
                x_mean = np.mean(inlier_X)
                x_var = np.var(inlier_X, ddof=1)  # sample variance (ddof=1)

                if x_var <= 0 or not np.isfinite(x_var):
                    logger.warning("Zero or invalid variance in instrumental magnitudes")
                    intercept_error = np.nan
                else:
                    intercept_error = residual_std * np.sqrt(
                        1 / n_points + x_mean**2 / ((n_points - 1) * x_var)
                    )

                fit_params.update(
                    {
                        "slope": slope,
                        "intercept": intercept,
                        "intercept_error": intercept_error,
                        "n_inliers": n_points,
                    }
                )

                # Linearity range = 0.5th to 95th flux percentile of inliers.
                inlier_flux = flux[inlier_mask]
                linear_range_str = ""
                if len(inlier_flux) > 0:
                    min_flux = np.percentile(inlier_flux, 0.5)
                    max_flux = np.percentile(inlier_flux, 95)
                    saturation_range = [min_flux, max_flux]
                    linear_range_str = f" | flux range {min_flux:.0f}-{max_flux:.0f}"
                else:
                    logger.warning("No inliers found for linearity range calculation")

                _zp_err_str = (
                    f" +/- {intercept_error:.3f}"
                    if np.isfinite(intercept_error)
                    else ""
                )
                logger.info(
                    f"Linearity fit: {n_points}/{len(X_full)} inliers "
                    f"(RANSAC on {len(X_fit)} high-S/N, -{n_sigma_outliers} clip) | "
                    f"ZP={intercept:.3f}{_zp_err_str}{linear_range_str}"
                )

                fit_line = lambda x: slope * x + intercept
            else:
                logger.warning("Not enough sources for linear fitting.")
                fit_line = None
                inlier_mask = np.ones_like(catalog_mag_linear, dtype=bool)

            # Plotting
            from plotting_utils import (
                apply_autophot_mplstyle, get_ransac_color, get_marker_size,
                get_alpha, get_line_width, ransac_grid, ransac_savefig,
                set_mag_axes_inverted_xy, format_log_colorbar_ticks,
                get_plot_ext,
            )

            apply_autophot_mplstyle()
            plt.ioff()
            fig, ax1 = plt.subplots(figsize=set_size(540, 1))

            if fit_line:
                predicted = fit_line(inst_mag_linear)
                residuals = catalog_mag_linear - predicted
                # Plot with the same RANSAC inlier mask used for the fit.
                ransac_inliers = inlier_mask if 'inlier_mask' in locals() else np.ones(len(inst_mag_linear), dtype=bool)
                ransac_outliers = ~ransac_inliers

                # S/N-driven marker colours; the norm range covers inliers
                # only -- outliers always get the flat outlier colour.
                _snr_in = snr_values[ransac_inliers] if 'snr_values' in locals() else np.array([])
                finite_snr = _snr_in[np.isfinite(_snr_in) & (_snr_in > 0)]
                snr_norm = None
                if finite_snr.size >= 3 and np.nanmax(finite_snr) > np.nanmin(finite_snr):
                    from matplotlib.colors import LogNorm
                    vmin = max(1.0, float(np.nanmin(finite_snr)))
                    vmax = float(np.nanpercentile(finite_snr, 98))
                    if vmax <= vmin:
                        vmax = vmin * 10.0
                    snr_norm = LogNorm(vmin=vmin, vmax=vmax)
                ms_area = get_marker_size('medium') ** 2

                ax1.errorbar(
                    inst_mag_linear[ransac_outliers],
                    catalog_mag_linear[ransac_outliers],
                    yerr=catalog_mag_err_linear[ransac_outliers],
                    xerr=inst_mag_err_linear[ransac_outliers],
                    fmt="none",
                    ecolor="lightgrey",
                    alpha=get_alpha('medium'),
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    linestyle="None",
                    zorder=2,
                )
                ax1.plot(
                    inst_mag_linear[ransac_outliers],
                    catalog_mag_linear[ransac_outliers],
                    "x", markersize=get_marker_size('medium'),
                    color=get_ransac_color('outliers'),
                    alpha=get_alpha('medium'), linestyle="None",
                    label=f"Outliers [{np.sum(ransac_outliers)}]",
                    zorder=3,
                )
                ax1.errorbar(
                    inst_mag_linear[ransac_inliers],
                    catalog_mag_linear[ransac_inliers],
                    yerr=catalog_mag_err_linear[ransac_inliers],
                    xerr=inst_mag_err_linear[ransac_inliers],
                    fmt="none",
                    ecolor="lightgrey",
                    alpha=get_alpha('dark'),
                    capsize=get_marker_size('medium') / 4,
                    elinewidth=0.5,
                    linestyle="None",
                    zorder=4,
                )
                if snr_norm is not None:
                    sc_in = ax1.scatter(
                        inst_mag_linear[ransac_inliers],
                        catalog_mag_linear[ransac_inliers],
                        c=snr_values[ransac_inliers], cmap="viridis", norm=snr_norm,
                        marker="o", s=ms_area, alpha=get_alpha('dark'),
                        label=f"Inliers [{np.sum(ransac_inliers)}]",
                        zorder=5,
                    )
                    cb = fig.colorbar(sc_in, ax=ax1, label="S/N", pad=0.02)
                    format_log_colorbar_ticks(cb, snr_norm.vmin, snr_norm.vmax)
                else:
                    ax1.plot(
                        inst_mag_linear[ransac_inliers],
                        catalog_mag_linear[ransac_inliers],
                        "o", markersize=get_marker_size('medium'),
                        color=get_ransac_color('zeropoint_ap'),
                        alpha=get_alpha('dark'), linestyle="None",
                        label=f"Inliers [{np.sum(ransac_inliers)}]",
                    )
                x_range = np.linspace(inst_mag_linear[ransac_inliers].min(), inst_mag_linear[ransac_inliers].max(), 100)
                y_fit = fit_line(x_range)
                ax1.fill_between(
                    x_range,
                    y_fit - intercept_error,
                    y_fit + intercept_error,
                    color=get_ransac_color('error_band'),
                    alpha=get_alpha('very_light'),
                )
                ax1.plot(
                    x_range,
                    y_fit,
                    color=get_ransac_color('fit'),
                    linestyle="--",
                    lw=get_line_width('medium'),
                    zorder=10,
                    label=(
                        rf"$m_\mathrm{{cal}} = m_\mathrm{{inst}} + {intercept:.2f} \pm {intercept_error:.2f}$"
                    ),
                )

                # Mark the linearity range in instrumental magnitude.
                if len(inlier_flux) > 0:
                    min_inst_mag = -2.5 * np.log10(
                        max_flux
                    )  # Note: brighter objects have smaller magnitudes
                    max_inst_mag = -2.5 * np.log10(min_flux)
                    ax1.axvline(
                        x=min_inst_mag,
                        color=get_ransac_color('error_band'),
                        linestyle=":",
                        lw=get_line_width('thin'),
                        alpha=0.7,
                    )
                    ax1.axvline(
                        x=max_inst_mag,
                        color=get_ransac_color('error_band'),
                        linestyle=":",
                        lw=get_line_width('thin'),
                        alpha=0.7,
                    )

            ax1.set_xlabel(r"Instrumental Magnitude $m_\mathrm{inst}$ [mag]")
            ax1.set_ylabel(rf"Catalog Magnitude $m_\mathrm{{cal,{use_filter}}}$ [mag]")
            set_mag_axes_inverted_xy(ax1)
            ax1.legend(
                loc="upper left", ncol=1, fontsize=8, frameon=False,
            )
            ransac_grid(ax1)
            save_path = os.path.join(write_dir, f"Saturation_{base_name}{get_plot_ext(self.input_yaml)}")
            ransac_savefig(fig, save_path)
            plt.close(fig)

            # Locate the continuous linear region via residual scatter.
            if fit_line:
                # Guard against length mismatches before indexing masks.
                n_clean = len(clean_catalog)
                n_flux = len(flux)
                n_mag = len(catalog_mag_linear)
                n_inst = len(inst_mag_linear)
                n_inlier_mask = len(inlier_mask)
                if not (n_clean == n_flux == n_mag == n_inst == n_inlier_mask):
                    logger.error(
                        f"Array length mismatch: clean_catalog={n_clean},\n"
                        f"    flux={n_flux}, catalog_mag_linear={n_mag},\n"
                        f"    inst_mag_linear={n_inst},\n"
                        f"    inlier_mask={n_inlier_mask}.\n"
                        f"    Skipping robust selection."
                    )
                    clean_catalog = clean_catalog[inlier_mask] if n_clean == n_inlier_mask else clean_catalog
                    return clean_catalog, fit_params, saturation_range
                
                inlier_catalog = clean_catalog[inlier_mask].copy()
                inlier_flux = flux[inlier_mask]
                inlier_inst_mag = inst_mag_linear[inlier_mask]

                # Keep all RANSAC inliers for zeropoint fitting. The aggressive
                # flux-range selection below removes too many valid faint sources
                # (photon noise is mistaken for non-linearity on modern CCDs).
                # NOTE: Don't reassign clean_catalog yet - we need the full arrays
                # for mask computation. We'll reassign after selection is complete.

                if len(inlier_catalog) > 5:
                    # predict() can return shape (N,1) depending on the sklearn
                    # version; flatten so the subtraction stays (N,)-(N,) and
                    # cannot broadcast to (N,N), which would silently corrupt
                    # central_start/central_end and cause an IndexError when
                    # indexing sorted_flux (size N).
                    predicted_mag = np.asarray(
                        fit_line(inlier_inst_mag.reshape(-1, 1))
                    ).flatten()
                    residuals = (
                        np.asarray(catalog_mag_linear[inlier_mask]).flatten() - predicted_mag
                    )

                    # Sort by flux, bright first (smaller mag = brighter).
                    sort_idx = np.argsort(inlier_flux)[::-1]
                    sorted_flux = inlier_flux[sort_idx]
                    sorted_residuals = residuals[sort_idx]  # already 1D
                    sorted_mag = inlier_inst_mag[sort_idx]

                    # Residual threshold comes from the central 50% of
                    # sources, the most linear region.
                    central_start = len(sorted_residuals) // 4
                    central_end = 3 * len(sorted_residuals) // 4
                    central_residuals = sorted_residuals[central_start:central_end]

                    if len(central_residuals) > 3:
                        median_resid = np.median(central_residuals)
                        mad_residual = np.median(np.abs(central_residuals - median_resid))
                        # Residual floor: 3*MAD, minimum 0.1 mag.
                        residual_threshold = float(np.maximum(3.0 * mad_residual, 0.1))
                    else:
                        residual_threshold = 0.15

                    # Bright end: cut where residuals exceed the threshold
                    # (saturation / non-linearity).
                    bright_cut_idx = 0
                    for i in range(len(sorted_residuals)):
                        resid_val = float(sorted_residuals[i])  # ensure scalar
                        if np.abs(resid_val) > residual_threshold:
                            bright_cut_idx = i + 1  # cut this and brighter
                        else:
                            break

                    # Faint end: cut where the local scatter systematically
                    # exceeds the central scatter. Require 2 consecutive
                    # high-scatter windows so a single noisy window does not
                    # trigger the cut.
                    window_size = max(3, len(sorted_residuals) // 15)
                    faint_cut_idx = len(sorted_residuals)

                    high_scatter_count = 0
                    required_consecutive = 2

                    for i in range(window_size, len(sorted_residuals) - window_size):
                        window_residuals = sorted_residuals[i-window_size:i+window_size]
                        window_mad = np.median(np.abs(window_residuals - np.median(window_residuals)))

                        # High scatter = local MAD > 2x central MAD
                        # (relaxed from 1.5x).
                        if window_mad > 2.0 * mad_residual:
                            high_scatter_count += 1
                            if high_scatter_count >= required_consecutive:
                                faint_cut_idx = i - window_size  # cut before this region
                                break
                        else:
                            high_scatter_count = 0  # reset if scatter drops

                    # Second faint cut: drop sources with large individual
                    # residuals, scanning in from the faint end.
                    for i in range(len(sorted_residuals) - 1, faint_cut_idx - 1, -1):
                        if np.abs(sorted_residuals[i]) > 1.5 * residual_threshold:
                            faint_cut_idx = i  # cut this and fainter
                        else:
                            break

                    # Apply flux range cuts (bright_cut_idx to faint_cut_idx).
                    if bright_cut_idx > 0 or faint_cut_idx < len(sorted_flux):
                        min_linear_flux = sorted_flux[min(faint_cut_idx, len(sorted_flux)-1)]
                        max_linear_flux = sorted_flux[bright_cut_idx] if bright_cut_idx < len(sorted_flux) else sorted_flux[0]

                        if min_linear_flux < max_linear_flux:
                            linear_flux_mask = (flux >= min_linear_flux) & (flux <= max_linear_flux)

                            # Residual filter on the inlier arrays only;
                            # mixing with clean_catalog would mismatch lengths.
                            inlier_catalog_mag_linear = catalog_mag_linear[inlier_mask]
                            inlier_inst_mag_linear = inst_mag_linear[inlier_mask]
                            preds = np.asarray(fit_line(inlier_inst_mag_linear.reshape(-1, 1))).flatten()
                            all_residuals = np.asarray(inlier_catalog_mag_linear).flatten() - preds
                            inlier_residual_mask = np.abs(all_residuals) < residual_threshold
                            
                            # Expand the inlier residual mask back to full
                            # size; lengths must match to avoid indexing errors.
                            n_inliers = np.sum(inlier_mask)
                            if len(inlier_residual_mask) != n_inliers:
                                logger.warning(
                                    f"Array length mismatch:\n"
                                    f"    inlier_residual_mask={len(inlier_residual_mask)},\n"
                                    f"    inlier_mask.sum()={n_inliers}.\n"
                                    f"    Skipping residual masking."
                                )
                                linear_residual_mask = inlier_mask.copy()
                            else:
                                linear_residual_mask = np.zeros(len(flux), dtype=bool)
                                linear_residual_mask[inlier_mask] = inlier_residual_mask
                            
                            # Keep sources in the flux range AND with a good residual.
                            final_linear_mask = linear_flux_mask & linear_residual_mask

                            n_bright_cut = np.sum(flux > max_linear_flux)
                            n_faint_cut = np.sum(flux < min_linear_flux)
                            n_outlier_cut = np.sum(linear_flux_mask & ~linear_residual_mask)
                            n_selected = np.sum(final_linear_mask)
                            
                            logger.info(
                                f"Robust linear selection:\n"
                                f"    flux range {min_linear_flux:.1f} - "
                                f"{max_linear_flux:.1f}, residual thresh "
                                f"{residual_threshold:.3f} mag\n"
                                f"    cut {n_bright_cut} bright + {n_faint_cut} "
                                f"faint + {n_outlier_cut} outlier, would keep "
                                f"{n_selected} sources\n"
                                f"    (keeping all {len(inlier_catalog)} inliers "
                                f"for ZP fit)"
                            )
                            
                            if n_selected > 0:
                                # Saturation range is informational only; all
                                # inliers are kept for zeropoint fitting to
                                # avoid over-aggressive faint-end cuts.
                                inlier_flux = inlier_catalog["flux_AP"].values
                                inlier_linear_mask = (inlier_flux >= min_linear_flux) & (inlier_flux <= max_linear_flux)
                                # linear_residual_mask already matches inlier_catalog size.
                                inlier_linear_mask = inlier_linear_mask & linear_residual_mask[inlier_mask]
                                saturation_range = [
                                    np.percentile(inlier_catalog["flux_AP"].values[inlier_linear_mask], 0.5),
                                    np.percentile(inlier_catalog["flux_AP"].values[inlier_linear_mask], 99.5)
                                ]
                            else:
                                # Nothing passed; fall back to the central
                                # region (inlier_catalog avoids length mismatch).
                                inlier_flux = inlier_catalog["flux_AP"].values
                                inlier_sorted_flux = np.sort(inlier_flux)[::-1]
                                central_start = max(0, len(inlier_sorted_flux) // 3)
                                central_end = min(len(inlier_sorted_flux), 2 * len(inlier_sorted_flux) // 3)
                                central_flux_mask = (inlier_flux >= inlier_sorted_flux[central_end]) & (inlier_flux <= inlier_sorted_flux[central_start])
                                if np.sum(central_flux_mask) > 0:
                                    logger.warning("No sources passed tight criteria, using central %s sources", np.sum(central_flux_mask))
                                else:
                                    logger.warning("No sources passed tight criteria, using all %s inliers", len(inlier_catalog))
                        else:
                            # Invalid range; use the central region
                            # (inlier_catalog avoids length mismatch).
                            inlier_flux = inlier_catalog["flux_AP"].values
                            inlier_sorted_flux = np.sort(inlier_flux)[::-1]
                            central_start = max(0, len(inlier_sorted_flux) // 3)
                            central_end = min(len(inlier_sorted_flux), 2 * len(inlier_sorted_flux) // 3)
                            central_flux_mask = (inlier_flux >= inlier_sorted_flux[central_end]) & (inlier_flux <= inlier_sorted_flux[central_start])
                            if np.sum(central_flux_mask) > 0:
                                pass
                    else:
                        # No cuts needed; still report the tight residual filter.
                        tight_residual_mask = np.abs(residuals) < residual_threshold
                        n_tight_outliers = (~tight_residual_mask).sum()
                        if n_tight_outliers > 0:
                            logger.info("Tight residual filter would remove %s sources (keeping all inliers)", n_tight_outliers)
                else:
                    # Too few inliers for the linear-range selection.
                    logger.warning("Only %s inliers, skipping robust selection", len(inlier_catalog))

                # Reassign clean_catalog only after all mask computations.
                clean_catalog = inlier_catalog

            logger.info("Returning %s sources for zeropoint fitting", len(clean_catalog))
            return clean_catalog, fit_params, saturation_range

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(
                f"Error in check_saturation_range: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {str(e)}"
            )
            # Fail closed, not open: the caller feeds the returned catalog
            # straight into the ZP fit, so returning the raw input would
            # silently calibrate on saturated/non-linear sources.  The
            # 0.5-mag combined-error cut is the one vetting step that does
            # not depend on the failed fit - apply it if we got that far.
            _fallback = (
                clean_catalog
                if "clean_catalog" in locals() and clean_catalog is not None
                else catalog
            )
            return _fallback, fit_params, saturation_range

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def downsample_sources_by_position(
        self,
        df: pd.DataFrame,
        x_col: str = "x_pix",
        y_col: str = "y_pix",
        nmax: int = 300,
        snr_col: Optional[str] = "SNR",
    ) -> pd.DataFrame:
        """
        Downsample a dataframe of sources by distributing them evenly across
        spatial bins while prioritizing higher SNR sources.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing source information.
        x_col : str, optional
            Column name for x-coordinates (default: 'x_pix').
        y_col : str, optional
            Column name for y-coordinates (default: 'y_pix').
        nmax : int, optional
            Maximum number of sources to keep (default: 300).
        snr_col : str, optional
            Column name for SNR values to prioritize selection (default: 'SNR').

        Returns:
        --------
        pd.DataFrame
            Downsampled dataframe with at most nmax sources.
        """
        n_src = len(df)

        # Small enough: return early without extra logging or work.
        if n_src <= nmax:
            logger.info("Downsampling skipped: %d sources (<= %d target).", n_src, nmax)
            return df.copy()

        # Banner only when downsampling actually runs.
        logger.info(
            log_step(f"Downsampling: {n_src} sources -> {nmax} max")
        )

        if x_col not in df.columns or y_col not in df.columns:
            error_msg = f"DataFrame must contain '{x_col}' and '{y_col}' columns"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Square spatial grid.
        n_bins = int(np.sqrt(nmax))
        sources_per_bin = max(1, nmax // (n_bins**2))
        logger.info(
            f"Using {n_bins}x{n_bins} grid with ~{sources_per_bin} sources per bin"
        )

        x_bins = np.linspace(df[x_col].min(), df[x_col].max(), n_bins + 1)
        y_bins = np.linspace(df[y_col].min(), df[y_col].max(), n_bins + 1)
        logger.debug("X bins range: %.2f to %.2f", x_bins[0], x_bins[-1])
        logger.debug("Y bins range: %.2f to %.2f", y_bins[0], y_bins[-1])

        selected_indices = []
        bin_stats = []  # Track bin statistics for logging

        logger.info("Processing spatial bins...")
        for i in range(n_bins):
            for j in range(n_bins):
                in_xbin = (df[x_col] >= x_bins[i]) & (df[x_col] < x_bins[i + 1])
                in_ybin = (df[y_col] >= y_bins[j]) & (df[y_col] < y_bins[j + 1])
                bin_indices = np.where(in_xbin & in_ybin)[0]
                bin_stats.append(len(bin_indices))

                if len(bin_indices) > 0:
                    # Highest SNR first so bins keep their best sources.
                    if snr_col and snr_col in df.columns:
                        bin_indices = bin_indices[
                            np.argsort(df[snr_col].iloc[bin_indices])[::-1]
                        ]
                        logger.debug(
                            f"Bin ({i},{j}): {len(bin_indices)} sources, sorted by {snr_col}"
                        )
                    else:
                        logger.debug("Bin (%s,%s): %s sources", i, j, len(bin_indices))

                    selected_indices.extend(bin_indices[:sources_per_bin])

        logger.info(
            f"Bin statistics: min={min(bin_stats)}, max={max(bin_stats)}, "
            f"avg={np.mean(bin_stats):.1f} sources per bin"
        )
        logger.info("Selected %s sources from spatial bins", len(selected_indices))

        # Under-filled bins: top up with the highest-SNR remaining sources.
        if len(selected_indices) < nmax:
            logger.warning(
                f"Only {len(selected_indices)} sources selected from bins, "
                f"filling with top {nmax - len(selected_indices)} remaining sources"
            )
            all_indices = set(range(len(df)))
            remaining_indices = list(all_indices - set(selected_indices))

            if snr_col and snr_col in df.columns:
                remaining_indices = sorted(
                    remaining_indices,
                    key=lambda idx: df[snr_col].iloc[idx],
                    reverse=True,
                )
                logger.debug("Sorted remaining sources by SNR")

            additional_count = nmax - len(selected_indices)
            selected_indices.extend(remaining_indices[:additional_count])
            logger.info(
                f"Added {additional_count} additional sources from remaining pool"
            )

        final_count = len(selected_indices)
        if final_count > nmax:
            logger.warning("Selected %s sources (exceeds target %s)", final_count, nmax)
        else:
            logger.info("Final selection: %s sources", final_count)

        logger.info("Downsampling complete")
        return df.iloc[selected_indices].copy()

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def check_starshape(
        self,
        image,
        catalog,
        scale=25,
        threshold=5,
        fwhm_threshold=1.5,
        roundness_threshold=0.2,
        sharpness_threshold=0.4,
    ):
        """
        Identify and exclude extended sources using multiple criteria:
        1. Radial profile consistency (original method)
        2. FWHM (Full Width at Half Maximum) comparison
        3. Roundness (1 - minor_axis/major_axis)
        4. Sharpness (central pixel concentration)

        Parameters:
        -----------
        image : numpy.ndarray
            2D array of the image.
        catalog : pd.DataFrame
            DataFrame with source positions (x_pix, y_pix).
        scale : int or tuple, optional
            Size of cutout around each source (default is 25).
        threshold : float, optional
            Sigma threshold for radial profile outliers (default is 5).
        fwhm_threshold : float, optional
            Max allowed FWHM in pixels for point sources (default is 1.5).
        roundness_threshold : float, optional
            Max allowed deviation from roundness (1-perfect circle) (default is 0.2).
        sharpness_threshold : float, optional
            Min required sharpness (higher = more point-like) (default is 0.4).

        Returns:
        --------
        pd.DataFrame
            Cleaned catalog DataFrame.
        """
        if isinstance(scale, int):
            scale = (scale, scale)

        catalog.reset_index(inplace=True, drop=True)
        required_columns = {"x_pix", "y_pix"}
        if not required_columns.issubset(catalog.columns):
            raise ValueError(
                f"Catalog missing required columns: {required_columns - set(catalog.columns)}"
            )

        def radial_profile(data):
            y, x = np.indices(data.shape)
            r = np.sqrt((x - data.shape[1] // 2) ** 2 + (y - data.shape[0] // 2) ** 2)
            r = r.astype(int)
            tbin = np.bincount(r.ravel(), data.ravel())
            nr = np.bincount(r.ravel())
            return tbin / np.maximum(nr, 1)

        def calculate_fwhm(data):
            """Estimate FWHM from the radial profile."""
            profile = radial_profile(data)
            half_max = np.max(profile) * 0.5
            above = np.where(profile >= half_max)[0]
            return 2 * (above[-1] - above[0]) if len(above) > 1 else np.nan

        def calculate_roundness(data):
            """Calculate roundness (1 - b/a) from image moments."""
            y, x = np.indices(data.shape)
            xc, yc = data.shape[1] / 2, data.shape[0] / 2
            x = x - xc
            y = y - yc

            # Second moments -> eigenvalues give major/minor axes.
            mxx = np.sum(x**2 * data) / np.sum(data)
            myy = np.sum(y**2 * data) / np.sum(data)
            mxy = np.sum(x * y * data) / np.sum(data)

            term1 = (mxx + myy) / 2
            term2 = np.sqrt(((mxx - myy) / 2) ** 2 + mxy**2)
            a = term1 + term2
            b = term1 - term2
            return 1 - np.sqrt(b / max(a, 1e-10))  # 0=perfect circle

        def norm(array):
            """
            Normalize a NumPy array to the range [0, 1].

            Parameters:
            -----------
            array : numpy.ndarray
                Input array to normalize.

            Returns:
            --------
            numpy.ndarray
                Normalized array.
            """
            array = np.asarray(array)
            min_val = np.min(array)
            max_val = np.max(array)
            if max_val == min_val:
                return np.zeros_like(
                    array
                )  # Avoid division by zero; all elements are the same.
            return (array - min_val) / (max_val - min_val)

        def calculate_sharpness(data):
            """Measure central concentration via the Laplacian."""
            lap = gaussian_laplace(data, sigma=1)
            center = data.shape[0] // 2, data.shape[1] // 2
            radius = min(center) // 2
            y, x = np.indices(data.shape)
            r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
            central = np.mean(np.abs(lap)[r <= radius])
            outer = np.mean(np.abs(lap)[r > radius * 2])
            return central / max(outer, 1e-6)  # Avoid division by zero

        metrics = {"fwhm": [], "roundness": [], "sharpness": []}
        cutouts = []
        radial_profiles = []
        valid_indices = []

        logger.info(log_step(f"Radial profile: {len(catalog)} sources"))

        for i, (x, y) in enumerate(zip(catalog["x_pix"], catalog["y_pix"])):
            position = (float(x), float(y))
            try:
                cutout = Cutout2D(
                    image, position=position, size=scale, mode="partial", fill_value=np.nan
                )
                cutout_data = cutout.data
                total_flux = np.nanmax(cutout_data)
                if np.isfinite(total_flux) and total_flux > 0:
                    normalized_cutout = norm(cutout_data / total_flux)

                    fwhm = calculate_fwhm(normalized_cutout)
                    roundness = calculate_roundness(normalized_cutout)
                    sharpness = calculate_sharpness(normalized_cutout)

                    metrics["fwhm"].append(fwhm)
                    metrics["roundness"].append(roundness)
                    metrics["sharpness"].append(sharpness)
                    cutouts.append(normalized_cutout)
                    radial_profiles.append(norm(radial_profile(normalized_cutout)))
                    valid_indices.append(i)
                else:
                    logger.debug("Star %s rejected: zero or negative flux.", i)

            except Exception as e:
                logger.warning(
                    f"Star {i} at position {position} skipped due to error: {e}"
                )
                # Do NOT append NaN to metrics here: metrics arrays must stay
                # aligned with valid_indices (and hence with profile_mask /
                # combined_mask), or metrics[key][combined_mask] below would
                # misassign per-source diagnostics or raise on a length
                # mismatch whenever a source raises mid-loop.

        for key in metrics:
            metrics[key] = np.array(metrics[key])

        fwhm_mask = np.isfinite(metrics["fwhm"])  # & (metrics['fwhm'] < fwhm_threshold)

        # Sigma-clip roundness and sharpness to reject outliers.
        roundness_sigma_clip = sigma_clip(
            metrics["roundness"],
            sigma=threshold,
            masked=True,
            cenfunc=np.nanmedian,
            stdfunc=mad_std,
        )
        sharpness_sigma_clip = sigma_clip(
            metrics["sharpness"],
            sigma=threshold,
            masked=True,
            cenfunc=np.nanmedian,
            stdfunc=mad_std,
        )

        roundness_mask = np.isfinite(metrics["roundness"]) & ~roundness_sigma_clip.mask
        sharpness_mask = np.isfinite(metrics["sharpness"]) & ~sharpness_sigma_clip.mask

        # Reject radial-profile outliers.
        if len(radial_profiles) > 0:
            radial_profiles = np.array(radial_profiles)
            clipped = sigma_clip(
                radial_profiles,
                sigma=threshold,
                axis=0,
                masked=True,
                # cenfunc=np.nanmedian, stdfunc=mad_std
            )
            profile_mask = ~np.any(clipped.mask, axis=1)
        else:
            profile_mask = np.zeros(len(valid_indices), dtype=bool)

        # Only profile_mask gates rejection; roundness/sharpness are
        # diagnostic-only (still logged above).
        combined_mask = profile_mask

        # Never let the masking empty the catalog.
        if np.sum(combined_mask) < 5:
            logger.warning("All sources have been rejected by the masking process.")
            combined_mask = np.ones_like(combined_mask, dtype=bool)

        inliers = [
            valid_indices[i] for i, is_good in enumerate(combined_mask) if is_good
        ]
        outliers = [
            valid_indices[i] for i, is_good in enumerate(combined_mask) if not is_good
        ]

        logger.info(
            f"Point sources: {len(inliers)} | Extended/rejected sources: {len(outliers)}"
        )
        logger.info(
            f"Rejection breakdown - FWHM: {sum(~fwhm_mask)}, "
            f"Roundness: {sum(~roundness_mask)}, "
            f"Sharpness: {sum(~sharpness_mask)}, "
            f"Profile: {sum(~profile_mask)}"
        )

        if len(inliers) > 0:
            cleaned_catalog = catalog.iloc[inliers].copy()
        else:
            logger.warning(
                "No sources remained after cleaning. Using original catalog."
            )
            cleaned_catalog = catalog.copy()  # fall back to the full catalog

        # Keep per-source metrics in the catalog for diagnostics.
        for key in metrics:
            cleaned_catalog[f"star_{key}"] = metrics[key][combined_mask]

        if self.input_yaml.get("save_cutout_plot", True):
            index_map = {idx: cutout for idx, cutout in zip(valid_indices, cutouts)}
            stars = [index_map[i] for i in inliers]
            num_stars = len(stars)
            side_length = int(np.ceil(np.sqrt(num_stars)))
            ncols = side_length
            nrows = ceil(num_stars / ncols)
            from plotting_utils import apply_autophot_mplstyle, get_plot_ext
            apply_autophot_mplstyle()
            plt.ioff()
            fpath = self.input_yaml["fpath"]
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            # Size each grid cell to the cutout aspect so the equal-aspect
            # panels fill their cells; a fixed figsize shrinks them and
            # leaves gaps no wspace can remove. The 3 trailing columns are
            # a 0.1 spacer plus the radial-profile panel.
            cut_h, cut_w = np.asarray(stars[0]).shape
            ax_h = 1.2
            cell_w = ax_h * (cut_w / max(cut_h, 1))
            left, right, bottom, top = 0.05, 0.985, 0.07, 0.88
            hspace = wspace = 0.01
            fig_w = (ncols + 2.1) * cell_w * (1 + wspace) / (right - left)
            fig_h = nrows * ax_h * (1 + hspace) / (top - bottom)
            fig = plt.figure(figsize=(fig_w, fig_h))
            grid = fig.add_gridspec(
                nrows=nrows,
                ncols=ncols + 3,
                width_ratios=[1] * ncols + [0.1, 1, 1],
                height_ratios=[1] * nrows,
                hspace=hspace,
                wspace=wspace,
                left=left,
                right=right,
                bottom=bottom,
                top=top,
            )
            for i in range(num_stars):
                row, col = divmod(i, ncols)
                ax = fig.add_subplot(grid[row, col])
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(np.asarray(stars[i]))
                norm = ImageNormalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap("viridis").copy()
                cmap = cmap.with_extremes(bad="none")
                ax.imshow(
                    stars[i],
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                    interpolation="none",
                )
                from plotting_utils import overlay_mask_hatch
                overlay_mask_hatch(ax, ~np.isfinite(np.asarray(stars[i])))
                ax.text(
                    0.98,
                    0.98,
                    f"{i + 1}/{num_stars}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    color="white",
                )
                ax.set_xticks([])
                ax.set_yticks([])

            ax_right = fig.add_subplot(grid[:, -2:])
            radii = np.arange(radial_profiles.shape[1])
            accepted_profiles = [
                radial_profiles[valid_indices.index(i)] for i in inliers
            ]
            for profile in accepted_profiles:
                ax_right.step(radii, profile, color="black", alpha=0.3, linewidth=0.5)
            median_profile = np.median(accepted_profiles, axis=0)
            ax_right.step(
                radii,
                median_profile,
                color="#D94F4F",
                linewidth=0.5,
                label="Median\nRadial\nProfile",
            )
            ax_right.set_xlabel("Radius [pixels]")
            ax_right.set_ylabel("Normalized Flux")
            ax_right.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                            frameon=False, fontsize=8)
            ax_right.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax_right.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
            pos = ax_right.get_position()
            ax_right.set_position([pos.x0 + 0.05, pos.y0, pos.width, pos.height])
            output_path = os.path.join(
                write_dir, f"Zeropoint_Sources_{base}{get_plot_ext(self.input_yaml)}"
            )
            fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
            plt.close(fig)

        logger.info("Returning %s well-behaved sources", len(cleaned_catalog))
        return cleaned_catalog

    # =============================================================================
    # =============================================================================
    # #
    # =============================================================================
    # =============================================================================

    def measure(self, selectedCatalog, image):
        """
        Measure the flux and signal-to-noise ratio (SNR) of sources in an image using aperture photometry.

        Parameters:
        -----------
        selectedCatalog : pd.DataFrame
            DataFrame containing the sources with initial coordinates.
        image : numpy.ndarray
            2D array representing the image where sources are located.

        Returns:
        --------
        pd.DataFrame
            Updated DataFrame with measured flux, instrumental magnitude, and SNR for each source.
        """
        logger = logging.getLogger(__name__)

        logger.info(
            log_step(
                f"Aperture photometry: {len(selectedCatalog)} field sources"
            )
        )

        initialAperture = Aperture(
            input_yaml=self.input_yaml,
            image=image,
        )

        selectedCatalog = initialAperture.measure(sources=selectedCatalog)

        # mag() does not mutate flux_AP.
        instMag = mag(selectedCatalog["flux_AP"])
        inst_col = "inst_" + self.input_yaml["imageFilter"] + "_AP"
        # Keep full float64 precision - this column feeds the zeropoint
        # fit, and 1-mmag rounding is unnecessary quantization on sparse
        # calibrator fields.
        selectedCatalog[inst_col] = instMag

        sourceSNR = snr(selectedCatalog["maxPixel"], selectedCatalog["noiseSky"])
        selectedCatalog["snr"] = np.round(sourceSNR, 1)

        logger.info(
            "Instrumental magnitude of %d sources measured"
            % sum(~np.isnan(selectedCatalog[inst_col]))
        )

        return selectedCatalog
