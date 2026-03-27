"""
Lightweight LPI-style local background prediction.

This implements a target-only, small-stamp infill of the structured background
behind a point source using local shifted samples and ridge regression in pixel
space (in the spirit of Saydjari & Finkbeiner 2022, ApJ 933:155).

Status: in development / experimental.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _extract_stamp_reflect(image: np.ndarray, x0: float, y0: float, half: int) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    ny, nx = img.shape
    _ = (ny, nx)  # keep locals referenced for readability / debug
    xi = int(np.rint(float(x0)))
    yi = int(np.rint(float(y0)))

    pad = int(max(0, half + 2))
    work = np.pad(img, pad_width=pad, mode="reflect")
    xi2 = xi + pad
    yi2 = yi + pad
    return work[yi2 - half : yi2 + half + 1, xi2 - half : xi2 + half + 1]


def predict_background_under_source(
    image: np.ndarray,
    *,
    x0: float,
    y0: float,
    inner_radius_px: float,
    outer_radius_px: float,
    stamp_half_size_px: int,
    n_samples: int = 250,
    sample_window_px: int = 30,
    min_shift_px: float = 0.0,
    ridge_lambda: float = 1e-2,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict structured background under a source.

    Returns
    -------
    bg_pred : np.ndarray
        2D stamp background prediction (same shape as stamp) with meaningful
        predictions only inside inner_radius (hidden region); other pixels are 0.
    bg_sigma : np.ndarray
        2D stamp per-pixel 1-sigma prediction scatter inside inner_radius; others 0.
    """
    half = int(stamp_half_size_px)
    if half < 5:
        half = 5
    inner = float(inner_radius_px)
    outer = float(outer_radius_px)
    if not np.isfinite(inner) or inner <= 0:
        raise ValueError("inner_radius_px must be finite and >0")
    if not np.isfinite(outer) or outer <= inner + 1:
        raise ValueError("outer_radius_px must be finite and > inner_radius_px")

    stamp0 = _extract_stamp_reflect(image, x0, y0, half=half)
    h, w = stamp0.shape
    cy, cx = half, half
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    hidden = rr <= inner
    good = (rr > inner) & (rr <= outer)

    if not np.any(hidden) or not np.any(good):
        raise ValueError("Invalid hidden/good pixel partition for given radii.")

    g_idx = np.flatnonzero(good)
    h_idx = np.flatnonzero(hidden)

    rng = np.random.default_rng(rng_seed)
    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []

    # Draw random shifted samples; reflect padding is handled by extractor.
    # Many trials may be rejected (min_shift, NaNs, etc.), so allow a much
    # larger number of attempts than n_samples to avoid frequent failures.
    max_trials = int(max(200, 10 * int(max(1, n_samples))))
    for _ in range(max_trials):
        dx = int(rng.integers(-sample_window_px, sample_window_px + 1))
        dy = int(rng.integers(-sample_window_px, sample_window_px + 1))
        if float(min_shift_px) > 0 and (dx * dx + dy * dy) < float(min_shift_px) ** 2:
            continue
        st = _extract_stamp_reflect(image, x0 + dx, y0 + dy, half=half)
        xv = st.ravel()[g_idx]
        yv = st.ravel()[h_idx]
        if not (np.all(np.isfinite(xv)) and np.all(np.isfinite(yv))):
            continue
        X_rows.append(xv.astype(float, copy=False))
        Y_rows.append(yv.astype(float, copy=False))
        if len(X_rows) >= n_samples:
            break

    if len(X_rows) < max(20, int(0.2 * n_samples)):
        raise RuntimeError(
            "Insufficient finite samples for LPI background prediction "
            f"(kept={len(X_rows)}/{n_samples}; trials={max_trials})."
        )

    X = np.asarray(X_rows, dtype=float)
    Y = np.asarray(Y_rows, dtype=float)

    # Ridge regression in pixel space: solve for B in X B ~ Y
    # B = (X^T X + lam I)^-1 X^T Y
    lam = float(ridge_lambda)
    XtX = X.T @ X
    XtX.flat[:: XtX.shape[0] + 1] += lam
    XtY = X.T @ Y
    B = np.linalg.solve(XtX, XtY)

    x0v = stamp0.ravel()[g_idx].astype(float, copy=False)
    y_pred = x0v @ B
    resid = Y - (X @ B)
    sigma = np.nanstd(resid, axis=0)

    bg_pred = np.zeros_like(stamp0, dtype=float)
    bg_sig = np.zeros_like(stamp0, dtype=float)
    bg_pred.ravel()[h_idx] = y_pred
    bg_sig.ravel()[h_idx] = sigma
    return bg_pred, bg_sig


def save_lpi_diagnostic_plot(
    *,
    image: np.ndarray,
    x0: float,
    y0: float,
    stamp_half_size_px: int,
    inner_radius_px: float,
    outer_radius_px: float,
    bg_pred: np.ndarray,
    bg_sig: np.ndarray,
    save_path: str,
    title: str | None = None,
) -> None:
    """
    Save a diagnostic plot for the LPI-style background prediction.
    This intentionally does not log "saved plot" messages (to keep logs clean).
    """
    import matplotlib.pyplot as plt

    half = int(stamp_half_size_px)
    stamp0 = _extract_stamp_reflect(image, x0, y0, half=half)
    corrected = stamp0 - bg_pred

    h, w = stamp0.shape
    cy, cx = half, half
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    hidden = rr <= float(inner_radius_px)
    good = (rr > float(inner_radius_px)) & (rr <= float(outer_radius_px))

    def _robust_limits(arr):
        v = np.asarray(arr, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (0.0, 1.0)
        lo, hi = np.nanpercentile(v, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.nanmin(v))
            hi = float(np.nanmax(v))
        return (float(lo), float(hi))

    v0 = _robust_limits(stamp0)
    v1 = _robust_limits(corrected)
    vp = _robust_limits(bg_pred[hidden])
    vs = _robust_limits(bg_sig[hidden])

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    ax = axes.ravel()

    im0 = ax[0].imshow(stamp0, origin="lower", cmap="bone", vmin=v0[0], vmax=v0[1])
    ax[0].set_title("Stamp (data)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03)

    m = np.zeros_like(stamp0, dtype=float)
    m[good] = 1.0
    m[hidden] = 2.0
    im1 = ax[1].imshow(m, origin="lower", cmap="viridis", vmin=0, vmax=2)
    ax[1].set_title("Mask (good=1, hidden=2)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.03)

    im2 = ax[2].imshow(bg_pred, origin="lower", cmap="bone", vmin=vp[0], vmax=vp[1])
    ax[2].set_title("Predicted background (hidden)")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03)

    im3 = ax[3].imshow(corrected, origin="lower", cmap="bone", vmin=v1[0], vmax=v1[1])
    ax[3].set_title("Corrected stamp (data - pred)")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.03)

    im4 = ax[4].imshow(bg_sig, origin="lower", cmap="magma", vmin=vs[0], vmax=vs[1])
    ax[4].set_title("Predicted sigma (hidden)")
    fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.03)

    # Simple radial profile view for intuition.
    r_flat = rr.ravel()
    s_flat = stamp0.ravel()
    c_flat = corrected.ravel()
    ok = np.isfinite(r_flat) & np.isfinite(s_flat) & np.isfinite(c_flat)
    ax[5].plot(r_flat[ok], s_flat[ok], ".", ms=1.2, alpha=0.25, label="data")
    ax[5].plot(r_flat[ok], c_flat[ok], ".", ms=1.2, alpha=0.25, label="corrected")
    ax[5].axvline(float(inner_radius_px), color="0.3", ls="--", lw=0.8, label="inner")
    ax[5].axvline(float(outer_radius_px), color="0.5", ls=":", lw=0.8, label="outer")
    ax[5].set_xlabel("r [px]")
    ax[5].set_ylabel("value")
    ax[5].set_title("Radial scatter")
    ax[5].legend(frameon=False, fontsize=8)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    if title:
        fig.suptitle(str(title), fontsize=11)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

