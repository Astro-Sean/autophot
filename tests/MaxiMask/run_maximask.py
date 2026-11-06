"""Run MaxiMask on the synthetic contaminated frame and make plots.

Reproduces the exercise in Paillassa, Bertin & Bouy (2020,
arXiv:1907.08298): a CCD-like image with injected contaminants is fed
through the MaxiMask network, and the per-class probability maps and
thresholded masks are compared with the known truth.

Running this file directly (``python run_maximask.py``) regenerates the
inputs, runs the TensorFlow inference and writes all artifacts into
``output/`` next to this file. The pytest wrapper in ``test_maximask.py``
reuses the same pipeline.
"""

from __future__ import annotations

import argparse
import logging as log
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from . import synthetic_image as syn
except ImportError:  # direct script execution
    import synthetic_image as syn

SEED = 7
BATCH_SIZE = 8

MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "output"
INPUT_FITS = OUTPUT_DIR / "maximask_test_input.fits"
TRUTH_FITS = OUTPUT_DIR / "maximask_test_truth.fits"
MASK_FITS = OUTPUT_DIR / "maximask_test_input.mask.fits"

CLASS_ABBR = [
    "CR",
    "HCL",
    "DCL",
    "HP",
    "DP",
    "P",
    "TRL",
    "FR",
    "NEB",
    "SAT",
    "SP",
    "OV",
    "BBG",
    "BG",
]
CLASS_NAMES = [
    "Cosmic Rays",
    "Hot Columns/Lines",
    "Dead Columns/Lines/Clusters",
    "Hot Pixels",
    "Dead Pixels",
    "Persistence",
    "TRaiLs",
    "residual FRinging",
    "NEBulosities",
    "SATurated pixels",
    "diffraction SPikes",
    "Overscan",
    "Bright BackGround",
    "BackGround",
]

# Distinct colours for overlaying the contaminant classes. BG is plain
# sky and is left out of the overlays.
CLASS_COLORS = {
    "CR": "#ff2d2d",
    "HCL": "#39ff14",
    "DCL": "#0f9b4a",
    "HP": "#ff00ff",
    "DP": "#a55eea",
    "P": "#95a5a6",
    "TRL": "#ff9500",
    "FR": "#00bcd4",
    "NEB": "#ff6b9d",
    "SAT": "#ffee00",
    "SP": "#00ffff",
    "OV": "#bcaa99",
    "BBG": "#3d5afe",
}


def write_inputs(seed: int = SEED, force: bool = False) -> dict:
    """Build the synthetic frame and persist image + truth to FITS."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = syn.make_contaminated_image(seed=seed)
    if force or not INPUT_FITS.exists():
        fits.writeto(INPUT_FITS, data["image"], data["header"], overwrite=True)
    if force or not TRUTH_FITS.exists():
        stack = np.stack([data["truth"][name] for name in syn.TRUTH_CLASSES]).astype(
            np.uint8
        )
        hdr = fits.Header()
        for k, name in enumerate(syn.TRUTH_CLASSES):
            hdr[f"MASK{k:02d}"] = name
        fits.writeto(TRUTH_FITS, stack, hdr, overwrite=True)
    return data


def run_inference(force: bool = False) -> np.ndarray:
    """Run MaxiMask and return the (14, ny, nx) prior-modified
    probability cube.

    Thresholding is left off so the plots can show continuous
    probability maps; the shipped per-class thresholds are applied
    afterwards in numpy.
    """
    from maximask_and_maxitrack import utils
    from maximask_and_maxitrack.maximask.maximask import MaxiMask_inference

    mm_dir = Path(utils.__file__).resolve().parent / "maximask"
    net_dir = mm_dir / "model"
    config_dir = mm_dir / "config"
    nb_classes = len(CLASS_ABBR)

    class_flags = utils.read_config_file(
        str(config_dir / "classes.flags"), nb_classes, log
    )
    priors = utils.read_config_file(
        str(config_dir / "classes.priors"), nb_classes, log, to_float=True
    )

    if force or not MASK_FITS.exists():
        mm_inf = MaxiMask_inference(
            str(INPUT_FITS), str(net_dir), class_flags, priors, None, False, BATCH_SIZE
        )
        mm_inf.process_all()

    with fits.open(MASK_FITS) as hdul:
        probs = hdul[0].data.astype(np.float32)
    return probs


def load_thresholds() -> np.ndarray:
    """Per-class decision thresholds shipped with MaxiMask."""
    from maximask_and_maxitrack import utils

    mm_dir = Path(utils.__file__).resolve().parent / "maximask"
    return utils.read_config_file(
        str(mm_dir / "config" / "classes.thresh"), len(CLASS_ABBR), log, to_float=True
    )


def _norm(image):
    return ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch())


def _overlay(ax, mask, color, alpha=0.55, extent=None):
    """Paint *mask* pixels in a flat translucent colour."""
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgb = to_rgb(color)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    ax.imshow(rgba, interpolation="none", origin="lower", extent=extent)


def plot_input(data, outdir: Path) -> Path:
    """Input frame with ground-truth contaminant contours."""
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.imshow(data["image"], norm=_norm(data["image"]), origin="lower", cmap="gray")
    handles = []
    for name in syn.TRUTH_CLASSES:
        m = data["truth"][name]
        if not m.any():
            continue
        ax.contour(
            m, levels=[0.5], colors=[CLASS_COLORS[name]], linewidths=0.6, alpha=0.9
        )
        handles.append(
            Line2D([0], [0], color=CLASS_COLORS[name], lw=1.5, label=f"{name} truth")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_title("Synthetic MaxiMask test frame - injected truth (arXiv:1907.08298)")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    path = outdir / "maximask_input.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overlay(data, masks, outdir: Path) -> Path:
    """Input frame with thresholded MaxiMask detections overlaid."""
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.imshow(data["image"], norm=_norm(data["image"]), origin="lower", cmap="gray")
    handles = []
    for i, abbr in enumerate(CLASS_ABBR):
        if abbr == "BG" or not masks[i].any():
            continue
        _overlay(ax, masks[i], CLASS_COLORS[abbr])
        handles.append(
            Patch(
                facecolor=CLASS_COLORS[abbr],
                alpha=0.55,
                label=f"{abbr} ({masks[i].sum()} px)",
            )
        )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_title("MaxiMask detections (prior-modified, shipped thresholds)")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    path = outdir / "maximask_overlay.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_probmaps(data, probs, masks, thresh, outdir: Path) -> Path:
    """Grid of all 14 class probability maps plus input and flag count."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.ravel()

    ax = axes[0]
    ax.imshow(data["image"], norm=_norm(data["image"]), origin="lower", cmap="gray")
    ax.set_title("input image", fontsize=9)

    n_flag = np.zeros(syn.IMAGE_SHAPE, dtype=int)
    for i, (abbr, name) in enumerate(zip(CLASS_ABBR, CLASS_NAMES)):
        ax = axes[i + 1]
        ax.imshow(probs[i], origin="lower", cmap="magma", vmin=0, vmax=1)
        n = int(masks[i].sum())
        ax.set_title(
            f"{abbr}: {name}\nthresh={thresh[i]:.2f}, flagged={n} px", fontsize=8
        )
        if abbr != "BG":
            n_flag += masks[i].astype(int)

    ax = axes[15]
    im = ax.imshow(n_flag, origin="lower", cmap="viridis", vmin=0)
    ax.set_title("flagged-class count per px", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("MaxiMask per-class probability maps (arXiv:1907.08298)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = outdir / "maximask_probmaps.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _zoom_windows(meta):
    sat_cy, sat_cx = meta["sat_center"]
    hot_col, dead_col = meta["defect_cols"]
    cr_y, cr_x = meta["cr_points"][0]
    neb_cy, neb_cx = meta["neb_center"]
    fr_y0, _, fr_x0, _ = meta["fringe_box"]
    ov_x = meta["overscan_x0"]
    return [
        (
            "saturated star + spikes",
            (sat_cy - 70, sat_cy + 70, sat_cx - 70, sat_cx + 70),
            ["SAT", "SP"],
        ),
        ("satellite trail", (480, 640, 650, 900), ["TRL"]),
        ("cosmic ray", (cr_y - 50, cr_y + 50, cr_x - 50, cr_x + 50), ["CR"]),
        ("hot column", (400, 650, hot_col - 30, hot_col + 30), ["HCL"]),
        ("dead column", (400, 650, dead_col - 30, dead_col + 30), ["DCL"]),
        (
            "nebulosity",
            (neb_cy - 120, neb_cy + 120, neb_cx - 130, neb_cx + 130),
            ["NEB", "BBG"],
        ),
        ("fringing region", (fr_y0, fr_y0 + 140, fr_x0, fr_x0 + 200), ["FR"]),
        ("overscan edge", (400, 650, ov_x - 90, syn.IMAGE_SHAPE[1] - 2), ["OV"]),
    ]


def plot_zooms(data, masks, outdir: Path) -> Path:
    """Zoomed panels comparing detections (fill) to truth (contour)."""
    windows = _zoom_windows(data["meta"])
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    idx = {a: i for i, a in enumerate(CLASS_ABBR)}
    for ax, (title, (y0, y1, x0, x1), classes) in zip(axes.ravel(), windows):
        y0, y1 = int(y0), int(y1)
        x0, x1 = int(x0), int(x1)
        cut = data["image"][y0:y1, x0:x1]
        ax.imshow(
            cut, norm=_norm(cut), origin="lower", cmap="gray", extent=[x0, x1, y0, y1]
        )
        handles = []
        for abbr in classes:
            det = masks[idx[abbr]][y0:y1, x0:x1]
            tru = data["truth"].get(abbr)
            if det.any():
                _overlay(
                    ax, det, CLASS_COLORS[abbr], alpha=0.5, extent=[x0, x1, y0, y1]
                )
            if tru is not None and tru[y0:y1, x0:x1].any():
                ax.contour(
                    tru[y0:y1, x0:x1],
                    levels=[0.5],
                    colors=["white"],
                    linewidths=0.8,
                    extent=[x0, x1, y0, y1],
                )
            handles.append(
                Patch(facecolor=CLASS_COLORS[abbr], alpha=0.55, label=f"{abbr} det.")
            )
        handles.append(Line2D([0], [0], color="white", lw=1, label="truth"))
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)
        ax.set_title(title, fontsize=10)
    fig.suptitle("MaxiMask detections vs injected truth (white contours)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    path = outdir / "maximask_zooms.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def summarise(data, masks, clean) -> str:
    """Per-class truth/flagged/recall table as text."""
    idx = {a: i for i, a in enumerate(CLASS_ABBR)}
    lines = [
        f"{'class':<4} {'truth_px':>9} {'flagged_px':>11} {'recall':>7} {'clean_fp':>9}",
    ]
    for abbr in CLASS_ABBR:
        det = masks[idx[abbr]]
        tru = data["truth"].get(abbr)
        n_det = int(det.sum())
        if tru is not None and tru.any():
            n_tru = int(tru.sum())
            recall = float((det & tru).sum()) / n_tru
            rec = f"{recall:7.3f}"
        else:
            n_tru = 0
            rec = "      -"
        fp = float((det & clean).sum()) / max(1, int(clean.sum()))
        lines.append(f"{abbr:<4} {n_tru:>9} {n_det:>11} {rec} {fp:>9.4f}")
    return "\n".join(lines)


def run_pipeline(seed: int = SEED, force: bool = False, figures: bool = True) -> dict:
    """Generate inputs, run inference, produce plots. Returns all arrays."""
    data = write_inputs(seed=seed, force=force)
    probs = run_inference(force=force)
    thresh = load_thresholds()
    masks = probs >= thresh[:, None, None]
    clean = data["clean"]

    paths = {}
    if figures:
        paths["input"] = plot_input(data, OUTPUT_DIR)
        paths["overlay"] = plot_overlay(data, masks, OUTPUT_DIR)
        paths["probmaps"] = plot_probmaps(data, probs, masks, thresh, OUTPUT_DIR)
        paths["zooms"] = plot_zooms(data, masks, OUTPUT_DIR)

    summary = summarise(data, masks, clean)
    (OUTPUT_DIR / "summary.txt").write_text(summary + "\n")
    print(summary)
    return {
        "data": data,
        "probs": probs,
        "masks": masks,
        "thresh": thresh,
        "paths": paths,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate inputs and rerun inference even if outputs exist",
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    run_pipeline(seed=args.seed, force=args.force, figures=not args.no_figures)


if __name__ == "__main__":
    main()
