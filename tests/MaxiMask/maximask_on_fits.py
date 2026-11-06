"""Run MaxiMask on a real FITS image and make diagnostic plots.

Same exercise as ``run_maximask.py`` but for arbitrary input frames:
loads the TF model once, calls ``process_hdu`` directly so results stay
in memory (the CLI-level ``process_all`` would write a multi-hundred-MB
``<image>.mask.fits`` next to the input file), applies the shipped
per-class thresholds and writes plots into an output directory.

Usage::

    python tests/MaxiMask/maximask_on_fits.py /path/to/image.fits \
        [--hdu 0] [--outdir DIR]
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
from scipy import ndimage

try:
    from . import run_maximask as mm
except ImportError:  # direct script execution
    import run_maximask as mm

REAL_DIR = mm.OUTPUT_DIR / "real"

# Classes drawn in the single-image overlay; BG/BBG are scene content,
# P is never triggered by these data.
OVERLAY_CLASSES = [
    "CR",
    "HCL",
    "DCL",
    "HP",
    "DP",
    "TRL",
    "FR",
    "NEB",
    "SAT",
    "SP",
    "OV",
]


def run_inference_on_hdu(fits_path: str, hdu: int, batch_size: int) -> np.ndarray:
    """Return the (14, ny, nx) prior-modified probability cube."""
    import tensorflow as tf
    from maximask_and_maxitrack import utils
    from maximask_and_maxitrack.maximask.maximask import MaxiMask_inference

    mm_dir = Path(utils.__file__).resolve().parent / "maximask"
    class_flags = utils.read_config_file(
        str(mm_dir / "config" / "classes.flags"), len(mm.CLASS_ABBR), log
    )
    priors = utils.read_config_file(
        str(mm_dir / "config" / "classes.priors"),
        len(mm.CLASS_ABBR),
        log,
        to_float=True,
    )
    mm_inf = MaxiMask_inference(
        fits_path,
        str(mm_dir / "model"),
        class_flags,
        priors,
        None,
        False,
        batch_size,
    )
    tf_model = tf.saved_model.load(str(mm_dir / "model"))
    probs, t = mm_inf.process_hdu(fits_path, hdu, "process", tf_model)
    log.getLogger(__name__).info(
        f"MaxiMask inference on {fits_path}[{hdu}] took {t:.1f}s"
    )
    return probs


def plot_real_overlay(image, masks, path: Path) -> None:
    """Image with thresholded detections overlaid, plus flagged counts."""
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.imshow(image, norm=mm._norm(image), origin="lower", cmap="gray")
    handles = []
    for abbr in OVERLAY_CLASSES:
        det = masks[mm.CLASS_ABBR.index(abbr)]
        if not det.any():
            continue
        mm._overlay(ax, det, mm.CLASS_COLORS[abbr])
        handles.append(
            matplotlib.patches.Patch(
                facecolor=mm.CLASS_COLORS[abbr],
                alpha=0.55,
                label=f"{abbr} ({det.sum()} px)",
            )
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_real_probmaps(image, probs, masks, thresh, path: Path) -> None:
    """4x4 grid: input, 14 class maps, flagged-class count map."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    axes[0].imshow(image, norm=mm._norm(image), origin="lower", cmap="gray")
    axes[0].set_title("input image", fontsize=9)
    n_flag = np.zeros(image.shape, dtype=int)
    for i, (abbr, name) in enumerate(zip(mm.CLASS_ABBR, mm.CLASS_NAMES)):
        ax = axes[i + 1]
        ax.imshow(probs[i], origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_title(
            f"{abbr}: {name}\nthresh={thresh[i]:.2f}, flagged={masks[i].sum()} px",
            fontsize=8,
        )
        if abbr != "BG":
            n_flag += masks[i].astype(int)
    im = axes[15].imshow(n_flag, origin="lower", cmap="viridis", vmin=0)
    axes[15].set_title("flagged-class count per px", fontsize=9)
    fig.colorbar(im, ax=axes[15], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _largest_blob_window(mask, pad=80):
    """Bounding window (y0,y1,x0,x1) around the largest labelled blob."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    blob = lab == int(np.argmax(sizes)) + 1
    ys, xs = np.where(blob)
    ny, nx = mask.shape
    return (
        max(0, ys.min() - pad),
        min(ny, ys.max() + pad),
        max(0, xs.min() - pad),
        min(nx, xs.max() + pad),
    )


def plot_real_zooms(image, masks, path: Path) -> None:
    """Zoom on the largest detected blob of the top flagged classes."""
    flagged = [
        (abbr, int(masks[mm.CLASS_ABBR.index(abbr)].sum())) for abbr in OVERLAY_CLASSES
    ]
    flagged = [(a, n) for a, n in flagged if n > 0]
    flagged.sort(key=lambda t: -t[1])
    panels = []
    for abbr, _ in flagged[:4]:
        win = _largest_blob_window(masks[mm.CLASS_ABBR.index(abbr)])
        if win:
            panels.append((abbr, win))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    axes = np.atleast_1d(axes)
    for ax, (abbr, (y0, y1, x0, x1)) in zip(axes, panels):
        cut = image[y0:y1, x0:x1]
        ax.imshow(
            cut,
            norm=mm._norm(cut),
            origin="lower",
            cmap="gray",
            extent=[x0, x1, y0, y1],
        )
        for other in OVERLAY_CLASSES:
            det = masks[mm.CLASS_ABBR.index(other)][y0:y1, x0:x1]
            if det.any():
                mm._overlay(
                    ax, det, mm.CLASS_COLORS[other], alpha=0.5, extent=[x0, x1, y0, y1]
                )
        ax.set_title(f"largest {abbr} blob", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits_path")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=mm.BATCH_SIZE)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)

    outdir = (
        Path(args.outdir) if args.outdir else (REAL_DIR / Path(args.fits_path).stem)
    )
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.fits_path, memmap=True) as hdul:
        image = np.asarray(hdul[args.hdu].data, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"HDU {args.hdu} is not a 2-D image: {image.shape}")

    probs = run_inference_on_hdu(args.fits_path, args.hdu, args.batch_size)
    thresh = mm.load_thresholds()
    masks = probs >= thresh[:, None, None]

    np.savez_compressed(outdir / "probmaps.npz", probs=probs)
    stem = Path(args.fits_path).stem
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.imshow(image, norm=mm._norm(image), origin="lower", cmap="gray")
    ax.set_title(stem)
    fig.savefig(outdir / "input.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    plot_real_overlay(image, masks, outdir / "overlay.png")
    plot_real_probmaps(image, probs, masks, thresh, outdir / "probmaps.png")
    plot_real_zooms(image, masks, outdir / "zooms.png")

    for abbr in mm.CLASS_ABBR:
        n = int(masks[mm.CLASS_ABBR.index(abbr)].sum())
        print(f"{abbr:<4} flagged {n:>9} px ({100.0 * n / image.size:.3f}%)")
    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    main()
