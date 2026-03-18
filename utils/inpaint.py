"""
Inpainting utilities for repairing small masked regions (e.g. broken/saturated cores).

This is intended as a pragmatic cosmetic/robustness tool for template subtraction:
it does NOT recover lost information in saturated cores, but it can reduce strong
discontinuities that otherwise propagate into convolution / subtraction artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class InpaintConfig:
    enabled: bool = False
    method: str = "biharmonic"  # currently only biharmonic is implemented
    saturate_frac: float = 0.90  # mask pixels >= saturate_frac * SATURATE
    dilate_radius: int = 6  # pixels; dilate the core mask before inpainting
    max_mask_fraction: float = 0.01  # safety: skip if mask too large (fraction of image)


def build_saturation_core_mask(
    image: np.ndarray,
    saturate: float,
    *,
    saturate_frac: float = 0.90,
    dilate_radius: int = 0,
) -> np.ndarray:
    """
    Build a boolean mask for saturated / broken cores in an image.

    Parameters
    ----------
    image : np.ndarray
        Image array (2D).
    saturate : float
        Saturation level in the same units as `image` (ADU in the pipeline).
    saturate_frac : float
        Fraction of `saturate` above which pixels are considered saturated.
    dilate_radius : int
        If >0, dilate the mask by this many pixels (disk structuring element).
    """
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("build_saturation_core_mask expects a 2D image.")
    if not np.isfinite(saturate) or saturate <= 0:
        return np.zeros(img.shape, dtype=bool)

    thr = float(saturate_frac) * float(saturate)
    core = np.isfinite(img) & (img >= thr)
    if dilate_radius and dilate_radius > 0 and np.any(core):
        try:
            from skimage.morphology import binary_dilation, disk

            core = binary_dilation(core, disk(int(dilate_radius)))
        except Exception:
            # Fall back to no dilation if skimage morphology isn't available.
            pass
    return np.asarray(core, dtype=bool)


def inpaint_image(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "biharmonic",
) -> np.ndarray:
    """
    Inpaint image values where mask is True.

    Notes
    -----
    - This function expects small holes. It is not suitable for large masked regions.
    - Masked pixels are replaced with interpolated values; unmasked pixels are unchanged.
    """
    img = np.asarray(image, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if img.shape != m.shape:
        raise ValueError("inpaint_image: image and mask must have the same shape.")
    if not np.any(m):
        return img

    method = str(method).strip().lower()
    if method != "biharmonic":
        raise ValueError(f"Unsupported inpaint method '{method}'.")

    from skimage.restoration import inpaint as sk_inpaint

    # biharmonic expects NaNs are allowed but we give the mask explicitly.
    # channel_axis=None for 2D images.
    return sk_inpaint.inpaint_biharmonic(img, m, channel_axis=None)


def inpaint_saturated_cores(
    image: np.ndarray,
    *,
    saturate: float,
    cfg: InpaintConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inpaint saturated/broken cores using a saturation-derived mask.

    Returns
    -------
    (image_inpainted, mask_used)
    """
    img = np.asarray(image, dtype=float)
    if not cfg.enabled:
        return img, np.zeros(img.shape, dtype=bool)

    mask = build_saturation_core_mask(
        img,
        saturate,
        saturate_frac=cfg.saturate_frac,
        dilate_radius=cfg.dilate_radius,
    )
    if not np.any(mask):
        return img, mask

    frac = float(np.mean(mask))
    if frac > float(cfg.max_mask_fraction):
        # Too much to inpaint safely; return unchanged.
        return img, mask

    out = inpaint_image(img, mask, method=cfg.method)
    return out, mask

