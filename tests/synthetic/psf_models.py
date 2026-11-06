"""Synthetic PSF models for AutoPhOT tests.

Provides analytically defined PSF models with known parameters so that
fitted results can be compared against ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_func


def gaussian_psf_2d(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    flux: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate a 2-D circular Gaussian PSF on a pixel grid.

    Parameters
    ----------
    x, y : np.ndarray
        Pixel coordinate meshes (from ``np.meshgrid``).
    x0, y0 : float
        PSF centre in pixel coordinates.
    flux : float
        Total integrated flux (counts).
    sigma : float
        Gaussian sigma in pixels.

    Returns
    -------
    np.ndarray
        PSF image with the same shape as *x*.
    """
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return flux / (2.0 * np.pi * sigma**2) * np.exp(-r2 / (2.0 * sigma**2))


def moffat_psf_2d(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    flux: float,
    alpha: float,
    beta: float = 4.765,
) -> np.ndarray:
    """Evaluate a 2-D circular Moffat PSF.

    Parameters
    ----------
    alpha : float
        Moffat scale radius (pixels).
    beta : float
        Moffat power-law index.
    """
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    # Normalisation: flux = A * pi * alpha^2 / (beta - 1)
    norm = (beta - 1.0) / (np.pi * alpha**2)
    return flux * norm * (1.0 + r2 / alpha**2) ** (-beta)


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert Gaussian FWHM to sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma: float) -> float:
    """Convert Gaussian sigma to FWHM."""
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


def fwhm_to_alpha(fwhm: float, beta: float = 4.765) -> float:
    """Convert Moffat FWHM to alpha."""
    return fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def moffat_fwhm(alpha: float, beta: float = 4.765) -> float:
    """Convert Moffat alpha to FWHM."""
    return 2.0 * alpha * np.sqrt(2.0 ** (1.0 / beta) - 1.0)


def encircled_energy_gaussian(radius: float, sigma: float) -> float:
    """Encircled energy fraction for a 2-D Gaussian at *radius*."""
    return 1.0 - np.exp(-radius**2 / (2.0 * sigma**2))


def encircled_energy_moffat(radius: float, alpha: float, beta: float = 4.765) -> float:
    """Encircled energy fraction for a 2-D Moffat at *radius*."""
    return 1.0 - (1.0 + radius**2 / alpha**2) ** (1.0 - beta)
