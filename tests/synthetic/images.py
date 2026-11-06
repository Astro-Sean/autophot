"""Synthetic image generators for AutoPhOT tests.

Creates images with known-truth sources, realistic noise, backgrounds,
and masks for validating the photometry pipeline.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .psf_models import gaussian_psf_2d, moffat_psf_2d, fwhm_to_sigma


def make_synthetic_image(
    catalog,
    image_shape: tuple[int, int] = (512, 512),
    fwhm: float = 4.0,
    background: float = 100.0,
    background_rms: float = 10.0,
    gain: float = 1.0,
    read_noise: float = 5.0,
    psf_model: str = "gaussian",
    moffat_beta: float = 4.765,
    add_poisson: bool = True,
    add_read_noise: bool = True,
    add_background_gradient: float = 0.0,
    mask_fraction: float = 0.0,
    seed: int | None = 42,
):
    """Create a synthetic image with known-truth sources.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with columns x_pix, y_pix, flux_true.
    image_shape : (ny, nx)
        Image dimensions.
    fwhm : float
        PSF FWHM in pixels.
    background : float
        Median background level (counts).
    background_rms : float
        Background RMS (counts) for the noise model.
    gain : float
        Detector gain (e-/ADU).
    read_noise : float
        Read noise (e-).
    psf_model : str
        "gaussian" or "moffat".
    moffat_beta : float
        Moffat beta (if psf_model="moffat").
    add_poisson : bool
        If True, add Poisson noise.
    add_read_noise : bool
        If True, add Gaussian read noise.
    add_background_gradient : float
        If > 0, add a linear gradient with this amplitude (counts/pixel).
    mask_fraction : float
        Fraction of pixels to mask (set to NaN).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        image : np.ndarray (ny, nx) - the synthetic image
        truth : pd.DataFrame - catalog with added columns
        header : astropy.io.fits.Header
        background_rms_map : np.ndarray - per-pixel RMS
        mask : np.ndarray - bad pixel mask (True = bad)
        noise_free : np.ndarray - image before noise addition
    """
    rng = np.random.default_rng(seed)
    ny, nx = image_shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    bkg = np.full((ny, nx), float(background), dtype=float)
    if add_background_gradient > 0:
        # Linear gradient from left to right
        gradient = add_background_gradient * np.linspace(-1, 1, nx)[np.newaxis, :]
        bkg += gradient

    noise_free = bkg.copy()
    sigma = fwhm_to_sigma(fwhm)

    for _, row in catalog.iterrows():
        x0 = float(row["x_pix"])
        y0 = float(row["y_pix"])
        flux = float(row["flux_true"])
        if psf_model == "gaussian":
            psf = gaussian_psf_2d(xx, yy, x0, y0, flux, sigma)
        elif psf_model == "moffat":
            from .psf_models import fwhm_to_alpha

            alpha = fwhm_to_alpha(fwhm, moffat_beta)
            psf = moffat_psf_2d(xx, yy, x0, y0, flux, alpha, moffat_beta)
        else:
            raise ValueError(f"Unknown psf_model: {psf_model}")
        noise_free += psf

    image = noise_free.copy()

    if add_poisson:
        # Poisson noise in electrons, then convert back to ADU
        image_e = image * gain
        noisy_e = rng.poisson(np.clip(image_e, 0, None).astype(np.float64))
        image = noisy_e / gain

    if add_read_noise:
        rn = rng.normal(0, read_noise / np.sqrt(gain), size=(ny, nx))
        image += rn

    # Background RMS map (per-pixel)
    # sigma_total = sqrt( (sqrt(max(image,0)*gain)/gain)^2 + (background_rms)^2 + (read_noise/gain)^2 )
    poisson_term = np.sqrt(np.clip(image, 0, None) * gain) / gain
    bkg_rms_map = np.sqrt(
        poisson_term**2 + background_rms**2 + (read_noise / gain) ** 2
    )

    mask = np.zeros((ny, nx), dtype=bool)
    if mask_fraction > 0:
        n_mask = int(mask_fraction * ny * nx)
        idx = rng.choice(ny * nx, size=n_mask, replace=False)
        mask.flat[idx] = True
        image[mask] = np.nan

    header = fits.Header()
    header["TELESCOP"] = "Synthetic"
    header["INSTRUME"] = "TestCam"
    header["FILTER"] = "TEST"
    header["EXPTIME"] = 30.0
    header["GAIN"] = gain
    header["RDNOISE"] = read_noise
    header["SATURATE"] = 50000.0
    header["MJD-OBS"] = 59000.0

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]  # 1 arcsec/pixel
    wcs.wcs.crval = [150.0, 30.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header.update(wcs.to_header())

    truth = catalog.copy()
    truth["fwhm_true"] = fwhm
    truth["background_true"] = background
    truth["background_rms_true"] = background_rms
    truth["gain_true"] = gain
    truth["read_noise_true"] = read_noise
    truth["psf_model_true"] = psf_model

    return {
        "image": image.astype(np.float64),
        "truth": truth,
        "header": header,
        "background_rms_map": bkg_rms_map,
        "mask": mask,
        "noise_free": noise_free,
        "wcs": wcs,
    }


def make_single_source_image(
    flux: float = 10000.0,
    x0: float | None = None,
    y0: float | None = None,
    fwhm: float = 4.0,
    image_shape: tuple[int, int] = (512, 512),
    background: float = 100.0,
    background_rms: float = 10.0,
    gain: float = 1.0,
    read_noise: float = 5.0,
    psf_model: str = "gaussian",
    add_poisson: bool = True,
    add_read_noise: bool = True,
    seed: int | None = 42,
):
    """Create a synthetic image with a single point source.

    Convenience wrapper around ``make_synthetic_image``.
    If *x0* or *y0* is None, the source is placed at the image centre.
    """
    import pandas as pd

    if x0 is None:
        x0 = image_shape[1] / 2.0
    if y0 is None:
        y0 = image_shape[0] / 2.0

    cat = pd.DataFrame({"x_pix": [x0], "y_pix": [y0], "flux_true": [flux]})
    return make_synthetic_image(
        catalog=cat,
        image_shape=image_shape,
        fwhm=fwhm,
        background=background,
        background_rms=background_rms,
        gain=gain,
        read_noise=read_noise,
        psf_model=psf_model,
        add_poisson=add_poisson,
        add_read_noise=add_read_noise,
        seed=seed,
    )


def make_faint_source_image(
    snr_target: float = 5.0,
    fwhm: float = 4.0,
    background_rms: float = 10.0,
    gain: float = 1.0,
    read_noise: float = 5.0,
    image_shape: tuple[int, int] = (256, 256),
    seed: int | None = 42,
):
    """Create a synthetic image with a source at a target S/N.

    The flux is set so that the PSF-fit S/N approximately equals *snr_target*.
    For a Gaussian PSF, the optimal S/N is:

        SNR = flux * sqrt(1 / (4*pi*sigma^2 * sigma_pix^2))

    where sigma_pix^2 = background_rms^2 + read_noise^2 / gain^2.
    """
    sigma = fwhm_to_sigma(fwhm)
    sigma_pix = np.sqrt(background_rms**2 + (read_noise / gain) ** 2)
    # SNR_psf = flux / sqrt(4*pi*sigma^2 * sigma_pix^2)
    flux = snr_target * np.sqrt(4 * np.pi * sigma**2 * sigma_pix**2)

    return make_single_source_image(
        flux=flux,
        fwhm=fwhm,
        background_rms=background_rms,
        gain=gain,
        read_noise=read_noise,
        image_shape=image_shape,
        seed=seed,
    )


def make_saturated_source_image(
    flux: float = 1000000.0,
    saturate: float = 50000.0,
    fwhm: float = 4.0,
    image_shape: tuple[int, int] = (256, 256),
    seed: int | None = 42,
):
    """Create a synthetic image with a saturated source.

    Pixels above *saturate* are clipped and masked.  The flux is set high
    enough that the peak exceeds the saturation limit.
    """
    data = make_single_source_image(
        flux=flux, fwhm=fwhm, image_shape=image_shape, seed=seed,
        background=100.0, background_rms=10.0,
    )
    image = data["image"]
    mask = data["mask"]
    sat_mask = image >= saturate
    image[sat_mask] = saturate
    mask = mask | sat_mask
    data["image"] = image
    data["mask"] = mask
    data["header"]["SATURATE"] = saturate
    return data


def make_edge_source_image(
    flux: float = 10000.0,
    fwhm: float = 4.0,
    edge_offset: float = 5.0,
    image_shape: tuple[int, int] = (256, 256),
    seed: int | None = 42,
):
    """Create a synthetic image with a source near the edge."""
    return make_single_source_image(
        flux=flux,
        x0=edge_offset,
        y0=image_shape[0] / 2,
        fwhm=fwhm,
        image_shape=image_shape,
        seed=seed,
    )


def make_nan_region_image(
    flux: float = 10000.0,
    fwhm: float = 4.0,
    nan_fraction: float = 0.1,
    image_shape: tuple[int, int] = (256, 256),
    seed: int | None = 42,
):
    """Create a synthetic image with NaN regions (chip gaps)."""
    data = make_single_source_image(
        flux=flux, fwhm=fwhm, image_shape=image_shape, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    image = data["image"]
    mask = data["mask"]
    ny, nx = image_shape
    # Add a horizontal band of NaNs
    band_height = int(ny * nan_fraction)
    band_start = rng.integers(0, ny - band_height)
    image[band_start:band_start + band_height, :] = np.nan
    mask[band_start:band_start + band_height, :] = True
    data["image"] = image
    data["mask"] = mask
    return data
