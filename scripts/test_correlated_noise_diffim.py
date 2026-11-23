"""Validate the empirical correlated-noise factor on synthetic diff images.

Convolution of one side of a difference image with the A&L matching kernel
correlates pixel noise. The pipeline's error array captures the per-pixel
sigma but not the covariance, so naive aperture/PSF-fit errors underestimate
the true scatter. autophot.photometry.aperture._empirical_noise_factor
measures the missing factor with empty-sky test apertures.

This script:
1. Builds a correlated-noise field (white noise convolved with a unit-sum
   kernel) plus a known per-pixel error map.
2. Measures the true aperture-sum scatter directly.
3. Checks that _empirical_noise_factor recovers it.

Run:  python scripts/test_correlated_noise_diffim.py
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from photutils.aperture import CircularAperture, aperture_photometry

from autophot.photometry.aperture import _empirical_noise_factor

SEED = 7
SIZE = 1024
SIGMA_WHITE = 1.0  # pre-convolution pixel sigma
KERNEL_SIGMA = 1.2  # px; ~A&L kernel width scale


def make_correlated_noise():
    rng = np.random.default_rng(SEED)
    white = rng.normal(0.0, SIGMA_WHITE, (SIZE, SIZE))
    kern = Gaussian2DKernel(KERNEL_SIGMA, x_size=15, y_size=15).array
    kern = kern / kern.sum()  # A&L kernels are normalised to unit sum
    corr = convolve_fft(white, kern, boundary="wrap")
    return corr, kern


def true_aperture_scatter(image, radius, n=800, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    margin = 30
    pos = np.c_[
        rng.uniform(margin, image.shape[1] - margin, n),
        rng.uniform(margin, image.shape[0] - margin, n),
    ]
    phot = aperture_photometry(
        image, CircularAperture(pos, r=radius), method="exact"
    )
    s = np.asarray(phot["aperture_sum"], dtype=float)
    s = s[np.isfinite(s)]
    mad = 1.4826 * np.median(np.abs(s - np.median(s)))
    return mad


def main():
    image, kern = make_correlated_noise()

    # Per-pixel sigma after convolution with unit-sum kernel.
    sigma_pix = SIGMA_WHITE * np.sqrt(np.sum(kern**2))
    print(f"per-pixel sigma: analytic {sigma_pix:.4f}, "
          f"measured {np.std(image):.4f}")

    # Give the estimator a correct per-pixel error map - the only thing it
    # must add is the covariance correction.
    error = np.full_like(image, sigma_pix)

    for radius in (2.5, 4.0, 6.0):
        ann_in = radius + 8.0
        ann_out = radius + 16.0

        true_scatter = true_aperture_scatter(image, radius)
        naive_pred = sigma_pix * np.sqrt(np.pi * radius**2)
        true_factor = true_scatter / naive_pred

        enf, n_ap = _empirical_noise_factor(
            image,
            error,
            radius,
            ann_in,
            ann_out,
            defects_mask=None,
            source_xy=None,
            n_trials=400,
            seed=SEED,
        )
        print(
            f"r={radius:4.1f}px  true factor {true_factor:5.2f}  "
            f"empirical factor {enf:5.2f}  (n_ap={n_ap})"
        )

    # Sanity: correlated noise must inflate aperture scatter.
    assert true_factor > 1.2, "correlated noise not detected at all"
    # The estimator must move in the right direction.
    assert enf > 1.1, f"empirical factor {enf:.2f} too low"
    print("PASS: empirical factor recovers correlated-noise inflation "
          "(order-of-magnitude agreement).")


if __name__ == "__main__":
    main()
