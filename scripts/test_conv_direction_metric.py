"""Test the FWHM+depth convolution-direction metric (Angulo et al. 2025).

arXiv:2508.10155 shows the standard "convolve the sharper image" rule is
wrong when the depths differ: the better direction is decided by a sloped
line in the (10*log10(FWHM ratio), delta m_5sigma) plane, not a vertical
line at equal FWHM.

Convention here matches SFFT/ForceConv in autophot:
  D = SCI - conv(REF)   -> ForceConv='REF'
  D = conv(SCI) - REF   -> ForceConv='SCI'

Metric (paper Eq. 1, global DECam slope ms=0.81):
  x = 10*log10(FWHM_sci / FWHM_ref)
  y = m5_sci - m5_ref          (5-sigma limiting depths, mag)
  convolve REF if  y < ms * x  (point below dividing line)

The script:
1. Implements choose_direction() and the template figure-of-merit (FoM).
2. Builds synthetic aligned image pairs over a grid of FWHM ratios and depth
   ratios, computes the matching kernel both ways (Wiener-regularised
   Fourier division - deconvolution of the broader image amplifies noise,
   which is exactly the failure mode the metric guards against), and scores
   each difference image by residual scatter around bright stars.
3. Reports agreement between the metric prediction and the empirical winner.

Run:  python scripts/test_conv_direction_metric.py
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import fftconvolve

GLOBAL_SLOPE = 0.81  # DECam combined fit; Swope ~0.3 - instrument dependent


def choose_direction(fwhm_sci, fwhm_ref, m5_sci, m5_ref, slope=GLOBAL_SLOPE):
    """Return 'REF' if the reference should be convolved (SCI - conv(REF))."""
    x = 10.0 * np.log10(fwhm_sci / fwhm_ref)
    y = m5_sci - m5_ref
    return ("REF", y - slope * x) if y < slope * x else ("SCI", y - slope * x)


def template_fom(fwhm_img, m5_img, fwhm_med, m5_med, slope=GLOBAL_SLOPE):
    """Signed FoM for ranking template candidates (paper Eq. 1).

    Smaller (more negative) = better template, relative to a reference
    point (median seeing/depth of the group).
    """
    x = 10.0 * np.log10(fwhm_img / fwhm_med)
    y = m5_img - m5_med
    return (y - slope * x) / np.sqrt(1.0 + slope**2)


# ---------------------------------------------------------------------------
# Synthetic subtraction experiment
# ---------------------------------------------------------------------------

def gauss_psf(size, fwhm):
    sig = fwhm / 2.3548
    c = (size - 1) / 2
    y, x = np.mgrid[:size, :size]
    g = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sig**2))
    return g / g.sum()


def make_image(size, fwhm, sky_sigma, star_xy, star_flux, rng):
    img = rng.normal(0.0, sky_sigma, (size, size))
    psf = gauss_psf(31, fwhm)
    for (x, y), f in zip(star_xy, star_flux):
        stamp = np.zeros_like(img)
        stamp[int(round(y)), int(round(x))] = f
        img += fftconvolve(stamp, psf, mode="same")
    return img, psf


def matching_kernel(psf_conv, psf_target, wiener=1e-4):
    """Kernel convolving image with PSF psf_conv to PSF psf_target.

    Wiener-regularised Fourier division. When psf_conv is broader than
    psf_target this is a deconvolution: the kernel amplifies high-frequency
    noise, which is the artifact the direction choice must avoid.
    """
    size = psf_conv.shape[0]
    A = fft2(fftshift(psf_conv))
    T = fft2(fftshift(psf_target))
    w = wiener * np.abs(A).max() ** 2
    K = np.real(ifft2(np.conj(A) * T / (np.abs(A) ** 2 + w)))
    return fftshift(K)


def diff_quality(diff, star_xy, r_in=6, r_out=12):
    """RMS of the difference image in annuli around bright stars.

    Good subtraction leaves only noise: low RMS. Bad convolution direction
    leaves ringing/residual flux correlated with star positions.
    """
    y, x = np.mgrid[: diff.shape[0], : diff.shape[1]]
    resid = []
    for sx, sy in star_xy:
        d = np.hypot(x - sx, y - sy)
        ann = diff[(d >= r_in) & (d < r_out)]
        ann = ann[np.isfinite(ann)]
        if ann.size:
            resid.append(np.std(ann))
    return float(np.median(resid))


def run_grid():
    rng = np.random.default_rng(11)
    size = 256
    nstars = 25
    star_xy = np.c_[
        rng.uniform(30, size - 30, nstars), rng.uniform(30, size - 30, nstars)
    ]
    star_flux = rng.uniform(3e4, 3e5, nstars)

    fwhm_ref = 3.0  # fixed template seeing
    cases = []
    for fwhm_sci in (2.2, 3.0, 4.0):
        for depth_ratio in (0.0, 1.0, 2.0):  # m5_sci - m5_ref, mag
            # depth ratio -> sky sigma ratio: m5 ~ -1.25 log10(sigma^2)
            sigma_ref = 3.0
            sigma_sci = sigma_ref * 10 ** (-depth_ratio / 2.5 / 2)
            img_s, psf_s = make_image(
                size, fwhm_sci, sigma_sci, star_xy, star_flux, rng
            )
            img_r, psf_r = make_image(
                size, fwhm_ref, sigma_ref, star_xy, star_flux, rng
            )
            m5_sci = -2.5 * np.log10(5 * sigma_sci)
            m5_ref = -2.5 * np.log10(5 * sigma_ref)

            # ForceConv=REF: D = SCI - conv(REF)
            k_r = matching_kernel(psf_r, psf_s)
            d_ref = img_s - fftconvolve(img_r, k_r, mode="same")
            # ForceConv=SCI: D = conv(SCI) - REF
            k_s = matching_kernel(psf_s, psf_r)
            d_sci = fftconvolve(img_s, k_s, mode="same") - img_r

            q_ref = diff_quality(d_ref, star_xy)
            q_sci = diff_quality(d_sci, star_xy)
            empirical = "REF" if q_ref < q_sci else "SCI"
            pred, margin = choose_direction(
                fwhm_sci, fwhm_ref, m5_sci, m5_ref
            )
            ok = pred == empirical
            cases.append(ok)
            print(
                f"FWHM sci/ref {fwhm_sci:.1f}/{fwhm_ref:.1f}  "
                f"dm5 {m5_sci - m5_ref:+.2f}  "
                f"q(REF) {q_ref:8.3f}  q(SCI) {q_sci:8.3f}  "
                f"empirical {empirical}  metric {pred}  "
                f"{'OK' if ok else 'MISS'}"
            )

    agree = np.mean(cases)
    print(f"\nmetric/empirical agreement: {agree:.0%} of {len(cases)}")
    print("note: agreement < 100% is expected near the dividing line; "
          "the key result is that depth shifts the boundary.")


def demo_fom():
    """Rank a small template pool against a group median."""
    fwhm_med, m5_med = 3.0, 22.0
    pool = [
        ("deep_sharp", 2.2, 23.5),
        ("deep_broad", 4.5, 23.5),
        ("shal_sharp", 2.2, 21.5),
        ("median", 3.0, 22.0),
    ]
    print("\ntemplate FoM ranking (smaller = better):")
    for name, f, m in sorted(
        pool, key=lambda p: template_fom(p[1], p[2], fwhm_med, m5_med)
    ):
        print(f"  {name:12s} fwhm={f:.1f} m5={m:.1f} "
              f"FoM={template_fom(f, m, fwhm_med, m5_med):+.3f}")


if __name__ == "__main__":
    run_grid()
    demo_fom()
