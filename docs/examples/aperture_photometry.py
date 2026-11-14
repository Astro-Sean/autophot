"""Aperture photometry: source aperture, sky annulus, and the curve of
growth that sets the optimum radius.

Net flux = counts inside the aperture minus (sky per pixel x area).
S/N rises with radius until the added sky noise beats the added flux.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autophot.core.plotting_utils import apply_autophot_mplstyle

rng = np.random.default_rng(7)

n = 61
c = (n - 1) / 2
yy, xx = np.mgrid[:n, :n]
r = np.hypot(xx - c, yy - c)

alpha, beta = 2.127, 2.5  # Moffat shape parameters
fwhm = 2 * alpha * np.sqrt(2 ** (1 / beta) - 1)  # = 2.4 px
psf = 30000 * (1 + (r / alpha) ** 2) ** -beta
sky_level = 800.0  # ADU per pixel
img = rng.normal(sky_level + psf, np.sqrt(sky_level + psf))

# Curve of growth and aperture S/N as a function of radius.
radii = np.linspace(0.5, 20, 200)
enclosed = np.array([psf[r <= rr].sum() for rr in radii])
frac = enclosed / enclosed[-1]
snr = enclosed / np.sqrt(enclosed + np.pi * radii**2 * (sky_level + 12.0**2))
r_opt = radii[np.argmax(snr)]
r_in, r_out = r_opt + fwhm, r_opt + 3 * fwhm  # annulus: 1 FWHM gap, 2 FWHM wide

apply_autophot_mplstyle()
plt.rcParams["pdf.fonttype"] = 42  # keep text editable in the PDF
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(6.378, 2.5))

ax0.imshow(img, cmap="gray_r", vmin=750, vmax=5000)
for rad, ls in [(r_opt, "-"), (r_in, "--"), (r_out, "--")]:
    ax0.add_patch(
        plt.Circle((c, c), rad, fill=False, edgecolor="#0072B2", ls=ls, lw=1.0)
    )
ax0.set(xlabel="x [px]", ylabel="y [px]")

ax1.plot(radii / fwhm, 100 * frac, color="#0072B2")
ax1.axvline(r_opt / fwhm, color="#D55E00", ls="--", lw=1.0)
ax1.set(
    xlabel="aperture radius [FWHM]",
    ylabel="enclosed flux [%]",
    ylim=(88, 102),
    xlim=(0, 4),
)

ax2.plot(radii / fwhm, snr, color="#0072B2")
ax2.axvline(r_opt / fwhm, color="#D55E00", ls="--", lw=1.0)
ax2.annotate(
    f"optimum = {r_opt / fwhm:.1f} FWHM",
    (3.9, snr.max() * 0.5),
    fontsize=8,
    ha="right",
)
ax2.set(xlabel="aperture radius [FWHM]", ylabel="S/N in aperture", xlim=(0, 4))

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures" / "fig_aperture.pdf"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
