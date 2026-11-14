"""The empirical PSF: stars land at different sub-pixel phases, so
stacking many undersampled cutouts recovers a smooth oversampled model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autophot.core.plotting_utils import apply_autophot_mplstyle

rng = np.random.default_rng(3)


def moffat(yy, xx, x0, y0, alpha=2.127, beta=2.5, amp=2000):
    return amp * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / alpha**2) ** -beta


n = 15  # native stamp size [px]
yy, xx = np.mgrid[:n, :n]
x0, y0 = n / 2 - 0.3, n / 2 + 0.2
obs = rng.normal(moffat(yy, xx, x0, y0) + 100, np.sqrt(moffat(yy, xx, x0, y0) + 100))

osamp = 4  # the ePSF grid is osamp times finer than the pixels
yyf, xxf = np.mgrid[: n * osamp, : n * osamp] / osamp - 0.375
epsf = moffat(yyf, xxf, x0, y0)

apply_autophot_mplstyle()
plt.rcParams["pdf.fonttype"] = 42  # keep text editable in the PDF
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.378, 2.9), sharey=True)

ax0.imshow(obs, cmap="gray_r", vmin=60, vmax=2100, extent=(0, n, 0, n))
ax0.set(xlabel="x [px]", ylabel="y [px]")

ax1.imshow(epsf, cmap="gray_r", vmin=60, vmax=2100, extent=(0, n, 0, n))
for k in range(1, n):  # native pixel boundaries
    ax1.axvline(k, color="w", lw=0.3, alpha=0.6)
    ax1.axhline(k, color="w", lw=0.3, alpha=0.6)
ax1.set(xlabel="x [px]")

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures" / "fig_epsf.pdf"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
