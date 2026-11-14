"""Sky background estimation: the mesh grid and the clipped sky histogram.

A mesh cell's pixels are mostly sky; the source population only adds a
bright tail. Sigma clipping the cell histogram recovers the sky level.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autophot.core.plotting_utils import apply_autophot_mplstyle

rng = np.random.default_rng(42)

ny, nx = 200, 200
yy, xx = np.mgrid[:ny, :nx]
sky = 800 + 0.8 * xx - 0.4 * yy  # smooth sky gradient [ADU]
img = rng.normal(sky, 12.0)
for x0, y0, amp in [(60, 140, 900), (150, 90, 700), (110, 60, 500), (40, 40, 400)]:
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    img += amp * np.exp(-r2 / (2 * 2.0**2))

apply_autophot_mplstyle()
plt.rcParams["pdf.fonttype"] = 42  # keep text editable in the PDF
fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(6.378, 2.9))

cell = 50  # mesh cell size [px]
ax_img.imshow(img, cmap="gray_r", vmin=780, vmax=900)
for k in range(1, nx // cell):
    ax_img.axvline(k * cell, color="#0072B2", lw=0.6)
    ax_img.axhline(k * cell, color="#0072B2", lw=0.6)
ax_img.add_patch(
    plt.Rectangle((100, 50), cell, cell, fill=False, edgecolor="#D55E00", lw=1.8)
)
ax_img.set(xlabel="x [px]", ylabel="y [px]")

pix = img[50 : 50 + cell, 100 : 100 + cell].ravel()  # highlighted cell
ax_hist.hist(pix, bins=40, color="#0072B2")
med = np.median(pix)
sig = 1.4826 * np.median(np.abs(pix - med))
ax_hist.axvline(med, color="#D55E00", lw=1.2, label="clipped median (sky)")
ax_hist.axvline(
    med + 3 * sig, color="#D55E00", lw=1.2, ls="--", label=r"median + 3$\sigma$"
)
ax_hist.set(
    xlabel="counts in highlighted cell [ADU]", ylabel="pixels", xlim=(790, 1000)
)
ax_hist.legend(frameon=False, loc="upper right", fontsize=7)
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "figures" / "fig_background.pdf"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
