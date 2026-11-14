"""Image subtraction: matching the template's flux scale and PSF to the
science frame, then subtracting, removes the sky and every static source
- only the transient survives.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autophot.core.plotting_utils import apply_autophot_mplstyle

rng = np.random.default_rng(9)

n = 121
yy, xx = np.mgrid[:n, :n]


def gauss(x0, y0, amp, fwhm=3.0):
    s = fwhm / 2.355
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s**2))


sky = 500 + 0.3 * xx
template = sky + gauss(40, 70, 2000) + gauss(90, 30, 800)
science = template + gauss(65, 60, 1500)  # the transient

sci_obs = rng.normal(science, np.sqrt(science))
tpl_obs = rng.normal(template, np.sqrt(template))
diff = sci_obs - tpl_obs

apply_autophot_mplstyle()
plt.rcParams["pdf.fonttype"] = 42  # keep text editable in the PDF
fig, axs = plt.subplots(1, 3, figsize=(6.378, 2.5), sharex=True, sharey=True)
panels = [
    (sci_obs, 400, 900, "(a) science"),
    (tpl_obs, 400, 900, "(b) template"),
    (diff, -150, 300, "(c) difference"),
]
for ax, (data, lo, hi, label) in zip(axs, panels):
    ax.imshow(data, cmap="gray_r", vmin=lo, vmax=hi)
    ax.plot(65, 60, "+", ms=8, color="#D55E00", mew=1.2)
    ax.annotate(label, (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=8)
    ax.set_xlabel("x [px]")
axs[0].set_ylabel("y [px]")

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures" / "fig_subtraction.pdf"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
