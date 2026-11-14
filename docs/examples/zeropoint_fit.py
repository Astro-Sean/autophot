"""Photometric zero point: catalogue magnitude minus instrumental
magnitude over the matched calibrators gives the offset ZP. Sigma
clipping removes the handful of bad calibrators before the fit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autophot.core.plotting_utils import apply_autophot_mplstyle

rng = np.random.default_rng(11)
zp_true = 24.7

m_inst = rng.uniform(12, 19, 80)  # -2.5 log10 of instrumental flux
m_cat = m_inst + zp_true + rng.normal(0, 0.03, m_inst.size)
bad = rng.choice(m_inst.size, 5, replace=False)  # blends, variables, ...
m_cat[bad] += rng.normal(0, 0.35, bad.size)

dm = m_cat - m_inst
med = np.median(dm)
mad = 1.4826 * np.median(np.abs(dm - med))
keep = np.abs(dm - med) < 3 * mad
zp = np.median(dm[keep])

apply_autophot_mplstyle()
plt.rcParams["pdf.fonttype"] = 42  # keep text editable in the PDF
fig, ax = plt.subplots(figsize=(4.7, 3.0))
ax.plot(m_inst[keep], dm[keep], "o", ms=3, color="#0072B2", label="calibrators")
ax.plot(m_inst[~keep], dm[~keep], "x", ms=5, color="#D55E00", label="clipped")
ax.axhline(zp, color="#D55E00", lw=1.2, label=f"ZP = {zp:.2f} mag")
ax.set(
    xlabel="instrumental magnitude [mag]", ylabel=r"$m_{\rm cat} - m_{\rm inst}$ [mag]"
)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "figures" / "fig_zeropoint.pdf"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
