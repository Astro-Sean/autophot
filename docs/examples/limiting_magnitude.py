"""Injection-recovery limits: artificial sources are added at known
magnitudes, and the magnitude at which 50% are recovered is the limiting
magnitude.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from autophot.core.plotting_utils import apply_autophot_mplstyle

rng = np.random.default_rng(5)

m_true = rng.uniform(18, 25, 600)
snr = 10 ** (0.2 * (23.5 - m_true)) * rng.lognormal(0, 0.15, m_true.size)
detected = snr > 3.0

bins = np.arange(18, 25.5, 0.5)
cen = 0.5 * (bins[1:] + bins[:-1])
frac = np.array(
    [detected[(m_true >= lo) & (m_true < hi)].mean() for lo, hi in pairwise(bins)]
)


def logistic(m, m50, k):
    return 1.0 / (1.0 + np.exp(k * (m - m50)))


(m50, k), _ = curve_fit(logistic, m_true, detected.astype(float), p0=[23, 1.5])
grid = np.linspace(18, 25, 300)

apply_autophot_mplstyle()
plt.rcParams["pdf.fonttype"] = 42  # keep text editable in the PDF
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.378, 2.7))

ax0.semilogy(m_true, snr, "o", ms=2, color="#0072B2", alpha=0.5)
ax0.axhline(3.0, color="#D55E00", ls="--", lw=1.0, label="detection threshold")
ax0.set(xlabel="injected magnitude [mag]", ylabel="recovery S/N", ylim=(0.8, 500))
ax0.legend(frameon=False, fontsize=7, loc="upper right")

ax1.plot(grid, 100 * logistic(grid, m50, k), color="#0072B2")
ax1.plot(cen, 100 * frac, "o", ms=3, color="#0072B2")
ax1.axhline(50, color="#D55E00", ls="--", lw=1.0, label="50% recovered")
ax1.axvline(m50, color="#D55E00", ls=":", lw=1.0, label=f"$m_{{50}}$ = {m50:.1f} mag")
ax1.legend(frameon=False, fontsize=7, loc="lower left")
ax1.set(xlabel="injected magnitude [mag]", ylabel="recovered [%]", ylim=(0, 105))

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures" / "fig_limiting.pdf"
fig.savefig(out)
plt.close(fig)
print(f"wrote {out}")
