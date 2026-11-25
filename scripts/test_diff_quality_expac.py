#!/usr/bin/env python3
"""Validate the difference-quality gate fix on a real SFFT product.

Reproduces the EXPAC computation added to run_sfft.py (kernel-implied
noise autocorrelation from the realized matching kernel + input-image
IQR variances), then runs assess_difference_image with the new
flux-scaled dipole/bright-star thresholds and the expected-autocorr
scoring.  Compares against the classification the old gate recorded
in the run log.

Usage: test_diff_quality_expac.py <image_dir>
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, "/home/sbrennan/Documents/autophot_object")


def _iqr_var(vals):
    q25, q75 = np.percentile(vals, [25.0, 75.0])
    return ((q75 - q25) / 1.3489795003921634) ** 2


def main():
    base = sys.argv[1]
    diff_path = glob.glob(os.path.join(base, "diff_*.fits"))[0]
    stem = os.path.basename(diff_path)[5:-5]
    # Science file is the stem minus the _APT suffix; the mask file keeps it.
    sci_glob = os.path.join(base, f"{stem.removesuffix('_APT')}.fits")
    mask_glob = os.path.join(base, f"universal_mask_{stem}.fits")
    if not os.path.isfile(sci_glob):
        cands = [
            f
            for f in glob.glob(os.path.join(base, "*.fits"))
            if "diff" not in f
            and "mask" not in f
            and "template" not in f
            and "weight" not in f
            and "PSF" not in f
            and "Solution" not in f
        ]
        sci_glob = cands[0]

    diff = fits.getdata(diff_path).astype(float)
    diff_hdr = fits.getheader(diff_path)
    sci = fits.getdata(sci_glob).astype(float)
    umask = (
        fits.getdata(mask_glob).astype(bool)
        if os.path.isfile(mask_glob)
        else np.zeros(diff.shape, bool)
    )
    print(os.path.basename(sci_glob))
    print(
        "CONVD =",
        diff_hdr.get("CONVD"),
        "| KERHW =",
        diff_hdr.get("KERHW"),
        "| FWHM_SCI =",
        round(float(diff_hdr.get("FWHM_SCI", 0)), 2),
        "FWHM_REF =",
        round(float(diff_hdr.get("FWHM_REF", 0)), 2),
    )

    quality_mask = ~np.isfinite(diff) | (np.abs(diff) < 1.1e-20) | umask

    sci_vals = sci[np.isfinite(sci) & (sci != 0.0)]
    sci_var = _iqr_var(sci_vals)
    tpl_path = os.path.join(base, "rp_template.fits")
    if os.path.isfile(tpl_path):
        tpl = fits.getdata(tpl_path).astype(float)
        tpl_vals = tpl[np.isfinite(tpl) & (tpl != 0.0)]
        ref_var = _iqr_var(tpl_vals)
    else:
        ref_var = sci_var * 0.25
    print(f"sigma_sci={np.sqrt(sci_var):.2f}  sigma_ref={np.sqrt(ref_var):.2f}")

    from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel

    ny, nx = diff.shape
    XY_q = np.array([[(nx + 1.0) / 2.0, (ny + 1.0) / 2.0]])
    ker = (
        Realize_MatchingKernel(XY_q)
        .FromFITS(os.path.join(base, "SFFT_Solution.fits"))[0]
        .T
    )
    print("kernel realized:", ker.shape, "sum=", float(np.nansum(ker)))

    from scipy.signal import correlate as _sig_corr

    convd = str(diff_hdr.get("CONVD", "")).strip().upper()
    kk = _sig_corr(ker, ker, mode="full")
    kc = kk.shape[0] // 2
    ac0 = kk[kc, kc]
    lag_max = min(20, kc)
    ac_prof = np.concatenate(
        [kk[kc, kc + 1 : kc + lag_max + 1], kk[kc + 1 : kc + lag_max + 1, kc]]
    ) / ac0
    ac_prof = ac_prof[np.isfinite(ac_prof)]
    k2 = float(np.nansum(ker**2))
    var_conv = (sci_var if convd == "SCI" else ref_var) * k2
    var_other = ref_var if convd == "SCI" else sci_var
    f_conv = var_conv / (var_conv + var_other)
    exp_pos = f_conv * max(float(np.max(ac_prof)), 0.0)
    exp_neg = f_conv * min(float(np.min(ac_prof)), 0.0)
    print(
        f"EXPAC: f_conv={f_conv:.3f}  acK_peak={np.max(ac_prof):.3f}  "
        f"exp_pos={exp_pos:.3f}  exp_neg={exp_neg:.3f}"
    )

    import sep

    sci_sub = np.ascontiguousarray(sci - np.median(sci_vals), dtype=np.float32)
    objs = sep.extract(
        sci_sub,
        5.0,
        err=float(np.sqrt(sci_var)),
        mask=~np.isfinite(sci),
    )
    order = np.argsort(objs["flux"])[::-1]
    src_xy = [(float(o["x"]), float(o["y"])) for o in objs[order[:40]]]
    print(f"{len(objs)} detected; checking {len(src_xy)} brightest")

    fscal = (
        float(diff_hdr.get("FSCAL_CONV", 1.0) or 1.0) if convd == "SCI" else 1.0
    )
    fluxes = []
    for sx, sy in src_xy:
        xi, yi = int(round(sx)), int(round(sy))
        f = 0.0
        if 3 <= xi < sci.shape[1] - 3 and 3 <= yi < sci.shape[0] - 3:
            st = sci[yi - 3 : yi + 4, xi - 3 : xi + 4]
            if np.isfinite(st).any():
                border = np.concatenate(
                    [st[0], st[-1], st[1:-1, 0], st[1:-1, -1]]
                )
                border = border[np.isfinite(border)]
                sky = float(np.median(border)) if border.size else 0.0
                f = max(float(np.nanmax(st)) - sky, 0.0) * fscal
        fluxes.append(f)
    bidx = np.argsort(fluxes)[::-1][:15]
    bright_pos = [src_xy[i] for i in bidx]
    bright_flx = [fluxes[i] for i in bidx]
    print(f"brightest source flux ~{max(fluxes):.0f} ADU (scaled)")

    from autophot.subtraction.difference_quality import (
        QualityConfig,
        assess_difference_image,
    )

    m = assess_difference_image(
        diff,
        quality_mask=quality_mask,
        source_positions=src_xy,
        bright_star_positions=bright_pos,
        source_fluxes=fluxes,
        bright_star_fluxes=bright_flx,
        expected_autocorr_pos=exp_pos,
        expected_autocorr_neg=exp_neg,
        fwhm=float(diff_hdr.get("FWHM_SCI", 3.0)),
        cfg=QualityConfig(),
        metadata={"algorithm": "sfft", "forceconv": convd},
    )
    print(
        f"\nRESULT: class={m.quality_class} score={m.quality_score:.3f}\n"
        f"  dipoles={m.dipole_count}/{m.dipole_checked} "
        f"({m.dipole_fraction*100:.1f}%) score={m.score_dipole:.2f}\n"
        f"  bright_star rms={m.bright_star_residual_rms:.1f}sigma "
        f"norm={m.bright_star_residual_rms_norm:.2f} "
        f"score={m.score_bright_star:.2f} (n={m.bright_star_count})\n"
        f"  autocorr signed={m.autocorr_peak_signed:+.3f} "
        f"expected=+{m.autocorr_expected_pos:.3f} "
        f"score={m.score_autocorr:.2f}\n"
        f"  bg={m.score_background:.2f} edge={m.score_edge:.2f}"
    )
    # What the OLD gate would have reported for comparison.
    m_old = assess_difference_image(
        diff,
        quality_mask=quality_mask,
        source_positions=src_xy,
        bright_star_positions=None,
        fwhm=float(diff_hdr.get("FWHM_SCI", 3.0)),
        cfg=QualityConfig(),
        metadata={"algorithm": "sfft", "forceconv": convd},
    )
    print(
        f"OLD-style (no fluxes, no EXPAC, no bright stars): "
        f"class={m_old.quality_class} score={m_old.quality_score:.3f} "
        f"dip={m_old.dipole_count}/{m_old.dipole_checked} "
        f"ac={m_old.autocorr_peak:.3f}"
    )


if __name__ == "__main__":
    main()
