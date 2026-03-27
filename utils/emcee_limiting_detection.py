#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emcee-style limiting detection (injection + PSF-MCMC recovery).

This is a standalone utility to estimate a limiting *instrumental* magnitude by
injecting an ePSF-shaped source into an image cutout and recovering it using
emcee-based PSF photometry (the same `MCMCFitter` used in `psf.py`).

Typical usage
-------------
python3 utils/emcee_limiting_detection.py \
  --fits science.fits \
  --epsf-pickle epsf_model.pkl \
  --x 512 --y 512 \
  --fwhm-px 4.2 \
  --beta-limit 0.5 \
  --snr-limit 3 \
  --recovery-frac 0.5

Notes
-----
* This script requires a pickled `epsf_model` (photutils EPSFModel / astropy model).
* It reports an *instrumental* limiting magnitude (add zeropoint externally).
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
from astropy.io import fits

from aperture import Aperture
from functions import beta_aperture, mag, points_in_circum
from psf import MCMCFitter


logger = logging.getLogger(__name__)


def _read_fits_2d(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FITS image at {path}, got shape={arr.shape}")
    return arr


def _extract_stamp(image: np.ndarray, x0: float, y0: float, half: int) -> tuple[np.ndarray, int, int]:
    ny, nx = image.shape
    xi = int(np.floor(float(x0)))
    yi = int(np.floor(float(y0)))
    x1 = max(0, xi - half)
    x2 = min(nx, xi + half + 2)
    y1 = max(0, yi - half)
    y2 = min(ny, yi + half + 2)
    return np.asarray(image[y1:y2, x1:x2], dtype=float), x1, y1


def _recover_psf_mcmc_snr(
    data2d: np.ndarray,
    epsf_model,
    *,
    x0: float,
    y0: float,
    background_rms: np.ndarray | None,
    gain: float,
    read_noise: float,
    nwalkers: int,
    nsteps: int | None,
    delta_px: float,
    burnin_frac: float,
    thin: int,
    adaptive_tau_target: int,
    min_autocorr_N: int,
    batch_steps: int,
    jitter_scale: float,
) -> tuple[float, float]:
    """
    Fit flux,x0,y0 with emcee and return (snr, flux_hat).
    """
    ny, nx = data2d.shape
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))

    model = epsf_model.copy()
    # Model coordinates are *absolute within this stamp* (not full image).
    # Set initial guesses.
    model.x_0.value = float(x0)
    model.y_0.value = float(y0)
    model.flux.value = max(1e-6, float(getattr(model, "flux", 1.0)))

    fitter = MCMCFitter(
        nwalkers=int(nwalkers),
        nsteps=int(nsteps) if nsteps is not None else None,
        delta=float(delta_px),
        burnin_frac=float(burnin_frac),
        thin=int(thin),
        adaptive_tau_target=int(adaptive_tau_target),
        min_autocorr_N=int(min_autocorr_N),
        batch_steps=int(batch_steps),
        jitter_scale=float(jitter_scale),
        use_nddata_uncertainty=True,
        gain=float(gain),
        readnoise=float(read_noise),
        background_rms=background_rms,
    )

    fitted = fitter(
        model,
        gx,
        gy,
        data2d,
        initial_params=np.array([model.flux.value, model.x_0.value, model.y_0.value], dtype=float)
        if list(model.param_names)[:3] == ["flux", "x_0", "y_0"]
        else None,
        use_nddata_uncertainty=True,
    )

    # Extract flux and its 1-sigma error from the fitter's covariance proxy.
    try:
        pnames = list(fitted.param_names)
        i_flux = pnames.index("flux")
        flux_hat = float(fitted.parameters[i_flux])
        flux_err = float(getattr(fitted, "stds", np.full_like(fitted.parameters, np.nan))[i_flux])
    except Exception:
        flux_hat, flux_err = np.nan, np.nan

    snr = flux_hat / flux_err if np.isfinite(flux_hat) and np.isfinite(flux_err) and flux_err > 0 else np.nan
    return float(snr), float(flux_hat)


def main() -> int:
    ap = argparse.ArgumentParser(description="emcee injection/recovery limiting magnitude")
    ap.add_argument("--fits", required=True, help="Science image FITS path")
    ap.add_argument("--epsf-pickle", required=True, help="Pickled photutils/astropy ePSF model")
    ap.add_argument("--rms-fits", default=None, help="Optional background RMS FITS path (same shape)")
    ap.add_argument("--x", type=float, required=True, help="Target x (pixel)")
    ap.add_argument("--y", type=float, required=True, help="Target y (pixel)")
    ap.add_argument("--fwhm-px", type=float, required=True, help="FWHM in pixels")
    ap.add_argument("--gain", type=float, default=1.0, help="Gain (e-/ADU)")
    ap.add_argument("--read-noise", type=float, default=0.0, help="Read noise (e-)")

    ap.add_argument("--inject-radius-fwhm", type=float, default=3.0, help="Injection ring radius in FWHM")
    ap.add_argument("--n-sites", type=int, default=10, help="Number of injection sites")
    ap.add_argument("--redo", type=int, default=3, help="Subpixel jitter repeats per site")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")

    ap.add_argument("--beta-limit", type=float, default=0.5, help="Beta threshold for recovery")
    ap.add_argument("--snr-limit", type=float, default=3.0, help="SNR threshold for recovery")
    ap.add_argument("--recovery-frac", type=float, default=0.5, help="Target recovery fraction (beta)")

    ap.add_argument("--m0", type=float, default=-5.0, help="Initial instrumental magnitude guess")
    ap.add_argument("--step", type=float, default=0.5, help="Bracket step (mag)")
    ap.add_argument("--max-bracket-steps", type=int, default=30, help="Max bracket steps")
    ap.add_argument("--max-bisect-steps", type=int, default=12, help="Max bisection steps")

    # emcee controls (reuse psf.py defaults)
    ap.add_argument("--nwalkers", type=int, default=20)
    ap.add_argument("--nsteps", type=int, default=5000, help="Steps per walker (set 0 for adaptive)")
    ap.add_argument("--delta-px", type=float, default=5.0, help="Centroid bound half-width (px)")
    ap.add_argument("--burnin-frac", type=float, default=0.3)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--adaptive-tau-target", type=int, default=50)
    ap.add_argument("--min-autocorr-N", type=int, default=100)
    ap.add_argument("--batch-steps", type=int, default=100)
    ap.add_argument("--jitter-scale", type=float, default=0.01)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    img = _read_fits_2d(args.fits)
    rms = _read_fits_2d(args.rms_fits) if args.rms_fits else None

    with open(args.epsf_pickle, "rb") as f:
        epsf_model = pickle.load(f)

    rng = np.random.default_rng(args.seed)

    # Use a cutout for speed.
    cut_half = int(np.ceil(max(20.0, 8.0 * float(args.fwhm_px))))
    cut, xoff, yoff = _extract_stamp(img, args.x, args.y, half=cut_half)
    rms_cut = None
    if rms is not None:
        rms_cut, _, _ = _extract_stamp(rms, args.x, args.y, half=cut_half)
        rms_cut = np.abs(np.asarray(rms_cut, dtype=float))

    # Centre coordinates in cutout frame.
    x_c = float(args.x) - float(xoff)
    y_c = float(args.y) - float(yoff)

    # Choose quiet injection sites (beta below threshold on the *original* cutout).
    inj_dist = float(args.inject_radius_fwhm) * float(args.fwhm_px)
    pts = points_in_circum(inj_dist, center=(x_c, y_c), n=int(args.n_sites))
    df_sites = pd.DataFrame({"x_pix": [p[0] for p in pts], "y_pix": [p[1] for p in pts]})
    ini_ap = Aperture(input_yaml={"fwhm": float(args.fwhm_px), "photometry": {}}, image=cut)
    df_meas = ini_ap.measure(sources=df_sites, plot=False, background_rms=rms_cut, verbose=0)
    p_det = df_meas.apply(
        lambda row: beta_aperture(n=3.0, flux_aperture=row["flux_AP"], sigma=row["noiseSky"], npix=row["area"]),
        axis=1,
    )
    quiet = df_meas[p_det < float(args.beta_limit)].copy()
    if quiet.empty:
        logger.warning("No quiet sites found; using cutout centre.")
        quiet = pd.DataFrame({"x_pix": [x_c], "y_pix": [y_c]})

    x_sites = quiet["x_pix"].to_numpy(dtype=float)
    y_sites = quiet["y_pix"].to_numpy(dtype=float)

    # Calibrate flux<->mag using unit PSF measured with aperture photometry.
    H, W = cut.shape
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    psf_unit = np.asarray(epsf_model.evaluate(x=gx, y=gy, flux=1.0, x_0=x_c, y_0=y_c), dtype=float)
    psf_unit /= np.nansum(psf_unit)
    psf_ap = Aperture(input_yaml={"fwhm": float(args.fwhm_px), "photometry": {}}, image=psf_unit)
    F_ref = float(psf_ap.measure(pd.DataFrame({"x_pix": [x_c], "y_pix": [y_c]}), plot=False, background_rms=None, verbose=0)["flux_AP"].iloc[0])
    m_ref = mag(F_ref)

    def flux_for_mag(m: float) -> float:
        return 10.0 ** (-0.4 * (float(m) - float(m_ref)))

    def trials_at_mag(m: float) -> float:
        F = flux_for_mag(m)
        det = []
        for (xs, ys) in zip(x_sites, y_sites):
            for _ in range(int(args.redo)):
                dx = float(rng.random() - 0.5)
                dy = float(rng.random() - 0.5)
                x_inj = float(xs + dx)
                y_inj = float(ys + dy)
                psf_img = np.asarray(
                    epsf_model.evaluate(x=gx, y=gy, flux=F, x_0=x_inj, y_0=y_inj),
                    dtype=float,
                )
                new_img = cut + psf_img

                # Beta (aperture-based)
                ap0 = Aperture(input_yaml={"fwhm": float(args.fwhm_px), "photometry": {}}, image=new_img)
                mres = ap0.measure(pd.DataFrame({"x_pix": [x_inj], "y_pix": [y_inj]}), plot=False, background_rms=rms_cut, verbose=0)
                beta_p = beta_aperture(
                    n=3.0,
                    flux_aperture=float(mres["flux_AP"].iloc[0]),
                    sigma=float(mres["noiseSky"].iloc[0]),
                    npix=float(mres["area"].iloc[0]),
                )
                if not (np.isfinite(beta_p) and beta_p >= float(args.beta_limit)):
                    det.append(False)
                    continue

                # emcee recovery in a small stamp
                half = int(np.ceil(max(6.0, 3.0 * float(args.fwhm_px))))
                stamp, sx0, sy0 = _extract_stamp(new_img, x_inj, y_inj, half=half)
                rms_stamp = None
                if rms_cut is not None:
                    rms_stamp, _, _ = _extract_stamp(rms_cut, x_inj, y_inj, half=half)
                    rms_stamp = np.abs(np.asarray(rms_stamp, dtype=float))

                snr_val, _ = _recover_psf_mcmc_snr(
                    stamp,
                    epsf_model,
                    x0=float(x_inj - sx0),
                    y0=float(y_inj - sy0),
                    background_rms=rms_stamp,
                    gain=float(args.gain),
                    read_noise=float(args.read_noise),
                    nwalkers=int(args.nwalkers),
                    nsteps=None if int(args.nsteps) <= 0 else int(args.nsteps),
                    delta_px=float(args.delta_px),
                    burnin_frac=float(args.burnin_frac),
                    thin=int(args.thin),
                    adaptive_tau_target=int(args.adaptive_tau_target),
                    min_autocorr_N=int(args.min_autocorr_N),
                    batch_steps=int(args.batch_steps),
                    jitter_scale=float(args.jitter_scale),
                )
                det.append(bool(np.isfinite(snr_val) and snr_val >= float(args.snr_limit)))

        return float(np.mean(det)) if det else 0.0

    # Bracket then bisect.
    t0 = time.time()
    m_bright = float(args.m0)
    c_bright = trials_at_mag(m_bright)
    going_faint = c_bright >= float(args.recovery_frac)
    m_faint, c_faint = m_bright, c_bright

    for _ in range(int(args.max_bracket_steps)):
        m_test = m_faint + float(args.step) if going_faint else m_bright - float(args.step)
        c_test = trials_at_mag(m_test)
        if going_faint:
            m_faint, c_faint = m_test, c_test
            if c_faint < float(args.recovery_frac):
                break
        else:
            m_bright, c_bright = m_test, c_test
            if c_bright >= float(args.recovery_frac):
                break

    if not ((c_bright >= float(args.recovery_frac)) and (c_faint < float(args.recovery_frac))):
        logger.error("Failed to bracket recovery fraction. Try a different --m0/--step or more sites.")
        return 2

    lo, hi = float(m_bright), float(m_faint)
    for _ in range(int(args.max_bisect_steps)):
        mid = 0.5 * (lo + hi)
        c_mid = trials_at_mag(mid)
        if c_mid >= float(args.recovery_frac):
            lo = mid
        else:
            hi = mid

    m_lim = 0.5 * (lo + hi)
    logger.info("Limiting instrumental mag (MCMC PSF recovery): %.3f  [elapsed %.1fs]", m_lim, time.time() - t0)
    print(f"{m_lim:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

