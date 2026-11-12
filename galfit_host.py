"""
Optional GALFIT host-galaxy modelling for embedded targets.

Runs the external ``galfit`` binary on the target cutout, fits Sersic
component(s) (+ optional PSF at the target position) and returns a
residual image with the host light removed.  Activated via the
``host_model`` block in the input YAML; silently skipped (with a
warning upstream) when disabled or when no galfit executable exists.
"""

import glob
import logging
import os
import shutil
import subprocess

import numpy as np
from astropy.io import fits


def find_galfit(cfg: dict | None = None) -> str | None:
    """Return a runnable galfit executable path, or None."""
    cfg = cfg or {}
    cand = cfg.get("galfit_path")
    if cand:
        cand = os.path.expanduser(str(cand))
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
        return None
    found = shutil.which("galfit")
    if found:
        return found
    # Common install location used by the bundled wrapper script.
    home_bin = os.path.expanduser("~/bin/galfit")
    if os.path.isfile(home_bin) and os.access(home_bin, os.X_OK):
        return home_bin
    return None


def _render_psf_stamp(epsf_model, fwhm: float) -> np.ndarray | None:
    """Render a unit-flux PSF stamp for the galfit convolution kernel.

    Prefers the pipeline ePSF; falls back to an analytic Moffat at the
    image FWHM so a stamp is always available.
    """
    size = int(np.ceil(8.0 * float(fwhm))) | 1
    cx = cy = (size - 1) / 2.0
    model = epsf_model
    if model is None:
        try:
            from limits import _analytic_psf_for_injection

            model = _analytic_psf_for_injection(float(fwhm))
        except Exception:
            model = None
    stamp = None
    if model is not None:
        try:
            yy, xx = np.indices((size, size), dtype=float)
            osamp = getattr(model, "oversampling", 1)
            osamp = int(np.asarray(osamp).ravel()[0])
            stamp = np.asarray(
                model.evaluate(
                    x=xx, y=yy, flux=1.0, x_0=cx, y_0=cy
                ),
                dtype=np.float32,
            )
            if osamp > 1:
                stamp = stamp / float(osamp**2)
        except Exception:
            stamp = None
    if stamp is None:
        # Moffat fallback: always available, close enough for a
        # convolution kernel at these SNR levels.
        beta = 2.5
        alpha = float(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        yy, xx = np.indices((size, size), dtype=float)
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        stamp = (1.0 + r2 / alpha**2) ** (-beta)
        stamp = stamp.astype(np.float32)
    if stamp is None or not np.any(np.isfinite(stamp)):
        return None
    stamp = np.where(np.isfinite(stamp), stamp, 0.0).astype(np.float32)
    tot = float(stamp.sum())
    if not np.isfinite(tot) or tot <= 0:
        return None
    return stamp / tot


def _brightest_unmasked_pixel(image, mask, avoid_xy, avoid_r, exclude_xy, exclude_r):
    """Seed position: brightest unmasked pixel, away from given exclusions."""
    img = np.asarray(image, dtype=float)
    bad = ~np.isfinite(img)
    if mask is not None:
        bad |= np.asarray(mask, dtype=bool)
    yy, xx = np.indices(img.shape)
    if exclude_xy is not None:
        bad |= (xx - exclude_xy[0]) ** 2 + (yy - exclude_xy[1]) ** 2 <= exclude_r**2
    if avoid_xy is not None:
        bad |= (xx - avoid_xy[0]) ** 2 + (yy - avoid_xy[1]) ** 2 <= avoid_r**2
    if np.all(bad):
        return None
    v = np.where(bad, -np.inf, img)
    iy, ix = np.unravel_index(int(np.argmax(v)), v.shape)
    if not np.isfinite(img[iy, ix]):
        return None
    return float(ix), float(iy), float(img[iy, ix])


def _write_feedme(
    path,
    *,
    sci,
    sig,
    psf,
    msk,
    out,
    shape,
    fwhm,
    pixel_scale,
    zeropoint,
    sky_init,
    components,
):
    """Write a GALFIT v3 feedme file."""
    H, W = shape
    conv = min(max(int(np.ceil(4 * float(fwhm))) | 1, 11), min(H, W))
    lines = [
        f"A) {sci}",
        f"B) {out}",
        f"C) {sig}",
        f"D) {psf}",
        "E) 1",
        f"F) {msk}",
        "G) none",
        f"H) 1 {W} 1 {H}",
        f"I) {conv} {conv}",
        f"J) {float(zeropoint):.4f}",
        f"K) {float(pixel_scale):.4f} {float(pixel_scale):.4f}",
        "O) regular",
        "P) 0",
        "",
    ]
    for i, comp in enumerate(components, start=1):
        lines.append(f"# Object number: {i}")
        lines.append(f" 0) {comp['type']}")
        lines.extend(comp["lines"])
        lines.append(" Z) 0")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def _sky_line(v, tog):
    return f" 1) {v:.4f}  {int(tog)}"


def _sersic_lines(x, y, mag, re, n, ba, pa, xy_toggles=(1, 1), n_tog=1):
    return [
        f" 1) {x:.3f} {y:.3f}  {xy_toggles[0]} {xy_toggles[1]}",
        f" 3) {mag:.3f}   1",
        f" 4) {re:.3f}   1",
        f" 5) {n:.3f}   {int(n_tog)}",
        f" 9) {ba:.3f}   1",
        f"10) {pa:.3f}   1",
    ]


def _psf_lines(x, y, mag, xy_toggles=(1, 1)):
    return [
        f" 1) {x:.3f} {y:.3f}  {xy_toggles[0]} {xy_toggles[1]}",
        f" 3) {mag:.3f}   1",
    ]


def _parse_fit_log(log_path):
    """Extract chi2nu and component summaries from fit.log (best effort)."""
    info = {"chi2nu": np.nan, "components": [], "starred_params": False}
    try:
        text = open(log_path, "r", errors="replace").read()
    except Exception:
        return info
    import re

    cur = []
    cur_starred = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Chi^2/nu"):
            # fit.log appends a summary block per run; keep the block that
            # just completed - the last completed block is the current fit.
            info["components"] = cur
            info["starred_params"] = cur_starred
            cur = []
            cur_starred = False
            continue
        for kind in ("sersic", "sky", "psf", "expdisk", "gaussian", "moffat"):
            if s.startswith(kind + " "):
                nums = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", s)
                cur.append({"type": kind, "values": nums})
                # GALFIT marks hit-boundary/unreliable parameters with
                # *...* in the log; keep them parsed but flag the fit.
                if "*" in s:
                    cur_starred = True
                break
    if cur:
        info["components"] = cur
        info["starred_params"] = cur_starred
    chis = re.findall(r"Chi\^2/nu\s*=\s*([-\d.eE+]+)", text)
    if chis:
        try:
            info["chi2nu"] = float(chis[-1])
        except ValueError:
            pass
    return info


def _sane_components(fit_info, box_w):
    """Return False if any fitted profile component ran away.

    A converged galfit exit does not imply a usable model: components can
    drift outside the frame or explode in size/magnitude on mostly-sky
    data.  fit_info values are (x, y, mag, re, n, ba, pa) for profile
    components.
    """
    for comp in fit_info.get("components", []):
        if comp["type"] == "psf":
            try:
                v = [float(q) for q in comp["values"]]
            except (TypeError, ValueError):
                v = []
            # psf values are (x, y, mag): a wandering point source is as
            # useless as a runaway Sersic.
            if len(v) >= 3 and (
                not (-0.5 * box_w < v[0] < 1.5 * box_w)
                or not (-0.5 * box_w < v[1] < 1.5 * box_w)
                or not np.isfinite(v[2])
                or v[2] > 40.0
            ):
                return False
            continue
        if comp["type"] not in ("sersic", "expdisk", "gaussian", "moffat"):
            continue
        try:
            v = [float(q) for q in comp["values"]]
        except (TypeError, ValueError):
            v = []
        if len(v) < 4:
            continue
        if (
            not (-0.5 * box_w < v[0] < 1.5 * box_w)
            or not (-0.5 * box_w < v[1] < 1.5 * box_w)
            or not (0.0 < v[3] < 2.0 * box_w)
            or not np.isfinite(v[2])
            or v[2] > 40.0
            # Sersic index pinned at the galfit cap: a runaway spike, not a
            # galaxy profile.
            or (comp["type"] == "sersic" and len(v) >= 5 and v[4] >= 19.0)
        ):
            return False
    return True


def fit_host_model(
    image,
    rms,
    mask,
    *,
    target_xy,
    fwhm,
    pixel_scale,
    zeropoint=22.5,
    cfg=None,
    workdir=None,
    epsf_model=None,
    seed_mask=None,
    logger=None,
):
    """Fit the host galaxy on a cutout and return a host-subtracted image.

    Parameters
    ----------
    image       : 2-D cutout (counts in a consistent flux unit)
    rms         : 2-D sigma map matching ``image`` (may be None)
    mask        : 2-D bool, True = excluded from the galfit fit
    target_xy   : (x, y) target position in cutout coordinates
    fwhm        : image FWHM in pixels (PSF stamp + seed scales)
    pixel_scale : arcsec/pixel for the feedme header
    zeropoint   : photometric ZP (cosmetic; only affects reported mags)
    cfg         : ``host_model`` config dict
    workdir     : directory for feedme/FITS products (created if absent)
    epsf_model  : optional ePSF model for the convolution kernel

    Returns
    -------
    dict with keys: success, residual, residual_skyrestored, model,
    fit_info, workdir.  ``residual`` keeps the fitted sky removed from
    the model (i.e. data - model + sky), so the pipeline's own local
    background treatment still applies.
    """
    log = logger or logging.getLogger(__name__)
    cfg = dict(cfg or {})
    img = np.asarray(image, dtype=float)
    out = {
        "success": False,
        "residual": None,
        "model": None,
        "fit_info": {},
        "workdir": workdir,
    }
    galfit_bin = find_galfit(cfg)
    if galfit_bin is None:
        out["error"] = "galfit executable not found"
        return out

    if workdir is None:
        import tempfile

        workdir = tempfile.mkdtemp(prefix="galfit_host_")
        out["workdir"] = workdir
    os.makedirs(workdir, exist_ok=True)

    fwhm = float(fwhm)
    tx, ty = float(target_xy[0]), float(target_xy[1])
    fit_psf = bool(cfg.get("fit_point_source", False))
    force_pos = bool(cfg.get("hold_target_position", True))
    n_sersic = int(cfg.get("n_sersic", 1))
    n_sersic = 1 if n_sersic < 1 else min(n_sersic, 4)
    n_psf = int(cfg.get("n_psf", 0))
    n_psf = 0 if n_psf < 1 else min(n_psf, 3)
    seed_exclude_r = float(cfg.get("seed_exclude_target_fwhm", 1.5)) * fwhm

    # ---- input products -------------------------------------------------
    sci_f = os.path.join(workdir, "sci.fits")
    sig_f = os.path.join(workdir, "sigma.fits")
    msk_f = os.path.join(workdir, "mask.fits")
    psf_f = os.path.join(workdir, "psf.fits")
    out_f = os.path.join(workdir, "out.fits")
    feedme_f = os.path.join(workdir, "galfit.feedme")

    fits.writeto(sci_f, img.astype(np.float32), overwrite=True)
    sig_name = "none"
    if rms is not None and np.asarray(rms).shape == img.shape:
        rms_a = np.abs(np.asarray(rms, dtype=float))
        rms_a = np.where(np.isfinite(rms_a) & (rms_a > 0), rms_a, np.nanmedian(rms_a[rms_a > 0]))
        fits.writeto(sig_f, rms_a.astype(np.float32), overwrite=True)
        sig_name = "sigma.fits"
    msk_name = "none"
    if mask is not None:
        fits.writeto(msk_f, np.asarray(mask, dtype=np.int16), overwrite=True)
        msk_name = "mask.fits"
    psf_name = "none"
    stamp = _render_psf_stamp(epsf_model, fwhm)
    if stamp is not None:
        fits.writeto(psf_f, stamp.astype(np.float32), overwrite=True)
        psf_name = "psf.fits"

    # ---- seeds ----------------------------------------------------------
    finite = img[np.isfinite(img)]
    sky_init = float(np.nanmedian(finite)) if finite.size else 0.0

    # Host seed: brightest pixel inside the reopened host footprint when
    # the caller supplies one, else the brightest unmasked pixel away
    # from the target.  The target itself (if it has flux) is masked out
    # of the galfit fit, so its flux must not steer the host centroid.
    host_seed = None
    _seed_reg = None
    if seed_mask is not None and np.any(np.asarray(seed_mask, dtype=bool)):
        _sm = np.asarray(seed_mask, dtype=bool)
        _v = np.where(_sm & np.isfinite(img), img, -np.inf)
        _iy, _ix = np.unravel_index(int(np.argmax(_v)), _v.shape)
        if np.isfinite(_v[_iy, _ix]):
            host_seed = (float(_ix), float(_iy), float(img[_iy, _ix]))
            _seed_reg = _sm
    if host_seed is None:
        host_seed = _brightest_unmasked_pixel(
            img, mask, None, 0.0, (tx, ty), seed_exclude_r
        )
    if host_seed is None:
        host_seed = (tx, ty, float(np.nanmax(finite)) if finite.size else 1.0)
    hx, hy, hpk = host_seed
    host_flux = float(np.nansum(np.clip(img - sky_init, 0, None)))
    mag_init = float(zeropoint) - 2.5 * np.log10(max(host_flux, 1.0))

    # Flux-weighted centroid and size of the seed region: a sane seed for
    # the extended component of a bulge+disk style fit.
    _reg_cen = None
    _reg_re = None
    if _seed_reg is not None:
        _ys, _xs = np.indices(img.shape)
        _w = np.where(_seed_reg & np.isfinite(img), img - sky_init, 0.0)
        _w = np.clip(_w, 0, None)
        _wsum = float(_w.sum())
        if _wsum > 0:
            _reg_cen = (
                float((_xs * _w).sum() / _wsum),
                float((_ys * _w).sum() / _wsum),
            )
            _reg_re = max(
                float(np.sqrt(np.count_nonzero(_w > 0) / np.pi)), fwhm
            )

    env = dict(os.environ)
    compat = os.path.expanduser("~/lib/galfit_compat")
    if os.path.isdir(compat):
        env["LD_LIBRARY_PATH"] = compat + os.pathsep + env.get(
            "LD_LIBRARY_PATH", ""
        )
    timeout_s = float(cfg.get("timeout_s", 120.0))

    def _attempt(n_try, np_try):
        """Write a feedme with n_try Sersic and np_try psf comps, run, parse."""
        components = [
            {
                "type": "sky",
                "lines": [
                    _sky_line(sky_init, 1),
                    " 2) 0.0   0",
                    " 3) 0.0   0",
                ],
            }
        ]
        # First component: compact, at the brightness peak (nucleus-like).
        components.append(
            {
                "type": "sersic",
                "lines": _sersic_lines(
                    hx, hy, mag_init, max(fwhm, 1.5), 1.5, 0.8, 0.0
                ),
            }
        )
        _seeds = [(hx, hy)]
        for _si in range(1, n_try):
            if _si == 1 and _reg_cen is not None:
                # Second component: extended body seeded at the footprint
                # centroid (bulge+disk style; co-centred seeds separate
                # cleanly during the fit).
                sx2, sy2 = _reg_cen
                re2 = _reg_re
                n2 = 1.0
            else:
                # Further components soak up substructure at the next
                # brightest seed pixel away from existing seeds.
                seed_k = None
                if _seed_reg is not None:
                    yy_s, xx_s = np.indices(img.shape)
                    _far = np.ones(img.shape, dtype=bool)
                    for _ax, _ay in _seeds:
                        _far &= (xx_s - _ax) ** 2 + (yy_s - _ay) ** 2 > (
                            2.0 * fwhm
                        ) ** 2
                    _vk = np.where(
                        _seed_reg & np.isfinite(img) & _far, img, -np.inf
                    )
                    _ik = np.unravel_index(int(np.argmax(_vk)), _vk.shape)
                    if np.isfinite(_vk[_ik]):
                        seed_k = (
                            float(_ik[1]), float(_ik[0]), float(img[_ik])
                        )
                if seed_k is None:
                    seed_k = _brightest_unmasked_pixel(
                        img, mask, (hx, hy), 2.0 * fwhm,
                        (tx, ty), seed_exclude_r,
                    )
                if seed_k is None:
                    continue
                sx2, sy2 = seed_k[0], seed_k[1]
                re2 = max(fwhm, 1.5)
                n2 = 4.0
            _seeds.append((sx2, sy2))
            # A fixed exponential index on the extended component keeps
            # multi-Sersic fits on irregular hosts from running away.
            components.append(
                {
                    "type": "sersic",
                    "lines": _sersic_lines(
                        sx2, sy2, mag_init + 1.5, re2, n2, 0.9, 0.0,
                        n_tog=0 if _si == 1 else 1,
                    ),
                }
            )
        # Unresolved peaks (a stellar/point-like host core, nearby stars)
        # cannot be modelled by a Sersic - the profile collapses and
        # galfit crashes.  Optional psf components take them instead.
        for _pi in range(np_try):
            seed_p = None
            if _seed_reg is not None:
                yy_s, xx_s = np.indices(img.shape)
                _far = np.ones(img.shape, dtype=bool)
                for _ax, _ay in _seeds:
                    _far &= (xx_s - _ax) ** 2 + (yy_s - _ay) ** 2 > (
                        2.0 * fwhm
                    ) ** 2
                _vp = np.where(
                    _seed_reg & np.isfinite(img) & _far, img, -np.inf
                )
                _ip = np.unravel_index(int(np.argmax(_vp)), _vp.shape)
                if np.isfinite(_vp[_ip]):
                    seed_p = (
                        float(_ip[1]), float(_ip[0]), float(img[_ip])
                    )
            if seed_p is None:
                seed_p = _brightest_unmasked_pixel(
                    img, mask, (hx, hy), 2.0 * fwhm,
                    (tx, ty), seed_exclude_r,
                )
            if seed_p is None:
                continue
            _seeds.append((seed_p[0], seed_p[1]))
            p_flux = float(np.nansum(
                np.clip(
                    img[
                        max(0, int(seed_p[1]) - 3): int(seed_p[1]) + 4,
                        max(0, int(seed_p[0]) - 3): int(seed_p[0]) + 4,
                    ]
                    - sky_init,
                    0,
                    None,
                )
            ))
            p_mag = float(zeropoint) - 2.5 * np.log10(max(p_flux, 1.0))
            components.append(
                {
                    "type": "psf",
                    "lines": _psf_lines(
                        seed_p[0], seed_p[1], p_mag, (1, 1)
                    ),
                }
            )
        psf_mag_init = None
        if fit_psf:
            txy_toggles = (0, 0) if force_pos else (1, 1)
            # Seed the point-source magnitude from aperture flux at target.
            r0 = max(fwhm, 2.0)
            yy, xx = np.indices(img.shape)
            in_ap = (xx - tx) ** 2 + (yy - ty) ** 2 <= r0 * r0
            ap_flux = float(np.nansum(np.where(in_ap, img - sky_init, 0.0)))
            psf_mag_init = float(zeropoint) - 2.5 * np.log10(
                max(ap_flux, 1.0)
            )
            components.append(
                {
                    "type": "psf",
                    "lines": _psf_lines(tx, ty, psf_mag_init, txy_toggles),
                }
            )

        _write_feedme(
            feedme_f,
            sci=os.path.basename(sci_f),
            sig=sig_name,
            psf=psf_name,
            msk=msk_name,
            out=os.path.basename(out_f),
            shape=img.shape,
            fwhm=fwhm,
            pixel_scale=pixel_scale,
            zeropoint=zeropoint,
            sky_init=sky_init,
            components=components,
        )
        # fit.log appends across runs and galfit.NN restart files let a new
        # run resume from a diverged solution; start each attempt clean.
        for stale in (
            [out_f, os.path.join(workdir, "fit.log")]
            + glob.glob(os.path.join(workdir, "galfit.[0-9]*"))
        ):
            try:
                os.remove(stale)
            except OSError:
                pass
        try:
            proc = subprocess.run(
                [galfit_bin, os.path.basename(feedme_f)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"galfit timed out after {timeout_s:.0f}s"}
        except Exception as exc:
            return {"error": str(exc)}
        # galfit exits 0 even on an internal crash, so stdout is saved
        # per attempt for debugging.
        try:
            with open(
                os.path.join(workdir, f"galfit_stdout_n{n_try}.log"), "w"
            ) as fh:
                fh.write(proc.stdout or "")
                fh.write(proc.stderr or "")
        except Exception:
            pass
        if proc.returncode != 0 or not os.path.isfile(out_f):
            return {
                "error": "galfit produced no output image "
                f"(exit {proc.returncode}; crashed internally)"
            }
        try:
            hdul = fits.open(out_f)
            model = np.asarray(hdul[2].data, dtype=float)
            resid = np.asarray(hdul[3].data, dtype=float)
        except Exception as exc:
            return {"error": f"could not read galfit output: {exc}"}
        fit_info = _parse_fit_log(os.path.join(workdir, "fit.log"))
        sane = _sane_components(fit_info, max(img.shape))
        # The real test: the residual around the target must not be worse
        # than the unsubtracted data.  A runaway or absorbing model can
        # pass parameter bounds yet wreck the target neighbourhood.
        if sane:
            yy_t, xx_t = np.indices(img.shape)
            _ring = (
                (xx_t - tx) ** 2 + (yy_t - ty) ** 2
                <= (2.5 * fwhm) ** 2
            ) & ((xx_t - tx) ** 2 + (yy_t - ty) ** 2 > (0.75 * fwhm) ** 2)
            if mask is not None:
                _ring &= ~np.asarray(mask, dtype=bool)
            _ring &= np.isfinite(img) & np.isfinite(resid)
            if np.count_nonzero(_ring) > 20:
                _rms_before = float(np.nanstd(img[_ring] - sky_init))
                _rms_after = float(np.nanstd(resid[_ring]))
                if np.isfinite(_rms_after) and _rms_after > 1.1 * max(
                    _rms_before, 1e-30
                ):
                    sane = False
                    fit_info["local_residual_worse"] = True
        return {
            "model": model,
            "resid_raw": resid,
            "fit_info": fit_info,
            "n_components": (n_try, np_try),
            "sane": sane,
        }

    # Try the requested component mix, then degrade (drop Sersic count,
    # then point sources) if the richer model runs away; keep the first
    # sane result.
    _plan = list(dict.fromkeys([(n_sersic, n_psf), (1, n_psf), (1, 0)]))
    _att = None
    for _n_try, _np_try in _plan:
        _att = _attempt(_n_try, _np_try)
        if _att.get("error"):
            log.warning(
                "GALFIT attempt (n_sersic=%d, n_psf=%d): %s",
                _n_try, _np_try, _att["error"],
            )
            if (_n_try, _np_try) == _plan[-1]:
                out["error"] = _att["error"]
                return out
            continue
        if _att.get("sane"):
            break
        log.warning(
            "GALFIT attempt (n_sersic=%d, n_psf=%d) produced an "
            "out-of-bounds component; %s",
            _n_try, _np_try,
            "retrying simpler model"
            if (_n_try, _np_try) != _plan[-1]
            else "giving up",
        )
    else:
        _att = None
    if _att is None or not _att.get("sane"):
        out["error"] = "all galfit component mixes ran out of bounds"
        log.warning("GALFIT host fit failed sanity checks; see %s", workdir)
        return out

    model = _att["model"]
    resid = _att["resid_raw"]
    fit_info = _att["fit_info"]
    if _att["n_components"] != (n_sersic, n_psf):
        fit_info["degraded_to"] = _att["n_components"]

    sky_fit = np.nan
    for comp in fit_info.get("components", []):
        # fit.log sky line is "[x, y]  sky  [dsky/dx]  [dsky/dy]" - the
        # sky value is the third numeric token, not the first.
        if comp["type"] == "sky" and len(comp["values"]) >= 3:
            try:
                sky_fit = float(comp["values"][2])
            except (TypeError, ValueError):
                pass
    if not np.isfinite(sky_fit):
        sky_fit = sky_init
    fit_info["sky"] = float(sky_fit)

    # data - (model - sky): removes fitted non-sky components while keeping
    # the measured sky level for the pipeline's own local-bkg stage.
    residual = resid + sky_fit
    # If a PSF component was fitted at the target, the galfit residual has
    # it removed; add the fitted component back so the pipeline still
    # measures the target itself.
    if fit_psf and stamp is not None:
        psf_fit_mag = None
        for comp in fit_info.get("components", []):
            if comp["type"] == "psf" and len(comp["values"]) >= 3:
                try:
                    psf_fit_mag = float(comp["values"][2])
                except (TypeError, ValueError):
                    pass
        if psf_fit_mag is not None and np.isfinite(psf_fit_mag):
            psf_flux = 10.0 ** (-0.4 * (psf_fit_mag - float(zeropoint)))
            sy, sx = stamp.shape
            x0 = int(round(tx)) - sx // 2
            y0 = int(round(ty)) - sy // 2
            add = np.zeros_like(residual)
            ys0, ys1 = max(0, y0), min(residual.shape[0], y0 + sy)
            xs0, xs1 = max(0, x0), min(residual.shape[1], x0 + sx)
            if ys1 > ys0 and xs1 > xs0:
                add[ys0:ys1, xs0:xs1] = (
                    stamp[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0] * psf_flux
                )
            residual = residual + add
            fit_info["psf_target_mag"] = psf_fit_mag
            fit_info["psf_target_flux"] = psf_flux

    out.update(
        success=True,
        residual=residual.astype(np.float64),
        model=model.astype(np.float64),
        fit_info=fit_info,
    )
    log.info(
        "GALFIT host fit done: chi2nu=%s, components=%s (workdir %s)",
        f"{fit_info.get('chi2nu', np.nan):.3f}"
        if np.isfinite(fit_info.get("chi2nu", np.nan))
        else "n/a",
        [c["type"] for c in fit_info.get("components", [])],
        workdir,
    )
    return out
