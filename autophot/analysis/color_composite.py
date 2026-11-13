"""RGB color-composite images from per-band reduced frames.

Each band is resampled onto a common pixel grid (the reference band's
WCS), scaled to a shared flux scale with the per-frame exposure time and
zeropoint when available, stretched, and mixed into an RGB image.  All
frames of one band may be median-combined first, which removes cosmic
rays and transient-like differences between the exposures.

Products written next to the requested output path:

- ``Color_Image.<ext>``: the stretched RGB image (plot_format aware).
- ``Color_Image.fits`` (optional): the pre-stretch linear RGB cube with
  the reference WCS, for reuse in other tools.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from ..core.functions import _celestial_wcs, data_path, read_output_csv
from ..core.plotting_utils import get_plot_ext

log = logging.getLogger(__name__.rsplit(".", 1)[-1])

_CHANNEL_KEYS = ("R", "G", "B")


def read_band_frame(path):
    """Read a 2D FITS image and its header, float64, NaN for bad pixels."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = None
        header = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) == 2:
                data = np.asarray(hdu.data, dtype=np.float64)
                header = hdu.header.copy()
                break
    if data is None:
        raise ValueError(f"No 2D image plane found in {path}")
    return data, header


def resample_to_grid(data, wcs, ref_wcs, shape_out):
    """Place ``data`` on the reference WCS grid.

    Same-grid frames pass through unchanged (a 0.5 px corner-parity test
    is cheaper than a resample and avoids unnecessary interpolation).
    Pixels outside the source footprint come back NaN so empty coverage
    shows as black rather than a fake sky level.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.shape == tuple(shape_out) and wcs is not None and ref_wcs is not None:
        ny, nx = data.shape
        same = True
        for px, py in ((0.0, 0.0), (nx - 1.0, ny - 1.0), (nx / 2.0, ny / 2.0)):
            ra, dec = wcs.all_pix2world(px, py, 0)
            xb, yb = ref_wcs.all_world2pix(ra, dec, 0)
            if not (np.isfinite(xb) and np.isfinite(yb) and np.hypot(xb - px, yb - py) <= 0.5):
                same = False
                break
        if same:
            return data
    if wcs is None or ref_wcs is None:
        if data.shape == tuple(shape_out):
            log.warning(
                "Missing WCS on a band frame; using it on the reference "
                "pixel grid without alignment."
            )
            return data
        raise ValueError("Band image lacks WCS and cannot be resampled.")
    from reproject import reproject_interp

    out, _ = reproject_interp(
        (data, wcs), ref_wcs, shape_out=tuple(shape_out), order="bilinear"
    )
    return np.asarray(out, dtype=np.float64)


def combine_frames(frames):
    """Median-combine resampled frames; NaN-tolerant, rejects CR outliers."""
    stack = np.asarray(frames, dtype=np.float64)
    if stack.ndim == 2:
        return stack
    if stack.shape[0] == 1:
        return stack[0]
    with np.errstate(all="ignore"):
        return np.nanmedian(stack, axis=0)


def flux_scale(exptime=None, zeropoint=None):
    """Display scale factor: equal physical flux -> equal pixel value.

    Instrumental flux density is data/exptime; converting to a common
    magnitude scale multiplies by 10^(0.4*ZP).  Missing calibration
    falls back to 1.0, in which case the robust black/white stretch
    still produces a usable (but not flux-faithful) image.
    """
    scale = 1.0
    if exptime is not None:
        try:
            e = float(exptime)
            if np.isfinite(e) and e > 0:
                scale /= e
        except (TypeError, ValueError):
            pass
    if zeropoint is not None:
        try:
            z = float(zeropoint)
            if np.isfinite(z):
                scale *= 10.0 ** (0.4 * z)
        except (TypeError, ValueError):
            pass
    return scale


def black_white_levels(data, black_percentile=1.0, white_percentile=99.5):
    """Robust (black, white) display levels from finite pixels."""
    finite = data[np.isfinite(data)]
    if finite.size < 100:
        return 0.0, 1.0
    black, white = np.percentile(
        finite, [float(black_percentile), float(white_percentile)]
    )
    if not np.isfinite(black):
        black = 0.0
    if not np.isfinite(white) or white <= black:
        white = black + 1.0
    return float(black), float(white)


def normalize_band(data, black, white):
    """Linear scale to [0, 1] between the black and white levels."""
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = (data - black) / (white - black)
    norm[~np.isfinite(norm)] = 0.0
    return np.clip(norm, 0.0, 1.0)


def stretch_channel(values, stretch="asinh", softening=10.0):
    """Non-linear stretch of normalized [0,1] channel values.

    'asinh' is the Lupton-style stretch: linear near the noise floor,
    logarithmic for bright structure, so star cores do not saturate
    while faint emission stays visible.
    """
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    q = max(float(softening), 1e-3)
    stretch = str(stretch).strip().lower()
    if stretch == "asinh":
        return np.arcsinh(q * v) / np.arcsinh(q)
    if stretch == "log":
        return np.log10(1.0 + q * v) / np.log10(1.0 + q)
    if stretch == "sqrt":
        return np.sqrt(v)
    if stretch == "linear":
        return v
    raise ValueError(f"Unknown stretch {stretch!r}; use asinh/log/sqrt/linear.")


def assign_channels(bands, channel_map=None, w_eff=None):
    """Map available bands onto R/G/B output channels.

    ``channel_map`` may name one band (``{"R": "i"}``) or a list of bands
    mixed with equal weight (``{"G": ["g", "V"]}``).  Without a map, the
    bands are sorted by effective wavelength: reddest -> R, middle -> G,
    bluest -> B.  With two bands the missing G is synthesized as the
    mean of R and B so stars do not render pure magenta.
    """
    bands = [str(b) for b in bands]
    if w_eff is None:
        w_eff = {}
    if channel_map:
        out = {}
        for key in _CHANNEL_KEYS:
            sel = channel_map.get(key)
            if sel is None:
                out[key] = []
            elif isinstance(sel, (list, tuple)):
                out[key] = [str(b) for b in sel if str(b) in bands]
            else:
                out[key] = [str(sel)] if str(sel) in bands else []
        if any(out[k] for k in _CHANNEL_KEYS):
            return out
        log.warning(
            "color_image_channels matched no available band; "
            "falling back to wavelength ordering."
        )
    ordered = sorted(bands, key=lambda b: float(w_eff.get(b, 0.0)), reverse=True)
    if len(ordered) >= 3:
        return {"R": [ordered[0]], "G": [ordered[len(ordered) // 2]], "B": [ordered[-1]]}
    if len(ordered) == 2:
        return {"R": [ordered[0]], "G": ["__synthetic__"], "B": [ordered[1]]}
    return {"R": ordered[:1], "G": ordered[:1], "B": ordered[:1]}


def compose_rgb(normalized_bands, channels):
    """Build the (ny, nx, 3) array from normalized [0,1] band images."""
    some = next(iter(normalized_bands.values()))
    rgb = np.zeros(some.shape + (3,), dtype=np.float64)
    for i, key in enumerate(_CHANNEL_KEYS):
        sel = channels.get(key) or []
        if sel == ["__synthetic__"] or not sel:
            continue
        mix = np.zeros_like(some)
        for band in sel:
            mix += normalized_bands[band]
        rgb[:, :, i] = np.clip(mix / len(sel), 0.0, 1.0)
    # Synthesized channels are deferred: they average the already filled
    # channels, so they must run after every real band is placed.
    for i, key in enumerate(_CHANNEL_KEYS):
        if (channels.get(key) or []) == ["__synthetic__"]:
            others = [rgb[:, :, j] for j in range(3) if j != i]
            rgb[:, :, i] = 0.5 * (others[0] + others[1])
    return rgb


def adjust_rgb(rgb, gamma=1.0, saturation=1.0):
    """Final cosmetic adjustments: gamma (brightness curve) and saturation."""
    out = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    g = float(gamma)
    if np.isfinite(g) and g > 0 and not np.isclose(g, 1.0):
        out = out ** (1.0 / g)
    s = float(saturation)
    if np.isfinite(s) and not np.isclose(s, 1.0):
        lum = (
            0.2126 * out[:, :, 0] + 0.7152 * out[:, :, 1] + 0.0722 * out[:, :, 2]
        )[:, :, None]
        out = np.clip(lum + s * (out - lum), 0.0, 1.0)
    return out


def save_rgb_png(rgb, path, title=None, dpi=200):
    """Write the RGB image through matplotlib so plot_format/mplstyle apply."""
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")

    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_axis_off()
    if title:
        import matplotlib.patheffects as pe

        ax.text(
            0.5,
            0.99,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=max(6, int(dpi / 20)),
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_rgb_fits(rgb, path, header=None):
    """Persist the linear (pre-stretch) RGB cube with the reference WCS."""
    from astropy.io import fits

    hdu = fits.PrimaryHDU(
        np.moveaxis(np.asarray(rgb, dtype=np.float32), -1, 0), header=header
    )
    hdu.header["NCHANNEL"] = (3, "RGB channel count")
    hdu.writeto(path, overwrite=True)
    return path


def generate_color_image(
    band_frames,
    output_base,
    reference=None,
    channel_map=None,
    stretch="asinh",
    softening=10.0,
    black_percentile=1.0,
    white_percentile=99.5,
    gamma=1.0,
    saturation=1.0,
    w_eff=None,
    save_fits=False,
    title=None,
    dpi=200,
):
    """Compose an RGB image from ``{band: [(data, header, exptime, zp)]}``.

    ``band_frames`` maps a band name to a list of per-frame records
    ``(data, header, exptime, zeropoint)``; every band's frames are
    resampled onto the reference grid and median-combined before
    normalization.  Returns the written PNG path (and FITS path when
    ``save_fits``) or None when nothing usable was produced.
    """
    if not band_frames:
        return None
    if w_eff is None:
        w_eff = _load_w_eff()

    ref_wcs, ref_shape = None, None
    if reference is not None:
        ref_wcs = _celestial_wcs(reference.get("header", reference))
        ref_shape = tuple(reference["data"].shape) if "data" in reference else None
    if ref_wcs is None or ref_shape is None:
        first_band = next(iter(band_frames))
        first = band_frames[first_band][0]
        ref_wcs = _celestial_wcs(first[1])
        ref_shape = first[0].shape
    if ref_wcs is None:
        log.warning(
            "No usable WCS on the reference frame; bands must already "
            "share a pixel grid."
        )

    scaled_bands = {}
    for band, frames in band_frames.items():
        placed = []
        for data, header, exptime, zp in frames:
            wcs = _celestial_wcs(header)
            try:
                resampled = resample_to_grid(data, wcs, ref_wcs, ref_shape)
            except Exception as exc:
                log.warning(
                    "Skipping a %s-band frame that could not be resampled: %s",
                    band, exc,
                )
                continue
            placed.append(resampled * flux_scale(exptime, zp))
        if placed:
            scaled_bands[str(band)] = combine_frames(placed)
    if not scaled_bands:
        log.warning("No band frames survived resampling; no color image.")
        return None

    normalized = {}
    for band, data in scaled_bands.items():
        black, white = black_white_levels(
            data, black_percentile, white_percentile
        )
        normalized[band] = normalize_band(data, black, white)

    channels = assign_channels(list(scaled_bands), channel_map, w_eff)
    rgb = compose_rgb(normalized, channels)
    rgb = np.stack(
        [stretch_channel(rgb[:, :, i], stretch, softening) for i in range(3)],
        axis=-1,
    )
    rgb = adjust_rgb(rgb, gamma=gamma, saturation=saturation)

    output_base = str(output_base)
    png_path = save_rgb_png(rgb, output_base, title=title, dpi=dpi)
    written = {"image": png_path}
    if save_fits:
        hdr = None
        if ref_wcs is not None:
            hdr = ref_wcs.to_header(relax=True)
        fits_path = os.path.splitext(output_base)[0] + ".fits"
        written["fits"] = save_rgb_fits(rgb, fits_path, header=hdr)
    return written


def _load_w_eff():
    """Band -> effective wavelength (A), filters.yml over built-ins."""
    try:
        from .lightcurve import BAND_WAVELENGTHS

        w_eff = dict(BAND_WAVELENGTHS)
    except Exception:
        w_eff = {}
    try:
        from ..core.functions import AutophotYaml

        w_eff.update(
            AutophotYaml(data_path("filters.yml")).load().get("W_eff", {}) or {}
        )
    except Exception:
        pass
    return w_eff


def _records_from_outputs(reduced_loc):
    """Collect (path, band, exptime, zp, fwhm) per processed image.

    The per-image ``Output_*.csv`` files record the FITS path, raw
    filter, exposure, measured FWHM and aperture zeropoint - everything
    needed to group frames into bands and put them on one flux scale.
    """
    from .lightcurve import (
        canonical_bands_from_filter_series,
        filter_value_matches_band,
        photometry_filter_series,
    )
    import pandas as pd

    rows = []
    for path in sorted(Path(reduced_loc).rglob("Output_*.csv")):
        try:
            df = read_output_csv(str(path))
        except Exception:
            continue
        if df.empty:
            continue
        rec = df.iloc[0]
        fpath = rec.get("filename_path", None)
        if not fpath or not os.path.exists(str(fpath)):
            continue
        zp = rec.get("zp_ap", np.nan)
        try:
            zp = float(zp)
        except (TypeError, ValueError):
            zp = np.nan
        if not np.isfinite(zp):
            try:
                zp = float(rec.get("zp_psf", np.nan))
            except (TypeError, ValueError):
                zp = np.nan
        rows.append(
            {
                "path": str(fpath),
                "filter": rec.get("filter", ""),
                "exptime": rec.get("exposure_time", np.nan),
                "fwhm": rec.get("image_fwhm", np.nan),
                "zp": zp,
            }
        )
    if not rows:
        return []
    meta = pd.DataFrame(rows)
    fser = photometry_filter_series(meta)
    if fser is not None and fser.notna().any():
        present = canonical_bands_from_filter_series(fser)
        raw_to_band = {}
        for raw in meta["filter"].astype(str).unique():
            raw_to_band[raw] = next(
                (b for b in present if filter_value_matches_band(b, raw)), raw
            )
        meta["band"] = meta["filter"].astype(str).map(raw_to_band)
    else:
        meta["band"] = meta["filter"].astype(str).str.strip()
    return meta.to_dict("records")


def generate_color_image_from_run(reduced_loc, input_yaml=None, log=None):
    """Driver entry: build the composite from a finished run's outputs.

    Bands with several frames are stacked after resampling, so rotating
    or dithered pointings still produce one seamless channel.  The
    sharpest frame of the reddest channel supplies the reference grid.
    """
    log = log or logging.getLogger(__name__.rsplit(".", 1)[-1])
    cfg = input_yaml or {}
    records = _records_from_outputs(reduced_loc)
    records = [
        r
        for r in records
        if str(r.get("band") or "").strip().lower() not in ("", "nan", "none")
    ]
    if not records:
        log.warning("color_image: no Output_*.csv frames found under %s", reduced_loc)
        return None

    band_map = {}
    for rec in records:
        band_map.setdefault(rec["band"], []).append(rec)

    w_eff = _load_w_eff()
    bands_sorted = sorted(
        band_map, key=lambda b: float(w_eff.get(b, 0.0)), reverse=True
    )
    max_frames = int(cfg.get("color_image_max_frames_per_band", 0) or 0)

    def _fwhm_key(rec):
        try:
            f = float(rec.get("fwhm"))
        except (TypeError, ValueError):
            return np.inf
        return f if np.isfinite(f) else np.inf

    band_frames = {}
    ref_record = None
    ref_data = ref_header = None
    for band in bands_sorted:
        recs = sorted(band_map[band], key=_fwhm_key)
        if max_frames > 0:
            recs = recs[:max_frames]
        frames = []
        for rec in recs:
            try:
                data, header = read_band_frame(rec["path"])
            except Exception as exc:
                log.warning(
                    "color_image: could not read %s (%s); skipped.", rec["path"], exc
                )
                continue
            frames.append((data, header, rec.get("exptime"), rec.get("zp")))
            if band == bands_sorted[0] and ref_record is None:
                ref_record, ref_data, ref_header = rec, data, header
        if frames:
            band_frames[band] = frames

    if len(band_frames) < 1 or ref_record is None:
        log.warning("color_image: no readable band frames; nothing produced.")
        return None

    reference = {"data": ref_data, "header": ref_header}
    ext = get_plot_ext(cfg)
    out_base = os.path.join(reduced_loc, "Color_Image")
    out_path = out_base + ext
    channel_map = cfg.get("color_image_channels") or None
    target_name = cfg.get("target_name")
    written = generate_color_image(
        band_frames,
        out_path,
        reference=reference,
        channel_map=channel_map,
        stretch=cfg.get("color_image_stretch", "asinh"),
        softening=cfg.get("color_image_softening", 10.0),
        black_percentile=cfg.get("color_image_black_percentile", 1.0),
        white_percentile=cfg.get("color_image_white_percentile", 99.5),
        gamma=cfg.get("color_image_gamma", 1.0),
        saturation=cfg.get("color_image_saturation", 1.0),
        w_eff=w_eff,
        save_fits=bool(cfg.get("color_image_save_fits", False)),
        title=(f"{target_name} - " if target_name else "")
        + "+".join(band_frames),
        dpi=int(cfg.get("color_image_dpi", 200) or 200),
    )
    if written:
        log.info("color_image: wrote %s", written["image"])
    return written


def _split_paths(value):
    return [p for p in (v.strip() for v in str(value).split(",")) if p]


def main(argv=None):
    """``autophot color-image`` entry point."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="autophot color-image",
        description=(
            "Build an RGB composite from per-band FITS frames. "
            "Each channel flag accepts a comma-separated list whose "
            "frames are stacked after resampling."
        ),
    )
    ap.add_argument("--red", action="append", default=[], help="R-channel FITS file(s)")
    ap.add_argument("--green", action="append", default=[], help="G-channel FITS file(s)")
    ap.add_argument("--blue", action="append", default=[], help="B-channel FITS file(s)")
    ap.add_argument("--out", default="Color_Image.png", help="Output image path")
    ap.add_argument("--reference", default=None, help="FITS file supplying the output WCS grid")
    ap.add_argument("--stretch", default="asinh", choices=["asinh", "log", "sqrt", "linear"])
    ap.add_argument("--softening", type=float, default=10.0, help="asinh/log stretch strength")
    ap.add_argument("--black-percentile", type=float, default=1.0)
    ap.add_argument("--white-percentile", type=float, default=99.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--saturation", type=float, default=1.0)
    ap.add_argument("--save-fits", action="store_true", help="Also write the linear RGB cube")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    channel_files = {
        "R": [p for v in args.red for p in _split_paths(v)],
        "G": [p for v in args.green for p in _split_paths(v)],
        "B": [p for v in args.blue for p in _split_paths(v)],
    }
    if not any(channel_files.values()):
        ap.error("provide at least one of --red/--green/--blue")

    band_frames = {}
    reference = None
    for key in ("R", "G", "B"):
        paths = channel_files[key]
        if not paths:
            continue
        band = f"__{key}__"
        band_frames[band] = []
        for path in paths:
            data, header = read_band_frame(path)
            band_frames[band].append((data, header, None, None))
        if key == "R" and reference is None:
            reference = {"data": band_frames[band][0][0], "header": band_frames[band][0][1]}
    if args.reference:
        data, header = read_band_frame(args.reference)
        reference = {"data": data, "header": header}

    channel_map = {
        k: ([f"__{k}__"] if channel_files[k] else []) for k in ("R", "G", "B")
    }
    written = generate_color_image(
        band_frames,
        args.out,
        reference=reference,
        channel_map=channel_map,
        stretch=args.stretch,
        softening=args.softening,
        black_percentile=args.black_percentile,
        white_percentile=args.white_percentile,
        gamma=args.gamma,
        saturation=args.saturation,
        w_eff={},
        save_fits=args.save_fits,
        dpi=args.dpi,
    )
    if written is None:
        return 1
    print(f"Wrote {written['image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
