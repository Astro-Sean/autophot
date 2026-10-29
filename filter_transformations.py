"""Published photometric transformations between survey systems.

Enabled by ``catalog.allow_filter_transformations``.  When a catalog
does not measure the image filter natively (e.g. a Johnson-Cousins
band for the SDSS, Pan-STARRS or Legacy Survey backends), the missing
band magnitudes are synthesized from the bands the survey does carry,
using published transformation equations.  Each published fit also
carries a measured scatter; it is folded in quadrature into the
propagated magnitude error so synthesized calibrators are weighted
correctly by the zeropoint fit.

Sources of the equations:

* Lupton (2005) - SDSS ugriz -> UBVRcIc, from the SDSS photometric
  equations page (www.sdss4.org/dr14/algorithms/sdssubvritransform).
  Applies to as-measured SDSS magnitudes, which is what the VizieR
  query returns.
* Jester et al. (2005) - ugriz -> U-B for Johnson U (all-stars fit,
  derived for Rc-Ic < 1.15).
* Tonry et al. (2012), Table 6 - Pan-STARRS1 -> Johnson-Cousins BVRI,
  quadratic fits in (g-r)_P1 from synthetic stellar photometry.
* Legacy Surveys DR10 description - PS1 -> DECam natural-system color
  terms, inverted here to recover PS1 magnitudes.  The color term
  (g-i)_PS1 is shared by all four equations, so the inversion needs
  an i-band measurement (DR10 only carries i in the south/DECam
  footprint).
* Toptun et al. (2024) - DECaLS and MzLS+BASS -> SDSS piecewise
  relations in (g-r) and (r-z).  Derived on galaxies and used only
  for LS rows without i, where the PS1 inversion is underdetermined;
  the published r.m.s. is propagated as the error floor.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Catalog-name aliases used by the resolver paths in catalog.py.
_CATALOG_ALIASES = {
    "sdss": "sdss",
    "panstarrs": "panstarrs",
    "pan_starrs": "panstarrs",
    "pan-starrs": "panstarrs",
    "ps1": "panstarrs",
    "legacy": "legacy",
    "ls": "legacy",
    "decals": "legacy",
}

# Bands each backend can synthesize.  catalog.py and prepare.py use
# this to declare which filters a catalog can serve before the
# per-source applicability is evaluated inside
# apply_filter_transformations.
TRANSFORMABLE_BANDS = {
    "sdss": frozenset({"U", "B", "V", "R", "I"}),
    "panstarrs": frozenset({"B", "V", "R", "I"}),
    "legacy": frozenset({"B", "V", "R", "I"}),
}

# Band columns recognized as transformation inputs/targets.
_ALL_BANDS = ("u", "g", "r", "i", "z", "U", "B", "V", "R", "I", "J", "H", "K", "Y", "w")

# Approximate declination split between the DECaLS (south) and
# BASS/MzLS (north) footprints used by the Legacy Surveys.
_LS_NORTH_DEC = 32.375


def _canonical_catalog(catalog_name):
    name = str(catalog_name or "").strip().lower()
    return _CATALOG_ALIASES.get(name)


def transformable_bands(catalog_name):
    """Bands the named catalog can synthesize, or an empty set."""
    name = _canonical_catalog(catalog_name)
    return TRANSFORMABLE_BANDS.get(name, frozenset())


def transformations_enabled(input_yaml):
    """True when catalog.allow_filter_transformations is set."""
    cat = (input_yaml or {}).get("catalog") or {}
    return bool(cat.get("allow_filter_transformations", False))


def _mag_array(df, band):
    """Band column as float array; non-physical entries become NaN."""
    if band not in df.columns:
        return None
    m = pd.to_numeric(df[band], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(m) & (m > 0), m, np.nan)


def _err_array(df, band):
    """<band>_err column as float array, or None when absent."""
    col = f"{band}_err"
    if col not in df.columns:
        return None
    e = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(e) & (e >= 0), e, np.nan)


def _frame(df):
    """Collect band/mag-error arrays for every known band in *df*."""
    mags, errs = {}, {}
    for b in _ALL_BANDS:
        m = _mag_array(df, b)
        if m is not None:
            mags[b] = m
            e = _err_array(df, b)
            if e is not None:
                errs[b] = e
    return mags, errs


def _fill(out, band, val, err, label):
    """First-wins merge of alternative rules for one target band.

    Later rules only fill rows the earlier ones could not produce,
    so rule order encodes preference (lower scatter, better coverage).
    """
    prev = out.get(band)
    if prev is None:
        out[band] = [np.asarray(val, float), np.asarray(err, float), label]
        return
    fill = ~np.isfinite(prev[0]) & np.isfinite(val)
    prev[0][fill] = np.asarray(val)[fill]
    prev[1][fill] = np.asarray(err)[fill]


def _propagate(coef_terms, errs, sigma, n):
    """sqrt(sigma^2 + sum coef_j^2 e_j^2); NaN errors propagate."""
    var = np.full(n, float(sigma) ** 2)
    for band, coef in coef_terms:
        e = errs.get(band)
        if e is not None:
            var = var + (coef * e) ** 2
    return np.sqrt(var)


def _linval(mags, errs, coefs, const, sigma):
    """Evaluate const + sum(coef*mag) with propagated errors.

    Coefficients come from expanding the published equation, so the
    error propagation is exact rather than an approximation through
    the color term.
    """
    terms = list(coefs.items())
    for band, _ in terms:
        if mags.get(band) is None:
            return None
    n = len(next(iter(mags.values())))
    val = np.full(n, float(const))
    for band, coef in terms:
        val = val + coef * mags[band]
    return val, _propagate(terms, errs, sigma, n)


def _tonry_eval(mags, errs, base, a0, a1, a2, sigma):
    """Tonry+2012 quadratic: out = m_base + a0 + a1*x + a2*x^2, x = g - r."""
    mg, mr, mb = mags.get("g"), mags.get("r"), mags.get(base)
    if mg is None or mr is None or mb is None:
        return None
    x = mg - mr
    val = mb + a0 + a1 * x + a2 * x * x
    dydx = a1 + 2.0 * a2 * x
    coef_map = {}
    for band, coef in (("g", dydx), ("r", -dydx), (base, 1.0)):
        coef_map[band] = coef_map.get(band, 0.0) + coef
    err = _propagate(list(coef_map.items()), errs, sigma, len(x))
    return val, err


def _piecewise_eval(mags, errs, base, c1, c2, segments, sigma):
    """Piecewise linear: out = m_base + a*(c1-c2) + b inside each segment.

    Segments are (lo, hi, slope, offset) tuples evaluated with
    ``lo < x <= hi``; ``None`` bounds are open.
    """
    mb, mc1, mc2 = mags.get(base), mags.get(c1), mags.get(c2)
    if mb is None or mc1 is None or mc2 is None:
        return None
    x = mc1 - mc2
    a = np.full(len(x), np.nan)
    b = np.full(len(x), np.nan)
    for lo, hi, ai, bi in segments:
        m = np.isfinite(x)
        if lo is not None:
            m &= x > lo
        if hi is not None:
            m &= x <= hi
        a[m] = ai
        b[m] = bi
    val = mb + a * x + b
    # Coefficients must be accumulated: base can coincide with a color
    # band (e.g. r via g-r), and a dict literal would drop it.
    coef_map = {}
    for band, coef in ((base, 1.0), (c1, a), (c2, -a)):
        coef_map[band] = coef_map.get(band, 0.0) + coef
    err = _propagate(list(coef_map.items()), errs, sigma, len(x))
    return val, err


# --- Lupton (2005): SDSS ugriz -> UBVRcIc -------------------------------
# Published form: target = m_base + k*(color) + offset.  Stored here as
# the expanded linear combination so error propagation is exact:
#   B = u - 0.8116(u-g) + 0.1313 -> 0.1884 u + 0.8116 g + 0.1313, etc.
# Rule order follows the published fit scatter (lower sigma first).
_LUPTON = {
    "B": [
        ({"u": 0.1884, "g": 0.8116}, 0.1313, 0.0095),
        ({"g": 1.3130, "r": -0.3130}, 0.2271, 0.0107),
    ],
    "V": [
        ({"g": 0.4216, "r": 0.5784}, -0.0038, 0.0054),
        ({"u": -0.2906, "g": 1.2906}, 0.0885, 0.0129),
    ],
    "R": [
        ({"r": 0.7064, "i": 0.2936}, -0.1439, 0.0072),
        ({"g": -0.1837, "r": 1.1837}, -0.0971, 0.0106),
    ],
    "I": [
        ({"i": 0.6220, "z": 0.3780}, -0.3974, 0.0063),
        ({"r": -0.2444, "i": 1.2444}, -0.3820, 0.0078),
    ],
}

# Jester et al. (2005), all-stars fit: U - B = 0.78*(u-g) - 0.88.
# Uses whatever B is already present (native or synthesized by the
# Lupton rules run before this one).
_JESTER_U = ({"B": 1.0, "u": 0.78, "g": -0.78}, -0.88, 0.05)

# --- Tonry et al. (2012), Table 6: PS1 -> Johnson-Cousins ---------------
# (base_band, A0, A1, A2, sigma) with x = (g-r)_P1.
_TONRY_BVRI = {
    "B": [("g", 0.212, 0.556, 0.034, 0.032)],
    "V": [
        ("r", 0.005, 0.462, 0.013, 0.012),
        ("g", 0.005, -0.536, 0.011, 0.012),
    ],
    "R": [("r", -0.137, -0.108, -0.029, 0.015)],
    "I": [("i", -0.366, -0.136, -0.018, 0.017)],
}


def _apply_lupton(out, mags, errs, bands, label="Lupton 2005"):
    for band in bands:
        for coefs, const, sigma in _LUPTON[band]:
            res = _linval(mags, errs, coefs, const, sigma)
            if res is not None:
                _fill(out, band, res[0], res[1], label)


def _transform_sdss(mags, errs):
    out = {}
    _apply_lupton(out, mags, errs, ("B", "V", "R", "I"))
    # B is needed first; it may itself have just been synthesized.
    if "B" in out:
        mags["B"], errs["B"] = out["B"][0], out["B"][1]
    if mags.get("B") is not None:
        res = _linval(mags, errs, *_JESTER_U)
        if res is not None:
            _fill(out, "U", res[0], res[1], "Jester+2005")
    return out


def _apply_tonry(out, mags, errs, bands, label):
    for band in bands:
        for base, a0, a1, a2, sigma in _TONRY_BVRI[band]:
            res = _tonry_eval(mags, errs, base, a0, a1, a2, sigma)
            if res is not None:
                _fill(out, band, res[0], res[1], label)


def _transform_panstarrs(mags, errs):
    out = {}
    _apply_tonry(out, mags, errs, ("B", "V", "R", "I"), "Tonry+2012")
    return out


# --- Legacy Surveys -----------------------------------------------------
# PS1 -> DECam color terms from the LS DR10 description:
#   m_DECam = m_PS + c0 + c1 x + c2 x^2 + c3 x^3,  x = (g-i)_PS,
# valid for 0.4 < x < 2.7.  Inverting them is the only published
# stellar transformation available for the LS natural system.
_DECAM_TERMS = {
    "g": (0.00062, 0.03604, 0.01028, -0.00613),
    "r": (0.00495, -0.08435, 0.03222, -0.01140),
    "i": (0.00904, -0.04171, 0.00566, -0.00829),
    "z": (0.02583, -0.07690, 0.02824, -0.00898),
}
# Residual scatter around the LS mean color terms is not published;
# a 0.02 mag floor keeps synthesized errors honest without letting
# them dominate the fit.
_LS_INVERT_SYS = 0.02


def _poly(c, x):
    return c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3


def _dpoly(c, x):
    return c[1] + 2.0 * c[2] * x + 3.0 * c[3] * x ** 2


def _legacy_to_ps1(mags, errs):
    """Invert the DR10 PS1->DECam color terms to recover PS1 mags.

    x = (g-i)_PS satisfies x = (g-i)_LS - F(x) with F = f_g - f_i;
    |F'| < 0.1 over the fit domain so fixed-point iteration converges
    in a few steps.  Rows without a finite i measurement stay NaN.
    """
    g, i = mags.get("g"), mags.get("i")
    ps_m, ps_e = {}, {}
    if g is None or i is None:
        return ps_m, ps_e
    F = tuple(a - b for a, b in zip(_DECAM_TERMS["g"], _DECAM_TERMS["i"]))
    d = g - i
    x = np.where(np.isfinite(d), d, np.nan)
    for _ in range(16):
        x_new = np.where(np.isfinite(d), d - _poly(F, x), np.nan)
        if np.allclose(x, x_new, atol=1e-12, equal_nan=True):
            x = x_new
            break
        x = x_new
    valid = np.isfinite(x) & (x >= 0.4) & (x <= 2.7)
    eg = errs.get("g")
    ei = errs.get("i")
    if eg is not None or ei is not None:
        var_x = 0.0
        if eg is not None:
            var_x = var_x + eg ** 2
        if ei is not None:
            var_x = var_x + ei ** 2
        sx = np.sqrt(var_x) / np.abs(1.0 + _dpoly(F, x))
    else:
        sx = np.zeros_like(x)
    for band, coef in _DECAM_TERMS.items():
        m = mags.get(band)
        if m is None:
            continue
        val = np.where(valid, m - _poly(coef, x), np.nan)
        var = np.full(len(x), _LS_INVERT_SYS ** 2)
        e = errs.get(band)
        if e is not None:
            var = var + e ** 2
        var = var + (_dpoly(coef, x) * sx) ** 2
        ps_m[band], ps_e[band] = val, np.sqrt(var)
    return ps_m, ps_e


# Toptun et al. (2024): piecewise DECaLS/MzLS+BASS -> SDSS relations.
# (color1, color2, segments, r.m.s.); "r_alt" is the published r-band
# fit in (r-z), used when g is unavailable.
_TOPTUN_DECALS = {
    "g": ("g", "r",
          [(None, 0.712, 0.014, 0.055),
           (0.712, 0.984, 0.404, -0.223),
           (0.984, None, 0.049, 0.126)], 0.072),
    "r": ("g", "r",
          [(None, 0.732, 0.038, 0.084),
           (0.732, 0.951, 0.252, -0.073),
           (0.951, None, 0.082, 0.089)], 0.087),
    "r_alt": ("r", "z",
              [(None, 0.629, 0.061, 0.099),
               (0.629, 1.280, 0.359, -0.088),
               (1.280, None, 0.054, 0.302)], 0.073),
    "z": ("r", "z",
          [(None, 0.779, -0.066, 0.177),
           (0.779, None, 0.191, -0.023)], 0.079),
}
_TOPTUN_BASS = {
    "g": ("g", "r",
          [(None, 0.664, -0.072, 0.071),
           (0.664, 1.102, 0.241, -0.137),
           (1.102, None, 0.017, 0.110)], 0.066),
    "r": ("g", "r",
          [(None, 0.732, -0.053, 0.120),
           (0.732, 0.951, 0.265, -0.099),
           (0.951, None, 0.048, 0.108)], 0.061),
    "r_alt": ("r", "z",
              [(None, 0.619, 0.040, 0.102),
               (0.619, 1.302, 0.262, -0.035),
               (1.302, None, 0.081, 0.201)], 0.072),
    "z": ("r", "z",
          [(None, 0.692, 0.235, -0.053),
           (0.692, 0.797, 0.047, 0.078),
           (0.797, None, -0.076, 0.175)], 0.081),
}


def _legacy_to_sdss_toptun(mags, errs, north, n):
    """LS grz -> SDSS grz via the Toptun+2024 piecewise relations.

    The fit depends on the imaging subsystem: rows north of
    dec ~32.4 deg use the MzLS+BASS coefficients, the rest DECaLS.
    """
    sd_m, sd_e = {}, {}
    cands = {"g": ("g",), "r": ("r", "r_alt"), "z": ("z",)}
    for band, keys in cands.items():
        val = np.full(n, np.nan)
        err = np.full(n, np.nan)
        for key in keys:
            for table, is_north in (
                (_TOPTUN_DECALS, False),
                (_TOPTUN_BASS, True),
            ):
                c1, c2, segments, sigma = table[key]
                res = _piecewise_eval(
                    mags, errs, band, c1, c2, segments, sigma
                )
                if res is None:
                    continue
                v, e = res
                region = north if is_north else ~north
                fill = region & ~np.isfinite(val) & np.isfinite(v)
                val[fill] = v[fill]
                err[fill] = e[fill]
        if np.isfinite(val).any():
            sd_m[band], sd_e[band] = val, err
    return sd_m, sd_e


def _transform_legacy(mags, errs, dec):
    # Stellar path: LS -> PS1 via the inverted DR10 color terms, then
    # Tonry PS1 -> BVRI.  Only rows with a finite i measurement are
    # invertible; the rest stay NaN and fall through to the fallback.
    ps_m, ps_e = _legacy_to_ps1(mags, errs)
    out_a = {}
    if ps_m:
        _apply_tonry(
            out_a, ps_m, ps_e, ("B", "V", "R", "I"),
            "LS DR10->PS1 + Tonry+2012",
        )
    # Fallback: galaxy-calibrated LS -> SDSS piecewise relations
    # (needs only g,r,z), then Lupton to BVRI.  I has no published
    # path without i, so it is left to the stellar path alone.
    n = len(next(iter(mags.values())))
    north = np.asarray(dec, dtype=float) > _LS_NORTH_DEC
    sd_m, sd_e = _legacy_to_sdss_toptun(mags, errs, north, n)
    out_b = {}
    if sd_m:
        _apply_lupton(
            out_b, sd_m, sd_e, ("B", "V", "R"),
            label="Toptun+2024 + Lupton 2005",
        )
    out = {}
    for band in ("B", "V", "R", "I"):
        a = out_a.get(band)
        b = out_b.get(band)
        if a is None and b is None:
            continue
        val = np.full(n, np.nan)
        err = np.full(n, np.nan)
        used = set()
        for src in (a, b):
            if src is None:
                continue
            fill = ~np.isfinite(val) & np.isfinite(src[0])
            val[fill] = src[0][fill]
            err[fill] = src[1][fill]
            if fill.any():
                used.add(src[2])
        if np.isfinite(val).any():
            out[band] = [val, err, " / ".join(sorted(used))]
    return out


def _write_band(df, band, val, err):
    """Fill only missing values; never overwrite a native catalog mag."""
    err_col = f"{band}_err"
    if band not in df.columns:
        df[band] = val
        df[err_col] = err
        return int(np.isfinite(val).sum())
    cur = pd.to_numeric(df[band], errors="coerce").to_numpy(dtype=float)
    fill = ~np.isfinite(cur) & np.isfinite(val)
    df[band] = np.where(fill, val, cur)
    if err_col in df.columns:
        cur_e = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
        df[err_col] = np.where(fill, err, cur_e)
    else:
        df[err_col] = np.where(fill, err, np.nan)
    return int(fill.sum())


def apply_filter_transformations(df, catalog_name):
    """Synthesize missing catalog bands in place on *df*.

    Returns ``{band: (label, n)}`` for each band that gained finite
    values; *label* names the published equations used.
    """
    if df is None or len(df) == 0:
        return {}
    name = _canonical_catalog(catalog_name)
    if name not in TRANSFORMABLE_BANDS:
        return {}
    mags, errs = _frame(df)
    if not mags:
        return {}
    if name == "sdss":
        out = _transform_sdss(mags, errs)
    elif name == "panstarrs":
        out = _transform_panstarrs(mags, errs)
    else:
        dec = df["DEC"] if "DEC" in df.columns else np.zeros(len(df))
        out = _transform_legacy(mags, errs, dec)
    report = {}
    for band, (val, err, label) in out.items():
        n = _write_band(df, band, val, err)
        if n:
            report[band] = (label, n)
    return report
