"""Additional-target support for AutoPhOT.

This module resolves, de-duplicates, and manages *additional targets* that
should be fit **simultaneously** with the primary target in a single grouped
PSF fit.  This is essential when a nearby source (e.g. another transient or a
bright host-galaxy nucleus) contaminates the primary target's PSF aperture.

User configuration (in ``default_input.yml`` / driver script)
-------------------------------------------------------------
``additional_targets``         : ``list[str]`` of object names (IAU transient
                                 names, SIMBAD-resolvable names, etc.).
``additional_targets_ra_dec``  : ``list[[RA, Dec]]`` of explicit coordinate
                                 pairs in degrees.  Used when names are not
                                 available or to skip name resolution.

Both keys may be used together; the resulting target lists are merged.

Resolution strategy
-------------------
1. For each name in ``additional_targets``:
   a. Try TNS (reusing the same ``Prepare.check_tns`` caching mechanism as the
      primary target, so repeated calls hit the on-disk cache).
   b. If TNS fails (no credentials, not a transient, network error), fall back
      to SIMBAD via ``tns.get_coords_simbad``.
2. For each ``[RA, Dec]`` pair in ``additional_targets_ra_dec``: use the
   coordinates directly (no name lookup).  A synthetic name
   ``AdditionalTarget_<index>`` is assigned if no name is supplied.
3. Crossmatch all resolved targets (primary + additional) and de-duplicate:
   any additional target within ``dedup_sep_arcsec`` of the primary target or
   another additional target is dropped (the earlier entry wins).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Separation (arcsec) below which two targets are considered the same.
DEDUP_SEP_ARCSEC = 3.0


def _safe_float(val, default=np.nan) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


def _resolve_one_name(
    name: str,
    input_yaml: Dict[str, Any],
    prepare_db=None,
) -> Optional[Dict[str, Any]]:
    """Resolve a single target name to ``{name, ra, dec, source}``.

    Tries TNS first (via ``Prepare.check_tns`` caching), then SIMBAD.
    Returns ``None`` if both fail.
    """
    name = str(name).strip()
    if not name:
        return None

    # --- TNS via Prepare.check_tns (uses on-disk cache) -----------------
    if prepare_db is not None:
        try:
            # Build a temporary config with this name so check_tns caches
            # under the correct filename.  check_tns skips the TNS query when
            # target_ra/target_dec are already set, so we must NOT copy those.
            from prepare import Prepare
            tmp_yaml = dict(input_yaml)
            tmp_yaml["target_name"] = name
            tmp_yaml.pop("target_ra", None)
            tmp_yaml.pop("target_dec", None)
            tmp_prepare = Prepare(default_input=tmp_yaml)

            tns_data = tmp_prepare.check_tns()
            if tns_data and tns_data.get("radeg") is not None:
                ra = _safe_float(tns_data["radeg"])
                dec = _safe_float(tns_data["decdeg"])
                if np.isfinite(ra) and np.isfinite(dec):
                    prefix = str(tns_data.get("name_prefix", "") or "").strip()
                    objname = str(tns_data.get("objname", name))
                    display = f"{prefix}{objname}" if prefix else objname
                    logger.info(
                        "Additional target '%s' resolved via TNS: %s  RA=%.6f  Dec=%.6f",
                        name, display, ra, dec,
                    )
                    return {"name": display, "ra": ra, "dec": dec, "source": "tns"}
        except Exception as exc:
            logger.debug("TNS resolution for '%s' failed: %s", name, exc)

    # --- SIMBAD fallback -------------------------------------------------
    try:
        from tns import get_coords_simbad
        simbad_data = get_coords_simbad(name)
        if simbad_data and simbad_data.get("radeg") is not None:
            ra = _safe_float(simbad_data["radeg"])
            dec = _safe_float(simbad_data["decdeg"])
            if np.isfinite(ra) and np.isfinite(dec):
                logger.info(
                    "Additional target '%s' resolved via SIMBAD: RA=%.6f  Dec=%.6f",
                    name, ra, dec,
                )
                return {"name": name, "ra": ra, "dec": dec, "source": "simbad"}
    except Exception as exc:
        logger.debug("SIMBAD resolution for '%s' failed: %s", name, exc)

    logger.warning("Could not resolve additional target '%s' via TNS or SIMBAD.", name)
    return None


def _angular_sep_arcsec(ra1, dec1, ra2, dec2) -> float:
    """Great-circle separation in arcsec (vectorised over ra2/dec2 arrays)."""
    ra1, dec1 = float(ra1), float(dec1)
    ra2 = np.atleast_1d(np.asarray(ra2, float))
    dec2 = np.atleast_1d(np.asarray(dec2, float))
    # Haversine
    dra = np.radians(ra2 - ra1) * np.cos(np.radians((dec1 + dec2) / 2.0))
    ddec = np.radians(dec2 - dec1)
    sep_rad = 2.0 * np.arcsin(np.sqrt(np.sin(ddec / 2.0) ** 2 + np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) * np.sin(dra / 2.0) ** 2))
    return np.degrees(sep_rad) * 3600.0


def resolve_additional_targets(
    input_yaml: Dict[str, Any],
    prepare_db=None,
    dedup_sep_arcsec: float = DEDUP_SEP_ARCSEC,
) -> List[Dict[str, Any]]:
    """Resolve and de-duplicate additional targets.

    Parameters
    ----------
    input_yaml : dict
        The full AutoPhOT configuration dict.  Reads ``additional_targets``
        (list of names) and ``additional_targets_ra_dec`` (list of [RA, Dec]).
    prepare_db : Prepare, optional
        Preparation helper used for TNS lookups (shares the on-disk cache).
    dedup_sep_arcsec : float
        Minimum allowed separation between any two targets (primary or
        additional).  Closer duplicates are dropped.

    Returns
    -------
    list of dict
        Each dict: ``{"name": str, "ra": float, "dec": float, "source": str}``.
        Empty list if no additional targets are configured.
    """
    names = input_yaml.get("additional_targets") or []
    ra_dec_pairs = input_yaml.get("additional_targets_ra_dec") or []

    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)):
        names = []
    names = [str(n).strip() for n in names if str(n).strip()]

    if not isinstance(ra_dec_pairs, (list, tuple)):
        ra_dec_pairs = []

    if not names and not ra_dec_pairs:
        return []

    resolved: List[Dict[str, Any]] = []

    # 1. Resolve names
    for name in names:
        entry = _resolve_one_name(name, input_yaml, prepare_db=prepare_db)
        if entry is not None:
            resolved.append(entry)

    # 2. Use explicit RA/Dec pairs
    for idx, pair in enumerate(ra_dec_pairs):
        try:
            if len(pair) < 2:
                continue
            ra = _safe_float(pair[0])
            dec = _safe_float(pair[1])
            if not (np.isfinite(ra) and np.isfinite(dec)):
                logger.warning("additional_targets_ra_dec[%d] has non-finite coords; skipping.", idx)
                continue
            # If a name was also supplied at the same index, use it; else synthesize.
            nm = None
            if idx < len(names) and names:
                nm = names[idx]
            if not nm:
                nm = f"Sub target {idx + 1}"
            resolved.append({"name": nm, "ra": ra, "dec": dec, "source": "manual"})
            logger.info(
                "Additional target '%s' from explicit coords: RA=%.6f  Dec=%.6f",
                nm, ra, dec,
            )
        except Exception as exc:
            logger.warning("Failed to parse additional_targets_ra_dec[%d]: %s", idx, exc)

    if not resolved:
        return []

    # 3. De-duplicate against the primary target and each other.
    primary_ra = _safe_float(input_yaml.get("target_ra"), np.nan)
    primary_dec = _safe_float(input_yaml.get("target_dec"), np.nan)

    kept: List[Dict[str, Any]] = []
    all_ras: List[float] = []
    all_decs: List[float] = []

    if np.isfinite(primary_ra) and np.isfinite(primary_dec):
        all_ras.append(primary_ra)
        all_decs.append(primary_dec)

    for entry in resolved:
        ra, dec = entry["ra"], entry["dec"]
        if not (np.isfinite(ra) and np.isfinite(dec)):
            continue
        if all_ras:
            seps = _angular_sep_arcsec(ra, dec, np.array(all_ras), np.array(all_decs))
            min_sep = float(np.min(seps))
            if min_sep < dedup_sep_arcsec:
                logger.warning(
                    "Dropping additional target '%s' (RA=%.6f, Dec=%.6f): %.2f arcsec from "
                    "an existing target (threshold %.1f arcsec).",
                    entry["name"], ra, dec, min_sep, dedup_sep_arcsec,
                )
                continue
        kept.append(entry)
        all_ras.append(ra)
        all_decs.append(dec)

    if kept:
        logger.info(
            "Resolved %d additional target(s) for simultaneous fitting: %s",
            len(kept),
            ", ".join(e["name"] for e in kept),
        )
    else:
        logger.info("No additional targets after de-duplication.")

    return kept


def sanitize_target_name_for_filename(name: str) -> str:
    """Make a target name safe for use in filenames."""
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
